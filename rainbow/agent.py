import torch
import numpy as np
import datetime
from .utils.memories import ReplayMemory, RNNReplayMemory, MultiStepsBuffer
from .utils.models import ModelBuilder
import os
import dill
import glob
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

class Rainbow:
    def __init__(self,
            nb_states,
            nb_actions,
            gamma,
            replay_capacity,
            learning_rate,
            batch_size,
            epsilon_function = lambda episode, step : max(0.001, (1 - 5E-5)** step),
            # Model builders
            window = 1, # 1 = Classic , >1 = RNN
            units = [32, 32],
            dropout = 0.,
            adversarial = False,
            noisy = False,
            # Double DQN
            tau = 500,
            # Multi Steps replay
            multi_steps = 1,
            # Distributional
            distributional = False,
            nb_atoms = 51,
            v_min= -200,
            v_max= 200,
            # Prioritized replay
            prioritized_replay = False,
            prioritized_replay_alpha =0.65,
            prioritized_replay_beta_function = lambda episode, step : min(1, 0.4 + 0.6*step/50_000),
            # Vectorized envs
            simultaneous_training_env = 1,
            train_every = 1,
            name = "Rainbow",
            # Scheduler
            scheduler = False,
            # Tensorboard
            tensorboard = False
        ):
        self.name = name
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.epsilon_function = epsilon_function if not noisy else lambda episode, step : 0
        self.replay_capacity = replay_capacity
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta_function = prioritized_replay_beta_function
        self.train_every = train_every
        self.multi_steps = multi_steps
        self.simultaneous_training_env = simultaneous_training_env
        self.tensorboard = tensorboard
        self.scheduler = scheduler

        self.recurrent = window > 1
        self.window = window

        self.nb_atoms = nb_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.distributional = distributional

        # Memory
        self.replay_memory = ReplayMemory(capacity= replay_capacity, nb_states= nb_states, prioritized = prioritized_replay, alpha= prioritized_replay_alpha)
        if self.recurrent:
            self.replay_memory = RNNReplayMemory(window= window, capacity= replay_capacity, nb_states= nb_states, prioritized = prioritized_replay, alpha= prioritized_replay_alpha)
        if self.multi_steps > 1:
            self.multi_steps_buffers = [MultiStepsBuffer(self.multi_steps, self.gamma) for _ in range(simultaneous_training_env)]

        # Models
        model_builder = ModelBuilder(
            units = units,
            dropout= dropout,
            nb_states= nb_states,
            nb_actions= nb_actions,
            l2_reg= None,
            window= window,
            distributional= distributional, nb_atoms= nb_atoms,
            adversarial= adversarial,
            noisy = noisy
        )
        input_shape = (nb_states,)
        self.model = model_builder.build_model(trainable= True)
        self.target_model = model_builder.build_model(trainable= False)
        self.target_model.load_state_dict(self.model.state_dict())
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1.5E-4)

        # History
        self.steps = 0
        self.losses = []
        self.episode_rewards = [[] for _ in range(self.simultaneous_training_env)]
        self.episode_count = [0 for _ in range(self.simultaneous_training_env)]
        self.episode_steps = [0 for _ in range(self.simultaneous_training_env)]

        # INITIALIZE CORE FUNCTIONS
        # Distributional training
        if self.distributional:
            self.delta_z = (v_max - v_min) / (nb_atoms - 1)
            self.zs = torch.linspace(v_min, v_max, nb_atoms)

        # LR Scheduler
        if self.scheduler:
            self.lr_scheduler = ReduceLROnPlateau(self.model_optimizer, mode='min', factor=0.1, patience=100)

        self.start_time = datetime.datetime.now()

        # Initialize Tensorboard
        if self.tensorboard:
            self.log_dir = f"logs/{name}_{self.start_time.strftime('%Y_%m_%d-%H_%M_%S')}"
            self.train_summary_writer = SummaryWriter(self.log_dir)

    def new_episode(self, i_env):
        self.episode_count[i_env] += 1
        self.episode_steps[i_env] = 0
        self.episode_rewards[i_env] = []

    def store_replay(self, state, action, reward, next_state, done, truncated, i_env=0):
        # Case where no multi-steps:
        if self.multi_steps == 1:
            self.replay_memory.store(
                state, action, reward, next_state, done
            )
        else:
            self.multi_steps_buffers[i_env].add(state, action, reward, next_state, done)
            if self.multi_steps_buffers[i_env].is_full():
                self.replay_memory.store(
                    *self.multi_steps_buffers[i_env].get_multi_step_replay()
                )

        # Store history
        self.episode_rewards[i_env].append(reward)

        if done or truncated:
            self.log(i_env)
            self.new_episode(i_env)

    def store_replays(self, states, actions, rewards, next_states, dones, truncateds):
        for i_env in range(len(actions)):
            self.store_replay(
                states[i_env], actions[i_env], rewards[i_env], next_states[i_env], dones[i_env], truncateds[i_env], i_env=i_env
            )

    def train(self):
        self.steps += 1
        for i_env in range(self.simultaneous_training_env):
            self.episode_steps[i_env] += 1
        if self.replay_memory.size() < self.batch_size or self.get_current_epsilon() >= 1:
            return

        if self.steps % self.tau == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if self.steps % self.train_every == 0:
            batch_indexes, states, actions, rewards, states_prime, dones, importance_weights = self.replay_memory.sample(
                self.batch_size,
                self.prioritized_replay_beta_function(sum(self.episode_count), self.steps)
            )

            loss_value, td_errors, metrics = self.train_step(states, actions, rewards, states_prime, dones, importance_weights)
            self.replay_memory.update_priority(batch_indexes, td_errors)

            self.losses.append(float(loss_value))

            # Tensorboard
            if self.tensorboard:
                self.train_summary_writer.add_scalar('Step Training Loss', loss_value, self.steps)
                self.train_summary_writer.add_scalar('Gradient Norm', metrics['grad_norm'], self.steps)
                self.train_summary_writer.add_scalar('Update Norm', metrics['update_norm'], self.steps)
                self.train_summary_writer.add_scalar('Momentum Norm', metrics['momentum_norm'], self.steps)
                if not np.isnan(metrics['grad_momentum_angle']):
                    self.train_summary_writer.add_scalar('Angle between Gradient and Momentum', metrics['grad_momentum_angle'], self.steps)
                if not np.isnan(metrics['grad_update_angle']):
                    self.train_summary_writer.add_scalar('Angle between Gradient and Update', metrics['grad_update_angle'], self.steps)
                self.train_summary_writer.add_scalar('Learning Rate', metrics['learning_rate'], self.steps)

    def log(self, i_env=0):
        text_print = f"\
↳ Env {i_env} : {self.episode_count[i_env]:03} : {self.steps:8d}   |   {self.format_time(datetime.datetime.now() - self.start_time)}   |   Epsilon : {self.get_current_epsilon()*100:4.2f}%   |   Mean Loss (last 10k) : {np.mean(self.losses[-10_000:]):0.4E}   |   Tot. Rewards : {np.sum(self.episode_rewards[i_env]):8.2f}   |   Rewards (/1000 steps) : {1000 * np.sum(self.episode_rewards[i_env]) / self.episode_steps[i_env]:8.2f}   |   Length : {self.episode_steps[i_env]:6.0f}"
        print(text_print)

    def get_current_epsilon(self, delta_episode=0, delta_steps=0):
        # if self.noisy: return 0
        return self.epsilon_function(sum(self.episode_count) + delta_episode, self.steps + delta_steps)

    def e_greedy_pick_action(self, state):
        epsilon = self.get_current_epsilon()
        if np.random.rand() < epsilon:
            return np.random.choice(self.nb_actions)
        with torch.no_grad():
            return self.pick_action(state)

    def e_greedy_pick_actions(self, states):
        epsilon = self.get_current_epsilon()
        if np.random.rand() < epsilon:
            return np.random.choice(self.nb_actions, size=self.simultaneous_training_env)
        with torch.no_grad():
            return self.pick_actions(states).detach().cpu().numpy()

    def format_time(self, t: datetime.timedelta):
        h = t.total_seconds() // (60 * 60)
        m = (t.total_seconds() % (60 * 60)) // 60
        s = t.total_seconds() % 60
        return f"{h:02.0f}:{m:02.0f}:{s:02.0f}"

    def train_step(self, *args, **kwargs):
        if self.distributional:
            return self._distributional_train_step(*args, **kwargs)
        return self._classic_train_step(*args, **kwargs)

    def validation_step(self, states, actions, rewards_n, states_prime_n, dones_n):
        if self.distributional:
            return self._distributional_validation_step(states, actions, rewards_n, states_prime_n, dones_n)
        return self._classic_validation_step(states, actions, rewards_n, states_prime_n, dones_n)

    def pick_action(self, *args, **kwargs):
        if self.distributional:
            return int(self._distributional_pick_action(*args, **kwargs))
        return int(self._classic_pick_action(*args, **kwargs))

    def pick_actions(self, *args, **kwargs):
        if self.distributional:
            return self._distributional_pick_actions(*args, **kwargs)
        return self._classic_pick_actions(*args, **kwargs)

    def _classic_td_errors(self, states, actions, rewards_n, states_prime_n, dones_n):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards_n = torch.tensor(rewards_n, dtype=torch.float32)
        states_prime_n = torch.tensor(states_prime_n, dtype=torch.float32)
        dones_n = torch.tensor(dones_n, dtype=torch.float32)

        best_ap = torch.argmax(self.model(states_prime_n).detach(), dim=1)
        with torch.no_grad():
            max_q_sp_ap = self.target_model(states_prime_n).gather(1, best_ap.unsqueeze(1)).squeeze(1)
        q_a_target = rewards_n + (1 - dones_n) * (self.gamma ** self.multi_steps) * max_q_sp_ap

        q_prediction = self.model(states)
        q_a_prediction = q_prediction.gather(1, actions.unsqueeze(1)).squeeze(1)
        return torch.abs(q_a_target - q_a_prediction)

    # Classic DQN Core Functions
    def _classic_train_step(self, states, actions, rewards_n, states_prime_n, dones_n, weights):
        weights = torch.tensor(weights, dtype=torch.float32)
        td_errors = self._classic_td_errors(states, actions, rewards_n, states_prime_n, dones_n)
        loss_value = (td_errors.pow(2) * weights).mean()

        return self._optimize_with_metrics(loss_value, td_errors)

    def _optimize_with_metrics(self, loss_value, td_errors):
        self.model_optimizer.zero_grad()
        loss_value.backward()

        # Opimizer metrics
        metrics = {}
        grad_norm, grad_vector = self.compute_grad()
        metrics['grad_norm'] = grad_norm

        # Save parameters before the update
        parameters_before = [p.clone().detach() for p in self.model.parameters()]
        momentum_norm, momentum_vector = self.compute_momentum()
        metrics['momentum_norm'] = momentum_norm
        metrics['grad_momentum_angle'] = self.compute_grad_momentum_angle(grad_vector, momentum_vector)

        # Update parameters
        self.model_optimizer.step()

        # Adjust Learning Rate
        learning_rate = self.model_optimizer.param_groups[0]['lr']
        metrics['learning_rate'] = learning_rate # self.lr_scheduler.get_last_lr()[0]
        if self.scheduler:
            self.lr_scheduler.step(metrics['grad_norm'])

        # Save parameters after the update
        update_norm, update_vector = self.compute_update(parameters_before)
        metrics['update_norm'] = update_norm
        metrics['grad_update_angle'] = self.compute_grad_update_angle(grad_vector, update_vector)

        return loss_value.item(), td_errors.detach().cpu().numpy(), metrics

    def _classic_pick_actions(self, states):
        states = torch.tensor(states, dtype=torch.float32)
        return torch.argmax(self.model(states), dim=1)

    def _classic_pick_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return torch.argmax(self.model(state), dim=1)

    # Distributional Core Functions
    def _distributional_pick_actions(self, states):
        states = torch.tensor(states, dtype=torch.float32)
        return torch.argmax(self._distributional_predict_q_a(self.model, states), dim=1)

    def _distributional_pick_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return torch.argmax(self._distributional_predict_q_a(self.model, state), dim=1)

    def _distributional_predict_q_a(self, model, s):
        p_a = model(s)
        q_a = torch.sum(p_a * self.zs, dim=-1)
        return q_a

    def _distributional_td_errors(self, states, actions, rewards, states_prime, dones):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        states_prime = torch.tensor(states_prime, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        batch_size = states.size(0)

        best_a_sp = torch.argmax(self._distributional_predict_q_a(self.model, states_prime).detach(), dim=1)

        Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * (self.gamma ** self.multi_steps) * self.zs.unsqueeze(0)
        Tz = Tz.clamp(self.v_min, self.v_max)

        b_j = (Tz - self.v_min) / self.delta_z
        l = b_j.floor().long()
        u = b_j.ceil().long()

        with torch.no_grad():
            p_max_ap_sp = self.target_model(states_prime).gather(1, best_a_sp.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.nb_atoms)).squeeze(1)

        m = torch.zeros(batch_size, self.nb_atoms)
        offset = torch.linspace(0, (batch_size - 1) * self.nb_atoms, batch_size).long().unsqueeze(1).expand(batch_size, self.nb_atoms)

        l_idx = l + offset
        u_idx = u + offset

        m.view(-1).index_add_(0, l_idx.view(-1), (p_max_ap_sp * (u - b_j)).view(-1))
        m.view(-1).index_add_(0, u_idx.view(-1), (p_max_ap_sp * (b_j - l)).view(-1))

        p_s_a = self.model(states).gather(1, actions.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.nb_atoms)).squeeze(1)
        p_s_a = p_s_a.clamp(1e-6, 1 - 1e-6)

        return -torch.sum(m * torch.log(p_s_a), dim=1)

    def _distributional_train_step(self, states, actions, rewards, states_prime, dones, weights):
        weights = torch.tensor(weights, dtype=torch.float32)
        td_errors = self._distributional_td_errors(states, actions, rewards, states_prime, dones)
        td_errors_weighted = td_errors * weights
        loss_value = td_errors_weighted.mean()

        return self._optimize_with_metrics(loss_value, td_errors)

    # Compute gradient norm
    def compute_grad(self):
        grad_norm = 0.0
        grad_vector = []
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
                grad_vector.append(p.grad.data.view(-1))
        grad_norm = grad_norm ** 0.5
        grad_vector = torch.cat(grad_vector)
        return grad_norm, grad_vector

    # Get optimizer state (momentum terms) before the update
    def compute_momentum(self):
        momentum_vector = []
        for p in self.model.parameters():
            state = self.model_optimizer.state[p]
            if 'exp_avg' in state:
                m = state['exp_avg']
                momentum_vector.append(m.data.view(-1))
        if len(momentum_vector) == 0:
            # Optimizer state is empty; return zero norm and zero vector
            total_params = sum(p.numel() for p in self.model.parameters())
            momentum_vector = torch.zeros(total_params)
            momentum_norm = 0.0
        else:
            momentum_vector = torch.cat(momentum_vector)
            momentum_norm = momentum_vector.norm(2).item()
        return momentum_norm, momentum_vector

    # Compute angle between gradient and momentum
    def compute_grad_momentum_angle(self, grad_vector, momentum_vector):
        if momentum_vector.numel() == 0 or momentum_vector.norm(2).item() == 0.0:
            # Momentum vector is zero; angle is undefined
            return np.nan  # or return None
        dot_product = torch.dot(grad_vector, momentum_vector)
        return torch.acos(
            dot_product / (grad_vector.norm(2) * momentum_vector.norm(2) + 1e-8)
        ).item()

    # Compute update norm
    def compute_update(self, parameters_before):
        update_vector = []
        update_norm = 0.0
        for p_before, p_after in zip(parameters_before, self.model.parameters()):
            delta = p_after - p_before
            update_norm += delta.data.norm(2).item() ** 2
            update_vector.append(delta.data.view(-1))
        update_norm = update_norm ** 0.5
        update_vector = torch.cat(update_vector)
        return update_norm, update_vector

    # Compute angle between gradient and update
    def compute_grad_update_angle(self, grad_vector, update_vector):
        if update_vector.numel() == 0 or update_vector.norm(2).item() == 0.0:
            # Update vector is zero; angle is undefined
            return np.nan  # or return None
        dot_product = torch.dot(grad_vector, update_vector)
        return torch.acos(
            dot_product / (grad_vector.norm(2) * update_vector.norm(2) + 1e-8)
        ).item()

    def _classic_validation_step(self, states, actions, rewards_n, states_prime_n, dones_n):
        with torch.no_grad():
            td_errors = self._classic_td_errors(states, actions, rewards_n, states_prime_n, dones_n)
            loss_value = (td_errors.pow(2)).mean()

        return loss_value.item()

    def _distributional_validation_step(self, states, actions, rewards, states_prime, dones):
        with torch.no_grad():
            td_errors = self._distributional_td_errors(states, actions, rewards, states_prime, dones)
            loss_value = td_errors.mean()

        return loss_value.item()


    def save(self, path, **kwargs):
        self.saved_path = path
        if not os.path.exists(path):
            os.makedirs(path)

        if self.model is not None:
            torch.save(self.model.state_dict(), f"{path}/model.pth")
        if self.target_model is not None:
            torch.save(self.target_model.state_dict(), f"{path}/target_model.pth")

        with open(f'{path}/agent.pkl', 'wb') as file:
            dill.dump(self, file)
        for key, element in kwargs.items():
            if isinstance(element, dict):
                with open(f'{path}/{key}.json', 'w') as file:
                    json.dump(element, file)
            else:
                with open(f'{path}/{key}.pkl', 'wb') as file:
                    dill.dump(element, file)

    def __getstate__(self):
        print("Saving agent ...")
        return_dict = self.__dict__.copy()
        return_dict.pop('model', None)
        return_dict.pop('target_model', None)
        return_dict.pop('replay_memory', None)
        return return_dict

def load_agent(path):
    with open(f'{path}/agent.pkl', 'rb') as file:
        unpickler = dill.Unpickler(file)
        agent = unpickler.load()

    # Rebuild model architectures if necessary
    # Assuming ModelBuilder and other necessary classes are available
    # You might need to pass necessary parameters to reconstruct the models
    # Here we assume that the models can be reconstructed using the saved agent's parameters
    model_builder = ModelBuilder(
        units=agent.model.units,
        dropout=agent.model.dropout,
        nb_states=agent.nb_states,
        nb_actions=agent.nb_actions,
        l2_reg=None,
        window=agent.window,
        distributional=agent.distributional,
        nb_atoms=agent.nb_atoms,
        adversarial=agent.adversarial,
        noisy=agent.noisy
    )
    agent.model = model_builder.build_model(trainable=True)
    agent.model.load_state_dict(torch.load(f'{path}/model.pth'))
    agent.target_model = model_builder.build_model(trainable=False)
    agent.target_model.load_state_dict(torch.load(f'{path}/target_model.pth'))

    other_elements = {}
    other_paths = glob.glob(f'{path}/*pkl')
    other_paths.extend(glob.glob(f'{path}/*json'))
    for element_path in other_paths:
        name = os.path.split(element_path)[-1].replace(".pkl", "").replace(".json", "")
        if name != "agent":
            if ".pkl" in element_path:
                with open(element_path, 'rb') as file:
                    other_elements[name] = dill.load(file)
            elif ".json" in element_path:
                with open(element_path, 'r') as file:
                    other_elements[name] = json.load(file)

    return agent, other_elements
