"""
Module implementing the Rainbow Deep Q-Network (DQN) agent.

This module contains the Rainbow class, which integrates multiple DQN extensions,
including Double DQN, Dueling Networks, Prioritized Experience Replay, Multi-step
Targets, Distributional RL, and Noisy Nets, resulting in a more robust and efficient
reinforcement learning algorithm.
"""

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
    """
    Rainbow DQN Agent.

    Combines several improvements to the traditional DQN algorithm:
    - Double DQN
    - Dueling Networks
    - Prioritized Experience Replay
    - Multi-step Targets
    - Distributional RL
    - Noisy Nets

    Parameters:
        nb_states (int): Number of dimensions in the state space.
        nb_actions (int): Number of possible actions.
        gamma (float): Discount factor for future rewards.
        replay_capacity (int): Capacity of the replay buffer.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Size of batches sampled from the replay buffer.
        epsilon_function (callable): Function to compute epsilon for epsilon-greedy policy.
        window (int): Sequence length for RNN; if window > 1, recurrent layers are used.
        units (list of int): List specifying the number of units in each hidden layer.
        dropout (float): Dropout rate.
        adversarial (bool): Whether to use dueling network architecture.
        noisy (bool): Whether to use Noisy Nets for exploration.
        tau (int): Update frequency for the target network.
        multi_steps (int): Number of steps for multi-step targets.
        distributional (bool): Whether to use distributional RL.
        nb_atoms (int): Number of atoms in the value distribution (for distributional RL).
        v_min (float): Minimum value of the support for distributional RL.
        v_max (float): Maximum value of the support for distributional RL.
        prioritized_replay (bool): Whether to use prioritized experience replay.
        prioritized_replay_alpha (float): Alpha parameter for prioritized replay.
        prioritized_replay_beta_function (callable): Function to compute beta for importance-sampling weights.
        simultaneous_training_env (int): Number of environments for simultaneous training.
        train_every (int): Number of steps between each training update.
        name (str): Name of the agent (used for logging).
        scheduler (bool): Whether to use a learning rate scheduler.
        tensorboard (bool): Whether to log metrics to TensorBoard.
    """
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
                 tau = 250,
                 # Multi-Step replay
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
                 modify_lr_steps = 100_000,
                 # TensorBoard
                 tensorboard = False
                 ):
        # Agent name
        self.name = name

        # Environment parameters
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.epsilon_function = epsilon_function if not noisy else lambda episode, step: 0
        self.replay_capacity = replay_capacity

        # Training parameters
        self.learning_rate = learning_rate
        self.tau = tau  # Target network update frequency
        self.batch_size = batch_size
        self.train_every = train_every
        self.multi_steps = multi_steps
        self.simultaneous_training_env = simultaneous_training_env
        self.tensorboard = tensorboard
        self.scheduler = scheduler
        self.modify_lr_steps = modify_lr_steps

        # Model parameters
        self.recurrent = window > 1
        self.window = window
        self.units = units
        self.dropout = dropout
        self.adversarial = adversarial
        self.noisy = noisy

        # Distributional RL parameters
        self.distributional = distributional
        self.nb_atoms = nb_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Prioritized Experience Replay parameters
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta_function = prioritized_replay_beta_function

        # Initialize replay memory
        if self.recurrent:
            self.replay_memory = RNNReplayMemory(window=self.window, capacity=self.replay_capacity,
                                                 nb_states=self.nb_states, prioritized=prioritized_replay,
                                                 alpha=self.prioritized_replay_alpha)
        else:
            self.replay_memory = ReplayMemory(capacity=self.replay_capacity, nb_states=self.nb_states,
                                              prioritized=prioritized_replay, alpha=self.prioritized_replay_alpha)
        # Initialize multi-step buffers if applicable
        if self.multi_steps > 1:
            self.multi_steps_buffers = [MultiStepsBuffer(self.multi_steps, self.gamma) for _ in range(self.simultaneous_training_env)]

        # Build models
        model_builder = ModelBuilder(
            units=self.units,
            dropout=self.dropout,
            nb_states=self.nb_states,
            nb_actions=self.nb_actions,
            l2_reg=None,
            window=self.window,
            distributional=self.distributional,
            nb_atoms=self.nb_atoms,
            adversarial=self.adversarial,
            noisy=self.noisy
        )
        self.model = model_builder.build_model(trainable=True)
        self.target_model = model_builder.build_model(trainable=False)
        # Initialize target model parameters
        self.target_model.load_state_dict(self.model.state_dict())
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1E-8)

        # Initialize distributional RL parameters
        if self.distributional:
            self.delta_z = (self.v_max - self.v_min) / (self.nb_atoms - 1)
            self.zs = torch.linspace(self.v_min, self.v_max, self.nb_atoms)

        # Learning rate scheduler
        if self.scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.model_optimizer, gamma=0.99)

        # Initialize training history
        self.steps = 0
        self.losses = []
        self.episode_rewards = [[] for _ in range(self.simultaneous_training_env)]
        self.episode_count = [0 for _ in range(self.simultaneous_training_env)]
        self.episode_steps = [0 for _ in range(self.simultaneous_training_env)]
        self.start_time = datetime.datetime.now()

        # Initialize TensorBoard logging
        if self.tensorboard:
            self.log_dir = f"logs/{name}_{self.start_time.strftime('%Y_%m_%d-%H_%M_%S')}"
            self.train_summary_writer = SummaryWriter(self.log_dir)
            self.train_summary_writer.add_custom_scalars({
                'Metrics': {
                    'Loss': ['Multiline', ['Loss/Training', 'Loss/Validation']],
                    'Norm': ['Multiline', ['Norm/Grad', 'Norm/Momentum', 'Norm/Update']],
                    'Angle': ['Multiline', ['Angle/Grad-Momentum', 'Angle/Grad-Update']],
                    'Episode Rewards': ['Multiline', ["Episode Rewards/Env " + str(env_i) for env_i in range(self.simultaneous_training_env)]],
                    'Rewards per 1k steps': ['Multiline', ["Rewards per 1k steps/Env " + str(env_i) for env_i in range(self.simultaneous_training_env)]],
                    'Episode Lengths': ['Multiline', ["Episode Lengths/Env " + str(env_i) for env_i in range(self.simultaneous_training_env)]],
                    'Epsilon (prod 100)': ['Multiline', ["Epsilon/Env " + str(env_i) for env_i in range(self.simultaneous_training_env)]],
                    'Episode Mean Loss (last 10k)': ['Multiline', ["Episode Mean Loss/Env " + str(env_i) for env_i in range(self.simultaneous_training_env)]],

                }
            })

    def new_episode(self, i_env):
        """
        Initialize a new episode for a specific environment.

        Args:
            i_env (int): Index of the environment.
        """
        self.episode_count[i_env] += 1
        self.episode_steps[i_env] = 0
        self.episode_rewards[i_env] = []

    def store_replay(self, state, action, reward, next_state, done, truncated, i_env=0):
        """
        Store a transition in the replay memory.

        Handles multi-step transitions if applicable.

        Args:
            state (array): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (array): Next state.
            done (bool): Whether the episode ended.
            truncated (bool): Whether the episode was truncated.
            i_env (int): Index of the environment.
        """
        if self.multi_steps == 1:
            # Single-step transition
            self.replay_memory.store(state, action, reward, next_state, done)
        else:
            # Multi-step transition
            self.multi_steps_buffers[i_env].add(state, action, reward, next_state, done)
            if self.multi_steps_buffers[i_env].is_full():
                multi_step_transition = self.multi_steps_buffers[i_env].get_multi_step_replay()
                self.replay_memory.store(*multi_step_transition)

        # Store episode reward
        self.episode_rewards[i_env].append(reward)

        # Check if episode is done
        if done or truncated:
            if self.tensorboard:
                total_rewards = np.sum(self.episode_rewards[i_env])
                episode_length = self.episode_steps[i_env]
                self.train_summary_writer.add_scalar('Episode Rewards/Env ' + str(i_env), total_rewards, self.steps)
                self.train_summary_writer.add_scalar('Rewards per 1k steps/Env ' + str(i_env), 1000 * total_rewards / episode_length, self.steps)
                self.train_summary_writer.add_scalar('Episode Lengths/Env ' + str(i_env), episode_length, self.steps)
                self.train_summary_writer.add_scalar('Epsilon/Env ' + str(i_env), self.get_current_epsilon()*100, self.steps)
                self.train_summary_writer.add_scalar('Episode Mean Loss/Env ' + str(i_env), np.mean(self.losses[-10_000:]), self.steps)
            self.log(i_env)
            self.new_episode(i_env)

    def store_replays(self, states, actions, rewards, next_states, dones, truncateds):
        """
        Store transitions from multiple environments.

        Args:
            states (list): List of current states for each environment.
            actions (list): List of actions taken in each environment.
            rewards (list): List of rewards received in each environment.
            next_states (list): List of next states for each environment.
            dones (list): List of done flags for each environment.
            truncateds (list): List of truncated flags for each environment.
        """
        for i_env in range(len(actions)):
            done = dones[i_env]
            truncated = truncateds[i_env]
            self.store_replay(states[i_env], actions[i_env], rewards[i_env],
                              next_states[i_env], done, truncated, i_env=i_env)


    def train(self):
        """
        Perform a training step if conditions are met.

        Updates the target network periodically.
        Logs training metrics to TensorBoard if enabled.
        """
        self.steps += 1
        # Increment episode steps for each environment
        for i_env in range(self.simultaneous_training_env):
            self.episode_steps[i_env] += 1

        # Check if ready to train
        if self.replay_memory.size() < self.batch_size or self.get_current_epsilon() >= 1:
            return

        # Update target network periodically
        if self.steps % self.tau == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # Train every 'train_every' steps
        if self.steps % self.train_every == 0:
            # Sample a batch from replay memory
            batch_indexes, states, actions, rewards, states_prime, dones, importance_weights = self.replay_memory.sample(
                self.batch_size,
                self.prioritized_replay_beta_function(sum(self.episode_count), self.steps)
            )
            # Perform training step
            td_errors, metrics = self.train_step(states, actions, rewards, states_prime, dones, importance_weights)
            # Update priorities in replay memory
            self.replay_memory.update_priority(batch_indexes, td_errors)
            # Store loss
            self.losses.append(float(metrics['Loss/Training']))

            # Log metrics to TensorBoard
            if self.tensorboard:
                self.train_summary_writer.add_scalar('Loss/Training', metrics['Loss/Training'], self.steps)
                self.train_summary_writer.add_scalar('Norm/Grad', metrics['Norm/Grad'], self.steps)
                self.train_summary_writer.add_scalar('Norm/Momentum', metrics['Norm/Momentum'], self.steps)
                self.train_summary_writer.add_scalar('Norm/Update', metrics['Norm/Update'], self.steps)
                if not np.isnan(metrics['Angle/Grad-Momentum']) and not np.isnan(metrics['Angle/Grad-Update']):
                    self.train_summary_writer.add_scalar('Angle/Grad-Momentum', metrics['Angle/Grad-Momentum'], self.steps)
                    self.train_summary_writer.add_scalar('Angle/Grad-Update', metrics['Angle/Grad-Update'], self.steps)
                self.train_summary_writer.add_scalar('Learning Rate', metrics['Learning Rate'], self.steps)

    def log(self, i_env=0):
        """
        Print training progress and statistics.

        Args:
            i_env (int): Index of the environment.
        """
        text_print = f"\
↳ Env {i_env} : {self.episode_count[i_env]:03} : {self.steps:8d}   |   {self.format_time(datetime.datetime.now() - self.start_time)}   |   Epsilon : {self.get_current_epsilon()*100:4.2f}%   |   Mean Loss (last 10k) : {np.mean(self.losses[-10_000:]):0.4E}   |   Tot. Rewards : {np.sum(self.episode_rewards[i_env]):8.2f}   |   Rewards (/1000 steps) : {1000 * np.sum(self.episode_rewards[i_env]) / self.episode_steps[i_env]:8.2f}   |   Length : {self.episode_steps[i_env]:6.0f}"
        print(text_print)

    def get_current_epsilon(self, delta_episode=0, delta_steps=0):
        """
        Compute the current epsilon value for epsilon-greedy policy.

        Args:
            delta_episode (int): Optional increment to episode count.
            delta_steps (int): Optional increment to steps.
        Returns:
            float: Current epsilon value.
        """
        return self.epsilon_function(sum(self.episode_count) + delta_episode, self.steps + delta_steps)

    def e_greedy_pick_action(self, state):
        """
        Pick an action using epsilon-greedy policy for a single state.

        Args:
            state (array): Current state.
        Returns:
            int: Selected action.
        """
        epsilon = self.get_current_epsilon()
        if np.random.rand() < epsilon:
            return np.random.choice(self.nb_actions)
        with torch.no_grad():
            return self.pick_action(state)

    def e_greedy_pick_actions(self, states):
        """
        Pick actions using epsilon-greedy policy for multiple states.

        Args:
            states (array): Array of states.
        Returns:
            array: Array of selected actions.
        """
        epsilon = self.get_current_epsilon()
        if np.random.rand() < epsilon:
            return np.random.choice(self.nb_actions, size=self.simultaneous_training_env)
        else:
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                actions = self.pick_actions(states).detach().cpu().numpy()
            self.model.train()  # Set model back to training mode
            return actions

    def format_time(self, t: datetime.timedelta):
        """
        Format timedelta into a string.

        Args:
            t (datetime.timedelta): Time delta.
        Returns:
            str: Formatted time string.
        """
        h = t.total_seconds() // (60 * 60)
        m = (t.total_seconds() % (60 * 60)) // 60
        s = t.total_seconds() % 60
        return f"{h:02.0f}:{m:02.0f}:{s:02.0f}"

    def train_step(self, *args, **kwargs):
        """
        Perform a single training step.

        Delegates to either distributional or classic train step based on configuration.

        Returns:
            tuple: (loss value, td_errors, metrics)
        """
        if self.distributional:
            return self._distributional_train_step(*args, **kwargs)
        return self._classic_train_step(*args, **kwargs)

    def validation_step(self, states, actions, rewards_n, states_prime_n, dones_n):
        """
        Compute validation loss for a batch of experiences.

        Args:
            states (array): Batch of states.
            actions (array): Batch of actions.
            rewards_n (array): Batch of rewards.
            states_prime_n (array): Batch of next states.
            dones_n (array): Batch of done flags.

        Returns:
            float: Validation loss value.
        """
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            if self.distributional:
                td_errors = self._distributional_td_errors(states, actions, rewards_n, states_prime_n, dones_n)
                loss_value = td_errors.mean()
            else:
                td_errors = self._classic_td_errors(states, actions, rewards_n, states_prime_n, dones_n)
                loss_value = (td_errors.pow(2)).mean()
        self.model.train()  # Set model back to training mode
        return loss_value.item()

    def pick_action(self, *args, **kwargs):
        """
        Pick action for a single state.

        Delegates to either distributional or classic pick action method.

        Returns:
            int: Selected action.
        """
        if self.distributional:
            return int(self._distributional_pick_action(*args, **kwargs))
        return int(self._classic_pick_action(*args, **kwargs))

    def pick_actions(self, *args, **kwargs):
        """
        Pick actions for multiple states.

        Delegates to either distributional or classic pick actions method.

        Returns:
            Tensor: Tensor of selected actions.
        """
        if self.distributional:
            return self._distributional_pick_actions(*args, **kwargs)
        return self._classic_pick_actions(*args, **kwargs)

    def _classic_td_errors(self, states, actions, rewards_n, states_prime_n, dones_n):
        """
        Compute temporal difference errors for classic DQN.

        Args:
            states (array): Batch of states.
            actions (array): Batch of actions.
            rewards_n (array): Batch of rewards.
            states_prime_n (array): Batch of next states.
            dones_n (array): Batch of done flags.

        Returns:
            Tensor: Tensor of TD errors.
        """
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards_n = torch.tensor(rewards_n, dtype=torch.float32)
        states_prime_n = torch.tensor(states_prime_n, dtype=torch.float32)
        dones_n = torch.tensor(dones_n, dtype=torch.float32)

        # Compute target Q-values
        best_ap = torch.argmax(self.model(states_prime_n).detach(), dim=1)
        with torch.no_grad():
            max_q_sp_ap = self.target_model(states_prime_n).gather(1, best_ap.unsqueeze(1)).squeeze(1)
        q_a_target = rewards_n + (1 - dones_n) * (self.gamma ** self.multi_steps) * max_q_sp_ap

        # Compute current Q-values
        q_prediction = self.model(states)
        q_a_prediction = q_prediction.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute TD errors
        return torch.abs(q_a_target - q_a_prediction)

    # Classic DQN Core Functions
    def _classic_train_step(self, states, actions, rewards_n, states_prime_n, dones_n, weights):
        """
        Perform a training step for classic DQN.

        Args:
            states (array): Batch of states.
            actions (array): Batch of actions.
            rewards_n (array): Batch of rewards.
            states_prime_n (array): Batch of next states.
            dones_n (array): Batch of done flags.
            weights (array): Importance-sampling weights.

        Returns:
            tuple: (loss value, td_errors, metrics)
        """
        weights = torch.tensor(weights, dtype=torch.float32)
        td_errors = self._classic_td_errors(states, actions, rewards_n, states_prime_n, dones_n)
        loss_value = (td_errors.pow(2) * weights).mean()

        return self._optimize_with_metrics(loss_value, td_errors)

    def _optimize_with_metrics(self, loss_value, td_errors):
        """
        Optimize the model and compute training metrics.

        Args:
            loss_value (Tensor): Computed loss.
            td_errors (Tensor): TD errors.

        Returns:
            tuple: (td_errors, metrics)
        """
        self.model_optimizer.zero_grad()
        loss_value.backward()

        metrics = {}

        # Optimizer metrics
        grad_norm, grad_vector = self.compute_grad()
        metrics['Norm/Grad'] = grad_norm

        # Save parameters before the update
        parameters_before = [p.clone().detach() for p in self.model.parameters()]
        momentum_norm, momentum_vector = self.compute_momentum()
        metrics['Norm/Momentum'] = momentum_norm
        metrics['Angle/Grad-Momentum'] = self.compute_grad_momentum_angle(grad_vector, momentum_vector)

        # Update parameters
        # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.model_optimizer.step()

        # Adjust Learning Rate
        learning_rate = self.model_optimizer.param_groups[0]['lr']
        metrics['Learning Rate'] = learning_rate
        if self.scheduler and self.steps % self.modify_lr_steps == 0:
            self.lr_scheduler.step()

        # Save parameters after the update
        update_norm, update_vector = self.compute_update(parameters_before)
        metrics['Norm/Update'] = update_norm
        metrics['Angle/Grad-Update'] = self.compute_grad_update_angle(grad_vector, update_vector)

        # Save loss_value
        metrics['Loss/Training'] = loss_value.item()

        return td_errors.detach().cpu().numpy(), metrics

    def _classic_pick_actions(self, states):
        """
        Pick actions for multiple states using classic DQN.

        Args:
            states (array): Array of states.

        Returns:
            Tensor: Tensor of selected actions.
        """
        states = torch.tensor(states, dtype=torch.float32)
        return torch.argmax(self.model(states), dim=1)

    def _classic_pick_action(self, state):
        """
        Pick an action for a single state using classic DQN.

        Args:
            state (array): State.

        Returns:
            Tensor: Selected action.
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return torch.argmax(self.model(state), dim=1)

    # Distributional Core Functions
    def _distributional_pick_actions(self, states):
        """
        Pick actions for multiple states using distributional DQN.

        Args:
            states (array): Array of states.

        Returns:
            Tensor: Tensor of selected actions.
        """
        states = torch.tensor(states, dtype=torch.float32)
        return torch.argmax(self._distributional_predict_q_a(self.model, states), dim=1)

    def _distributional_pick_action(self, state):
        """
        Pick an action for a single state using distributional DQN.

        Args:
            state (array): State.

        Returns:
            Tensor: Selected action.
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return torch.argmax(self._distributional_predict_q_a(self.model, state), dim=1)

    def _distributional_predict_q_a(self, model, s):
        """
        Predict Q-values for all actions using distributional DQN.

        Args:
            model (nn.Module): The model to use for prediction.
            s (Tensor): States.

        Returns:
            Tensor: Q-values for all actions.
        """
        p_a = model(s)
        q_a = torch.sum(p_a * self.zs, dim=-1)
        return q_a

    def _distributional_td_errors(self, states, actions, rewards, states_prime, dones):
        """
        Compute temporal difference errors for distributional DQN.

        Args:
            states (array): Batch of states.
            actions (array): Batch of actions.
            rewards (array): Batch of rewards.
            states_prime (array): Batch of next states.
            dones (array): Batch of done flags.

        Returns:
            Tensor: Tensor of TD errors.
        """
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        states_prime = torch.tensor(states_prime, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        batch_size = states.size(0)

        # Compute target distributions
        best_a_sp = torch.argmax(self._distributional_predict_q_a(self.model, states_prime).detach(), dim=1)

        Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * (self.gamma ** self.multi_steps) * self.zs.unsqueeze(0)
        Tz = Tz.clamp(self.v_min, self.v_max)

        b_j = (Tz - self.v_min) / self.delta_z
        l = b_j.floor().long()
        u = b_j.ceil().long()

        with torch.no_grad():
            p_max_ap_sp = self.target_model(states_prime).gather(
                1, best_a_sp.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.nb_atoms)
            ).squeeze(1)

        m = torch.zeros(batch_size, self.nb_atoms)
        offset = torch.linspace(0, (batch_size - 1) * self.nb_atoms, batch_size).long().unsqueeze(1).expand(batch_size,
                                                                                                            self.nb_atoms)

        l_idx = l + offset
        u_idx = u + offset

        m.view(-1).index_add_(0, l_idx.view(-1), (p_max_ap_sp * (u - b_j)).view(-1))
        m.view(-1).index_add_(0, u_idx.view(-1), (p_max_ap_sp * (b_j - l)).view(-1))

        p_s_a = self.model(states).gather(
            1, actions.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.nb_atoms)
        ).squeeze(1)
        p_s_a = p_s_a.clamp(1e-6, 1 - 1e-6)

        td_errors = -torch.sum(m * torch.log(p_s_a), dim=1)

        return td_errors

    def _distributional_train_step(self, states, actions, rewards, states_prime, dones, weights):
        """
        Perform a training step for distributional DQN.

        Args:
            states (array): Batch of states.
            actions (array): Batch of actions.
            rewards (array): Batch of rewards.
            states_prime (array): Batch of next states.
            dones (array): Batch of done flags.
            weights (array): Importance-sampling weights.

        Returns:
            tuple: (td_errors, metrics)
        """
        weights = torch.tensor(weights, dtype=torch.float32)
        td_errors = self._distributional_td_errors(states, actions, rewards, states_prime, dones)
        td_errors_weighted = td_errors * weights
        loss_value = td_errors_weighted.mean()

        return self._optimize_with_metrics(loss_value, td_errors)

    # Compute gradient norm
    def compute_grad(self):
        """
        Compute the norm of the gradients and return gradient vector.

        Returns:
            tuple: (grad_norm, grad_vector)
        """
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
        """
        Compute the norm and vector of the momentum terms from the optimizer state.

        Returns:
            tuple: (momentum_norm, momentum_vector)
        """
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
        """
        Compute the angle between the gradient and momentum vectors.

        Args:
            grad_vector (Tensor): Flattened gradient vector.
            momentum_vector (Tensor): Flattened momentum vector.

        Returns:
            float: Angle in radians between gradient and momentum.
        """
        if momentum_vector.numel() == 0 or momentum_vector.norm(2).item() == 0.0:
            # Momentum vector is zero; angle is undefined
            return np.nan
        dot_product = torch.dot(grad_vector, momentum_vector)
        return torch.acos(
            dot_product / (grad_vector.norm(2) * momentum_vector.norm(2) + 1e-8)
        ).item()

    # Compute update norm
    def compute_update(self, parameters_before):
        """
        Compute the norm and vector of the parameter updates.

        Args:
            parameters_before (list): List of parameters before the update.

        Returns:
            tuple: (update_norm, update_vector)
        """
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
        """
        Compute the angle between the gradient and update vectors.

        Args:
            grad_vector (Tensor): Flattened gradient vector.
            update_vector (Tensor): Flattened update vector.

        Returns:
            float: Angle in radians between gradient and update.
        """
        if update_vector.numel() == 0 or update_vector.norm(2).item() == 0.0:
            # Update vector is zero; angle is undefined
            return np.nan
        dot_product = torch.dot(grad_vector, update_vector)
        return torch.acos(
            dot_product / (grad_vector.norm(2) * update_vector.norm(2) + 1e-8)
        ).item()

    def save(self, path, **kwargs):
        """
        Save the agent's state and parameters to a specified path.

        Args:
            path (str): Path to save the agent.
            **kwargs: Additional elements to save.
        """
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
        """
        Customize pickling behavior to exclude certain attributes.

        Returns:
            dict: State dictionary without excluded attributes.
        """
        print("Saving agent ...")
        return_dict = self.__dict__.copy()
        return_dict.pop('model', None)
        return_dict.pop('target_model', None)
        return_dict.pop('replay_memory', None)
        return return_dict

def load_agent(path):
    """
    Load an agent from a specified path.

    Args:
        path (str): Path to load the agent from.

    Returns:
        tuple: (agent, other_elements)
    """
    with open(f'{path}/agent.pkl', 'rb') as file:
        unpickler = dill.Unpickler(file)
        agent = unpickler.load()

    # Rebuild model architectures if necessary
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

    # Load additional elements
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
