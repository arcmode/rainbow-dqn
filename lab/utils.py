import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import sys
import gym_trading_env
import nest_asyncio
import dill
import pickle
import torch

from sklearn.preprocessing import robust_scale

from rainbow.agent import Rainbow

def add_features(df):
    df["feature_close"] = robust_scale(df["close"].pct_change())
    df["feature_open"] = robust_scale(df["open"]/df["close"])
    df["feature_high"] = robust_scale(df["high"]/df["close"])
    df["feature_low"] = robust_scale(df["low"]/df["close"])
    df["feature_volume"] = robust_scale(df["volume"] / df["volume"].rolling(7*24).max())
    df.dropna(inplace= True)
    return df

def reward_function(history):
    return 800*np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2]) #log (p_t / p_t-1 )

def max_drawdown(history):
    networth_array = history['portfolio_valuation']
    _max_networth = networth_array[0]
    _max_drawdown = 0
    for networth in networth_array:
        if networth > _max_networth:
            _max_networth = networth
        drawdown = ( networth - _max_networth ) / _max_networth
        if drawdown < _max_drawdown:
            _max_drawdown = drawdown
    return f"{_max_drawdown*100:5.2f}%"

def make_env(dir):
    print(f"Using  diectory: {dir}")
    dataset_dir = os.path.join(os.getcwd(), dir)
    print(f"Using dataset directory: {dataset_dir}")
    env = gym.make(
        "MultiDatasetTradingEnv",
        dataset_dir= dir,
        preprocess= add_features,
        windows= 15,
        positions = [ -1, -0.5, 0, 1, 2], # From -1 (=SHORT), to +1 (=LONG)
        initial_position = 0,
        trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (= 1h here)
        reward_function = reward_function,
        portfolio_initial_value = 1000, # here, in USDT

        verbose= 1,
    )
    env.unwrapped.add_metric('Position Changes', lambda history : f"{ 100*np.sum(np.diff(history['position']) != 0)/len(history['position']):5.2f}%" )
    env.unwrapped.add_metric('Max Drawdown', max_drawdown)
    return env

def dump_agent(agent):
    #agent.model = None
    #agent.target_model = None
    #agent.replay_memory = None

    with open("test.pkl", "wb") as file:
        dill.dump(agent, file)

def plot_agent(agent):
    batch_indexes, states, actions, rewards, states_prime, dones, importance_weights = agent.replay_memory.sample(
        256,
        agent.prioritized_replay_beta_function(agent.episode_count, agent.steps)
    )
    results = agent.model(states)

    action_colors=["blue", "orange","purple","red"]
    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(16,9), dpi=300)
    for action in range(4):
        for i in range(256):
            axes[action%2, action//2%2].plot(agent.zs, results[i, action, :], color = action_colors[action], alpha = 0.2)

def make_agent():
    return Rainbow(
        simultaneous_training_env = 5,

        #Distributional
        distributional= True,
        v_min= -200,
        v_max = 250,
        nb_atoms= 51,
        # Prioritized Replay
        prioritized_replay = False,
        prioritized_replay_alpha= 0.5,
        prioritized_replay_beta_function = lambda episode, step : min(1, 0.5 + 0.5*step/150_000),

        # General
        multi_steps = 3,
        nb_states = 7,
        nb_actions = 4,
        gamma = 0.99,
        replay_capacity = 1E8,
        tau = 2000,

        # Model
        window= 15,
        units = [16,16, 16],
        dropout= 0.2,
        adversarial= True,
        noisy= True,
        learning_rate = 3*2.5E-5,

        batch_size= 128,
        train_every = 10,
        epsilon_function = lambda episode, step : max(0.001, (1 - 5E-5)** step),
        name = "Rainbow",

        tensorboard = True,
    )

def train(agent, training_envs, validation_envs, steps=100_000, eval_every=5000):
    print("___________________________________________ TRAINING ___________________________________________")
    if 'obs' not in globals():
        global obs
        obs, info = training_envs.reset()
    for step in range(steps):
        actions = agent.e_greedy_pick_actions(obs)
        next_obs, rewards, dones, truncateds, infos = training_envs.step(actions)

        agent.store_replays(obs, actions, rewards, next_obs, dones, truncateds)
        agent.train()

        obs = next_obs

        # Perform evaluation every eval_every steps
        if step % eval_every == 0 and step > 0:
            evaluation(agent, validation_envs)

def evaluation(agent, validation_envs, validation_batch_size=256):
    print("___________________________________________ VALIDATION ___________________________________________")
    val_obs, info = validation_envs.reset()
    check = np.array([False for _ in range(val_obs.shape[0])])
    # Create lists to store validation experiences
    states_list = []
    actions_list = []
    rewards_list = []
    next_states_list = []
    dones_list = []

    while not np.all(check):
        with torch.no_grad():
            actions = agent.e_greedy_pick_actions(val_obs)
        next_obs, rewards, dones, truncateds, infos = validation_envs.step(actions)
        # Store experiences
        states_list.append(val_obs)
        actions_list.append(actions)
        rewards_list.append(rewards)
        next_states_list.append(next_obs)
        dones_list.append(dones)
        val_obs = next_obs
        check += dones + truncateds

        # When enough experiences are collected, compute validation loss
        if len(states_list) * len(states_list[0]) >= validation_batch_size:
            # Convert lists to arrays
            states_array = np.concatenate(states_list)
            actions_array = np.concatenate(actions_list)
            rewards_array = np.concatenate(rewards_list)
            next_states_array = np.concatenate(next_states_list)
            dones_array = np.concatenate(dones_list)

            # Now, sample validation_batch_size experiences
            idx = np.random.choice(len(states_array), validation_batch_size, replace=False)
            validation_loss = agent.validation_step(
                states_array[idx],
                actions_array[idx],
                rewards_array[idx],
                next_states_array[idx],
                dones_array[idx]
            )
            # Write validation loss to tensorboard
            if agent.tensorboard:
                agent.train_summary_writer.add_scalar('Validation Loss', validation_loss, agent.steps)
            # Break after computing validation loss
            break

    # If episodes end before collecting enough experiences
    if len(states_list) * len(states_list[0]) < validation_batch_size:
        # Convert lists to arrays
        states_array = np.concatenate(states_list)
        actions_array = np.concatenate(actions_list)
        rewards_array = np.concatenate(rewards_list)
        next_states_array = np.concatenate(next_states_list)
        dones_array = np.concatenate(dones_list)

        if len(states_array) > 0:
            validation_loss = agent.validation_step(
                states_array,
                actions_array,
                rewards_array,
                next_states_array,
                dones_array
            )
            if agent.tensorboard:
                agent.train_summary_writer.add_scalar('Validation Loss', validation_loss, agent.steps)
