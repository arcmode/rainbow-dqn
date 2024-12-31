"""
Utilities for setting up and training a Rainbow DQN agent in a trading environment.

This module includes functions for feature preprocessing, environment creation,
agent instantiation, training loops, and evaluation procedures.
"""

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
    """
    Preprocess the dataframe by adding custom features.

    Parameters:
    df (pd.DataFrame): The original dataframe containing market data.

    Returns:
    pd.DataFrame: The dataframe with additional features.
    """
    df["feature_close"] = robust_scale(df["close"].pct_change())
    df["feature_open"] = robust_scale(df["open"]/df["close"])
    df["feature_high"] = robust_scale(df["high"]/df["close"])
    df["feature_low"] = robust_scale(df["low"]/df["close"])
    df["feature_volume"] = robust_scale(df["volume"] / df["volume"].rolling(7*24).max())
    df.dropna(inplace= True)
    return df

def reward_function(history):
    """
    Custom reward function based on the logarithmic return of the portfolio valuation.

    Parameters:
    history (dict): A history dictionary containing 'portfolio_valuation'.

    Returns:
    float: The computed reward.
    """
    return 800*np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2]) # log(p_t / p_{t-1})

def max_drawdown(history):
    """
    Compute the maximum drawdown of the portfolio.

    Parameters:
    history (dict): A history dictionary containing 'portfolio_valuation'.

    Returns:
    str: The maximum drawdown as a formatted string percentage.
    """
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
    """
    Create a trading environment using the specified directory for datasets.

    Parameters:
    dir (str): The directory path where dataset files are located.

    Returns:
    gym.Env: The instantiated trading environment.
    """
    print(f"Using  directory: {dir}")
    dataset_dir = os.path.join(os.getcwd(), dir)
    print(f"Using dataset directory: {dataset_dir}")
    env = gym.make(
        "MultiDatasetTradingEnv",
        dataset_dir= dir,
        preprocess= add_features,
        windows= 15,
        positions = [ -1, -0.5, 0, 1, 2],  # From -1 (=SHORT), to +1 (=LONG)
        initial_position = 0,
        trading_fees = 0.01/100,           # 0.01% per stock buy/sell (Binance fees)
        borrow_interest_rate= 0.0003/100,  # 0.0003% per timestep (= 1h here)
        reward_function = reward_function,
        portfolio_initial_value = 1000,    # Initial portfolio value in USDT
        verbose= 1,
    )
    # Add custom metrics to the environment for evaluation
    env.unwrapped.add_metric('Position Changes', lambda history : f"{ 100*np.sum(np.diff(history['position']) != 0)/len(history['position']):5.2f}%" )
    env.unwrapped.add_metric('Max Drawdown', max_drawdown)
    return env

def dump_agent(agent):
    """
    Serialize and save the agent to a file.

    Parameters:
    agent (Rainbow): The agent instance to be saved.
    """
    # agent.model = None
    # agent.target_model = None
    # agent.replay_memory = None

    with open("test.pkl", "wb") as file:
        dill.dump(agent, file)

def plot_agent(agent):
    """
    Plot the agent's learned Q-value distributions for analysis.

    Parameters:
    agent (Rainbow): The agent instance from which to sample and plot distributions.
    """
    batch_indexes, states, actions, rewards, states_prime, dones, importance_weights = agent.replay_memory.sample(
        256,
        agent.prioritized_replay_beta_function(agent.episode_count, agent.steps)
    )
    results = agent.model(states)

    action_colors = ["blue", "orange", "purple", "red"]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,9), dpi=300)
    for action in range(4):
        for i in range(256):
            axes[action%2, action//2%2].plot(agent.zs, results[i, action, :], color=action_colors[action], alpha=0.2)

def make_agent():
    """
    Instantiate a Rainbow DQN agent with specific hyperparameters.

    Returns:
    Rainbow: The initialized agent instance.
    """
    return Rainbow(
        simultaneous_training_env = 5,

        # Distributional parameters
        distributional= True,
        v_min= -200,
        v_max = 250,
        nb_atoms= 51,

        # Prioritized Replay parameters
        prioritized_replay = False,
        prioritized_replay_alpha= 0.5,
        prioritized_replay_beta_function = lambda episode, step : min(1, 0.5 + 0.5*step/150_000),

        # General parameters
        multi_steps = 3,
        nb_states = 7,
        nb_actions = 4,
        gamma = 0.99,
        replay_capacity = 1E8,
        tau = 2000,

        # Model architecture parameters
        window= 15,
        units = [16,16,16],
        dropout= 0.2,
        adversarial= True,
        noisy= True,
        learning_rate = 2*2.5E-5,
        scheduler=True,

        batch_size= 192,
        train_every = 10,
        epsilon_function = lambda episode, step : max(0.001, (1 - 5E-5)** step),
        name = "Rainbow",

        tensorboard = True,
    )

def train(agent, training_envs, validation_envs, steps=100_000, eval_every=5000):
    """
    Train the agent within the training environments for a specified number of steps.
    Periodically evaluate the agent using validation environments.

    Parameters:
    agent (Rainbow): The agent to be trained.
    training_envs (gym.Env): The training environment(s).
    validation_envs (gym.Env): The validation environment(s).
    steps (int): Total number of training steps.
    eval_every (int): Frequency of evaluation (in steps).
    """
    print("___________________________________________ TRAINING ___________________________________________")
    if 'obs' not in globals():
        global obs
        obs, info = training_envs.reset()
    for step in range(steps):
        # Agent selects actions based on current observations
        actions = agent.e_greedy_pick_actions(obs)
        # Environment responds to actions
        next_obs, rewards, dones, truncateds, infos = training_envs.step(actions)

        # Store experiences and train the agent
        agent.store_replays(obs, actions, rewards, next_obs, dones, truncateds)
        agent.train()

        obs = next_obs

        # Perform evaluation every 'eval_every' steps
        if step % eval_every == 0 and step > 0:
            evaluation(agent, validation_envs)

def evaluation(agent, validation_envs, validation_batch_size=256):
    """
    Evaluate the agent's performance on validation environments and compute validation loss.

    Parameters:
    agent (Rainbow): The agent to be evaluated.
    validation_envs (gym.Env): The validation environments.
    validation_batch_size (int): The number of experiences to use for validation loss computation.
    """
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
            # Agent selects actions without gradient tracking
            actions = agent.e_greedy_pick_actions(val_obs)
        # Environment responds to actions
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
        total_samples = sum(len(s) for s in states_list)
        if total_samples >= validation_batch_size:
            # Convert lists to arrays
            states_array = np.concatenate(states_list)
            actions_array = np.concatenate(actions_list)
            rewards_array = np.concatenate(rewards_list)
            next_states_array = np.concatenate(next_states_list)
            dones_array = np.concatenate(dones_list)

            # Sample a batch for validation
            idx = np.random.choice(len(states_array), validation_batch_size, replace=False)
            validation_loss = agent.validation_step(
                states_array[idx],
                actions_array[idx],
                rewards_array[idx],
                next_states_array[idx],
                dones_array[idx]
            )
            # Write validation loss to TensorBoard
            if agent.tensorboard:
                agent.train_summary_writer.add_scalar('Loss/Validation', validation_loss, agent.steps)
            # Break after computing validation loss
            break

    # If episodes end before collecting enough experiences
    if sum(len(s) for s in states_list) < validation_batch_size:
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
                agent.train_summary_writer.add_scalar('Loss/Validation', validation_loss, agent.steps)
