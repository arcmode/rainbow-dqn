# Rainbow-RL-Agent
Modified version of Reinforcement Learning **Rainbow Agent** with Pytorch from paper "Rainbow: Combining Improvements in Deep Reinforcement Learning".

Based on original implementation in Tensorflow by [Clement Perroud](https://github.com/ClementPerroud/Rainbow-Agent) which includes support for Recurrent Neural Nets and Multi Parallelized Environments.

The Rainbow Agent is a DQN agent with strong improvments:
- **DoubleQ-learning** : Adding a Target Network that is used in the loss function and upgrade once every `tau` steps. See paper [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- **Distributional RL** : Approximating the probability distributions of the Q-values instead of the Q-values themself. See paper : [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
- **Prioritizedreplay** : Sampling method that prioritize experiences with big *Temporal Difference(TD) errors* (~loss) at the beginning of a training. See paper : [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- **Dueling Networks**: Divide neural net stream into two branches, an action stream and a value stream. Both of them combined formed the Q-action values. See paper : [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1509.06461)
- **Multi-step learning** : Making Temporal Difference bigger than classic DQN (where TD = 1). See paper [Multi-step Reinforcement Learning: A Unifying Algorithm](https://arxiv.org/abs/1703.01327)
- **NoisyNets** : Replace classic epsilon-greedy exploration/exploitation with noise in the Neural Net. [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)

## How to use?

### Install
TODO

### Import
```python
from rainbow.agent import Rainbow
```

### Usage
```python
agent = Rainbow(
    simultaneous_training_env = 5,

    #Distributional
    distributional= True,
    v_min= -200,
    v_max = 250,
    nb_atoms= 51,

    # Prioritized Replay
    prioritized_replay = True,
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
    learning_rate = 3*2.5E-4,

    batch_size= 128,
    train_every = 10,
    # epsilon_function = lambda episode, step : max(0.001, (1 - 5E-5)** step), # Useless if noisy == True
    name = "Rainbow",
)
```

#### Training

##### Use Optimizer Metrics to Diagnose Training Behavior

By tracking these statistics, you can gain insights into the training dynamics:

- **Gradient Norm**: Indicates how large the gradients are. Small gradients may suggest that the model is near a minimum or stuck in a flat region.
- **Update Norm**: Shows how much the parameters are being updated. Small updates may indicate convergence or slow progress.
- **Momentum Norm**: Reflects the accumulated gradient information used by the optimizer to accelerate convergence.
- **Angle Between Gradient and Momentum**: Provides insight into whether the momentum is aligned with the current gradient. A small angle suggests the momentum is reinforcing the current gradient direction.
- **Angle Between Gradient and Update**: Helps understand how the optimizer is combining the gradient and momentum to update the parameters.

**Interpretation:**

- **Convergence to a Local Minimum**:
  - **Gradient Norm**: Small and decreasing.
  - **Update Norm**: Small and decreasing.
  - **Momentum Norm**: Decreasing.
  - **Angles**: May fluctuate as the optimizer fine-tunes the parameters.

- **Stuck in a Flat Region**:
  - **Gradient Norm**: Small but not decreasing.
  - **Update Norm**: Small but possibly larger than gradient norm.
  - **Momentum Norm**: Remaining relatively constant or not decreasing as expected.
  - **Angles**: Large angles might indicate that momentum is not helping to escape the flat region.

**Actionable Steps:**

- **If Converged to a Local Minimum**:
  - Training may be complete.
  - Consider evaluating the model's performance.

- **If Stuck in a Flat Region**:
  - **Increase the Learning Rate**: Helps the optimizer make larger updates to escape the flat region.
  - **Adjust Learning Rate Schedule**: Implement a dynamic learning rate scheduler that increases the learning rate when progress stalls.
  - **Modify the Optimizer Parameters**: Tweak optimizer parameters like `eps` in Adam.
  - **Consider Different Optimizers**: Experiment with optimizers like RMSprop or SGD with momentum.

### Example of Dynamic Learning Rate Adjustment

Implement a learning rate scheduler that increases the learning rate when the gradient norm falls below a threshold:

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

# In your agent's __init__ function
self.lr_scheduler = ReduceLROnPlateau(self.model_optimizer, mode='min', factor=1.1, patience=1000)

# In your train_step function, after updating the optimizer
self.lr_scheduler.step(metrics['grad_norm'])
```

**Note:** Adjust the `factor`, `patience`, and other parameters of `ReduceLROnPlateau` to suit your needs.

### Conclusion

By integrating the tracking of optimizer statistics into your training loop and logging them with TensorBoard, you can monitor the training process more comprehensively. This allows you to make informed decisions about adjusting hyperparameters like the learning rate to address issues like slow convergence in flat regions of the loss landscape.

**References:**

- [Understanding Adam Optimizer](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c)
- [PyTorch Optimizer Documentation](https://pytorch.org/docs/stable/optim.html)
- [TensorBoard for PyTorch](https://pytorch.org/docs/stable/tensorboard.html)
