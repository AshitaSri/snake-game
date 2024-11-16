# **Deep Q-Learning with PyTorch**

This project demonstrates the implementation of **Deep Q-Learning (DQN)**, a popular reinforcement learning algorithm, using **PyTorch**. The agent learns to maximize rewards in a simulated environment using **experience replay** and **target networks** to stabilize training.
https://github.com/user-attachments/assets/5815daba-c844-4f57-bb04-1e95489c2b90

## **Table of Contents**

- [Introduction](#introduction)
- [Key Concepts](#key-concepts)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Details](#algorithm-details)
- [Hyperparameters](#hyperparameters)
- [Results](#results)
- [References](#references)

## **Introduction**

Deep Q-Learning is an advanced version of the traditional Q-Learning algorithm that leverages a neural network to approximate the Q-value function. This approach enables the agent to handle environments with large state and action spaces. The goal is to train the agent to take optimal actions to maximize cumulative rewards over time.

## **Key Concepts**

### 1. **Reinforcement Learning (RL)**
   - A learning paradigm where an agent interacts with an environment to learn a policy that maximizes the total expected reward.

### 2. **Q-Learning**
   - An off-policy algorithm where the agent learns the optimal action-value function using the Bellman equation.

### 3. **Deep Q-Network (DQN)**
   - A neural network-based approach to approximate the Q-values, allowing the algorithm to handle complex, high-dimensional environments.

### 4. **Experience Replay**
   - A technique where the agent stores experiences in a buffer and samples from it to break the correlation between consecutive experiences.

### 5. **Target Network**
   - A separate neural network used to generate stable Q-value targets, reducing the risk of oscillations in training.

## **Project Structure**

```
├── model/
│   └── model.pth                  # Saved model weights
├── main.py                        # Main file to train and test the agent
├── agent.py                       # Deep Q-Learning agent implementation
├── model.py                       # Neural network architecture
├── trainer.py                     # Training loop and Q-learning updates
├── environment.py                 # Environment setup (optional)
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

## **Installation**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/deep-q-learning-pytorch.git
   cd deep-q-learning-pytorch
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Dependencies include**:
   - `torch` - PyTorch for building and training neural networks
   - `numpy` - Numerical computations
   - `matplotlib` (optional) - For visualizing results

## **Usage**

### **Training the Agent**
   Run the training script to train the DQN agent:
   ```bash
   python main.py
   ```

   You can modify the training parameters in `main.py` to change the learning rate, discount factor, epsilon decay, etc.

### **Testing the Agent**
   After training, the agent can be tested:
   ```bash
   python main.py --test
   ```

   This will load the saved model weights and test the agent's performance in the environment.

## **Algorithm Details**

### **Deep Q-Learning Update Rule**

The Q-value update is based on the Bellman equation:

\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q'(s', a') - Q(s, a) \right]
\]

- **State (`s`)**: Current state of the environment.
- **Action (`a`)**: Action taken by the agent.
- **Reward (`r`)**: Reward received after taking action `a`.
- **Next State (`s'`)**: State after taking action `a`.
- **Discount Factor (`γ`)**: Determines the importance of future rewards.
- **Learning Rate (`α`)**: Controls the step size during parameter updates.
- **Target Network (`Q'`)**: A fixed Q-value network used for stable updates.

### **Experience Replay**
   - **Replay Buffer**: Stores past experiences as tuples (`state`, `action`, `reward`, `next_state`, `done`).
   - **Sampling**: Randomly samples mini-batches from the buffer to update the Q-network.

### **Target Network**
   - Updated periodically to provide stable Q-value targets.

## **Hyperparameters**

The key hyperparameters used in training the DQN agent:

| Hyperparameter   | Description                           | Value (Default) |
|------------------|---------------------------------------|-----------------|
| `learning_rate`  | Step size for gradient descent        | 0.001           |
| `gamma`          | Discount factor for future rewards    | 0.99            |
| `epsilon_start`  | Initial exploration rate              | 1.0             |
| `epsilon_min`    | Minimum exploration rate              | 0.01            |
| `epsilon_decay`  | Decay rate of exploration             | 0.995           |
| `batch_size`     | Number of samples per training batch  | 64              |
| `buffer_size`    | Size of the replay buffer             | 10,000          |
| `target_update`  | Steps before updating target network  | 1000            |

## **Results**

The agent learns to make optimal decisions through training, gradually increasing its cumulative reward. You can visualize the training progress using **Matplotlib**:

```python
import matplotlib.pyplot as plt

# Plot training rewards
plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.title('DQN Training Progress')
plt.show()
```

## **References**

- **Deep Q-Learning Paper**: ["Playing Atari with Deep Reinforcement Learning"](https://arxiv.org/abs/1312.5602) by Mnih et al.
- **Markov Decision Processes**: Sutton & Barto's book, "Reinforcement Learning: An Introduction."
- **PyTorch Documentation**: [PyTorch Official Docs](https://pytorch.org/docs/stable/index.html)
