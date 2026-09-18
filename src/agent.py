import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from src.network import DQNNetwork


class DQNAgent:
    """Deep Q-Network (DQN) Agent for continuous driving environments with discrete actions."""

    def __init__(
        self,
        state_size: int = 259,
        hidden_size: int = 512,
        action_size: int = 6,
        lr: float = 3e-4,
        gamma: float = 0.995,
        device=None,
    ):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Dual networks: Policy Q-network and Frozen Target network
        self.policy_net = DQNNetwork(state_size, hidden_size, action_size=action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, hidden_size, action_size=action_size).to(self.device)

        # Initialize target network with identical weights
        self.update_target_network()
        self.target_net.eval()

        # 2. Optimizer and Loss Function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy exploration strategy.
        
        Ensures input state tensor is placed onto self.device.
        """
        if random.random() < epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(torch.argmax(q_values).item())

    def learn(self, experiences) -> float:
        """Applies the Bellman optimality equation and backpropagates MSE loss."""
        states, actions, rewards, next_states, dones = experiences

        states = states.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        rewards = rewards.to(self.device, non_blocking=True)
        next_states = next_states.to(self.device, non_blocking=True)
        dones = dones.to(self.device, non_blocking=True)

        # 1. Current Q-values: Q(s, a)
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 2. Target Q-values using Target Network: y = r + (1 - d) * gamma * max(Q_target(s', a'))
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        # 3. Mean Squared Error Loss
        loss = self.loss_fn(current_q, target_q)

        # 4. Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self):
        """Hard update: copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath: str):
        """Saves policy network weights to disk."""
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath: str, map_location=None):
        """Loads policy network weights from disk."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Model checkpoint not found: '{filepath}'")
        if map_location is None:
            map_location = self.device
        state_dict = torch.load(filepath, map_location=map_location, weights_only=True)
        self.policy_net.load_state_dict(state_dict)
        self.update_target_network()