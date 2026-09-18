import random
from collections import deque
import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity experience replay buffer for off-policy reinforcement learning."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action: int, reward: float, next_state, done: bool):
        """Appends a transition tuple (s, a, r, s', d) to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Samples a mini-batch of transitions uniformly at random and converts to PyTorch tensors."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.from_numpy(np.asarray(states, dtype=np.float32)),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.from_numpy(np.asarray(next_states, dtype=np.float32)),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)