import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """Deep Q-Network: 3-layer MLP predicting Q-values for discrete driving actions.
    
    Architecture:
        Input (state_size=259) -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output (action_size=6)
    """

    def __init__(self, state_size: int = 259, hidden_size: int = 512, action_size: int = 6):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing Q(s, a) for all discrete actions."""
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)