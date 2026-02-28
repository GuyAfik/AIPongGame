"""
Layer 3 — AI Brain: Neural Network
PyTorch DQN policy network.
Imports from Layer 0 (config) only.
"""

import torch
import torch.nn as nn
from src.utils.config import AI_CONFIG


class DQNetwork(nn.Module):
    """
    Deep Q-Network: maps a game state vector to Q-values for each action.

    Architecture:
        Input  (6)  → Linear → ReLU
        Hidden (128) → Linear → ReLU
        Hidden (128) → Linear
        Output (3)  — raw Q-values, no activation

    The three output neurons correspond to actions:
        0 = move UP
        1 = move DOWN
        2 = STAY
    """

    def __init__(self) -> None:
        super().__init__()
        cfg = AI_CONFIG

        self.network = nn.Sequential(
            nn.Linear(cfg.STATE_SIZE, cfg.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(cfg.HIDDEN_SIZE, cfg.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(cfg.HIDDEN_SIZE, cfg.ACTION_SIZE),
        )

        # Initialize weights with Xavier uniform for stable early training
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: state tensor of shape (batch_size, STATE_SIZE) or (STATE_SIZE,)

        Returns:
            Q-value tensor of shape (batch_size, ACTION_SIZE) or (ACTION_SIZE,)
        """
        return self.network(x)
