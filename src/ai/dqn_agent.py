"""
Layer 3 — AI Brain: DQN Agent
Owns the policy network, target network, replay buffer, and optimizer.
Implements epsilon-greedy action selection and the DQN learning update.
Imports from Layer 0 (config) and Layer 3 (DQNetwork, ReplayBuffer).
"""

import random
import pathlib

import torch
import torch.nn as nn
import numpy as np

from src.utils.config import AI_CONFIG
from src.ai.neural_net import DQNetwork
from src.ai.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent.

    Responsibilities:
    - Select actions via epsilon-greedy policy
    - Store experiences in the replay buffer
    - Learn from sampled mini-batches using the Bellman equation
    - Periodically sync the target network
    - Save and load model weights
    """

    def __init__(self) -> None:
        self._cfg = AI_CONFIG

        # For this tiny network (6 inputs, batch=64), CPU is faster than MPS/CUDA
        # because GPU kernel launch overhead dominates at small batch sizes.
        self._device = torch.device("cpu")

        # Networks
        self.policy_net = DQNetwork().to(self._device)
        self.target_net = DQNetwork().to(self._device)
        self.sync_target_network()          # start with identical weights
        self.target_net.eval()              # target net is never trained directly

        # Optimizer
        self._optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self._cfg.LEARNING_RATE,
        )
        self._loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer()

        # Exploration state
        self.epsilon: float = self._cfg.EPSILON_START
        self.steps_done: int = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        state: list[float],
        epsilon_override: float | None = None,
    ) -> int:
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: normalized 6-float game state vector
            epsilon_override: if provided, use this epsilon instead of self.epsilon
                              (used for difficulty levels in play mode)

        Returns:
            action index: 0=UP, 1=DOWN, 2=STAY
        """
        eps = epsilon_override if epsilon_override is not None else self.epsilon

        if random.random() < eps:
            return random.randint(0, self._cfg.ACTION_SIZE - 1)

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self._device
        ).unsqueeze(0)  # shape: (1, STATE_SIZE)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)  # shape: (1, ACTION_SIZE)

        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def store_experience(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool,
    ) -> None:
        """Push one transition into the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self) -> float | None:
        """
        Sample a mini-batch and perform one gradient update.

        Returns:
            The scalar loss value, or None if the buffer is not ready yet.
        """
        if not self.replay_buffer.is_ready():
            return None

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample_arrays()
        )

        # Convert to tensors
        states_t      = torch.tensor(states,      device=self._device)
        actions_t     = torch.tensor(actions,     device=self._device)
        rewards_t     = torch.tensor(rewards,     device=self._device)
        next_states_t = torch.tensor(next_states, device=self._device)
        dones_t       = torch.tensor(dones,       device=self._device)

        # Current Q-values: Q(s, a) for the actions that were taken
        q_current = self.policy_net(states_t)                       # (B, A)
        q_taken   = q_current.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

        # Target Q-values: r + γ * max_a'[Q_target(s', a')] * (1 - done)
        with torch.no_grad():
            q_next_max = self.target_net(next_states_t).max(dim=1).values  # (B,)
            q_target   = rewards_t + self._cfg.GAMMA * q_next_max * (1.0 - dones_t)

        # Compute loss and backpropagate
        loss = self._loss_fn(q_taken, q_target)

        self._optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self._optimizer.step()

        self.steps_done += 1
        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Decay epsilon by one step (call once per episode)."""
        self.epsilon = max(
            self._cfg.EPSILON_END,
            self.epsilon * self._cfg.EPSILON_DECAY,
        )

    # ------------------------------------------------------------------
    # Target network sync
    # ------------------------------------------------------------------

    def sync_target_network(self) -> None:
        """Copy policy network weights to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Save policy network weights to disk."""
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "epsilon":    self.epsilon,
                "steps_done": self.steps_done,
            },
            path,
        )

    def load(self, path: str | pathlib.Path) -> None:
        """Load policy network weights from disk."""
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon    = checkpoint.get("epsilon",    self._cfg.EPSILON_END)
        self.steps_done = checkpoint.get("steps_done", 0)
        self.target_net.eval()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DQNAgent(epsilon={self.epsilon:.4f}, "
            f"steps={self.steps_done}, "
            f"buffer={len(self.replay_buffer)}, "
            f"device={self._device})"
        )
