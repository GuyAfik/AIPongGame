"""
Layer 3 — AI Brain: Replay Buffer
Circular experience memory for DQN training.
Imports from Layer 0 (config) only.
"""

import random
from collections import deque
from typing import NamedTuple

import numpy as np
from src.utils.config import AI_CONFIG


class Experience(NamedTuple):
    """A single transition stored in the replay buffer."""
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class ReplayBuffer:
    """
    Fixed-capacity circular buffer of Experience tuples.

    Responsibilities:
    - Store (state, action, reward, next_state, done) transitions
    - Evict oldest entries when capacity is exceeded (FIFO via deque)
    - Sample a random mini-batch for DQN training
    """

    def __init__(self, capacity: int | None = None) -> None:
        cap = capacity if capacity is not None else AI_CONFIG.REPLAY_BUFFER_SIZE
        self._buffer: deque[Experience] = deque(maxlen=cap)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool,
    ) -> None:
        """Add one experience to the buffer. Oldest entry is evicted if full."""
        self._buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int | None = None) -> list[Experience]:
        """
        Sample a random mini-batch of experiences without replacement.

        Args:
            batch_size: number of experiences to sample.
                        Defaults to AIConfig.BATCH_SIZE.

        Returns:
            List of Experience namedtuples.

        Raises:
            ValueError: if buffer has fewer entries than batch_size.
        """
        size = batch_size if batch_size is not None else AI_CONFIG.BATCH_SIZE
        if len(self._buffer) < size:
            raise ValueError(
                f"Buffer has {len(self._buffer)} entries, "
                f"but batch_size={size} was requested."
            )
        return random.sample(list(self._buffer), size)

    def sample_arrays(
        self, batch_size: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a mini-batch and return pre-stacked numpy arrays.

        Returns:
            (states, actions, rewards, next_states, dones)
            Each is a numpy array with shape (batch_size, ...).
        """
        batch = self.sample(batch_size)
        states      = np.array([e.state      for e in batch], dtype=np.float32)
        actions     = np.array([e.action     for e in batch], dtype=np.int64)
        rewards     = np.array([e.reward     for e in batch], dtype=np.float32)
        next_states = np.array([e.next_state for e in batch], dtype=np.float32)
        dones       = np.array([e.done       for e in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def is_ready(self, batch_size: int | None = None) -> bool:
        """Return True if the buffer has enough entries to sample a batch."""
        size = batch_size if batch_size is not None else AI_CONFIG.BATCH_SIZE
        return len(self._buffer) >= size

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self._buffer)}, capacity={self._buffer.maxlen})"
