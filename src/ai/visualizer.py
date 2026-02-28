"""
Layer 3 — AI Brain: Training Visualizer
Live matplotlib dashboard updated during training.
Imports from Layer 0 (config) only.
"""

from __future__ import annotations

import pathlib
import numpy as np
import matplotlib
# Select the best available interactive backend for the platform.
# Must be called before importing pyplot.
import sys
_backend_set = False
if sys.platform == "darwin":
    for _backend in ("macosx", "TkAgg", "Agg"):
        try:
            matplotlib.use(_backend)
            _backend_set = True
            break
        except Exception:
            continue
else:
    for _backend in ("TkAgg", "Qt5Agg", "Agg"):
        try:
            matplotlib.use(_backend)
            _backend_set = True
            break
        except Exception:
            continue

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.utils.config import AI_CONFIG


def _rolling_average(values: list[float], window: int) -> list[float]:
    """Compute a rolling average over a list of floats. Utility function."""
    if not values:
        return []
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(values[start : i + 1])))
    return result


class TrainingVisualizer:
    """
    Live 2×2 matplotlib dashboard shown during training.

    Panels:
        Top-left:     Episode reward (raw + rolling average)
        Top-right:    Win rate over last 100 episodes
        Bottom-left:  Epsilon decay curve
        Bottom-right: Steps per episode

    Usage:
        visualizer.record(episode_reward, won, epsilon, steps)
        if episode % visualizer.update_interval == 0:
            visualizer.update()
        visualizer.save("models/training_plot.png")
        visualizer.close()
    """

    def __init__(self, update_interval: int = 50) -> None:
        self.update_interval: int = update_interval

        # Tracking lists
        self._episode_rewards: list[float] = []
        self._wins: list[bool] = []
        self._epsilons: list[float] = []
        self._steps_per_ep: list[int] = []

        # Matplotlib state
        self._fig: plt.Figure | None = None
        self._axes: list[plt.Axes] = []
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Data recording
    # ------------------------------------------------------------------

    def record(
        self,
        episode_reward: float,
        won: bool,
        epsilon: float,
        steps: int,
    ) -> None:
        """Append one episode's data to all tracking lists."""
        self._episode_rewards.append(episode_reward)
        self._wins.append(won)
        self._epsilons.append(epsilon)
        self._steps_per_ep.append(steps)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Redraw all four panels with the latest data (non-blocking)."""
        if not self._initialized:
            self._init_figure()

        episodes = list(range(1, len(self._episode_rewards) + 1))
        if not episodes:
            return

        rolling_rewards = _rolling_average(self._episode_rewards, window=100)

        # Win rate: rolling 100-episode window
        win_rate: list[float] = []
        for i in range(len(self._wins)):
            start = max(0, i - 99)
            window_wins = self._wins[start : i + 1]
            win_rate.append(sum(window_wins) / len(window_wins) * 100.0)

        ax_reward, ax_winrate, ax_epsilon, ax_steps = self._axes

        # --- Top-left: Episode reward ---
        ax_reward.cla()
        ax_reward.plot(episodes, self._episode_rewards, alpha=0.3, color="#4a9eff", linewidth=0.8, label="Raw")
        ax_reward.plot(episodes, rolling_rewards, color="#4a9eff", linewidth=2.0, label="Avg(100)")
        ax_reward.set_title("Episode Reward", color="white")
        ax_reward.set_xlabel("Episode", color="#aaaaaa")
        ax_reward.set_ylabel("Reward", color="#aaaaaa")
        ax_reward.legend(facecolor="#2a2a2a", labelcolor="white", fontsize=8)
        ax_reward.axhline(0, color="#555555", linewidth=0.5)

        # --- Top-right: Win rate ---
        ax_winrate.cla()
        ax_winrate.plot(episodes, win_rate, color="#50e050", linewidth=2.0)
        ax_winrate.set_ylim(0, 100)
        ax_winrate.set_title("Win Rate (last 100 eps)", color="white")
        ax_winrate.set_xlabel("Episode", color="#aaaaaa")
        ax_winrate.set_ylabel("Win %", color="#aaaaaa")
        ax_winrate.axhline(50, color="#555555", linewidth=0.5, linestyle="--")

        # --- Bottom-left: Epsilon ---
        ax_epsilon.cla()
        ax_epsilon.plot(episodes, self._epsilons, color="#ffaa44", linewidth=2.0)
        ax_epsilon.set_ylim(0, 1.05)
        ax_epsilon.set_title("Epsilon (Exploration Rate)", color="white")
        ax_epsilon.set_xlabel("Episode", color="#aaaaaa")
        ax_epsilon.set_ylabel("ε", color="#aaaaaa")

        # --- Bottom-right: Steps per episode ---
        ax_steps.cla()
        ax_steps.plot(episodes, self._steps_per_ep, alpha=0.4, color="#cc66ff", linewidth=0.8)
        rolling_steps = _rolling_average(self._steps_per_ep, window=50)
        ax_steps.plot(episodes, rolling_steps, color="#cc66ff", linewidth=2.0)
        ax_steps.set_title("Steps per Episode", color="white")
        ax_steps.set_xlabel("Episode", color="#aaaaaa")
        ax_steps.set_ylabel("Steps", color="#aaaaaa")

        # Style all axes
        for ax in self._axes:
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="#aaaaaa")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")

        self._fig.canvas.draw()
        plt.pause(0.001)

    def save(self, path: str | pathlib.Path = "models/training_plot.png") -> None:
        """Save the current figure to disk as a PNG."""
        if self._fig is not None:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#111111")

    def close(self) -> None:
        """Close the matplotlib figure."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._initialized = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_figure(self) -> None:
        """Create the figure and 2×2 subplot grid."""
        plt.ion()  # interactive mode — non-blocking
        self._fig = plt.figure(figsize=(12, 7), facecolor="#111111")
        self._fig.suptitle("DQN Training Progress", color="white", fontsize=14, fontweight="bold")

        gs = gridspec.GridSpec(2, 2, figure=self._fig, hspace=0.45, wspace=0.35)
        self._axes = [
            self._fig.add_subplot(gs[0, 0]),  # reward
            self._fig.add_subplot(gs[0, 1]),  # win rate
            self._fig.add_subplot(gs[1, 0]),  # epsilon
            self._fig.add_subplot(gs[1, 1]),  # steps
        ]
        self._initialized = True

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TrainingVisualizer("
            f"episodes={len(self._episode_rewards)}, "
            f"update_interval={self.update_interval})"
        )
