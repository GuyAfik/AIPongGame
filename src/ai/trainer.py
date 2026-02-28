"""
Layer 3 — AI Brain: Trainer
Orchestrates the full training loop.
Owns PongEnvironment, DQNAgent, and TrainingVisualizer.
Imports from Layer 0, Layer 3 (environment, dqn_agent, visualizer).
"""

from __future__ import annotations

import pathlib
import sys
import time

from src.utils.config import AI_CONFIG, GAME_CONFIG
from src.ai.environment import PongEnvironment
from src.ai.dqn_agent import DQNAgent
from src.ai.visualizer import TrainingVisualizer


_MODEL_DIR = pathlib.Path("models")
_CHECKPOINT_PATH = _MODEL_DIR / "checkpoint.pth"
_FINAL_MODEL_PATH = _MODEL_DIR / "trained_model.pth"


class Trainer:
    """
    Orchestrates the DQN training loop.

    Responsibilities:
    - Run episodes: reset → select action → step → store → learn → repeat
    - Sync target network every TARGET_UPDATE_FREQ steps
    - Decay epsilon after each episode
    - Checkpoint model every 500 episodes
    - Feed data to TrainingVisualizer and trigger redraws
    - Save final model on completion
    """

    def __init__(
        self,
        env: PongEnvironment,
        agent: DQNAgent,
        visualizer: TrainingVisualizer | None = None,
        checkpoint_interval: int = 500,
    ) -> None:
        self._env = env
        self._agent = agent
        self._visualizer = visualizer
        self._checkpoint_interval = checkpoint_interval
        self._cfg = AI_CONFIG
        self._game_cfg = GAME_CONFIG

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, episodes: int | None = None) -> None:
        """
        Run the full training loop.

        Args:
            episodes: number of episodes to train for.
                      Defaults to AIConfig.EPISODES.
        """
        total_episodes = episodes if episodes is not None else self._cfg.EPISODES
        max_steps = self._cfg.MAX_STEPS_PER_EPISODE

        print(f"\n{'='*60}", flush=True)
        print(f"  DQN Training — {total_episodes} episodes", flush=True)
        print(f"  Device: {self._agent._device}", flush=True)
        print(f"{'='*60}\n", flush=True)

        start_time = time.time()

        for episode in range(1, total_episodes + 1):
            state = self._env.reset()
            episode_reward = 0.0
            steps = 0
            won = False

            for _ in range(max_steps):
                # Select action
                action = self._agent.select_action(state)

                # Step environment
                next_state, reward, done = self._env.step(action)

                # Store experience
                self._agent.store_experience(state, action, reward, next_state, done)

                # Learn
                self._agent.learn()

                # Sync target network
                if self._agent.steps_done % self._cfg.TARGET_UPDATE_FREQ == 0:
                    self._agent.sync_target_network()

                episode_reward += reward
                steps += 1
                state = next_state

                if done:
                    winner = self._env.game_state.get_winner()
                    won = winner == "right"
                    break

            # Post-episode updates
            self._agent.decay_epsilon()

            # Feed visualizer
            if self._visualizer is not None:
                self._visualizer.record(episode_reward, won, self._agent.epsilon, steps)
                if episode % self._visualizer.update_interval == 0:
                    self._visualizer.update()

            # Console logging
            if episode % 100 == 0 or episode == 1:
                elapsed = time.time() - start_time
                self._log_progress(episode, total_episodes, episode_reward, steps, elapsed)

            # Checkpoint
            if episode % self._checkpoint_interval == 0:
                self._agent.save(_CHECKPOINT_PATH)
                if self._visualizer is not None:
                    self._visualizer.save()
                print(f"  [Checkpoint saved at episode {episode}]", flush=True)

        # Final save
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._agent.save(_FINAL_MODEL_PATH)
        if self._visualizer is not None:
            self._visualizer.update()
            self._visualizer.save()

        elapsed = time.time() - start_time
        print(f"\n{'='*60}", flush=True)
        print(f"  Training complete in {elapsed:.1f}s", flush=True)
        print(f"  Model saved → {_FINAL_MODEL_PATH}", flush=True)
        print(f"{'='*60}\n", flush=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_progress(
        self,
        episode: int,
        total: int,
        reward: float,
        steps: int,
        elapsed: float,
    ) -> None:
        pct = episode / total * 100
        eps_str = f"{self._agent.epsilon:.3f}"
        buf_str = f"{len(self._agent.replay_buffer):>6}"
        print(
            f"  Ep {episode:>5}/{total}  ({pct:5.1f}%)  "
            f"reward={reward:>8.2f}  steps={steps:>5}  "
            f"ε={eps_str}  buf={buf_str}  t={elapsed:.0f}s",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Trainer(env={self._env!r}, agent={self._agent!r})"
        )
