"""
Layer 3 — AI Brain: PongEnvironment
Gym-like wrapper around GameState. Applies DifficultyConfig constraints.
Boundary between the game engine (Layer 2) and the AI brain (Layer 3).
Imports from Layer 0, Layer 2 (GameState, Renderer).
"""

from __future__ import annotations

from src.utils.config import (
    GAME_CONFIG,
    AI_CONFIG,
    Difficulty,
    DifficultyConfig,
    DIFFICULTY_PRESETS,
)
from src.game.game_state import GameState
from src.game.renderer import Renderer


class PongEnvironment:
    """
    Gym-like interface wrapping GameState.

    Responsibilities:
    - Expose reset() → initial state vector
    - Expose step(action) → (next_state, reward, done)
    - Apply DifficultyConfig constraints to the AI paddle
    - Optionally render each frame (play mode) or run headless (train mode)
    - Provide a rule-based bot action for the left (opponent) paddle
    """

    def __init__(
        self,
        headless: bool = True,
        difficulty: Difficulty = Difficulty.HARD,
    ) -> None:
        """
        Args:
            headless: if True, no pygame window is created (training mode).
            difficulty: controls AI speed, reaction delay, epsilon, ball speed.
        """
        self._headless = headless
        self._difficulty = difficulty
        self._diff_cfg: DifficultyConfig = DIFFICULTY_PRESETS[difficulty]
        self._game_cfg = GAME_CONFIG
        self._ai_cfg = AI_CONFIG

        # Build GameState with difficulty-adjusted speeds and speed-up timer
        self._game_state = GameState(
            ai_speed_multiplier=self._diff_cfg.paddle_speed_multiplier,
            ball_speed_multiplier=self._diff_cfg.ball_speed_multiplier,
            speed_up_interval_seconds=self._diff_cfg.speed_up_interval_seconds,
            speed_up_amount=self._diff_cfg.speed_up_amount,
        )

        # Renderer (only in play mode)
        self._renderer: Renderer | None = None
        if not headless:
            self._renderer = Renderer()
            self._renderer.init()

        # Difficulty: reaction delay state
        self._skip_frames: int = self._diff_cfg.action_skip_frames
        self._last_ai_action: int = 2   # default: STAY
        self._frame_counter: int = 0

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    def reset(self) -> list[float]:
        """
        Reset the game to its initial state.

        Returns:
            Initial 6-float normalized state vector.
        """
        self._game_state.reset()
        self._last_ai_action = 2
        self._frame_counter = 0
        return self._game_state.get_ai_state()

    def step(self, ai_action: int) -> tuple[list[float], float, bool]:
        """
        Advance the game by one frame.

        Applies reaction delay: if action_skip_frames > 0, the AI reuses
        its last action for N frames before accepting a new one.

        Args:
            ai_action: action chosen by the DQN agent (0=UP, 1=DOWN, 2=STAY)

        Returns:
            (next_state, reward, done)
        """
        # Reaction delay: only update action every (skip_frames + 1) frames
        if self._skip_frames > 0:
            if self._frame_counter % (self._skip_frames + 1) == 0:
                self._last_ai_action = ai_action
            effective_action = self._last_ai_action
        else:
            effective_action = ai_action

        self._frame_counter += 1

        # Rule-based bot for the left (opponent) paddle
        bot_action = self._bot_action()

        reward, done = self._game_state.step(bot_action, effective_action)
        next_state = self._game_state.get_ai_state()

        return next_state, reward, done

    def render(self) -> None:
        """Draw the current frame (no-op in headless mode)."""
        if self._renderer is not None:
            self._renderer.draw(self._game_state)
            self._renderer.tick()

    def close(self) -> None:
        """Clean up renderer if present."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def game_state(self) -> GameState:
        return self._game_state

    @property
    def renderer(self) -> Renderer | None:
        return self._renderer

    @property
    def difficulty_config(self) -> DifficultyConfig:
        return self._diff_cfg

    # ------------------------------------------------------------------
    # Rule-based opponent bot (left paddle)
    # ------------------------------------------------------------------

    def _bot_action(self) -> int:
        """
        Simple rule-based bot: always move toward the ball's y position.

        Returns:
            0=UP, 1=DOWN, 2=STAY
        """
        ball_y = self._game_state.ball.y
        paddle_center = self._game_state.paddle_left.get_center_y()
        dead_zone = 5.0  # pixels — prevents jitter at center

        if ball_y < paddle_center - dead_zone:
            return 0   # UP
        elif ball_y > paddle_center + dead_zone:
            return 1   # DOWN
        else:
            return 2   # STAY

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PongEnvironment("
            f"headless={self._headless}, "
            f"difficulty={self._difficulty.value!r})"
        )
