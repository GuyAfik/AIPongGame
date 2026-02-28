"""
Layer 2 — Game Engine: GameState
Central state manager. Owns all game entities and orchestrates each step.
Imports from Layer 0 (config) and Layer 1 (Ball, Paddle) and Layer 2 (Physics).
"""

import time
from src.utils.config import GAME_CONFIG, REWARD_CONFIG
from src.game.ball import Ball
from src.game.paddle import Paddle
from src.game.physics import Physics


class GameState:
    """
    Owns the ball, both paddles, and the physics engine.

    Responsibilities:
    - Reset the full game state
    - Advance the game by one step given two actions
    - Compute and return rewards for the AI paddle (right)
    - Expose a 6-float normalized state vector for the AI
    - Track scores and the done flag
    - Trigger timed ball speed-ups based on real-world elapsed time
    """

    def __init__(
        self,
        ai_speed_multiplier: float = 1.0,
        ball_speed_multiplier: float = 1.0,
        speed_up_interval_seconds: float = 0.0,
        speed_up_amount: float = 1.0,
    ) -> None:
        """
        Args:
            ai_speed_multiplier:       scales AI paddle speed
            ball_speed_multiplier:     scales initial ball speed
            speed_up_interval_seconds: every N real seconds the ball gets faster (0 = disabled)
            speed_up_amount:           px/frame added to ball speed each interval
        """
        self._cfg = GAME_CONFIG
        self._reward_cfg = REWARD_CONFIG
        self._ball_speed_multiplier = ball_speed_multiplier
        self._speed_up_interval = speed_up_interval_seconds
        self._speed_up_amount = speed_up_amount

        self.ball = Ball()
        self.paddle_left = Paddle("left")                              # human
        self.paddle_right = Paddle("right", ai_speed_multiplier)      # AI
        self.physics = Physics()

        self.score_left: int = 0
        self.score_right: int = 0
        self.done: bool = False

        # Speed-up timer state
        self._game_start_time: float = 0.0
        self._last_speedup_time: float = 0.0
        self._speedup_count: int = 0          # how many speed-ups have fired this game
        self.speed_up_triggered: bool = False  # True for exactly one step after a speed-up fires

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset ball, paddles, scores, done flag, and speed-up timer."""
        initial_speed = self._cfg.BALL_SPEED_INITIAL * self._ball_speed_multiplier
        self.ball.reset(speed_override=initial_speed)
        self.paddle_left.reset()
        self.paddle_right.reset()
        self.score_left = 0
        self.score_right = 0
        self.done = False
        self._game_start_time = time.monotonic()
        self._last_speedup_time = self._game_start_time
        self._speedup_count = 0
        self.speed_up_triggered = False

    def reset_round(self) -> None:
        """Reset only the ball (after a point is scored). Scores and timer are preserved."""
        initial_speed = self._cfg.BALL_SPEED_INITIAL * self._ball_speed_multiplier
        self.ball.reset(speed_override=initial_speed)

    def step(
        self,
        left_action: int,
        right_action: int,
    ) -> tuple[float, bool]:
        """
        Advance the game by one frame.

        Args:
            left_action:  action for left paddle  (0=UP, 1=DOWN, 2=STAY)
            right_action: action for right paddle (0=UP, 1=DOWN, 2=STAY)

        Returns:
            (reward_right, done)
            reward_right: reward signal for the AI (right paddle)
            done:         True when a player reaches the winning score
        """
        if self.done:
            return 0.0, True

        # --- Timed speed-up check ---
        self.speed_up_triggered = False
        if self._speed_up_interval > 0:
            now = time.monotonic()
            elapsed_since_last = now - self._last_speedup_time
            if elapsed_since_last >= self._speed_up_interval:
                self._apply_speed_up()
                self._last_speedup_time = now
                self.speed_up_triggered = True

        # Apply paddle actions
        self.paddle_left.apply_action(left_action)
        self.paddle_right.apply_action(right_action)

        # Advance ball
        self.ball.update()

        # Physics: wall bounce
        self.physics.check_wall_collision(self.ball)

        # Physics: paddle collision
        hit_side = self.physics.check_paddle_collision(
            self.ball, self.paddle_left, self.paddle_right
        )

        # Compute reward
        reward: float = self._reward_cfg.SURVIVE_STEP

        # Proximity shaping: reward AI for keeping paddle close to ball's y
        paddle_center = self.paddle_right.get_center_y()
        ball_y = self.ball.y
        dist = abs(paddle_center - ball_y) / self._cfg.WINDOW_HEIGHT
        reward += 0.3 * (1.0 - dist)

        if hit_side == "right":
            # AI (right paddle) returned the ball
            reward += self._reward_cfg.RETURN_BALL

        # Physics: score detection
        scorer = self.physics.check_score(self.ball)

        if scorer == "right":
            # Right player (AI) scored
            self.score_right += 1
            reward += self._reward_cfg.SCORE_POINT
            self.reset_round()
        elif scorer == "left":
            # Left player (human / bot) scored — AI missed
            self.score_left += 1
            reward += self._reward_cfg.MISS_POINT
            self.reset_round()

        # Check win condition
        if (
            self.score_right >= self._cfg.WINNING_SCORE
            or self.score_left >= self._cfg.WINNING_SCORE
        ):
            self.done = True

        return reward, self.done

    def get_ai_state(self) -> list[float]:
        """
        Return a 6-float normalized state vector for the DQN agent.

        Indices:
            0: ball_x          normalized to [0, 1]
            1: ball_y          normalized to [0, 1]
            2: ball_vx         normalized to [0, 1]  (shifted from [-max, +max])
            3: ball_vy         normalized to [0, 1]
            4: ai_paddle_y     normalized to [0, 1]  (right paddle)
            5: player_paddle_y normalized to [0, 1]  (left paddle)
        """
        bx, by, bvx, bvy = self.ball.get_state()
        ai_y = self.paddle_right.get_state()
        player_y = self.paddle_left.get_state()
        return [bx, by, bvx, bvy, ai_y, player_y]

    def is_done(self) -> bool:
        """Return True when the game has ended."""
        return self.done

    def get_winner(self) -> str | None:
        """Return 'left', 'right', or None if game is still in progress."""
        if self.score_right >= self._cfg.WINNING_SCORE:
            return "right"
        if self.score_left >= self._cfg.WINNING_SCORE:
            return "left"
        return None

    def get_elapsed_seconds(self) -> float:
        """Return total real-world seconds since the game started."""
        return time.monotonic() - self._game_start_time

    def get_speedup_count(self) -> int:
        """Return how many speed-ups have fired this game."""
        return self._speedup_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_speed_up(self) -> None:
        """
        Increase the ball's current speed by speed_up_amount, capped at BALL_SPEED_MAX.
        Preserves the ball's direction.
        """
        import math
        self._speedup_count += 1
        current_speed = math.hypot(self.ball.vx, self.ball.vy)
        new_speed = min(
            current_speed + self._speed_up_amount,
            self._cfg.BALL_SPEED_MAX,
        )
        if current_speed > 0:
            scale = new_speed / current_speed
            self.ball.vx *= scale
            self.ball.vy *= scale
        self.ball.speed = new_speed

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GameState(score={self.score_left}:{self.score_right}, "
            f"done={self.done}, speedups={self._speedup_count}, ball={self.ball!r})"
        )
