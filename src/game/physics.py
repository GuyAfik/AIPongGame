"""
Layer 2 — Game Engine: Physics
Stateless collision detection service.
Imports from Layer 0 (config) and Layer 1 (Ball, Paddle).
"""

import random
from src.utils.config import GAME_CONFIG
from src.game.ball import Ball
from src.game.paddle import Paddle


class Physics:
    """
    Stateless collision detection and response service.

    All methods are pure functions of their arguments — Physics holds no
    mutable state of its own. It modifies Ball velocity in-place when a
    collision is detected and returns a descriptive result.
    """

    def __init__(self) -> None:
        self._cfg = GAME_CONFIG

    # ------------------------------------------------------------------
    # Wall collision
    # ------------------------------------------------------------------

    def check_wall_collision(self, ball: Ball) -> bool:
        """
        Detect and respond to ball hitting top or bottom wall.

        Reverses ball.vy and clamps ball.y to stay within bounds.

        Returns:
            True if a wall collision occurred, False otherwise.
        """
        hit = False

        if ball.y - self._cfg.BALL_RADIUS <= 0:
            ball.vy = abs(ball.vy)          # bounce downward
            ball.y = float(self._cfg.BALL_RADIUS)
            hit = True

        elif ball.y + self._cfg.BALL_RADIUS >= self._cfg.WINDOW_HEIGHT:
            ball.vy = -abs(ball.vy)         # bounce upward
            ball.y = float(self._cfg.WINDOW_HEIGHT - self._cfg.BALL_RADIUS)
            hit = True

        return hit

    # ------------------------------------------------------------------
    # Paddle collision
    # ------------------------------------------------------------------

    def check_paddle_collision(
        self,
        ball: Ball,
        paddle_left: Paddle,
        paddle_right: Paddle,
    ) -> str | None:
        """
        Detect and respond to ball hitting either paddle.

        On hit:
        - Reverses ball.vx
        - Adds slight random vy perturbation for unpredictability
        - Calls ball.increase_speed()

        Returns:
            'left' | 'right' if a paddle was hit, None otherwise.
        """
        ball_rect = ball.get_rect()

        for paddle, side in ((paddle_left, "left"), (paddle_right, "right")):
            if not ball_rect.colliderect(paddle.get_rect()):
                continue

            # Reverse horizontal direction
            if side == "left":
                ball.vx = abs(ball.vx)      # ensure ball moves right
                ball.x = float(
                    paddle.x + paddle.width + self._cfg.BALL_RADIUS
                )
            else:
                ball.vx = -abs(ball.vx)     # ensure ball moves left
                ball.x = float(
                    paddle.x - self._cfg.BALL_RADIUS
                )

            # Add slight vertical randomness to prevent predictable loops
            ball.vy += random.uniform(-0.5, 0.5)

            # Clamp vy so it never becomes near-zero (boring horizontal rally)
            if abs(ball.vy) < 1.0:
                ball.vy = 1.0 if ball.vy >= 0 else -1.0

            ball.increase_speed()
            return side

        return None

    # ------------------------------------------------------------------
    # Score detection
    # ------------------------------------------------------------------

    def check_score(self, ball: Ball) -> str | None:
        """
        Detect whether the ball has passed a paddle (scoring event).

        Does NOT reset the ball — that is GameState's responsibility.

        Returns:
            'right' if right player scored (ball passed left edge),
            'left'  if left player scored (ball passed right edge),
            None    if no score yet.
        """
        if ball.x < 0:
            return "right"   # ball exited left → right player scores
        if ball.x > self._cfg.WINDOW_WIDTH:
            return "left"    # ball exited right → left player scores
        return None
