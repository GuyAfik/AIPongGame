"""
Layer 1 — Core Game Entity: Ball
Owns position, velocity, and collision rect.
Only imports from Layer 0 (config).
"""

import random
import pygame
from src.utils.config import GAME_CONFIG


class Ball:
    """
    Represents the Pong ball.

    Responsibilities:
    - Track position (x, y) and velocity (vx, vy)
    - Advance position each frame via update()
    - Provide a pygame.Rect for collision detection
    - Provide a normalized state tuple for the AI
    - Reset to center with a random direction
    """

    def __init__(self) -> None:
        self._cfg = GAME_CONFIG
        self.x: float = 0.0
        self.y: float = 0.0
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.speed: float = self._cfg.BALL_SPEED_INITIAL
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, speed_override: float | None = None) -> None:
        """Place ball at screen center with a random diagonal direction."""
        self.x = float(self._cfg.WINDOW_WIDTH // 2)
        self.y = float(self._cfg.WINDOW_HEIGHT // 2)
        self.speed = speed_override if speed_override is not None else self._cfg.BALL_SPEED_INITIAL

        # Random angle: avoid near-horizontal shots (±30° from horizontal)
        angle_sign_x = random.choice([-1, 1])
        angle_sign_y = random.choice([-1, 1])
        self.vx = angle_sign_x * self.speed
        self.vy = angle_sign_y * self.speed * random.uniform(0.5, 0.9)

    def update(self) -> None:
        """Advance ball position by its current velocity. No collision here."""
        self.x += self.vx
        self.y += self.vy

    def increase_speed(self) -> None:
        """Increment ball speed after a successful paddle return, capped at max."""
        self.speed = min(
            self.speed + self._cfg.BALL_SPEED_INCREMENT,
            self._cfg.BALL_SPEED_MAX,
        )
        # Scale velocity vector to new speed while preserving direction
        magnitude = (self.vx ** 2 + self.vy ** 2) ** 0.5
        if magnitude > 0:
            scale = self.speed / magnitude
            self.vx *= scale
            self.vy *= scale

    def get_rect(self) -> pygame.Rect:
        """Return a pygame.Rect centered on the ball for collision detection."""
        r = self._cfg.BALL_RADIUS
        return pygame.Rect(
            int(self.x) - r,
            int(self.y) - r,
            r * 2,
            r * 2,
        )

    def get_state(self) -> tuple[float, float, float, float]:
        """
        Return normalized (x, y, vx, vy) in [0, 1] range for the AI state vector.
        vx/vy are normalized relative to BALL_SPEED_MAX.
        """
        w = self._cfg.WINDOW_WIDTH
        h = self._cfg.WINDOW_HEIGHT
        max_v = self._cfg.BALL_SPEED_MAX
        return (
            self.x / w,
            self.y / h,
            (self.vx + max_v) / (2 * max_v),   # shift to [0,1]
            (self.vy + max_v) / (2 * max_v),
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Ball(x={self.x:.1f}, y={self.y:.1f}, "
            f"vx={self.vx:.2f}, vy={self.vy:.2f}, speed={self.speed:.2f})"
        )
