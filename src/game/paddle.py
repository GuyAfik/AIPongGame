"""
Layer 1 — Core Game Entity: Paddle
Owns position, dimensions, and movement logic.
Only imports from Layer 0 (config).
"""

import pygame
from src.utils.config import GAME_CONFIG


class Paddle:
    """
    Represents a Pong paddle (left = human, right = AI).

    Responsibilities:
    - Track top-left position (x, y)
    - Move up/down while clamping to screen bounds
    - Provide a pygame.Rect for collision detection
    - Provide a normalized y-position for the AI state vector
    """

    def __init__(self, side: str, speed_multiplier: float = 1.0) -> None:
        """
        Args:
            side: 'left' (human player) or 'right' (AI player).
            speed_multiplier: scales paddle speed (used for difficulty).
        """
        if side not in ("left", "right"):
            raise ValueError(f"Paddle side must be 'left' or 'right', got '{side}'")

        self._cfg = GAME_CONFIG
        self.side: str = side
        self.width: int = self._cfg.PADDLE_WIDTH
        self.height: int = self._cfg.PADDLE_HEIGHT
        self.speed: float = self._cfg.PADDLE_SPEED * speed_multiplier

        # Compute initial x position based on side
        margin = self._cfg.PADDLE_MARGIN
        if side == "left":
            self.x: float = float(margin)
        else:
            self.x = float(self._cfg.WINDOW_WIDTH - margin - self.width)

        # Start vertically centered
        self.y: float = float(
            (self._cfg.WINDOW_HEIGHT - self.height) // 2
        )

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move_up(self) -> None:
        """Move paddle upward, clamped to top screen edge."""
        self.y = max(0.0, self.y - self.speed)

    def move_down(self) -> None:
        """Move paddle downward, clamped to bottom screen edge."""
        self.y = min(
            float(self._cfg.WINDOW_HEIGHT - self.height),
            self.y + self.speed,
        )

    def stay(self) -> None:
        """No movement — explicit no-op for clarity."""
        pass

    def reset(self) -> None:
        """Return paddle to vertical center."""
        self.y = float((self._cfg.WINDOW_HEIGHT - self.height) // 2)

    def apply_action(self, action: int) -> None:
        """
        Apply an action index to the paddle.

        Args:
            action: 0 = UP, 1 = DOWN, 2 = STAY
        """
        if action == 0:
            self.move_up()
        elif action == 1:
            self.move_down()
        else:
            self.stay()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def get_rect(self) -> pygame.Rect:
        """Return a pygame.Rect for collision detection."""
        return pygame.Rect(int(self.x), int(self.y), self.width, self.height)

    def get_center_y(self) -> float:
        """Return the vertical center of the paddle."""
        return self.y + self.height / 2.0

    def get_state(self) -> float:
        """Return normalized y position in [0, 1] for the AI state vector."""
        return self.y / float(self._cfg.WINDOW_HEIGHT - self.height)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Paddle(side={self.side!r}, x={self.x:.1f}, y={self.y:.1f}, "
            f"speed={self.speed:.2f})"
        )
