"""
Layer 2 — Game Engine: Renderer
Owns the pygame surface and draws the full game scene each frame.
Also renders the difficulty selection menu screen.
Imports from Layer 0 (config) and Layer 1/2 types for type hints only.
"""

from __future__ import annotations

import pygame
from src.utils.config import GAME_CONFIG, Difficulty, DIFFICULTY_PRESETS


class Renderer:
    """
    Manages the pygame window and draws all game elements.

    Responsibilities:
    - Initialize and own the pygame display surface
    - Draw background, center line, ball, paddles, and score each frame
    - Render the difficulty selection menu before play mode starts
    - Enforce frame rate via clock.tick()
    - Clean up pygame on close()
    """

    def __init__(self) -> None:
        self._cfg = GAME_CONFIG
        self._screen: pygame.Surface | None = None
        self._font_score: pygame.font.Font | None = None
        self._font_menu: pygame.font.Font | None = None
        self._font_small: pygame.font.Font | None = None
        self._clock: pygame.time.Clock | None = None
        self._initialized: bool = False

        # Speed-up flash state
        self._speedup_flash_frames: int = 0   # frames remaining to show flash
        self._FLASH_DURATION: int = 90         # ~1.5 seconds at 60 FPS

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Initialize pygame and create the display window."""
        pygame.init()
        self._screen = pygame.display.set_mode(
            (self._cfg.WINDOW_WIDTH, self._cfg.WINDOW_HEIGHT)
        )
        pygame.display.set_caption("AI Pong")
        self._font_score = pygame.font.SysFont("monospace", 48, bold=True)
        self._font_menu = pygame.font.SysFont("monospace", 36, bold=True)
        self._font_small = pygame.font.SysFont("monospace", 22)
        self._clock = pygame.time.Clock()
        self._initialized = True

    def close(self) -> None:
        """Quit pygame cleanly."""
        pygame.quit()
        self._initialized = False

    # ------------------------------------------------------------------
    # Game scene rendering
    # ------------------------------------------------------------------

    def draw(self, game_state: object) -> None:
        """
        Draw the full game scene for one frame.

        Args:
            game_state: a GameState instance (typed as object to avoid
                        circular import — Layer 2 → Layer 2 is fine but
                        we keep the import local to avoid coupling).
        """
        assert self._initialized, "Renderer.init() must be called before draw()"
        screen = self._screen
        cfg = self._cfg

        # Background
        screen.fill(cfg.COLOR_BACKGROUND)

        # Center dashed line
        self._draw_center_line()

        # Ball
        ball = game_state.ball
        pygame.draw.circle(
            screen,
            cfg.COLOR_FOREGROUND,
            (int(ball.x), int(ball.y)),
            cfg.BALL_RADIUS,
        )

        # Paddles
        for paddle in (game_state.paddle_left, game_state.paddle_right):
            pygame.draw.rect(screen, cfg.COLOR_FOREGROUND, paddle.get_rect())

        # Score
        self._draw_score(game_state.score_left, game_state.score_right)

        # Speed indicator (current ball speed)
        self._draw_speed_indicator(ball.speed)

        # Speed-up flash — trigger if game_state just fired one
        if getattr(game_state, "speed_up_triggered", False):
            self._speedup_flash_frames = self._FLASH_DURATION

        if self._speedup_flash_frames > 0:
            self._draw_speedup_flash()
            self._speedup_flash_frames -= 1

        pygame.display.flip()

    def tick(self, fps: int | None = None) -> None:
        """Enforce frame rate. Uses GameConfig.FPS if fps not specified."""
        assert self._clock is not None
        self._clock.tick(fps or self._cfg.FPS)

    # ------------------------------------------------------------------
    # Difficulty menu
    # ------------------------------------------------------------------

    def show_difficulty_menu(self) -> Difficulty:
        """
        Block until the player selects a difficulty level.

        Renders a full-screen menu and waits for key 1-4 or ESC.

        Returns:
            The selected Difficulty enum value.
        """
        assert self._initialized, "Renderer.init() must be called first"

        options: list[tuple[str, Difficulty, tuple[int, int, int]]] = [
            ("1  —  Easy",   Difficulty.EASY,   self._cfg.COLOR_EASY),
            ("2  —  Medium", Difficulty.MEDIUM, self._cfg.COLOR_MEDIUM),
            ("3  —  Hard",   Difficulty.HARD,   self._cfg.COLOR_HARD),
            ("4  —  Insane", Difficulty.INSANE, self._cfg.COLOR_INSANE),
        ]

        key_map = {
            pygame.K_1: Difficulty.EASY,
            pygame.K_2: Difficulty.MEDIUM,
            pygame.K_3: Difficulty.HARD,
            pygame.K_4: Difficulty.INSANE,
        }

        selected: Difficulty | None = None

        while selected is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise SystemExit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.close()
                        raise SystemExit(0)
                    if event.key in key_map:
                        selected = key_map[event.key]

            self._screen.fill(self._cfg.COLOR_BACKGROUND)

            # Title
            title = self._font_menu.render("AI  PONG", True, self._cfg.COLOR_FOREGROUND)
            self._screen.blit(
                title,
                (self._cfg.WINDOW_WIDTH // 2 - title.get_width() // 2, 80),
            )

            subtitle = self._font_small.render(
                "Select Difficulty", True, self._cfg.COLOR_SCORE
            )
            self._screen.blit(
                subtitle,
                (self._cfg.WINDOW_WIDTH // 2 - subtitle.get_width() // 2, 150),
            )

            # Options
            for i, (label, difficulty, color) in enumerate(options):
                surf = self._font_menu.render(label, True, color)
                y = 230 + i * 70
                self._screen.blit(
                    surf,
                    (self._cfg.WINDOW_WIDTH // 2 - surf.get_width() // 2, y),
                )

            # Controls hint
            hint = self._font_small.render(
                "W / S  to move    ESC to quit", True, self._cfg.COLOR_SCORE
            )
            self._screen.blit(
                hint,
                (
                    self._cfg.WINDOW_WIDTH // 2 - hint.get_width() // 2,
                    self._cfg.WINDOW_HEIGHT - 60,
                ),
            )

            pygame.display.flip()
            assert self._clock is not None
            self._clock.tick(30)

        return selected

    def show_game_over(self, winner: str, score_left: int, score_right: int) -> bool:
        """
        Display a game-over screen.

        Args:
            winner: 'left' (human won) or 'right' (AI won)
            score_left: final left score
            score_right: final right score

        Returns:
            True if player wants to play again, False to quit.
        """
        assert self._initialized

        if winner == "right":
            msg = "AI  WINS!"
            color = self._cfg.COLOR_INSANE
        else:
            msg = "YOU  WIN!"
            color = self._cfg.COLOR_EASY

        waiting = True
        play_again = False

        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        play_again = True
                        waiting = False
                    elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                        waiting = False

            self._screen.fill(self._cfg.COLOR_BACKGROUND)

            title = self._font_menu.render(msg, True, color)
            self._screen.blit(
                title,
                (self._cfg.WINDOW_WIDTH // 2 - title.get_width() // 2, 180),
            )

            score_surf = self._font_score.render(
                f"{score_left}  :  {score_right}", True, self._cfg.COLOR_FOREGROUND
            )
            self._screen.blit(
                score_surf,
                (self._cfg.WINDOW_WIDTH // 2 - score_surf.get_width() // 2, 270),
            )

            hint = self._font_small.render(
                "R = Play Again    ESC = Quit", True, self._cfg.COLOR_SCORE
            )
            self._screen.blit(
                hint,
                (self._cfg.WINDOW_WIDTH // 2 - hint.get_width() // 2, 380),
            )

            pygame.display.flip()
            assert self._clock is not None
            self._clock.tick(30)

        return play_again

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_speedup_flash(self) -> None:
        """
        Render a centered 'SPEED UP!' flash that fades out over _FLASH_DURATION frames.
        """
        # Fade: alpha goes from 255 → 0 over the flash duration
        alpha = int(255 * self._speedup_flash_frames / self._FLASH_DURATION)
        color = (255, min(80 + alpha, 255), 0)   # orange-red, fades to dim

        flash_surf = self._font_menu.render("⚡  SPEED UP!  ⚡", True, color)
        x = self._cfg.WINDOW_WIDTH // 2 - flash_surf.get_width() // 2
        y = self._cfg.WINDOW_HEIGHT // 2 - flash_surf.get_height() // 2
        self._screen.blit(flash_surf, (x, y))

    def _draw_speed_indicator(self, ball_speed: float) -> None:
        """
        Render a small speed readout in the bottom-center of the screen.
        """
        max_speed = self._cfg.BALL_SPEED_MAX
        pct = min(ball_speed / max_speed, 1.0)

        # Color: green → yellow → red as speed increases
        r = int(min(pct * 2, 1.0) * 220)
        g = int(min((1.0 - pct) * 2, 1.0) * 200)
        color = (r, g, 40)

        label = self._font_small.render(
            f"Speed: {ball_speed:.1f} / {max_speed:.0f}", True, color
        )
        self._screen.blit(
            label,
            (
                self._cfg.WINDOW_WIDTH // 2 - label.get_width() // 2,
                self._cfg.WINDOW_HEIGHT - 28,
            ),
        )

    def _draw_center_line(self) -> None:
        """Draw a dashed vertical center line."""
        x = self._cfg.WINDOW_WIDTH // 2
        dash_height = 15
        gap = 10
        y = 0
        while y < self._cfg.WINDOW_HEIGHT:
            pygame.draw.rect(
                self._screen,
                self._cfg.COLOR_CENTER_LINE,
                pygame.Rect(x - 1, y, 2, dash_height),
            )
            y += dash_height + gap

    def _draw_score(self, score_left: int, score_right: int) -> None:
        """Render score digits on either side of the center line."""
        cx = self._cfg.WINDOW_WIDTH // 2

        left_surf = self._font_score.render(str(score_left), True, self._cfg.COLOR_SCORE)
        self._screen.blit(left_surf, (cx - 80 - left_surf.get_width(), 20))

        right_surf = self._font_score.render(str(score_right), True, self._cfg.COLOR_SCORE)
        self._screen.blit(right_surf, (cx + 80, 20))
