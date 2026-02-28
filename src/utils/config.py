"""
Layer 0 — Configuration
Single source of truth for all constants and hyperparameters.
No logic. No imports from other project modules.
"""

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Game Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GameConfig:
    WINDOW_WIDTH: int = 800
    WINDOW_HEIGHT: int = 600
    FPS: int = 60

    BALL_RADIUS: int = 10
    BALL_SPEED_INITIAL: float = 5.0
    BALL_SPEED_MAX: float = 12.0
    BALL_SPEED_INCREMENT: float = 0.3  # speed increase per successful return

    PADDLE_WIDTH: int = 15
    PADDLE_HEIGHT: int = 90
    PADDLE_SPEED: float = 6.0
    PADDLE_MARGIN: int = 20  # distance from screen edge

    WINNING_SCORE: int = 5   # lower for training speed; raise to 10 for play mode

    # Colors (RGB)
    COLOR_BACKGROUND: tuple[int, int, int] = (0, 0, 0)
    COLOR_FOREGROUND: tuple[int, int, int] = (255, 255, 255)
    COLOR_SCORE: tuple[int, int, int] = (200, 200, 200)
    COLOR_CENTER_LINE: tuple[int, int, int] = (80, 80, 80)
    COLOR_EASY: tuple[int, int, int] = (100, 220, 100)
    COLOR_MEDIUM: tuple[int, int, int] = (220, 220, 100)
    COLOR_HARD: tuple[int, int, int] = (220, 140, 60)
    COLOR_INSANE: tuple[int, int, int] = (220, 60, 60)


# ---------------------------------------------------------------------------
# AI / DQN Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AIConfig:
    STATE_SIZE: int = 6        # [ball_x, ball_y, ball_vx, ball_vy, ai_paddle_y, player_paddle_y]
    ACTION_SIZE: int = 3       # 0=UP, 1=DOWN, 2=STAY

    HIDDEN_SIZE: int = 128

    LEARNING_RATE: float = 0.0005
    GAMMA: float = 0.99        # discount factor

    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.02
    EPSILON_DECAY: float = 0.998   # slower decay → more exploration

    REPLAY_BUFFER_SIZE: int = 50_000
    BATCH_SIZE: int = 128
    TARGET_UPDATE_FREQ: int = 200   # steps between target network syncs

    EPISODES: int = 10_000
    MAX_STEPS_PER_EPISODE: int = 2_000  # enough for a full 5-point game


# ---------------------------------------------------------------------------
# Reward Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardConfig:
    SCORE_POINT: float = 10.0     # AI scores a point
    MISS_POINT: float = -10.0     # AI misses the ball
    RETURN_BALL: float = 1.0      # AI successfully returns the ball
    SURVIVE_STEP: float = 0.1     # each timestep the AI stays alive


# ---------------------------------------------------------------------------
# Difficulty
# ---------------------------------------------------------------------------

class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    INSANE = "insane"


@dataclass(frozen=True)
class DifficultyConfig:
    """Behavioral constraints applied to the trained AI at inference time."""
    paddle_speed_multiplier: float        # scales AI paddle speed (1.0 = full speed)
    action_skip_frames: int               # AI reuses last action for N frames (reaction delay)
    epsilon_override: float               # forced random-action rate (0.0 = pure exploit)
    ball_speed_multiplier: float          # scales initial ball speed
    speed_up_interval_seconds: float      # ball gets faster every N real-world seconds (0 = disabled)
    speed_up_amount: float                # how much to add to ball speed each interval


# Preset difficulty configurations
DIFFICULTY_PRESETS: dict[Difficulty, DifficultyConfig] = {
    Difficulty.EASY: DifficultyConfig(
        paddle_speed_multiplier=0.50,
        action_skip_frames=4,
        epsilon_override=0.30,
        ball_speed_multiplier=1.0,
        speed_up_interval_seconds=120.0,  # speed up every 2 minutes
        speed_up_amount=1.0,
    ),
    Difficulty.MEDIUM: DifficultyConfig(
        paddle_speed_multiplier=0.75,
        action_skip_frames=2,
        epsilon_override=0.10,
        ball_speed_multiplier=1.0,
        speed_up_interval_seconds=60.0,   # speed up every 1 minute
        speed_up_amount=1.5,
    ),
    Difficulty.HARD: DifficultyConfig(
        paddle_speed_multiplier=1.00,
        action_skip_frames=0,
        epsilon_override=0.00,
        ball_speed_multiplier=1.0,
        speed_up_interval_seconds=20.0,   # speed up every 20 seconds
        speed_up_amount=2.0,
    ),
    Difficulty.INSANE: DifficultyConfig(
        paddle_speed_multiplier=1.00,
        action_skip_frames=0,
        epsilon_override=0.00,
        ball_speed_multiplier=1.4,
        speed_up_interval_seconds=20.0,   # speed up every 20 seconds
        speed_up_amount=2.5,
    ),
}


# ---------------------------------------------------------------------------
# Singleton instances (import these directly throughout the project)
# ---------------------------------------------------------------------------

GAME_CONFIG = GameConfig()
AI_CONFIG = AIConfig()
REWARD_CONFIG = RewardConfig()
