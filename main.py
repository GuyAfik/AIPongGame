"""
Layer 4 — Application Entry Point
Parses CLI arguments and wires together all components.
Supports --train and --play modes with --difficulty selection.

Usage:
    uv run python main.py --train
    uv run python main.py --train --episodes 3000
    uv run python main.py --play
    uv run python main.py --play --difficulty easy
    uv run python main.py --play --difficulty insane
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pygame

from src.utils.config import Difficulty, DIFFICULTY_PRESETS, AI_CONFIG
from src.ai.environment import PongEnvironment
from src.ai.dqn_agent import DQNAgent
from src.ai.trainer import Trainer
from src.ai.visualizer import TrainingVisualizer

_MODEL_PATH = pathlib.Path("models/trained_model.pth")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai-pong",
        description="Pong with a locally-trained DQN AI opponent.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--train",
        action="store_true",
        help="Train the DQN agent (headless, no window).",
    )
    mode.add_argument(
        "--play",
        action="store_true",
        help="Play against the trained AI.",
    )

    parser.add_argument(
        "--difficulty",
        choices=[d.value for d in Difficulty],
        default=None,
        help=(
            "AI difficulty for play mode: easy | medium | hard | insane. "
            "If omitted, a menu is shown in-game."
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=f"Number of training episodes (default: {AI_CONFIG.EPISODES}).",
    )

    parser.add_argument(
        "--no-visualizer",
        action="store_true",
        help="Disable the live training dashboard (faster training).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Train mode
# ---------------------------------------------------------------------------

def run_train(args: argparse.Namespace) -> None:
    """Set up and run the training loop."""
    print("\n[Mode: TRAIN]")

    env = PongEnvironment(headless=True, difficulty=Difficulty.HARD)
    agent = DQNAgent()

    visualizer: TrainingVisualizer | None = None
    if not args.no_visualizer:
        visualizer = TrainingVisualizer(update_interval=50)

    trainer = Trainer(env, agent, visualizer=visualizer)

    try:
        trainer.train(episodes=args.episodes)
    except KeyboardInterrupt:
        print("\n[Training interrupted — saving checkpoint...]")
        agent.save("models/interrupted_checkpoint.pth")
        if visualizer:
            visualizer.save()
    finally:
        if visualizer:
            visualizer.close()
        env.close()


# ---------------------------------------------------------------------------
# Play mode
# ---------------------------------------------------------------------------

def run_play(args: argparse.Namespace) -> None:
    """Load the trained model and run the interactive play loop."""
    print("\n[Mode: PLAY]")

    if not _MODEL_PATH.exists():
        print(
            f"\n[ERROR] No trained model found at '{_MODEL_PATH}'.\n"
            "Run training first:  uv run python main.py --train\n"
        )
        sys.exit(1)

    # Determine difficulty
    if args.difficulty is not None:
        difficulty = Difficulty(args.difficulty)
    else:
        difficulty = None  # will be selected via in-game menu

    # We need a renderer to show the menu, so create env with a placeholder
    # difficulty first, then potentially override after menu selection.
    env = PongEnvironment(
        headless=False,
        difficulty=difficulty or Difficulty.HARD,
    )

    # Show difficulty menu if not specified on CLI
    if difficulty is None and env.renderer is not None:
        difficulty = env.renderer.show_difficulty_menu()
        # Rebuild env with the chosen difficulty
        env.close()
        env = PongEnvironment(headless=False, difficulty=difficulty)

    diff_cfg = DIFFICULTY_PRESETS[difficulty or Difficulty.HARD]
    print(f"  Difficulty: {(difficulty or Difficulty.HARD).value.upper()}")

    # Load agent
    agent = DQNAgent()
    agent.load(_MODEL_PATH)
    agent.epsilon = 0.0   # pure exploitation in play mode

    play_again = True
    while play_again:
        play_again = _play_one_game(env, agent, diff_cfg.epsilon_override)

    env.close()


def _play_one_game(
    env: PongEnvironment,
    agent: DQNAgent,
    epsilon_override: float,
) -> bool:
    """
    Run one complete game (until winning score is reached).

    Returns:
        True if the player wants to play again, False to quit.
    """
    state = env.reset()
    renderer = env.renderer
    game_state = env.game_state
    clock_fps = env._game_cfg.FPS

    running = True
    while running:
        # --- Handle pygame events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        # --- Human input (left paddle) ---
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            human_action = 0   # UP
        elif keys[pygame.K_s]:
            human_action = 1   # DOWN
        else:
            human_action = 2   # STAY

        # --- AI action (right paddle) ---
        ai_action = agent.select_action(state, epsilon_override=epsilon_override)

        # Step both paddles together via game_state
        reward, done = game_state.step(human_action, ai_action)
        state = game_state.get_ai_state()

        # --- Render ---
        if renderer is not None:
            renderer.draw(game_state)
            renderer.tick(clock_fps)

        if done:
            running = False

    # Game over screen
    winner = game_state.get_winner() or "right"
    if renderer is not None:
        return renderer.show_game_over(winner, game_state.score_left, game_state.score_right)

    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.train:
        run_train(args)
    elif args.play:
        run_play(args)


if __name__ == "__main__":
    main()
