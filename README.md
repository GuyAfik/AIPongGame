# 🏓 AI Pong Game

> 🎬 **[Watch Demo Video](https://github.com/GuyAfik/AIPongGame/blob/master/docs/recording.mp4)**

A classic Pong game built with **pygame** and a locally-trained **Deep Q-Network (DQN)** AI opponent. No cloud APIs, no LLMs — the AI learns entirely on your machine using Reinforcement Learning.

---

## 🤖 How the AI Works

The AI is a **DQN (Deep Q-Network)** — the same algorithm DeepMind used to beat Atari games in 2013.

- **Observes** 6 game values: ball position, ball velocity, both paddle positions
- **Chooses** one of 3 actions: move UP, move DOWN, or STAY
- **Receives** rewards for scoring and penalties for missing
- **Learns** over thousands of games which actions lead to winning

No internet. No API keys. Trains in ~30 minutes on a laptop CPU.

---

## 🏗️ Architecture

Strict **5-layer OOP architecture** — lower layers cannot import higher layers:

```
Layer 4  main.py                    Entry point, CLI
Layer 3  src/ai/                    AI brain (DQN, environment, trainer, visualizer)
Layer 2  src/game/                  Game engine (physics, state, renderer)
Layer 1  src/game/ball.py, paddle.py  Game entities
Layer 0  src/utils/config.py        All constants (no logic)
```

See [`plans/design.md`](plans/design.md) for the full design document.

---

## ⚙️ Setup

### Prerequisites

Install [uv](https://docs.astral.sh/uv/) (fast Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
git clone https://github.com/yourname/AIPongGame.git
cd AIPongGame
uv sync
```

This installs Python 3.13 + all dependencies into a local `.venv` automatically.

---

## 🚀 Usage

### Train the AI

```bash
# Full training (10,000 episodes, ~5 hours, with live dashboard)
uv run python main.py --train

# Training without the live chart (faster terminal output, live logs)
PYTHONUNBUFFERED=1 uv run python main.py --train --no-visualizer

# Quick test run (200 episodes)
PYTHONUNBUFFERED=1 uv run python main.py --train --episodes 200 --no-visualizer
```

During training, a **live matplotlib dashboard** shows:
- Episode reward (raw + rolling average)
- Win rate over last 100 episodes
- Epsilon decay curve
- Steps per episode

The trained model is saved to `models/trained_model.pth`.

### Play against the AI

```bash
# Show in-game difficulty menu
uv run python main.py --play

# Skip menu, start directly at a difficulty
uv run python main.py --play --difficulty easy
uv run python main.py --play --difficulty medium
uv run python main.py --play --difficulty hard
uv run python main.py --play --difficulty insane
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `W` | Move your paddle UP |
| `S` | Move your paddle DOWN |
| `R` | Play again (on game over screen) |
| `ESC` | Quit |

**You** play on the **left**. The **AI** plays on the **right**.

---

## 🎯 Difficulty Levels

| Level | AI Speed | Reaction Delay | Random Moves | Ball Speed |
|-------|---------|---------------|-------------|-----------|
| Easy | 50% | Every 4 frames | 30% random | Normal |
| Medium | 75% | Every 2 frames | 10% random | Normal |
| Hard | 100% | No delay | Pure exploit | Normal |
| Insane | 100% | No delay | Pure exploit | 1.4× faster |

Difficulty does **not** retrain the model — it applies behavioral constraints at inference time.

---

## 📁 Project Structure

```
AIPongGame/
├── src/
│   ├── game/
│   │   ├── ball.py          # Layer 1 — Ball entity
│   │   ├── paddle.py        # Layer 1 — Paddle entity
│   │   ├── physics.py       # Layer 2 — Collision detection
│   │   ├── game_state.py    # Layer 2 — State manager
│   │   └── renderer.py      # Layer 2 — pygame rendering
│   ├── ai/
│   │   ├── neural_net.py    # Layer 3 — PyTorch DQN network
│   │   ├── replay_buffer.py # Layer 3 — Experience memory
│   │   ├── dqn_agent.py     # Layer 3 — DQN agent
│   │   ├── environment.py   # Layer 3 — Gym-like wrapper
│   │   ├── visualizer.py    # Layer 3 — Training dashboard
│   │   └── trainer.py       # Layer 3 — Training loop
│   └── utils/
│       └── config.py        # Layer 0 — All constants
├── models/
│   ├── trained_model.pth    # Saved model weights
│   └── training_plot.png    # Training progress chart
├── plans/
│   └── design.md            # Full architecture design document
├── main.py                  # Entry point
├── pyproject.toml           # uv project manifest
└── uv.lock                  # Locked dependencies
```

---

## 🧠 Neural Network

```
Input  (6):   ball_x, ball_y, ball_vx, ball_vy, ai_paddle_y, player_paddle_y
Hidden (128): Linear → ReLU
Hidden (128): Linear → ReLU
Output (3):   Q(UP), Q(DOWN), Q(STAY)
```

All inputs normalized to `[0, 1]`. Action = `argmax` of output Q-values.

---

## 📊 Reward Function

| Event | Reward |
|-------|--------|
| AI scores a point | +10 |
| AI misses the ball | −10 |
| AI returns the ball | +1 |
| Each step alive | +0.1 |

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| Game engine | `pygame 2.6` |
| Neural network | `PyTorch 2.10` |
| Numerics | `numpy 2.4` |
| Training charts | `matplotlib 3.10` |
| Package manager | `uv` |

**100% local. No cloud. No API keys.**
