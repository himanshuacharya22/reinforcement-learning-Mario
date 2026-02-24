# 🍄 Super Mario Bros — Reinforcement Learning

Train a PPO agent to play Super Mario Bros using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

## Features

- **Tuned PPO hyperparameters** — linear LR decay, optimized clip range, entropy coefficient
- **Multi-signal reward shaping** — forward progress, score, time penalty, death penalty, idle penalty, level-clear bonus
- **Professional preprocessing** — frame skip, grayscale, 84×84 resize, 4-frame stack, channels-first CNN
- **COMPLEX_MOVEMENT action space** — 12 actions including run+jump combinations
- **TensorBoard logging** — monitor training in real time
- **Checkpoint & best-model saving** — never lose progress
- **CLI overrides** — tune any parameter without editing code
- **Separate evaluation script** — watch your agent play with episode statistics

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python train.py                          # Full training (2M steps)
python train.py --total_timesteps 500000 # Quick run
python train.py --num_envs 4             # Parallel environments
```

### 3. Monitor

```bash
tensorboard --logdir logs/
```

### 4. Evaluate

```bash
python evaluate.py                                            # Best model
python evaluate.py --model_path models/checkpoint_50000.zip   # Specific checkpoint
python evaluate.py --episodes 10                              # 10 episodes
```

### 5. Resume training

```bash
python train.py --resume_from models/checkpoint_100000.zip
```

## Project Structure

```
├── config.py              # All hyperparameters & paths (dataclass-based)
├── env_factory.py         # Environment construction pipeline
├── train.py               # Training script with CLI args
├── evaluate.py            # Evaluation & playback script
├── callbacks.py           # Checkpoint & metric callbacks
├── wrappers/
│   ├── __init__.py
│   ├── reward_wrapper.py  # Multi-signal reward shaping
│   ├── skip_frame.py      # Frame skip with max-pooling
│   └── resize_obs.py      # Observation resizing (84×84)
├── requirements.txt
└── README.md
```

## Configuration

All parameters are in `config.py`. Override any value via CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--total_timesteps` | 2,000,000 | Total training steps |
| `--learning_rate` | 2.5e-4 | PPO learning rate |
| `--n_steps` | 512 | Rollout steps per update |
| `--batch_size` | 64 | Minibatch size |
| `--n_epochs` | 10 | PPO epochs per update |
| `--ent_coef` | 0.01 | Entropy coefficient |
| `--gamma` | 0.99 | Discount factor |
| `--num_envs` | 1 | Parallel environments |
| `--frame_skip` | 4 | Frames to skip |
| `--checkpoint_freq` | 25,000 | Steps between checkpoints |
| `--eval_freq` | 25,000 | Steps between evaluations |
| `--seed` | 42 | Random seed |
| `--resume_from` | — | Checkpoint path to resume |

## License

MIT
