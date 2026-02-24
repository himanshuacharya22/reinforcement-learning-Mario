"""
Centralized configuration for Mario RL training.

All hyperparameters, paths, and reward weights are defined here
as dataclass fields for easy tuning and CLI overrides.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RewardConfig:
    """Weights for multi-signal reward shaping."""

    forward_progress_scale: float = 1.0     # Reward per pixel of forward movement
    death_penalty: float = -50.0            # Penalty on death
    level_clear_bonus: float = 100.0        # Bonus for completing the level
    score_delta_scale: float = 0.025        # Reward per point of score increase
    time_penalty: float = -0.01             # Small penalty per step to encourage speed
    idle_threshold: int = 30                # Steps before idle penalty kicks in
    idle_penalty: float = -1.0              # Penalty per step when idle


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters (tuned for Mario)."""

    learning_rate: float = 2.5e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_linear_lr_decay: bool = True


@dataclass
class EnvConfig:
    """Environment configuration."""

    game_id: str = "SuperMarioBros-v0"
    frame_skip: int = 4
    frame_size: int = 84
    frame_stack: int = 4
    num_envs: int = 1                       # Number of parallel envs
    use_subprocess: bool = False            # Use SubprocVecEnv vs DummyVecEnv


@dataclass
class TrainConfig:
    """Training session configuration."""

    total_timesteps: int = 2_000_000
    checkpoint_freq: int = 25_000           # Save model every N steps
    eval_freq: int = 25_000                 # Evaluate every N steps
    eval_episodes: int = 5                  # Episodes per evaluation
    log_interval: int = 1                   # Log every N PPO updates
    seed: int = 42

    # Paths
    model_dir: Path = field(default_factory=lambda: Path("models"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    video_dir: Path = field(default_factory=lambda: Path("videos"))

    # Resume training from a checkpoint
    resume_from: str | None = None


@dataclass
class Config:
    """Top-level configuration aggregating all sub-configs."""

    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
