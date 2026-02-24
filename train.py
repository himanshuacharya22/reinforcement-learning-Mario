"""
Training script for Super Mario Bros RL agent.

Usage:
  python train.py                           # Train with defaults (2M steps)
  python train.py --total_timesteps 500000  # Quick run
  python train.py --resume_from models/checkpoint_100000.zip
"""

import argparse
import logging
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import get_linear_fn

from callbacks import TrainCallback
from config import Config
from env_factory import make_env

# ── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-18s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a PPO agent on Super Mario Bros")
    p.add_argument("--total_timesteps", type=int, default=None, help="Total training timesteps")
    p.add_argument("--learning_rate", type=float, default=None, help="PPO learning rate")
    p.add_argument("--n_steps", type=int, default=None, help="Rollout steps per update")
    p.add_argument("--batch_size", type=int, default=None, help="Minibatch size")
    p.add_argument("--n_epochs", type=int, default=None, help="PPO epochs per update")
    p.add_argument("--ent_coef", type=float, default=None, help="Entropy coefficient")
    p.add_argument("--gamma", type=float, default=None, help="Discount factor")
    p.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments")
    p.add_argument("--frame_skip", type=int, default=None, help="Frame skip count")
    p.add_argument("--checkpoint_freq", type=int, default=None, help="Checkpoint every N steps")
    p.add_argument("--eval_freq", type=int, default=None, help="Evaluate every N steps")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint .zip to resume from")
    return p.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Apply CLI argument overrides to the config."""
    if args.total_timesteps is not None:
        cfg.train.total_timesteps = args.total_timesteps
    if args.learning_rate is not None:
        cfg.ppo.learning_rate = args.learning_rate
    if args.n_steps is not None:
        cfg.ppo.n_steps = args.n_steps
    if args.batch_size is not None:
        cfg.ppo.batch_size = args.batch_size
    if args.n_epochs is not None:
        cfg.ppo.n_epochs = args.n_epochs
    if args.ent_coef is not None:
        cfg.ppo.ent_coef = args.ent_coef
    if args.gamma is not None:
        cfg.ppo.gamma = args.gamma
    if args.num_envs is not None:
        cfg.env.num_envs = args.num_envs
    if args.frame_skip is not None:
        cfg.env.frame_skip = args.frame_skip
    if args.checkpoint_freq is not None:
        cfg.train.checkpoint_freq = args.checkpoint_freq
    if args.eval_freq is not None:
        cfg.train.eval_freq = args.eval_freq
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.resume_from is not None:
        cfg.train.resume_from = args.resume_from
    return cfg


def main():
    args = parse_args()
    cfg = apply_overrides(Config(), args)

    # Ensure output directories exist
    cfg.train.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.train.log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Super Mario Bros — PPO Training")
    logger.info("=" * 60)
    logger.info("Total timesteps : %s", f"{cfg.train.total_timesteps:,}")
    logger.info("Learning rate   : %s", cfg.ppo.learning_rate)
    logger.info("Num envs        : %s", cfg.env.num_envs)
    logger.info("Frame skip      : %s", cfg.env.frame_skip)
    logger.info("Frame stack     : %s", cfg.env.frame_stack)
    logger.info("Model dir       : %s", cfg.train.model_dir.resolve())
    logger.info("Log dir         : %s", cfg.train.log_dir.resolve())
    logger.info("=" * 60)

    # ── Build environments ──────────────────────────────────────────────
    logger.info("Building training environment...")
    train_env = make_env(cfg)

    logger.info("Building evaluation environment...")
    import copy
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.env.num_envs = 1
    eval_cfg.env.use_subprocess = False
    eval_env = make_env(eval_cfg)

    # ── Build or load model ─────────────────────────────────────────────
    lr = cfg.ppo.learning_rate
    if cfg.ppo.use_linear_lr_decay:
        lr = get_linear_fn(cfg.ppo.learning_rate, 0.0, 1.0)

    if not cfg.train.resume_from:
        auto_resume_path = cfg.train.model_dir / "final_model.zip"
        if auto_resume_path.exists():
            cfg.train.resume_from = str(auto_resume_path)
            logger.info("Auto-detected existing model → %s", auto_resume_path)

    if cfg.train.resume_from:
        logger.info("Resuming from checkpoint: %s", cfg.train.resume_from)
        model = PPO.load(
            cfg.train.resume_from,
            env=train_env,
            tensorboard_log=str(cfg.train.log_dir),
        )
        # Override learning rate for continued training
        model.learning_rate = lr
    else:
        model = PPO(
            policy="CnnPolicy",
            env=train_env,
            learning_rate=lr,
            n_steps=cfg.ppo.n_steps,
            batch_size=cfg.ppo.batch_size,
            n_epochs=cfg.ppo.n_epochs,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
            clip_range=cfg.ppo.clip_range,
            ent_coef=cfg.ppo.ent_coef,
            vf_coef=cfg.ppo.vf_coef,
            max_grad_norm=cfg.ppo.max_grad_norm,
            verbose=1,
            tensorboard_log=str(cfg.train.log_dir),
            seed=cfg.train.seed,
        )

    # ── Callbacks ───────────────────────────────────────────────────────
    train_callback = TrainCallback(
        save_dir=cfg.train.model_dir,
        checkpoint_freq=cfg.train.checkpoint_freq,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(cfg.train.model_dir),
        log_path=str(cfg.train.log_dir),
        eval_freq=cfg.train.eval_freq,
        n_eval_episodes=cfg.train.eval_episodes,
        deterministic=True,
        render=False,
    )

    # ── Train ───────────────────────────────────────────────────────────
    logger.info("Starting training...")
    try:
        model.learn(
            total_timesteps=cfg.train.total_timesteps,
            callback=[train_callback, eval_callback],
            log_interval=cfg.train.log_interval,
            tb_log_name="PPO_Mario",
            reset_num_timesteps=cfg.train.resume_from is None,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    finally:
        # Save final model
        final_path = cfg.train.model_dir / "final_model_ki"
        model.save(str(final_path))
        logger.info("Final model saved → %s", final_path)

        train_env.close()
        eval_env.close()

    logger.info("Training complete!")
    logger.info("  - Best model : %s/best_model.zip", cfg.train.model_dir)
    logger.info("  - Final model: %s/final_model.zip", cfg.train.model_dir)
    logger.info("  - Logs       : %s", cfg.train.log_dir)
    logger.info("Run `tensorboard --logdir %s` to visualize.", cfg.train.log_dir)


if __name__ == "__main__":
    main()
