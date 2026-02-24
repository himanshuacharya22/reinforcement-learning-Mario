"""
Environment factory for Super Mario Bros.

Builds the complete preprocessing pipeline:
  JoypadSpace → RewardWrapper → SkipFrame → GrayScale → Resize → VecEnv → FrameStack
"""

import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from config import Config
from wrappers import MarioRewardWrapper, SkipFrame, ResizeObservation


def _make_single_env(cfg: Config, render_mode: str | None = None):
    """Return a factory function that creates a single wrapped Mario env."""

    def _init():
        env = gym_super_mario_bros.make(cfg.env.game_id)

        # Restrict action space to COMPLEX_MOVEMENT (includes run + jump combos)
        env = JoypadSpace(env, COMPLEX_MOVEMENT)

        # Custom reward shaping
        env = MarioRewardWrapper(env, reward_cfg=cfg.reward)

        # Frame skip (repeat action for N frames, max-pool last 2)
        env = SkipFrame(env, skip=cfg.env.frame_skip)

        # Grayscale (reduces 3 channels → 1)
        env = GrayScaleObservation(env, keep_dim=True)

        # Resize to 84x84
        env = ResizeObservation(env, size=cfg.env.frame_size)

        # Convert Gym v0.25 API to Gymnasium API for SB3 v2.x compatibility
        import shimmy
        env = shimmy.openai_gym_compatibility.GymV21CompatibilityV0(env=env, render_mode=render_mode)

        return env

    return _init


def make_env(cfg: Config | None = None, render_mode: str | None = None):
    """
    Build the full vectorized environment pipeline.

    Args:
        cfg: Configuration object. Uses defaults if None.
        render_mode: Optional render mode (e.g., "human" for evaluation).

    Returns:
        A vectorized, frame-stacked environment ready for PPO training.
    """
    if cfg is None:
        cfg = Config()

    env_fns = [_make_single_env(cfg, render_mode) for _ in range(cfg.env.num_envs)]

    if cfg.env.use_subprocess and cfg.env.num_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # Stack N frames along the channel dimension
    vec_env = VecFrameStack(vec_env, n_stack=cfg.env.frame_stack, channels_order="last")

    # Transpose to channels-first for CnnPolicy: (N, H, W, C) → (N, C, H, W)
    vec_env = VecTransposeImage(vec_env)

    return vec_env
