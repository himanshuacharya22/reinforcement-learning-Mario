"""
Evaluation script - load a trained Mario agent and watch it play.

Usage:
  python evaluate.py                                          # Uses best model
  python evaluate.py --model_path models/checkpoint_50000.zip
  python evaluate.py --episodes 10
"""

import argparse
import logging

import cv2
import imageio
import numpy as np
from stable_baselines3 import PPO

from config import Config
from env_factory import make_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained Mario PPO agent")
    p.add_argument(
        "--model_path",
        type=str,
        default="models/best_model",
        help="Path to the model (without .zip extension)",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions (default: True)",
    )
    p.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        help="Use stochastic actions",
    )
    p.add_argument(
        "--no-record",
        dest="record",
        action="store_false",
        default=True,
        help="Disable recording runs to MP4 video",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        default=True,
        help="Show live preview window (default: True)",
    )
    return p.parse_args()


def _extract_frame(vec_env):
    """Return an RGB frame from the first vec env instance, if available."""
    if not hasattr(vec_env, "get_images"):
        return None

    frames = vec_env.get_images()
    if not frames:
        return None

    frame = frames[0]
    if isinstance(frame, np.ndarray) and frame.size > 0:
        return frame
    return None


def _load_model_with_fallback(model_path: str, env):
    """
    Load model with compatibility fallbacks for checkpoints coming from other machines.
    """
    try:
        # Safe-first load:
        # Colab/Linux checkpoints may serialize schedule callables that can crash
        # on Windows during deserialization before Python can catch exceptions.
        model = PPO.load(
            model_path,
            env=env,
            device="auto",
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.2,
            },
        )
        return model
    except Exception as first_err:
        logger.warning("Safe model load failed: %s", first_err)
        logger.info("Retrying load without custom_objects...")

    try:
        model = PPO.load(model_path, env=env, device="auto")
        return model
    except Exception as second_err:
        logger.error("Fallback model load also failed.")
        logger.error("  - Checkpoint path: %s", model_path)
        logger.error("  - Local SB3/Gym stack must match training stack.")
        logger.error("  - Original error: %s", second_err)
        raise


def main():
    args = parse_args()
    cfg = Config()

    logger.info("=" * 60)
    logger.info("  Super Mario Bros - Agent Evaluation")
    logger.info("=" * 60)
    logger.info("Model           : %s", args.model_path)
    logger.info("Episodes        : %d", args.episodes)
    logger.info("Deterministic   : %s", args.deterministic)
    logger.info("Live Preview    : %s", args.preview)
    logger.info("Record Video    : %s", args.record)
    logger.info("=" * 60)

    # Build environment (single env, rgb_array rendering for video/preview)
    cfg.env.num_envs = 1
    env = make_env(cfg, render_mode="rgb_array")

    # Load model
    model = _load_model_with_fallback(args.model_path, env=env)
    logger.info("Model loaded successfully!")

    if args.record:
        cfg.train.video_dir.mkdir(parents=True, exist_ok=True)

    all_rewards = []
    all_x_pos = []
    flags_gotten = 0

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        max_x = 0

        frames = []

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, infos = env.step(action)

            reward = float(rewards[0] if isinstance(rewards, np.ndarray) else rewards)
            done = bool(dones[0] if isinstance(dones, np.ndarray) else dones)
            info = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}

            total_reward += reward
            steps += 1

            if isinstance(info, dict):
                x_pos = int(info.get("x_pos", 0))
                max_x = max(max_x, x_pos)
                if info.get("flag_get", False):
                    flags_gotten += 1

            if args.record or args.preview:
                frame = _extract_frame(env)
                if frame is not None:
                    if args.preview:
                        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Super Mario Bros (Live Preview)", bgr_frame)
                        cv2.waitKey(1)

                    if args.record:
                        frames.append(frame)

        if args.record and frames:
            video_path = cfg.train.video_dir / f"eval_ep{ep}.mp4"
            imageio.mimsave(str(video_path), frames, fps=30)
            logger.info("Saved video -> %s", video_path)

        all_rewards.append(total_reward)
        all_x_pos.append(max_x)

        logger.info(
            "Episode %d/%d | Reward: %7.1f | Max X: %5d | Steps: %5d",
            ep,
            args.episodes,
            total_reward,
            max_x,
            steps,
        )

    env.close()
    if args.preview:
        cv2.destroyAllWindows()

    logger.info("-" * 60)
    logger.info("  Evaluation Summary")
    logger.info("-" * 60)
    logger.info("  Mean Reward    : %.1f +/- %.1f", np.mean(all_rewards), np.std(all_rewards))
    logger.info("  Mean Max X     : %.0f +/- %.0f", np.mean(all_x_pos), np.std(all_x_pos))
    logger.info(
        "  Levels Cleared : %d / %d (%.0f%%)",
        flags_gotten,
        args.episodes,
        100 * flags_gotten / args.episodes,
    )
    logger.info("-" * 60)


if __name__ == "__main__":
    main()
