"""
Evaluation script — load a trained Mario agent and watch it play.

Usage:
  python evaluate.py                                          # Uses best model
  python evaluate.py --model_path models/checkpoint_50000.zip
  python evaluate.py --episodes 10
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
from stable_baselines3 import PPO

from config import Config
from env_factory import make_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-18s │ %(levelname)-7s │ %(message)s",
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
        help="Disable recording runs to MP4 video",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        default=True,
        help="Show live preview window (default: True)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    logger.info("=" * 60)
    logger.info("  Super Mario Bros — Agent Evaluation")
    logger.info("=" * 60)
    logger.info("Model           : %s", args.model_path)
    logger.info("Episodes        : %d", args.episodes)
    logger.info("Deterministic   : %s", args.deterministic)
    logger.info("Live Preview    : %s", args.preview)
    logger.info("Record Video    : %s", args.record)
    logger.info("=" * 60)

    # Build environment (single env, rgb_array rendering for video)
    cfg.env.num_envs = 1
    env = make_env(cfg, render_mode="rgb_array")

    # Load model
    model = PPO.load(args.model_path, env=env)
    logger.info("Model loaded successfully!")

    if args.record:
        cfg.train.video_dir.mkdir(parents=True, exist_ok=True)

    # ── Play episodes ───────────────────────────────────────────────────
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
            obs, reward, done, info = env.step(action)
            total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            steps += 1

            # Track progress
            if isinstance(info, list) and len(info) > 0:
                x_pos = info[0].get("x_pos", 0)
                max_x = max(max_x, x_pos)
                if info[0].get("flag_get", False):
                    flags_gotten += 1

            # We must call render() to get the frame if previewing or recording
            if args.record or args.preview:
                frame = env.render()
                if isinstance(frame, np.ndarray):
                    # Gymnasium/Shimmy returns RGB format.
                    # OpenCV expects BGR format.
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    if args.preview:
                        cv2.imshow("Super Mario Bros (Live Preview)", bgr_frame)
                        # Wait 1ms so OpenCV can process window events
                        cv2.waitKey(1)
                        
                    if args.record:
                        # Append the original RGB frame for imageio
                        frames.append(frame)

        if args.record and frames:
            video_path = cfg.train.video_dir / f"eval_ep{ep}.mp4"
            # Frames are often smaller than native due to scaling wrappers, 
            # save them directly to avoid scaling artifacts.
            imageio.mimsave(str(video_path), frames, fps=30)
            logger.info("Saved video \u2192 %s", video_path)

        all_rewards.append(total_reward)
        all_x_pos.append(max_x)

        logger.info(
            "Episode %d/%d │ Reward: %7.1f │ Max X: %5d │ Steps: %5d",
            ep, args.episodes, total_reward, max_x, steps,
        )

    env.close()
    if args.preview:
        cv2.destroyAllWindows()

    # ── Summary ─────────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("  Evaluation Summary")
    logger.info("─" * 60)
    logger.info("  Mean Reward    : %.1f ± %.1f", np.mean(all_rewards), np.std(all_rewards))
    logger.info("  Mean Max X     : %.0f ± %.0f", np.mean(all_x_pos), np.std(all_x_pos))
    logger.info("  Levels Cleared : %d / %d (%.0f%%)", flags_gotten, args.episodes, 100 * flags_gotten / args.episodes)
    logger.info("─" * 60)


if __name__ == "__main__":
    main()
