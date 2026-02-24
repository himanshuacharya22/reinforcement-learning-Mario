"""
Training callbacks for Mario RL.

Provides:
  - TrainCallback: periodic checkpointing with best-model tracking and
    TensorBoard-compatible metric logging.
"""

import os
import logging
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class TrainCallback(BaseCallback):
    """
    Custom training callback that handles:
      1. Periodic model checkpointing
      2. Best-model tracking based on mean episode reward
      3. Logging episode-level metrics
    """

    def __init__(
        self,
        save_dir: str | Path,
        checkpoint_freq: int = 25_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.checkpoint_freq = checkpoint_freq
        self.best_mean_reward = -np.inf
        self._episode_rewards: list[float] = []

    def _init_callback(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        # ── Periodic checkpoint ─────────────────────────────────────────
        if self.n_calls % self.checkpoint_freq == 0:
            ckpt_path = self.save_dir / f"checkpoint_{self.n_calls}"
            self.model.save(str(ckpt_path))
            logger.info("Saved checkpoint → %s", ckpt_path)

        # ── Episode tracking ────────────────────────────────────────────
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                ep_reward = ep_info["r"]
                self._episode_rewards.append(ep_reward)

                # Log to TensorBoard
                self.logger.record("episode/reward", ep_reward)
                self.logger.record("episode/length", ep_info["l"])
                if "x_pos" in info:
                    self.logger.record("episode/x_pos", info["x_pos"])
                if "flag_get" in info:
                    self.logger.record("episode/flag_get", int(info["flag_get"]))

        # ── Best model saving ───────────────────────────────────────────
        if (
            len(self._episode_rewards) >= 10
            and self.n_calls % self.checkpoint_freq == 0
        ):
            recent = self._episode_rewards[-100:]
            mean_reward = np.mean(recent)

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_path = self.save_dir / "best_model"
                self.model.save(str(best_path))
                logger.info(
                    "New best model! mean_reward=%.2f → %s",
                    mean_reward,
                    best_path,
                )

        return True  # Continue training
