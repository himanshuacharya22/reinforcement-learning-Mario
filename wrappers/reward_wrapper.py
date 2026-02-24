"""
Multi-signal reward shaping wrapper for Super Mario Bros.

Shapes the reward using several game-state signals to accelerate learning:
  - Forward progress (x-position delta)
  - Score delta
  - Time penalty (per step)
  - Death penalty
  - Idle penalty (if x-position doesn't change)
  - Level clear bonus
"""

import gym
import numpy as np

from config import RewardConfig


class MarioRewardWrapper(gym.Wrapper):
    """Wraps the Mario environment to provide a shaped reward signal."""

    def __init__(self, env: gym.Env, reward_cfg: RewardConfig | None = None):
        super().__init__(env)
        self.cfg = reward_cfg or RewardConfig()

        # State tracking
        self._prev_x_pos: int = 0
        self._prev_score: int = 0
        self._idle_steps: int = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._prev_x_pos = 0
        self._prev_score = 0
        self._idle_steps = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = self._shape_reward(reward, done, info)
        return obs, reward, done, info

    def _shape_reward(self, base_reward: float, done: bool, info: dict) -> float:
        """Compute the shaped reward from multiple game-state signals."""
        shaped = 0.0

        current_x = info.get("x_pos", self._prev_x_pos)
        current_score = info.get("score", self._prev_score)

        # 1. Forward progress — proportional to distance gained
        x_delta = current_x - self._prev_x_pos
        shaped += x_delta * self.cfg.forward_progress_scale

        # 2. Score delta — small bonus for killing enemies / collecting coins
        score_delta = current_score - self._prev_score
        if score_delta > 0:
            shaped += score_delta * self.cfg.score_delta_scale

        # 3. Time penalty — encourage faster completion
        shaped += self.cfg.time_penalty

        # 4. Idle penalty — escalates when Mario stands still
        if x_delta <= 0:
            self._idle_steps += 1
            if self._idle_steps > self.cfg.idle_threshold:
                shaped += self.cfg.idle_penalty
        else:
            self._idle_steps = 0

        # 5. Death / level completion
        if done:
            if info.get("flag_get", False):
                shaped += self.cfg.level_clear_bonus
            elif info.get("life", 2) < 2:
                shaped += self.cfg.death_penalty

        # Update state for next step
        self._prev_x_pos = current_x
        self._prev_score = current_score

        return np.float32(shaped)
