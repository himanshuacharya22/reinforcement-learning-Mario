"""
Frame-skip (action repeat) wrapper.

Repeats the chosen action for N frames and returns the max-pooled
observation of the last two frames (standard Atari-style preprocessing).
This reduces the effective decision frequency and speeds up training.
"""

import gym
import numpy as np


class SkipFrame(gym.Wrapper):
    """Skip N frames, summing rewards and max-pooling the last two observations."""

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        obs_shape = env.observation_space.shape
        # Buffer to hold the last two raw frames for max-pooling
        self._obs_buffer = np.zeros((2, *obs_shape), dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}

        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward

            # Shift buffer and store newest observation
            self._obs_buffer[0] = self._obs_buffer[1]
            self._obs_buffer[1] = obs

            if done:
                break

        # Max-pool over the last two frames to reduce flickering
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._obs_buffer[0] = obs
        self._obs_buffer[1] = obs
        return obs
