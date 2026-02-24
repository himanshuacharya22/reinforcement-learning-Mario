"""
Observation resizing wrapper.

Down-scales observations to a smaller spatial resolution (default 84×84)
to reduce the input dimensionality for the CNN policy, speeding up both
forward passes and learning.
"""

import gym
import numpy as np
from gym.spaces import Box

try:
    import cv2
except ImportError:
    cv2 = None


class ResizeObservation(gym.ObservationWrapper):
    """Resize observations to (size × size) using area interpolation."""

    def __init__(self, env: gym.Env, size: int = 84):
        super().__init__(env)
        self._size = size

        old_shape = self.observation_space.shape
        if len(old_shape) == 3:
            # (H, W, C)  →  (size, size, C)
            new_shape = (size, size, old_shape[2])
        else:
            # (H, W)  →  (size, size)
            new_shape = (size, size)

        self.observation_space = Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if cv2 is not None:
            resized = cv2.resize(
                obs, (self._size, self._size), interpolation=cv2.INTER_AREA
            )
        else:
            # Fallback: nearest-neighbor via numpy (slower but no dependency)
            h, w = obs.shape[:2]
            row_idx = (np.arange(self._size) * h // self._size).astype(int)
            col_idx = (np.arange(self._size) * w // self._size).astype(int)
            resized = obs[np.ix_(row_idx, col_idx)]

        # Preserve channel dimension if it was present
        if len(self.observation_space.shape) == 3 and resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        return resized.astype(np.uint8)
