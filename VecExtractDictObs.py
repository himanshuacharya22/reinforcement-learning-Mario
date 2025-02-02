from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv
import numpy as np


class VecExtractDictObs(VecEnvWrapper):
    last_x_pos = 0
    same_point_count = 0

    def __init__(self, venv: VecEnv, key: str):
        self.key = key
        super().__init__(venv=venv, observation_space=venv.observation_space)
        self.info = None


    def reset(self):
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset(seed=self._seeds[env_idx])
            self._save_obs(env_idx, obs)
        # Seeds are only used once
        self._reset_seeds()
        return self._obs_from_buf()

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self):
        try:
            _state, _reward, _done, _info=self.venv.step_wait()
            self.info=_info[0]
            # print(self.info)

            _reward = self.reward(_reward)
            self.last_x_pos = self.info["x_pos"]

            return _state, _reward, _done, _info
        except Exception as e:
            print(e)

            return self.venv.step([0])

    def reward(self, _reward):
        custom_reward = _reward
        try:
            current_x_pos = self.info["x_pos"]
            # print(current_x_pos)

            if self.last_x_pos <= current_x_pos:
                self.same_point_count += 1
                if self.same_point_count > 10:
                    custom_reward -= 5
                    self.same_point_count = 0
                    # print("minus")

        except KeyError:
            pass

        return custom_reward
