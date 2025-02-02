from gym import RewardWrapper


class CustomRewardWrapper(RewardWrapper):
    last_x_pos = 0
    same_point_count = 0

    def __init__(self, _env):
        super(CustomRewardWrapper, self).__init__(_env)
        self.info = None

    def step(self, action):
        _state, _reward, _done, _info = self.env.step(action)
        self.info = _info  # Capture the info dictionary
        _reward = self.reward(_reward)  # Apply custom reward
        # print("last x pos",self.last_x_pos)
        self.last_x_pos = self.info["x_pos"]

        # print("step taken")

        return _state, _reward, _done, _info

    def reward(self, _reward):
        # Modify the reward function here
        custom_reward = _reward
        try:
            current_x_pos = self.info["x_pos"]
            # print(current_x_pos)

            if self.last_x_pos <= current_x_pos:
                self.same_point_count += 1
                if self.same_point_count > 10:
                    custom_reward -= 5
                    self.same_point_count = 0
                    print("minus")

        except KeyError:
            pass
            # print("some error")
        # print("reward given")

        return custom_reward
