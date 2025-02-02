import gym_super_mario_bros
from gym import Env
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from CallBackClass import TrainAndLoggingCallback
from VecExtractDictObs import VecExtractDictObs


# Initialize the environment
Env.render_mode = "human"
env = gym_super_mario_bros.make("SuperMarioBros-v0",new_step_api=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
# env = CustomRewardWrapper(env)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecExtractDictObs(env, key="obs")
env = VecFrameStack(env, 4, channels_order="last")

state = env.reset()
print(SIMPLE_MOVEMENT)
CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

print(env.action_space.sample())
# env.step(0)
model = PPO.load(r"best_model_100000.zip", env=env, tensorboard_log=LOG_DIR)
print("Model loaded successfully!")
done = True
for step in range(10000):
    if done:
        env.reset()
    action, _ = model.predict(state)

    state, reward, done, info = env.step(action)
    # print("pp")

    # print(state, reward, done, info)
    env.render()
env.close()





