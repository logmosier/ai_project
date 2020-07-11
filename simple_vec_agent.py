import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

# multiprocess environment

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v2')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = make_vec_env(lambda: env, n_envs=4)

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs_joypad_vec/',
                                         name_prefix='rl_model')

model = PPO2(CnnPolicy, env, verbose=1,tensorboard_log="./mario_tensorboard/")
model.learn(total_timesteps=2000000, callback=checkpoint_callback)
model.save("mario2M_simple")
