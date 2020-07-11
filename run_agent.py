import gym
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv
import gym_super_mario_bros
from stable_baselines import PPO2

env_id = 'SuperMarioBros-1-1-v2'
video_folder = 'logs/videos/'
video_length = 1000

env = DummyVecEnv([lambda: gym.make(env_id)])

obs = env.reset()
model = PPO2.load("marioM")

# Record the video starting at the first step
env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix="random-agent-{}".format(env_id))

env.reset()
for _ in range(video_length + 1):
  action, _states = model.predict(obs)
  obs, _, _, _ = env.step(action)
# Save the video
env.close()