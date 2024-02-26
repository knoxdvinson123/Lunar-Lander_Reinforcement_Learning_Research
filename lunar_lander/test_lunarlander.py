import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = "LunarLander-v2"
env = gym.make(environment_name, render_mode = "human")

PPO_path = os.path.join('Training', 'Saved Models', 'PPO_Model_LunarLander')
model = PPO.load(PPO_path, env=env)

#Example of running random actions for 5 episodes and printing score--------------------------------------
episodes = 5
for episode in range(1, episodes+1):
    obs, _info = env.reset()
    done = False
    score = 0

    while not done:
      env.render()
      #action, _states = model.predict(obs)
      action, _ = model.predict(obs) #now using the trained model
      obs, reward, done, truncated, info = env.step(action)
      score += reward
    print('Episode:{} Score:{}'.format(episode, score))
#to view the tensor board to view stats
#training_log_path = os.path.join(log_path, 'PPO_2')
#tensorboard --logdir={training_log_path) (which is used in command prompt)