import os
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = 'LunarLander-v2'
env = gym.make(environment_name, render_mode = "human")

#PPO_path = os.path.join('Training', 'Saved Models', 'A2C_Model_LunarLander')
#model = A2C('MlpPolicy', env, verbose=1) #defines the agent
A2C_path = os.path.join('Training', 'Saved Models', 'A2C_Model_LunarLander2')

model = A2C.load(A2C_path, env=env)

print(evaluate_policy(model, env, n_eval_episodes=10, render=True))


#env.close()


