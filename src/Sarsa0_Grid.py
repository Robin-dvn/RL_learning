import gymnasium as gym
import gym_custom

env = gym.make("GridWorld-v0",render_mode = "human")
env.reset(seed=1)