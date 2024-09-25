import gymnasium as gym
import gym
import numpy as np


policy = lambda x: np.random.randint(2)

def main_TD0(nb_ep):
    env = gym.make('WalkFive-v0') 
    v = np.zeros(env.observation_space.n)
    for t in range(nb_ep):
        s = env.reset()
        terminated,truncated = False,False
        while not (terminated):
            a = policy(s)
            news,r,terminated,_ = env.step(a)
            v[s] = v[s] + 0.01*(r+v[news]-v(s))
            s = news

    env.close()
    return v

v = main_TD0(10)