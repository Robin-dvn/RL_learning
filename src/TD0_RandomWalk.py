import gymnasium as gym
import numpy as np
import gym_custom

policy = lambda x: np.random.randint(2)

def main_TD0(nb_ep):
    env = gym.make('GymWalk-v0',render_mode = "human") 
    v = np.zeros(env.observation_space.n)
    for t in range(nb_ep):
        s,info = env.reset()
        terminated,truncated = False,False
        while not (terminated):
            a = policy(s)
            news,r,terminated,_,info = env.step(a)
            v[s] = v[s] + 0.01*(r+v[news]-v[s])
            s = news

    env.close()
    return v

v = main_TD0(5)