import gymnasium as gym
import numpy as np
import gym_custom
#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

policy = lambda x: np.random.randint(2)
#pip install --upgrade git+https://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@github.com/Robin-dvn/gym-custom-exemples.git

def main_TDlambda(nb_ep,lambd):
    env = gym.make('GymWalk-v0') 
    v = np.zeros(env.observation_space.n)
    e = np.zeros(env.observation_space.n)
    for t in range(nb_ep):
        e = np.zeros(env.observation_space.n)
        s,info = env.reset()
        terminated,truncated = False,False
        while not (terminated):
            a = policy(s)
            news,r,terminated,_,info = env.step(a)
            e[news] += 1

            for i in range(v.size):
                v[s] = v[s] + 0.3*(r+v[news]-v[s])*e[news]
                e[i] = lambd*e[i]
                
            s = news       
    env.close()
    return v,e

v,e = main_TDlambda(100000,0.3)