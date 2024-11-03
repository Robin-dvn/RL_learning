import gymnasium as gym
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


def epsilon_greedy_policy(s,Q,eps):
    if random.uniform(0,1) <eps:
        action = np.random.choice(len(Q[s]))
    else:
        action = np.argmax(Q[s])
    return int(action)



def train_QLearning(nbep,alp,eps,epsdecay,policy):

    env = gym.make("FrozenLake-v1",map_name = "8x8",is_slippery = False)
    Q = dict([(x,np.zeros(env.action_space.n)) for x in range(env.observation_space.n)])
    nbsuc = 0
    tr=0
    for ep in tqdm(range(nbep)): 

        s,info = env.reset()
        eps = max(0,eps-epsdecay)

        

        done = False
        step = 0
        while not done:
            step+=1
            a = policy(s,Q,eps)
            s_new,r,terminated,truncated,info = env.step(a)
            
            if r==1: 
                nbsuc+=1
                
            if eps==0: alpha = 0.0001
            #augmentation de la trace d'élégibilité de la paire (s,a)

            #tirage de la nouvelle action
            a_new = np.argmax(Q[s_new])

            Q[s][a] = Q[s][a] + alp*(r+0.9*np.max(Q[s_new]) - Q[s][a])

            
            if (terminated or truncated): done = True
            if step >=198: tr+=1
            s = s_new
    env.close()
    return Q,nbsuc,tr

Q,nbsuc,tr = train_QLearning(20000,0.9,1,0.00005,epsilon_greedy_policy)
