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

def train_SARSAlambda(nbep,alp,lam,eps,epsdecay,policy):
    env = gym.make("FrozenLake-v1",map_name = "8x8",is_slippery = False)
    Q = dict([(x,np.zeros(env.action_space.n)) for x in range(env.observation_space.n)])
    nbsuc = 0
    tr=0
    for ep in tqdm(range(nbep)): 
        #if ep%1000== 0: ep*=0.5
        s,info = env.reset()
        eps = max(0,eps-epsdecay)
        a = policy(s,Q,eps)
        E = dict([(x,np.zeros(env.action_space.n)) for x in range(env.observation_space.n)])
        done = False
        step = 0
        while not done:
            step+=1
            s_new,r,terminated,truncated,info = env.step(a)
            
            if r==1: 
                nbsuc+=1
                
            if eps==0: alp = 0.0001
            #augmentation de la trace d'élégibilité de la paire (s,a)
            E[s][a]= E[s][a]+1

            #tirage de la nouvelle action
            a_new = policy(s_new,Q,eps)
            if lam !=0.:
                for state in range(env.observation_space.n):
                    for action in range(env.action_space.n):

                        Q[state][action] = Q[state][action] + alp*(r+Q[s_new][a_new] - Q[s][a])*E[state][action]
                        #réduction de la trace d'élégibilité
                        E[state][action] = E[state][action]*lam
            else:
                Q[s][a] = Q[s][a] + alp*(r+0.9*Q[s_new][a_new] - Q[s][a])
            if (terminated or truncated): done = True
            if step >=150: tr+=1
            s = s_new
            a = a_new
    env.close()
    return Q,nbsuc,tr

def greedy_play(Q,nbep):
    env = gym.make("FrozenLake-v1",map_name="8x8",is_slippery=False,render_mode="human")
    for ep in range(nbep):

        done = False
    
        s,info = env.reset()
        while not done:
            a = int(np.argmax(Q[s]))
            s,r,terminated,truncated,info = env.step(a)
            if (terminated or truncated): done = True

    env.close()

def printresAlpha(res,lambd,eps):
    nbsucs = []
    alphas = [0.1,0.3,0.5,0.7,0.9]
    for alpha in alphas:
        nbsucs.append(res[lambd][alpha][eps]["nbsuc"])
    plt.figure()
    plt.plot(alphas,nbsucs)
    plt.show()
    

    


def main():
    lambdas = [0.]
    alphas = [0.1,0.3,0.5,0.7,0.9]
    epsdecays = [0.0002]
    results = {}

    for lambd in tqdm(lambdas):
        results[lambd] = {}
        for alpha in alphas:
            results[lambd][alpha] = {}
            for epsdecay in epsdecays:
                results[lambd][alpha][epsdecay] = {}
                Q,nbsuc = train_SARSAlambda(5000,alpha,lambd,1,epsdecay,epsilon_greedy_policy)
                results[lambd][alpha][epsdecay]["Q"] = Q
                results[lambd][alpha][epsdecay]["nbsuc"] = nbsuc
    
    
    return results
    
Q,nbsuc,tr = train_SARSAlambda(200000,0.9,0.,1,0.000005,epsilon_greedy_policy)
print(nbsuc) 
greedy_play(Q,9)
        
# results = main()
# printresAlpha(results,0.,0.0002)