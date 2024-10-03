import gymnasium as gym
import numpy as np
import gym_custom
import matplotlib.pyplot as plt
#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
def rmse(v1,v2):
    m = np.mean((v1-v2)**2)
    return np.sqrt(m)


policy = lambda x: np.random.randint(2)
#pip install --upgrade git+https://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@github.com/Robin-dvn/gym-custom-exemples.git
def train_TDl_On(nb_ep,lambd,alpha,history):
    V_true = np.linspace(0, 1, 21)
    env = gym.make('GymWalk19-v0') 
    v = np.zeros(env.observation_space.n)
    #v[1:-1] = 0.5
    e = np.zeros(env.observation_space.n)
    for t in range(nb_ep):
        e = np.zeros(env.observation_space.n)
        s,info = env.reset()
        terminated,truncated = False,False
        while not (terminated):
            a = policy(s)
            news,r,terminated,_,info = env.step(a)
            e[s] += 1

            for i in range(v.size-2):
                v[i+1] = v[i+1] + alpha*(r+v[news]-v[s])*e[i+1]
                e[i+1] = lambd*e[i+1]
                
            s = news  
    
    history[lambd]["rmse"].append(float(rmse(V_true[1:-1],v[1:-1])))
    history[lambd]["alphas"].append(alpha)
    env.close()
    return history

def main(nb_ep):
    
    alphas = [0.001,0.005,0.01,0.025,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
    lambdas = [0.,0.4,0.8,0.9,0.95,0.975,0.99,1]
    history = {}
    plt.figure(figsize=(10,6))
    for l in lambdas:
        history[l] = {}
        history[l]["rmse"] = []
        history[l]["alphas"] = []
        for alpha in alphas:
            history = train_TDl_On(nb_ep,l,alpha,history)
        plt.plot(history[l]["alphas"],history[l]["rmse"])
        plt.ylim(0,0.55)
    plt.show()
    return history




v= train_TDl_On(10,0.,0.8)
his = main(10)
plt.figure(figsize=(10,6))
plt.plot(his[0.8]["alphas"],his[0.8]["rmse"])
plt.ylim(0,0.55)
plt.show()
