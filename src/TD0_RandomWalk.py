import gymnasium as gym
import numpy as np
import gym_custom
import matplotlib.pyplot as plt

def rmse(v1,v2):
    m = np.mean((v1-v2)**2)
    return np.sqrt(m)

policy = lambda x: np.random.randint(2)
vtrue = np.array([0,1/6,1/3,1/2,2/3,5/6,0])


def train_TD0(nb_ep):
    history = {"rmse_alpha": {}}   
    for alpha in [0.05,0.1,0.15]:
        history["rmse_alpha"][alpha] = []
        env = gym.make('GymWalk-v0') 
        v = np.zeros(env.observation_space.n)
        v[1:-1] = 0.5
        for t in range(nb_ep):
            s,info = env.reset()
            terminated,truncated = False,False
            while not (terminated):
                a = policy(s)
                news,r,terminated,_,info = env.step(a)
                v[s] = v[s] + alpha*(r+v[news]-v[s])
                
                s = news
            history["rmse_alpha"][alpha].append(float(rmse(vtrue[1:-1],v[1:-1])))
        env.close()
        print(v)
    return history




def main(nb_ep):
    his = train_TD0(nb_ep)

    # Créer le plot
    plt.figure(figsize=(10, 6))

    # Parcourir les alphas et les courbes RMSE associées
    for alpha, rmse_values in his["rmse_alpha"].items():
        plt.plot(rmse_values, label=f'alpha={alpha}')

    # Ajouter un titre et des labels
    plt.title('Courbes RMSE pour différents alphas')
    plt.xlabel('Itération')
    plt.ylabel('RMSE')
    plt.legend()

    # Afficher le plot
    plt.show()
    return his

his = main(200)

    