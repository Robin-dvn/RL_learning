import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def policy(observation):
    player_sum,dealer_card,usable_ace = observation
    if player_sum >= 18:
        return 0
    else:
        return 1

def check_presence(lst, s, t):
    for element in lst[:t]:  # On parcourt les i premiers éléments de la liste
        previous_s = element[0]       # On récupère le tuple a
        if s == previous_s:           # On compare b avec a
            return True
    return False

def main_montecarlo(nb_ep):
    v = np.zeros((21,11,2))
    n = np.zeros((21,11,2))
    env = gym.make('Blackjack',natural='false',sab='false')

    for _ in range(nb_ep):
        trajectoire = []
        observation,info = env.reset()
        terminated,truncated = False,False
        g= 0

        while not (terminated or truncated):
            action = policy(observation)
            observation,reward,terminated,truncated,info = env.step(action)
            trajectoire.append((observation,reward))

        for t in range(len(trajectoire)-1-1,-1,-1):
            g = g+trajectoire[t+1][1] #reward Rt+1
            #vérification que l'état n'a pas été visité avant
            s = trajectoire[t][0]
            if not check_presence(trajectoire,s,t):
                i = s[0]-1
                j = s[1]-1
                k = s[2]
                n[i,j,k]+=1
                v[i,j,k]+=(g-v[i,j,k])/n[i,j,k]

                



    env.close()
    return v




v = main_montecarlo(500000)



# Créer des coordonnées pour le meshgrid
x = np.linspace(0, 10, 11)  # 11 points
y = np.linspace(0, 20, 21)  # 21 points
X, Y = np.meshgrid(x, y)

# Créer la figure et l'axe 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Afficher la surface
ax.plot_surface(X, Y, v[:, :, 0], cmap='viridis')

# Ajouter des labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Valeurs')
ax.set_title('Affichage 3D de la matrice 21x11')

plt.show()

for i in range(1,-1,-1):
    print(i)
