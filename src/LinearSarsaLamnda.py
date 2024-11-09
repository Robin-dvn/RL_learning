import gymnasium as gym
import numpy as np
from tile_coding import IHT,tiles
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
class LinearSarsaLambda():


    def __init__(self,alpha=0.01,lambd = 0.1,nbeps = 20,eps = 0.1, eps_decay = 0, gamma = 1,nb_tilings = 5,tile_grid = 9):
        self.alpha = alpha
        self.eps = eps
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.lambd = lambd
        self.tile_grid = tile_grid
        self.nbeps = nbeps
        self.nb_tilings = nb_tilings
        self.Iht_size = tile_grid *tile_grid * nb_tilings * 3
        self.iht = IHT(self.Iht_size)
        self.env = gym.make("MountainCar-v0")
        self.w = np.zeros(self.Iht_size)  
        self.e = np.zeros(self.Iht_size) 

        self.steps_per_eps = np.zeros(nbeps)

    def mytiles(self,x,v,a):
        x_scale = self.tile_grid/(0.6+1.2)
        y_scale = self.tile_grid/(0.07+0.07)
        return tiles(self.iht,self.nb_tilings,list((x*x_scale,v*y_scale)),[a])
    
    def evaluate(self,s,a):
        x,v = s
        tiless = self.mytiles(x,v,a)
        ev = 0 
        for tile in tiless:
            ev += self.w[tile]
        return ev
    
    def pi(self,s,eps):
        if random.uniform(0,1) < eps:
            a = self.env.action_space.sample()
            return a
        else:
            a = np.argmax(np.r_[self.evaluate(s,0),self.evaluate(s,1),self.evaluate(s,2)])
            return a


    
    def train(self):


        for nb in range(0,self.nbeps):
            total_reward = 0
            total_steps = 0
            self.e = np.zeros(self.Iht_size)
            s = self.env.reset()[0]
            a= self.pi(s,self.eps)
            done = False
            while not done:
                self.e *= self.lambd *self.gamma
                tiless = self.mytiles(s[0],s[1],a)

                s_new, r, terminated,truncated,info = self.env.step(a)
                
                delta = r
                for tile in tiless:
                    delta-= self.w[tile]
                    self.e[tile] = 1
                
                if terminated :
                    if nb %50 ==0:
                        pass
                        # print(f"Episode {nb} est terminé avec un totalreward de {total_reward}")
                    self.w = self.w + delta * self.alpha * self.e
                    break
                
                a_new = self.pi(s,self.eps)
                
                self.w += self.alpha * (delta + self.gamma * self.evaluate(s_new,a_new)) * self.e
                total_reward-=1
                total_steps+=1

                s = s_new
                a = a_new
            self.steps_per_eps[nb] = total_steps
            self.eps = max(0,self.eps-self.eps_decay)

        self.env.close()
        
    def play(self,nbeps):
        env = gym.make("MountainCar-v0",render_mode = "human") 
        for nb in range(nbeps):
            done = False
            s = np.array(env.reset()[0])
            a = self.pi(s,0)
            done = False
            while not done:
                s_new,r,terminated,truncated,info = env.step(a) 
                a_new= self.pi(s_new,0)
                s = s_new
                a = a_new
                done = terminated  
        env.close()


lsl = LinearSarsaLambda(lambd=1.,alpha=0.1/5)
lsl.train()
lsl.play(2)  

class Simulator():
    
    def __init__(self):
        # self.lambdas = np.r_[0.95,0.9,0.8,0.7,0.4,0.]
        self.alphas = np.r_[0.1/5,0.3/5,0.5/5,0.7/5,0.9/5]
        self.lambdas = np.r_[0.9]
        # self.alpha = np.r_[0.5]
        self.result = {}
    
    def simulate(self,nbeps,runs):
        for lam in self.lambdas:
            self.result[lam] = np.zeros(len(self.alphas))
            for (i,alpha) in enumerate(self.alphas):
                print(f"lambda : {lam}, alpha:{alpha}")
                sum_mean_step_per_eps = 0
                for run in tqdm(range(runs)):
                    lsl = LinearSarsaLambda(eps=0,lambd=lam,alpha=alpha,nb_tilings=5,tile_grid=9,nbeps=nbeps)
                    lsl.train()
                    sum_mean_step_per_eps += np.mean(lsl.steps_per_eps)
                final_mean = sum_mean_step_per_eps/runs
                self.result[lam][i] = final_mean
    
    def plot(self):
        plt.figure(figsize=(5,5))
        colors = ["black","red","blue","yellow","purple","green","brown","pink"]
        for (i,lam) in enumerate(self.lambdas):
            step_per_eps  = self.result[lam]
            plt.plot(self.alphas*5, step_per_eps,color = colors[i],label = str(self.lambdas[i]))
        plt.xlabel("alpha * 5")
        plt.ylabel("Number of step per episode avergad over the firqt 20 episond and 30 runs")
        plt.ylim(400,800)
        plt.title("number of step per episod w.r.t lambda and alpha")
        plt.show()

sim = Simulator()
sim.simulate(20,30)
sim.plot() 
results = sim.result 
lambdas = np.r_[0.95,0.9,0.8,0.7,0.4,0.]
alphas = np.r_[0.1/5,0.3/5,0.5/5,0.7/5,0.9/5]
plt.figure(figsize=(5,5))
colors = ["black","red","blue","yellow","purple","green","brown","pink"]
for (i,lam) in enumerate(lambdas):
    step_per_eps  = results[lam]
    print(f"lambda is: lam")
    plt.plot(alphas*5, step_per_eps,color = colors[i],label = str(lambdas[i]))
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("Multiple Curves with Annotations and Special Markers")

plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Création des données
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x) / 10  # Divisé pour ne pas être trop grand

# Création de la figure
plt.figure(figsize=(10, 6))

# Tracé des courbes avec des couleurs, des marqueurs et des annotations
plt.plot(x, y1, color='blue', marker='s', label='sin(x)')
plt.plot(x, y2, color='green', marker='o', label='cos(x)')
plt.plot(x, y3, color='red', marker='^', label='tan(x)')

# Annotation pour chaque courbe
plt.text(5, np.sin(5), "sin(x)", color='blue', fontsize=12, ha='right')
plt.text(5, np.cos(5), "cos(x)", color='green', fontsize=12, ha='right')
plt.text(5, np.tan(5)/10, "tan(x)", color='red', fontsize=12, ha='right')

# Titres des axes
plt.xlabel("X axis")
plt.ylabel("Y axis")

# Titre principal
plt.title("Multiple Curves with Annotations and Special Markers")

# Affichage de la grille pour plus de lisibilité
plt.grid(True)

# Affichage
plt.show()
