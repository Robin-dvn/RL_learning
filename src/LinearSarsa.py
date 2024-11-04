import gymnasium as gym
import numpy as np
from tile_coding import IHT,tiles
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
class LinearSarsa():


    def __init__(self,alpha=0.01,nbeps = 500,eps = 0.1, eps_decay = 0, gamma = 1,nb_tilings = 16):
        self.alpha = alpha
        self.eps = eps
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.nbeps = nbeps
        self.nb_tilings = nb_tilings
        self.iht = IHT(nb_tilings**4)
        self.env = gym.make("MountainCar-v0")
        self.w = np.random.uniform(low= -0.05,high=0.05,size =nb_tilings**4)    

    def mytiles(self,x,v,a):
        x_scale = 10/(0.6+1.2)
        y_scale = 10/(0.07+0.07)
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


        for nb in tqdm(range(0,self.nbeps)):
            total_reward = 0

            s = self.env.reset()[0]
            a= self.pi(s,self.eps)
            done = False
            while not done:
                s_new,r,terminated,truncated,info = self.env.step(a) 
                a_new= self.pi(s_new,self.eps)
                tderror = r+self.evaluate(s_new,a_new)- self.evaluate(s,a)
                tiless = self.mytiles(s[0],s[1],a)

                if terminated:
                    if nb % 50 ==0:
                        print("Nombre d'épisode: ",nb, " reward: ",total_reward)
                    self.w[tile] += self.alpha*self.gamma *(r-self.evaluate(s,a))
                else:

                    for tile in tiless:
                        self.w[tile] += self.alpha * self.gamma * tderror 
                total_reward+=r
                s = s_new
                a = a_new

                done = terminated

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
ls = LinearSarsa()
ls.train()
ls.play(2)  