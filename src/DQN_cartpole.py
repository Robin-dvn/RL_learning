import gymnasium as gym
import numpy as np
import torch.nn as nn
from collections import deque
import itertools
import random
import torch.optim as optim
import torch 

BUFFER_SIZE = 50000 # taille du buffer d'expérience replay
BATCH_SIZE = 32 # taille du minibatch pris dans le buffer 
TARGET_RESET_FREQ = 1000 # au bout de C step on switch de target (hard switch) on copie les poids
MIN_BUFFERS_SIZE = 1000
EPS_START = 1. # epsilon de la epsilongready
EPS_END = 0.02 # epsilon minimum
EPS_DECAY_STEPS = 10000 # chaque step eps = eps -epsdecay
GAMMA = 0.99 # dicount factor

class Q_hat(nn.Module):
    def __init__(self,env: gym.Env):
        super().__init__()
        in_featurs = int(np.prod(env.observation_space.shape))
        self.seq = nn.Sequential(
            nn.Linear(in_featurs,64),
            nn.Tanh(),
            nn.Linear(64,env.action_space.n)
        )
    
    def forward(self,x):
        return self.seq(x)
    
    def act(self,obs):
        obs_t = torch.as_tensor(obs,dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values,dim=1)[0]
        action = max_q_index.detach().item()
        return action


env = gym.make("CartPole-v1")

online_q_hat = Q_hat(env)
target_q_hat = Q_hat(env)

target_q_hat.load_state_dict(online_q_hat.state_dict()) # initialisation du target network

episode_buffer = deque(maxlen=BUFFER_SIZE) # buffer en mode chained list deque
reward_buffer = deque([0.0],maxlen=100)

# initialisation du buffer
obs,info = env.reset()
for _ in range(MIN_BUFFERS_SIZE):
    a = env.action_space.sample()
    new_obs, r, terminated, truncated, info = env.step(a)
    done = truncated or terminated
    episode_buffer.append((obs,a,r,new_obs,done))
    obs = new_obs 

    if done:
        obs,info = env.reset()

# entrainement de DQN
obs, info = env.reset()
cumul_reward = 0
opti = optim.Adam(online_q_hat.parameters(),lr=5e-4)

for step in itertools.count():
    if len(reward_buffer)>= 100:
        if np.mean(reward_buffer)>=220:
            env = gym.make("CartPole-v1",render_mode="human")
            obs,info = env.reset()
            while True:
                a = online_q_hat.act(obs)
                new_obs,r,truncated,terminated,info = env.step(a)
                obs = new_obs
                env.render()
                if terminated or truncated:
                    obs,info = env.reset()

    epsilon = np.interp(step, [0, EPS_DECAY_STEPS],[EPS_START,EPS_END])
    random_sample = random.random()
    if random_sample <= epsilon:
        a = env.action_space.sample()
    else:
        a = online_q_hat.act(obs)
    new_obs,r,terminated,truncated,info = env.step(a)
    done = terminated or truncated
    transition = (obs,a,r,new_obs,done)
    episode_buffer.append(transition) 
    cumul_reward += r
    obs = new_obs
    if done:
        obs,info = env.reset() 
        reward_buffer.append(cumul_reward)
        cumul_reward = 0
    
    # descente de gradient
    list_of_trans =random.sample(episode_buffer,BATCH_SIZE)
    obsv = np.asarray([t[0] for t in list_of_trans])
    act = np.asarray([t[1] for t in list_of_trans])
    rew = np.asarray([t[2] for t in list_of_trans])
    new_obsv = np.asarray([t[3] for t in list_of_trans])
    dones = np.asarray([t[4] for t in list_of_trans])

    #conversion en tensor
    obsv_t  = torch.as_tensor(obsv,dtype=torch.float32)
    act_t  = torch.as_tensor(act,dtype=torch.int64).unsqueeze(-1)
    rew_t  = torch.as_tensor(rew,dtype=torch.float32).unsqueeze(-1)
    new_obsv_t  = torch.as_tensor(new_obsv,dtype=torch.float32)
    dones_t  = torch.as_tensor(dones,dtype=torch.float32).unsqueeze(-1)

    q_values = online_q_hat(obsv_t)
    action_q_values = torch.gather(input = q_values,dim = 1,index=act_t)

    q_targets = target_q_hat(new_obsv_t)
    max_q_targets = q_targets.max(dim = 1,keepdim=True)[0]

    targets = rew_t + GAMMA * max_q_targets * (1-dones_t)

    loss = nn.functional.smooth_l1_loss(action_q_values, targets)
    opti.zero_grad()
    loss.backward()
    opti.step()

    if step % TARGET_RESET_FREQ == 0:
        target_q_hat.load_state_dict(online_q_hat.state_dict())

    if step %1000 == 0:
        print()
        print('Step ',step)
        print('Avg reward ', np.mean(reward_buffer) )
    


