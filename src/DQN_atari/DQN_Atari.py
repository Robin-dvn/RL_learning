import torch.optim as optim
import gymnasium as gym 
import random
import matplotlib.pyplot as plt
from collections import deque
import ale_py
from DQN_Atari_Preprocessing import phi,phi_frame
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from tqdm import tqdm
from time import time
from datetime import timedelta
import gc
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="DQN-Atari-Breakout",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "CNN",
    "frames": 10000,
    }
)

MIN_BUFFER_SIZE = 20000
REPLAY_BUFFER_SIZE = 20000
BATCH_SIZE = 32
AGENT_HISTORY_LENGTH = 4
TARGET_UPDATE_FREQUENCY = 1000
GAMMA = 0.99
ACTION_REPEAT = 4
UPDATE_FREQUENCY = 4
LR = 0.00025
EPS_START = 1
EPS_END = 0.1
EPS_MAX_FRAME = 10000
NOOP_MAX = 30
NB_FRAME_TRAIN = 10000

device = "cuda" if torch.cuda.is_available()  else "cpu"

gym.register_envs(ale_py)


class Qnetwort(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=4,out_channels=32,kernel_size=(8,8),stride=(4,4)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(4,4),stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(1,1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7,512),
            nn.ReLU(),
            nn.Linear(512,env.action_space.n),
        )
    def forward(self,x):
        return self.seq(x)
    def act(self,obs):
        obs_t = torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0).to(device)
        q_values = self(obs_t)
        max_q_index = torch.argmax(q_values)
        action = max_q_index.detach().item()
        return action

env = gym.make("ALE/Breakout-v5",frameskip=ACTION_REPEAT,repeat_action_probability = 0.)

obs,info = env.reset()

online_net = Qnetwort(env).to(device)
target_net = Qnetwort(env).to(device)

target_net.load_state_dict(online_net.state_dict())

replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
reward_buffer = deque(maxlen=100)

## Remplissage du buffer initiale
last_m_images = deque(maxlen=AGENT_HISTORY_LENGTH+1)
last_m_images.append(obs)

init_start_time = time()
noop_compteur = 0
for _ in tqdm(range(MIN_BUFFER_SIZE)):
    while len(last_m_images) <= AGENT_HISTORY_LENGTH:
        a = env.action_space.sample()
        obs,r,ter,trun,info = env.step(a)
        last_m_images.append(obs)
    a = 0
    if noop_compteur == 30:

        while a == 0:
            a = env.action_space.sample()
    else:
        a = env.action_space.sample()
        if a == 0 : noop_compteur+=1
    

    new_obs,r,ter,trun,info = env.step(a)
    if r > 0: r = 1
    if r < 0: r = -1
    done = ter or trun

    obs_stacked = phi(last_m_images,agent_history_length=AGENT_HISTORY_LENGTH)
    last_m_images.append(new_obs)

    new_obs_stacked = np.copy(obs_stacked)
    new_obs_stacked[-1] = phi_frame(new_obs,obs)

    transition = (obs_stacked,a,r,new_obs_stacked,done)
    replay_buffer.append(transition)
    
    obs = new_obs


    if done:
        wandb.log({"noop":noop_compteur})
        noop_compteur=0
        obs,info = env.reset()
        last_m_images.clear()
        last_m_images.append(obs)

init_end_time = time()

print(f"[INFO] INitialisation du Replay Buffer de taille {MIN_BUFFER_SIZE} terminé en {timedelta(seconds=round(init_end_time - init_start_time,2))} secondes")  


print(f"[INFO] Début de l'entrainement pour {NB_FRAME_TRAIN} frames")

optimizer = optim.RMSprop(online_net.parameters(),lr=0.0001) 
cumul_reward = 0
criterion = nn.MSELoss()

for frame in tqdm(range(NB_FRAME_TRAIN)):
    while len(last_m_images) <= AGENT_HISTORY_LENGTH:
        a = env.action_space.sample()
        obs,r,ter,trun,info = env.step(a)
        last_m_images.append(obs)
    ### Act ###
    epsilon = np.interp(frame, [0, EPS_MAX_FRAME],[EPS_START,EPS_END])
    random_sample = random.random()
    if random_sample <= epsilon:
        a = env.action_space.sample()
    else:
        stacked_obs = phi(last_m_images,AGENT_HISTORY_LENGTH)
        a = online_net.act(stacked_obs)
    
    
    new_obs,r,ter,trun,info = env.step(a)
    cumul_reward+=r
    if r >0 : r = 1
    if r < 0 : r = -1 
    done = ter or trun

    obs_stacked = phi(last_m_images,agent_history_length=AGENT_HISTORY_LENGTH)
    last_m_images.append(new_obs)
    

    new_obs_stacked = np.copy(obs_stacked)
    new_obs_stacked[-1] = phi_frame(new_obs,obs)

    transition = (obs_stacked,a,r,new_obs_stacked,done)
    # removed_item = re.popleft()  # L'élément à gauche est retiré
    # del removed_item 
    replay_buffer.append(transition)
    obs = new_obs
    if done:
        wandb.log({"reward":cumul_reward})
        reward_buffer.append(cumul_reward)
        obs,info = env.reset()
        last_m_images.clear()
        last_m_images.append(obs)
        cumul_reward= 0 

    ### Learn ###
    if frame % UPDATE_FREQUENCY == 0:
        ### Sample ###
        transition_mb = random.sample(replay_buffer,BATCH_SIZE)
        observations = np.asarray([t[0] for t in transition_mb])
        actions = np.asarray([t[1] for t in transition_mb])
        rewards = np.asarray([t[2] for t in transition_mb])
        new_observations = np.asarray([t[3] for t in transition_mb])
        dones = np.asarray([t[4] for t in transition_mb])

        ### Convert to tensors/dimensions ###
        observations_t = torch.as_tensor(observations,dtype=torch.float32).to(device)
        actions_t = torch.as_tensor(actions,dtype=torch.int64).unsqueeze(-1).to(device) # pour passer de [1 2 3] à [[1] [2] [3]] (3) -> (3,1)
        rewards_t = torch.as_tensor(rewards,dtype=torch.float32).unsqueeze(-1).to(device)
        new_observations_t = torch.as_tensor(new_observations,dtype=torch.float32).to(device)
        dones_t = torch.as_tensor(dones,dtype=torch.float32).unsqueeze(-1).to(device)

        ### Compute Targets ###
        target_q_values = target_net(new_observations_t)
        max_target_values = torch.max(input=target_q_values,dim=1,keepdim=True)[0] # ca renvoie max,indices d'ou le [0] 

        online_q_values = online_net(observations_t)
        action_online_values = torch.gather(input=online_q_values,dim=1,index=actions_t)

        targets = rewards_t + GAMMA * max_target_values * (1-dones_t)

        ### Loss ###
        loss = criterion(action_online_values,targets)
        wandb.log({"loss": loss.item()})

        ### Optimizer ###
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del observations_t
        del actions_t
        del rewards_t
        del new_observations_t
        del dones_t
        del observations
        del actions
        del rewards
        del new_observations
        del dones

        torch.cuda.empty_cache()
    if frame % 500 == 0:
        gc.collect()
    ### TARGET UPDATE ###
    if (frame//4) % TARGET_UPDATE_FREQUENCY == 0:
        target_net.load_state_dict(online_net.state_dict())

    ### Log ###
    
print(f"[INFO] Entrainement terminé et l'average reward sur les 100 derniers episode est : {np.mean(reward_buffer)}")
print(reward_buffer)

torch.save(online_net.state_dict(),"Qnetworkstatedict.pth")


# print(env.observation_space.shape)
# m = 4
# last_m_images = deque(maxlen=m+1)
# last_m_images.append(obs)


# while len(last_m_images) != m+1:
#     obs,r,ter,tru,info = env.step(env.action_space.sample())
#     last_m_images.append(obs)

# first_input = phi(last_m_images)
# print(first_input.shape)
# done = False
# plt.imshow(first_input[:,:,0],cmap="gray")
# plt.show()
# while not done:
#     obs,r,ter,tru,info = env.step(env.action_space.sample())
#     # print(f"Reward : {r} \n")
#     done = tru or ter
env.close()