import gymnasium as gym 
import matplotlib.pyplot as plt
from collections import deque
import ale_py
from DQN_Atari_Preprocessing import phi,phi_frame
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from tqdm import tqdm

MIN_BUFFER_SIZE = 50000
REPLAY_BUFFER_SIZE = 1000000
BATCH_SIZE = 32
AGENT_HISTORY_LENGTH = 4
TARGET_UPDATE_FREQUENCY = 10000
GAMMA = 0.99
ACTION_REPEAT = 4
UPDATE_FREQUENCY = 4
LR = 0.00025
GRAD_MOMENTUM = 0.95
SQR_GRAD_MOMENTUM = 0.95
MIN_SQR_GRAD = 0.01
EPS_START = 1
EPS_END = 0.1
EPS_MAX_FRAME = 1000000
NOOP_MAX = 30

gym.register_envs(ale_py)


class Qnetwort(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()
        seq = nn.Sequential(
            nn.Conv2d(in_channels=4,out_channels=32,kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,env.action_space.n),
        )
    def forward(self,x):
        return self.seq(x)
    def act(self,obs):
        pass

env = gym.make("ALE/Breakout-v5",frameskip=ACTION_REPEAT,repeat_action_probability = 0.)

obs,info = env.reset()

online_net = Qnetwort(env)
target_net = Qnetwort(env)
target_net.load_state_dict(online_net.state_dict())

replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
reward_buffer = deque(maxlen=100)

## Remplissage du buffer initiale
last_m_images = deque(maxlen=AGENT_HISTORY_LENGTH+1)
last_m_images.append(obs)


for _ in tqdm(range(MIN_BUFFER_SIZE)):
    while len(last_m_images) <= AGENT_HISTORY_LENGTH:
        a = env.action_space.sample()
        obs,r,ter,trun,info = env.step(a)
        last_m_images.append(obs)
    
    a = env.action_space.sample()
    

    new_obs,r,ter,trun,info = env.step(a)
    if r > 0: r = 1
    if r < 0: r = -1
    done = ter or trun

    obs_stacked = phi(last_m_images,agent_history_length=AGENT_HISTORY_LENGTH)
    last_m_images.append(new_obs)

    new_obs_stacked = np.copy(obs_stacked)
    new_obs_stacked[:,:,-1] = phi_frame(new_obs,obs)

    transition = (obs_stacked,a,r,new_obs_stacked,done)
    replay_buffer.append(transition)
    
    obs = new_obs


    if done:
        obs,info = env.reset()
        last_m_images.clear()
        last_m_images.append(obs)



print(len(replay_buffer))

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