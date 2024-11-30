import torch
import torch.nn as nn
import gymnasium as gym
import ale_py
from collections import deque
import numpy as np
from DQN_Atari_Preprocessing import phi,phi_frame
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
        obs_t = torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0)
        q_values = self(obs_t)
        print(q_values)
        max_q_index = torch.argmax(q_values)
        action = max_q_index.detach().item()
        return action

env = gym.make('ALE/Breakout-v5',render_mode = "human")
qNet = Qnetwort(env)
qNet.load_state_dict(torch.load("Qnetworkstatedict.pth",map_location=torch.device('cpu'),weights_only=True))
last_m_images = deque(maxlen=5)
obs,info = env.reset()
for _ in range(1000):
    while len(last_m_images) <= 4:
        a = env.action_space.sample()
        obs,r,ter,trun,info = env.step(a)
        last_m_images.append(obs)
    
    obs_stacked = phi(last_m_images,agent_history_length=4)
    a = qNet.act(obs_stacked)
    

    new_obs,r,ter,trun,info = env.step(a)

    if r > 0: r = 1
    if r < 0: r = -1
    done = ter or trun


    last_m_images.append(new_obs)

    new_obs_stacked = np.copy(obs_stacked)
    new_obs_stacked[-1] = phi_frame(new_obs,obs)

    transition = (obs_stacked,a,r,new_obs_stacked,done)
    # replay_buffer.append(transition)
    
    obs = new_obs


    if done:
        obs,info = env.reset()
        last_m_images.clear()
        last_m_images.append(obs)

env.close()