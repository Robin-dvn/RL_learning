import torch 
import torch.nn as nn

import gymnasium as gym
# from same_wrapper import AtariWrapper
from Wrappers import AtariWrapper

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.monitor import Monitor

# from stable_baselines3.common.atari_wrappers import AtariWrapper


import torch.nn as nn
import torch.optim as optim
import torch

import numpy as np
from tqdm import tqdm
import random
import time
import psutil
import os
import wandb

import ale_py
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
        return self.seq(x/255)


def make_env(env_id,seed):
    def thunk():
        env = gym.make(env_id,render_mode="human",repeat_action_probability=0.)
        env = AtariWrapper(env,max_noops=30)

        
        return env
    return thunk
    

if __name__ == "__main__":
    NB_FRAME_PLAY = 10000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    funcs = [make_env("BreakoutNoFrameskip-v4",i) for i in range(1)]

    dummy_vec_env = DummyVecEnv(funcs)
    envs = VecFrameStack(dummy_vec_env,4,"first")

    model = Qnetwort(envs)
    model.load_state_dict(torch.load("qnetparameters.pth",map_location=torch.device('cpu'),weights_only = True))
    
    obsv = envs.reset()
    for frame in range(NB_FRAME_PLAY):
        epsilon = 0.1
        random_sample = random.random()
        if random_sample <= epsilon:
            actions = np.array([envs.action_space.sample()  for _ in range(1)])
        else:
            q_values = model(torch.tensor(obsv,dtype=torch.float32).to(device))
            actions = torch.argmax(q_values,dim=1).detach().cpu().numpy()

        new_obs,rewards,dones,infos = envs.step(actions)
        obsv = new_obs


