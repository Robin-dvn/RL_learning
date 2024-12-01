import gymnasium as gym
from Wrappers import AtariWrapper

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack


import torch.nn as nn
import torch.optim as optim
import torch

import numpy as np

import ale_py
gym.register_envs(ale_py)



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

device = "cuda" if torch.cuda.is_available() else "cpu"

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

##### Gestion de l'environement #####

def make_env(env_id,seed):
    def thunk():
        env = gym.make(env_id,render_mode="rgb_array",repeat_action_probability=0.)
        env = AtariWrapper(env)
        
        return env
    return thunk

nb_envs = 1
funcs = [make_env("BreakoutNoFrameskip-v4",i) for i in range(nb_envs)]

dummy_vec_env = DummyVecEnv(funcs)
envs = VecFrameStack(dummy_vec_env,4,"last")

rp = ReplayBuffer(
    REPLAY_BUFFER_SIZE,
    dummy_vec_env.observation_space,
    dummy_vec_env.action_space,
    device,
    nb_envs,
    True,
    False
)




