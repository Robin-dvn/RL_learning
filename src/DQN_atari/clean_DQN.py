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


NB_VEC_ENVS = 1
MIN_BUFFER_SIZE = 10000
REPLAY_BUFFER_SIZE = 20000
BATCH_SIZE = 32
TARGET_UPDATE_FREQUENCY = 2000
UPDATE_FREQUENCY = 4
GAMMA = 0.99
LR = 0.0001
EPS_START = 1
EPS_END = 0.1
EPS_MAX_FRAME = 100000
NOOP_MAX = 30
NB_FRAME_TRAIN = 30000

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


##### Gestion de l'environement #####

def make_env(env_id,seed):
    def thunk():
        env = gym.make(env_id,render_mode="rgb_array",repeat_action_probability=0.)
        env = AtariWrapper(env,max_noops=NOOP_MAX)

        
        return env
    return thunk

if __name__ == "__main__":

    wandb.init(
        project="DQN-Atari-Brekout",
        config={
            "learning_rate":LR,
            "Number_envs": NB_VEC_ENVS,
            "Total_timestep": NB_FRAME_TRAIN,
            "Replay_buffer_size": REPLAY_BUFFER_SIZE,
            "Min_replay_buffer_size":MIN_BUFFER_SIZE,
            "Batch_size":BATCH_SIZE,
            "Target_update_frequency":TARGET_UPDATE_FREQUENCY,
            "seed":1,
        }

    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    funcs = [make_env("BreakoutNoFrameskip-v4",i) for i in range(NB_VEC_ENVS)]

    dummy_vec_env = DummyVecEnv(funcs)
    envs = VecFrameStack(dummy_vec_env,4,"first")

    rp = ReplayBuffer(
        REPLAY_BUFFER_SIZE,
        envs.observation_space,
        envs.action_space,
        device,
        NB_VEC_ENVS,
        True,
        False
    )


    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic =True


    online_net = Qnetwort(envs).to(device)
    target_net = Qnetwort(envs).to(device)
    target_net.load_state_dict(online_net.state_dict())

    start_time = time.time()
    obs = envs.reset()

    criterion = torch.nn.MSELoss()
    opti = optim.Adam(online_net.parameters(),lr=LR)

    for frame in tqdm(range(NB_FRAME_TRAIN)):

        epsilon = np.interp(frame, [0, EPS_MAX_FRAME],[EPS_START,EPS_END])
        random_sample = random.random()
        if random_sample <= epsilon:
            actions = np.array([envs.action_space.sample()  for _ in range(NB_VEC_ENVS)])
        else:
            q_values = online_net(torch.tensor(obs,dtype=torch.float32).to(device))
            actions = torch.argmax(q_values,dim=1).detach().cpu().numpy()
    
        new_obs,rewards,dones,infos = envs.step(actions)
        rp.add(obs,new_obs,actions,rewards,dones,infos)
        
        obs = new_obs

        for info in infos:
            if "terminal_observation" in info.keys():
                ep = info["episode"]
                wandb.log({"reward":ep["r"],"time":ep["t"]})

        if frame > MIN_BUFFER_SIZE:
            
        

            if frame % UPDATE_FREQUENCY == 0:
                transitions = rp.sample(BATCH_SIZE)
                with torch.no_grad():
                    target_max,_ = target_net(transitions.next_observations).max(dim=1)
                    td_target = transitions.rewards.flatten() + GAMMA*target_max* (1-transitions.dones.flatten())
                old_max = online_net(transitions.observations).gather(1,transitions.actions).squeeze()
                loss = criterion(old_max,td_target)

                opti.zero_grad()
                loss.backward()
                opti.step()
                wandb.log({"loss":loss.item()})
            if frame % TARGET_UPDATE_FREQUENCY == 0 :
                target_net.load_state_dict(online_net.state_dict())
        

    process = psutil.Process(os.getpid())
    print(f"RAM utilisée après remplissage du buffer : {process.memory_info().rss / 1e9} GB")



