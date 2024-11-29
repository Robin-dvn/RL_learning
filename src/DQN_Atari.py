import gymnasium as gym 
import matplotlib.pyplot as plt
from collections import deque
import ale_py
from DQN_Atari_Preprocessing import phi
import torch
import torch.nn as nn
import torch.functional as F

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
            nn.Linear(512,env.action_space.shape),
        )
    def forward(self,x):
        return self.seq(x)
    def act(self,obs):
        pass



gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5",render_mode="human")
obs,info = env.reset()

print(env.observation_space.shape)
m = 4
last_m_images = deque(maxlen=m+1)
last_m_images.append(obs)


while len(last_m_images) != m+1:
    obs,r,ter,tru,info = env.step(env.action_space.sample())
    last_m_images.append(obs)

first_input = phi(last_m_images)
print(first_input.shape)
done = False
plt.imshow(first_input[:,:,0],cmap="gray")
plt.show()
while not done:
    obs,r,ter,tru,info = env.step(env.action_space.sample())
    # print(f"Reward : {r} \n")
    done = tru or ter
env.close()