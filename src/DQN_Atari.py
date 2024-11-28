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
TARGET_NETWORK_UPDATE_FREQUENCY = 10000
GAMMA = 0.99
ACTION_REPEAT = 4
UPDATE_FREQUENCY = 4
LR = 0.00025
GRAD_MOMENTUM = 0.95
SQR_GRAD_MOMENTUM = 0.95


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