from gymnasium import spaces
from typing import Dict, SupportsFloat

import wandb
import gymnasium as gym
import numpy as np
import numpy as np
import cv2
from stable_baselines3.common.type_aliases import AtariResetReturn, AtariStepReturn
from stable_baselines3.common.monitor import Monitor
cv2.ocl.setUseOpenCL(False)
from typing import SupportsFloat


class MaxAndSkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    def __init__(self, env:gym.Env ,frame_skip:int =4):
        super().__init__(env)
        self._frame_skip = frame_skip
        self._obs = np.zeros((2,*env.observation_space.shape),dtype=env.observation_space.dtype)
    def step(self, action: int) -> AtariStepReturn:

        total_reward = 0.
        terminated = truncated=False
        for i in range(self._frame_skip):
            obs,r,terminated,truncated,info = self.env.step(action)
            if self._frame_skip - (i+1) == 0:
                self._obs[0] = obs
            if self._frame_skip - (i+1) == 1:
                self._obs[1] = obs
            total_reward+=float(r)
            done = terminated or truncated
            if done:
                break
        max_frame = self._obs.max(axis=0)
        return max_frame,total_reward,terminated,truncated,info
    
class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env:gym.Env):
        super().__init__(env)

    def reward(self, reward: SupportsFloat):

        return np.sign(float(reward))

class EpisodicLifeEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]): 
    def __init__(self, env:gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done =  True

    def step(self, action:int)-> AtariStepReturn:
        obs,reward,terminated,truncated,info = self.env.step(action)
        self.was_real_done = terminated or truncated

        lives = self.env.unwrapped.ale.lives()
        if 0<lives < self.lives:
            terminated = True

        self.lives = lives

        return obs,reward,terminated,truncated,info

    def reset(self, **kwargs)-> AtariResetReturn:

        if self.was_real_done:
            obs,info = self.env.reset(**kwargs)
        else:
            obs,_,terminated,truncated,info = self.env.step(0)
            
            if terminated or truncated:
                obs,info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs,info

class NoopResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    def __init__(self, env:gym.Env,max_noop:int = 30):
        super().__init__(env)
        self.max_noop = max_noop

    def reset(self, **kwargs)-> AtariResetReturn:
        obs,info = self.env.reset(**kwargs) 
        noop = self.unwrapped.np_random.integers(1,self.max_noop+1)
        obs = np.zeros(0)
        info: Dict = {} # au cas ou noops =0
        for _ in range(noop):
            obs,_,terminated,truncated,info = self.env.step(0)
            if terminated or truncated:
                obs,info = self.env.reset(**kwargs)
        return obs,info


class FireResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    def __init__(self, env:gym.Env):
        super().__init__(env)
    
    def reset(self, **kwargs)-> AtariResetReturn:
        self.env.reset()
        obs,_,terminated,truncated,_ = self.env.step(1) #firring
        if terminated or truncated:
            self.env.reset()
        obs,_,terminated,truncated,_ = self.env.step(2) # pour gérer les jeux ou il faut deux actions 
        if terminated or truncated:
            self.env.reset()

        return obs, {} 

class WarpEnv(gym.ObservationWrapper):
    def __init__(self, env:gym.Env, width: int = 84, height:int =84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1,self.height, self.width),
            dtype=env.observation_space.dtype,  # type: ignore[arg-type]

        )

    def observation(self, observation):
        observation = cv2.cvtColor(observation,cv2.COLOR_RGB2GRAY)
        observation = cv2.resize(observation,(self.width,self.height),interpolation=cv2.INTER_AREA)

        
        return observation[None,:,:] # ajoute une dimension "channel"
    

class AtariWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):

    def __init__(
        self, 
        env:gym.Env,
        screen_size : int = 84,         
        frame_skip: int  = 4,
        max_noops: int = 30,
        teerminal_on_life_loss:bool = True,
        clip_reward:bool = True,
    )->None:
        if max_noops > 0:
            env = NoopResetEnv(env,max_noops)
        if frame_skip > 1:
            env = MaxAndSkipEnv(env,frame_skip)      
        if teerminal_on_life_loss:

            env = EpisodicLifeEnv(env)
        env = Monitor(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpEnv(env,screen_size,screen_size)
        if clip_reward:
            env = ClipRewardEnv(env)
        super().__init__(env)