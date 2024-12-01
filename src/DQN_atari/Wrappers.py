from gymnasium import Wrapper,RewardWrapper,Env,ObservationWrapper
from gymnasium import spaces
import numpy as np
import cv2
from typing import SupportsFloat

class MaxAndSkipEnv(Wrapper):
    def __init__(self, env: Env ,frame_skip:int =4):
        super().__init__(env)
        self._frame_skip = frame_skip
        self._obs = np.zeros((2,*env.observation_space.shape),dtype=env.observation_space.dtype)
    def step(self, action: int):

        total_reward = 0
        for i in range(self._frame_skip):
            obs,r,terminated,truncated,info = self.get_wrapper_attr('env').step(action)
            if self._frame_skip - (i+1) == 0:
                self._obs[0] = obs
            if self._frame_skip - (i+1) == 1:
                self._obs[1] = obs
            total_reward+=r
            done = terminated or truncated
            if done:
                break
        max_frame = self._obs.max(axis=0)
        return max_frame,total_reward,terminated,truncated,info
    
class ClipRewardEnv(RewardWrapper):
    def __init__(self, env:Env):
        super().__init__(env)

    def reward(self, reward: SupportsFloat):

        return np.sign(float(reward))

class EpisodicLifeEnv(Wrapper): 
    def __init__(self, env:Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done: True

    def step(self, action:int):
        obs,reward,terminated,truncated,info = self.get_wrapper_attr('env').step(action)
        self.was_real_done = terminated or truncated

        lives = self.unwrapped.ale.lives()
        if 0<lives < self.lives:
            terminated = True
        self.lives = lives

        return obs,reward,terminated,truncated,info

    def reset(self, **kwargs):

        if self.was_real_done:
            obs,info = self.get_wrapper_attr('env').reset(**kwargs)
        else:
            obs,_,terminated,truncated,info = self.get_wrapper_attr('env').step(0)
            done = terminated or truncated
            if done:
                obs,info = self.get_wrapper_attr('env').reset(**kwargs)
        self.lives = self.unwrapped.ale.lives()
        return obs,info

class NoopResetEnv(Wrapper):
    def __init__(self, env:Env,max_noop:int = 30):
        super().__init__(env)
        self.max_noop = max_noop

    def reset(self, **kwargs):
        
        noop = self.unwrapped.np_random.integers(1,self.max_noop+1)
        obs = np.zeros(0)
        info:dict = {} # au cas ou noops =0
        for _ in range(noop):
            obs,_,terminated,truncated,info = self.get_wrapper_attr('env').step(0)
            if terminated or truncated:
                obs,info = self.get_wrapper_attr('env').reset(**kwargs)
        return obs,info


class FireResetEnv(Wrapper):
    def __init__(self, env:Env):
        super().__init__(env)
    
    def reset(self, **kwargs):
        obs,info = self.get_wrapper_attr('env').reset()
        obs,_,terminated,truncated,info = self.get_wrapper_attr('env').step(1) #firring
        if terminated or truncated:
            obs,info = self.get_wrapper_attr('env').reset()
        obs,_,terminated,truncated,info = self.get_wrapper_attr('env').step(2) # pour gérer les jeux ou il faut deux actions 
        if terminated or truncated:
            obs,info = self.get_wrapper_attr('env').reset()

        return obs, {} 

class WarpEnv(ObservationWrapper):
    def __init__(self, env:Env, width: int = 84, height:int =84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=self.unwrapped.observation_space.dtype,  # type: ignore[arg-type]
        )

    def observation(self, observation):
        image_gray = cv2.cvtColor(observation,cv2.COLOR_BGR2GRAY)
        resized_im = cv2.resize(image_gray,(self.width,self.height),interpolation=cv2.INTER_LINEAR)
        return resized_im[:,:,None] # ajoute une dimension "channel"
    

class AtariWrapper(Wrapper):

    def __init__(
        self, 
        env:Env,
        screen_size : int = 84,         
        frame_skip: int  = 84,
        max_noops: int = 30,
        teerminal_on_life_loss:bool = True,
        clip_reward:bool = True,
    ):
        if max_noops > 0:
            env = NoopResetEnv(env,max_noops)
        if teerminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if frame_skip > 1:
            env = MaxAndSkipEnv(env,frame_skip)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpEnv(env,screen_size,screen_size)
        if clip_reward:
            env = ClipRewardEnv(env)
        super().__init__(env)