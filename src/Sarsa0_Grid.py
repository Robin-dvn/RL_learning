import gymnasium as gym
import gym_custom
import numpy as np
import random



def pos_to_indice(x,y,n):
    i = x*n + y
    return i

def indice_to_pos(i,n):
    x = i // n
    y = i % n
    return (x,y)

def epsilon_greedy_policy(s,Q,epsilon):
    i_agent,i_target = s
    # prendre le a qui maximise de Q(s,a) sachant s
    # avec une propabilité epsilon, prendre une action au hasard
    if random.uniform(0,1) < epsilon:
        action = random.choice([0,1,2,3])
    else:
        action = np.argmax(Q[i_agent,i_target])

    return int(action)




def trainSarsa0(nbep,seed,policy,size,alpha,epsilon):
    env = gym.make("GridWorld-v0",size = size)
    Q = np.zeros(shape=(size**2,size**2,env.action_space.n))
    for ep in range(nbep):
        if ep%10000 ==0 : epsilon/=10
        done =False
        s,info = env.reset()
        a = policy(s,Q,epsilon)
        while not done:
            i_agent,i_target = s
            snew,r,terminated,truncated,info = env.step(a)
            if r ==1:
                r = 0
            else: r = -1
            if (terminated or truncated): done = True

            anew = policy(snew,Q,epsilon)
            i_agent_new,i_target_new = snew
            Q[i_agent,i_target,a] = Q[i_agent,i_target,a] + alpha*(r+Q[i_agent_new,i_target_new,anew]-Q[i_agent,i_target,a])
            a = anew
            s = snew
    env.close()
    return Q

Q=trainSarsa0(100000,None,epsilon_greedy_policy,5,0.1,0.5)

def playpolicy(s,Q,size):

    i_agent,i_target = s
    action = np.argmax(Q[i_agent,i_target])

    return int(action)


def play(nbep,policy,Q):
    env = gym.make("GridWorld-v0",size = 5,render_mode="human")
    for ep in range(nbep):
        done = False
        s,info = env.reset()
        while not done:

            a = policy(s,Q,5)
            snew,r,terminated,truncated,info = env.step(a)
            s = snew
            if (terminated or truncated): done = True
    env.close()
play(5,playpolicy,Q)
    
