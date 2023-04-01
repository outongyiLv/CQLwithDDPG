import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import gym
import gc
import numpy as np
from DDPG import DDPG
from Buffer import Buffer
from matplotlib import  pyplot as plt
env = gym.make('BipedalWalker-v3',render_mode="human")
MAX_EPISODES = 1500
MAX_STEPS = 100
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]
ram = Buffer(MAX_BUFFER)
trainer=DDPG(S_DIM, A_DIM, A_MAX, ram)#24,4,1.0,ram
trainer.load_model()
max_reward=-1000000
s1=[]
a1=[]
r1=[]
s2=[]
for epoch in range(1):
    for _ep in range(1000):
        observation = env.reset()
        observation1 = observation[0]
        sum_reward = 0
        state3 = np.float32(observation1)
        for r in range(MAX_STEPS):
            state = state3
            action=trainer.get_action(state)
            new_observation,reward,done,info,k=env.step(action)
            if done:
                new_state = None
            else:
                new_state = np.float32(new_observation)
                s1.append(state)
                a1.append(action)
                r1.append(reward)
                s2.append(new_state)
            state3=new_state
            if done:
                print("epoch",_ep)
                break
s1=np.array(s1)
a1=np.array(a1)
r1=np.array(r1)
s2=np.array(s2)
np.save("npydata/s1.npy", s1)
np.save("npydata/a1.npy", a1)
np.save("npydata/r1.npy", r1)
np.save("npydata/s2.npy", s2)