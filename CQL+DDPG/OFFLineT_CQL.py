import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import gym
import gc
import numpy as np
from OffLineDDPGCQL import DDPG
from Buffer import Buffer
from matplotlib import  pyplot as plt
import pandas as pd
env = gym.make('BipedalWalker-v3')
MAX_EPISODES = 500000
MAX_STEPS = 100
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]
ram = Buffer(MAX_BUFFER)
##loading_offline_data
s1_data=np.load("npydata/s1.npy")
a1_data=np.load("npydata/a1.npy")
r1_data=np.load("npydata/r1.npy")
s2_data=np.load("npydata/s2.npy")
for range1 in range(len(s1_data)):
    ram.add(s1_data[range1],a1_data[range1],r1_data[range1],s2_data[range1])
trainer=DDPG(S_DIM, A_DIM, A_MAX, ram)
max_reward=-1000000
pic=[]

for _ep in range(MAX_EPISODES):#training
    trainer.optimizer()#training
    if(_ep%2000==0):
        observation=env.reset()
        observation1= observation[0]
        state3=np.float32(observation1)
        tsum_reward = 0
        for r in range(MAX_STEPS):
            env.render()
            state=state3
            action=trainer.tget_action(state)
            new_observation, reward, done, info, k = env.step(action)
            if done:
                new_state = None
            else:
                new_state = np.float32(new_observation)
            state3=new_state
            tsum_reward+=reward
            if done:
                break
        if(tsum_reward>max_reward):
            max_reward=tsum_reward
            print("max_reward:",max_reward)
            trainer.save_model()
        print("episode:",_ep,tsum_reward)
        pic.append(tsum_reward)
        plt.plot(pic)
        plt.savefig("Result_OffLineCQL.jpg")
        gc.collect()
        data1=pd.DataFrame([pic])
        data1.to_csv("OffLineCQLresult.csv")

