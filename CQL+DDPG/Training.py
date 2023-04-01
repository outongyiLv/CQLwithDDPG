import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import gym
import gc
import numpy as np
from DDPG import DDPG
from Buffer import Buffer
from matplotlib import  pyplot as plt
env = gym.make('BipedalWalker-v3')
MAX_EPISODES = 1500
MAX_STEPS = 100
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]
ram = Buffer(MAX_BUFFER)
trainer=DDPG(S_DIM, A_DIM, A_MAX, ram)#24,4,1.0,ram
max_reward=-1000000
pic=[]


for _ep in range(MAX_EPISODES):
    observation=env.reset()
    observation1=observation[0]
    sum_reward=0
    state3=np.float32(observation1)
    for r in range(MAX_STEPS):
        state=state3
        action=trainer.get_action(state)
        new_observation, reward, done, info,k= env.step(action)
        if done:
            new_state=None
        else:
            new_state=np.float32(new_observation)
            ram.add(state,action,reward,new_state)
        state3=new_state
        sum_reward+=reward
        if(ram.len>32):
            trainer.optimizer()
        if done:
            break
    if(_ep%10==0):
        observation = env.reset()
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
        print("episode:",_ep, sum_reward, tsum_reward)
        pic.append(tsum_reward)
        plt.plot(pic)
        plt.savefig("Result.jpg")
        gc.collect()





