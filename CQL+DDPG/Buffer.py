import numpy as np
import random
from collections import deque
class Buffer:
    def __init__(self,buffer_size):
        self.buffer=deque(maxlen=buffer_size)
        self.maxSize=buffer_size
        self.len=0
    def sample(self,count):
        count=min(count, self.len)
        batch=random.sample(self.buffer, count)
        s_arr=np.float32([arr[0] for arr in batch])
        a_arr=np.float32([arr[1] for arr in batch])
        r_arr=np.float32([arr[2] for arr in batch])
        s1_arr=np.float32([arr[3] for arr in batch])
        return s_arr, a_arr, r_arr, s1_arr
    def len(self):
        return self.len
    def add(self,s,a,r,s1):
        transition=(s,a,r,s1)
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)




