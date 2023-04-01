import torch
import numpy as np
device=torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
EPS=0.003
def para_init(size,fanin=False):
    fanin=fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v,v)

class Critic(torch.nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fcs1 = torch.nn.Linear(state_dim, 256)
        self.fcs1.weight.data=para_init(self.fcs1.weight.data.size())
        self.fcs2 = torch.nn.Linear(256,128)
        self.fcs2.weight.data=para_init(self.fcs2.weight.data.size())

        self.fca1 = torch.nn.Linear(action_dim, 128)
        self.fca1.weight.data = para_init(self.fca1.weight.data.size())

        self.fc2 = torch.nn.Linear(256, 128)
        self.fc2.weight.data = para_init(self.fc2.weight.data.size())

        self.fc3 = torch.nn.Linear(128, 1)
        self.fc3.weight.data.uniform_(-EPS, EPS)

    def forward(self,state,action):
        state=state.to(device)#get it to cuda
        action=action.to(device)#get it to cuda

        s1 = torch.nn.functional.relu(self.fcs1(state))
        s2 = torch.nn.functional.relu(self.fcs2(s1))
        a1 = torch.nn.functional.relu(self.fca1(action))
        x = torch.cat((s2, a1), dim=1)

        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Actor(torch.nn.Module):
    def __init__(self,state_dim, action_dim, action_lim):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim

        self.fc1 = torch.nn.Linear(state_dim, 256)
        self.fc1.weight.data = para_init(self.fc1.weight.data.size())

        self.fc2 = torch.nn.Linear(256, 128)
        self.fc2.weight.data = para_init(self.fc2.weight.data.size())

        self.fc3 = torch.nn.Linear(128, 64)
        self.fc3.weight.data = para_init(self.fc3.weight.data.size())

        self.fc4 = torch.nn.Linear(64, action_dim)
        self.fc4.weight.data.uniform_(-EPS, EPS)

    def forward(self,state):
        state=state.to(device)
        x = torch.nn.functional.relu(self.fc1(state))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        action = torch.nn.functional.tanh(self.fc4(x))

        action = action * self.action_lim
        return action



