import torch
from Model import Actor,Critic
import utils
BATCH_SIZE=256
LEARNING_RATE=0.0005
GAMMA=0.99
TAU=0.0005
device=torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
class DDPG:
    def __init__(self, state_dim, action_dim, action_lim, ram):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.ram = ram
        self.iter = 0
        self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.actor = Actor(self.state_dim, self.action_dim, self.action_lim)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_lim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),0.0005)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),0.0005)

        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)

        self.actor = self.actor.to(device)
        self.target_actor = self.target_actor.to(device)
        self.critic = self.critic.to(device)
        self.target_critic = self.target_critic.to(device)

    def tget_action(self,state):
        state = torch.autograd.Variable(torch.from_numpy(state))
        action= self.target_actor.forward(state).detach().cpu()
        return action

    def optimizer(self):
        s1, a1, r1, s2=self.ram.sample(BATCH_SIZE)
        s1 = torch.autograd.Variable(torch.from_numpy(s1))
        a1 = torch.autograd.Variable(torch.from_numpy(a1))
        r1 = torch.autograd.Variable(torch.from_numpy(r1))
        s2 = torch.autograd.Variable(torch.from_numpy(s2))
        # update-critic-network
        with torch.no_grad():
            noise=(torch.FloatTensor(a1.size()).data.normal_(0,0.1)).to(device)
            a2=(self.target_actor.forward(s2)+noise).clamp(-1,1).detach()#下一个action是？
            next_val=torch.squeeze(self.target_critic.forward(s2,a2).detach())  # 下一个value是？
            r1=r1.to(device)
            TD_error=r1+GAMMA*next_val
        y_predicted=torch.squeeze(self.critic.forward(s1,a1))
        loss_critic=torch.nn.functional.smooth_l1_loss(y_predicted,TD_error)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        #更新action网络
        pred_a1=self.actor.forward(s1)#预估动作#(Batch*4)
        #loss2=torch.sum((pred_a1-a1)**2,dim=1)#Batch*4------->Batch*1
        #loss2=torch.sqrt(loss2)
        #loss2=torch.mean(loss2)
        #print("loss2",loss2,"loss1",torch.sum(self.critic.forward(s1,pred_a1)))
        loss_actor=-1*(torch.sum(self.critic.forward(s1,pred_a1))).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        utils.soft_update(self.target_actor, self.actor,TAU)
        utils.soft_update(self.target_critic, self.critic, TAU)

    def save_model(self):
        torch.save(self.target_actor.state_dict(),'DDPG_weight/lotyoffline_actor.pt')
        torch.save(self.target_critic.state_dict(), 'DDPG_weight/lotyoffline_critic.pt')
        print('Models saved successfully')
    def load_model(self):
        self.actor.load_state_dict(torch.load('lotyactor.pt'))
        self.critic.load_state_dict(torch.load('lotycritic.pt'))
        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic,self.critic)
        print('Models loaded succesfully')









