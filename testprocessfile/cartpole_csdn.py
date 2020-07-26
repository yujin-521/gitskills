import gym
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import numpy as np
from io import BytesIO
import pickle
import multiprocessing as mp

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Agentsave(object):
    def __init__(self):
        self.model = None
        self.model_data = None
        self.model_state_dict = None  # for load dict #
        pass

    def _create_model(self):
        model = Net(4,256,2)
        # for name, param in model.named_parameters():
        #     print(name,param,"init-----------------------")
        # model.cpu()
        return model

    def _torch_save_model(self, model):
        # for torch.save #
        f = BytesIO()
        torch.save(model, f)
        self.model_data = f.getvalue()

    def _torch_load_model(self):
        # for torch.load #
        f = BytesIO()
        f.write(self.model_data)
        f.seek(0)
        model = torch.load(f)
        return model

    def _torch_save_model_dict(self, model):
        f = BytesIO()
        torch.save(model.state_dict(), f)
        self.model_state_dict = f.getvalue()
        f.close()

    def _torch_load_model_dict(self):
        f = BytesIO()
        f.write(self.model_state_dict)
        f.seek(0)

        model = self._create_model()
        model.load_state_dict(torch.load(f))
        return model


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.eval_net = Net(self.state_space_dim, 256, self.action_space_dim)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.buffer = []
        self.steps = 0

    def act(self, s0):
        self.steps += 1

        self.eval_net = self.eval_net.to(device)

        epsi = self.epsi_low + (self.epsi_high - self.epsi_low) * (math.exp(-1.0 * self.steps / self.decay))
        if random.random() < epsi:
            a0 = random.randrange(self.action_space_dim)
        else:
            s0 = torch.tensor(s0, dtype=torch.float).view(1, -1).to(device)
            a0 = torch.argmax(self.eval_net(s0)).item()
        return a0

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self):
        if (len(self.buffer)) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)
        # print(samples)
        s0, a0, r1, s1 = zip(*samples)
        s0 = torch.tensor(s0, dtype=torch.float).to(device)
        a0 = torch.tensor(a0, dtype=torch.long).view(self.batch_size, -1).to(device)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1).to(device)
        s1 = torch.tensor(s1, dtype=torch.float).to(device)
        # print(s0)
        self.eval_net = self.eval_net.to(device)

        # for name, param in self.eval_net.named_parameters():
        #     print(name,param)
        # print('*****************ago*****************')
        # agentput = Agentsave()
        # # print(agentput.model_state_dict, 0)
        # # agentput._torch_save_model_dict(self.eval_net)
        # agentput._torch_save_model(self.eval_net)
        # # print(agentput.model_state_dict, 1)
        # # self.eval_net = agentput._torch_load_model_dict().to(device)
        # self.eval_net = agentput._torch_load_model()
        # # print(agentput.model_state_dict, 2)
        # for name, param in self.eval_net.named_parameters():
        #     print(name,param)
        # print('*****************save*****************')

        # torch.save(self.eval_net, "net.pkl")
        # self.eval_net = torch.load("net.pkl")

        y_true = r1 + self.gamma * torch.max(self.eval_net(s1).detach(), dim=1)[0].view(self.batch_size, -1)
        y_pred = self.eval_net(s0).gather(1, a0)
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y_true)
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':

    # 设置随机数种子
    setup_seed(521)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    env = gym.make('CartPole-v0')
    params = {
        'gamma': 0.8,
        'epsi_high': 0.9,
        'epsi_low': 0.05,
        'decay': 200,  # exploration的衰减率
        'lr': 0.001,
        'capacity': 10000,
        'batch_size': 128,
        'state_space_dim': env.observation_space.shape[0],
        'action_space_dim': env.action_space.n
    }
    agent = Agent(**params)
    score = []
    mean = []
    maxepisode = 101
    count_whole_score = 0
    start_time = datetime.now()  # 获得当前时间
    print('STARTTIME', start_time)

    for episode in range(maxepisode):
        s0 = env.reset()
        total_reward = 1
        while True:
            # env.render()
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)
            # print(r1)
            if done:
                r1 = -1
            agent.put(s0, a0, r1, s1)
            if done:
                break
            total_reward += r1
            s0 = s1

            agent.learn()


        if episode % 20 == 0:
            print(episode, ': ', total_reward)



        # if total_reward > 199:
        #     count_whole_score += 1
        #     # print(count_whole_score)
        #     if count_whole_score > 10:
        #         break
        # else:
        #     count_whole_score = 0

        # score.append(total_reward)
        # mean.append(sum(score[-100:]) / 100)

    finish_time = datetime.now() #获得当前时间
    print('total time of training', (finish_time-start_time).seconds)