import gym
import math
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
import redis
import pickle, io
import warnings

warnings.filterwarnings("ignore")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def strip_ddp_state_dict(state_dict):
    """ Workaround the fact that DistributedDataParallel prepends 'module.' to
    every key, but the sampler models will not be wrapped in
    DistributedDataParallel. (Solution from PyTorch forums.)"""
    clean_state_dict = type(state_dict)()
    for k, v in state_dict.items():
        key = k[7:] if k[:7] == "module." else k
        clean_state_dict[key] = v
    return clean_state_dict


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class MemoryDataset(Dataset):

    def __init__(self, data_length, memory_buffer):  # input_size,

        self.len = data_length
        self.data = memory_buffer  # torch.randn(data_length, input_size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class Agent(object):

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.act_net = Net(self.state_space_dim, 256, self.action_space_dim)
        self.eval_net = Net(self.state_space_dim, 256, self.action_space_dim)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.buffer = []
        self.steps = 0
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    def act(self, s0):
        self.steps += 1
        epsi = self.epsi_low + (self.epsi_high - self.epsi_low) * (math.exp(-1.0 * self.steps / self.decay))
        if random.random() < epsi:
            a0 = random.randrange(self.action_space_dim)
        else:
            self.act_net = self.act_net.cpu()
            s0 = torch.tensor(s0, dtype=torch.float).view(1, -1)
            a0 = torch.argmax(self.act_net(s0)).item()
        return a0

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self, gpu, args):
        if self.steps % 49 == 0:
            print('gpu-now', gpu)
        rank = args.nr * args.gpus + gpu
        os.environ['MASTER_ADDR'] = '192.168.0.116'  # '10.57.23.164'  # 告诉Multiprocessing模块去哪个IP地址找process 0以确保初始同步所有进程
        os.environ['MASTER_PORT'] = '37439'  # '8888'  # process 0所在的端口
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
        ##########################################
        torch.cuda.set_device(gpu)

        if (len(self.buffer)) < self.batch_size:
            if gpu == 0:
                # =========================== Redis =========================== #
                r = redis.Redis(host='localhost', port=6379, decode_responses=False)
                io_buffer = io.BytesIO()
                pickle.dump(self.eval_net.state_dict(), io_buffer)
                model_byte_data = io_buffer.getvalue()
                r.set('eval_net_para', model_byte_data)
                # print('*******************  ------------------  ==============', model_byte_data)
                # ====================================================== #
            return
        self.eval_net = self.eval_net.cuda(gpu)
        self.eval_net = nn.parallel.DistributedDataParallel(self.eval_net, device_ids=[gpu])
        samples = random.sample(self.buffer, self.batch_size)
        DQNmemory = MemoryDataset(self.batch_size, samples)
        train_sampler = torch.utils.data.distributed.DistributedSampler(DQNmemory,
                                                                        num_replicas=args.world_size,
                                                                        rank=rank)
        sample_loader = torch.utils.data.DataLoader(dataset=DQNmemory,
                                                    batch_size=self.batch_size,
                                                    shuffle=False,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    sampler=train_sampler)

        for data in sample_loader:
            s0, a0, r1, s1 = data
            s0 = s0.float().cuda(gpu)
            a0 = a0.long().view(self.batch_size//2, -1).cuda(gpu)
            r1 = r1.float().view(self.batch_size//2, -1).cuda(gpu)
            s1 = s1.float().cuda(gpu)

            y_true = r1 + self.gamma * torch.max(self.eval_net(s1).detach(), dim=1)[0].view(self.batch_size//2, -1)
            y_pred = self.eval_net(s0).gather(1, a0)

            loss_fn = nn.MSELoss().cuda(gpu)
            loss = loss_fn(y_pred, y_true)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if gpu == 0:
            print('gpu=', gpu)
            # =========================== Redis =========================== #
            r = redis.Redis(host='localhost', port=6379, decode_responses=False)
            io_buffer = io.BytesIO()
            pickle.dump(self.eval_net.state_dict(), io_buffer)
            model_byte_data = io_buffer.getvalue()
            r.set('eval_net_para', model_byte_data)
            # print('*******************  ------------------  ==============', model_byte_data)
            # ====================================================== #


if __name__ == '__main__':
    #########################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes  # 等于GPU的总数
    smp = mp.get_context('spawn')
    #########################################################
    # 设置随机数种子
    setup_seed(521)
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
            if done:
                r1 = -1
            agent.put(s0, a0, r1, s1)
            if done:
                break
            total_reward += r1
            s0 = s1
            #########################################################
            gpu_num1 = 0
            gpu_num2 = 1  # redis中key值相同会覆盖
            p1 = smp.Process(target=agent.learn, args=(gpu_num1, args))  # q 后面必须有逗号，
            p2 = smp.Process(target=agent.learn, args=(gpu_num2, args))
            p1.start()
            p2.start()
            p1.join()  # 先执行该子进程中的代码，结束后再执行主进程的其他代码
            p2.join()
            #########################################################
            # =========================== Redis =========================== #
            r = redis.Redis(host='localhost', port=6379, decode_responses=False)
            modeldata = r.get('eval_net_para')
            io_buffer = io.BytesIO()
            io_buffer.write(modeldata)
            io_buffer.seek(0)
            modelpara = pickle.load(io_buffer)
            # print('*******************  ------------------  ==============', modelpara)
            agent.act_net.load_state_dict(strip_ddp_state_dict(modelpara))

        if episode % 1 == 0:
            print(episode, ': ', total_reward)

    finish_time = datetime.now()  # 获得当前时间
    print('total time of training', (finish_time-start_time).seconds)
