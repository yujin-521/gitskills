from torch.nn import init
import redis
import pickle, io
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self, input_size=4, hidden_size=10000, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class NetS(nn.Module):
    def __init__(self, input_size=4, hidden_size=10000, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def job1(model2, num_process):
    r = redis.Redis(host='localhost', port=6379, decode_responses=False)
    model1 = Net()
    model3 = model2
    if num_process == 1:
        print('==========================model1========================')
        for name, param in model1.named_parameters():
            print(name, param)
        print('********************************************************')
        print('==========================model3========================')
        for name, param in model3.named_parameters():
            print(name, param)
        print('********************************************************')
        io_buffer = io.BytesIO()
        pickle.dump(model3.state_dict(), io_buffer)
        model_byte_data = io_buffer.getvalue()
        # print('*******************  ------------------  ==============', model_byte_data)
        r.set('modelgpu', model_byte_data)
        # q.put(model_byte_data, )
        # print(q.empty())
        # print(q.full())


def mppro():
    # q = mp.Queue(0)  # 1 代表可以存放的数据个数，不是单个数据的长度
    r = redis.Redis(host='localhost', port=6379, decode_responses=False)
    model2 = NetS()
    output_process = 1
    outnone_process = 0  # redis中key值相同会覆盖
    p1 = mp.Process(target=job1, args=(model2, output_process))  # q 后面必须有>逗号，
    p2 = mp.Process(target=job1, args=(model2, outnone_process))
    p1.start()
    p2.start()
    p1.join()  # 先执行该子进程中的代码，结束后再执行主进程的其他代码
    p2.join()
    # modeldata = q.get()
    modeldata = r.get('modelgpu')
    io_buffer = io.BytesIO()
    io_buffer.write(modeldata)
    io_buffer.seek(0)
    modelpara = pickle.load(io_buffer)
    # print('*******************  ------------------  ==============', modelpara)
    # model2 = NetS()
    print('======================COPY model2========================')
    model2.load_state_dict(modelpara)
    for name, param in model2.named_parameters():
        print(name, param)
    print('********************************************************')


if __name__ == '__main__':  # 多线程、多进程必须有这一行
    for i in range(100):
        mppro()
        print(i)
