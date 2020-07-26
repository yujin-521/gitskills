# **********  MNIST with Multi-GPU *********#
import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision import datasets
from torch.utils.data import DataLoader
import redis
import pickle, io
# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp
'''
args.nodes 是我们使用的结点数
args.gpus 是每个结点的GPU数.
args.nr 是当前结点的阶序rank，这个值的取值范围是 0 到 args.nodes - 1.
'''


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def strip_ddp_state_dict(state_dict):
    """ Workaround the fact that DistributedDataParallel prepends 'module.' to
    every key, but the sampler models will not be wrapped in
    DistributedDataParallel. (Solution from PyTorch forums.)"""
    clean_state_dict = type(state_dict)()
    for k, v in state_dict.items():
        key = k[7:] if k[:7] == "module." else k
        clean_state_dict[key] = v
    return clean_state_dict


def test(model):
    # 下载训练集 MNIST 手写数字训练集
    batch_size = 100
    test_dataset = datasets.MNIST(
        root='./data', train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = model.cpu()
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        with torch.no_grad():
            img = img
            label = label
        out = model(img)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    #########################################################
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes  # 等于GPU的总数
    modeltrain = ConvNet()
    smp = mp.get_context('spawn')
    for i in range(11):
    #########################################################
        print('epoch = ', i)
        gpu_num1 = 0
        gpu_num2 = 1  # redis中key值相同会覆盖
        p1 = smp.Process(target=train, args=(gpu_num1, args, modeltrain))  # q 后面必须有逗号，
        p2 = smp.Process(target=train, args=(gpu_num2, args, modeltrain))
        p1.start()
        p2.start()
        p1.join()  # 先执行该子进程中的代码，结束后再执行主进程的其他代码
        p2.join()
    # =========================== Redis =========================== #
        r = redis.Redis(host='localhost', port=6379, decode_responses=False)
        modeldata = r.get('modelgpu')
        io_buffer = io.BytesIO()
        io_buffer.write(modeldata)
        io_buffer.seek(0)
        modelpara = pickle.load(io_buffer)
        # print('*******************  ------------------  ==============', modelpara)
        modeltrain.load_state_dict(strip_ddp_state_dict(modelpara))
    # ====================================================== #
        modeltest = modeltrain
        # print('==========================modeltest========================')
        # for name, param in modeltest.named_parameters():
        #     print(name, param)
        # print('********************************************************')
        test(modeltest)
    #########################################################
        # mp.spawn(train, nprocs=args.gpus, args=(args,))  # 我们需要生成 args.gpus 个进程, 每个进程都运行 train(i, args), 其中 i 从 0 到 args.gpus - 1



def train(gpu, args, modeltrain):
    ##########################################
    rank = args.nr * args.gpus + gpu
    os.environ['MASTER_ADDR'] = '192.168.0.116'  # '10.57.23.164'  # 告诉Multiprocessing模块去哪个IP地址找process 0以确保初始同步所有进程
    os.environ['MASTER_PORT'] = '37439'  # '8888'  # process 0所在的端口
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    ##########################################
    torch.manual_seed(0)
    model = modeltrain
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    ###############################################################
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    ###############################################################

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print("gpu_now", gpu, 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step, loss.item()))
            if (i + 1) % 100 == 0 and gpu == 1:
                print("gpu_now", gpu, 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step, loss.item()))
    if gpu == 0:
        print('gpu=', gpu,  "Training complete in: " + str(datetime.now() - start))
        # =========================== Redis =========================== #
        # print('========================== modeltrain gpu 0 ========================')
        # for name, param in model.named_parameters():
        #     print(name, param)
        # print('********************************************************')
        r = redis.Redis(host='localhost', port=6379, decode_responses=False)
        io_buffer = io.BytesIO()
        pickle.dump(model.state_dict(), io_buffer)
        model_byte_data = io_buffer.getvalue()
        r.set('modelgpu', model_byte_data)
        # print('*******************  ------------------  ==============', model_byte_data)
        # ====================================================== #
    print("gpu_now", gpu, "Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':
    main()

