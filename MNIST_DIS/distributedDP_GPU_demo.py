import torch
import time
from datetime import datetime
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import argparse
import torch.distributed as dist

# 定义超参数
batch_size = 128
learning_rate = 1e-2 * 2
num_epoches = 100


def to_np(x):
    return x.cpu().data.numpy()

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)
#print(args.local_rank)

# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())


train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)


#test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

#test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler)


#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# 定义 Convolution Network 模型
class Cnn(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True), nn.MaxPool2d(2, 2))

        self.fc = nn.Sequential(
            nn.Linear(400, 120), nn.Linear(120, 84), nn.Linear(84, n_class))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
#        print(x)
        return out


model = Cnn(1, 10)  # 图片大小是28x28
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
print('START  :  ',time.ctime())
start_time = datetime.now() #获得当前时间
# 开始训练
for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    datasize = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        if use_gpu:
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
        # 向前传播
        out = model(img)
        datasize += len(img)
        loss = criterion(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.item()
        # 向后传播
        optimizer.zero_grad()
        # print(loss)
        loss.backward()
        optimizer.step()
#        if i % 300 == 0:
#            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
#                epoch + 1, num_epoches, running_loss / (batch_size * i),
#                running_acc / (batch_size * i)))
#        print(len(train_dataset))

#    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
#        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
#            train_dataset))))
#    print(datasize)
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (datasize), running_acc / datasize))


    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        if use_gpu:
            with torch.no_grad():
                img = img.cuda()
                label = label.cuda()
        else:
            with torch.no_grad():
                img = img
                label = label
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    print('FInish a epoch  :  ',time.ctime())
    print('================')
finish_time = datetime.now() #获得当前时间
print('total time of training', (finish_time-start_time).seconds)
