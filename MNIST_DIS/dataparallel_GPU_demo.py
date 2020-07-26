# ====================== nn.DataParallel =========================== #
import torch.distributed as dist
import torch
import time
from datetime import datetime
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

print("We have", torch.cuda.device_count(), "GPUs!")
num_gpu = torch.cuda.device_count()
gpus = []
for i in range(num_gpu):
    gpus.append(i)
print(gpus)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")

# train_dataset = ...
#
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=...)

# 定义超参数
batch_size = 128
learning_rate = 1e-2
num_epoches = 50


def to_np(x):
    return x.cpu().data.numpy()


# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
        print(x.squeeze(0).size())
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # print(out.squeeze(0).size())
      #  print("\tIn Model: input size", x.size(), "output size", out.size())
        return out


model = Cnn(1, 10)  # 图片大小是28x28
model = nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus[0])
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
print(use_gpu)
if use_gpu:
    model = model.cuda()
# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# for batch_idx, (data, target) in enumerate(train_loader):
#    images = images.cuda(non_blocking=True)
#    target = target.cuda(non_blocking=True)
#    ...
#    output = model(images)
#    loss = criterion(output, target)
#    ...
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
start_time = datetime.now() #获得当前时间
print('STARTTIME', time.ctime())
for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.cuda(non_blocking=True)
        # print(img.squeeze(0).size())
        label = label.cuda(non_blocking=True)
        # 向前传播
        out = model(img)
       # print('A batch over')
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
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))
    model.eval()  # 在model(test)之前，需要加上model.eval()，否则的话，有输入数据，即使不训练，它也会改变权>值。这是model中含>有batch normalization层所带来的的性质。
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        with torch.no_grad():
            img = img.cuda()
            label = label.cuda()
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    print('Finish training a epoch',time.ctime())
    print('================')
finish_time = datetime.now()  #获得当前时间
print('total time of training', (finish_time-start_time).seconds)
