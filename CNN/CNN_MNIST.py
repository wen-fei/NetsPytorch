#!/usr/bin/env python  
# -*- coding: utf-8 -*-

"""
@author: TenYun  
@contact: qq282699766@gmail.com  
@time: 2018/5/24 16:37
搭建CNN进行MNIST分类任务
4层卷积，2层最大池化，3层全连接层
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),  # b, 16, 26, 26
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # b, 32, 24, 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # b, 32, 12, 12
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # b, 64, 10, 10
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  # b, 128, 8, 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # b, 128, 4, 4
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(128*4*4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = CNN_MNIST()
batch_size = 64
learning_rate = 1e-2
num_epoches = 20
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))]
)
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=data_tf, download=False
)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        # data = Variable(data.view(-1, 28*28))
        data  = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train epoch : {}[{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.item()
            ))


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        # data = Variable(data.view(-1, 26*26))
        data = Variable(data)
        target = Variable(target)
        output = model(data)
        test_loss = criterion(output, target)
        pred = output.data.max(1)
        correct += (pred[1] == target.data).cpu().sum()
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, num_epoches + 1):
    train(num_epoches)
    test(num_epoches)



