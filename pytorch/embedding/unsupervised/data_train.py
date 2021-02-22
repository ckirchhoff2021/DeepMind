import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as opt

from torchvision import transforms, models
from torch.utils.data import DataLoader

from cifar_dataset import CIFAR10Instance
from loss import BatchCriterion
from normalize import Normalize

from sklearn.neighbors import KNeighborsClassifier
from common_path import *


def mobile_net():
    net = models.mobilenet_v2()
    conv1 = net.features[0]
    conv1[0].stride = (1, 1)
    classifier = net.classifier
    classifier[1] = nn.Linear(in_features=1280, out_features=128, bias=True)
    return net

def resnet():
    net = models.resnet18()
    net.conv1 = nn.Conv2d(3,64,3, stride=1, padding=1, bias=False)
    net.fc = nn.Sequential(
        nn.Linear(512, 128, bias=True)
    )
    return net
    

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.feature = resnet()
        self.l2norm = Normalize(2)
    
    def forward(self, x):
        y = self.feature(x)
        y = self.l2norm(y)
        return y


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def start_train():
    data_path = data_path
    train_data = CIFAR10Instance(root=data_path, train=True, download=False, transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, drop_last=True)
    
    test_data = CIFAR10Instance(root=data_path, train=False, download=False, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False)
    
    data_num = len(train_data)
    
    net = EmbeddingNet()
    criterion = BatchCriterion(0.1)
    optimizer = opt.SGD(net.parameters(), lr=0.001, momentum=0.9)
    best_acc = 0.0
    
    for epoch in range(0, 301):
        net.train()
        train_loss = 0.0
        for index, (inputs1, inputs2, targets) in enumerate(train_loader):
            x = torch.cat((inputs1, inputs2), dim=0)
            y = net(x)
            loss = criterion(y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print('Epoch: %d, idx = %d, train loss = %f' % (epoch, index, loss.item()))
    
        average_loss = train_loss / len(train_loader)
        print('-- Average loss= ', average_loss)
        acc = KNN_classify()
        print('Acc = %f', acc)
        if acc > best_acc:
            torch.save(net.state_dict(), 'output/model.pth')


def knn_test():
    net = EmbeddingNet()
    net.load_state_dict(torch.load('output/model.pth'))
    net.eval()

    train_data = CIFAR10Instance(root=data_path, train=True, download=False, transform=test_transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    test_data = CIFAR10Instance(root=data_path, train=False, download=False, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

    data_num = len(train_data)
    x_datas = np.zeros((128, data_num))
    x_labels = []
    
    with torch.no_grad():
        for index, (inputs1, inputs2, targets) in enumerate(train_loader):
            x = inputs1
            outputs = net(x)
            data = outputs.data.numpy().T
            x_labels.extend(targets)
            if (index + 1) * 128 <= data_num:
                x_datas[:,index*128:(index+1)*128] = data
            else:
                x_datas[:, index*128:data_num] = data
        print('train datas:', x_datas.shape)
        print('train labels:', len(x_labels))
        
        train_features = torch.Tensor(x_datas)
        train_labels = torch.Tensor(x_labels)
        
        correct = total = 0
        for index, (inputs1, inputs2, targets) in enumerate(test_loader):
            x = inputs1
            outputs = net(x)
            dist = torch.mm(outputs, train_features)
            yd, yi = dist.topk(200,dim=1,largest=True, sorted=True)
            pred_values = train_labels[yi].numpy().int()

            predicts = []
            for pred in pred_values:
                pred_label = 0
                num_arr = np.zeros(10)
                for i in pred:
                    num_arr[pred[i]] += 1
                pred_label = np.argmax(num_arr)
                predicts.append(pred_label)
                
            predicts = torch.tensor(predicts)
            success = predicts.eq(targets).sum().item()
            correct += success
            total += len(targets)
            
        print('Accuracy: ', correct / total)

        
if __name__ == '__main__':
    # start_train()
    net = resnet()
    print(net)
    # x = torch.randn(12, 3, 32, 32)
    # y = net(x)
    # print(y.size())