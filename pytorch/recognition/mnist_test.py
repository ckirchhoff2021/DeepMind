import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets

import argparse
from models.resnet import *
from utils import *


def run():
    net = resnet18()
    net.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)

    train_dataset, test_dataset = mnist_dataset()
    print('==> start training loop...')
    print('==> train datas: ', len(train_dataset))
    print('==> test datas: ', len(test_dataset))

    if cuda:
        # net = nn.DataParallel(net)
        net = net.cuda()

    epochs = 2
    best_acc = 0.0
    batch_size = 32
    learning_rate = 0.002

    batches = int(len(train_dataset) / batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    opt = optimizer.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        net.train()
        losses = 0.0
        counts = successes = 0
        for index, (data, label) in enumerate(train_loader):
            if cuda:
                inputs, targets = data.cuda(), label.cuda()
            else:
                inputs, targets = data, label

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            losses += loss.item()
            predicts = outputs.max(1)[1]
            success = predicts.eq(targets).sum().item()
            count = data.size(0)
            opt.zero_grad()
            loss.backward()
            opt.step()

            successes += success
            counts += count
            if index % 50 == 0:
                print('==> Epoch-[%d]/[%d]-[%d]/[%d], training process : loss = %.4f, acc = %.4f' %
                      (epoch, epochs, index, batches, loss.item(), success / count))

        epoch_loss = losses / len(train_loader)
        epoch_acc = successes / counts
        print('=> Epoch-[%d], training average : loss = %.4f, acc = %.4f' % (epoch, epoch_loss, epoch_acc))

        net.eval()
        counts = successes = 0
        for index, (data, label) in enumerate(test_loader):
            if cuda:
                inputs, targets = data.cuda(), label.cuda()
            else:
                inputs, targets = data, label

            outputs = net(inputs)
            predicts = outputs.max(1)[1]
            successes += targets.eq(predicts).sum().item()
            counts += data.size(0)

        test_acc = successes / counts
        print('=> Epoch-[%d], testing acc = %.4f' % (epoch, test_acc))
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'net': net.state_dict(),
                'epoch': epoch,
                'acc': best_acc
            }
            torch.save(state, 'model.pth')


if __name__ == '__main__':
    run()
