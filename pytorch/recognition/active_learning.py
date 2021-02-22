import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from models.resnet import *
from train import start_build
from datasets import CommonDataset
from common_path import *


class ChildDataset(Dataset):
    def __init__(self, full_dataset, indices):
        super(ChildDataset, self).__init__()
        self.full_dataset = full_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        return self.full_dataset[index]


class ActiveLearning:
    def __init__(self, net, train_datas, test_datas):
        self.cuda = torch.cuda.is_available()
        self.net = net
        self.labeled = set()
        self.unlabeled = set()

        self.train_datas = train_datas
        self.test_datas = test_datas
        self.prob_threshold = 0.5
        self.acc_threshold = 0.9

        self.initialize()

    def validate(self):
        state = torch.load('output/model.pth')
        print('==> acc: ', state['acc'])
        self.net.load_state_dict(state['net'])
        return state['acc']

        self.net.eval()
        test_loader = DataLoader(self.test_datas, batch_size=64, shuffle=True)
        with torch.no_grad():
            correct = total = 0
            for index, (data, label) in enumerate(test_loader):
                if self.cuda:
                    inputs, targets = data.cuda(), label.cuda()
                else:
                    inputs, targets = data, label

                outputs = self.net(inputs)
                preds = outputs.max(1)[1]
                correct += preds.eq(targets.view_as(preds)).sum().item()
                total += data.size(0)
            acc = correct / total
            print('==> Accuracy = %f' % acc)
        return acc

    def least_certain_query(self):
        self.net.eval()
        choose_datas = ChildDataset(self.train_datas, self.unlabeled)
        choose_loader = DataLoader(choose_datas, batch_size=64)

        with torch.no_grad():
            for index, (data, label) in enumerate(choose_loader):
                num = data.size(0)
                indices = np.arange(num)
                indices += index * 64
                if cuda:
                    inputs, targets = data.cuda(), label.cuda()
                else:
                    inputs, targets = data, label

                outputs = self.net(inputs)
                probs = F.softmax(outputs, dim=1)
                probs = probs.max(1)[0]
                chosen = probs < self.prob_threshold
                if self.cuda:
                    chosen = chosen.cpu().numpy().astype(np.bool)
                else:
                    chosen = chosen.numpy().astype(np.bool)

                chosen = set(indices[chosen])

                self.labeled = self.labeled | chosen
                self.unlabeled = self.unlabeled - chosen

            print('==> Labeled: ', len(self.labeled))
            print('==> Unlabeled: ', len(self.unlabeled))

    def margin_query(self):
        self.net.eval()
        choose_datas = ChildDataset(self.train_datas, self.unlabeled)
        choose_loader = DataLoader(choose_datas, batch_size=64)

        with torch.no_grad():
            diffs = list()
            for index, (data, label) in enumerate(choose_loader):
                if cuda:
                    inputs, targets = data.cuda(), label.cuda()
                else:
                    inputs, targets = data, label

                outputs = self.net(inputs)
                probs = F.softmax(outputs, dim=1)
                probs = probs.topk(2, dim=1, largest=True, sorted=True)[0]
                diff = probs[:, 0] - probs[:, 1]
                diffs.extend(diff.numpy())

            indices = np.argsort(diffs)
            chosen = set(indices[:1000])
            self.labeled = self.labeled | chosen
            self.unlabeled = self.unlabeled - chosen

            print('==> Labeled: ', len(self.labeled))
            print('==> Unlabeled: ', len(self.unlabeled))

    def entropy_query(self):
        self.net.eval()
        choose_datas = ChildDataset(self.train_datas, self.unlabeled)
        choose_loader = DataLoader(choose_datas, batch_size=64)

        with torch.no_grad():
            entropy = list()
            for index, (data, label) in enumerate(choose_loader):
                if cuda:
                    inputs, targets = data.cuda(), label.cuda()
                else:
                    inputs, targets = data, label

                outputs = self.net(inputs)
                probs = F.softmax(outputs, dim=1)

                value = probs * torch.log(probs) * (-1)
                value = torch.sum(value, dim=1)
                entropy.extend(value.numpy())

            indices = np.argsort(entropy)
            chosen = set(indices[-1000:])
            self.labeled = self.labeled | chosen
            self.unlabeled = self.unlabeled - chosen

            print('==> Labeled: ', len(self.labeled))
            print('==> Unlabeled: ', len(self.unlabeled))

    def initialize(self):
        targets = self.train_datas.targets
        self.unlabeled = set(np.arange(len(targets)))

        datas = list()
        for i in range(10):
            datas.append(list())

        for index, value in enumerate(targets):
            datas[value].append(index)

        for index, value in enumerate(datas):
            random.shuffle(value)
            chosen = value[:200]
            self.labeled = self.labeled | set(chosen)

        self.unlabeled = self.unlabeled - self.labeled

    def train(self):
        train_datas = ChildDataset(self.train_datas, self.labeled)
        start_train(self.net, train_datas, self.test_datas, 20, 64, 'output/model.pth', 0.01, self.cuda)

    def run(self):
        self.train()
        iteration = 0
        ret = list()

        while len(self.unlabeled) > 0 and iteration < 10:
            print('==> current iteration:', iteration)
            iteration += 1
            acc = self.validate()

            ret.append({
                'acc': acc,
                'iteration': iteration,
                'unlabeled': len(self.unlabeled),
                'labeled': len(self.labeled)
            })

            if acc > self.acc_threshold:
                break
            self.least_certain_query()
            self.train()


def main():
    net = resnet18v2(pretrained=True)
    dataset = CommonDataset(data_path)
    train_datas, test_datas = dataset.cifar10(transform=True)
    AL = ActiveLearning(net, train_datas, test_datas)

    print('==> Labeled :', len(AL.labeled))
    print('==> Unlabeled :', len(AL.unlabeled))

    AL.run()


if __name__ == '__main__':
    main()








