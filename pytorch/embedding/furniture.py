import os
import json
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models



data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def resnet18(feature_dim=128, pretrained=False):
    net = models.resnet18(pretrained=pretrained)
    fc_in_num = net.fc.in_features
    net.fc = nn.Linear(fc_in_num, feature_dim)
    return net


def resnet50(feature_dim=128, pretrained=False):
    net = models.resnet50(pretrained=pretrained)
    fc_in_num = net.fc.in_features
    net.fc = nn.Linear(fc_in_num, feature_dim)
    return net


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.feature = resnet50(feature_dim=128, pretrained=True)
        self.norm = Normalize(power=2)

    def forward(self, x):
        y = self.feature(x)
        yn = self.norm(y)
        return yn


class TBDataset(Dataset):
    def __init__(self, image_list, image_folder):
        super(TBDataset, self).__init__()
        self.data_files = list()
        self.initialize(image_list, image_folder)

    def __getitem__(self, index):
        image_file = self.data_files[index]
        try:
            image_data = Image.open(image_file)
        except:
            image_data = Image.open(self.data_files[10])

        x1 = image_data.crop([0, 0, 256, 256])
        t1 = data_transform(x1)
        x2 = image_data.crop([256, 0, 512, 256])
        x3 = image_data.crop([512, 0, 768, 256])
        x4 = image_data.crop([768, 0, 1024, 256])
        x5 = image_data.crop([1024, 0, 1280, 256])

        datas = list()
        datas.append(x2)
        datas.append(x3)
        datas.append(x4)
        datas.append(x5)
        k = np.random.randint(4)
        t2 = data_transform(datas[k])
        return t1, t2

    def __len__(self):
        return len(self.data_files)

    def initialize(self, image_list, image_folder):
        for image in image_list:
            if not image.endswith('.jpg'):
                continue
            image_file = os.path.join(image_folder, image)
            self.data_files.append(image_file)


class TB2Dataset(Dataset):
    def __init__(self, root):
        super(TB2Dataset, self).__init__()
        self.data_files = list()
        self.initialize(root)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, item):
        try:
            image_file = self.data_files[item]
            image_data = Image.open(image_file)
            x1 = data_transform(image_data)
            x2 = data_transform(image_data)
            return x1, x2
        except:
            return self.__getitem__(100)

    def initialize(self, root):
        children = os.listdir(root)
        for child in children:
            folder = os.path.join(root, child)
            if not os.path.isdir(folder):
                continue
            images_list = os.listdir(folder)
            for str_image in images_list:
                if not str_image.endswith('.png') and not str_image.endswith('.jpg'):
                    continue
                image_file = os.path.join(folder, str_image)
                self.data_files.append(image_file)


class NCELoss(nn.Module):
    def __init__(self, T, batch_size=128):
        super(NCELoss, self).__init__()
        self.T = T
        self.batch_size = batch_size

    def forward(self, y):
        y1 = y.narrow(0, 0, self.batch_size)
        y2 = y.narrow(0, self.batch_size, self.batch_size)
        num = y1.size(0)
        y3 = torch.mm(y1, y2.t()).div_(self.T).exp_()
        y4 = y3 / y3.sum(1, keepdim=True)
        prob_p = y4.diag().log_().sum(0)

        y6 = torch.mm(y1, y1.t()).div_(self.T).exp_()
        y7 = (1.0 - y6 / y6.sum(1, keepdim=True)).log_()
        y8 = y7.sum(1) - y7.diag()
        prob_n = y8.sum(0)
        loss = (-prob_p - prob_n) / num
        return loss


def start_train():
    image_folder = 'datas'
    datas = json.load(open('tb_datas.json', 'r'))
    tb_datas = TBDataset(datas['train'], image_folder)

    print('==> start training loop...')
    nbs = 64
    batches = int(len(tb_datas) / nbs)
    print('=> data num: ', len(tb_datas))
    data_loader = DataLoader(tb_datas, batch_size=nbs, shuffle=True, num_workers=4, drop_last=True)
    net = EmbeddingNet()
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load('output/tbs-res50.pth'))
    net = net.cuda()

    net.train()
    opt = optimizer.SGD(net.parameters(), lr=0.003, momentum=0.8, weight_decay=5e-4)
    criterion = NCELoss(0.1, batch_size=nbs)
    criterion = criterion.cuda()

    epochs = 100
    for epoch in range(epochs):
        losses = 0.0
        for index, (datas, targets) in enumerate(data_loader):
            x = torch.cat((datas, targets), dim=0).cuda()
            y = net(x)
            loss = criterion(y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses += loss.item()
            if index % 10 == 0:
                print('==> Epoch: training [%d]/[%d]-[%d]/[%d], batch loss = %f' %
                      (epoch, epochs, index, batches, loss.item()))

        train_loss = losses / len(data_loader)
        print('==> Average loss:  %f' % train_loss)
        torch.save(net.state_dict(), 'output/tbs-res50.pth')


if __name__ == '__main__':
    start_train()





