import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

cuda  = torch.cuda.is_available()


def resnet18(feature_dim=128, pretrained=False):
    net = models.resnet18(pretrained=pretrained)
    fc_in_num = net.fc.in_features
    net.fc = nn.Linear(fc_in_num, feature_dim)
    return net


data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class RoomDataset(Dataset):
    def __init__(self, image_folder):
        super(RoomDataset, self).__init__()
        self.data_files = list()
        self.initialize(image_folder)

    def __getitem__(self, index):
        image_file = self.data_files[index]
        try:
            image_data = Image.open(image_file)
        except:
            image_data = Image.open(self.data_files[10])

        angles = [90, 180, 270]
        k1 = np.random.randint(3)
        A1 = image_data.rotate(angles[k1])
        k2 = np.random.randint(3)
        A2 = image_data.rotate(angles[k2])
        data = target_transform(A1)
        target = target_transform(A2)
        return data, target

    def __len__(self):
        return len(self.data_files)

    def initialize(self, image_folder):
        image_list = os.listdir(image_folder)
        for image in image_list:
            if not image.endswith('.jpg'):
                continue
            image_file = os.path.join(image_folder, image)
            self.data_files.append(image_file)


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.feature = resnet18(feature_dim=128, pretrained=True)
        self.norm = Normalize(power=2)

    def forward(self, x):
        y = self.feature(x)
        yn = self.norm(y)
        return yn


class NCELoss(nn.Module):
    def __init__(self, T):
        super(NCELoss, self).__init__()
        self.T = T

    def forward(self, y):
        y1 = y.narrow(0, 0, 128)
        y2 = y.narrow(0, 128, 128)
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
    print('==> start training loop...')
    room_datas = RoomDataset(image_folder)
    batches = int(len(room_datas) / 128)
    print('=> data num: ', len(room_datas))
    data_loader = DataLoader(room_datas, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    net = EmbeddingNet()
    state = torch.load('output/res18-120.pth')
    net.load_state_dict(state)
    net = net.cuda() if cuda else net
    net.train()

    opt = optimizer.SGD(net.parameters(), lr=0.003, momentum=0.8, weight_decay=5e-4)
    criterion = NCELoss(0.1)
    criterion = criterion.cuda() if cuda else criterion

    epochs = 20
    for epoch in range(epochs):
        losses = 0.0
        for index, (datas, targets) in enumerate(data_loader):
            x = torch.cat((datas, targets), dim=0)
            x = x.cuda()
            y = net(x)
            loss = criterion(y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses += loss.item()
            if index % 10 == 0:
                print('==> Epoch: [%d]/[%d]-[%d]/[%d], batch loss = %f' % (epoch, epochs, index, batches, loss.item()))
        train_loss = losses / len(data_loader)
        print('==> Average loss:  %f' % train_loss)
        torch.save(net.state_dict(), 'output/res18-120-plus.pth')


if __name__ == '__main__':
    start_train()




