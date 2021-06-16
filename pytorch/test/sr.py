import os
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torch.optim as optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid

from torch.utils.tensorboard import SummaryWriter


data_root = '/Users/chenxiang/Downloads/dataset/DIV2K/DIV2K_valid_HR'

randomCrop = transforms.RandomCrop((128, 128))

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.trunk = models.vgg16(pretrained=True).eval()
        self.feature = self.trunk.features[:22]
        for param in self.feature.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        y1 = self.feature(x1)
        y2 = self.feature(x2)
        y = F.mse_loss(y1, y2)
        return y


class DIV2KDatas(Dataset):
    def __init__(self):
        super(DIV2KDatas, self).__init__()
        self.files = list()
        self.initialize()

    def initialize(self):
        image_files = os.listdir(data_root)
        for file in image_files:
            if not file.endswith('.png'):
                continue
            self.files.append(os.path.join(data_root, file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        image_data = Image.open(file)
        image_HR = randomCrop(image_data)
        image_LR = image_HR.resize((32,32))
        x1 = image_transform(image_LR)
        x2 = image_transform(image_HR)
        return x1, x2



class SRResnet(nn.Module):
    def __init__(self):
        super(SRResnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1)
        )

        self.upSampling = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1) + y1
        y3 = self.upSampling(y2)
        y4 = self.conv3(y3)
        return y4


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        y = self.conv(x) + x
        return y


class SRGenerator(nn.Module):
    def __init__(self):
        super(SRGenerator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        residual_list = list()
        for i in range(6):
            residual_list.append(ResidualBlock())
        self.residual_blocks = nn.Sequential(*residual_list)

        self.upSampling = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Conv2d(64,3,3,1,1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.residual_blocks(y1)
        y3 = self.upSampling(y2)
        y = self.conv2(y3)
        return y
    
    
class SRDiscriminator(nn.Module):
    def __init__(self):
        super(SRDiscriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.02, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 4, 2, 1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, 4, 2, 1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.BatchNorm2d(512)
        )

        self.fc = nn.Sequential(
            nn.Linear(32768, 1024),
            nn.LeakyReLU(0.02),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = y2.view(y2.size(0), -1)
        y4 = self.fc(y3)
        return y4


def postprocess(x):
    mean = torch.tensor([0.485, 0.456, 0.406])
    mean = mean.view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225])
    std = std.view(1,3,1,1)
    y = x * std + mean
    y = y.clamp(0, 1)
    return y


def start_train():
    print('==> start training loop ...... ')
    datas = DIV2KDatas()
    print('==> train datas: ', len(datas))
    data_loader = DataLoader(datas, shuffle=True, batch_size=4)
    net = SRResnet()
    criterion = nn.L1Loss()
    opt = optimizer.Adam(net.parameters(), lr=0.002)
    epochs = 50
    net.train()

    summary = SummaryWriter(comment='sr_loss')
    for epoch in range(epochs):
        losses = 0.0
        for index, (x1, x2) in enumerate(data_loader):
            y1 = net(x1)
            loss = criterion(y1, x2)
            summary.add_scalar('train/loss', loss.item(), index + len(data_loader) * epoch)

            v1 = postprocess(y1)
            v2 = postprocess(x2)

            images = make_grid(torch.cat([v1, v2], dim=0), nrow=4)
            summary.add_image('train/images', images)

            if index % 10 == 0:
                print('==> Epoch:[%d]/[%d]-[%d]/[%d], loss = %f' % (epoch, epochs, index, len(data_loader), loss.item()))

            losses += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()

        print('==> Epoch:[%d], average loss = %f' % (epoch, losses / len(data_loader)))
        torch.save(net, os.path.join('output', 'srresnet.pth'))

    summary.close()


def gan_train():
    print('==> start training loop ...... ')
    datas = DIV2KDatas()
    print('==> train datas: ', len(datas))
    data_loader = DataLoader(datas, shuffle=True, batch_size=4, num_workers=4)
    G = SRGenerator()
    D = SRDiscriminator()

    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss()
    criterion = nn.BCELoss()

    d_opt = optimizer.Adam(D.parameters(), lr=0.0005)
    g_opt = optimizer.Adam(G.parameters(), lr=0.0005)
    epochs = 50
    D.train()
    G.train()

    summary = SummaryWriter(comment='SRGAN_loss')
    for epoch in range(epochs):
        d_losses = 0.0
        g_losses = 0.0
        for index, (x1, x2) in enumerate(data_loader):
            real_y = torch.ones(x1.size(0), 1)
            fake_y = torch.zeros(x1.size(0), 1)

            y1 = G(x1)
            fake_p = D(y1)
            real_p = D(x2)

            fake_loss = criterion(fake_p, fake_y)
            real_loss = criterion(real_p, real_y)
            d_loss = fake_loss + real_loss
            summary.add_scalar('train/d_loss', d_loss.item(), index + len(data_loader) * epoch)

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            y1 = G(x1)
            real_p = D(y1)
            loss1 = l1_loss(y1, x2)
            loss2 = perceptual_loss(y1, x2)
            loss3 = criterion(real_p, real_y)
            g_loss = loss1 + loss2 * 0.002 + loss3 * 0.005
            summary.add_scalar('train/g_loss', g_loss.item(), index + len(data_loader) * epoch)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            v1 = postprocess(y1)
            v2 = postprocess(x2)
            images = make_grid(torch.cat([v1, v2], dim=0), nrow=4)
            summary.add_image('train/images', images)

            if index % 10 == 0:
                print('==> Epoch:[%d]/[%d]-[%d]/[%d], d_loss = %f, g_loss = %f, real_score = %f, fake_score = %f' %
                      (epoch, epochs, index, len(data_loader), d_loss.item(), g_loss.item(), real_p.mean().item(), fake_p.mean().item()))

            d_losses += d_loss.item()
            g_losses += g_loss.item()

        print('==> Epoch:[%d], average d_loss = %f, g_loss = %f' % (epoch, d_losses / len(data_loader), g_losses / len(data_loader)))
        summary.add_scalar('train/average_d_loss', d_losses / len(data_loader), index + len(data_loader) * epoch)
        summary.add_scalar('train/average_g_loss', g_losses / len(data_loader), index + len(data_loader) * epoch)

        torch.save(G, os.path.join('output', 'srGan.pth'))

    summary.close()




if __name__ == '__main__':
    # start_train()
    # net = models.vgg16(pretrained=True)
    # func = net.features[:22]
    # print(func)

    # perceptual = PerceptualLoss()
    # x1 = torch.randn(1,3,128,128)
    # y1 = perceptual(x1)
    # print(y1.size())

    # G = SRGenerator()
    # x1 = torch.randn(1, 3, 32, 32)
    # y1 = G(x1)
    # print(y1.size())

    # D = SRDiscriminator()
    # x1 = torch.randn(1, 3, 128, 128)
    # y1 = D(x1)
    # print(y1.size())

    gan_train()


