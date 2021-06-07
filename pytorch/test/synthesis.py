import os
import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image

image_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

data_root = '/Users/chenxiang/Downloads/Gitlab/Deepmind/datas'

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(10, 256, 4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        y = self.conv(x)
        return y

class ConditionGenerator(nn.Module):
    def __init__(self):
        super(ConditionGenerator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(10, 128, 4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(10, 128, 4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, condition):
        y1 = self.conv1(x)
        y2 = self.conv2(condition)
        x1 = torch.cat([y1,y2], dim=1)
        y = self.conv3(x1)
        return y


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02),

            nn.Conv2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02),

            nn.Conv2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),

            nn.Conv2d(64, 1, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size(0), -1)
        return y


class ConditionDiscriminator(nn.Module):
    def __init__(self):
        super(ConditionDiscriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02),

            nn.Conv2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),

            nn.Conv2d(64, 1, 4),
            nn.Sigmoid()
        )

    def forward(self, x, condition):
        y1 = self.conv1(x)
        y2 = self.conv2(condition)
        x1 = torch.cat([y1, y2], dim=1)
        y = self.conv(x1)
        y = y.view(y.size(0), -1)
        return y


def start_train():
    print('==> start training loop ...')
    D = Discriminator()
    G = Generator()

    batch_size = 128
    datas = datasets.MNIST(data_root, train=True, transform=image_transform)
    data_loader = DataLoader(datas, shuffle=True, batch_size=batch_size)
    print('==> train datas Num: ', len(datas))

    criterion = nn.BCELoss()
    d_optimizer = optimizer.Adam(D.parameters(), lr=0.005)
    g_optimizer = optimizer.Adam(G.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        D.train()
        G.train()

        for index, (data, label) in enumerate(data_loader):
            x, _ = data, label
            real_y = torch.ones(x.size(0), 1)
            fake_y = torch.zeros(x.size(0), 1)
            real_p = D(x)

            noise = torch.randn(x.size(0), 10, 1, 1)
            fake_x = G(noise)
            fake_p = D(fake_x)

            real_score = real_p.mean().item()
            fake_score = fake_p.mean().item()

            real_loss = criterion(real_p, real_y)
            fake_loss = criterion(fake_p, fake_y)
            d_loss = real_loss + fake_loss

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            g_losses = 0.0
            for i in range(3):
                noise = torch.randn(x.size(0), 10, 1, 1)
                fake_x = G(noise)
                fake_p = D(fake_x)
                g_loss = criterion(fake_p, real_y)
                g_losses += g_loss.item()

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            if index % 50 == 0:
                print('epoch: [%d]/[%d]-[%d]/[%d], d_loss = %f, g_loss = %f, real_score = %f, fake_score = %f'
                      %(epoch, epochs, index, len(data_loader), d_loss.item(), g_loss.item(), real_score, fake_score))

        state = {
            'D': D.state_dict(),
            'G': G.state_dict(),
        }

        torch.save(state,  'mnistGan.pth')
        G.eval()
        noise = torch.randn(100, 10, 1, 1)
        fake_x = G(noise)
        fake_x = (fake_x + 1) * 0.5
        fake_x = fake_x.clamp(0, 1)
        save_image(fake_x, os.path.join('images', str(epoch) +'.png'), nrow=10)


def condition_train():
    print('==> start training loop ...')
    D = ConditionDiscriminator()
    G = ConditionGenerator()

    batch_size = 128
    datas = datasets.MNIST(data_root, train=True, transform=image_transform)
    data_loader = DataLoader(datas, shuffle=True, batch_size=batch_size)
    print('==> train datas Num: ', len(datas))

    criterion = nn.BCELoss()
    d_optimizer = optimizer.Adam(D.parameters(), lr=0.005)
    g_optimizer = optimizer.Adam(G.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        D.train()
        G.train()

        for index, (data, label) in enumerate(data_loader):
            x, y = data, label
            real_y = torch.ones(x.size(0), 1)
            fake_y = torch.zeros(x.size(0), 1)

            dc = torch.zeros(x.size(0), 10, 32, 32)
            dc[range(x.size(0)), label, :, :] = 1
            real_p = D(x, dc)

            gc = torch.zeros(x.size(0), 10, 1, 1)
            gc[range(x.size(0)), label, :, :] = 1

            noise = torch.randn(x.size(0), 10, 1, 1)
            fake_x = G(noise, gc)
            fake_p = D(fake_x, dc)

            real_score = real_p.mean().item()
            fake_score = fake_p.mean().item()

            real_loss = criterion(real_p, real_y)
            fake_loss = criterion(fake_p, fake_y)
            d_loss = real_loss + fake_loss

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            g_losses = 0.0
            for i in range(3):
                noise = torch.randn(x.size(0), 10, 1, 1)
                fake_x = G(noise, gc)
                fake_p = D(fake_x, dc)
                g_loss = criterion(fake_p, real_y)
                g_losses += g_loss.item()

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            if index % 50 == 0:
                print('epoch: [%d]/[%d]-[%d]/[%d], d_loss = %f, g_loss = %f, real_score = %f, fake_score = %f'
                      % (epoch, epochs, index, len(data_loader), d_loss.item(), g_loss.item(), real_score, fake_score))

        state = {
            'D': D.state_dict(),
            'G': G.state_dict(),
        }

        torch.save(state, 'mnistcGan.pth')
        G.eval()
        noise = torch.randn(100, 10, 1, 1)
        condition = torch.zeros(100, 10, 1, 1)
        label = [int(x/10) for x in range(100)]
        condition[range(100), label, :, :] = 1
        fake_x = G(noise, condition)
        fake_x = (fake_x + 1) * 0.5
        fake_x = fake_x.clamp(0, 1)
        save_image(fake_x, os.path.join('images', 'condition_' + str(epoch) + '.png'), nrow=10)


if __name__ == '__main__':
    # start_train()
    condition_train()