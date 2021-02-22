import cgan
import cdcgan

import torch
import torch.nn as nn

from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import os
import numpy as np
from PIL import Image
from common_path import *

image_path = os.path.join(output_path, 'images')

cuda = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def to_image(x):
    y = (x + 1) * 0.5
    y = y.clamp(0, 1)
    return y


def train_cGAN():
    param = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    data_loader = DataLoader(
        datasets.MNIST(mnist_path, train=True, transform=param, download=True),
        batch_size=256, shuffle=True
    )

    D = cgan.Discriminator()
    D.apply(weights_init)
    D.to(cuda)
    D.train()

    G = cgan.Generator()
    G.apply(weights_init)
    G.to(cuda)
    G.train()

    print('Start Training Loop....')
    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

    for epoch in range(20):
        for index, (data, label) in enumerate(data_loader):
            n_batch = data.size(0)
            data_flatten = data.view(n_batch, -1)
            label_onehot = torch.zeros(n_batch, 10).scatter_(1, label.view(n_batch, 1), 1)

            real_data = data_flatten.to(cuda)
            real_label = torch.ones(n_batch, 1).to(cuda)
            real_out = D(real_data, label_onehot)

            noise = torch.randn(n_batch, 100).to(cuda)
            fake_data = G(noise, label_onehot)
            fake_label = torch.zeros(n_batch, 1).to(cuda)
            fake_out = D(fake_data, label_onehot)

            real_score = real_out.mean().item()
            fake_score = fake_out.mean().item()

            d_real_loss = criterion(real_out, real_label)
            d_fake_loss = criterion(fake_out, fake_label)

            d_loss = d_real_loss + d_fake_loss
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            for idx in range(10):
                fake_data = G(noise, label_onehot)
                fake_out = D(fake_data, label_onehot)
                g_loss = criterion(fake_out, real_label)

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            if (index + 1) % 100 == 0:
                print('Epoch: %d, real_score = %f, fake_score = %f, d_loss = %f, g_loss = %f' % (
                    epoch, real_score, fake_score, d_loss.item(), g_loss.item()
                ))

            if (index + 1) % 200 == 0:
                fake_data = fake_data.view(fake_data.size(0), 1, 28, 28)
                image = to_image(fake_data)
                image_name = 'cGAN' + str(epoch) + '-' + str(index + 1) + '.png'
                save_image(image, os.path.join(IMAGE_PATH, image_name), nrow=16)

        torch.save(D.state_dict(), os.path.join(MODEL_PATH, 'cGAN_D.pkl'))
        torch.save(D.state_dict(), os.path.join(MODEL_PATH, 'cGAN_G.pkl'))

        noise = torch.randn(100, 100).to(cuda)
        label = torch.zeros(100, 10).to(cuda)
        for k in range(10):
            label[k * 10:(k + 1) * 10, k] = 1
        fake_data = G(noise, label)
        fake_data = fake_data.view(fake_data.size(0), 1, 28, 28)
        image = to_image(fake_data)
        image_name = 'cGAN' + str(epoch) + '.png'
        save_image(image, os.path.join(IMAGE_PATH, image_name), nrow=10)


def train_cdcGAN():
    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_datas = datasets.MNIST(mnist_path, train=True, transform=data_transform)
    test_datas = datasets.MNIST(mnist_path, train=False, transform=data_transform)

    train_loader = DataLoader(train_datas, batch_size=256, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_datas, batch_size=256, shuffle=True)

    D = cdcgan.Discriminator()
    D.apply(weights_init)
    D.cuda()

    G = cdcgan.Generator()
    G.apply(weights_init)
    G.cuda()

    print('Start Training Loop....')
    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

    for epoch in range(20):
        D.train()
        G.train()
        for index, (data, label) in enumerate(train_loader):
            n_batch = data.size(0)
            label_matrix = torch.zeros(n_batch, 10, 32, 32)
            label_matrix[np.arange(n_batch), label.numpy(), :, :] = 1
            label_matrix = label_matrix.cuda()

            real_data = data.cuda()
            real_label = torch.ones(n_batch, 1).to(cuda)
            real_out = D(real_data, label_matrix)

            noise = torch.randn(n_batch, 100, 1, 1).cuda()
            label_onehot = torch.zeros(n_batch, 10).scatter_(1, label.view(n_batch, 1), 1).view(n_batch, 10, 1,
                                                                                                1).cuda()
            fake_data = G(noise, label_onehot)
            fake_label = torch.zeros(n_batch, 1).cuda()
            fake_out = D(fake_data, label_matrix)

            real_score = real_out.mean().item()
            fake_score = fake_out.mean().item()

            d_real_loss = criterion(real_out, real_label)
            d_fake_loss = criterion(fake_out, fake_label)

            d_loss = d_real_loss + d_fake_loss
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            for idx in range(10):
                fake_data = G(noise, label_onehot)
                fake_out = D(fake_data, label_extd)
                g_loss = criterion(fake_out, real_label)

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            if (index + 1) % 100 == 0:
                print('Epoch: %d, real_score = %f, fake_score = %f, d_loss = %f, g_loss = %f' % (
                    epoch, real_score, fake_score, d_loss.item(), g_loss.item()
                ))

            if (index + 1) % 200 == 0:
                image = to_image(fake_data)
                image_name = 'cDCGAN' + str(epoch) + '-' + str(index + 1) + '.png'
                save_image(image, os.path.join(IMAGE_PATH, image_name), nrow=16)

        torch.save(D.state_dict(), os.path.join(MODEL_PATH, 'cDCGAN_D.pkl'))
        torch.save(D.state_dict(), os.path.join(MODEL_PATH, 'cDCGAN_G.pkl'))

        noise = torch.randn(100, 100, 1, 1).to(cuda)
        label = torch.zeros(100, 10).to(cuda)
        for k in range(10):
            label[k * 10:(k + 1) * 10, k] = 1
        label = label.view(100, 10, 1, 1)
        fake_data = G(noise, label)
        image = to_image(fake_data)
        image_name = 'cDCGAN' + str(epoch) + '.png'
        save_image(image, os.path.join(IMAGE_PATH, image_name), nrow=10)


def main():
    # train_cGAN()
    train_cdcGAN()


if __name__ == '__main__':
    main()