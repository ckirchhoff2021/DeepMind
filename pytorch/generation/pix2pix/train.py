import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from pix2pix.pix2pix_net import UnetGenerator, PixelDiscriminator
from torchvision.utils import save_image

data_folder = '/Users/chenxiang/Downloads/dataset/datas/facades'

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class FacadesDatas(Dataset):
    def __init__(self, train=False):
        super(FacadesDatas, self).__init__()
        self.train = train
        self.image_list = []
        self.initialize()

    def initialize(self):
        if self.train:
            image_folder = os.path.join(data_folder, 'train')
        else:
            image_folder = os.path.join(data_folder, 'test')

        image_list = os.listdir(image_folder)
        for image in image_list:
            if not image.endswith('.jpg'):
                continue
            self.image_list.append(os.path.join(image_folder, image))


    def __getitem__(self, index):
        image_file = self.image_list[index]
        image_data = Image.open(image_file)
        random = np.random.randint(2)
        if random == 0:
            image_data = image_data.transpose(Image.FLIP_LEFT_RIGHT)
        image_tensor = image_transform(image_data)
        data = image_tensor[:, :, 0:256]
        target = image_tensor[:, :, 256:]

        if random == 0:
            return data, target
        else:
            return target, data


    def __len__(self):
        return len(self.image_list)


def recover_image(data):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    value = data * std + mean
    value = value.clamp(0, 1)
    func = transforms.ToPILImage()
    return func(value)


def start_train():
    print('==> start training loop...')
    train_datas = FacadesDatas(train=True)
    train_loader = DataLoader(train_datas, batch_size=4, shuffle=True)

    print('==> train datas: ', len(train_datas))
    test_datas = FacadesDatas(train=False)
    test_loader = DataLoader(test_datas, batch_size=4, shuffle=True)
    print('==> test datas: ', len(test_datas))

    G = UnetGenerator(3, 3)
    g_optimizer = optimizer.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    D = PixelDiscriminator(3)
    d_optimizer = optimizer.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # gan_criterion = nn.BCELoss()
    gan_criterion = nn.MSELoss()
    epochs = 50
    batches = len(train_datas) / 4
    for epoch in range(epochs):
        G.train()
        D.train()
        for index, (datas, targets) in enumerate(train_loader):
            num = datas.size(0)
            outputs = G(datas)
            real_label = torch.ones(datas.size())
            real_out = D(targets)
            real_score = torch.mean(real_out)
            real_loss = gan_criterion(real_out, real_label)

            fake_label = torch.zeros(datas.size())
            fake_out = D(outputs)
            fake_score = torch.mean(fake_out)
            fake_loss = gan_criterion(fake_out, fake_label)

            d_loss = real_loss + fake_loss
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            outputs = G(datas)
            fake_out = D(outputs)
            # l2_loss = F.mse_loss(outputs, targets)
            l1_loss = F.l1_loss(outputs, targets)
            gan_loss = gan_criterion(fake_out, real_label)
            g_loss = gan_loss + l1_loss

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            print('==> [%d]/[%d]-[%d]/[%d], d_loss = %f, g_loss = %f' % (epoch, epochs, index, batches, d_loss.item(), g_loss.item()))
            print('--  real_score = %f, fake_score = %f, gan_loss = %f, l1_loss = %f' % (real_score.item(), fake_score.item(), gan_loss.item(), l1_loss.item()))

            fake_data = recover_image(outputs)
            str_image = 'output/' + str(epoch) + '-' + str(index) + '.jpg'
            save_image(fake_data, str_image, nrow=2)

            torch.save(G.state_dict(), 'output/net.pth')


def unit_test():
    state = torch.load('output/net.pth', map_location='cpu')
    G = UnetGenerator(3, 3)
    G.load_state_dict(state)
    G.eval()

    datas = FacadesDatas(train=False)
    x, y = datas[0]
    x = x.unsqueeze(0)

    y2 = G(x)
    f1 = recover_image(y2.squeeze())
    f2 = recover_image(y)
    f1.show()
    f2.show()



if __name__ == '__main__':
    # start_train()
    unit_test()

    # datas = FacadesDatas(train=False)
    # x, y = datas[0]
    # f1 = recover_image(x)
    # f1.show()

