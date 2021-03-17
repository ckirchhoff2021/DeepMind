import torch
import torch.nn as nn
import torch.optim as optimizer
from torchvision import models, transforms
from PIL import Image
from common_path import *
from tensorboardX import SummaryWriter

import cv2
import numpy as np



class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.net = models.vgg19(pretrained=True)
        # print(self.net)
        self.features = self.net.features
        self.layers = list(self.features.children())
        self.middle_feature = nn.Sequential(*self.layers[:17])
        self.deep_features = nn.Sequential(*self.layers[:26])

    def forward(self, x):
        y1 = self.middle_feature(x)
        y2 = self.deep_features(x)
        return y1, y2


class AutoEncoder(nn.Module):
    def __init__(self, sample_cnt):
        super(AutoEncoder, self).__init__()
        self.sample_cnt = sample_cnt
        self.down_sampling = self._make_downsampling_layers_(3, 16)
        self.up_sampling = self._make_upsampling_layers(16, 3)
        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True)
        )

    def _make_downsampling_layers_(self, in_chn, out_chn):
        layers = list()
        c1 = in_chn
        for i in range(self.sample_cnt):
            c2 = c1 * 2 if i < self.sample_cnt - 1 else out_chn
            layers += [
                nn.Conv2d(c1, c2, 4, stride=2, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True)
            ]
            c1 = c2
        return nn.Sequential(*layers)

    def _make_upsampling_layers(self, in_chn, out_chn):
        layers = list()
        c1 = in_chn
        for i in range(self.sample_cnt):
            c2 = c1 * 2 if i < self.sample_cnt - 1 else out_chn
            layers += [
                nn.ConvTranspose2d(c1, c2, 4, stride=2, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True)
            ]
            c1 = c2
        return nn.Sequential(*layers)


    def forward(self, x):
        y1 = self.down_sampling(x)
        y1 = y1.view(y1.size(0), -1)
        y1 = self.fc1(y1)
        y2 = self.fc2(y1)
        y2 = y2.view(y2.size(0), 16, 4, 4)
        y2 = self.up_sampling(y2)
        return y2

def main():
    net = AutoEncoder(6)
    x = torch.randn(1,3,256,256)
    y = net(x)
    print(y.size())
    
    image_file = os.path.join(data_path, 'gauss.png')
    image_data = Image.open(image_file)
    image_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    reverse = transforms.ToPILImage()

    x = image_transform(image_data)
    x = x.unsqueeze(0)

    criterion = nn.MSELoss()
    opt = optimizer.Adam(net.parameters(), lr=0.005)
    net.train()

    perceptural_loss = VGGLoss()
    for param in perceptural_loss.parameters():
        param.requires_grad = False


    writer = SummaryWriter(os.path.join(output_path, 'summary'))

    for epoch in range(300):
        y = net(x)
        x1, x2 = perceptural_loss(x)
        y1, y2 = perceptural_loss(y)

        mse = criterion(x, y)
        l1 = criterion(x1, y1)
        l2 = criterion(x2, y2)
        loss = mse + 0.05 * l1 + 0.06 * l2

        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('mse', mse, global_step=epoch)
        writer.add_scalar('l1', l1, global_step=epoch)
        writer.add_scalar('l2', l2, global_step=epoch)

        if epoch % 10 == 0:
            print('==> epoch: [%d]/[%d], loss = %f' % (epoch, 100, loss.item()))
            torch.save(net.state_dict(), os.path.join(output_path, 'encoder.pth'))

    '''
    net.load_state_dict(torch.load(os.path.join(output_path, 'encoder.pth')))
    net.eval()

    x1 = reverse(x.squeeze())
    y = net(x)
    y1 = reverse(y.squeeze())
    new_image = Image.new('RGB', (512,256))
    new_image.paste(x1, (0,0, 256, 256))
    new_image.paste(y1, (256,0,512,256))
    new_image.show()
    '''


if __name__ == '__main__':
    main()

    '''
    k1 = cv2.getGaussianKernel(256, 128)
    k2 = k1.T
    k3 = k1 * k2
    k3 = k3 / np.max(k3) * 255.0
    k3 = k3.astype(np.uint8)
    heat = cv2.applyColorMap(k3, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(data_path, 'gauss.png'), heat)
    '''
