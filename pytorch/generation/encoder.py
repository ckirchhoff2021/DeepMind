import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image
import torch.optim as optimizer
from common_path import *

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.down_sampling = self._make_down_sample_layers_(3, 16, 6)
        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )

        self.up_sampling = self._make_up_sample_layers_(16, 3, 6)
        self.fc2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True)
        )

        # self.average_pool = nn.AdaptiveAvgPool2d((7, 7))

    def _make_down_sample_layers_(self, in_chn, out_chn, layer_cnt):
        layers = list()
        c1 = in_chn
        for i in range(layer_cnt):
            c2 = c1 * 2 if i < layer_cnt - 1 else out_chn
            layers += [
                nn.Conv2d(c1, c2, 4, stride=2, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True)
            ]
            c1 = c2
        return nn.Sequential(*layers)

    def _make_up_sample_layers_(self, in_chn, out_chn, layer_cnt):
        layers = list()
        c1 = in_chn
        for i in range(layer_cnt):
            c2 = c1 * 2 if i < layer_cnt - 1 else out_chn
            layers += [
                nn.ConvTranspose2d(c1, c2, 4, stride=2, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True)
            ]
            c1 = c2
        return nn.Sequential(*layers)

    def forward(self, x):
        d1 = self.down_sampling(x)
        d1 = d1.view(d1.size(0), -1)
        f1 = self.fc1(d1)
        f2 = self.fc2(f1)
        f2 = f2.view(f2.size(0), 16, 4, 4)
        u1 = self.up_sampling(f2)
        return u1





if __name__ == '__main__':
    net = AutoEncoder()
    x = torch.randn(1,3,256,256)
    y = net(x)
    print(y.size())

    file = '/Users/chenx/Downloads/s1.jpg'
    image_data = Image.open(file)

    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image_tensor = image_transform(image_data)
    x = image_tensor.unsqueeze(0)

    optimizer = optimizer.Adam(net.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    '''
    net.train()
    for epoch in range(100):
        y = net(x)
        loss = criterion(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('==> Epoch: [%d]/[%d], loss=%f' %(epoch, 100, loss.item()))
        if epoch % 10 == 0:
            torch.save(net.state_dict(), os.path.join(output_path, 'eocoder.pth'))
    '''

    state = torch.load(os.path.join(output_path, 'eocoder.pth'))
    net.load_state_dict(state)
    net.eval()

    y = net(x)
    reverse = transforms.ToPILImage()
    yp = reverse(y.squeeze())
    xp = reverse(x.squeeze())
    xp.show()
    yp.show()

