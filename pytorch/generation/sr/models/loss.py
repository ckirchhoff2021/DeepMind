import torch
import torch.nn as nn
from torchvision.models.vgg import vgg19


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.l2_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=2e-8):
        super(TVLoss, self). __init__()
        self.tv_loss_weight = tv_loss_weight
    
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def main():
    net = PerceptualLoss()
    print(net)

    x = torch.randn(12, 3, 256, 256)
    f = net.loss_network
    y = f(x)
    print(y.size())
    

if __name__ == '__main__':
    main()