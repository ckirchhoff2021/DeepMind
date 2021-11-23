import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

from PIL import Image
import torch.optim as optimizer
import torch.nn.functional as F
from collections import namedtuple
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from datas import *
from torch.utils.tensorboard import SummaryWriter


cuda = torch.cuda.is_available()



def calc_mean_std(features):
    """

    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std


def adain(content_features, style_features):
    """
    Adaptive Instance Normalization

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features


class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images, output_last_feature=False):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        if output_last_feature:
            return h4
        else:
            return h1, h2, h3, h4


class RC(nn.Module):
    """A wrapper of ReflectionPad2d and Conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1, activated=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.activated = activated

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        if self.activated:
            return F.relu(h)
        else:
            return h


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc1 = RC(512, 256, 3, 1)
        self.rc2 = RC(256, 256, 3, 1)
        self.rc3 = RC(256, 256, 3, 1)
        self.rc4 = RC(256, 256, 3, 1)
        self.rc5 = RC(256, 128, 3, 1)
        self.rc6 = RC(128, 128, 3, 1)
        self.rc7 = RC(128, 64, 3, 1)
        self.rc8 = RC(64, 64, 3, 1)
        self.rc9 = RC(64, 3, 3, 1, False)

    def forward(self, features):
        h = self.rc1(features)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc2(h)
        h = self.rc3(h)
        h = self.rc4(h)
        h = self.rc5(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc6(h)
        h = self.rc7(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc8(h)
        h = self.rc9(h)
        return h


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_encoder = VGGEncoder()
        self.decoder = Decoder()

    def generate(self, content_images, style_images, alpha=1.0):
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(style_images, output_last_feature=True)
        t = adain(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features
        out = self.decoder(t)
        return out

    @staticmethod
    def calc_content_loss(out_features, t):
        return F.mse_loss(out_features, t)

    @staticmethod
    def calc_style_loss(content_middle_features, style_middle_features):
        loss = 0
        for c, s in zip(content_middle_features, style_middle_features):
            c_mean, c_std = calc_mean_std(c)
            s_mean, s_std = calc_mean_std(s)
            loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
        return loss

    def forward(self, content_images, style_images, alpha=1.0, lam=10):
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(style_images, output_last_feature=True)
        t = adain(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features
        out = self.decoder(t)

        output_features = self.vgg_encoder(out, output_last_feature=True)
        output_middle_features = self.vgg_encoder(out, output_last_feature=False)
        style_middle_features = self.vgg_encoder(style_images, output_last_feature=False)

        loss_c = self.calc_content_loss(output_features, t)
        loss_s = self.calc_style_loss(output_middle_features, style_middle_features)
        loss = loss_c + lam * loss_s
        return loss


def recover_tensor(image_tensor, cuda=True):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    if cuda: mean, std = mean.cuda(), std.cuda()
    out = image_tensor * std + mean
    out = out.clamp(0, 1)
    return out


def start_train():
    print('==> start training loop...')
    style_datas = StyleDataset()
    print('==> datas: ', len(style_datas))

    batch_size = 8
    train_loader = DataLoader(style_datas, batch_size=batch_size, shuffle=True)
    test_loader =  DataLoader(style_datas, batch_size=batch_size, shuffle=True)
    test_iter = iter(test_loader)
    batches = int(len(style_datas) / batch_size) + 1

    net = Model()
    if cuda: net.cuda()

    opt = optimizer.Adam(net.parameters(), lr=5e-5)
    epochs = 20
    net.train()

    summary = SummaryWriter('out')
    for epoch in range(epochs):
        losses = 0.0
        for index, (inputs, targets) in enumerate(train_loader):
            content, style = inputs, targets
            if cuda: content, style = inputs.cuda(), targets.cuda()

            loss = net(content, style)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses += loss.item()
            summary.add_scalar('train/batch_loss', loss.item())

            print('==> Epoch: [%d]/[%d]-[%d]/[%d], batch loss = %f' % (epoch, epochs, index, batches, loss.item()))
            if index % 1000 == 0:
                content, style = next(test_iter)
                if cuda: content, style = content.cuda(), style.cuda()
                with torch.no_grad():
                    out = net.generate(content, style)
                content = recover_tensor(content, cuda=cuda)
                style = recover_tensor(style, cuda=cuda)
                out = recover_tensor(out, cuda=cuda)
                res = torch.cat([content, style, out], dim=0)
                res = res.to('cpu')
                save_image(res, 'out/' + str(epoch) +'-' + str(index) + '.png', nrow=batch_size)

        train_loss = losses / len(train_loader)
        summary.add_scalar('train/epoch_loss', train_loss)

        print('==> Epoch: [%d]/[%d], train loss = %f' % (epoch, epochs, train_loss) )
        torch.save(net.state_dict(), 'out/style-net.pth')


def test():
    net = Model()
    state = torch.load('output/out/style-net.pth', map_location='cpu')
    net.load_state_dict(state)
    net.eval()

    content = 'pytorch/style/res/content/lenna.jpg'
    style = 'pytorch/style/res/style/sky.jpg'
    out = net.generate(content, style)
    out.show()




if __name__ == '__main__':
    start_train()
    # net = models.vgg19()
    # print(net.features)
    # test()
