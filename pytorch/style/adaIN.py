import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
import torch.optim as optimizer
import torch.nn.functional as F
from collections import namedtuple
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from datas import *


class VGGEncoder(nn.Module):
    def __init__(self, required_grad=False):
        super(VGGEncoder, self).__init__()
        vgg_pretrained = models.vgg19(pretrained=True).features
        vgg_pretrained = vgg_pretrained.eval()

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained[x])

        if not required_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        vgg_features = namedtuple('vgg_features', ['h1', 'h2', 'h3', 'h4'])
        features = vgg_features(h1, h2, h3, h4)
        return features


def compute_mean_std(features):
    a, b, w, h = features.size()
    mean_value = features.view(a, b, -1).mean(dim=2).view(a, b, 1, 1)
    std_value = features.view(a, b, -1).std(dim=2).view(a, b, 1, 1)
    return mean_value, std_value


def adaIN_transform(content_features, style_features):
    content_mean, content_std = compute_mean_std(content_features)
    style_mean, style_std = compute_mean_std(style_features)
    norm = style_std * (content_features - content_mean) / content_std + style_mean
    return norm


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )
        self.up_sample = nn.Upsample(mode='nearest', scale_factor=2)

    def forward(self, x):
        y = self.up_sample(self.conv1(x))
        y = self.up_sample(self.conv2(y))
        y = self.up_sample(self.conv3(y))
        y = self.conv4(y)
        return y


class AdaINNet(nn.Module):
    def __init__(self):
        super(AdaINNet, self).__init__()
        self.encode = VGGEncoder()
        self.decode = Decoder()

    @staticmethod
    def compute_content_loss(out_feature, content_feature):
        loss = F.mse_loss(out_feature, content_feature)
        return loss

    @staticmethod
    def compute_style_loss(content_features, style_features):
        loss = 0
        for c, s in zip(content_features, style_features):
            c_mean, c_std = compute_mean_std(c)
            s_mean, s_std = compute_mean_std(s)
            loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
        return loss

    def generate(self, content_file, style_file, alpha=1.0):
        content_image = Image.open(content_file).resize((256,256))
        content_tensor = image_transform(content_image).unsqueeze(0)
        style_image = Image.open(style_file).resize((256,256))
        style_tensor = image_transform(style_image).unsqueeze(0)
        content_features = self.encode(content_tensor)
        style_features = self.encode(style_tensor)
        f1 = adaIN_transform(content_features[3], style_features[3])
        f2 = alpha * f1 + (1.0 - alpha) * content_features[3]
        out = self.decode(f2)
        out_image = recover_tensor(out)
        recover_func = transforms.ToPILImage()
        out_image = recover_func(out_image[0])
        images = Image.new('RGB', (768, 256))
        images.paste(content_image, (0,0,256,256))
        images.paste(style_image, (256,0,512,256))
        images.paste(out_image, (512,0,768,256))
        return images

    def forward(self, content_tensor, style_tensor, alpha=1.0):
        content_features = self.encode(content_tensor)
        style_features = self.encode(style_tensor)
        f1 = adaIN_transform(content_features[3], style_features[3])
        f2 = alpha * f1 + (1.0 - alpha) * content_features[3]

        output = self.decode(f2)
        output_features = self.encode(output)
        style_features = self.encode(style_tensor)
        style_loss = self.compute_style_loss(output_features, style_features)
        content_loss = self.compute_content_loss(output_features[3], f2)
        loss = content_loss + style_loss * 10.0
        return loss


def recover_tensor(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    out = image_tensor * std + mean
    out = out.clamp(0, 1)
    return out


def start_train():
    print('==> start training loop...')
    style_datas = StyleDataset()
    print('==> datas: ', len(style_datas))
    data_loader = DataLoader(style_datas, batch_size=4, shuffle=True)
    batches = int(len(style_datas) / 4)

    net = AdaINNet()
    net.train()
    opt = optimizer.Adam(net.parameters(), lr=5e-5)
    epochs = 100
    for epoch in range(epochs):
        losses = 0.0
        for index, (content, style) in enumerate(data_loader):
            loss = net(content, style)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses += loss.item()
            print('==> Epoch: [%d]/[%d]-[%d]/[%d], batch loss = %f' % (epoch, epochs, index, batches, loss.item()))
            if index % 10 == 0:
                out = net.generate(content, style)
                out = recover_tensor(out)
                save_image(out, 'out/' + str(epoch) +'-' + str(index) + '.png', nrow=2)

        train_loss = losses / len(data_loader)
        print('==> Epoch: [%d]/[%d], train loss = %f' % (epoch, epochs, train_loss) )
        torch.save(net, 'out/style-net.pth')


def test():
    net = AdaINNet()
    state = torch.load('out/style-net.pth', map_location='cpu')
    net.load_state_dict(state)
    net.eval()
    out = net.generate('res/content/lenna.jpg', 'res/style/sky.jpg')
    out.show()



def main():
    net = VGGEncoder()
    x = torch.randn(1,3,256,256)
    h1, h2, h3, h4 = net(x)
    print(h4.size())

    decode = Decoder()
    y = decode(h4)
    print(y.size())



if __name__ == '__main__':
    # main()
    # start_train()
    # net = models.vgg19()
    # print(net.features)
    test()



