import os
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
from collections import namedtuple
from torch.utils.data import DataLoader, Dataset
from torchvision import models, datasets, transforms
from torchvision.utils import save_image
from style.transformNet import TransformNet



data_folder = '/Users/chenxiang/Downloads/dataset/datas/celeA'
style_image = 'res/style5.jpg'

# torch.distributed.init_process_group(backend='nccl')

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CeleADatas(Dataset):
    def __init__(self, image_folder):
        super(CeleADatas, self).__init__()
        self.image_files = []
        self.initialize(image_folder)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        try:
            image_data = Image.open(image_file)
        except:
            image_data = Image.open(self.image_files[10])
        image_tensor = image_transform(image_data)
        return image_tensor, image_file

    def __len__(self):
        return len(self.image_files)

    def initialize(self, image_folder):
        image_list = os.listdir(image_folder)
        for image in image_list:
            if not image.endswith('.jpg'):
                continue
            image_file = os.path.join(image_folder, image)
            self.image_files.append(image_file)


class VGG16(nn.Module):
    def __init__(self, required_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained = models.vgg16(pretrained=True).features
        vgg_pretrained = vgg_pretrained.eval()

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained[x])

        if not required_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        y = self.slice1(x)
        h1 = y
        y = self.slice2(y)
        h2 = y
        y = self.slice3(y)
        h3 = y
        y = self.slice4(y)
        h4 = y
        y = self.slice5(y)
        h5 = y
        vgg_features = namedtuple('vgg_features', ['h1', 'h2', 'h3', 'h4', 'h5'])
        features = vgg_features(h1, h2, h3, h4, h5)
        return features


def gram_matrix(x):
    a, b, c, d = x.size()
    features = x.view(a, b, c * d)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (b * c * d)
    return gram


def compute_content_loss(source, target):
    loss = F.mse_loss(source, target)
    return loss


def compute_style_loss(source, target):
    A = gram_matrix(source)
    B = gram_matrix(target)
    loss = F.mse_loss(A, B)
    return loss


def recover_tensor(source):
    mean = torch.tensor([0.485, 0.456, 0.406])
    mean = mean.view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225])
    std = std.view(1,3, 1, 1)
    dst = source * std + mean
    dst = dst.clamp(0, 1)
    return dst



def train():
    print('==> start training loop...')
    datas = datasets.ImageFolder(data_folder, transform=image_transform)
    print('==> all datas: ', len(datas))
    data_loader = DataLoader(datas, batch_size=256, shuffle=True, num_workers=4)
    batches = int(len(datas) / 256)

    style_weight = 1e5
    tv_weight = 1e-6
    content_weight = 1.0

    net = TransformNet()
    net = torch.nn.parallel.DistributedDataParallel(net)
    net.cuda()
    net.train()

    vgg = VGG16()
    style_data = Image.open(style_image)
    style_tensor = image_transform(style_data)
    style_tensor = style_tensor.unsqueeze(0)
    style_features = vgg(style_tensor)

    opt = optimizer.Adam(net.parameters(), lr=1e-3)
    epochs = 200
    for epoch in range(200):
        losses = 0.0
        for index, (content_image, _) in enumerate(data_loader):
            content_image = content_image.cuda()
            outputs = net(content_image)
            outputs = outputs.clamp(-3, 3)

            content_features = vgg(content_image)
            outputs_features = vgg(outputs)

            content_loss = compute_content_loss(outputs_features[1], content_features[1])
            content_loss = content_loss * content_weight

            style_loss = 0.0
            for i in range(5):
                value = compute_style_loss(outputs_features[i], style_features[i])
                style_loss += value
            style_loss = style_loss * style_weight

            tv_loss = torch.sum(torch.abs(outputs[:,:,:,:-1] - outputs[:,:,:,1:])) + \
                      torch.sum(torch.abs(outputs[:,:,:-1,:] - outputs[:,:,1:,:]))
            tv_loss = tv_loss * tv_weight

            loss = content_loss  + style_loss + tv_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses += loss.item()
            if index % 50 == 0:
                print('==> Epoch: [%d]/[%d]-[%d]/[%d], content_loss = %f, style_loss = %f, tv_loss = %f' % (
                    epoch, epochs, index, batches, content_loss.item(), style_loss.item(), tv_loss.item()
                ))

            if index % 300 == 0:
                outputs = recover_tensor(outputs)
                str_image = 'output/' + str(epoch) + '-' + str(index) + '.png'
                save_image(outputs, str_image, nrow=16)

        losses = losses / batches
        print('==> Epoch: [%d]/[%d], train loss = %f' % (epoch, epochs, losses.item()))



if __name__ == '__main__':
    # train()
    # datas = CeleADatas(os.path.join(data_folder, 'celeba'))
    # print('Num: ', len(datas))

    datas = datasets.CocoDetection('../datas/COCO')
    print(len(datas))

