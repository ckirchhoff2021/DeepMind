import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

data_root = '/Users/chenxiang/Downloads/Gitlab/Deepmind/datas'

image_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class TestDNN(nn.Module):
    def __init__(self, in_chn=3, n_cls=10):
        super(TestDNN, self).__init__()
        self.conv1 = nn.Conv2d(in_chn, 128, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_cls)
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        p1 = self.avg_pool(y2)
        p2 = p1.view(p1.size(0), -1)
        yt = self.fc(p2)
        return yt


class ResidualBlock(nn.Module):
    def __init__(self, in_chn, expansion):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chn, in_chn, 3, 1, 1),
            nn.BatchNorm2d(in_chn),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_chn, in_chn, 3, 1, 1),
            nn.BatchNorm2d(in_chn),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_chn, expansion * in_chn, 3, 1, 1),
            nn.BatchNorm2d(expansion * in_chn),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()
        if expansion != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chn, expansion * in_chn, 3, 1, 1),
                nn.BatchNorm2d(expansion * in_chn),
                nn.ReLU(inplace=True)
            )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(expansion * in_chn, expansion * in_chn // 4),
            nn.ReLU(inplace=True),
            nn.Linear(expansion * in_chn // 4, expansion * in_chn),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv(x) + self.shortcut(x)
        y = nn.ReLU(inplace=True)(y)

        y2 = self.pool(y)
        y2 = y2.view(y2.size(0), -1)
        w = self.channel_attention(y2)
        w = w.view(y2.size(0),-1, 1, 1)
        y = (1.0 + w) * y
        return y


class TestResnet(nn.Module):
    def __init__(self, in_chn, n_cls):
        super(TestResnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chn, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.residual_block1 = ResidualBlock(64, 1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.residual_block2 = ResidualBlock(64, 2)
        self.residual_block3 = ResidualBlock(128, 2)
        self.pool2 = nn.AdaptiveAvgPool2d(4)

        self.fc = nn.Sequential(
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_cls)
        )

    def forward(self, x):
        y1 = self.residual_block1(self.conv1(x))
        y1 = self.pool1(y1)
        y2 = self.residual_block2(y1)
        y2 = self.pool1(y2)
        y3 = self.residual_block3(y2)
        y3 = self.pool2(y3)
        y3 = y3.view(y3.size(0), -1)
        y = self.fc(y3)
        return y


def start_train():
    net = TestDNN(3, 10)
    batch_size = 128

    train_datas = datasets.CIFAR10(data_root, train=True, transform=image_transform)
    train_loader = DataLoader(train_datas, batch_size=batch_size, shuffle=True)
    test_datas = datasets.CIFAR10(data_root, train=False, transform=image_transform)
    test_loader = DataLoader(test_datas, batch_size=batch_size, shuffle=True)

    print('==> start training loop ...')
    print('==> train datas: ', len(train_datas))
    print('==> test datas: ', len(test_datas))

    n_batch = int(len(train_datas) / batch_size) + 1
    epochs = 20
    best_acc = 0

    scalar_summary = SummaryWriter(comment='metric')
    criterion = nn.CrossEntropyLoss()
    opt = optimizer.Adam(net.parameters(), lr=0.005)
    for epoch in range(epochs):
        net.train()
        losses = 0.0
        counts = 0
        corrects = 0
        for index, (data, label) in enumerate(train_loader):
            y = net(data)
            loss = criterion(y, label)
            losses += loss.item()
            preds = y.max(1)[1]
            correct = preds.eq(label).sum().item()
            corrects += correct
            counts += len(data)

            scalar_summary.add_scalar('loss', loss, index + epoch * len(train_loader))
            scalar_summary.add_scalar('accuracy', correct / len(data), index + epoch * len(train_loader))

            if index % 100 == 0:
                print('==> training: epoch [%d]/[%d]-[%d]/[%d], loss = %f, acc = %f' %
                      (epoch, epochs, index, n_batch, loss.item(), correct/len(data)))

            opt.zero_grad()
            loss.backward()
            opt.step()

        train_acc = corrects / counts
        train_loss = losses / n_batch
        scalar_summary.add_scalar('average_loss', train_loss, epoch)
        scalar_summary.add_scalars('average_accuracy', {"train": train_acc}, epoch)
        print('==> training: epoch [%d]/[%d], loss = %f, acc = %f' % (epoch, 2, train_loss, train_acc))

        net.eval()
        counts = 0
        corrects = 0
        for index, (data, label) in enumerate(test_loader):
            y = net(data)
            preds = y.max(1)[1]
            correct = preds.eq(label).sum().item()
            corrects += correct
            counts += len(data)
        test_acc = corrects / counts
        scalar_summary.add_scalars('average_accuracy', {"test": test_acc}, epoch)
        print('==> testing: epoch [%d]/[%d],  acc = %f' % (epoch, 2, test_acc))

        if best_acc < test_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'net': net.state_dict(),
                'acc': test_acc
            }

            torch.save(state, '../output/cls.pth')


def update(a):
    a['x'] += 1


if __name__ == '__main__':
    # net = TestDNN(3, 10)
    # x = torch.randn(1,3, 64, 64)
    # y = net(x)
    # print(y.size())
    # start_train()

    # net = TestResnet(3, 10)
    # x = torch.randn(1, 3, 32, 32)
    # y = net(x)
    # print(y.size())
    # summary(net, (3, 32, 32))

    net = models.resnet50(pretrained=True)
    layers = list(net.children())
    feature = nn.Sequential(*layers[:-1])
    x = torch.randn(1,3,224,224)
    y = feature(x)
    print(y.size())

    pt = 1.0 /3
    print('{:.3f}'.format(pt))
    print("%.3f"% pt)

    a = {'x': 12, 'y': 13}
    update(a)
    update(a)
    update(a)
    print(a)