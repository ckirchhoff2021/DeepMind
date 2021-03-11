import torch
import torch.nn as nn
from seblock import SEUnit


class TestNet(nn.Module):
    def __init__(self, in_chn=3, nclass=10):
        super(TestNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_chn, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
        )

        self.classify = nn.Linear(128, nclass)

    def get_feature_map(self, x):
        y_conv = self.feature(x)
        y_pool = self.avgpool(y_conv)
        y_pool = y_pool.view(y_pool.size(0), -1)
        y_feature = self.fc(y_pool)
        return y_feature

    def forward(self, x):
        feature = self.get_feature_map(x)
        y = self.classify(feature)
        return y


class VGGLike(nn.Module):
    def __init__(self, nclass=10, nchn=3):
        super(VGGLike, self).__init__()

        self.conv1 = nn.Conv2d(nchn, 64, 3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 128, 3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool2 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 256, 3, padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(256, 512, 3, padding=1, stride=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(512, 512, 3, padding=1, stride=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(8192, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 128),
            nn.Linear(128, nclass)
        )

    def forward(self, x):
        y11 = self.relu1(self.bn1(self.conv1(x)))
        y12 = self.relu2(self.bn2(self.conv2(y11)))
        p11 = self.pool1(y12)

        y21 = self.relu3(self.bn3(self.conv3(p11)))
        y22 = self.relu4(self.bn4(self.conv4(y21)))
        p21 = self.pool2(y22)

        y31 = self.relu5(self.bn5(self.conv5(p21)))
        y32 = self.relu6(self.bn6(self.conv6(y31)))
        y33 = self.relu7(self.bn7(self.conv7(y32)))

        p33 = self.avgpool(y33)

        x_fc = p33.view(p33.size(0), -1)
        y_out = self.fc(x_fc)
        return y_out


class SEVGGLike(nn.Module):
    def __init__(self, nclass=10, nchn=3):
        super(SEVGGLike, self).__init__()

        self.conv1 = nn.Conv2d(nchn, 64, 3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.se1 = SEUnit(128, coef=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 128, 3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        self.se2 = SEUnit(256, coef=2)
        self.pool2 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 256, 3, padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(256, 512, 3, padding=1, stride=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(512, 512, 3, padding=1, stride=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU(inplace=True)

        self.se3 = SEUnit(512, coef=2)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(8192, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 128),
            nn.Linear(128, nclass)
        )

    def forward(self, x):
        y11 = self.relu1(self.bn1(self.conv1(x)))
        y12 = self.relu2(self.bn2(self.conv2(y11)))
        y13 = self.se1(y12)
        y14 = y12 * y13
        p11 = self.pool1(y14)

        y21 = self.relu3(self.bn3(self.conv3(p11)))
        y22 = self.relu4(self.bn4(self.conv4(y21)))
        y23 = self.se2(y22)
        y24 = y22 * y23
        p21 = self.pool2(y24)

        y31 = self.relu5(self.bn5(self.conv5(p21)))
        y32 = self.relu6(self.bn6(self.conv6(y31)))
        y33 = self.relu7(self.bn7(self.conv7(y32)))
        y34 = self.se3(y33)
        y35 = y33 * y34
        p33 = self.avgpool(y35)

        x_fc = p33.view(p33.size(0), -1)
        y_out = self.fc(x_fc)
        return y_out


def main():
    net = SEVGGLike()
    print(net)

    x = torch.randn(1, 3, 224, 224)
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    main()