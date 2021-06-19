import os

import torch
import torch.nn as nn
import torch.optim as optimizer

import misc_utils as utils
import cv2

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 2)
            # nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return y


def main():
    x = torch.randn(100, 2)
    y =  ((x[:, 0] + x[:,1]) > 0).long()
    net = LinearModel()
    criterion = nn.CrossEntropyLoss()
    opt = optimizer.Adam(net.parameters(), lr=0.005)
    for epoch in range(40):
        y1 = net(x)
        loss = criterion(y1, y)
        y2 = y1.max(1)[1]
        acc = y2.eq(y).sum().item() / y.size(0)
        print('==> epoch: [%d]/[%d], loss = %f, acc = %f' % (epoch, 10, loss.item(), acc))

        opt.zero_grad()
        loss.backward()
        opt.step()


def test():
    x = torch.randn(1, 12, 3, 128, 128)
    f = nn.Conv3d(12, 10, (3, 4, 4), stride=2, padding=1)
    y = f(x)
    print(y.size())



def video_test():
    root = '/Users/chenxiang/Downloads/KTH/boxing'
    video_file = os.path.join(root, 'person01_boxing_d2_uncomp.avi')
    capture = cv2.VideoCapture(video_file)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(capture.get(cv2.CAP_PROP_FPS))

    print(frame_count)
    print(frame_width)
    print(frame_height)
    print(frame_fps)

    retaining = True
    count = 0
    while count < frame_count and retaining:
        retaining, frame = capture.read()
        if frame is None:
            continue

        cv2.imwrite(os.path.join('../../output/videos', '0000{}.jpg'.format(str(count))), frame)
        count += 1

    capture.release()


if __name__ == '__main__':
    # main()
    # utils.color_print('test', 2)
    # test()
    video_test()