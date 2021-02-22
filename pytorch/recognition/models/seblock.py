import torch
import torch.nn as nn
import torch.nn.functional as F


class SEUnit(nn.Module):
    def __init__(self, inchn, coef=4):
        super(SEUnit, self).__init__()
        self.ratio = coef
        mid = inchn // coef
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.feature = nn.Sequential(
            nn.Linear(inchn, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, inchn),
            nn.Sigmoid()
        )

    def forward(self, x):
        yp = self.gap(x)
        yp = yp.view(yp.size(0), -1)
        yf = self.feature(yp)
        yw = yf.view(yf.size(0), yf.size(1), 1, 1)
        return yw
