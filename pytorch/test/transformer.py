import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, datasets

from cls_train import start_train


data_root = '/Users/chenxiang/Downloads/Gitlab/Deepmind/datas'

image_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class cifar10Patches(Dataset):
    def __init__(self, train=True, transform=None):
        super(cifar10Patches, self).__init__()
        self.datas = datasets.CIFAR10(data_root, train=train, transform=transform)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        x, y = self.datas[index]
        patches = list()
        for i in range(4):
            for j in range(4):
                p = x[:,8*i:8*(i+1), 8*j:8*(j+1)]
                p = p.unsqueeze(0)
                patches.append(p)
        patches = torch.cat(patches, dim=0)
        return patches, y


class AttentionBlock(nn.Module):
    def __init__(self, num_seq, input_dims, output_dims):
        super(AttentionBlock, self).__init__()
        self.feature_dim = input_dims
        self.Q = nn.Parameter(torch.randn(num_seq, input_dims, output_dims))
        self.K = nn.Parameter(torch.randn(num_seq, input_dims, output_dims))
        self.V = nn.Parameter(torch.randn(num_seq, input_dims, output_dims))

    def forward(self, x):
        fea_dim = self.feature_dim
        nseqs = x.size(1) // fea_dim
        scores = list()
        vectors = list()
        for i in range(nseqs):
            vec = x[:,fea_dim * i: fea_dim * (i+1)]
            wQ = self.Q[i]
            wK = self.K[i]
            wV = self.V[i]
            q = torch.mm(vec, wQ)
            k = torch.mm(vec, wK)
            v = torch.mm(vec, wV)
            score = torch.mul(q, k)
            score = score.sum(1, keepdim=True)
            scores.append(score)
            vectors.append(v)
        scores = torch.cat(scores, dim=1)
        weights = scores / scores.norm(dim=1, keepdim=True)
        y = 0.0
        for k in range(nseqs):
            w = weights[:, k]
            w = w.unsqueeze(1)
            v = vectors[k]
            y += w * v
        return y


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, nheads, num_seq, input_dims, output_dims):
        super(MultiHeadAttentionBlock, self).__init__()
        self.nheads = nheads
        self.attetion_blocks = list()
        for i in range(nheads):
            block = AttentionBlock(num_seq, input_dims, output_dims)
            self.attetion_blocks.append(block)

    def forward(self, x):
        outputs = list()
        for block in self.attetion_blocks:
            y = block(x)
            outputs.append(y)
        y = torch.cat(outputs, dim=1)
        return y



class VisTransformer(nn.Module):
    def __init__(self):
        super(VisTransformer, self).__init__()
        self.attentionblock_1 = MultiHeadAttentionBlock(3, 16, 64, 32)
        self.shortcut = nn.Sequential(
            nn.Linear(1024, 96),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Conv2d(128, 1, 3, stride=1, padding=1)

    def cnn_feature(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1) + f1
        f3 = self.conv3(f2)
        f4 = self.conv4(f3) + f3
        f5 = self.conv5(f4)
        y = f5.view(f5.size(0), -1)
        return y

    def forward(self, x):
        count = x.size(1)
        sequences = list()
        for i in range(count):
            patch = x[:,i,:,:,:]
            feature = self.cnn_feature(patch)
            sequences.append(feature)

        seqs = torch.cat(sequences, dim=1)
        outputs = self.attentionblock_1(seqs) + self.shortcut(seqs)
        outputs = self.fc(outputs)
        return outputs


def main():
    net = VisTransformer()
    x = torch.randn(8, 16, 3, 8, 8)
    y = net(x)
    print(y.size())

    cifar10 = cifar10Patches(transform=image_transform)
    x, y = cifar10[10]
    print(x.size())


def train_loop():
    net = VisTransformer()
    # net = nn.DataParallel(net)
    # net = net.cuda()

    train_datas = cifar10Patches(train=True, transform=image_transform)
    test_datas = cifar10Patches(train=False, transform=image_transform)
    start_train(net, train_datas, test_datas, 30, 0.001, 'summary', 'model/transformer.pth')



if __name__ == '__main__':
    # main()
    train_loop()
