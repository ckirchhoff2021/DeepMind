import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CommonDataset:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomRotation(degrees=6),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    def __init__(self, folder):
        self.root = folder
        print('==> Experiment data initialize...')

    def cifar10(self, transform=True):
        '''
        cifar10
        :param transform: bool
        :return:
        '''
        t1 = None
        t2 = None

        if transform:
            t1 = self.train_transform
            t2 = self.test_transform

        train_datas = datasets.CIFAR10(root=self.root, train=True, transform=t1)
        test_datas = datasets.CIFAR10(root=self.root, train=False, transform=t2)

        return train_datas, test_datas

    def mnist(self, transform=True, download=False):
        '''
        mnist
        :param transform: bool
        :return:
        '''
        image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
            ]
        )
        t1 = None
        t2 = None

        if transform:
            t1 = image_transform
            t2 = image_transform

        train_datas = datasets.MNIST(root=self.root, train=True, transform=t1, download=download)
        test_datas = datasets.MNIST(root=self.root, train=False, transform=t2, download=download)

        return train_datas, test_datas

    def cifar100(self, transform=True):
        '''
        :param transform: bool
        :return:
        '''
        t1 = None
        t2 = None

        if transform:
            t1 = self.train_transform
            t2 = self.test_transform

        train_datas = datasets.CIFAR100(root=self.root, train=True, transform=t1)
        test_datas = datasets.CIFAR100(root=self.root, train=False, transform=t2)

        return train_datas, test_datas


def test():
    datas = datasets.CIFAR100(root=data_path, train=True)
    print(datas.class_to_idx)
    targets = datas.targets
    cnt = [0] * 100
    for value in targets:
        cnt[value] += 1
    print(cnt)

    image, label = datas[12]
    print(image.size)
    image.show()
    print(label)


def test2():
    folder = '../../../datas'
    dataset = CommonDataset(folder)

    train_datas, test_datas = dataset.mnist(transform=False, download=True)
    a1, b1 = train_datas[22]
    a2 = a1.resize((64, 64))
    a2.show()


if __name__ == '__main__':
    test2()