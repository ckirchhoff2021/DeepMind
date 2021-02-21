import torch
import torch.optim as opt
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from models.resnet import *
from models.vgg import *
from models.mobilenet import *

from utils import *

import argparse

parser = argparse.ArgumentParser(description='Pytorch recognition training...')
parser.add_argument('--net', type=str, default='resnet50', help='backbone')
parser.add_argument('--nclass', type=int, default=10, help='类别数')
parser.add_argument('--dims', type=int, default=128, help='feature维度')
parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
parser.add_argument('--reweighting', type=bool, default=False, help='reweighting方法')
parser.add_argument('--epochs', type=int, default=30, help='epochs Num')
parser.add_argument('--train', type=str, default='', help='train datas')
parser.add_argument('--test', type=str, default='', help='test datas')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--outfile', type=str, default='output/model.pth', help='模型保存文件名')
args = parser.parse_args()


def start_build(train_dataset, test_dataset, net, batch_size, epochs, model_file):
    print('==> start building models...')
    train_num = len(train_dataset)
    batches = int(train_num) / batch_size

    print('-- train num: ', len(train_dataset))
    print('-- test num: ', len(test_dataset))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=64, num_workers=4)
    criterion = nn.CrossEntropyLoss()

    if cuda:
        net = nn.DataParallel(net)
        net.cuda()
        criterion.cuda()

    # optimizer = opt.Adam(net.parameters(), lr=0.001)
    optimizer = opt.SGD(net.parameters(), lr=0.003, momentum=0.8)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    best_acc = 0.0
    for epoch in range(epochs):
        net.train()

        losses = 0.0
        correct = total = 0
        for index, (data, label) in enumerate(train_loader):
            if cuda:
                inputs, targets = data.cuda(), label.cuda()
            else:
                inputs, targets = data, label

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            predicts = outputs.max(1)[1]

            success = predicts.eq(targets.view_as(predicts)).sum().item()
            accuracy = success / data.size(0)

            losses += loss.item()
            correct += success
            total += data.size(0)

            if index % 10 == 0:
                print('-- Epoch: [%d]/[%d]-[%d]/[%d], training: loss = %f, acc = %f' %
                      (epoch, epochs, index, batches, loss.item(), accuracy))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        average_loss = losses / len(train_loader)
        average_acc = correct / total

        print('-- training: average loss: {}, average acc: {}'.format(average_loss, average_acc))

        net.eval()
        with torch.no_grad():
            correct = total = 0
            for index, (data, label) in enumerate(test_loader):
                if cuda:
                    inputs, targets = data.cuda(), label.cuda()
                else:
                    inputs, targets = data, label
                outputs = net(inputs)
                predicts = outputs.max(1)[1]

                success = predicts.eq(targets.view_as(predicts)).sum().item()
                correct += success
                total += data.size(0)

            average_acc = correct / total
            print('-- testing: average acc: {}'.format(average_acc))

            if average_acc > best_acc:
                best_acc = average_acc
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'acc': average_acc,
                    'epoch': epoch
                }

                torch.save(state, model_file)

    print('All complete...')


def main():
    train_dataset = DatasetInstance(args.train, train=True)
    test_dataset = DatasetInstance(args.test, train=False)
    if args.net == 'resnet50':
        print('Backbone: ', args.net)
        net = resnet50(out_num=args.nclass, mid_dim=args.dims, pretrained=True)
    elif args.net == 'resnet152':
        print('Backbone: ', args.net)
        net = resnet152(out_num=args.nclass, mid_dim=args.dims, pretrained=True)
    else:
        print('Backbone: ', args.net)
        net = mobileV2(out_num=args.nclass, mid_dim=args.dims, pretrained=True)

    start_build(train_dataset, test_dataset, net, args.batch_size, args.epochs, args.outfile)


if __name__ == '__main__':
    main()


#  python train.py --train=../../output/sofa_train.json --test=../../output/sofa_test.json --outfile=../../output/sofa.pth --nclass=2 --epochs=2 --net=mobilenetV2
