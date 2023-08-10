import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
import torch.optim as Optimizer
from dataset import ThusNewsData
from attention import AttentionModel, GroupAttentionModel


def train_loop():
    train_file = '/home/cx/A100/datas/words/train.json'
    val_file = '/home/cx/A100/datas/words/val.json'

    train_data = ThusNewsData(train_file)
    print('==> train data Num: ', len(train_data))
    val_data = ThusNewsData(val_file)
    print('==> val data Num: ',  len(val_data))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=128)
    print('==> train loader num: ', len(train_loader))
    val_loader = DataLoader(val_data, shuffle=True, batch_size=64)
    print('==> val loader num: ', len(val_loader))

    # model = AttentionModel()
    model = GroupAttentionModel()
    model = model.cuda()
    # model = nn.DataParallel(model, device_ids=[0,1,2,3])
    opt = Optimizer.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    summary = SummaryWriter('summary/groupAttention', comment='metric')

    best_acc = 0.0
    batches = len(train_loader)
    epochs = 30

    for epoch in range(epochs):
        model.train()
        metric = {'corrects': 0, 'counts': 0, 'losses': 0.0}
        for index, (data, label) in enumerate(train_loader):
            inputs, targets = data.cuda(), label.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            predicts = outputs.max(1)[1]
            corrects = predicts.eq(targets).sum().item()
            acc = corrects / data.size(0)
            steps = index + epoch * batches

            summary.add_scalar('train/batch_acc', acc, steps)
            summary.add_scalar('train/batch_loss', loss.item(), steps)

            metric['counts'] += data.size(0)
            metric['corrects'] += corrects
            metric['losses'] += loss.item()

            if index % 20 == 0:
                print('==> Epoch:[%d]/[%d]-[%d]/[%d], training: batch_loss = %f, batch_acc = %f' %
                      (epoch, epochs, index, batches, loss.item(), acc))

            opt.zero_grad()
            loss.backward()
            opt.step()

        epoch_loss = metric['losses'] / len(train_loader)
        epoch_acc = metric['corrects'] / metric['counts']
        summary.add_scalar('train/epoch_acc', epoch_acc, epoch)
        summary.add_scalar('train/epoch_loss', epoch_loss, epoch)

        print('==> Epcoh: [%d]/[%d], training:  epoch_loss = %f, epoch_acc = %f' %
              (epoch, epochs, epoch_loss, epoch_acc))

        model.eval()
        metric = {'corrects': 0, 'counts': 0, 'losses': 0.0}
        for index, (data, label) in enumerate(val_loader):
            inputs, targets = data.cuda(), label.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            predicts = outputs.max(1)[1]
            corrects = predicts.eq(targets).sum().item()
            acc = corrects / data.size(0)
            steps = index + epoch * batches

            summary.add_scalar('test/batch_acc', acc, steps)
            summary.add_scalar('test/batch_loss', loss.item(), steps)

            metric['counts'] += data.size(0)
            metric['corrects'] += corrects
            metric['losses'] += loss.item()

        epoch_loss = metric['losses'] / len(val_loader)
        epoch_acc = metric['corrects'] / metric['counts']
        summary.add_scalar('test/epoch_acc', epoch_acc, epoch)
        summary.add_scalar('test/epoch_loss', epoch_loss, epoch)
        print('==> Epcoh: [%d]/[%d], testing: epoch loss = %f, epoch_acc = %f' %
              (epoch, epochs, epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            state = {
                'net': model.state_dict(),
                'epoch': epochs,
                'acc': best_acc
            }
            torch.save(state, 'output/group_attention.pt')


if __name__ == '__main__':
    train_loop()
