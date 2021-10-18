import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.tensorboard import SummaryWriter

from config import *
from thnews import *
from transformers import BertModel, AdamW


class CLSModel(nn.Module):
    def __init__(self):
        super(CLSModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_root)
        for param in self.bert.parameters():
            param.requres_grad = False

        self.fc = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 14)
        )

    def forward(self, x, mask):
        outputs = self.bert(x, attention_mask=mask)
        feature = outputs[1]
        y = self.fc(feature)
        return y


def build(net, train_datas, test_datas, epochs, learning_rate, save_file, summary_folder):
    print('==> start building loop ...')
    print('==> train datas: ', len(train_datas))
    print('==> test datas: ', len(test_datas))

    best_acc = 0.0
    batch_size = 256
    train_loader = DataLoader(train_datas, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_datas, batch_size=(batch_size // 2), shuffle=True, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    opt = optimizer.Adam(net.parameters(), lr=learning_rate)
    summary = SummaryWriter(log_dir=summary_folder, comment='metric')
    batches = len(train_loader)

    for epoch in range(epochs):
        net.train()
        metric = {'corrects': 0, 'counts': 0, 'losses': 0.0}
        for index, (data, mask, label) in enumerate(train_loader):
            inputs, masks, targets = data.cuda(), mask.cuda(), label.cuda()
            outputs = net(inputs, masks)
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

        net.eval()
        metric = {'corrects': 0, 'counts': 0, 'losses': 0.0}
        for index, (data, mask, label) in enumerate(test_loader):
            inputs, masks, targets = data.cuda(), mask.cuda(), label.cuda()
            outputs = net(inputs, masks)
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

        epoch_loss = metric['losses'] / len(test_loader)
        epoch_acc = metric['corrects'] / metric['counts']
        summary.add_scalar('test/epoch_acc', epoch_acc, epoch)
        summary.add_scalar('test/epoch_loss', epoch_loss, epoch)
        print('==> Epcoh: [%d]/[%d], testing: epoch loss = %f, epoch_acc = %f' %
              (epoch, epochs, epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            state = {
                'net': net.state_dict(),
                'epoch': epochs,
                'acc': best_acc
            }
            torch.save(state, save_file)


def main():
    net = CLSModel()
    x = torch.LongTensor([[0,0,11,145,23]])
    mask = torch.tensor([[1,1,1,1,1]])
    y = net(x, mask)
    print(y.size())

    train_datas = THUDatas(train=True)
    test_datas = THUDatas(train=False)
    epochs = 30
    learning_rate = 0.005
    save_file = 'output/text.pth'
    summary_folder = 'summary'

    net = nn.DataParallel(net)
    net = net.cuda()
    build(net, train_datas, test_datas, 30, learning_rate, save_file, summary_folder)


if __name__ == '__main__':
    main()
