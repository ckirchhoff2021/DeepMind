import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def start_train(net, train_datas, test_datas, epochs, learning_rate, sumarry_dir, save_file):
    print('==> start building loop ...')
    print('==> train datas: ', len(train_datas))
    print('==> test datas: ', len(test_datas))

    best_acc = 0.0
    batch_size = 128
    train_loader = DataLoader(train_datas, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_datas, batch_size=(batch_size//2), shuffle=True, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    opt = optimizer.Adam(net.parameters(), lr=learning_rate)
    scalar_summary = SummaryWriter(log_dir=sumarry_dir, comment='metric')
    batches = len(train_loader)

    for epoch in range(epochs):
        net.train()
        ncorrect = 0
        ncount = 0
        losses = 0
        for index, (data, label) in enumerate(train_loader):
            inputs, targets = data, label
            outputs = net(inputs)
            predicts = outputs.max(1)[1]
            corrects = predicts.eq(targets).sum().item()
            acc = corrects / data.size(0)
            scalar_summary.add_scalar('train_batch_acc', acc, index + epoch * batches)
            loss = criterion(outputs, targets)
            scalar_summary.add_scalar('train_batch_loss', loss.item(), index + epoch * batches)

            ncount += data.size(0)
            ncorrect += corrects
            losses += loss.item()
            if index % 2 == 0:
                print('==> Epoch:[%d]/[%d]-[%d]/[%d], train loss = %f, train accuracy = %f' % (epoch, epochs, index, batches, loss.item(), acc))

            opt.zero_grad()
            loss.backward()
            opt.step()

        average_loss = losses / len(train_loader)
        average_acc = ncorrect / ncount
        scalar_summary.add_scalar('train_epoch_acc', average_acc, epoch)
        scalar_summary.add_scalar('train_epoch_loss', average_loss, epoch)
        print('==> Epcoh: [%d]/[%d], training: average loss = %f, average accuracy = %f' % (epoch, epochs, average_loss, average_acc))

        net.eval()
        ncorrect = 0
        ncount = 0
        losses = 0
        for index, (data, label) in enumerate(test_loader):
            inputs, targets = data, label
            outputs = net(inputs)
            predicts = outputs.max(1)[1]
            corrects = predicts.eq(targets).sum().item()
            acc = corrects / data.size(0)
            scalar_summary.add_scalar('test_batch_acc', acc, index + epoch * batches)
            loss = criterion(outputs, targets)
            scalar_summary.add_scalar('test_batch_loss', loss.item(), index + epoch * batches)

            ncorrect += corrects
            ncount += data.size(0)
            losses += loss.item()

        average_acc = ncorrect / ncount
        average_loss = losses / len(test_loader)
        scalar_summary.add_scalar('test_epoch_acc', average_acc, epoch)
        scalar_summary.add_scalar('test_epoch_loss', average_loss, epoch)
        print('==> Epcoh: [%d]/[%d], testing: average loss = %f, average accuracy = %f' % (epoch, epochs, average_loss, average_acc))

        if average_acc > best_acc:
            best_acc = average_acc
            state = {
                'net': net.state_dict(),
                'epoch': epochs,
                'acc': best_acc
            }

            torch.save(state, save_file)

