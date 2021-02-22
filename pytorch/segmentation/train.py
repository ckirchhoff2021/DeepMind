import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import DataLoader

from fcn import FCN8s, FCN16s, FCN32s
from dataset import DataInstance


def start_train():
    print('start training loop ...')
    train_datas = DataInstance('data/train')
    test_datas = DataInstance('data/test')
    
    print('--- train num: ', len(train_datas))
    print('--- test num: ', len(test_datas))
    
    train_loader = DataLoader(train_datas, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_datas, batch_size=16, shuffle=True, num_workers=4)
    
    net = FCN8s(2)
    criterion = nn.BCEWithLogitsLoss()
    opt = optimizer.SGD(net.parameters(), lr=0.001)
    
    epochs = 10
    batches = int(len(train_loader) / 32)
    best_acc = 0.0
    
    for epoch in range(epochs):
        net.train()
        losses = 0.0
        mAps = 0.0
        for index, (data, label) in enumerate(train_loader):
            inputs, targets = data, label
            outputs = net(inputs)
            loss = criterion(outputs, x)
            losses += loss.item()
            
            pred = outputs.max(1)[1]
            success = pred.eq(targets).sum().item()
            pixels = data.size(2) * data.size(3)
            mAp = success / (pixels * data.size(0))
            mAps += mAp
            
            if index % 5 == 0:
                print('Epoch: [%d]/[%d]-[%d]/[%d], training loss = %f, mAp = %f' % (epoch, epochs, index, batches, loss.item(), mAp))
                
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        average_loss = losses / len(train_loader)
        average_mAp = mAps / len(train_loader)
        print('-- average loss = %f, average mAp = %f' % (average_loss, average_mAp))
        
        with torch.no_grad():
            net.eval()
            losses = 0.0
            mAps = 0.0
            for index, (data, label) in enumerate(test_loader):
                inputs, targets = data, label
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                losses += loss.item()

                pred = outputs.max(1)[1]
                success = pred.eq(targets).sum().item()
                pixels = data.size(2) * data.size(3)
                mAp = success / (pixels * data.size(0))
                mAps += mAp
                
            average_loss = losses / len(test_loader)
            average_mAp = mAps / len(test_loader)
            print('-- average loss = %f, average mAp = %f' % (average_loss, average_mAp))
            
            if average_mAp > best_acc:
                best_acc = average_mAp
                state = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'mAp': best_acc
                }

                torch.save(state, 'output/fcn8s.pth')

def main():
    start_train()


if __name__ == '__main__':
    main()