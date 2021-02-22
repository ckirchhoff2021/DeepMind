import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
import torch.nn.functional as F

from detection.voc import *
from detection.loss import YoloLoss
from collections import OrderedDict
from pycocotools.coco import COCO

from torchvision import models, transforms, datasets

model_path = 'output/yolo.pth'

def residual_net():
    net = models.resnet18(pretrained=True, progress=True)
    fc_in_num = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(fc_in_num, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 1470)
    )
    return net

class YoloNet(nn.Module):
    def __init__(self):
        super(YoloNet, self).__init__()
        self.net = residual_net()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = self.net(x)
        y = y.view(y.size(0), 7, 7, 30)
        y1 = F.softmax(y[:,:,:,0:20], dim=1)
        y2 = self.sigmoid(y[:,:,:,20:])
        y_out = torch.cat((y1,y2), dim=3)
        return y_out


def start_train():
    print('==> start training loop......')
    net = residual_net()
    net = nn.DataParallel(net)
    net.cuda()
    
    optimizer = opt.Adam(net.parameters(), lr = 0.001)
    criterion = YoloLoss()
    criterion.cuda()
    
    train_dataset = VocDataset(train=True)
    batches = int(len(train_dataset) / 64)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64, num_workers=4)
    
    test_dataset = VocDataset(test=True)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=32)
    
    print('--All data : train - %d, test - %d' % (len(train_dataset), len(test_dataset)))

    for epoch in range(30):
        net.train()
        losses = 0.0
        for index, (datas, labels) in enumerate(train_dataloader):
            inputs, targets = datas.cuda(), labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            losses += loss.item()
            
            if index % 10 == 0:
                print('Epoch : [%d]-[%d][%d], training loss = %f' % (epoch, index, batches, loss.item()))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        average_loss = losses / len(train_dataloader)
        print('--Average loss: ', average_loss)
        torch.save(net.state_dict(), model_path)
        
      
def single_predict(image):
    image_data = image.resize((448,448))
    image_tensor = train_transforms(image_data)
    x = image_tensor.unsqueeze(0)

    net = load_yolo_net()
    net.eval()

    score_threshold = 0.7
    iou_threshold = 0.3
    
    with torch.no_grad():
        y = net(x)
        y_pred = y.squeeze()
       
        ret = list()
        for i in range(20):
            score_list = list()
            for row in range(7):
                for col in range(7):
                    prob = y_pred[row, col, i].item()
                    c1 = y_pred[row, col, 20].item()
                    box1 = y_pred[row, col, 22:26]
                    score1 = prob * c1
                    
                    if score1 > score_threshold:
                        if len(score_list) == 0:
                            score_list.append([score1, i, box1])
                        else:
                            for item in score_list:
                                item_box = item[2]
                                iou_value = YoloLoss.calculate_IOU(box1, item_box)
                                if iou_value < iou_threshold:
                                    score_list.append([score1, i, box1])
                    
                    c2 = y_pred[row, col, 21].item()
                    box2 = y_pred[row, col, 26:]
                    score2 = prob * c2
                    
                    if score2 > score_threshold:
                        if len(score_list) == 0:
                            score_list.append([score2, i, box2])
                        else:
                            for item in score_list:
                                item_box = item[2]
                                iou_value = YoloLoss.calculate_IOU(box2, item_box)
                                if iou_value < iou_threshold:
                                    score_list.append([score2, i, box2])
                                
                ret.extend(score_list)
            
        return ret


def load_yolo_net():
    net = YoloNet()
    state = torch.load('bin/yolo.pth', map_location='cpu')
    state_dict = OrderedDict()
    for k, v in state.items():
        name = k[7:]
        state_dict[name] = v
    net.load_state_dict(state_dict)
    return net

def predict():
    # test_dataset = VocDataset(train=False)
    voc = datasets.VOCDetection('../datas', download=False)
    data = voc[-1]
    data[0].show()
    scores = single_predict(data[0])
    print(scores)
    
    
    
def main():
    net = residual_net()
    x = torch.randn(3,3,500,327)
    y = net(x)
    y = y.view(-1,7,7,30)
    print(y.size())



if __name__ == '__main__':
    # main()
    # predict()
    coco = datasets.CocoDetection(root="/Users/chenxiang/Downloads/dataset/cocos/train2014",
                                   annFile="/Users/chenxiang/Downloads/dataset/cocos/annotations/instances_train2014.json")
    coco = COCO(annotation_file="/Users/chenxiang/Downloads/dataset/cocos/annotations/instances_train2014.json")
    print(len(coco))

    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cate in cates]
    print('COCO categories: \n {} \n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n'.format(' '.join(nms)))