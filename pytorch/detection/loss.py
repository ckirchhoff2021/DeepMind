import torch
import torch.nn as nn


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.w_coord = 5
        self.w_noobj = 0.5
    
    @staticmethod
    def calculate_IOU(b1, b2):
        dw = (b1[0] - b2[0]).abs()
        dh = (b1[1] - b2[1]).abs()
        
        sw = (b1[2] + b2[2]) * 0.5
        sh = (b1[3] + b2[3]) * 0.5
        
        iw_mask = (sw - dw) > 0
        iw = iw_mask.float() * (sw - dw)
        
        ih_mask = (sh - dh) > 0
        ih = ih_mask.float() * (sh - dh)
        
        I_area = iw * ih
        U_area = b1[2] * b1[3] + b2[2] * b2[3] - I_area
        IOU = I_area.div(U_area)
        return IOU
    
    def forward(self, outputs, targets):
        obj_exists = targets[:,:,:,20]
        nums = outputs.size(0)
        
        prob_loss = 0.0
        position_loss = 0.0
        size_loss = 0.0
        confidence_loss1 = 0.0
        confidence_loss2 = 0.0
        
        for inum in range(nums):
            for irow in range(7):
                for icol in range(7):
                    
                    if obj_exists[inum, irow, icol] == 0:
                        continue
                    
                    probs = (targets[inum, irow, icol, :20] - outputs[inum, irow, icol, :20]) ** 2
                    prob_loss += probs.sum()
                        
                    gt_box = targets[inum, irow, icol, 22:26]
                    pred_box1 = outputs[inum, irow, icol, 22:26]
                    iou1 = self.calculate_IOU(gt_box, pred_box1)

                    pred_box2 = outputs[inum, irow, icol, 26:]
                    iou2 = self.calculate_IOU(gt_box, pred_box2)
                    
                    if iou1 > iou2:
                        chosen_box = pred_box1
                        pred_c1 = outputs[inum, irow, icol, 20]
                        pred_c2 = outputs[inum, irow, icol, 21]
                    else:
                        chosen_box = pred_box2
                        pred_c1 = outputs[inum, irow, icol, 21]
                        pred_c2 = outputs[inum, irow, icol, 20]
                    
                    x1, y1, w1, h1 = gt_box
                    x2, y2, w2, h2 = chosen_box
                    
                    position_loss += ((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    size_loss += ((torch.sqrt(w1) - torch.sqrt(w2)) ** 2 + (torch.sqrt(h1) - torch.sqrt(h2)) ** 2)
                    confidence_loss1 += (pred_c1 - 1.0) ** 2
                    confidence_loss2 += (pred_c2 - 1.0) ** 2
        
        losses = self.w_coord * (position_loss + size_loss) + confidence_loss1 + self.w_noobj * confidence_loss2 + prob_loss
        return losses
                    
                    
                    
        