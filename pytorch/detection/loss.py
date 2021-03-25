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
        gt_b1 = targets[:, :, :, 20]
        gt_x1 = targets[:, :, :, 22]
        gt_y1 = targets[:, :, :, 23]
        gt_w1 = targets[:, :, :, 24]
        gt_h1 = targets[:, :, :, 25]

        gt_b2 = targets[:, :, :, 21]
        gt_x2 = targets[:, :, :, 26]
        gt_y2 = targets[:, :, :, 27]
        gt_w2 = targets[:, :, :, 28]
        gt_h2 = targets[:, :, :, 29]

        yt_b1 = outputs[:, :, :, 20]
        yt_x1 = outputs[:, :, :, 22]
        yt_y1 = outputs[:, :, :, 23]
        yt_w1 = outputs[:, :, :, 24]
        yt_h1 = outputs[:, :, :, 25]

        yt_b2 = outputs[:, :, :, 21]
        yt_x2 = outputs[:, :, :, 26]
        yt_y2 = outputs[:, :, :, 27]
        yt_w2 = outputs[:, :, :, 28]
        yt_h2 = outputs[:, :, :, 29]

        gt_p = targets[:, :, :, 0:20]
        yt_p = outputs[:, :, :, 0:20]

        loss_position = gt_b1 * ((yt_x1 - gt_x1) ** 2 + (yt_y1 - gt_y1) ** 2) + \
                      gt_b2 * ((yt_x2 - gt_x2) ** 2 + (yt_y2 - gt_y2) ** 2)

        loss_size = gt_b1 * ((yt_w1 ** 0.5 - gt_w1 ** 0.5) ** 2 + (yt_h1 ** 0.5 - gt_h1 ** 0.5) ** 2) + \
                      gt_b2 * ((yt_w2 ** 0.5 - gt_w2 ** 0.5) ** 2 + (yt_h2 ** 0.5 - gt_h2 ** 0.5) ** 2)

        loss_cls = (gt_b2 + gt_b1).unsqueeze(3) * ((gt_p - yt_p) ** 2)

        loss_cf_obj = gt_b1 * ((yt_b1 -gt_b1) ** 2)  + gt_b2 * ((yt_b2 - gt_b2) ** 2)
        loss_cf_noobj = (1 - gt_b1) * ((yt_b1 -gt_b1) ** 2)  + (1 - gt_b2) * ((yt_b2 - gt_b2) ** 2)

        loss_position = torch.mean(loss_position)
        loss_size = torch.mean(loss_size)
        loss_cls = torch.mean(loss_cls)
        loss_cf_obj = torch.mean(loss_cf_obj)
        loss_cf_noobj = torch.mean(loss_cf_noobj)
        print('==> loss distribution: loss_position = %.3f, loss_size = %.3f, loss_cls = %.3f, loss_cf_obj = %.3f, loss_cf_noobj = %.3f' % (
            loss_position.item(), loss_size.item(), loss_cls.item(), loss_cf_obj.item(), loss_cf_noobj.item()))

        losses = self.w_coord * (loss_position + loss_size) + loss_cf_obj +  self.w_noobj * loss_cf_noobj + loss_cls
        return losses


if __name__ == '__main__':
    criterion = YoloLoss()
    x1 = torch.randn(1,7,7,30)
    x2 = torch.randn(1,7,7,30)
    activate = nn.Sigmoid()
    x1[:, :, :, 20:] = activate(x1[:, :, :, 20:])
    x2[:, :, :, 20:] = activate(x2[:, :, :, 20:])

    loss = criterion(x1, x2)
    print(loss)
                    
        