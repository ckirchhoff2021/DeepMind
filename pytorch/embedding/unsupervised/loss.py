import torch
from torch.autograd import Function
from torch import nn
import math


class BatchCriterion(nn.Module):
    def __init__(self, T):
        super(BatchCriterion, self).__init__()
        self.T = T
    
    def slowly_forward(self, y1, y2):
        num = y1.size(0)
        p1 = list()
        for i in range(num):
            fi = y1[i]
            fi_hat = y2[i]
            d1 = (fi * fi_hat).sum().div(self.T).exp()
            d2 = 0
            for k in range(num):
                fk = y1[k]
                d2 += (fk * fi_hat).sum().div(self.T).exp()
            value = (d1 / d2).log()
            p1.append(value)
        
        p2 = list()
        for i in range(num):
            fi = y1[i]
            log_sum = 0
            for j in range(num):
                if i == j:
                    continue
                fj = y1[j]
                d1 = (fi * fj).sum().div(self.T).exp()
                d2 = 0
                for k in range(num):
                    fk = y1[k]
                    d2 += (fk * fj).sum().div(self.T).exp()
                value = d1 / d2
                log_sum += (1 - value).log()
            p2.append(log_sum)
        
        loss = (-sum(p1) - sum(p2)) / num
        return loss
    
    def forward(self, y):
        y1 = y.narrow(0, 0, 128)
        y2 = y.narrow(0, 128, 128)
        num = y1.size(0)
        y3 = torch.mm(y1, y2.t()).div_(self.T).exp_()
        y4 = y3 / y3.sum(1, keepdim=True)
        prob_p = y4.diag().log_().sum(0)
        
        y6 = torch.mm(y1, y1.t()).div_(self.T).exp_()
        y7 = (1.0 - y6 / y6.sum(1,keepdim=True)).log_()
        y8 = y7.sum(1) - y7.diag()
        prob_n = y8.sum(0)
        loss = (-prob_p - prob_n) / num
        return loss
        
    def modified_forward(self, y):
        num = y.size(0)
        y1 = y
        y2 = torch.cat((y1.narrow(0,128,128), y1.narrow(0,0,128)), 0)
        positive = (y1 * y2).sum(1).div_(self.T).exp_()
        values = torch.mm(y1, y1.t()).div_(self.T).exp_()
        
        diag_filter = 1.0 - torch.eye(128)
        s1 = values[0:128,0:128]
        s2 = values[0:128,128:256]
        
        s3 = values[128:256,0:128]
        s4 = values[128:256,128:256]
        
        s5 = torch.cat((s2,s4), dim=0)
        p1 = (positive / s5.sum(1)).log_()
        
        s1 = (1.0 - s1.div(s1.sum(1,keepdim=True)) * diag_filter).log_().sum()
        s3 = (1.0 - s3.div(s3.sum(1,keepdim=True)) * diag_filter).log_().sum()
        
        prob_n = s1 + s3
        prob_p = p1.sum()
        loss = (-prob_p - prob_n) / num
        return loss
        
    def official_forward(self, y):
        num = y.size(0)
        diag_filter = 1.0 - torch.eye(256)
        y1 = y
        y2 = torch.cat((y1.narrow(0, 128, 128), y1.narrow(0, 0, 128)), 0)
        positive = (y1 * y2).sum(1).div_(self.T).exp_()
        probs = torch.mm(y1, y1.t()).div_(self.T).exp_() * diag_filter
        
        sum_values = probs.sum(1)
        prob_p = torch.div(positive, sum_values)
        
        values = sum_values.repeat(num, 1)
        prob_n = torch.div(probs, values.t())
        prob_n = -prob_n.add(-1)
        prob_n.log_()
        
        prob_n = prob_n.sum(1) - (-prob_p.add(-1)).log_()
        prob_p.log_()
      
        prob_n_sum = prob_n.sum(0)
        prob_p_sum = prob_p.sum(0)
        loss = - (prob_p_sum + prob_n_sum) / num
        
        return loss

    
        
        
        
        