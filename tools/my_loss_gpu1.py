# -*- coding: utf-8 -*-
"""
# @file name  : my_loss.py
# @author     : https://github.com/TingsongYu
# @date       : 2021-02-28 10:08:00
# @brief      : 新的loss
"""
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import os

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print("device", device)

torch.cuda.set_device(1)
class MWNLoss():
    """
    Multi Weighted New loss
    Args:
        gamma (float): the hyper-parameter of focal loss
        beta (float, 0.0 - 0.4):
        type: "zero", "fix", "decrease"
        sigmoid: "normal", "enlarge"
    """
    
    def __init__(self, para_dict=None):
        self.num_class_list = np.array(para_dict["num_class_list"])
        self.no_of_class = len(self.num_class_list)
        self.device = para_dict["device"]
        
        self.cfg = para_dict["cfgs"]
        self.class_weight_power = self.cfg.LOSS.WEIGHT_POWER  # 1.1
        self.class_extra_weight = np.array(self.cfg.LOSS.EXTRA_WEIGHT)  # [1. 1. 1.]
        self.scheduler = self.cfg.LOSS.SCHEDULER  # cls
        self.drw_epoch = self.cfg.LOSS.DRW_EPOCH  # 50
        self.cls_epoch_min = self.cfg.LOSS.CLS_EPOCH_MIN
        self.cls_epoch_max = self.cfg.LOSS.CLS_EPOCH_MAX
        self.gamma = self.cfg.LOSS.MWNL.GAMMA
        self.beta = self.cfg.LOSS.MWNL.BETA
        self.type = self.cfg.LOSS.MWNL.TYPE
        self.sigmoid = self.cfg.LOSS.MWNL.SIGMOID
        if self.beta > 0.4 or self.beta < 0.0:
            raise AttributeError(
                "For MWNLoss, the value of beta must be between 0.0 and 0.0 .")
    
    def __call__(self, x, target, epoch):
        x, target = x.to(device), target.to(device)
        
        if self.scheduler == "default":  # the weights of all classes are "1.0"
            per_cls_weights = np.array([1.0] * self.no_of_class)
        elif self.scheduler == "re_weight":
            per_cls_weights = 1.0 / (self.num_class_list.astype(np.float64))
            per_cls_weights = per_cls_weights * self.class_extra_weight
            per_cls_weights = [math.pow(num, self.class_weight_power) for num in per_cls_weights]
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * self.no_of_class
        elif self.scheduler == "drw":  # two-stage strategy using re-weighting at the second stage
            if epoch < self.drw_epoch:
                per_cls_weights = np.array([1.0] * self.no_of_class)
            else:
                per_cls_weights = 1.0 / (self.num_class_list.astype(np.float64))
                per_cls_weights = per_cls_weights * self.class_extra_weight
                per_cls_weights = [math.pow(num, self.class_weight_power) for num in per_cls_weights]
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * self.no_of_class
        elif self.scheduler == "cls":  # cumulative learning strategy
            if epoch <= self.cls_epoch_min:
                now_power = 0
            elif epoch < self.cls_epoch_max:
                now_power = ((epoch - self.cls_epoch_min) / (self.cls_epoch_max - self.cls_epoch_min)) ** 2
                now_power = now_power * self.class_weight_power
            else:
                now_power = self.class_weight_power
            
            per_cls_weights = 1.0 / (self.num_class_list.astype(np.float64))#[0.0002777  0.00107991 0.00018709]
            per_cls_weights = per_cls_weights * self.class_extra_weight#[0.0002777  0.00107991 0.00018709]
            per_cls_weights = [math.pow(num, now_power) for num in per_cls_weights]
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * self.no_of_class
        else:
            raise AttributeError(
                "loss scheduler can only be 'default', 're_weight', 'drw' and 'cls'.")
        #print("per_cls_weights: {}".format(per_cls_weights))
        weights = torch.FloatTensor(per_cls_weights).to(self.device)
        
        
        labels_one_hot = F.one_hot(target, self.no_of_class).float().to(self.device)
        # weights = self.weight
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_class)
        
        loss = F.binary_cross_entropy_with_logits(input=x, target=labels_one_hot, reduction="none")
        
        if self.beta > 0.0:
            th = - math.log(self.beta)
            if self.type == "zero":
                other = torch.zeros(loss.shape).to(self.device)
                loss = torch.where(loss <= th, loss, other)
            elif self.type == "fix":
                other = torch.ones(loss.shape).to(self.device)
                other = other * th
                loss = torch.where(loss <= th, loss, other)
            elif self.type == "decrease":
                pt = torch.exp(-1.0 * loss)
                loss = torch.where(loss <= th, loss, pt * th / self.beta)
        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels_one_hot * x
                                  - self.gamma * torch.log(1 + torch.exp(-1.0 * x)))
        
        loss = modulator * loss
        
        weighted_loss = weights * loss
        if self.sigmoid == "enlarge":
            weighted_loss = torch.mean(weighted_loss) * 30
        else:
            weighted_loss = weighted_loss.sum() / weights.sum()
        return weighted_loss


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)  # log_p 向量  【64，102】
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))  # Q向量  【target：64】】
        loss = (-weight * log_prob).sum(dim=-1).mean()  # log_p * Q 再相加
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average # 对batch里面的数据取均值/求和

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss) #一个样本的交叉熵
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class CB_loss(nn.Module):
    def __init__(self, gamma=2, beta=0.999, samples_per_cls=[3601, 926, 5345], no_of_classes=3, size_average=True,
                 ignore_index=255, ):
        super(CB_loss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.beta = beta
        self.gamma = gamma
        self.no_of_classes = no_of_classes
        self.ignore_index = ignore_index
        self.size_average = size_average  # 对batch里面的数据取均值/求和
    
    def forward(self, inputs, targets):
        
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes
        
        labels_one_hot = F.one_hot(targets, self.no_of_classes).float()
        
        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes)
        # BCLoss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        BCLoss = F.binary_cross_entropy_with_logits(inputs, labels_one_hot, reduction="none")
        
        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels_one_hot * inputs - self.gamma * torch.log(1 +
                                                                                                 torch.exp(
                                                                                                     -1.0 * inputs)))
        
        loss = modulator * BCLoss
        
        weighted_loss = weights * loss
        focal_loss = torch.sum(weighted_loss)
        
        focal_loss /= torch.sum(targets)
        return focal_loss



class CrossEntropy(nn.Module):
    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__()

    def forward(self, output, target):
        output = output
        loss = F.cross_entropy(output, target)
        return loss


class CSCE(nn.Module):

    def __init__(self, para_dict=None):
        super(CSCE, self).__init__()
        self.num_class_list = para_dict["num_class_list"]
        self.device = para_dict["device"]

        cfg = para_dict["cfg"]
        #scheduler = cfg.LOSS.CSCE.SCHEDULER
        scheduler = cfg.LOSS.SCHEDULER
        
        #self.step_epoch = cfg.LOSS.CSCE.DRW_EPOCH
        self.step_epoch = cfg.LOSS.DRW_EPOCH
        

        if scheduler == "drw":
            self.betas = [0, 0.999999]
        elif scheduler == "default":
            self.betas = [0.999999, 0.999999]
        self.weight = None

    def update_weight(self, beta):
        effective_num = 1.0 - np.power(beta, self.num_class_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).to(self.device)

    def reset_epoch(self, epoch):
        idx = (epoch-1) // self.step_epoch
        beta = self.betas[idx]
        self.update_weight(beta)

    def forward(self, x, target, **kwargs):
        #target=target.cuda()
        #x=x.cuda()
        target = target.to(self.device)
        x = x.to(self.device)
        
        
        return F.cross_entropy(x, target, weight=self.weight)


# The LDAMLoss class is copied from the official PyTorch implementation in LDAM (https://github.com/kaidic/LDAM-DRW).
class LDAMLoss(nn.Module):

    def __init__(self, para_dict=None):
        super(LDAMLoss, self).__init__()
        s = 30
        self.num_class_list = para_dict["num_class_list"]
        self.device = para_dict["device"]

        cfg = para_dict["cfgs"]
        max_m = cfg.LOSS.LDAM.MAX_MARGIN
        m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(self.device)
        self.m_list = m_list
        assert s > 0

        self.s = s
        self.step_epoch = cfg.LOSS.LDAM.DRW_EPOCH
        self.weight = None

    def reset_epoch(self, epoch):
        idx = (epoch-1) // self.step_epoch
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], self.num_class_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).to(self.device)

    def forward(self, x, target):
        x=x.cuda()
        target=target.cuda()
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.FloatTensor)
        index_float = index_float.to(self.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x=x.cuda()
        x_m = x - batch_m
        x_m=x_m.cuda()
        output = torch.where(index, x_m.cuda(), x.cuda())
        
        return F.cross_entropy(self.s * output, target, weight=self.weight)



if __name__ == '__main__':
   
    
    output = torch.tensor([[4.0, 5.0, 10.0], [1.0, 5.0, 4.0], [1.0, 15.0, 4.0]])
    label = torch.tensor([2, 1, 1], dtype=torch.int64)

    #criterion = LabelSmoothLoss(0.01)+FocalLoss()
    #loss_f=CB_loss()
    loss_f=MWNLoss()
    loss=loss_f(output,label)
    #loss = criterion(output, label)

    print("CrossEntropy:{}".format(loss))



