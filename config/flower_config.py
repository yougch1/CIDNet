# -*- coding: utf-8 -*-
"""
# @file name  : flower_config.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-04-23
# @brief      : 花朵分类参数配置
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import torchvision.transforms as transforms
from easydict import EasyDict

cfg = EasyDict()  # 访问属性的方式去使用key-value 即通过 .key获得value

cfg.model_name = "resnet18"
# cfg.model_name = "vgg16_bn"
# cfg.model_name = "se_resnet50"

cfg.mixup = True  # 是否采用mixup
cfg.mixup_alpha = 1.  # beta分布的参数. beta分布是一组定义在(0,1) 区间的连续概率分布。
cfg.label_smooth = True  # 是否采用标签平滑
cfg.label_smooth_eps = 0.01  # 标签平滑超参数 eps

data_dir = os.path.join(BASE_DIR, "..", "..", "data")
cfg.path_resnet18 = os.path.join(data_dir, "pretrained_model", "resnet18-5c106cde.pth")
cfg.path_vgg16bn = os.path.join(data_dir, "pretrained_model", "vgg16_bn-6c64b313.pth")
cfg.path_se_res50 = os.path.join(data_dir, "pretrained_model", "seresnet50-60a8950a85b2b.pkl")

cfg.train_bs = 64
cfg.valid_bs = 8
cfg.workers = 2

cfg.lr_init = 0.01
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.factor = 0.1
cfg.milestones = [30, 45]
cfg.max_epoch = 50
 
cfg.log_interval = 10

norm_mean = [0.485, 0.456, 0.406]  # imagenet 120万图像统计得来
norm_std = [0.229, 0.224, 0.225]
normTransform = transforms.Normalize(norm_mean, norm_std)

cfg.transforms_train = transforms.Compose([
        transforms.Resize((256)),  # (256, 256) 区别； （256） 最短边256
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normTransform,
])
cfg.transforms_valid = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normTransform,
])







