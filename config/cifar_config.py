# -*- coding: utf-8 -*-
"""
# @file name  : flower_config.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-04-23
# @brief      : cifar-10分类参数配置
"""
import os
import sys
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import torchvision.transforms as transforms
from easydict import EasyDict
from albumentations.pytorch import ToTensorV2
cfg = EasyDict()  # 访问属性的方式去使用key-value 即通过 .key获得value

cfg.pb = True  # 是否采用渐进式采样
# cfg.pb = False  # 是否采用渐进式采样
#cfg.mixup = True  # 是否采用mixup
cfg.mixup = False  # 是否采用mixup
#cfg.mixup_alpha = 1.  # beta分布的参数. beta分布是一组定义在(0,1) 区间的连续概率分布。
cfg.mixup_alpha = 0.8

cfg.label_smooth = False  # 是否采用标签平滑
cfg.label_smooth_eps = 0.01  # 标签平滑超参数 eps

cfg.train_bs = 32
cfg.valid_bs = 32
cfg.workers = 0

cfg.lr_init = 0.000001 #0.0005+batchsize/512 0.063
cfg.momentum = 0.9
cfg.weight_decay = 0.001
cfg.factor = 0.1
cfg.milestones = [160, 180,300,400,500,600,700,1000,1400,1500]
cfg.epochs = 400
 
cfg.log_interval = 10


# 数据预处理设置
cfg.norm_mean = [0.4914, 0.4822, 0.4465]    # cifar10 from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
cfg.norm_std = [0.2023, 0.1994, 0.2010]

normTransform = transforms.Normalize(cfg.norm_mean, cfg.norm_std)
# cfg.transforms_train = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
#
#     transforms.Resize((int(224 * 1.25), int(224 * 1.25))),
#     transforms.RandomRotation(15),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     normTransform
# ])
CFG = {
    'img_size': 224,
    'verbose_step': 1,
}
cfg.transforms_train =Compose([
        RandomResizedCrop(CFG['img_size'], CFG['img_size']),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        Cutout(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.)


cfg.transforms_valid =Compose([
        CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
        Resize(CFG['img_size'], CFG['img_size']),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)





cfg.transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normTransform
])


