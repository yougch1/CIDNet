# -*- coding: utf-8 -*-
"""
# @file name  : model_trainer.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-29
# @brief      : 模型训练类
"""
import torch
from torch.cuda.amp import autocast, GradScaler

import numpy as np
from collections import Counter
from tools.mixup import mixup_criterion, mixup_data
from tensorboardX import SummaryWriter

from tools.my_loss import LabelSmoothLoss,FocalLoss

writer = SummaryWriter(log_dir='../results/events/')


def _log_stats_train(train_results, epoch):
    tag_value = {'training_loss': train_results['train_loss'],
                 'training_accuracy': train_results['train_accuracy']}
    for tag, value in tag_value.items():
        writer.add_scalar(tag, value, epoch)


def _log_stats_val(val_results, epoch):
    tag_value = {'validation_loss': val_results['val_loss'],
                 'validation_accuracy': val_results['val_accuracy']}
    for tag, value in tag_value.items():
        writer.add_scalar(tag, value, epoch)


class ModelTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, scheduler, epoch_idx, device, cfg, logger):
        model.train()

        class_num = data_loader.dataset.cls_num#2
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        loss_mean = 0
        acc_avg = 0
        path_error = []
        label_list = []
        for i, data in enumerate(data_loader):#data torch.Size([128, 3, 32, 32])。
            # _, labels = data
            inputs, labels, path_imgs = data
           # label_list.extend(labels.tolist())
            label_list.extend(list(labels))
            labels=torch.tensor(labels,dtype=torch.long)

           
            # inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            #print("label_type",type(labels))

            # mixup
            if cfg.mixup:
                mixed_inputs, label_a, label_b, lam = mixup_data(inputs, labels, cfg.mixup_alpha, device)
                inputs = mixed_inputs

            # forward & backward
            outputs = model(inputs)#torch.Size([128, 10])
            # print("type_outputs",type(outputs))
            # print("type_labels",type(labels))
            
            optimizer.zero_grad()
            # loss 计算
            if cfg.mixup:
                #loss = mixup_criterion(loss_f, outputs.cpu(), label_a.cpu(), label_b.cpu(), lam)
                loss = mixup_criterion(loss_f, outputs.cpu(), label_a.cpu(), label_b.cpu(), epoch_idx,lam)
            else:
                #loss = loss_f(outputs.cpu(), labels.cpu(),epoch_idx)
                loss = loss_f(outputs[0][0], labels, epoch_idx)
            loss.backward()
            #scaler = GradScaler()
          #  scaler.scale(loss).backward()
            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()
            # optimizer.zero_grad()

            # 统计loss
            loss_sigma.append(loss.item())
            loss_mean = np.mean(loss_sigma)
            
            _, predicted = torch.max(outputs[0][0].data, 1)
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.
                if cate_i != pre_i:
                    path_error.append((cate_i, pre_i, path_imgs[j]))    # 记录错误样本的信息
            acc_avg = conf_mat.trace() / conf_mat.sum()
            
            

            # 每10个iteration 打印一次训练信息
            if i % cfg.log_interval == cfg.log_interval - 1:
                logger.info("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".
                            format(epoch_idx + 1, cfg.epochs, i + 1, len(data_loader), loss_mean, acc_avg))
        logger.info("epoch:{} sampler: {}".format(epoch_idx, Counter(label_list)))
        return loss_mean, acc_avg, conf_mat, path_error

    

    # @staticmethod
    # def valid(data_loader, model, loss_f, device):
    #     model.eval()
    #
    #     class_num = data_loader.dataset.cls_num
    #     conf_mat = np.zeros((class_num, class_num))
    #     loss_sigma = []
    #     path_error = []
    #
    #     for i, data in enumerate(data_loader):
    #         inputs, labels, path_imgs = data
    #         # inputs, labels = data
    #         inputs, labels = inputs.to(device), labels.to(device)
    #
    #         outputs = model(inputs)
    #         loss = loss_f(outputs.cpu(), labels.cpu())
    #
    #         # 统计混淆矩阵
    #         _, predicted = torch.max(outputs.data, 1)
    #         for j in range(len(labels)):
    #             cate_i = labels[j].cpu().numpy()
    #             pre_i = predicted[j].cpu().numpy()
    #             conf_mat[cate_i, pre_i] += 1.
    #             if cate_i != pre_i:
    #                 path_error.append((cate_i, pre_i, path_imgs[j]))   # 记录错误样本的信息
    #
    #         # 统计loss
    #         loss_sigma.append(loss.item())
    #
    #     acc_avg = conf_mat.trace() / conf_mat.sum()
    #
    #     return np.mean(loss_sigma), acc_avg, conf_mat, path_error

    @staticmethod
    def valid(data_loader, model, loss_f, epoch,device):
        model.eval()


        class_num = data_loader.dataset.cls_num
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        path_error = []
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs, labels, path_imgs = data
                # inputs, labels = data
                labels = torch.tensor(labels, dtype=torch.long)
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = loss_f(outputs[0][0], labels,epoch)

                # 统计混淆矩阵
                _, predicted = torch.max(outputs[0][0].data, 1)
                for j in range(len(labels)):
                    cate_i = labels[j].cpu().numpy()
                    pre_i = predicted[j].cpu().numpy()
                    conf_mat[cate_i, pre_i] += 1.
                    if cate_i != pre_i:
                        path_error.append((cate_i, pre_i, path_imgs[j]))   # 记录错误样本的信息

                # 统计loss
                loss_sigma.append(loss.item())

            acc_avg = conf_mat.trace() / conf_mat.sum()

            return np.mean(loss_sigma), acc_avg, conf_mat, path_error

    @staticmethod
    def valid_probability(data_loader, model, loss_f, epoch,device):
        model.eval()


        class_num = data_loader.dataset.cls_num
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        path_error = []
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs, labels, path_imgs = data
                # inputs, labels = data
                labels = torch.tensor(labels, dtype=torch.long)
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = loss_f(outputs.cpu()[0][0], labels.cpu(),epoch)

                # 统计混淆矩阵
                _, predicted = torch.max(outputs[0][0].data, 1)
                for j in range(len(labels)):
                    cate_i = labels[j].cpu().numpy()
                    pre_i = predicted[j].cpu().numpy()
                    conf_mat[cate_i, pre_i] += 1.
                    if cate_i != pre_i:
                        path_error.append((cate_i, pre_i, path_imgs[j]))   # 记录错误样本的信息

                # 统计loss
                loss_sigma.append(loss.item())

            acc_avg = conf_mat.trace() / conf_mat.sum()

            return np.mean(loss_sigma), acc_avg, conf_mat, path_error