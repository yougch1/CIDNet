

import sys

BASE_DIR = r'/home/gwj/Intussption_classification'
BASE_DIR1 = r'/home/gwj/Intussption_classification/models'
# # # BASE_DIR2 = r'/home/gwj/Intussption_classification/config'
BASE_DIR3 = r'/home/gwj/Intussption_classification/util'
# # #
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR1)
# sys.path.append(BASE_DIR2)
sys.path.append(BASE_DIR3)

import datasets.samplers as samplers

print(sys.path)

import argparse
from tools.model_trainerO import ModelTrainer

from tools.common_tools import *

from timm.optim import Nadam
from misc import NestedTensor

#from models.backbone_convnext import *
from models.backbone_efficientnet import *
from models.backbone import build_WSIFeatureMapBackbone

from models.deformable_transformer import build_deforamble_transformer
from timm.scheduler import create_scheduler
from config import cfg_c, update_config, update_cfg_name
from torch.utils.data import DataLoader
from tools.my_loss import *
import torch.nn.functional as F
from config.cifar_config_improve import cfg
from datetime import datetime
from datasets.cifar_longtail_convnext_gs import CifarDataset

from tools.progressively_balance import ProgressiveSampler
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from pytorchtool import EarlyStopping
from tensorboardX import SummaryWriter
from utils.scheduler import *
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# setup_seed(12345)  # 先固定随机种子
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='code for diagnose intussusception')
parser.add_argument(
	"--cfg_c",
	help="decide which cfg to use",
	required=False,
	default="../config/Intussusception_2022.yaml",
	type=str,
)
parser.add_argument("--tensorboard_events", type=str, default='../results/events/',
                    help="path to tensorboard events")
parser.add_argument('--sched', default='poly', type=str, metavar='SCHEDULER',
                    choices=['cosine', 'tanh', 'step', 'multistep', 'poly'],
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', default=0.0001, help='learning rate')

parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
parser.add_argument('--lr_backbone', default=2e-5, type=float)
parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)

parser.add_argument('--lr_drop', default=40, type=int)
parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
parser.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='gradient clipping max norm')

parser.add_argument('--two_stage', default=False, action='store_true')

# Model parameters
parser.add_argument('--frozen_weights', type=str, default=None,
                    help="Path to the pretrained model. If set, only the mask head will be trained")

# parser.add_argument('--backbone', default='resnet152', type=str,
#                     help="Name of the convolutional backbone to use")
parser.add_argument('--backbone', default='efficientnet_b7', type=str,
                    help="Name of the convolutional backbone to use")


parser.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                    help="position / size * scale")

parser.add_argument('--num_feature_levels', default=1, type=int, help='number of feature levels')

#parser.add_argument('--num_class', default=3, type=int, help='number of class')
parser.add_argument('--map_size', default=22, type=int, help='size of merge feature map')

# * Transformer
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=1024, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")

parser.add_argument('--num_queries', default=100, type=int,
                    help="Number of query slots")
parser.add_argument('--dec_n_points', default=4, type=int)
parser.add_argument('--enc_n_points', default=4, type=int)
parser.add_argument('--mask_loss_coef', default=1, type=float)
parser.add_argument('--dice_loss_coef', default=1, type=float)
parser.add_argument('--cls_loss_coef', default=2, type=float)
parser.add_argument('--focal_alpha', default=0.25, type=float)
parser.add_argument('--output_dir', default='../results',
                    help='path where to save, empty for no saving')

parser.add_argument('--masks', action='store_true',
                    help="Train segmentation head if the flag is provided")

#parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

# custom param
parser.add_argument('--num_input_channels', default=1280, type=int)
parser.add_argument('--out_channels', default=256, type=int)
parser.add_argument('--feat_map_size', default=60, type=int)  # size of featuremap

parser.add_argument('--model_name', default='transformer', type=str)  #
parser.add_argument('--drop_decoder', action='store_true')



parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--num_class', default=3, type=int, help='number of class')

#
# parser.add_argument("--lr_policy", type=str, default='cosine',
#                     choices=['poly', 'step', 'multi_step', 'exponential', 'cosine', 'lambda', 'onecycle'],
#                     help="learning rate scheduler policy")
parser.add_argument('--bs', default=32, help='training batch size')
parser.add_argument('--epochs', default=400)
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
# parser.add_argument('--data_root_dir', default=r"G:\deep_learning_data\cifar10",
#                     help="path to your dataset")
parser.add_argument('--data_root_dir', default=r"../data/",
                    help="path to your dataset")
parser.add_argument("--optimizer", type=str, default='adam',
                    choices=['sgd', 'Nadam', 'adam', 'AdamW', 'adamw', 'adadelta', 'rmsprop', 'rmsproptf',
                             'fusedadamw'],
                    help="choose optimizer")

parser.add_argument(
	"opts",
	help="modify config options using the command-line",
	default=None,
	nargs=argparse.REMAINDER,
)
# 显卡配置
parser.add_argument("--gpu_id", type=str, default='0',
                    help="GPU ID")
parser.add_argument("--multi_gpu", action='store_true', default=False)
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')


def log_stats_train(train_results, epoch):
	tag_value = {'train_efficienet_gs_loss': train_results['train_loss'],
	             'train_efficienet_gs_accuracy': train_results['train_accuracy']}
	for tag, value in tag_value.items():
		writer.add_scalar(tag, value, epoch)


def log_stats_val(val_results, epoch):
	tag_value = {'validation_efficienet_gs_loss': val_results['val_loss'],
	             'validation_efficienet_gs_accuracy': val_results['val_accuracy']}
	for tag, value in tag_value.items():
		writer.add_scalar(tag, value, epoch)


def check_data_dir(path_data):
	if not os.path.exists(path_data):
		print("文件夹不存在，请检查数据是否存放到data_dir变量:{}".format(path_data))

class DeformableTransformerMIL(nn.Module):
    """ This is the DT-MIL module that performs WSI Classification """

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model


        # Modify class_embed for all queries embbeding，and then perform classification on the whole WSI
        self.class_embed = nn.Linear(hidden_dim * num_queries, num_classes)

        self.num_feature_levels = num_feature_levels


        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        print(backbone.num_channels)
        if num_feature_levels > 1:

            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers

        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])


    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()

            # src.requires_grad_(False)
            # mask.requires_grad_(False)
            # print(src.shape)
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = self.query_embed.weight

        #hs=[32,100,256]
        hs, encoder_intermediate_result = self.transformer(srcs, masks, pos, query_embeds)

        hs = hs.view(hs.shape[0], -1)

        outputs_class = self.class_embed[-1](hs)

        out = {'pred_logits': outputs_class, }
        return out['pred_logits']

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]






if __name__ == "__main__":
	args = parser.parse_args()
	update_config(cfg_c, args)
	update_cfg_name(cfg_c)  # modify the cfg.NAME
	print("cfg_c",cfg_c)
	import models.misc as utils
	
	utils.init_distributed_mode(args)
	#print("git:\n  {}\n".format(utils.get_sha()))
	
	# 显卡环境
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	cfg.lr_init = args.lr if args.lr else cfg.lr_init
	cfg.train_bs = args.bs if args.bs else cfg.train_bs
	cfg.epochs = args.epochs if args.epochs else cfg.epochs
	
	# 日志 & Tensorboard
	# train_logger = logger.get_logger(opts.logs + opts.dataset)
	
	writer = SummaryWriter(log_dir=args.tensorboard_events)
	
	# update_config(cfgs, args)
	# update_cfg_name(cfgs)  # modify
	# step0: setting path
	train_dir = args.data_root_dir
	valid_dir = args.data_root_dir
	# train_dir = os.path.join(args.data_root_dir, "cifar10_train")#'/Users/hello/Downloads/cifar-10/cifar10_train'
	# valid_dir = os.path.join(args.data_root_dir, "cifar10_test")#'/Users/hello/Downloads/cifar-10/cifar10_test'
	check_data_dir(train_dir)
	check_data_dir(valid_dir)
	num_classes = 3
	# num_class_list=['normal','sleeve_sign','concentric_circle_sign']
	num_class_list = [3601, 926, 5345]
	para_dict = {
		"num_classes": num_classes,
		"num_class_list": num_class_list,
		"cfgs": cfg_c,
		"device": device
	}
	
	# 创建logger
	# res_dir = os.path.join(BASE_DIR, "..", "..", "results")#'/Users/hello/PycharmProjects/MTX/src/../../results'
	# res_dir = os.path.join(BASE_DIR, "results_densnet_2022_0328")
	# res_dir = os.path.join(BASE_DIR, "results_CoAtNet_2022_0427_0938AM")
	res_dir = os.path.join(BASE_DIR, "results_efficienet_gs_2022_0518_0958PM")
	logger, log_dir = make_logger(res_dir)
	
	# step1： 数据集
	# 构建MyDataset实例， 构建DataLoder
	# train_data = CifarDataset(root_dir=train_dir, transform=cfg.transforms_train, isTrain=True)
	# valid_data = CifarDataset(root_dir=valid_dir, transform=cfg.transforms_valid, isTrain=False)
	
	# train_data = CifarDataset(root_dir=train_dir,transform=cfg.transforms_train,mode="train")
	# valid_data = CifarDataset(root_dir=valid_dir, transform=cfg.transforms_train, mode="val")
	import models.misc as utils
	train_data = CifarDataset(root_dir=train_dir, transform=cfg.transforms_train,mode="train", do_fmix=False,
	                          do_cutmix=False)
	valid_data = CifarDataset(root_dir=valid_dir, transform=cfg.transforms_train,  mode="val", do_fmix=False,
	                          do_cutmix=False)
	
	# train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.workers)
	# valid_loader = DataLoader(dataset=valid_data, batch_size=cfg.valid_bs, num_workers=cfg.workers)
	if args.distributed:
		
		if args.cache_mode:
			sampler_train = samplers.NodeDistributedSampler(train_data)
			sampler_val = samplers.NodeDistributedSampler(valid_data, shuffle=False)
		else:
			sampler_train = samplers.DistributedSampler(train_data)
			sampler_val = samplers.DistributedSampler(valid_data, shuffle=False)
	else:
		sampler_train = torch.utils.data.RandomSampler(train_data)
		sampler_val = torch.utils.data.SequentialSampler(valid_data)
	
	batch_sampler_train = torch.utils.data.BatchSampler(
		sampler_train, args.bs, drop_last=True)
	
	# train_loader = DataLoader(train_data, batch_sampler=batch_sampler_train,
	#                                collate_fn=utils.collate_fn, num_workers=args.num_workers,
	#                                pin_memory=True)
	
	valid_loader = DataLoader(valid_data, args.bs,
	                             drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
	                             pin_memory=True)
	
	#train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, collate_fn=utils.collate_fn,shuffle=True, num_workers=cfg.workers)
	#valid_loader = DataLoader(dataset=valid_data, batch_size=cfg.valid_bs, collate_fn=utils.collate_fn, num_workers=cfg.workers)
	
	
	
	if cfg.pb:  # true
		sampler_generator = ProgressiveSampler(train_data,
		                                       cfg.epochs)  # <tools.progressively_balance.ProgressiveSampler object at 0x7fea26815f40>
	
	# step2: 模型
	
	num_classes = args.num_class
	
	#backbone = build_WSIFeatureMapBackbone(args)
	backbone = build_backbone(args)
	
	
	transformer = build_deforamble_transformer(args)
	model = DeformableTransformerMIL(
		backbone,
		transformer,
		num_classes=num_classes,
		num_queries=args.num_queries,
		num_feature_levels=args.num_feature_levels,
	)
	model.to(device)

	early_stopping = EarlyStopping(10, verbose=True)

	
	# step3: 损失函数、优化器
	if cfg.label_smooth:
		loss_f = LabelSmoothLoss(cfg.label_smooth_eps)
	else:
		# loss_f = FocalLoss()
		# loss_f=CB_loss()
		loss_f = MWNLoss(para_dict)
	# loss_f = nn.CrossEntropyLoss()
	# loss_f = nn.nn.BCELoss()
	
	# 优化器
	if args.optimizer == 'sgd':
		# optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9,
		# weight_decay=opts.weight_decay)
		optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr_init, momentum=cfg.momentum,
		                            weight_decay=cfg.weight_decay)
	if args.optimizer == 'adam':
		optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_init)
	if args.optimizer == 'AdamW':
		optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_init, betas=(0.9, 0.999), eps=1e-08,
		                              weight_decay=cfg.weight_decay, amsgrad=False,
		                              maximize=False)
	if args.optimizer == 'Nadam':
		optimizer = Nadam(model.parameters(), lr=cfg.lr_init, weight_decay=cfg.weight_decay, eps=1e-8)
	
	# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)
	# scheduler = option(args,optimizer, args.lr_policy, args.max_epoch)
	scheduler, num_epochs = create_scheduler(args, optimizer)
	# step4: 迭代训练
	# 记录训练所采用的模型、损失函数、优化器、配置参数cfg
	logger.info("cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n model:\n{}".format(
		cfg, loss_f, scheduler, optimizer, model))
	
	loss_rec = {"train": [], "valid": []}
	acc_rec = {"train": [], "valid": []}
	best_acc, best_epoch = 0, 0
	print("start training")
	for epoch in range(cfg.epochs):
		if cfg.pb:
			sampler, _ = sampler_generator(epoch)  # sampler就是第几张图像的索引
			#train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=False,
			                         # num_workers=cfg.workers,
			                         # sampler=sampler)
		
		if epoch < 10:
			
			train_data = CifarDataset(root_dir=train_dir, transform=cfg.transforms_train,blur_kernel_size=(65, 65), sigma=16, mode="train", do_fmix=False,
			                          do_cutmix=False)
			  # , collate_fn=utils.collate_fn)
		elif epoch >= 10 and epoch < 15:
			train_data = CifarDataset(root_dir=train_dir, transform=cfg.transforms_train, blur_kernel_size=(33, 33),
			                          sigma=8, mode="train", do_fmix=False,
			                          do_cutmix=False)
			
			
	
		elif epoch >= 15 and epoch < 20:
			train_data = CifarDataset(root_dir=train_dir, transform=cfg.transforms_train, blur_kernel_size=(17, 17),
			                          sigma=4, mode="train", do_fmix=False,
			                          do_cutmix=False)
			
			
		elif epoch >= 20 and epoch < 25:
			train_data = CifarDataset(root_dir=train_dir, transform=cfg.transforms_train, blur_kernel_size=(9, 9),
			                          sigma=2, mode="train", do_fmix=False,
			                          do_cutmix=False)
			
		
		elif epoch >= 25 and epoch < 30:
			train_data = CifarDataset(root_dir=train_dir, transform=cfg.transforms_train, blur_kernel_size=(5, 5),
			                          sigma=1, mode="train", do_fmix=False,
			                          do_cutmix=False)
			
		
		else:
			train_data = CifarDataset(root_dir=train_dir, transform=cfg.transforms_train, to_blur=False, mode="train", do_fmix=False,
			                          do_cutmix=False)
			
		
		train_loader = DataLoader(train_data,collate_fn=utils.collate_fn, batch_size=args.bs,num_workers=args.num_workers,sampler=sampler)
		
		loss_train, acc_train, mat_train, path_error_train = ModelTrainer.train(
			train_loader, model, loss_f, optimizer, scheduler, epoch, device, cfg, logger)
		
		loss_valid, acc_valid, mat_valid, path_error_valid = ModelTrainer.valid(
			valid_loader, model, loss_f, epoch, device)
		
		train_results = {"train_loss": loss_train, 'train_accuracy': acc_train}
		log_stats_train(train_results, epoch)
		
		# print("train_results",train_results["train_loss"])
		# print("train_accuracy", train_results["train_accuracy"])
		
		val_results = {'val_loss': loss_valid, 'val_accuracy': acc_valid}
		
		log_stats_val(val_results, epoch)
		
		logger.info("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}". \
		            format(epoch + 1, cfg.epochs, acc_train, acc_valid, loss_train, loss_valid,
		                   optimizer.param_groups[0]["lr"]))
		scheduler.step(epoch)
		# 记录训练信息
		loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
		acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)
		
		# 保存混淆矩阵图
		show_confMat(mat_train, train_data.names, "train", log_dir, epoch=epoch, verbose=epoch == cfg.epochs - 1)
		show_confMat(mat_valid, valid_data.names, "valid", log_dir, epoch=epoch, verbose=epoch == cfg.epochs - 1)
		# 保存loss曲线， acc曲线
		plt_x = np.arange(1, epoch + 2)
		plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
		plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)
		
		# 模型保存
		if best_acc < acc_valid or epoch == cfg.epochs - 1:
			best_epoch = epoch if best_acc < acc_valid else best_epoch
			best_acc = acc_valid if best_acc < acc_valid else best_acc
			checkpoint = {"model_state_dict": model.state_dict(),
			              "optimizer_state_dict": optimizer.state_dict(),
			              "epoch": epoch,
			              "best_acc": best_acc}
			pkl_name = "checkpoint_densnet_{}.pkl".format(epoch) if epoch == cfg.epochs - 1 else "checkpoint_best.pkl"
			path_checkpoint = os.path.join(log_dir, pkl_name)
			torch.save(checkpoint, path_checkpoint)
			torch.cuda.empty_cache()
			
			# 保存错误图片的路径
			err_ims_name = "error_imgs_{}.pkl".format(epoch) if epoch == cfg.epochs - 1 else "error_imgs_best.pkl"
			path_err_imgs = os.path.join(log_dir, err_ims_name)
			error_info = {}
			error_info["train"] = path_error_train
			error_info["valid"] = path_error_valid
			pickle.dump(error_info, open(path_err_imgs, 'wb'))
		early_stopping(loss_valid, model)
		if early_stopping.early_stop:
			print("Early stopping")
			# 结束模型训练
			break
	logger.info("{} done, best acc: {} in :{}".format(
		datetime.strftime(datetime.now(), '%m-%d_%H-%M'), best_acc, best_epoch))
