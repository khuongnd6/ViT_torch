# %%
import argparse
import datetime
import json
import random
import time
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from models.swin import SwinTransformerOD, get_swin_model_od

from PIL import Image
import torchvision
# from torchvision.transforms.transforms import Resize
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# %%
os.environ['TORCH_HOME'] = '/host/ubuntu/torch'
time_stamp = time.strftime('%y%m%d_%H%M%S')


# %%
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    # parser.add_argument('--backbone', default='resnet50', type=str,
    #                     help="Name of the convolutional backbone to use")
    parser.add_argument('--backbone', default='swin_large_patch4_window12_384_22k', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/host/ubuntu/torch/coco2017')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=0, type=int)
    parser.add_argument('--train_limit', default=0, type=int)
    parser.add_argument('--val_limit', default=0, type=int)
    # parser.add_argument('--class_limit', default=0, type=int)
    parser.add_argument('--classes', default=[], nargs='+', type=int)
    parser.add_argument('--num_classes', default=91, type=int)
    
    return parser

parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

# %%
# num_classes = 20
# num_classes_dict = {
#     'coco': 91,
#     'coco_panoptic': 250,
# }
# if args.dataset_file in num_classes_dict:
#     num_classes = num_classes_dict[args.dataset_file]
if isinstance(args.classes, list) and len(args.classes) > 0:
    args.num_classes = len(args.classes) + 1

# args.num_classes = num_classes
print('num_classes:', args.num_classes)

# %%
print('args:', json.dumps(args.__dict__, indent=4))

# %%
dataset_train = build_dataset(image_set='train', args=args)
dataset_val = build_dataset(image_set='val', args=args)

# %%
# if args.distributed:
#     sampler_train = DistributedSampler(dataset_train)
#     sampler_val = DistributedSampler(dataset_val, shuffle=False)
# else:
if args.train_limit > 0:
    dataset_train = torch.utils.data.Subset(dataset_train, torch.arange(args.train_limit))
if args.val_limit > 0:
    dataset_val = torch.utils.data.Subset(dataset_val, torch.arange(args.val_limit))
sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)


# %%
batch_sampler_train = torch.utils.data.BatchSampler(
    sampler_train, args.batch_size, drop_last=True)

data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

# %%
if args.dataset_file == "coco_panoptic":
    # We also evaluate AP during panoptic training, on original coco DS
    coco_val = datasets.coco.build("val", args)
    base_ds = get_coco_api_from_dataset(coco_val)
else:
    base_ds = get_coco_api_from_dataset(dataset_val)

_ds = {
    'sets': {
        'train': dataset_train,
        'val': dataset_val,
    },
    'loaders': {
        'train': data_loader_train,
        'val': data_loader_val,
    },
}
print(json.dumps({
    k: {
        k1: len(v1)
        for k1, v1 in v.items()
    }
    for k, v in _ds.items()
}, indent=4))

# %%
def get_model_FRCNN(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


# %%
def do_training(
            model,
            dataloaders=[],
            base_ds=None,
            criterion=None,
            postprocessors=None,
            num_epochs=10,
            stats_fp='./logs/stats_{}.json'.format(time_stamp),
            telem={},
            info={},
            lr=0.001,
            lr_schedule_step=4,
            lr_schedule_gamma=0.3,
            initial_validation=True,
            device='cuda',
            ):
    
    # train on the GPU or on the CPU, if a GPU is not available
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print("Using device %s" % device)
    
    # move model to the right device
    model.to(device)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005,
    )
    # optimizer = torch.optim.AdamW(params, lr=args.lr,
    #                               weight_decay=args.weight_decay)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_schedule_step,
        gamma=lr_schedule_gamma,
    )
    
    _logging_stat = False
    stats = {
        'info': {**info},
        'telem': {
            'device': str(device),
            **telem,
        },
        'logs': [],
    }
    metric_loggers = []
    coco_evaluators = []
    if isinstance(stats_fp, str) and stats_fp.endswith('.json'):
        _dp = os.path.split(stats_fp)[0]
        if not os.path.isdir(_dp):
            os.makedirs(_dp)
        _logging_stat = True
    
    def save_stats(epoch, train_stats=None, val_stats=None, time_costs={}):
        _stat = {
            'epoch': epoch,
            **time_costs,
        }
        precision_keys = ['coco_eval_bbox']
        for _split, _stats in zip(['train', 'val'], [train_stats, val_stats]):
            if isinstance(_stats, dict):
                # print(list(_stats.keys()))
                
                # for k, v in _stats.items():
                #     if not isinstance(v, (int, float)):
                #         print(_split, k, v)
                _stat[_split] = {
                    str(k): float(v) for k, v in _stats.items()
                    if k not in precision_keys
                }
                for k in precision_keys:
                    if k not in _stats:
                        continue
                    for j, ap_type in enumerate(['ap', 'ar']):
                        for i, _variant in enumerate(['', '50', '75', 's', 'm', 'l']):
                            _stat[_split]['{}.{}{}'.format(k, ap_type, _variant)] = _stats[k][j * 6 + i]
        stats['logs'].append(_stat)
        with open(stats_fp, 'w') as f:
            json.dump(stats, f, indent=4)
        
    
    for epoch in range(num_epochs):
        
        # if epoch == 0 and initial_validation and len(dataloaders) >= 2:
        #     # _coco_evaluator = evaluate(model, dataloaders[1], device=device)
            
        #     test_stats, coco_evaluator = evaluate(
        #         model, criterion, postprocessors, dl, base_ds, device, args.output_dir
        #     )
        #     if _logging_stat:
        #         save_stats(
        #             epoch=0,
        #             val_stats=test_stats,
        #             time_costs={},
        #         )
        
        time_costs = {
            'time_start': time.time(),
            'time_finish': None,
            'time_cost': None,
        }
        for dl, _training in zip(dataloaders, [True, False]):
            _time_start = time.time()
            # if epoch == -1 and initial_validation:
            if _training:
                # _metric_logger = train_one_epoch(model, optimizer, dl, device, epoch, print_freq=10)
                
                train_stats = train_one_epoch(
                    model, criterion, dl, optimizer, device, epoch,
                    args.clip_max_norm)
                
                
                lr_scheduler.step()
                time_costs['time_train'] = time.time() - _time_start
                # print()
                # print('train_stats')
                # print(json.dumps(train_stats, indent=4))
            else:
                # _coco_evaluator = evaluate(model, dl, device=device)
                
                val_stats, coco_evaluator = evaluate(
                    model, criterion, postprocessors, dl, base_ds, device, args.output_dir
                )
                
                time_costs['time_val'] = time.time() - _time_start
                
                # print()
                # print('val_stats')
                # print(json.dumps(val_stats, indent=4))
        
        
        time_costs['time_finish'] = time.time()
        time_costs['time_cost'] = time_costs['time_finish'] - time_costs['time_start']
        
        # metric_loggers.append(_metric_logger)
        # coco_evaluators.append(_coco_evaluator)
        
        if _logging_stat:
            save_stats(
                epoch=epoch + 1,
                train_stats=train_stats,
                val_stats=val_stats,
                time_costs=time_costs,
            )
        
    # return metric_loggers, coco_evaluators

# %%
model, criterion, postprocessors = build_model(args)
model

# %%

# %%

model.to('cuda')
# %%
do_training(
    model=model,
    dataloaders=[data_loader_train, data_loader_val],
    base_ds=base_ds,
    num_epochs=args.epochs,
    stats_fp='./logs/stats_{}.json'.format(time_stamp),
    # stats_fp=None,
    telem={**args.__dict__},
    info={
        'time_stamp': time_stamp,
    },
    lr=0.001,
    lr_schedule_step=4,
    lr_schedule_gamma=0.3,
    initial_validation=False,
    device='cuda',
    postprocessors=postprocessors,
    criterion=criterion,
)

# %%