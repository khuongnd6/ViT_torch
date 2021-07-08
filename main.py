# %%
import time, os, json, string, re, datetime
import argparse
import numpy as np
from tabulate import Line

import torch
from torch.nn.modules.activation import GELU
from torch.nn.modules.linear import Linear
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from functools import partial

from torchvision.transforms.functional import InterpolationMode
# from xcit import XCiT

# from utils_progressbar import *
from utils_args import ARGS
# from utils_smi import NVIDIA_SMI

from utils_datasets import Datasets
from utils_dataset_tire import get_tire_dataset
from models.vision_all import VisionModelZoo
from utils_network import Network

from multiprocessing import Pool



# %%


# %%
time_start_master = time.time()
time_stamp = time.strftime('%y%m%d_%H%M%S')

# %%
config = [
    ('device',  'cuda',  str,  ['cuda', 'cpu']),
    ('epoch',  100,  int, None, 'number of training epochs'),
    ('dataset', 'tire', str, None, 'name of the dataset'),
    ('data_path', '/host/ubuntu/torch/tire/tire_500', str, None, 'path to the local image folder'),
    ('bs', 512, int, None, 'batch size'),
    ('root_path', '/host/ubuntu/torch', str, None, 'path of the folder to put the pretrained models and download datasets'),
    ('arch', 'dino_vits16', str, None, 'backbone network architecture'),
    ('lr', 0.001, float, None, 'initial learning rate'),
    ('lr_schedule_half', 10, int, None, 'number of epochs between halving the learning rate'),
    ('limit_train', 0, int, None, 'set to int >0 to limit the number of training samples'),
    ('limit_test', 0, int, None, 'set to int >0 to limit the number of testing samples'),
    ('stats_json', ''.format(time_stamp), str),
    ('master_stats_json', './logs/stats_master.json', str),
    # ('test_only', False, bool),
    # ('train_only', False, bool),
    ('lineareval', False, bool),
    # ('earlystopping', True, bool),
    ('pretrained', False, bool),
    ('note', '', str, None, 'note to recognize the run'),
    ('opt', 'sgd', str, None, 'set the optimizer'),
    ('fc', [], int, None, 'the units for the additional fc layers'),
    # ('multiple_frozen', 1, int, None, 'multiply the frozen train dataset by this amount'),
    # ('update_frozen_rate', 2, int, None, 'set to int >0 to update the frozen values every <this> epochs'),
    ('aug_auto_imagenet', 1, int, [0, 1]),
    ('aug_random_crop', 1, int, [0, 1]),
    ('aug_color_jitter', 1, int, [0, 1]),
    ('image_size', 0, int),
    ('tire_settings', 0, int, None, 'settings [0-3] for tire dataset preprocessing'),
]

def main():
    A = ARGS(config=config)
    
    A.set_and_parse_args('')

    args = A.args
    print('args:', json.dumps(A.info, indent=4))
    
    os.environ['TORCH_HOME'] = args['root_path'] # '/host/ubuntu/torch'
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # smi = NVIDIA_SMI(device_id=0)
    
    _image_channels = 3
    if args['dataset'] in Datasets.available_datasets:
        # _transform_resize = []
        # if args['image_size'] > 0:
        #     _transform_resize = [transforms.Resize(args['image_size'],InterpolationMode.BICUBIC)]
        ds = Datasets(
            dataset=args['dataset'],
            image_size=args['image_size'],
            root_path=args['root_path'],
            batchsize=args['bs'],
            # transform_pre=[],
            download=True,
            splits=['train', 'test'],
            shuffle=True,
            num_workers=4,
            limit_train=args['limit_train'],
            limit_test=args['limit_test'],
        )
    elif args['dataset'] == 'tire':
        if args['tire_settings'] == 0:
            lbp_methods = ['r', 'g', 'b', 'default', 'uniform', 'ror', 'nri_uniform']
        elif args['tire_settings'] == 1:
            lbp_methods = ['l', 'default', 'uniform']
        
        _image_channels = len(lbp_methods)
        assert args['image_size'] > 0, 'must provide arg `image_size` of int >0'
        ds = get_tire_dataset(
            data_path=args['data_path'],
            batchsize=args['bs'],
            shuffle=True,
            num_workers=16,
            train_ratio=0.8,
            limit=0,
            image_size=args['image_size'],
            # force_reload=False,
            lbp={'radius': 1, 'point_mult': 8, 'methods': lbp_methods},
            random_crop=bool(args['aug_random_crop']),
            color_jitter=bool(args['aug_color_jitter']),
            autoaugment_imagenet=bool(args['aug_auto_imagenet']),
        )
    else:
        raise ValueError('arg `dataset` [{}] is not supported!'.format(args['dataset']))
    print('dataset: {}'.format(json.dumps(ds.info, indent=4)))
    
    if args['lineareval']:
        _model_backbone = VisionModelZoo.get_model(
            args['arch'],
            pretrained=args['pretrained'],
            image_channels=_image_channels,
            classifier=None,
            root_path=args['root_path']
        )
        _input_shape = [1, _image_channels, args['image_size'], args['image_size']]
        _output_dim = VisionModelZoo.get_output_shape(_model_backbone, _input_shape, args['device'])[-1]
        
        frozen_model_bottom = [_model_backbone]
        model = VisionModelZoo.get_classifier_head(
            _output_dim,
            [*args['fc'], ds.num_labels],
            # GELU(),
        )
    else:
        frozen_model_bottom = []
        model = VisionModelZoo.get_model(
            args['arch'],
            pretrained=args['pretrained'],
            image_channels=_image_channels,
            classifier=[*args['fc'], ds.num_labels],
        )
    
    net = Network(
        model=model,
        frozen_model_bottom=frozen_model_bottom,
        # frozen_model_top=[],
        opt='sgd',
        loss_fn=nn.CrossEntropyLoss(),
        lr=args['lr'],
        # lr_schedule_type='step',
        lr_schedule_step=args['lr_schedule_half'],
        lr_schedule_gamma=0.5,
        # pretrained=False,
        device=args['device'],
        # metrics_best=None,
    )
    
    run_mode = 'finetune'
    if args.get('lineareval'):
        run_mode = 'lineareval'

    time_start = time.time()
    stats = {
        'info': {**args},
        'telem': {
            'hardware': '1x3090',
            # 'gpu_total_memory': float(smi.info['total']),
            'sample_count_train': ds.info['sample_count']['train'],
            'sample_count_val': ds.info['sample_count']['test'],
            'completed': False,
            'time_stamp': time_stamp,
            'time_start': time_start,
            'time_finish': None,
            'time_elapsed': None,
            'mode': run_mode,
        },
        **{_split: [] for _split in ['train', 'val']},
    }
    net.fit(
        dataloader_train=ds.loaders['train'],
        dataloader_val=ds.loaders['test'],
        epochs=args['epoch'],
        # epoch_start=0,
        fp_json_master=args['master_stats_json'],
        time_stamp=time_stamp,
        stats=stats,
    )

    
# %%
if __name__ == '__main__':
    main()