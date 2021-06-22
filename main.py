# %%
import torch
from torch.nn.modules.activation import GELU
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import time, os, json, string, re, datetime
import numpy as np

from utils_progressbar import *
from utils_args import ARGS

# %%
import argparse

# %%
import nvidia_smi

nvidia_smi.nvmlInit()

# %%
# def_args = {
#     'device': 'cuda',
#     'epoch': 10,
#     'dataset': 'cifar10',
#     ('batchsize', 'bs'): 512,
#     'data_path': '/host/ubuntu/torch',
#     'arch': 'dino_vits16',
#     ('learningrate', 'lr'): 0.001,
# }
# args = {
#     _key: v
#     for k, v in def_args.items()
#     for _key in ([k] if isinstance(k, str) else k)
# }
# args = object()
# args.blink = 'blah'
# args, args.blink


# %%
time_start_master = time.time()
time_stamp = time.strftime('%y%m%d_%H%M%S')

# %%
config = [
    # {'key': 'device', 'def': 'cuda', 'set': ['cuda', 'cpu'], 'type': str},
    ('device',  'cuda',  str,  ['cuda', 'cpu']),
    ('epoch',  10,  int),
    ('dataset', 'cifar10', str, ['cifar10', 'cifar100', 'stl10']),
    ('bs', 512, int),
    ('data_path', '/host/ubuntu/torch'),
    ('arch', 'dino_vits16', str, ['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8', 'dino_resnet50']),
    ('lr', 0.001),
    ('stats_json', './logs/stats/stats_{}.json'.format(time_stamp), str),
    ('master_stats_json', './logs/stats_master.json'.format(time_stamp), str),
    ('train', True, bool),
    ('test', True, bool),
]
A = ARGS(config=config)

A.set_and_parse_args('')

args = A.args
print('args:', json.dumps(A.info, indent=4))

# %%
nvml_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

def get_nvml_info(gpu_id=0):
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(nvml_handle)
    r = {
        'total': info.total / 1024**3,
        'used': info.used / 1024**3,
        'free': info.free / 1024**3,
        'usage': info.used / info.total,
    }
    return r



# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

fn_args_base = {
    'root': args['data_path'],
    'download': True,
    'transform': transform,
}

datasets = {
    'cifar10': {
        'fn': torchvision.datasets.CIFAR10,
        'fn_args': fn_args_base,
        'fn_args_train': {
            'train': True,
        },
        'fn_args_test': {
            'train': False,
        },
        'fc': [64, 10],
    },
    'cifar100': {
        'fn': torchvision.datasets.CIFAR100,
        'fn_args': fn_args_base,
        'fn_args_train': {
            'train': True,
        },
        'fn_args_test': {
            'train': False,
        },
        'fc': [256, 100],
    },
    'stl10': {
        'fn': torchvision.datasets.STL10,
        'fn_args': fn_args_base,
        'fn_args_train': {
            'split': 'train',
        },
        'fn_args_test': {
            'split': 'test',
        },
        'fc': [64, 10],
    },
}

assert args['dataset'] in datasets
ds_config = datasets[args['dataset']]
_dataset_fn = ds_config['fn']

trainset = _dataset_fn(
    **ds_config['fn_args'],
    **ds_config['fn_args_train']
)
testset = _dataset_fn(
    **ds_config['fn_args'],
    **ds_config['fn_args_test']
)
print('dataset length: {} + {}'.format(len(trainset), len(testset)))
# trainset = _dataset_fn(
#     root=args['data_path'],
#     train=True,
#     download=True,
#     transform=transform,
# )
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args['bs'],
    shuffle=True,
    num_workers=2,
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args['bs'],
    shuffle=False,
    num_workers=2,
)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %%
# archs = [
#     'dino_vits16'
# ]
# assert args['arch'] in archs, '--arch must be one of [{}]'.format('|'.join(archs))

# if not torch.cuda.is_available():
#     args['device'] = 'cpu'

model_backbone = torch.hub.load('facebookresearch/dino:main', args['arch'])
model_backbone.to(args['device'])


fc_units = datasets[args['dataset']]['fc']
fc_units = [model_backbone.num_features, *fc_units]

fc_layers = [
    _layer
    for i in range(len(fc_units) - 1)
    for _layer in [
        nn.Linear(fc_units[i], fc_units[i + 1], bias=True, device=args['device'])
    ] + ([] if i == len(fc_units) - 1 else [nn.GELU()])
]


model = nn.Sequential(
    model_backbone,
    *fc_layers,
    # nn.Linear(model_backbone.num_features, 64, bias=True, device=args['device']),
    # nn.GELU(),
    # nn.Linear(64, 10, bias=False, device=args['device']),
    # nn.Softmax(1),
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9)

# %%
def count_correct(outputs, labels):
    # print(outputs.shape)
    # print(labels.size())
    outputs_class = torch.argmax(outputs, dim=-1)
    correct = (outputs_class == labels)
    return correct.sum().item()
    return correct

def run_single_epoch(epoch, dataloader, training=True, key='train', _pb=None):
    losses = []
    _loss_avg = -1.
    gpu_info = get_nvml_info(0)
    peak_vram_gb = gpu_info['used']
    
    batch_count = len(dataloader)
    time_start = time.time()
    correct_count = 0
    sample_count = 0
    acc_percent = 0.0
    time_ett = 1.0
    if _pb is None:
        _pb = lambda *args: None
    _pb(epoch, 0, 0., _loss_avg, 0.0, gpu_info['used'], 0.0, time_ett)
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        
        inputs, labels = data
        inputs = inputs.to(args['device'])
        labels = labels.to(args['device'])
        
        if training:
            # zero the parameter gradients
            optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        if training:
            loss.backward()
            optimizer.step()
        
        _correct = count_correct(outputs, labels)
        correct_count += _correct
        sample_count += outputs.size(0)
        acc_percent = correct_count / max(sample_count, 1) * 100
        
        losses.append(loss.item())
        _loss_avg = sum(losses) / len(losses)
        
        if i % 2 == 0:
            gpu_info = get_nvml_info(0)
            peak_vram_gb = max(peak_vram_gb, gpu_info['used'])
        
        _progress_percent = (i + 1) / batch_count * 100
        
        time_elapsed = time.time() - time_start
        time_ett = time_elapsed / max(0.000001, _progress_percent / 100)
        time_eta = time_ett * (1 - _progress_percent / 100)
        
        _pb(epoch, i + 1, _progress_percent, _loss_avg, acc_percent, peak_vram_gb, time_elapsed, time_ett)
    
    
    time_current = time.time()
    time_elapsed = time_current - time_start
    gpu_info = get_nvml_info(0)
    peak_vram_gb = max(peak_vram_gb, gpu_info['used'])
    _pb(epoch + 1, batch_count, 100., _loss_avg, acc_percent, peak_vram_gb, time_elapsed, time_elapsed)
    _stat = {
        'epoch': epoch + 1,
        'time_start': time_start,
        'time_finish': time_current,
        'time_cost': time_elapsed,
        'loss': float(_loss_avg),
        'acc': float(acc_percent / 100),
        'vram': peak_vram_gb,
    }
    return _stat

# %%
def save_stats(_stats={}):
    fp = args['stats_json']
    fp_master = args['master_stats_json']
    
    if isinstance(fp, str) and fp.endswith('.json'):
        _dp = os.path.split(fp)[0]
        if not os.path.isdir(_dp):
            os.makedirs(_dp)
        _ = json.dump(_stats, open(fp, 'w'), indent=4)
    
    if isinstance(fp_master, str) and fp_master.endswith('.json'):
        _stats_master = {
            time_stamp: _stats,
        }
        if os.path.isfile(fp_master):
            try:
                _stats_old = json.load(open(fp_master, 'r'))
                if isinstance(_stats_old, dict):
                    # _stats_old[time_stamp] = _stats
                    _stats = {
                        **_stats_old,
                        **_stats_master,
                    }
            except:
                pass
        _dp = os.path.split(fp_master)[0]
        if not os.path.isdir(_dp):
            os.makedirs(_dp)
        _ = json.dump(_stats_master, open(fp_master, 'w'), indent=4)
    


# %%
batch_count = len(trainloader)
batch_count_test = len(testloader)

pb_percent = ProgressBarFloat(0.0, '{:5.1f}%')
pb_epoch = ProgressBarTextMultiInt([0, args['epoch']], 1, '0')
pb_step = ProgressBarTextMultiInt([0, batch_count], 1, ' ')
pb_step_test = ProgressBarTextMultiInt([0, batch_count_test], 1, ' ')
pb_loss = ProgressBarFloat(.0, '{:.6f}')
pb_acc = ProgressBarFloat(.0, '{:6.2f}%')
pb_vram = ProgressBarFloat(.0, '{:4.1f}GB')
pb_time_elp = ProgressBarTime(.0, '{:.1f}')
pb_time_ett = ProgressBarTime(.0, '{:.1f}')
pb_time_eta = ProgressBarTime(.0, '{:.1f}')

progress_str = '[{}/{}] step[{}/{}][{}] loss[{}] acc[{}] vram[{}] time[{}/{}]   '
pb = ProgressBar(
    'Train' + progress_str,
    pb_epoch,
    pb_step,
    pb_percent,
    pb_loss,
    pb_acc,
    pb_vram,
    pb_time_elp,
    pb_time_ett,
    # pb_time_eta,
)
pb_test = ProgressBar(
    ' Test' + progress_str,
    pb_epoch,
    pb_step_test,
    pb_percent,
    pb_loss,
    pb_acc,
    pb_vram,
    pb_time_elp,
    pb_time_ett,
    pb_time_eta,
)

_loss_avg = -1.
stats = {
    'info': A.info,
    'telem': {
        'hardware': '1x3090',
        'gpu_total_memory': float(get_nvml_info()['total']),
        'sample_count_train': len(trainset),
        'sample_count_val': len(testset),
        'completed': False,
        'time_stamp': time_stamp,
        'time_start': time_start_master,
        'time_finish': None,
        'time_elapsed': None,
    },
    'train': [],
    'val': [],
}
save_stats(stats)

assert args['train'] or args['test'], 'at least one of args `train` and `test` must be True'

_training_bool_values = []
if args['train']:
    _training_bool_values.append(True)
if args['test']:
    _training_bool_values.append(False)

for _epoch in range(args['epoch']):
    for _training in _training_bool_values:
        _key = 'train' if _training else 'val'
        _pb = pb if _training else pb_test
        _dataloader = trainloader if _training else testloader
        _stat = run_single_epoch(
            epoch=_epoch,
            dataloader=_dataloader,
            training=_training,
            key=_key,
            _pb=_pb,
        )
        stats[_key].append(_stat)
        print()
    
    save_stats(stats)
    
    continue
    

print()
print('Finished Training')

time_current = time.time()
stats['telem']['time_finish'] = time_current
stats['telem']['time_elapsed'] = time_current - time_start_master
stats['telem']['completed'] = True
save_stats(stats)

print('Stats saved at <{}> and <{}>'.format(args['stats_json'], args['master_stats_json']))

# %% clean up
nvidia_smi.nvmlShutdown()

# %%