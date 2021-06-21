# %%
import torch
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
# %%
config = [
    # {'key': 'device', 'def': 'cuda', 'set': ['cuda', 'cpu'], 'type': str},
    ('device',  'cuda',  str,  ['cuda', 'cpu']),
    ('epoch',  10,  int),
    ('dataset', 'cifar10', str, ['cifar10', 'cifar100']),
    ('bs', 512, int),
    ('data_path', '/host/ubuntu/torch'),
    ('arch', 'dino_vits16', str, ['dino_vits16']),
    ('lr', 0.001),
    ('stats_json', './logs/stats_latest.json', str)
]
A = ARGS(config=config)

A.set_and_parse_args('')

args = A.args
print('args:', json.dumps(A.info, indent=4))

# %%

import subprocess as sp
import os

def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print(memory_free_values)
    return memory_free_values


# parser = argparse.ArgumentParser('Vision_Transformer_torch')
# for k, v in def_args.items():
#     _keys = [k]
#     if isinstance(k, tuple):
#         _keys = list(k)
#     parser.add_argument(*['--{}'.format(_key) for _key in _keys], default=v, type=type(v),)

# args = {**args, **parser.parse_args().__dict__}

# %%
# print('args:', json.dumps(args, indent=4))
# assert args['device'] in ['cuda', 'cpu']


# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset_fns = {
    'cifar10': torchvision.datasets.CIFAR10,
}

assert args['dataset'] in dataset_fns
_dataset_fn = dataset_fns[args['dataset']]

trainset = _dataset_fn(root=args['data_path'], train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['bs'],
                                          shuffle=True, num_workers=2)

testset = _dataset_fn(root=args['data_path'], train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args['bs'],
                                         shuffle=False, num_workers=2)

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

model = nn.Sequential(
    model_backbone,
    nn.Linear(model_backbone.num_features, 64, bias=True, device=args['device']),
    nn.GELU(),
    nn.Linear(64, 10, bias=False, device=args['device']),
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
    return correct.item().sum()
    return correct

def run_single_epoch(epoch, training=True, key='train', _pb=None):
    losses = []
    _loss_avg = -1.
    vram_usage = get_gpu_memory()
    
    time_start = time.time()
    correct_count = 0
    sample_count = 0
    acc_percent = 0.0
    if _pb is None:
        _pb = lambda *args: None
    _pb(0., epoch, 0, _loss_avg, 0.0, vram_usage)
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        
        inputs, labels = data
        inputs = inputs.to(args['device'])
        labels = labels.to(args['device'])
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _correct = count_correct(outputs, labels)
        correct_count += _correct
        sample_count += outputs.size(0)
        acc_percent = correct_count / max(sample_count, 1) * 100
        
        losses.append(loss.item())
        _loss_avg = sum(losses) / len(losses)
        
        _progress_percent = (epoch + (i + 1) / batch_count) / args['epoch'] * 100
        if i % 10 == 0:
            vram_usage = get_gpu_memory()
        
        time_elapsed = time.time() - time_start
        time_eta = time_elapsed / max(0.000001, _progress_percent) * (100 - _progress_percent)
        
        _pb(_progress_percent, epoch, i + 1, _loss_avg, acc_percent, vram_usage, time_elapsed, time_eta)
        
    _pb(100., epoch + 1, batch_count, _loss_avg, acc_percent, vram_usage, time_elapsed, time_eta)
    
    vram_usage = get_gpu_memory()
    time_current = time.time()
    time_elapsed = time_current - time_start
    _stat = {
        'epoch': epoch + 1,
        'time_start': time_start,
        'time_finish': time_current,
        'time_cost': time_elapsed,
        'loss': float(_loss_avg),
        'acc': float(acc_percent / 100),
        'vram': vram_usage
    }
    return _stat

# %%
batch_count = len(trainloader)
batch_count_test = len(testloader)

pb_percent = ProgressBarFloat(0.0, '{:5.1f}%')
pb_epoch = ProgressBarTextMultiInt([0, args['epoch']], 1, '0')
pb_step = ProgressBarTextMultiInt([0, batch_count], 1, ' ')
pb_step_test = ProgressBarTextMultiInt([0, batch_count_test], 1, ' ')
pb_loss = ProgressBarFloat(.0, '{:.6f}')
pb_acc = ProgressBarFloat(.0, '{:6.2f}%')
pb_vram = ProgressBarFloat(.0, '{:8.1f}%')
pb_time_elp = ProgressBarTime(.0, '{:.1f}')
pb_time_eta = ProgressBarTime(.0, '{:.1f}')

pb = ProgressBar(
    '[Train] [{}] epoch[{}/{}] step[{}/{}] loss[{}] acc[{}] vram[{}MB] time[{}+{}]   ',
    pb_percent,
    pb_epoch,
    pb_step,
    pb_loss,
    pb_acc,
    pb_vram,
    pb_time_elp,
    pb_time_eta,
)
pb_test = ProgressBar(
    '[Test]  [{}] epoch[{}/{}] step[{}/{}] loss[{}] acc[{}] vram[{}MB]    ',
    pb_percent,
    pb_epoch,
    pb_step_test,
    pb_loss,
    pb_acc,
    pb_vram,
    pb_time_elp,
    pb_time_eta,
)

_loss_avg = -1.
stats = {
    'info': A.info,
    'train': [],
    'val': [],
}
for epoch in range(args['epoch']):
    for _training in [True, False]:
        _key = 'train' if _training else 'val'
        _pb = pb if _training else pb_test
        _stat = run_single_epoch(
            epoch,
            training=_training,
            key=_key,
            _pb=_pb,
        )
        stats[_key].append
    
    
    
    continue
    
    losses = []
    vram_usage = get_gpu_memory()
    
    time_start = time.time()
    correct_count = 0
    sample_count = 0
    acc_percent = 0.0
    pb(0., epoch, 0, _loss_avg, 0.0, vram_usage)
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        
        inputs, labels = data
        inputs = inputs.to(args['device'])
        labels = labels.to(args['device'])
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _correct = count_correct(outputs, labels)
        correct_count += _correct
        sample_count += outputs.size(0)
        acc_percent = correct_count / max(sample_count, 1) * 100
        
        losses.append(loss.item())
        _loss_avg = sum(losses) / len(losses)
        
        _progress_percent = (epoch + (i + 1) / batch_count) / args['epoch'] * 100
        if i % 10 == 0:
            vram_usage = get_gpu_memory()
        
        time_elapsed = time.time() - time_start
        time_eta = time_elapsed / max(0.000001, _progress_percent) * (100 - _progress_percent)
        
        pb(_progress_percent, epoch, i + 1, _loss_avg, acc_percent, vram_usage, time_elapsed, time_eta)
        
    pb(100., epoch + 1, batch_count, _loss_avg, acc_percent, vram_usage, time_elapsed, time_eta)
    stats['train'].append({
        'epoch': epoch + 1,
        'time_start': time_start,
        'time_finish': time.time(),
        'loss': float(_loss_avg),
        'acc': float(acc_percent / 100),
    })
    print()
    
    
    time_start = time.time()
    correct_count = 0
    sample_count = 0
    acc_percent = 0.0
    pb_test(0., epoch, 0, _loss_avg, 0.0)
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        
        inputs, labels = data
        inputs = inputs.to(args['device'])
        labels = labels.to(args['device'])
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        _correct = count_correct(outputs, labels)
        correct_count += _correct
        sample_count += outputs.size(0)
        acc_percent = correct_count / max(sample_count, 1) * 100
        
        losses.append(loss.item())
        _loss_avg = sum(losses) / len(losses)
        
        _progress_percent = (epoch + (i + 1) / batch_count) / args['epoch'] * 100
        pb_test(_progress_percent, epoch, i + 1, _loss_avg, acc_percent)
        
    pb_test(100., epoch + 1, batch_count, _loss_avg, acc_percent)
    
    stats['val'].append({
        'epoch': epoch + 1,
        'time_start': time_start,
        'time_finish': time.time(),
        'loss': float(_loss_avg),
        'acc': float(acc_percent / 100),
    })
    print()
    print()
    
    if isinstance(args['stats_json'], str) and args['stats_json'].endwith('.json'):
        _dp = os.path.split(args['stats_json'])[0]
        if not os.path.isdir(_dp):
            os.makedirs(_dp)
        _ = json.dump(stats, open(args['stats_json'], 'w'), indent=4)
    
    # print('something something something')

print()
print('Finished Training')


# %%