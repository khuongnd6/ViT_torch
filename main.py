# %%
import torch
from torch.nn.modules.activation import GELU
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import time, os, json, string, re, datetime
import numpy as np

from utils_progressbar import *
from utils_args import ARGS
from utils_smi import NVIDIA_SMI
from utils_datasets import *

# %%
import argparse

time_start_master = time.time()
time_stamp = time.strftime('%y%m%d_%H%M%S')

# %%
config = [
    ('device',  'cuda',  str,  ['cuda', 'cpu']),
    ('epoch',  100,  int, None, 'number of training epochs'),
    ('dataset', 'cifar10', str, ['cifar10', 'cifar100', 'stl10'], 'name of the dataset'),
    ('bs', 512, int, None, 'batch size'),
    ('data_path', '/host/ubuntu/torch', str, None, 'path of the folder to put the pretrained models and datasets'),
    ('arch', 'dino_vits16', str, ['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8', 'dino_resnet50']),
    ('lr', 0.001, float, None,'initial learning rate'),
    ('lr_schedule_half', 5, int, None, 'number of epochs between halving the learning rate'),
    ('limit_train', 0, int, None, 'set to int >0 to limit the number of training examples'),
    ('limit_test', 0, int, None, 'set to int >0 to limit the number of testing examples'),
    ('stats_json', './logs/stats/stats_{}.json'.format(time_stamp), str),
    ('master_stats_json', './logs/stats_master.json'.format(time_stamp), str),
    ('test_only', False, bool),
    ('train_only', False, bool),
    ('lineareval', False, bool),
]
A = ARGS(config=config)

A.set_and_parse_args('')

args = A.args
print('args:', json.dumps(A.info, indent=4))

# %%
os.environ['TORCH_HOME'] = args['data_path']
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
smi = NVIDIA_SMI(device_id=0)

# %%
ds = Datasets(
    dataset=args['dataset'],
    root_path=args['data_path'],
    batchsize=args['bs'],
    transform=[],
    download=True,
    splits=['train', 'test'],
    shuffle=True,
    num_workers=4,
    limit_train=args['limit_train'],
    limit_test=args['limit_test'],
)
print('dataset: {}'.format(json.dumps(ds.info, indent=4)))


# %% DISABLED DUE TO CUDA COMPATIBILITY BUG
# class LinearClassifier(nn.Module):
#     """Linear layer to train on top of frozen features"""
#     def __init__(self, dims=[384], num_labels=1000, act=nn.GELU):
#         super(LinearClassifier, self).__init__()
#         self.num_labels = num_labels
        
#         self._act = act
        
#         self.linears = []
#         for i in range(len(dims)):
#             _linear = nn.Linear(dims[i], num_labels if i >= len(dims) - 1 else dims[i + 1])
#             _linear.weight.data.normal_(mean=0.0, std=0.01)
#             _linear.bias.data.zero_()
#             self.linears.append(_linear)

#     def forward(self, x):
#         # flatten
#         x = x.view(x.size(0), -1)

#         # linear layer
#         for i, _linear in enumerate(self.linears):
#             x = _linear(x)
#             if i < len(self.linears) - 1:
#                 x = self._act(x)
#         return x

# fc_units = datasets[args['dataset']]['fc']
# fc_units = [model_backbone.num_features, *fc_units]

# model_classifier = LinearClassifier(
#     dims=[model_backbone.num_features, *datasets[args['dataset']]['fc']],
#     num_labels=datasets[args['dataset']]['class_count'],
# )
# model_classifier.to(args['device'])

# %% MODEL
model_backbone = torch.hub.load('facebookresearch/dino:main', args['arch'])
model_backbone.to(args['device'])

_fc_units = {
    'stl10': [128],
    # 'cifar10': [128],
    'cifar10': [],
    'cifar100': [],
    # 'imagenet': [],
}

units = [
    model_backbone.num_features,
    *_fc_units.get(args['dataset'], []),
    ds.num_labels,
]
linear_layers = [
    _layer
    for i, u0, u1 in zip(range(len(units)), units[:-1], units[1:])
    for _layer in [
        nn.Linear(u0, u1, bias=True, device=args['device'])
    ] + ([] if i >= len(units) - 2 else [nn.GELU()])
]

model = nn.Sequential(
    model_backbone,
    # model_classifier,
    *linear_layers,
)
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9)
lr_scheduler = StepLR(optimizer, step_size=int(args['lr_schedule_half']), gamma=0.5)


# %% RUN FUNCTIONS
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
                    _stats_master = {
                        **_stats_old,
                        **_stats_master,
                    }
            except:
                pass
        _dp = os.path.split(fp_master)[0]
        if not os.path.isdir(_dp):
            os.makedirs(_dp)
        _ = json.dump(_stats_master, open(fp_master, 'w'), indent=4)

def count_correct(outputs, labels):
    outputs_class = torch.argmax(outputs, dim=-1)
    correct = (outputs_class == labels)
    return correct.sum().item()

def train_single_epoch(model, epoch, dataloader, frozen_backbone=None, debug=False, **kwargs):
    losses = []
    _loss_avg = -1.
    peak_vram_gb = smi.get_vram_used()
    
    batch_count = len(dataloader)
    time_start = time.time()
    correct_count = 0
    sample_count = 0
    acc_percent = 0.0
    time_ett = 1.0
    _outputs = []
    _labels = []
    
    _pb = ProgressBar(
        'Train[{}] step[{}][{}] loss[{}] acc[{}] vram[{}] time[{}{}]   ',
        ProgressBarTextMultiInt([0, args['epoch']], '{}/{}', 1, '0'), # epoch
        ProgressBarTextMultiInt([0, batch_count], '{}/{}', 1, ' '), # step
        ProgressBarFloat(0.0, '{:5.1f}%'), # progress percent
        ProgressBarFloat(.0, '{:.6f}'), # loss
        ProgressBarFloat(.0, '{:6.2f}%'), # acc
        ProgressBarFloat(.0, '{:4.1f}GB'), # vram
        ProgressBarTime(.0, '{:.1f}'), # time elapsed
        ProgressBarTime(.0, '/{:.1f}'), # time total estimated
    )
    _pb(epoch + 1, 0, 0., _loss_avg, 0.0, peak_vram_gb, 0.0, time_ett)
    
    model.train()
    # set frozen_backbone to eval mode to freeze it
    if frozen_backbone is not None:
        frozen_backbone.eval()
    
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(args['device'])
        labels = labels.to(args['device'])
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        if debug:
            _outputs.append(outputs.cpu().detach().numpy())
            _labels.append(labels.cpu().detach().numpy())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _correct = count_correct(outputs, labels)
        correct_count += _correct
        sample_count += outputs.size(0)
        acc_percent = correct_count / max(sample_count, 1) * 100
        
        losses.append(loss.item())
        _loss_avg = sum(losses) / len(losses)
        
        if i % 5 == 0:
            peak_vram_gb = max(peak_vram_gb, smi.get_vram_used())
        
        _progress_percent = (i + 1) / batch_count * 100
        
        time_elapsed = time.time() - time_start
        time_ett = time_elapsed / max(0.000001, _progress_percent / 100)
        time_eta = time_ett * (1 - _progress_percent / 100)
        
        _pb(epoch + 1, i + 1, _progress_percent, _loss_avg, acc_percent, peak_vram_gb, time_elapsed, time_ett)
    
    time_current = time.time()
    time_elapsed = time_current - time_start
    peak_vram_gb = max(peak_vram_gb, smi.get_vram_used())
    _pb(epoch + 1, batch_count, 100., _loss_avg, acc_percent, peak_vram_gb, time_elapsed, None)
    _stat = {
        'epoch': epoch + 1,
        'time_start': time_start,
        'time_finish': time_current,
        'time_cost': time_elapsed,
        'loss': float(_loss_avg),
        'acc': float(acc_percent / 100),
        'vram': peak_vram_gb,
    }
    if debug:
        _dp = './logs/tmp'
        _outputs = np.array([v1 for v in _outputs for v1 in v], np.float32)
        _labels = np.array([v1 for v in _labels for v1 in v], np.int64)
        if not os.path.isdir(_dp):
            os.makedirs(_dp)
        np.save(open(os.path.join(_dp, 'outputs'), 'wb'), _outputs)
        np.save(open(os.path.join(_dp, 'labels'), 'wb'), _labels)
    return _stat

@torch.no_grad()
def validate_single_epoch(model, epoch, dataloader, debug=False, **kwargs):
    losses = []
    _loss_avg = -1.
    peak_vram_gb = smi.get_vram_used()
    
    batch_count = len(dataloader)
    time_start = time.time()
    correct_count = 0
    sample_count = 0
    acc_percent = 0.0
    time_ett = 1.0
    _outputs = []
    _labels = []
    
    
    _pb = ProgressBar(
        '  Val[{}] step[{}][{}] loss[{}] acc[{}] vram[{}] time[{}{}]   ',
        ProgressBarTextMultiInt([0, args['epoch']], '{}/{}', 1, '0'), # epoch
        ProgressBarTextMultiInt([0, batch_count], '{}/{}', 1, ' '), # step
        ProgressBarFloat(0.0, '{:5.1f}%'), # progress percent
        ProgressBarFloat(.0, '{:.6f}'), # loss
        ProgressBarFloat(.0, '{:6.2f}%'), # acc
        ProgressBarFloat(.0, '{:4.1f}GB'), # vram
        ProgressBarTime(.0, '{:.1f}'), # time elapsed
        ProgressBarTime(.0, '/{:.1f}'), # time total estimated
    )
    _pb(epoch + 1, 0, 0., _loss_avg, 0.0, peak_vram_gb, 0.0, time_ett)
    
    model.eval()
    
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(args['device'])
        labels = labels.to(args['device'])
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        if debug:
            _outputs.append(outputs.cpu().detach().numpy())
            _labels.append(labels.cpu().detach().numpy())
        
        _correct = count_correct(outputs, labels)
        correct_count += _correct
        sample_count += outputs.size(0)
        acc_percent = correct_count / max(sample_count, 1) * 100
        
        losses.append(loss.item())
        _loss_avg = sum(losses) / len(losses)
        
        if i % 5 == 0:
            peak_vram_gb = max(peak_vram_gb, smi.get_vram_used())
        
        _progress_percent = (i + 1) / batch_count * 100
        
        time_elapsed = time.time() - time_start
        time_ett = time_elapsed / max(0.000001, _progress_percent / 100)
        time_eta = time_ett * (1 - _progress_percent / 100)
        
        _pb(epoch + 1, i + 1, _progress_percent, _loss_avg, acc_percent, peak_vram_gb, time_elapsed, time_ett)
    
    time_current = time.time()
    time_elapsed = time_current - time_start
    peak_vram_gb = max(peak_vram_gb, smi.get_vram_used())
    _pb(epoch + 1, batch_count, 100., _loss_avg, acc_percent, peak_vram_gb, time_elapsed, None)
    _stat = {
        'epoch': epoch + 1,
        'time_start': time_start,
        'time_finish': time_current,
        'time_cost': time_elapsed,
        'loss': float(_loss_avg),
        'acc': float(acc_percent / 100),
        'vram': peak_vram_gb,
    }
    if debug:
        _dp = './logs/tmp'
        _outputs = np.array([v1 for v in _outputs for v1 in v], np.float32)
        _labels = np.array([v1 for v in _labels for v1 in v], np.int64)
        if not os.path.isdir(_dp):
            os.makedirs(_dp)
        np.save(open(os.path.join(_dp, 'outputs'), 'wb'), _outputs)
        np.save(open(os.path.join(_dp, 'labels'), 'wb'), _labels)
    return _stat


# %% RUN
def main():
    time_start = time.time()
    assert not (args['train_only'] and args['test_only']), 'at max only one of flags `train_only` and `test_only` allowed'

    run_splits = ['train', 'val']
    if args['train_only']:
        run_splits = ['train']
    if args['test_only']:
        run_splits = ['val']

    run_mode = 'finetune'
    if args.get('lineareval'):
        run_mode = 'lineareval'

    stats = {
        'info': A.info,
        'telem': {
            'hardware': '1x3090',
            'gpu_total_memory': float(smi.info['total']),
            'sample_count_train': ds.info['sample_count']['train'],
            'sample_count_val': ds.info['sample_count']['test'],
            'completed': False,
            'time_stamp': time_stamp,
            'time_start': time_start,
            'time_finish': None,
            'time_elapsed': None,
            'mode': run_mode,
        },
        **{_mode: [] for _mode in run_splits},
    }
    save_stats(stats)

    for _epoch in range(args['epoch']):
        for _mode in run_splits:
            if _mode == 'train':
                _stat = train_single_epoch(
                    model=model,
                    epoch=_epoch,
                    dataloader=ds.loaders['train'],
                    frozen_backbone=model_backbone if run_mode == 'lineareval' else None,
                    debug=False,
                )
                lr_scheduler.step()
            elif _mode == 'val':
                _stat = validate_single_epoch(
                    model=model,
                    epoch=_epoch,
                    dataloader=ds.loaders['test'],
                    debug=False,
                )
            else:
                continue
            stats[_mode].append(_stat)
            print()
        
        print()
        save_stats(stats)
        

    print()
    print('Finished Training')

    time_current = time.time()
    stats['telem']['time_finish'] = time_current
    stats['telem']['time_elapsed'] = time_current - time_start
    stats['telem']['completed'] = True
    save_stats(stats)

    print('Stats saved at <{}> and <{}>'.format(args['stats_json'], args['master_stats_json']))

# %% clean up
if __name__ == '__main__':
    main()
    smi.dispose()

# %%
