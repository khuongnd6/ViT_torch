# %%
from types import ClassMethodDescriptorType
from numpy import core
import torch
from torch._C import Value
from torch.autograd.grad_mode import no_grad
from torch.jit import Error
import torch.optim as optim
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.modules.activation import GELU
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.transforms as transforms

# from functools import partial
# from xcit import XCiT
from adabelief_pytorch import AdaBelief

# %%
import numpy as np
# import scipy
from PIL import Image

import time, os, json, re, string, math, random, datetime
import sys

# %%
class TimerLog:
    def __init__(self, format='time[{elapsed:.1f}/{total:.1f}{unit}]'):
        self.format = format
        self.time_start = time.time()
        self.progress = 0.000000001
        self.update()
    
    def __str__(self) -> str:
        self.update()
        secs_limit = max(self.time_elapsed, self.time_total)
        _unit = self.format_time(secs_limit)[-1]
        values = {
            'start': self.time_start,
            'current': self.time_current,
            'elapsed': self.format_time(self.time_elapsed, secs_limit)[0],
            'total': self.format_time(self.time_total, secs_limit)[0],
            'remain': self.format_time(self.time_remain, secs_limit)[0],
            'unit': _unit,
        }
        return self.format.format(**{
            k: v
            for k, v in values.items()
        })
    
    def start(self):
        self.time_start = time.time()
        self.update()
        
    def update(self, progress=None):
        self.time_current = time.time()
        self.time_elapsed = self.time_current - self.time_start
        if progress:
            self.progress = float(np.clip(progress, 0.000000001, 1.0))
        # self.progress
        self.time_total = self.time_elapsed / self.progress
        self.time_remain = self.time_total - self.time_elapsed
    
    @classmethod
    def format_time(self, secs=0, secs_limit=0):
        units = {
            'd': 864000,
            'h': 3600,
            'm': 60,
            's': 1,
        }
        for k, v in units.items():
            if k == 's' or secs >= v or secs_limit >= v:
                return [secs / v, k]

# %%
class Metrics:
    def __init__(self):
        pass
    
    def update(self):
        pass

class AccuracyMetrics(Metrics):
    def __init__(self, format='{percent:6.2f}%', fn=None, value=0.0, best_str='(best)', last_best=1.0):
        self.fn = fn
        self.format = format
        self.value = value
        self.percent = self.value * 100
        self.value_best = 0.0
        self.percent_best = self.value_best * 100
        self.last_is_best = False
        self.is_best = False
        self.corrects = []
        self.best_str = best_str
        self.last_best = last_best
        self.with_best = False
    
    def reset(self):
        self.corrects = []
    
    def update(self, corrects=None, with_best=None):
        if corrects is not None:
            if isinstance(corrects, list):
                self.corrects.extend([int(bool(v)) for v in corrects])
            elif isinstance(corrects, np.ndarray):
                self.corrects.extend([int(bool(v)) for v in corrects.reshape(-1)])
            else:
                raise ValueError()
                # self.corrects.append(int(bool(corrects)))
        if len(self.corrects) > 0:
            self.value = float(np.mean(self.corrects))
        else:
            self.value = 0.0
        self.percent = self.value * 100
        # self.last_is_best = False
        # if self.value > self.value_best:
        #     self.value_best = self.value
        #     self.percent_best = self.value_best * 100
        #     self.last_is_best = True
        self.is_best = self.value > self.last_best
        if with_best is not None:
            self.with_best = bool(with_best)
    
    def calc(self, *args, **kwargs):
        _corrects = self.fn(*args, **kwargs)
        self.update_count(_corrects)
        return None
    
    def __str__(self) -> str:
        return self.format.format(**{
            'percent': self.percent,
            'value': self.value,
            'value_best': self.value_best,
            'percent_best': self.percent_best,
            # 'best': self.best_str if self.last_is_best else '',
            'best': self.best_str if (self.is_best and self.with_best) else '',
        })
    

class AverageMetrics(Metrics):
    def __init__(self, format='{}', value=0.0, lower_is_better=False):
        self.format = format
        self.value = value
        self.percent = self.value * 100
        self.value_best = self.value
        self.percent_best = self.value_best * 100
        self.last_is_best = False
        self.lower_is_better = bool(lower_is_better)
        self.values = []
    
    def reset(self):
        self.values = []
    
    def update(self, values=None):
        if values is not None:
            if isinstance(values, list):
                self.values.extend([v for v in values])
            else:
                self.values.append(values)
        if len(self.values) > 0:
            self.value = float(np.mean(self.values))
        # else:
        #     self.value = 0.0
        self.percent = self.value * 100
        self.last_is_best = False
        _is_best = False
        if not self.lower_is_better and self.value > self.value_best:
            _is_best = True
        if self.lower_is_better and self.value < self.value_best:
            _is_best = True
        if _is_best:
            self.value_best = self.value
            self.percent_best = self.value_best * 100
            self.last_is_best = True
    
    def __str__(self) -> str:
        return self.format.format(**{
            'percent': self.percent,
            'value': self.value,
            'value_best': self.value_best,
            'percent_best': self.percent_best,
        })

class CounterLog:
    box_chars = ['█', '▉','▊','▋', '▌', '▍', '▎', '▏', ' '][::-1]
    def __init__(self,
                total=10,
                value=0,
                format='[{value}/{total}]',
                format_start=None,
                format_finish=None,
                bar_len=40,
                ):
        self.value = value
        self.total = total
        self.bar_len = bar_len
        self.format = format
        self.format_start = format
        self.format_finish = format
        if format_start:
            self.format_start = format_start
        if format_finish:
            self.format_finish = format_finish
    
    def update(self, value=None):
        if value:
            self.value = value
        self.progress = self.value / self.total
        self.progress_percent = self.progress * 100
        
        self.bar = self.get_box_string(self.progress, self.bar_len)
    
    def __str__(self) -> str:
        return self.format.format(**{
            'value': self.value,
            'total': self.total,
            'progress': self.progress,
            'progress_percent': self.progress_percent,
            'bar': self.bar,
        })
    
    @classmethod
    def get_box_string(cls, value=0.2, length=10):
        box_count = float(np.clip(value, 0.0, 1.0) * length)
        full_box_count = int(np.floor(box_count))
        partial_box_count = box_count - full_box_count
        partial_box_index = int(np.clip(
            np.round(partial_box_count * (len(cls.box_chars) - 1)),
            0,
            len(cls.box_chars) - 1,
        ))
        s = cls.box_chars[-1] * full_box_count + cls.box_chars[partial_box_index]
        s = s.ljust(length, ' ')[:length]
        return s


class ProgressLog:
    box_chars = ['█', '▉','▊','▋', '▌', '▍', '▎', '▏', ' '][::-1]
    
    def __init__(self, name='', epochs=None, steps=None, timer='step', counters={}, metrics={}, bar_len=20, **kwargs):
        self.fields = {}
        self.bar_len = int(max(2, bar_len))
        self.name = str(name)
        
        if isinstance(epochs, int) and epochs > 0:
            self.fields['epoch'] = CounterLog(
                total=epochs,
                value=0,
                format='epoch[{value}/{total}]',
            )
        if isinstance(steps, int) and steps > 0:
            self.fields['step'] = CounterLog(
                total=steps,
                value=0,
                format='step[{value}/{total}][{bar}]',
                # format_finish='step[{value}/{total}][{bar}]',
                bar_len=bar_len,
            )
        self.timer = timer
        if timer:
            self.fields['timer'] = TimerLog(
                format='time[{elapsed:.1f}/{total:.1f}{unit}]',
            )
        for k, v in kwargs.items():
            self.fields[k] = v
    
    def __str__(self):
        return self.get_str()
    
    def get_str(self):
        ss = [self.name]
        for k, v in self.fields.items():
            v.update()
            s = str(v)
            ss.append(s)
        return ' '.join(ss) + ' ' * 10
    
    def print(self, in_place=True):
        _str = self.get_str()
        if in_place:
            print('\r' + _str, end='')
        else:
            print(_str)
    
    def update(self, print_in_place=False, **kwargs):
        for k, v in kwargs.items():
            if k in self.fields:
                self.fields[k].update(v)
                # self.fields[k]['value'] = v
                # if self.fields[k].get('_type', None) == 'counter':
                #     self.fields[k]['ratio'] = self.fields[k]['value'] / self.fields[k]['total']
                #     self.fields[k]['percentage'] = self.fields[k]['ratio'] * 100
                #     self.fields[k]['bar'] = self.get_box_string(self.fields[k]['ratio'], self.bar_len)
        for k, v in self.fields.items():
            if isinstance(v, TimerLog):
                _progress = None
                if self.timer in self.fields:
                    _progress = self.fields[self.timer].progress
                v.update(progress=_progress)
        if print_in_place:
            self.print(True)
    
    # def time_start(self):
    #     self.time_start = time.time()
    
    # def time_update(self, progress=None):
    #     self.time_current = time.time()
    #     self.time_elapsed = self.time_current - self.time_start
    #     self.time_ett = 0.0
            
    
    # @classmethod
    # def get_box_string(cls, value=0.2, length=10):
    #     box_count = float(np.clip(value, 0.0, 1.0) * length)
    #     full_box_count = int(np.floor(box_count))
    #     partial_box_count = box_count - full_box_count
    #     partial_box_index = int(np.round(partial_box_count / (len(cls.box_chars) - 1)))
    #     s = cls.box_chars[-1] * full_box_count + cls.box_chars[partial_box_index]
    #     s = s.ljust(length, ' ')[:length]
    #     return s


# PL = ProgressLog(
#     epochs=21,
#     steps=49,
#     acc=AccuracyMetrics('acc[{percent:6.2f}%]'),
#     loss=AverageMetrics('loss[{value:.6f}]')
# )
# time.sleep(0.5)
# PL.update(
#     step=7,
#     epoch=3,
#     acc=[bool(v) for v in np.random.randint(2, size=[20])],
#     loss=[0.45, 0.1],
# )
# str(PL)

# %%
def classification_count_correct(outputs, labels):
    # "outputs have class dim at the last dim and labels are class indices"
    # if first_run_count_correct:
    #     print(outptus.shape, labels.shape)
    #     first_run_count_correct = False
    with torch.no_grad():
        outputs_class = torch.argmax(outputs, dim=-1)
        correct = (outputs_class == labels)
        # correct = correct.sum().item()
        correct = np.array(correct.cpu().detach().numpy()).reshape(-1)
        return correct

# %%
class EmptyWith:
    def __init__(self, *args, **kwargs):
        pass
    
    def __enter__(self, *args, **kwargs):
        return self
    
    def __exit__(self, *args, **kwargs):
        return self

# EW = EmptyWith()

# with EW:
#     a = 10
#     print(a)

# %%
class Model(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        return 0

# %%
class Network:
    arch_fns = {
        
        # 'dino_vits16': lambda **kwargs: torch.hub.load('facebookresearch/dino:main', 'dino_vits16'),
        # 'dino_vits8': lambda **kwargs: torch.hub.load('facebookresearch/dino:main', 'dino_vits8'),
        # 'dino_vitb16': lambda **kwargs: torch.hub.load('facebookresearch/dino:main', 'dino_vitb16'),
        # 'dino_vitb8': lambda **kwargs: torch.hub.load('facebookresearch/dino:main', 'dino_vitb8'),
        # 'dino_resnet50': lambda **kwargs: torch.hub.load('facebookresearch/dino:main', 'dino_resnet50'),
        
        # 'resnext50_32x4d': torchvision.models.resnext50_32x4d,
        # 'resnext101_32x8d': torchvision.models.resnext101_32x8d,
        # 'wide_resnet50_2': torchvision.models.wide_resnet50_2,
        # 'wide_resnet101_2': torchvision.models.wide_resnet101_2,
        
        # 'densenet201': torchvision.models.densenet201,
        # 'densenet169': torchvision.models.densenet169,
        # 'densenet121': torchvision.models.densenet121,
        
    }
    
    optimizer_fns = {
        'sgd': lambda **kwargs: optim.SGD(**{'momentum': 0.9, **kwargs}),
        'adam': lambda **kwargs: optim.Adam(**kwargs),
        'adadelta': lambda **kwargs: optim.Adadelta(**kwargs),
        'adagrad': lambda **kwargs: optim.Adagrad(**kwargs),
        'adamw': lambda **kwargs: optim.AdamW(**kwargs),
        'adabelief': lambda **kwargs: AdaBelief(**{'eps': 1e-16, 'betas': (0.9,0.999), 'weight_decouple': True, 'rectify': True, **kwargs})
    }
    
    metrics_best_classification = {
        'acc': {
            'value': 0.0,
            'higher_is_better': True,
        }
    }
    
    log_print_info = {
        None: {
            'len': 0,
        }
    }
    
    def __init__(self,
                model='wide_resnet50_2',
                frozen_model_bottom=[],
                frozen_model_top=[],
                opt='sgd',
                loss_fn=lambda *args: 0.0,
                lr=0.001,
                lr_schedule_type='step',
                lr_schedule_step=10,
                lr_schedule_gamma=0.5,
                pretrained=True,
                device='cuda',
                metrics=[],
                metrics_best=None,
                ):
        if isinstance(model, nn.Module):
            self.model = model
        else:
            raise ValueError('model must be of type <torch.nn.Module>')
            # self.model = self.get_model(arch=model, pretrained=pretrained)
        
        self.frozen_model_bottom = frozen_model_bottom
        if isinstance(self.frozen_model_bottom, nn.Module):
            self.frozen_model_bottom = [self.frozen_model_bottom]
        self.frozen_model_top = frozen_model_top
        if isinstance(self.frozen_model_top, nn.Module):
            self.frozen_model_top = [self.frozen_model_top]
        
        self.loss_fn = loss_fn
        if not callable(self.loss_fn):
            raise ValueError('loss_fn [{}] must be callable'.format(self.loss_fn))
        self.lr = float(max(0.000000001, lr))
        self.optimizer = self.get_optimizer(name=opt, model=self.model, lr=self.lr)
        self.lr_scheduler = self.get_lr_scheduler(
            type=lr_schedule_type,
            optimizer=self.optimizer,
            step_size=lr_schedule_step,
            gamma=lr_schedule_gamma,
        )
        self.device = device
        self.metrics = metrics
    
    def fit(self,
                dataloader_train=None,
                dataloader_val=None,
                epochs=10,
                epoch_start=0,
                fp_json_master=None,
                time_stamp='<latest>',
                stats={}
                ):
        metrics_best = {
            'train': {'acc': 0.0},
            'val': {'acc': 0.0},
        }
        epoch_final = epoch_start + epochs
        for _epoch in range(epoch_start, epoch_final):
            print()
            for _dataloader, _training in zip([dataloader_train, dataloader_val], [True, False]):
                _split = ['val', 'train'][int(_training)]
                if _dataloader is None:
                    continue
                dls = [_dataloader]
                if isinstance(_dataloader, list):
                    dls = _dataloader
                _stat = {}
                for _dl in dls:
                    _stat = self.run_one_epoch(
                        dataloader=_dl,
                        epoch=_epoch,
                        training=_training,
                        print_fps=30,
                        metrics_best=metrics_best[_split],
                        epoch_final=epoch_final,
                    )
                    print()
                for k in metrics_best[_split]:
                    if k in _stat:
                        metrics_best[_split][k] = max(metrics_best[_split][k], _stat[k])
                if _training:
                    self.lr_scheduler.step()
                if _split not in stats:
                    stats[_split] = []
                stats[_split].append(_stat)
                # save json master stats
                if isinstance(fp_json_master, str):
                    self.save_stats(
                        stats,
                        fp_json_master=fp_json_master,
                        time_stamp=time_stamp,
                    )
    
    def run_one_epoch(self,
                dataloader,
                epoch=0,
                training=True,
                print_fps=30,
                metrics_best={},
                epoch_final=None,
                ):
        
        losses = []
        _loss_avg = -1.
        batch_count = len(dataloader)
        time_start = time.time()
        time_last_print = time_start
        correct_count = 0
        sample_count = 0
        acc_percent = 0.0
        time_ett = 1.0
        _outputs = []
        _labels = []
        
        if epoch_final is None:
            epoch_final = epoch
        
        self.model.to(self.device)
        for m in self.frozen_model_bottom:
            m.to(self.device)
        for m in self.frozen_model_top:
            m.to(self.device)
        metrics_acc = AccuracyMetrics(
            'acc[{percent:6.2f}%{best}]',
            value=0.0,
            best_str='(best)',
            last_best=metrics_best.get('acc', 1.0),
        )
        pl = ProgressLog(
            name='Train' if training else '  Val',
            epochs=epoch_final,
            steps=batch_count,
            timer='step',
            loss=AverageMetrics('loss[{value:8.6f}]', value=99.9, lower_is_better=True),
            acc=metrics_acc,
        )
        pl.update(
            epoch=epoch,
            step=0,
        )
        pl.update(epoch=epoch, step=0)
        pl.print()
        
        context = [torch.no_grad, EmptyWith][int(bool(training))]
        
        with context():
            for batch_index, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                x = inputs
                with torch.no_grad():
                    for m in self.frozen_model_bottom:
                        x = m(x)
                x = self.model(x)
                with torch.no_grad():
                    for m in self.frozen_model_top:
                        x = m(x)
                outputs = x
                
                loss = self.loss_fn(outputs, labels)
                
                # if return_values or debug:
                #     _outputs.append(outputs.cpu().detach().numpy())
                #     _labels.append(labels.cpu().detach().numpy())
                
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                # metrics = {
                #     _name: _m['fn'](outputs, labels)
                #     for _name, _m in self.metrics.items()
                # }
                _corrects = classification_count_correct(outputs, labels)
                # correct_count += _correct
                # sample_count += outputs.size(0)
                # acc_percent = correct_count / max(sample_count, 1) * 100
                
                _loss_value = float(loss.item())
                losses.append(_loss_value)
                _loss_avg = sum(losses) / len(losses)
                
                # if i % max(20, int(batch_count // 4)) == 0:
                #     peak_vram_gb = max(peak_vram_gb, get_vram_fn())
                
                _progress_percent = (batch_index + 1) / batch_count * 100
                
                time_current = time.time()
                time_elapsed = time_current - time_start
                time_ett = time_elapsed / max(0.000001, _progress_percent / 100)
                time_eta = time_ett * (1 - _progress_percent / 100)
                
                metrics_acc.update(_corrects)
                pl.update(
                    step=batch_index + 1,
                    # acc=_corrects,
                    loss=_loss_value,
                )
                if print_fps > 0 and time_current >= time_last_print + 1 / print_fps:
                    time_last_print = max(time_last_print + 1/30, time_current - 1/60)
                    pl.print()
            
        time_current = time.time()
        time_elapsed = time_current - time_start
        metrics_acc.update(None, True)
        pl.print()
        _stat = {
            'epoch': epoch + 1,
            'time_start': time_start,
            'time_finish': time_current,
            'time_cost': time_elapsed,
            'loss': float(_loss_avg),
            'acc': float(metrics_acc.value),
            # 'vram': peak_vram_gb,
        }
        return _stat
    
    # @classmethod
    # def get_model(cls, arch, pretrained=None, include_top=True, prev_features=None, **kwargs):
    #     # assert isinstance(arch, str)
    #     # dino_archs = [
    #     #     'dino_vits16',
    #     #     'dino_vits8',
    #     #     'dino_vitb16',
    #     #     'dino_vitb8',
    #     #     'dino_resnet50',
    #     # ]
    #     # xcit_urls = {
    #     #     'xcit_small_12_p16': 'https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p16_224.pth',
    #     # }
    #     # cnn_fns = {
    #     #     'resnext50_32x4d': torchvision.models.resnext50_32x4d,
    #     #     'resnext101_32x8d': torchvision.models.resnext101_32x8d,
    #     #     'wide_resnet50_2': torchvision.models.wide_resnet50_2,
    #     #     'wide_resnet101_2': torchvision.models.wide_resnet101_2,
    #     #     'densenet201': torchvision.models.densenet201,
    #     #     'densenet169': torchvision.models.densenet169,
    #     #     'densenet121': torchvision.models.densenet121,
    #     # }
        
    #     # torch.hub.load('facebookresearch/dino:main', arch)
    #     if isinstance(arch, nn.Module):
    #         return arch
        
    #     if isinstance(arch, int):
    #         print(prev_features)
    #         if isinstance(prev_features, nn.Module):
    #             prev_features = cls.get_output_shape(prev_features)[-1]
    #         assert isinstance(prev_features, int)
    #         return nn.Linear(prev_features, arch, bias=True)
        
    #     if isinstance(arch, list):
    #         assert len(arch) >= 1
    #         if len(arch) == 1:
    #             return cls.get_model(
    #                 arch[0],
    #                 pretrained=pretrained,
    #                 include_top=include_top,
    #                 prev_features=None,
    #                 **kwargs,
    #             )
    #         else:
    #             _model_0 = cls.get_model(
    #                 arch[0],
    #                 pretrained=pretrained,
    #                 include_top=include_top,
    #                 prev_features=None,
    #                 **kwargs,
    #             )
    #             _model_1 = cls.get_model(
    #                 arch[1],
    #                 pretrained=pretrained,
    #                 include_top=include_top,
    #                 prev_features=_model_0,
    #                 **kwargs,
    #             )
    #             print(type(_model_0))
    #             print(type(_model_1))
    #             _model_01 = nn.Sequential([_model_0, _model_1])
    #             return nn.Sequential([
    #                 _model_01,
    #                 *[
    #                     cls.get_model(
    #                         v,
    #                         pretrained=pretrained,
    #                         include_top=include_top,
    #                         prev_features=_model_01,
    #                         **kwargs,
    #                     )
    #                     for v in arch[2:]
    #                 ],
    #             ])
        
    #     if not isinstance(arch, str):
    #         return arch
    #     if arch in cls.arch_fns:
    #         _kwargs = {**kwargs}
    #         if pretrained is not None:
    #             _kwargs = {
    #                 **_kwargs,
    #                 'pretrained': pretrained,
    #             }
    #         _model = cls.arch_fns[arch](**_kwargs)
    #         if include_top == False:
    #             if any([arch.startswith(v) for v in ['resne', 'wide_resne']]):
    #                 _model.fc = nn.Identity()
                
    #             if any([arch.startswith(v) for v in ['densenet']]):
    #                 _model.classifier = nn.Identity()
    #         return _model
    #     else:
    #         raise ValueError('arch `{}` is currently not supported! must be one of [ {} ]'.format(arch, ' | '.join([
    #             str(v) for v in cls.arch_fns.keys()
    #         ])))
    
    @classmethod
    def get_optimizer(cls, name='sgd', model=None, lr=0.001, **kwargs):
        _params = model.parameters()
        if name in cls.optimizer_fns:
            return cls.optimizer_fns[name](params=_params, lr=lr, **kwargs)
        else:
            return None
            # raise ValueError('optimizer `{}` is currently not supported! must be one of [ {} ]'.format(arch, ' | '.join([
            #     str(v) for v in cls.optimizer_fns.keys()
            # ])))
    
    @classmethod
    def get_lr_scheduler(cls, type='step', optimizer=None, step_size=10, gamma=0.5, **kwargs):
        if type == 'step':
            lr_scheduler = StepLR(
                optimizer,
                step_size=int(max(1, step_size)),
                gamma=np.clip(gamma, 0.0, 1.0),
            )
        else:
            print('lr_scheduler [{}] could not be recognized | default to none')
            lr_scheduler = lambda *args, **kwargs: None
        return lr_scheduler
    
    @classmethod
    def get_output_shape(cls, model, input_shape=[1, 3, 224, 224], device='cuda'):
        # print('getting the output shape of the backbone model with {}'.format(input_shape))
        model.to(device)
        output_shape = model(torch.rand(*input_shape).to(device)).data.shape
        # num_features = int(output_shape[-1])
        # print('found shape: [N, {}]'.format(num_features))
        return output_shape
    
    @classmethod
    def save_stats(cls, _stats={}, fp_json_master='./logs/master2.json', time_stamp='<latest>'):
        # fp = args['stats_json']
        # fp_master = args['master_stats_json']
        fp_master = fp_json_master
        
        # if isinstance(fp, str) and fp.endswith('.json'):
        #     _dp = os.path.split(fp)[0]
        #     if not os.path.isdir(_dp):
        #         os.makedirs(_dp)
        #     _ = json.dump(_stats, open(fp, 'w'), indent=4)
        
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



# %%
# cnn_archs = [
#     'dino_vits8',
#     'resnext50_32x4d',
#     # 'resnext101_32x8d',
#     'wide_resnet50_2',
#     # 'wide_resnet101_2',
#     # 'densenet201',
#     # 'densenet169',
#     'densenet121',
# ]
# for a in cnn_archs:
#     model = Network.get_model(a)
#     model.to('cuda')
#     r = {}
#     r['arch'] = a
#     b = Network.get_output_shape(model)
#     r['output_full'] = b
#     r['fc'] = None
#     r['output_without_top'] = None
#     try:
#         r['fc'] = str(model.fc)
#         model.fc = nn.Identity()
#         c = Network.get_output_shape(model)
#         r['output_without_top'] = c
#     except:
#         pass
    
#     print(json.dumps(r, indent=4))

# %%

# %%