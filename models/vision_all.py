# %%
# from utils_model import classification_count_correct
from timm.models.factory import create_model
import torch
from torch._C import Value
import torch.nn as nn
from torch.nn.modules import linear
from torch.nn.modules.activation import GELU
from functools import partial
from torch.nn.modules.linear import Linear
import torchvision

import timm
# from timm.models.vision_transformer import VisionTransformer, _cfg
# from timm.models.registry import register_model
# from timm.models.layers import trunc_normal_
# from models_deit import DistilledVisionTransformer, VisionTransformer
from models import deit as DEIT
from models import cait as CAIT

import os, time, datetime, string, json
import torchvision
import numpy as np
import pandas as pd


# %%
class VisionModelZoo:
    archs_types = {
        # **{
        #     v: 'deit'
        #     for v in DEIT.__all__
        # },
        'cait': list(CAIT.__all__),
        'dino': [
            'dino_vits16',
            'dino_vits8',
            'dino_vitb16',
            'dino_vitb8',
            'dino_resnet50',
        ],
        'resnet': [
            'resnext50_32x4d',
            'resnext101_32x8d',
            'wide_resnet50_2',
            'wide_resnet101_2',
        ],
        # 'densenet': [
        #     'densenet201',
        #     'densenet169',
        #     'densenet121',
        # ],
    }
    
    @classmethod
    def get_model(cls,
                arch=None,
                pretrained=True,
                image_channels=3,
                classifier=None,
                classifier_act=GELU(),
                root_path='/host/ubuntu/torch',
                *args,
                **kwargs,
                ):
        os.environ['TORCH_HOME'] = root_path
        if arch is None:
            if isinstance(classifier, (int, list)):
                return cls.get_classifier_head(
                    in_features=image_channels,
                    classifier_units=classifier,
                    classifier_act=classifier_act,
                )
            else:
                return nn.Identity()
        _type = None
        for k, v in cls.archs_types.items():
            if arch in v:
                _type = k
                break
        if _type is None:
            raise ValueError('arch [{}] not found!'.format(arch))
        if _type == 'dino':
            return cls.get_model_dino(
                arch=arch,
                pretrained=pretrained,
                image_channels=image_channels,
                classifier=classifier,
                classifier_act=classifier_act,
                *args,
                **kwargs,
            )
        if _type == 'cait':
            return cls.get_model_cait(
                arch=arch,
                pretrained=pretrained,
                image_channels=image_channels,
                classifier=classifier,
                classifier_act=classifier_act,
                *args,
                **kwargs,
            )
        if _type == 'deit':
            raise NotImplementedError('arch [{}] of type [{}] is being implemented, usage is limited due to the requirement of input image size'.format(
                arch, _type
            ))
        raise NotImplementedError('arch [{}] of type [{}] is being implemented, will be available soon'.format(
            arch, _type
        ))
    
    @classmethod
    def get_model_dino(cls, arch='dino_vits16', pretrained=True, image_channels=3, classifier=None, classifier_act=GELU(), return_separate=False):
        model = torch.hub.load('facebookresearch/dino:main', arch, pretrained=bool(pretrained))
        if pretrained == False:
            cls.reset_parameters(model)
        if image_channels != 3 and image_channels is not None:
            assert isinstance(image_channels, int) and image_channels > 0
            model.patch_embed.proj = torch.nn.Conv2d(
                in_channels=image_channels,
                out_channels=model.patch_embed.proj.out_channels,
                stride=model.patch_embed.proj.stride,
                kernel_size=model.patch_embed.proj.kernel_size,
                padding=model.patch_embed.proj.padding,
            )
        if isinstance(classifier, (int, list)):
            backbone_features = model.norm.weight.data.shape[-1]
            model.head = torch.nn.Identity()
            classsification_head = cls.get_classifier_head(
                in_features=backbone_features,
                classifier_units=classifier,
                classifier_act=classifier_act,
            )
            model = torch.nn.Sequential(
                model,
                classsification_head,
            )
        return model
    
    @classmethod
    def get_model_cait(cls, arch='cait_M36', pretrained=True, image_channels=3, classifier=None, classifier_act=GELU(), return_separate=False):
        model = create_model(
            arch,
            pretrained=bool(pretrained),
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )
        if image_channels != 3 and image_channels is not None:
            assert isinstance(image_channels, int) and image_channels > 0
            model.patch_embed.proj = torch.nn.Conv2d(
                in_channels=image_channels,
                out_channels=model.patch_embed.proj.out_channels,
                stride=model.patch_embed.proj.stride,
                kernel_size=model.patch_embed.proj.kernel_size,
                padding=model.patch_embed.proj.padding,
            )
        if classifier is False:
            model.head = nn.Identity()
            model.head_dist = model.head
        elif isinstance(classifier, (int, list)):
            backbone_features = model.norm.weight.data.shape[-1]
            model.head = cls.get_classifier_head(
                in_features=backbone_features,
                classifier_units=classifier,
                classifier_act=classifier_act,
            )
            model.head_dist = model.head
        return model
    
    @classmethod
    def get_classifier_head(cls, in_features, classifier_units=None, classifier_act=GELU()):
        linear_layers = []
        if isinstance(classifier_units, int):
            classifier_units = [classifier_units]
        if isinstance(classifier_units, list):
            for i, v in enumerate(classifier_units):
                if i == 0:
                    _in_features = in_features
                else:
                    _in_features = classifier_units[i - 1]
                is_not_last = True
                if i >= len(classifier_units) - 1:
                    is_not_last = False
                linear_layers.append(torch.nn.Linear(
                    in_features=_in_features,
                    out_features=v,
                    bias=is_not_last,
                ))
                if is_not_last:
                    linear_layers.append(classifier_act)
        return torch.nn.Sequential(*linear_layers)
    
    @classmethod
    def reset_parameters(cls, m=[]):
        if isinstance(m, list):
            _ = [cls.reset_parameters(v) for v in m]
        if hasattr(m, 'children'):
            cls.reset_parameters(list(m.children()))
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    
    @classmethod
    def get_output_shape(cls, model, input_shape=[1, 3, 32, 32], device='cuda'):
        _temp_inputs = torch.rand(*input_shape)
        _ = model.to(device)
        _output = model(_temp_inputs.to(device)).cpu().detach().numpy()
        return _output.shape
        

# %%
# m = VisionModelZoo.get_model_cait(pretrained=False, image_channels=9, classifier=[49, 31, 7])
# _ = m

# # %%
# input_shape = [1, 2, 224, 224]
# temp_inputs = torch.rand(*input_shape)
# m = VisionModelZoo.get_model(
#     'dino_vits8',
#     pretrained=False,
#     image_channels=input_shape[1],
#     classifier=[49, 31, 7],
# )
# _ = m.to('cuda')
# _output = m(temp_inputs.to('cuda')).cpu().detach().numpy()
# print(input_shape)
# print(_output.shape)

# # %%
# import sys
# sys.exit()

# %%