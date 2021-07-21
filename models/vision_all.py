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
from models.swin import SwinTransformer
from models.swin import get_swin_model

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
        'swin': [
            # 'swin_base_patch4_window7_224',
            # 'swin_base_patch4_window7_224',
            # 'swin_base_patch4_window12_384',
            # 'swin_large_patch4_window7_224',
            # 'swin_large_patch4_window12_384',
            
            'swin_tiny_patch4_window7_224',
            'swin_small_patch4_window7_224',
            'swin_base_patch4_window7_224',
            'swin_base_patch4_window12_384',
            'swin_base_patch4_window7_224_22k',
            'swin_base_patch4_window7_224_22kto1k',
            'swin_base_patch4_window12_384_22k',
            'swin_base_patch4_window12_384_22kto1k',
            'swin_large_patch4_window7_224_22k',
            'swin_large_patch4_window7_224_22kto1k',
            'swin_large_patch4_window12_384_22k',
            'swin_large_patch4_window12_384_22kto1k',
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
        if isinstance(root_path, str):
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
        if _type == 'swin':
            return cls.get_model_swin(
                arch=arch,
                pretrained=pretrained,
                image_channels=image_channels,
                classifier=classifier,
                classifier_act=classifier_act,
                *args,
                **kwargs,
            )
        if _type == 'resnet':
            return cls.get_model_resnet(
                arch=arch,
                pretrained=pretrained,
                image_channels=image_channels,
                classifier=classifier,
                classifier_act=classifier_act,
                *args,
                **kwargs,
            )
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
            model.head = cls.get_classifier_head(
                in_features=backbone_features,
                classifier_units=classifier,
                classifier_act=classifier_act,
            )
        
        if return_separate:
            _head = model.head
            model.head = nn.Identity()
            print(model.head, _head)
            return model, _head
        
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
        
        if return_separate:
            _head = model.head
            model.head = nn.Identity()
            print(model.head, _head)
            return model, _head
        
        return model
    
    @classmethod
    def get_model_swin(cls, arch='swin_base_patch4_window7_224', pretrained=True, image_channels=3, classifier=None, classifier_act=GELU(), return_separate=False):
        # TYPE: swin
        # NAME: swin_base_patch4_window7_224
        # DROP_PATH_RATE: 0.5
        # SWIN:
        #     EMBED_DIM: 128
        #     DEPTHS: [ 2, 2, 18, 2 ]
        #     NUM_HEADS: [ 4, 8, 16, 32 ]
        #     WINDOW_SIZE: 7
        
        assert image_channels == 3
        # assert arch == 'swin_base_patch4_window7_224'
        _num_classes = 1000
        if classifier is False or isinstance(classifier, (int, list)):
            if classifier != 1000:
                _num_classes = 0
        
        model = get_swin_model(arch, pretrained=pretrained)
        
        # model = SwinTransformer(
        #     img_size=224,
        #     patch_size=4,
        #     in_chans=image_channels,
        #     num_classes=_num_classes,
        #     embed_dim=128,
        #     depths=[2, 2, 18, 2],
        #     num_heads=[4, 8, 16, 32],
        #     window_size=7,
        #     mlp_ratio=4.,
        #     qkv_bias=True,
        #     qk_scale=None,
        #     drop_rate=0.0,
        #     drop_path_rate=0.5,
        #     ape=False,
        #     patch_norm=True,
        #     use_checkpoint=False,
        # )
        
        if isinstance(classifier, (int, list)):
            backbone_features = model.norm.weight.data.shape[-1]
            model.head = cls.get_classifier_head(
                in_features=backbone_features,
                classifier_units=classifier,
                classifier_act=classifier_act,
            )
        
        # WIP classifier not dealt with
        # if classifier is False:
        #     model.head = nn.Identity()
        #     model.head_dist = model.head
        # elif isinstance(classifier, (int, list)):
        #     backbone_features = model.norm.weight.data.shape[-1]
        #     model.head = cls.get_classifier_head(
        #         in_features=backbone_features,
        #         classifier_units=classifier,
        #         classifier_act=classifier_act,
        #     )
        #     model.head_dist = model.head
        
        # 'swin_base_patch4_window7_224': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth'
        # 'swin_base_patch4_window12_384': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth'
        # 'swin_large_patch4_window7_224': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth'
        # 'swin_large_patch4_window12_384': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth'




        if return_separate:
            _head = model.head
            model.head = nn.Identity()
            print(model.head, _head)
            return model, _head
        
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
    def get_model_resnet(cls, arch='wide_resnet50_2', pretrained=True, image_channels=3, classifier=None, classifier_act=GELU(), return_separate=False):
        
        model_fns = {
            'resnext50_32x4d': torchvision.models.resnext50_32x4d,
            'resnext101_32x8d': torchvision.models.resnext101_32x8d,
            'wide_resnet50_2': torchvision.models.wide_resnet50_2,
            'wide_resnet101_2': torchvision.models.wide_resnet101_2,
        }
        assert arch in model_fns
        model = model_fns[arch](pretrained=pretrained)
        if pretrained == False:
            cls.reset_parameters(model)
        # if image_channels != 3 and image_channels is not None:
        #     assert isinstance(image_channels, int) and image_channels > 0
        #     model.patch_embed.proj = torch.nn.Conv2d(
        #         in_channels=image_channels,
        #         out_channels=model.patch_embed.proj.out_channels,
        #         stride=model.patch_embed.proj.stride,
        #         kernel_size=model.patch_embed.proj.kernel_size,
        #         padding=model.patch_embed.proj.padding,
        #     )
        if isinstance(classifier, (int, list)):
            model.fc = nn.Identity()
            _output_shape = cls.get_output_shape(model, input_shape=[1, 3, 128, 128])
            backbone_features = _output_shape[-1]
            model.fc = cls.get_classifier_head(
                in_features=backbone_features,
                classifier_units=classifier,
                classifier_act=classifier_act,
            )
        
        if return_separate:
            _head = model.fc
            model.fc = nn.Identity()
            print(model.fc, _head)
            return model, _head
        
        return model
    
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