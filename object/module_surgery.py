# %%
from torchsummary import summary
import torch
from swin_transformer_backbone import SwinTransformer

# %%
def analyze_model(model, input_shape=(3, 224, 224), device='cuda'):
    model.to(device)
    _ = summary(model, input_size=input_shape)
    return _
    
# %%
model_swin = SwinTransformer(
    hidden_dim=96,
    layers=(2, 2, 6, 2),
    heads=(3, 6, 12, 24),
    channels=3,
    num_classes=10,
    head_dim=32,
    window_size=7,
    downscaling_factors=(4, 2, 2, 2),
    relative_pos_embedding=True
)

# %%
dummy_x = torch.randn(1, 3, 224, 224)
logits = model_swin(dummy_x)  # (1,3)
print(model_swin)
print(logits)

# %%
model_swin.to('cuda')
summary(model_swin, input_size=(3, 224, 224))

# %%
# model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
# model

# # %%
# for c in model.children():
#     print(c)

# # %%
# model.children()[2]
# %%






# %%
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import os
os.environ['TORCH_HOME'] = '/host/ubuntu/torch'

from models.swin import get_swin_model, get_swin_model_od, SwinTransformer
from models.vision_all import VisionModelZoo

# %%
# model = get_swin_model('swin_tiny_patch4_window7_224', False)
# model = get_swin_model_od('swin_tiny_patch4_window7_224', True)
model = get_swin_model_od('swin_base_patch4_window12_384', True)

model.to('cuda')
_output_shape = VisionModelZoo.get_output_shape(model, (2, 3, 224, 224))
_output_shape
model.out_channels = _output_shape[1]
model.img_size
model

# %%
# load a pre-trained model for classification and return
# only the features
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it here
backbone.out_channels = 1280
VisionModelZoo.get_output_shape(backbone, (2, 3, 224, 224))

# %%
# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# %%
# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

# %%
# put the pieces together inside a FasterRCNN model
model_swin_frcnn = FasterRCNN(
    model,
    min_size=model.img_size,
    max_size=model.img_size,
    num_classes=2,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler,
)

model_swin_frcnn

# %%
model_swin_frcnn.eval()
_temp_inputs = torch.rand(*(2, 3, 224, 224))
_ = model_swin_frcnn.to('cuda')
_output = model_swin_frcnn(_temp_inputs.to('cuda'))[0]
# len(_output)
{k: v.cpu().detach().numpy().shape for k, v in _output.items()}

# %%
model_r50_fpn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model_r50_fpn.eval()
_temp_inputs = torch.rand(*(1, 3, 224, 224))
_ = model_r50_fpn.to('cuda')
_output = model_r50_fpn(_temp_inputs.to('cuda'))[0]
_output.keys()

