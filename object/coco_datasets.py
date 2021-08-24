# %%
from shutil import ExecError
from sys import exec_prefix
from plotly import express
import torch
import torchvision
import torchvision.transforms as T
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os, json
import os.path
from typing import Any, Callable, Optional, Tuple, List
import numpy as np
from torchvision.transforms.transforms import Resize

# %%
def analyze_object(a):
    if isinstance(a, dict):
        return 'dict[{}] [{}]'.format(len(a), '\n '.join([v1 for v1 in [analyze_object(v) for v in a.values()] if isinstance(v1, str)]))
    if isinstance(a, list):
        return 'list[{}] [{}]'.format(len(a), '\n '.join([v1 for v1 in [analyze_object(v) for v in a] if isinstance(v1, str)]))
    return None

# %%
def get_fit_to_od(img, target=[], shape=1000, fill=0, interpolation=Image.BICUBIC):
    if isinstance(shape, int):
        assert shape > 0
        shape = [shape, shape]
    assert isinstance(shape, (list, tuple)) and len(shape) == 2
    
    s0 = img.size[:2]
    s1 = shape
    
    scales = [v1 / v0 for v0, v1 in zip(s0, s1)]
    scale = min(scales)
    pad_axis = np.argmax(scales)
    
    s2 = [*s1]
    s2[pad_axis] = int(np.ceil(s0[pad_axis] * scale))
    offset = [0, 0]
    offset[pad_axis] = int(np.floor((s1[pad_axis] - s2[pad_axis]) / 2))
    s2
    
    if img.mode in ['RGB', 'L']:
        img1 = Image.new(img.mode, s1, fill)
    else:
        raise ValueError('img must have be rank 2 or 3')
    img2 = img.resize(s2, resample=interpolation)
    img1.paste(img2, offset)
    
    # print(target)
    # print(type(target))
    # print(len(target))
    # target is a list of dicts with keys 'segmentation', 'bbox', 'area' that need modified
    # print(json.dumps(target, indent=4))
    # print(str(analyze_object(target,)))
    # print(type(target), len(target), {k: ((str(type(v)), len(v)) if isinstance(v, (list, dict, tuple)) else type(v)) for k, v in target[0].items()})
    # print('target - {} [{}] keys {}'.format(str(type(target)), len(target), list(target[0].keys())))
    # print('target[0][area] - {} {}'.format(type(target[0]['area']), target[0]['area']))
    # print('target[0][bbox] - {} {}'.format(type(target[0]['bbox']), target[0]['bbox']))
    # print('target[0][segmentation] - {} {}'.format(type(target[0]['bbox']), target[0]['bbox']))
    # print()
    target1 = []
    for t in target:
        t1 = {
            k: v for k, v in t.items() if k not in ['segmentation', 'bbox', 'area']
        }
        # t1['segmentation'] = [
        #     []
        #     for segms in t['segmentation']
        # ]
        # try:
        #     t1['segmentation'] = [
        #         [
        #             v * scale + offset[i % 2]
        #             for i, v in enumerate(segms)
        #         ]
        #         for segms in t['segmentation']
        #     ]
        # except Exception as e:
        #     print('segmentation error')
        #     print(t['segmentation'])
        #     raise e
        # t1['bbox'] = [
        #     v * scale + (offset[i] if i < 2 else 0)
        #     for i, v in enumerate(t['bbox'])
        # ]
        t1['bbox'] = [
            v * scale + (offset[i % 2])
            for i, v in enumerate(t['bbox'])
        ]
        t1['area'] = t['area'] * (scale ** 2)
        target1.append(t1)
    # boxes1 = [
    #     [
    #         v[0] * scale + offset[0],
    #         v[1] * scale + offset[1],
    #         v[2] * scale + offset[0],
    #         v[3] * scale + offset[1],
    #     ]
    #     for v in target['boxes']
    # ]
    # areas1 = [
    #     v * (scale ** 2)
    #     for v in target['areas']
    # ]
    # target1 = {
    #     **target,
    #     'boxes': boxes1,
    #     'areas': areas1,
    # }
    # with open('logs/temp_transform2.json', 'w') as f:
    #     json.dump(
    #         {'before': target, 'after': target1},
    #         f,
    #         indent=4,
    #     )
    # raise ValueError('debug done')
    
    return img1, target1

def fit_to_od(shape=1000, fill=0, interpolation=Image.BICUBIC):
    return T.Lambda(lambda img, target: get_fit_to_od(
        img,
        target,
        shape=shape,
        fill=fill,
        interpolation=interpolation,
    ))


# %%
class CocoDetectionCustom(torchvision.datasets.CocoDetection):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        limit: int = 0,
        # annFile: str = '',
        image_size: int = 0,
        shuffle_raw: bool = False,
        # transform: Optional[Callable] = None,
        # target_transform: Optional[Callable] = None,
        # transforms: Optional[Callable] = None,
    ):
        data_dir = os.path.join(root, 'data')
        # super().__init__(data_dir, transforms, transform, target_transform)
        super().__init__(data_dir, None, None, None)
        from pycocotools.coco import COCO
        
        self.coco = COCO(os.path.join(root, 'labels.json'))
        self.limit = limit
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.image_size = image_size
        if isinstance(self.image_size, int) and self.image_size > 0:
            self.image_size = [self.image_size, self.image_size]
        if not isinstance(self.image_size, (list, tuple)):
            self.image_size = None
        else:
            self.image_size = list(self.image_size)
        if shuffle_raw:
            self.ids = list(np.random.choice(self.ids, size=len(self.ids), replace=False))
        if self.limit > 0:
            self.ids = self.ids[:self.limit]
        self.transforms = T.Compose([T.ToTensor()])

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        
        # if isinstance(self.image_size, (list, tuple)):
        #     image, target = get_fit_to_od(
        #         image, target,
        #         shape=self.image_size,
        #         fill=0,
        #     )
        #     image, target = Resize(self.image_size)(image, target)
        #     assert self.image is None
        
        
        target_output = {}
        target_output["image_id"] = torch.as_tensor([id])
        target_output["boxes"] = torch.as_tensor([v['bbox'] for v in target], dtype=torch.float32)
        target_output["labels"] = torch.as_tensor([v['category_id'] for v in target], dtype=torch.int64)
        target_output["area"] = torch.as_tensor([v['area'] for v in target], dtype=torch.float32)
        target_output["iscrowd"] = torch.as_tensor([v['iscrowd'] for v in target], dtype=torch.int64)
        
        # "area": 237.63384999999988,
        # "iscrowd": 0,
        # "image_id": 872,
        # "bbox": [
        #     408.03,
        #     172.04,
        #     19.38,
        #     16.53
        # ],
        # "category_id": 37,
        # image = T.ToTensor()(image)
        
        # if self.transforms is not None:
        #     # image, target = self.transforms(image, target)
        #     image = self.transforms(image)
        
        # target = T.ToTensor()(target)
        # if self.transforms is not None:
        #     image, target = self.transforms(image, target)
        
        return image, target_output
        # return image, target

    # def __getitem__(self, index: int) -> Tuple[Any, Any]:
    #     id = self.ids[index]
    #     image = self._load_image(id)
    #     target = self._load_target(id)

    #     if self.transforms is not None:
    #         image, target = self.transforms(image, target)

    #     return image, target

    def __len__(self) -> int:
        return len(self.ids)



# %%
# _split = 'validation'
# _split = 'train'
# ds = CocoDetectionCustom(
#     root='/host/ubuntu/torch/coco2017/{}'.format(_split),
#     annFile='/host/ubuntu/torch/coco2017/{}/labels.json'.format(_split),
#     # transform=T.Compose([]),
#     # target_transform=T.Compose([]),
#     limit=200,
# )
# type(ds), len(ds)

# %%


# %%

