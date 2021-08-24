# %%
import time, json, os

import torch
import torchvision

import fiftyone as fo
import coco_utils

os.environ['TORCH_HOME'] = '/host/ubuntu/torch'

# %%
# The directory containing the source images
# data_path = '/host/ubuntu/torch/coco2017/train'
data_path = '/host/ubuntu/torch/coco2017/validation/data'

# The path to the COCO labels JSON file
labels_path = '/host/ubuntu/torch/coco2017/validation/labels.json'

# Import the dataset
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=data_path,
    labels_path=labels_path,
)
dataset


# %%

class FiftyOneTorchDataset(torch.utils.data.Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.
    
    Args:
        fiftyone_dataset: a FiftyOne dataset or view that will be used for training or testing
        transforms (None): a list of PyTorch transforms to apply to images and targets when loading
        gt_field ("ground_truth"): the name of the field in fiftyone_dataset that contains the 
            desired labels to load
        classes (None): a list of class strings that are used to define the mapping between
            class names and indices. If None, it will use all classes present in the given fiftyone_dataset.
    """
    
    def __init__(
        self,
        fiftyone_dataset,
        transforms=None,
        gt_field="ground_truth",
        classes=None,
    ):
        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.gt_field = gt_field
        
        self.img_paths = self.samples.values("filepath")
        
        self.classes = classes
        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.samples.distinct(
                "%s.detections.label" % gt_field
            )
        
        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes
        
        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        metadata = sample.metadata
        img = Image.open(img_path).convert("RGB")
        
        boxes = []
        labels = []
        area = []
        iscrowd = []
        detections = sample[self.gt_field].detections
        for det in detections:
            coco_obj = fouc.COCOObject.from_detection(
                det, metadata, labels_map_rev=self.labels_map_rev
            )
            x, y, w, h = coco_obj.bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(coco_obj.category_id)
            area.append(coco_obj.area)
            iscrowd.append(coco_obj.iscrowd)
        
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.as_tensor([idx])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        if self.transforms is not None:
            # img, target = self.transforms(img, target)
            img = self.transforms(img)
            # target = self.transforms(target)
        
        return img, target
    
    def __len__(self):
        return len(self.img_paths)
    
    def get_classes(self):
        return self.classes


# %%
from fiftyone import ViewField as F
# %%
_set = dataset
            
labels_list = [
    'car', 'truck', 'bus', 'bicycle',
    'chair', 'person', 'bench',
    'bird', 'cat', 'dog',
]

_set = _set.filter_labels(
    "ground_truth",
    F("label").is_in(labels_list),
)

# filtered_count = len(_set)
# _limit = raw_count
# if isinstance(limits, int):
#     _limit = limits
# elif isinstance(limits, (list, tuple)) and len(limits) > i:
#     _limit = limits[i]
_view = _set.take(32, seed=0)
_view
torch_dataset = FiftyOneTorchDataset(
    fiftyone_dataset=_view,
    # transforms=transform[_split],
    # gt_field="ground_truth",
    # classes=None,
    classes=labels_list,
)
torch_loader = torch.utils.data.DataLoader(
    torch_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=16,
    collate_fn=coco_utils.collate_fn,
)