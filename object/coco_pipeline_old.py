# %%
import fiftyone as fo
import fiftyone.zoo as foz
import torch
import torchvision
import time, json, os

# %%
import torch
import fiftyone.utils.coco as fouc
from PIL import Image
from torchvision.transforms.transforms import Resize
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from fiftyone import ViewField as F

# From the torchvision references we cloned
import torchvision.transforms as T
from engine import train_one_epoch, evaluate
import torch_utils
import argparse

# %%
time_stamp = time.strftime('%y%m%d_%H%M%S')
time_start = time.time()

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--train_limit', type=int, default=500)
parser.add_argument('--val_limit', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--bs', type=int, default=2)
args = parser.parse_args()

# %%
train_limit = args.train_limit
val_limit = args.val_limit
_lr = args.lr
_bs = args.bs

# %%
# dataset = foz.load_zoo_dataset(
#     name='coco-2017',
#     split='validation',
#     # split='train',
#     # splits=None,
#     # label_field=None,
#     # dataset_name=None,
#     dataset_dir='/host/ubuntu/torch/coco2017',
#     # download_if_necessary=True,
#     # drop_existing_dataset=False,
#     # overwrite=False,
#     # cleanup=True,
# )
dataset_dir = os.path.join('/host/ubuntu/torch/coco2017', 'validation')
dataset_type = fo.types.COCODetectionDataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
    max_samples=1000,
    shuffle=False,
)
time_elapsed = time.time() - time_start
print('Done loading dataset in {:.0f}s'.format(time_elapsed))


# %%
dataset.compute_metadata()
print('type: ', type(dataset))
print()
print(dataset)


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
def get_model_FRCNN(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# %%

# Define simple image transforms
# train_transforms = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.5)])
train_transforms = T.Compose([T.ToTensor()])
val_transforms = T.Compose([T.ToTensor()])

labels_list = None
data_view = dataset

all_labels_train = [
    # 'background',
    'airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']

labels_list = all_labels_train

# labels_list = [
#     'car', 'truck', 'bus', 'bicycle',
#     'chair', 'person', 'bench',
#     'bird', 'cat', 'dog',
# ]
data_view = dataset.filter_labels(
    "ground_truth",
    F("label").is_in(labels_list),
)
# matching_view = data_view.match(
#     F("predictions.detections").filter(F("label").is_in(labels_list)).length() > 0
# )

raw_count = len(dataset)
filtered_count = len(data_view)
if val_limit <= 0:
    val_limit = int(filtered_count * 0.2)
if train_limit <= 0:
    filtered_count - val_limit
assert filtered_count > val_limit > 0
assert train_limit > 0
train_view = data_view.take(min(train_limit, len(data_view) - val_limit), seed=0)
val_view = data_view.exclude(train_view).take(val_limit)

print('\nCreated views: raw[{}] total_filtered[{}] train[{}] test[{}]\n'.format(
    raw_count,
    filtered_count,
    len(train_view), len(val_view)))

# %%
torch_dataset = FiftyOneTorchDataset(
    fiftyone_dataset=train_view,
    # transforms=torchvision.transforms.Compose([
    #     # torchvision.transforms.Resize([224, 224]),
    #     torchvision.transforms.Resize(224),
    #     torchvision.transforms.CenterCrop(224),
    #     torchvision.transforms.RandomHorizontalFlip(),
    #     torchvision.transforms.ToTensor(),
    # ]),
    transforms=train_transforms,
    # gt_field="ground_truth",
    # classes=None,
    classes=labels_list,
)

# %%
torch_dataset
torch_dataset_val = FiftyOneTorchDataset(val_view, val_transforms, 
        classes=labels_list)

# %%
# loader = torch.utils.data.DataLoader(
#     torch_dataset,
#     batch_size=4,
#     shuffle=False,
#     num_workers=8,
#     # collate_fn=utils.collate_fn,
#     collate_fn=_collate_fn,
# )
# # loader = torch.utils.data.DataLoader(
# #     ds,
# #     batch_size=4,
# #     shuffle=False,
# #     num_workers=8,
# # )
# loader
# for i, data in enumerate(loader):
#     print('\r{}/{}'.format(i, len(loader)).ljust(32), end='')


# %%
def do_training(model, torch_dataset, torch_dataset_val, num_epochs=4, stats_fp='./logs/stats_{}.json'.format(time_stamp)):
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        torch_dataset, batch_size=_bs, shuffle=True, num_workers=8,
        collate_fn=torch_utils.collate_fn)
    
    data_loader_val = torch.utils.data.DataLoader(
        torch_dataset_val, batch_size=1, shuffle=False, num_workers=8,
        collate_fn=torch_utils.collate_fn)
    
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device %s" % device)
    
    # move model to the right device
    model.to(device)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=_lr,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=4,
                                                    gamma=0.3)
    
    _logging_stat = False
    stats = {
        'info': {
            'lr': _lr,
        },
        'telem': {
            'time_stamp': time_stamp,
            'sample_count_train': len(torch_dataset),
            'sample_count_val': len(torch_dataset_val),
            'epochs': num_epochs,
        },
        'logs': [],
    }
    if isinstance(stats_fp, str) and stats_fp.endswith('.json'):
        _dp = os.path.split(stats_fp)[0]
        if not os.path.isdir(_dp):
            os.makedirs(_dp)
        _logging_stat = True
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        _metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        _coco_evaluator = evaluate(model, data_loader_val, device=device)
        
        if _logging_stat:
            train_stats = {str(k): float(v.global_avg) for k, v in _metric_logger.meters.items()}
            val_stats = {
                str(m): {
                    str(k): float(v)
                    for k, v in zip(
                        [f'{t}{a}' for t in ['ap', 'ar'] for a in ['', '50', '75', 's', 'm', 'l']],
                        c.stats,
                    )
                }
                for m, c in _coco_evaluator.coco_eval.items()
            }
            stats['logs'].append({
                'train': train_stats,
                'val': val_stats,
            })
            with open(stats_fp, 'w') as f:
                json.dump(stats, f, indent=4)
        
    return _metric_logger, _coco_evaluator


# %%
num_classes=len(torch_dataset.get_classes())
print('class count:', num_classes)

model = get_model_FRCNN(num_classes)

do_training(model, torch_dataset, torch_dataset_val, num_epochs=20)

# %%