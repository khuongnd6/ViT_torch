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
import utils
import argparse

# %%
time_stamp = time.strftime('%y%m%d_%H%M%S')
time_start = time.time()

# %%
parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=10,)
parser.add_argument('--bs', type=int, default=2)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_schedule_step', type=int, default=4,)
parser.add_argument('--lr_schedule_gamma', type=float, default=0.3,)

parser.add_argument('--train_limit', type=int, default=20000)
parser.add_argument('--val_limit', type=int, default=5000)

parser.add_argument('--data_path', type=str, default='/host/ubuntu/torch/coco2017')

args = parser.parse_args()

# %%
train_limit = args.train_limit
val_limit = args.val_limit
_lr = args.lr
_bs = args.bs

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
class Datasets_COCO2017:
    all_labels_train = [
            # 'background',
            'airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench',
            'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot',
            'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut',
            'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse',
            'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange',
            'oven', 'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich',
            'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign',
            'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush',
            'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']
    
    def __init__(self,
                splits=['train', 'val'],
                labels=None,
                data_path='./data',
                limits=None,
                coco_split_overwrite=None,
                bs=4,
                shuffle=True,
                num_workers=8,
                ):
        self.dataset_name = 'coco-2017'
        self.split_mapping = {
            'train': 'train',
            'val': 'validation',
        }
        self.splits = splits
        self.views = {}
        self.sets = {}
        self.loaders = {}
        self.num_labels = None
        self.data_path = data_path
        transform = {
            _split: T.Compose([T.ToTensor()])
            for _split in self.splits
        }
        self.bs = bs
        
        self.labels = None
        if isinstance(labels, (list)):
            self.labels = [v for v in labels if v in self.all_labels_train]
        if not isinstance(self.labels, list) or len(self.labels) <= 0:
            self.labels = self.all_labels_train
        
        self.num_labels = len(self.labels)
        assert self.num_labels > 0
        
        self.info = {
            'dataset': self.dataset_name,
            'sample_count_raw': {},
            'sample_count_filtered': {},
            'sample_count': {},
            'batch_count': {},
            'num_labels': self.num_labels,
        }
        
        for i, _split in enumerate(self.splits):
            _training = _split == 'train'
            if not isinstance(coco_split_overwrite, str):
                if _split not in self.split_mapping:
                    raise ValueError('split [{}] is not supported with dataset [{}]'.format(
                        _split, self.dataset_name
                    ))
            coco_split = coco_split_overwrite
            coco_split = self.split_mapping[_split]
            _foz_set = foz.load_zoo_dataset(
                name=self.dataset_name,
                split=coco_split,
                # splits=None,
                # label_field=None,
                # dataset_name=None,
                dataset_dir=self.data_path,
                # download_if_necessary=True,
                # drop_existing_dataset=False,
                # overwrite=False,
                # cleanup=True,
            )
            _foz_set.compute_metadata()
            raw_count = len(_foz_set)
            _set = _foz_set
            
            # labels_list = [
            #     'car', 'truck', 'bus', 'bicycle',
            #     'chair', 'person', 'bench',
            #     'bird', 'cat', 'dog',
            # ]
            if _training:
                _set = _set.filter_labels(
                    "ground_truth",
                    F("label").is_in(self.labels),
                )
                # _set = _set.match(
                #     F("predictions.detections").filter(F("label").is_in(self.labels)).length() > 0
                # )
            
            filtered_count = len(_set)
            _limit = raw_count
            if isinstance(limits, int):
                _limit = limits
            elif isinstance(limits, (list, tuple)) and len(limits) > i:
                _limit = limits[i]
            _view = _set.take(_limit, seed=0)
            _view
            self.views[_split] = _view
            torch_dataset = FiftyOneTorchDataset(
                fiftyone_dataset=_view,
                transforms=transform[_split],
                # gt_field="ground_truth",
                # classes=None,
                classes=self.labels,
            )
            _bs = self.bs
            if not _training:
                # lock val/test bs to 1 (for now)
                _bs = 1
            torch_loader = torch.utils.data.DataLoader(
                torch_dataset,
                batch_size=_bs,
                shuffle=shuffle and _training,
                num_workers=num_workers,
                collate_fn=utils.collate_fn,
            )
            
            self.sets[_split] = torch_dataset
            self.loaders[_split] = torch_loader
            self.info['batch_count'][_split] = len(torch_loader)
            self.info['sample_count'][_split] = len(torch_dataset)
            self.info['sample_count_raw'][_split] = raw_count
            self.info['sample_count_filtered'][_split] = filtered_count


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
def do_training(
            model,
            dataloaders=[],
            num_epochs=10,
            stats_fp='./logs/stats_{}.json'.format(time_stamp),
            telem={},
            info={},
            lr=0.001,
            lr_schedule_step=4,
            lr_schedule_gamma=0.3,
            initial_validation=True,
            ):
    
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device %s" % device)
    
    # move model to the right device
    model.to(device)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005,
    )
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_schedule_step,
        gamma=lr_schedule_gamma,
    )
    
    _logging_stat = False
    stats = {
        'info': {**info},
        'telem': {
            'device': str(device),
            **telem,
        },
        'logs': [],
    }
    metric_loggers = []
    coco_evaluators = []
    if isinstance(stats_fp, str) and stats_fp.endswith('.json'):
        _dp = os.path.split(stats_fp)[0]
        if not os.path.isdir(_dp):
            os.makedirs(_dp)
        _logging_stat = True
    
    def save_stats(epoch, metric_logger=None, coco_evaluator=None, time_costs={}):
        _stat = {
            'epoch': epoch,
            **time_costs,
        }
        if metric_logger is not None:
            _stat['train'] = {str(k): float(v.global_avg) for k, v in metric_logger.meters.items()}
        if coco_evaluator is not None:
            _stat['val'] = {
                str(m): {
                    str(k): float(v)
                    for k, v in zip(
                        [f'{t}{a}' for t in ['ap', 'ar'] for a in ['', '50', '75', 's', 'm', 'l']],
                        c.stats,
                    )
                }
                for m, c in coco_evaluator.coco_eval.items()
            }
        stats['logs'].append(_stat)
        with open(stats_fp, 'w') as f:
            json.dump(stats, f, indent=4)
        
    
    for epoch in range(num_epochs):
        
        if epoch == 0 and initial_validation and len(dataloaders) >= 2:
            _coco_evaluator = evaluate(model, dataloaders[1], device=device)
            if _logging_stat:
                save_stats(
                    epoch=0,
                    # metric_logger=_metric_logger,
                    coco_evaluator=_coco_evaluator,
                    time_costs={},
                )
        
        time_costs = {
            'time_start': time.time(),
            'time_finish': None,
            'time_cost': None,
        }
        for dl, _training in zip(dataloaders, [True, False]):
            _time_start = time.time()
            # if epoch == -1 and initial_validation:
            if _training:
                _metric_logger = train_one_epoch(model, optimizer, dl, device, epoch, print_freq=10)
                lr_scheduler.step()
                time_costs['time_train'] = time.time() - _time_start
            else:
                _coco_evaluator = evaluate(model, dl, device=device)
                time_costs['time_val'] = time.time() - _time_start
        time_costs['time_finish'] = time.time()
        time_costs['time_cost'] = time_costs['time_finish'] - time_costs['time_start']
        
        metric_loggers.append(_metric_logger)
        coco_evaluators.append(_coco_evaluator)
        
        if _logging_stat:
            save_stats(
                epoch=epoch + 1,
                metric_logger=_metric_logger,
                coco_evaluator=_coco_evaluator,
                time_costs=time_costs,
            )
        
    return metric_loggers, coco_evaluators


# %%
ds = Datasets_COCO2017(
    splits=['train', 'val'],
    labels=None,
    data_path=args.data_path,
)
print('loaded dataset [{}]: {}'.format(ds.dataset_name, json.dumps(ds.info, indent=4)))

model = get_model_FRCNN(ds.num_labels)

do_training(
    model=model,
    dataloaders=[ds.loaders[k] for k in ['train', 'val']],
    num_epochs=args.epoch,
    stats_fp='./logs/stats_{}.json'.format(time_stamp),
    info={**args.__dict__},
    telem={},
    lr=args.lr,
    lr_schedule_step=args.lr_schedule_step,
    lr_schedule_gamma=args.lr_schedule_gamma,
    # initial_validation=True,
)

# %%