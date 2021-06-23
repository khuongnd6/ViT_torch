# %%
from numpy.lib.arraysetops import isin
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
from torch._C import Value
import torchvision
import torchvision.transforms as transforms
# from utils_datasets import Cutout, CIFAR10Policy, ImageNetPolicy

# %%
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.
        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.
        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SVHNPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.
        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),

            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),

            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img


# %%
class Datasets:
    _datasets_config = {
        'stl10': {
            'dataset_fn': torchvision.datasets.STL10,
            'norm_values': {
                'mean': [0.44671062065972217, 0.43980983983523964, 0.40664644709967324],
                'std': [0.2603409782662331, 0.25657727311344447, 0.27126738145225493],
            },
            'num_labels': 10,
            'transform': {
                'train': [
                    transforms.RandomCrop(96, padding=4, fill=128), # fill parameter needs torchvision installed from source
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ],
                'test': [
                    transforms.ToTensor(),
                ],
            },
            'split': {
                'train': {'split': 'train'},
                'test': {'split': 'test'},
            },
            'shape': [96, 96, 3],
        },
        'cifar10': {
            'dataset_fn': torchvision.datasets.CIFAR10,
            'norm_values': {
                'mean': [0.4914, 0.4822, 0.4465],
                'std': [0.247, 0.243, 0.261],
            },
            'num_labels': 10,
            'transform': {
                'train': [
                    transforms.RandomCrop(32, padding=4, fill=128), # fill parameter needs torchvision installed from source
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    Cutout(n_holes=1, length=16), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
                ],
                'test': [
                    transforms.ToTensor(),
                ],
            },
            'split': {
                'train': {'train': True},
                'test': {'train': False},
            },
        },
        'cifar100': {
            'dataset_fn': torchvision.datasets.CIFAR100,
            'norm_values': {
                'mean': [0.50707516,  0.48654887,  0.44091784],
                'std': [0.26733429,  0.25643846,  0.27615047],
            },
            'num_labels': 100,
            'transform': {
                'train': [
                    transforms.RandomCrop(32, padding=4, fill=128), # fill parameter needs torchvision installed from source
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    Cutout(n_holes=1, length=16), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
                ],
                'test': [
                    transforms.ToTensor(),
                ],
            },
            'split': {
                'train': {'train': True},
                'test': {'train': False},
            },
        },
        # 'WorkInProgress_imagenet': {
        #     'dataset_fn': torchvision.datasets.ImageFolder,
        #     'norm_values': {
        #         'mean': [0.485, 0.456, 0.406],
        #         'std': [0.229, 0.224, 0.225],
        #     },
        #     'num_labels': 1000,
        #     'transform': {
        #         'train': [
        #             transforms.RandomHorizontalFlip(),
        #             ImageNetPolicy(),
        #         ],
        #         'test': [],
        #     },
        # },
        # '<other>': {
        #     'norm_values': {
        #         'mean': [0.5, 0.5, 0.5],
        #         'std': [0.5, 0.5, 0.5],
        #     },
        #     'num_labels': 100,
        # },
    }
    
    def __init__(self,
                dataset='stl10',
                root_path=None,
                batchsize=128,
                transform=[],
                download=True,
                shuffle=True,
                num_workers=4,
                splits=['train', 'test'],
                limit_train=0,
                limit_test=0,
                ):
        self.dataset = str(dataset).lower()
        if self.dataset not in self._datasets_config:
            raise ValueError('[ERROR] dataset [{}] is currently not supported!\nUse one of [{}]'.format(
                self.dataset,
                '|'.join(list(self._datasets_config.values()))
            ))
        self.root_path = root_path
        self.bs = batchsize
        if isinstance(self.bs, int):
            self.bs = [self.bs for _ in range(len(splits))]
        if isinstance(self.bs, (list, tuple)):
            assert len(self.bs) == len(splits)
            self.bs = {
                k: v
                for v, k in zip(self.bs, splits)
            }
        self.config = self._datasets_config[self.dataset]
        self.num_labels = int(self.config['num_labels'])
        self.sets = {}
        self.loaders = {}
        self.info = {
            'dataset': self.dataset,
            'batch_count': {},
            'sample_count': {},
            'num_labels': self.num_labels,
        }
        self.limits = {
            'train': limit_train,
            'test': limit_test,
        }
        for _split in splits:
            # _ds_kwargs = self._get_ds_kwargs(
            #     transform=self.config['transform'][_split],
            #     norm_values=self.config['norm_values'],
            #     download=self.download,
            #     root=self.root_path,
            #     **self.config['split'][_split],
            # )
            # _set = self.config['dataset_fn'](**_ds_kwargs)
            # _loader = torch.utils.data.DataLoader(
            #     _set,
            #     batch_size=self.bs[_split],
            #     shuffle=bool(shuffle),
            #     num_workers=num_workers,
            # )
            # _batch_count = len(_loader)
            # _sample_count = len(_set)
            
            
            r = self._get_dataset(
                config=self.config,
                split=_split,
                download=bool(download),
                root_path=self.root_path,
                shuffle=(shuffle and _split == 'train'),
                num_workers=num_workers,
                bs=self.bs[_split],
                limit=self.limits[_split],
            )
            
            self.sets[_split] = r['set']
            self.loaders[_split] = r['loader']
            self.info['batch_count'][_split] = r['batch_count']
            self.info['sample_count'][_split] = r['sample_count']
        
    @classmethod
    def download_and_prepare(cls, dataset=None, path='./data'):
        if dataset is None:
            dataset = list(cls._datasets_config.keys())
        if isinstance(dataset, list):
            return [cls.download_and_prepare(v) for v in dataset]
        if dataset in cls._datasets_config:
            _config = cls._datasets_config[dataset]
            for _split in ['train', 'test']:
                _ = cls._get_dataset(
                    config=_config,
                    split='train',
                    download=True,
                    root_path=path,
                    shuffle=False,
                    num_workers=4,
                    bs=512,
                )
                del _
            return True
        else:
            print('Skipped [{}], not supported'.format(dataset))
    
    @classmethod
    def _get_ds_kwargs(cls,
                transform=[transforms.ToTensor()],
                norm_values=None,
                download=True,
                **kwargs,
                ):
        
        if transform is None:
            transform = []
        
        if norm_values is None:
            norm_values = {
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5],
            }
        
        _ds_kwargs = {
            **kwargs,
            'transform': transforms.Compose([
                *transform,
                transforms.Normalize(**norm_values),
            ]),
        }
        return _ds_kwargs
    
    @classmethod
    def _get_dataset(cls, config, split, download, root_path, shuffle, num_workers=4, bs=128, limit=0):
        _ds_kwargs = cls._get_ds_kwargs(
            transform=config['transform'][split],
            norm_values=config['norm_values'],
            download=download,
            root=root_path,
            **config['split'][split],
        )
        _set = config['dataset_fn'](**_ds_kwargs)
        
        if isinstance(limit, int) and limit > 0:
            _set = torch.utils.data.Subset(_set, torch.arange(limit))
        _loader = torch.utils.data.DataLoader(
            _set,
            batch_size=bs,
            shuffle=bool(shuffle),
            num_workers=num_workers,
        )
        _batch_count = len(_loader)
        _sample_count = len(_set)
        return {
            'set': _set,
            'loader': _loader,
            'batch_count': _batch_count,
            'sample_count': _sample_count,
        }


# %%
if __name__ == '__main__':
    ds = Datasets(
        dataset='cifar10',
        root_path='/host/ubuntu/torch',
        batchsize=[128, 128],
        transform=[],
        download=True,
        splits=['train', 'test'],
        shuffle=True,
        num_workers=4,
    )
    ds.info


# %%

