# ViT_torch
Repos for reproducing and experiments with Vision Transformer in PyTorch

## ViT Architectures
Currently supporting:
- All DINO Transformer backbones (S/16, S/8, B/16, B/8)
- All CAIT Transformer backbones
- Resnet CNN: resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
- Fast/FasterRCNN
- Swin-based Transformer (work in progress)
## Datasets/Tasks
### Classification:
- STL-10
- Cifar10, Cifar100
- ImageNet1k (coming soon)
### Object Detection:
- COCO-2017
## Installation
- Install python>=3.8
```
python -m pip install -r requirements.txt
```
## Run - Classification
- Finetune
```bash
# (example with 30 epochs on stl10 from pretrained dino_vitb8 with sgd at 0.001 initial learning rate, 4 fc layers)

python main.py --epoch 30 --dataset stl10 --root_path /host/ubuntu/torch --bs 128 --pretrained --arch dino_vitb8 --opt sgd --lr 0.001 --fc 256 128 32

# (example with 50 epochs on cifar10 from pretrained dino_vitb8 with sgd at 0.01 initial learning rate, 1 fc layers)

python main.py --epoch 50 --dataset cifar10 --root_path /host/ubuntu/torch --bs 128 --pretrained --arch dino_vits16 --opt sgd --lr 0.01
```
- Linear Evaluation
```bash
# (example with 30 epochs on stl10 from pretrained dino_vitb8 with adamw at 0.001 initial learning rate, 3 fc layers)

python main.py --epoch 30 --dataset stl10 --root_path /host/ubuntu/torch --bs 128 --pretrained --arch dino_vitb8 --lineareval --opt adamw --lr 0.001 --fc 256 128 32
```
