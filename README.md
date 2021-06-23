# ViT_torch
repos for all things Vision Transformer in PyTorch

## ViT Architectures
Currently supporting:
- All DINO backbones (S/16, S/8, B/16, B/8)
## Datasets/Tasks
Currently supporting:
- STL-10
- Cifar10, Cifar100
## Installation
- Install python>=3.8
```
python -m pip install -r requirements.txt
```
## Run
- Finetune (example)
```
python main.py --epoch 50 --dataset stl10 --bs 2048 --lr 0.002 --arch dino_vits16
```
- Linear Evaluation (example)
```
python main.py --epoch 50 --dataset stl10 --bs 2048 --lr 0.002 --arch dino_vits16 --lineareval 1
```
