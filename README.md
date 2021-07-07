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
- Finetune
(example with 30 epochs on stl10 from pretrained dino_vitb8 with sgd at 0.001 initial learning rate, 3 fc layers)
```
python main.py --epoch 30 --dataset stl10 --root_path /host/ubuntu/torch --bs 128 --pretrained --arch dino_vitb8 --opt sgd --lr 0.001 --fc 256 128 32
```
- Linear Evaluation
(example with 30 epochs on stl10 from pretrained dino_vitb8 with adamw at 0.001 initial learning rate, 3 fc layers)
```
python main.py --epoch 30 --dataset stl10 --root_path /host/ubuntu/torch --bs 128 --pretrained --arch dino_vitb8 --lineareval --opt adamw --lr 0.001 --fc 256 128 32
```
