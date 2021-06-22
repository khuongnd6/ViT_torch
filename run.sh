

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia



python main_dino.py --arch "vit_small" --patch_size 8 --out_dim 65536 \
    --norm_last_layer false --warmup_teacher_temp 0.04 --teacher_temp 0.07 \
    --warmup_teacher_temp_epochs 30 --use_fp16 false --weight_decay 0.04 --weight_decay_end 0.4 \
    --clip_grad 3.0 --batch_size_per_gpu 16 --epochs 800 --freeze_last_layer 1 \
    --lr 0.0005 --warmup_epochs 10 --min_lr 1e-06 \
    --local_crops_scale 0.05 0.4 --global_crops_scale 0.4 1.0 \
    --local_crops_number 10 --seed 0 \
    --num_workers 10 \
    --optimizer "adamw" --momentum_teacher 0.996 --use_bn_in_head false \
    --data_path /host/ubuntu/data/imagenet_extract/train


    # --world_size 64 --ngpus 8 --nodes 8 \
    #  \



python main_dino.py --arch "vit_small" --patch_size 8 --out_dim 65536 \
    --norm_last_layer false --warmup_teacher_temp 0.04 --teacher_temp 0.07 \
    --warmup_teacher_temp_epochs 30 --use_fp16 false --weight_decay 0.04 --weight_decay_end 0.4 \
    --clip_grad 3.0 --batch_size_per_gpu 8 --epochs 800 --freeze_last_layer 1 \
    --lr 0.0005 --warmup_epochs 10 --min_lr 1e-06 \
    --local_crops_scale 0.05 0.4 --global_crops_scale 0.4 1.0 \
    --local_crops_number 10 --seed 0 \
    --num_workers 10 \
    --optimizer "adamw" --momentum_teacher 0.996 --use_bn_in_head false \
    --data_path /host/ubuntu/data/imagenet_extract/train




python main.py --epoch 10 --dataset cifar10 --bs 512 --lr 0.001 --arch dino_vits16
python main.py --epoch 10 --dataset cifar10 --bs 1024 --lr 0.001 --arch dino_vits16
python main.py --epoch 10 --dataset cifar10 --bs 2048 --lr 0.001 --arch dino_vits16
python main.py --epoch 10 --dataset cifar10 --bs 4096 --lr 0.001 --arch dino_vits16




python main.py --epoch 5 --dataset stl10 --bs 512 --lr 0.001 --arch dino_vits16
python main.py --epoch 5 --dataset stl10 --bs 1024 --lr 0.001 --arch dino_vits16
python main.py --epoch 5 --dataset stl10 --bs 2048 --lr 0.001 --arch dino_vits16

python main.py --epoch 5 --dataset stl10 --bs 128 --lr 0.001 --arch dino_vits8
python main.py --epoch 5 --dataset stl10 --bs 256 --lr 0.001 --arch dino_vits8
python main.py --epoch 5 --dataset stl10 --bs 512 --lr 0.001 --arch dino_vits8
python main.py --epoch 5 --dataset stl10 --bs 1024 --lr 0.001 --arch dino_vits8

python main.py --epoch 5 --dataset stl10 --bs 128 --lr 0.001 --arch dino_vitb16
python main.py --epoch 5 --dataset stl10 --bs 256 --lr 0.001 --arch dino_vitb16
python main.py --epoch 5 --dataset stl10 --bs 512 --lr 0.001 --arch dino_vitb16

python main.py --epoch 5 --dataset stl10 --bs 64 --lr 0.001 --arch dino_vitb8
python main.py --epoch 5 --dataset stl10 --bs 128 --lr 0.001 --arch dino_vitb8
python main.py --epoch 5 --dataset stl10 --bs 256 --lr 0.001 --arch dino_vitb8



python main.py --epoch 10 --dataset cifar10 --bs 512 --lr 0.001 --arch dino_vits16
python main.py --epoch 10 --dataset cifar10 --bs 1024 --lr 0.001 --arch dino_vits16
python main.py --epoch 10 --dataset cifar10 --bs 2048 --lr 0.001 --arch dino_vits16
python main.py --epoch 10 --dataset cifar10 --bs 4096 --lr 0.001 --arch dino_vits16
python main.py --epoch 10 --dataset cifar10 --bs 8192 --lr 0.001 --arch dino_vits16





