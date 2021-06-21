

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




python torch_data.py --epoch 10 --dataset cifar10 --bs 512 --lr 0.001 --arch dino_vits16



    ('device',  'cuda',  str,  ['cuda', 'cpu']),
    ('epoch',  10,  int),
    ('dataset', 'cifar10', str, ['cifar10', 'cifar100']),
    (['batchsize', 'bs'], 512, int),
    ('data_path', '/host/ubuntu/torch'),
    (['arch', 'a'], 'dino_vits16', str, ['dino_vits16']),
    (['learningrate', 'lr'], 0.001),
    ('stats_json', './logs/stats_latest.json', str)
