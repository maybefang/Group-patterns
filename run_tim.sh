python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --model=sparse_tim_vit --learning_rate=0.02 --batch_size=32 \
                             --affix=sparse_tim_vit_25_lr02 --gpu=6 --max_epoch=50 --mask --ratio=0.25 &> sparse_tim_vit_25.txt

python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --model=sparse_tim_vit --learning_rate=0.02 --batch_size=32 \
                             --affix=sparse_tim_vit_50_lr02 --gpu=6 --max_epoch=50 --mask --ratio=0.5 &> sparse_tim_vit_50.txt

python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --model=sparse_tim_vit --learning_rate=0.02 --batch_size=32 \
                             --affix=sparse_tim_vit_75_lr02 --gpu=6 --max_epoch=50 --mask --ratio=0.75 &> sparse_tim_vit_75.txt
