python train.py --data_root=/home/data/cifar10 --model sparse_vgg16 --batch_size 64 --affix=sparse_vgg16_10_lr02 --milestones 100 --learning_rate=0.02 --gpu=7 --max_epoch=200 --ratio 0.1 &> sparse_vgg16_10_lr02.txt

python train.py --data_root=/home/data/cifar10 --model sparse_vgg16 --batch_size 64 --affix=sparse_vgg16_5_lr02 --milestones 100 --learning_rate=0.02 --gpu=7 --max_epoch=200 --ratio 0.05 &> sparse_vgg16_5_lr02.txt

#python train.py --data_root=/home/data/cifar10 --model sparse_vgg16 --batch_size 64 --affix=sparse_vgg16_7_5_lr02 --milestones 100 --learning_rate=0.02 --gpu=7 --max_epoch=200 --ratio 0.075 &> sparse_vgg16_7_5_lr02.txt





