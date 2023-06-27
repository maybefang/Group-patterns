python train.py --data_root=/home/data/cifar10 --model VGG16 --batch_size 64 --affix=vgg16_5.5e-3_lr01 --milestones 60 120 --learning_rate=0.01 --gpu=4 --max_epoch=150 --alpha 5.5e-3 --mask &> vgg16_5.5e-3_01lr.txt

python train.py --data_root=/home/data/cifar10 --model VGG16 --batch_size 64 --affix=vgg16_6e-3_lr01 --milestones 60 120 --learning_rate=0.01 --gpu=4 --max_epoch=150 --alpha 6e-3 --mask &> vgg16_6e-3_01lr.txt

python train.py --data_root=/home/data/cifar10 --model VGG16 --batch_size 64 --affix=vgg16_5e-3_lr02_mile60 --milestones 60 --learning_rate=0.02 --gpu=4 --max_epoch=150 --alpha 5e-3 --mask &> vgg16_5e-3_mile60.txt