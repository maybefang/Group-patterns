# python train.py --data_root=/home/data/cifar10 --affix=vgg16_1e-6alpha_lr001 --learning_rate=0.001 --gpu=7 --max_epoch=5 --mask --alpha=1e-6


# python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --affix=tim_vgg16_1e-3alpha_lr01 --model=tim_VGG16 --learning_rate=0.01 --gpu=4 --max_epoch=100 --mask --alpha=1e-3  2>&1|tee 1e-3timvgg75_log.txt;

# python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --affix=tim_vgg16_5e-3alpha_lr01 --model=tim_VGG16 --learning_rate=0.01 --gpu=4 --max_epoch=100 --mask --alpha=5e-3  2>&1|tee 5e-3timvgg50_log.txt;

# python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --affix=tim_vgg16_7e-2alpha_lr01 --model=tim_VGG16 --learning_rate=0.01 --gpu=4 --max_epoch=100 --mask --alpha=7e-3  2>&1|tee 7e-2timvgg25_log.txt;


# #vit

# python train.py --data_root=/home/data/cifar10 --model vit --batch_size 32 --affix=vit_5e-3alpha_lr02 --learning_rate=0.02 --gpu=7 --max_epoch=100 --mask --alpha=5e-3 &> vit_5e-3alpha_lr01.txt

# python train.py --data_root=/home/data/cifar10 --model vit --batch_size 32 --affix=vit_5e-1alpha_lr02 --learning_rate=0.02 --gpu=7 --max_epoch=100 --mask --alpha=5e-1 &> vit_5e-1alpha_lr01.txt

# python train.py --data_root=/home/data/cifar10 --model vit --batch_size 32 --affix=vit_1alpha_lr02 --learning_rate=0.02 --gpu=7 --max_epoch=100 --mask --alpha=1 &> vit_1alpha_lr01.txt

# #tim_vit

# python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --model=tim_vit --learning_rate=0.01 --batch_size=32 \
#                              --affix=tim_vit_5e-3alpha_lr01 --gpu=7 --max_epoch=50 --mask --alpha=5e-3 &> timvit_5e-3alpha_lr01.txt

# python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --model=tim_vit --learning_rate=0.01 --batch_size=32 \
#                              --affix=tim_vit_5e-1alpha_lr01 --gpu=7 --max_epoch=50 --mask --alpha=5e-1 &> timvit_5e-1alpha_lr01.txt

# python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --model=tim_vit --learning_rate=0.01 --batch_size=32 \
#                              --affix=tim_vit_5alpha_lr01 --gpu=7 --max_epoch=50 --mask --alpha=5 &> timvit_5alpha_lr01.txt


#sparse_vit
# python train.py --data_root=/home/data/cifar10 --model sparse_vit --batch_size 32 --affix=sparse_vit_25_lr02 --learning_rate=0.02 --gpu=5 --max_epoch=100 --mask --ratio=0.25 &> sparse_vit_25.txt

#python train.py --data_root=/home/data/cifar10 --model sparse_vit --batch_size 32 --affix=sparse_vit_50_lr02 --learning_rate=0.02 --gpu=5 --max_epoch=100 --mask --ratio=0.5 &> sparse_vit_50.txt

#python train.py --data_root=/home/data/cifar10 --model sparse_vit --batch_size 32 --affix=sparse_vit_75_lr02 --learning_rate=0.02 --gpu=5 --max_epoch=100 --mask --ratio=0.75 &> sparse_vit_75.txt


#saprse_tim_vit 
                              
# python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --model=sparse_tim_vit --learning_rate=0.02 --batch_size=32 \
#                              --affix=sparse_tim_vit_25_lr02 --gpu=5 --max_epoch=50 --mask --ratio=0.25 &> sparse_tim_vit_25.txt

# python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --model=sparse_tim_vit --learning_rate=0.02 --batch_size=32 \
#                              --affix=sparse_tim_vit_50_lr02 --gpu=5 --max_epoch=50 --mask --ratio=0.5 &> sparse_tim_vit_50.txt

# python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --model=sparse_tim_vit --learning_rate=0.02 --batch_size=32 \
#                              --affix=sparse_tim_vit_75_lr02 --gpu=5 --max_epoch=50 --mask --ratio=0.75 &> sparse_tim_vit_75.txt


# python train.py --data_root=/home/data/cifar10 --model ResNet50 --batch_size 64 --affix=resnet50_9e-3_lr02 --learning_rate=0.02 --gpu=6 --max_epoch=100 --mask --alpha 9e-3 &> resnet50_9e-3.txt

# python train.py --data_root=/home/data/cifar10 --model ResNet50 --batch_size 64 --affix=resnet50_9e-4_lr02 --learning_rate=0.02 --gpu=6 --max_epoch=100 --mask --alpha 9e-4 &> resnet50_9e-4.txt

# python train.py --data_root=/home/data/cifar10 --model ResNet50 --batch_size 64 --affix=resnet50_1e-2_lr02 --learning_rate=0.02 --gpu=6 --max_epoch=100 --mask --alpha 1e-2 &> resnet50_1e-2.txt

# python train.py --data_root=/home/data/cifar10 --model ResNet50 --batch_size 64 --affix=resnet50_1e-2_lr02 --learning_rate=0.02 --gpu=6 --max_epoch=100 --mask --alpha 1e-2 &> resnet50_1e-2.txt


# python train.py --data_root=/home/data/cifar10 --model VGG16 --batch_size 64 --affix=vgg16_5.5e-3_lr02 --milestones 60 120 --learning_rate=0.02 --gpu=6 --max_epoch=150 --alpha 5.5e-3 --mask &> vgg16_5.5e-3.txt

# python train.py --data_root=/home/data/cifar10 --model VGG16 --batch_size 64 --affix=vgg16_6e-3_lr02 --milestones 60 120 --learning_rate=0.02 --gpu=6 --max_epoch=150 --alpha 6e-3 --mask &> vgg16_6e-3.txt

# python train.py --data_root=/home/data/cifar10 --model VGG16 --batch_size 64 --affix=vgg16_5e-3_lr01 --milestones 60 120 --learning_rate=0.01 --gpu=6 --max_epoch=150 --alpha 5e-3 --mask &> vgg16_5e-3_01lr.txt

#python train.py --data_root=/home/data/cifar100 --dataset cifar100 --model VGG16 --batch_size 64 --affix=vgg16_cifar100_nop_lr01 --milestones 60 --learning_rate=0.01 \
#                --gpu=0 --max_epoch=100 &> vgg16_cifar100_nop_01lr.txt

#python train.py --data_root=/home/data/cifar100 --dataset cifar100 --model VGG16 --batch_size 64 --affix=vgg16_cifar100_5e-4_lr01 --milestones 60 --learning_rate=0.01 \
#                --gpu=0 --max_epoch=100 --alpha 5e-4 --mask &> vgg16_cifar100_5e-4_01lr.txt

#python train.py --data_root=/home/data/cifar100 --dataset cifar100 --model VGG16 --batch_size 64 --affix=vgg16_cifar100_2e-6_lr01 --milestones 60 --learning_rate=0.01 \
#                --gpu=0 --max_epoch=100 --alpha 2e-6 --mask &> vgg16_cifar100_2e-6_01lr.txt

#python train.py --data_root=/home/data/cifar100 --dataset cifar100 --model VGG16 --batch_size 64 --affix=vgg16_cifar100_5e-3_lr01 --milestones 60 --learning_rate=0.01 \
#                --gpu=0 --max_epoch=100 --alpha 5e-3 --mask &> vgg16_cifar100_5e-3_01lr.txt

#python train.py --data_root=/home/data/cifar100 --dataset cifar100 --model VGG16 --batch_size 64 --affix=vgg16_cifar100_7e-4_lr01 --milestones 60 --learning_rate=0.01 \
#                 --gpu=0 --max_epoch=100 --alpha 7e-4 --mask &> vgg16_cifar100_7e-4_01lr.txt

#python train.py --data_root=/home/data/cifar100 --dataset cifar100 --model VGG16 --batch_size 64 --affix=vgg16_cifar100_7e-3_lr01 --milestones 60 --learning_rate=0.01 \
#                 --gpu=0 --max_epoch=100 --alpha 7e-3 --mask &> vgg16_cifar100_7e-3_01lr.txt

#python train.py --data_root=/home/data/cifar100 --dataset cifar100 --model VGG16 --batch_size 64 --affix=vgg16_cifar100_5e-6_lr01 --milestones 60 --learning_rate=0.01 \
#                 --gpu=0 --max_epoch=100 --alpha 5e-6 --mask &> vgg16_cifar100_5e-6_01lr.txt

#python train.py --data_root=/home/data/cifar100 --dataset cifar100 --model VGG16 --batch_size 64 --affix=vgg16_cifar100_7e-2_lr01 --milestones 60 --learning_rate=0.01 \
#                 --gpu=0 --max_epoch=100 --alpha 7e-2 --mask &> vgg16_cifar100_7e-2_01lr.txt

#python train.py --data_root=/home/data/cifar100 --dataset cifar100 --model VGG16 --batch_size 64 --affix=vgg16_cifar100_1e-1_lr01 --milestones 60 --learning_rate=0.01 \
#                --gpu=0 --max_epoch=100 --alpha 1e-1 --mask &> vgg16_cifar100_1e-1_01lr.txt



# python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --affix=tim_vgg19_7e-3alpha_lr01 --model=tim_VGG19 --learning_rate=0.01 --gpu=0 --max_epoch=100 --mask --alpha=7e-3 &> timvgg19_7e-3.txt

#python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --affix=tim_vgg19_1e-1alpha_lr01 --model=tim_VGG19 --learning_rate=0.01 --gpu=0 --max_epoch=100  &> timvgg19_nop.txt

#python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --affix=tim_vgg19_1e-1alpha_lr01 --model=tim_VGG19 --learning_rate=0.01 --gpu=0 --max_epoch=100 --mask --alpha=1e-1  &> timvgg19_1e-1.txt

#python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --affix=tim_vgg19_5e-2alpha_lr01 --model=tim_VGG19 --learning_rate=0.01 --gpu=0 --max_epoch=100 --mask --alpha=5e-2  &> timvgg19_5e-2.txt

# python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --affix=tim_vgg19_5e-5alpha_lr01 --model=tim_VGG19 --learning_rate=0.01 --gpu=0 --max_epoch=100 --mask --alpha=5e-5  &> timvgg19_5e-5.txt

# python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --affix=tim_vgg19_5e-6alpha_lr01 --model=tim_VGG19 --learning_rate=0.01 --gpu=0 --max_epoch=100 --mask --alpha=5e-6  &> timvgg19_5e-6.txt


python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --affix=tim_resnet50_7e-3alpha_lr01 --model=tim_ResNet50 --learning_rate=0.01 --gpu=0 --max_epoch=100 --mask --alpha=7e-3 &> timresnet50_7e-3.txt

python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --affix=tim_resnet50_1e-1alpha_lr01 --model=tim_ResNet50 --learning_rate=0.01 --gpu=0 --max_epoch=100  &> timresnet50_nop.txt

python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --affix=tim_resnet50_1e-1alpha_lr01 --model=tim_ResNet50 --learning_rate=0.01 --gpu=0 --max_epoch=100 --mask --alpha=1e-1  &> timresnet50_1e-1.txt

python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --affix=tim_resnet50_5e-2alpha_lr01 --model=tim_ResNet50 --learning_rate=0.01 --gpu=0 --max_epoch=100 --mask --alpha=5e-2  &> timresnet50_5e-2.txt

python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --affix=tim_resnet50_5e-4alpha_lr01 --model=tim_ResNet50 --learning_rate=0.01 --gpu=0 --max_epoch=100 --mask --alpha=5e-4  &> timresnet50_5e-4.txt

python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --affix=tim_resnet50_5e-5alpha_lr01 --model=tim_ResNet50 --learning_rate=0.01 --gpu=0 --max_epoch=100 --mask --alpha=5e-5  &> timresnet50_5e-5.txt

python train_tinyimagenet.py --data_root=/home/data/tiny_imagenet/tiny-imagenet-200 --affix=tim_resnet50_5e-6alpha_lr01 --model=tim_ResNet50 --learning_rate=0.01 --gpu=0 --max_epoch=100 --mask --alpha=5e-6  &> timresnet50_5e-6.txt


























