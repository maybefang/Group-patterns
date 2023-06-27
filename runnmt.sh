python train_multi30k.py --data_root=/home/lyj --model=nmt --affix=nmt_5e-1alpha_lr02 --learning_rate=0.02 --batch_size=128 --milestones 60 80 --gpu=3 --max_epoch=100 --mask --alpha=5e-1 2>&1|tee nmt_log5e-1.txt

python train_multi30k.py --data_root=/home/lyj --model=nmt --affix=nmt_5e-3alpha_lr02 --learning_rate=0.02 --batch_size=128 --milestones 60 80 --gpu=3 --max_epoch=100 --mask --alpha=5e-3 2>&1|tee nmt_log5e-3.txt

python train_multi30k.py --data_root=/home/lyj --model=nmt --affix=nmt_5e-5alpha_lr02 --learning_rate=0.02 --batch_size=128 --milestones 60 80 --gpu=3 --max_epoch=100 --mask --alpha=5e-5 2>&1|tee nmt_log5e-5.txt

python train_multi30k.py --data_root=/home/lyj --model=nmt --affix=nmt_5e-7alpha_lr02 --learning_rate=0.02 --batch_size=128 --milestones 60 80 --gpu=3 --max_epoch=100 --mask --alpha=5e-7 2>&1|tee nmt_log5e-7.txt


