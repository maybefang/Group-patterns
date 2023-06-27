import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn

from train_argument import parser, print_args

from time import time
from utils import * 
from models import *
from trainer import *


def main(args):
    save_folder = args.affix
    data_dir = args.data_root

    #log_folder = os.path.join(args.log_root, save_folder)
    log_folder = os.path.join(args.model_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, 'train', 'info')
    print_args(args, logger)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    #if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    
    if args.model == "tim_VGG16":
        net = tim_vgg(dataset=args.dataset, depth=16)
        if args.mask:
            net = tim_masked_vgg(dataset=args.dataset, depth=16)
    elif args.model == "tim_VGG19":
        net = tim_vgg(dataset=args.dataset, depth=19)
        if args.mask:
            net = tim_masked_vgg(dataset=args.dataset, depth=19)
    elif args.model == "tim_ResNet18":
        net = tim_ResNet18(dataset=args.dataset)
        if args.mask:
            net = tim_masked_ResNet18(dataset=args.dataset)
    elif args.model == "tim_ResNet50":
        net = tim_ResNet50(dataset=args.dataset)
        if args.mask:
            net = tim_masked_ResNet50(dataset=args.dataset)
    elif args.model == "tim_vit":
        net = vit_tiny_patch16_64()
        if args.mask:
            net = masked_vit_tiny_patch16_64()
    elif args.model == "sparse_tim_vit":
        net = sparse_vit_tiny_patch16_64(ratio = args.ratio)
   
    net.to(device)
    
    trainer = Trainer(args, logger)
    
    loss = nn.CrossEntropyLoss()
 
    
    kwargs = {'num_workers': 3, 'pin_memory': True} #if torch.cuda.is_available() else {}
    
    
    from tiny_imagenet_dataset import TinyImageNet
    dataset_train = TinyImageNet(args.data_root, split='train', transform=transforms.Compose([
                            #transforms.Pad(4),
                            #transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ]))
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)


    dataset_val = TinyImageNet(args.data_root, split='val', transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ]))
    test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, **kwargs)
        
    
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    if 'sparse' in args.model:
        trainer.sparse_train(net, loss, device, train_loader, test_loader, optimizer=optimizer, scheduler=scheduler)
    else:
        trainer.train(net, loss, device, train_loader, test_loader, optimizer=optimizer, scheduler=scheduler)
    


if __name__ == '__main__':
    args = parser()
    #print_args(args)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
