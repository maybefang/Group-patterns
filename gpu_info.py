import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn
import numpy as np

from train_argument import parser, print_args

from time import time
from utils import *
from inference_models import *
#from models import *
#from trainer import *
import pynvml

import argparse



def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()

def watch_gpu(gpu, csv_writer):
    
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    i=0
    
    while True:
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        csv_writer.writerow([i, meminfo.total, meminfo.used, meminfo.free,
                              utilization.memory, utilization.gpu])
        i+=1
        # print(i,meminfo.total, meminfo.used, meminfo.free,
        #                       utilization.memory, utilization.gpu)
        time.sleep(0.001)

def watch_gpu_main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu number",
    )

    parser.add_argument(
        "--filename",
        default='gpu_info.csv',
        type=str,
        help="Path to record gpu info",
    )

    args = parser.parse_args()

    device = torch.device("cuda")
    torch.cuda.set_device(args.gpu)

    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    
    net = masked_vgg(dataset='cifar10', depth=16)
    #net = vgg(dataset='cifar10', depth=16)
    # net = masked_vgg(dataset='cifar10', depth=13)
    # net = masked_ResNet20(dataset='cifar10')
    # net = masked_ResNet18(dataset='cifar10')

    # model_dir = 'checkpoint/vgg16_5e-3alpha_lr02_bn64/best_keepratio_model.pth'  #25.57
    # model_dir = 'checkpoint/vgg16_5e-5alpha_lr02_bn64/best_acc_model.pth'  #50(49.9)
    model_dir = 'checkpoint/vgg16_1.5e-6alpha_lr02_bn64_bak75/best_acc_model.pth'  #75(73.84)
    #model_dir = 'checkpoint/vgg13_5e-5alpha_lr02_bn64/best_keepratio_model.pth'
    # model_dir = 'checkpoint/resnet18_1e-2alpha_lr02_bn128/best_keepratio_model.pth'#39.35
    # model_dir = 'checkpoint/resnet18_5e-4alpha_lr02_bn128/best_acc_model.pth' #46.76
    # model_dir = 'checkpoint/resnet18_2e-6alpha_lr02_bn128/best_keepratio_model.pth' #77.27
    # model_dir = 'checkpoint/vgg16_1.5e-6alpha_lr02_bn64_bak75/best_acc_model.pth'
    #model_dir = 'checkpoint/vgg16_nopruning/best_acc_model.pth'
    #model_dir = os.path.join(model_folder, "best_acc_model.pth")
    net.load_state_dict(
            torch.load(model_dir, map_location=lambda storage, loc: storage))

    #device = torch.device("cuda")
    #torch.cuda.set_device(args.gpu)

    net.init_model()
    net.to(device)
    
    #for l in net.modules():
    #    if isinstance(l,nn.BatchNorm2d):
    #        print(l.state_dict())

    file_name = 'checkpoint/vgg16_5e-3alpha_lr02_bn64/my_model.pth'
    #torch.save(net, file_name)
    
    input_data = torch.randn(64,3,32,32).cuda()
    for i in range(1):
        out = net(input_data)
    
    import csv
    
    f=open(args.filename,'w',encoding='gbk', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["time(s)", "total_mem", "ues_mem", "free_mem", "utilization(mem)", "utilization(gpu)"])
    
    from multiprocessing import Process

    process = Process(target=watch_gpu, args=(args.gpu, csv_writer))
    torch.cuda.synchronize()
    process.start()
    time.sleep(0.5)

    out = net(input_data)

    torch.cuda.synchronize()
    #print(re.shape)
    time.sleep(2)
    process.terminate()
    f.close()
    
    print(f"save the gpu info in {args.filename}.")
    


if __name__ == '__main__':
    # print_args(args)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main()
