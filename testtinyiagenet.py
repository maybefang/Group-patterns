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


def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    
    # net = masked_vit_tiny_patch16_64()
    net = masked_vgg(dataset='tiny_imagenet', depth=16)
    # net = masked_vgg2015(dataset='tiny_imagenet', depth=16)
    # net = masked_ResNet18_2015(dataset='tiny_imagenet')
    # net = masked_ResNet18(dataset='tiny_imagenet')
    # net = masked_ResNet50(dataset='tiny_imagenet')

    
    # model_dir = 'checkpoint/sparse_tim_vit_75_lr02/best_acc_model.pth'
    # model_dir = 'checkpoint/timresnet50_1e-2_lr02/best_acc_model.pth'
    model_dir = 'checkpoint/tim_vgg16_7e-2alpha_lr01/best_keepratio_model.pth'  #30.81
    # model_dir = 'checkpoint/tim_vgg16_1e-3alpha_lr01/best_keepratio_model-92.pth'  #50(49.23)
    # model_dir = 'checkpoint/tim_vgg16_7e-2alpha_lr01/best_keepratio_model-36.pth'  #50(49.16)
    # model_dir = 'checkpoint/tim_vgg16_1e-5alpha_lr01/best_keepratio_model-45.pth'  #75(71.74)
    
    # model_dir = 'checkpoint/tim_resnet18_1e-1alpha_lr01/best_acc_model.pth'#29.55
    # model_dir = 'checkpoint/tim_resnet18_1e-3alpha_lr01/best_acc_model.pth' #49.23
    # model_dir = 'checkpoint/tim_resnet18_1e-6alpha_lr01/best_keepratio_model-32.pth' #75.4
    # model_dir = 'checkpoint/tim_resnet18_1e-3alpha_lr01/best_keepratio_model-4.pth' #74.74

    # model_dir = 'checkpoint/sparse_tim_vit_25_lr02/best_acc_model.pth'
    # model_dir = 'checkpoint/sparse_tim_vit_50_lr02/best_acc_model.pth'
    # model_dir = 'checkpoint/sparse_tim_vit_75_lr02/best_acc_model.pth'


    # model_dir = 'checkpoint/vgg16_1.5e-6alpha_lr02_bn64_bak75/best_acc_model.pth'
    #model_dir = 'checkpoint/vgg16_nopruning/best_acc_model.pth'
    #model_dir = os.path.join(model_folder, "best_acc_model.pth")
    net.load_state_dict(
            torch.load(model_dir, map_location=lambda storage, loc: storage), strict=False)
    

    #device = torch.device("cuda")
    #torch.cuda.set_device(args.gpu)

    net.init_model()
    net.to(device)
    
    #for l in net.modules():
    #    if isinstance(l,nn.BatchNorm2d):
    #        print(l.state_dict())

    file_name = 'checkpoint/vgg16_5e-3alpha_lr02_bn64/my_model.pth'
    #torch.save(net, file_name)
    '''
    data_dir = '/home/data/tiny_imagenet/tiny-imagenet-200'
    
    from tiny_imagenet_dataset import TinyImageNet
    
    kwargs = {'num_workers': 0, 'pin_memory': True}
    dataset_val = TinyImageNet(data_dir, split='val', transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ]))
    test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=64, shuffle=True, **kwargs)
    for data, label in test_loader:
        input_data, label = data.to(device), label.to(device)
        break
    '''
    input_data = torch.randn(16,3,64,64).cuda()
    
    
    for i in range(10):
        out = net(input_data)
    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    rep=100
    all_time=0
    for i in range(rep):
        start.record()
        out = net(input_data)
        end.record()
        torch.cuda.synchronize()
        all_time += start.elapsed_time(end)
    time = all_time/rep
    print(time)

    '''
    total_acc = 0.0
    num = 0
    net.eval()
    loss = nn.CrossEntropyLoss()
    std_loss = 0. 
    iteration = 0.
    with torch.no_grad():
        for data, label in loader:#for j in range(rep):
            data, label = data.to(device), label.to(device)
            start.record()
            output = net(data)
            end.record()
            torch.cuda.synchronize()
            all_time += start.elapsed_time(end)

            pred = torch.max(output, dim=1)[1]
            te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum') 
            total_acc += te_acc
            num += output.shape[0]  
            std_loss += loss(output, label)
            iteration += 1
    std_acc = total_acc/num*100.
    std_loss /= iteration
    print("Test accuracy {:.2f}%, Test loss {:.3f}".format(std_acc, std_loss))

    my_time=all_time/num
    print(my_time) 
    '''

if __name__ == '__main__':
    # print_args(args)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main()
