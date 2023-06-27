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
# from inference_models.inference import MaskedConv2d,MaskedMLP
from inference_models.inference_cusparse import MaskedConv2d,MaskedMLP #cusparse
from inference_models.inference_2015 import MaskedConv2d2015


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()

def print_layer_keep_ratio(model):
    #ratio = 0_number/1_number
    total = 0. 
    keep = 0.
    for layer in model.modules():
        if isinstance(layer, MaskedMLP) or isinstance(layer,MaskedDEMLP):
            # print("+++++++++++++++++++++++sum MLP++++++++++++++++++++")

            abs_weight = torch.abs(layer.weight)
            #threshold = layer.threshold.view(abs_weight.shape[0], -1)
            threshold = layer.threshold
            abs_weight = abs_weight-threshold
            mask = layer.step(abs_weight)
            ratio = torch.sum(mask) / mask.numel()
            total += mask.numel()
            keep += torch.sum(mask)
            # logger.info("Layer threshold {:.4f}".format(layer.threshold[0]))
            print("{}, keep ratio {:.4f}".format(layer, ratio))
            '''
            weight_shape = layer.weight.shape
            mask = layer.mask.repeat(weight_shape[1], 1)
            ratio = torch.sum(mask) / mask.numel()
            total += mask.numel()
            keep += torch.sum(mask)
            logger.info("use mask {}, keep ratio {:.4f}".format(layer, ratio))
            '''
        elif isinstance(layer, MaskedConv2d) or isinstance(layer, MaskedConv2d2015):
            # print("+++++++++++++++++++++++sum conv2d++++++++++++++++++++")
            weight_shape = layer.weight.shape
            #threshold = layer.threshold.view(weight_shape[0], -1) #layer.threshold.shape=[1]
            threshold = layer.threshold
            weight = torch.abs(layer.weight)
            weight = weight.view(weight_shape[0], -1)
            weight = weight - threshold
            mask = layer.step(weight)
            ratio = torch.sum(mask) / mask.numel()
            total += mask.numel()
            keep += torch.sum(mask)
            #logger.info("Layer threshold {:.4f}".format(layer.threshold[0]))
            print("{}, keep ratio {:.4f}".format(layer, ratio))
            '''
            weight_shape = layer.weight.shape
            mask = layer.mask.repeat(weight_shape[0], 1, 1, 1)
            ratio = torch.sum(mask) / mask.numel()
            total += mask.numel()
            keep += torch.sum(mask)
            logger.info("{}, keep ratio {:.4f}".format(layer, ratio))
            '''
        elif isinstance(layer, MaskedGRU) or isinstance(layer, MaskedbiGRU):
            wlist = [i for i in layer.named_parameters()]
            # tid=0
            # threshold = layer.threshold
            for i in wlist:
                # if 'weight' in i[0] and 'mask' not in i[0]:
                #     # wtemp = getattr(self,i)
                #     # weight.append(get_weight(wtemp,tid,i))
                #     weight_shape = i[1].shape
                #     #threshold = layer.threshold.view(weight_shape[0], -1) #layer.threshold.shape=[1]
                #     weight = torch.abs(i[1])
                #     weight = weight.view(weight_shape[0], -1)
                #     weight = weight - threshold[tid]
                #     mask = layer.step(weight)
                #     ratio = torch.sum(mask) / mask.numel()
                #     total += mask.numel()
                #     keep += torch.sum(mask)
                #     tid+=1
                if 'weight' in i[0] and 'mask' in i[0]:
                    # mask = layer.getattr(i[0])
                    ratio = torch.sum(i[1]) / i[1].numel()
                    total += i[1].numel()
                    keep += torch.sum(i[1])
            #logger.info("Layer threshold {:.4f}".format(layer.threshold[0]))
            print("{}, keep ratio {:.4f}".format(layer, ratio))
    print("Model keep ratio {:.4f}".format(keep/total))
    return keep / total

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    
    #net = masked_vgg_imagenet(dataset='cifar10', depth=16)
    net = masked_vit_cifar_patch4_32()
    # net = masked_vgg(dataset='cifar10', depth=16)
    # net = masked_vgg2015(dataset='cifar10', depth=16)
    #net = vgg(dataset='cifar10', depth=16)
    #net = masked_vgg(dataset='cifar10', depth=13)
    # net = masked_ResNet20(dataset='cifar10')
    # net = masked_ResNet18(dataset='cifar10')
    # net = masked_ResNet50(dataset='cifar10')

    ###############vgg16#########################
    # model_dir = 'checkpoint/vgg16_1.5e-6alpha_lr02_bn64_bak75/best_acc_model.pth'  #vgg16 75
    # model_dir = 'checkpoint/vgg16_5e-3alpha_lr02_bn64/best_keepratio_model.pth'  #vgg16 25
    # model_dir = 'checkpoint/vgg16_5e-5alpha_lr02_bn64/best_acc_model.pth'  #vgg16 50

    # model_dir = 'checkpoint/vit_5e-3alpha_lr02/best_acc_model.pth'  #75(73.84)
    # model_dir = 'checkpoint/sparse_vit_25_lr02/best_acc_model.pth'
    # model_dir = 'checkpoint/sparse_vit_50_lr02/best_acc_model.pth'
    # model_dir = 'checkpoint/sparse_vit_75_lr02/best_acc_model.pth'


    # model_dir = 'checkpoint/sparse_vgg19_50_lr02/best_acc_model.pth'
    # model_dir = 'checkpoint/resnet50_1e-2_lr02/best_keepratio_model-1.pth'

    # model_dir = 'checkpoint/resnet18_2e-6alpha_lr02_bn128/best_keepratio_model.pth'#77.27
    # model_dir = 'checkpoint/resnet18_5e-4alpha_lr02_bn128/best_acc_model.pth' #46.76
    # model_dir = 'checkpoint/resnet18_1e-2alpha_lr02_bn128/best_keepratio_model.pth' #39.35

    # model_dir = 'checkpoint/resnet50_7e-4_lr02/best_acc_model.pth'#50
    # model_dir = 'checkpoint/sparse_vgg16_5_lr01/best_acc_model.pth'
    # model_dir = 'checkpoint/no_prune_vit/best_acc_model.pth'
    #model_dir = 'checkpoint/vgg16_nopruning/best_acc_model.pth'
    #model_dir = os.path.join(model_folder, "best_acc_model.pth")
    net.load_state_dict(
            torch.load(model_dir, map_location=lambda storage, loc: storage), strict=False)

    #device = torch.device("cuda")
    #torch.cuda.set_device(args.gpu)
    # print(net)
    # print_layer_keep_ratio(net)

    net.init_model()
    
    net.to(device)
    getModelSize(net)
    #for l in net.modules():
    #    if isinstance(l,nn.BatchNorm2d):
    #        print(l.state_dict())

    file_name = 'checkpoint/vit_5e-3alpha_lr02/my_model.pth'
    #torch.save(net, file_name)
    '''
    data_dir = '/home/data/cifar10'
    
    kwargs = {'num_workers': 0, 'pin_memory': True}
    loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=100, shuffle=True, **kwargs)
    '''
    input_data = torch.randn(16,3,32,32).cuda()
    for i in range(5):
        out = net(input_data)
    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    rep=100
    all_time=0
    start.record()
    for i in range(rep):
        # start.record()
        out = net(input_data)
        # end.record()
        # torch.cuda.synchronize()
        # all_time += start.elapsed_time(end)
    end.record()
    torch.cuda.synchronize()
    all_time = start.elapsed_time(end)
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
    # net = ResNet50()
    # getModelSize(net)
    # net = ResNet18()
    # getModelSize(net)
