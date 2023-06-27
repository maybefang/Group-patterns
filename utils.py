import os
import sys
import time
import math
import json
import logging
import pickle


import torch
import torch.nn as nn
import torch.nn.init as init

import numpy as np
from models import MaskedMLP, MaskedConv2d, MaskedbiGRU, MaskedGRU, MaskedDEMLP


def list2cuda(_list):
    array = np.array(_list)
    return numpy2cuda(array)

def numpy2cuda(array):
    tensor = torch.from_numpy(array)

    return tensor2cuda(tensor)

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor

def one_hot(ids, n_class):
    assert len(ids.shape) == 1, 'the ids should be 1-D'

    out_tensor = torch.zeros(len(ids), n_class)

    out_tensor.scatter_(1, ids.cpu().unsqueeze(1), 1.)

    return out_tensor
    
def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()


def create_logger(save_path='', file_type='', level='debug'):

    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_model_with_mask(model, file_name, maskname):
    model.load_state_dict(
            torch.load(file_name, map_location=lambda storage, loc: storage))
    mask_val = pickle.load(maskname)
    for l in model.modules:
        if isinstance(model.modules[l], MaskedConv2d):
            model.modules[l].mask = mask_val[l]


def save_model(model, file_name, maskname):
    torch.save(model.state_dict(), file_name)
    mask_val = {}
    for l in model.modules():
        if isinstance(l, MaskedConv2d) or isinstance(l, MaskedMLP):
            mask_val[l]=l.mask
        if isinstance(l, MaskedbiGRU) or isinstance(l, MaskedGRU):
            for i in l.named_parameters():
                if 'mask' in i[0]:
                    mask_val[l]=i[1]
    with open(maskname,"wb") as file:
        pickle.dump(mask_val,file)


def print_layer_keep_ratio(model, logger):
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
            logger.info("{}, keep ratio {:.4f}".format(layer, ratio))
            '''
            weight_shape = layer.weight.shape
            mask = layer.mask.repeat(weight_shape[1], 1)
            ratio = torch.sum(mask) / mask.numel()
            total += mask.numel()
            keep += torch.sum(mask)
            logger.info("use mask {}, keep ratio {:.4f}".format(layer, ratio))
            '''
        elif isinstance(layer, MaskedConv2d):
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
            logger.info("{}, keep ratio {:.4f}".format(layer, ratio))
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
            logger.info("{}, keep ratio {:.4f}".format(layer, ratio))
    logger.info("Model keep ratio {:.4f}".format(keep/total))
    return keep / total

