import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from train_argument import parser, print_args

from time import time
from utils import * 
from models import *
from trainer import *

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time

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
    
    
    
    TRG_PAD_IDX = 1
    loss = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
 
    
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    """We create the tokenizers."""

    def tokenize_de(text):
        # Tokenizes German text from a string into a list of strings
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        # Tokenizes English text from a string into a list of strings
        return [tok.text for tok in spacy_en.tokenizer(text)]

    """The fields remain the same as before."""

    SRC = Field(tokenize = tokenize_de, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)

    TRG = Field(tokenize = tokenize_en, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)

    """Load the data."""

    train_data, valid_data, test_data = Multi30k.splits(root=args.data_root, exts = ('.de', '.en'),fields = (SRC, TRG))

    """Build the vocabulary."""

    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    net = seq2seq_attention(INPUT_DIM, OUTPUT_DIM, dataset=args.dataset, device=device)
    if args.mask:
        net = masked_seq2seq_attention(INPUT_DIM, OUTPUT_DIM, dataset=args.dataset, device=device)

    net.to(device)
    
    trainer = seq_Trainer(args, logger)
    
    
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = args.batch_size,
        device = device)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    trainer.train(net, loss, device, train_iterator, valid_iterator, optimizer=optimizer, scheduler=scheduler)
    


if __name__ == '__main__':
    args = parser()
    #print_args(args)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
