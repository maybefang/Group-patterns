import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F
from time import time
import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from models.binarization import MaskedbiGRU, MaskedGRU, MaskedMLP, MaskedDEMLP

class seq_Trainer():
    def __init__(self, args, logger, attack=None):
        self.args = args
        self.logger = logger
        self.attack = attack

    def train(self, model, criterion, device, train_iterator, valid_iterator=None, optimizer=None, scheduler=None):
        args = self.args
        logger = self.logger

        _iter = 1

        begin_time = time()
        best_valid_loss = float('inf')
        keep_ratio_at_best_loss = 0.
        best_keep_ratio = 1.
        loss_at_best_keep_ratio = 0.

        for epoch in range(1, args.max_epoch+1):
            logger.info("-"*30 + "Epoch start" + "-"*30)

            train_loss = self.ep_train(model, train_iterator, optimizer, criterion)
            valid_loss = self.evaluate(model, valid_iterator, criterion)

            if args.mask:
                current_keep_ratio = print_layer_keep_ratio(model, logger)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if args.mask:
                    keep_ratio_at_best_acc = current_keep_ratio
                filename = os.path.join(args.model_folder, 'best_loss_model.pth')
                maskname = os.path.join(args.model_folder, 'best_loss_mask.pkl')
                save_model(model, filename, maskname)
            if args.mask and current_keep_ratio < best_keep_ratio:
                best_keep_ratio = current_keep_ratio
                loss_at_best_keep_ratio = valid_loss
                filename = os.path.join(args.model_folder, 'best_keepratio_model.pth')
                maskname = os.path.join(args.model_folder, 'best_keepratio_mask.pkl')
                save_model(model, filename, maskname)
            filename = os.path.join(args.model_folder, 'best_keepratio_model-'+str(epoch)+'.pth')
            maskname = os.path.join(args.model_folder, 'best_keepratio_mask-'+str(epoch)+'.pkl')
            save_model(model, filename, maskname)
            if scheduler is not None:
                scheduler.step()
            
            logger.info(f'Epoch: {epoch:02}')
            logger.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            logger.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        logger.info(">>>>> Training process finish")
        if args.mask:
            logger.info("Best keep ratio {:.4f}, acc at best keep ratio {:.4f}".format(best_keep_ratio, acc_at_best_keep_ratio))
            logger.info("Best acc {:.4f}, keep ratio at best acc {:.4f}".format(best_acc, keep_ratio_at_best_acc))
        else:
            logger.info("Best test accuracy {:.4f}".format(best_acc))
        file_name = os.path.join(args.model_folder, 'final_model.pth')
        maskname = os.path.join(args.model_folder, 'final_model_mask.pkl')
        save_model(model, file_name, maskname)

    def ep_train(self, model, iterator, optimizer, criterion):
        args = self.args
        model.train()    
        epoch_loss = 0
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg # trg = [trg_len, batch_size]

            # pred = [trg_len, batch_size, pred_dim]
            pred = model(src, trg)
            
            pred_dim = pred.shape[-1]
            
            # trg = [(trg len - 1) * batch size]
            # pred = [(trg len - 1) * batch size, pred_dim]
            trg = trg[1:].view(-1)
            pred = pred[1:].view(-1, pred_dim)
            
            loss = criterion(pred, trg)

            print(f'\t{i} step Loss: {loss:.3f} | PPL: {math.exp(loss):7.3f}',end=' | ')
            if args.mask:
                for layer in model.modules():
                    if isinstance(layer, MaskedMLP) or isinstance(layer, MaskedGRU) or isinstance(layer, MaskedbiGRU): 
                        loss += args.alpha * torch.sum(torch.exp(-layer.threshold))
                    elif isinstance(layer, MaskedDEMLP):
                        loss += args.alpha *10* torch.sum(torch.exp(-layer.threshold))
            print("new loss:",loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
        return epoch_loss / len(iterator)

    """...and the evaluation loop, remembering to set the model to `eval` mode and turn off teaching forcing."""

    def evaluate(self, model, iterator, criterion):
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = batch.src
                trg = batch.trg # trg = [trg_len, batch_size]

                # output = [trg_len, batch_size, output_dim]
                output = model(src, trg, 0) # turn off teacher forcing
            
                output_dim = output.shape[-1]
                
                # trg = [(trg_len - 1) * batch_size]
                # output = [(trg_len - 1) * batch_size, output_dim]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

                loss = criterion(output, trg)
                epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    
    