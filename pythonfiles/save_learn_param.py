# -*- coding: utf-8 -*-
"""
Created on Sun May 29 18:01:42 2022

@author: kaito
"""

import os
import gc
import sys
import time
import copy
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
from glob import glob
from IPython.core.display import display
from natsort import natsorted
from collections import defaultdict
import matplotlib.pyplot as plt

# PyTorch 
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

import segmentation_models_pytorch as smp

from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

from Make_Dataset import BuildDataset, prepare_loaders

from colorama import Fore,Style

c_  = Fore.GREEN
sr_ = Style.RESET_ALL

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 50000)

class CFG:
    debug         = False # set debug=False for Full Training
    debug_late    = 0.2  # use debug_late * allImage
    seed          = 101
    comment       = 'unet-efficientnet_b0-320x384'
    model_name    = 'Unet'
    backbone      = 'efficientnet-b4'
    num_classes   = 3
    epochs        = 20
    lr            = 2e-3
    min_lr        = 1e-6
    wd            = 1e-6
    n_fold        = 5
    train_bs      = 32
    valid_bs      = train_bs * 2
    n_accumulate  = max(1, 32//train_bs)
    T_max         = int(30000/train_bs*epochs)+50
    T_0           = 25
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def rand_nodup(upper, lower, rand_len):
    rand_list = []
    while len(rand_list) < rand_len:
        n = random.randint(lower, upper)
        if not n in rand_list:
            rand_list.append(n)
    return rand_list

#------------Model------------
def build_model():
    model = smp.Unet(
        encoder_name=CFG.backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=CFG.num_classes,        # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to(CFG.device)
    return model

#------------Loss Function------------
JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss    = smp.losses.DiceLoss(mode='multilabel')
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(alpha=0.3, mode='multilabel', log_loss=False)

def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):

    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

def criterion(y_pred, y_true):
    return 0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)

#-----------Training one epoch---------------
def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss   = criterion(y_pred, masks)
            loss   = loss / CFG.n_accumulate
            
        scaler.scale(loss).backward()
    
        if (step + 1) % CFG.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_mem=f'{mem:0.2f} GB')
        torch.cuda.empty_cache()
        gc.collect()
    
    return epoch_loss

@torch.no_grad()
def valid_one_epoch(model, optimizer, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    val_scores = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (images, masks) in pbar:        
        images  = images.to(device, dtype=torch.float)
        masks   = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        y_pred  = model(images)
        loss    = criterion(y_pred, masks)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        y_pred = nn.Sigmoid()(y_pred)
        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_memory=f'{mem:0.2f} GB')
    val_scores  = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss, val_scores

def run_training(model, optimizer, scheduler, device, num_epochs, train_loader, valid_loader, fold):
    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice      = -np.inf
    best_epoch     = -1
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}')
        train_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CFG.device, epoch=epoch)
        
        val_loss, val_scores = valid_one_epoch(model, optimizer, 
                                               dataloader=valid_loader, 
                                               device=CFG.device, epoch=epoch)
        val_dice, val_jaccard = val_scores
    
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)
        
        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')
        
        # deep copy the model
        if val_dice >= best_dice:
            print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice    = val_dice
            best_jaccard = val_jaccard
            best_epoch   = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"../output/best_epoch-{fold:02d}.pth"
            torch.save(model.state_dict(), PATH)

            print(f"Model Saved{sr_}")
            
        last_model_wts = copy.deepcopy(model.state_dict())
        PATH = f"../output/last_epoch-{fold:02d}.pth"
        torch.save(model.state_dict(), PATH)
            
        print(); print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Dice: {:.4f}     Best Jaccard: {:.4f}     Best Epoch: {:.4f}"
          .format(best_dice,best_jaccard,best_epoch))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history


def main():
    df = pd.DataFrame(natsorted(glob('../input/seg_train/images/*')), columns=['image_path'])
    df['mask_path'] = df.image_path.str.replace('image','mask')
    df['id'] = df.image_path.map(lambda x: x.split('/')[-1].replace('.png','').replace('images\\',''))
    df['case'] =  df.id.map(lambda x: x.split('_')[0].replace('case',''))
    
    df2 = pd.read_csv('../input/train.csv')
    df2 = df2.sort_values('id').reset_index(drop=True)
    df2['segmentation'] = df2.segmentation.fillna('')
    df2['rle_len'] = df2.segmentation.map(len)
    
    df3 = df2.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index() # rle list of each id
    df3 = df3.merge(df2.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index()) # total length of all rles of each id
    
    df2 = df2.drop(columns=['segmentation', 'class', 'rle_len'])
    df2 = df2.groupby(['id']).head(1).reset_index(drop=True)
    df2 = df2.merge(df3, on=['id'])
    df2['empty'] = (df2.rle_len==0)
    #df2['empty'].value_counts().plot.bar()
    
    df['rle_len'] = df2['rle_len']
    df['empty'] = df2['empty']
    fault1 = 'case7_day0'
    fault2 = 'case81_day30'
    df = df[~df['id'].str.contains(fault1) & ~df['id'].str.contains(fault2)].reset_index(drop=True)

    df['empty'].value_counts().plot.bar()
    
    if CFG.debug:
        debug_len = len(df['id']) - (CFG.debug_late*len(df['id']))
        delete_idx = rand_nodup(len(df['id'])-1, 0, debug_len)
        for delete in delete_idx:
            df = df.drop(index=delete)
        df = df.reset_index()
        print("#################DEBUG MODE#################")
    
    skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups = df["case"])):
        df.loc[val_idx, 'fold'] = fold  
    display(df.groupby(['fold','empty'])['id'].count())
    
    
    for fold in range(CFG.n_fold):
        print('#'*15)
        print(f'### Fold: {fold}')
        print('#'*15)
        
        #-----load data, model-----
        train_loader, valid_loader = prepare_loaders(df, fold, CFG.train_bs, CFG.valid_bs)
        model = build_model()
        
        #-----print data-----      
        imgs,msks = next(iter(train_loader))
        print(f' imgsize: {imgs.size()} \n msksize: {msks.size()}')
        #summary(model,(3,320,384)) #print model
        #sys.exit()
        
        #-----train-----
        optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG.T_max, eta_min=CFG.min_lr)
        model, history = run_training(model, optimizer, scheduler, #return best model and history
                                      device=CFG.device,
                                      num_epochs=CFG.epochs,
                                      train_loader=train_loader,
                                      valid_loader=valid_loader,
                                      fold=fold)
        
        #-----plt result-----
        fig, axs = plt.subplots(2,2,figsize=(12,12))
        for f, ax in zip(history, axs.ravel()):
            ax.set_xlabel(f)
            ax.plot(history[f])
        savepath = "../output/learn" + str(fold) + ".png"
        plt.savefig(savepath)
    
if __name__ == '__main__':
    main()
    
    
