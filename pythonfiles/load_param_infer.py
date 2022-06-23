# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 18:03:26 2022

@author: maiko
"""

import gc
import cupy as cp
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
# PyTorch 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

from save_learn_param import CFG,build_model
from Make_Dataset import BuildDataset

debug = True

def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def mask2rle(msk, thr=0.5):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    msk    = cp.array(msk)
    pixels = msk.flatten()
    pad    = cp.array([0])
    pixels = cp.concatenate([pad, pixels, pad])
    runs   = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def masks2rles(msks, ids, heights, widths):
    pred_strings = []; pred_ids = []; pred_classes = [];
    for idx in range(msks.shape[0]):
        msk = msks[idx]
        height = heights[idx].item()
        width = widths[idx].item()
        shape0 = np.array([height, width])
        resize = np.array([320, 384])
        if np.any(shape0!=resize):
            diff = resize - shape0
            pad0 = diff[0]
            pad1 = diff[1]
            pady = [pad0//2, pad0//2 + pad0%2]
            padx = [pad1//2, pad1//2 + pad1%2]
            msk = msk[pady[0]:-pady[1], padx[0]:-padx[1], :]
            msk = msk.reshape((*shape0, 3))
        rle = [None]*3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(msk[...,midx])
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]]*len(rle))
        pred_classes.extend(['large_bowel', 'small_bowel', 'stomach'])
    return pred_strings, pred_ids, pred_classes

@torch.no_grad()
def infer(model_paths, test_loader, num_log=1, thr=CFG.thr):
    msks = []; imgs = [];
    pred_strings = []; pred_ids = []; pred_classes = [];
    for idx, (img, ids, heights, widths) in enumerate(tqdm(test_loader, total=len(test_loader), desc='Infer ')):
        img = img.to(CFG.device, dtype=torch.float) # .squeeze(0)
        size = img.size()
        msk = []
        msk = torch.zeros((size[0], 3, size[2], size[3]), device=CFG.device, dtype=torch.float32)
        for path in model_paths:
            model = load_model(path)
            out   = model(img) # .squeeze(0) # removing batch axis
            out   = nn.Sigmoid()(out) # removing channel axis
            msk+=out/len(model_paths)
        msk = (msk.permute((0,2,3,1))>thr).to(torch.uint8).cpu().detach().numpy() # shape: (n, h, w, c)
        result = masks2rles(msk, ids, heights, widths)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])
        if idx<num_log and debug:
            img = img.permute((0,2,3,1)).cpu().detach().numpy()
            imgs.append(img[::5])
            msks.append(msk[::5])
        del img, msk, out, model, result
        gc.collect()
        torch.cuda.empty_cache()
    return pred_strings, pred_ids, pred_classes, imgs, msks

def main():
    model_paths = 'input/best_epoch-00.pth'
    #model_paths  = glob(f'input/best_epoch*.pth')
    test_dataset = BuildDataset(test_df, transforms=data_transforms['valid'])
    test_loader  = DataLoader(test_dataset, batch_size=CFG.valid_bs, 
                              num_workers=4, shuffle=False, pin_memory=False)
    pred_strings, pred_ids, pred_classes, imgs, msks = infer(model_paths, test_loader)
    
    if debug:
        for img, msk in zip(imgs[0][:5], msks[0][:5]):
            plt.figure(figsize=(12, 7))
            plt.subplot(1, 3, 1); plt.imshow(img, cmap='bone');
            plt.axis('OFF'); plt.title('image')
            plt.subplot(1, 3, 2); plt.imshow(msk*255); plt.axis('OFF'); plt.title('mask')
            plt.subplot(1, 3, 3); plt.imshow(img, cmap='bone'); plt.imshow(msk*255, alpha=0.4);
            plt.axis('OFF'); plt.title('overlay')
            plt.tight_layout()
            plt.show()

if __name__ == '__main__':
    main()


