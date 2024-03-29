# -*- coding: utf-8 -*-
"""
Created on Sun May 29 12:16:14 2022

@author: kaito
"""

import os
import cv2
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from glob import glob
from tqdm import tqdm

IMG_SIZE = (320,384)

def rle_decode(mask_rle, shape):
    s = np.array(mask_rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2] 
    ends = starts + lengths
    h, w = shape
    img = np.zeros((h * w,), dtype = np.uint8) #flatten
    for lo, hi in zip(starts, ends): #start-1 + length = end (length include start)
        img[lo : hi] = 1
    return img.reshape(shape)

def load_img(path, size=IMG_SIZE):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    shape0 = np.array(img.shape[:2])
    resize = np.array(size)
    if np.any(shape0!=resize):
        diff = resize - shape0
        pad0 = diff[0]
        pad1 = diff[1]
        pady = [pad0//2, pad0//2 + pad0%2]
        padx = [pad1//2, pad1//2 + pad1%2]
        img = np.pad(img, [pady, padx],mode='constant')
        img = img.reshape(*resize)
        
#     img = np.tile(img[...,None], [1, 1, 3]) # gray to rgb
#     img = img.astype('float32') # original is uint16
#     mx = np.max(img)
#     if mx:
#         img/=mx # scale image to [0, 1]
    return img

def change_shape_msk(msks, size=IMG_SIZE):
    msk = np.stack([msks[k] for k in msks], -1)
    shape0 = np.array(msk.shape[:2])
    resize = np.array(size)
    if np.any(shape0!=resize):
        diff = resize - shape0
        pad0 = diff[0]
        pad1 = diff[1]
        pady = [pad0//2, pad0//2 + pad0%2]
        padx = [pad1//2, pad1//2 + pad1%2]
        msk = np.pad(msk, [pady, padx, [0,0]], mode='constant')
        msk = msk.reshape(*resize, 3)
    return msk

def pathcheck(path):
    if os.path.exists(path) == True:
        if os.path.isfile(path) == True :
            print('Reset File: {}'.format(path))
            os.remove(path)
        if os.path.isdir(path) == True:
            print('Reset Directory: {}'.format(path))
            shutil.rmtree(path)
            os.makedirs(path)
    elif len(path.split('.')) == 1 or len(path.split('.')) == 3:
        os.makedirs(path)
    else: print("no reset")
    
    
pathcheck('../input/seg_train/images')
pathcheck('../input/seg_train/masks')

df = pd.read_csv("../input/train.csv") #read train data
df = df.sort_values(["id", "class"]).reset_index(drop = True) #id:case_day_slice
df["patient"] = df.id.apply(lambda x: x.split("_")[0]) # case
df["days"] = df.id.apply(lambda x: "_".join(x.split("_")[:2])) #day


all_image_files = sorted(glob('../input/train/*/*/scans/*.png'),key = lambda x: x.split("/")[1] + x.split("/")[2])
#slice[0]_slicenum[1]_sizex[2]_sizey[3]_spacingx[4]_spacingy[5]        ex.slice_0001_266_266_1.50_1.50
size_x = [int(os.path.basename(_)[:-4].split("_")[-4]) for _ in all_image_files] #[:-4] -> remove ".png"
size_y = [int(os.path.basename(_)[:-4].split("_")[-3]) for _ in all_image_files]
spacing_x = [float(os.path.basename(_)[:-4].split("_")[-2]) for _ in all_image_files]
spacing_y = [float(os.path.basename(_)[:-4].split("_")[-1]) for _ in all_image_files]
df["image_files"] = np.repeat(all_image_files, 3) #3->"stomach" "large_bowel" "small_bowel"
df["spacing_x"] = np.repeat(spacing_x, 3)
df["spacing_y"] = np.repeat(spacing_y, 3)
df["size_x"] = np.repeat(size_x, 3)
df["size_y"] = np.repeat(size_y, 3)
df["slice"] = np.repeat([int(os.path.basename(_)[:-4].split("_")[-5]) for _ in all_image_files], 3) #slice

cnt=0
pbar = tqdm(df.groupby("days"), total=len(df.groupby("days")), desc='Saving ')
for day, group in pbar: #144 scans per day -> imgs,msks
    imgs,msks,file_names = [],[],[]
    for file_name in group.image_files.unique(): #1group -> "stomach" "large_bowel" "small_bowel"(3labels)
        #img = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH) #(266,266) ...fluctuate xy size but size are mostly it
        img = load_img(file_name) #set (IMGSIZE)
        segms = group.loc[group.image_files == file_name] 
        masks = {} #3label mask
        for segm, label in zip(segms.segmentation, segms['class']): #1lebel + 1segm -> 1mask
            if not pd.isna(segm):
                mask = rle_decode(segm,[segms.size_y.iloc[0], segms.size_x.iloc[0]])
                masks[label] = mask
            else:
                masks[label] = np.zeros((segms.size_y.iloc[0], segms.size_x.iloc[0]), dtype = np.uint8)
                
        masks = change_shape_msk(masks,IMG_SIZE) # (inputsize) -> (IMGSIZE)     
        imgs.append(img) #  imgs : (144,266,266)    img : (1,266,266)
        msks.append(masks) #msks : (144,266,266,3) masks: (1,266,266,3)
    
    #print(f'{len(imgs)}, {len(imgs[0])}, {len(imgs[0][0])}')
    imgs = np.stack(imgs, 0) #(144,266,266) ... ndarray
    #imgs = np.array(imgs)
    msks = np.stack(msks, 0) #(144,266,266,3) ...ndarray
    #msks = np.array(msks)
    
    for i in range(msks.shape[0]):
        cnt += 1
        img = imgs[[max(0, i - 2), i, min(imgs.shape[0] - 1, i + 2)]].transpose(1,2,0)    
        msk = msks[i]
        
        new_file_name = f"{day}_slice_{str(i+1).zfill(4)}.png"
        cv2.imwrite(f"../input/seg_train/images/{new_file_name}", img)
        cv2.imwrite(f"../input/seg_train/masks/{new_file_name}", msk)
        pbar.set_postfix(saveimg=f'{cnt}/{len(df.id)/3}')
    