# -*- coding: utf-8 -*-
"""
Created on Sun May 29 12:16:14 2022

@author: kaito
"""

import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

df = pd.read_csv("../input/train.csv") #read train data
df = df.sort_values(["id", "class"]).reset_index(drop = True) #id:case_day_slice
df["patient"] = df.id.apply(lambda x: x.split("_")[0]) # case
df["days"] = df.id.apply(lambda x: "_".join(x.split("_")[:2])) #day

all_image_files = sorted(glob('../input/train/*/*/scans/*.png'),key = lambda x: x.split("/")[1] + "_" + x.split("/")[2])
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

def rle_decode(mask_rle, shape):
    s = np.array(mask_rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2] 
    ends = starts + lengths
    h, w = shape
    img = np.zeros((h * w,), dtype = np.uint8) #flatten
    for lo, hi in zip(starts, ends): #start-1 + length = end (length include start)
        img[lo : hi] = 1
    return img.reshape(shape)


for day, group in tqdm(df.groupby("days")): #144 scans per day -> imgs,msks
    patient = group.patient.iloc[0]
    imgs,msks,file_names = [],[],[]
    for file_name in group.image_files.unique(): #1group -> "stomach" "large_bowel" "small_bowel"(3labels)
        img = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH) #(266,266) ...fluctuate xy size but size are mostly it
        segms = group.loc[group.image_files == file_name] 
        masks = {} #3label mask
        for segm, label in zip(segms.segmentation, segms['class']): #1lebel + 1segm -> 1mask
            if not pd.isna(segm):
                mask = rle_decode(segm,[segms.size_x.iloc[0], segms.size_y.iloc[0]])
                masks[label] = mask
            else:
                masks[label] = np.zeros((segms.size_x.iloc[0], segms.size_y.iloc[0]), dtype = np.uint8)
        masks = np.stack([masks[k] for k in sorted(masks)], -1)
        imgs.append(img)
        msks.append(masks)
        
    imgs = np.stack(imgs, 0) #(144,266,266) ...fluctuate xy size
    msks = np.stack(msks, 0) #(144,266,266,3) ...fluctuate xy size
    
    for i in range(msks.shape[0]):
        img = imgs[[max(0, i - 2), i, min(imgs.shape[0] - 1, i + 2)]].transpose(1,2,0)           
        msk = msks[i]
        
        new_file_name = f"{day}_{i}.png"
        cv2.imwrite(f"../input/seg_train/images/{new_file_name}", img)
        cv2.imwrite(f"../input/seg_train/masks/{new_file_name}", msk)