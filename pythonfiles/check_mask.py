# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 11:37:09 2022

@author: maiko
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    #img = np.tile(img[...,None], [1, 1, 3]) # gray to rgb
    img = img.astype('float32') # original is uint16
    mx = np.max(img)
    if mx:
        img/=mx # scale image to [0, 1]
    return img

def show_img(img, mask=None):
    plt.imshow(img, cmap='bone')
    
    if mask is not None:
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles,labels)
    plt.axis('off')
    
img = load_img('../input/seg_train/images/case2_day1_slice_0085.png')
mask = load_img('../input/seg_train/masks/case2_day1_slice_0085.png')

show_img(img,mask)