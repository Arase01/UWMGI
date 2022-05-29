# -*- coding: utf-8 -*-
"""
Created on Sat May 28 17:39:16 2022

@author: kaito
"""

import numpy as np
import matplotlib.pyplot as plt

def mask_from_segmentation(segmentation, shape):

    # Get a list of numbers from the initial segmentation
    segm = np.asarray(segmentation.split(), dtype=int)

    # Get start point and length between points
    start_point = segm[0::2] - 1
    length_point = segm[1::2]

    # Compute the location of each endpoint
    end_point = start_point + length_point

    # Create an empty list mask the size of the original image
    # take onl
    case_mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    # Change pixels from 0 to 1 that are within the segmentation
    for start, end in zip(start_point, end_point):
        case_mask[start:end] = 1
        
    print(case_mask.shape)    
    case_mask = case_mask.reshape((shape[0], shape[1]))
    
    return case_mask

# Example
segmentation = '45601 5 45959 10 46319 12 46678 14 47037 16 47396 18 47756 18 48116 19 48477 18 48837 19 \
                49198 19 49558 19 49919 19 50279 20 50639 20 50999 21 51359 21 51719 22 52079 22 52440 22 52800 22 53161 21 \
                53523 20 53884 20 54245 19 54606 19 54967 18 55328 17 55689 16 56050 14 56412 12 56778 4 57855 7 58214 9 58573 12 \
                58932 14 59292 15 59651 16 60011 17 60371 17 60731 17 61091 17 61451 17 61812 15 62172 15 62532 15 62892 14 \
                63253 12 63613 12 63974 10 64335 7'

shape = (310, 360)

case_mask = mask_from_segmentation(segmentation, shape)

plt.figure(figsize=(5, 5))
plt.title("Mask Example:")
plt.imshow(case_mask,cmap="binary")
plt.axis("off")
plt.show()