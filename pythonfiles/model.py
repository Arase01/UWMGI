# -*- coding: utf-8 -*-
"""
Created on Sun May 29 18:01:42 2022

@author: kaito
"""

import os
from glob import glob

all_image_files = glob("../input/seg_train/images/*")
patients = [os.path.basename(_).split("_")[0] for _ in all_image_files]
print(len(patients))