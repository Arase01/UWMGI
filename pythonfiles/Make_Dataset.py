import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from glob import glob
import torchvision.transforms as transforms


class BuildDataset(Dataset):
    def __init__(self, df, label=True, transforms=None):
        #self.img_paths  = sorted(glob(f'{image_paths}/*.png'), key = lambda x: x.split('\\')[1].split('_')[1])
        #self.img_masks  = sorted(glob(f'{mask_paths}/*.png'), key = lambda x: x.split('\\')[1].split('_')[1])
        self.df = df
        self.img_paths  = df['image_path']
        self.mask_paths  = df['mask_path']
        self.label = label
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path  = self.img_paths[index]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype('float32')
        img /= img.max(axis=(0,1))
        
        if self.label:
            mask_path = self.mask_paths[index]
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype('float32')
#             if self.transforms:
#                 data = self.transforms(image=img, mask=msk)
#                 img  = data['image']
#                 msk  = data['mask']
            img = np.transpose(img, (2, 0, 1))  #(3, x, y)
            mask = np.transpose(mask, (2, 0, 1))  #(3, x, y)
            return torch.tensor(img), torch.tensor(mask)
        else:
#             if self.transforms:
#                 data = self.transforms(image=img)
#                 img  = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)

def prepare_loaders(df, current_fold, train_bs, valid_bs):
    train_df = df.query("fold!=@current_fold").reset_index(drop=True)
    valid_df = df.query("fold==@current_fold").reset_index(drop=True)
    
    train_dataset = BuildDataset(train_df)
    valid_dataset = BuildDataset(valid_df)
    
    train_loader = DataLoader(train_dataset, batch_size=train_bs, pin_memory=True,shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_bs, pin_memory=True,shuffle=False, num_workers=4)
    
    return train_loader, valid_loader

# image_paths = '../input/seg_train/images'
# mask_paths = '../input/seg_train/masks'
# train_dataset = BuildDataset(image_paths, mask_paths)
# train_loader = DataLoader(train_dataset, batch_size=128, pin_memory=True,shuffle=True, num_workers=4)

# imgs, msks = next(iter(train_loader))
# print(imgs.size(), msks.size())