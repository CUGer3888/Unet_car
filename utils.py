import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
def getloaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
               train_transform, val_transform,
               batch_size, numworkers, pin_memory=True):
    train_ds = CarvanaDataset(image_dic=train_img_dir, mask_dic=train_mask_dir, transform=train_transform)
    val_ds = CarvanaDataset(image_dic=val_img_dir, mask_dic=val_mask_dir, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=numworkers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=numworkers, pin_memory=pin_memory)
    return train_loader, val_loader
