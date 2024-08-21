import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image

class CarvanaDataset(Dataset):
    def __init__(self,image_dic,mask_dic,transform = None):
        self.image_dic = image_dic
        self.mask_dic = mask_dic
        self.transform = transform
        self.images = os.listdir(self.image_dic)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dic,self.images[idx])
        mask_path = os.path.join(self.mask_dic,self.images[idx].replace('.jpg','.png'))
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        if self.transform is not None:
            augmentation = self.transform(image=image,mask=mask)
            image = augmentation['image']
            mask = augmentation['mask']
        return image,mask