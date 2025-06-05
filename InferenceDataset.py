#this is to build a one shot testing dataset

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F

class InferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Collect all image paths
        self.image_paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(('.png')):
                    self.image_paths.append(os.path.join(dirpath, filename))

    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, idx,testing = False):
        image = Image.open(self.image_paths[idx]).convert('L')
        #print(image.shape)
        if self.transform:
            image = self.transform(image)
        
        return image,  self.image_paths[idx]# (input, target)
    
