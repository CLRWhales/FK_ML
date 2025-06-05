#this builds out the dataset class for the fk nets
#first attempt at building an autoencoder
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F


# === Custom Dataset with Optional Masking ===
class MaskedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_prob=0.0):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_prob = mask_prob
        # Collect all image paths
        self.image_paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(('.png')):
                    self.image_paths.append(os.path.join(dirpath, filename))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        #print(image.shape)
        if self.transform:
            image = self.transform(image)

        masked_image = image.clone()

        if self.mask_prob > 0:
            mask = torch.rand_like(masked_image[0]) < self.mask_prob
            masked_image[0][mask] = 0
        return masked_image, image  # (input, target)
    
