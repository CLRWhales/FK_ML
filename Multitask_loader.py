#this is a new loader for multitask inference
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F
from torchvision import transforms


# === Custom Dataset with Optional Masking ===
class Multitask_loader(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.totensor = transforms.ToTensor()
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
        
        image = self.totensor(image)
        blurred_image = image.clone()

        name = os.path.basename(self.image_paths[idx]).split('_')
        toff = float(name[0].split('T')[1])/2048
        xoff = float(name[1].split('X')[1])/10240
        F = float(name[2].split('F')[1])/256
        K = float(name[3].split('K')[1])/512
        V = float(name[4].split('V')[1])/10000
        reg = torch.tensor([toff,xoff,F,K,V], dtype= torch.float32)

        if self.transform:
            blurred_image = self.transform(image)
        
        return image,blurred_image,reg  # (input, target)

