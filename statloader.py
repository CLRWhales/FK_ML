#this is a mean and variance exploration file of the 1 hour dataset

import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader#, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Custom Dataset with Optional Masking ===
class imageStats(Dataset):
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

    def __getitem__(self, idx):
        #image = Image.open(self.image_paths[idx]).convert('L')
        name = os.path.basename(self.image_paths[idx]).split('_')
        toff = name[1].split('T')[1]
        xoff = name[2].split('X')[1]
        ent = float(name[3].split('E')[1])
        tabs = name[4].split('.')[0]

        # if self.transform:
        #     image = self.transform(image)
        
        # image *=max
        # image +=min
    
        # mean = image.mean()
        # sd = image.std()

        return toff,xoff,tabs,ent#,mean,sd


if __name__ == '__main__':

    datapath = 'D:\\DAS\\FK\\DS3_x512_NoShoreNoNorm20250609T174925'
    batch_size = 256
    num_workers = 8
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor() #this also scales uint 8 inputs to range 0-1
    ])

    toff = []
    xoff= []
    tabs = []
    ent = []


    dataset = imageStats(datapath, transform=transform)
    statloader = DataLoader(dataset, batch_size=batch_size,num_workers=num_workers)

    for to, xo,ta,e in statloader:
        toff.extend(to)
        xoff.extend(xo)
        tabs.extend(ta)
        ent.extend(e)

    toff = [float(t)/256 for t in toff]

    xoff = [float(t)*12/1000 for t in xoff]
    #mean = [t.item() for t in mean]
    #sd = [t.item() for t in sd]
    timestamps = pd.to_datetime(tabs, format= '%Y%m%dT%H%M%SZ')
    toffsets = pd.to_timedelta(toff, unit = 's')
    truetime = timestamps + toffsets
    # outputs = pd.DataFrame({
    #     'trutime': truetime,
    #     'toff': toff,
    #     'xoff': xoff,
    #     'mean': mean,
    #     'sd': sd
    # })

    outname = os.path.join(datapath, 'stats.csv')
    #outputs.to_csv(outname, index=False)

    #mu = np.mean(mean)
    #dvar = np.mean([s**2 + (m - mu)**2 for s, m in zip(sd, mean)])
    #dstd = np.sqrt(dvar)
    #print(mu)
    #print(dstd)

    plt.figure()
    plt.hexbin(xoff,ent)
    plt.title('density')
    plt.xlabel('Fiber distance from shore (km)')
    plt.ylabel('mean pixel value')
    fname = os.path.join(datapath,'meanoverspace.jpeg')
    plt.savefig(fname)
    plt.close()

    # plt.figure()
    # plt.hexbin(xoff,sd)
    # plt.title('density')
    # plt.xlabel('Fiber distance from shore (km)')
    # plt.ylabel('standard deviation of pixel value')
    # fname = os.path.join(datapath,'stdoverspace.jpeg')
    # plt.savefig(fname)
    # plt.close()

    # plt.figure()
    # plt.scatter(xoff,ent)
    # plt.title('scatter')
    # plt.xlabel('Fiber distance from shore (km)')
    # plt.ylabel('mean pixel value')
    # fname = os.path.join(datapath,'meanoverspace_scatter.jpeg')
    # plt.savefig(fname)
    # plt.close()

    plt.figure()
    plt.scatter(xoff,ent)
    plt.title('scatter')
    plt.xlabel('Fiber distance from shore (km)')
    plt.ylabel('standard deviation of pixel value')
    fname = os.path.join(datapath,'stdoverspace_scatter.jpeg')
    plt.savefig(fname)
    plt.close()

    plt.figure()
    plt.scatter(truetime,xoff,c = ent)
    plt.colorbar()
    plt.title('mean')
    plt.xlabel('Time')
    plt.ylabel('Fiber distance from shore (km)')
    fname = os.path.join(datapath,'meanspacetime.jpeg')
    plt.savefig(fname)
    plt.close()

    # plt.figure()
    # plt.scatter(truetime,xoff,c = sd)
    # plt.colorbar()
    # plt.title('standard deviation')
    # plt.xlabel('Time')
    # plt.ylabel('Fiber distance from shore (km)')
    # fname = os.path.join(datapath,'stdspacetime.jpeg')
    # plt.savefig(fname)
    # plt.close()