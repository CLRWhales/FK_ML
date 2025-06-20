#setup 
import os#, glob
import torch
#import torch.nn as nn
from torch.utils.data import DataLoader#, random_split
from torchvision import transforms
#from PIL import Image
#import time
#from torchvision.transforms import functional as F
import pandas as pd
#import datetime
#import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import numpy as np
#net and loader
from InferenceDataset import InferenceDataset
from FK_ver1 import Autoencoder as mod



#inputs
datapath = "D:\\DAS\\FK\\testset"
mpath = "C:\\Users\\Calder\\Models\\FK_ver1.2_20250612T114618\\Epoch16_model.pth"
batch_size = 64
num_workers = 8
# LR = 1e-3
# mask_prob = 0.0
# test_proportion = 0.8
# nepochs = 100
# patience = 5
# min_delta = 1e-5
# name = 'UNET'
# test = False


if __name__ == '__main__':
    dst_dir = os.path.split(mpath)[0]
    # GPU setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        #RemoveBottomRow(), #goes from 257x512 to 256x512 removing the nyquist row
        transforms.ToTensor() #this also scales uint 8 inputs to range 0-1
        #transforms.Normalize(mean = [0.0],std = [1.0])
    ])

    #accumulators
    latents = []
    filenames =[]
    toff = []
    xoff= []
    tabs = []

    dataset = InferenceDataset(datapath, transform=transform)
    inferLoader = DataLoader(dataset, batch_size=batch_size,num_workers=num_workers ,shuffle=False,pin_memory=True)
    model = mod().to(device)
    model.load_state_dict(torch.load(mpath,weights_only=True))
    model.eval()
  
    with torch.no_grad():
        for input_img, to,xo,ta,paths in inferLoader:
            input_img = input_img.to(device, non_blocking = True)
            latent,*_ = model.encode(input_img)
            latents.append(latent.cpu().numpy())
            filenames.extend(paths)

            #locations
            toff.extend(to)
            xoff.extend(xo)
            tabs.extend(ta)
    
    #save results
    latents = np.concatenate(latents,axis = 0)
    df = pd.DataFrame(latents)

    toff = [float(t)/256 for t in toff]
    xoff = [float(t)*12/1000 for t in xoff]
    timestamps = pd.to_datetime(tabs, format= '%Y%m%dT%H%M%SZ')
    toffsets = pd.to_timedelta(toff, unit = 's')
    truetime = timestamps + toffsets

    df['xoff'] = xoff
    df['toff'] = toff
    df['timestamp'] = truetime
    df['filename'] = filenames
    outname = os.path.join(dst_dir,'latent_states.csv')
    df.to_csv(outname,index = False)
