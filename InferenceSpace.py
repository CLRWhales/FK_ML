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
from FK_skipcon import Autoencoder as mod



#inputs
datapath = 'D:\\DAS\\FK\\NoShore_DS3_x512_20250604T143445'
mpath = ''
batch_size = 64
num_workers = 4
LR = 1e-3
mask_prob = 0.0
test_proportion = 0.8
nepochs = 100
patience = 5
min_delta = 1e-5
name = 'UNET'
test = False


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

    dataset = InferenceDataset(datapath, transform=transform)
    inferLoader = DataLoader(dataset, batch_size=batch_size,num_workers=num_workers ,shuffle=False,pin_memory=True)

    # Model, loss, optimizer 
    model = mod().to(device)
    model.load_state_dict(torch.load(mpath,weights_only=True))
    model.eval()
    latents = []
    filenames =[]
    with torch.no_grad():
        for input_img, paths in inferLoader:
            input_img = input_img.to(device, non_blocking = True)
            latent,*_ = model.encode(input_img)
            latents.append(latent.cpu().numpy())
            filenames.extend(paths)
    
    #save results
    latents = np.concatenate(latents,axis = 0)
    df = pd.DataFrame(latents)
    df['filename'] = filenames
    outname = os.path.join(dst_dir,'latent_states.csv')
    df.to_csv(outname,index = False)
