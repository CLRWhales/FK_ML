#inference space 2:

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
from Multitask_loader import Multitask_loader
from FK_ver3 import Autoencoder as mod



#inputs
datapath = "D:\\DAS\\FK\\meta_halfday_newspeed20250621T123640"
mpath = "C:\\Users\\Calder\\Models\\FK_ver3_subset_transform_20250622T182320\\Epoch26_final.pth"
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

    
    #accumulators
    latents = []
    filenames =[]
    tabs = []
    regtrue = []
    regest = []

    dataset = Multitask_loader(datapath)
    inferLoader = DataLoader(dataset, batch_size=batch_size,num_workers=num_workers ,shuffle=False,pin_memory=True)
    model = mod(128).to(device)
    model.load_state_dict(torch.load(mpath,weights_only=True))
    model.eval()
  
    with torch.no_grad():
        for input_img, _,regs,name in inferLoader:
            input_img = input_img.to(device, non_blocking = True)
            _,reg,latent= model(input_img)
            latents.append(latent.cpu().numpy())
            regest.append(reg.cpu().numpy())
            regtrue.append(regs)
            filenames.extend(name)
    
    #save results
    latents = np.concatenate(latents,axis = 0)
    regest = np.concatenate(regest, axis = 0)
    regtrue = np.concatenate(regtrue,axis = 0)

    
    df = pd.DataFrame(latents)
    df1 = pd.DataFrame(regtrue)
    df2 = pd.DataFrame(regest)
    df3 = pd.DataFrame(filenames)

    outname = os.path.join(dst_dir,'latent_states.csv')
    outname1 = os.path.join(dst_dir,'regest.csv')
    outname2 = os.path.join(dst_dir,'regtrue.csv')
    outnames = os.path.join(dst_dir,'names.csv')
    df.to_csv(outname,index = False)
    df1.to_csv(outname1,index=False)
    df2.to_csv(outname2,index= False)
    df3.to_csv(outnames,index = False)