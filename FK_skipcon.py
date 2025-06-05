#first attempt at building an autoencoder
#import os, glob
import torch
import torch.nn as nn
#from torch.utils.data import Dataset, DataLoader, random_split
#from torchvision import transforms
#from PIL import Image
#import time
#from torchvision.transforms import functional as F
#import pandas as pd
#import datetime
#import matplotlib.pyplot as plt
#import matplotlib.colors as colors


# === Autoencoder === following the example of Cchien et al (billy jenkins paper)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        def conv_block(in_ch, out_ch, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def deconv_block(in_ch, out_ch, stride=2):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, output_padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def recoded(in_ch, out_ch):
            return nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = conv_block(1, 8)
        self.enc2 = conv_block(8, 16)
        self.enc3 = conv_block(16, 32)
        self.enc4 = conv_block(32, 64)
        self.enc5 = conv_block(64, 128)
        self.enc6 = conv_block(128, 128) #this is an extra layer to bring the number of points down closer to the implementation of Billy jenkins

        self.flat1 = nn.Flatten()
        self.bottleneck = recoded(128 * 4 * 8, 9)

        # Decoder
        self.open = recoded(9, 128 * 4 * 8)
        self.uflat1 = nn.Unflatten(1, (128, 4, 8))

        self.dec6 = deconv_block(256, 128) #undoes the extra layer
        self.dec5 = deconv_block(256, 64)
        self.dec4 = deconv_block(128, 32)
        self.dec3 = deconv_block(64, 16)
        self.dec2 = deconv_block(32, 8)
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
                                  nn.Sigmoid()) # back to 1 channel, seperate because sigmoid activation (ie no activation)
    
    def encode(self,x): #could likely make all these sequential but this framing allows for future use with ski connections
        e1 = self.enc1(x)   # [B, 8, 128, 256]
        e2 = self.enc2(e1)   # [B, 16, 64, 128]
        e3 = self.enc3(e2)   # [B, 32, 32, 64]
        e4 = self.enc4(e3)   # [B, 64, 16, 32]
        e5 = self.enc5(e4)   # [B, 128, 8, 16]
        e6 = self.enc6(e5)
        flat = self.flat1(e6)  # [B, 128*8*16]
        bot = self.bottleneck(flat)  # [B, 9]
        return bot, e1,e2,e3,e4,e5,e6

    def decode(self,bot,e1,e2,e3,e4,e5,e6):
        op = self.open(bot)   # [B, 128*8*16]
        uflat = self.uflat1(op)  # [B, 128, 8, 16]
        dec6 = self.dec6(torch.concat([e6,uflat], axis=1))
        dec5 = self.dec5(torch.concat([e5,dec6], axis = 1))  # [B, 64, 16, 32]
        dec4 = self.dec4(torch.concat([e4,dec5], axis = 1))  # [B, 32, 32, 64]
        dec3 = self.dec3(torch.concat([e3,dec4], axis = 1))  # [B, 16, 64, 128]
        dec2 = self.dec2(torch.concat([e2,dec3], axis = 1))  # [B, 8, 128, 256]
        dec1 = self.dec1(torch.concat([e1,dec2],axis = 1))  # [B, 1, 256, 512]
        return dec1

    def forward(self, x):
        bot, e1,e2,e3,e4,e5,e6 = self.encode(x)
        dec1 = self.decode(bot,e1,e2,e3,e4,e5,e6)
        return dec1
