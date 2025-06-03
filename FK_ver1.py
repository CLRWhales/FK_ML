#first attempt at building an autoencoder
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import time
from torchvision.transforms import functional as F
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
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
        self.dec6 = deconv_block(128, 128) #undoes the extra layer
        self.dec5 = deconv_block(128, 64)
        self.dec4 = deconv_block(64, 32)
        self.dec3 = deconv_block(32, 16)
        self.dec2 = deconv_block(16, 8)
        self.dec1 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1) # back to 1 channel, seperate because no/linear activation

    def encode(self,x): #could likely make all these sequential but this framing allows for future use with ski connections
        e1 = self.enc1(x)   # [B, 8, 128, 256]
        e2 = self.enc2(e1)   # [B, 16, 64, 128]
        e3 = self.enc3(e2)   # [B, 32, 32, 64]
        e4 = self.enc4(e3)   # [B, 64, 16, 32]
        e5 = self.enc5(e4)   # [B, 128, 8, 16]
        e6 = self.enc6(e5)
        flat = self.flat1(e6)  # [B, 128*8*16]
        bot = self.bottleneck(flat)  # [B, 9]
        return bot
    
    def decode(self,bot):
        op = self.open(bot)   # [B, 128*8*16]
        uflat = self.uflat1(op)  # [B, 128, 8, 16]
        dec6 = self.dec6(uflat)
        dec5 = self.dec5(dec6)  # [B, 64, 16, 32]
        dec4 = self.dec4(dec5)  # [B, 32, 32, 64]
        dec3 = self.dec3(dec4)  # [B, 16, 64, 128]
        dec2 = self.dec2(dec3)  # [B, 8, 128, 256]
        dec1 = self.dec1(dec2)  # [B, 1, 256, 512]
        return dec1

    def forward(self, x):
        bot = self.encode(x)
        dec1 = self.decode(bot)
        return dec1

#=== custom crop to remove nyquist ===
class RemoveBottomRow:
    def __call__(self, img):
        w, h = img.size  # PIL Image: (width, height)
        return F.crop(img, top=0, left=0, height=h - 1, width=w)



# === Setup and Example Usage ===
if __name__ == '__main__':
    datapath = 'D:\\DAS\\FK\\DS3_x512_20250530T153412'
    batch_size = 256
    num_workers = 4
    LR = 1e-3
    mask_prob = 0.0
    test_proportion = 0.8
    nepochs = 100
    patience = 5
    min_delta = 1e-4
    name = 'v2_refac'
    

    inputlist = [batch_size, num_workers, LR, mask_prob, test_proportion,nepochs,patience,min_delta]

    tnow = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    outputdir = os.path.join(datapath,name + '_' + tnow)
    os.makedirs(outputdir, exist_ok = True)
    txtname = os.path.join(outputdir, 'params.txt')
    with open(txtname, 'w') as f:
        for line in inputlist:
            f.write(f"{line}\n") 

    # GPU setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        RemoveBottomRow(), #goes from 257x512 to 256x512 removing the nyquist row
        transforms.ToTensor() #this also scales uint 8 inputs to range 0-1
        #transforms.Normalize(mean = [0.0],std = [1.0])
    ])

    # Dataset and Dataloaders
    dataset = MaskedImageDataset(datapath, transform=transform, mask_prob=mask_prob)
    train_size = int(test_proportion * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size,num_workers=num_workers ,shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,shuffle=False, pin_memory=True)

    # Model, loss, optimizer 
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    train_losses = []
    eval_losses = []
    counter = 0
    patience_track= []
    # Training loop with validation
    for epoch in range(nepochs):
        e_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        for input_img, target_img in train_loader:
            input_img = input_img.to(device, non_blocking = True)
            target_img = target_img.to(device, non_blocking = True)

            optimizer.zero_grad()
            #print(type(input_img), input_img.shape)
            output = model(input_img)
            loss = criterion(output, target_img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_img, target_img in val_loader:
                input_img, target_img = input_img.to(device, non_blocking = True), target_img.to(device, non_blocking = True)
                output = model(input_img)
                loss = criterion(output, target_img)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        eval_losses.append(avg_val_loss)

        e_end = time.perf_counter()
        tdif = e_end - e_start

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Time (s) = {tdif:.4f}")

       
        # Save checkpoint if best
        if avg_val_loss < best_val_loss - min_delta:
            counter = 0
            best_val_loss = avg_val_loss
            pathname = os.path.join(outputdir, 'Epoch' + str(epoch) + '_model.pth')
            torch.save(model.state_dict(), pathname)
            print("✅ Saved new best model.")
        else:
            counter += 1
        
        patience_track.append(counter)
        if counter >= patience:
            print('Run out of patience :(, stopping early')
            break
    
    df = pd.DataFrame({'train_loss': train_losses, 'val_loss': eval_losses,'patience':patience_track})
    fname = os.path.join(outputdir,'loss_log.csv')
    df.to_csv(fname, index=False)

    figure, axis = plt.subplots(2,1)

    axis[0].plot(train_losses, label = 'Training')
    axis[0].plot(eval_losses, label = 'Validation')
    axis[0].set_xlabel('epoch')
    axis[0].set_ylabel('MSE')
    axis[0].set_yscale('log')
    axis[0].legend(loc = 'upper right')

    axis[1].plot(patience_track)
    axis[1].set_xlabel('epoch')
    axis[1].set_ylabel('patience')
    pname = os.path.join(outputdir,'train_plot.jpeg')
    plt.tight_layout()
    plt.savefig(pname)
    plt.close()
     