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
                nn.ReLU(inplace=True)
            )

        def deconv_block(in_ch, out_ch, stride=2):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, output_padding=1),
                nn.ReLU(inplace=True)
            )

        def recoded(in_ch, out_ch):
            return nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = conv_block(1, 8)
        self.enc2 = conv_block(8, 16)
        self.enc3 = conv_block(16, 32)
        self.enc4 = conv_block(32, 64)
        self.enc5 = conv_block(64, 128)

        self.flat1 = nn.Flatten()
        self.bottleneck = recoded(128 * 8 * 16, 9)

        # Decoder
        self.open = recoded(9, 128 * 8 * 16)
        self.uflat1 = nn.Unflatten(1, (128, 8, 16))

        self.dec5 = deconv_block(128, 64)
        self.dec4 = deconv_block(64, 32)
        self.dec3 = deconv_block(32, 16)
        self.dec2 = deconv_block(16, 8)
        self.dec1 = deconv_block(8, 1)  # back to 1 channel

    def forward(self, x):
        # Encode
        x = self.enc1(x)   # [B, 8, 128, 256]
        x = self.enc2(x)   # [B, 16, 64, 128]
        x = self.enc3(x)   # [B, 32, 32, 64]
        x = self.enc4(x)   # [B, 64, 16, 32]
        x = self.enc5(x)   # [B, 128, 8, 16]

        x = self.flat1(x)  # [B, 128*8*16]
        x = self.bottleneck(x)  # [B, 9]
        x = self.open(x)   # [B, 128*8*16]
        x = self.uflat1(x)  # [B, 128, 8, 16]

        # Decode (mirror of encoder)
        x = self.dec5(x)  # [B, 64, 16, 32]
        x = self.dec4(x)  # [B, 32, 32, 64]
        x = self.dec3(x)  # [B, 16, 64, 128]
        x = self.dec2(x)  # [B, 8, 128, 256]
        x = self.dec1(x)  # [B, 1, 256, 512]

        return x

#=== custom crop to remove nyquist ===
class RemoveBottomRow:
    def __call__(self, img):
        w, h = img.size  # PIL Image: (width, height)
        return F.crop(img, top=0, left=0, height=h - 1, width=w)



# === Setup and Example Usage ===
if __name__ == '__main__':
    datapath = 'D:\\DAS\\FK\\DS3_x512_20250530T153412'

    # GPU setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        RemoveBottomRow(), #goes from 257x512 to 256x512 removing the nyquist row
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5],std = [0.5])
    ])

    # Dataset and Dataloaders
    dataset = MaskedImageDataset(datapath, transform=transform, mask_prob=0.3)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=256,num_workers=12 ,shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, num_workers=12,shuffle=False, pin_memory=True)

    # Model, loss, optimizer 
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    train_losses = []
    eval_losses = []

    # Training loop with validation
    for epoch in range(60):
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
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Save checkpoint if best
        e_end = time.perf_counter()
        tdif = e_end - e_start
        print(tdif)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            pathname = os.path.join(datapath, 'Epoch' + str(epoch) + '_model.pth')
            torch.save(model.state_dict(), pathname)
            print("✅ Saved new best model.")
    
    df = pd.DataFrame({'train_loss': train_losses, 'val_loss': eval_losses})
    fname = os.path.join(datapath,'loss_log.csv')
    df.to_csv(fname, index=False)
