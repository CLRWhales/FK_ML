import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down(x)

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 32),         # 256x512 → 256x512
            Downsample(32, 64),       # → 128x256
            ConvBlock(64, 64),
            Downsample(64, 128),      # → 64x128
            ConvBlock(128, 128),
            Downsample(128, 128),     # → 32x64
            ConvBlock(128, 128),
            Downsample(128, 64),      # → 16x32
            ConvBlock(64, 64),
            Downsample(64, 32),       # → 8x16
            ConvBlock(32, 32),        # 8x16
            Downsample(32, 16),       # → 4x8
        )

        self.flatten = nn.Flatten()              # → 16*4*8 = 512
        self.fc = nn.Linear(512, latent_dim)     # → latent vector

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z

class Decoder (nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 32),         # 256x512 → 256x512
            Upsample(32, 64),       # → 128x256
            ConvBlock(64, 64),
            Downsample(64, 128),      # → 64x128
            ConvBlock(128, 128),
            Downsample(128, 128),     # → 32x64
            ConvBlock(128, 128),
            Downsample(128, 64),      # → 16x32
            ConvBlock(64, 64),
            Downsample(64, 32),       # → 8x16
            ConvBlock(32, 32),        # 8x16
            Downsample(32, 16),       # → 4x8
        )

        self.flatten = nn.Flatten()              # → 16*4*8 = 512
        self.fc = nn.Linear(512, latent_dim)     # → latent vector

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z
    

class preprocessing(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch, kernel_size=(1,2),stride = (1,2))
        )

    def forward(self, x):
        return self.block(x)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.blockA = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.blockB = nn.Sequential(
            nn.AvgPool2d(kernel_size=2,stride = 2),
            nn.Conv2d(in_ch,out_ch,kernel_size=1)
        )
    def forward(self, x):
        return self.blockA(x)