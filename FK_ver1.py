#updated autoencoder based on spectrogram classification
import torch.nn as nn

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
        self.enc1 = conv_block(1, 8) #n 8 128 256
        self.enc2 = conv_block(8, 16) # n 16 64 128
        self.enc3 = conv_block(16, 32) # n 32 32 64
        self.enc4 = conv_block(32, 64) # n 64 16 32
        self.enc5 = conv_block(64, 128) # n 128 8 16
        self.enc6 = conv_block(128, 128) # n 128 4 8
        self.flat1 = nn.Flatten()
        self.bottleneck = recoded(128 * 4 * 8, 128)

        # Decoder
        self.open = recoded(128, 128 * 4 * 8)
        self.uflat1 = nn.Unflatten(1, (128, 4, 8))
        self.dec6 = deconv_block(128, 128) #n 128 8 16
        self.dec5 = deconv_block(128, 64) # n 64 16 32
        self.dec4 = deconv_block(64, 32) # n 32 32 64
        self.dec3 = deconv_block(32, 16) #n 16 64 128
        self.dec2 = deconv_block(16, 8) #n 8 128 256
        self.dec1 = deconv_block(8, 8) #n 8 256 512
        self.dec0 = nn.Conv2d(in_channels=8,out_channels=1,kernel_size=1,stride=1)# back to 1 channel, seperate because no/linear activation

    def encode(self,x): #could likely make all these sequential but this framing allows for future use with ski connections
        x = self.enc1(x)   # [B, 8, 128, 256]
        x = self.enc2(x)   # [B, 16, 64, 128]
        x = self.enc3(x)   # [B, 32, 32, 64]
        x = self.enc4(x)   # [B, 64, 16, 32]
        x = self.enc5(x)   # [B, 128, 8, 16]
        x = self.enc6(x)   # [B, 128, 4, 8]
        x = self.flat1(x)  # [B, 128*8*16]
        x = self.bottleneck(x)  # [B, 9]
        return x
    
    def decode(self,x):
        x = self.open(x)   # [B, 128*8*16]
        x = self.uflat1(x)  # [B, 128, 8, 16]
        x = self.dec6(x)
        x = self.dec5(x)  # [B, 64, 16, 32]
        x = self.dec4(x)  # [B, 32, 32, 64]
        x = self.dec3(x)  # [B, 16, 64, 128]
        x = self.dec2(x)  # [B, 8, 128, 256]
        x = self.dec1(x)  # [B, 8, 256, 512]
        x = self.dec0(x) # B,1,256,512
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

