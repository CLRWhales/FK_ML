#this is the same unet architecture as the deno das framework as fara as i can tell
import torch
import torch.nn as nn

class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def deconv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(out_ch,int(out_ch/2), kernel_size=2,stride=2)
            )
        def deconv_final(in_ch,out_ch,end):
           return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch,end, kernel_size=1,stride=1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

        # Encoder
        self.pool = nn.MaxPool2d(kernel_size=2,stride = 2)
        self.enc1 = conv_block(1, 16)
        self.enc2 = conv_block(16, 32)
        self.enc3 = conv_block(32, 64)
        self.enc4 = conv_block(64, 128)
        self.enc5 = conv_block(128, 256)

        self.drop = nn.Sequential(nn.Dropout2d(),
                                  nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride = 2))

        #decoder
        self.dec4 = deconv_block(256,128)
        self.dec3 = deconv_block(128,64)
        self.dec2 = deconv_block(64,32)
        self.dec1 = deconv_final(32,16,1)

    def encode(self,x): 
        skip1 = self.enc1(x)  
        skip2 = self.enc2(self.pool(skip1)) 
        skip3 = self.enc3(self.pool(skip2)) 
        skip4 = self.enc4(self.pool(skip3))
        lowest = self.enc5(self.pool(skip4))   
        return skip1,skip2,skip3,skip4,lowest

    def decode(self,skip1,skip2,skip3,skip4,lowest):
        L_out = self.drop(lowest)  # [B, 128*8*16]
        L_out = self.dec4(torch.concat([skip4,L_out], axis = 1))  
        L_out = self.dec3(torch.concat([skip3,L_out], axis = 1))
        L_out = self.dec2(torch.concat([skip2,L_out], axis = 1))
        L_out = self.dec1(torch.concat([skip1,L_out], axis = 1))
        return L_out

    def forward(self, x):
        skip1,skip2,skip3,skip4,lowest = self.encode(x)
        dec1 = self.decode(skip1,skip2,skip3,skip4,lowest)
        return dec1

# if __name__ == '__main__':
#     from torchsummary import summary
#     model = UNET()
#     summary(model,input_size=(256,512))