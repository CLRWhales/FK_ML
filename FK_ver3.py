import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.prepro = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2,stride = 2) # n 32 128 256
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # n 64 64 128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # n 128 32 64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # n 256 16 32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), # n 512 8 16
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),  # n 1024 4 8
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, 3, stride=2, padding=1), # n 2048 2 4
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout1d()
        )
        self.fc = nn.Linear(2048*2*4, latent_dim)

    def forward(self, x):
        x = self.prepro(x)
        x = self.cnn(x)
        z = self.fc(x)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 2048*2*4)
        
        self.deconv = nn.Sequential(
            nn.Unflatten(dim = 1,unflattened_size=(2048,2,4)), # n 2048 2 4
            nn.ConvTranspose2d(2048, 1024, 3, stride=2, output_padding=1, padding=1), # n 1024 4 8
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 3, stride=2, output_padding=1, padding=1), # n 512 8 16
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, stride=2, output_padding=1, padding=1), # n 256 16 32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, output_padding=1, padding=1), # n 128 32 64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1, padding=1), # n 64 64 128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1, padding=1), # n 32 128 256
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1, padding=1), # n 1 256 512
            nn.ReLU(),
            nn.Conv2d(16,1,1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.deconv(x)
        return x

class Regressor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(64, 5),  # 5 target variables
            nn.Sigmoid()  # values between 0 and 1
        )

    def forward(self, z):
        return self.mlp(z)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.regressor = Regressor(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        y_pred = self.regressor(z)
        return x_recon, y_pred, z
