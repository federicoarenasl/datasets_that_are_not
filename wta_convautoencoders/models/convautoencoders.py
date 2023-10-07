import torch.nn as nn

class ConvAutoencoder64(nn.Module):
    def __init__(self):
        super(ConvAutoencoder64, self).__init__()
        self.name = "ConvAutoencoder64"
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=11, padding=5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvAutoencoder128(nn.Module):
    def __init__(self):
        super(ConvAutoencoder128, self).__init__()
        self.name = "ConvAutoencoder128"
         # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=11, padding=5), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
