import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import Sequential

class ConvAutoencoder(nn.Module):
    def __init__(self):
        """Convolutional autoencoder"""
        super(ConvAutoencoder, self).__init__()
        self.encoder = Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class WTAConvAutoencoder(nn.Module):
    def __init__(self, k_percentage=0.1):
        """Convolutional autoencoder with winner-take-all (WTA) layer"""
        super(WTAConvAutoencoder, self).__init__()
        self.encoder = Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.wta = WTALayer(k_percentage)
        self.decoder = Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.wta(x)
        x = self.decoder(x)
        return x

class WTALayer(nn.Module):
    def __init__(self, k_percentage=0.1):
        """Winner-take-all layer"""
        super(WTALayer, self).__init__()
        self.k_percentage = k_percentage

    def forward(self, x):
        """Forward pass"""
        # Flatten the input
        x = x.view(x.size(0), -1)
        # Get the number of features
        n_features = x.size(1)
        # Get the number of winners
        k = int(n_features * self.k_percentage)
        # Get the top-k winners
        _, indices = x.topk(k, dim=1)
        # Create the mask
        mask = torch.zeros_like(x)
        # Set the top-k winners to 1
        mask.scatter_(1, indices, 1)
        # Apply the mask
        x = x * mask
        # Reshape the input
        x = x.view(x.size(0), 64, 1, 1)
        return x