import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import Sequential


class WTAConvAutoencoder(nn.Module):
    def __init__(self):
        super(WTAConvAutoencoder, self).__init__()
        self.name = "WTAConvAutoencoder"
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max-pooling operation
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=11, padding=5), nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the network using winner-takes-all sparsity, which
        works by zeroing out all but the largest activation within a feature map.
        """
        # Winner-Takes-All mechanism in the encoder
        encoded = self.encoder(x)
        winner_indices = torch.argmax(encoded, dim=1, keepdim=True)
        encoded_wta = torch.zeros_like(encoded)
        encoded_wta.scatter_(1, winner_indices, encoded.gather(1, winner_indices))

        decoded = self.decoder(encoded_wta)
        return decoded


class WTALifetimeSparseConvAutoencoder(nn.Module):
    def __init__(self, k_percentage: float = 0.1):
        super(WTALifetimeSparseConvAutoencoder, self).__init__()
        self.name = "WTALifetimeSparseConvAutoencoder"
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max-pooling operation
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=11, padding=5), nn.Sigmoid()
        )

        self.k_percent = k_percentage  # Choose your desired k percent

    def forward(self, x):
        """
        Forward pass through the network using winner-takes-all lifetime sparsity,
        which works by zeroing out all but the top k% activations within a feature.
        """
        encoded = self.encoder(x)

        # Calculate the number of top activations to keep
        num_activations = encoded.size(0) * self.k_percent
        num_activations = int(num_activations)

        # Apply lifetime sparsity to each filter across the entire batch
        top_activations, _ = torch.topk(
            encoded.view(encoded.size(0), -1), num_activations, dim=1
        )
        threshold = top_activations[:, -1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        encoded_wta = torch.where(
            encoded >= threshold, encoded, torch.zeros_like(encoded)
        )

        decoded = self.decoder(encoded_wta)
        return decoded
