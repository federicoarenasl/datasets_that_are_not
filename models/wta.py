import torch
from models.convautoencoders import ConvAutoencoder64, ConvAutoencoder128

def wta_spatial_sparsity(encoded_x):
    """
    Applies spatial sparsity to the encoded feature maps by zeroing out all but
    the largest activation within a feature map.
    """
    winner_indices = torch.argmax(encoded_x, dim=1, keepdim=True)
    encoded_wta = torch.zeros_like(encoded_x)
    encoded_wta.scatter_(1, winner_indices, encoded_x.gather(1, winner_indices))
    return encoded_wta

def wta_lifetime_sparsity(encoded_x, k_percent: float = 0.1):
    """
    Applies lifetime sparsity to the encoded feature maps by zeroing out all but
    the top k% activations within a feature map.
    """
    # Calculate the number of top activations to keep
    num_activations = encoded_x.size(0) * k_percent
    num_activations = int(num_activations)
    # Apply lifetime sparsity to each filter across the entire batch
    top_activations, _ = torch.topk(
        encoded_x.view(encoded_x.size(0), -1), num_activations, dim=1
    )
    threshold = top_activations[:, -1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    encoded_wta = torch.where(
        encoded_x >= threshold, encoded_x, torch.zeros_like(encoded_x)
    )
    return encoded_wta


class WTASpatialConvAutoencoder(ConvAutoencoder128):
    def __init__(self):
        super(WTASpatialConvAutoencoder, self).__init__()
        self.name = "WTASpatialConvAutoencoder"

    def forward(self, x):
        """
        Forward pass through the network using winner-takes-all sparsity, which
        works by zeroing out all but the largest activation within a feature map.
        """
        # Spatial only Winner-Takes-All mechanism in the encoder
        encoded = self.encoder(x)
        spatial_encoded = wta_spatial_sparsity(encoded)
        decoded = self.decoder(spatial_encoded)
        return decoded

class WTALifetimeSparse(ConvAutoencoder128):
    def __init__(self, k_percentage: float = 0.1):
        super(WTALifetimeSparse, self).__init__()
        self.name = "WTALifetimeSparseConvAutoencoder"
        self.k_percent = k_percentage

    def forward(self, x):
        """
        Forward pass through the network using winner-takes-all lifetime sparsity,
        which works by zeroing out all but the top k% activations within a feature.
        """
        encoded = self.encoder(x)
        lifetime_encoded = wta_lifetime_sparsity(encoded, self.k_percent)
        decoded = self.decoder(lifetime_encoded)
        return decoded

class WTA128(ConvAutoencoder128):
    def __init__(self, k_percentage: float = 0.1):
        super(WTA128, self).__init__()
        self.name = "WTAConvAutoencoder128"
        self.k_percent = k_percentage
        
    def forward(self, x):
        """
        Forward pass through the network using winner-takes-all spatial sparsity
        to select top activation within each feature map, then use lifetime 
        sparsity, which selects top k% activations across the entire batch.
        """
        encoded = self.encoder(x)
        spatial_encoded = wta_spatial_sparsity(encoded)
        lifetime_encoded = wta_lifetime_sparsity(spatial_encoded, self.k_percent)
        decoded = self.decoder(lifetime_encoded)

        return decoded

class WTA64(ConvAutoencoder64):
    def __init__(self, k_percentage: float = 0.1):
        super(WTA64, self).__init__()
        self.name = "WTAConvAutoencoder64"
        self.k_percent = k_percentage 

    def forward(self, x):
        """
        Forward pass through the network using winner-takes-all spatial sparsity
        to select top activation within each feature map, then use lifetime 
        sparsity, which selects top k% activations across the entire batch.
        """
        encoded = self.encoder(x)
        spatial_encoded = wta_spatial_sparsity(encoded)
        lifetime_encoded = wta_lifetime_sparsity(spatial_encoded, self.k_percent)
        decoded = self.decoder(lifetime_encoded)

        return decoded

