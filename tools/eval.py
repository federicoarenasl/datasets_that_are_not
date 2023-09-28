import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_reconstructions(
        model:torch.nn.Sequential, 
        test_loader:torch.utils.data.DataLoader, 
        device:torch.device, 
        n_samples=5):
    """
    Visualize the first 5 images of the test set side 
    by side with their reconstructions.
    """
    for i in range(n_samples):
        images, labels = next(iter(test_loader))
        images = images.to(device)
        outputs = model(images)
        outputs = outputs.view(outputs.size(0), 1, 28, 28)
        outputs = outputs.cpu().data.numpy()
        images = images.cpu().data.numpy()
        # Plot the original images and their reconstructions
        fig, ax = plt.subplots(1, 2, figsize=(5, 5))
        ax[0].imshow(np.squeeze(images[i]), cmap="gray")
        ax[1].imshow(np.squeeze(outputs[i]), cmap="gray")
        # Add titles to the plots
        ax[0].set_title("Original")
        ax[1].set_title("Reconstruction")