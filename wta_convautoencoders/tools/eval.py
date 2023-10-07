import numpy as np
import matplotlib.pyplot as plt
import torch

def load_model_from_checkpoint(model, checkpoint_path, device, eval=False):
    """
    Load model from checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    if eval:
        model.eval()
    return model

def visualize_filters(model, weight):
    """
    Visualize learned convolutional filters
    """
    model.eval()
    fig = plt.figure(figsize=(20, 20))
    for i, filter in enumerate(weight):
        ax = fig.add_subplot(12, 12, i + 1)
        ax.imshow(filter.detach().cpu().squeeze(), cmap="gray")
        ax.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")

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
        images, _ = next(iter(test_loader))
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