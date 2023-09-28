import torch
from torch import nn
from torch.nn import functional as F
from datetime import datetime
from models import convautoencoders
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import datasets, transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from tools.dataset import split_dataset

def train_for_n_epochs(
    N_EPOCHS: int,
    VISUALIZE_EVERY: int,
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    validation_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    writer: SummaryWriter = SummaryWriter(
        f"runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
    checkpoint_dir: str = "models/",
    ):
    """
    Train the model for N_EPOCHS epochs, visualizing the results every

    Parameters
    ----------
    N_EPOCHS : int
        Number of epochs to train the model for
    VISUALIZE_EVERY : int
        Visualize the results every VISUALIZE_EVERY epochs
    model : nn.Module
        Model to train
    train_dataloader : torch.utils.data.DataLoader
        Training data loader
    validation_dataloader : torch.utils.data.DataLoader
        Validation data loader
    optimizer : torch.optim.Optimizer
        Optimizer to use for training
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to use for training
    writer : SummaryWriter
        TensorBoard writer
    checkpoint_dir : str
        Directory to save checkpoints to
    """
    i = 0
    print(f"Training {model.name} for {N_EPOCHS} epochs...")
    for epoch in tqdm(range(0, N_EPOCHS)):
        train_loss = 0.0
        for i, (images, _) in enumerate(train_dataloader):
            optimizer.zero_grad()
            images = images.to(device)
            out = model(images)
            loss = criterion(out, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            writer.add_scalar(f"train_loss", loss.item(), i)
            i += images.size(0)

        val_loss = 0.0
        if epoch % VISUALIZE_EVERY == 0:
            with torch.no_grad():
                val_images, _ = next(iter(validation_dataloader))
                val_images = val_images.to(device)
                # Forward pass
                output = model(val_images)
                # Calculate reconstruction loss
                loss = criterion(output, val_images)
                val_loss += loss.item()
                # Log images to TensorBoard
                writer.add_images("Input Images", val_images, global_step=epoch)
                writer.add_images("Reconstructed Images", output, global_step=epoch)

        # print avg training statistics
        train_loss = train_loss / len(train_dataloader)
        val_loss = val_loss / len(validation_dataloader)
        writer.add_scalar(f"avg_train_loss", train_loss, epoch)
        writer.add_scalar(f"avg_val_loss", train_loss, epoch)

        # Save checkpoints to checkpoint_path
        checkpoint_path = \
            f"{model.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"+\
            f"/epoch_{epoch}.pth"
        torch.save(\
            model.state_dict(), 
            checkpoint_path=f"{checkpoint_dir}/{checkpoint_path}")


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="ConvAutoencoder", help="Model to train"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--visualize_every", type=int, default=1, help="Visualize every"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    if args.model not in [
        "ConvAutoencoder",
        "WTAConvAutoencoder",
        "WTALifetimeSparseConvAutoencoder",
    ]:
        raise ValueError("Model not supported")
    # Get model from models.py based on --model argument
    device = torch.device(args.device)
    model = getattr(convautoencoders, args.model)()
    if args.model == "WTALifetimeSparseConvAutoencoder":
        parser.add_argument("--k_percent", type=float, default=0.1, help="k percent")
        args = parser.parse_args()
        model.k_percentage = args.k_percent
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    # Load dataset and split into training and validation sets
    dataset = datasets.MNIST(
        root="data", train=True, transform=transforms.ToTensor(), download=True
    )
    train_loader, validation_loader = split_dataset(
        dataset, batch_size=args.batch_size, validation_split=0.2, random_seed=args.seed
    )
    # Set optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    # Train the model
    train_for_n_epochs(
        N_EPOCHS=args.epochs,
        VISUALIZE_EVERY=args.visualize_every,
        model=model,
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )
