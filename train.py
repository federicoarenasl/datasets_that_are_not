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

def train_for_n_epochs(
    N_EPOCHS: int,
    VISUALIZE_EVERY: int,
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    validation_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
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
    """
    # Set TensorBoard writer
    writer = SummaryWriter(
        f"logs/{model.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    i = 0
    # Train the model
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

        if epoch % VISUALIZE_EVERY == 0:
            with torch.no_grad():
                val_images, _ = next(iter(validation_dataloader))
                val_images = val_images.to(device)

                # Forward pass
                output = model(val_images)

                # Log images to TensorBoard
                writer.add_images("Input Images", val_images, global_step=epoch)
                writer.add_images("Reconstructed Images", output, global_step=epoch)

        # print avg training statistics
        train_loss = train_loss / len(train_dataloader)
        writer.add_scalar(f"avg_train_loss", train_loss, epoch)


def load_mnist(
    batch_size: int,
    validation_split: float = 0.2,
    random_seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load MNIST dataset

    Parameters
    ----------
    batch_size : int
        Batch size to use for training
    """
    # Load MNIST dataset based on validatio split
    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # load the training and test datasets
    mnist_dataset = datasets.MNIST(
        root="~/.pytorch/MNIST_data/", train=True, download=True, transform=transform
    )

    # Define the split ratio for validation data
    validation_split = 0.2
    num_train = len(mnist_dataset)
    indices = list(range(num_train))
    split = int(validation_split * num_train)

    # Shuffle the indices
    random.seed(random_seed)  # For reproducibility
    random.shuffle(indices)

    # Split the indices into training and validation sets
    train_indices, val_indices = indices[split:], indices[:split]

    # Define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Prepare data loaders (combine dataset and sampler)
    train_dataloader = torch.utils.data.DataLoader(
        mnist_dataset, batch_size=batch_size, sampler=train_sampler
    )
    validation_dataloader = torch.utils.data.DataLoader(
        mnist_dataset, batch_size=batch_size, sampler=val_sampler
    )

    return train_dataloader, validation_dataloader

if __name__ == "__main__":
    import argparse
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="ConvAutoencoder", help="Model to train"
    )
    if args.model not in ["ConvAutoencoder", "WTAConvAutoencoder"]:
        raise ValueError("Model not supported")
    if args.model == "WTALifetimeSparseConvAutoencoder":
        parser.add_argument("--k_percent", type=float, default=0.1, help="k percent")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--visualize_every", type=int, default=1, help="Visualize every"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    # Get model from models.py based on --model argument
    model = getattr(convautoencoders, args.model)()
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    # Set device
    device = torch.device(args.device)
    # Load MNIST dataset
    train_loader, validation_loader = load_mnist(args.batch_size, random_seed=args.seed)
    # Set optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    # Train the model
    train(
        N_EPOCHS=args.epochs,
        VISUALIZE_EVERY=args.visualize_every,
        model=model,
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )
