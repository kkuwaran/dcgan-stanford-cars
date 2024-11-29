from typing import List, Dict
from IPython.display import clear_output

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

import torch
from torchvision.utils import make_grid



def show_tensor_shape(x: torch.Tensor) -> None:
    """Prints the shape of a tensor."""

    b, c, w, h = x.shape
    print(f"(B, C, W, H): ({b}, {c}, {w}, {h})")


def visualize_batch(batch: torch.Tensor) -> None:
    """Visualizes a batch of images."""

    # Detach the batch from the GPU and move it to the CPU
    b = batch.detach().cpu()

    # Create a figure and axes
    fig, axes = plt.subplots(dpi=150)

    # Make a grid of the images in the batch and normalize them
    image_grid = make_grid(b, padding=0, normalize=True)

    # Make the image grid into a numpy array and transpose it to HWC
    image_grid = np.transpose(image_grid, (1, 2, 0))

    # Display the image grid
    axes.imshow(image_grid)
    axes.axis("off")


def training_tracking(metrics: Dict[str, List[float]], image_batch: torch.Tensor) -> plt.Figure:
    """Visualizes training metrics and generated images.
    metrics (dict): Dictionary containing training metrics with keys "D_loss", "G_loss", and "D_acc".
    """

    # Create a new figure with a specified DPI
    fig = plt.figure(dpi=150)
    gs = gridspec.GridSpec(2, 8)

    # Create subplots
    ax_a = fig.add_subplot(gs[0, :3])  # Top-left subplot for losses
    ax_b = fig.add_subplot(gs[1, :3])  # Bottom-left subplot for accuracy
    ax_c = fig.add_subplot(gs[:, 4:])  # Right subplot spanning both rows for images

    # Plot Discriminator and Generator losses
    ax_a.plot(metrics["D_loss"], label="Discriminator")
    ax_a.plot(metrics["G_loss"], label="Generator")
    ax_a.legend()
    ax_a.set_ylabel("Loss")
    ax_a.set_xlabel("Epoch")
    ax_a.set_title("Discriminator and Generator Losses")
    ax_a.grid(True)

    # Plot Discriminator accuracy
    ax_b.plot(metrics["D_acc"])
    ax_b.set_ylabel("Accuracy")
    ax_b.set_xlabel("Epoch")
    ax_b.set_title("Discriminator Accuracy")
    ax_b.grid(True)

    # Prepare and display the grid of generated images
    image_grid = make_grid(image_batch, padding=0, normalize=True, nrow=4)
    image_grid = np.transpose(image_grid, (1, 2, 0))  # Convert from CHW to HWC format

    ax_c.imshow(image_grid)
    ax_c.axis("off")  # Hide axes for the image grid
    fig.tight_layout()  # Adjust subplots to fit into the figure area

    return fig


def generate_and_save_frame(G: torch.nn.Module, noise: torch.Tensor, metrics: Dict[str, List[float]], 
                            outdir: str, n_frame: int) -> int:
    """Generates images using the generator model, visualizes training metrics, and saves the visualization."""

    # Generate images with the generator model without computing gradients
    with torch.no_grad():
        image_batch = G(noise)
    # Clear the current output in the notebook
    clear_output(wait=True)

    # Create a figure visualizing the training metrics and generated images
    image_batch = image_batch.detach().cpu()
    fig = training_tracking(metrics, image_batch)
    # Save the figure
    fig.savefig(f"{outdir}/frame_{n_frame:05d}.png")
    n_frame += 1
    # Display the figure
    plt.show()

    return n_frame