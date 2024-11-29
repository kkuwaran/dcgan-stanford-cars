import torch
import torch.nn as nn


def initialize_weights(model: nn.Module) -> None:
    """Custom weight initialization."""

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


def get_positive_labels(size: int, device: torch.device, smoothing_ratio: float = 0.5, 
                        flip_ratio: float = 0.05) -> torch.Tensor:
    """Generates positive labels with optional label smoothing and flipping."""

    # Create a tensor of positive labels (1.0)
    labels = torch.full((size, ), 1.0, device=device)

    # Apply label smoothing if smoothing_ratio is greater than 0
    if smoothing_ratio > 0:
        num_to_smooth = int(smoothing_ratio * size)
        indices = torch.randperm(size)[:num_to_smooth]
        labels[indices] = 0.85  # Smooth the selected labels to 0.9

    # Apply label flipping if flip_ratio is greater than 0
    if flip_ratio > 0:
        num_to_flip = int(flip_ratio * size)
        indices = torch.randperm(size)[:num_to_flip]
        labels[indices] = 0  # Flip the selected labels to 0

    return labels