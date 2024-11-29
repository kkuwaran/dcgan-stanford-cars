
from typing import Union
import torch
import torch.nn as nn
import numpy as np

import torch.nn.utils.spectral_norm as spectral_norm


class Generator(nn.Module):
    """Generator class for DCGAN. The generator takes a latent vector and generates an image."""

    def __init__(self, image_size: int, latent_dim: int, feat_map_size: int, num_channels: int) -> None:
        """Initialize the generator with the given hyperparameters."""

        super(Generator, self).__init__()

        # Initialize the sequential container for the blocks
        blocks = nn.Sequential()

        # Number of blocks between the first and the last (excluded)
        n_blocks = int(np.log2(image_size)) - 3

        # Define the first block: 4x4 kernel with stride 1 and no padding
        in_channels = latent_dim
        out_channels = feat_map_size * (2**(n_blocks))
        block = self._transposed_conv_block(in_channels, out_channels, 4, 1, 0, nn.LeakyReLU(0.2))
        blocks.add_module("block_0", block)

        # Define the intermediate blocks: 4x4 kernel with stride 2 and padding 1
        for n in range(n_blocks, 0, -1):
            in_channels = feat_map_size * (2**n)
            out_channels = feat_map_size * (2**(n-1))
            block = self._transposed_conv_block(in_channels, out_channels, 4, 2, 1, nn.LeakyReLU(0.2))
            blocks.add_module(f"block_{n}", block)

        # Define the last block: 4x4 kernel with stride 2 and padding 1
        in_channels = out_channels  # out_channels = feat_map_size
        out_channels = num_channels
        block = self._transposed_conv_block(in_channels, out_channels, 4, 2, 1, nn.Tanh(), bn_flag=False)
        blocks.add_module(f"block_{n_blocks+1}", block)

        # Set the model to the sequential container
        self.model = blocks


    def _transposed_conv_block(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple], 
                               stride: Union[int, tuple], padding: Union[int, tuple], 
                               activation: nn.Module, bn_flag: bool = True) -> nn.Sequential:
        """Creates a transposed convolutional block with optional batch normalization and activation."""

        # Create a transposed convolutional layer
        transposed_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # Apply batch normalization if bn_flag is True, otherwise use an identity layer
        batch_norm = nn.BatchNorm2d(out_channels) if bn_flag else nn.Identity()
        # Create a sequential container with the transposed convolution, batch normalization, and activation
        block = nn.Sequential(transposed_conv, batch_norm, activation)
        return block
    

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Forward pass of the generator."""
        return self.model(latents)
    


class Discriminator(nn.Module):
    """Discriminator class for DCGAN. The discriminator takes an image and classifies it as real or fake."""

    def __init__(self, image_size: int, feat_map_size: int, num_channels: int, dropout: float = 0.0) -> None:
        """Initialize the discriminator with the given hyperparameters."""

        super(Discriminator, self).__init__()

        # Number of blocks between the first and the last (excluded)
        n_blocks = int(np.log2(image_size)) - 3
        # NOTE: total number of blocks is log2(image_size) - 1

        # Initialize the sequential container for the blocks
        blocks = nn.Sequential()

        # Define the first block: 4x4 kernel with stride 2 and padding 1
        in_channels = num_channels
        out_channels = feat_map_size
        block = self._conv_block(in_channels, out_channels, 4, 2, 1, dropout, nn.LeakyReLU(0.2, inplace=True), bn_flag=False)
        blocks.add_module("block_0", block)

        # Define the intermediate blocks: 4x4 kernel with stride 2 and padding 1
        for i in range(n_blocks):
            in_channels = feat_map_size * (2**i)
            out_channels = feat_map_size * (2**(i+1))
            block = self._conv_block(in_channels, out_channels, 4, 2, 1, dropout, nn.LeakyReLU(0.2, inplace=True), bn_flag=True)
            blocks.add_module(f"block_{i+1}", block)

        # Define the last block: 4x4 kernel with stride 1 and no padding
        in_channels = feat_map_size * (2**n_blocks)
        out_channels = 1
        block = self._conv_block(in_channels, out_channels, 4, 1, 0, 0.0, nn.Sigmoid(), bn_flag=False)
        blocks.add_module(f"block_{n_blocks+1}", block)

        # Set the model to the sequential container
        self.model = blocks


    def _conv_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, 
                    dropout_prob: float, activation: nn.Module, bn_flag: bool = True) -> nn.Sequential:
        """Creates a convolutional block with optional batch normalization, dropout, and activation."""
        
        # Create a convolutional layer and apply spectral normalization to stabilize training
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not bn_flag)
        normalized_conv = spectral_norm(conv)
        # Apply dropout if dropout_prob > 0, otherwise use an identity layer
        dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        # Apply batch normalization if bn_flag is True, otherwise use an identity layer
        batch_norm = nn.BatchNorm2d(out_channels) if bn_flag else nn.Identity()
        # Create a sequential container with the convolution, dropout, batch normalization, and activation
        block = nn.Sequential(normalized_conv, dropout, batch_norm, activation)
        return block
    
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass of the discriminator."""
        return self.model(images).view(-1)