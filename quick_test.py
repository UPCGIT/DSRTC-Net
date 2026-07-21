"""Minimal synthetic quick test for DSRTC-Net.

This script verifies that the model, decoder, and DSM-guided regularization
can complete one forward and backward pass without requiring external data.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F

from DSRTCNet import DSRTCNet
from loss_function import DSMEnhancedRegularizationLoss


def main() -> None:
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small synthetic tensors for a fast functionality check.
    batch_size = 1
    num_bands = 16
    num_endmembers = 4
    lidar_dims = 8
    height = 8
    width = 8

    hsi = torch.rand(batch_size, num_bands, height, width, device=device)
    lidar = torch.rand(batch_size, lidar_dims, height, width, device=device)
    dsm = torch.rand(height, width, device=device)

    model = DSRTCNet(num_bands, num_endmembers, lidar_dims).to(device)
    model.train()

    abundance, endmembers, reconstructed = model(hsi, lidar)

    dse_reg = DSMEnhancedRegularizationLoss(lambda_reg=1e-3, alpha=0.05)
    reconstruction_loss = F.mse_loss(reconstructed, hsi)
    regularization_loss = dse_reg(abundance, dsm)
    total_loss = reconstruction_loss + regularization_loss
    total_loss.backward()

    assert abundance.shape == (batch_size, num_endmembers, height, width)
    assert reconstructed.shape == hsi.shape
    assert endmembers.shape == (num_bands, num_endmembers, 1, 1)
    assert torch.isfinite(total_loss).item()

    print("DSRTC-Net quick test passed.")
    print(f"Device: {device}")
    print(f"Abundance shape: {tuple(abundance.shape)}")
    print(f"Endmember shape: {tuple(endmembers.shape)}")
    print(f"Reconstruction shape: {tuple(reconstructed.shape)}")
    print(f"Total loss: {total_loss.item():.6f}")


if __name__ == "__main__":
    main()
