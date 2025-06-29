import torch
import numpy as np
from data import *


def decay_mask(seq_length, gamma):
    # Create a tensor with the powers of gamma
    powers = torch.arange(seq_length).unsqueeze(1) - torch.arange(seq_length).unsqueeze(0)
    # Create the mask using the condition
    mask = torch.where(powers >= 0, gamma ** powers.float(), torch.zeros_like(powers, dtype=torch.float))
    
    return mask


def prefix_mask(seq_length, num_prefixes):
    """
    Create a prefix mask of shape (1, seq_length, seq_length) where the first K entries 
    in each row are 1, and starting from the K+1 row, the first j entries are 1 for row j.

    Args:
        seq_length (int): The sequence length.
        num_prefixes (int): The value determining the behavior of the mask.

    Returns:
        torch.Tensor: A prefix mask tensor of shape (seq_length, seq_length).
    """
    # Initialize the mask with zeros
    mask = torch.zeros(seq_length, seq_length, dtype=torch.float32)
    
    # Set the first K rows to 1
    mask[:num_prefixes, :num_prefixes] = 1

    # Set the remaining rows according to the pattern
    for i in range(num_prefixes, seq_length):
        mask[i, :i + 1] = 1

    return mask

def window_mask(seq_length, window_size):
    """
    Create a window mask of shape (1, seq_length, seq_length) where for each row j,
    the entries from max(0, j-w+1) to j are 1, and the rest are 0.

    Args:
        seq_length (int): The sequence length.
        window_size (int): The size of the window (w).

    Returns:
        torch.Tensor: A window mask tensor of shape (1, seq_length, seq_length).
    """
    # Initialize the mask with zeros
    mask = torch.zeros(seq_length, seq_length, dtype=torch.float32)

    # Fill the mask row by row
    for j in range(seq_length):
        start_idx = max(0, j - window_size + 1)
        mask[j, start_idx:j + 1] = 1

    return mask


