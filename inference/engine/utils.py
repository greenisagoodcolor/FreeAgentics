"""
Module for FreeAgentics Active Inference implementation.
"""

import torch


def to_one_hot(tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Converts a tensor of indices to a one-hot representation"""
    # PyTorch one_hot requires LongTensor
    if tensor.dtype != torch.long:
        tensor = tensor.long()
    return torch.nn.functional.one_hot(tensor, num_classes=num_classes)


def normalize_beliefs(beliefs: torch.Tensor) -> torch.Tensor:
    """Normalizes a belief distribution"""
    return beliefs / beliefs.sum(dim=-1, keepdim=True)
