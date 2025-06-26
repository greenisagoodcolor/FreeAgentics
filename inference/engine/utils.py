import torch


def to_one_hot(tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Converts a tensor of indices to a one-hot representation."""
    return torch.nn.functional.one_hot(tensor, num_classes=num_classes)


def normalize_beliefs(beliefs: torch.Tensor) -> torch.Tensor:
    """Normalizes a belief distribution."""
    return beliefs / beliefs.sum(dim=-1, keepdim=True)
