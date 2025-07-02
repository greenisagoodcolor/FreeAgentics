"""Minimal script to test for global isinstance/type errors."""

import torch

print("torch version:", torch.__version__)
t = torch.tensor([1.0, 2.0, 3.0])
print("Tensor:", t)
print("isinstance(t, torch.Tensor):", isinstance(t, torch.Tensor))
print("type(t):", type(t))
