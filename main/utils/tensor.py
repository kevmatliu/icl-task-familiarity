import torch


def convert_tensor_scalar(value):
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()

    return value
