import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def has_torch():
    return torch is not None


def get_torch():
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for the torch backend. Install a CPU or CUDA build of torch first."
        )
    return torch


def resolve_device(device="auto"):
    torch_mod = get_torch()
    if device == "auto":
        return "cuda" if torch_mod.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch_mod.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return "cuda"
    if device == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device: {device}")


def to_torch(array, device, dtype=None):
    torch_mod = get_torch()
    if isinstance(array, torch_mod.Tensor):
        return array.to(device=device, dtype=dtype or array.dtype)
    return torch_mod.as_tensor(array, device=device, dtype=dtype)


def to_numpy(array):
    if torch is not None and isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)
