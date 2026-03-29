import numpy as np
import librosa
from scipy.ndimage import convolve1d
from scipy.optimize import nnls

from backend import has_torch, resolve_device, to_numpy, to_torch


def mel_filter(sr, n_fft, n_mels=128, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0
    return librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=False,
        norm="slaney",
    )


def mel_forward(F_mel, A, device="cpu"):
    if has_torch() and device != "cpu":
        torch = __import__("torch")
        device = resolve_device(device)
        F_t = to_torch(F_mel, device=device, dtype=torch.float32)
        A_t = to_torch(A, device=device, dtype=torch.float32)
        return to_numpy(F_t @ A_t)
    return F_mel @ A


def mel_inverse_pinv(F_mel, M, device="cpu"):
    if has_torch() and device != "cpu":
        torch = __import__("torch")
        device = resolve_device(device)
        F_t = to_torch(F_mel, device=device, dtype=torch.float32)
        M_t = to_torch(M, device=device, dtype=torch.float32)
        F_pinv = torch.linalg.pinv(F_t)
        A_hat = F_pinv @ M_t
        return to_numpy(A_hat.clamp_min(0.0))
    F_pinv = np.linalg.pinv(F_mel)
    A_hat = F_pinv @ M
    return np.clip(A_hat, 0.0, None)


def mel_inverse_nnls(F_mel, M):
    n_mels, T = M.shape
    n_freq = F_mel.shape[1]
    A_hat = np.zeros((n_freq, T), dtype=np.float32)
    for t in range(T):
        A_hat[:, t] = nnls(F_mel, M[:, t])[0]
    return np.clip(A_hat, 0.0, None)


def _gaussian_kernel_1d(sigma_bins, truncate=3.0):
    sigma_bins = float(sigma_bins)
    if sigma_bins <= 0.0:
        return None
    radius = max(1, int(np.ceil(truncate * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def linear_frequency_blur(A, sigma_bins, device="cpu"):
    kernel = _gaussian_kernel_1d(sigma_bins)
    if kernel is None:
        return np.asarray(A, dtype=np.float32)

    if has_torch() and device != "cpu":
        torch = __import__("torch")
        torch_f = __import__("torch.nn.functional", fromlist=["functional"])
        device = resolve_device(device)
        A_t = to_torch(A, device=device, dtype=torch.float32)
        kernel_t = to_torch(kernel, device=device, dtype=torch.float32).view(1, 1, -1)
        radius = kernel.shape[0] // 2
        A_batch = A_t.transpose(0, 1).unsqueeze(1)
        A_pad = torch_f.pad(A_batch, (radius, radius), mode="replicate")
        A_blur = torch_f.conv1d(A_pad, kernel_t)
        return to_numpy(A_blur.squeeze(1).transpose(0, 1))

    return convolve1d(np.asarray(A, dtype=np.float32), kernel, axis=0, mode="nearest")


def thicken_mel_in_linear_frequency(F_mel, M, sigma_bins, device="cpu", clip_upper=None):
    sigma_bins = float(sigma_bins)
    if sigma_bins <= 0.0:
        return np.clip(np.asarray(M, dtype=np.float32), 0.0, None)

    A_seed = mel_inverse_pinv(F_mel, M, device=device)
    A_blur = linear_frequency_blur(A_seed, sigma_bins, device=device)
    A_thick = np.maximum(A_seed, A_blur)
    M_thick = mel_forward(F_mel, A_thick, device=device)

    if clip_upper is not None:
        M_thick = np.minimum(M_thick, clip_upper)
    return np.clip(M_thick, 0.0, None)


def build_thickened_mel_basis(F_mel, sigma_bins, device="cpu", normalize="peak"):
    n_mels = F_mel.shape[0]
    eye = np.eye(n_mels, dtype=np.float32)
    basis = thicken_mel_in_linear_frequency(
        F_mel,
        eye,
        sigma_bins=sigma_bins,
        device=device,
        clip_upper=None,
    )

    if normalize == "peak":
        scale = np.maximum(np.max(basis, axis=0, keepdims=True), 1e-8)
        basis = basis / scale
    elif normalize == "sum":
        scale = np.maximum(np.sum(basis, axis=0, keepdims=True), 1e-8)
        basis = basis / scale

    return np.clip(basis.astype(np.float32), 0.0, None)


def fit_topk_thickened_mel_bands(M, center_mask, F_mel, sigma_bins, device="cpu", clip_upper=True):
    sigma_bins = float(sigma_bins)
    M = np.asarray(M, dtype=np.float32)
    center_mask = np.asarray(center_mask, dtype=bool)

    if sigma_bins <= 0.0:
        return np.where(center_mask, M, 0.0).astype(np.float32), np.where(center_mask, M, 0.0).astype(np.float32)

    basis = build_thickened_mel_basis(F_mel, sigma_bins=sigma_bins, device=device, normalize="peak")
    n_mels, T = M.shape
    R = np.zeros_like(M, dtype=np.float32)
    coeff_map = np.zeros_like(M, dtype=np.float32)

    for t in range(T):
        idx = np.where(center_mask[:, t])[0]
        if idx.size == 0:
            continue
        B = basis[:, idx]
        coeff = nnls(B, M[:, t])[0].astype(np.float32)
        frame = B @ coeff
        if clip_upper:
            frame = np.minimum(frame, M[:, t])
        R[:, t] = np.clip(frame, 0.0, None)
        coeff_map[idx, t] = coeff

    return R, coeff_map
