from __future__ import annotations

import numpy as np
import librosa
from scipy.ndimage import convolve1d
from scipy.optimize import nnls

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def build_mel_filter(sr, n_fft, n_mels=128, fmin=0.0, fmax=None):
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
    ).astype(np.float32)


def mel_forward(F_mel, A):
    return np.asarray(F_mel, dtype=np.float32) @ np.asarray(A, dtype=np.float32)


def mel_inverse_pinv(F_mel, M):
    F_pinv = np.linalg.pinv(np.asarray(F_mel, dtype=np.float32))
    A_hat = F_pinv @ np.asarray(M, dtype=np.float32)
    return np.clip(A_hat, 0.0, None).astype(np.float32)


def _gaussian_kernel_1d(sigma_bins, truncate=3.0):
    sigma_bins = float(sigma_bins)
    if sigma_bins <= 0.0:
        return None
    radius = max(1, int(np.ceil(truncate * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def linear_frequency_blur(A, sigma_bins):
    kernel = _gaussian_kernel_1d(sigma_bins)
    if kernel is None:
        return np.asarray(A, dtype=np.float32)
    return convolve1d(np.asarray(A, dtype=np.float32), kernel, axis=0, mode="nearest")


def thicken_mel_in_linear_frequency(F_mel, M, sigma_bins, clip_upper=None):
    sigma_bins = float(sigma_bins)
    if sigma_bins <= 0.0:
        return np.clip(np.asarray(M, dtype=np.float32), 0.0, None)

    A_seed = mel_inverse_pinv(F_mel, M)
    A_blur = linear_frequency_blur(A_seed, sigma_bins)
    A_thick = np.maximum(A_seed, A_blur)
    M_thick = mel_forward(F_mel, A_thick)

    if clip_upper is not None:
        M_thick = np.minimum(M_thick, clip_upper)
    return np.clip(M_thick, 0.0, None).astype(np.float32)


def build_thickened_mel_basis(F_mel, sigma_bins, normalize="peak"):
    n_mels = F_mel.shape[0]
    eye = np.eye(n_mels, dtype=np.float32)
    basis = thicken_mel_in_linear_frequency(
        F_mel,
        eye,
        sigma_bins=sigma_bins,
        clip_upper=None,
    )

    if normalize == "peak":
        scale = np.maximum(np.max(basis, axis=0, keepdims=True), 1e-8)
        basis = basis / scale
    elif normalize == "sum":
        scale = np.maximum(np.sum(basis, axis=0, keepdims=True), 1e-8)
        basis = basis / scale
    elif normalize not in (None, "none"):
        raise ValueError(f"Unsupported normalize={normalize}")

    return np.clip(basis.astype(np.float32), 0.0, None)


def extract_center_idx_matrix(ridge_raw, k=None, sort_by="mel_asc"):
    ridge_raw = np.asarray(ridge_raw, dtype=np.float32)
    if ridge_raw.ndim != 2:
        raise ValueError("ridge_raw must have shape [n_mels, T].")

    n_mels, T = ridge_raw.shape
    support = ridge_raw > 0.0
    max_active = int(np.max(np.sum(support, axis=0))) if T > 0 else 0
    k = max_active if k is None else int(k)
    center_idx = np.full((k, T), -1, dtype=np.int64)
    valid_mask = np.zeros((k, T), dtype=bool)

    for t in range(T):
        idx = np.where(support[:, t])[0]
        if idx.size == 0:
            continue

        if sort_by == "mel_asc":
            ordered = idx
        elif sort_by == "value_desc":
            order = np.argsort(ridge_raw[idx, t], kind="stable")[::-1]
            ordered = idx[order]
        else:
            raise ValueError(f"Unsupported sort_by={sort_by}")

        ordered = ordered[:k]
        count = ordered.size
        center_idx[:count, t] = ordered
        valid_mask[:count, t] = True

    return center_idx, valid_mask


def slot_strength_to_coeff_map(center_idx, strength, n_mels):
    center_idx = np.asarray(center_idx, dtype=np.int64)
    strength = np.asarray(strength, dtype=np.float32)
    if center_idx.shape != strength.shape:
        raise ValueError("center_idx and strength must have the same shape [k, T].")

    _, T = center_idx.shape
    coeff_map = np.zeros((n_mels, T), dtype=np.float32)
    for t in range(T):
        valid = center_idx[:, t] >= 0
        idx = center_idx[valid, t]
        coeff_map[idx, t] = strength[valid, t]
    return coeff_map


def coeff_map_to_slot_strength(center_idx, coeff_map):
    center_idx = np.asarray(center_idx, dtype=np.int64)
    coeff_map = np.asarray(coeff_map, dtype=np.float32)
    k, T = center_idx.shape
    strength = np.zeros((k, T), dtype=np.float32)

    for t in range(T):
        valid = center_idx[:, t] >= 0
        idx = center_idx[valid, t]
        strength[valid, t] = coeff_map[idx, t]
    return strength


def render_mel_from_slots_numpy(center_idx, strength, basis, clip_upper=None):
    center_idx = np.asarray(center_idx, dtype=np.int64)
    strength = np.asarray(strength, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)

    if center_idx.shape != strength.shape:
        raise ValueError("center_idx and strength must have the same shape [k, T].")
    if basis.ndim != 2 or basis.shape[0] != basis.shape[1]:
        raise ValueError("basis must have shape [n_mels, n_mels].")

    safe_idx = np.clip(center_idx, 0, basis.shape[1] - 1)
    valid = (center_idx >= 0).astype(np.float32)
    selected_basis = basis[:, safe_idx]
    rendered = np.sum(selected_basis * strength[None, :, :] * valid[None, :, :], axis=1)

    if clip_upper is not None:
        rendered = np.minimum(rendered, np.asarray(clip_upper, dtype=np.float32))
    return np.clip(rendered, 0.0, None).astype(np.float32)


def render_mel_from_slots_torch(center_idx, strength, basis, clip_upper=None, device=None):
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")

    basis_t = torch.as_tensor(basis, dtype=torch.float32, device=device)
    center_idx_t = torch.as_tensor(center_idx, dtype=torch.long, device=device)
    valid_t = (center_idx_t >= 0).to(dtype=basis_t.dtype)
    safe_idx_t = center_idx_t.clamp(min=0)
    flat_idx = safe_idx_t.reshape(-1)
    selected_basis = basis_t.index_select(1, flat_idx).reshape(
        basis_t.shape[0],
        center_idx_t.shape[0],
        center_idx_t.shape[1],
    )

    if isinstance(strength, torch.Tensor):
        strength_t = strength.to(device=basis_t.device, dtype=basis_t.dtype)
    else:
        strength_t = torch.as_tensor(strength, dtype=basis_t.dtype, device=basis_t.device)

    rendered = (selected_basis * strength_t.unsqueeze(0) * valid_t.unsqueeze(0)).sum(dim=1)
    if clip_upper is not None:
        clip_upper_t = torch.as_tensor(clip_upper, dtype=basis_t.dtype, device=basis_t.device)
        rendered = torch.minimum(rendered, clip_upper_t)
    return rendered.clamp_min(0.0)


def fit_strength_matrix_nnls(M, center_idx, basis, clip_upper=True):
    M = np.asarray(M, dtype=np.float32)
    center_idx = np.asarray(center_idx, dtype=np.int64)
    basis = np.asarray(basis, dtype=np.float32)
    if M.ndim != 2:
        raise ValueError("M must have shape [n_mels, T].")
    if basis.shape[0] != M.shape[0] or basis.shape[1] != M.shape[0]:
        raise ValueError("basis must have shape [n_mels, n_mels].")
    if center_idx.ndim != 2 or center_idx.shape[1] != M.shape[1]:
        raise ValueError("center_idx must have shape [k, T].")

    k, T = center_idx.shape
    strength = np.zeros((k, T), dtype=np.float32)

    for t in range(T):
        valid = center_idx[:, t] >= 0
        idx = center_idx[valid, t]
        if idx.size == 0:
            continue
        B = basis[:, idx]
        coeff = nnls(B, M[:, t])[0].astype(np.float32)
        strength[valid, t] = coeff

    clip_target = M if clip_upper else None
    rendered = render_mel_from_slots_numpy(center_idx, strength, basis, clip_upper=clip_target)
    return strength, rendered
