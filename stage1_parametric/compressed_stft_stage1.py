from __future__ import annotations

import numpy as np
import librosa
from scipy.ndimage import convolve1d

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from .parametric_stage1 import (
    extract_center_idx_matrix,
    fit_strength_matrix_nnls,
    render_from_slots_numpy,
    render_from_slots_torch,
    slot_strength_to_coeff_map,
)


def whisper_power_spectrogram(
    y,
    sr,
    target_sr=16000,
    n_fft=400,
    hop_length=160,
    win_length=400,
    center=True,
    drop_last=True,
):
    y = np.asarray(y, dtype=np.float32)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    S = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=center,
    )
    power = np.abs(S) ** 2
    if drop_last and power.shape[1] > 0:
        power = power[:, :-1]
    return power.astype(np.float32), sr


def compress_whisper_power(
    power,
    clamp_min=1e-10,
    top_db=8.0,
    offset=4.0,
    scale=4.0,
    reference_max_log10=None,
    return_stats=False,
):
    power = np.asarray(power, dtype=np.float32)
    log_spec = np.log10(np.clip(power, clamp_min, None))
    max_log10 = float(np.max(log_spec)) if reference_max_log10 is None else float(reference_max_log10)
    floor_log10 = max_log10 - float(top_db)
    log_spec = np.maximum(log_spec, floor_log10)
    compressed = (log_spec + float(offset)) / float(scale)

    if not return_stats:
        return compressed.astype(np.float32)

    stats = {
        "max_log10": max_log10,
        "floor_log10": floor_log10,
        "compressed_floor": (floor_log10 + float(offset)) / float(scale),
        "compressed_max": float((max_log10 + float(offset)) / float(scale)),
    }
    return compressed.astype(np.float32), stats


def compress_whisper_power_torch(
    power,
    clamp_min=1e-10,
    top_db=8.0,
    offset=4.0,
    scale=4.0,
    reference_max_log10=None,
):
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")

    if not isinstance(power, torch.Tensor):
        power = torch.as_tensor(power, dtype=torch.float32)
    power = power.to(dtype=torch.float32)

    log_spec = power.clamp_min(clamp_min).log10()
    if reference_max_log10 is None:
        max_log10 = log_spec.max()
    else:
        max_log10 = torch.as_tensor(
            reference_max_log10,
            dtype=log_spec.dtype,
            device=log_spec.device,
        )
    floor_log10 = max_log10 - float(top_db)
    log_spec = torch.maximum(log_spec, floor_log10)
    return (log_spec + float(offset)) / float(scale)


def decompress_whisper_compressed_stft(compressed, offset=4.0, scale=4.0):
    compressed = np.asarray(compressed, dtype=np.float32)
    log_spec = compressed * float(scale) - float(offset)
    power = np.power(10.0, log_spec, dtype=np.float32)
    return power.astype(np.float32)


def shift_compressed_stft_for_support(compressed, floor_value=None):
    compressed = np.asarray(compressed, dtype=np.float32)
    if floor_value is None:
        floor_value = float(np.min(compressed))
    shifted = compressed - float(floor_value)
    return shifted.astype(np.float32), float(floor_value)


def _gaussian_kernel_1d(sigma_bins, truncate=3.0):
    sigma_bins = float(sigma_bins)
    if sigma_bins <= 0.0:
        return None
    radius = max(1, int(np.ceil(truncate * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def build_thickened_stft_basis(n_bins, sigma_bins, normalize="peak", blend_mode="max"):
    n_bins = int(n_bins)
    eye = np.eye(n_bins, dtype=np.float32)
    kernel = _gaussian_kernel_1d(sigma_bins)
    if kernel is None:
        basis = eye
    else:
        blurred = convolve1d(eye, kernel, axis=0, mode="nearest")
        if blend_mode == "max":
            basis = np.maximum(eye, blurred)
        elif blend_mode == "sum":
            basis = eye + blurred
        elif blend_mode == "blur":
            basis = blurred
        else:
            raise ValueError(f"Unsupported blend_mode={blend_mode}")

    if normalize == "peak":
        scale = np.maximum(np.max(basis, axis=0, keepdims=True), 1e-8)
        basis = basis / scale
    elif normalize == "sum":
        scale = np.maximum(np.sum(basis, axis=0, keepdims=True), 1e-8)
        basis = basis / scale
    elif normalize not in (None, "none"):
        raise ValueError(f"Unsupported normalize={normalize}")

    return np.clip(basis.astype(np.float32), 0.0, None)


def fit_topk_thickened_stft_bands(power, center_mask, sigma_bins, clip_upper=True):
    power = np.asarray(power, dtype=np.float32)
    center_mask = np.asarray(center_mask, dtype=bool)
    if power.shape != center_mask.shape:
        raise ValueError("power and center_mask must have the same shape [n_bins, T].")

    max_active = int(np.max(np.sum(center_mask, axis=0))) if center_mask.shape[1] > 0 else 0
    center_idx, _ = extract_center_idx_matrix(center_mask.astype(np.float32), k=max_active, sort_by="mel_asc")
    basis = build_thickened_stft_basis(power.shape[0], sigma_bins=sigma_bins, normalize="peak")
    strength, rendered = fit_strength_matrix_nnls(power, center_idx, basis, clip_upper=clip_upper)
    coeff_map = slot_strength_to_coeff_map(center_idx, strength, power.shape[0])
    return rendered, coeff_map, center_idx, basis


def render_stft_power_from_slots_numpy(center_idx, strength, basis, clip_upper=None):
    return render_from_slots_numpy(center_idx, strength, basis, clip_upper=clip_upper)


def render_stft_power_from_slots_torch(center_idx, strength, basis, clip_upper=None, device=None):
    return render_from_slots_torch(
        center_idx,
        strength,
        basis,
        clip_upper=clip_upper,
        device=device,
    )
