from __future__ import annotations

import os

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import convolve1d
from scipy.optimize import nnls
from scipy.signal import find_peaks

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - optional dependency
    cp = None


def load_audio(path, sr=None, mono=True):
    data, native_sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr is not None and sr != native_sr:
        data = librosa.resample(data.astype(np.float32), orig_sr=native_sr, target_sr=sr)
        native_sr = sr
    if mono and data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), native_sr


def save_audio(path, audio, sr):
    sf.write(path, audio, sr)


def has_torch():
    return torch is not None


def resolve_device(device="auto"):
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return "cuda"
    if device == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device: {device}")


def to_torch(array, device, dtype=None):
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=dtype or array.dtype)
    return torch.as_tensor(array, device=device, dtype=dtype)


def to_numpy(array):
    if torch is not None and isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def peak_mask(M, distance=2, prominence=None, height=None, dilate=False):
    n_bins, T = M.shape
    mask = np.zeros((n_bins, T), dtype=bool)
    for t in range(T):
        peaks, _ = find_peaks(
            M[:, t],
            distance=distance,
            prominence=prominence,
            height=height,
        )
        if peaks.size > 0:
            mask[peaks, t] = True
    if dilate:
        dilated = mask.copy()
        if n_bins > 1:
            dilated[1:, :] |= mask[:-1, :]
            dilated[:-1, :] |= mask[1:, :]
        mask = dilated
    return mask


def _solve_ridge_cvxpy(M, V, lam_sparse, lam_tv, solver):
    if cp is None:
        raise RuntimeError("CVXPY is not installed, so the cvxpy backend is unavailable.")
    n_bins, T = M.shape
    V_float = V.astype(np.float32)
    R = cp.Variable((n_bins, T))
    tv_time = cp.sum(cp.abs(R[:, 1:] - R[:, :-1])) if T > 1 else 0
    objective = 0.5 * cp.sum_squares(R - M) + lam_sparse * cp.norm1(R) + lam_tv * tv_time
    constraints = [R >= 0, R <= cp.multiply(M, V_float)]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve(solver=solver, verbose=False)
    except Exception:
        problem.solve(solver="SCS", verbose=False)
    R_val = R.value
    if R_val is None:
        raise RuntimeError("Ridge optimization failed to converge")
    R_val = np.maximum(R_val, 0.0)
    R_val[V == 0] = 0.0
    return R_val.astype(np.float32)


def _forward_diff_time(x):
    return x[:, 1:] - x[:, :-1]


def _adjoint_diff_time(y):
    x = y.new_zeros((y.shape[0], y.shape[1] + 1))
    x[:, 0] = -y[:, 0]
    x[:, 1:-1] = y[:, :-1] - y[:, 1:]
    x[:, -1] = y[:, -1]
    return x


def _prox_data_l1_box(v, M, U, lam_sparse, tau):
    x = (v + tau * (M - lam_sparse)) / (1.0 + tau)
    return x.clamp_min(0.0).minimum(U)


def _solve_ridge_torch(M, V, lam_sparse, lam_tv, device, max_iter, tol):
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")
    device = resolve_device(device)
    M_t = to_torch(M, device=device, dtype=torch.float32)
    V_t = to_torch(V.astype(np.float32), device=device, dtype=torch.float32)
    U_t = M_t * V_t

    x = (M_t - lam_sparse).clamp_min(0.0).minimum(U_t)
    if x.shape[1] <= 1 or lam_tv <= 0.0:
        x = x.clamp_min(0.0).minimum(U_t)
        return to_numpy(x).astype(np.float32)

    y = torch.zeros((x.shape[0], x.shape[1] - 1), device=device, dtype=x.dtype)
    x_bar = x.clone()
    tau = 0.49
    sigma = 0.49
    theta = 1.0

    for it in range(max_iter):
        y = (y + sigma * _forward_diff_time(x_bar)).clamp(-lam_tv, lam_tv)
        x_next = _prox_data_l1_box(x - tau * _adjoint_diff_time(y), M_t, U_t, lam_sparse, tau)
        x_bar = x_next + theta * (x_next - x)

        if tol > 0 and (it + 1) % 25 == 0:
            diff = torch.linalg.vector_norm(x_next - x).item()
            base = max(torch.linalg.vector_norm(x_next).item(), 1.0)
            if diff <= tol * base:
                x = x_next
                break

        x = x_next

    x = x.clamp_min(0.0).minimum(U_t)
    return to_numpy(x).astype(np.float32)


def _topk_per_frame(R, V, k):
    R_topk = np.zeros_like(R, dtype=np.float32)
    for t in range(R.shape[1]):
        idx = np.where(V[:, t])[0]
        if idx.size == 0:
            continue
        values = R[idx, t]
        if idx.size > k:
            topk_local = np.argpartition(values, -k)[-k:]
            keep = idx[topk_local]
        else:
            keep = idx
        R_topk[keep, t] = R[keep, t]
    return R_topk


def ridge_optimize(
    M,
    V,
    k=3,
    lam_sparse=0.05,
    lam_tv=0.1,
    solver="SCS",
    ridge_refine=True,
    backend="auto",
    device="auto",
    max_iter=300,
    tol=1e-4,
):
    use_torch = backend == "torch" or (backend == "auto" and has_torch())
    if use_torch:
        R_raw = _solve_ridge_torch(M, V, lam_sparse, lam_tv, device, max_iter, tol)
    else:
        R_raw = _solve_ridge_cvxpy(M, V, lam_sparse, lam_tv, solver)
    R_topk = _topk_per_frame(R_raw, V, k)

    if ridge_refine:
        V2 = R_topk > 0
        if V2.any():
            if use_torch:
                R_refined = _solve_ridge_torch(M, V2, lam_sparse, lam_tv, device, max_iter, tol)
            else:
                R_refined = _solve_ridge_cvxpy(M, V2, lam_sparse, lam_tv, solver)
            R_topk = _topk_per_frame(R_refined, V2, k)

    R_topk[V == 0] = 0.0
    return R_topk.astype(np.float32)


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
        max_log10 = torch.as_tensor(reference_max_log10, dtype=log_spec.dtype, device=log_spec.device)
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


def extract_center_idx_matrix(ridge_raw, k=None, sort_by="mel_asc"):
    ridge_raw = np.asarray(ridge_raw, dtype=np.float32)
    if ridge_raw.ndim != 2:
        raise ValueError("ridge_raw must have shape [n_bins, T].")
    support = ridge_raw > 0.0
    max_active = int(np.max(np.sum(support, axis=0))) if ridge_raw.shape[1] > 0 else 0
    k = max_active if k is None else int(k)
    center_idx = np.full((k, ridge_raw.shape[1]), -1, dtype=np.int64)
    valid_mask = np.zeros((k, ridge_raw.shape[1]), dtype=bool)

    for t in range(ridge_raw.shape[1]):
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
        center_idx[: ordered.size, t] = ordered
        valid_mask[: ordered.size, t] = True
    return center_idx, valid_mask


def slot_strength_to_coeff_map(center_idx, strength, n_bins):
    center_idx = np.asarray(center_idx, dtype=np.int64)
    strength = np.asarray(strength, dtype=np.float32)
    coeff_map = np.zeros((n_bins, center_idx.shape[1]), dtype=np.float32)
    for t in range(center_idx.shape[1]):
        valid = center_idx[:, t] >= 0
        idx = center_idx[valid, t]
        coeff_map[idx, t] = strength[valid, t]
    return coeff_map


def coeff_map_to_slot_strength(center_idx, coeff_map):
    center_idx = np.asarray(center_idx, dtype=np.int64)
    coeff_map = np.asarray(coeff_map, dtype=np.float32)
    strength = np.zeros(center_idx.shape, dtype=np.float32)
    for t in range(center_idx.shape[1]):
        valid = center_idx[:, t] >= 0
        idx = center_idx[valid, t]
        strength[valid, t] = coeff_map[idx, t]
    return strength


def render_from_slots_numpy(center_idx, strength, basis, clip_upper=None):
    center_idx = np.asarray(center_idx, dtype=np.int64)
    strength = np.asarray(strength, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)
    safe_idx = np.clip(center_idx, 0, basis.shape[1] - 1)
    valid = (center_idx >= 0).astype(np.float32)
    selected_basis = basis[:, safe_idx]
    rendered = np.sum(selected_basis * strength[None, :, :] * valid[None, :, :], axis=1)
    if clip_upper is not None:
        rendered = np.minimum(rendered, np.asarray(clip_upper, dtype=np.float32))
    return np.clip(rendered, 0.0, None).astype(np.float32)


def render_from_slots_torch(center_idx, strength, basis, clip_upper=None, device=None):
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
    strength = np.zeros(center_idx.shape, dtype=np.float32)
    for t in range(center_idx.shape[1]):
        valid = center_idx[:, t] >= 0
        idx = center_idx[valid, t]
        if idx.size == 0:
            continue
        B = basis[:, idx]
        coeff = nnls(B, M[:, t])[0].astype(np.float32)
        strength[valid, t] = coeff
    clip_target = M if clip_upper else None
    rendered = render_from_slots_numpy(center_idx, strength, basis, clip_upper=clip_target)
    return strength, rendered


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


def reconstruct_griffinlim(
    power,
    n_fft=400,
    hop_length=160,
    win_length=400,
    center=True,
    griffin_lim_iters=32,
    griffin_lim_momentum=0.99,
):
    mag = np.sqrt(np.clip(np.asarray(power, dtype=np.float32), 0.0, None)).astype(np.float32)
    if mag.size == 0 or not np.any(mag > 0.0):
        audio = np.zeros(win_length, dtype=np.float32)
    else:
        audio = librosa.griffinlim(
            mag,
            n_iter=griffin_lim_iters,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            window="hann",
            center=center,
            momentum=griffin_lim_momentum,
            random_state=0,
        ).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak > 0.999:
        audio = audio / peak * 0.999
    return audio.astype(np.float32)


def run_stage1_pipeline(
    input_path,
    sr=None,
    n_fft=400,
    hop_length=160,
    win_length=400,
    center=True,
    peak_distance=2,
    peak_prominence=None,
    peak_height=None,
    peak_dilate=0,
    k=3,
    lam_sparse=0.05,
    lam_tv=0.1,
    backend="auto",
    device="auto",
    solver="SCS",
    ridge_refine=True,
    ridge_max_iter=300,
    ridge_tol=1e-4,
    ridge_stft_sigma=2.0,
    griffin_lim_iters=32,
    griffin_lim_momentum=0.99,
    compute_grad_norm=True,
):
    y, input_sr = load_audio(input_path, sr=sr, mono=True)
    power, actual_sr = whisper_power_spectrogram(
        y,
        input_sr,
        target_sr=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        drop_last=True,
    )
    compressed, comp_stats = compress_whisper_power(power, return_stats=True)
    power_clamped = decompress_whisper_compressed_stft(compressed)
    shifted, floor_value = shift_compressed_stft_for_support(compressed)

    V = peak_mask(
        shifted,
        distance=peak_distance,
        prominence=peak_prominence,
        height=peak_height,
        dilate=bool(peak_dilate),
    )
    ridge_support = ridge_optimize(
        shifted,
        V,
        k=k,
        lam_sparse=lam_sparse,
        lam_tv=lam_tv,
        solver=solver,
        ridge_refine=ridge_refine,
        backend=backend,
        device=device,
        max_iter=ridge_max_iter,
        tol=ridge_tol,
    )
    center_idx, valid_mask = extract_center_idx_matrix(ridge_support, k=k, sort_by="mel_asc")
    basis = build_thickened_stft_basis(power_clamped.shape[0], sigma_bins=ridge_stft_sigma, normalize="peak")
    slot_strength, ridge_power = fit_strength_matrix_nnls(power_clamped, center_idx, basis, clip_upper=True)
    coeff_map = slot_strength_to_coeff_map(center_idx, slot_strength, power_clamped.shape[0])

    reference_max_log10 = comp_stats["max_log10"]
    ridge_compressed = compress_whisper_power(ridge_power, reference_max_log10=reference_max_log10)
    residual_power = np.clip(power_clamped - ridge_power, 0.0, None)
    residual_compressed = compress_whisper_power(residual_power, reference_max_log10=reference_max_log10)
    recon_compressed = compress_whisper_power(
        ridge_power + residual_power,
        reference_max_log10=reference_max_log10,
    )
    ridge_audio = reconstruct_griffinlim(
        ridge_power,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        griffin_lim_iters=griffin_lim_iters,
        griffin_lim_momentum=griffin_lim_momentum,
    )

    grad_norm = None
    if compute_grad_norm and torch is not None:
        strength_t = torch.tensor(slot_strength, dtype=torch.float32, requires_grad=True)
        ridge_power_t = render_from_slots_torch(center_idx, strength_t, basis, clip_upper=None)
        ridge_comp_t = compress_whisper_power_torch(ridge_power_t, reference_max_log10=reference_max_log10)
        loss = ridge_comp_t.square().mean() + 1e-4 * strength_t.square().mean()
        loss.backward()
        grad_norm = float(strength_t.grad.norm().item())

    return {
        "input_path": os.path.abspath(input_path),
        "input_audio": y,
        "input_sr": input_sr,
        "actual_sr": actual_sr,
        "power": power,
        "compressed": compressed,
        "comp_stats": comp_stats,
        "power_clamped": power_clamped,
        "shifted": shifted,
        "support_floor_value": floor_value,
        "peak_mask": V,
        "ridge_support": ridge_support,
        "center_idx": center_idx,
        "valid_mask": valid_mask,
        "basis": basis,
        "slot_strength": slot_strength,
        "coeff_map": coeff_map,
        "ridge_power": ridge_power,
        "ridge_compressed": ridge_compressed,
        "residual_power": residual_power,
        "residual_compressed": residual_compressed,
        "recon_compressed": recon_compressed,
        "ridge_audio": ridge_audio,
        "grad_norm": grad_norm,
        "config": {
            "sr": sr,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "center": center,
            "peak_distance": peak_distance,
            "peak_prominence": peak_prominence,
            "peak_height": peak_height,
            "peak_dilate": peak_dilate,
            "k": k,
            "lam_sparse": lam_sparse,
            "lam_tv": lam_tv,
            "backend": backend,
            "device": device,
            "solver": solver,
            "ridge_refine": ridge_refine,
            "ridge_max_iter": ridge_max_iter,
            "ridge_tol": ridge_tol,
            "ridge_stft_sigma": ridge_stft_sigma,
            "griffin_lim_iters": griffin_lim_iters,
            "griffin_lim_momentum": griffin_lim_momentum,
        },
    }


def save_stage1_artifacts(output_dir, result):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "compressed_stft.npy"), result["compressed"])
    np.save(os.path.join(output_dir, "shifted_stft.npy"), result["shifted"])
    np.save(os.path.join(output_dir, "power.npy"), result["power"])
    np.save(os.path.join(output_dir, "power_clamped.npy"), result["power_clamped"])
    np.save(os.path.join(output_dir, "ridge_support.npy"), result["ridge_support"])
    np.save(os.path.join(output_dir, "center_idx.npy"), result["center_idx"])
    np.save(os.path.join(output_dir, "valid_mask.npy"), result["valid_mask"].astype(np.uint8))
    np.save(os.path.join(output_dir, "slot_strength.npy"), result["slot_strength"])
    np.save(os.path.join(output_dir, "coeff_map.npy"), result["coeff_map"])
    np.save(os.path.join(output_dir, "basis.npy"), result["basis"])
    np.save(os.path.join(output_dir, "ridge_power.npy"), result["ridge_power"])
    np.save(os.path.join(output_dir, "ridge_compressed.npy"), result["ridge_compressed"])
    np.save(os.path.join(output_dir, "residual_power.npy"), result["residual_power"])
    np.save(os.path.join(output_dir, "residual_compressed.npy"), result["residual_compressed"])
    np.save(os.path.join(output_dir, "recon_compressed.npy"), result["recon_compressed"])
    save_audio(os.path.join(output_dir, "ridge_griffinlim.wav"), result["ridge_audio"], result["actual_sr"])
