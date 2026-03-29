import numpy as np
import librosa
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
