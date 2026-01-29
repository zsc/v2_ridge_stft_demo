import numpy as np
import librosa
from scipy.optimize import nnls


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


def mel_forward(F_mel, A):
    return F_mel @ A


def mel_inverse_pinv(F_mel, M):
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
