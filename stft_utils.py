import librosa
import numpy as np

from backend import has_torch, resolve_device, to_numpy, to_torch


def _use_torch_backend(backend, device):
    if backend == "torch":
        return True
    if backend == "librosa":
        return False
    return has_torch() and device != "cpu"


def stft(
    y,
    n_fft=2048,
    hop_length=256,
    win_length=2048,
    window="hann",
    center=True,
    backend="auto",
    device="cpu",
):
    if _use_torch_backend(backend, device):
        if window != "hann":
            raise ValueError("The torch STFT backend currently supports only the hann window.")
        torch = __import__("torch")
        device = resolve_device(device)
        y_t = to_torch(np.asarray(y, dtype=np.float32), device=device, dtype=torch.float32)
        window_t = torch.hann_window(win_length, periodic=True, device=device, dtype=torch.float32)
        S = torch.stft(
            y_t,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window_t,
            center=center,
            pad_mode="constant",
            return_complex=True,
        )
        return to_numpy(S)
    return librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
    )


def istft(
    S,
    hop_length=256,
    win_length=2048,
    window="hann",
    center=True,
    length=None,
    backend="auto",
    device="cpu",
):
    if _use_torch_backend(backend, device):
        if window != "hann":
            raise ValueError("The torch ISTFT backend currently supports only the hann window.")
        torch = __import__("torch")
        device = resolve_device(device)
        S_t = to_torch(S, device=device, dtype=torch.complex64)
        window_t = torch.hann_window(win_length, periodic=True, device=device, dtype=torch.float32)
        y = torch.istft(
            S_t,
            n_fft=(S_t.shape[0] - 1) * 2,
            hop_length=hop_length,
            win_length=win_length,
            window=window_t,
            center=center,
            length=length,
        )
        return to_numpy(y)
    return librosa.istft(
        S,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        length=length,
    )


def mag_phase(S):
    return np.abs(S), np.angle(S)
