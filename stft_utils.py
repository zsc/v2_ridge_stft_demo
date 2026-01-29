import librosa
import numpy as np


def stft(y, n_fft=2048, hop_length=256, win_length=2048, window="hann", center=True):
    return librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
    )


def istft(S, hop_length=256, win_length=2048, window="hann", center=True, length=None):
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
