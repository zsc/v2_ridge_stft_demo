import numpy as np
import soundfile as sf
import librosa


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
