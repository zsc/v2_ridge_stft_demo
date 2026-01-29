import numpy as np
from scipy.signal import find_peaks


def peak_mask(M, distance=2, prominence=None, height=None, dilate=False):
    n_mels, T = M.shape
    mask = np.zeros((n_mels, T), dtype=bool)
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
        if n_mels > 1:
            dilated[1:, :] |= mask[:-1, :]
            dilated[:-1, :] |= mask[1:, :]
        mask = dilated
    return mask
