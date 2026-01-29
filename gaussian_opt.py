import numpy as np
import cvxpy as cp


def _gaussian_1d(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) ** 2) / (sigma ** 2))


def build_time_centers(T, spacing_frames=None):
    if spacing_frames is None:
        spacing_frames = max(1, int(round(T / 32)))
    spacing_frames = max(1, int(spacing_frames))
    centers = np.arange(0, T, spacing_frames, dtype=int)
    return centers


def pick_freq_centers(R, V, M, t_c, k):
    idx = np.where(R[:, t_c] > 0)[0]
    if idx.size == 0:
        idx = np.where(V[:, t_c])[0]
    if idx.size == 0:
        idx = np.array([int(np.argmax(M[:, t_c]))])
    if idx.size > k:
        values = R[idx, t_c] if (R[:, t_c] > 0).any() else M[idx, t_c]
        topk_local = np.argpartition(values, -k)[-k:]
        idx = idx[topk_local]
    return idx.tolist()


def build_dictionary(n_mels, T, time_centers, freq_centers_map, sigma_t, sigma_f_list):
    mel_axis = np.arange(n_mels)
    time_axis = np.arange(T)

    basis_list = []
    for t_c in time_centers:
        g_t = _gaussian_1d(time_axis, t_c, sigma_t)
        for f_c in freq_centers_map[t_c]:
            for sigma_f in sigma_f_list:
                g_f = _gaussian_1d(mel_axis, f_c, sigma_f)
                basis = np.outer(g_f, g_t)
                basis_list.append(basis.reshape(-1))
    if not basis_list:
        return np.zeros((n_mels * T, 0), dtype=np.float32)
    B = np.stack(basis_list, axis=1).astype(np.float32)
    return B


def gaussian_optimize(E, R, V, M, k, spacing_frames, sigma_t, sigma_f_list, lam_g, solver="SCS"):
    n_mels, T = E.shape
    time_centers = build_time_centers(T, spacing_frames)
    freq_centers_map = {}
    for t_c in time_centers:
        freq_centers_map[t_c] = pick_freq_centers(R, V, M, t_c, k)

    B = build_dictionary(n_mels, T, time_centers, freq_centers_map, sigma_t, sigma_f_list)
    if B.shape[1] == 0:
        return np.zeros_like(E)

    target = E.reshape(-1)
    c = cp.Variable(B.shape[1])
    objective = 0.5 * cp.sum_squares(B @ c - target) + lam_g * cp.norm1(c)
    constraints = [c >= 0]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve(solver=solver, verbose=False)
    except Exception:
        problem.solve(solver="SCS", verbose=False)
    c_val = c.value
    if c_val is None:
        raise RuntimeError("Gaussian optimization failed to converge")

    G = (B @ c_val).reshape(n_mels, T)
    return np.clip(G, 0.0, None)
