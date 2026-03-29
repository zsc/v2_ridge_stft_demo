import numpy as np

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - optional dependency
    cp = None

from backend import has_torch, resolve_device, to_numpy, to_torch


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


def _build_dictionary_torch(n_mels, T, time_centers, freq_centers_map, sigma_t, sigma_f_list, device):
    torch = __import__("torch")
    mel_axis = torch.arange(n_mels, device=device, dtype=torch.float32)
    time_axis = torch.arange(T, device=device, dtype=torch.float32)

    basis_list = []
    for t_c in time_centers:
        g_t = torch.exp(-0.5 * ((time_axis - float(t_c)) ** 2) / (float(sigma_t) ** 2))
        for f_c in freq_centers_map[t_c]:
            for sigma_f in sigma_f_list:
                g_f = torch.exp(-0.5 * ((mel_axis - float(f_c)) ** 2) / (float(sigma_f) ** 2))
                basis_list.append((g_f[:, None] * g_t[None, :]).reshape(-1))
    if not basis_list:
        return torch.zeros((n_mels * T, 0), device=device, dtype=torch.float32)
    return torch.stack(basis_list, dim=1)


def _estimate_lipschitz(B, n_iter=20):
    torch = __import__("torch")
    if B.shape[1] == 0:
        return 1.0
    v = torch.ones(B.shape[1], device=B.device, dtype=B.dtype)
    v = v / max(torch.linalg.vector_norm(v).item(), 1e-12)
    for _ in range(n_iter):
        v = B.T @ (B @ v)
        norm = torch.linalg.vector_norm(v).item()
        if norm <= 1e-12:
            return 1.0
        v = v / norm
    Bv = B @ v
    return max(float(torch.dot(Bv, Bv).item()), 1e-6)


def _gaussian_optimize_torch(
    E,
    R,
    V,
    M,
    k,
    spacing_frames,
    sigma_t,
    sigma_f_list,
    lam_g,
    device,
    max_iter,
    tol,
):
    torch = __import__("torch")
    device = resolve_device(device)

    n_mels, T = E.shape
    time_centers = build_time_centers(T, spacing_frames)
    freq_centers_map = {}
    for t_c in time_centers:
        freq_centers_map[t_c] = pick_freq_centers(R, V, M, t_c, k)

    B = _build_dictionary_torch(
        n_mels,
        T,
        time_centers,
        freq_centers_map,
        sigma_t,
        sigma_f_list,
        device,
    )
    if B.shape[1] == 0:
        return np.zeros_like(E)

    target = to_torch(E.reshape(-1), device=device, dtype=torch.float32)
    L = _estimate_lipschitz(B)
    step = 0.99 / L

    c = torch.zeros(B.shape[1], device=device, dtype=torch.float32)
    z = c.clone()
    t_k = 1.0

    for it in range(max_iter):
        residual = B @ z - target
        grad = B.T @ residual
        c_next = (z - step * grad - step * lam_g).clamp_min(0.0)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
        z = c_next + ((t_k - 1.0) / t_next) * (c_next - c)

        if tol > 0 and (it + 1) % 25 == 0:
            diff = torch.linalg.vector_norm(c_next - c).item()
            base = max(torch.linalg.vector_norm(c_next).item(), 1.0)
            if diff <= tol * base:
                c = c_next
                break

        c = c_next
        t_k = t_next

    G = (B @ c).reshape(n_mels, T).clamp_min(0.0)
    return to_numpy(G)


def _gaussian_optimize_cvxpy(E, R, V, M, k, spacing_frames, sigma_t, sigma_f_list, lam_g, solver="SCS"):
    if cp is None:
        raise RuntimeError("CVXPY is not installed, so the cvxpy backend is unavailable.")
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


def gaussian_optimize(
    E,
    R,
    V,
    M,
    k,
    spacing_frames,
    sigma_t,
    sigma_f_list,
    lam_g,
    solver="SCS",
    backend="auto",
    device="auto",
    max_iter=300,
    tol=1e-4,
):
    use_torch = backend == "torch" or (backend == "auto" and has_torch())
    if use_torch:
        return _gaussian_optimize_torch(
            E,
            R,
            V,
            M,
            k,
            spacing_frames,
            sigma_t,
            sigma_f_list,
            lam_g,
            device,
            max_iter,
            tol,
        )
    return _gaussian_optimize_cvxpy(
        E,
        R,
        V,
        M,
        k,
        spacing_frames,
        sigma_t,
        sigma_f_list,
        lam_g,
        solver=solver,
    )
