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


def is_voiced_frame(
    R,
    V,
    M,
    t_c,
    voiced_min_peak_count=2,
    voiced_min_ridge_ratio=0.03,
    voiced_max_flatness=0.55,
):
    frame = M[:, t_c].astype(np.float32)
    total = float(np.sum(frame))
    if total <= 1e-8:
        return False

    peak_count = int(np.count_nonzero(V[:, t_c]))
    ridge_ratio = float(np.sum(R[:, t_c]) / total)

    frame_safe = np.maximum(frame, 1e-8)
    flatness = float(np.exp(np.mean(np.log(frame_safe))) / np.mean(frame_safe))

    return peak_count >= voiced_min_peak_count and (
        ridge_ratio >= voiced_min_ridge_ratio or flatness <= voiced_max_flatness
    )


def pick_freq_centers(
    R,
    V,
    M,
    t_c,
    k,
    voiced_only=True,
    voiced_min_peak_count=2,
    voiced_min_ridge_ratio=0.03,
    voiced_max_flatness=0.55,
):
    if voiced_only and not is_voiced_frame(
        R,
        V,
        M,
        t_c,
        voiced_min_peak_count=voiced_min_peak_count,
        voiced_min_ridge_ratio=voiced_min_ridge_ratio,
        voiced_max_flatness=voiced_max_flatness,
    ):
        return []

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

def build_dictionary(
    n_mels,
    T,
    time_centers,
    freq_centers_map,
    sigma_t,
    sigma_f_list,
    balls_per_center=2,
    width_growth=0.4,
):
    mel_axis = np.arange(n_mels)
    time_axis = np.arange(T)

    basis_list = []
    for t_c in time_centers:
        for f_c in freq_centers_map[t_c]:
            for ball_idx in range(max(1, int(balls_per_center))):
                scale = 1.0 + float(ball_idx) * float(width_growth)
                sigma_t_eff = sigma_t * scale
                g_t = _gaussian_1d(time_axis, t_c, sigma_t_eff)
                for sigma_f in sigma_f_list:
                    sigma_f_eff = sigma_f * scale
                    g_f = _gaussian_1d(mel_axis, f_c, sigma_f_eff)
                    basis = np.outer(g_f, g_t)
                    basis_list.append(basis.reshape(-1))
    if not basis_list:
        return np.zeros((n_mels * T, 0), dtype=np.float32)
    B = np.stack(basis_list, axis=1).astype(np.float32)
    return B


def _build_dictionary_torch(
    n_mels,
    T,
    time_centers,
    freq_centers_map,
    sigma_t,
    sigma_f_list,
    device,
    balls_per_center=2,
    width_growth=0.4,
):
    torch = __import__("torch")
    mel_axis = torch.arange(n_mels, device=device, dtype=torch.float32)
    time_axis = torch.arange(T, device=device, dtype=torch.float32)

    basis_list = []
    for t_c in time_centers:
        for f_c in freq_centers_map[t_c]:
            for ball_idx in range(max(1, int(balls_per_center))):
                scale = 1.0 + float(ball_idx) * float(width_growth)
                sigma_t_eff = float(sigma_t) * scale
                g_t = torch.exp(-0.5 * ((time_axis - float(t_c)) ** 2) / (sigma_t_eff ** 2))
                for sigma_f in sigma_f_list:
                    sigma_f_eff = float(sigma_f) * scale
                    g_f = torch.exp(-0.5 * ((mel_axis - float(f_c)) ** 2) / (sigma_f_eff ** 2))
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
    voiced_only,
    voiced_min_peak_count,
    voiced_min_ridge_ratio,
    voiced_max_flatness,
    balls_per_center,
    width_growth,
):
    torch = __import__("torch")
    device = resolve_device(device)

    n_mels, T = E.shape
    time_centers = build_time_centers(T, spacing_frames)
    freq_centers_map = {}
    for t_c in time_centers:
        freq_centers_map[t_c] = pick_freq_centers(
            R,
            V,
            M,
            t_c,
            k,
            voiced_only=voiced_only,
            voiced_min_peak_count=voiced_min_peak_count,
            voiced_min_ridge_ratio=voiced_min_ridge_ratio,
            voiced_max_flatness=voiced_max_flatness,
        )

    B = _build_dictionary_torch(
        n_mels,
        T,
        time_centers,
        freq_centers_map,
        sigma_t,
        sigma_f_list,
        device,
        balls_per_center=balls_per_center,
        width_growth=width_growth,
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


def _gaussian_optimize_cvxpy(
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
    voiced_only=True,
    voiced_min_peak_count=2,
    voiced_min_ridge_ratio=0.03,
    voiced_max_flatness=0.55,
    balls_per_center=2,
    width_growth=0.4,
):
    if cp is None:
        raise RuntimeError("CVXPY is not installed, so the cvxpy backend is unavailable.")
    n_mels, T = E.shape
    time_centers = build_time_centers(T, spacing_frames)
    freq_centers_map = {}
    for t_c in time_centers:
        freq_centers_map[t_c] = pick_freq_centers(
            R,
            V,
            M,
            t_c,
            k,
            voiced_only=voiced_only,
            voiced_min_peak_count=voiced_min_peak_count,
            voiced_min_ridge_ratio=voiced_min_ridge_ratio,
            voiced_max_flatness=voiced_max_flatness,
        )

    B = build_dictionary(
        n_mels,
        T,
        time_centers,
        freq_centers_map,
        sigma_t,
        sigma_f_list,
        balls_per_center=balls_per_center,
        width_growth=width_growth,
    )
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
    voiced_only=True,
    voiced_min_peak_count=2,
    voiced_min_ridge_ratio=0.03,
    voiced_max_flatness=0.55,
    balls_per_center=2,
    width_growth=0.4,
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
            voiced_only,
            voiced_min_peak_count,
            voiced_min_ridge_ratio,
            voiced_max_flatness,
            balls_per_center,
            width_growth,
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
        voiced_only=voiced_only,
        voiced_min_peak_count=voiced_min_peak_count,
        voiced_min_ridge_ratio=voiced_min_ridge_ratio,
        voiced_max_flatness=voiced_max_flatness,
        balls_per_center=balls_per_center,
        width_growth=width_growth,
    )
