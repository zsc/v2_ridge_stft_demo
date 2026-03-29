import numpy as np

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - optional dependency
    cp = None

from backend import has_torch, resolve_device, to_numpy, to_torch


def _solve_ridge_cvxpy(M, V, lam_sparse, lam_tv, solver):
    if cp is None:
        raise RuntimeError("CVXPY is not installed, so the cvxpy backend is unavailable.")
    n_mels, T = M.shape
    V_float = V.astype(np.float32)
    R = cp.Variable((n_mels, T))
    tv_time = cp.sum(cp.abs(R[:, 1:] - R[:, :-1])) if T > 1 else 0
    objective = (
        0.5 * cp.sum_squares(R - M)
        + lam_sparse * cp.norm1(R)
        + lam_tv * tv_time
    )
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
    return R_val


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
    torch = __import__("torch")
    device = resolve_device(device)
    M_t = to_torch(M, device=device, dtype=torch.float32)
    V_t = to_torch(V.astype(np.float32), device=device, dtype=torch.float32)
    U_t = M_t * V_t

    x = (M_t - lam_sparse).clamp_min(0.0).minimum(U_t)
    if x.shape[1] <= 1 or lam_tv <= 0.0:
        x = x.clamp_min(0.0).minimum(U_t)
        return to_numpy(x)

    y = torch.zeros((x.shape[0], x.shape[1] - 1), device=device, dtype=x.dtype)
    x_bar = x.clone()
    tau = 0.49
    sigma = 0.49
    theta = 1.0

    for it in range(max_iter):
        y = (y + sigma * _forward_diff_time(x_bar)).clamp(-lam_tv, lam_tv)
        x_next = _prox_data_l1_box(
            x - tau * _adjoint_diff_time(y),
            M_t,
            U_t,
            lam_sparse,
            tau,
        )
        x_bar = x_next + theta * (x_next - x)

        if tol > 0 and (it + 1) % 25 == 0:
            diff = torch.linalg.vector_norm(x_next - x).item()
            base = max(torch.linalg.vector_norm(x_next).item(), 1.0)
            if diff <= tol * base:
                x = x_next
                break

        x = x_next

    x = x.clamp_min(0.0).minimum(U_t)
    return to_numpy(x)


def _topk_per_frame(R, V, k):
    n_mels, T = R.shape
    R_topk = np.zeros_like(R)
    for t in range(T):
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
                R_refined = _solve_ridge_torch(
                    M,
                    V2,
                    lam_sparse,
                    lam_tv,
                    device,
                    max_iter,
                    tol,
                )
            else:
                R_refined = _solve_ridge_cvxpy(M, V2, lam_sparse, lam_tv, solver)
            R_topk = _topk_per_frame(R_refined, V2, k)

    R_topk[V == 0] = 0.0
    return R_topk
