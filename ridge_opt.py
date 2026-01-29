import numpy as np
import cvxpy as cp


def _solve_ridge(M, V, lam_sparse, lam_tv, solver):
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
    k=5,
    lam_sparse=0.05,
    lam_tv=0.1,
    solver="SCS",
    ridge_refine=True,
):
    R_raw = _solve_ridge(M, V, lam_sparse, lam_tv, solver)
    R_topk = _topk_per_frame(R_raw, V, k)

    if ridge_refine:
        V2 = R_topk > 0
        if V2.any():
            R_refined = _solve_ridge(M, V2, lam_sparse, lam_tv, solver)
            R_topk = _topk_per_frame(R_refined, V2, k)

    R_topk[V == 0] = 0.0
    return R_topk
