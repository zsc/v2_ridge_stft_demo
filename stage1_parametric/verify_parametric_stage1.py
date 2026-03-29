from __future__ import annotations

import argparse
import os
import sys
from html import escape

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from audio import load_audio
from peaks import peak_mask
from ridge_opt import ridge_optimize
from report import save_mel_png
from stft_utils import mag_phase, stft
from mel_utils import mel_filter, mel_forward, fit_topk_thickened_mel_bands

from stage1_parametric.parametric_stage1 import (
    build_thickened_mel_basis,
    coeff_map_to_slot_strength,
    extract_center_idx_matrix,
    fit_strength_matrix_nnls,
    render_mel_from_slots_torch,
    slot_strength_to_coeff_map,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Verify self-contained stage-1 parameterization vs legacy code.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            os.path.join(ROOT, "input.wav"),
            os.path.join(ROOT, "examples", "denoise_prompt.wav"),
        ],
        help="Input wav paths to verify.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(ROOT, "html", "stage1_parametric_verify"),
        help="Directory for HTML reports.",
    )
    parser.add_argument("--sr", type=int, default=None)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--win_length", type=int, default=2048)
    parser.add_argument("--center", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--stft_backend", choices=["auto", "librosa", "torch"], default="auto")
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--peak_distance", type=int, default=2)
    parser.add_argument("--peak_prominence", type=float, default=None)
    parser.add_argument("--peak_height", type=float, default=None)
    parser.add_argument("--peak_dilate", type=int, default=0)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--lam_sparse", type=float, default=0.05)
    parser.add_argument("--lam_tv", type=float, default=0.1)
    parser.add_argument("--backend", choices=["auto", "torch", "cvxpy"], default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--solver", choices=["SCS", "ECOS", "OSQP"], default="SCS")
    parser.add_argument("--ridge_refine", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--ridge_max_iter", type=int, default=300)
    parser.add_argument("--ridge_tol", type=float, default=1e-4)
    parser.add_argument("--ridge_linear_thicken_sigma", type=float, default=2.0)
    parser.add_argument("--max_abs_tol", type=float, default=1e-5)
    parser.add_argument("--relative_l2_tol", type=float, default=1e-6)
    parser.add_argument("--coeff_tol", type=float, default=1e-6)
    return parser.parse_args()


def slugify(path):
    name = os.path.splitext(os.path.basename(path))[0]
    keep = [ch if ch.isalnum() else "_" for ch in name]
    return "".join(keep).strip("_") or "sample"


def save_slot_png(path, matrix, title, ylabel):
    plt.figure(figsize=(6, 3))
    plt.imshow(np.asarray(matrix, dtype=np.float32), origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_report(path, title, summary_rows, image_rows):
    summary_html = "".join(
        f"<tr><th>{escape(key)}</th><td>{escape(value)}</td></tr>" for key, value in summary_rows
    )
    columns = "".join(
        f"""
  <div class=\"column\">
    <h2>{escape(item['title'])}</h2>
    <img src=\"{escape(item['src'])}\" alt=\"{escape(item['title'])}\" />
  </div>"""
        for item in image_rows
    )
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f1f1f; }}
table {{ border-collapse: collapse; margin-bottom: 20px; }}
th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 18px; }}
img {{ width: 100%; height: auto; border: 1px solid #ccc; background: #111; }}
@media (min-width: 1400px) {{
  .grid {{ grid-template-columns: repeat(4, minmax(280px, 1fr)); }}
}}
</style>
</head>
<body>
<h1>{escape(title)}</h1>
<table>
{summary_html}
</table>
<div class="grid">
{columns}
</div>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def write_index(path, records):
    sections = []
    toc = []
    for record in records:
        toc.append(f'<li><a href="#{escape(record["slug"])}">{escape(record["name"])}</a></li>')
        sections.append(
            f"""
<section class="item" id="{escape(record['slug'])}">
  <div class="item-header">
    <h2>{escape(record['name'])}</h2>
    <p><a href="{escape(record['report_html'])}">Per-file report</a></p>
    <p class="metrics">
      max_abs_diff={record['max_abs_diff']:.8e}
      mean_abs_diff={record['mean_abs_diff']:.8e}
      relative_l2_diff={record['relative_l2_diff']:.8e}
      torch_grad_norm={record['torch_grad_norm']}
    </p>
  </div>
  <div class="grid">
    <div class="column">
      <h3>GT Mel</h3>
      <img src="{escape(record['gt_png'])}" alt="GT Mel" />
    </div>
    <div class="column">
      <h3>Legacy Stage-1</h3>
      <img src="{escape(record['legacy_png'])}" alt="Legacy Stage-1" />
    </div>
    <div class="column">
      <h3>Parametric Stage-1</h3>
      <img src="{escape(record['param_png'])}" alt="Parametric Stage-1" />
    </div>
    <div class="column">
      <h3>Absolute Difference</h3>
      <img src="{escape(record['diff_png'])}" alt="Absolute Difference" />
    </div>
  </div>
</section>"""
        )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Stage-1 Parametric Verification</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f1f1f; }}
a {{ color: #0b57d0; }}
ul {{ columns: 2; padding-left: 20px; }}
.item {{ margin-bottom: 40px; padding-bottom: 24px; border-bottom: 1px solid #ddd; }}
.metrics {{ font-family: monospace; font-size: 12px; white-space: pre-wrap; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 18px; }}
img {{ width: 100%; height: auto; border: 1px solid #ccc; background: #111; }}
@media (min-width: 1400px) {{
  .grid {{ grid-template-columns: repeat(4, minmax(280px, 1fr)); }}
}}
</style>
</head>
<body>
<h1>Stage-1 Parametric Verification</h1>
<p>Total files: {len(records)}</p>
<ul>
{''.join(toc)}
</ul>
{''.join(sections)}
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def run_single(input_path, args):
    slug = slugify(input_path)
    sample_dir = os.path.join(args.output_dir, slug)
    os.makedirs(sample_dir, exist_ok=True)

    y, sr = load_audio(input_path, sr=args.sr, mono=True)
    S = stft(
        y,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        center=args.center,
        backend=args.stft_backend,
        device=args.device,
    )
    A, _ = mag_phase(S)
    F_mel = mel_filter(sr, args.n_fft, args.n_mels, args.fmin, args.fmax)
    M = mel_forward(F_mel, A, device=args.device)
    V = peak_mask(
        M,
        distance=args.peak_distance,
        prominence=args.peak_prominence,
        height=args.peak_height,
        dilate=bool(args.peak_dilate),
    )

    R_raw = ridge_optimize(
        M,
        V,
        k=args.k,
        lam_sparse=args.lam_sparse,
        lam_tv=args.lam_tv,
        solver=args.solver,
        ridge_refine=args.ridge_refine,
        backend=args.backend,
        device=args.device,
        max_iter=args.ridge_max_iter,
        tol=args.ridge_tol,
    )
    legacy_r, legacy_coeff_map = fit_topk_thickened_mel_bands(
        M,
        R_raw > 0,
        F_mel,
        sigma_bins=args.ridge_linear_thicken_sigma,
        device=args.device,
        clip_upper=True,
    )

    basis = build_thickened_mel_basis(
        F_mel,
        sigma_bins=args.ridge_linear_thicken_sigma,
        normalize="peak",
    )
    center_idx, valid_mask = extract_center_idx_matrix(R_raw, k=args.k, sort_by="mel_asc")
    strength, parametric_r = fit_strength_matrix_nnls(M, center_idx, basis, clip_upper=True)
    parametric_coeff_map = slot_strength_to_coeff_map(center_idx, strength, args.n_mels)
    legacy_slot_strength = coeff_map_to_slot_strength(center_idx, legacy_coeff_map)
    diff = np.abs(legacy_r - parametric_r)

    torch_grad_norm = None
    if torch is not None:
        strength_t = torch.tensor(strength, dtype=torch.float32, requires_grad=True)
        render_t = render_mel_from_slots_torch(center_idx, strength_t, basis, clip_upper=None)
        loss = (render_t.square().mean() + 1e-4 * strength_t.square().mean())
        loss.backward()
        torch_grad_norm = float(strength_t.grad.norm().item())

    np.save(os.path.join(sample_dir, "legacy_ridge.npy"), legacy_r)
    np.save(os.path.join(sample_dir, "parametric_ridge.npy"), parametric_r)
    np.save(os.path.join(sample_dir, "diff_abs.npy"), diff)
    np.save(os.path.join(sample_dir, "center_idx.npy"), center_idx)
    np.save(os.path.join(sample_dir, "valid_mask.npy"), valid_mask.astype(np.uint8))
    np.save(os.path.join(sample_dir, "slot_strength.npy"), strength)
    np.save(os.path.join(sample_dir, "legacy_slot_strength.npy"), legacy_slot_strength)
    np.save(os.path.join(sample_dir, "basis.npy"), basis)

    gt_png = os.path.join(sample_dir, "gt_mel.png")
    legacy_png = os.path.join(sample_dir, "legacy_stage1.png")
    param_png = os.path.join(sample_dir, "parametric_stage1.png")
    diff_png = os.path.join(sample_dir, "abs_diff.png")
    center_png = os.path.join(sample_dir, "center_idx.png")
    strength_png = os.path.join(sample_dir, "slot_strength.png")

    save_mel_png(gt_png, M, sr, args.hop_length, "GT Mel (direct)")
    save_mel_png(legacy_png, legacy_r, sr, args.hop_length, "Legacy Stage-1")
    save_mel_png(param_png, parametric_r, sr, args.hop_length, "Parametric Stage-1")
    save_mel_png(diff_png, diff, sr, args.hop_length, "Abs Diff")
    save_slot_png(center_png, np.where(center_idx >= 0, center_idx, np.nan), "Center Index Matrix", "slot k")
    save_slot_png(strength_png, strength, "Strength Matrix", "slot k")

    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    rel_l2 = float(np.linalg.norm((legacy_r - parametric_r).ravel()) / max(np.linalg.norm(legacy_r.ravel()), 1e-8))
    coeff_max_abs = float(np.max(np.abs(legacy_coeff_map - parametric_coeff_map)))
    slot_max_abs = float(np.max(np.abs(legacy_slot_strength - strength)))
    active_ratio = float(valid_mask.mean())

    if max_abs > args.max_abs_tol:
        raise AssertionError(
            f"{input_path}: max_abs_diff={max_abs:.8e} exceeds tol={args.max_abs_tol:.8e}"
        )
    if rel_l2 > args.relative_l2_tol:
        raise AssertionError(
            f"{input_path}: relative_l2_diff={rel_l2:.8e} exceeds tol={args.relative_l2_tol:.8e}"
        )
    if max(coeff_max_abs, slot_max_abs) > args.coeff_tol:
        raise AssertionError(
            f"{input_path}: coeff diff exceeds tol={args.coeff_tol:.8e}"
        )

    report_path = os.path.join(sample_dir, "report.html")
    summary_rows = [
        ("input", input_path),
        ("shape", f"M={tuple(M.shape)} center_idx={tuple(center_idx.shape)}"),
        ("max_abs_diff", f"{max_abs:.8e}"),
        ("mean_abs_diff", f"{mean_abs:.8e}"),
        ("relative_l2_diff", f"{rel_l2:.8e}"),
        ("coeff_map_max_abs_diff", f"{coeff_max_abs:.8e}"),
        ("slot_strength_max_abs_diff", f"{slot_max_abs:.8e}"),
        ("active_slot_ratio", f"{active_ratio:.4f}"),
        (
            "tolerances",
            (
                f"max_abs<={args.max_abs_tol:.1e}, "
                f"relative_l2<={args.relative_l2_tol:.1e}, "
                f"coeff<={args.coeff_tol:.1e}"
            ),
        ),
        ("torch_grad_norm", "n/a" if torch_grad_norm is None else f"{torch_grad_norm:.8e}"),
    ]
    image_rows = [
        {"title": "GT Mel", "src": "gt_mel.png"},
        {"title": "Legacy Stage-1", "src": "legacy_stage1.png"},
        {"title": "Parametric Stage-1", "src": "parametric_stage1.png"},
        {"title": "Absolute Difference", "src": "abs_diff.png"},
        {"title": "Center Index Matrix", "src": "center_idx.png"},
        {"title": "Strength Matrix", "src": "slot_strength.png"},
    ]
    write_report(report_path, f"Stage-1 Parametric Check: {os.path.basename(input_path)}", summary_rows, image_rows)

    return {
        "name": os.path.basename(input_path),
        "slug": slug,
        "report_html": f"{slug}/report.html",
        "gt_png": f"{slug}/gt_mel.png",
        "legacy_png": f"{slug}/legacy_stage1.png",
        "param_png": f"{slug}/parametric_stage1.png",
        "diff_png": f"{slug}/abs_diff.png",
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "relative_l2_diff": rel_l2,
        "torch_grad_norm": "n/a" if torch_grad_norm is None else f"{torch_grad_norm:.8e}",
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    records = [run_single(os.path.abspath(path), args) for path in args.inputs]
    index_path = os.path.join(args.output_dir, "index.html")
    write_index(index_path, records)
    print(index_path)


if __name__ == "__main__":
    main()
