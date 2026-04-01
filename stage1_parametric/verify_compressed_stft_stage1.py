from __future__ import annotations

import argparse
import os
import sys
from html import escape

import matplotlib.pyplot as plt
import numpy as np
import librosa

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from audio import load_audio, save_audio
from peaks import peak_mask
from ridge_opt import ridge_optimize
from stage1_parametric.compressed_stft_stage1 import (
    build_thickened_stft_basis,
    compress_whisper_power,
    compress_whisper_power_torch,
    decompress_whisper_compressed_stft,
    fit_topk_thickened_stft_bands,
    render_stft_power_from_slots_torch,
    shift_compressed_stft_for_support,
    whisper_power_spectrogram,
)
from stage1_parametric.parametric_stage1 import (
    coeff_map_to_slot_strength,
    extract_center_idx_matrix,
    fit_strength_matrix_nnls,
    slot_strength_to_coeff_map,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Verify stage-1 parameterization in compressed Whisper-STFT space.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            os.path.join(ROOT, "input.wav"),
            os.path.join(ROOT, "examples", "denoise_prompt.wav"),
        ],
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(ROOT, "html", "stage1_compressed_stft_verify"),
    )
    parser.add_argument("--sr", type=int, default=None)
    parser.add_argument("--n_fft", type=int, default=400)
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--win_length", type=int, default=400)
    parser.add_argument("--center", type=lambda x: x.lower() == "true", default=True)
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
    parser.add_argument("--ridge_stft_sigma", type=float, default=2.0)
    parser.add_argument("--griffin_lim_iters", type=int, default=32)
    parser.add_argument("--griffin_lim_momentum", type=float, default=0.99)
    parser.add_argument("--max_abs_tol", type=float, default=1e-5)
    parser.add_argument("--relative_l2_tol", type=float, default=1e-6)
    parser.add_argument("--coeff_tol", type=float, default=1e-6)
    return parser.parse_args()


def slugify(path):
    name = os.path.splitext(os.path.basename(path))[0]
    keep = [ch if ch.isalnum() else "_" for ch in name]
    return "".join(keep).strip("_") or "sample"


def save_matrix_png(path, matrix, title, ylabel):
    plt.figure(figsize=(6, 4))
    plt.imshow(np.asarray(matrix, dtype=np.float32), origin="lower", aspect="auto", cmap="magma")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


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
  <div class="column">
    <h2>{escape(item['title'])}</h2>
    <img src="{escape(item['src'])}" alt="{escape(item['title'])}" />
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
  .grid {{ grid-template-columns: repeat(3, minmax(280px, 1fr)); }}
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
    toc = []
    sections = []
    for record in records:
        toc.append(f'<li><a href="#{escape(record["slug"])}">{escape(record["name"])}</a></li>')
        sections.append(
            f"""
<section class="item" id="{escape(record['slug'])}">
  <div class="item-header">
    <h2>{escape(record['name'])}</h2>
    <p><a href="{escape(record['report_html'])}">Per-file report</a></p>
    <p class="metrics">
      raw_min={record['raw_min']:.6f}
      raw_max={record['raw_max']:.6f}
      floor_value={record['floor_value']:.6f}
      recon_max_abs_diff={record['recon_max_abs_diff']:.8e}
      coeff_max_abs_diff={record['coeff_max_abs_diff']:.8e}
      grad_norm={record['grad_norm']}
    </p>
  </div>
  <div class="grid">
    <div class="column">
      <h3>Compressed STFT</h3>
      <img src="{escape(record['compressed_png'])}" alt="Compressed STFT" />
    </div>
    <div class="column">
      <h3>Stage-1 Ridge (Compressed View)</h3>
      <img src="{escape(record['ridge_png'])}" alt="Stage-1 Ridge" />
      <audio controls preload="none" src="{escape(record['ridge_wav'])}"></audio>
    </div>
    <div class="column">
      <h3>Residual (Compressed View)</h3>
      <img src="{escape(record['residual_png'])}" alt="Residual" />
    </div>
  </div>
</section>"""
        )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Compressed STFT Stage-1 Verification</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f1f1f; }}
a {{ color: #0b57d0; }}
ul {{ columns: 2; padding-left: 20px; }}
.item {{ margin-bottom: 40px; padding-bottom: 24px; border-bottom: 1px solid #ddd; }}
.metrics {{ font-family: monospace; font-size: 12px; white-space: pre-wrap; }}
.grid {{ display: grid; grid-template-columns: repeat(3, minmax(280px, 1fr)); gap: 18px; }}
img {{ width: 100%; height: auto; border: 1px solid #ccc; background: #111; }}
audio {{ width: 100%; margin-top: 10px; }}
</style>
</head>
<body>
<h1>Compressed STFT Stage-1 Verification</h1>
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
    power, actual_sr = whisper_power_spectrogram(
        y,
        sr,
        target_sr=16000,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        center=args.center,
        drop_last=True,
    )
    compressed, comp_stats = compress_whisper_power(power, return_stats=True)
    power_clamped = decompress_whisper_compressed_stft(compressed)
    shifted, floor_value = shift_compressed_stft_for_support(compressed)

    V = peak_mask(
        shifted,
        distance=args.peak_distance,
        prominence=args.peak_prominence,
        height=args.peak_height,
        dilate=bool(args.peak_dilate),
    )
    ridge_support = ridge_optimize(
        shifted,
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

    center_idx, valid_mask = extract_center_idx_matrix(ridge_support, k=args.k, sort_by="mel_asc")
    basis = build_thickened_stft_basis(power_clamped.shape[0], sigma_bins=args.ridge_stft_sigma, normalize="peak")
    slot_strength, ridge_power = fit_strength_matrix_nnls(power_clamped, center_idx, basis, clip_upper=True)
    coeff_map = slot_strength_to_coeff_map(center_idx, slot_strength, power_clamped.shape[0])

    legacy_ridge_power, legacy_coeff_map, legacy_center_idx, legacy_basis = fit_topk_thickened_stft_bands(
        power_clamped,
        ridge_support > 0,
        sigma_bins=args.ridge_stft_sigma,
        clip_upper=True,
    )
    legacy_slot_strength = coeff_map_to_slot_strength(center_idx, legacy_coeff_map)

    if legacy_center_idx.shape != center_idx.shape or not np.array_equal(legacy_center_idx, center_idx):
        raise AssertionError(f"{input_path}: center_idx mismatch between wrapper and explicit path")
    if legacy_basis.shape != basis.shape or np.max(np.abs(legacy_basis - basis)) > 1e-7:
        raise AssertionError(f"{input_path}: basis mismatch between wrapper and explicit path")

    reference_max_log10 = comp_stats["max_log10"]
    ridge_compressed = compress_whisper_power(ridge_power, reference_max_log10=reference_max_log10)
    residual_power = np.clip(power_clamped - ridge_power, 0.0, None)
    residual_compressed = compress_whisper_power(residual_power, reference_max_log10=reference_max_log10)
    recon_compressed = compress_whisper_power(
        ridge_power + residual_power,
        reference_max_log10=reference_max_log10,
    )
    decompressed_again = decompress_whisper_compressed_stft(compressed)

    recon_diff = np.abs(compressed - recon_compressed)
    power_inv_diff = np.abs(power_clamped - decompressed_again)
    coeff_max_abs_diff = float(np.max(np.abs(coeff_map - legacy_coeff_map)))
    slot_max_abs_diff = float(np.max(np.abs(slot_strength - legacy_slot_strength)))
    ridge_power_diff = np.abs(ridge_power - legacy_ridge_power)

    if float(np.max(recon_diff)) > args.max_abs_tol:
        raise AssertionError(f"{input_path}: compressed recon diff exceeds max_abs_tol")
    if float(np.max(ridge_power_diff)) > args.max_abs_tol:
        raise AssertionError(f"{input_path}: wrapper ridge power diff exceeds max_abs_tol")
    if max(coeff_max_abs_diff, slot_max_abs_diff) > args.coeff_tol:
        raise AssertionError(f"{input_path}: coeff diff exceeds coeff_tol")
    rel_l2 = float(
        np.linalg.norm((ridge_power - legacy_ridge_power).ravel())
        / max(np.linalg.norm(ridge_power.ravel()), 1e-8)
    )
    if rel_l2 > args.relative_l2_tol:
        raise AssertionError(f"{input_path}: wrapper relative_l2 diff exceeds relative_l2_tol")

    grad_norm = None
    if torch is not None:
        strength_t = torch.tensor(slot_strength, dtype=torch.float32, requires_grad=True)
        ridge_power_t = render_stft_power_from_slots_torch(center_idx, strength_t, basis, clip_upper=None)
        ridge_comp_t = compress_whisper_power_torch(ridge_power_t, reference_max_log10=reference_max_log10)
        loss = ridge_comp_t.square().mean() + 1e-4 * strength_t.square().mean()
        loss.backward()
        grad_norm = float(strength_t.grad.norm().item())

    ridge_wav_path = os.path.join(sample_dir, "ridge_griffinlim.wav")
    ridge_mag = np.sqrt(np.clip(ridge_power, 0.0, None)).astype(np.float32)
    if ridge_mag.size == 0 or not np.any(ridge_mag > 0.0):
        ridge_audio = np.zeros(args.win_length, dtype=np.float32)
    else:
        ridge_audio = librosa.griffinlim(
            ridge_mag,
            n_iter=args.griffin_lim_iters,
            hop_length=args.hop_length,
            win_length=args.win_length,
            n_fft=args.n_fft,
            window="hann",
            center=args.center,
            momentum=args.griffin_lim_momentum,
            random_state=0,
        ).astype(np.float32)
    peak = float(np.max(np.abs(ridge_audio))) if ridge_audio.size > 0 else 0.0
    if peak > 0.999:
        ridge_audio = ridge_audio / peak * 0.999
    save_audio(ridge_wav_path, ridge_audio.astype(np.float32), actual_sr)

    np.save(os.path.join(sample_dir, "compressed_stft.npy"), compressed)
    np.save(os.path.join(sample_dir, "shifted_stft.npy"), shifted)
    np.save(os.path.join(sample_dir, "power.npy"), power)
    np.save(os.path.join(sample_dir, "power_clamped.npy"), power_clamped)
    np.save(os.path.join(sample_dir, "ridge_support.npy"), ridge_support)
    np.save(os.path.join(sample_dir, "center_idx.npy"), center_idx)
    np.save(os.path.join(sample_dir, "valid_mask.npy"), valid_mask.astype(np.uint8))
    np.save(os.path.join(sample_dir, "slot_strength.npy"), slot_strength)
    np.save(os.path.join(sample_dir, "coeff_map.npy"), coeff_map)
    np.save(os.path.join(sample_dir, "basis.npy"), basis)
    np.save(os.path.join(sample_dir, "ridge_power.npy"), ridge_power)
    np.save(os.path.join(sample_dir, "ridge_compressed.npy"), ridge_compressed)
    np.save(os.path.join(sample_dir, "residual_power.npy"), residual_power)
    np.save(os.path.join(sample_dir, "residual_compressed.npy"), residual_compressed)
    np.save(os.path.join(sample_dir, "recon_compressed.npy"), recon_compressed)

    compressed_png = os.path.join(sample_dir, "compressed_stft.png")
    shifted_png = os.path.join(sample_dir, "shifted_stft.png")
    support_png = os.path.join(sample_dir, "ridge_support.png")
    ridge_png = os.path.join(sample_dir, "ridge_compressed.png")
    residual_png = os.path.join(sample_dir, "residual_compressed.png")
    center_png = os.path.join(sample_dir, "center_idx.png")
    strength_png = os.path.join(sample_dir, "slot_strength.png")

    save_matrix_png(compressed_png, compressed, "Compressed Whisper STFT", "STFT bins")
    save_matrix_png(shifted_png, shifted, "Shifted Support Domain", "STFT bins")
    save_matrix_png(support_png, ridge_support, "Stage-1 Support Strength", "STFT bins")
    save_matrix_png(ridge_png, ridge_compressed, "Stage-1 Ridge (Compressed View)", "STFT bins")
    save_matrix_png(residual_png, residual_compressed, "Residual (Compressed View)", "STFT bins")
    save_slot_png(center_png, np.where(center_idx >= 0, center_idx, np.nan), "Center Index Matrix", "slot k")
    save_slot_png(strength_png, slot_strength, "Power-Domain Strength Matrix", "slot k")

    report_path = os.path.join(sample_dir, "report.html")
    summary_rows = [
        ("input", input_path),
        ("actual_sr", str(actual_sr)),
        ("shape", f"compressed={tuple(compressed.shape)} center_idx={tuple(center_idx.shape)}"),
        ("compressed_min", f"{float(compressed.min()):.8e}"),
        ("compressed_max", f"{float(compressed.max()):.8e}"),
        ("negative_ratio", f"{float((compressed < 0).mean()):.6f}"),
        ("support_floor_value", f"{floor_value:.8e}"),
        ("reference_max_log10", f"{reference_max_log10:.8e}"),
        ("power_clamped_inverse_max_abs_diff", f"{float(np.max(power_inv_diff)):.8e}"),
        ("ridge_wrapper_max_abs_diff", f"{float(np.max(ridge_power_diff)):.8e}"),
        ("ridge_wrapper_relative_l2_diff", f"{rel_l2:.8e}"),
        ("coeff_map_max_abs_diff", f"{coeff_max_abs_diff:.8e}"),
        ("slot_strength_max_abs_diff", f"{slot_max_abs_diff:.8e}"),
        ("compressed_recon_max_abs_diff", f"{float(np.max(recon_diff)):.8e}"),
        ("torch_grad_norm", "n/a" if grad_norm is None else f"{grad_norm:.8e}"),
    ]
    image_rows = [
        {"title": "Compressed STFT", "src": "compressed_stft.png"},
        {"title": "Shifted Support Domain", "src": "shifted_stft.png"},
        {"title": "Stage-1 Support Strength", "src": "ridge_support.png"},
        {"title": "Stage-1 Ridge (Compressed View)", "src": "ridge_compressed.png"},
        {"title": "Residual (Compressed View)", "src": "residual_compressed.png"},
        {"title": "Center Index Matrix", "src": "center_idx.png"},
        {"title": "Power-Domain Strength Matrix", "src": "slot_strength.png"},
    ]
    write_report(report_path, f"Compressed STFT Stage-1 Check: {os.path.basename(input_path)}", summary_rows, image_rows)

    return {
        "name": os.path.basename(input_path),
        "slug": slug,
        "report_html": f"{slug}/report.html",
        "compressed_png": f"{slug}/compressed_stft.png",
        "ridge_png": f"{slug}/ridge_compressed.png",
        "residual_png": f"{slug}/residual_compressed.png",
        "ridge_wav": f"{slug}/ridge_griffinlim.wav",
        "raw_min": float(compressed.min()),
        "raw_max": float(compressed.max()),
        "floor_value": floor_value,
        "recon_max_abs_diff": float(np.max(recon_diff)),
        "coeff_max_abs_diff": coeff_max_abs_diff,
        "grad_norm": "n/a" if grad_norm is None else f"{grad_norm:.8e}",
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
