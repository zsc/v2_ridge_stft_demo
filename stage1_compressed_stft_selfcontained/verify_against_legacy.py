from __future__ import annotations

import argparse
import os
import sys
from html import escape

import librosa
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from audio import load_audio as legacy_load_audio
from audio import save_audio as legacy_save_audio
from peaks import peak_mask as legacy_peak_mask
from ridge_opt import ridge_optimize as legacy_ridge_optimize
from stage1_parametric.compressed_stft_stage1 import (
    build_thickened_stft_basis as legacy_build_thickened_stft_basis,
    compress_whisper_power as legacy_compress_whisper_power,
    compress_whisper_power_torch as legacy_compress_whisper_power_torch,
    decompress_whisper_compressed_stft as legacy_decompress_whisper_compressed_stft,
    whisper_power_spectrogram as legacy_whisper_power_spectrogram,
    shift_compressed_stft_for_support as legacy_shift_compressed_stft_for_support,
)
from stage1_parametric.parametric_stage1 import (
    extract_center_idx_matrix as legacy_extract_center_idx_matrix,
    fit_strength_matrix_nnls as legacy_fit_strength_matrix_nnls,
    render_from_slots_torch as legacy_render_from_slots_torch,
    slot_strength_to_coeff_map as legacy_slot_strength_to_coeff_map,
)
from stage1_compressed_stft_selfcontained.stage1_stft import (
    run_stage1_pipeline,
    save_audio,
    save_stage1_artifacts,
)

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def parse_args():
    parser = argparse.ArgumentParser(description="Verify self-contained compressed STFT stage-1 vs legacy implementation.")
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
        default=os.path.join(ROOT, "html", "stage1_compressed_stft_selfcontained_verify"),
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
    parser.add_argument("--array_max_abs_tol", type=float, default=1e-5)
    parser.add_argument("--array_rel_l2_tol", type=float, default=1e-6)
    parser.add_argument("--audio_max_abs_tol", type=float, default=1e-5)
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


def write_report(path, title, summary_rows, image_rows, audio_rows):
    summary_html = "".join(
        f"<tr><th>{escape(key)}</th><td>{escape(value)}</td></tr>" for key, value in summary_rows
    )
    image_html = "".join(
        f"""
  <div class="column">
    <h2>{escape(item['title'])}</h2>
    <img src="{escape(item['src'])}" alt="{escape(item['title'])}" />
  </div>"""
        for item in image_rows
    )
    audio_html = "".join(
        f"""
  <div class="audio-card">
    <h2>{escape(item['title'])}</h2>
    <audio controls preload="none" src="{escape(item['src'])}"></audio>
  </div>"""
        for item in audio_rows
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
.audio-grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 18px; margin-top: 18px; }}
img {{ width: 100%; height: auto; border: 1px solid #ccc; background: #111; }}
audio {{ width: 100%; }}
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
{image_html}
</div>
<div class="audio-grid">
{audio_html}
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
      ridge_power_max_abs_diff={record['ridge_power_max_abs_diff']:.8e}
      ridge_compressed_max_abs_diff={record['ridge_compressed_max_abs_diff']:.8e}
      coeff_map_max_abs_diff={record['coeff_map_max_abs_diff']:.8e}
      ridge_audio_max_abs_diff={record['ridge_audio_max_abs_diff']:.8e}
      center_idx_mismatch_count={record['center_idx_mismatch_count']}
    </p>
  </div>
  <div class="grid">
    <div class="column">
      <h3>Legacy Ridge</h3>
      <img src="{escape(record['legacy_ridge_png'])}" alt="Legacy Ridge" />
      <audio controls preload="none" src="{escape(record['legacy_ridge_wav'])}"></audio>
    </div>
    <div class="column">
      <h3>Self-contained Ridge</h3>
      <img src="{escape(record['new_ridge_png'])}" alt="Self-contained Ridge" />
      <audio controls preload="none" src="{escape(record['new_ridge_wav'])}"></audio>
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
<title>Compressed STFT Self-contained Verification</title>
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
<h1>Compressed STFT Self-contained Verification</h1>
<p>Total files: {len(records)}</p>
<ul>
{''.join(toc)}
</ul>
{''.join(sections)}
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def run_legacy_pipeline(input_path, args):
    y, sr = legacy_load_audio(input_path, sr=args.sr, mono=True)
    power, actual_sr = legacy_whisper_power_spectrogram(
        y,
        sr,
        target_sr=16000,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        center=args.center,
        drop_last=True,
    )
    compressed, comp_stats = legacy_compress_whisper_power(power, return_stats=True)
    power_clamped = legacy_decompress_whisper_compressed_stft(compressed)
    shifted, floor_value = legacy_shift_compressed_stft_for_support(compressed)

    V = legacy_peak_mask(
        shifted,
        distance=args.peak_distance,
        prominence=args.peak_prominence,
        height=args.peak_height,
        dilate=bool(args.peak_dilate),
    )
    ridge_support = legacy_ridge_optimize(
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
    center_idx, valid_mask = legacy_extract_center_idx_matrix(ridge_support, k=args.k, sort_by="mel_asc")
    basis = legacy_build_thickened_stft_basis(power_clamped.shape[0], sigma_bins=args.ridge_stft_sigma, normalize="peak")
    slot_strength, ridge_power = legacy_fit_strength_matrix_nnls(power_clamped, center_idx, basis, clip_upper=True)
    coeff_map = legacy_slot_strength_to_coeff_map(center_idx, slot_strength, power_clamped.shape[0])

    reference_max_log10 = comp_stats["max_log10"]
    ridge_compressed = legacy_compress_whisper_power(ridge_power, reference_max_log10=reference_max_log10)
    residual_power = np.clip(power_clamped - ridge_power, 0.0, None)
    residual_compressed = legacy_compress_whisper_power(residual_power, reference_max_log10=reference_max_log10)
    recon_compressed = legacy_compress_whisper_power(
        ridge_power + residual_power,
        reference_max_log10=reference_max_log10,
    )

    grad_norm = None
    if torch is not None:
        strength_t = torch.tensor(slot_strength, dtype=torch.float32, requires_grad=True)
        ridge_power_t = legacy_render_from_slots_torch(center_idx, strength_t, basis, clip_upper=None)
        ridge_comp_t = legacy_compress_whisper_power_torch(ridge_power_t, reference_max_log10=reference_max_log10)
        loss = ridge_comp_t.square().mean() + 1e-4 * strength_t.square().mean()
        loss.backward()
        grad_norm = float(strength_t.grad.norm().item())

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

    return {
        "input_path": os.path.abspath(input_path),
        "input_audio": y,
        "input_sr": sr,
        "actual_sr": actual_sr,
        "power": power,
        "compressed": compressed,
        "comp_stats": comp_stats,
        "power_clamped": power_clamped,
        "shifted": shifted,
        "support_floor_value": floor_value,
        "peak_mask": V,
        "ridge_support": ridge_support,
        "center_idx": center_idx,
        "valid_mask": valid_mask,
        "basis": basis,
        "slot_strength": slot_strength,
        "coeff_map": coeff_map,
        "ridge_power": ridge_power,
        "ridge_compressed": ridge_compressed,
        "residual_power": residual_power,
        "residual_compressed": residual_compressed,
        "recon_compressed": recon_compressed,
        "ridge_audio": ridge_audio.astype(np.float32),
        "grad_norm": grad_norm,
    }


def max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


def relative_l2_diff(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(a.ravel()), 1e-8))


def run_single(input_path, args):
    slug = slugify(input_path)
    sample_dir = os.path.join(args.output_dir, slug)
    legacy_dir = os.path.join(sample_dir, "legacy")
    new_dir = os.path.join(sample_dir, "selfcontained")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(legacy_dir, exist_ok=True)
    os.makedirs(new_dir, exist_ok=True)

    legacy = run_legacy_pipeline(input_path, args)
    new = run_stage1_pipeline(
        input_path,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        center=args.center,
        peak_distance=args.peak_distance,
        peak_prominence=args.peak_prominence,
        peak_height=args.peak_height,
        peak_dilate=args.peak_dilate,
        k=args.k,
        lam_sparse=args.lam_sparse,
        lam_tv=args.lam_tv,
        backend=args.backend,
        device=args.device,
        solver=args.solver,
        ridge_refine=args.ridge_refine,
        ridge_max_iter=args.ridge_max_iter,
        ridge_tol=args.ridge_tol,
        ridge_stft_sigma=args.ridge_stft_sigma,
        griffin_lim_iters=args.griffin_lim_iters,
        griffin_lim_momentum=args.griffin_lim_momentum,
        compute_grad_norm=True,
    )

    save_stage1_artifacts(new_dir, new)
    legacy_save_audio(os.path.join(legacy_dir, "ridge_griffinlim.wav"), legacy["ridge_audio"], legacy["actual_sr"])

    diffs = {
        "compressed_max_abs_diff": max_abs_diff(legacy["compressed"], new["compressed"]),
        "shifted_max_abs_diff": max_abs_diff(legacy["shifted"], new["shifted"]),
        "ridge_support_max_abs_diff": max_abs_diff(legacy["ridge_support"], new["ridge_support"]),
        "basis_max_abs_diff": max_abs_diff(legacy["basis"], new["basis"]),
        "slot_strength_max_abs_diff": max_abs_diff(legacy["slot_strength"], new["slot_strength"]),
        "coeff_map_max_abs_diff": max_abs_diff(legacy["coeff_map"], new["coeff_map"]),
        "ridge_power_max_abs_diff": max_abs_diff(legacy["ridge_power"], new["ridge_power"]),
        "ridge_compressed_max_abs_diff": max_abs_diff(legacy["ridge_compressed"], new["ridge_compressed"]),
        "residual_compressed_max_abs_diff": max_abs_diff(legacy["residual_compressed"], new["residual_compressed"]),
        "ridge_audio_max_abs_diff": max_abs_diff(legacy["ridge_audio"], new["ridge_audio"]),
        "ridge_power_relative_l2_diff": relative_l2_diff(legacy["ridge_power"], new["ridge_power"]),
        "center_idx_mismatch_count": int(np.count_nonzero(legacy["center_idx"] != new["center_idx"])),
    }

    if diffs["center_idx_mismatch_count"] != 0:
        raise AssertionError(f"{input_path}: center_idx mismatch")
    for key in (
        "compressed_max_abs_diff",
        "shifted_max_abs_diff",
        "ridge_support_max_abs_diff",
        "basis_max_abs_diff",
        "slot_strength_max_abs_diff",
        "coeff_map_max_abs_diff",
        "ridge_power_max_abs_diff",
        "ridge_compressed_max_abs_diff",
        "residual_compressed_max_abs_diff",
    ):
        if diffs[key] > args.array_max_abs_tol:
            raise AssertionError(f"{input_path}: {key}={diffs[key]:.8e} exceeds tol={args.array_max_abs_tol:.8e}")
    if diffs["ridge_power_relative_l2_diff"] > args.array_rel_l2_tol:
        raise AssertionError(
            f"{input_path}: ridge_power_relative_l2_diff={diffs['ridge_power_relative_l2_diff']:.8e} exceeds tol={args.array_rel_l2_tol:.8e}"
        )
    if diffs["ridge_audio_max_abs_diff"] > args.audio_max_abs_tol:
        raise AssertionError(
            f"{input_path}: ridge_audio_max_abs_diff={diffs['ridge_audio_max_abs_diff']:.8e} exceeds tol={args.audio_max_abs_tol:.8e}"
        )

    legacy_ridge_png = os.path.join(sample_dir, "legacy_ridge.png")
    new_ridge_png = os.path.join(sample_dir, "selfcontained_ridge.png")
    diff_png = os.path.join(sample_dir, "ridge_abs_diff.png")
    compressed_png = os.path.join(sample_dir, "compressed_stft.png")
    legacy_support_png = os.path.join(sample_dir, "legacy_support.png")
    new_support_png = os.path.join(sample_dir, "selfcontained_support.png")
    legacy_center_png = os.path.join(sample_dir, "legacy_center_idx.png")
    new_center_png = os.path.join(sample_dir, "selfcontained_center_idx.png")

    ridge_diff = np.abs(legacy["ridge_compressed"] - new["ridge_compressed"])
    save_matrix_png(compressed_png, legacy["compressed"], "Compressed Whisper STFT", "STFT bins")
    save_matrix_png(legacy_ridge_png, legacy["ridge_compressed"], "Legacy Ridge (Compressed)", "STFT bins")
    save_matrix_png(new_ridge_png, new["ridge_compressed"], "Self-contained Ridge (Compressed)", "STFT bins")
    save_matrix_png(diff_png, ridge_diff, "Absolute Difference", "STFT bins")
    save_matrix_png(legacy_support_png, legacy["ridge_support"], "Legacy Support", "STFT bins")
    save_matrix_png(new_support_png, new["ridge_support"], "Self-contained Support", "STFT bins")
    save_slot_png(
        legacy_center_png,
        np.where(legacy["center_idx"] >= 0, legacy["center_idx"], np.nan),
        "Legacy Center Index Matrix",
        "slot k",
    )
    save_slot_png(
        new_center_png,
        np.where(new["center_idx"] >= 0, new["center_idx"], np.nan),
        "Self-contained Center Index Matrix",
        "slot k",
    )

    report_path = os.path.join(sample_dir, "report.html")
    summary_rows = [
        ("input", os.path.abspath(input_path)),
        ("shape", f"compressed={tuple(legacy['compressed'].shape)} center_idx={tuple(legacy['center_idx'].shape)}"),
        ("compressed_max_abs_diff", f"{diffs['compressed_max_abs_diff']:.8e}"),
        ("shifted_max_abs_diff", f"{diffs['shifted_max_abs_diff']:.8e}"),
        ("ridge_support_max_abs_diff", f"{diffs['ridge_support_max_abs_diff']:.8e}"),
        ("basis_max_abs_diff", f"{diffs['basis_max_abs_diff']:.8e}"),
        ("slot_strength_max_abs_diff", f"{diffs['slot_strength_max_abs_diff']:.8e}"),
        ("coeff_map_max_abs_diff", f"{diffs['coeff_map_max_abs_diff']:.8e}"),
        ("ridge_power_max_abs_diff", f"{diffs['ridge_power_max_abs_diff']:.8e}"),
        ("ridge_power_relative_l2_diff", f"{diffs['ridge_power_relative_l2_diff']:.8e}"),
        ("ridge_compressed_max_abs_diff", f"{diffs['ridge_compressed_max_abs_diff']:.8e}"),
        ("ridge_audio_max_abs_diff", f"{diffs['ridge_audio_max_abs_diff']:.8e}"),
        ("center_idx_mismatch_count", str(diffs["center_idx_mismatch_count"])),
        (
            "tolerances",
            (
                f"array_max_abs<={args.array_max_abs_tol:.1e}, "
                f"array_rel_l2<={args.array_rel_l2_tol:.1e}, "
                f"audio_max_abs<={args.audio_max_abs_tol:.1e}"
            ),
        ),
    ]
    image_rows = [
        {"title": "Compressed STFT", "src": "compressed_stft.png"},
        {"title": "Legacy Ridge", "src": "legacy_ridge.png"},
        {"title": "Self-contained Ridge", "src": "selfcontained_ridge.png"},
        {"title": "Absolute Difference", "src": "ridge_abs_diff.png"},
        {"title": "Legacy Support", "src": "legacy_support.png"},
        {"title": "Self-contained Support", "src": "selfcontained_support.png"},
        {"title": "Legacy Center Index", "src": "legacy_center_idx.png"},
        {"title": "Self-contained Center Index", "src": "selfcontained_center_idx.png"},
    ]
    audio_rows = [
        {"title": "Legacy Ridge Audio", "src": "legacy/ridge_griffinlim.wav"},
        {"title": "Self-contained Ridge Audio", "src": "selfcontained/ridge_griffinlim.wav"},
    ]
    write_report(
        report_path,
        f"Compressed STFT Self-contained Check: {os.path.basename(input_path)}",
        summary_rows,
        image_rows,
        audio_rows,
    )

    return {
        "name": os.path.basename(input_path),
        "slug": slug,
        "report_html": f"{slug}/report.html",
        "legacy_ridge_png": f"{slug}/legacy_ridge.png",
        "new_ridge_png": f"{slug}/selfcontained_ridge.png",
        "diff_png": f"{slug}/ridge_abs_diff.png",
        "legacy_ridge_wav": f"{slug}/legacy/ridge_griffinlim.wav",
        "new_ridge_wav": f"{slug}/selfcontained/ridge_griffinlim.wav",
        **diffs,
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
