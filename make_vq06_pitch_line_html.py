#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_JOB_NAME = "bifrost-2026031608521600-zhousc6"
DEFAULT_OUTPUT_HTML = ROOT / "html" / "vq06_pitch_lines_20260322.html"
TASK_ROOT = (
    "/dataset-cpfs2/vtm-foundation-pretrain/temp/zhousc6/"
    "wav2vq0206_lb_completed_20260318_v2/vq06_mel_pairs_edp"
)


def _run_sofy_python(job_name: str, rank: int, python_source: str, timeout_sec: float = 1800.0) -> subprocess.CompletedProcess[str]:
    remote_cmd = "python3 - <<'PY'\n" + python_source + "\nPY"
    return subprocess.run(
        [
            "sofy",
            "job",
            "shell",
            "--job-name",
            job_name,
            "--rank",
            str(rank),
            "--timeout",
            str(timeout_sec),
            "--",
            "bash",
            "-lc",
            remote_cmd,
        ],
        text=True,
        capture_output=True,
        check=True,
    )


def _remote_report_script(sample_count: int, seed: int, candidate_count: int) -> str:
    config_json = json.dumps(
        {
            "task_root": TASK_ROOT,
            "sample_count": int(sample_count),
            "seed": int(seed),
            "candidate_count": int(candidate_count),
        },
        ensure_ascii=False,
    )
    template = """
        import base64
        import html
        import io
        import json
        import math
        import random
        from pathlib import Path

        import librosa
        import numpy as np
        import soundfile as sf
        from PIL import Image, ImageDraw
        from scipy.signal import find_peaks

        CONFIG = json.loads(__CONFIG_JSON__)
        TASK_ROOT = Path(CONFIG["task_root"])
        TARGET_SR = 16000
        WHISPER_N_FFT = 400
        WHISPER_HOP = 160
        WHISPER_N_MELS = 128
        LINEAR_FREQ_STEP_HZ = 10.0
        F0_MIN_HZ = 60.0
        F0_MAX_HZ = 600.0
        PEAK_CFG = {
            "distance": 3,
            "prominence": 0.015,
            "height": 0.05,
            "dilate": True,
        }
        RIDGE_CFG = {
            "k": 1,
            "lam_sparse": 0.02,
            "lam_tv": 0.10,
            "ridge_refine": True,
            "max_iter": 160,
            "step_size": 0.20,
            "smooth_eps": 1.0e-3,
            "tol": 1.0e-5,
        }
        VOICED_CFG = {
            "min_ridge_value": 0.020,
            "min_peak_value": 0.080,
            "min_peak_margin": 0.015,
            "min_voiced_run_frames": 3,
            "bridge_unvoiced_gap_frames": 2,
        }


        class MelAxis:
            def __init__(self, sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> None:
                self.sr = int(sr)
                self.n_fft = int(n_fft)
                self.n_mels = int(n_mels)
                self.fmin = float(fmin)
                self.fmax = float(fmax)
                self.center_frequencies_hz = librosa.mel_frequencies(
                    n_mels=self.n_mels,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    htk=False,
                ).astype(np.float64)

            def hz_to_melbin(self, frequency_hz):
                frequencies = np.asarray(frequency_hz, dtype=np.float64)
                clipped = np.clip(frequencies, self.fmin, self.fmax)
                return np.interp(
                    clipped,
                    self.center_frequencies_hz,
                    np.arange(self.n_mels, dtype=np.float64),
                )

            def melbin_to_hz(self, bin_index):
                bins = np.asarray(bin_index, dtype=np.float64)
                clipped = np.clip(bins, 0.0, self.n_mels - 1.0)
                return np.interp(
                    clipped,
                    np.arange(self.n_mels, dtype=np.float64),
                    self.center_frequencies_hz,
                )


        def normalize_mono_wav(wav: np.ndarray) -> np.ndarray:
            array = np.asarray(wav, dtype=np.float32)
            if array.ndim == 2:
                array = array.mean(axis=1)
            max_abs = float(np.max(np.abs(array))) if array.size else 0.0
            if max_abs > 0.6:
                array = array / max_abs * 0.6
            return array.astype(np.float32)


        def energy_norm_fn(wav: np.ndarray) -> np.ndarray:
            max_data = float(np.max(np.abs(wav))) if wav.size else 0.0
            scale = max(max_data, 0.01)
            return (wav / scale * 0.999).astype(np.float32)


        def trim_silence(
            audio: np.ndarray,
            sr: int,
            keep_left_time: float = 0.05,
            keep_right_time: float = 0.22,
            hop_size: int = 240,
        ) -> np.ndarray:
            if audio.size == 0:
                return np.zeros(0, dtype=np.float32)
            _, index = librosa.effects.trim(audio, top_db=20, frame_length=512, hop_length=128)
            num_frames = int(math.ceil(max(index[1] - index[0], 0) / hop_size))
            left_sil_samples = int(keep_left_time * sr)
            start_idx = index[0] - left_sil_samples
            trim_wav = audio
            if start_idx > 0:
                trim_wav = trim_wav[start_idx:]
            else:
                trim_wav = np.pad(trim_wav, (abs(start_idx), 0), mode="constant", constant_values=0.0)
            wav_len = len(trim_wav)
            out_len = int(num_frames * hop_size + (keep_left_time + keep_right_time) * sr)
            if out_len < wav_len:
                trim_wav = trim_wav[:out_len]
            else:
                trim_wav = np.pad(trim_wav, (0, out_len - wav_len), mode="constant", constant_values=0.0)
            return trim_wav.astype(np.float32)


        def preprocess_wav(wav: np.ndarray, sample_rate: int) -> np.ndarray:
            array = normalize_mono_wav(wav)
            if int(sample_rate) != TARGET_SR:
                array = librosa.resample(array, orig_sr=int(sample_rate), target_sr=TARGET_SR)
            array = energy_norm_fn(array)
            array = trim_silence(array, TARGET_SR)
            return array.astype(np.float32)


        def compute_vq06_mel(audio: np.ndarray) -> np.ndarray:
            stft = librosa.stft(
                audio,
                n_fft=WHISPER_N_FFT,
                hop_length=WHISPER_HOP,
                win_length=WHISPER_N_FFT,
                window="hann",
                center=True,
                pad_mode="reflect",
            )
            if stft.shape[1] > 0:
                stft = stft[:, :-1]
            power = np.abs(stft) ** 2
            mel_filter = librosa.filters.mel(
                sr=TARGET_SR,
                n_fft=WHISPER_N_FFT,
                n_mels=WHISPER_N_MELS,
            ).astype(np.float32)
            mel = mel_filter @ power
            log_spec = np.log10(np.clip(mel, 1.0e-10, None))
            peak = float(np.max(log_spec)) if log_spec.size else 0.0
            log_spec = np.maximum(log_spec, peak - 8.0)
            return ((log_spec + 4.0) / 4.0).astype(np.float32)

        def mel_inverse_pinv(F_mel: np.ndarray, M: np.ndarray) -> np.ndarray:
            F_pinv = np.linalg.pinv(F_mel)
            A_hat = F_pinv @ M
            return np.clip(A_hat, 0.0, None)


        def mel_to_linear_spectrum(mel: np.ndarray) -> np.ndarray:
            mel_log10 = mel.astype(np.float64) * 4.0 - 4.0
            mel_power = np.power(10.0, mel_log10, dtype=np.float64)
            mel_filter = librosa.filters.mel(
                sr=TARGET_SR,
                n_fft=WHISPER_N_FFT,
                n_mels=WHISPER_N_MELS,
            ).astype(np.float64)
            linear_power = mel_inverse_pinv(mel_filter, mel_power)
            return linear_power.astype(np.float32)


        def upsample_linear_spectrum(linear_power: np.ndarray):
            source_hz = np.linspace(0.0, TARGET_SR / 2.0, linear_power.shape[0], dtype=np.float64)
            target_hz = np.arange(0.0, TARGET_SR / 2.0 + LINEAR_FREQ_STEP_HZ, LINEAR_FREQ_STEP_HZ, dtype=np.float64)
            upsampled = np.empty((target_hz.size, linear_power.shape[1]), dtype=np.float32)
            for frame_index in range(linear_power.shape[1]):
                upsampled[:, frame_index] = np.interp(
                    target_hz,
                    source_hz,
                    linear_power[:, frame_index].astype(np.float64),
                ).astype(np.float32)
            return upsampled, target_hz.astype(np.float32)


        def compute_frequency_acf_image(linear_power: np.ndarray):
            upsampled, target_hz = upsample_linear_spectrum(linear_power)
            lag_min = max(1, int(np.floor(F0_MIN_HZ / LINEAR_FREQ_STEP_HZ)))
            lag_max = min(upsampled.shape[0] - 2, int(np.ceil(F0_MAX_HZ / LINEAR_FREQ_STEP_HZ)))
            lag_hz = target_hz[lag_min : lag_max + 1]
            rows = []
            for frame_index in range(upsampled.shape[1]):
                spectrum = np.sqrt(np.clip(upsampled[:, frame_index], 0.0, None)).astype(np.float64)
                peak = float(np.max(spectrum)) if spectrum.size else 0.0
                if peak > 0.0:
                    spectrum = spectrum / peak
                centered = spectrum - float(np.mean(spectrum))
                corr = np.correlate(centered, centered, mode="full")
                positive = corr[corr.size // 2 :]
                scale = float(max(float(positive[0]), 1.0e-6))
                normalized = positive / scale
                normalized = np.clip(normalized, 0.0, None)
                rows.append(normalized[lag_min : lag_max + 1].astype(np.float32))
            return np.asarray(rows, dtype=np.float32).T, lag_hz.astype(np.float32)


        def _fill_short_gaps(ridge: np.ndarray, voiced_mask: np.ndarray, max_gap: int):
            if max_gap <= 0:
                return ridge, voiced_mask
            filled_mask = voiced_mask.copy()
            filled_ridge = ridge.copy()
            total = voiced_mask.size
            index = 0
            while index < total:
                if filled_mask[index]:
                    index += 1
                    continue
                gap_start = index
                while index < total and not filled_mask[index]:
                    index += 1
                gap_end = index
                gap_size = gap_end - gap_start
                if gap_start == 0 or gap_end == total or gap_size > max_gap:
                    continue
                left = gap_start - 1
                right = gap_end
                if not (filled_mask[left] and filled_mask[right]):
                    continue
                values = np.linspace(filled_ridge[left], filled_ridge[right], gap_size + 2)[1:-1]
                filled_ridge[gap_start:gap_end] = values
                filled_mask[gap_start:gap_end] = True
            return filled_ridge, filled_mask


        def _remove_short_runs(ridge: np.ndarray, voiced_mask: np.ndarray, min_run: int):
            if min_run <= 1:
                return ridge, voiced_mask
            cleaned_mask = voiced_mask.copy()
            total = voiced_mask.size
            index = 0
            while index < total:
                if not cleaned_mask[index]:
                    index += 1
                    continue
                start = index
                while index < total and cleaned_mask[index]:
                    index += 1
                end = index
                if end - start < min_run:
                    cleaned_mask[start:end] = False
                    ridge[start:end] = np.nan
            return ridge, cleaned_mask


        def peak_mask_local(M, distance=2, prominence=None, height=None, dilate=False):
            n_bins, n_frames = M.shape
            mask = np.zeros((n_bins, n_frames), dtype=bool)
            for frame_index in range(n_frames):
                peaks, _ = find_peaks(
                    M[:, frame_index],
                    distance=distance,
                    prominence=prominence,
                    height=height,
                )
                if peaks.size > 0:
                    mask[peaks, frame_index] = True
            if dilate:
                dilated = mask.copy()
                if n_bins > 1:
                    dilated[1:, :] |= mask[:-1, :]
                    dilated[:-1, :] |= mask[1:, :]
                mask = dilated
            return mask


        def _ridge_objective(R: np.ndarray, M: np.ndarray, lam_sparse: float, lam_tv: float, smooth_eps: float) -> float:
            diff = R[:, 1:] - R[:, :-1] if R.shape[1] > 1 else np.zeros((R.shape[0], 0), dtype=R.dtype)
            sparse_term = np.sqrt(R * R + smooth_eps * smooth_eps).sum()
            tv_term = np.sqrt(diff * diff + smooth_eps * smooth_eps).sum()
            return float(0.5 * np.square(R - M).sum() + lam_sparse * sparse_term + lam_tv * tv_term)


        def _ridge_gradient(R: np.ndarray, M: np.ndarray, lam_sparse: float, lam_tv: float, smooth_eps: float) -> np.ndarray:
            grad = R - M
            grad += lam_sparse * (R / np.sqrt(R * R + smooth_eps * smooth_eps))
            if R.shape[1] > 1:
                diff = R[:, 1:] - R[:, :-1]
                scaled = diff / np.sqrt(diff * diff + smooth_eps * smooth_eps)
                grad[:, :-1] -= lam_tv * scaled
                grad[:, 1:] += lam_tv * scaled
            return grad


        def _solve_ridge(M, V, lam_sparse, lam_tv, solver):
            del solver
            upper = np.where(V, M, 0.0).astype(np.float32)
            R = np.clip(upper.copy(), 0.0, upper)
            Y = R.copy()
            t_value = 1.0
            step = float(RIDGE_CFG["step_size"])
            smooth_eps = float(RIDGE_CFG["smooth_eps"])
            tol = float(RIDGE_CFG["tol"])
            max_iter = int(RIDGE_CFG["max_iter"])
            prev_obj = _ridge_objective(R, M, lam_sparse, lam_tv, smooth_eps)
            for _ in range(max_iter):
                grad = _ridge_gradient(Y, M, lam_sparse, lam_tv, smooth_eps)
                local_step = step
                accepted = False
                candidate = R
                candidate_obj = prev_obj
                for _ in range(8):
                    proposal = np.clip(Y - local_step * grad, 0.0, upper)
                    proposal[V == 0] = 0.0
                    proposal_obj = _ridge_objective(proposal, M, lam_sparse, lam_tv, smooth_eps)
                    if proposal_obj <= prev_obj or local_step <= 1.0e-4:
                        candidate = proposal.astype(np.float32)
                        candidate_obj = proposal_obj
                        step = local_step
                        accepted = True
                        break
                    local_step *= 0.5
                if not accepted:
                    candidate = np.clip(Y - step * grad, 0.0, upper).astype(np.float32)
                    candidate[V == 0] = 0.0
                    candidate_obj = _ridge_objective(candidate, M, lam_sparse, lam_tv, smooth_eps)
                if abs(prev_obj - candidate_obj) <= tol * max(prev_obj, 1.0):
                    R = candidate
                    break
                t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_value * t_value))
                Y = candidate + ((t_value - 1.0) / t_next) * (candidate - R)
                R = candidate
                t_value = t_next
                prev_obj = candidate_obj
            R = np.clip(R, 0.0, upper)
            R[V == 0] = 0.0
            return R.astype(np.float32)


        def _topk_per_frame(R, V, k):
            n_bins, n_frames = R.shape
            topk = np.zeros_like(R)
            for frame_index in range(n_frames):
                idx = np.where(V[:, frame_index])[0]
                if idx.size == 0:
                    continue
                values = R[idx, frame_index]
                if idx.size > k:
                    topk_local = np.argpartition(values, -k)[-k:]
                    keep = idx[topk_local]
                else:
                    keep = idx
                topk[keep, frame_index] = R[keep, frame_index]
            return topk


        def ridge_optimize_local(M, V, k=1, lam_sparse=0.05, lam_tv=0.1, solver="projected_fista", ridge_refine=True):
            R_raw = _solve_ridge(M, V, lam_sparse, lam_tv, solver)
            R_topk = _topk_per_frame(R_raw, V, k)
            if ridge_refine:
                V2 = R_topk > 0
                if V2.any():
                    R_refined = _solve_ridge(M, V2, lam_sparse, lam_tv, solver)
                    R_topk = _topk_per_frame(R_refined, V2, k)
            R_topk[V == 0] = 0.0
            return R_topk.astype(np.float32)


        def extract_pitch_line(acf_image: np.ndarray, lag_hz: np.ndarray):
            peak_mask = peak_mask_local(
                acf_image,
                distance=int(PEAK_CFG["distance"]),
                prominence=float(PEAK_CFG["prominence"]),
                height=float(PEAK_CFG["height"]),
                dilate=bool(PEAK_CFG["dilate"]),
            )
            if not peak_mask.any():
                raise RuntimeError("no acf peaks")
            ridge_energy = ridge_optimize_local(
                acf_image,
                peak_mask,
                k=int(RIDGE_CFG["k"]),
                lam_sparse=float(RIDGE_CFG["lam_sparse"]),
                lam_tv=float(RIDGE_CFG["lam_tv"]),
                ridge_refine=bool(RIDGE_CFG["ridge_refine"]),
            )
            n_lags, n_frames = ridge_energy.shape
            best_idx = np.argmax(ridge_energy, axis=0)
            frame_indices = np.arange(n_frames, dtype=np.int32)
            ridge_value = ridge_energy[best_idx, frame_indices]
            peak_value = acf_image[best_idx, frame_indices]
            peak_margin = np.zeros(n_frames, dtype=np.float32)
            for frame_index in range(n_frames):
                row = acf_image[:, frame_index].copy()
                center = int(best_idx[frame_index])
                start = max(0, center - 2)
                end = min(n_lags, center + 3)
                row[start:end] = 0.0
                peak_margin[frame_index] = float(max(peak_value[frame_index] - row.max(initial=0.0), 0.0))
            confidence = ridge_value * peak_margin
            voiced_mask = ridge_value >= float(VOICED_CFG["min_ridge_value"])
            voiced_mask &= peak_value >= float(VOICED_CFG["min_peak_value"])
            voiced_mask &= peak_margin >= float(VOICED_CFG["min_peak_margin"])
            pitch_hz = lag_hz[best_idx].astype(np.float32)
            pitch_hz[~voiced_mask] = np.nan
            pitch_hz, voiced_mask = _remove_short_runs(
                pitch_hz,
                voiced_mask,
                int(VOICED_CFG["min_voiced_run_frames"]),
            )
            pitch_hz, voiced_mask = _fill_short_gaps(
                pitch_hz,
                voiced_mask,
                int(VOICED_CFG["bridge_unvoiced_gap_frames"]),
            )
            confidence[~voiced_mask] = 0.0
            return pitch_hz.astype(np.float32), voiced_mask, confidence.astype(np.float32), peak_value.astype(np.float32)


        def image_to_data_uri(image: Image.Image) -> str:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


        def prepare_display_image(mel: np.ndarray, pitch_melbin: np.ndarray, voiced_mask: np.ndarray):
            mel_min = float(np.min(mel)) if mel.size else 0.0
            mel_max = float(np.max(mel)) if mel.size else 1.0
            if mel_max <= mel_min:
                mel_max = mel_min + 1.0e-6
            norm = np.clip((mel - mel_min) / (mel_max - mel_min), 0.0, 1.0)
            uint8_data = (np.flipud(norm) * 255.0).astype(np.uint8)
            base = Image.fromarray(uint8_data, mode="L")
            width = max(1, base.width)
            height = max(1, base.height)
            scale = min(160.0 / height, 720.0 / width if width > 0 else 1.0)
            if scale <= 0:
                scale = 1.0
            out_w = max(1, int(round(width * scale)))
            out_h = max(1, int(round(height * scale)))
            if out_w != width or out_h != height:
                base = base.resize((out_w, out_h), resample=Image.Resampling.BICUBIC)
            overlay = base.convert("RGB")
            draw = ImageDraw.Draw(overlay)
            points = []
            x_scale = out_w / max(1, mel.shape[1])
            y_scale = out_h / max(1, mel.shape[0])
            for frame_index, (bin_value, is_voiced) in enumerate(zip(pitch_melbin, voiced_mask)):
                if not is_voiced or not np.isfinite(bin_value):
                    if len(points) > 1:
                        draw.line(points, fill=(255, 104, 73), width=max(2, out_h // 64))
                    points = []
                    continue
                x = min(out_w - 1, max(0, int(round(frame_index * x_scale))))
                y = min(out_h - 1, max(0, int(round((mel.shape[0] - 1 - float(bin_value)) * y_scale))))
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=(255, 104, 73), width=max(2, out_h // 64))
            return image_to_data_uri(base), image_to_data_uri(overlay)


        def reservoir_candidates(meta_root: Path, candidate_count: int, seed: int):
            rng = random.Random(seed)
            sample = []
            seen = 0
            for path in sorted(meta_root.glob("*/*.jsonl")):
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        seen += 1
                        if len(sample) < candidate_count:
                            sample.append(row)
                            continue
                        index = rng.randrange(seen)
                        if index < candidate_count:
                            sample[index] = row
            rng.shuffle(sample)
            return sample, seen


        def process_candidate(row: dict):
            summary_path = (
                TASK_ROOT
                / "output"
                / "summary"
                / str(row["dataset_key"])
                / f"{row['chunk_base']}.summary.json"
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            wav_path = Path(summary["input_root"]) / str(row["wav_relpath"])
            wav, sr = sf.read(str(wav_path))
            processed = preprocess_wav(wav, int(sr))
            mel = compute_vq06_mel(processed)
            if mel.shape[1] < 8:
                raise RuntimeError("mel too short")
            linear_spectrum = mel_to_linear_spectrum(mel)
            acf_image, lag_hz = compute_frequency_acf_image(linear_spectrum)
            pitch_hz, voiced_mask, confidence, peak_value = extract_pitch_line(acf_image, lag_hz)
            if int(np.count_nonzero(voiced_mask)) < 6:
                raise RuntimeError("too few voiced frames")
            mel_axis = MelAxis(TARGET_SR, WHISPER_N_FFT, WHISPER_N_MELS, 0.0, TARGET_SR / 2.0)
            pitch_melbin = mel_axis.hz_to_melbin(np.nan_to_num(pitch_hz, nan=0.0)).astype(np.float32)
            pitch_melbin[~voiced_mask] = np.nan
            mel_uri, overlay_uri = prepare_display_image(mel, pitch_melbin, voiced_mask)
            voiced_hz = pitch_hz[voiced_mask] if np.any(voiced_mask) else np.zeros(0, dtype=np.float32)
            return {
                "dataset_key": str(row["dataset_key"]),
                "chunk_base": str(row["chunk_base"]),
                "task_name": str(row.get("task_name", "")),
                "wav_relpath": str(row["wav_relpath"]),
                "utterance": str(row.get("utterance", "")),
                "source_sr": int(sr),
                "frames": int(mel.shape[1]),
                "bins": int(mel.shape[0]),
                "duration_sec": float(len(processed) / TARGET_SR),
                "voiced_ratio": float(np.mean(voiced_mask.astype(np.float32))),
                "voiced_frames": int(np.count_nonzero(voiced_mask)),
                "confidence_mean": float(confidence[voiced_mask].mean()) if np.any(voiced_mask) else 0.0,
                "acf_peak_mean": float(peak_value[voiced_mask].mean()) if np.any(voiced_mask) else 0.0,
                "f0_median_hz": float(np.median(voiced_hz)) if voiced_hz.size else 0.0,
                "f0_mean_hz": float(np.mean(voiced_hz)) if voiced_hz.size else 0.0,
                "mel_uri": mel_uri,
                "overlay_uri": overlay_uri,
            }


        def build_html(items: list[dict], seen_total: int) -> str:
            cards = []
            for index, item in enumerate(items, start=1):
                cards.append(
                    f\"\"\"
                    <article class="card">
                      <div class="card-head">
                        <div>
                          <div class="index">#{index:03d}</div>
                          <div class="dataset">{html.escape(item["dataset_key"])}</div>
                          <div class="task">{html.escape(item["task_name"])}</div>
                          <div class="sub">{html.escape(item["wav_relpath"])}</div>
                        </div>
                        <div class="stats">
                          <div>{item["frames"]} x {item["bins"]}</div>
                          <div>{item["duration_sec"]:.2f} s</div>
                          <div>voiced {item["voiced_ratio"] * 100:.1f}%</div>
                          <div>median {item["f0_median_hz"]:.1f} Hz</div>
                          <div>conf {item["confidence_mean"]:.4f}</div>
                          <div>acf {item["acf_peak_mean"]:.4f}</div>
                        </div>
                      </div>
                      <div class="utt">{html.escape(item["utterance"])}</div>
                      <div class="pair">
                        <figure>
                          <div class="label">vq06 mel</div>
                          <img src="{item["mel_uri"]}" alt="vq06 mel">
                        </figure>
                        <figure>
                          <div class="label">pitch line overlay</div>
                          <img src="{item["overlay_uri"]}" alt="pitch line overlay">
                        </figure>
                      </div>
                    </article>
                    \"\"\".strip()
                )
            return f\"\"\"<!doctype html>
        <html lang="zh-CN">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>vq06 Mel Pitch Lines</title>
          <style>
            :root {{
              --bg: #f1ecdf;
              --paper: rgba(255, 252, 245, 0.94);
              --ink: #1d2524;
              --muted: #66716d;
              --line: #d8cfbf;
              --accent: #8c4c2e;
            }}
            * {{ box-sizing: border-box; }}
            body {{
              margin: 0;
              color: var(--ink);
              font-family: "IBM Plex Sans", "Helvetica Neue", "PingFang SC", sans-serif;
              background:
                radial-gradient(circle at top left, rgba(140,76,46,0.10), transparent 30%),
                linear-gradient(180deg, #f8f4ea 0%, #ece3d2 100%);
            }}
            main {{
              max-width: 1640px;
              margin: 0 auto;
              padding: 28px 24px 48px;
            }}
            h1 {{
              margin: 0 0 10px;
              font-size: 34px;
              letter-spacing: -0.04em;
            }}
            .summary {{
              color: var(--muted);
              margin-bottom: 24px;
              line-height: 1.6;
            }}
            .grid {{
              display: grid;
              gap: 18px;
            }}
            .card {{
              background: var(--paper);
              border: 1px solid var(--line);
              border-radius: 18px;
              padding: 18px;
              box-shadow: 0 18px 42px rgba(43, 33, 25, 0.08);
            }}
            .card-head {{
              display: flex;
              justify-content: space-between;
              gap: 18px;
              align-items: flex-start;
            }}
            .index {{
              color: var(--accent);
              font-size: 12px;
              letter-spacing: 0.08em;
              text-transform: uppercase;
              margin-bottom: 4px;
            }}
            .dataset {{
              color: var(--accent);
              font-size: 13px;
              letter-spacing: 0.08em;
              text-transform: uppercase;
              margin-bottom: 4px;
            }}
            .task {{
              font-size: 14px;
              color: var(--muted);
              word-break: break-all;
            }}
            .sub {{
              margin-top: 8px;
              color: var(--muted);
              font-size: 12px;
              word-break: break-all;
            }}
            .stats {{
              color: var(--muted);
              font-size: 13px;
              text-align: right;
              white-space: nowrap;
              line-height: 1.5;
            }}
            .utt {{
              margin: 14px 0 16px;
              font-size: 18px;
              line-height: 1.45;
            }}
            .pair {{
              display: grid;
              grid-template-columns: repeat(2, minmax(0, 1fr));
              gap: 16px;
              align-items: start;
            }}
            figure {{
              margin: 0;
            }}
            .label {{
              color: var(--accent);
              font-size: 12px;
              letter-spacing: 0.08em;
              text-transform: uppercase;
              margin-bottom: 8px;
            }}
            img {{
              display: block;
              width: 100%;
              border: 1px solid var(--line);
              border-radius: 12px;
              background: #faf7f1;
              object-fit: contain;
            }}
            @media (max-width: 980px) {{
              .pair {{ grid-template-columns: 1fr; }}
              .card-head {{ flex-direction: column; }}
              .stats {{ text-align: left; }}
            }}
          </style>
        </head>
        <body>
          <main>
            <h1>vq06-mel Random Pitch Lines</h1>
            <div class="summary">
              {len(items)} samples from {seen_total} total rows under `vq06_mel_pairs_edp/output/meta`.
              Remote analysis follows the current repo's ridge logic more closely:
              vq06 mel -> inverse mel to linear spectrum -> frequency-axis ACF -> vertical peak mask + convex ridge with voiced/unvoiced post-filtering.
              Left is raw vq06 mel; right is the same mel with extracted pitch line overlay.
            </div>
            <section class="grid">
              {''.join(cards)}
            </section>
          </main>
        </body>
        </html>
        \"\"\"


        candidate_rows, seen_total = reservoir_candidates(
            TASK_ROOT / "output" / "meta",
            int(CONFIG["candidate_count"]),
            int(CONFIG["seed"]),
        )
        results = []
        failures = []
        for row in candidate_rows:
            if len(results) >= int(CONFIG["sample_count"]):
                break
            try:
                results.append(process_candidate(row))
            except Exception as exc:
                failures.append({
                    "dataset_key": row.get("dataset_key"),
                    "chunk_base": row.get("chunk_base"),
                    "wav_relpath": row.get("wav_relpath"),
                    "error": str(exc),
                })
        if not results:
            raise SystemExit("no valid samples produced")
        report_path = Path("/tmp") / f"vq06_pitch_lines_{CONFIG['seed']}_{CONFIG['sample_count']}.html"
        report_path.write_text(build_html(results, seen_total), encoding="utf-8")
        print(
            json.dumps(
                {
                    "remote_html": str(report_path),
                    "sample_count": len(results),
                    "candidate_count": len(candidate_rows),
                    "seen_total": seen_total,
                    "failures": failures[:20],
                },
                ensure_ascii=False,
            )
        )
    """
    return textwrap.dedent(template.replace("__CONFIG_JSON__", repr(config_json))).strip()


def _remote_fetch_script(remote_html: str) -> str:
    return textwrap.dedent(
        f"""
        from pathlib import Path
        import sys

        sys.stdout.write(Path({remote_html!r}).read_text(encoding="utf-8"))
        """
    ).strip()


def _remote_cleanup_script(remote_html: str) -> str:
    return textwrap.dedent(
        f"""
        from pathlib import Path

        path = Path({remote_html!r})
        if path.exists():
            path.unlink()
        """
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample random vq06-mel rows on a SOFY job, extract pitch lines, and write a self-contained local HTML."
    )
    parser.add_argument("--job-name", default=DEFAULT_JOB_NAME, type=str)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--sample-count", default=100, type=int)
    parser.add_argument("--seed", default=20260322, type=int)
    parser.add_argument("--candidate-multiplier", default=6, type=int)
    parser.add_argument("--output-html", default=str(DEFAULT_OUTPUT_HTML), type=str)
    parser.add_argument("--open", action="store_true", help="Open the generated local HTML with the default browser.")
    args = parser.parse_args()

    output_html = Path(args.output_html).expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    candidate_count = max(int(args.sample_count), int(args.sample_count) * int(args.candidate_multiplier))

    print(
        f"[vq06-pitch] job={args.job_name} rank={args.rank} sample_count={args.sample_count} "
        f"seed={args.seed} candidate_count={candidate_count}",
        flush=True,
    )
    report_proc = _run_sofy_python(
        job_name=args.job_name,
        rank=args.rank,
        python_source=_remote_report_script(
            sample_count=int(args.sample_count),
            seed=int(args.seed),
            candidate_count=candidate_count,
        ),
    )
    summary_lines = [line for line in report_proc.stdout.splitlines() if line.strip()]
    if not summary_lines:
        raise RuntimeError("remote report step returned no JSON summary")
    summary = json.loads(summary_lines[-1])
    remote_html = str(summary["remote_html"])
    print(
        f"[vq06-pitch] remote_html={remote_html} valid={summary['sample_count']} seen={summary['seen_total']}",
        flush=True,
    )
    if summary.get("failures"):
        print(f"[vq06-pitch] first_failures={json.dumps(summary['failures'][:3], ensure_ascii=False)}", flush=True)

    fetch_proc = _run_sofy_python(
        job_name=args.job_name,
        rank=args.rank,
        python_source=_remote_fetch_script(remote_html),
    )
    output_html.write_text(fetch_proc.stdout, encoding="utf-8")
    print(f"[vq06-pitch] wrote {output_html}", flush=True)

    try:
        _run_sofy_python(
            job_name=args.job_name,
            rank=args.rank,
            python_source=_remote_cleanup_script(remote_html),
        )
    except subprocess.CalledProcessError:
        pass

    if args.open:
        subprocess.run(["open", str(output_html)], check=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            sys.stderr.write(exc.stdout)
            if not exc.stdout.endswith("\n"):
                sys.stderr.write("\n")
        if exc.stderr:
            sys.stderr.write(exc.stderr)
            if not exc.stderr.endswith("\n"):
                sys.stderr.write("\n")
        raise
