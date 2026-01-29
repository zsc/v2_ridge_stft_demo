import os
import numpy as np
import librosa
import matplotlib.pyplot as plt


def mel_spectrogram(y, sr, n_fft, hop_length, win_length, n_mels, fmin, fmax):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=1.0,
    )
    return S


def save_mel_png(path, M, sr, hop_length, title):
    eps = 1e-8
    M_db = 20.0 * np.log10(M + eps)
    plt.figure(figsize=(6, 4))
    plt.imshow(M_db, origin="lower", aspect="auto", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Mel bins")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_html(
    path,
    gt_png,
    recon_png,
    gt_wav,
    recon_wav,
    params=None,
    ridge_png=None,
    ridge_wav=None,
    ridge_direct_png=None,
):
    summary = ""
    if params:
        summary_items = " ".join(f"{k}={v}" for k, v in params.items())
        summary = f"<p class=\"summary\">{summary_items}</p>"
    ridge_column = ""
    if ridge_png and ridge_wav:
        ridge_column = f"""
  <div class=\"column\">
    <h2>Ridge Only</h2>
    <img src=\"{ridge_png}\" alt=\"Ridge Mel\" />
    <audio controls src=\"{ridge_wav}\"></audio>
  </div>"""
    ridge_direct_column = ""
    if ridge_direct_png:
        ridge_direct_column = f"""
  <div class=\"column\">
    <h2>R (Direct)</h2>
    <img src=\"{ridge_direct_png}\" alt=\"Ridge Direct\" />
  </div>"""
    html = f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\">
<title>GT vs Reconstruction</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
.container {{ display: flex; gap: 24px; flex-wrap: wrap; }}
.column {{ flex: 1; }}
img {{ width: 100%; height: auto; border: 1px solid #ccc; }}
.summary {{ color: #333; font-size: 12px; margin-bottom: 12px; }}
</style>
</head>
<body>
{summary}
<div class=\"container\">
  <div class=\"column\">
    <h2>Ground Truth</h2>
    <img src=\"{gt_png}\" alt=\"GT Mel\" />
    <audio controls src=\"{gt_wav}\"></audio>
  </div>
  {ridge_column}
  {ridge_direct_column}
  <div class=\"column\">
    <h2>Reconstruction</h2>
    <img src=\"{recon_png}\" alt=\"Recon Mel\" />
    <audio controls src=\"{recon_wav}\"></audio>
  </div>
</div>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
