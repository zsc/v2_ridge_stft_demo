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
    gaussian_direct_png=None,
    fit_sum_direct_png=None,
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
    gaussian_direct_column = ""
    if gaussian_direct_png:
        gaussian_direct_column = f"""
  <div class=\"column\">
    <h2>Gaussian (Direct)</h2>
    <img src=\"{gaussian_direct_png}\" alt=\"Gaussian Direct\" />
  </div>"""
    fit_sum_direct_column = ""
    if fit_sum_direct_png:
        fit_sum_direct_column = f"""
  <div class=\"column\">
    <h2>R + G (Direct)</h2>
    <img src=\"{fit_sum_direct_png}\" alt=\"Fit Sum Direct\" />
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
  {gaussian_direct_column}
  {fit_sum_direct_column}
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


def write_gallery_html(path, records, title="Batch Mel Gallery"):
    sections = []
    toc = []
    for record in records:
        name = record["name"]
        slug = record["slug"]
        toc.append(f'<li><a href="#{slug}">{name}</a></li>')
        sections.append(
            f"""
<section class=\"item\" id=\"{slug}\">
  <div class=\"item-header\">
    <h2>{name}</h2>
    <p>
      <a href=\"{record['report_html']}\">Per-file report</a>
      <a href=\"{record['gt_wav']}\">GT wav</a>
      <a href=\"{record['ridge_wav']}\">Ridge wav</a>
      <a href=\"{record['recon_wav']}\">Recon wav</a>
    </p>
  </div>
  <div class=\"grid\">
    <div class=\"column\">
      <h3>GT Mel</h3>
      <img src=\"{record['gt_png']}\" alt=\"GT Mel\" />
    </div>
    <div class=\"column\">
      <h3>Ridge Component</h3>
      <img src=\"{record['ridge_direct_png']}\" alt=\"Ridge Component\" />
    </div>
    <div class=\"column\">
      <h3>Gaussian Component</h3>
      <img src=\"{record['gaussian_direct_png']}\" alt=\"Gaussian Component\" />
    </div>
    <div class=\"column\">
      <h3>Fit Sum</h3>
      <img src=\"{record['fit_sum_direct_png']}\" alt=\"Fit Sum\" />
    </div>
  </div>
</section>"""
        )

    html = f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\">
<title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f1f1f; }}
a {{ color: #0b57d0; margin-right: 12px; }}
ul {{ columns: 2; padding-left: 20px; }}
.item {{ margin-bottom: 40px; padding-bottom: 24px; border-bottom: 1px solid #ddd; }}
.item-header {{ margin-bottom: 12px; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 18px; }}
.column {{ margin: 0; }}
img {{ width: 100%; height: auto; border: 1px solid #ccc; background: #111; }}
@media (min-width: 1400px) {{
  .grid {{ grid-template-columns: repeat(4, minmax(280px, 1fr)); }}
}}
</style>
</head>
<body>
<h1>{title}</h1>
<p>Total files: {len(records)}</p>
<ul>
{''.join(toc)}
</ul>
{''.join(sections)}
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
