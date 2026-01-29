import argparse
import os
import numpy as np

from audio import load_audio, save_audio
from stft_utils import stft, istft, mag_phase
from mel_utils import mel_filter, mel_forward, mel_inverse_pinv, mel_inverse_nnls
from peaks import peak_mask
from ridge_opt import ridge_optimize
from gaussian_opt import gaussian_optimize
from report import mel_spectrogram, save_mel_png, write_html


def parse_args():
    parser = argparse.ArgumentParser(description="WAV -> STFT -> Mel -> Ridge + Gaussian -> Recon -> HTML")
    parser.add_argument("--input", required=True, help="Input wav path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--sr", type=int, default=None, help="Resample rate (default: keep original)")

    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--win_length", type=int, default=2048)
    parser.add_argument("--center", type=lambda x: x.lower() == "true", default=True)

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
    parser.add_argument("--solver", type=str, default="SCS", choices=["SCS", "ECOS", "OSQP"])
    parser.add_argument("--ridge_refine", type=lambda x: x.lower() == "true", default=True)

    parser.add_argument("--spacing_frames", type=int, default=None)
    parser.add_argument("--sigma_t", type=float, default=1.0)
    parser.add_argument("--sigma_f_list", type=str, default="1,2,4,8,16")
    parser.add_argument("--lam_g", type=float, default=0.01)

    parser.add_argument("--mel_inverse", type=str, default="pinv", choices=["pinv", "nnls"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_intermediates", type=lambda x: x.lower() == "true", default=True)

    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    y, sr = load_audio(args.input, sr=args.sr, mono=True)
    save_audio(os.path.join(args.output_dir, "gt.wav"), y, sr)

    S = stft(
        y,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        center=args.center,
    )
    A, P = mag_phase(S)

    if args.save_intermediates:
        np.save(os.path.join(args.output_dir, "stft_mag.npy"), A)
        np.save(os.path.join(args.output_dir, "stft_phase.npy"), P)

    F_mel = mel_filter(sr, args.n_fft, args.n_mels, args.fmin, args.fmax)
    M = mel_forward(F_mel, A)
    if args.save_intermediates:
        np.save(os.path.join(args.output_dir, "mel_mag.npy"), M)

    V = peak_mask(
        M,
        distance=args.peak_distance,
        prominence=args.peak_prominence,
        height=args.peak_height,
        dilate=bool(args.peak_dilate),
    )
    if args.save_intermediates:
        np.save(os.path.join(args.output_dir, "peak_mask.npy"), V)

    R = ridge_optimize(
        M,
        V,
        k=args.k,
        lam_sparse=args.lam_sparse,
        lam_tv=args.lam_tv,
        solver=args.solver,
        ridge_refine=args.ridge_refine,
    )
    if args.save_intermediates:
        np.save(os.path.join(args.output_dir, "ridge.npy"), R)

    # Ridge-only reconstruction
    if args.mel_inverse == "pinv":
        A_hat_ridge = mel_inverse_pinv(F_mel, R)
    else:
        A_hat_ridge = mel_inverse_nnls(F_mel, R)
    S_hat_ridge = A_hat_ridge * np.exp(1j * P)
    y_ridge = istft(
        S_hat_ridge,
        hop_length=args.hop_length,
        win_length=args.win_length,
        center=args.center,
        length=len(y),
    )
    y_ridge = np.clip(y_ridge, -1.0, 1.0)
    save_audio(os.path.join(args.output_dir, "ridge.wav"), y_ridge, sr)

    E = np.clip(M - R, 0.0, None)
    sigma_f_list = [float(x) for x in args.sigma_f_list.split(",") if x.strip()]
    G = gaussian_optimize(
        E,
        R,
        V,
        M,
        k=args.k,
        spacing_frames=args.spacing_frames,
        sigma_t=args.sigma_t,
        sigma_f_list=sigma_f_list,
        lam_g=args.lam_g,
        solver=args.solver,
    )
    if args.save_intermediates:
        np.save(os.path.join(args.output_dir, "gaussian.npy"), G)

    M_hat = np.clip(R + G, 0.0, None)
    if args.save_intermediates:
        np.save(os.path.join(args.output_dir, "mel_hat.npy"), M_hat)

    if args.mel_inverse == "pinv":
        A_hat = mel_inverse_pinv(F_mel, M_hat)
    else:
        A_hat = mel_inverse_nnls(F_mel, M_hat)
    if args.save_intermediates:
        np.save(os.path.join(args.output_dir, "stft_mag_hat.npy"), A_hat)

    S_hat = A_hat * np.exp(1j * P)
    y_hat = istft(
        S_hat,
        hop_length=args.hop_length,
        win_length=args.win_length,
        center=args.center,
        length=len(y),
    )
    y_hat = np.clip(y_hat, -1.0, 1.0)
    save_audio(os.path.join(args.output_dir, "recon.wav"), y_hat, sr)

    gt_mel = mel_spectrogram(
        y,
        sr,
        args.n_fft,
        args.hop_length,
        args.win_length,
        args.n_mels,
        args.fmin,
        args.fmax,
    )
    recon_mel = mel_spectrogram(
        y_hat,
        sr,
        args.n_fft,
        args.hop_length,
        args.win_length,
        args.n_mels,
        args.fmin,
        args.fmax,
    )
    ridge_mel = mel_spectrogram(
        y_ridge,
        sr,
        args.n_fft,
        args.hop_length,
        args.win_length,
        args.n_mels,
        args.fmin,
        args.fmax,
    )
    gt_png = os.path.join(args.output_dir, "gt_mel.png")
    recon_png = os.path.join(args.output_dir, "recon_mel.png")
    ridge_png = os.path.join(args.output_dir, "ridge_mel.png")
    ridge_direct_png = os.path.join(args.output_dir, "ridge_direct.png")
    save_mel_png(gt_png, gt_mel, sr, args.hop_length, "Ground Truth Mel")
    save_mel_png(recon_png, recon_mel, sr, args.hop_length, "Reconstruction Mel")
    save_mel_png(ridge_png, ridge_mel, sr, args.hop_length, "Ridge Only Mel")
    save_mel_png(ridge_direct_png, R, sr, args.hop_length, "Ridge Direct")

    params = {
        "n_fft": args.n_fft,
        "hop": args.hop_length,
        "n_mels": args.n_mels,
        "k": args.k,
        "lam_sparse": args.lam_sparse,
        "lam_tv": args.lam_tv,
        "lam_g": args.lam_g,
    }
    write_html(
        os.path.join(args.output_dir, "report.html"),
        "gt_mel.png",
        "recon_mel.png",
        "gt.wav",
        "recon.wav",
        params=params,
        ridge_png="ridge_mel.png",
        ridge_wav="ridge.wav",
        ridge_direct_png="ridge_direct.png",
    )


if __name__ == "__main__":
    main()
