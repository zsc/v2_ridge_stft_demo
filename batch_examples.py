import argparse
import glob
import os
import subprocess
import sys

from report import write_gallery_html


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-run main.py on a glob of wavs and build a gallery HTML.")
    parser.add_argument("--input_glob", required=True, help="Glob for input wav files")
    parser.add_argument("--output_dir", required=True, help="Directory for batch outputs and gallery HTML")
    parser.add_argument("--python_exec", default=sys.executable, help="Python executable used to run main.py")
    parser.add_argument(
        "--main_script",
        default=os.path.join(os.path.dirname(__file__), "main.py"),
        help="Path to main.py",
    )
    parser.add_argument("--title", default="Batch Mel Gallery")
    return parser.parse_known_args()


def _has_flag(extra_args, flag):
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in extra_args)


def main():
    args, extra_args = parse_args()
    input_paths = sorted(glob.glob(os.path.expanduser(args.input_glob)))
    if not input_paths:
        raise FileNotFoundError(f"No wav files matched: {args.input_glob}")

    os.makedirs(args.output_dir, exist_ok=True)

    if not _has_flag(extra_args, "--save_intermediates"):
        extra_args = [*extra_args, "--save_intermediates", "False"]

    records = []
    for wav_path in input_paths:
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        item_dir = os.path.join(args.output_dir, stem)
        cmd = [
            args.python_exec,
            args.main_script,
            "--input",
            wav_path,
            "--output_dir",
            item_dir,
            *extra_args,
        ]
        print("[batch]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        records.append(
            {
                "name": stem,
                "slug": stem,
                "report_html": f"{stem}/report.html",
                "gt_wav": f"{stem}/gt.wav",
                "ridge_wav": f"{stem}/ridge.wav",
                "recon_wav": f"{stem}/recon.wav",
                "gt_png": f"{stem}/gt_mel.png",
                "ridge_direct_png": f"{stem}/ridge_direct.png",
                "gaussian_direct_png": f"{stem}/gaussian_direct.png",
                "fit_sum_direct_png": f"{stem}/fit_sum_direct.png",
            }
        )

    gallery_path = os.path.join(args.output_dir, "index.html")
    write_gallery_html(gallery_path, records, title=args.title)
    print(f"[batch] wrote gallery: {gallery_path}")


if __name__ == "__main__":
    main()
