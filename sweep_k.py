import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep multiple k values and build per-k batch galleries.")
    parser.add_argument("--input_glob", required=True, help="Glob for input wav files")
    parser.add_argument("--output_dir", required=True, help="Top-level directory for sweep outputs")
    parser.add_argument("--k_values", default="1,2,3,4,5", help="Comma-separated k values")
    parser.add_argument("--python_exec", default=sys.executable, help="Python executable")
    parser.add_argument(
        "--batch_script",
        default=os.path.join(os.path.dirname(__file__), "batch_examples.py"),
        help="Path to batch_examples.py",
    )
    parser.add_argument("--title_prefix", default="Step-Audio-EditX k sweep")
    return parser.parse_known_args()


def _write_index_html(path, entries, title):
    links = "".join(
        f'<li><a href="{entry["subdir"]}/index.html">k={entry["k"]}</a> <span>{entry["title"]}</span></li>'
        for entry in entries
    )
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f1f1f; }}
a {{ color: #0b57d0; }}
li {{ margin: 10px 0; }}
</style>
</head>
<body>
<h1>{title}</h1>
<ul>
{links}
</ul>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    args, extra_args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    entries = []
    for k in k_values:
        subdir = f"k_{k}"
        title = f"{args.title_prefix} (k={k})"
        cmd = [
            args.python_exec,
            args.batch_script,
            "--input_glob",
            args.input_glob,
            "--output_dir",
            os.path.join(args.output_dir, subdir),
            "--title",
            title,
            "--k",
            str(k),
            *extra_args,
        ]
        print("[sweep]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        entries.append({"k": k, "subdir": subdir, "title": title})

    _write_index_html(os.path.join(args.output_dir, "index.html"), entries, args.title_prefix)
    print(f"[sweep] wrote index: {os.path.join(args.output_dir, 'index.html')}")


if __name__ == "__main__":
    main()
