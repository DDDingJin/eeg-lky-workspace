from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT / "data" / "processed" / "reference_splits"


DATASET_ALIASES = {
    "sample": "hugo_sample_tf64",
    "hugo": "hugo_sample_tf64",
    "weissbart": "weissbart_tf64",
    "etard": "etard_tf64",
    "etard_p00": "etard_tf64_p00_test",
}

MODEL_COMMANDS = {
    "adt_exact": ROOT / "scripts" / "run_adt_exact_reference.py",
    "vlaai_exact": ROOT / "scripts" / "run_vlaai_exact_reference.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hugo_sample_tf64", "weissbart_tf64"],
        help="reference_splits dataset folders or aliases such as sample/weissbart/etard",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["adt_exact", "vlaai_exact"],
        choices=sorted(MODEL_COMMANDS),
    )
    parser.add_argument("--participants", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-length", type=int, default=320)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_dataset(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def pretty_cmd(cmd: list[str]) -> str:
    return " ".join(cmd)


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    print(pretty_cmd(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    for model_name in args.models:
        script_path = MODEL_COMMANDS[model_name]
        model_root = "adt_exact_reference" if model_name == "adt_exact" else "vlaai_exact_reference"
        for dataset_name in args.datasets:
            resolved_name = resolve_dataset(dataset_name)
            input_dir = REFERENCE_ROOT / resolved_name
            if not input_dir.exists():
                raise SystemExit(f"missing input dataset: {input_dir}")

            run_tag = f"{resolved_name}_all_e{args.epochs}" if not args.participants else f"{resolved_name}_{'-'.join(args.participants)}_e{args.epochs}"
            output_dir = ROOT / "experiments" / model_root / run_tag
            cmd = [
                args.python,
                str(script_path),
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
                "--window-length",
                str(args.window_length),
                "--hop-length",
                str(args.hop_length),
                "--learning-rate",
                str(args.learning_rate),
                "--min-lr",
                str(args.min_lr),
                "--device",
                args.device,
            ]
            if args.participants:
                cmd.extend(["--participants", *args.participants])
            run_cmd(cmd, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
