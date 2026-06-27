from __future__ import annotations

from pathlib import Path
import argparse
import json
import statistics
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repro.adt_exact import train_adt_exact_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "reference_splits" / "hugo_sample_tf64",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "adt_exact_reference" / "hugo_sample_tf64",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-length", type=int, default=320)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--participants", nargs="+")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    _, summary = train_adt_exact_reference(
        args.input_dir,
        participants=args.participants,
        output_dir=args.output_dir,
        seq_len=args.window_length,
        hop_length=args.hop_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        device=device,
        seed=args.seed,
    )

    metrics = {subject: {"loss": item.loss, "pearson_metric": item.pearson_metric} for subject, item in summary.subject_metrics.items()}
    pearsons = [item["pearson_metric"] for item in metrics.values()]
    result = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "device": device,
        "metric_name": "pearson_metric",
        "epochs_requested": args.epochs,
        "window_length": args.window_length,
        "hop_length": args.hop_length,
        "batch_size": args.batch_size,
        "best_epoch": summary.best_epoch,
        "best_val_loss": summary.best_val_loss,
        "best_val_pearson_metric": summary.best_val_metric,
        "mean_test_metric": statistics.mean(pearsons) if pearsons else None,
        "std_test_metric": statistics.pstdev(pearsons) if len(pearsons) > 1 else 0.0 if pearsons else None,
        "history": summary.history,
        "subjects": metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ["device", "epochs_requested", "best_epoch", "best_val_pearson_metric", "mean_test_metric", "std_test_metric"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
