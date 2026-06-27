from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repro.happyquokka_reference import evaluate_happyquokka_reference, train_happyquokka_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--win-len-seconds", type=int, default=10)
    parser.add_argument("--sample-rate", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lamda", type=float, default=0.2)
    parser.add_argument("--g-con", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    model, summary = train_happyquokka_reference(
        input_dir,
        win_len_seconds=args.win_len_seconds,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        lamda=args.lamda,
        g_con=args.g_con,
        device=device,
        channels=range(64),
    )
    result = evaluate_happyquokka_reference(
        input_dir,
        state_dict=summary.state_dict,
        subject_to_id=summary.subject_to_id,
        win_len_seconds=args.win_len_seconds,
        sample_rate=args.sample_rate,
        dropout=args.dropout,
        g_con=args.g_con,
        device=device,
        channels=range(64),
    )
    torch.save(summary.state_dict, output_dir / "happyquokka_state_dict.pt")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "device": device,
                "epochs_requested": args.epochs,
                "batch_size": args.batch_size,
                "win_len_seconds": args.win_len_seconds,
                "sample_rate": args.sample_rate,
                "learning_rate": args.learning_rate,
                "dropout": args.dropout,
                "lamda": args.lamda,
                "g_con": args.g_con,
                "best_epoch": summary.best_epoch,
                "best_val_metric": summary.best_val_metric,
                "epochs_completed": summary.epochs_completed,
                "history": summary.history,
                **result,
            },
            f,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
