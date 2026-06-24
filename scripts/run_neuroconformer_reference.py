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

from repro.neuroconformer_reference import evaluate_neuroconformer_reference, train_neuroconformer_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--win-len-seconds", type=int, default=10)
    parser.add_argument("--sample-rate", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--g-con", action="store_true")
    parser.add_argument("--windows-per-sample", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-inner", type=int, default=1024)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--conv-kernel-size", type=int, default=31)
    parser.add_argument("--use-sinusoidal-pos", action="store_true")
    parser.add_argument("--no-relative-pos", action="store_true")
    parser.add_argument("--no-macaron-ffn", action="store_true")
    parser.add_argument("--no-gated-residual", action="store_true")
    parser.add_argument("--no-mlp-head", action="store_true")
    parser.add_argument("--gradient-scale", type=float, default=1.0)
    parser.add_argument("--no-skip-cnn", action="store_true")
    parser.add_argument("--no-se", action="store_true")
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
    model, summary = train_neuroconformer_reference(
        input_dir,
        win_len_seconds=args.win_len_seconds,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        g_con=args.g_con,
        device=device,
        channels=range(64),
        windows_per_sample=args.windows_per_sample,
        d_model=args.d_model,
        d_inner=args.d_inner,
        n_head=args.n_head,
        n_layers=args.n_layers,
        conv_kernel_size=args.conv_kernel_size,
        use_relative_pos=not args.no_relative_pos,
        use_macaron_ffn=not args.no_macaron_ffn,
        use_sinusoidal_pos=args.use_sinusoidal_pos,
        use_gated_residual=not args.no_gated_residual,
        use_mlp_head=not args.no_mlp_head,
        gradient_scale=args.gradient_scale,
        skip_cnn=not args.no_skip_cnn,
        use_se=not args.no_se,
    )
    result = evaluate_neuroconformer_reference(
        input_dir,
        state_dict=summary.state_dict,
        subject_to_id=summary.subject_to_id,
        config=summary.config,
        win_len_seconds=args.win_len_seconds,
        sample_rate=args.sample_rate,
        dropout=args.dropout,
        g_con=args.g_con,
        device=device,
        channels=range(64),
    )
    torch.save(summary.state_dict, output_dir / "neuroconformer_state_dict.pt")
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
                "g_con": args.g_con,
                "windows_per_sample": args.windows_per_sample,
                "best_epoch": summary.best_epoch,
                "best_val_metric": summary.best_val_metric,
                "epochs_completed": summary.epochs_completed,
                "history": summary.history,
                "config": summary.config,
                **result,
            },
            f,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
