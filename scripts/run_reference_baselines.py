from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import pickle
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
UPSTREAM = ROOT / "external" / "upstream" / "mldecoders"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from pipeline.dnn import CNN, FCNN
from repro.mldecoders.models import EEGNetRegressor
from repro.reference_baselines import (
    DNNReferenceTrainResult,
    evaluate_dnn_reference_on_test,
    evaluate_reference_cca,
    evaluate_reference_cca_canonical,
    evaluate_reference_backward_trf_ridge,
    evaluate_reference_forward_ridge,
    evaluate_reference_linear_variant,
    evaluate_reference_ridge,
    evaluate_reference_sparse_decoder,
    fit_reference_cca,
    fit_reference_cca_canonical,
    fit_reference_cca_multilag,
    fit_reference_backward_trf_ridge,
    fit_reference_forward_ridge,
    fit_reference_linear_variant,
    fit_reference_ridge,
    fit_reference_sparse_decoder,
    list_reference_subjects,
    train_dnn_reference_logged,
)


DATASET_ALIASES = {
    "weissbart": "weissbart_tf64",
    "etard": "etard_tf64",
    "sample": "hugo_sample_tf64",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="reference_splits dataset folder or alias")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "cca", "fcnn", "cnn", "eegnet"],
        choices=[
            "backward_trf_ridge",
            "forward_ridge",
            "ridge",
            "lasso",
            "elastic_net",
            "avgdec_ridge",
            "avgdec_lasso",
            "avgcorr_ridge",
            "avgcorr_lasso",
            "cca",
            "cca_canonical",
            "cca_multilag",
            "fcnn",
            "cnn",
            "eegnet",
        ],
    )
    parser.add_argument("--participants", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_csv(path: Path, row: dict, fieldnames: list[str]) -> None:
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def resolve_dataset(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def run_linear_variant(dataset_dir: Path, out_dir: Path, participant: str, variant: str) -> None:
    marker = out_dir / f"{participant}_summary.json"
    if marker.exists():
        print(f"{variant} {participant}: skip")
        return
    model, best = fit_reference_linear_variant(dataset_dir, participant, variant, channels=range(64))
    test = evaluate_reference_linear_variant(dataset_dir, participant, variant, model, channels=range(64))
    with open(out_dir / f"{participant}.pkl", "wb") as f:
        pickle.dump(model, f)
    np.save(out_dir / f"{participant}_pred.npy", test["prediction"])
    np.save(out_dir / f"{participant}_target.npy", test["target"])
    payload = {
        "participant": participant,
        "variant": variant,
        "best_alpha": best["best_alpha"],
        "val_pearson": best["val_pearson"],
        "test_pearson": test["pearson"],
    }
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    append_csv(out_dir / "metrics.csv", payload, ["participant", "variant", "best_alpha", "val_pearson", "test_pearson"])
    print(f"{variant} {participant}: {test['pearson']:.4f}")


def run_cca(dataset_dir: Path, out_dir: Path, participant: str) -> None:
    marker = out_dir / f"{participant}_summary.json"
    if marker.exists():
        print(f"cca {participant}: skip")
        return
    model, fit_metrics = fit_reference_cca(dataset_dir, participant, channels=range(64))
    test = evaluate_reference_cca(dataset_dir, participant, model, channels=range(64))
    with open(out_dir / f"{participant}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    np.save(out_dir / f"{participant}_pred.npy", test["prediction"])
    np.save(out_dir / f"{participant}_target.npy", test["target"])
    payload = {
        "participant": participant,
        "n_components": fit_metrics["n_components"],
        "x_pca_components": fit_metrics["x_pca_components"],
        "y_pca_components": fit_metrics["y_pca_components"],
        "recon_alpha": fit_metrics["recon_alpha"],
        "val_recon_corr": fit_metrics["val_recon_corr"],
        "test_canonical_corr": test["canonical_corr"],
        "test_recon_corr": test["recon_corr"],
        "test_match_mismatch_accuracy": test["match_mismatch_accuracy"],
        "test_match_mismatch_margin_mean": test["match_mismatch_margin_mean"],
    }
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    append_csv(
        out_dir / "metrics.csv",
        payload,
        [
            "participant",
            "n_components",
            "x_pca_components",
            "y_pca_components",
            "recon_alpha",
            "val_recon_corr",
            "test_canonical_corr",
            "test_recon_corr",
            "test_match_mismatch_accuracy",
            "test_match_mismatch_margin_mean",
        ],
    )
    print(f"cca {participant}: recon={test['recon_corr']:.4f}")


def run_cca_canonical(dataset_dir: Path, out_dir: Path, participant: str) -> None:
    marker = out_dir / f"{participant}_summary.json"
    if marker.exists():
        print(f"cca_canonical {participant}: skip")
        return
    model, fit_metrics = fit_reference_cca_canonical(dataset_dir, participant, channels=range(64))
    test = evaluate_reference_cca_canonical(dataset_dir, participant, model, channels=range(64))
    with open(out_dir / f"{participant}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    np.save(out_dir / f"{participant}_pred.npy", test["prediction"])
    np.save(out_dir / f"{participant}_target.npy", test["target"])
    payload = {
        "participant": participant,
        "n_components": fit_metrics["n_components"],
        "x_pca_components": fit_metrics["x_pca_components"],
        "y_pca_components": fit_metrics["y_pca_components"],
        "val_canonical_corr": fit_metrics["val_canonical_corr"],
        "test_canonical_corr": test["canonical_corr"],
        "test_recon_corr": test["recon_corr"],
    }
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    append_csv(
        out_dir / "metrics.csv",
        payload,
        [
            "participant",
            "n_components",
            "x_pca_components",
            "y_pca_components",
            "val_canonical_corr",
            "test_canonical_corr",
            "test_recon_corr",
        ],
    )
    print(f"cca_canonical {participant}: canonical={test['canonical_corr']:.4f}")


def run_cca_multilag(dataset_dir: Path, out_dir: Path, participant: str) -> None:
    marker = out_dir / f"{participant}_summary.json"
    if marker.exists():
        print(f"cca_multilag {participant}: skip")
        return
    model, fit_metrics = fit_reference_cca_multilag(dataset_dir, participant, channels=range(64))
    test = evaluate_reference_cca(dataset_dir, participant, model, channels=range(64))
    with open(out_dir / f"{participant}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    np.save(out_dir / f"{participant}_pred.npy", test["prediction"])
    np.save(out_dir / f"{participant}_target.npy", test["target"])
    payload = {
        "participant": participant,
        "start_lag": fit_metrics["start_lag"],
        "end_lag": fit_metrics["end_lag"],
        "n_components": fit_metrics["n_components"],
        "x_pca_components": fit_metrics["x_pca_components"],
        "y_pca_components": fit_metrics["y_pca_components"],
        "recon_alpha": fit_metrics["recon_alpha"],
        "val_recon_corr": fit_metrics["val_recon_corr"],
        "test_canonical_corr": test["canonical_corr"],
        "test_recon_corr": test["recon_corr"],
        "test_match_mismatch_accuracy": test["match_mismatch_accuracy"],
        "test_match_mismatch_margin_mean": test["match_mismatch_margin_mean"],
    }
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    append_csv(
        out_dir / "metrics.csv",
        payload,
        [
            "participant",
            "start_lag",
            "end_lag",
            "n_components",
            "x_pca_components",
            "y_pca_components",
            "recon_alpha",
            "val_recon_corr",
            "test_canonical_corr",
            "test_recon_corr",
            "test_match_mismatch_accuracy",
            "test_match_mismatch_margin_mean",
        ],
    )
    print(f"cca_multilag {participant}: recon={test['recon_corr']:.4f}")


def run_dnn(
    dataset_dir: Path,
    out_dir: Path,
    participant: str,
    model_name: str,
    model_handle,
    model_kwargs: dict,
    train_kwargs: dict,
    *,
    device: str,
    epochs: int,
    patience: int,
    seed: int,
) -> None:
    marker = out_dir / f"{participant}_history.json"
    if marker.exists():
        print(f"{model_name} {participant}: skip")
        return
    result: DNNReferenceTrainResult = train_dnn_reference_logged(
        dataset_dir,
        participant,
        model_handle,
        model_kwargs,
        epochs=epochs,
        lr=train_kwargs["lr"],
        weight_decay=train_kwargs["weight_decay"],
        batch_size=train_kwargs["batch_size"],
        early_stopping_patience=patience,
        device=device,
        seed=seed,
        channels=range(model_kwargs["num_input_channels"]),
    )
    predictions, targets, test_score = evaluate_dnn_reference_on_test(
        dataset_dir,
        participant,
        model_handle,
        model_kwargs,
        result.state_dict,
        device=device,
        channels=range(model_kwargs["num_input_channels"]),
    )
    torch.save(result.state_dict, out_dir / f"{participant}_state_dict.pt")
    np.save(out_dir / f"{participant}_predictions.npy", predictions)
    np.save(out_dir / f"{participant}_targets.npy", targets)
    payload = {
        "participant": participant,
        "best_epoch": result.best_epoch,
        "epochs_completed": result.epochs_completed,
        "best_val_score": result.best_val_score,
        "test_pearson": test_score,
        "epochs_requested": epochs,
        "device": device,
    }
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(
            {
                **payload,
                "val_history": result.val_history,
            },
            f,
            indent=2,
        )
    append_csv(
        out_dir / "metrics.csv",
        payload,
        ["participant", "best_epoch", "epochs_completed", "best_val_score", "test_pearson", "epochs_requested", "device"],
    )
    print(f"{model_name} {participant}: {test_score:.4f}")


def main() -> int:
    args = parse_args()
    dataset_id = resolve_dataset(args.dataset)
    dataset_dir = ROOT / "data" / "processed" / "reference_splits" / dataset_id
    if not dataset_dir.exists():
        raise SystemExit(f"missing dataset_dir: {dataset_dir}")
    participants = args.participants or list_reference_subjects(dataset_dir)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    root_out = ROOT / "experiments" / "reference_baselines" / dataset_id
    ensure_dir(root_out)

    for model_name in args.models:
        out_dir = root_out / model_name
        ensure_dir(out_dir)
        if model_name in {"avgdec_ridge", "avgdec_lasso", "avgcorr_ridge", "avgcorr_lasso", "ridge", "forward_ridge", "backward_trf_ridge", "lasso", "elastic_net"}:
            for participant in participants:
                if model_name == "ridge":
                    marker = out_dir / f"{participant}_summary.json"
                    if marker.exists():
                        print(f"ridge {participant}: skip")
                        continue
                    model, best = fit_reference_ridge(dataset_dir, participant, channels=range(64))
                    test = evaluate_reference_ridge(dataset_dir, participant, model, channels=range(64))
                    with open(out_dir / f"{participant}.pkl", "wb") as f:
                        pickle.dump(model, f)
                    np.save(out_dir / f"{participant}_pred.npy", test["prediction"])
                    np.save(out_dir / f"{participant}_target.npy", test["target"])
                    payload = {
                        "participant": participant,
                        "variant": "ridge",
                        "best_alpha": best["best_alpha"],
                        "val_pearson": best["val_pearson"],
                        "test_pearson": test["pearson"],
                    }
                    with open(marker, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                    append_csv(out_dir / "metrics.csv", payload, ["participant", "variant", "best_alpha", "val_pearson", "test_pearson"])
                    print(f"ridge {participant}: {test['pearson']:.4f}")
                elif model_name == "backward_trf_ridge":
                    marker = out_dir / f"{participant}_summary.json"
                    if marker.exists():
                        print(f"backward_trf_ridge {participant}: skip")
                        continue
                    model, best = fit_reference_backward_trf_ridge(dataset_dir, participant, channels=range(64))
                    test = evaluate_reference_backward_trf_ridge(dataset_dir, participant, model, channels=range(64))
                    with open(out_dir / f"{participant}.pkl", "wb") as f:
                        pickle.dump(model, f)
                    np.save(out_dir / f"{participant}_pred.npy", test["prediction"])
                    np.save(out_dir / f"{participant}_target.npy", test["target"])
                    payload = {
                        "participant": participant,
                        "variant": "backward_trf_ridge",
                        "best_alpha": best["best_alpha"],
                        "start_lag": best["start_lag"],
                        "end_lag": best["end_lag"],
                        "val_pearson": best["val_pearson"],
                        "test_pearson": test["pearson"],
                    }
                    with open(marker, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                    append_csv(
                        out_dir / "metrics.csv",
                        payload,
                        ["participant", "variant", "best_alpha", "start_lag", "end_lag", "val_pearson", "test_pearson"],
                    )
                    print(f"backward_trf_ridge {participant}: {test['pearson']:.4f}")
                elif model_name == "forward_ridge":
                    marker = out_dir / f"{participant}_summary.json"
                    if marker.exists():
                        print(f"forward_ridge {participant}: skip")
                        continue
                    model, best = fit_reference_forward_ridge(dataset_dir, participant, channels=range(64))
                    test = evaluate_reference_forward_ridge(dataset_dir, participant, model, channels=range(64))
                    with open(out_dir / f"{participant}.pkl", "wb") as f:
                        pickle.dump(model, f)
                    np.save(out_dir / f"{participant}_pred.npy", test["prediction"])
                    np.save(out_dir / f"{participant}_target.npy", test["target"])
                    np.save(out_dir / f"{participant}_channel_corrs.npy", test["channel_corrs"])
                    payload = {
                        "participant": participant,
                        "variant": "forward_ridge",
                        "best_alpha": best["best_alpha"],
                        "val_mean_channel_corr": best["val_mean_channel_corr"],
                        "test_mean_channel_corr": test["mean_channel_corr"],
                    }
                    with open(marker, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                    append_csv(out_dir / "metrics.csv", payload, ["participant", "variant", "best_alpha", "val_mean_channel_corr", "test_mean_channel_corr"])
                    print(f"forward_ridge {participant}: {test['mean_channel_corr']:.4f}")
                elif model_name in {"lasso", "elastic_net"}:
                    marker = out_dir / f"{participant}_summary.json"
                    if marker.exists():
                        print(f"{model_name} {participant}: skip")
                        continue
                    model, best = fit_reference_sparse_decoder(dataset_dir, participant, model_name, channels=range(64))
                    test = evaluate_reference_sparse_decoder(dataset_dir, participant, model, channels=range(64))
                    with open(out_dir / f"{participant}.pkl", "wb") as f:
                        pickle.dump(model, f)
                    np.save(out_dir / f"{participant}_pred.npy", test["prediction"])
                    np.save(out_dir / f"{participant}_target.npy", test["target"])
                    payload = {
                        "participant": participant,
                        "variant": model_name,
                        "best_alpha": best["best_alpha"],
                        "start_lag": best["start_lag"],
                        "end_lag": best["end_lag"],
                        "val_pearson": best["val_pearson"],
                        "test_pearson": test["pearson"],
                    }
                    if "best_l1_ratio" in best:
                        payload["best_l1_ratio"] = best["best_l1_ratio"]
                    with open(marker, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                    fieldnames = ["participant", "variant", "best_alpha", "start_lag", "end_lag", "val_pearson", "test_pearson"]
                    if "best_l1_ratio" in payload:
                        fieldnames.insert(3, "best_l1_ratio")
                    append_csv(out_dir / "metrics.csv", payload, fieldnames)
                    print(f"{model_name} {participant}: {test['pearson']:.4f}")
                else:
                    run_linear_variant(dataset_dir, out_dir, participant, model_name)
        elif model_name == "cca":
            for participant in participants:
                run_cca(dataset_dir, out_dir, participant)
        elif model_name == "cca_canonical":
            for participant in participants:
                run_cca_canonical(dataset_dir, out_dir, participant)
        elif model_name == "cca_multilag":
            for participant in participants:
                run_cca_multilag(dataset_dir, out_dir, participant)
        elif model_name == "fcnn":
            for participant in participants:
                run_dnn(
                    dataset_dir,
                    out_dir,
                    participant,
                    model_name,
                    FCNN,
                    {"num_hidden": 3, "dropout_rate": 0.45, "num_input_channels": 64},
                    {"lr": 1e-4, "batch_size": 256, "weight_decay": 1e-4},
                    device=device,
                    epochs=args.epochs,
                    patience=args.patience,
                    seed=args.seed,
                )
        elif model_name == "cnn":
            for participant in participants:
                run_dnn(
                    dataset_dir,
                    out_dir,
                    participant,
                    model_name,
                    CNN,
                    {"dropout_rate": 0.20, "F1": 8, "D": 8, "F2": 64, "num_input_channels": 64},
                    {"lr": 1e-2, "batch_size": 256, "weight_decay": 1e-8},
                    device=device,
                    epochs=args.epochs,
                    patience=args.patience,
                    seed=args.seed,
                )
        elif model_name == "eegnet":
            for participant in participants:
                run_dnn(
                    dataset_dir,
                    out_dir,
                    participant,
                    model_name,
                    EEGNetRegressor,
                    {"num_input_channels": 64, "input_length": 50, "temporal_filters": 8, "depth_multiplier": 2, "separable_filters": 16, "dropout_rate": 0.25},
                    {"lr": 1e-3, "batch_size": 256, "weight_decay": 1e-4},
                    device=device,
                    epochs=args.epochs,
                    patience=args.patience,
                    seed=args.seed,
                )
        else:
            raise SystemExit(f"unknown model: {model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
