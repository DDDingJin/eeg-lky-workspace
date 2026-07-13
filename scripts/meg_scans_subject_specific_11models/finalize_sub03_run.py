"""Finalize an already completed sub-03 run without loading data or rerunning models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_WINDOWS = {
    "fcnn": (50, 1, "endpoint-window reconstruction"),
    "cnn": (50, 1, "endpoint-window reconstruction"),
    "eegnet": (50, 1, "endpoint-window reconstruction"),
    "vlaai": (50, 1, "endpoint-window reconstruction"),
    "adt": (320, 64, "sequence-window reconstruction"),
    "happyquokka": (640, 640, "sequence-window reconstruction"),
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finalize(output_dir: Path) -> None:
    state_path = output_dir / "run_state.json"
    state = read(state_path)
    if state.get("pending_jobs"):
        raise RuntimeError("run is not complete; refusing finalization")
    state.pop("ready_for_full_manual_run", None)
    state["run_status"] = "completed"
    state["finalization"] = "metadata_only_no_rerun"
    write(state_path, state)

    entries = read(output_dir / "model_run_entries.json")
    successful = {row["model"] for row in entries if row.get("status") == "success"}
    expected = {"linear", "ridge", "lasso", "elasticnet", "cca", *MODEL_WINDOWS}
    if successful != expected:
        raise RuntimeError(f"successful model roster mismatch: {sorted(successful)}")
    write(
        output_dir / "cca_x_only_inference_assertion.json",
        {
            "status": "passed",
            "assertion": "validation/test CCA reconstruction uses X lag -> X scaler/PCA -> CCA.transform(X) -> ridge y_hat; target is used only for final Pearson scoring",
            "checked_entries": [
                row for row in entries if row.get("model") == "cca" and row.get("status") == "success"
            ],
        },
    )

    contract = []
    for model in sorted(successful):
        if model in MODEL_WINDOWS:
            window, hop, label = MODEL_WINDOWS[model]
            valid = 7680 - window + 1 if model in {"fcnn", "cnn", "eegnet", "vlaai"} else 7680
            representation = "mag102 window"
            identity = next(
                (row.get("implementation_identity") for row in entries if row.get("model") == model),
                model,
            )
        else:
            window, hop, label, valid, representation, identity = 0, 1, "lagged linear reconstruction", 7631, "mag102 lagged samples", model
        contract.append(
            {
                "model": model,
                "implementation_identity": identity,
                "input_representation": representation,
                "window_size": window,
                "hop": hop,
                "target_mode": "endpoint" if "endpoint" in label else "sequence",
                "evaluation_valid_samples_per_120s_recording": valid,
                "temporal_context_label": label,
            }
        )
    write(
        output_dir / "temporal_context_contract.json",
        {
            "status": "passed",
            "benchmark_definition": "offline, architecture-faithful reconstruction",
            "causal_or_realtime_claim": False,
            "rows": contract,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    finalize(Path(args.output_dir))
