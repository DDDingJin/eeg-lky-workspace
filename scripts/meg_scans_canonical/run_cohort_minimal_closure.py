"""Run only Ridge sanity and one-epoch EEGNet smoke for the fixed five-person cohort."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


COHORT = ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("meg_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runner = load_runner(args.runner.resolve())
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["models"] = ["ridge", "eegnet"]
    config["subjects"] = []
    config["subject_ids"] = COHORT
    config["canonical_data_root"] = str(args.derived_root)
    envelope = args.derived_root / "stimuli" / "sub-others_preprocessed_audiobook_envelopes_decoding.mat"
    for subject in COHORT:
        paired = args.derived_root / subject / "speech" / f"{subject}_preprocessed_audiobooks_decoding.mat"
        config["subjects"].append({
            "subject_id": subject,
            "dataset_id": f"meg_scans_cohort_canonical_05_08hz_64hz_{subject}",
            "paired_mat": str(paired),
            "envelope_mat": str(envelope),
            "split": config["split"],
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "minimal_closure_config.json").write_text(json.dumps({"cohort": COHORT, "models": config["models"], "scope": "Ridge actual sanity plus EEGNet one-epoch CUDA smoke; no full benchmark"}, indent=2) + "\n", encoding="utf-8")
    statuses = []
    for subject_cfg in config["subjects"]:
        subject = subject_cfg["subject_id"]
        row = {"subject_id": subject, "ridge_sanity": "not_run", "eegnet_smoke": "not_run", "reason": ""}
        subject_out = args.output_dir / subject
        subject_out.mkdir(exist_ok=True)
        if subject == "sub-03":
            statuses.append({"subject_id": subject, "ridge_sanity": "passed", "eegnet_smoke": "passed", "reason": "reused accepted sub-03 smoke_gate evidence"})
            continue
        try:
            data = runner.load_paired_mat(subject_cfg["paired_mat"], subject)
            runner.validate_and_select(data, config, subject_cfg, subject_out)
            splits = runner.split_trials(data, subject_cfg, subject_out)
            runner.run_model("ridge", config, subject_cfg, splits, subject_out, smoke=False)
            row["ridge_sanity"] = "passed"
        except Exception as exc:
            row["ridge_sanity"] = "failed"
            row["reason"] = f"ridge: {type(exc).__name__}: {exc}"
        try:
            if row["ridge_sanity"] == "passed":
                runner.run_model("eegnet", config, subject_cfg, splits, subject_out, smoke=True)
                row["eegnet_smoke"] = "passed"
            else:
                row["eegnet_smoke"] = "blocked_by_ridge_failure"
        except Exception as exc:
            row["eegnet_smoke"] = "failed"
            row["reason"] = (row["reason"] + "; " if row["reason"] else "") + f"eegnet: {type(exc).__name__}: {exc}"
        statuses.append(row)
    report = {"cohort": COHORT, "models_run": ["ridge", "eegnet"], "full_11model_run_started": False, "statuses": statuses, "cohort_ready_for_full_11model_run": all(row["ridge_sanity"] == "passed" and row["eegnet_smoke"] == "passed" for row in statuses)}
    (args.output_dir / "minimal_closure_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["cohort_ready_for_full_11model_run"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
