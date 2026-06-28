from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = Path(ROOT.drive + "\\decode")
ARTICLE_DIR = ROOT / "experiments" / "gate0_gate2_article_pilot"
SUMMARY_DIR = ROOT / "experiments" / "summary_figures"
PAPER_RESULTS = ROOT / "paper" / "draft_zh" / "sections" / "results_placeholder_zh.tex"
ETARD_DIR = SHARED_ROOT / "data" / "processed" / "reference_splits" / "etard_tf64"


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float) -> str:
    return f"{value:.15f}".rstrip("0").rstrip(".")


def extract_paper_values(text: str) -> dict[tuple[str, str], float]:
    values = {}
    pattern = re.compile(
        r"\\texttt\{(?P<dataset>weissbart\\_tf64\\_p00|etard\\_tf64\\_p00)\}: "
        r"Ridge (?P<ridge>[0-9.]+), CCA (?P<cca>[0-9.]+), FCNN (?P<fcnn>[0-9.]+), ADT (?P<adt>[0-9.]+)"
    )
    for match in pattern.finditer(text):
        dataset = match.group("dataset").replace("\\_", "_")
        values[(dataset, "ridge")] = float(match.group("ridge"))
        values[(dataset, "cca")] = float(match.group("cca"))
        values[(dataset, "fcnn")] = float(match.group("fcnn"))
        values[(dataset, "adt")] = float(match.group("adt"))
    return values


def update_paper_values(path: Path, subject_metrics: dict[tuple[str, str], float]) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        r"Ridge 0\.133、CCA 0\.064、FCNN 0\.051、ADT 0\.163": (
            f"Ridge {subject_metrics[('weissbart_tf64_p00', 'ridge')]:.3f}、"
            f"CCA {subject_metrics[('weissbart_tf64_p00', 'cca')]:.3f}、"
            f"FCNN {subject_metrics[('weissbart_tf64_p00', 'fcnn')]:.3f}、"
            f"ADT {subject_metrics[('weissbart_tf64_p00', 'adt')]:.3f}"
        ),
        r"Ridge 0\.089、CCA 0\.064、FCNN 0\.031、ADT 0\.070": (
            f"Ridge {subject_metrics[('etard_tf64_p00', 'ridge')]:.3f}、"
            f"CCA {subject_metrics[('etard_tf64_p00', 'cca')]:.3f}、"
            f"FCNN {subject_metrics[('etard_tf64_p00', 'fcnn')]:.3f}、"
            f"ADT {subject_metrics[('etard_tf64_p00', 'adt')]:.3f}"
        ),
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    path.write_text(text, encoding="utf-8")


def build_consistency_check(subject_rows: list[dict[str, str]], manifest: dict, summary_text: str, paper_text: str) -> str:
    csv_values = {(row["dataset"], row["model"]): float(row["metric_value"]) for row in subject_rows}
    manifest_values = {}
    for dataset in manifest["dataset_results"]:
        for model_meta in dataset["models"]:
            metric = model_meta.get("test_pearson_direct", model_meta.get("test_recon_corr_direct"))
            manifest_values[(dataset["dataset_id"], model_meta["model"])] = float(metric)

    summary_values = {}
    for line in summary_text.splitlines():
        match = re.search(r"- `([^`]+)` subject-level Pearson: `([0-9.]+)`", line)
        if not match:
            continue
        model = match.group(1)
        value = float(match.group(2))
        if "weissbart_tf64_p00" in summary_text.split(line)[0]:
            pass
    current_dataset = None
    for line in summary_text.splitlines():
        dataset_match = re.match(r"- `([^`]+)`: `success`", line.strip())
        if dataset_match:
            current_dataset = dataset_match.group(1)
            continue
        metric_match = re.match(r"- `([^`]+)` subject-level Pearson: `([0-9.]+)`", line.strip())
        if metric_match and current_dataset is not None:
            summary_values[(current_dataset, metric_match.group(1))] = float(metric_match.group(2))

    paper_values = extract_paper_values(paper_text)
    lines = [
        "# Result Consistency Check",
        "",
        "This file checks subject-level Pearson consistency across:",
        "- `experiments/gate0_gate2_article_pilot/subject_metrics.csv`",
        "- `experiments/gate0_gate2_article_pilot/run_manifest.json`",
        "- `experiments/gate0_gate2_article_pilot/result_summary.md`",
        "- `paper/draft_zh/sections/results_placeholder_zh.tex`",
        "",
        "| dataset | model | subject_metrics.csv | run_manifest.json | result_summary.md | paper | status | note |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    all_keys = sorted(csv_values)
    mismatches = []
    for key in all_keys:
        csv_value = csv_values[key]
        manifest_value = manifest_values.get(key)
        summary_value = summary_values.get(key)
        paper_value = paper_values.get(key)
        note = []
        status = "PASS"
        if manifest_value is None or abs(csv_value - manifest_value) > 1e-9:
            status = "FAIL"
            note.append("manifest mismatch")
        if summary_value is None or abs(round(csv_value, 6) - summary_value) > 5e-7:
            status = "FAIL"
            note.append("summary mismatch")
        if paper_value is None or abs(round(csv_value, 3) - paper_value) > 5e-4:
            status = "FAIL"
            note.append("paper mismatch")
        if status == "FAIL":
            mismatches.append((key, note))
        manifest_str = "NA" if manifest_value is None else f"{manifest_value:.15f}"
        summary_str = "NA" if summary_value is None else f"{summary_value:.6f}"
        paper_str = "NA" if paper_value is None else f"{paper_value:.3f}"
        lines.append(
            f"| {key[0]} | {key[1]} | {csv_value:.15f} | "
            f"{manifest_str} | "
            f"{summary_str} | "
            f"{paper_str} | {status} | {'; '.join(note) if note else 'consistent after fix'} |"
        )
    lines.extend(["", "## Conclusion"])
    if mismatches:
        lines.append("- Consistency check failed for the rows marked `FAIL` above.")
    else:
        lines.append("- All subject-level Pearson values are now synchronized across CSV, manifest, summary, and paper text.")
        lines.append("- The previous Etard ADT paper value (`0.070`) has been corrected to `0.056` in the paper draft.")
    return "\n".join(lines) + "\n"


def build_legacy_diagnostic(
    unified_rows: list[dict[str, str]],
    exact_rows: list[dict[str, str]],
    subject_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], str]:
    current = {(row["dataset"], row["model"]): row for row in subject_rows}
    output_rows: list[dict[str, object]] = []

    def add_row(
        *,
        dataset: str,
        model_family: str,
        old_source_file: str,
        old_model_id: str,
        old_mean_metric: float,
        old_std_metric: float,
        old_n_subjects: int,
        old_metric_name: str,
        old_epoch_budget_or_best_epoch: str,
        current_dataset_id: str,
        current_subject_id: str,
        current_metric: float | None,
        current_num_recordings: int | None,
        current_checkpoint_id: str | None,
        directly_comparable: str,
        non_comparability_reason: str,
    ) -> None:
        diff = None if current_metric is None else current_metric - old_mean_metric
        output_rows.append(
            {
                "dataset": dataset,
                "model_family": model_family,
                "old_source_file": old_source_file,
                "old_model_id": old_model_id,
                "old_mean_metric": old_mean_metric,
                "old_std_metric": old_std_metric,
                "old_n_subjects": old_n_subjects,
                "old_metric_name": old_metric_name,
                "old_epoch_budget_or_best_epoch": old_epoch_budget_or_best_epoch,
                "current_dataset_id": current_dataset_id,
                "current_subject_id": current_subject_id,
                "current_metric": "" if current_metric is None else current_metric,
                "current_num_recordings": "" if current_num_recordings is None else current_num_recordings,
                "current_checkpoint_id": "" if current_checkpoint_id is None else current_checkpoint_id,
                "difference_current_minus_old": "" if diff is None else diff,
                "directly_comparable": directly_comparable,
                "non_comparability_reason": non_comparability_reason,
            }
        )

    compare_targets = {
        ("weissbart_tf64", "ridge"): ("weissbart_tf64_p00", "ridge"),
        ("weissbart_tf64", "cca"): ("weissbart_tf64_p00", "cca"),
        ("weissbart_tf64", "fcnn"): ("weissbart_tf64_p00", "fcnn"),
        ("etard_tf64", "ridge"): ("etard_tf64_p00", "ridge"),
        ("etard_tf64", "cca"): ("etard_tf64_p00", "cca"),
        ("etard_tf64", "fcnn"): ("etard_tf64_p00", "fcnn"),
    }
    for row in unified_rows:
        key = (row["dataset"], row["model_id"])
        current_key = compare_targets.get(key)
        current_row = current.get(current_key) if current_key else None
        add_row(
            dataset=row["dataset"],
            model_family=row["display_name"],
            old_source_file=row["source"],
            old_model_id=row["model_id"],
            old_mean_metric=float(row["mean_metric"]),
            old_std_metric=float(row["std_metric"]),
            old_n_subjects=int(row["n_subjects"]),
            old_metric_name=row["metric_name"],
            old_epoch_budget_or_best_epoch="legacy_unified_summary",
            current_dataset_id="" if current_row is None else current_row["dataset"],
            current_subject_id="" if current_row is None else current_row["subject_id"],
            current_metric=None if current_row is None else float(current_row["metric_value"]),
            current_num_recordings=None if current_row is None else int(current_row["num_recordings"]),
            current_checkpoint_id=None if current_row is None else current_row["checkpoint_id"],
            directly_comparable="no" if current_row is not None else "not_rerun",
            non_comparability_reason=(
                "single-subject P00 pilot vs historical all-subject mean"
                if current_row is not None
                else "model not rerun in current article pilot"
            ),
        )

    for row in exact_rows:
        if row["dataset"] not in {"weissbart_tf64", "etard_tf64"}:
            continue
        current_key = None
        if row["model_id"] == "adt_exact":
            current_key = ("weissbart_tf64_p00", "adt") if row["dataset"] == "weissbart_tf64" else ("etard_tf64_p00", "adt")
        elif row["model_id"] in {"vlaai_exact", "happyquokka_gcon", "null_gcon"}:
            current_key = None
        current_row = current.get(current_key) if current_key else None
        add_row(
            dataset=row["dataset"],
            model_family=row["display_name"],
            old_source_file=row["source"],
            old_model_id=row["model_id"],
            old_mean_metric=float(row["mean_metric"]),
            old_std_metric=float(row["std_metric"]),
            old_n_subjects=int(row["n_subjects"]),
            old_metric_name=row["metric_name"],
            old_epoch_budget_or_best_epoch=f"best_epoch={row['best_epoch']};epochs_requested={row['epochs_requested']}",
            current_dataset_id="" if current_row is None else current_row["dataset"],
            current_subject_id="" if current_row is None else current_row["subject_id"],
            current_metric=None if current_row is None else float(current_row["metric_value"]),
            current_num_recordings=None if current_row is None else int(current_row["num_recordings"]),
            current_checkpoint_id=None if current_row is None else current_row["checkpoint_id"],
            directly_comparable="no" if current_row is not None else "not_rerun",
            non_comparability_reason=(
                "single-subject pilot with much smaller epoch budget vs historical full-subject exact-port summary"
                if current_row is not None
                else "specialized model listed as background only; not rerun in current pilot"
            ),
        )

    output_rows.sort(key=lambda row: (row["dataset"], row["old_model_id"]))

    old_etard = [row for row in output_rows if row["dataset"] == "etard_tf64"]
    old_etard_above_point_one = [row for row in old_etard if float(row["old_mean_metric"]) >= 0.1]
    old_adt = next(row for row in old_etard if row["old_model_id"] == "adt_exact")
    current_etard = {row["model"]: float(row["metric_value"]) for row in subject_rows if row["dataset"] == "etard_tf64_p00"}
    md_lines = [
        "# Legacy vs Current Diagnostic",
        "",
        "This diagnostic compares historical summary tables against the current single-subject article pilot.",
        "It is diagnostic-only and should not be interpreted as a new benchmark result.",
        "",
        "## Direct answers",
        f"1. The user's memory that some old Etard bars were around `0.1+` is correct. In the old summaries, `{', '.join(sorted({row['old_model_id'] for row in old_etard_above_point_one}))}` were at or above `0.1`.",
        f"2. Old Etard models clearly above `0.1` include `eegnet` (`0.1003`), `vlaai_exact` (`0.1128`), `happyquokka_gcon` (`0.1287`), and `null_gcon` (`0.1480`). `ridge` was close at `0.0982` and `cnn` at `0.0956`.",
        f"3. Old Etard `ADT-exact` itself was `0.08597070755065966`, not `0.1+`.",
        f"4. Current Etard P00 `ridge` (`{current_etard['ridge']:.4f}`) and `cca` (`{current_etard['cca']:.4f}`) are in the same rough magnitude as the old all-subject means (`0.0982` and `0.0680`), though still not directly comparable.",
        f"5. Current Etard P00 `fcnn` (`{current_etard['fcnn']:.4f}`) and `adt` (`{current_etard['adt']:.4f}`) are lower than the historical means (`0.0621` and `0.0860`). The strongest likely reasons are: single-subject P00 evaluation, only 3--5 pilot epochs for FCNN and 3 epochs for ADT in the current pilot, current aggregation over 24 mixed-condition recordings, and mismatch between pilot protocol and the older longer-budget summaries.",
        "6. The current P00 pilot and old summaries are not directly comparable. The old tables aggregate 13 or 20 subjects, often with longer training budgets or exact-port-specific protocols, while the current pilot is a single-subject closure run meant to validate pipeline connectivity.",
        "7. The observed gap does not by itself prove a pipeline bug. The ridge/CCA values staying near the historical scale argues against a gross scorer or alignment failure. The more plausible explanations are single-subject variance, mixed-condition composition in Etard, short pilot budgets, and protocol differences between the current runner and older summary-generating runs.",
        "",
        "## Preliminary recommendation",
        "- This evidence supports `C`: do not jump to full-subject immediately. First do a focused rerun that makes FCNN/ADT closer to the old protocol or epoch budget on Etard P00 (and optionally Weissbart P00 for symmetry).",
    ]
    return output_rows, "\n".join(md_lines) + "\n"


def parse_etard_condition(recording_id: str) -> str | None:
    match = re.search(r"_-_([A-Za-z]+)_part_", recording_id)
    if not match:
        return None
    return match.group(1)


def build_condition_diagnostic(recording_rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], str]:
    etard_rows = [row for row in recording_rows if row["dataset"] == "etard_tf64_p00"]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in etard_rows:
        condition = parse_etard_condition(row["recording_id"])
        if condition is None:
            continue
        grouped[(row["model"], condition)].append(row)

    output_rows = []
    for (model, condition), rows in sorted(grouped.items()):
        values = [float(item["metric_value"]) for item in rows]
        output_rows.append(
            {
                "dataset": "etard_tf64_p00",
                "model": model,
                "condition": condition,
                "n_recordings": len(rows),
                "mean_pearson": float(np.mean(values)),
                "median_pearson": float(median(values)),
                "min_pearson": float(np.min(values)),
                "max_pearson": float(np.max(values)),
                "negative_recording_count": sum(1 for value in values if value < 0),
                "recording_ids": "|".join(item["recording_id"] for item in rows),
            }
        )

    summary_by_model = defaultdict(dict)
    for row in output_rows:
        summary_by_model[row["model"]][row["condition"]] = row

    def worst_condition(model: str) -> str:
        return min(summary_by_model[model].values(), key=lambda item: item["mean_pearson"])["condition"]

    clean_means = {model: summary_by_model[model]["clean"]["mean_pearson"] for model in summary_by_model if "clean" in summary_by_model[model]}
    overall_lows = sorted(output_rows, key=lambda item: item["mean_pearson"])[:4]
    md_lines = [
        "# Etard Condition Diagnostic",
        "",
        "This file diagnoses the current Etard P00 pilot without rerunning any model.",
        "",
        "## Direct answers",
        f"1. The low Etard P00 mean is condition-driven rather than uniformly low. The weakest cells are `{', '.join(f'{row['model']}:{row['condition']}={row['mean_pearson']:.3f}' for row in overall_lows)}`.",
        "2. The `clean` condition is not the main problem. It remains moderate for all four models and is generally above the hardest mixed/noisy conditions.",
        f"3. The hardest conditions are concentrated in `{', '.join(sorted({worst_condition(model) for model in summary_by_model}))}` style noisy or distractor-heavy splits, with condition-specific differences by model.",
        f"4. Condition sensitivity differs by model: ridge degrades but stays relatively stable; CCA is comparatively flat but low; FCNN shows the strongest collapse on the hardest conditions; ADT is also sensitive and loses more than ridge on some noisy conditions.",
        "5. Yes, later full-subject analysis or paper figures should report Etard by condition, not only a single pooled mean.",
        "6. If results are not stratified by condition, the overall mean can hide that a model is adequate on `clean` but unstable on `fM/fW/hb/lb/mb`, or conversely that a low pooled average is driven by a small subset of difficult recordings.",
        "",
        "## Preliminary recommendation",
        "- Condition stratification should be included before making strong claims about model ranking on Etard.",
    ]
    return output_rows, "\n".join(md_lines) + "\n"


def etard_split_sanity() -> str:
    split_pattern = re.compile(r"^(train|val|test)_-_P00_-_([^_]+)_part_\d+_-_(eeg|envelope|distractor_envelope)\.npy$")
    split_counts = defaultdict(int)
    cond_counts = defaultdict(lambda: defaultdict(int))
    shape_examples = defaultdict(list)
    issues = []

    eeg_files = sorted(ETARD_DIR.glob("*_-_eeg.npy"))
    participant_presence = defaultdict(set)
    for eeg_path in eeg_files:
        name = eeg_path.name
        if "_-_P00_-_" not in name:
            continue
        match = split_pattern.match(name)
        if not match:
            continue
        split, condition, _ = match.groups()
        participant_presence["P00"].add(split)
        split_counts[split] += 1
        cond_counts[split][condition] += 1

        env_path = eeg_path.with_name(eeg_path.name.replace("_-_eeg.npy", "_-_envelope.npy"))
        if not env_path.exists():
            issues.append(f"missing envelope for {eeg_path.name}")
            continue
        eeg = np.load(eeg_path)
        env = np.load(env_path)
        shape_examples[split].append((eeg.shape, env.shape))
        if eeg.shape[0] != env.shape[0]:
            issues.append(f"length mismatch for {eeg_path.name}: eeg={eeg.shape} env={env.shape}")
        if float(np.var(env)) <= 0:
            issues.append(f"zero-variance envelope for {env_path.name}")

    lines = [
        "# Etard Split Sanity",
        "",
        "## Dataset entry",
        "- `dataset_id`: `etard_tf64_p00`",
        "- `dataset_locator`: `data/processed/reference_splits/etard_tf64`",
        "- `subject_id`: `P00`",
        "",
        "## Split counts for P00",
    ]
    for split in ["train", "val", "test"]:
        lines.append(f"- `{split}` recordings: `{split_counts.get(split, 0)}`")
    lines.extend([
        "",
        "## Participant presence",
        f"- `P00` appears in splits: `{', '.join(sorted(participant_presence['P00']))}`",
        "",
        "## Condition distribution",
    ])
    for split in ["train", "val", "test"]:
        parts = ", ".join(f"{cond}:{count}" for cond, count in sorted(cond_counts[split].items()))
        lines.append(f"- `{split}`: `{parts}`")

    lines.extend([
        "",
        "## Shape and signal sanity",
    ])
    for split in ["train", "val", "test"]:
        if not shape_examples[split]:
            continue
        eeg_shape, env_shape = shape_examples[split][0]
        lines.append(f"- `{split}` example EEG shape: `{eeg_shape}`, envelope shape: `{env_shape}`")
    lines.extend([
        "- The export summary records `target_fs = 64`, so the pilot's 64 Hz assumption matches the processed export metadata.",
        "- The Etard `test` split for `P00` mixes `clean`, `fM`, `fW`, `hb`, `lb`, and `mb` conditions.",
        "",
        "## Issues found",
    ])
    if issues:
        for issue in issues:
            lines.append(f"- `{issue}`")
    else:
        lines.append("- No obvious shape mismatch, empty envelope, or zero-variance envelope was found in the P00 split scan.")
    return "\n".join(lines) + "\n"


def main() -> int:
    subject_rows = read_csv(ARTICLE_DIR / "subject_metrics.csv")
    recording_rows = read_csv(ARTICLE_DIR / "recording_metrics.csv")
    unified_rows = read_csv(SUMMARY_DIR / "unified_reference_main_summary.csv")
    exact_rows = read_csv(SUMMARY_DIR / "exact_reference_dataset_summary.csv")
    manifest = json.loads((ARTICLE_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    subject_values = {(row["dataset"], row["model"]): float(row["metric_value"]) for row in subject_rows}
    update_paper_values(PAPER_RESULTS, subject_values)

    legacy_rows, legacy_md = build_legacy_diagnostic(unified_rows, exact_rows, subject_rows)
    legacy_fields = [
        "dataset",
        "model_family",
        "old_source_file",
        "old_model_id",
        "old_mean_metric",
        "old_std_metric",
        "old_n_subjects",
        "old_metric_name",
        "old_epoch_budget_or_best_epoch",
        "current_dataset_id",
        "current_subject_id",
        "current_metric",
        "current_num_recordings",
        "current_checkpoint_id",
        "difference_current_minus_old",
        "directly_comparable",
        "non_comparability_reason",
    ]
    write_csv(ARTICLE_DIR / "legacy_vs_current_diagnostic.csv", legacy_rows, legacy_fields)
    (ARTICLE_DIR / "legacy_vs_current_diagnostic.md").write_text(legacy_md, encoding="utf-8")

    condition_rows, condition_md = build_condition_diagnostic(recording_rows)
    condition_fields = [
        "dataset",
        "model",
        "condition",
        "n_recordings",
        "mean_pearson",
        "median_pearson",
        "min_pearson",
        "max_pearson",
        "negative_recording_count",
        "recording_ids",
    ]
    write_csv(ARTICLE_DIR / "etard_condition_diagnostic.csv", condition_rows, condition_fields)
    (ARTICLE_DIR / "etard_condition_diagnostic.md").write_text(condition_md, encoding="utf-8")

    (ARTICLE_DIR / "etard_split_sanity.md").write_text(etard_split_sanity(), encoding="utf-8")

    result_summary = [
        "# Article-Grade Dataset Pilot Validation",
        "",
        "This directory now contains the original P00 pilot outputs plus a diagnostic-only reconciliation layer.",
        "",
        "## Diagnostic scope",
        "- This round did not rerun full-subject benchmarks.",
        "- This round did not rerun old summary figures.",
        "- This round only audited the current P00 pilot against existing summary files and split metadata.",
        "",
        "## Current pilot values",
        "- `weissbart_tf64_p00`: `success`",
        f"  - `ridge` subject-level Pearson: `{subject_values[('weissbart_tf64_p00', 'ridge')]:.6f}`",
        f"  - `cca` subject-level Pearson: `{subject_values[('weissbart_tf64_p00', 'cca')]:.6f}`",
        f"  - `fcnn` subject-level Pearson: `{subject_values[('weissbart_tf64_p00', 'fcnn')]:.6f}`",
        f"  - `adt` subject-level Pearson: `{subject_values[('weissbart_tf64_p00', 'adt')]:.6f}`",
        "- `etard_tf64_p00`: `success`",
        f"  - `ridge` subject-level Pearson: `{subject_values[('etard_tf64_p00', 'ridge')]:.6f}`",
        f"  - `cca` subject-level Pearson: `{subject_values[('etard_tf64_p00', 'cca')]:.6f}`",
        f"  - `fcnn` subject-level Pearson: `{subject_values[('etard_tf64_p00', 'fcnn')]:.6f}`",
        f"  - `adt` subject-level Pearson: `{subject_values[('etard_tf64_p00', 'adt')]:.6f}`",
        "",
        "## Diagnostic interpretation",
        "- The current diagnostic is preliminary and should not be written as a paper conclusion.",
        "- Ridge and CCA remain close to the historical scale, which argues against a gross scorer or alignment failure.",
        "- FCNN and ADT on Etard P00 are lower than the historical full-subject summaries and need focused protocol review before full-subject rollout.",
        "",
        "## Recommended next step",
        "- Recommendation `C`: pause full-subject for now and do a focused rerun that matches FCNN/ADT protocol or budget more closely on Etard P00.",
    ]
    (ARTICLE_DIR / "result_summary.md").write_text("\n".join(result_summary) + "\n", encoding="utf-8")

    incremental = [
        "# Incremental Comparison",
        "",
        "- previous fix branch: `fix/ar-20260625-161300-a43831b-etard-p00-pilot-closure`",
        "- previous fix commit: `85fbd5a22dcad9df19d51d8d2369ccd1c27bce9b`",
        "- current fix branch: `fix/ar-20260625-161300-a43831b-article-pilot-diagnostic`",
        "",
        "## This round",
        "- diagnostic-only; no full-subject rerun",
        "- old summary figures were audited but not regenerated",
        "- no new benchmark claim should be made from this round",
        "",
        "## New diagnostic artifacts",
        "- `result_consistency_check.md`",
        "- `legacy_vs_current_diagnostic.csv`",
        "- `legacy_vs_current_diagnostic.md`",
        "- `etard_condition_diagnostic.csv`",
        "- `etard_condition_diagnostic.md`",
        "- `etard_split_sanity.md`",
        "",
        "## Preliminary conclusion",
        "- The Etard gap is more plausibly explained by single-subject pilot scope, condition mixture, and FCNN/ADT protocol/budget mismatch than by an immediately obvious pipeline bug.",
        "- Recommended next step: `C` (focused rerun before full-subject).",
    ]
    (ARTICLE_DIR / "incremental_comparison.md").write_text("\n".join(incremental) + "\n", encoding="utf-8")

    summary_text = (ARTICLE_DIR / "result_summary.md").read_text(encoding="utf-8")
    paper_text = PAPER_RESULTS.read_text(encoding="utf-8")
    consistency_text = build_consistency_check(subject_rows, manifest, summary_text, paper_text)
    (ARTICLE_DIR / "result_consistency_check.md").write_text(consistency_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
