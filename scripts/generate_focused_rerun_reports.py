from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "experiments" / "gate0_gate2_article_pilot"
FOCUSED_DIR = ROOT / "experiments" / "gate0_gate2_etard_p00_focused_rerun"
SUMMARY_DIR = ROOT / "experiments" / "summary_figures"
PAPER_RESULTS = ROOT / "paper" / "draft_zh" / "sections" / "results_placeholder_zh.tex"


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    pilot_rows = read_csv(PILOT_DIR / "subject_metrics.csv")
    focused_rows = read_csv(FOCUSED_DIR / "subject_metrics.csv")
    unified_rows = read_csv(SUMMARY_DIR / "unified_reference_main_summary.csv")
    exact_rows = read_csv(SUMMARY_DIR / "exact_reference_dataset_summary.csv")
    focused_manifest = json.loads((FOCUSED_DIR / "run_manifest.json").read_text(encoding="utf-8"))

    pilot = {row["model"]: row for row in pilot_rows if row["dataset"] == "etard_tf64_p00" and row["model"] in {"fcnn", "adt"}}
    focused = {row["model"]: row for row in focused_rows if row["dataset"] == "etard_tf64_p00"}
    focused_meta = {row["model"]: row for row in focused_manifest["models"]}

    legacy_map = {}
    for row in unified_rows:
        if row["dataset"] == "etard_tf64" and row["model_id"] == "fcnn":
            legacy_map["fcnn"] = {
                "metric": float(row["mean_metric"]),
                "source": row["source"],
                "n_subjects": int(row["n_subjects"]),
            }
    for row in exact_rows:
        if row["dataset"] == "etard_tf64" and row["model_id"] == "adt_exact":
            legacy_map["adt"] = {
                "metric": float(row["mean_metric"]),
                "source": row["source"],
                "n_subjects": int(row["n_subjects"]),
            }

    rows = []
    for model in ["fcnn", "adt"]:
        pilot_metric = float(pilot[model]["metric_value"])
        focused_metric = float(focused[model]["metric_value"])
        meta = focused_meta[model]
        old = legacy_map[model]
        if model == "fcnn":
            pilot_epochs = 5
        else:
            pilot_epochs = 3
        interpretation = (
            "focused rerun did not improve over pilot; budget alone is unlikely to explain the gap"
            if abs(focused_metric - pilot_metric) < 1e-9
            else "focused rerun improved over pilot; budget is a meaningful factor"
        )
        rows.append(
            {
                "dataset": "etard_tf64_p00",
                "subject_id": "P00",
                "model": model,
                "old_summary_metric": old["metric"],
                "old_summary_source": old["source"],
                "old_n_subjects": old["n_subjects"],
                "pilot_metric": pilot_metric,
                "pilot_epochs": pilot_epochs,
                "focused_metric": focused_metric,
                "focused_epochs_completed": meta["epochs_completed"],
                "focused_best_epoch": meta["best_epoch"],
                "focused_best_val_metric": meta.get("best_val_score", meta.get("best_val_metric")),
                "focused_minus_pilot": focused_metric - pilot_metric,
                "focused_minus_old_mean": focused_metric - old["metric"],
                "directly_comparable_with_old": "no",
                "interpretation": interpretation,
            }
        )

    fieldnames = [
        "dataset",
        "subject_id",
        "model",
        "old_summary_metric",
        "old_summary_source",
        "old_n_subjects",
        "pilot_metric",
        "pilot_epochs",
        "focused_metric",
        "focused_epochs_completed",
        "focused_best_epoch",
        "focused_best_val_metric",
        "focused_minus_pilot",
        "focused_minus_old_mean",
        "directly_comparable_with_old",
        "interpretation",
    ]
    write_csv(FOCUSED_DIR / "focused_vs_pilot_comparison.csv", rows, fieldnames)

    fcnn_row = next(row for row in rows if row["model"] == "fcnn")
    adt_row = next(row for row in rows if row["model"] == "adt")
    summary_lines = [
        "# Focused Rerun Summary",
        "",
        "This directory contains the Etard `P00` focused rerun for `fcnn` and `adt` under the same unified scorer and result schema.",
        "",
        "## Focused rerun values",
        f"- `fcnn`: `{fcnn_row['focused_metric']:.6f}`",
        f"- `adt`: `{adt_row['focused_metric']:.6f}`",
        "",
        "## Training metadata",
        f"- `fcnn`: `best_epoch={fcnn_row['focused_best_epoch']}`, `epochs_completed={fcnn_row['focused_epochs_completed']}`, `best_val_metric={fcnn_row['focused_best_val_metric']:.6f}`",
        f"- `adt`: `best_epoch={adt_row['focused_best_epoch']}`, `epochs_completed={adt_row['focused_epochs_completed']}`, `best_val_metric={adt_row['focused_best_val_metric']:.6f}`",
        "",
        "## Interpretation",
        "- This is a focused training-budget diagnostic, not a benchmark conclusion.",
        "- The unified scorer and schema were preserved.",
    ]
    (FOCUSED_DIR / "focused_rerun_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    md_lines = [
        "# Focused vs Pilot Comparison",
        "",
        "This comparison is diagnostic-only. It contrasts the current Etard `P00` article pilot with a focused rerun under a larger training budget, while keeping the same split, scorer, and result schema.",
        "",
        "## Direct answers",
        f"1. `FCNN` did not improve: pilot `{fcnn_row['pilot_metric']:.6f}` vs focused `{fcnn_row['focused_metric']:.6f}`. `ADT` improved clearly: pilot `{adt_row['pilot_metric']:.6f}` vs focused `{adt_row['focused_metric']:.6f}`.",
        "2. The ADT improvement indicates that training budget is a major contributor for ADT. The FCNN non-improvement indicates that budget alone is not the main explanation for FCNN.",
        "3. Because FCNN did not improve, the next suspects are model protocol mismatch, window definition, target alignment, or implementation differences relative to the old baseline run, not simply insufficient epochs.",
        f"4. Focused rerun ADT (`{adt_row['focused_metric']:.6f}`) is now very close to the old Etard ADT-exact mean (`{adt_row['old_summary_metric']:.6f}`). Focused rerun FCNN (`{fcnn_row['focused_metric']:.6f}`) remains well below the old FCNN mean (`{fcnn_row['old_summary_metric']:.6f}`).",
        "5. Even after focused rerun, this is still a single-subject P00 diagnostic and should not be written as a formal benchmark conclusion.",
        "6. The result supports a mixed recommendation: ADT is now less concerning and could be taken into a larger single-seed rollout, but FCNN still needs a protocol-focused check before promoting the whole suite to full-subject.",
        "",
        "## Recommendation",
        "- Recommendation `C`: continue focused protocol checking before full-subject. Specifically, ADT no longer shows the same level of concern, but FCNN still does.",
    ]
    (FOCUSED_DIR / "focused_vs_pilot_comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    paper_text = PAPER_RESULTS.read_text(encoding="utf-8")
    diagnostic_note = (
        "\n\n进一步地，我们对 \\texttt{etard\\_tf64\\_p00} 的 FCNN 与 ADT 做了 focused rerun，"
        "在不改变 split、scorer 与 result schema 的前提下，仅增加训练预算。结果显示，ADT 从 pilot 的 0.056 提升到 0.085，"
        "已经接近旧的全被试 ADT-exact 均值；而 FCNN 从 0.031 几乎没有提升。"
        "这说明 Etard 上的 ADT 低值主要受短训练预算影响，而 FCNN 的差距更可能还涉及模型协议或实现层面的差异。"
        "这一步仍然只是 training-budget diagnostic，不应写成正式 benchmark 结论。"
    )
    if diagnostic_note not in paper_text:
        paper_text = paper_text.rstrip() + diagnostic_note + "\n"
        PAPER_RESULTS.write_text(paper_text, encoding="utf-8")

    article_legacy = PILOT_DIR / "legacy_vs_current_diagnostic.md"
    legacy_text = article_legacy.read_text(encoding="utf-8")
    focused_addendum = (
        "\n## Focused rerun addendum\n"
        f"- ADT focused rerun reached `{adt_row['focused_metric']:.6f}` from pilot `{adt_row['pilot_metric']:.6f}`, which strongly supports training budget as a major cause for the pilot underestimation.\n"
        f"- FCNN focused rerun stayed at `{fcnn_row['focused_metric']:.6f}` from pilot `{fcnn_row['pilot_metric']:.6f}`, which weakens the short-budget explanation for FCNN and points more toward protocol mismatch or implementation differences.\n"
        "- Therefore the Etard low-value explanation is now model-specific rather than uniform across all deep models.\n"
    )
    if "## Focused rerun addendum" not in legacy_text:
        article_legacy.write_text(legacy_text.rstrip() + "\n" + focused_addendum, encoding="utf-8")

    result_summary = PILOT_DIR / "result_summary.md"
    result_text = result_summary.read_text(encoding="utf-8")
    result_text = result_text.rstrip() + (
        "\n\n## Focused rerun note\n"
        "- This round also added a focused rerun on `etard_tf64_p00` for `fcnn` and `adt`; it is still diagnostic-only and not a full-subject benchmark rerun.\n"
        f"- `adt` improved from `0.056455` to `{adt_row['focused_metric']:.6f}`, which reduces concern that the earlier ADT pilot was purely a pipeline failure.\n"
        f"- `fcnn` remained at `{fcnn_row['focused_metric']:.6f}`, so FCNN still requires protocol-focused investigation before broad rollout.\n"
    ) + "\n"
    result_summary.write_text(result_text, encoding="utf-8")

    incremental = PILOT_DIR / "incremental_comparison.md"
    incremental_text = incremental.read_text(encoding="utf-8")
    incremental_text = incremental_text.rstrip() + (
        "\n\n## Focused rerun follow-up\n"
        "- This round did not rerun old summary figures.\n"
        "- This round added a focused P00 rerun for Etard FCNN and ADT only.\n"
        "- The focused rerun changes the interpretation: ADT low pilot values are now largely explainable by training budget, while FCNN still appears protocol-limited.\n"
    ) + "\n"
    incremental.write_text(incremental_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
