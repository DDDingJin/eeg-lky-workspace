from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "summary_figures"


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#22303A",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linewidth": 0.7,
            "grid.color": "#BEC7CF",
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "savefig.dpi": 300,
        }
    )


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_sample_suite() -> pd.DataFrame:
    rows: list[dict] = []
    sample_csv = ROOT / "experiments" / "summary_figures" / "model_mean_overview_full_subjects.csv"
    df = pd.read_csv(sample_csv)
    for _, row in df.iterrows():
        rows.append(
            {
                "dataset": "hugo_sample_tf64",
                "model_id": row["model"],
                "display_name": row["display_name"],
                "group": row["group"],
                "mean_metric": float(row["mean"]),
                "n_subjects": int(row["n_subjects"]),
                "metric_name": row["metric"],
                "run_scope": "full_subject",
                "training_budget": "existing_local_run",
                "source": str(sample_csv),
            }
        )

    exact_runs = [
        ("adt_exact", "ADT-exact", ROOT / "experiments" / "adt_exact_reference" / "hugo_sample_tf64_all_e100" / "summary.json"),
        ("vlaai_exact", "VLAAI-exact", ROOT / "experiments" / "vlaai_exact_reference" / "hugo_sample_tf64_all_e100" / "summary.json"),
    ]
    for model_id, display_name, path in exact_runs:
        summary = load_json(path)
        rows.append(
            {
                "dataset": "hugo_sample_tf64",
                "model_id": model_id,
                "display_name": display_name,
                "group": "Exact structural port",
                "mean_metric": float(summary["mean_test_metric"]),
                "n_subjects": int(len(summary["subjects"])),
                "metric_name": summary["metric_name"],
                "run_scope": "full_subject",
                "training_budget": f"epochs={summary['epochs_requested']},early_stop",
                "source": str(path),
            }
        )
    happyquokka_path = ROOT / "experiments" / "happyquokka_reference" / "hugo_sample_tf64_e20_gcon" / "summary.json"
    if happyquokka_path.exists():
        summary = load_json(happyquokka_path)
        rows.append(
            {
                "dataset": "hugo_sample_tf64",
                "model_id": "happyquokka_gcon",
                "display_name": "HappyQuokka (g-con)",
                "group": "Subject-conditioned deep model",
                "mean_metric": float(summary["mean_test_metric"]),
                "n_subjects": int(len(summary["subjects"])),
                "metric_name": summary["metric_name"],
                "run_scope": "full_subject",
                "training_budget": f"epochs={summary['epochs_requested']},fixed",
                "source": str(happyquokka_path),
            }
        )
    return pd.DataFrame(rows)


def collect_exact_dataset_suite() -> pd.DataFrame:
    runs = [
        ("hugo_sample_tf64", "adt_exact", "ADT-exact", ROOT / "experiments" / "adt_exact_reference" / "hugo_sample_tf64_all_e100" / "summary.json"),
        ("hugo_sample_tf64", "vlaai_exact", "VLAAI-exact", ROOT / "experiments" / "vlaai_exact_reference" / "hugo_sample_tf64_all_e100" / "summary.json"),
        ("hugo_sample_tf64", "happyquokka_gcon", "HappyQuokka (g-con)", ROOT / "experiments" / "happyquokka_reference" / "hugo_sample_tf64_e20_gcon" / "summary.json"),
        ("weissbart_tf64", "adt_exact", "ADT-exact", ROOT / "experiments" / "adt_exact_reference" / "weissbart_tf64_all_e100" / "summary.json"),
        ("weissbart_tf64", "vlaai_exact", "VLAAI-exact", ROOT / "experiments" / "vlaai_exact_reference" / "weissbart_tf64_all_e100" / "summary.json"),
        ("weissbart_tf64", "happyquokka_gcon", "HappyQuokka (g-con)", ROOT / "experiments" / "happyquokka_reference" / "weissbart_tf64_e100_gcon" / "summary.json"),
        ("etard_tf64", "adt_exact", "ADT-exact", ROOT / "experiments" / "adt_exact_reference" / "etard_tf64_all_e100" / "summary.json"),
        ("etard_tf64", "vlaai_exact", "VLAAI-exact", ROOT / "experiments" / "vlaai_exact_reference" / "etard_tf64_all_e100" / "summary.json"),
        ("etard_tf64", "happyquokka_gcon", "HappyQuokka (g-con)", ROOT / "experiments" / "happyquokka_reference" / "etard_tf64_e100_gcon" / "summary.json"),
    ]
    rows: list[dict] = []
    for dataset, model_id, display_name, path in runs:
        summary = load_json(path)
        rows.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "display_name": display_name,
                "mean_metric": float(summary["mean_test_metric"]),
                "std_metric": float(summary["std_test_metric"]),
                "n_subjects": int(len(summary["subjects"])),
                "metric_name": summary["metric_name"],
                "best_epoch": int(summary["best_epoch"]),
                "epochs_requested": int(summary["epochs_requested"]),
                "source": str(path),
            }
        )
    return pd.DataFrame(rows)


def collect_unified_reference_main_suite() -> pd.DataFrame:
    rows: list[dict] = []
    expected_subjects = {"weissbart_tf64": 13, "etard_tf64": 20}
    baseline_specs = [
        ("weissbart_tf64", "avgdec_ridge", "AvgDec-Ridge", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "avgdec_ridge" / "metrics.csv", "test_pearson"),
        ("weissbart_tf64", "avgdec_lasso", "AvgDec-LASSO", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "avgdec_lasso" / "metrics.csv", "test_pearson"),
        ("weissbart_tf64", "avgcorr_ridge", "AvgCorr-Ridge", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "avgcorr_ridge" / "metrics.csv", "test_pearson"),
        ("weissbart_tf64", "avgcorr_lasso", "AvgCorr-LASSO", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "avgcorr_lasso" / "metrics.csv", "test_pearson"),
        ("weissbart_tf64", "ridge", "Ridge", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "ridge" / "metrics.csv", "test_pearson"),
        ("weissbart_tf64", "cca", "CCA", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "cca" / "metrics.csv", "test_recon_corr"),
        ("weissbart_tf64", "fcnn", "FCNN", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "fcnn" / "metrics.csv", "test_pearson"),
        ("weissbart_tf64", "cnn", "CNN", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "cnn" / "metrics.csv", "test_pearson"),
        ("weissbart_tf64", "eegnet", "EEGNet", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "eegnet" / "metrics.csv", "test_pearson"),
        ("etard_tf64", "avgdec_ridge", "AvgDec-Ridge", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "avgdec_ridge" / "metrics.csv", "test_pearson"),
        ("etard_tf64", "avgdec_lasso", "AvgDec-LASSO", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "avgdec_lasso" / "metrics.csv", "test_pearson"),
        ("etard_tf64", "avgcorr_ridge", "AvgCorr-Ridge", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "avgcorr_ridge" / "metrics.csv", "test_pearson"),
        ("etard_tf64", "avgcorr_lasso", "AvgCorr-LASSO", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "avgcorr_lasso" / "metrics.csv", "test_pearson"),
        ("etard_tf64", "ridge", "Ridge", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "ridge" / "metrics.csv", "test_pearson"),
        ("etard_tf64", "cca", "CCA", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "cca" / "metrics.csv", "test_recon_corr"),
        ("etard_tf64", "fcnn", "FCNN", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "fcnn" / "metrics.csv", "test_pearson"),
        ("etard_tf64", "cnn", "CNN", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "cnn" / "metrics.csv", "test_pearson"),
        ("etard_tf64", "eegnet", "EEGNet", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "eegnet" / "metrics.csv", "test_pearson"),
    ]
    for dataset, model_id, display_name, path, metric_col in baseline_specs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if len(df) < expected_subjects[dataset]:
            continue
        rows.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "display_name": display_name,
                "group": "Unified baseline",
                "mean_metric": float(df[metric_col].mean(skipna=True)),
                "std_metric": float(df[metric_col].std(skipna=True)),
                "n_subjects": int(len(df)),
                "metric_name": metric_col,
                "source": str(path),
            }
        )

    exact_specs = [
        ("weissbart_tf64", "adt_exact", "ADT-exact", "Exact structural port", ROOT / "experiments" / "adt_exact_reference" / "weissbart_tf64_all_e100" / "summary.json"),
        ("weissbart_tf64", "vlaai_exact", "VLAAI-exact", "Exact structural port", ROOT / "experiments" / "vlaai_exact_reference" / "weissbart_tf64_all_e100" / "summary.json"),
        ("weissbart_tf64", "happyquokka_gcon", "HappyQuokka (g-con)", "Subject-conditioned deep model", ROOT / "experiments" / "happyquokka_reference" / "weissbart_tf64_e100_gcon" / "summary.json"),
        ("etard_tf64", "adt_exact", "ADT-exact", "Exact structural port", ROOT / "experiments" / "adt_exact_reference" / "etard_tf64_all_e100" / "summary.json"),
        ("etard_tf64", "vlaai_exact", "VLAAI-exact", "Exact structural port", ROOT / "experiments" / "vlaai_exact_reference" / "etard_tf64_all_e100" / "summary.json"),
        ("etard_tf64", "happyquokka_gcon", "HappyQuokka (g-con)", "Subject-conditioned deep model", ROOT / "experiments" / "happyquokka_reference" / "etard_tf64_e100_gcon" / "summary.json"),
    ]
    for dataset, model_id, display_name, group, path in exact_specs:
        summary = load_json(path)
        rows.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "display_name": display_name,
                "group": group,
                "mean_metric": float(summary["mean_test_metric"]),
                "std_metric": float(summary["std_test_metric"]),
                "n_subjects": int(len(summary["subjects"])),
                "metric_name": summary["metric_name"],
                "source": str(path),
            }
        )
    return pd.DataFrame(rows)


def collect_unified_classical_family_suite() -> pd.DataFrame:
    rows: list[dict] = []
    specs = [
        ("weissbart_tf64", "forward_ridge", "Forward-Ridge", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "forward_ridge" / "metrics.csv", "test_mean_channel_corr"),
        ("weissbart_tf64", "ridge", "Ridge", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "ridge" / "metrics.csv", "test_pearson"),
        ("weissbart_tf64", "avgdec_ridge", "AvgDec-Ridge", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "avgdec_ridge" / "metrics.csv", "test_pearson"),
        ("weissbart_tf64", "avgdec_lasso", "AvgDec-LASSO", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "avgdec_lasso" / "metrics.csv", "test_pearson"),
        ("weissbart_tf64", "avgcorr_ridge", "AvgCorr-Ridge", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "avgcorr_ridge" / "metrics.csv", "test_pearson"),
        ("weissbart_tf64", "avgcorr_lasso", "AvgCorr-LASSO", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "avgcorr_lasso" / "metrics.csv", "test_pearson"),
        ("weissbart_tf64", "cca_recon", "CCA-recon", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "cca" / "metrics.csv", "test_recon_corr"),
        ("weissbart_tf64", "cca_match_mismatch", "CCA-MM-acc", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "cca" / "metrics.csv", "test_match_mismatch_accuracy"),
        ("etard_tf64", "forward_ridge", "Forward-Ridge", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "forward_ridge" / "metrics.csv", "test_mean_channel_corr"),
        ("etard_tf64", "ridge", "Ridge", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "ridge" / "metrics.csv", "test_pearson"),
        ("etard_tf64", "avgdec_ridge", "AvgDec-Ridge", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "avgdec_ridge" / "metrics.csv", "test_pearson"),
        ("etard_tf64", "avgdec_lasso", "AvgDec-LASSO", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "avgdec_lasso" / "metrics.csv", "test_pearson"),
        ("etard_tf64", "avgcorr_ridge", "AvgCorr-Ridge", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "avgcorr_ridge" / "metrics.csv", "test_pearson"),
        ("etard_tf64", "avgcorr_lasso", "AvgCorr-LASSO", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "avgcorr_lasso" / "metrics.csv", "test_pearson"),
        ("etard_tf64", "cca_recon", "CCA-recon", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "cca" / "metrics.csv", "test_recon_corr"),
        ("etard_tf64", "cca_match_mismatch", "CCA-MM-acc", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "cca" / "metrics.csv", "test_match_mismatch_accuracy"),
    ]
    for dataset, model_id, display_name, path, metric_col in specs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        rows.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "display_name": display_name,
                "metric_name": metric_col,
                "mean_metric": float(df[metric_col].mean()),
                "std_metric": float(df[metric_col].std()),
                "n_subjects": int(len(df)),
                "source": str(path),
            }
        )
    return pd.DataFrame(rows)


def plot_sample_suite(df: pd.DataFrame) -> None:
    order = [
        "Ridge",
        "AvgDec-Ridge",
        "AvgCorr-Ridge",
        "CCA",
        "FCNN",
        "CNN",
        "EEGNet",
        "ADT-lite",
        "VLAAI-lite",
        "ADT-exact",
        "VLAAI-exact",
        "HappyQuokka (g-con)",
    ]
    color_map = {
        "Linear / correlation baseline": "#537A8A",
        "Neural benchmark": "#C66C3D",
        "Approximate deep port": "#8E63A5",
        "Exact structural port": "#2E8B57",
        "Subject-conditioned deep model": "#C14C64",
    }

    plot_df = df.copy()
    plot_df["display_name"] = pd.Categorical(plot_df["display_name"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("display_name").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13.0, 6.8))
    for idx, row in plot_df.iterrows():
        ax.bar(
            idx,
            row["mean_metric"],
            color=color_map[row["group"]],
            edgecolor="#22303A",
            linewidth=0.8,
            width=0.72,
        )
        ax.text(idx, row["mean_metric"] + 0.006, f"{row['mean_metric']:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["display_name"], rotation=28, ha="right")
    ax.set_ylabel("Mean test correlation")
    ax.set_title("Sample Dataset: Current Comparable Methods")
    ax.set_ylim(0.0, float(plot_df["mean_metric"].max()) + 0.08)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=color_map["Linear / correlation baseline"], label="Linear / correlation baseline"),
            Patch(facecolor=color_map["Neural benchmark"], label="Neural benchmark"),
            Patch(facecolor=color_map["Approximate deep port"], label="Approximate deep port"),
            Patch(facecolor=color_map["Exact structural port"], label="Exact structural port"),
            Patch(facecolor=color_map["Subject-conditioned deep model"], label="Subject-conditioned deep model"),
        ],
        ncol=2,
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sample_all_methods_overview.png")
    plt.close(fig)


def plot_exact_dataset_suite(df: pd.DataFrame) -> None:
    dataset_order = ["hugo_sample_tf64", "weissbart_tf64", "etard_tf64"]
    model_order = ["ADT-exact", "VLAAI-exact", "HappyQuokka (g-con)"]
    color_map = {"ADT-exact": "#2E8B57", "VLAAI-exact": "#3F6C7A", "HappyQuokka (g-con)": "#C14C64"}

    plot_df = df.copy()
    plot_df["dataset"] = pd.Categorical(plot_df["dataset"], categories=dataset_order, ordered=True)
    plot_df["display_name"] = pd.Categorical(plot_df["display_name"], categories=model_order, ordered=True)
    plot_df = plot_df.sort_values(["dataset", "display_name"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    width = 0.24
    xs = list(range(len(dataset_order)))
    for model_idx, model_name in enumerate(model_order):
        sub = plot_df[plot_df["display_name"] == model_name].set_index("dataset").loc[dataset_order].reset_index()
        offsets = [x + (model_idx - 1.0) * width for x in xs]
        ax.bar(
            offsets,
            sub["mean_metric"],
            width=width,
            yerr=sub["std_metric"],
            color=color_map[model_name],
            edgecolor="#22303A",
            linewidth=0.8,
            capsize=3,
            label=model_name,
        )
        for off, value in zip(offsets, sub["mean_metric"]):
            ax.text(off, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(xs)
    ax.set_xticklabels(dataset_order)
    ax.set_ylabel("Mean test correlation")
    ax.set_title("Exact Structural Ports Across Unified Datasets")
    ax.set_ylim(0.0, float(plot_df["mean_metric"].max() + plot_df["std_metric"].max()) + 0.08)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exact_reference_dataset_overview.png")
    plt.close(fig)


def plot_unified_reference_main_suite(df: pd.DataFrame) -> None:
    dataset_order = ["weissbart_tf64", "etard_tf64"]
    model_order = ["Ridge", "CCA", "FCNN", "CNN", "EEGNet", "ADT-exact", "VLAAI-exact", "HappyQuokka (g-con)"]
    color_map = {
        "Ridge": "#537A8A",
        "CCA": "#6D8A96",
        "FCNN": "#C66C3D",
        "CNN": "#D37A3F",
        "EEGNet": "#E08A46",
        "ADT-exact": "#2E8B57",
        "VLAAI-exact": "#3D6F8A",
        "HappyQuokka (g-con)": "#C14C64",
    }

    plot_df = df.copy()
    plot_df["dataset"] = pd.Categorical(plot_df["dataset"], categories=dataset_order, ordered=True)
    plot_df["display_name"] = pd.Categorical(plot_df["display_name"], categories=model_order, ordered=True)
    plot_df = plot_df.sort_values(["dataset", "display_name"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), sharey=True)
    for ax, dataset in zip(axes, dataset_order):
        sub = plot_df[plot_df["dataset"] == dataset].set_index("display_name").loc[model_order].reset_index()
        xs = list(range(len(model_order)))
        ax.bar(
            xs,
            sub["mean_metric"],
            yerr=sub["std_metric"],
            color=[color_map[name] for name in sub["display_name"]],
            edgecolor="#22303A",
            linewidth=0.8,
            capsize=3,
            width=0.72,
        )
        for idx, row in sub.iterrows():
            ax.text(idx, row["mean_metric"] + row["std_metric"] + 0.006, f"{row['mean_metric']:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(sub["display_name"], rotation=28, ha="right")
        ax.set_title(dataset)
        ax.set_ylim(0.0, float(plot_df["mean_metric"].max() + plot_df["std_metric"].max()) + 0.08)
    axes[0].set_ylabel("Mean test correlation")
    fig.suptitle("Unified Dataset Baselines And Exact Ports")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "unified_reference_main_overview.png")
    plt.close(fig)


def write_manifest(sample_df: pd.DataFrame, exact_df: pd.DataFrame) -> None:
    manifest = {
        "generated_files": [
            "sample_all_methods_summary.csv",
            "sample_all_methods_overview.png",
            "exact_reference_dataset_summary.csv",
            "exact_reference_dataset_overview.png",
            "unified_reference_main_summary.csv",
            "unified_reference_main_overview.png",
            "unified_classical_baseline_family_summary.csv",
            "happyquokka_reference/*.json",
        ],
        "sample_rows": int(len(sample_df)),
        "exact_rows": int(len(exact_df)),
    }
    with open(OUT_DIR / "benchmark_archive_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    sample_df = collect_sample_suite()
    exact_df = collect_exact_dataset_suite()
    unified_df = collect_unified_reference_main_suite()
    classical_df = collect_unified_classical_family_suite()
    sample_df.to_csv(OUT_DIR / "sample_all_methods_summary.csv", index=False)
    exact_df.to_csv(OUT_DIR / "exact_reference_dataset_summary.csv", index=False)
    unified_df.to_csv(OUT_DIR / "unified_reference_main_summary.csv", index=False)
    classical_df.to_csv(OUT_DIR / "unified_classical_baseline_family_summary.csv", index=False)
    plot_sample_suite(sample_df)
    plot_exact_dataset_suite(exact_df)
    plot_unified_reference_main_suite(unified_df)
    write_manifest(sample_df, exact_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
