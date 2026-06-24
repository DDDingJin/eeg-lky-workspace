from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "summary_figures"


RUNS = [
    ("hugo_sample_tf64", "NULL g_con=True", ROOT / "experiments" / "neuroconformer_reference" / "hugo_sample_tf64_e10_gcon" / "summary.json"),
    ("hugo_sample_tf64", "NULL g_con=False", ROOT / "experiments" / "neuroconformer_reference" / "hugo_sample_tf64_e10_nogcon" / "summary.json"),
    ("weissbart_tf64", "NULL g_con=True", ROOT / "experiments" / "neuroconformer_reference" / "weissbart_tf64_e10_gcon" / "summary.json"),
    ("weissbart_tf64", "NULL g_con=False", ROOT / "experiments" / "neuroconformer_reference" / "weissbart_tf64_e10_nogcon" / "summary.json"),
    ("etard_tf64", "NULL g_con=True", ROOT / "experiments" / "neuroconformer_reference" / "etard_tf64_e10_gcon" / "summary.json"),
    ("etard_tf64", "NULL g_con=False", ROOT / "experiments" / "neuroconformer_reference" / "etard_tf64_e10_nogcon" / "summary.json"),
]


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
            "grid.alpha": 0.14,
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


def collect_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    curve_rows: list[dict] = []
    for dataset, label, path in RUNS:
        summary = load_json(path)
        rows.append(
            {
                "dataset": dataset,
                "label": label,
                "g_con": bool(summary["g_con"]),
                "epochs_requested": int(summary["epochs_requested"]),
                "epochs_completed": int(summary["epochs_completed"]),
                "best_epoch": int(summary["best_epoch"]),
                "best_val_metric": float(summary["best_val_metric"]),
                "mean_test_metric": float(summary["mean_test_metric"]),
                "std_test_metric": float(summary["std_test_metric"]),
                "n_subjects": int(len(summary["subjects"])),
                "source": str(path),
            }
        )
        for idx, (train_loss, val_loss, val_metric, lr) in enumerate(
            zip(
                summary["history"]["train_loss"],
                summary["history"]["val_loss"],
                summary["history"]["val_metric"],
                summary["history"]["lr"],
            ),
            start=1,
        ):
            curve_rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "g_con": bool(summary["g_con"]),
                    "epoch": idx,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_metric": float(val_metric),
                    "lr": float(lr),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(curve_rows)


def plot_conditioning_summary(df: pd.DataFrame) -> None:
    dataset_order = ["hugo_sample_tf64", "weissbart_tf64", "etard_tf64"]
    condition_order = ["NULL g_con=False", "NULL g_con=True"]
    color_map = {
        "NULL g_con=False": "#537A8A",
        "NULL g_con=True": "#B5483A",
    }
    plot_df = df.copy()
    plot_df["dataset"] = pd.Categorical(plot_df["dataset"], categories=dataset_order, ordered=True)
    plot_df["label"] = pd.Categorical(plot_df["label"], categories=condition_order, ordered=True)
    plot_df = plot_df.sort_values(["dataset", "label"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    width = 0.34
    xs = list(range(len(dataset_order)))
    for model_idx, label in enumerate(condition_order):
        sub = plot_df[plot_df["label"] == label].set_index("dataset").loc[dataset_order].reset_index()
        offsets = [x + (model_idx - 0.5) * width for x in xs]
        ax.bar(
            offsets,
            sub["mean_test_metric"],
            width=width,
            yerr=sub["std_test_metric"],
            color=color_map[label],
            edgecolor="#22303A",
            linewidth=0.8,
            capsize=3,
            label=label,
        )
        for off, value in zip(offsets, sub["mean_test_metric"]):
            ax.text(off, value + 0.005, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(dataset_order)
    ax.set_ylabel("Mean test correlation")
    ax.set_title("NULL Conditioning Comparison")
    ax.set_ylim(0.0, float(plot_df["mean_test_metric"].max() + plot_df["std_test_metric"].max()) + 0.08)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "null_conditioning_overview.png")
    plt.close(fig)


def plot_training_curves(curve_df: pd.DataFrame) -> None:
    dataset_order = ["hugo_sample_tf64", "weissbart_tf64", "etard_tf64"]
    label_order = ["NULL g_con=False", "NULL g_con=True"]
    color_map = {
        "NULL g_con=False": "#537A8A",
        "NULL g_con=True": "#B5483A",
    }
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 11.6), sharex=True)
    for row_idx, dataset in enumerate(dataset_order):
        for col_idx, metric_name in enumerate(["val_metric", "train_loss"]):
            ax = axes[row_idx, col_idx]
            sub = curve_df[curve_df["dataset"] == dataset]
            for label in label_order:
                series = sub[sub["label"] == label]
                ax.plot(series["epoch"], series[metric_name], color=color_map[label], linewidth=2.0, label=label)
            title_metric = "Validation correlation" if metric_name == "val_metric" else "Training loss"
            ax.set_title(f"{dataset}: {title_metric}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title_metric)
            if row_idx == 0 and col_idx == 1:
                ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "null_training_curves.png")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    summary_df, curve_df = collect_summary()
    summary_df.to_csv(OUT_DIR / "null_conditioning_summary.csv", index=False)
    curve_df.to_csv(OUT_DIR / "null_training_curves.csv", index=False)
    plot_conditioning_summary(summary_df)
    plot_training_curves(curve_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
