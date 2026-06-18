from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "experiments" / "summary_figures"
DOC_DIR = ROOT / "docs"


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#24313B",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.14,
            "grid.linewidth": 0.7,
            "grid.color": "#BEC7CF",
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "savefig.dpi": 300,
        }
    )


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def format_float(value: object, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def dataframe_to_longtable(
    df: pd.DataFrame,
    columns: list[str],
    headers: list[str],
    colspec: str,
) -> str:
    parts = [
        rf"\begin{{longtable}}{{{colspec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in df.iterrows():
        values = [latex_escape(row[col]) for col in columns]
        parts.append(" & ".join(values) + r" \\")
    parts.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(parts)


def aggregate_history_from_participant_jsons(paths: list[Path], key: str = "val_history") -> tuple[list[float], list[float], int]:
    series = []
    for path in paths:
        data = load_json(path)
        history = data[key]
        series.append([float(x) for x in history])
    max_len = max(len(x) for x in series)
    means = []
    stds = []
    for idx in range(max_len):
        values = [item[idx] for item in series if idx < len(item)]
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values, ddof=0)))
    return means, stds, len(series)


def collect_curve_rows() -> pd.DataFrame:
    rows: list[dict] = []
    grouped_specs = [
        ("hugo_sample_tf64", "fcnn", "FCNN", ROOT / "experiments" / "mldecoders" / "all_subjects" / "fcnn"),
        ("hugo_sample_tf64", "cnn", "CNN", ROOT / "experiments" / "mldecoders" / "all_subjects" / "cnn"),
        ("hugo_sample_tf64", "eegnet", "EEGNet", ROOT / "experiments" / "mldecoders" / "all_subjects" / "eegnet"),
        ("weissbart_tf64", "fcnn", "FCNN", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "fcnn"),
        ("weissbart_tf64", "cnn", "CNN", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "cnn"),
        ("weissbart_tf64", "eegnet", "EEGNet", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "eegnet"),
        ("etard_tf64", "fcnn", "FCNN", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "fcnn"),
        ("etard_tf64", "cnn", "CNN", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "cnn"),
        ("etard_tf64", "eegnet", "EEGNet", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "eegnet"),
    ]
    for dataset, model_id, display_name, folder in grouped_specs:
        paths = sorted(folder.glob("*_history.json"))
        if not paths:
            continue
        means, stds, n_runs = aggregate_history_from_participant_jsons(paths)
        for epoch_idx, (mean_val, std_val) in enumerate(zip(means, stds), start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "model_id": model_id,
                    "display_name": display_name,
                    "epoch": epoch_idx,
                    "mean_val_metric": mean_val,
                    "std_val_metric": std_val,
                    "n_runs": n_runs,
                    "history_source": "participant_average",
                }
            )

    exact_specs = [
        ("hugo_sample_tf64", "adt_exact", "ADT-exact", ROOT / "experiments" / "adt_exact_reference" / "hugo_sample_tf64_all_e100" / "summary.json"),
        ("hugo_sample_tf64", "vlaai_exact", "VLAAI-exact", ROOT / "experiments" / "vlaai_exact_reference" / "hugo_sample_tf64_all_e100" / "summary.json"),
        ("weissbart_tf64", "adt_exact", "ADT-exact", ROOT / "experiments" / "adt_exact_reference" / "weissbart_tf64_all_e100" / "summary.json"),
        ("weissbart_tf64", "vlaai_exact", "VLAAI-exact", ROOT / "experiments" / "vlaai_exact_reference" / "weissbart_tf64_all_e100" / "summary.json"),
        ("etard_tf64", "adt_exact", "ADT-exact", ROOT / "experiments" / "adt_exact_reference" / "etard_tf64_all_e100" / "summary.json"),
        ("etard_tf64", "vlaai_exact", "VLAAI-exact", ROOT / "experiments" / "vlaai_exact_reference" / "etard_tf64_all_e100" / "summary.json"),
    ]
    for dataset, model_id, display_name, path in exact_specs:
        summary = load_json(path)
        history = summary["history"]["val_pearson_metric"]
        for epoch_idx, value in enumerate(history, start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "model_id": model_id,
                    "display_name": display_name,
                    "epoch": epoch_idx,
                    "mean_val_metric": float(value),
                    "std_val_metric": 0.0,
                    "n_runs": 1,
                    "history_source": "single_global_run",
                }
            )
    return pd.DataFrame(rows)


def plot_curves(curve_df: pd.DataFrame) -> None:
    dataset_order = ["hugo_sample_tf64", "weissbart_tf64", "etard_tf64"]
    color_map = {
        "FCNN": "#C66C3D",
        "CNN": "#D9793E",
        "EEGNet": "#E59B47",
        "ADT-exact": "#2E8B57",
        "VLAAI-exact": "#3F6C7A",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), sharey=False)
    for ax, dataset in zip(axes, dataset_order):
        sub = curve_df[curve_df["dataset"] == dataset].copy()
        for display_name in ["FCNN", "CNN", "EEGNet", "ADT-exact", "VLAAI-exact"]:
            part = sub[sub["display_name"] == display_name].sort_values("epoch")
            if part.empty:
                continue
            ax.plot(part["epoch"], part["mean_val_metric"], label=display_name, color=color_map[display_name], linewidth=1.8)
            if (part["std_val_metric"] > 0).any():
                ax.fill_between(
                    part["epoch"].to_numpy(),
                    (part["mean_val_metric"] - part["std_val_metric"]).to_numpy(),
                    (part["mean_val_metric"] + part["std_val_metric"]).to_numpy(),
                    color=color_map[display_name],
                    alpha=0.12,
                )
        ax.set_title(dataset)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation correlation")
    axes[0].legend(ncol=1, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reproduction_audit_training_curves.png")
    plt.close(fig)


def collect_protocol_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "family": "unified_baseline",
                "model": "Ridge",
                "implementation_type": "lagged linear decoder",
                "input_shape": "time x 64",
                "target_type": "scalar envelope sample",
                "window_or_lag": "lags 0:50",
                "optimizer_or_solver": "sklearn Ridge",
                "selection_rule": "best validation correlation over alpha grid",
                "train_mode": "subject-specific",
                "notes": "train/val/test are per-subject recording splits",
            },
            {
                "family": "unified_baseline",
                "model": "CCA",
                "implementation_type": "CCA reconstruction plus match/mismatch",
                "input_shape": "time x 64",
                "target_type": "continuous envelope",
                "window_or_lag": "lags 0:50, PCA + CCA + ridge reconstruction",
                "optimizer_or_solver": "sklearn CCA + Ridge",
                "selection_rule": "best validation reconstruction correlation",
                "train_mode": "subject-specific",
                "notes": "main table uses reconstruction correlation rather than canonical correlation",
            },
            {
                "family": "unified_baseline",
                "model": "FCNN",
                "implementation_type": "upstream mldecoders-style DNN",
                "input_shape": "64 x 50 window",
                "target_type": "scalar envelope sample",
                "window_or_lag": "window 50, target at last sample",
                "optimizer_or_solver": "NAdam lr=1e-4 wd=1e-4",
                "selection_rule": "best validation correlation, patience=10",
                "train_mode": "subject-specific",
                "notes": "per-participant histories available",
            },
            {
                "family": "unified_baseline",
                "model": "CNN",
                "implementation_type": "upstream mldecoders CNN",
                "input_shape": "64 x 50 window",
                "target_type": "scalar envelope sample",
                "window_or_lag": "window 50, target at last sample",
                "optimizer_or_solver": "NAdam lr=1e-2 wd=1e-8",
                "selection_rule": "best validation correlation, patience=10",
                "train_mode": "subject-specific",
                "notes": "F1=8, D=8, F2=64, dropout=0.20",
            },
            {
                "family": "unified_baseline",
                "model": "EEGNet",
                "implementation_type": "EEGNet-style regressor",
                "input_shape": "64 x 50 window",
                "target_type": "scalar envelope sample",
                "window_or_lag": "window 50, target at last sample",
                "optimizer_or_solver": "NAdam lr=1e-3 wd=1e-4",
                "selection_rule": "best validation correlation, patience=10",
                "train_mode": "subject-specific",
                "notes": "temporal_filters=8, depth_multiplier=2, separable_filters=16, dropout=0.25",
            },
            {
                "family": "exact_port",
                "model": "ADT-exact",
                "implementation_type": "sequence-to-sequence structural port",
                "input_shape": "320 x 64 window",
                "target_type": "320 x 1 envelope sequence",
                "window_or_lag": "window 320, hop 64",
                "optimizer_or_solver": "Adam lr=1e-3 + ReduceLROnPlateau",
                "selection_rule": "best pooled validation Pearson checkpoint, patience=10",
                "train_mode": "pooled multi-subject train/val, subject-wise test report",
                "notes": "spatio-temporal conv frontend + anti-causal transformer decoder",
            },
            {
                "family": "exact_port",
                "model": "VLAAI-exact",
                "implementation_type": "sequence-to-sequence structural port",
                "input_shape": "320 x 64 window",
                "target_type": "320 x 1 envelope sequence",
                "window_or_lag": "window 320, hop 64",
                "optimizer_or_solver": "Adam lr=1e-3 + ReduceLROnPlateau",
                "selection_rule": "best pooled validation Pearson checkpoint, patience=10",
                "train_mode": "pooled multi-subject train/val, subject-wise test report",
                "notes": "shared extractor/output-context modules, 4 blocks",
            },
        ]
    )


def collect_run_metadata() -> pd.DataFrame:
    rows: list[dict] = []
    baseline_specs = [
        ("weissbart_tf64", "FCNN", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "fcnn"),
        ("weissbart_tf64", "CNN", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "cnn"),
        ("weissbart_tf64", "EEGNet", ROOT / "experiments" / "reference_baselines" / "weissbart_tf64" / "eegnet"),
        ("etard_tf64", "FCNN", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "fcnn"),
        ("etard_tf64", "CNN", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "cnn"),
        ("etard_tf64", "EEGNet", ROOT / "experiments" / "reference_baselines" / "etard_tf64" / "eegnet"),
    ]
    for dataset, model, folder in baseline_specs:
        vals = [load_json(path) for path in sorted(folder.glob("*_history.json"))]
        if not vals:
            continue
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "run_type": "subject_specific_baseline",
                "epochs_requested": float(np.mean([item["epochs_requested"] for item in vals])),
                "epochs_completed_mean": float(np.mean([item["epochs_completed"] for item in vals])),
                "best_epoch_mean": float(np.mean([item["best_epoch"] for item in vals])),
                "n_histories": len(vals),
            }
        )
    exact_specs = [
        ("hugo_sample_tf64", "ADT-exact", ROOT / "experiments" / "adt_exact_reference" / "hugo_sample_tf64_all_e100" / "summary.json"),
        ("hugo_sample_tf64", "VLAAI-exact", ROOT / "experiments" / "vlaai_exact_reference" / "hugo_sample_tf64_all_e100" / "summary.json"),
        ("weissbart_tf64", "ADT-exact", ROOT / "experiments" / "adt_exact_reference" / "weissbart_tf64_all_e100" / "summary.json"),
        ("weissbart_tf64", "VLAAI-exact", ROOT / "experiments" / "vlaai_exact_reference" / "weissbart_tf64_all_e100" / "summary.json"),
        ("etard_tf64", "ADT-exact", ROOT / "experiments" / "adt_exact_reference" / "etard_tf64_all_e100" / "summary.json"),
        ("etard_tf64", "VLAAI-exact", ROOT / "experiments" / "vlaai_exact_reference" / "etard_tf64_all_e100" / "summary.json"),
    ]
    for dataset, model, path in exact_specs:
        item = load_json(path)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "run_type": "pooled_exact_port",
                "epochs_requested": item["epochs_requested"],
                "epochs_completed_mean": len(item["history"]["val_loss"]),
                "best_epoch_mean": item["best_epoch"],
                "n_histories": 1,
            }
        )
    return pd.DataFrame(rows)


def collect_implementation_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "role": "unified baseline logic",
                "path": "src/repro/reference_baselines.py",
                "purpose": "window construction, linear baselines, CCA, DNN train/eval",
            },
            {
                "role": "baseline batch runner",
                "path": "scripts/run_reference_baselines.py",
                "purpose": "per-dataset, per-subject batch execution and metric export",
            },
            {
                "role": "ADT exact model",
                "path": "src/repro/adt_exact.py",
                "purpose": "exact structural port, pooled sequence training, subject-wise test report",
            },
            {
                "role": "VLAAI exact model",
                "path": "src/repro/vlaai_exact.py",
                "purpose": "exact structural port, pooled sequence training, subject-wise test report",
            },
            {
                "role": "exact batch runner",
                "path": "scripts/run_reference_exact_suite.py",
                "purpose": "standardized launcher for exact reference runs",
            },
            {
                "role": "archive summary builder",
                "path": "scripts/archive_benchmark_results.py",
                "purpose": "aggregates method outputs into comparison CSV and publication-style figures",
            },
        ]
    )


def write_tex(
    protocol_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    implementation_df: pd.DataFrame,
    unified_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    exact_df: pd.DataFrame,
) -> None:
    protocol_tex = dataframe_to_longtable(
        protocol_df.fillna(""),
        columns=[
            "family",
            "model",
            "implementation_type",
            "input_shape",
            "target_type",
            "window_or_lag",
            "optimizer_or_solver",
            "selection_rule",
            "train_mode",
            "notes",
        ],
        headers=[
            "Family",
            "Model",
            "Implementation",
            "Input",
            "Target",
            "Window/Lag",
            "Optimizer",
            "Selection",
            "Train mode",
            "Notes",
        ],
        colspec=r"p{1.4cm} p{1.6cm} p{2.2cm} p{1.8cm} p{1.8cm} p{2.0cm} p{2.4cm} p{2.6cm} p{2.0cm} p{2.8cm}",
    )

    metadata_fmt = metadata_df.copy()
    for col in ["epochs_requested", "epochs_completed_mean", "best_epoch_mean"]:
        metadata_fmt[col] = metadata_fmt[col].map(lambda x: format_float(x, 1))
    metadata_tex = dataframe_to_longtable(
        metadata_fmt.fillna(""),
        columns=[
            "dataset",
            "model",
            "run_type",
            "epochs_requested",
            "epochs_completed_mean",
            "best_epoch_mean",
            "n_histories",
        ],
        headers=[
            "Dataset",
            "Model",
            "Run type",
            "Epochs req.",
            "Epochs done",
            "Best epoch",
            "Histories",
        ],
        colspec=r"p{2.1cm} p{1.7cm} p{3.0cm} p{1.5cm} p{1.7cm} p{1.5cm} p{1.2cm}",
    )

    implementation_tex = dataframe_to_longtable(
        implementation_df,
        columns=["role", "path", "purpose"],
        headers=["Role", "Path", "Purpose"],
        colspec=r"p{2.2cm} p{4.1cm} p{7.0cm}",
    )

    unified_fmt = unified_df.copy()
    unified_fmt["mean_metric"] = unified_fmt["mean_metric"].map(format_float)
    unified_fmt["std_metric"] = unified_fmt["std_metric"].map(format_float)
    unified_tex = dataframe_to_longtable(
        unified_fmt,
        columns=["dataset", "display_name", "group", "metric_name", "mean_metric", "std_metric", "n_subjects"],
        headers=["Dataset", "Model", "Group", "Metric", "Mean", "Std", "N"],
        colspec=r"p{2.1cm} p{1.8cm} p{2.5cm} p{2.1cm} p{1.2cm} p{1.2cm} p{0.9cm}",
    )

    sample_fmt = sample_df.copy()
    if "std_metric" not in sample_fmt.columns:
        sample_fmt["std_metric"] = ""
    sample_fmt["mean_metric"] = sample_fmt["mean_metric"].map(format_float)
    sample_fmt["std_metric"] = sample_fmt["std_metric"].map(lambda x: format_float(x) if str(x) not in {"", "nan"} else "")
    sample_tex = dataframe_to_longtable(
        sample_fmt,
        columns=["display_name", "group", "metric_name", "mean_metric", "std_metric", "n_subjects"],
        headers=["Model", "Group", "Metric", "Mean", "Std", "N"],
        colspec=r"p{2.1cm} p{2.7cm} p{2.2cm} p{1.2cm} p{1.2cm} p{0.9cm}",
    )

    exact_fmt = exact_df.copy()
    exact_fmt["mean_metric"] = exact_fmt["mean_metric"].map(format_float)
    exact_fmt["std_metric"] = exact_fmt["std_metric"].map(format_float)
    exact_tex = dataframe_to_longtable(
        exact_fmt,
        columns=["dataset", "display_name", "metric_name", "mean_metric", "std_metric", "n_subjects"],
        headers=["Dataset", "Model", "Metric", "Mean", "Std", "N"],
        colspec=r"p{2.2cm} p{2.0cm} p{2.3cm} p{1.2cm} p{1.2cm} p{0.9cm}",
    )

    tex = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{hyperref}
\usepackage{longtable}
\usepackage{array}

\title{Current Auditory EEG Reproduction Audit Note}
\date{2026-06-18}

\begin{document}
\maketitle

\section*{Purpose}
This note was generated to support external review of the current reproduction state. It records the present evaluation setup, the current result tables, and the training-curve figures that are available for audit.

\section*{Why This Document Is Longer Than A Usual Summary}
The current codebase contains more than one evaluation layer:
\begin{itemize}
\item a historical sample-dataset development layer
\item a newer unified \texttt{reference\_splits} layer
\item exact structural ports whose training mode is not identical to the subject-specific baselines
\end{itemize}
Therefore, a short result table is not sufficient. A reviewer needs enough detail to decide whether an unexpected score is caused by:
\begin{itemize}
\item an implementation bug
\item an optimizer or checkpoint-selection issue
\item a data split mismatch
\item a non-comparable training regime
\end{itemize}

\section*{Evaluation Setup}
The current unified reconstruction datasets are:
\begin{itemize}
\item \texttt{hugo\_sample\_tf64}: development dataset
\item \texttt{weissbart\_tf64}: 13-subject reconstruction dataset
\item \texttt{etard\_tf64}: 20-subject reconstruction dataset
\end{itemize}

For the exact structural ports (\texttt{ADT-exact}, \texttt{VLAAI-exact}), the train/validation/test procedure is:
\begin{enumerate}
\item train on the train split
\item evaluate on the validation split every epoch
\item retain the best validation checkpoint
\item stop after a fixed patience without validation improvement
\item evaluate the retained best-validation checkpoint once on the test split
\end{enumerate}

\paragraph{Important comparability boundary.}
The current main table should not yet be interpreted as a perfect apples-to-apples leaderboard because the training modes differ:
\begin{itemize}
\item \texttt{Ridge}, \texttt{CCA}, \texttt{FCNN}, \texttt{CNN}, and \texttt{EEGNet} are currently run as \emph{subject-specific} models
\item \texttt{ADT-exact} and \texttt{VLAAI-exact} are currently run as \emph{pooled multi-subject} models on the training and validation windows, followed by subject-wise test reporting
\end{itemize}

\section*{Implementation Summary}
\subsection*{Implementation files to inspect}
__IMPLEMENTATION_TABLE__

\subsection*{Model protocols}
__PROTOCOL_TABLE__

\subsection*{Run metadata}
__METADATA_TABLE__

\subsection*{Additional audit CSV files}
The following files were generated to support review:
\begin{itemize}
\item \texttt{experiments/summary\_figures/reproduction\_audit\_model\_protocols.csv}
\item \texttt{experiments/summary\_figures/reproduction\_audit\_run\_metadata.csv}
\end{itemize}
The first CSV describes the intended implementation and training mode for each model family. The second CSV describes how many epochs were requested, how many epochs were actually completed, and whether the history comes from a subject-specific average or a single pooled run.

\paragraph{Why the metadata table matters.}
The user explicitly raised concern about very short-looking deep-learning runs. In the current pipeline, a small \texttt{best\_epoch} does not mean the run terminated immediately. It means the best validation checkpoint occurred early and was later restored after patience-based early stopping. The run metadata table is therefore necessary for distinguishing:
\begin{itemize}
\item requested training budget
\item actually completed epochs
\item checkpoint-selection epoch
\end{itemize}

\section*{Current Main Table}
The main unified comparison table is stored in:
\begin{itemize}
\item \texttt{experiments/summary\_figures/unified\_reference\_main\_summary.csv}
\end{itemize}

The current unified results table is reproduced below for direct review:

__UNIFIED_TABLE__

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{../experiments/summary_figures/unified_reference_main_overview.png}
\caption{Current unified comparison on Weissbart and Etard.}
\end{figure}

\section*{Training Curves}
The aggregated validation-curve CSV is:
\begin{itemize}
\item \texttt{experiments/summary\_figures/reproduction\_audit\_training\_curves.csv}
\end{itemize}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{../experiments/summary_figures/reproduction_audit_training_curves.png}
\caption{Validation curves used for audit. For FCNN/CNN/EEGNet, shaded bands indicate participant-to-participant variation. For exact ports, the curves correspond to the single pooled-data training run.}
\end{figure}

\paragraph{What a reviewer should infer from the curves.}
These curves are useful for identifying at least three different kinds of problems:
\begin{itemize}
\item training instability or collapse
\item apparently healthy optimization with poor final generalization
\item suspiciously flat or very short histories that may indicate a stopping-rule or implementation issue
\end{itemize}
They do not prove correctness by themselves, but they make it much easier to separate implementation failure from protocol mismatch.

\section*{Sample Dataset Method Panorama}
The current sample-only method panorama is stored in:
\begin{itemize}
\item \texttt{experiments/summary\_figures/sample\_all\_methods\_summary.csv}
\end{itemize}

For completeness, the current sample-dataset table is reproduced below:

__SAMPLE_TABLE__

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{../experiments/summary_figures/sample_all_methods_overview.png}
\caption{Current comparable methods on the development sample dataset.}
\end{figure}

\section*{Exact-Port Cross-Dataset Table}
The exact structural ports can already be compared across the currently unified datasets:

__EXACT_TABLE__

\section*{Interpretation}
At the current stage, the results support the following:
\begin{itemize}
\item the code paths run end-to-end on the unified datasets
\item \texttt{VLAAI-exact} is currently stronger than \texttt{ADT-exact} on both Weissbart and Etard
\item \texttt{EEGNet} is a strong baseline on the current reconstruction setup
\item \texttt{Ridge} remains competitive and should not be treated as a trivial baseline
\end{itemize}

\paragraph{What should not be over-claimed.}
The current results are meaningful, but they are not yet a final journal table because:
\begin{itemize}
\item pooled exact-port training and subject-specific baseline training are not identical regimes
\item scalar-window regressors and sequence-to-sequence models do not optimize exactly the same training target
\item the historical sample panorama includes shorter-budget development runs
\end{itemize}

\paragraph{Why different implementations can diverge substantially.}
This benchmark is sensitive to details that are easy to overlook:
\begin{itemize}
\item whether the prediction target is a single center/last-sample scalar or a full output sequence
\item whether training is subject-specific or pooled across participants
\item whether validation selects alpha, checkpoint epoch, or both
\item whether the metric is reconstruction correlation, canonical correlation, or match/mismatch accuracy
\item whether window length and hop length match the original implementation
\end{itemize}
Therefore, disagreement between two implementations is not automatically evidence of a bug. It may reflect a real protocol mismatch. This is exactly why the present package exposes both code pointers and run-level metadata.

At the same time, the current benchmark is not yet a final paper-ready leaderboard because:
\begin{itemize}
\item external generalization datasets are not yet integrated into the same adapter layer
\item true competing-speaker AAD datasets are not yet in the same reporting table
\item some sample-dataset baselines were historical short-budget runs and should be interpreted as development comparisons rather than final claims
\end{itemize}

\section*{Recommended Review Order}
A technically informed reviewer should inspect the package in the following order:
\begin{enumerate}
\item \texttt{docs/CURRENT\_EVALUATION\_STATUS.md}
\item \texttt{unified\_reference\_main\_summary.csv} and the corresponding figure
\item \texttt{reproduction\_audit\_training\_curves.csv} and the corresponding figure
\item \texttt{reproduction\_audit\_model\_protocols.csv}
\item \texttt{reproduction\_audit\_run\_metadata.csv}
\item the implementation files for the exact ports and the unified baseline runner
\end{enumerate}

\paragraph{Likely reviewer questions.}
The most important questions are likely to be:
\begin{itemize}
\item whether the exact ports are architecturally faithful enough
\item whether the pooled exact-port training regime is acceptable for the claimed comparison scope
\item whether the subject-specific baselines are being judged under a harder or easier setup than the exact ports
\item whether the observed validation curves look plausible rather than degenerate
\end{itemize}

\section*{Files Provided For External Review}
\begin{itemize}
\item \texttt{docs/CURRENT\_EVALUATION\_STATUS.md}
\item \texttt{experiments/summary\_figures/reproduction\_audit\_model\_protocols.csv}
\item \texttt{experiments/summary\_figures/reproduction\_audit\_run\_metadata.csv}
\item \texttt{experiments/summary\_figures/unified\_reference\_main\_summary.csv}
\item \texttt{experiments/summary\_figures/unified\_reference\_main\_overview.png}
\item \texttt{experiments/summary\_figures/reproduction\_audit\_training\_curves.csv}
\item \texttt{experiments/summary\_figures/reproduction\_audit\_training\_curves.png}
\item \texttt{experiments/summary\_figures/sample\_all\_methods\_summary.csv}
\item \texttt{experiments/summary\_figures/sample\_all\_methods\_overview.png}
\end{itemize}

\end{document}
"""
    tex = tex.replace("__IMPLEMENTATION_TABLE__", implementation_tex)
    tex = tex.replace("__PROTOCOL_TABLE__", protocol_tex)
    tex = tex.replace("__METADATA_TABLE__", metadata_tex)
    tex = tex.replace("__UNIFIED_TABLE__", unified_tex)
    tex = tex.replace("__SAMPLE_TABLE__", sample_tex)
    tex = tex.replace("__EXACT_TABLE__", exact_tex)
    (DOC_DIR / "reproduction_audit_note.tex").write_text(tex, encoding="utf-8")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    curve_df = collect_curve_rows()
    protocol_df = collect_protocol_rows()
    metadata_df = collect_run_metadata()
    implementation_df = collect_implementation_rows()
    unified_df = pd.read_csv(FIG_DIR / "unified_reference_main_summary.csv")
    sample_df = pd.read_csv(FIG_DIR / "sample_all_methods_summary.csv")
    exact_df = pd.read_csv(FIG_DIR / "exact_reference_dataset_summary.csv")
    curve_df.to_csv(FIG_DIR / "reproduction_audit_training_curves.csv", index=False)
    protocol_df.to_csv(FIG_DIR / "reproduction_audit_model_protocols.csv", index=False)
    metadata_df.to_csv(FIG_DIR / "reproduction_audit_run_metadata.csv", index=False)
    plot_curves(curve_df)
    write_tex(protocol_df, metadata_df, implementation_df, unified_df, sample_df, exact_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
