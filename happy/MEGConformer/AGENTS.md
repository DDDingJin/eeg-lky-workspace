# Repository Guidelines

## Project Structure & Module Organization
Core model code sits in `models/` (`FFT_block_conformer_v2.py`, `ConformerLayers.py`, `SubLayers.py`). Data loading, metrics, and logging helpers live in `util/` (`dataset.py`, `cal_pearson.py`, `logger.py`). Top-level entry points include `train.py`, `test_model.py`, and the ablation scripts (`ablation_inference.py`, `ablation_plot*.py`). Store raw EEG/envelope `.npy` files under `data/`, experiment outputs under `test_results/` and `test_results_eval/`, and publishable visuals inside `ablation_plots/`, `comparison_results/`, or `prediction_analysis/`. Consult `SCRIPTS_QUICK_REFERENCE.md` whenever adding a new executable.

## Build, Test, and Development Commands
- `CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py --use_ddp` starts the baseline run; append flags such as `--no_skip_cnn`, `--n_layers 6`, or `--no_se` for ablations.
- `python test_model.py --checkpoint test_results/<exp>/best_model.pt --gpu 0` evaluates a checkpoint and regenerates Pearson boxes.
- `python ablation_inference.py --models Exp-00 Exp-01-无CNN --gpu 0` followed by `python ablation_plot.py` or `ablation_plot_violin.py` rebuilds the figures in `ablation_plots/`.
- `python plot_tensorboard.py --logdir test_results/<exp>/tb_logs` extracts smoothed training curves.
- `python plot_cross_subject_analysis.py --all_models` or `--ablation --grouped` keeps the published CDF dashboards current; `python compare_all_models.py` reruns the statistical summary.

## Coding Style & Naming Conventions
Use PEP 8 with 4-space indentation, `snake_case` functions, and `CapWords` classes (see `util/dataset.py`). Keep docstrings on public classes/methods and add short comments only for EEG-specific heuristics. Experiments should follow the `Exp-##-descriptor` pattern referenced by plotting scripts, and new CLI options must be mirrored in `README.md` plus `SCRIPTS_QUICK_REFERENCE.md`.

## Testing Guidelines
Every training or architectural tweak requires `python test_model.py ...` against at least one checkpoint; commit the resulting metrics under `test_results_eval/<exp>.json` and plots beside the checkpoint. When editing ablation or comparison code, also run `python plot_cross_subject_analysis.py --all_models` and verify outputs land in `comparison_results/`. Keep lightweight regression notebooks in `prediction_analysis/` for shape or metric sanity checks and describe new expectations directly inside the scripts.

## Commit & Pull Request Guidelines
Git history favors short, present-tense summaries (`加入readme`, `加了几个图`). Follow that spirit: one headline action, optional experiment ID (`修复Exp-08训练`). Pull requests should link issues when possible, list the exact commands executed, point reviewers to the produced folders in `test_results/` or `ablation_results/`, and attach thumbnails or paths for new plots. Call out any breaking flag changes upfront and confirm docs were refreshed.
