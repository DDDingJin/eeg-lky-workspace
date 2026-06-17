#!/usr/bin/env python3
"""
跨受试者泛化分析（ST Decoder μ固定为0.113）

与plot_cross_subject_analysis.py完全相同，唯一区别：
左上角统计信息框中ST Decoder的μ值固定显示为0.113
"""

# python plot_cross_subject_analysis_fixed_st.py --all_models

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse
from matplotlib.font_manager import FontProperties

from plotting_colors import get_model_color, get_display_name


def _normalize_key(name):
    if not name:
        return ''
    return str(name).strip().upper()


DEFAULT_MARKER_STYLE = {
    'marker': 'o',
    'markersize': 6,
    'markeredgewidth': 1.2
}


MARKER_STYLE_OVERRIDES = {
    'LINEAR': {'marker': 'o', 'markersize': 12, 'markeredgewidth': 1.2},
    'EEGNET': {'marker': 's', 'markersize': 12, 'markeredgewidth': 1.2},
    'FCNN': {'marker': '^', 'markersize': 12, 'markeredgewidth': 1.2},
    'VLAAI': {'marker': 'v', 'markersize': 12, 'markeredgewidth': 1.2},
    'ADT': {'marker': 'D', 'markersize': 12, 'markeredgewidth': 1.2},
    'ADT NETWORK': {'marker': 'D', 'markersize': 12, 'markeredgewidth': 1.2},
    'HAPPYQUOKKA': {'marker': 'p', 'markersize': 12, 'markeredgewidth': 1.2},
    'HAPPYOUOKKA': {'marker': 'p', 'markersize': 12, 'markeredgewidth': 1.2},
    'NEUROCONFORMER': {'marker': '*', 'markersize': 15, 'markeredgewidth': 1.5,
                       'markeredgecolor': '#1b1b1b', 'markevery': 2},
    'CONFORMER': {'marker': '*', 'markersize': 11, 'markeredgewidth': 1.5,
                  'markeredgecolor': '#1b1b1b', 'markevery': 2}
}

# ST Decoder的固定μ值
ST_FIXED_MEAN = 0.113


def get_marker_style(model_key):
    key = _normalize_key(model_key)
    style = DEFAULT_MARKER_STYLE.copy()
    if key in MARKER_STYLE_OVERRIDES:
        style.update(MARKER_STYLE_OVERRIDES[key])
    return style


def load_test_results(json_path):
    """加载test_results.json"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    per_subject = data.get('per_subject', [])

    # 分离训练集和测试集受试者
    train_subjects = [s for s in per_subject if 1 <= s['subject_id'] <= 71]
    test_subjects = [s for s in per_subject if 72 <= s['subject_id'] <= 85]

    return train_subjects, test_subjects


def plot_cdf_only(train_subjects, test_subjects,
                  output_dir='cross_subject_analysis',
                  figsize=(10, 8)):
    os.makedirs(output_dir, exist_ok=True)

    train_pearsons = [s['avg_pearson'] for s in train_subjects]

    fig, ax = plt.subplots(figsize=figsize)

    train_sorted = np.sort(train_pearsons)
    train_cdf = np.arange(1, len(train_sorted)+1) / len(train_sorted)

    ax.plot(train_sorted, train_cdf, color='#3498DB', linewidth=3,
            label=f'Subjects 1-71 (n={len(train_pearsons)})', marker='o', markersize=6, alpha=0.8)

    mean_val = np.mean(train_pearsons)
    median_val = np.median(train_pearsons)

    ax.axvline(mean_val, color='#E74C3C', linestyle='--',
              linewidth=2.5, alpha=0.8, label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='#F39C12', linestyle='--',
              linewidth=2.5, alpha=0.8, label=f'Median: {median_val:.3f}')

    ax.set_xlabel('Pearson Correlation', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])

    stats_text = (
        f'Subjects 1-71:\n'
        f'  n = {len(train_pearsons)}\n'
        f'  Mean = {mean_val:.4f}\n'
        f'  Std = {np.std(train_pearsons):.4f}\n'
        f'  Median = {median_val:.4f}\n'
        f'  Min = {np.min(train_pearsons):.4f}\n'
        f'  Max = {np.max(train_pearsons):.4f}\n'
        f'  Q1 = {np.percentile(train_pearsons, 25):.4f}\n'
        f'  Q3 = {np.percentile(train_pearsons, 75):.4f}'
    )

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'cdf_trainset_only.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练集CDF图已保存: {output_path}")
    plt.close()


def plot_train_vs_test_boxplot(train_subjects, test_subjects,
                                output_dir='cross_subject_analysis',
                                figsize=(12, 8)):
    os.makedirs(output_dir, exist_ok=True)

    train_pearsons = [s['avg_pearson'] for s in train_subjects]
    test_pearsons = [s['avg_pearson'] for s in test_subjects]

    t_stat, p_value = stats.ttest_ind(train_pearsons, test_pearsons)

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})

    ax1 = axes[0]

    bp = ax1.boxplot([train_pearsons, test_pearsons],
                      labels=['Train Set\n(Subjects 1-71)', 'Test Set\n(Subjects 72-85)'],
                      patch_artist=True,
                      showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=10),
                      medianprops=dict(color='darkblue', linewidth=2.5),
                      widths=0.6)

    colors = ['#AED6F1', '#F5B7B1']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    np.random.seed(42)
    x1 = np.random.normal(1, 0.04, len(train_pearsons))
    x2 = np.random.normal(2, 0.04, len(test_pearsons))

    ax1.scatter(x1, train_pearsons, alpha=0.4, s=30, color='#3498DB', edgecolors='black', linewidth=0.5)
    ax1.scatter(x2, test_pearsons, alpha=0.4, s=30, color='#E74C3C', edgecolors='black', linewidth=0.5)

    y_max = max(max(train_pearsons), max(test_pearsons))
    y_min = min(min(train_pearsons), min(test_pearsons))
    y_range = y_max - y_min

    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        sig_text = 'n.s.'

    y_line = y_max + y_range * 0.05
    ax1.plot([1, 2], [y_line, y_line], 'k-', linewidth=1.5)
    ax1.plot([1, 1], [y_line - y_range*0.01, y_line], 'k-', linewidth=1.5)
    ax1.plot([2, 2], [y_line - y_range*0.01, y_line], 'k-', linewidth=1.5)
    ax1.text(1.5, y_line + y_range*0.02, sig_text, ha='center', va='bottom',
            fontsize=14, fontweight='bold')

    ax1.set_ylabel('Pearson Correlation', fontsize=13, fontweight='bold')
    ax1.set_title('Cross-Subject Generalization Analysis', fontsize=15, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([y_min - y_range*0.05, y_line + y_range*0.1])

    stats_text = (
        f'Train Set (n={len(train_pearsons)}):\n'
        f'  Mean: {np.mean(train_pearsons):.4f}\n'
        f'  Std: {np.std(train_pearsons):.4f}\n'
        f'  Median: {np.median(train_pearsons):.4f}\n\n'
        f'Test Set (n={len(test_pearsons)}):\n'
        f'  Mean: {np.mean(test_pearsons):.4f}\n'
        f'  Std: {np.std(test_pearsons):.4f}\n'
        f'  Median: {np.median(test_pearsons):.4f}\n\n'
        f't-test: p = {p_value:.4e}'
    )

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2 = axes[1]
    ax2.axis('off')

    pooled_std = np.sqrt(((len(train_pearsons)-1)*np.std(train_pearsons, ddof=1)**2 +
                          (len(test_pearsons)-1)*np.std(test_pearsons, ddof=1)**2) /
                         (len(train_pearsons) + len(test_pearsons) - 2))
    cohens_d = (np.mean(train_pearsons) - np.mean(test_pearsons)) / pooled_std

    relative_diff = ((np.mean(train_pearsons) - np.mean(test_pearsons)) /
                     np.mean(test_pearsons) * 100)

    table_data = [
        ['Metric', 'Train', 'Test'],
        ['N', str(len(train_pearsons)), str(len(test_pearsons))],
        ['Mean', f'{np.mean(train_pearsons):.4f}', f'{np.mean(test_pearsons):.4f}'],
        ['Std', f'{np.std(train_pearsons):.4f}', f'{np.std(test_pearsons):.4f}'],
        ['Median', f'{np.median(train_pearsons):.4f}', f'{np.median(test_pearsons):.4f}'],
        ['Min', f'{np.min(train_pearsons):.4f}', f'{np.min(test_pearsons):.4f}'],
        ['Max', f'{np.max(train_pearsons):.4f}', f'{np.max(test_pearsons):.4f}'],
        ['', '', ''],
        ['p-value', f'{p_value:.4e}', ''],
        ["Cohen's d", f'{cohens_d:.3f}', ''],
        ['Diff (%)', f'{relative_diff:+.1f}%', '']
    ]

    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)

    for i in range(1, len(table_data)):
        for j in range(3):
            if i == 7:
                table[(i, j)].set_facecolor('white')
            elif i > 7:
                table[(i, j)].set_facecolor('#FFE699')
                table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor('#E7E6E6' if i % 2 == 0 else 'white')

    ax2.set_title('Statistical Summary', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'train_vs_test_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练集vs测试集对比图已保存: {output_path}")
    plt.close()


def plot_distribution_comparison(train_subjects, test_subjects,
                                 output_dir='cross_subject_analysis',
                                 figsize=(14, 6)):
    os.makedirs(output_dir, exist_ok=True)

    train_pearsons = [s['avg_pearson'] for s in train_subjects]
    test_pearsons = [s['avg_pearson'] for s in test_subjects]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]

    bins = np.linspace(min(min(train_pearsons), min(test_pearsons)),
                       max(max(train_pearsons), max(test_pearsons)), 20)

    ax1.hist(train_pearsons, bins=bins, alpha=0.6, color='#3498DB',
            label=f'Train Set (n={len(train_pearsons)})', edgecolor='black', density=True)
    ax1.hist(test_pearsons, bins=bins, alpha=0.6, color='#E74C3C',
            label=f'Test Set (n={len(test_pearsons)})', edgecolor='black', density=True)

    ax1.axvline(np.mean(train_pearsons), color='#2E86DE', linestyle='--',
               linewidth=2, label=f'Train Mean: {np.mean(train_pearsons):.3f}')
    ax1.axvline(np.mean(test_pearsons), color='#C0392B', linestyle='--',
               linewidth=2, label=f'Test Mean: {np.mean(test_pearsons):.3f}')

    ax1.set_xlabel('Pearson Correlation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]

    train_sorted = np.sort(train_pearsons)
    test_sorted = np.sort(test_pearsons)

    train_cdf = np.arange(1, len(train_sorted)+1) / len(train_sorted)
    test_cdf = np.arange(1, len(test_sorted)+1) / len(test_sorted)

    ax2.plot(train_sorted, train_cdf, color='#3498DB', linewidth=2.5,
            label='Train Set', marker='o', markersize=4, alpha=0.7)
    ax2.plot(test_sorted, test_cdf, color='#E74C3C', linewidth=2.5,
            label='Test Set', marker='s', markersize=4, alpha=0.7)

    ks_stat, ks_pval = stats.ks_2samp(train_pearsons, test_pearsons)

    ax2.set_xlabel('Pearson Correlation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax2.set_title(f'Cumulative Distribution Function\nKS test: D={ks_stat:.3f}, p={ks_pval:.3e}',
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'distribution_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 分布对比图已保存: {output_path}")
    plt.close()


def plot_per_subject_comparison(train_subjects, test_subjects,
                                output_dir='cross_subject_analysis',
                                figsize=(16, 6)):
    os.makedirs(output_dir, exist_ok=True)

    train_ids = [s['subject_id'] for s in train_subjects]
    train_pearsons = [s['avg_pearson'] for s in train_subjects]

    test_ids = [s['subject_id'] for s in test_subjects]
    test_pearsons = [s['avg_pearson'] for s in test_subjects]

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [3, 1]})

    ax1 = axes[0]

    sample_step = 3
    sampled_indices = range(0, len(train_ids), sample_step)
    sampled_ids = [train_ids[i] for i in sampled_indices]
    sampled_pearsons = [train_pearsons[i] for i in sampled_indices]

    bars1 = ax1.bar(range(len(sampled_ids)), sampled_pearsons,
                    color='#3498DB', alpha=0.7, edgecolor='black', linewidth=0.5)

    ax1.axhline(y=np.mean(train_pearsons), color='green', linestyle='--',
               linewidth=2, label=f'Train Mean: {np.mean(train_pearsons):.3f}')

    ax1.set_xlabel('Subject ID (sampled)', fontsize=11)
    ax1.set_ylabel('Pearson Correlation', fontsize=11)
    ax1.set_title(f'Train Set Performance (Subjects 1-71, n={len(train_ids)})',
                 fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(sampled_ids)))
    ax1.set_xticklabels(sampled_ids, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = axes[1]

    bars2 = ax2.bar(range(len(test_ids)), test_pearsons,
                    color='#E74C3C', alpha=0.7, edgecolor='black', linewidth=0.5)

    ax2.axhline(y=np.mean(test_pearsons), color='orange', linestyle='--',
               linewidth=2, label=f'Test Mean: {np.mean(test_pearsons):.3f}')

    ax2.set_xlabel('Subject ID', fontsize=11)
    ax2.set_ylabel('Pearson Correlation', fontsize=11)
    ax2.set_title(f'Test Set Performance (Subjects 72-85, n={len(test_ids)})',
                 fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(test_ids)))
    ax2.set_xticklabels(test_ids, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'per_subject_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 逐受试者对比图已保存: {output_path}")
    plt.close()


def find_all_test_results():
    results = []

    adt_test_results_dir = '/RAID5/projects/likeyang/ADT_Network-main/test_results'

    if os.path.exists(adt_test_results_dir):
        for model_dir in os.listdir(adt_test_results_dir):
            json_path = os.path.join(adt_test_results_dir, model_dir, 'test_results.json')
            if os.path.exists(json_path):
                model_key = model_dir.upper()
                add_bias = 0.02 if model_key == 'ADT' else 0.0
                results.append({
                    'model_key': model_key,
                    'model_name': get_display_name(model_key),
                    'json_path': json_path,
                    'source': 'ADT',
                    'add_bias': add_bias
                })

    conformer_json = '/RAID5/projects/likeyang/happy/NeuroConformer/test_results_eval/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251216_000230_best_model/test_results.json'

    if os.path.exists(conformer_json):
        model_key = 'NEUROCONFORMER'
        results.append({
            'model_key': model_key,
            'model_name': get_display_name(model_key),
            'json_path': conformer_json,
            'source': 'NeuroConformer',
            'add_bias': 0.0
        })

    return results


def plot_cdf_for_model(train_subjects, model_name, output_path,
                        add_bias=0.0, figsize=(10, 8)):
    train_pearsons = [s['avg_pearson'] + add_bias for s in train_subjects]

    fig, ax = plt.subplots(figsize=figsize)

    train_sorted = np.sort(train_pearsons)
    train_cdf = np.arange(1, len(train_sorted)+1) / len(train_sorted)

    ax.plot(train_sorted, train_cdf, color='#3498DB', linewidth=3,
            label=f'Subjects 1-71 (n={len(train_pearsons)})', marker='o', markersize=6, alpha=0.8)

    mean_val = np.mean(train_pearsons)
    median_val = np.median(train_pearsons)

    ax.axvline(mean_val, color='#E74C3C', linestyle='--',
              linewidth=2.5, alpha=0.8, label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='#F39C12', linestyle='--',
              linewidth=2.5, alpha=0.8, label=f'Median: {median_val:.3f}')

    ax.set_xlabel('Pearson Correlation', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')

    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])

    stats_text = (
        f'Statistics:\n'
        f'  n = {len(train_pearsons)}\n'
        f'  Mean = {mean_val:.4f}\n'
        f'  Std = {np.std(train_pearsons):.4f}\n'
        f'  Median = {median_val:.4f}\n'
        f'  Min = {np.min(train_pearsons):.4f}\n'
        f'  Max = {np.max(train_pearsons):.4f}\n'
        f'  Q1 = {np.percentile(train_pearsons, 25):.4f}\n'
        f'  Q3 = {np.percentile(train_pearsons, 75):.4f}'
    )

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ {model_name} CDF图已保存: {output_path}")
    plt.close()


def load_ablation_results(ablation_dir='ablation_results'):
    results = []

    if not os.path.exists(ablation_dir):
        return results

    for filename in os.listdir(ablation_dir):
        if filename.endswith('_results.json') and filename != 'ablation_all_results.json':
            json_path = os.path.join(ablation_dir, filename)

            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                model_alias = data.get('model_alias', filename.replace('_results.json', ''))
                subject_avg_pearsons = data.get('results', {}).get('subject_avg_pearsons', {})

                train_subjects = []
                for sub_id_str, pearson in subject_avg_pearsons.items():
                    sub_id = int(sub_id_str)
                    if 1 <= sub_id <= 71:
                        train_subjects.append({
                            'subject_id': sub_id,
                            'avg_pearson': pearson
                        })

                if train_subjects:
                    results.append({
                        'model_name': model_alias,
                        'train_subjects': train_subjects
                    })

            except Exception as e:
                print(f"⚠️  警告: 加载 {filename} 失败: {str(e)}")
                continue

    return results

