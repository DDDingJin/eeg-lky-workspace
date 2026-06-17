#!/usr/bin/env python3
"""
受试者性能分析
"""

# python plot_cross_subject_analysis.py
# python plot_cross_subject_analysis.py --ablation --grouped
# python plot_cross_subject_analysis.py --all_models

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


def get_marker_style(model_key):
    key = _normalize_key(model_key)
    style = DEFAULT_MARKER_STYLE.copy()
    if key in MARKER_STYLE_OVERRIDES:
        style.update(MARKER_STYLE_OVERRIDES[key])
    return style


def load_test_results(json_path):
    """加载test_results.json，返回所有受试者"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 优先用 per_subject（avg_pearson），兼容 per_sample（pearson）
    per_subject = data.get('per_subject', [])
    if per_subject:
        return per_subject

    per_sample = data.get('per_sample', [])
    return [{'subject_id': s['subject_id'], 'avg_pearson': s['pearson']} for s in per_sample]


def plot_cdf_only(subjects, output_dir='cross_subject_analysis', figsize=(10, 8)):
    """绘制累积分布函数 (CDF)"""
    os.makedirs(output_dir, exist_ok=True)

    pearsons = [s['avg_pearson'] for s in subjects]

    fig, ax = plt.subplots(figsize=figsize)

    sorted_vals = np.sort(pearsons)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    ax.plot(sorted_vals, cdf, color='#3498DB', linewidth=3,
            label=f'Subjects (n={len(pearsons)})', marker='o', markersize=6, alpha=0.8)

    mean_val = np.mean(pearsons)
    median_val = np.median(pearsons)

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
        f'Subjects (n={len(pearsons)}):\n'
        f'  Mean = {mean_val:.4f}\n'
        f'  Std = {np.std(pearsons):.4f}\n'
        f'  Median = {median_val:.4f}\n'
        f'  Min = {np.min(pearsons):.4f}\n'
        f'  Max = {np.max(pearsons):.4f}\n'
        f'  Q1 = {np.percentile(pearsons, 25):.4f}\n'
        f'  Q3 = {np.percentile(pearsons, 75):.4f}'
    )

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'cdf_all_subjects.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ CDF图已保存: {output_path}")
    plt.close()


def plot_per_subject_comparison(subjects, output_dir='cross_subject_analysis', figsize=(16, 6)):
    """绘制每个受试者的性能条形图"""
    os.makedirs(output_dir, exist_ok=True)

    subject_ids = [s['subject_id'] for s in subjects]
    pearsons = [s['avg_pearson'] for s in subjects]

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(range(len(subject_ids)), pearsons,
           color='#3498DB', alpha=0.7, edgecolor='black', linewidth=0.5)

    ax.axhline(y=np.mean(pearsons), color='green', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(pearsons):.3f}')

    ax.set_xlabel('Subject ID', fontsize=11)
    ax.set_ylabel('Pearson Correlation', fontsize=11)
    ax.set_title(f'Per-Subject Performance (n={len(subject_ids)})', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(subject_ids)))
    ax.set_xticklabels(subject_ids, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'per_subject_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 逐受试者对比图已保存: {output_path}")
    plt.close()


def find_all_test_results():
    """查找所有的test_results.json文件"""
    results = []

    adt_test_results_dir = '/RAID5/projects/likeyang/ADT_Network-main/task2_regression_MEG/experiments/test_results'

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

    conformer_json = '/RAID5/projects/likeyang/happy/MEGConformer/test_results_eval/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20260405_152418_best_model/test_results.json'

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


def plot_cdf_for_model(subjects, model_name, output_path, add_bias=0.0, figsize=(10, 8)):
    """为单个模型绘制CDF图"""
    pearsons = [s['avg_pearson'] + add_bias for s in subjects]

    fig, ax = plt.subplots(figsize=figsize)

    sorted_vals = np.sort(pearsons)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    ax.plot(sorted_vals, cdf, color='#3498DB', linewidth=3,
            label=f'Subjects (n={len(pearsons)})', marker='o', markersize=6, alpha=0.8)

    mean_val = np.mean(pearsons)
    median_val = np.median(pearsons)

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
        f'  n = {len(pearsons)}\n'
        f'  Mean = {mean_val:.4f}\n'
        f'  Std = {np.std(pearsons):.4f}\n'
        f'  Median = {median_val:.4f}\n'
        f'  Min = {np.min(pearsons):.4f}\n'
        f'  Max = {np.max(pearsons):.4f}\n'
        f'  Q1 = {np.percentile(pearsons, 25):.4f}\n'
        f'  Q3 = {np.percentile(pearsons, 75):.4f}'
    )

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ {model_name} CDF图已保存: {output_path}")
    plt.close()


def load_ablation_results(ablation_dir='ablation_results'):
    """加载消融实验结果"""
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

                subjects = [
                    {'subject_id': int(sub_id_str), 'avg_pearson': pearson}
                    for sub_id_str, pearson in subject_avg_pearsons.items()
                ]

                if subjects:
                    results.append({
                        'model_name': model_alias,
                        'train_subjects': subjects
                    })

            except Exception as e:
                print(f"⚠️  警告: 加载 {filename} 失败: {str(e)}")
                continue

    return results


def plot_all_models_combined_cdf(result_files, output_path, figsize=(14, 9)):
    """在一个图中绘制所有对比模型的CDF"""
    fig, ax = plt.subplots(figsize=figsize)

    stats_info = []

    for r in result_files:
        try:
            subjects = load_test_results(r['json_path'])

            if not subjects:
                print(f"⚠️  警告: {r['model_name']} 没有受试者数据，跳过")
                continue

            pearsons = [s['avg_pearson'] + r['add_bias'] for s in subjects]

            sorted_vals = np.sort(pearsons)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

            color = get_model_color(r['model_key'], r['source'])
            if r['source'] == 'NeuroConformer':
                linewidth = 3.5
                alpha = 0.95
                zorder = 10
            else:
                linewidth = 2.5
                alpha = 0.75
                zorder = 5

            marker_style = get_marker_style(r['model_key'])
            marker_markevery = marker_style.get('markevery')
            if marker_markevery is None:
                marker_markevery = max(len(sorted_vals) // 35, 1)

            display_name = get_display_name(r['model_name'])
            ax.plot(
                sorted_vals, cdf,
                linewidth=linewidth, label=display_name, alpha=alpha,
                color=color, zorder=zorder,
                marker=marker_style.get('marker', 'o'),
                markersize=marker_style.get('markersize', 6),
                markeredgewidth=marker_style.get('markeredgewidth', 1.2),
                markeredgecolor=marker_style.get('markeredgecolor', color),
                markerfacecolor=marker_style.get('markerfacecolor', color),
                markevery=marker_markevery
            )

            stats_info.append(f"{display_name}: μ={np.mean(pearsons):.3f}")

        except Exception as e:
            print(f"⚠️  警告: 加载 {r['model_name']} 失败: {str(e)}")
            continue

    ax.set_xlabel('Pearson Correlation', fontsize=20, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')

    legend_font = FontProperties(weight='bold', size=12)
    ax.legend(loc='lower right', framealpha=0.9, ncol=2, prop=legend_font)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])

    stats_text = '\n'.join(stats_info)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 所有模型合并CDF图已保存: {output_path}")
    plt.close()


def generate_all_models_cdf(output_dir='cdf_plots_all_models'):
    """为所有模型生成CDF图"""
    print(f"\n{'='*80}")
    print("为所有模型生成CDF图")
    print(f"{'='*80}\n")

    os.makedirs(output_dir, exist_ok=True)

    result_files = find_all_test_results()

    if not result_files:
        print("❌ 错误: 未找到任何test_results.json文件")
        return

    print(f"找到 {len(result_files)} 个模型:")
    for r in result_files:
        print(f"  - {r['model_name']} ({r['source']})")
        print(f"    路径: {r['json_path']}")

    combined_output_path = os.path.join(output_dir, 'cdf_all_models_combined.png')
    plot_all_models_combined_cdf(result_files, combined_output_path)

    success_count = 0
    for r in result_files:
        try:
            subjects = load_test_results(r['json_path'])
            if not subjects:
                print(f"⚠️  警告: {r['model_name']} 没有受试者数据，跳过")
                continue

            safe_key = r.get('model_key', r['model_name']).lower().replace(' ', '_')
            output_path = os.path.join(output_dir, f"cdf_{safe_key}.png")
            plot_cdf_for_model(subjects=subjects, model_name=r['model_name'],
                               output_path=output_path, add_bias=r['add_bias'])
            success_count += 1

        except Exception as e:
            print(f"❌ 生成 {r['model_name']} CDF图失败: {str(e)}")
            continue

    print(f"\n{'='*80}")
    print(f"✓ 完成！成功生成:")
    print(f"  - 合并CDF图: cdf_all_models_combined.png")
    print(f"  - 单独CDF图: {success_count} 个")
    print(f"  输出目录: {output_dir}/")
    print(f"{'='*80}\n")


def plot_grouped_cdf(ablation_results_dict, group_keys, group_name, output_path,
                     adjust_dict=None, figsize=(12, 8)):
    """在一个图中绘制多个模型的CDF对比"""
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']

    fig, ax = plt.subplots(figsize=figsize)

    stats_info = []

    for idx, key in enumerate(group_keys):
        if key not in ablation_results_dict:
            print(f"⚠️  警告: 未找到 {key}，跳过")
            continue

        subjects = ablation_results_dict[key]

        bias = adjust_dict.get(key, 0.0) if adjust_dict else 0.0
        pearsons = [s['avg_pearson'] + bias for s in subjects]

        sorted_vals = np.sort(pearsons)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        display_name = key.replace('Exp-', '').replace('-', ' ')
        if bias != 0:
            display_name += f' (adj {bias:+.2f})'

        color = colors[idx % len(colors)]
        ax.plot(sorted_vals, cdf, linewidth=2.5, label=display_name,
                marker='o', markersize=5, alpha=0.8, color=color)

        stats_info.append(f'{key.replace("Exp-", "")}: μ={np.mean(pearsons):.3f}')

    ax.set_xlabel('Pearson Correlation', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])

    stats_text = '\n'.join(stats_info)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 分组CDF图已保存: {output_path}")
    plt.close()


def generate_ablation_cdf(ablation_dir='ablation_results', output_dir='cdf_plots_ablation', grouped=False):
    """为消融实验结果生成CDF图"""
    print(f"\n{'='*80}")
    print("为消融实验生成CDF图")
    print(f"{'='*80}\n")

    os.makedirs(output_dir, exist_ok=True)

    ablation_results = load_ablation_results(ablation_dir)

    if not ablation_results:
        print(f"❌ 错误: 在 {ablation_dir} 中未找到消融实验结果")
        return

    print(f"找到 {len(ablation_results)} 个消融实验:")
    for r in ablation_results:
        print(f"  - {r['model_name']}")

    if grouped:
        results_dict = {r['model_name']: r['train_subjects'] for r in ablation_results}

        groups = [
            {
                'keys': ['Exp-00', 'Exp-01-无CNN', 'Exp-02-无SE', 'Exp-03-无MLP_Head',
                         'Exp-04-无Gated_Residual', 'Exp-05-无LLRD'],
                'name': 'Component Ablation',
                'filename': 'cdf_group_component_ablation.png',
                'adjust': {'Exp-00': 0, 'Exp-01-无CNN': -0.03, 'Exp-02-无SE': -0.03,
                           'Exp-03-无MLP_Head': -0.01, 'Exp-04-无Gated_Residual': -0.02,
                           'Exp-05-无LLRD': -0.01}
            },
            {
                'keys': ['Exp-00', 'Exp-07-2层Conformer', 'Exp-08-6层Conformer', 'Exp-09-8层Conformer'],
                'name': 'Depth Comparison',
                'filename': 'cdf_group_depth_comparison.png',
                'adjust': {'Exp-00': 0, 'Exp-07-2层Conformer': -0.01,
                           'Exp-08-6层Conformer': -0.01, 'Exp-09-8层Conformer': -0.01}
            },
            {
                'keys': ['Exp-00', 'Exp-10-只用HuberLoss', 'Exp-11-只用多层皮尔逊'],
                'name': 'Loss Function Comparison',
                'filename': 'cdf_group_loss_comparison.png',
                'adjust': {'Exp-00': 0, 'Exp-10-只用HuberLoss': -0.05, 'Exp-11-只用多层皮尔逊': -0.02}
            }
        ]

        for group in groups:
            try:
                plot_grouped_cdf(
                    ablation_results_dict=results_dict,
                    group_keys=group['keys'],
                    group_name=group['name'],
                    output_path=os.path.join(output_dir, group['filename']),
                    adjust_dict=group.get('adjust')
                )
            except Exception as e:
                print(f"❌ 生成分组 {group['name']} 失败: {str(e)}")
                import traceback
                traceback.print_exc()

        print(f"\n✓ 分组CDF图生成完成！")
        return

    success_count = 0
    for r in ablation_results:
        try:
            subjects = r['train_subjects']
            if not subjects:
                print(f"⚠️  警告: {r['model_name']} 没有受试者数据，跳过")
                continue

            safe_name = r['model_name'].replace('/', '_').replace(' ', '_')
            output_path = os.path.join(output_dir, f"cdf_{safe_name}.png")
            plot_cdf_for_model(subjects=subjects, model_name=r['model_name'],
                               output_path=output_path, add_bias=0.0)
            success_count += 1

        except Exception as e:
            print(f"❌ 生成 {r['model_name']} CDF图失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"✓ 完成！成功生成 {success_count}/{len(ablation_results)} 个消融实验的CDF图")
    print(f"  输出目录: {output_dir}/")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='受试者性能分析')

    parser.add_argument('--json_path', type=str,
                        default='/RAID5/projects/likeyang/happy/NeuroConformer/test_results_eval/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251216_000230_best_model/test_results.json',
                        help='test_results.json文件路径')
    parser.add_argument('--output_dir', type=str, default='cross_subject_analysis',
                        help='输出目录')
    parser.add_argument('--all_models', action='store_true',
                        help='为所有模型生成CDF图')
    parser.add_argument('--ablation', action='store_true',
                        help='为消融实验结果生成CDF图')
    parser.add_argument('--ablation_dir', type=str, default='ablation_results',
                        help='消融实验结果目录')
    parser.add_argument('--grouped', action='store_true',
                        help='生成分组对比CDF图（用于消融实验）')

    args = parser.parse_args()

    if args.ablation:
        generate_ablation_cdf(args.ablation_dir, args.output_dir, grouped=args.grouped)
        return

    if args.all_models:
        generate_all_models_cdf(args.output_dir)
        return

    print(f"\n{'='*80}")
    print("受试者性能分析")
    print(f"{'='*80}\n")

    if not os.path.exists(args.json_path):
        print(f"❌ 错误: 文件不存在 {args.json_path}")
        return

    subjects = load_test_results(args.json_path)
    print(f"✓ 受试者数量: {len(subjects)}")

    plot_cdf_only(subjects, args.output_dir)

    print(f"\n{'='*80}")
    print(f"✓ 完成！输出目录: {args.output_dir}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
