#!/usr/bin/env python3
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

data = [
    (0.710585605, 27),
    (0.680757575, 25),
    (0.665756861, 21),
    (0.637535009, 24),
    (0.632976261, 27),
    (0.631716069, 18),
    (0.628079422, 22),
    (0.617908970, 18),
    (0.583690362, 15),
    (0.528625492, 25),
    (0.524731171, 15),
    (0.516567315, 10),
    (0.505651726, 15),
    (0.487749485, 20),
    (0.472976060,  8),
    (0.466116238,  9),
    (0.464532596,  7),
]

pearsons = np.array([d[0] for d in data])
scores   = np.array([d[1] for d in data])

r, p = stats.pearsonr(scores, pearsons)
m, b = np.polyfit(scores, pearsons, 1)

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(scores, pearsons, color='#3498DB', s=80, zorder=3, label='Subject')

x_line = np.linspace(scores.min() - 1, scores.max() + 1, 200)
ax.plot(x_line, m * x_line + b, color='#E74C3C', linewidth=2, label='Linear fit')

# 95% 置信区间
n = len(scores)
se = np.sqrt(np.sum((pearsons - (m * scores + b))**2) / (n - 2))
t_val = stats.t.ppf(0.975, df=n - 2)
ci = t_val * se * np.sqrt(1/n + (x_line - scores.mean())**2 / np.sum((scores - scores.mean())**2))
ax.fill_between(x_line, m * x_line + b - ci, m * x_line + b + ci,
                color='#E74C3C', alpha=0.15, label='95% CI')

p_str = f'p = {p:.3f}' if p >= 0.001 else f'p < 0.001'
ax.text(0.05, 0.93, f'r = {r:.3f}, {p_str}', transform=ax.transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

ax.set_xlabel('Attention Score', fontsize=13, fontweight='bold')
ax.set_ylabel('Decoding Efficiency', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('subject_pearson_age.png', dpi=300, bbox_inches='tight')
print(f"✓ 已保存: subject_pearson_age.png  (r={r:.3f}, {p_str})")
plt.close()
