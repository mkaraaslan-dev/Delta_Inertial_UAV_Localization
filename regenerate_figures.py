"""
regenerate_figures.py
=====================
Sadece R2 ve Delta MAPE iceren gorselleri yeniden uretir.
Tum egitimi yeniden calistirmak gerekmez.

Guncellenen gorseller:
  results/summary/comparison_plots/
    bar_mean_std.png        <- Delta R2 grafigi kaldirildi, sadece Pos RMSE kaldi
    heatmap_comparison.png  <- Delta R2 satiri kaldirildi
    summary_table.png       <- Delta R2 ve Delta MAPE sutunlari kaldirildi

Kullanim:
  python regenerate_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SUMMARY_DIR  = 'results/summary'
COMP_DIR     = os.path.join(SUMMARY_DIR, 'comparison_plots')
STATS_PATH   = os.path.join(SUMMARY_DIR, 'statistics.xlsx')
ALL_RES_PATH = os.path.join(SUMMARY_DIR, 'all_results.xlsx')

os.makedirs(COMP_DIR, exist_ok=True)

MODEL_NAMES  = ['LSTMModel', 'BiLSTMModel', 'GRUModel', 'AHLSTMModel']
MODEL_LABELS = ['LSTM', 'BiLSTM', 'GRU', 'AHLSTM']
COLORS       = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

# ===========================================================================
# Veri Yukleme
# ===========================================================================
print("Loading statistics...")
df_pos   = pd.read_excel(STATS_PATH, sheet_name='Position_Stats')
df_delta = pd.read_excel(STATS_PATH, sheet_name='Delta_Stats')
df_all   = pd.read_excel(ALL_RES_PATH)
print("Done.\n")

def get_pos(col):
    return [float(df_pos[df_pos['Model'] == m][col].values[0]) for m in MODEL_NAMES]

def get_delta(col):
    return [float(df_delta[df_delta['Model'] == m][col].values[0]) for m in MODEL_NAMES]

# ===========================================================================
# 1. bar_mean_std.png — Pos RMSE Mean +- Std, eski kod stili
# ===========================================================================
means = get_pos('Pos_RMSE_mean')
stds  = get_pos('Pos_RMSE_std')
x     = np.arange(len(MODEL_LABELS))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x, means, yerr=stds, capsize=7,
              color=COLORS, alpha=0.75,
              edgecolor='black', linewidth=0.8,
              error_kw={'linewidth': 1.5, 'capthick': 1.5})
ax.set_xticks(x)
ax.set_xticklabels(MODEL_LABELS, fontsize=11)
ax.set_ylabel('Pos_RMSE', fontsize=10)
ax.set_title('Position RMSE Mean ± Std (9 Flights)', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + max(stds) * 0.02,
            f'{mean:.4f}',
            ha='center', va='bottom', fontsize=9)
ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.25)
plt.tight_layout()
path = os.path.join(COMP_DIR, 'bar_mean_std.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

# ===========================================================================
# 2. heatmap_comparison.png — Delta R2 satiri kaldirildi
# ===========================================================================
HEAT_METRICS = [
    ('pos',   'Pos_RMSE_mean',       'Pos\nRMSE'),
    ('pos',   'Pos_MAE_mean',        'Pos\nMAE'),
    ('delta', 'Delta_RMSE_mean',     'Delta\nRMSE'),
    ('all',   'Best_Test_RMSE_mean', 'Best\nTest'),
]

# Best_Test_RMSE_mean -> all_results'tan hesapla
best_test = []
for m in MODEL_NAMES:
    val = df_all[df_all['Model'] == m]['Best_Test_RMSE'].mean()
    best_test.append(float(val))

heat_data = []
for sheet, col, _ in HEAT_METRICS:
    if sheet == 'pos':
        row = [float(df_pos[df_pos['Model'] == m][col].values[0]) for m in MODEL_NAMES]
    elif sheet == 'delta':
        row = [float(df_delta[df_delta['Model'] == m][col].values[0]) for m in MODEL_NAMES]
    else:
        row = best_test
    heat_data.append(row)

heat_data  = np.array(heat_data)    # (n_metrics, n_models)
heat_labels = [h[2] for h in HEAT_METRICS]

# Normalize her satir kendi icinde
heat_norm = np.zeros_like(heat_data)
for i in range(heat_data.shape[0]):
    row     = heat_data[i]
    rmin    = np.nanmin(row)
    rmax    = np.nanmax(row)
    heat_norm[i] = (row - rmin) / (rmax - rmin + 1e-9)

fig, ax = plt.subplots(figsize=(10, 4.5))
im = ax.imshow(heat_norm, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(range(len(MODEL_LABELS)))
ax.set_xticklabels(MODEL_LABELS, fontsize=11)
ax.set_yticks(range(len(heat_labels)))
ax.set_yticklabels(heat_labels, fontsize=11)

for i in range(len(HEAT_METRICS)):
    for j in range(len(MODEL_NAMES)):
        val      = heat_data[i, j]
        norm_val = heat_norm[i, j]
        txt_col  = 'white' if norm_val > 0.6 else 'black'
        ax.text(j, i, f'{val:.4f}',
                ha='center', va='center',
                fontsize=10, color=txt_col, fontweight='bold')

plt.colorbar(im, ax=ax, label='Normalized value (lower = better)')
ax.set_title('Model Comparison Heatmap — 9-Flight Average',
             fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
path = os.path.join(COMP_DIR, 'heatmap_comparison.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

# ===========================================================================
# 3. summary_table.png — Delta R2 ve Delta MAPE sutunlari kaldirildi
# ===========================================================================
table_data = []
for m_name, m_label in zip(MODEL_NAMES, MODEL_LABELS):
    pos_row   = df_pos[df_pos['Model'] == m_name].iloc[0]
    delta_row = df_delta[df_delta['Model'] == m_name].iloc[0]
    best_val  = df_all[df_all['Model'] == m_name]['Best_Test_RMSE'].mean()

    table_data.append([
        m_label,
        f"{pos_row['Pos_RMSE_mean']:.4f}",
        f"{pos_row['Pos_RMSE_std']:.4f}",
        f"{pos_row['Pos_MAE_mean']:.4f}",
        f"{delta_row['Delta_RMSE_mean']:.6f}",
        f"{delta_row['Delta_MAE_mean']:.6f}",
        f"{best_val:.4f}",
    ])

col_labels = [
    'Model',
    'Pos RMSE\nMean',
    'Pos RMSE\nStd',
    'Pos MAE\nMean',
    'Delta RMSE\nMean',
    'Delta MAE\nMean',
    'Best Test\nRMSE Mean',
]

fig, ax = plt.subplots(figsize=(16, 3.2))
ax.axis('off')

tbl = ax.table(
    cellText   = table_data,
    colLabels  = col_labels,
    cellLoc    = 'center',
    loc        = 'center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.0)

# Header stil
for j in range(len(col_labels)):
    cell = tbl[0, j]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(color='white', fontweight='bold')

# Satir renklendirme
for i in range(1, len(table_data) + 1):
    bg = '#F2F2F2' if i % 2 == 0 else 'white'
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor(bg)

ax.set_title('Summary Table — 9-Flight Average (All Models)',
             fontsize=12, fontweight='bold', pad=16)
plt.tight_layout()
path = os.path.join(COMP_DIR, 'summary_table.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path}")

print(f"\nDone. Updated figures saved to: {COMP_DIR}/")
