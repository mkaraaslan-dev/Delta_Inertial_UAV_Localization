"""
comparison_plots.py
===================
Baseline vs Noise seviyelerini karsilastiran gorseller ve tablo.

Gerekli dosyalar:
  results/summary/statistics.xlsx
  results_noise_low/summary/statistics.xlsx
  results_noise_medium/summary/statistics.xlsx
  results_noise_high/summary/statistics.xlsx

Ciktilar:
  comparison_plots/
    bench_result_pos_rmse_noise.png
    bench_result_delta_rmse_noise.png
    bench_result_heatmap.png
    noise_comparison_table.xlsx
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = 'comparison_plots'
os.makedirs(OUT_DIR, exist_ok=True)

# ===========================================================================
# 0. Ayarlar
# ===========================================================================
SCENARIOS = {
    'Baseline':   'results/summary/statistics.xlsx',
    'Low Noise':  'results_noise_low/summary/statistics.xlsx',
    'Med Noise':  'results_noise_medium/summary/statistics.xlsx',
    'High Noise': 'results_noise_high/summary/statistics.xlsx',
}

MODEL_NAMES  = ['LSTMModel', 'BiLSTMModel', 'GRUModel', 'AHLSTMModel']
MODEL_LABELS = ['LSTM', 'BiLSTM', 'GRU', 'AHLSTM']

# Her senaryo farkli renk
SCENARIO_COLORS = ['#2c7bb6', '#abd9e9', '#fdae61', '#d7191c']
SCENARIO_HATCH  = ['',        '///',     '...',     'xxx'    ]

# ===========================================================================
# 1. Veri Yukleme
# ===========================================================================
data = {}
for label, path in SCENARIOS.items():
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — skipping {label}")
        continue
    data[label] = {
        'pos':   pd.read_excel(path, sheet_name='Position_Stats'),
        'delta': pd.read_excel(path, sheet_name='Delta_Stats'),
    }

available = list(data.keys())
print(f"Loaded: {available}\n")

# ===========================================================================
# 2. Metrik Okuma
# ===========================================================================

def get_metric(scenario, sheet, col):
    df       = data[scenario][sheet]
    col_mean = f'{col}_mean'
    vals = []
    for m in MODEL_NAMES:
        row = df[df['Model'] == m]
        if row.empty or col_mean not in row.columns:
            vals.append(float('nan'))
        else:
            vals.append(float(row[col_mean].values[0]))
    return vals

# ===========================================================================
# 3. Grouped Bar — Her senaryo farkli renk, her bar uzerinde deger etiketi
# ===========================================================================

def grouped_bar_plot(metric_dict, title, ylabel, fmt, fname):
    scenarios = list(metric_dict.keys())
    n_m       = len(MODEL_LABELS)
    n_s       = len(scenarios)
    width     = 0.8 / n_s          # toplam genislik 0.8, esit bolunur
    x         = np.arange(n_m)

    fig, ax = plt.subplots(figsize=(12, 6))

    for s_idx, scenario in enumerate(scenarios):
        vals   = metric_dict[scenario]
        offset = (s_idx - (n_s - 1) / 2) * width
        bars   = ax.bar(
            x + offset, vals,
            width     = width * 0.92,
            color     = SCENARIO_COLORS[s_idx],
            hatch     = SCENARIO_HATCH[s_idx],
            edgecolor = 'black',
            linewidth = 0.7,
            label     = scenario,
            alpha     = 0.88,
        )
        # Her bar uzerine deger yaz
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(
                        v for vv in metric_dict.values() for v in vv
                        if not np.isnan(v)
                    ) * 0.012,
                    fmt.format(val),
                    ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold', rotation=0,
                )

    all_vals = [v for vv in metric_dict.values() for v in vv if not np.isnan(v)]
    ax.set_ylim(0, max(all_vals) * 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=10, loc='upper right',
              framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.25, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


# ---- Pos RMSE ----
if available:
    pos_rmse = {s: get_metric(s, 'pos', 'Pos_RMSE') for s in available}
    grouped_bar_plot(
        pos_rmse,
        title = 'Position RMSE Comparison — Baseline vs Noise Levels (Lower is Better ↓)',
        ylabel= 'Position RMSE (m)',
        fmt   = '{:.3f}',
        fname = os.path.join(OUT_DIR, 'bench_result_pos_rmse_noise.png'),
    )

# ---- Delta RMSE ----
if available:
    delta_rmse = {s: get_metric(s, 'delta', 'Delta_RMSE') for s in available}
    grouped_bar_plot(
        delta_rmse,
        title = 'Delta RMSE Comparison — Baseline vs Noise Levels (Lower is Better ↓)',
        ylabel= 'Delta RMSE (m)',
        fmt   = '{:.5f}',
        fname = os.path.join(OUT_DIR, 'bench_result_delta_rmse_noise.png'),
    )

# ===========================================================================
# 4. Heatmap — Delta R2 kaldirilmis
# ===========================================================================
if 'Baseline' in data:
    pos_df   = data['Baseline']['pos']
    delta_df = data['Baseline']['delta']

    HEAT_METRICS = [
        ('pos',   'Pos_RMSE',   'Pos\nRMSE'),
        ('pos',   'Pos_MAE',    'Pos\nMAE'),
        ('delta', 'Delta_RMSE', 'Delta\nRMSE'),
    ]

    heat_data = np.full((len(MODEL_NAMES), len(HEAT_METRICS)), np.nan)
    for j, (sheet, col, _) in enumerate(HEAT_METRICS):
        df  = data['Baseline'][sheet]
        col_mean = f'{col}_mean'
        for i, m in enumerate(MODEL_NAMES):
            row = df[df['Model'] == m]
            if not row.empty and col_mean in row.columns:
                heat_data[i, j] = float(row[col_mean].values[0])

    heat_norm = np.zeros_like(heat_data)
    for j in range(heat_data.shape[1]):
        col     = heat_data[:, j]
        col_min = np.nanmin(col)
        col_max = np.nanmax(col)
        heat_norm[:, j] = (col - col_min) / (col_max - col_min + 1e-9)

    heat_labels = [h[2] for h in HEAT_METRICS]

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(heat_norm.T, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(MODEL_LABELS)))
    ax.set_xticklabels(MODEL_LABELS, fontsize=11)
    ax.set_yticks(range(len(heat_labels)))
    ax.set_yticklabels(heat_labels, fontsize=11)

    for i in range(len(MODEL_NAMES)):
        for j in range(len(HEAT_METRICS)):
            val       = heat_data[i, j]
            norm_val  = heat_norm[i, j]
            txt_color = 'white' if norm_val > 0.6 else 'black'
            if not np.isnan(val):
                ax.text(i, j, f'{val:.4f}',
                        ha='center', va='center',
                        fontsize=10, color=txt_color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Normalized value (lower = better)')
    ax.set_title('Model Comparison Heatmap — 9-Flight Average',
                 fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, 'bench_result_heatmap.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")

# ===========================================================================
# 5. Noise Karsilastirma Tablosu — Excel
# ===========================================================================
METRICS_FOR_TABLE = [
    ('pos',   'Pos_RMSE',     'Pos RMSE Mean (m)'),
    ('pos',   'Pos_MAE',      'Pos MAE Mean (m)'),
    ('pos',   'Pos_MAPE',     'Pos MAPE Mean'),
    ('delta', 'Delta_RMSE',   'Delta RMSE Mean (m)'),
    ('delta', 'Delta_MAE',    'Delta MAE Mean (m)'),
    ('delta', 'Delta_Avg_RMSE', 'Delta Avg RMSE Mean (m)'),
]

rows = []
for sheet, col, display_name in METRICS_FOR_TABLE:
    for m_name, m_label in zip(MODEL_NAMES, MODEL_LABELS):
        row = {'Metric': display_name, 'Model': m_label}
        for scenario in available:
            df      = data[scenario][sheet]
            col_mean = f'{col}_mean'
            df_row  = df[df['Model'] == m_name]
            if not df_row.empty and col_mean in df_row.columns:
                row[scenario] = round(float(df_row[col_mean].values[0]), 6)
            else:
                row[scenario] = float('nan')

        # Degisim yuzdeleri — baseline varsa
        if 'Baseline' in row and not np.isnan(row.get('Baseline', float('nan'))):
            base = row['Baseline']
            for s in available:
                if s != 'Baseline' and s in row and not np.isnan(row[s]):
                    pct = ((row[s] - base) / base) * 100
                    row[f'{s} Change (%)'] = round(pct, 2)

        rows.append(row)

df_table = pd.DataFrame(rows)

# Sutun sirasi
col_order = ['Metric', 'Model'] + available
pct_cols  = [f'{s} Change (%)' for s in available if s != 'Baseline']
col_order += [c for c in pct_cols if c in df_table.columns]
df_table  = df_table[[c for c in col_order if c in df_table.columns]]

table_path = os.path.join(OUT_DIR, 'noise_comparison_table.xlsx')
with pd.ExcelWriter(table_path, engine='openpyxl') as writer:
    df_table.to_excel(writer, sheet_name='All_Metrics', index=False)

    # Her metrik icin ayri sayfa
    for sheet, col, display_name in METRICS_FOR_TABLE:
        df_sub = df_table[df_table['Metric'] == display_name].copy()
        safe_name = display_name[:30].replace('/', '_').replace(' ', '_')
        df_sub.to_excel(writer, sheet_name=safe_name, index=False)

print(f"Saved: {table_path}")

print(f"\nAll outputs saved to: {OUT_DIR}/")
