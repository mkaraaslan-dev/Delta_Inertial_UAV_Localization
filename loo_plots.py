"""
loo_plots.py
============
LOO sonuclari icin tum gorsellestime fonksiyonlari.
Per-condition ve combined plot'lar.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS       = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
MODEL_LABELS = ['LSTM', 'BiLSTM', 'GRU', 'AHLSTM']
MODEL_NAMES  = ['LSTMModel', 'BiLSTMModel', 'GRUModel', 'AHLSTMModel']
NOISE_LEVELS = ['baseline', 'low', 'medium', 'high']
NOISE_LABELS = ['Baseline', 'Low', 'Medium', 'High']
NOISE_COLORS = ['#2c7bb6', '#abd9e9', '#fdae61', '#d7191c']

# ===========================================================================
# PER-CONDITION PLOTS
# ===========================================================================

def plot_loss_curves(loss_store, noise_label, out_dir, num_epochs=250):
    """4 subplot, her model icin 9 fold loss egrisi."""
    epochs      = np.arange(1, num_epochs + 1)
    fold_colors = plt.cm.tab10(np.linspace(0, 0.9, 9))

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    for ax, model_name, m_color in zip(axes, MODEL_NAMES, COLORS):
        if model_name not in loss_store or not loss_store[model_name]:
            continue
        for f_idx, (tr_loss, te_loss) in enumerate(loss_store[model_name]):
            ax.plot(epochs[:len(tr_loss)], tr_loss,
                    color=fold_colors[f_idx], linewidth=1.0, alpha=0.6, linestyle='-')
            ax.plot(epochs[:len(te_loss)], te_loss,
                    color=fold_colors[f_idx], linewidth=1.0, alpha=0.6, linestyle='--')

        tr_mean = np.mean([l[0] for l in loss_store[model_name]], axis=0)
        te_mean = np.mean([l[1] for l in loss_store[model_name]], axis=0)
        ax.plot(epochs[:len(tr_mean)], tr_mean,
                color='black', linewidth=2.5, linestyle='-',  label='Train Mean')
        ax.plot(epochs[:len(te_mean)], te_mean,
                color='black', linewidth=2.5, linestyle='--', label='Test Mean')

        handles = [plt.Line2D([0],[0], color=fold_colors[i], linewidth=1.5,
                               label=f'Fold {i+1}') for i in range(9)]
        handles += [
            plt.Line2D([0],[0], color='black', linewidth=2, linestyle='-',  label='Train Mean'),
            plt.Line2D([0],[0], color='black', linewidth=2, linestyle='--', label='Test Mean'),
        ]
        ax.legend(handles=handles, fontsize=7, ncol=2, loc='upper right')
        ax.set_title(model_name.replace('Model',''), fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE Loss')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'LOO Loss Curves ({noise_label}) — Train / Test | Bold = Mean across Folds',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, f'loss_curves_{noise_label.lower()}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_fold_rmse(df_all, noise_label, out_dir):
    """Fold bazli Pos RMSE bar chart — gruplu."""
    fold_ids = sorted(df_all['Fold'].unique())
    n_folds  = len(fold_ids)
    n_models = len(MODEL_NAMES)
    width    = 0.18
    x        = np.arange(n_folds)

    fig, ax = plt.subplots(figsize=(14, 6))

    for m_idx, (model_name, m_color) in enumerate(zip(MODEL_NAMES, COLORS)):
        vals   = []
        for f in fold_ids:
            row = df_all[(df_all['Model'] == model_name) & (df_all['Fold'] == f)]
            vals.append(float(row['Pos_RMSE'].values[0]) if len(row) else float('nan'))
        offset = (m_idx - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               color=m_color, alpha=0.85, edgecolor='black',
               linewidth=0.6, label=model_name.replace('Model',''))

    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}\n(Test F{f})' for f in fold_ids], fontsize=9)
    ax.set_xlabel('Fold (Test Flight)', fontsize=11)
    ax.set_ylabel('Position RMSE (m)', fontsize=11)
    ax.set_title(f'LOO Position RMSE per Fold — {noise_label}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(out_dir, f'fold_rmse_{noise_label.lower()}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ===========================================================================
# COMBINED PLOTS
# ===========================================================================

def plot_noise_vs_rmse(all_stats, out_dir):
    """Noise vs Pos RMSE line plot — 4 model, X: noise level."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, m_color in zip(MODEL_NAMES, COLORS):
        vals = []
        for nl in NOISE_LEVELS:
            if nl in all_stats and model_name in all_stats[nl]:
                vals.append(all_stats[nl][model_name]['Pos_RMSE_mean'])
            else:
                vals.append(float('nan'))
        ax.plot(NOISE_LABELS, vals, marker='o', color=m_color,
                linewidth=2.5, markersize=9,
                label=model_name.replace('Model',''))
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.annotate(f'{v:.4f}', (NOISE_LABELS[i], v),
                            textcoords="offset points", xytext=(0, 8),
                            ha='center', fontsize=8)

    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('Position RMSE Mean (m)', fontsize=12)
    ax.set_title('Position RMSE vs Noise Level — LOO Protocol', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'combined_noise_vs_pos_rmse.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_noise_vs_delta_rmse(all_stats, out_dir):
    """Noise vs Delta RMSE line plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, m_color in zip(MODEL_NAMES, COLORS):
        vals = []
        for nl in NOISE_LEVELS:
            if nl in all_stats and model_name in all_stats[nl]:
                vals.append(all_stats[nl][model_name]['Delta_RMSE_mean'])
            else:
                vals.append(float('nan'))
        ax.plot(NOISE_LABELS, vals, marker='o', color=m_color,
                linewidth=2.5, markersize=9,
                label=model_name.replace('Model',''))
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.annotate(f'{v:.5f}', (NOISE_LABELS[i], v),
                            textcoords="offset points", xytext=(0, 8),
                            ha='center', fontsize=8)

    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('Delta RMSE Mean (m)', fontsize=12)
    ax.set_title('Delta RMSE vs Noise Level — LOO Protocol', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'combined_noise_vs_delta_rmse.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_combined_boxplot(all_dfs, out_dir):
    """4 noise seviyesi yan yana boxplot — Pos RMSE."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)

    for ax, noise_level, noise_label, n_color in zip(
            axes, NOISE_LEVELS, NOISE_LABELS, NOISE_COLORS):
        if noise_level not in all_dfs:
            continue
        df = all_dfs[noise_level]
        data = [df[df['Model'] == m]['Pos_RMSE'].values for m in MODEL_NAMES]
        bp   = ax.boxplot(data, patch_artist=True)
        for patch, m_color in zip(bp['boxes'], COLORS):
            patch.set_facecolor(m_color)
            patch.set_alpha(0.75)
        ax.set_title(noise_label, fontsize=12, fontweight='bold')
        ax.set_xticklabels([m.replace('Model','') for m in MODEL_NAMES], fontsize=9)
        ax.set_ylabel('Position RMSE (m)' if ax == axes[0] else '', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Position RMSE Distribution — LOO Protocol across Noise Levels', fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, 'combined_boxplot_pos_rmse.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_combined_axis_rmse(all_dfs, out_dir):
    """4 noise seviyesi yan yana — X, Y, Z channel RMSE."""
    channels = ['Pos_X_RMSE', 'Pos_Y_RMSE', 'Pos_Z_RMSE']
    ch_labels = ['X-Axis', 'Y-Axis', 'Z-Axis']

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharey='row')

    for row_idx, (ch, ch_label) in enumerate(zip(channels, ch_labels)):
        for col_idx, (noise_level, noise_label) in enumerate(
                zip(NOISE_LEVELS, NOISE_LABELS)):
            ax = axes[row_idx][col_idx]
            if noise_level not in all_dfs:
                continue
            df   = all_dfs[noise_level]
            data = [df[df['Model'] == m][ch].values for m in MODEL_NAMES]
            bp   = ax.boxplot(data, patch_artist=True)
            for patch, m_color in zip(bp['boxes'], COLORS):
                patch.set_facecolor(m_color)
                patch.set_alpha(0.75)
            if row_idx == 0:
                ax.set_title(noise_label, fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'{ch_label} RMSE (m)', fontsize=10)
            ax.set_xticklabels([m.replace('Model','') for m in MODEL_NAMES],
                               fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Per-Channel Position RMSE — LOO Protocol across Noise Levels', fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, 'combined_axis_rmse.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_extended_heatmap(all_stats, speed_data, out_dir):
    """
    Genisletilmis heatmap: model x noise seviyesi
    Rows: Pos RMSE, Pos MAE, Delta RMSE, Inference Time
    Cols: LSTM-base, LSTM-low, ..., AHLSTM-high
    """
    metrics     = ['Pos_RMSE_mean', 'Pos_MAE_mean', 'Delta_RMSE_mean']
    metric_lbls = ['Pos RMSE', 'Pos MAE', 'Delta RMSE']

    # Speed row ekle
    if speed_data:
        metrics.append('speed')
        metric_lbls.append('Latency (ms)')

    col_labels = []
    for m_label in MODEL_LABELS:
        for n_label in NOISE_LABELS:
            col_labels.append(f'{m_label}\n{n_label}')

    heat_data = np.full((len(metrics), len(col_labels)), np.nan)

    col_idx = 0
    for model_name, m_label in zip(MODEL_NAMES, MODEL_LABELS):
        for noise_level, n_label in zip(NOISE_LEVELS, NOISE_LABELS):
            for row_idx, metric in enumerate(metrics):
                if metric == 'speed':
                    if speed_data and model_name in speed_data:
                        heat_data[row_idx, col_idx] = speed_data[model_name]
                elif noise_level in all_stats and model_name in all_stats[noise_level]:
                    heat_data[row_idx, col_idx] = all_stats[noise_level][model_name].get(metric, np.nan)
            col_idx += 1

    # Normalize per row
    heat_norm = np.zeros_like(heat_data)
    for i in range(heat_data.shape[0]):
        row    = heat_data[i]
        r_min  = np.nanmin(row)
        r_max  = np.nanmax(row)
        heat_norm[i] = (row - r_min) / (r_max - r_min + 1e-9)

    fig_w = max(18, len(col_labels) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    im = ax.imshow(heat_norm, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(metric_lbls)))
    ax.set_yticklabels(metric_lbls, fontsize=11)

    for i in range(len(metrics)):
        for j in range(len(col_labels)):
            val      = heat_data[i, j]
            norm_val = heat_norm[i, j]
            if not np.isnan(val):
                fmt = f'{val:.4f}' if metrics[i] != 'speed' else f'{val:.2f}'
                txt = 'white' if norm_val > 0.6 else 'black'
                ax.text(j, i, fmt, ha='center', va='center',
                        fontsize=7.5, color=txt, fontweight='bold')

    # Model grup cizgileri
    for m_idx in range(1, len(MODEL_NAMES)):
        ax.axvline(x=m_idx * len(NOISE_LEVELS) - 0.5,
                   color='white', linewidth=2.5)

    plt.colorbar(im, ax=ax, label='Normalized value (lower = better)')
    ax.set_title('Extended Model Comparison Heatmap — LOO Protocol (All Noise Levels)',
                 fontsize=12, pad=12)
    plt.tight_layout()
    path = os.path.join(out_dir, 'combined_heatmap_extended.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_speed_vs_accuracy(speed_data, all_stats, out_dir):
    """Speed vs Accuracy scatter — reviewer favorite."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for model_name, m_label, m_color in zip(MODEL_NAMES, MODEL_LABELS, COLORS):
        if model_name not in speed_data:
            continue
        latency = speed_data[model_name]
        rmse    = all_stats.get('baseline', {}).get(model_name, {}).get('Pos_RMSE_mean', np.nan)
        if not np.isnan(rmse):
            ax.scatter(latency, rmse, color=m_color, s=200, zorder=5,
                       edgecolors='black', linewidth=1.5)
            ax.annotate(m_label, (latency, rmse),
                        textcoords="offset points", xytext=(10, 5),
                        fontsize=12, fontweight='bold', color=m_color)

    ax.set_xlabel('Inference Latency (ms/step)', fontsize=12)
    ax.set_ylabel('Position RMSE Mean (m)', fontsize=12)
    ax.set_title('Speed vs Accuracy — LOO Baseline', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'speed_vs_accuracy.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def save_per_condition_table(df_stats, noise_label, out_path):
    """Her noise seviyesi icin ayri istatistik tablosu."""
    cols = ['Model',
            'Pos_RMSE_mean', 'Pos_RMSE_std',
            'Pos_MAE_mean',  'Pos_MAE_std',
            'Delta_RMSE_mean', 'Delta_RMSE_std',
            'Best_Test_RMSE_mean']
    existing = [c for c in cols if c in df_stats.columns]
    df_out   = df_stats[existing].copy()
    df_out.to_excel(out_path, index=False)
    print(f"Saved: {out_path}")


def save_combined_table(all_stats, out_path):
    """Birlesik karsilastirma tablosu — Baseline | Low | Med | High | Change %"""
    rows = []
    for metric, metric_label in [
        ('Pos_RMSE_mean',   'Position RMSE (m)'),
        ('Pos_MAE_mean',    'Position MAE (m)'),
        ('Delta_RMSE_mean', 'Delta RMSE (m)'),
    ]:
        for model_name, m_label in zip(MODEL_NAMES, MODEL_LABELS):
            baseline_val = all_stats.get('baseline', {}).get(model_name, {}).get(metric, np.nan)
            row = {'Metric': metric_label, 'Model': m_label, 'Baseline': baseline_val}
            for noise_level, n_label in zip(['low','medium','high'], ['Low','Medium','High']):
                val = all_stats.get(noise_level, {}).get(model_name, {}).get(metric, np.nan)
                row[n_label] = val
                if not np.isnan(baseline_val) and not np.isnan(val) and baseline_val != 0:
                    row[f'{n_label} Change (%)'] = round((val - baseline_val) / baseline_val * 100, 2)
                else:
                    row[f'{n_label} Change (%)'] = np.nan
            rows.append(row)

    df = pd.DataFrame(rows)
    col_order = ['Metric', 'Model', 'Baseline', 'Low', 'Medium', 'High',
                 'Low Change (%)', 'Medium Change (%)', 'High Change (%)']
    df = df[[c for c in col_order if c in df.columns]]
    df.to_excel(out_path, index=False)
    print(f"Saved: {out_path}")
