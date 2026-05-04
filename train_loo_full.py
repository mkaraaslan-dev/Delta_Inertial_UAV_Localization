"""
train_loo_full.py
=================
LOO protokolu ile tum noise seviyeleri icin egitim ve gorsellestime.

Tek calistirma ile tum sonuclari uretir:
  - Baseline (no noise)
  - Low noise
  - Medium noise
  - High noise

Her seviye icin: 9 fold x 4 model = 36 egitim oturumu
Toplam: 4 x 36 = 144 egitim oturumu

Ciktilar:
  results_loo/
    baseline/   low/   medium/   high/
      fold_01/ ... fold_09/
        LSTMModel / BiLSTMModel / GRUModel / AHLSTMModel
          *_model.pth, *_losses.csv
          *_loss_curve.png (per-fold)
          *_results.xlsx
      summary/
        all_results.xlsx
        statistics.xlsx
        per_condition_table.xlsx
        plots/
          loss_curves_{level}.png
          fold_rmse_{level}.png
    combined/
      combined_noise_vs_pos_rmse.png
      combined_noise_vs_delta_rmse.png
      combined_boxplot_pos_rmse.png
      combined_axis_rmse.png
      combined_heatmap_extended.png
      speed_vs_accuracy.png
      combined_table.xlsx
"""

import os
import gc
import pickle
import json
from math import sqrt

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error as mae,
    mean_absolute_percentage_error as mape,
    mean_squared_error as mse,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models import AHLSTMModel, LSTMModel, BiLSTMModel, GRUModel
from noise_augmentation import apply_noise

# ===========================================================================
# 0. Ayarlar
# ===========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SEED            = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

INPUT_SIZE      = 16
OUTPUT_SIZE     = 3
HIDDEN_SIZE     = 256
NUM_LAYERS      = 2
DROPOUT         = 0.4
LEARNING_RATE   = 0.001
BATCH_SIZE      = 64
NUM_EPOCHS      = 250
TRAIN_RATIO     = 0.80   # test flight icin GPS loss bolumunu belirler
SEQUENCE_LENGTH = 20
BASE_OUT_DIR    = 'results_loo'
COMBINED_DIR    = os.path.join(BASE_OUT_DIR, 'combined')

INPUT_COLS = [
    'qx', 'qy', 'qz', 'qw',
    'roll', 'yaw', 'pitch',
    'roll_a', 'pitch_a', 'yaw_a',
    'acc_x', 'acc_y', 'acc_z',
    'c_x', 'c_y', 'c_z',
]
OUTPUT_COLS = ['x_artis', 'y_artis', 'z_artis']
FILE_PATHS  = [f'dataset/{i}/sonuc_dosya_adı.csv' for i in range(1, 10)]
N_FLIGHTS   = len(FILE_PATHS)

MODEL_CONFIGS = [
    ('LSTMModel',   LSTMModel),
    ('BiLSTMModel', BiLSTMModel),
    ('GRUModel',    GRUModel),
    ('AHLSTMModel', AHLSTMModel),
]

# Noise seviyeleri: (label, noise_augmentation key)
NOISE_CONFIGS = [
    ('baseline', 'none'),
    ('low',      'low'),
    ('medium',   'medium'),
    ('high',     'high'),
]

os.makedirs(COMBINED_DIR, exist_ok=True)

# Speed data: inference_analysis.py ciktisinden okunur
# Eger dosya yoksa None olarak gecilir
SPEED_CACHE = os.path.join(BASE_OUT_DIR, 'speed_data.json')

# ===========================================================================
# 1. Helper Fonksiyonlar
# ===========================================================================

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def compute_metrics(y_true, y_pred):
    MSE_val  = float(mse(y_true, y_pred))
    MAE_val  = float(mae(y_true, y_pred))
    RMSE_val = sqrt(MSE_val)
    R2_val   = float(r2_score(y_true, y_pred))
    MAPE_val = float(mape(y_true, y_pred))
    ch_rmse  = [sqrt(float(mse(y_true[:, i], y_pred[:, i]))) for i in range(OUTPUT_SIZE)]
    return {
        'MSE':              MSE_val,
        'MAE':              MAE_val,
        'RMSE':             RMSE_val,
        'R2':               R2_val,
        'MAPE':             MAPE_val,
        'Channel_X_RMSE':   ch_rmse[0],
        'Channel_Y_RMSE':   ch_rmse[1],
        'Channel_Z_RMSE':   ch_rmse[2],
        'Avg_Channel_RMSE': float(np.mean(ch_rmse)),
    }


def load_speed_data():
    """inference_results/inference_summary.xlsx dosyasindan speed data yukle."""
    path = 'inference_results/inference_summary.xlsx'
    if not os.path.exists(path):
        print(f"  Speed data not found: {path}. Skipping speed plots.")
        return None
    df   = pd.read_excel(path)
    data = {}
    for _, row in df.iterrows():
        model_name = str(row['Model'])
        if not model_name.endswith('Model'):
            model_name = model_name + 'Model'
        data[model_name] = float(row['Latency (ms)'])
    return data


# ===========================================================================
# 2. Ana Egitim Fonksiyonu — Tek Noise Seviyesi
# ===========================================================================

def run_loo_for_noise(noise_label, noise_key, all_dfs):
    """
    Belirli bir noise seviyesi icin tam LOO egitimi yapar.
    Tum fold ve model sonuclari ile loss store'u dondurur.
    """
    from loo_plots import plot_loss_curves, plot_fold_rmse, save_per_condition_table

    level_dir   = os.path.join(BASE_OUT_DIR, noise_label)
    summary_dir = os.path.join(level_dir, 'summary')
    plots_dir   = os.path.join(summary_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  NOISE LEVEL: {noise_label.upper()}")
    print(f"{'='*65}")

    all_results = []
    loss_store  = {name: [] for name, _ in MODEL_CONFIGS}

    for fold_idx in range(N_FLIGHTS):
        fold_id   = fold_idx + 1
        test_df   = all_dfs[fold_idx]
        train_dfs = [all_dfs[i] for i in range(N_FLIGHTS) if i != fold_idx]

        print(f"\n  Fold {fold_id}/{N_FLIGHTS} | Test: Flight {fold_id}")

        # Train pool
        df_train_pool = pd.concat(train_dfs, axis=0, ignore_index=True)
        X_tr_raw = df_train_pool[INPUT_COLS].values
        y_tr_raw = df_train_pool[OUTPUT_COLS].values

        # Test: tam ucus
        X_te_raw = test_df[INPUT_COLS].values
        y_te_raw = test_df[OUTPUT_COLS].values

        # Gurultu uygula (sadece train'e, scaler'dan once)
        if noise_key != 'none':
            X_tr_raw = apply_noise(X_tr_raw, level=noise_key, seed=SEED + fold_idx)

        # Scaler — sadece train pool'a fit
        scaler_X = StandardScaler().fit(X_tr_raw)
        scaler_y = StandardScaler().fit(y_tr_raw)

        X_tr_s = scaler_X.transform(X_tr_raw)
        y_tr_s = scaler_y.transform(y_tr_raw)
        X_te_s = scaler_X.transform(X_te_raw)
        y_te_s = scaler_y.transform(y_te_raw)

        X_train_seq, y_train_seq = create_sequences(X_tr_s, y_tr_s, SEQUENCE_LENGTH)
        X_test_seq,  y_test_seq  = create_sequences(X_te_s, y_te_s, SEQUENCE_LENGTH)

        X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_seq, dtype=torch.float32)
        X_test_t  = torch.tensor(X_test_seq,  dtype=torch.float32)
        y_test_t  = torch.tensor(y_test_seq,  dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=BATCH_SIZE, shuffle=True,
        )

        # Baslangic konumu sifir
        last_known_pos = np.array([0.0, 0.0, 0.0])

        fold_dir = os.path.join(level_dir, f'fold_{fold_id:02d}')
        os.makedirs(fold_dir, exist_ok=True)

        for model_name, model_class in MODEL_CONFIGS:
            model_dir = os.path.join(fold_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            model = model_class(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                                OUTPUT_SIZE, DROPOUT).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            train_losses, test_losses = [], []

            for epoch in range(NUM_EPOCHS):
                model.train()
                batch_losses = []
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    out  = model(X_batch)
                    loss = torch.sqrt(criterion(out, y_batch))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())

                train_loss = float(np.mean(batch_losses))
                train_losses.append(train_loss)

                model.eval()
                with torch.no_grad():
                    te_out   = model(X_test_t.to(device))
                    te_loss  = torch.sqrt(criterion(te_out, y_test_t.to(device))).item()
                test_losses.append(te_loss)

                if (epoch + 1) % 50 == 0:
                    print(f"    [{model_name}] Epoch {epoch+1}/{NUM_EPOCHS} "
                          f"Train: {train_loss:.4f} Test: {te_loss:.4f}")

            # Tahmin
            model.eval()
            with torch.no_grad():
                pred_s = model(X_test_t.to(device)).cpu().numpy()

            actual    = scaler_y.inverse_transform(y_test_t.numpy())
            predicted = scaler_y.inverse_transform(pred_s)

            delta_m = compute_metrics(actual, predicted)

            cum_actual    = last_known_pos + np.cumsum(actual,    axis=0)
            cum_predicted = last_known_pos + np.cumsum(predicted, axis=0)
            pos_m = compute_metrics(cum_actual, cum_predicted)

            result_row = {
                'Fold':             fold_id,
                'Noise':            noise_label,
                'Model':            model_name,
                'Train_Flights':    str([i+1 for i in range(N_FLIGHTS) if i != fold_idx]),
                'Test_Flight':      fold_id,
                'Train_Samples':    len(X_train_seq),
                'Test_Samples':     len(X_test_seq),
                'Delta_MSE':        delta_m['MSE'],
                'Delta_MAE':        delta_m['MAE'],
                'Delta_RMSE':       delta_m['RMSE'],
                'Delta_R2':         delta_m['R2'],
                'Delta_MAPE':       delta_m['MAPE'],
                'Delta_X_RMSE':     delta_m['Channel_X_RMSE'],
                'Delta_Y_RMSE':     delta_m['Channel_Y_RMSE'],
                'Delta_Z_RMSE':     delta_m['Channel_Z_RMSE'],
                'Delta_Avg_RMSE':   delta_m['Avg_Channel_RMSE'],
                'Pos_MSE':          pos_m['MSE'],
                'Pos_MAE':          pos_m['MAE'],
                'Pos_RMSE':         pos_m['RMSE'],
                'Pos_R2':           pos_m['R2'],
                'Pos_MAPE':         pos_m['MAPE'],
                'Pos_X_RMSE':       pos_m['Channel_X_RMSE'],
                'Pos_Y_RMSE':       pos_m['Channel_Y_RMSE'],
                'Pos_Z_RMSE':       pos_m['Channel_Z_RMSE'],
                'Pos_Avg_RMSE':     pos_m['Avg_Channel_RMSE'],
                'Final_Train_RMSE': train_losses[-1],
                'Final_Test_RMSE':  test_losses[-1],
                'Best_Test_RMSE':   min(test_losses),
            }
            all_results.append(result_row)
            loss_store[model_name].append((train_losses, test_losses))

            # Kaydet
            torch.save(model.state_dict(),
                       os.path.join(model_dir, f'{model_name}_model.pth'))
            with open(os.path.join(model_dir, f'{model_name}_scaler_X.pkl'), 'wb') as f:
                pickle.dump(scaler_X, f)
            with open(os.path.join(model_dir, f'{model_name}_scaler_y.pkl'), 'wb') as f:
                pickle.dump(scaler_y, f)
            pd.DataFrame({'epoch': range(1, NUM_EPOCHS+1),
                          'train_rmse': train_losses,
                          'test_rmse':  test_losses}
                        ).to_csv(os.path.join(model_dir, f'{model_name}_losses.csv'),
                                 index=False)

            df_delta = pd.DataFrame([{'Fold': fold_id, 'Model': model_name,
                                       'Noise': noise_label, **delta_m}])
            df_pos   = pd.DataFrame([{'Fold': fold_id, 'Model': model_name,
                                       'Noise': noise_label, **pos_m}])
            with pd.ExcelWriter(os.path.join(model_dir, f'{model_name}_results.xlsx')) as w:
                df_delta.to_excel(w, sheet_name='Delta_Metrics',    index=False)
                df_pos.to_excel(  w, sheet_name='Position_Metrics', index=False)

            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            print(f"    -> Pos RMSE: {pos_m['RMSE']:.4f} m | "
                  f"Delta RMSE: {delta_m['RMSE']:.6f}")

    # ---- Summary ----
    df_all = pd.DataFrame(all_results)
    df_all.to_excel(os.path.join(summary_dir, 'all_results.xlsx'), index=False)

    model_names  = [n for n, _ in MODEL_CONFIGS]
    metric_cols  = [
        'Delta_RMSE', 'Delta_MAE', 'Delta_Avg_RMSE',
        'Pos_RMSE', 'Pos_MAE', 'Pos_Avg_RMSE',
        'Pos_X_RMSE', 'Pos_Y_RMSE', 'Pos_Z_RMSE',
        'Best_Test_RMSE',
    ]
    stat_rows = []
    for mn in model_names:
        df_m = df_all[df_all['Model'] == mn]
        row  = {'Model': mn}
        for col in metric_cols:
            row[f'{col}_mean']  = df_m[col].mean()
            row[f'{col}_std']   = df_m[col].std()
            row[f'{col}_best']  = df_m[col].min()
            row[f'{col}_worst'] = df_m[col].max()
        stat_rows.append(row)

    df_stats = pd.DataFrame(stat_rows)

    stats_path = os.path.join(summary_dir, 'statistics.xlsx')
    with pd.ExcelWriter(stats_path) as writer:
        df_all.to_excel(  writer, sheet_name='All_Results',     index=False)
        df_stats.to_excel(writer, sheet_name='Summary_Stats',   index=False)

        pos_cols = ['Model'] + [f'{c}_{s}'
            for c in ['Pos_RMSE','Pos_MAE','Pos_Avg_RMSE']
            for s in ['mean','std','best','worst']]
        df_stats[[c for c in pos_cols if c in df_stats.columns]].to_excel(
            writer, sheet_name='Position_Stats', index=False)

        delta_cols = ['Model'] + [f'{c}_{s}'
            for c in ['Delta_RMSE','Delta_MAE','Delta_Avg_RMSE']
            for s in ['mean','std','best','worst']]
        df_stats[[c for c in delta_cols if c in df_stats.columns]].to_excel(
            writer, sheet_name='Delta_Stats', index=False)

    save_per_condition_table(df_stats, noise_label,
        os.path.join(summary_dir, 'per_condition_table.xlsx'))

    # ---- Per-condition plots ----
    noise_display = noise_label.capitalize()
    plot_loss_curves(loss_store, noise_display, plots_dir, NUM_EPOCHS)
    plot_fold_rmse(df_all,       noise_display, plots_dir)

    print(f"\n  {noise_label.upper()} complete. "
          f"Results: {summary_dir}")

    # all_stats icin model bazli mean degerler
    stats_dict = {}
    for mn in model_names:
        row = df_stats[df_stats['Model'] == mn]
        if not row.empty:
            stats_dict[mn] = {
                'Pos_RMSE_mean':   float(row['Pos_RMSE_mean'].values[0]),
                'Pos_MAE_mean':    float(row['Pos_MAE_mean'].values[0]),
                'Delta_RMSE_mean': float(row['Delta_RMSE_mean'].values[0]),
            }

    return df_all, stats_dict


# ===========================================================================
# 3. Ana Akis
# ===========================================================================
if __name__ == '__main__':

    # Verileri bir kez yukle
    print("Loading all flight data...")
    all_dfs = []
    for path in FILE_PATHS:
        df = pd.read_csv(path).dropna()
        all_dfs.append(df)
        print(f"  {path}: {len(df)} rows")

    # Speed data
    speed_data = load_speed_data()
    if speed_data:
        with open(SPEED_CACHE, 'w') as f:
            json.dump(speed_data, f)
        print(f"Speed data loaded: {speed_data}")

    # Her noise seviyesi icin egitim
    all_dfs_results = {}   # noise_label -> df_all
    all_stats       = {}   # noise_label -> {model -> {metric -> val}}

    for noise_label, noise_key in NOISE_CONFIGS:
        df_all, stats_dict = run_loo_for_noise(noise_label, noise_key, all_dfs)
        all_dfs_results[noise_label] = df_all
        all_stats[noise_label]       = stats_dict

    # ===========================================================================
    # 4. Combined Plots
    # ===========================================================================
    from loo_plots import (
        plot_noise_vs_rmse, plot_noise_vs_delta_rmse,
        plot_combined_boxplot, plot_combined_axis_rmse,
        plot_extended_heatmap, plot_speed_vs_accuracy,
        save_combined_table,
    )

    print(f"\n{'='*65}")
    print("  GENERATING COMBINED PLOTS...")
    print(f"{'='*65}")

    plot_noise_vs_rmse(      all_stats,          COMBINED_DIR)
    plot_noise_vs_delta_rmse(all_stats,          COMBINED_DIR)
    plot_combined_boxplot(   all_dfs_results,    COMBINED_DIR)
    plot_combined_axis_rmse( all_dfs_results,    COMBINED_DIR)
    plot_extended_heatmap(   all_stats, speed_data, COMBINED_DIR)

    if speed_data:
        plot_speed_vs_accuracy(speed_data, all_stats, COMBINED_DIR)

    save_combined_table(all_stats,
        os.path.join(COMBINED_DIR, 'combined_table.xlsx'))

    # ===========================================================================
    # 5. Konsol Ozet
    # ===========================================================================
    print(f"\n{'='*65}")
    print("  FINAL SUMMARY — LOO Pos RMSE Mean (m)")
    print(f"{'='*65}")
    print(f"  {'Model':<14}", end='')
    for nl, _ in NOISE_CONFIGS:
        print(f"  {nl.capitalize():>10}", end='')
    print()
    print(f"  {'-'*60}")

    for model_name in [n for n, _ in MODEL_CONFIGS]:
        m_label = model_name.replace('Model', '')
        print(f"  {m_label:<14}", end='')
        for nl, _ in NOISE_CONFIGS:
            val = all_stats.get(nl, {}).get(model_name, {}).get('Pos_RMSE_mean', float('nan'))
            print(f"  {val:>10.4f}", end='')
        print()

    print(f"{'='*65}")
    print(f"\nAll outputs saved to: {BASE_OUT_DIR}/")
    print(f"Combined plots:       {COMBINED_DIR}/")
    print("Training complete.")
