"""
UAV GPS Loss - Per-Flight Training and Evaluation
==================================================
Strategy:
  - Each flight is trained and tested independently
  - First 80% of each flight -> train
  - Last 20% of each flight  -> test  (simulates GPS loss on that segment)
  - 4 models x 9 flights = 36 training sessions

This is methodologically stronger than pooling:
  - The model learns from a flight, then predicts the final segment
  - Mirrors real-world GPS loss scenario within each flight
  - Per-flight and aggregate metrics reported

Outputs:
  results/
    flight_01/ ... flight_09/
      LSTMModel / BiLSTMModel / GRUModel / AHLSTMModel
        - *_model.pth, *_scaler_X.pkl, *_scaler_y.pkl
        - *_losses.csv, *_loss_curve.png
        - *_scenario_2d.png, *_scenario_3d.png, *_scenario_axes.png
        - *_results.xlsx
    summary/
      all_results.xlsx
      statistics.xlsx
      comparison_plots/
        loss_curves_all_models.png
        boxplot_position_rmse.png
        bar_mean_std.png
        channel_rmse_comparison.png
        heatmap_comparison.png
        per_flight_pos_rmse.png
"""

import os
import pickle
from math import sqrt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# ===========================================================================
# 0. Settings
# ===========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SEED = 42
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
TRAIN_RATIO     = 0.80
SEQUENCE_LENGTH = 20
BASE_OUT_DIR    = 'results'

INPUT_COLS = [
    'qx', 'qy', 'qz', 'qw',
    'roll', 'yaw', 'pitch',
    'roll_a', 'pitch_a', 'yaw_a',
    'acc_x', 'acc_y', 'acc_z',
    'c_x', 'c_y', 'c_z',
]
OUTPUT_COLS = ['x_artis', 'y_artis', 'z_artis']
FILE_PATHS  = [f'dataset/{i}/sonuc_dosya_adı.csv' for i in range(1, 10)]

MODEL_CONFIGS = [
    ('LSTMModel',   LSTMModel),
    ('BiLSTMModel', BiLSTMModel),
    ('GRUModel',    GRUModel),
    ('AHLSTMModel', AHLSTMModel),
]

os.makedirs(BASE_OUT_DIR, exist_ok=True)

# ===========================================================================
# 1. Helper Functions
# ===========================================================================

def create_sequences(X, y, seq_len):
    """Sliding window: output at t paired with inputs [t-seq_len : t]."""
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def build_model(name, model_class):
    return model_class(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)


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


def save_loss_plot(train_losses, test_losses, model_name, out_dir, flight_id):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train RMSE', color='steelblue', linewidth=1.5)
    plt.plot(test_losses,  label='Test RMSE',  color='tomato',    linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE Loss')
    plt.title(f'{model_name} | Flight {flight_id:02d} | Train vs Test Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f'{model_name}_loss_curve.png')
    plt.savefig(path, dpi=130)
    plt.close()


def save_scenario_plots(train_cumpos, test_cumpos_actual, test_cumpos_predicted,
                        last_known_pos, model_name, flight_id, out_dir):
    """
    Three scenario plots:
      1. 2D position (XY plane)
      2. 3D position
      3. Per-axis time series (X, Y, Z separately)
    """
    n_train = len(train_cumpos)
    n_test  = len(test_cumpos_actual)

    # ---- 2D ----
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(train_cumpos[:, 0], train_cumpos[:, 1],
            color='royalblue', linewidth=2, label='Actual (80% - GPS Available)')
    ax.scatter(last_known_pos[0], last_known_pos[1],
               color='black', s=120, zorder=5, label='GPS Loss Point')
    ax.plot(test_cumpos_actual[:, 0], test_cumpos_actual[:, 1],
            color='green', linewidth=2, linestyle='--', label='Actual (20% - GPS Lost)')
    ax.plot(test_cumpos_predicted[:, 0], test_cumpos_predicted[:, 1],
            color='tomato', linewidth=2, label='Model Prediction (20%)')
    x_vals = np.concatenate([test_cumpos_actual[:, 0], test_cumpos_predicted[:, 0]])
    ax.axvspan(x_vals.min() - 0.5, x_vals.max() + 0.5,
               alpha=0.06, color='red', label='GPS Loss Region')
    ax.set_title(f'{model_name} | Flight {flight_id:02d} | 2D Position\n'
                 f'Seq={SEQUENCE_LENGTH} | GPS Loss: last 20%', fontsize=12)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{model_name}_scenario_2d.png'), dpi=130)
    plt.close()

    # ---- 3D ----
    fig = plt.figure(figsize=(12, 7))
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.plot(*train_cumpos.T,           color='royalblue', linewidth=2,
             label='Actual (80% - GPS Available)')
    ax3.scatter(*last_known_pos,         color='black', s=120, zorder=5,
                label='GPS Loss Point')
    ax3.plot(*test_cumpos_actual.T,      color='green', linewidth=2, linestyle='--',
             label='Actual (20% - GPS Lost)')
    ax3.plot(*test_cumpos_predicted.T,   color='tomato', linewidth=2,
             label='Model Prediction (20%)')
    ax3.set_title(f'{model_name} | Flight {flight_id:02d} | 3D Position\n'
                  f'Seq={SEQUENCE_LENGTH} | GPS Loss: last 20%', fontsize=12)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{model_name}_scenario_3d.png'), dpi=130)
    plt.close()

    # ---- Per-axis time series ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    for i, (ax, lbl) in enumerate(zip(axes, ['X (m)', 'Y (m)', 'Z (m)'])):
        ax.plot(range(n_train), train_cumpos[:, i],
                color='royalblue', linewidth=1.5, label='Actual (80% - GPS Available)')
        ax.axvline(x=n_train - 1, color='black', linewidth=1.5,
                   linestyle=':', label='GPS Loss Point')
        ax.plot(range(n_train, n_train + n_test), test_cumpos_actual[:, i],
                color='green', linewidth=1.5, linestyle='--',
                label='Actual (20% - GPS Lost)')
        ax.plot(range(n_train, n_train + n_test), test_cumpos_predicted[:, i],
                color='tomato', linewidth=1.5, label='Model Prediction (20%)')
        ax.axvspan(n_train, n_train + n_test, alpha=0.07, color='red')
        ax.set_ylabel(lbl, fontsize=11)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time Step (×100 ms)', fontsize=11)
    fig.suptitle(f'{model_name} | Flight {flight_id:02d} | Per-Axis Position\n'
                 f'Red region: GPS Loss (last 20%)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{model_name}_scenario_axes.png'), dpi=130)
    plt.close()


# ===========================================================================
# 2. Main Loop: Per-Flight, Per-Model Training
# ===========================================================================
all_results = []   # one row per (flight, model)

# Store loss curves for combined plots later
# loss_store[model_name] = list of (train_losses, test_losses) per flight
loss_store = {name: [] for name, _ in MODEL_CONFIGS}

for flight_idx, flight_path in enumerate(FILE_PATHS):
    flight_id = flight_idx + 1
    print(f"\n{'='*65}")
    print(f"  FLIGHT {flight_id}/9  |  {flight_path}")
    print(f"{'='*65}")

    # ----------------------------------------------------------------
    # Load and split this flight's data
    # ----------------------------------------------------------------
    df    = pd.read_csv(flight_path).dropna()
    split = int(len(df) * TRAIN_RATIO)

    df_train = df.iloc[:split]
    df_test  = df.iloc[split:]

    X_tr_raw = df_train[INPUT_COLS].values
    y_tr_raw = df_train[OUTPUT_COLS].values
    X_te_raw = df_test[INPUT_COLS].values
    y_te_raw = df_test[OUTPUT_COLS].values

    print(f"  Total samples : {len(df)}")
    print(f"  Train (80%)   : {len(df_train)}")
    print(f"  Test  (20%)   : {len(df_test)}")

    # Fit scaler on THIS flight's train data only
    scaler_X = StandardScaler().fit(X_tr_raw)
    scaler_y = StandardScaler().fit(y_tr_raw)

    X_tr_s = scaler_X.transform(X_tr_raw)
    y_tr_s = scaler_y.transform(y_tr_raw)
    X_te_s = scaler_X.transform(X_te_raw)   # transform only
    y_te_s = scaler_y.transform(y_te_raw)

    # Sliding window sequences
    X_train_seq, y_train_seq = create_sequences(X_tr_s, y_tr_s, SEQUENCE_LENGTH)
    X_test_seq,  y_test_seq  = create_sequences(X_te_s, y_te_s, SEQUENCE_LENGTH)

    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test_seq,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test_seq,  dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=False,  # time series -- preserve order
    )

    # Cumulative position for scenario plots
    # (use raw deltas, not sequences -- full flight)
    train_cumpos   = np.cumsum(y_tr_raw, axis=0)
    last_known_pos = train_cumpos[-1]

    flight_dir = os.path.join(BASE_OUT_DIR, f'flight_{flight_id:02d}')
    os.makedirs(flight_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Train each model on this flight
    # ----------------------------------------------------------------
    for model_name, model_class in MODEL_CONFIGS:
        print(f"\n  [{flight_id}/9] {model_name}")

        model_dir = os.path.join(flight_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        model     = build_model(model_name, model_class)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_losses, test_losses = [], []

        for epoch in range(NUM_EPOCHS):
            model.train()
            batch_losses = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss    = torch.sqrt(criterion(outputs, y_batch))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            train_loss = float(np.mean(batch_losses))
            train_losses.append(train_loss)

            model.eval()
            with torch.no_grad():
                test_out  = model(X_test_t.to(device))
                test_loss = torch.sqrt(criterion(test_out, y_test_t.to(device))).item()
            test_losses.append(test_loss)

            if (epoch + 1) % 50 == 0:
                print(f"    Epoch [{epoch+1:>3}/{NUM_EPOCHS}]  "
                      f"Train: {train_loss:.4f}  Test: {test_loss:.4f}")

        # ---- Predict ----
        model.eval()
        with torch.no_grad():
            pred_scaled = model(X_test_t.to(device)).cpu().numpy()

        actual    = scaler_y.inverse_transform(y_test_t.numpy())
        predicted = scaler_y.inverse_transform(pred_scaled)

        # Delta metrics (R2 meaningful here)
        delta_m = compute_metrics(actual, predicted)

        # Cumulative position metrics
        # Start from last known GPS position (end of 80% region)
        # Use raw test deltas (not sequence-trimmed) for full coverage
        # For metrics, use sequence-aligned versions
        test_cumpos_actual    = last_known_pos + np.cumsum(actual,    axis=0)
        test_cumpos_predicted = last_known_pos + np.cumsum(predicted, axis=0)
        pos_m = compute_metrics(test_cumpos_actual, test_cumpos_predicted)

        # ---- Scenario plots ----
        # Full actual test deltas (not sequence-trimmed) for complete visual
        actual_full    = y_te_raw
        # Predict on full test set with context from train end
        X_ctx_s   = scaler_X.transform(df_train[INPUT_COLS].values[-SEQUENCE_LENGTH:])
        X_full_s  = np.concatenate([X_ctx_s, X_te_s], axis=0)
        X_full_seq = np.array([X_full_s[i - SEQUENCE_LENGTH:i]
                                for i in range(SEQUENCE_LENGTH, len(X_full_s))],
                               dtype=np.float32)
        with torch.no_grad():
            pred_full = model(torch.tensor(X_full_seq).to(device)).cpu().numpy()
        pred_full_inv = scaler_y.inverse_transform(pred_full)

        test_cumpos_actual_full    = last_known_pos + np.cumsum(actual_full,   axis=0)
        test_cumpos_predicted_full = last_known_pos + np.cumsum(pred_full_inv, axis=0)

        save_loss_plot(train_losses, test_losses, model_name, model_dir, flight_id)
        save_scenario_plots(train_cumpos, test_cumpos_actual_full,
                            test_cumpos_predicted_full,
                            last_known_pos, model_name, flight_id, model_dir)

        # ---- Save results row ----
        result_row = {
            'Flight':           flight_id,
            'Model':            model_name,
            'Train_Samples':    len(X_train_seq),
            'Test_Samples':     len(X_test_seq),
            # Delta metrics
            'Delta_MSE':        delta_m['MSE'],
            'Delta_MAE':        delta_m['MAE'],
            'Delta_RMSE':       delta_m['RMSE'],
            'Delta_R2':         delta_m['R2'],
            'Delta_MAPE':       delta_m['MAPE'],
            'Delta_X_RMSE':     delta_m['Channel_X_RMSE'],
            'Delta_Y_RMSE':     delta_m['Channel_Y_RMSE'],
            'Delta_Z_RMSE':     delta_m['Channel_Z_RMSE'],
            'Delta_Avg_RMSE':   delta_m['Avg_Channel_RMSE'],
            # Position metrics
            'Pos_MSE':          pos_m['MSE'],
            'Pos_MAE':          pos_m['MAE'],
            'Pos_RMSE':         pos_m['RMSE'],
            'Pos_R2':           pos_m['R2'],
            'Pos_MAPE':         pos_m['MAPE'],
            'Pos_X_RMSE':       pos_m['Channel_X_RMSE'],
            'Pos_Y_RMSE':       pos_m['Channel_Y_RMSE'],
            'Pos_Z_RMSE':       pos_m['Channel_Z_RMSE'],
            'Pos_Avg_RMSE':     pos_m['Avg_Channel_RMSE'],
            # Loss summary
            'Final_Train_RMSE': train_losses[-1],
            'Final_Test_RMSE':  test_losses[-1],
            'Best_Test_RMSE':   min(test_losses),
        }
        all_results.append(result_row)
        loss_store[model_name].append((train_losses, test_losses))

        # ---- Save per-flight Excel ----
        common = {'Flight': flight_id, 'Model': model_name,
                  'Sequence_Length': SEQUENCE_LENGTH, 'Dropout': DROPOUT,
                  'Epochs': NUM_EPOCHS}
        df_delta = pd.DataFrame([{**common, **delta_m}])
        df_pos   = pd.DataFrame([{**common,
                                   'MSE':    pos_m['MSE'], 'MAE': pos_m['MAE'],
                                   'RMSE':   pos_m['RMSE'], 'MAPE': pos_m['MAPE'],
                                   'X_RMSE': pos_m['Channel_X_RMSE'],
                                   'Y_RMSE': pos_m['Channel_Y_RMSE'],
                                   'Z_RMSE': pos_m['Channel_Z_RMSE'],
                                   'Avg_RMSE': pos_m['Avg_Channel_RMSE'],
                                   'R2_NOTE': 'NOT APPLICABLE for cumulative pos'}])
        excel_path = os.path.join(model_dir, f'{model_name}_results.xlsx')
        with pd.ExcelWriter(excel_path) as writer:
            df_delta.to_excel(writer, sheet_name='Delta_Metrics',    index=False)
            df_pos.to_excel(  writer, sheet_name='Position_Metrics', index=False)

        # ---- Save model and scalers ----
        torch.save(model.state_dict(),
                   os.path.join(model_dir, f'{model_name}_model.pth'))
        with open(os.path.join(model_dir, f'{model_name}_scaler_X.pkl'), 'wb') as f:
            pickle.dump(scaler_X, f)
        with open(os.path.join(model_dir, f'{model_name}_scaler_y.pkl'), 'wb') as f:
            pickle.dump(scaler_y, f)

        # Save loss CSV
        pd.DataFrame({'epoch': range(1, NUM_EPOCHS+1),
                      'train_rmse': train_losses,
                      'test_rmse':  test_losses}
                    ).to_csv(os.path.join(model_dir, f'{model_name}_losses.csv'),
                             index=False)

        print(f"    -> Delta RMSE: {delta_m['RMSE']:.6f}  "
              f"| Pos RMSE: {pos_m['RMSE']:.4f} m  "
              f"| Delta R2: {delta_m['R2']:.4f}")

    print(f"\n  Flight {flight_id} complete.")


# ===========================================================================
# 3. Statistical Summary
# ===========================================================================
print(f"\n{'='*65}")
print("  COMPUTING STATISTICAL SUMMARY...")
print(f"{'='*65}")

summary_dir = os.path.join(BASE_OUT_DIR, 'summary')
os.makedirs(summary_dir, exist_ok=True)

df_all = pd.DataFrame(all_results)

# Save full results table
df_all.to_excel(os.path.join(summary_dir, 'all_results.xlsx'), index=False)

metric_cols = [
    'Delta_MSE', 'Delta_MAE', 'Delta_RMSE', 'Delta_R2', 'Delta_MAPE',
    'Delta_X_RMSE', 'Delta_Y_RMSE', 'Delta_Z_RMSE', 'Delta_Avg_RMSE',
    'Pos_MSE', 'Pos_MAE', 'Pos_RMSE', 'Pos_MAPE',
    'Pos_X_RMSE', 'Pos_Y_RMSE', 'Pos_Z_RMSE', 'Pos_Avg_RMSE',
    'Final_Train_RMSE', 'Final_Test_RMSE', 'Best_Test_RMSE',
]

model_names = [n for n, _ in MODEL_CONFIGS]
colors      = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
color_map   = dict(zip(model_names, colors))

# Per-model stats across 9 flights
stat_rows = []
for model_name in model_names:
    df_m = df_all[df_all['Model'] == model_name]
    row  = {'Model': model_name}
    for col in metric_cols:
        row[f'{col}_mean']  = df_m[col].mean()
        row[f'{col}_std']   = df_m[col].std()
        row[f'{col}_best']  = df_m[col].min()
        row[f'{col}_worst'] = df_m[col].max()
    stat_rows.append(row)

df_stats = pd.DataFrame(stat_rows)

# Save statistics Excel (3 sheets)
stats_path = os.path.join(summary_dir, 'statistics.xlsx')
with pd.ExcelWriter(stats_path) as writer:
    df_all.to_excel(writer, sheet_name='All_Results', index=False)

    delta_cols = ['Model'] + [
        f'{c}_{s}' for c in ['Delta_RMSE','Delta_MAE','Delta_R2','Delta_MAPE','Delta_Avg_RMSE']
        for s in ['mean','std','best','worst']]
    df_stats[delta_cols].to_excel(writer, sheet_name='Delta_Stats', index=False)

    pos_cols = ['Model'] + [
        f'{c}_{s}' for c in ['Pos_RMSE','Pos_MAE','Pos_MAPE','Pos_Avg_RMSE']
        for s in ['mean','std','best','worst']]
    df_stats[pos_cols].to_excel(writer, sheet_name='Position_Stats', index=False)

    test_cols = ['Model'] + [
        f'{c}_{s}' for c in ['Final_Test_RMSE','Best_Test_RMSE']
        for s in ['mean','std','best','worst']]
    df_stats[test_cols].to_excel(writer, sheet_name='Test_RMSE_Stats', index=False)

print(f"Statistics saved: {stats_path}")


# ===========================================================================
# 4. Comparison and Summary Plots
# ===========================================================================
comp_dir = os.path.join(summary_dir, 'comparison_plots')
os.makedirs(comp_dir, exist_ok=True)

epochs = np.arange(1, NUM_EPOCHS + 1)

# ---- 4a. Loss curves: 4 subplots (one per model), 9 lines each flight ----
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
axes = axes.flatten()
flight_colors = plt.cm.tab10(np.linspace(0, 0.9, 9))

for ax, (model_name, _), m_color in zip(axes, MODEL_CONFIGS, colors):
    for f_idx, (tr_loss, te_loss) in enumerate(loss_store[model_name]):
        ax.plot(epochs, tr_loss, color=flight_colors[f_idx],
                linewidth=1.0, alpha=0.6, linestyle='-')
        ax.plot(epochs, te_loss, color=flight_colors[f_idx],
                linewidth=1.0, alpha=0.6, linestyle='--')

    # Mean across flights
    tr_mean = np.mean([l[0] for l in loss_store[model_name]], axis=0)
    te_mean = np.mean([l[1] for l in loss_store[model_name]], axis=0)
    ax.plot(epochs, tr_mean, color='black', linewidth=2.5,
            linestyle='-',  label='Train Mean')
    ax.plot(epochs, te_mean, color='black', linewidth=2.5,
            linestyle='--', label='Test Mean')

    # Legend: flight numbers
    handles = [plt.Line2D([0], [0], color=flight_colors[i], linewidth=1.5,
                           label=f'Flight {i+1}') for i in range(9)]
    handles += [plt.Line2D([0], [0], color='black', linewidth=2, linestyle='-',  label='Train Mean'),
                plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Test Mean')]
    ax.legend(handles=handles, fontsize=7, ncol=2, loc='upper right')
    ax.set_title(f'{model_name}', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE Loss')
    ax.grid(True, alpha=0.3)

plt.suptitle('Loss Curves per Flight (— Train / -- Test) | Bold = Mean across Flights',
             fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(comp_dir, 'loss_curves_all_models.png'), dpi=150)
plt.close()
print("Saved: loss_curves_all_models.png")

# ---- 4b. Per-flight Position RMSE (line plot, 4 models × 9 flights) ----
fig, ax = plt.subplots(figsize=(13, 6))
flight_ids = list(range(1, 10))

for model_name, m_color in zip(model_names, colors):
    vals = [df_all[(df_all['Model'] == model_name) &
                   (df_all['Flight'] == f)]['Pos_RMSE'].values[0]
            for f in flight_ids]
    ax.plot(flight_ids, vals, marker='o', color=m_color,
            linewidth=2, markersize=7, label=model_name)

ax.set_xticks(flight_ids)
ax.set_xlabel('Flight ID', fontsize=11)
ax.set_ylabel('Position RMSE (m)', fontsize=11)
ax.set_title('Position RMSE per Flight — All Models', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(comp_dir, 'per_flight_pos_rmse.png'), dpi=150)
plt.close()
print("Saved: per_flight_pos_rmse.png")

# ---- 4c. Boxplot: distribution across 9 flights ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, (metric, title) in zip(axes, [
    ('Pos_RMSE',   'Position RMSE (m) — 9 Flights'),
    ('Delta_RMSE', 'Delta RMSE — 9 Flights'),
]):
    data = [df_all[df_all['Model'] == m][metric].values for m in model_names]
    bp   = ax.boxplot(data, patch_artist=True, notch=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([m.replace('Model','') for m in model_names], fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(metric, fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Metric Distribution Across 9 Flights', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(comp_dir, 'boxplot_rmse.png'), dpi=150)
plt.close()
print("Saved: boxplot_rmse.png")

# ---- 4d. Bar chart: mean ± std ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, (metric, title) in zip(axes, [
    ('Pos_RMSE',   'Position RMSE Mean ± Std (9 Flights)'),
    ('Delta_R2',   'Delta R² Mean ± Std (9 Flights)'),
]):
    means = df_stats[f'{metric}_mean'].values
    stds  = df_stats[f'{metric}_std'].values
    x     = np.arange(len(model_names))
    bars  = ax.bar(x, means, yerr=stds, capsize=7,
                   color=colors, alpha=0.75, edgecolor='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('Model','') for m in model_names], fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(metric, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + max(stds)*0.02,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(comp_dir, 'bar_mean_std.png'), dpi=150)
plt.close()
print("Saved: bar_mean_std.png")

# ---- 4e. Per-channel Position RMSE ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (ch, title) in zip(axes, [
    ('Pos_X_RMSE', 'X-Axis Position RMSE'),
    ('Pos_Y_RMSE', 'Y-Axis Position RMSE'),
    ('Pos_Z_RMSE', 'Z-Axis Position RMSE'),
]):
    data = [df_all[df_all['Model'] == m][ch].values for m in model_names]
    bp   = ax.boxplot(data, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([m.replace('Model','') for m in model_names], fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('RMSE (m)')
    ax.grid(True, alpha=0.3)

plt.suptitle('Per-Channel Position RMSE — 9 Flights', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(comp_dir, 'channel_rmse_comparison.png'), dpi=150)
plt.close()
print("Saved: channel_rmse_comparison.png")

# ---- 4f. Heatmap: model × metric ----
heat_metrics = ['Pos_RMSE_mean', 'Pos_MAE_mean', 'Delta_RMSE_mean',
                'Delta_R2_mean', 'Best_Test_RMSE_mean']
heat_labels  = ['Pos\nRMSE', 'Pos\nMAE', 'Delta\nRMSE', 'Delta\nR²', 'Best\nTest']

heat_data = df_stats[heat_metrics].values.astype(float)
heat_norm = (heat_data - heat_data.min(0)) / (heat_data.max(0) - heat_data.min(0) + 1e-9)

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(heat_norm.T, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(range(len(model_names)))
ax.set_xticklabels([m.replace('Model','') for m in model_names], fontsize=11)
ax.set_yticks(range(len(heat_labels)))
ax.set_yticklabels(heat_labels, fontsize=11)

for i in range(len(model_names)):
    for j in range(len(heat_metrics)):
        ax.text(i, j, f'{heat_data[i, j]:.4f}',
                ha='center', va='center', fontsize=9,
                color='black' if heat_norm[i, j] < 0.65 else 'white')

plt.colorbar(im, ax=ax, label='Normalized value (lower=better, except R²)')
ax.set_title('Model Comparison Heatmap — 9-Flight Average', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(comp_dir, 'heatmap_comparison.png'), dpi=150)
plt.close()
print("Saved: heatmap_comparison.png")

# ---- 4g. All metrics table as a figure ----
summary_table = df_stats[['Model',
    'Pos_RMSE_mean', 'Pos_RMSE_std', 'Pos_RMSE_best', 'Pos_RMSE_worst',
    'Pos_MAE_mean',  'Delta_RMSE_mean', 'Delta_R2_mean',
    'Best_Test_RMSE_mean']].copy()
summary_table.columns = [
    'Model', 'Pos RMSE\nMean', 'Pos RMSE\nStd', 'Pos RMSE\nBest', 'Pos RMSE\nWorst',
    'Pos MAE\nMean', 'Delta RMSE\nMean', 'Delta R²\nMean', 'Best Test\nRMSE Mean']

fig, ax = plt.subplots(figsize=(16, 3))
ax.axis('off')
tbl = ax.table(
    cellText=summary_table.values.tolist(),
    colLabels=summary_table.columns.tolist(),
    cellLoc='center', loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.8)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#f0f4f8')
plt.title('Summary Table — 9-Flight Average (All Models)', fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(comp_dir, 'summary_table.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: summary_table.png")


# ===========================================================================
# 5. Console Summary
# ===========================================================================
print(f"\n{'='*65}")
print(f"  FINAL SUMMARY — 9-Flight Average Position RMSE (m)")
print(f"{'='*65}")
print(f"  {'Model':<15} {'Pos_RMSE':>10} {'±Std':>8} {'Best':>8} {'Worst':>8}  {'Delta_R2 Mean':>14}")
print(f"  {'-'*65}")
for _, row in df_stats.iterrows():
    print(f"  {row['Model']:<15} "
          f"{row['Pos_RMSE_mean']:>10.4f} "
          f"{row['Pos_RMSE_std']:>8.4f} "
          f"{row['Pos_RMSE_best']:>8.4f} "
          f"{row['Pos_RMSE_worst']:>8.4f}  "
          f"{row['Delta_R2_mean']:>14.4f}")
print(f"{'='*65}")
print(f"\nAll outputs saved to: {BASE_OUT_DIR}/")
print("Training complete.")