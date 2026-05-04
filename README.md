# IMU-Based UAV Navigation Under GPS Loss: Cross-Flight Generalization and Noise Robustness of Recurrent Deep Learning Architectures

This repository contains the dataset, source code, and results associated with the paper:

**"IMU-Based UAV Navigation Under GPS Loss: Cross-Flight Generalization and Noise Robustness of Recurrent Deep Learning Architectures"**

---

## Overview

This study evaluates four recurrent deep learning architectures for continuous UAV position estimation during GPS loss using raw IMU sensor data. Incremental position changes (Δx, Δy, Δz) are directly estimated at each time step and cumulatively summed from the last known GPS position to obtain instantaneous position. Models are evaluated under a Leave-One-Out (LOO) cross-flight protocol across four noise conditions: baseline, low, medium, and high synthetic IMU noise.

![System Architecture](gps_prediction-inertialtrindex1.drawio.png)

---

## Repository Structure

```
├── models.py                  # Model definitions: LSTM, BiLSTM, GRU, AHLSTM
├── train_loo_full.py          # Main training script (LOO protocol, all noise levels)
├── loo_plots.py               # Visualization module (per-condition and combined plots)
├── noise_augmentation.py      # Synthetic IMU noise injection module
├── inference_analysis.py      # Inference time, FPS, and VRAM analysis
├── data/                      # Flight trajectory CSV files (9 flights)
└── README.md
```

---

## Models

All model architectures are defined in `models.py`:

| Model    | Description                                      |
|----------|--------------------------------------------------|
| LSTM     | Long Short-Term Memory                           |
| BiLSTM   | Bidirectional LSTM                               |
| GRU      | Gated Recurrent Unit                             |
| AHLSTM   | Attention-based Hierarchical LSTM                |

All models share the same configuration:
- Input size: 16
- Hidden size: 256
- Number of layers: 2
- Output size: 3 (Δx, Δy, Δz)
- Dropout: 0.4

---

## Training Protocol

A Leave-One-Out (LOO) cross-flight protocol is used. At each fold, one flight is held out entirely as the test flight while the remaining eight flights form the training pool. This directly tests inter-flight generalization under real GPS-loss deployment conditions.

- **Train**: 8 complete flights (all rows)
- **Test**: 1 unseen flight (full trajectory, prediction from origin)
- **Folds**: 9 (one per flight)
- **Noise levels**: Baseline, Low, Medium, High
- **Total training sessions**: 9 folds × 4 models × 4 noise levels = 144

The scaler is fit only on the 8-flight training pool to prevent data leakage.

---

## Getting Started

### Requirements

```bash
pip install torch scikit-learn pandas numpy matplotlib openpyxl
```

### Running the Full Training (All Noise Levels)

```bash
python train_loo_full.py
```

The script will automatically run all four noise conditions sequentially and generate all outputs:

1. Load all 9 flight CSV files from `data/`
2. Run LOO training for each noise level: baseline → low → medium → high
3. Save per-fold model weights, loss curves, and metrics
4. Generate per-condition plots (loss curves, fold RMSE bar charts)
5. Generate combined comparison plots (noise vs RMSE line plots, boxplots, heatmap, speed vs accuracy)
6. Save all statistical summaries as Excel files

### Running Inference Analysis

```bash
python inference_analysis.py
```

Measures latency (ms/step), throughput (fps), and peak VRAM for each model on CUDA.

---

## Output Structure

```
results_loo/
  baseline/   low/   medium/   high/
    fold_01/ ... fold_09/
      LSTMModel / BiLSTMModel / GRUModel / AHLSTMModel
        *_model.pth
        *_scaler_X.pkl, *_scaler_y.pkl
        *_losses.csv
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
    combined_heatmap_extended.png
    speed_vs_accuracy.png
    combined_table.xlsx

inference_results/
  inference_summary.xlsx
  bench_result_latency.png
  bench_result_throughput.png
  bench_result_vram.png

> **Note:** The `results_loo/` directory is not included in this repository due to file size.
> Run `train_loo_full.py` to reproduce all outputs locally.
```

---

## Key Results

Under the LOO baseline protocol, GRU achieves the lowest average position RMSE at **0.5849 m** with the most consistent cross-flight generalization. Under high noise, GRU's position RMSE increases by **41.29%** compared to **93.28%** for AHLSTM. All models satisfy the 10 Hz real-time requirement, with GRU achieving **1552.8 fps** at **0.644 ms/step**.

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{karaaslan2025imu,
  title   = {IMU-Based UAV Navigation Under GPS Loss: Cross-Flight Generalization
             and Noise Robustness of Recurrent Deep Learning Architectures},
  author  = {Karaaslan, Mahmut and Kaya, Ersin},
  year    = {2025}
}
```

---

## License

This project is open-source and available under the [MIT License](LICENSE).
