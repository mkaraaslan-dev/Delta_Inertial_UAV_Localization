import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gc
import time
import platform

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import AHLSTMModel, LSTMModel, BiLSTMModel, GRUModel

# ===========================================================================
# AYARLAR
# ===========================================================================
SEQUENCE_LENGTH   = 20
INPUT_SIZE        = 16
OUTPUT_SIZE       = 3
HIDDEN_SIZE       = 256
NUM_LAYERS        = 2
DROPOUT           = 0.0

ITERATIONS        = 500
WARMUP            = 50
DEVICE            = 'cuda' if torch.cuda.is_available() else 'cpu'
REALTIME_LIMIT_MS = 100.0       # 10 Hz UAV -> max 100 ms/adim

BASE_RESULTS_DIR  = 'results'
OUT_DIR           = 'inference_results'
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_CONFIGS = [
    ('LSTMModel',   LSTMModel),
    ('BiLSTMModel', BiLSTMModel),
    ('GRUModel',    GRUModel),
    ('AHLSTMModel', AHLSTMModel),
]

print(f"\nBENCHMARK: UAV IMU Position Models @ seq={SEQUENCE_LENGTH}, input={INPUT_SIZE}")
print(f"   Device    : {DEVICE.upper()}")
if DEVICE == 'cuda':
    print(f"   GPU       : {torch.cuda.get_device_name(0)}")
print(f"   Platform  : {platform.processor()}")
print(f"   Iterations: {ITERATIONS}  |  Warmup: {WARMUP}")
print("-" * 60)

# ===========================================================================
# YARDIMCI FONKSIYONLAR
# ===========================================================================

def load_model(model_class, model_name):
    model = model_class(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                        OUTPUT_SIZE, DROPOUT).to(DEVICE)
    for flight_id in range(1, 10):
        path = os.path.join(BASE_RESULTS_DIR,
                            f'flight_{flight_id:02d}',
                            model_name,
                            f'{model_name}_model.pth')
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location='cpu'))
            break
    model.eval()
    return model


def get_model_size_mb(model_name):
    for flight_id in range(1, 10):
        path = os.path.join(BASE_RESULTS_DIR,
                            f'flight_{flight_id:02d}',
                            model_name,
                            f'{model_name}_model.pth')
        if os.path.exists(path):
            return os.path.getsize(path) / (1024 * 1024)
    return float('nan')


# ===========================================================================
# BENCHMARK DONGUSU
# ===========================================================================
dummy_input = torch.randn(1, SEQUENCE_LENGTH, INPUT_SIZE,
                          dtype=torch.float32).to(DEVICE)
rows = []

for model_name, model_class in MODEL_CONFIGS:
    print(f"\nAnalyzing: {model_name}")

    model  = load_model(model_class, model_name)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    size   = get_model_size_mb(model_name)

    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warm-up
    print("   Warming up...")
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(dummy_input)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()

    # Hiz testi
    print(f"   Running {ITERATIONS} iterations...")

    if DEVICE == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            for _ in range(ITERATIONS):
                _ = model(dummy_input)
        end_event.record()
        torch.cuda.synchronize()
        total_ms = start_event.elapsed_time(end_event)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(ITERATIONS):
                _ = model(dummy_input)
        total_ms = (time.perf_counter() - t0) * 1000

    latency = total_ms / ITERATIONS
    fps     = 1000.0 / latency
    vram    = torch.cuda.max_memory_allocated() / (1024 ** 2) if DEVICE == 'cuda' else float('nan')

    rt_ok = latency < REALTIME_LIMIT_MS
    print(f"   FPS       : {fps:.2f}")
    print(f"   Latency   : {latency:.4f} ms")
    print(f"   VRAM      : {vram:.2f} MB" if DEVICE == 'cuda' else "   VRAM      : N/A (CPU)")
    print(f"   Params    : {params:.4f} M")
    print(f"   Size      : {size:.3f} MB")
    print(f"   Real-time : {'OK (< 100 ms)' if rt_ok else 'EXCEEDS LIMIT'}")

    rows.append({
        'Model':              model_name,
        'Parameters (M)':     round(params, 4),
        'Size (MB)':          round(size, 3),
        'Latency (ms)':       round(latency, 4),
        'FPS':                round(fps, 2),
        'Peak VRAM (MB)':     round(vram, 2),
        'Real-time (100ms)':  'OK' if rt_ok else 'EXCEEDS',
    })

    del model
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

# ===========================================================================
# KAYIT
# ===========================================================================
df = pd.DataFrame(rows)
df.to_csv(  os.path.join(OUT_DIR, 'inference_summary.csv'),   index=False)
df.to_excel(os.path.join(OUT_DIR, 'inference_summary.xlsx'),  index=False)
print(f"\nSaved: inference_summary.csv / .xlsx")

# ===========================================================================
# GRAFIKLER
# ===========================================================================
colors      = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
model_names = [r['Model'].replace('Model', '') for r in rows]
latencies   = [r['Latency (ms)']    for r in rows]
fps_vals    = [r['FPS']             for r in rows]
vram_vals   = [r['Peak VRAM (MB)']  for r in rows]

def styled_bar(ax, names, values, colors, fmt, ylabel, title, pad=0.08):
    x      = np.arange(len(names))
    bars   = ax.bar(x, values, color=colors, alpha=0.85,
                    edgecolor='black', linewidth=0.8, width=0.5)
    margin = (max(values) - min(values)) * pad + max(values) * 0.02
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + margin * 0.4,
                fmt.format(val), ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(0, max(values) + margin * 2.5)
    ax.grid(True, alpha=0.25, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return bars

# ---- Latency ----
fig, ax = plt.subplots(figsize=(9, 6))
styled_bar(ax, model_names, latencies, colors,
           fmt='{:.3f} ms', ylabel='Latency (ms)',
           title='Inference Latency Comparison (Lower is Better ↓)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'bench_result_latency.png'), dpi=150)
plt.close()
print("Saved: bench_result_latency.png")

# ---- FPS ----
fig, ax = plt.subplots(figsize=(9, 6))
styled_bar(ax, model_names, fps_vals, colors,
           fmt='{:.1f} fps', ylabel='Throughput (fps)',
           title='Inference Speed Comparison (Higher is Better ↑)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'bench_result_throughput.png'), dpi=150)
plt.close()
print("Saved: bench_result_throughput.png")

# ---- VRAM (sadece CUDA) ----
if DEVICE == 'cuda' and not any(np.isnan(vram_vals)):
    fig, ax = plt.subplots(figsize=(9, 6))
    styled_bar(ax, model_names, vram_vals, colors,
               fmt='{:.2f} MB', ylabel='Peak VRAM (MB)',
               title='Peak GPU Memory Usage (Lower is Better ↓)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'bench_result_vram.png'), dpi=150)
    plt.close()
    print("Saved: bench_result_vram.png")

# ===========================================================================
# KONSOL OZET
# ===========================================================================
print(f"\n{'='*60}")
print(f"  RESULTS — {DEVICE.upper()} | n={ITERATIONS}")
print(f"{'='*60}")
print(f"  {'Model':<14} {'Latency(ms)':>12} {'FPS':>8} {'VRAM(MB)':>10} {'RT?':>6}")
print(f"  {'-'*54}")
for r in rows:
    rt = 'YES' if r['Real-time (100ms)'] == 'OK' else 'NO'
    vr = f"{r['Peak VRAM (MB)']:.2f}" if not np.isnan(r['Peak VRAM (MB)']) else 'N/A'
    print(f"  {r['Model'].replace('Model',''):<14} "
          f"{r['Latency (ms)']:>12.4f} "
          f"{r['FPS']:>8.1f} "
          f"{vr:>10} "
          f"{rt:>6}")
print(f"{'='*60}")
print(f"\nAll outputs saved to: {OUT_DIR}/")
