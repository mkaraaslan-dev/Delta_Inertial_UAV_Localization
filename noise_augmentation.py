"""
noise_augmentation.py
=====================
IMU sensör verisi için gürültü enjeksiyonu modülü.

Literatürde kullanılan üç temel gürültü türü uygulanmaktadır:
  1. Gaussian White Noise  — sıfır ortalama, sabit standart sapma
  2. Bias Drift            — her eksene sabit bir offset eklenir
  3. Scale Factor Error    — ölçüm değeri belirli bir yüzde oranında sapma gösterir

Referanslar:
  - Choi et al. (2023), Sensors: IMU data augmentation for orientation estimation
  - Han et al. (2021), Micromachines: Hybrid RNN for MEMS-IMU noise reduction
  - Liu et al. (2022), IEEE Sensors Journal: Deep learning for GPS outage bridging

Kullanım:
    from noise_augmentation import apply_noise

    # X_train: (N, 16) numpy array — ham, ölçeklendirilmemiş IMU verisi
    X_noisy = apply_noise(X_train, level='low')
    # level: 'none' | 'low' | 'medium' | 'high'
"""

import numpy as np

# ---------------------------------------------------------------------------
# Sütun grupları — INPUT_COLS sıralamasıyla eşleşmeli
# ---------------------------------------------------------------------------
# INPUT_COLS = ['qx','qy','qz','qw', 'roll','yaw','pitch',
#               'roll_a','pitch_a','yaw_a', 'acc_x','acc_y','acc_z',
#               'c_x','c_y','c_z']

QUATERNION_COLS  = [0, 1, 2, 3]        # qx, qy, qz, qw
EULER_COLS       = [4, 5, 6]           # roll, yaw, pitch
GYRO_COLS        = [7, 8, 9]           # roll_a, pitch_a, yaw_a
ACC_COLS         = [10, 11, 12]        # acc_x, acc_y, acc_z
COMPASS_COLS     = [13, 14, 15]        # c_x, c_y, c_z

# ---------------------------------------------------------------------------
# Gürültü parametreleri — her seviye için (sigma, bias_range, scale_range)
# ---------------------------------------------------------------------------
# sigma       : Gaussian gürültü standart sapması (sensörün tipik gürültü seviyesine göre)
# bias_range  : Bias drift için düzgün dağılım aralığı [-b, +b]
# scale_range : Scale factor hatası için yüzde aralığı (örn. 0.02 → ±%2)

NOISE_PARAMS = {
    'none': {
        'gyro':    {'sigma': 0.0,    'bias': 0.0,    'scale': 0.0},
        'acc':     {'sigma': 0.0,    'bias': 0.0,    'scale': 0.0},
        'euler':   {'sigma': 0.0,    'bias': 0.0,    'scale': 0.0},
        'compass': {'sigma': 0.0,    'bias': 0.0,    'scale': 0.0},
        'quat':    {'sigma': 0.0,    'bias': 0.0,    'scale': 0.0},
    },
    'low': {
        # Gerçekçi düşük kaliteli MEMS IMU seviyesi
        'gyro':    {'sigma': 0.005,  'bias': 0.002,  'scale': 0.005},
        'acc':     {'sigma': 0.010,  'bias': 0.005,  'scale': 0.005},
        'euler':   {'sigma': 0.005,  'bias': 0.002,  'scale': 0.003},
        'compass': {'sigma': 0.008,  'bias': 0.003,  'scale': 0.003},
        'quat':    {'sigma': 0.003,  'bias': 0.001,  'scale': 0.002},
    },
    'medium': {
        # Orta seviye bozulma
        'gyro':    {'sigma': 0.020,  'bias': 0.010,  'scale': 0.020},
        'acc':     {'sigma': 0.040,  'bias': 0.020,  'scale': 0.020},
        'euler':   {'sigma': 0.020,  'bias': 0.010,  'scale': 0.010},
        'compass': {'sigma': 0.030,  'bias': 0.015,  'scale': 0.010},
        'quat':    {'sigma': 0.010,  'bias': 0.005,  'scale': 0.008},
    },
    'high': {
        # Yüksek gürültü — elektromanyetik girişim / sıcaklık kayması senaryosu
        'gyro':    {'sigma': 0.050,  'bias': 0.030,  'scale': 0.050},
        'acc':     {'sigma': 0.100,  'bias': 0.050,  'scale': 0.050},
        'euler':   {'sigma': 0.050,  'bias': 0.030,  'scale': 0.030},
        'compass': {'sigma': 0.080,  'bias': 0.040,  'scale': 0.030},
        'quat':    {'sigma': 0.025,  'bias': 0.015,  'scale': 0.020},
    },
}


def _add_gaussian(data, sigma):
    """Gaussian beyaz gürültü ekle."""
    if sigma == 0.0:
        return data
    return data + np.random.normal(0.0, sigma, data.shape)


def _add_bias(data, bias_range):
    """
    Her eksene bağımsız sabit bir bias offset ekle.
    Bias, her çağrıda bir kez örneklenir — tüm zaman adımlarına aynı değer eklenir.
    Bu, gerçek sensör bias drift davranışını yansıtır.
    """
    if bias_range == 0.0:
        return data
    n_cols = data.shape[1]
    bias   = np.random.uniform(-bias_range, bias_range, (1, n_cols))
    return data + bias


def _add_scale_error(data, scale_range):
    """
    Scale factor hatası: her eksene bağımsız bir ölçek çarpanı uygula.
    Çarpan 1 ± scale_range aralığından örneklenir.
    """
    if scale_range == 0.0:
        return data
    n_cols = data.shape[1]
    scale  = 1.0 + np.random.uniform(-scale_range, scale_range, (1, n_cols))
    return data * scale


def _apply_to_group(X, col_indices, params):
    """Belirtilen sütun grubuna üç gürültü türünü sırayla uygula."""
    group = X[:, col_indices].copy()
    group = _add_gaussian(group,      params['sigma'])
    group = _add_bias(group,          params['bias'])
    group = _add_scale_error(group,   params['scale'])
    X[:, col_indices] = group
    return X


def apply_noise(X_train, level='low', seed=None):
    """
    Ham IMU eğitim verisine gürültü enjeksiyonu uygula.

    Parametreler
    ------------
    X_train : np.ndarray, shape (N, 16)
        Ham, ölçeklendirilmemiş eğitim giriş verisi.
    level : str
        Gürültü seviyesi: 'none' | 'low' | 'medium' | 'high'
    seed : int veya None
        Tekrarlanabilirlik için rastgele tohum.

    Döndürür
    --------
    X_noisy : np.ndarray, shape (N, 16)
        Gürültü eklenmiş eğitim verisi. Orijinal dizi değiştirilmez.
    """
    if level not in NOISE_PARAMS:
        raise ValueError(f"Geçersiz gürültü seviyesi: '{level}'. "
                         f"Geçerli seçenekler: {list(NOISE_PARAMS.keys())}")

    if seed is not None:
        np.random.seed(seed)

    params = NOISE_PARAMS[level]
    X_noisy = X_train.copy().astype(np.float64)

    X_noisy = _apply_to_group(X_noisy, QUATERNION_COLS, params['quat'])
    X_noisy = _apply_to_group(X_noisy, EULER_COLS,      params['euler'])
    X_noisy = _apply_to_group(X_noisy, GYRO_COLS,       params['gyro'])
    X_noisy = _apply_to_group(X_noisy, ACC_COLS,        params['acc'])
    X_noisy = _apply_to_group(X_noisy, COMPASS_COLS,    params['compass'])

    return X_noisy.astype(np.float32)


def noise_summary(level):
    """Seçilen gürültü seviyesinin parametrelerini yazdır."""
    if level not in NOISE_PARAMS:
        print(f"Bilinmeyen seviye: {level}")
        return
    print(f"\nGürültü Seviyesi: {level.upper()}")
    print(f"{'Grup':<12} {'Sigma':>8} {'Bias':>8} {'Scale':>8}")
    print("-" * 40)
    for group, p in NOISE_PARAMS[level].items():
        print(f"  {group:<10} {p['sigma']:>8.4f} {p['bias']:>8.4f} {p['scale']:>8.4f}")
