# extract_features.py
# Run on LAPTOP after collect_data.py
# Reads raw streams from data_hardware_raw/
# Outputs feature CSVs to data_hardware/

import os
import numpy as np
import pandas as pd
from preprocess import preprocess

# ── Config — must match training script ───────────────────────────────────────
SAMPLE_RATE  = 50
WINDOW_SEC   = 5.0
OVERLAP      = 0.75
WINDOW_SIZE  = int(SAMPLE_RATE * WINDOW_SEC)       # 250
STEP_SIZE    = int(WINDOW_SIZE * (1 - OVERLAP))    # 62

RAW_DIR  = "data_hardware_raw"
FEAT_DIR = "data_hardware"

IMU_NAMES  = ["CHEST", "LEFTARM", "RIGHTARM", "LEFTLEG", "RIGHTLEG"]
AXIS_NAMES = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ",
              "MagX", "MagY", "MagZ", "AccMag"]
IMU_COLS   = [f"{imu}_{ax}" for imu in IMU_NAMES for ax in AXIS_NAMES]


def extract_features(window_df, sampling_rate=50):
    features = {}

    # ── Per-axis features ─────────────────────────────────────────────────────
    for imu in IMU_NAMES:
        for ax in AXIS_NAMES:
            col = f"{imu}_{ax}"
            signal = window_df[col].to_numpy(dtype=float)

            features[f"{col}_mean"]  = float(np.mean(signal))
            features[f"{col}_std"]   = float(np.std(signal))
            features[f"{col}_min"]   = float(np.min(signal))
            features[f"{col}_max"]   = float(np.max(signal))
            features[f"{col}_range"] = float(np.max(signal) - np.min(signal))

            # Zero crossing rate
            zc = np.diff(np.sign(signal - np.mean(signal)))
            features[f"{col}_zcr"] = float(np.sum(zc != 0) / len(signal))

            # Jerk
            jerk = np.diff(signal) * sampling_rate
            features[f"{col}_jerk_mean"] = float(np.mean(np.abs(jerk)))
            features[f"{col}_jerk_std"]  = float(np.std(jerk))

            signal = signal[np.isfinite(signal)]
            if len(signal) < 2:
                for k in ["dominant_freq", "spectral_energy", "skewness", "kurtosis"]:
                    features[f"{col}_{k}"] = 0.0
                continue

            signal_d  = signal - np.mean(signal)
            fft_vals  = np.fft.fft(signal_d)
            fft_freqs = np.fft.fftfreq(len(signal), d=1.0 / sampling_rate)
            idx = np.argmax(np.abs(fft_vals[1:])) + 1
            features[f"{col}_dominant_freq"]   = float(fft_freqs[idx])
            features[f"{col}_spectral_energy"] = float(np.sum(np.abs(fft_vals) ** 2))

            std = np.std(signal)
            if std == 0:
                features[f"{col}_skewness"] = 0.0
                features[f"{col}_kurtosis"] = 0.0
            else:
                z = (signal - np.mean(signal)) / std
                features[f"{col}_skewness"] = float(np.mean(z ** 3))
                features[f"{col}_kurtosis"] = float(np.mean(z ** 4) - 3)

    # ── Per-IMU features ──────────────────────────────────────────────────────
    for imu in IMU_NAMES:
        acc = [window_df[f"{imu}_{ax}"].to_numpy(dtype=float) for ax in ["AccX", "AccY", "AccZ"]]

        # SMA
        features[f"{imu}_acc_sma"] = float(
            np.sum([np.sum(np.abs(s)) for s in acc]) / len(acc[0])
        )

        # Inter-axis correlations
        for (ax1, ax2) in [("AccX", "AccY"), ("AccX", "AccZ"), ("AccY", "AccZ")]:
            s1 = window_df[f"{imu}_{ax1}"].to_numpy(dtype=float)
            s2 = window_df[f"{imu}_{ax2}"].to_numpy(dtype=float)
            corr = float(np.corrcoef(s1, s2)[0, 1]) if np.std(s1) > 0 and np.std(s2) > 0 else 0.0
            features[f"{imu}_{ax1}_{ax2}_corr"] = corr

        # Vertical dominance
        acc_z = window_df[f"{imu}_AccZ"].to_numpy(dtype=float)
        acc_x = window_df[f"{imu}_AccX"].to_numpy(dtype=float)
        acc_y = window_df[f"{imu}_AccY"].to_numpy(dtype=float)
        features[f"{imu}_vertical_dominance"] = float(
            np.std(acc_z) / (np.std(acc_x) + np.std(acc_y) + 1e-6)
        )

    return features


def process_file(filepath):
    filename = os.path.basename(filepath)
    activity = '_'.join(filename.replace('.csv', '').split('_')[:-1])

    print(f"Processing {filename} → activity: {activity}")

    df = pd.read_csv(filepath)

    missing = [c for c in IMU_COLS if c not in df.columns]
    if missing:
        print(f"  Skipping — missing columns: {missing[:3]}")
        return None

    df, fs = preprocess(df)
    print(f"  Preprocessed — {len(df)} rows remaining at {fs}Hz")

    df_imu = df[IMU_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)

    if len(df_imu) < WINDOW_SIZE:
        print(f"  Skipping — not enough rows ({len(df_imu)} < {WINDOW_SIZE})")
        return None

    rows = []
    for start in range(0, len(df_imu) - WINDOW_SIZE, STEP_SIZE):
        window = df_imu.iloc[start:start + WINDOW_SIZE]
        feats  = extract_features(window, SAMPLE_RATE)
        feats["label"] = activity
        rows.append(feats)

    print(f"  {len(rows)} windows extracted")
    return pd.DataFrame(rows)


def main():
    os.makedirs(FEAT_DIR, exist_ok=True)

    for f in os.listdir(FEAT_DIR):
        if f.endswith(".csv"):
            os.remove(os.path.join(FEAT_DIR, f))
            print(f"Removed old: {f}")

    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if not raw_files:
        print(f"No CSV files found in {RAW_DIR}")
        return

    all_dfs = []
    for file in raw_files:
        path = os.path.join(RAW_DIR, file)
        result = process_file(path)
        if result is not None:
            all_dfs.append(result)

    if not all_dfs:
        print("No data processed")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = os.path.join(FEAT_DIR, "hardware_features.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} total windows → {out_path}")
    print("\nClass counts:")
    print(combined["label"].value_counts())


if __name__ == '__main__':
    main()