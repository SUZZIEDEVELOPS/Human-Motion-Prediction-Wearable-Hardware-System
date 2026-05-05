# pi_realtime.py
# Role    : Receive ESP32 stream → window → extract features →
#           select (via JSON) → normalise → infer → majority vote → send label
# Run     : python3 pi_realtime.py

import serial
import numpy as np
import pandas as pd
import joblib
import json
import os
from collections import deque

# ── Serial config ─────────────────────────────────────────────────────────────
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE   = 115200

# ── ESP32 stream column names (56 total) ──────────────────────────────────────
ESP32_IMU_NAMES  = ["CHEST", "LEFTARM", "RIGHTARM", "LEFTLEG", "RIGHTLEG"]
ESP32_AXIS_NAMES = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ",
                    "MagX", "MagY", "MagZ", "AccMag"]
ESP32_IMU_COLS   = [f"{imu}_{ax}"
                    for imu in ESP32_IMU_NAMES
                    for ax in ESP32_AXIS_NAMES]   # 50 cols
GPS_COLS         = ["GPS_LAT", "GPS_LNG", "GPS_ALT_M",
                    "GPS_SPEED_KMH", "GPS_SATS", "GPS_VALID"]
ALL_COLS         = ESP32_IMU_COLS + GPS_COLS      # 56 cols
N_TOTAL          = len(ALL_COLS)

# ── Feature names — must match training pipeline exactly ──────────────────────
FEAT_NAMES = ["mean", "std", "min", "max", "range",
              "dominant_freq", "spectral_energy", "skewness", "kurtosis",
              "zcr", "jerk_mean", "jerk_std"]  # 12 features per axis

# Full 625-feature column list — same as ALL_COLS in training main.py
TRAINING_FEATURE_COLS = [
    f"{imu}_{ax}_{fn}"
    for imu in ESP32_IMU_NAMES
    for ax in ESP32_AXIS_NAMES
    for fn in FEAT_NAMES
] + [f"{imu}_acc_sma" for imu in ESP32_IMU_NAMES] \
  + [f"{imu}_{ax1}_{ax2}_corr"
     for imu in ESP32_IMU_NAMES
     for ax1, ax2 in [("AccX","AccY"),("AccX","AccZ"),("AccY","AccZ")]] \
  + [f"{imu}_vertical_dominance" for imu in ESP32_IMU_NAMES]

# ── Windowing — MUST match training pipeline ──────────────────────────────────
SAMPLE_RATE  = 50
WINDOW_SEC   = 5.0
OVERLAP      = 0.75
WINDOW_SIZE  = int(SAMPLE_RATE * WINDOW_SEC)        # 250 samples
STEP_SIZE    = int(WINDOW_SIZE * (1 - OVERLAP))     # 62 samples

# ── Majority vote ─────────────────────────────────────────────────────────────
VOTE_WINDOW   = 6
VOTE_REQUIRED = 5

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading model, scaler, selected features...")
model  = joblib.load("activity_adaboost.pkl")
scaler = joblib.load("scaler.pkl")

with open("selected_features.json") as f:
    selected_features = json.load(f)

print(f"Model loaded OK")
print(f"Selected features ({len(selected_features)}): {selected_features}")

# ── Terminal display ──────────────────────────────────────────────────────────
def display_activity_terminal(raw_label, confirmed, tally, buffer_len):
    os.system('clear')
    print("=" * 35)
    print("      ACTIVITY MONITOR (Pi)")
    print("=" * 35)

    if buffer_len < WINDOW_SIZE:
        print(f"\n  Buffering: {buffer_len}/{WINDOW_SIZE} samples")
        print(f"  ({WINDOW_SIZE - buffer_len} more needed for first prediction)")
    else:
        print(f"\n  Current:   {raw_label}")
        print(f"  Confirmed: {confirmed or 'Waiting for stable vote...'}")
        print(f"\n  Votes: {tally}")

    print("=" * 35)

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(window_df, sampling_rate=50):
    """
    window_df : DataFrame shape (WINDOW_SIZE, 50) — IMU columns only
    Returns   : dict with 625 keys matching training feature names
    """
    features = {}

    # ── Per-axis features ─────────────────────────────────────────────────────
    for imu in ESP32_IMU_NAMES:
        for ax in ESP32_AXIS_NAMES:
            col_name = f"{imu}_{ax}"

            if col_name not in window_df.columns:
                for fn in FEAT_NAMES:
                    features[f"{col_name}_{fn}"] = 0.0
                continue

            signal = window_df[col_name].to_numpy(dtype=float)

            # Time domain
            features[f"{col_name}_mean"]  = float(np.mean(signal))
            features[f"{col_name}_std"]   = float(np.std(signal))
            features[f"{col_name}_min"]   = float(np.min(signal))
            features[f"{col_name}_max"]   = float(np.max(signal))
            features[f"{col_name}_range"] = float(np.max(signal) - np.min(signal))

            # Zero crossing rate
            zc = np.diff(np.sign(signal - np.mean(signal)))
            features[f"{col_name}_zcr"] = float(np.sum(zc != 0) / len(signal))

            # Jerk
            jerk = np.diff(signal) * sampling_rate
            features[f"{col_name}_jerk_mean"] = float(np.mean(np.abs(jerk)))
            features[f"{col_name}_jerk_std"]  = float(np.std(jerk))

            signal = signal[np.isfinite(signal)]
            if len(signal) < 2:
                for fn in ["dominant_freq", "spectral_energy",
                           "skewness", "kurtosis"]:
                    features[f"{col_name}_{fn}"] = 0.0
                continue

            # Frequency domain
            signal_d  = signal - np.mean(signal)
            fft_vals  = np.fft.fft(signal_d)
            fft_freqs = np.fft.fftfreq(len(signal), d=1.0 / sampling_rate)
            idx = np.argmax(np.abs(fft_vals[1:])) + 1
            features[f"{col_name}_dominant_freq"]   = float(fft_freqs[idx])
            features[f"{col_name}_spectral_energy"] = float(
                np.sum(np.abs(fft_vals) ** 2))

            # Skewness and kurtosis
            std = np.std(signal)
            if std == 0:
                features[f"{col_name}_skewness"] = 0.0
                features[f"{col_name}_kurtosis"] = 0.0
            else:
                z = (signal - np.mean(signal)) / std
                features[f"{col_name}_skewness"] = float(np.mean(z ** 3))
                features[f"{col_name}_kurtosis"] = float(np.mean(z ** 4) - 3)

    # ── Per-IMU features ──────────────────────────────────────────────────────
    for imu in ESP32_IMU_NAMES:
        acc = [window_df[f"{imu}_{ax}"].to_numpy(dtype=float)
               if f"{imu}_{ax}" in window_df.columns
               else np.zeros(WINDOW_SIZE)
               for ax in ["AccX", "AccY", "AccZ"]]

        # SMA
        features[f"{imu}_acc_sma"] = float(
            np.sum([np.sum(np.abs(s)) for s in acc]) / len(acc[0])
        )

        # Inter-axis correlations
        for (ax1, ax2) in [("AccX","AccY"),("AccX","AccZ"),("AccY","AccZ")]:
            col1 = f"{imu}_{ax1}"
            col2 = f"{imu}_{ax2}"
            s1 = window_df[col1].to_numpy(dtype=float) if col1 in window_df.columns else np.zeros(WINDOW_SIZE)
            s2 = window_df[col2].to_numpy(dtype=float) if col2 in window_df.columns else np.zeros(WINDOW_SIZE)
            corr = float(np.corrcoef(s1, s2)[0, 1]) if np.std(s1) > 0 and np.std(s2) > 0 else 0.0
            features[f"{imu}_{ax1}_{ax2}_corr"] = corr

        # Vertical dominance
        acc_z = window_df[f"{imu}_AccZ"].to_numpy(dtype=float) if f"{imu}_AccZ" in window_df.columns else np.zeros(WINDOW_SIZE)
        acc_x = window_df[f"{imu}_AccX"].to_numpy(dtype=float) if f"{imu}_AccX" in window_df.columns else np.zeros(WINDOW_SIZE)
        acc_y = window_df[f"{imu}_AccY"].to_numpy(dtype=float) if f"{imu}_AccY" in window_df.columns else np.zeros(WINDOW_SIZE)
        features[f"{imu}_vertical_dominance"] = float(
            np.std(acc_z) / (np.std(acc_x) + np.std(acc_y) + 1e-6)
        )

    return features

# ── Majority vote ─────────────────────────────────────────────────────────────
prediction_history = deque(maxlen=VOTE_WINDOW)
confirmed_label    = None

def majority_vote(new_label):
    global confirmed_label
    prediction_history.append(new_label)

    if len(prediction_history) < VOTE_WINDOW:
        return None

    votes = {}
    for label in prediction_history:
        votes[label] = votes.get(label, 0) + 1

    top_label, top_count = max(votes.items(), key=lambda x: x[1])

    if top_count >= VOTE_REQUIRED:
        if top_label != confirmed_label:
            confirmed_label = top_label
            return confirmed_label

    return None

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print(f"Opening {SERIAL_PORT} at {BAUD_RATE} baud...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected. Window={WINDOW_SIZE} samples ({WINDOW_SEC}s), "
          f"Step={STEP_SIZE} samples")

    buffer       = []
    sample_count = 0

    while True:

        # 1. Read line
        try:
            raw = ser.readline().decode('utf-8', errors='ignore').strip()
        except Exception as e:
            print(f"Serial error: {e}")
            continue

        if not raw:
            continue
        if raw.startswith('#'):
            print(f"ESP32: {raw}")
            continue

        # 2. Parse CSV — tolerant of ovf, nan, empty fields
        try:
            values = [float(x) if x not in ('ovf', 'nan', '') else 0.0
                      for x in raw.split(',')]
        except ValueError:
            print(f"Bad row (skipped): {raw[:50]}")
            continue

        if len(values) != N_TOTAL:
            print(f"Column mismatch: got {len(values)}, expected {N_TOTAL}")
            continue

        # 3. Buffer
        buffer.append(values)
        sample_count += 1

        if len(buffer) > WINDOW_SIZE * 2:
            buffer = buffer[-WINDOW_SIZE:]

        # 4. Show buffering progress in terminal
        if len(buffer) < WINDOW_SIZE:
            display_activity_terminal(None, confirmed_label, {}, len(buffer))
            continue

        # 5. Only infer every STEP_SIZE samples
        if sample_count % STEP_SIZE != 0:
            continue

        # 6. Build window DataFrame (IMU cols only)
        window_np  = np.array(buffer[-WINDOW_SIZE:], dtype=np.float32)
        window_df  = pd.DataFrame(window_np, columns=ALL_COLS)
        window_imu = window_df[ESP32_IMU_COLS]

        # 7. Feature extraction → 625 features
        feats_dict = extract_features(window_imu, SAMPLE_RATE)
        feats_df   = pd.DataFrame([feats_dict])[TRAINING_FEATURE_COLS].fillna(0)

        # 8. Feature selection → 150 features
        X_selected = feats_df[selected_features].to_numpy()

        # 9. Normalise
        X_scaled = scaler.transform(X_selected)

        # 10. Inference
        try:
            raw_label = model.predict(X_scaled)[0]
        except Exception as e:
            print(f"Inference error: {e}")
            ser.write(b"error\n")
            continue

        # 11. Majority vote
        stable = majority_vote(raw_label)

        tally = {l: list(prediction_history).count(l)
                 for l in set(prediction_history)}

        # 12. Display in terminal
        display_activity_terminal(raw_label, confirmed_label, tally, len(buffer))

        # 13. Send confirmed label to ESP32
        if stable:
            print(f"\n>>> Sending to ESP32: {stable}")
            ser.write((stable + '\n').encode('utf-8'))

if __name__ == '__main__':
    main()