import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ── Fixed hardware config ─────────────────────────────────────────────────────
FS = 50.0  # ESP32 samples at exactly 50Hz — no need to detect

IMU_NAMES  = ["CHEST", "LEFTARM", "RIGHTARM", "LEFTLEG", "RIGHTLEG"]
AXIS_NAMES = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ",
              "MagX", "MagY", "MagZ", "AccMag"]
GPS_COLS   = ["GPS_LAT", "GPS_LNG", "GPS_ALT_M",
              "GPS_SPEED_KMH", "GPS_SATS", "GPS_VALID"]

ACC_AXES   = ["AccX", "AccY", "AccZ"]
GYRO_AXES  = ["GyroX", "GyroY", "GyroZ"]
MAG_AXES   = ["MagX", "MagY", "MagZ"]


def butter_lowpass_filter(data, cutoff, fs=FS, order=4):
    """Apply zero-phase Butterworth lowpass filter to a 1D array."""
    nyq    = 0.5 * fs
    normal = cutoff / nyq
    normal = np.clip(normal, 1e-4, 0.9999)
    b, a   = butter(order, normal, btype='low', analog=False)
    # Need at least padlen samples — skip filter if too short
    if len(data) < 3 * max(len(a), len(b)):
        return data
    return filtfilt(b, a, data)


def preprocess(df, cutoff_acc=5.0, cutoff_gyro=10.0, order=4):
    """
    Clean and filter a raw ESP32 CSV dataframe.

    Steps:
      1. Convert all sensor columns to numeric
      2. Drop rows where ALL sensor values are NaN
      3. Forward-fill then zero-fill remaining NaNs
      4. Butterworth lowpass filter on accel columns (cutoff=5Hz)
      5. Butterworth lowpass filter on gyro columns  (cutoff=10Hz)
      6. Recompute AccMag from filtered accel values

    Parameters
    ----------
    df          : pd.DataFrame  — raw rows from collect_data.py CSV
    cutoff_acc  : float         — accel lowpass cutoff frequency (Hz)
    cutoff_gyro : float         — gyro  lowpass cutoff frequency  (Hz)
    order       : int           — Butterworth filter order

    Returns
    -------
    df : pd.DataFrame — cleaned and filtered dataframe (same columns)
    fs : float        — sampling rate (always 50.0 for this hardware)
    """
    df = df.copy()

    # ── 1. Force numeric (coerce bad values to NaN) ───────────────────────────
    sensor_cols = [c for c in df.columns if c not in GPS_COLS]
    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # GPS columns — coerce but keep separately
    for col in GPS_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 2. Drop rows where ALL IMU sensor values are NaN ─────────────────────
    imu_cols = [c for c in sensor_cols if c not in GPS_COLS]
    df = df.dropna(subset=imu_cols, how="all").copy()

    if len(df) < 2:
        print("WARNING: DataFrame too short after NaN drop")
        return df, FS

    # ── 3. Fill remaining NaNs ────────────────────────────────────────────────
    df[imu_cols] = df[imu_cols].ffill().fillna(0.0)

    # ── 4. Butterworth lowpass on ACCEL columns ───────────────────────────────
    for imu in IMU_NAMES:
        for ax in ACC_AXES:
            col = f"{imu}_{ax}"
            if col in df.columns:
                df[col] = butter_lowpass_filter(
                    df[col].to_numpy(),
                    cutoff=cutoff_acc,
                    fs=FS,
                    order=order
                )

    # ── 5. Butterworth lowpass on GYRO columns ────────────────────────────────
    for imu in IMU_NAMES:
        for ax in GYRO_AXES:
            col = f"{imu}_{ax}"
            if col in df.columns:
                df[col] = butter_lowpass_filter(
                    df[col].to_numpy(),
                    cutoff=cutoff_gyro,
                    fs=FS,
                    order=order
                )

    # ── 6. Recompute AccMag from filtered accel values ────────────────────────
    for imu in IMU_NAMES:
        ax_col = f"{imu}_AccX"
        ay_col = f"{imu}_AccY"
        az_col = f"{imu}_AccZ"
        am_col = f"{imu}_AccMag"
        if all(c in df.columns for c in [ax_col, ay_col, az_col]):
            df[am_col] = np.sqrt(
                df[ax_col]**2 + df[ay_col]**2 + df[az_col]**2
            )

    return df, FS