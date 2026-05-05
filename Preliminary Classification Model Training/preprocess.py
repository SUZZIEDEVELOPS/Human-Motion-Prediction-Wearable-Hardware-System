

import numpy as np
import pandas as pd
from utils.filters import butter_lowpass_filter

def preprocess(
    df,
    time_col="Timestamp",
    cutoff_acc=5,
    acc_cols=("X_Acceleration", "Y_Acceleration", "Z_Acceleration"),
    gyro_cols=("X_AngularVelocity", "Y_AngularVelocity", "Z_AngularVelocity"),
    mag_cols=("X_MagneticField", "Y_MagneticField", "Z_MagneticField"),
    orient_cols=("X_Orientation", "Y_Orientation", "Z_Orientation"),
    order=4
):
    df = df.copy()

    # =========================================================
    # 1) PARSE TIME (handles datetime OR time-only OR numeric)
    # =========================================================

    col = df[time_col]

    # try datetime first (auto detect ANY format)
    t = pd.to_datetime(col, errors="coerce")

    # if mostly NaT → maybe numeric seconds
    if t.isna().mean() > 0.5:
        t = pd.to_numeric(col, errors="coerce")

    # if still mostly NaT → maybe time-only HH:MM:SS
    if isinstance(t, pd.Series) and t.isna().mean() > 0.5:
        t = pd.to_datetime(col, format="%H:%M:%S", errors="coerce")

    df[time_col] = t
    df = df.dropna(subset=[time_col])

    if len(df) < 2:
        return df, None

    # =========================================================
    # 2) TRUE SAMPLING RATE (ROBUST — handles coarse timestamps)
    # =========================================================

    if np.issubdtype(df[time_col].dtype, np.datetime64):
        t_sec = df[time_col].astype("int64") / 1e9
    else:
        t_sec = df[time_col].astype(float)

    diffs = np.diff(t_sec)
    diffs = diffs[diffs > 0]  # remove duplicate timestamps

    # if timestamps only have second resolution → cannot compute fs
    if len(diffs) == 0 or np.min(diffs) > 0.2:
        fs = 50.0  # known phone rate
        print("Coarse/duplicate timestamps → assuming fs=50Hz")
        return df, fs

    dt = np.min(diffs)  # smallest true interval
    fs = float(np.clip(1.0 / dt, 1, 500))

    # =========================================================
    # 3) DROP ROWS WITH MISSING SENSOR DATA
    # =========================================================
    # =========================================================
    # 3) DROP ONLY ROWS WHERE ALL SENSOR VALUES ARE NaN
    # =========================================================
    sensor_cols = [c for c in df.columns if c != time_col]

    df = df.dropna(subset=sensor_cols, how="all").copy()

    # =========================================================
    # 4) LOWPASS FILTER ACCEL ONLY
    # =========================================================
    for c in acc_cols:
        if c in df.columns:
            df[c] = butter_lowpass_filter(
                df[c].to_numpy(),
                cutoff=cutoff_acc,
                fs=fs,
                order=order
            )

    # =========================================================
    # 5) ACC MAGNITUDE
    # =========================================================
    if all(c in df.columns for c in acc_cols):
        ax, ay, az = acc_cols
        df["acc_mag"] = np.sqrt(df[ax]**2 + df[ay]**2 + df[az]**2)

    # =========================================================
    return df, fs
