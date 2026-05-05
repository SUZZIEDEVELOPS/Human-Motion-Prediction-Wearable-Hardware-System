# main_regression_hardware.py
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from src.feature_selection_reg import select_features
from src.normalize import normalize_features
from src.train_reg import train_regressor
from src.evaluate_reg import evaluate_model

# ── Config ────────────────────────────────────────────────────────────────────
RAW        = "data_hardware/hardware_features.csv"
K_FEATURES = 50

INPUT_IMUS = ["LEFTARM", "RIGHTLEG"]

AXIS_NAMES = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ",
              "MagX", "MagY", "MagZ", "AccMag"]
FEAT_NAMES = ["mean", "std", "min", "max", "range",
              "dominant_freq", "spectral_energy", "skewness", "kurtosis",
              "zcr", "jerk_mean", "jerk_std"]

ALL_COLS = [
    f"{imu}_{ax}_{fn}"
    for imu in INPUT_IMUS
    for ax in AXIS_NAMES
    for fn in FEAT_NAMES
] + [f"{imu}_acc_sma" for imu in INPUT_IMUS] \
  + [f"{imu}_{ax1}_{ax2}_corr"
     for imu in INPUT_IMUS
     for ax1, ax2 in [("AccX","AccY"),("AccX","AccZ"),("AccY","AccZ")]] \
  + [f"{imu}_vertical_dominance" for imu in INPUT_IMUS]

TARGET_COLS = [
    # Direction (ay, az only)
    "LEFTARM_AccY_mean",
    "LEFTARM_AccZ_mean",
    "RIGHTLEG_AccY_mean",
    "RIGHTLEG_AccZ_mean",
    # Cadence
    "LEFTARM_AccX_dominant_freq",
    "LEFTARM_AccY_dominant_freq",
    "LEFTARM_AccZ_dominant_freq",
    "LEFTARM_AccMag_dominant_freq",
    "RIGHTLEG_AccX_dominant_freq",
    "RIGHTLEG_AccY_dominant_freq",
    "RIGHTLEG_AccZ_dominant_freq",
    "RIGHTLEG_AccMag_dominant_freq",
]

# ── Remove target cols from features to prevent data leakage ──────────────────
FEATURE_COLS = [c for c in ALL_COLS if c not in TARGET_COLS]

leaked = [c for c in TARGET_COLS if c in FEATURE_COLS]
if leaked:
    raise ValueError(f"DATA LEAKAGE — targets found in features: {leaked}")
else:
    print(f"Leakage check passed ✓ ({len(FEATURE_COLS)} feature cols, "
          f"{len(TARGET_COLS)} target cols, no overlap)")

# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading:", os.path.abspath(RAW))
df = pd.read_csv(RAW)
print(f"Loaded {len(df)} windows")

missing = [c for c in FEATURE_COLS + TARGET_COLS + ["label"] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing[:5]}")

all_numeric_cols = list(dict.fromkeys(FEATURE_COLS + TARGET_COLS))
df[all_numeric_cols] = df[all_numeric_cols].apply(pd.to_numeric, errors="coerce")

print(f"\nClass counts:")
print(df["label"].value_counts())

# ── Pair window i → window i+1 within each activity group ────────────────────
X_features   = []
y_targets    = []
y_activities = []

for activity, group in df.groupby("label"):
    group = group.reset_index(drop=True)
    pairs = 0
    for i in range(len(group) - 1):
        x_row = group[FEATURE_COLS].iloc[i].values.astype(float)
        y_row = group[TARGET_COLS].iloc[i + 1].values.astype(float)

        if not (np.all(np.isfinite(x_row)) and np.all(np.isfinite(y_row))):
            continue

        X_features.append(x_row)
        y_targets.append(y_row)
        y_activities.append(activity)
        pairs += 1

    print(f"  {activity:<25s}  {pairs} pairs")

if not X_features:
    raise ValueError("No valid window pairs found.")

X_df       = pd.DataFrame(X_features, columns=FEATURE_COLS).fillna(0)
y          = np.array(y_targets)
activities = np.array(y_activities)

print(f"\nFeature matrix : {X_df.shape}")
print(f"Target matrix  : {y.shape}")

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, act_train, act_test = train_test_split(
    X_df, y, activities,
    test_size=0.2,
    random_state=42,
    stratify=activities
)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

# ── Feature Selection ─────────────────────────────────────────────────────────
X_train_sel, selector = select_features(X_train, y_train, k=K_FEATURES)
selected_cols         = X_train.columns[selector.get_support()].tolist()
X_test_sel            = X_test[selected_cols]

print(f"\nSelected {K_FEATURES} features:")
for c in selected_cols:
    print(f"  {c}")

# ── Normalise ─────────────────────────────────────────────────────────────────
X_train_norm, scaler = normalize_features(X_train_sel)
X_test_norm,  _      = normalize_features(X_test_sel, scaler=scaler)

assert np.all(np.isfinite(X_train_norm)), "NaN in training features"
assert np.all(np.isfinite(y_train)),      "NaN in training targets"
print(f"\nTraining on {len(X_train_norm)} clean samples ✓")

# ── Train ─────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

model = train_regressor(
    X_train_norm,
    y_train,
    model_path="models/trajectory_rf.pkl"
)
print("[SAVED] Regressor → models/trajectory_rf.pkl")

joblib.dump(scaler, "models/scaler_reg.pkl")
print("[SAVED] Scaler → models/scaler_reg.pkl")

with open("models/selected_features_reg.json", "w") as f:
    json.dump(selected_cols, f, indent=2)
print("[SAVED] Selected features → models/selected_features_reg.json")

with open("models/target_cols_reg.json", "w") as f:
    json.dump(TARGET_COLS, f, indent=2)
print("[SAVED] Target columns → models/target_cols_reg.json")

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n=== EVALUATION ON TEST SET ===")
metrics = evaluate_model(model, X_test_norm, y_test, act_test)

print("\n=== FINAL SUMMARY ===")
print(f"  ADE  : {metrics['ade']:.4f}")
print(f"  RMSE : {metrics['rmse']:.4f}")
print(f"  R²   : {metrics['r2']:.4f}")
print(f"  P50  : {metrics['p50']:.4f}")
print(f"  P90  : {metrics['p90']:.4f}")

print("\nPIPELINE FINISHED SUCCESSFULLY")