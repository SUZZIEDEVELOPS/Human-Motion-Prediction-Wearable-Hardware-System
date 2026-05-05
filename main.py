# main.py
# Runs on LAPTOP after extract_features.py has been run
# Reads feature CSVs from data_hardware/
# Trains AdaBoost, saves model + scaler + selected features to models/


import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.feature_selection import select_features
from src.normalize import normalize_features
from src.train import train_classifier
from src.evaluate import evaluate_model

# ── Config ────────────────────────────────────────────────────────────────────
RAW = "data_hardware"   # folder containing feature CSVs from extract_features.py

IMU_NAMES  = ["CHEST", "LEFTARM", "RIGHTARM", "LEFTLEG", "RIGHTLEG"]
AXIS_NAMES = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ",
              "MagX", "MagY", "MagZ", "AccMag"]

FEAT_NAMES = ["mean", "std", "min", "max", "range",
              "dominant_freq", "spectral_energy", "skewness", "kurtosis",
              "zcr", "jerk_mean", "jerk_std"]
#Per-axis (5 IMU × 10 axes × 12 features)600
ALL_COLS = [
    f"{imu}_{ax}_{fn}"
    for imu in IMU_NAMES
    for ax in AXIS_NAMES
    for fn in FEAT_NAMES
] + [f"{imu}_acc_sma" for imu in IMU_NAMES] \
  + [f"{imu}_{ax1}_{ax2}_corr"
     for imu in IMU_NAMES
     for ax1, ax2 in [("AccX","AccY"),("AccX","AccZ"),("AccY","AccZ")]] \
  + [f"{imu}_vertical_dominance" for imu in IMU_NAMES]


# ── Load Data ─────────────────────────────────────────────────────────────────
print("RAW folder:", os.path.abspath(RAW))
all_files = [f for f in os.listdir(RAW) if f.endswith(".csv")]
print("Total CSV files:", len(all_files))

dfs = []

for file in all_files:
    path = os.path.join(RAW, file)
    df   = pd.read_csv(path)

    # Validate columns
    missing = [c for c in ALL_COLS + ["label"] if c not in df.columns]
    extra   = [c for c in df.columns if c not in set(ALL_COLS + ["label"])]

    if missing:
        print(f"Skipping {file}: missing columns: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        continue
    if extra:
        print(f"Note: {file} has extra columns (ignored): {extra}")

    # Force exact column order to match Pi inference
    df = df[ALL_COLS + ["label"]]
    dfs.append(df)
    print(f"Loaded {file}: {df.shape[0]} windows")

if not dfs:
    raise ValueError("No valid feature CSV files found in data_hardware/. "
                     "Run extract_features.py first.")

# ── Combine All Sessions ──────────────────────────────────────────────────────
full_df = pd.concat(dfs, ignore_index=True)

X_df = full_df[ALL_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
y    = full_df["label"].astype(str)

print("\nFeature matrix shape:", X_df.shape)   # (n_windows, 450)
print("Label vector shape:  ", y.shape)
print("\nClass counts:")
print(y.value_counts())

# ── Train / Test Split ────────────────────────────────────────────────────────
# 80% training, 20% testing
# stratify=y ensures each activity is proportionally represented in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X_df,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain: {len(X_train)} windows")
print(f"Test:  {len(X_test)} windows")

# ── Feature Selection ─────────────────────────────────────────────────────────
# Only fit on training data — never test data
X_train_sel, selector  = select_features(X_train, y_train)
selected_cols          = X_train.columns[selector.get_support()].tolist()
X_test_sel             = X_test[selected_cols]

print(f"\nSelected {len(selected_cols)} features:")
for col in selected_cols:
    print(f"  {col}")

# ── Normalise ─────────────────────────────────────────────────────────────────
# Fit scaler on training data only, apply same scaler to test data
X_train_norm, scaler = normalize_features(X_train_sel)
X_test_norm, _       = normalize_features(X_test_sel, scaler=scaler)

# ── Train ─────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

model = train_classifier(
    X_train_norm,
    y_train,
    model_path="models/activity_adaboost.pkl"
)
print("\n[SAVED] Model → models/activity_adaboost.pkl")

# ── Save Scaler ───────────────────────────────────────────────────────────────
joblib.dump(scaler, "models/scaler.pkl")
print("[SAVED] Scaler → models/scaler.pkl")

# ── Save Selected Feature Names ───────────────────────────────────────────────
# Pi loads this JSON to know which 20 features to extract at runtime
with open("models/selected_features.json", "w") as f:
    json.dump(selected_cols, f, indent=2)
print(f"[SAVED] Selected features → models/selected_features.json")

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n=== EVALUATION ON TEST SET ===")
evaluate_model(model, X_test_norm, y_test)

print("\nTrain class counts:\n", y_train.value_counts())
print("\nTest class counts:\n",  y_test.value_counts())
print("\nPIPELINE FINISHED SUCCESSFULLY")
print("\nFiles ready to copy to Pi:")
print("  models/activity_adaboost.pkl")
print("  models/scaler.pkl")
print("  models/selected_features.json")
