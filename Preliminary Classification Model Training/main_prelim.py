import os
import pandas as pd
import numpy as np
from src.load_data import process_file
#from src.load_data import load_label_from_filename
from src.preprocess import preprocess
from utils.windowing import sliding_window
from sklearn.model_selection import train_test_split
from src.feature_engineering import extract_features
from src.feature_selection import select_features
from src.normalize import normalize_features
from src.train import train_classifier
from src.evaluate import evaluate_model
from sklearn.metrics import classification_report


RAW = "data_raw"

print("RAW folder:", os.path.abspath(RAW))
all_files = os.listdir(RAW)
print("Total files in RAW:", len(all_files))


X_features = []
y_activity = []


# --------- LOOP OVER FILES --------------
for file in os.listdir(RAW):
    if not file.endswith(".csv"):
        continue

    path = os.path.join(RAW, file)

    # Load + metadata + sensor detection
    df, activity, mount, sensors = process_file(path)

    # PREPROCESS (df IS DEFINED HERE)
    df_clean, fs = preprocess(df)

    if fs is None:
        print(f"Skipping {file}: cannot compute sampling rate")
        continue

    # ----- WINDOW SETTINGS (1 second windows, 75% overlap) -----
    #window_seconds = 1.0  # samples per "" second
    window_seconds = 5.0 # samples per "" second testing diff window sizes
    overlap = 0.75 # 75%
    window_size = int(fs * window_seconds)
    step_size = int(window_size * (1 - overlap))

    #windows = sliding_window(df_clean, window_size, step_size)
    windows = list(sliding_window(df_clean, window_size, step_size))
    print("Num windows:", len(windows))

    #Debug windows
    print(f"\nProcessing {file}")
    print("Samples:", len(df_clean))
    print("fs:", fs)
    print("window_size:", window_size)
    print("step_size:", step_size)

    #--------- FEATURE EXTRACTION per window and attach SAME file labels---------
    for w in windows:
        # Keep only numeric columns (drops Timestamp automatically)
        feature_cols = w.select_dtypes(include=[np.number]).columns
        #feature_cols = w.columns  # or restrict per sensor later
        feats = extract_features(w, feature_cols, fs)
        X_features.append(feats)
        y_activity.append(activity)

# Convert list of dicts -> feature matrix
X_df = pd.DataFrame(X_features).fillna(0)
y = pd.Series(y_activity, name="label")

print("Feature matrix shape:", X_df.shape)
print("Label vector shape:", y.shape)

# Class distribution in feature matrix
print("\n--- Activity Class Distribution ---")
print(y.value_counts())
print("\nAs percentages:")
print(y.value_counts(normalize=True).mul(100).round(1).astype(str) + '%')


# ----- SPLIT FIRST -----
X_train, X_test, y_train, y_test = train_test_split(
    X_df,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----- FEATURE SELECTION (train only) -----
X_train_sel, selector = select_features(X_train, y_train)
#X_test_sel = X_test.loc[:, selector.get_support()]
selected_cols = X_train.columns[selector.get_support()]
X_test_sel = X_test[selected_cols]

# ----- NORMALISE -----
X_train_norm, scaler = normalize_features(X_train_sel)
X_test_norm, _ = normalize_features(X_test_sel, scaler=scaler)

# ----- TRAIN -----
model = train_classifier(
    X_train_norm,
    y_train,
    model_path="models/activity_adaboost.pkl"
)

# ----- EVALUATE -----
evaluate_model(model, X_test_norm, y_test)



print("Test set class counts:")
print(y_test.value_counts())

print("All unique labels:")
print(y.unique())

print("Train counts:\n", y_train.value_counts())
print("Test counts:\n", y_test.value_counts())

print("PIPELINE FINISHED SUCCESSFULLY")


