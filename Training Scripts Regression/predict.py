"""
predict.py — visualise actual predictions vs ground truth.
Called automatically at end of main_prelim_regression.py.
Standalone: python src/predict.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from src.load_data import process_file
from preprocess import preprocess
from src.build_targets import TARGET_KEYS, features_to_target
from src.feature_engineering import extract_features
from utils.windowing import sliding_window

SHORT = {
    'X_Acceleration_mean': 'ax mean',
    'Y_Acceleration_mean': 'ay mean',
    'Z_Acceleration_mean': 'az mean',
    'X_Acceleration_std':  'ax std',
    'Y_Acceleration_std':  'ay std',
    'Z_Acceleration_std':  'az std',
}

ACCEL_COL_SETS = [
    ('X_Acceleration', 'Y_Acceleration', 'Z_Acceleration'),
    ('x-axis', 'y-axis', 'z-axis'),
    ('ax', 'ay', 'az'),
    ('X', 'Y', 'Z'),
    ('linear acceleration x (m/s^2)',
     'linear acceleration y (m/s^2)',
     'linear acceleration z (m/s^2)'),
]

def find_accel_cols(df):
    lower_map = {c.lower(): c for c in df.columns}
    for col_set in ACCEL_COL_SETS:
        if all(c.lower() in lower_map for c in col_set):
            return tuple(lower_map[c.lower()] for c in col_set)
    accel_cols = [c for c in df.columns if 'accel' in c.lower()]
    return tuple(accel_cols[:3]) if len(accel_cols) >= 3 else None

def remap_accel_cols(df, found_cols):
    standard = ('X_Acceleration', 'Y_Acceleration', 'Z_Acceleration')
    rename   = {found_cols[i]: standard[i] for i in range(3)
                if found_cols[i] != standard[i]}
    return df.rename(columns=rename) if rename else df


def run_prediction_demo(
    raw_folder      = "data_raw",
    model_path      = "models/trajectory_rf.pkl",
    scaler_path     = "models/scaler.pkl",
    window_secs     = 5.0,
    overlap         = 0.75,
    n_examples      = 6,
    activity_filter = None
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model at {model_path} — run main_prelim_regression.py first."
        )

    model               = joblib.load(model_path)
    scaler              = joblib.load(scaler_path)
    n_features_expected = scaler.n_features_in_

    # ── Collect window pairs ──
    all_pairs = []

    for file in sorted(os.listdir(raw_folder)):
        if not file.endswith(".csv"):
            continue

        path = os.path.join(raw_folder, file)
        df, activity, mount, sensors = process_file(path)

        if activity_filter and activity != activity_filter:
            continue

        accel_cols = find_accel_cols(df)
        if accel_cols is None:
            continue

        df       = remap_accel_cols(df, accel_cols)
        df_clean, fs = preprocess(df)
        if fs is None:
            continue

        imu_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns
                    if c not in ['latitude','longitude','altitude',
                                 'speed','course','hacc']]

        window_size = int(fs * window_secs)
        step_size   = int(window_size * (1 - overlap))
        windows     = list(sliding_window(df_clean, window_size, step_size))

        if len(windows) < 2:
            continue

        file_feats = [extract_features(w, imu_cols, fs) for w in windows]

        for i in range(len(file_feats) - 1):
            target = features_to_target(file_feats[i + 1])
            if target is None:
                continue
            all_pairs.append((
                file_feats[i],
                target,
                windows[i + 1],
                activity,
                fs
            ))

    if not all_pairs:
        print("No pairs found for prediction demo.")
        return

    print(f"\nTotal window pairs for prediction: {len(all_pairs)}")

    # ── Pick n_examples evenly spaced across dataset ──
    indices  = np.linspace(0, len(all_pairs) - 1, n_examples, dtype=int)
    examples = [all_pairs[i] for i in indices]

    # ── Build feature matrix aligned to scaler ──
    rows   = [pd.Series(feats).fillna(0) for feats, *_ in examples]
    X_demo = pd.DataFrame(rows).fillna(0)
    X_demo = X_demo.replace([np.inf, -np.inf], 0).fillna(0)

    if X_demo.shape[1] > n_features_expected:
        X_demo = X_demo.iloc[:, :n_features_expected]
    elif X_demo.shape[1] < n_features_expected:
        for pad in range(n_features_expected - X_demo.shape[1]):
            X_demo[f'_pad_{pad}'] = 0.0

    X_scaled = scaler.transform(X_demo)
    y_preds  = model.predict(X_scaled)          # (n_examples, n_targets)

    labels = [SHORT.get(k, k) for k in TARGET_KEYS]
    x_pos  = np.arange(len(TARGET_KEYS))
    width  = 0.35
    colours_bar = ['#4878CF','#E87838','#3CB371',
                   '#CC3333','#9B59B6','#8B4513']

    # ══════════════════════════════════════════════════
    # FIGURE 1 — bar chart: predicted vs actual per example
    # ══════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(n_examples, 1,
                               figsize=(14, 3.5 * n_examples))
    if n_examples == 1:
        axes1 = [axes1]
    fig1.suptitle('Predicted vs actual — next-window IMU statistics',
                  fontsize=12, fontweight='bold')

    for idx, (ax, (feats, target, w_next, activity, fs), y_pred) in \
            enumerate(zip(axes1, examples, y_preds)):

        error = float(np.linalg.norm(y_pred - target))

        bars_act  = ax.bar(x_pos - width/2, target, width,
                           label='Actual',    color='steelblue',
                           alpha=0.85, edgecolor='white')
        bars_pred = ax.bar(x_pos + width/2, y_pred, width,
                           label='Predicted', color='orange',
                           alpha=0.85, edgecolor='white')

        # Value labels on bars
        for bar in bars_act:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    h + (0.02 * abs(h) if h >= 0 else -0.08 * abs(h)),
                    f'{h:.2f}', ha='center', va='bottom', fontsize=6.5)
        for bar in bars_pred:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    h + (0.02 * abs(h) if h >= 0 else -0.08 * abs(h)),
                    f'{h:.2f}', ha='center', va='bottom', fontsize=6.5,
                    color='darkorange')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
        ax.set_ylabel('Value (m/s²)', fontsize=8)
        ax.set_title(
            f'Example {idx+1}  |  activity: {activity}  |  '
            f'Euclidean error: {error:.4f}',
            fontsize=9
        )
        ax.legend(fontsize=8, loc='upper right')
        ax.axhline(0, color='black', lw=0.5)

    plt.tight_layout()
    plt.savefig('models/prediction_examples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved → models/prediction_examples.png")

    # ══════════════════════════════════════════════════
    # FIGURE 2 — raw signal + predicted vs actual mean
    # accel overlay (first 3 examples)
    # ══════════════════════════════════════════════════
    n_sig    = min(3, n_examples)
    fig2, axes2 = plt.subplots(n_sig, 1, figsize=(14, 4.5 * n_sig))
    if n_sig == 1:
        axes2 = [axes2]
    fig2.suptitle(
        'Current window signal — predicted vs actual next-window mean acceleration',
        fontsize=11, fontweight='bold'
    )

    sig_cols    = ['X_Acceleration', 'Y_Acceleration', 'Z_Acceleration']
    sig_colours = ['#4878CF', '#E87838', '#3CB371']

    for idx, (ax, (feats, target, w_next, activity, fs), y_pred) in \
            enumerate(zip(axes2, examples[:n_sig], y_preds[:n_sig])):

        t_axis = np.arange(len(w_next)) / fs

        # Raw signal of current window
        for col, col_colour in zip(sig_cols, sig_colours):
            if col not in w_next.columns:
                continue
            ax.plot(t_axis, w_next[col].to_numpy(),
                    color=col_colour, alpha=0.35, lw=0.9,
                    label=col.replace('_Acceleration', '') + ' (current window)')

        # Actual next-window means — dashed lines
        ax.axhline(target[0], color=sig_colours[0], linestyle='--', lw=2,
                   label=f'actual ax_mean = {target[0]:.3f}')
        ax.axhline(target[1], color=sig_colours[1], linestyle='--', lw=2,
                   label=f'actual ay_mean = {target[1]:.3f}')
        ax.axhline(target[2], color=sig_colours[2], linestyle='--', lw=2,
                   label=f'actual az_mean = {target[2]:.3f}')

        # Predicted next-window means — dotted lines
        ax.axhline(y_pred[0], color=sig_colours[0], linestyle=':', lw=2.5,
                   label=f'pred  ax_mean = {y_pred[0]:.3f}')
        ax.axhline(y_pred[1], color=sig_colours[1], linestyle=':', lw=2.5,
                   label=f'pred  ay_mean = {y_pred[1]:.3f}')
        ax.axhline(y_pred[2], color=sig_colours[2], linestyle=':', lw=2.5,
                   label=f'pred  az_mean = {y_pred[2]:.3f}')

        ax.set_xlabel('Time within current window (s)', fontsize=8)
        ax.set_ylabel('Acceleration (m/s²)', fontsize=8)
        ax.set_title(f'Example {idx+1}  |  activity: {activity}', fontsize=9)
        ax.legend(fontsize=7, ncol=2, loc='upper right')

    plt.tight_layout()
    plt.savefig('models/prediction_signal_overlay.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved → models/prediction_signal_overlay.png")

    # ══════════════════════════════════════════════════
    # FIGURE 3 — per-target error breakdown across examples
    # ══════════════════════════════════════════════════
    targets_arr = np.array([ex[1] for ex in examples])
    errors_per_target = np.abs(y_preds - targets_arr)   # (n_examples, n_targets)

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    act_labels = [f"Ex{i+1}\n{ex[3]}" for i, ex in enumerate(examples)]
    x3 = np.arange(n_examples)
    bar_w = 0.13

    for ti, (tkey, col) in enumerate(zip(TARGET_KEYS, colours_bar)):
        offset = (ti - len(TARGET_KEYS)/2 + 0.5) * bar_w
        ax3.bar(x3 + offset, errors_per_target[:, ti], bar_w,
                label=SHORT.get(tkey, tkey), color=col, alpha=0.8,
                edgecolor='white')

    ax3.set_xticks(x3)
    ax3.set_xticklabels(act_labels, fontsize=8)
    ax3.set_ylabel('Absolute error per target', fontsize=9)
    ax3.set_title('Per-target prediction error by example', fontsize=10)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.axhline(0, color='black', lw=0.5)

    plt.tight_layout()
    plt.savefig('models/prediction_per_target_error.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved → models/prediction_per_target_error.png")

    # ── Terminal summary table ──
    n_t = len(TARGET_KEYS)
    print(f"\n{'Ex':<4} {'Activity':<22} "
          f"{'ax actual':>10} {'ax pred':>9} "
          f"{'ay actual':>10} {'ay pred':>9} "
          f"{'az actual':>10} {'az pred':>9} "
          f"{'Error':>8}")
    print("-" * 100)
    for idx, ((feats, target, w_next, activity, fs), y_pred) in \
            enumerate(zip(examples, y_preds)):
        err = float(np.linalg.norm(y_pred - target))
        print(f"{idx+1:<4} {activity:<22} "
              f"{target[0]:>10.3f} {y_pred[0]:>9.3f} "
              f"{target[1]:>10.3f} {y_pred[1]:>9.3f} "
              f"{target[2]:>10.3f} {y_pred[2]:>9.3f} "
              f"{err:>8.4f}")


if __name__ == "__main__":
    run_prediction_demo()