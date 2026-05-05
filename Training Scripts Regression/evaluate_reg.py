import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


TARGET_KEYS = [
    "LEFTARM_AccY_mean",
    "LEFTARM_AccZ_mean",
    "RIGHTLEG_AccY_mean",
    "RIGHTLEG_AccZ_mean",
    "LEFTARM_AccX_dominant_freq",
    "LEFTARM_AccY_dominant_freq",
    "LEFTARM_AccZ_dominant_freq",
    "LEFTARM_AccMag_dominant_freq",
    "RIGHTLEG_AccX_dominant_freq",
    "RIGHTLEG_AccY_dominant_freq",
    "RIGHTLEG_AccZ_dominant_freq",
    "RIGHTLEG_AccMag_dominant_freq",
]

SHORT_NAMES = {
    "LEFTARM_AccY_mean":            "LA ay mean",
    "LEFTARM_AccZ_mean":            "LA az mean",
    "RIGHTLEG_AccY_mean":           "RL ay mean",
    "RIGHTLEG_AccZ_mean":           "RL az mean",
    "LEFTARM_AccX_dominant_freq":   "LA ax freq",
    "LEFTARM_AccY_dominant_freq":   "LA ay freq",
    "LEFTARM_AccZ_dominant_freq":   "LA az freq",
    "LEFTARM_AccMag_dominant_freq": "LA amag freq",
    "RIGHTLEG_AccX_dominant_freq":  "RL ax freq",
    "RIGHTLEG_AccY_dominant_freq":  "RL ay freq",
    "RIGHTLEG_AccZ_dominant_freq":  "RL az freq",
    "RIGHTLEG_AccMag_dominant_freq":"RL amag freq",
}


def evaluate_model(model, X, y, activities=None):
    y      = np.asarray(y)
    y_pred = model.predict(X)

    mask = np.all(np.isfinite(y), axis=1) & np.all(np.isfinite(y_pred), axis=1)
    y, y_pred = y[mask], y_pred[mask]
    if activities is not None:
        activities = np.asarray(activities)[mask]

    errors     = np.linalg.norm(y_pred - y, axis=1)
    ade        = float(np.mean(errors))
    rmse       = float(np.sqrt(np.mean((y_pred - y) ** 2)))
    p50        = float(np.median(errors))
    p90        = float(np.percentile(errors, 90))
    r2_outputs = r2_score(y, y_pred, multioutput='raw_values')
    r2_overall = float(r2_score(y, y_pred))

    print("\n" + "=" * 55)
    print("  OVERALL REGRESSION PERFORMANCE")
    print("=" * 55)
    print(f"  ADE   (mean Euclidean error)  : {ade:.4f}")
    print(f"  RMSE  (across all outputs)    : {rmse:.4f}")
    print(f"  R²    (overall)               : {r2_overall:.4f}")
    print(f"  Median error                  : {p50:.4f}")
    print(f"  90th percentile               : {p90:.4f}")

    print("\n  Per-output R²:")
    for name, r2 in zip(TARGET_KEYS, r2_outputs):
        bar  = "█" * max(0, int(r2 * 25))
        flag = " ✓" if r2 > 0.6 else (" ~" if r2 > 0.3 else " ✗")
        print(f"    {SHORT_NAMES.get(name, name):<14s}  "
              f"R²={r2:+.3f}  {bar}{flag}")

    act_metrics = {}
    if activities is not None:
        acts = np.unique(activities)
        print(f"\n  Per-activity breakdown  (n={len(errors)} test samples):")
        print(f"  {'Activity':<25s} {'N':>4} {'ADE':>7} "
              f"{'RMSE':>7} {'R²':>7} {'P90':>7}")
        print("  " + "-" * 58)
        for act in sorted(acts):
            m = activities == act
            n = int(m.sum())
            if n < 5:
                continue
            e_a   = errors[m]
            r2_a  = float(np.clip(r2_score(y[m], y_pred[m]), -1.0, 1.0))
            ade_a = float(np.mean(e_a))
            rms_a = float(np.sqrt(np.mean((y_pred[m] - y[m]) ** 2)))
            p90_a = float(np.percentile(e_a, 90))
            print(f"  {act:<25s} {n:>4} {ade_a:>7.4f} "
                  f"{rms_a:>7.4f} {r2_a:>7.4f} {p90_a:>7.4f}")
            act_metrics[act] = {
                'n': n, 'ade': ade_a, 'rmse': rms_a,
                'r2': r2_a, 'p90': p90_a
            }

    # ── Figure ────────────────────────────────────────────────────────────────
    # 1 hist + 12 scatter + 3 activity panels = 16 slots → (3, 6) = 18 slots
    fig = plt.figure(figsize=(22, 12))
    fig.suptitle('Next-window IMU motion prediction — full evaluation',
                 fontsize=13, fontweight='bold')

    # Panel 1: error distribution
    ax0 = fig.add_subplot(3, 6, 1)
    ax0.hist(errors, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    ax0.axvline(ade, color='red',    linestyle='--', lw=1.5,
                label=f'ADE={ade:.3f}')
    ax0.axvline(p50, color='orange', linestyle='--', lw=1.5,
                label=f'Med={p50:.3f}')
    ax0.set_title('Error distribution', fontsize=9)
    ax0.set_xlabel('Euclidean error')
    ax0.set_ylabel('Count')
    ax0.legend(fontsize=7)

    # Panels 2-13: predicted vs actual per output
    colours = plt.cm.tab20(np.linspace(0, 1, len(TARGET_KEYS)))
    for i, (name, col) in enumerate(zip(TARGET_KEYS, colours)):
        ax = fig.add_subplot(3, 6, i + 2)
        lim = max(np.abs(y[:, i]).max(), np.abs(y_pred[:, i]).max()) * 1.1
        ax.scatter(y[:, i], y_pred[:, i], alpha=0.25, s=6, color=col)
        ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
        ax.set_title(
            f'{SHORT_NAMES.get(name, name)}\nR²={r2_outputs[i]:.3f}',
            fontsize=8)
        ax.set_xlabel('Actual', fontsize=7)
        ax.set_ylabel('Predicted', fontsize=7)
        ax.tick_params(labelsize=7)

    # Panels 16-18: per-activity plots
    if activities is not None and len(act_metrics) > 1:
        acts_s   = sorted(act_metrics, key=lambda a: act_metrics[a]['ade'])
        ade_vals = [act_metrics[a]['ade'] for a in acts_s]
        r2_vals  = [act_metrics[a]['r2']  for a in acts_s]
        n_vals   = [act_metrics[a]['n']   for a in acts_s]
        cmap     = plt.cm.tab10(np.linspace(0, 0.9, len(acts_s)))

        ax16 = fig.add_subplot(3, 6, 16)
        ax16.barh(acts_s, ade_vals, color=cmap, edgecolor='white')
        ax16.axvline(ade, color='red', linestyle='--', lw=1,
                     label=f'Overall={ade:.3f}')
        ax16.set_xlabel('ADE')
        ax16.set_title('ADE by activity', fontsize=9)
        ax16.legend(fontsize=7)

        ax17 = fig.add_subplot(3, 6, 17)
        box_data = [errors[activities == a] for a in acts_s]
        bp = ax17.boxplot(box_data, vert=False, patch_artist=True,
                          medianprops=dict(color='red', lw=1.5))
        for patch, c in zip(bp['boxes'], cmap):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax17.set_yticks(range(1, len(acts_s) + 1))
        ax17.set_yticklabels(acts_s, fontsize=7)
        ax17.set_xlabel('Euclidean error')
        ax17.set_title('Error spread by activity', fontsize=9)

        ax18 = fig.add_subplot(3, 6, 18)
        ax18.barh(acts_s, r2_vals, color=cmap, edgecolor='white')
        ax18.axvline(0,          color='black', lw=0.5)
        ax18.axvline(r2_overall, color='red', linestyle='--', lw=1,
                     label=f'Overall={r2_overall:.3f}')
        ax18.set_xlabel('R²')
        ax18.set_title('R² by activity', fontsize=9)
        ax18.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('models/regression_evaluation.png', dpi=100,
                bbox_inches='tight')
    plt.close()
    print("\nPlot saved → models/regression_evaluation.png")

    return {
        'ade': ade, 'rmse': rmse, 'r2': r2_overall,
        'p50': p50, 'p90': p90,
        'r2_per_output': dict(zip(TARGET_KEYS, r2_outputs.tolist())),
        'per_activity':  act_metrics
    }