from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay)

import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    print("\n=== MODEL PERFORMANCE ===")
    print("Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred, zero_division=0))



    # ---- Confusion Matrix ----
    labels = np.unique(y)
    # cm = confusion_matrix(y, y_pred, labels=labels)
    cm = confusion_matrix(y, y_pred, labels=labels, normalize="true")

    # 🔹 Make figure bigger
    fig, ax = plt.subplots(figsize=(10, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(
        cmap="Blues",
        ax=ax,
        xticks_rotation=45
    )

    # 🔹 Align labels properly
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.show()

    # ---- Confusion Matrix ----
    labels = np.unique(y)
    cm = confusion_matrix(y, y_pred, labels=labels, normalize="true")

    fig, ax = plt.subplots(figsize=(14, 9))  # wider

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, colorbar=True)

    # IMPORTANT: set ticks AFTER disp.plot()
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_title("Confusion Matrix")

    # give labels more room (tight_layout often isn't enough here)
    fig.subplots_adjust(bottom=0.28, left=0.22)

    plt.show()

