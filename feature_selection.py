# src/feature_selection.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif


class RFSelector:
    def __init__(self, top_k_idx, n_features):
        self.top_k_idx_  = top_k_idx
        self.n_features_ = n_features

    def get_support(self):
        mask = np.zeros(self.n_features_, dtype=bool)
        mask[self.top_k_idx_] = True
        return mask


def select_features(X, y, k=150, random_state=0):
    """
    Selects top-k features by combining Random Forest importance
    and Mutual Information scores (averaged rank).
    """
    # ── Random Forest importance ──────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X, y)
    rf_importances = rf.feature_importances_

    # ── Mutual Information ────────────────────────────────────────────────────
    mi_scores = mutual_info_classif(X, y, random_state=random_state)

    # ── Combine by averaging rank (lower rank = more important) ───────────────
    n = X.shape[1]
    rf_ranks = np.argsort(np.argsort(-rf_importances))  # rank 0 = best
    mi_ranks = np.argsort(np.argsort(-mi_scores))        # rank 0 = best
    combined_ranks = rf_ranks + mi_ranks

    # Top-k by combined rank, sorted to keep original column order
    top_k_idx = np.sort(np.argsort(combined_ranks)[:k])

    selector = RFSelector(top_k_idx, n_features=n)
    X_new = X.iloc[:, top_k_idx] if hasattr(X, 'iloc') else X[:, top_k_idx]

    return X_new, selector