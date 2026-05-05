# src/feature_selection.py
from sklearn.feature_selection import mutual_info_classif, SelectKBest

def select_features(X, y, k=40, random_state=0):
    """
    Selects top-k features using mutual information.
    """
    selector = SelectKBest(
        score_func=lambda X_, y_: mutual_info_classif(X_, y_, random_state=random_state),
        k=k
    )
    X_new = selector.fit_transform(X, y)
    return X_new, selector
