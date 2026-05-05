import numpy as np
from sklearn.feature_selection import mutual_info_regression, SelectKBest

def select_features(X, y, k=50, random_state=0):
    y_arr = np.asarray(y)
    y_1d  = np.linalg.norm(y_arr, axis=1) if y_arr.ndim == 2 else y_arr

    selector = SelectKBest(
        score_func=lambda X_, y_: mutual_info_regression(
            X_, y_, random_state=random_state
        ),
        k=k
    )
    X_new = selector.fit_transform(X, y_1d)
    return X_new, selector