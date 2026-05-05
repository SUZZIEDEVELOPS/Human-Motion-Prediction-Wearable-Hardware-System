# src/normalize.py


from sklearn.preprocessing import StandardScaler
import joblib
import os

def normalize_features(X, scaler=None, save_path="models/scaler.pkl"):
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


