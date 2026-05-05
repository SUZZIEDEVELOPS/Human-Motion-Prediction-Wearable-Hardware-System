from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_regressor(X, y, model_path="models/trajectory_rf.pkl"):
    model = RandomForestRegressor(
        n_estimators=100,       # more trees = more stable
        max_depth=None,         # full trees
        min_samples_leaf=3,     # more flexible
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved → {model_path}")
    return model