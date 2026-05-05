from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import os


def train_classifier(X, y, model_path="models/activity_adaboost_ada.pkl"):
    """Trains an AdaBoost classifier and saves it to disk."""

    # Base learner (small decision tree)
    base_tree = DecisionTreeClassifier(
        max_leaf_nodes=20,
        random_state=0
    )

    # AdaBoost ensemble
    model = AdaBoostClassifier(
        estimator=base_tree,
        n_estimators=40,
        learning_rate=0.1,
        algorithm="SAMME",
        random_state=0
    )
    model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[SAVED] AdaBoost → {model_path}")

    return model
