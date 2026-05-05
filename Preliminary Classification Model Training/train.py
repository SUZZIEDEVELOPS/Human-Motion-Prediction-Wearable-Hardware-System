from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import joblib
import os

def train_classifier(X, y, model_path="models/adaboost_tree.pkl"):
    """
    Trains a tree-based ensemble classifier.
    """

    # Base learner (small decision tree)
    base_tree = DecisionTreeClassifier(
        max_leaf_nodes=20,
        random_state=0
    )

    # AdaBoost ensemble
    model = AdaBoostClassifier(
        estimator=base_tree,
        n_estimators=30,
        learning_rate=0.1,
        algorithm="SAMME",
        random_state=0
    )

    # Train
    model.fit(X, y)

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    return model
