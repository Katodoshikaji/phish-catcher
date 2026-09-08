"""
Model Inference Layer — training script.

Trains a RandomForestClassifier on the labeled feature dataset produced by
app/data/generate_dataset.py and persists the fitted model (+ metadata) to
app/model/rf_model.joblib so the inference/serving layer can load it without
retraining on every request (this is the "cloud classifier" artifact that
would normally be stored in S3 / GCS / a model registry and loaded once per
container/function cold-start).
"""

import json
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.feature_extraction.extractor import FEATURE_NAMES  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_NAMES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importances": dict(
            sorted(
                zip(FEATURE_NAMES, clf.feature_importances_.tolist()),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
    }

    joblib.dump(
        {"model": clf, "feature_names": FEATURE_NAMES, "sklearn_version_trained_with": __import__("sklearn").__version__},
        MODEL_PATH,
    )
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model trained and saved to", MODEL_PATH)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print("Confusion matrix:", metrics["confusion_matrix"])
    print("\nTop 8 feature importances:")
    for name, imp in list(metrics["feature_importances"].items())[:8]:
        print(f"  {name:28s} {imp:.4f}")


if __name__ == "__main__":
    main()
