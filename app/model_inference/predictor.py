"""
Model Inference Layer — serving.

Loads the trained RandomForest classifier once (cold start) and exposes a
predict() function that the API layer calls per-request. In a real cloud
deployment this module is what would run inside the "cloud classifier"
compute unit (e.g. a SageMaker endpoint, a Cloud Run container, or a Lambda
function with the model artifact pulled from S3/GCS at cold start).
"""

import os

import joblib
import numpy as np
import pandas as pd

from app.feature_extraction.extractor import FEATURE_NAMES, extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "rf_model.joblib")

_model_bundle = None


def _load_model():
    global _model_bundle
    if _model_bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_PATH}. "
                "Run `python app/model_inference/train.py` first."
            )
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


def predict_url(url: str) -> dict:
    """Run the full feature-extraction + inference pipeline for one URL.

    Returns a dict with:
      - features: the extracted numeric feature dict
      - phishing_probability: float in [0, 1]
      - predicted_label: 0 (legitimate) or 1 (phishing)
      - top_contributing_features: list of (feature, value, importance) tuples
        used by the response layer to build a human-readable explanation.
    """
    bundle = _load_model()
    model = bundle["model"]
    feature_names = bundle.get("feature_names", FEATURE_NAMES)

    feats = extract_features(url)
    row = pd.DataFrame([[feats[name] for name in feature_names]], columns=feature_names)

    proba = model.predict_proba(row)[0]
    phishing_proba = float(proba[1])
    predicted_label = int(phishing_proba >= 0.5)

    importances = model.feature_importances_
    ranked = sorted(
        zip(feature_names, importances),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top_features = []
    for name, importance in ranked[:5]:
        top_features.append(
            {
                "feature": name,
                "value": feats[name],
                "importance": round(float(importance), 4),
            }
        )

    return {
        "features": feats,
        "phishing_probability": round(phishing_proba, 4),
        "predicted_label": predicted_label,
        "top_contributing_features": top_features,
    }


def model_info() -> dict:
    bundle = _load_model()
    model = bundle["model"]
    return {
        "model_type": type(model).__name__,
        "n_estimators": getattr(model, "n_estimators", None),
        "n_features": len(bundle.get("feature_names", FEATURE_NAMES)),
        "trained_with_sklearn": bundle.get("sklearn_version_trained_with"),
    }
