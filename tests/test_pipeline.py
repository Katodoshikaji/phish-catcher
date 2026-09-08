"""
Basic smoke tests covering all four layers end-to-end.
Run with: python -m pytest tests/ -v   (or: python tests/test_pipeline.py)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.feature_extraction.extractor import extract_features, FEATURE_NAMES
from app.model_inference.predictor import predict_url
from app.response.formatter import build_response


def test_feature_extraction_shape():
    feats = extract_features("https://www.google.com/search?q=test")
    assert set(feats.keys()) == set(FEATURE_NAMES)
    assert feats["has_https"] == 1
    assert feats["has_ip_address"] == 0


def test_feature_extraction_ip_and_suspicious():
    feats = extract_features("http://192.168.10.5/verify-account/login")
    assert feats["has_ip_address"] == 1
    assert feats["has_suspicious_words"] == 1
    assert feats["has_https"] == 0


def test_legitimate_url_scores_low():
    result = predict_url("https://www.google.com/search?q=machine+learning")
    assert result["phishing_probability"] < 0.5
    resp = build_response("https://www.google.com/search?q=machine+learning", result)
    assert resp["label"] == "legitimate"
    assert resp["risk_level"] in ("low", "medium")


def test_phishing_url_scores_high():
    url = "http://paypal-secure-login-update.verify-account.xyz/webscr"
    result = predict_url(url)
    assert result["phishing_probability"] > 0.5
    resp = build_response(url, result)
    assert resp["label"] == "phishing"
    assert resp["risk_level"] in ("medium", "high")
    assert isinstance(resp["explanation"], list)


def test_response_schema():
    result = predict_url("http://example.com")
    resp = build_response("http://example.com", result)
    for key in ("url", "score", "label", "risk_level"):
        assert key in resp


if __name__ == "__main__":
    test_feature_extraction_shape()
    test_feature_extraction_ip_and_suspicious()
    test_legitimate_url_scores_low()
    test_phishing_url_scores_high()
    test_response_schema()
    print("All smoke tests passed.")
