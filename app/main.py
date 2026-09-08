"""
Client / API Layer
-------------------
This module is the entry point that clients (browsers, mobile apps, or any
service issuing an API request containing a URL) talk to. It is a thin
Flask REST API — deployable as-is inside a Docker container to any cloud
(AWS ECS/App Runner, GCP Cloud Run, Azure Container Apps), or adapted to a
serverless function (AWS Lambda via a WSGI adapter, GCP Cloud Functions,
Azure Functions) with minimal changes, since all the real logic lives in
the feature_extraction / model_inference / response layers below it.

Endpoints
---------
GET  /                      -> serves the demo web client (browser UI)
GET  /health                -> liveness/readiness probe
GET  /model-info            -> metadata about the loaded cloud classifier
POST /api/v1/predict        -> { "url": "..." } -> classification response
POST /api/v1/predict/batch  -> { "urls": ["...", "..."] } -> list of responses
"""

import os

from flask import Flask, jsonify, request, send_from_directory

from app.model_inference.predictor import predict_url, model_info
from app.response.formatter import build_response

CLIENT_DIR = os.path.join(os.path.dirname(__file__), "client")

app = Flask(__name__, static_folder=CLIENT_DIR, static_url_path="")


def _classify(url: str, include_explanation: bool = True) -> dict:
    inference_result = predict_url(url)
    return build_response(url, inference_result, include_explanation=include_explanation)


@app.get("/")
def index():
    return send_from_directory(CLIENT_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/model-info")
def get_model_info():
    try:
        return jsonify(model_info())
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503


@app.post("/api/v1/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    url = payload.get("url", "").strip()
    include_explanation = payload.get("explain", True)

    if not url:
        return jsonify({"error": "Field 'url' is required."}), 400
    if len(url) > 2048:
        return jsonify({"error": "URL exceeds maximum length of 2048 characters."}), 400

    try:
        result = _classify(url, include_explanation=include_explanation)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Failed to classify URL: {e}"}), 500

    return jsonify(result)


@app.post("/api/v1/predict/batch")
def predict_batch():
    payload = request.get_json(silent=True) or {}
    urls = payload.get("urls", [])

    if not isinstance(urls, list) or not urls:
        return jsonify({"error": "Field 'urls' must be a non-empty list."}), 400
    if len(urls) > 100:
        return jsonify({"error": "Batch size limited to 100 URLs per request."}), 400

    results = []
    for url in urls:
        url = (url or "").strip()
        if not url:
            continue
        try:
            results.append(_classify(url))
        except Exception as e:  # noqa: BLE001
            results.append({"url": url, "error": str(e)})

    return jsonify({"count": len(results), "results": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("DEBUG", "false").lower() == "true")
