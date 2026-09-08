# Phishing Website Classification via URL Features (Cloud Random Forest)

A 4-layer cloud architecture for classifying URLs as phishing or legitimate.

## Layers
1. **Client Layer** (`app/client/`, any browser/mobile app/API caller) — sends a URL to the API.
2. **Feature Extraction Layer** (`app/feature_extraction/extractor.py`) — converts a URL into 25 numeric lexical/structural features (no live network calls; fast & stateless).
3. **Model Inference Layer** (`app/model_inference/`) — a RandomForestClassifier ("cloud classifier") scores the feature vector.
4. **Response Layer** (`app/response/formatter.py`) — returns `score`, `label`, `risk_level`, and an optional `explanation`.

## Quickstart
```bash
pip install -r requirements.txt
python app/data/generate_dataset.py     # build labeled dataset
python app/model_inference/train.py     # train + save the RF model
python app/main.py                      # run API on :8080
```
Open http://localhost:8080 for the demo web client, or call the API directly:
```bash
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://paypal-secure-login.xyz/verify"}'
```

## Docker
```bash
docker build -t phishing-classifier .
docker run -p 8080:8080 phishing-classifier
```
The same image can be deployed unchanged to AWS App Runner/ECS, GCP Cloud Run, or Azure Container Apps.

## Tests
```bash
python -m pytest tests/ -v
```

## API
- `GET /health` — liveness probe
- `GET /model-info` — metadata about the loaded classifier
- `POST /api/v1/predict` — `{"url": "..."}` -> classification response
- `POST /api/v1/predict/batch` — `{"urls": ["...", "..."]}` -> list of responses

## Dataset note
This environment has no reliable access to public dataset mirrors, so `app/data/generate_dataset.py` programmatically generates a labeled dataset (1500 legitimate + 1500 phishing URLs) using the same lexical patterns documented in the UCI "Phishing Websites" feature literature (suspicious TLDs, IP-hosted URLs, brand impersonation in subdomains, `@` redirection tricks, keyword stuffing, URL shorteners, etc). Swap in a real dataset (e.g. PhishTank + Tranco top domains) by replacing `app/data/dataset.csv` with the same column schema and re-running `train.py`.
