FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Model is trained at image-build time so the container is self-contained
# and cold-starts fast (no training on request path). To retrain with fresh
# data, re-run this build step or run train.py against a mounted volume.
RUN python app/data/generate_dataset.py && python app/model_inference/train.py

ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "app.main:app"]
