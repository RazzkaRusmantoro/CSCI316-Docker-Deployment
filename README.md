# CSCI316 Project 2

This project serves a Tamil-English code-switched sentiment classifier through a Flask API.

## What this API does

- Accepts text input via `POST /predict`
- Returns:
  - predicted sentiment label
  - label id
  - class probabilities
- Health check is available at `GET /health` (and basic info at `GET /`)

## 1) Local setup (Python + requirements)

1. Create and activate a virtual environment (recommended):
   - Windows PowerShell:
     - `python -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Place your fine-tuned model files inside `model/`
   - Example required files include:
     - `config.json`
     - `tokenizer_config.json`
     - `special_tokens_map.json`
     - model weights (for example `model.safetensors`)

## 2) Run with Docker (pull image, then run)

If an image is already published to GitHub Container Registry (GHCR), pull and run it:

1. Pull:
   - `docker pull ghcr.io/<github-username>/<repo-name>:latest`
2. Run
   - `docker run --rm -p 5000:5000 ghcr.io/razzkarusmantoro/csci316-docker-deployment:latest`

Then test:
- `http://localhost:5000/`
- `http://localhost:5000/health`

## 3) If pull is not available: build locally, then run

1. Build image from this repo:
   - `docker build -t csci316-sentiment:latest .`
2. Run container:
   - `docker run --rm -p 5000:5000 csci316-sentiment:latest`

## 4) Run locally without Docker

After you install requirements and place the model in `model/`, start the app directly:

- `python app.py`

The service starts on port `5000` by default.

## Example prediction request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"This movie was super, but konjam long-a irundhudhu\"}"
```

PowerShell alternative:

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:5000/predict" `
  -ContentType "application/json" `
  -Body '{"text":"This movie was super, but konjam long-a irundhudhu"}'
```