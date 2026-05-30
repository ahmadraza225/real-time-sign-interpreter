# Real-Time Sign Language Interpreter

**Tagline:** _Breaking the communication barrier, one gesture at a time._

A production-oriented full-stack project for real-time ASL (A–Z) static gesture interpretation using webcam frames, MediaPipe landmark extraction, and two baseline ML models (RandomForest + Keras MLP).

## Repository Structure

- `/frontend` – React + TypeScript + React Router + Tailwind CSS UI
- `/backend` – FastAPI service, model inference, training/evaluation scripts
- `/notebooks` – EDA and training walkthrough notebook(s)
- `/report` – report scaffold and generated assets
- `/demo` – demo instructions and placeholder media locations

## Features

- Modern responsive frontend with pages:
  - Landing
  - Live Interpreter
  - Results
  - Analytics
  - About
- Accessibility support:
  - keyboard-friendly controls
  - ARIA labels on interactive controls
  - high-contrast mode
- Live interpreter capabilities:
  - webcam capture
  - periodic frame upload to backend `/predict`
  - sign + confidence display
  - detection history
  - speech output via Web Speech API
- Backend APIs:
  - `GET /health`
  - `POST /predict`
  - `GET /history`
  - `GET /metrics`
- AI/ML pipeline:
  - MediaPipe hand landmark extraction (21 points)
  - normalized wrist-relative features
  - Model A: RandomForest
  - Model B: Keras MLP
  - metrics generation and confusion matrix assets
- Phase 2 dynamic gesture scaffold (LSTM-oriented placeholder)

## Frontend Setup

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Build:

```bash
npm run build
```

## Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Reference

### `GET /health`
Response:
```json
{ "status": "ok" }
```

### `POST /predict`
- Multipart form field: `frame` (image file)

Response example:
```json
{
  "predicted_sign": "A",
  "confidence": 0.92,
  "model_used": "random_forest",
  "landmarks": [{"x":0.1,"y":0.2,"z":-0.03}],
  "bounding_box": {"x_min":0.1,"y_min":0.2,"x_max":0.7,"y_max":0.9},
  "timestamp": "2026-05-30T00:00:00+00:00"
}
```

If no trained model exists, API returns a clear stub message with `model_used: "stub"`.

### `GET /history`
Returns recent in-memory detections.

### `GET /metrics`
Returns model comparison metrics and confusion matrix data.

## Dataset Placement

Place ASL alphabet dataset in folder format:

```text
backend/data/asl_alphabet_train/
  A/
    img1.jpg
  B/
    img1.jpg
  ...
  Z/
    img1.jpg
```

## Training and Evaluation

Train RF + NN models:

```bash
cd backend
python scripts/train_models.py --dataset data/asl_alphabet_train
```

Evaluate and generate report assets:

```bash
python scripts/evaluate_models.py --dataset data/asl_alphabet_train
```

Generate sample metrics payload for frontend demo (no training):

```bash
python scripts/generate_sample_metrics.py
```

Artifacts are stored in `backend/models/v1/` with version marker.
Metrics and figures are written to `report/assets/`.

## Phase 2 Dynamic Gesture Scaffold

See `backend/scripts/dynamic_gesture_scaffold.py` for sequence dataset format and implementation roadmap for LSTM-based dynamic gesture recognition.

## Report and Demo

- Report scaffold: `report/README.md`
- Demo placeholders: `demo/README.md`

## Deployment Notes

- Frontend can be deployed on Vercel/Netlify.
- Backend can be deployed with Uvicorn/Gunicorn on Render/Railway/Fly.io.
- Configure frontend `VITE_API_BASE_URL` to point at deployed backend.

## Team

Add team names/roles in this section.

## Screenshots / Video Placeholders

- `demo/screenshots/landing.png`
- `demo/screenshots/live-interpreter.png`
- `demo/video/demo.mp4`
