# Backend — Real-Time Sign Language Interpreter

This folder contains the **Flask API** that serves predictions from the two
trained models (the static ASL-alphabet MLP and the dynamic word-level LSTM)
to the React frontend and to the `test_api.py` client.

The backend is deliberately thin:

- All machine-learning logic lives in `src/inference` (the `StaticPredictor`
  and `DynamicPredictor` classes), wrapped by `backend/model_manager.py`.
- All landmark normalization lives in `src/common/landmarks.py`.
- This API only validates requests, forwards **raw** landmarks to the model
  manager, and shapes the JSON responses.

> **No webcam required.** The backend never opens a camera. It only receives
> already-extracted landmark numbers (from the browser via `@mediapipe/hands`,
> or from `src/inference/webcam_demo.py`). The webcam lives in the *frontend*
> and the demo script, never here.

---

## 1. Prerequisites

From the **project root** (`c:/SignInterpreter`), install both the shared ML
dependencies and the backend web dependencies:

```powershell
# Project root requirements (torch, mediapipe, opencv-python, numpy, ...)
pip install -r requirements.txt

# Backend web requirements (flask, flask-cors, flask-socketio, requests)
pip install -r backend/requirements.txt
```

The models must have been trained first (so that `models/static_mlp.pt` and/or
`models/dynamic_lstm.pt` exist). If a model file is missing, the server still
starts — that model's predict endpoint simply returns **503** until it is
trained, and `/api/health` reports it as `false`.

---

## 2. Running the server

Always run from the **project root** so that `import config` and the
`src.*` imports resolve. Either of these works:

```powershell
# Preferred: run as a module
python -m backend.app

# Or run the file directly (the script bootstraps sys.path itself)
python backend/app.py
```

The server listens on **`http://0.0.0.0:5000`** (reachable as
`http://localhost:5000`). The React dev server runs **separately** on
`http://localhost:3000`; CORS is configured to allow that origin (and `*` for
local convenience).

Enable Flask debug mode (auto-reload, verbose errors) by setting an
environment variable before launching:

```powershell
$env:FLASK_DEBUG = "1"
python -m backend.app
```

On startup the log prints which models were loaded, e.g.:

```
[INFO] backend.app: Startup complete -> device=cuda, static_model=True, dynamic_model=False
[WARNING] backend.app: Dynamic model not loaded: /api/predict/dynamic will return 503 ...
```

### Optional WebSocket (Socket.IO)

If `flask-socketio` is installed, the server also exposes a Socket.IO channel:
the client emits a `"landmarks"` event `{ landmarks: [63 floats] }` and the
server emits a `"prediction"` event with the same shape as the static HTTP
response. If `flask-socketio` is **not** installed, the import is guarded and
the server falls back to a plain Flask app — everything else still works.

---

## 3. Endpoint reference

Base URL: `http://localhost:5000`

| Method | Path                     | Body                                   | Success response                                                                 | Error codes |
|--------|--------------------------|----------------------------------------|----------------------------------------------------------------------------------|-------------|
| GET    | `/api/health`            | —                                      | `{"status":"ok","device":<str>,"models":{"static":<bool>,"dynamic":<bool>}}`     | —           |
| GET    | `/api/labels`            | —                                      | `{"static":[...class strings...],"dynamic":[...gloss strings...]}`               | —           |
| POST   | `/api/predict/static`    | `{"landmarks":[63 RAW floats]}`        | `{"prediction":<str>,"confidence":<float>,"top_k":[{"label":<str>,"confidence":<float>}, ... up to 3]}` | 400 invalid input; 503 model not loaded |
| POST   | `/api/predict/dynamic`   | `{"sequence":[[126 RAW floats], ...]}` | `{"prediction":<gloss>,"confidence":<float>,"top_k":[...]}`                       | 400 invalid input; 503 model not loaded |

Notes:

- **Static** input must be exactly **63** numbers (one hand, `[x,y,z]*21`).
- **Dynamic** input is a list of frames, each exactly **126** numbers (two
  hands). The sequence may be any non-empty length; it is padded/truncated to
  `config.SEQUENCE_LENGTH` (30) **server-side** inside the predictor.
- Landmarks are sent **raw** (un-normalized). Normalization is applied once,
  inside the predictors, so models always see normalized input.

---

## 4. Example calls

### PowerShell (`Invoke-RestMethod`)

```powershell
# Health check
Invoke-RestMethod -Uri http://localhost:5000/api/health -Method Get

# Available labels
Invoke-RestMethod -Uri http://localhost:5000/api/labels -Method Get

# Static prediction (63 raw floats; zeros shown here as a placeholder)
$body = @{ landmarks = @(0) * 63 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/api/predict/static `
    -Method Post -ContentType "application/json" -Body $body

# Dynamic prediction (a sequence of frames, each 126 raw floats)
$frame = ,(@(0) * 126)            # one frame; comma forces a nested array
$seq   = @{ sequence = $frame * 10 } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://localhost:5000/api/predict/dynamic `
    -Method Post -ContentType "application/json" -Body $seq
```

### curl

```bash
# Health check
curl http://localhost:5000/api/health

# Available labels
curl http://localhost:5000/api/labels

# Static prediction (replace the zeros with 63 real landmark values)
curl -X POST http://localhost:5000/api/predict/static \
     -H "Content-Type: application/json" \
     -d "{\"landmarks\": [0,0,0, ... 63 numbers total ...]}"

# Dynamic prediction (a list of frames, each with 126 numbers)
curl -X POST http://localhost:5000/api/predict/dynamic \
     -H "Content-Type: application/json" \
     -d "{\"sequence\": [[0, ... 126 numbers ...], [0, ...]]}"
```

A ready-made Python client lives at `src/inference/test_api.py` (it uses
`requests`, which is included in this folder's `requirements.txt`).

---

## 5. Files in this folder

| File               | Purpose                                                                 |
|--------------------|-------------------------------------------------------------------------|
| `app.py`           | Flask app: routes, validation, CORS, optional Socket.IO, entry point.   |
| `model_manager.py` | `ModelManager` wrapping the predictors; graceful handling of missing models; `ModelUnavailable`. |
| `requirements.txt` | Web/server dependencies (ML deps are in the root `requirements.txt`).   |
| `__init__.py`      | Marks `backend` as a Python package (so `python -m backend.app` works). |
| `README.md`        | This file.                                                              |
