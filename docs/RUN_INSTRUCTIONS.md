# Run Instructions (exact ordered commands)

This is the **step-by-step runbook** for the Real-Time Sign Language
Interpreter. Follow the steps **in order** — each builds on the previous one.
All commands are PowerShell and assume you are at the project root with the
virtual environment activated.

> **Before you start:** complete [`SETUP.md`](SETUP.md) (venv + dependencies)
> and [`CUDA_GUIDE.md`](CUDA_GUIDE.md) (GPU). Then:
>
> ```powershell
> Set-Location c:\SignInterpreter
> .\.venv\Scripts\Activate.ps1
> ```
>
> All `python -m src.*` scripts bootstrap `sys.path` to the project root, so
> `import config` and `from src.common.landmarks import ...` resolve as long
> as you run them **from `c:\SignInterpreter`**.

---

## Pipeline at a glance

```
1) preprocess ASL   →  data/processed/asl_landmarks.npz
2) train static     →  models/static_mlp.pt  + models/static_labels.json
3) preprocess WLASL →  data/processed/wlasl_sequences.npz
4) train dynamic    →  models/dynamic_lstm.pt + models/dynamic_labels.json
5) run backend      →  Flask API on http://localhost:5000
6) run frontend     →  React UI on http://localhost:3000
7) test API         →  src/inference/test_api.py (no webcam needed)
8) webcam demo      →  src/inference/webcam_demo.py (laptop with a camera)
```

---

## Step 1 — Preprocess the ASL alphabet (STATIC)

Extracts one-hand landmarks (63 floats) from every alphabet image using
MediaPipe and saves them to `data/processed/asl_landmarks.npz`. **No webcam
is used.**

```powershell
python -m src.preprocessing.preprocess_asl_alphabet
```

- Reads from `datasets/asl/asl_alphabet_train/asl_alphabet_train/<CLASS>/*.jpg`.
- Output: `data/processed/asl_landmarks.npz` (features + integer labels +
  class names).
- This is the longest preprocessing step (tens of thousands of images); a
  `tqdm` progress bar is shown. Tip: the script supports a per-class image cap
  if you want a faster first run — see its `--help` / top-of-file docstring.

---

## Step 2 — Train the static MLP

Trains `StaticSignMLP` (63 → 256 → 128 → 29) on the normalized landmarks.

```powershell
python -m src.training.train_static
```

- Input: `data/processed/asl_landmarks.npz`.
- Prints the active device (expect `cuda`) and per-epoch loss/accuracy.
- Output:
  - `models/static_mlp.pt` — checkpoint `{state_dict, arch, classes}`.
  - `models/static_labels.json` — JSON list of class names (index → label).

Verify the GPU is busy in another window with `nvidia-smi -l 1`
(see [`CUDA_GUIDE.md`](CUDA_GUIDE.md)).

---

## Step 3 — Preprocess the WLASL videos (DYNAMIC)

Selects the top-K glosses by video count (default **K = 20**), extracts
two-hand landmarks (126 floats) per frame, pads/truncates each clip to
`SEQUENCE_LENGTH = 30` frames, and saves sequences to
`data/processed/wlasl_sequences.npz`. **No webcam is used** — it reads the
`.mp4` files with OpenCV.

```powershell
python -m src.preprocessing.preprocess_wlasl
```

- Reads from `datasets/archive/dataset/SL/<gloss>/<video_id>.mp4`.
- Output: `data/processed/wlasl_sequences.npz` (sequences + labels + glosses).
- To use a different number of glosses, see the script's options
  (e.g. `--num-glosses 30`). More glosses + augmentation = a stronger model,
  at the cost of fewer samples per class.

---

## Step 4 — Train the dynamic LSTM

Trains `DynamicSignLSTM` (126 → LSTM(128, 2 layers) → K glosses) on the
normalized sequences.

```powershell
python -m src.training.train_dynamic
```

- Input: `data/processed/wlasl_sequences.npz`.
- Prints the active device and per-epoch metrics.
- Output:
  - `models/dynamic_lstm.pt` — checkpoint `{state_dict, arch, classes}`.
  - `models/dynamic_labels.json` — JSON list of glosses (index → label).

---

## Step 5 — Run the Flask backend

Loads both trained models and serves the prediction API on
`http://localhost:5000` (CORS allows `http://localhost:3000`).

```powershell
# run as a module from the project root (the backend lives in backend/, not src/)
python -m backend.app
# equivalent: python backend/app.py
```

Endpoints:

| Method & Path | Purpose |
|---------------|---------|
| `GET  /api/health` | `{"status":"ok","device":...,"models":{"static":bool,"dynamic":bool}}` |
| `GET  /api/labels` | `{"static":[...], "dynamic":[...]}` |
| `POST /api/predict/static`  | body `{"landmarks":[63 RAW floats]}` → prediction + top-k |
| `POST /api/predict/dynamic` | body `{"sequence":[[126 RAW floats], ...]}` → prediction + top-k |

If `flask-socketio` is installed, a real-time `landmarks` → `prediction`
Socket.IO channel is also enabled (the app runs without it too). Leave this
window running.

---

## Step 6 — Run the React frontend

In a **new** PowerShell window (keep the backend running):

```powershell
Set-Location c:\SignInterpreter\frontend
npm install   # first time only
npm start
```

- Opens `http://localhost:3000` in your browser.
- Grant **camera permission** when prompted.
- The app captures hand landmarks with `@mediapipe/hands`, sends **RAW**
  landmarks to the backend, displays the predicted letter/word, and **speaks**
  it via the Web Speech API.

---

## Step 7 — Test the backend WITHOUT a webcam

The GPU desktop has no camera. This script exercises every API endpoint using
synthetic / file-derived landmarks, so you can validate the backend before
touching a webcam.

```powershell
# with the backend (Step 5) running:
python src/inference/test_api.py
```

- Calls `/api/health`, `/api/labels`, `/api/predict/static`, and
  `/api/predict/dynamic`, printing each response and a pass/fail summary.

---

## Step 8 — Run the OpenCV webcam demo (laptop)

Run this **on a machine with a webcam** (e.g. a laptop). It opens an OpenCV
window, tracks your hand(s), and overlays live predictions. This is the only
Python file that uses a webcam.

```powershell
python src/inference/webcam_demo.py
```

- Press the key shown in the on-screen help to toggle between **static**
  (alphabet) and **dynamic** (word) modes; press `q` / `Esc` to quit.
- It can run fully local (loading the `.pt` models directly) — no backend
  required.

---

## End-to-end summary (copy/paste)

```powershell
Set-Location c:\SignInterpreter
.\.venv\Scripts\Activate.ps1

python -m src.preprocessing.preprocess_asl_alphabet   # 1
python -m src.training.train_static                   # 2
python -m src.preprocessing.preprocess_wlasl          # 3
python -m src.training.train_dynamic                  # 4
python src/backend/app.py                             # 5 (leave running)

# new window:
Set-Location c:\SignInterpreter\frontend ; npm install ; npm start   # 6

# another window (backend running):
python src/inference/test_api.py                      # 7

# on a laptop with a camera:
python src/inference/webcam_demo.py                   # 8
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: config` | Run from `c:\SignInterpreter` (not from inside `src/`). |
| `FileNotFoundError: asl_landmarks.npz` | Run Step 1 before Step 2. |
| `FileNotFoundError: wlasl_sequences.npz` | Run Step 3 before Step 4. |
| Backend returns `503` on predict | The corresponding model isn't trained yet — complete Steps 2 / 4. |
| Frontend can't reach backend | Ensure Step 5 is running and CORS allows `http://localhost:3000`. |
| Training runs on CPU | See [`CUDA_GUIDE.md`](CUDA_GUIDE.md) to install the CUDA PyTorch build. |
| Webcam demo: "no camera" | Run it on a device with a webcam (the GPU desktop has none). |
