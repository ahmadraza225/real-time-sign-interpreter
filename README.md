# Real-Time Sign Language Interpreter

> A university-level AI system that recognizes American Sign Language (ASL) in
> real time from a webcam, translating both the **static fingerspelling
> alphabet** and **dynamic word-level signs (glosses)** into on-screen text
> and synthesized speech.

---

## 1. Overview

Sign language is the primary means of communication for millions of Deaf and
hard-of-hearing people, yet most computing interfaces assume spoken or typed
input. This project bridges that gap with a real-time interpreter that:

1. Captures hand motion through a standard webcam.
2. Extracts **21 hand landmarks per hand** using Google's MediaPipe Hands.
3. Classifies the gesture with one of two neural networks:
   * a **Static MLP** for the 29-class ASL alphabet (A-Z plus `del`,
     `nothing`, `space`), and
   * a **Dynamic LSTM** for word-level signs over a 30-frame sequence.
4. Displays the recognized letter/word and speaks it aloud via the browser's
   **Web Speech API**.

The design deliberately separates concerns: heavy training and preprocessing
run **offline on the GPU desktop with no webcam**, while live capture happens
only in the lightweight React frontend and an optional laptop webcam demo.

---

## 2. System Architecture

```
                           REAL-TIME SIGN LANGUAGE INTERPRETER
                           ====================================

   ┌──────────────────────────────────────────────────────────────────────┐
   │  PHASE 1: OFFLINE PREPROCESSING  (GPU desktop, NO webcam)              │
   │                                                                        │
   │  ASL alphabet images ─┐                       WLASL gloss videos ─┐    │
   │  (datasets/asl/...)   │                       (datasets/.../SL)   │    │
   │                       ▼                                           ▼    │
   │             MediaPipe Hands (static)               MediaPipe Hands     │
   │             extract 1 hand → 63 floats             extract 2 hands     │
   │                       │                            → 126 floats/frame  │
   │                       ▼                                   │            │
   │            data/processed/                                ▼            │
   │            asl_landmarks.npz               data/processed/             │
   │                                            wlasl_sequences.npz         │
   └──────────────────────────────────────────────────────────────────────┘
                       │                                   │
                       ▼                                   ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │  PHASE 2: TRAINING  (PyTorch on CUDA / RTX 4060 Ti)                    │
   │                                                                        │
   │   StaticSignMLP (63 → 256 → 128 → 29)      DynamicSignLSTM             │
   │        normalized landmarks                (126 → LSTM(128,2) → K)     │
   │             │                                       │                  │
   │             ▼                                       ▼                  │
   │   models/static_mlp.pt  + labels        models/dynamic_lstm.pt + labels│
   └──────────────────────────────────────────────────────────────────────┘
                       │                                   │
                       └─────────────────┬─────────────────┘
                                         ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │  PHASE 3: REAL-TIME INFERENCE                                          │
   │                                                                        │
   │   ┌─────────────────────┐   RAW landmarks   ┌────────────────────────┐ │
   │   │  React Frontend     │  (JSON over HTTP) │  Flask Backend (5000)  │ │
   │   │  @mediapipe/hands   │ ────────────────► │  /api/predict/static   │ │
   │   │  webcam capture     │                   │  /api/predict/dynamic  │ │
   │   │  Web Speech API     │ ◄──────────────── │  normalize + classify  │ │
   │   │  (text + voice)     │   prediction+conf │  on GPU/CPU            │ │
   │   └─────────────────────┘                   └────────────────────────┘ │
   │                                                                        │
   │   Alt. client: src/inference/webcam_demo.py (OpenCV window, laptop)    │
   └──────────────────────────────────────────────────────────────────────┘
```

**Key invariant:** clients always send **RAW (un-normalized)** landmarks.
Landmark normalization lives in exactly one place — `src/common/landmarks.py`
— and is applied server-side and inside the training datasets, so the models
*always* see normalized input.

---

## 3. The Three Phases

| Phase | What happens | Where it runs |
|-------|--------------|---------------|
| **1. Preprocessing** | Convert raw images/videos into compact landmark feature arrays (`.npz`). | Offline, GPU desktop, **no webcam**. |
| **2. Training** | Train the static MLP and dynamic LSTM in PyTorch on the GPU; save checkpoints + label files. | Offline, CUDA. |
| **3. Inference** | Serve the trained models via Flask; capture live webcam landmarks in React (or the OpenCV demo) and display + speak predictions. | Real time. |

---

## 4. Technology Stack

**Machine Learning / CV**
- [PyTorch](https://pytorch.org/) (CUDA build for the RTX 4060 Ti)
- [MediaPipe Hands](https://developers.google.com/mediapipe) — 21-landmark hand tracking
- OpenCV (`opencv-python`) — image/video decoding
- NumPy, scikit-learn (train/test split, metrics), tqdm

**Backend**
- Flask + flask-cors (REST API on `http://localhost:5000`)
- flask-socketio (optional real-time channel; the app runs fine without it)

**Frontend**
- React (plain JavaScript, Create-React-App style)
- `@mediapipe/hands`, `@mediapipe/camera_utils`, `@mediapipe/drawing_utils`
- Browser **Web Speech API** for text-to-speech

**Hardware target**
- NVIDIA RTX 4060 Ti (16 GB, CUDA), Ryzen 7 7700X, 32 GB RAM, Windows 10
- Python 3.11, Node 24

---

## 5. Datasets

Both datasets are already present on disk. **Exact on-disk paths** (do not move them):

### 5.1 ASL Alphabet — STATIC images
- **Train:** `c:/SignInterpreter/datasets/asl/asl_alphabet_train/asl_alphabet_train/<CLASS>/*.jpg`
- **Test:** `c:/SignInterpreter/datasets/asl/asl_alphabet_test/asl_alphabet_test/<LETTER>_test.jpg`
- **29 classes:** `A`–`Z`, `del`, `nothing`, `space` (~3000 images per class).
- Used to train the **StaticSignMLP** (one hand → 63-dim feature vector).

### 5.2 WLASL Word-Level — DYNAMIC videos
- **Path:** `c:/SignInterpreter/datasets/archive/dataset/SL/<gloss>/<video_id>.mp4`
- ~2000 gloss folders, ~5–7 videos each. **The gloss is the folder name** —
  there is **no JSON metadata file**; labels are derived from folder names.
- By default we select the **top-K glosses ranked by number of videos**
  (`K = config.WLASL_DEFAULT_NUM_GLOSSES = 20`).
- Used to train the **DynamicSignLSTM** (two hands → 126-dim per frame,
  30-frame sequences).

> **Note on scale:** word-level samples are very few per class (demo-scale).
> For a stronger model, increase the number of glosses, add more videos per
> gloss, and apply temporal/spatial augmentation. This is documented as a
> known limitation and future-work item.

---

## 6. Repository Layout

```
SignInterpreter/
├── config.py                 # ALL shared constants + paths + get_device()
├── requirements.txt          # Python deps (see CUDA note inside)
├── README.md                 # this file
├── .gitignore
├── data/
│   ├── raw/                  # (reserved) raw scratch space
│   └── processed/            # generated .npz feature archives
├── models/                   # trained .pt checkpoints + *_labels.json
├── docs/
│   ├── SETUP.md              # environment setup (venv, pip, npm)
│   ├── CUDA_GUIDE.md         # GPU / CUDA PyTorch installation & checks
│   └── RUN_INSTRUCTIONS.md   # exact ordered commands to run everything
└── src/
    ├── common/landmarks.py   # SINGLE source of truth: normalization + MediaPipe
    ├── models/               # StaticSignMLP, DynamicSignLSTM + save/load
    ├── preprocessing/        # preprocess_asl_alphabet.py, preprocess_wlasl.py
    ├── training/             # train_static.py, train_dynamic.py
    ├── inference/            # predictors, webcam_demo.py, test_api.py
    └── backend/              # Flask app.py, model_manager.py
```

---

## 7. Quickstart

The full, copy-pasteable command sequence lives in
[`docs/RUN_INSTRUCTIONS.md`](docs/RUN_INSTRUCTIONS.md). In short:

1. **Set up the environment** → [`docs/SETUP.md`](docs/SETUP.md)
2. **Enable the GPU** → [`docs/CUDA_GUIDE.md`](docs/CUDA_GUIDE.md)
3. **Preprocess → train → serve → run the UI** → [`docs/RUN_INSTRUCTIONS.md`](docs/RUN_INSTRUCTIONS.md)

```powershell
# from the project root: c:\SignInterpreter
python -m src.preprocessing.preprocess_asl_alphabet   # 1) build static features
python -m src.training.train_static                   # 2) train alphabet MLP
python -m src.preprocessing.preprocess_wlasl          # 3) build dynamic features
python -m src.training.train_dynamic                  # 4) train word LSTM
python -m backend.app                                 # 5) start Flask API (:5000)
# 6) in frontend/:  npm install ; npm start            # React UI (:3000)
python src/inference/test_api.py                      # 7) test API, no webcam
python src/inference/webcam_demo.py                   # 8) OpenCV webcam demo (laptop)
```

---

## 8. Results

Both models were trained on an NVIDIA RTX 4060 Ti (CUDA), fully offline (no
webcam). Reproduce with the commands in Section 7.

| Model | Task | Classes | Samples | Validation accuracy |
|-------|------|---------|---------|---------------------|
| Static MLP (`static_mlp.pt`) | ASL alphabet (single-frame landmarks) | 29 (A–Z, space, del, nothing) | ~87k images → 63-d landmark vectors | **~99%** |
| Dynamic LSTM (`dynamic_lstm.pt`) | Word-level signs (landmark sequences) | 10 glosses | 151 video sequences (30×126) | **~84%** (best) |

- **Static model:** converges within ~10 epochs with no over-fitting (validation
  loss tracks training loss). Training curves: `models/static_history.png`.
- **Dynamic model:** the 10 best-populated WLASL glosses — *cousin, before, cool,
  thin, drink, go, computer, help, inform, take*. Per-class report and curves:
  `models/dynamic_history.png`. `before` and `take` reach perfect recall; the
  ~84% figure is strong given only 14–17 clips per class.
- The static alphabet model is the headline result; the dynamic model
  demonstrates the temporal (sequence) pipeline on a deliberately data-poor set.

---

## 9. Academic Notes

- **Reproducibility:** preprocessing and training are fully offline and
  deterministic given fixed seeds; no webcam is required to build the models.
- **Separation of concerns:** a single normalization implementation guarantees
  that training-time and inference-time features are identical.
- **Limitations & future work:** demo-scale word data, single-frame static
  recognition, and dependence on MediaPipe detection quality. Future work:
  larger gloss vocabulary, data augmentation, transformer sequence models, and
  continuous (sentence-level) recognition.

---

*This project was developed as a university AI coursework submission.*
