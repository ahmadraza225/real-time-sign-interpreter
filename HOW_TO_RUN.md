# How to Run — Sign Language Interpreter

A simple, step-by-step guide. The two AI models are **already trained and
included**, so you can run the app straight away. You only need to retrain if you
want to.

**Team:** Ahmed Raza · Humdia Amjad · Hafsa Irfan

---

## What you need

| Tool | Why | Note |
|------|-----|------|
| Python 3.11 | Backend + AI models | Already used to train the models |
| Node.js 18+ | React website | Run `node --version` to check |
| A webcam | Live signing | Optional — the app still opens without one |
| NVIDIA GPU (CUDA) | Only for **retraining** | Models were trained on an RTX 4060 Ti |

---

## Fastest start (models already trained)

You do **not** need the datasets or a GPU for this. Open **two** terminals in the
project folder `C:\SignInterpreter`.

**Terminal 1 — start the backend (the AI server):**

```powershell
pip install torch torchvision        # CPU build is fine just to run it
pip install -r requirements.txt
python -m backend.app                 # serves the models at http://localhost:5000
```

**Terminal 2 — start the website:**

```powershell
cd frontend
npm install
copy .env.example .env
npm start                             # opens http://localhost:3000
```

That's it. Allow the camera when the browser asks, and start signing.

> **Windows tip:** if `npm` shows `running scripts is disabled on this system`,
> that is a PowerShell setting, not a bug. Fix it once with
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, **or**
> just type `npm.cmd` instead of `npm` (for example `npm.cmd start`).

---

## No webcam? Test the AI without a camera

The website will show a "Camera unavailable" card, which is normal. You can still
prove the AI works:

```powershell
python -m src.inference.test_api      # sends sample data to the backend and prints the result
```

---

## Retrain from scratch (optional — needs the datasets + GPU)

Run these from `C:\SignInterpreter`, in order:

```powershell
python -m src.preprocessing.preprocess_asl_alphabet   # 1. letters: images -> hand points
python -m src.training.train_static                   # 2. train the letter model
python -m src.preprocessing.preprocess_wlasl --num-glosses 10   # 3. words: videos -> sequences
python -m src.training.train_dynamic                  # 4. train the word model
```

The trained models are saved into the `models/` folder.

---

## On a laptop with a camera (no browser needed)

```powershell
python -m src.inference.webcam_demo   # opens a window, shows live letters; press q to quit
```

---

## Quick fixes

| Problem | Fix |
|---------|-----|
| `npm` won't run on Windows | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or use `npm.cmd` |
| Website says "Backend offline" | Make sure Terminal 1 (`python -m backend.app`) is running |
| `torch.cuda.is_available()` is False | Reinstall torch from the CUDA link in `docs/CUDA_GUIDE.md` (only matters for retraining) |
| `ModuleNotFoundError: config` | Run commands from the `C:\SignInterpreter` folder |

Full technical details are in [`README.md`](README.md) and [`docs/`](docs/).
