# Setup Guide (Windows 10 + PowerShell)

This guide gets the **Real-Time Sign Language Interpreter** running from a
fresh checkout on Windows 10 using PowerShell. Run every command from the
project root unless told otherwise.

> Project root: `c:\SignInterpreter`
> Targets: Python **3.11**, Node **24**, NVIDIA **RTX 4060 Ti** (CUDA).

---

## 0. Prerequisites

Confirm the required tools are installed and on your PATH:

```powershell
# Python 3.11 (must report 3.11.x)
python --version

# pip
python -m pip --version

# Node + npm (for the React frontend)
node --version
npm --version
```

If `python` does not point to 3.11, install Python 3.11 from python.org and
either use the launcher (`py -3.11 ...`) or re-create your PATH accordingly.

For GPU acceleration you also need a recent NVIDIA driver — see
[`CUDA_GUIDE.md`](CUDA_GUIDE.md).

---

## 1. Open PowerShell at the project root

```powershell
Set-Location c:\SignInterpreter
```

If PowerShell blocks the virtual-environment activation script later, allow
local scripts for the current user (one-time):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

## 2. Create and activate a virtual environment

```powershell
# Create a venv named ".venv" using Python 3.11
python -m venv .venv

# Activate it (PowerShell)
.\.venv\Scripts\Activate.ps1
```

After activation your prompt is prefixed with `(.venv)`. Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

---

## 3. Install PyTorch with CUDA support (do this FIRST)

The default PyPI `torch` wheel is **CPU-only** on Windows. Install the
CUDA-enabled build *before* the rest of the requirements so pip does not pull
the CPU wheel. For the RTX 4060 Ti the CUDA 12.1 build is recommended:

```powershell
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

(If you prefer CUDA 12.4: `pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124`.)

Full details, driver checks, and troubleshooting are in
[`CUDA_GUIDE.md`](CUDA_GUIDE.md).

---

## 4. Install the remaining Python dependencies

```powershell
pip install -r requirements.txt
```

`requirements.txt` pins the same `torch`/`torchvision` versions, so pip sees
them as already satisfied and installs only the other packages (mediapipe,
opencv-python, numpy, scikit-learn, tqdm, flask, flask-cors, flask-socketio,
requests, matplotlib).

---

## 5. Verify the Python environment

```powershell
# Confirm CUDA is visible to PyTorch (should print: True)
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# Confirm the config module imports and resolves paths
python -c "import config; print('Device:', config.DEVICE); print('ASL train dir:', config.ASL_TRAIN_DIR)"

# Confirm MediaPipe and OpenCV import cleanly
python -c "import mediapipe, cv2, numpy, sklearn; print('CV/ML stack OK')"
```

If `CUDA available` prints `False`, see [`CUDA_GUIDE.md`](CUDA_GUIDE.md). The
project still runs on CPU, just slower.

---

## 6. Install the React frontend dependencies

The frontend lives in `c:\SignInterpreter\frontend` (created by the frontend
subsystem). Install its npm packages:

```powershell
Set-Location c:\SignInterpreter\frontend
npm install
Set-Location c:\SignInterpreter
```

> **If `npm` fails with `running scripts is disabled on this system` /
> `UnauthorizedAccess`:** that is PowerShell's execution policy blocking
> `npm.ps1` — not a project problem. Fix it with EITHER:
>
> ```powershell
> # Option A (permanent, per-user, no admin) — then use `npm` normally:
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned   # answer Y
>
> # Option B (no policy change) — call the .cmd shim, which PS does not block:
> npm.cmd install
> npm.cmd start
> ```

This installs React plus `@mediapipe/hands`, `@mediapipe/camera_utils`, and
`@mediapipe/drawing_utils`. Text-to-speech uses the browser's built-in Web
Speech API (no package needed).

---

## 7. Confirm dataset paths

The datasets are already on disk. Confirm the expected folders exist:

```powershell
Test-Path c:\SignInterpreter\datasets\asl\asl_alphabet_train\asl_alphabet_train   # True
Test-Path c:\SignInterpreter\datasets\asl\asl_alphabet_test\asl_alphabet_test      # True
Test-Path c:\SignInterpreter\datasets\archive\dataset\SL                            # True
```

All three must return `True` before preprocessing.

---

## 8. Next steps

You are ready to build and run the system. Follow the exact ordered commands
in [`RUN_INSTRUCTIONS.md`](RUN_INSTRUCTIONS.md):

1. Preprocess the ASL alphabet images.
2. Train the static MLP.
3. Preprocess the WLASL videos.
4. Train the dynamic LSTM.
5. Start the Flask backend.
6. Start the React frontend.
7. Test the backend without a webcam.
8. Run the OpenCV webcam demo (on a laptop with a camera).

---

## Troubleshooting quick reference

| Symptom | Fix |
|---------|-----|
| `Activate.ps1 cannot be loaded` | Run the `Set-ExecutionPolicy` command in step 1. |
| `npm : ... npm.ps1 cannot be loaded` / `running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or use `npm.cmd` instead of `npm`. See step 6. |
| `torch.cuda.is_available()` is `False` | Reinstall torch from the CUDA index-url; update the NVIDIA driver. See `CUDA_GUIDE.md`. |
| `ModuleNotFoundError: config` | Run scripts from the project root (`c:\SignInterpreter`); entry scripts bootstrap `sys.path`. |
| `mediapipe` install fails | Ensure Python is 3.11 (mediapipe wheels target specific versions). |
| Dataset `Test-Path` returns `False` | Verify the datasets were extracted to the exact paths in step 7. |
