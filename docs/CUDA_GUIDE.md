# CUDA / GPU Guide (NVIDIA RTX 4060 Ti)

This guide explains how to get PyTorch running on the GPU for training and
inference on the project's hardware:

- **GPU:** NVIDIA GeForce RTX 4060 Ti, **16 GB VRAM** (Ada Lovelace, `sm_89`)
- **CPU:** AMD Ryzen 7 7700X, **32 GB** system RAM
- **OS:** Windows 10, PowerShell
- **Python:** 3.11

The RTX 4060 Ti is fully supported by current PyTorch CUDA builds. The only
common mistake is accidentally installing the **CPU-only** wheel.

---

## 1. Verify the NVIDIA driver

PyTorch's CUDA wheels bundle their own CUDA runtime — you do **not** need to
install the CUDA Toolkit separately — but you **do** need a recent NVIDIA
display driver. Check it with:

```powershell
nvidia-smi
```

You should see a table listing the **GeForce RTX 4060 Ti**, the **Driver
Version**, and a **CUDA Version** (this number is the *maximum* CUDA version
the driver supports, not what is installed). For PyTorch cu121/cu124 wheels,
a driver reporting CUDA 12.1 or newer is sufficient.

If `nvidia-smi` is not found or shows no GPU:
- Install/update the driver from <https://www.nvidia.com/Download/index.aspx>
  (choose *Game Ready* or *Studio* driver for the RTX 4060 Ti).
- Reboot, then re-run `nvidia-smi`.

---

## 2. Install CUDA-enabled PyTorch

Inside your activated virtual environment (see `SETUP.md`), install the
GPU build of PyTorch from the dedicated index. **Recommended for the RTX
4060 Ti (CUDA 12.1):**

```powershell
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

Alternative (CUDA 12.4):

```powershell
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
```

> The `--index-url` is essential. Without it, `pip install torch` pulls the
> CPU-only wheel from PyPI and `torch.cuda.is_available()` will be `False`.

If you previously installed the CPU wheel, remove it first:

```powershell
pip uninstall -y torch torchvision
```

then re-run the CUDA install command above.

---

## 3. Verify PyTorch sees the GPU

```powershell
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Expected output (versions may vary):

```
torch: 2.3.1+cu121
CUDA available: True
CUDA runtime: 12.1
GPU: NVIDIA GeForce RTX 4060 Ti
```

The project's `config.py` performs the same detection: `config.DEVICE` will be
`"cuda"` and `config.get_device()` returns `torch.device("cuda")` when the GPU
is available. You can confirm with:

```powershell
python -c "import config; print('config.DEVICE =', config.DEVICE)"
```

---

## 4. Confirm the GPU is actually used during training

The training scripts (`src/training/train_static.py`,
`src/training/train_dynamic.py`) move the model and batches to
`config.get_device()` and print the active device at startup, e.g.:

```
[train_static] Using device: cuda
```

To watch GPU utilization live while training runs, open a **second**
PowerShell window and poll `nvidia-smi`:

```powershell
# refresh every 1 second
nvidia-smi -l 1
```

While training you should see:
- a non-zero **GPU-Util %**,
- the python process listed under **Processes** with allocated **GPU Memory**.

If GPU-Util stays at 0% and the python process is absent, PyTorch is running
on CPU — revisit steps 2 and 3.

---

## 5. VRAM tips for the 16 GB card

The 4060 Ti's 16 GB is generous for this project (landmark vectors are tiny —
63 or 126 floats), so memory is rarely the bottleneck. Still, useful habits:

- **Batch size:** the defaults (static `batch_size=256`, dynamic
  `batch_size=64`) fit comfortably. You can raise the static batch size to
  512–1024 to speed up training; raise the dynamic batch only if VRAM allows
  (sequences are 30 × 126 floats, still small).
- **Free cached memory** between large runs if needed:
  ```python
  import torch; torch.cuda.empty_cache()
  ```
- **Mixed precision** (optional speedup): the data here is so small that fp32
  is fine, but `torch.autocast("cuda")` can be added for larger experiments.
- **Single-process inference:** the Flask backend loads both models once into
  GPU memory at startup — well under 1 GB total — leaving plenty of headroom.
- **Monitor allocation** from Python:
  ```python
  import torch
  print(torch.cuda.memory_allocated() / 1e6, "MB allocated")
  print(torch.cuda.max_memory_allocated() / 1e6, "MB peak")
  ```

---

## 6. Common pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CUDA available: False` | CPU-only wheel installed | Reinstall via `--index-url https://download.pytorch.org/whl/cu121`. |
| `torch.__version__` ends in `+cpu` | CPU build | Uninstall and reinstall the cu121/cu124 build. |
| `nvidia-smi` not found | Driver not installed / not on PATH | Install the NVIDIA driver, reboot. |
| `CUDA error: no kernel image is available` | torch too old for Ada `sm_89` | Use torch ≥ 2.1 with cu121/cu124. |
| Training uses CPU despite CUDA available | Tensors not moved to device | Project scripts already call `.to(get_device())`; ensure you run the provided scripts. |
| `DLL load failed` on `import torch` | Missing VC++ runtime | Install the latest *Microsoft Visual C++ Redistributable (x64)*. |

---

## 7. CPU fallback

The project is designed to run on CPU as well — `config.DEVICE` falls back to
`"cpu"` automatically. Preprocessing (MediaPipe) is CPU-bound regardless;
training the small MLP/LSTM on CPU is slower but still feasible for this
demo-scale data. For the best experience, use the RTX 4060 Ti as described
above.
