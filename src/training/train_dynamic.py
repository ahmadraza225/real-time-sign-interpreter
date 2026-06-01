"""
train_dynamic.py
================

Trains the DYNAMIC (word-level) sign classifier (:class:`DynamicSignLSTM`) on
landmark *sequences* extracted from the WLASL-style video dataset.

Pipeline
--------
1. Load pre-extracted sequences from ``wlasl_sequences.npz`` (created by the
   dynamic preprocessing step). The archive stores RAW (un-normalized)
   sequences in ``X`` with shape ``(N, SEQUENCE_LENGTH, 126)``, labels in ``y``
   and an ordered ``classes`` array (index -> gloss).
2. Apply ``normalize_sequence`` to EVERY sample so the model only ever sees the
   single canonical normalization (each hand is normalized independently per
   frame). This keeps one normalization implementation shared with the live
   predictors and the frontend pipeline.
3. Stratified train / validation split. Because this demo-scale dataset has
   very few samples per gloss, a fully stratified split can be impossible (a
   class may have < 2 samples). In that case we WARN and fall back to a plain
   (non-stratified) random split.
4. Train an LSTM on the GPU when available, tracking and saving the best
   validation accuracy.
5. Print a scikit-learn classification report and save a training-history plot.

Note on data scale
-------------------
WLASL gloss folders contain only ~5-7 videos each. With the default top-20
glosses, per-class sample counts are tiny, so validation metrics will be noisy.
For a stronger model, increase the number of glosses and add data augmentation
(e.g. temporal jitter, horizontal flip with hand-label swap, small landmark
noise). This script is intentionally robust to the small-data regime.

Run from the project root::

    python -m src.training.train_dynamic
    # or
    python src/training/train_dynamic.py
"""

import argparse
import os
import sys
import warnings

# ---------------------------------------------------------------------------
# sys.path bootstrap (project root computed from __file__, never hard-coded) so
# that ``import config`` and ``from src.common.landmarks import ...`` resolve.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

# matplotlib is OPTIONAL — only used for the training-history PNG. Training
# runs fully without it (we just skip the plot).
try:
    import matplotlib  # noqa: E402

    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt  # noqa: E402

    HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover - environment dependent
    plt = None
    HAS_MATPLOTLIB = False

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from sklearn.metrics import classification_report  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from tqdm import tqdm  # noqa: E402

import config  # noqa: E402
from src.common.landmarks import normalize_sequence  # noqa: E402
from src.models.dynamic_model import DynamicSignLSTM, save_dynamic_model  # noqa: E402


def parse_args():
    """Parse command-line hyper-parameters with sensible contract defaults."""
    parser = argparse.ArgumentParser(
        description="Train the dynamic (word-level) sign LSTM classifier."
    )
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--processed-path",
        type=str,
        default=config.WLASL_SEQ_NPZ,
        help="Path to the wlasl_sequences.npz feature archive.",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Fraction of data held out for validation."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def set_seed(seed):
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(processed_path):
    """Load and normalize the dynamic sequence dataset.

    Parameters
    ----------
    processed_path : str
        Path to ``wlasl_sequences.npz`` containing ``X``
        ``(N, SEQUENCE_LENGTH, 126)`` RAW sequences, ``y`` (N,) labels and an
        ordered ``classes`` array (index -> gloss).

    Returns
    -------
    tuple
        ``(X_norm, y_idx, classes)`` with float32 normalized sequences, int64
        class indices and the ordered list of gloss strings.
    """
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"Processed sequence file not found: {processed_path}\n"
            "Run the dynamic preprocessing step first to create wlasl_sequences.npz."
        )

    archive = np.load(processed_path, allow_pickle=True)
    X = np.asarray(archive["X"], dtype=np.float32)
    y_raw = np.asarray(archive["y"])
    # The WLASL preprocessing step stores the ordered label list under the
    # "glosses" key (per the project data contract). Accept "classes" too so the
    # loader is robust to either naming.
    if "glosses" in archive.files:
        label_array = archive["glosses"]
    elif "classes" in archive.files:
        label_array = archive["classes"]
    else:
        raise KeyError(
            "wlasl_sequences.npz must contain a 'glosses' (or 'classes') array; "
            f"found keys: {archive.files}"
        )
    classes = [str(c) for c in label_array.tolist()]

    if X.ndim != 3 or X.shape[2] != config.DYNAMIC_FEATURE_DIM:
        raise ValueError(
            f"Expected X of shape (N, T, {config.DYNAMIC_FEATURE_DIM}); got {X.shape}."
        )

    if y_raw.dtype.kind in ("i", "u"):
        y_idx = y_raw.astype(np.int64)
    else:
        label_to_idx = {label: i for i, label in enumerate(classes)}
        y_idx = np.array([label_to_idx[str(label)] for label in y_raw], dtype=np.int64)

    # Apply the SINGLE canonical normalization to every sequence.
    X_norm = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        X_norm[i] = normalize_sequence(X[i])

    return X_norm, y_idx, classes


def stratified_or_random_split(X, y, val_split, seed):
    """Split into train / val, preferring stratification but degrading safely.

    A stratified split requires every class to have at least 2 samples (one per
    side). On the tiny WLASL demo subset this may not hold, so we detect that
    situation, emit a clear warning, and fall back to a plain random split.

    Returns
    -------
    tuple
        ``(X_train, X_val, y_train, y_val, used_stratify)``.
    """
    # Count samples per class to decide whether stratification is feasible.
    _, counts = np.unique(y, return_counts=True)
    min_class_count = int(counts.min())

    if min_class_count >= 2:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_split, random_state=seed, stratify=y
            )
            return X_train, X_val, y_train, y_val, True
        except ValueError as exc:
            # E.g. val_split too small to allocate one per class. Warn + fall back.
            warnings.warn(
                f"[train_dynamic] Stratified split failed ({exc}); "
                "falling back to a non-stratified random split."
            )

    else:
        warnings.warn(
            "[train_dynamic] At least one gloss has fewer than 2 samples "
            f"(min per-class count = {min_class_count}). Stratification is "
            "impossible; falling back to a non-stratified random split. "
            "Consider using more glosses or data augmentation."
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=None
    )
    return X_train, X_val, y_train, y_val, False


def make_loaders(X, y, val_split, batch_size, seed, pin_memory):
    """Create train / validation DataLoaders (stratified when possible).

    Returns
    -------
    tuple
        ``(train_loader, val_loader, X_val_t, y_val_np, used_stratify)``.
    """
    X_train, X_val, y_train, y_val, used_stratify = stratified_or_random_split(
        X, y, val_split, seed
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, drop_last=False
    )
    return train_loader, val_loader, torch.from_numpy(X_val), y_val, used_stratify


def evaluate(model, loader, criterion, device):
    """Run one evaluation pass; return ``(avg_loss, accuracy)``."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total_count += xb.size(0)
    avg_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs):
    """Train for a single epoch; return ``(avg_loss, accuracy)``."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
    for xb, yb in progress:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total_count += xb.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy


def plot_history(history, out_path):
    """Save loss + accuracy curves to ``out_path`` (PNG).

    No-op (with a notice) when matplotlib is unavailable.
    """
    if not HAS_MATPLOTLIB:
        print("[train_dynamic] matplotlib not available — skipping history plot.")
        return

    epochs_axis = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs_axis, history["train_loss"], label="train loss")
    ax_loss.plot(epochs_axis, history["val_loss"], label="val loss")
    ax_loss.set_title("Dynamic LSTM — Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs_axis, history["train_acc"], label="train acc")
    ax_acc.plot(epochs_axis, history["val_acc"], label="val acc")
    ax_acc.set_title("Dynamic LSTM — Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = config.get_device()
    print(f"[train_dynamic] Using device: {device}")
    if device.type == "cuda":
        print(f"[train_dynamic] CUDA device: {torch.cuda.get_device_name(device)}")
    pin_memory = device.type == "cuda"

    # ----------------------------------------------------------------- data --
    print(f"[train_dynamic] Loading sequences from: {args.processed_path}")
    X, y, classes = load_dataset(args.processed_path)
    num_glosses = len(classes)
    print(
        f"[train_dynamic] Loaded {X.shape[0]} sequences of shape "
        f"({X.shape[1]}, {X.shape[2]}) across {num_glosses} glosses."
    )

    train_loader, val_loader, X_val_t, y_val_np, used_stratify = make_loaders(
        X, y, args.val_split, args.batch_size, args.seed, pin_memory
    )
    print(
        f"[train_dynamic] Split: {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} val "
        f"({'stratified' if used_stratify else 'non-stratified'})."
    )

    # ---------------------------------------------------------------- model --
    model = DynamicSignLSTM(
        num_glosses=num_glosses,
        in_dim=config.DYNAMIC_FEATURE_DIM,
        hidden=128,
        layers=2,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --------------------------------------------------------------- training --
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dynamic_model(model, classes, config.DYNAMIC_MODEL_PATH)
            marker = "  <- best (saved)"

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}{marker}"
        )

    print(f"[train_dynamic] Best validation accuracy: {best_val_acc:.4f}")
    print(f"[train_dynamic] Best model saved to: {config.DYNAMIC_MODEL_PATH}")
    print(f"[train_dynamic] Labels saved to:     {config.DYNAMIC_LABELS_PATH}")

    # ----------------------------------------------------- classification report --
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t.to(device))
        val_preds = val_logits.argmax(dim=1).cpu().numpy()

    print("\n[train_dynamic] Validation classification report:")
    print(
        classification_report(
            y_val_np,
            val_preds,
            labels=list(range(num_glosses)),
            target_names=classes,
            zero_division=0,
        )
    )

    # ------------------------------------------------------------- history plot --
    history_path = os.path.join(config.MODELS_DIR, "dynamic_history.png")
    plot_history(history, history_path)
    print(f"[train_dynamic] Training history plot saved to: {history_path}")


if __name__ == "__main__":
    main()
