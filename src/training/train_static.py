"""
train_static.py
================

Trains the STATIC sign classifier (:class:`StaticSignMLP`) on the
landmark features extracted from the ASL alphabet dataset.

Pipeline
--------
1. Load the pre-extracted landmark features from ``asl_landmarks.npz`` (created
   by the preprocessing step). The archive stores RAW (un-normalized) 63-dim
   landmark rows in ``X`` and integer/string labels in ``y``, plus an ordered
   ``classes`` array (index -> label).
2. Apply ``normalize_hand`` to EVERY row so the model only ever sees the single,
   canonical normalization (translation + scale invariant). This is done here
   (not at preprocessing time) so there is exactly one normalization
   implementation, shared with the live predictors and the frontend pipeline.
3. Stratified train / validation split (scikit-learn).
4. Train with Adam + CrossEntropy on the GPU when available, tracking the best
   validation accuracy and saving that checkpoint.
5. Print a scikit-learn classification report on the validation set and save a
   training-history plot (loss + accuracy curves).

Run from the project root::

    python -m src.training.train_static
    # or
    python src/training/train_static.py
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# sys.path bootstrap: insert the PROJECT ROOT so that ``import config`` and
# ``from src.common.landmarks import ...`` both resolve, regardless of how this
# script is launched. The root is computed from this file's location
# (src/training/train_static.py -> two directories up), never hard-coded.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

# matplotlib is OPTIONAL — it is only used to save a training-history PNG.
# If it is not installed (e.g. its build failed on a machine without a C
# compiler), training still runs fully; we just skip the plot.
try:
    import matplotlib  # noqa: E402

    matplotlib.use("Agg")  # headless backend: required on a server / no-display box
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
from src.common.landmarks import normalize_hand  # noqa: E402
from src.models.static_model import StaticSignMLP, save_static_model  # noqa: E402


def parse_args():
    """Parse command-line hyper-parameters with sensible contract defaults."""
    parser = argparse.ArgumentParser(
        description="Train the static (ASL alphabet) sign MLP classifier."
    )
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--processed-path",
        type=str,
        default=config.ASL_LANDMARKS_NPZ,
        help="Path to the asl_landmarks.npz feature archive.",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15, help="Fraction of data held out for validation."
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
    """Load and normalize the static landmark dataset.

    Parameters
    ----------
    processed_path : str
        Path to ``asl_landmarks.npz`` containing ``X`` (N, 63) RAW landmarks,
        ``y`` (N,) labels, and ``classes`` (ordered label list).

    Returns
    -------
    tuple
        ``(X_norm, y_idx, classes)`` where ``X_norm`` is float32 (N, 63)
        normalized features, ``y_idx`` is an int64 array of class indices, and
        ``classes`` is the ordered list of class-name strings.
    """
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"Processed feature file not found: {processed_path}\n"
            "Run the static preprocessing step first to create asl_landmarks.npz."
        )

    archive = np.load(processed_path, allow_pickle=True)
    X = np.asarray(archive["X"], dtype=np.float32)
    y_raw = np.asarray(archive["y"])

    # The ``classes`` array (index -> label) is authoritative for ordering.
    classes = [str(c) for c in archive["classes"].tolist()]

    if X.ndim != 2 or X.shape[1] != config.STATIC_FEATURE_DIM:
        raise ValueError(
            f"Expected X of shape (N, {config.STATIC_FEATURE_DIM}); got {X.shape}."
        )

    # Labels may be stored either as integer indices or as class-name strings.
    # Build an int64 index array consistent with ``classes`` either way.
    if y_raw.dtype.kind in ("i", "u"):
        y_idx = y_raw.astype(np.int64)
    else:
        label_to_idx = {label: i for i, label in enumerate(classes)}
        y_idx = np.array([label_to_idx[str(label)] for label in y_raw], dtype=np.int64)

    # Apply the SINGLE canonical normalization to every sample.
    X_norm = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        X_norm[i] = normalize_hand(X[i])

    return X_norm, y_idx, classes


def make_loaders(X, y, val_split, batch_size, seed, pin_memory):
    """Create stratified train / validation DataLoaders.

    Returns
    -------
    tuple
        ``(train_loader, val_loader, X_val_t, y_val_np)`` where the last two
        are kept for the final classification report.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, drop_last=False
    )
    return train_loader, val_loader, torch.from_numpy(X_val), y_val


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

    No-op (with a notice) when matplotlib is unavailable so training never
    fails just because the optional plotting dependency is missing.
    """
    if not HAS_MATPLOTLIB:
        print("[train_static] matplotlib not available — skipping history plot.")
        return

    epochs_axis = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs_axis, history["train_loss"], label="train loss")
    ax_loss.plot(epochs_axis, history["val_loss"], label="val loss")
    ax_loss.set_title("Static MLP — Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs_axis, history["train_acc"], label="train acc")
    ax_acc.plot(epochs_axis, history["val_acc"], label="val acc")
    ax_acc.set_title("Static MLP — Accuracy")
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
    print(f"[train_static] Using device: {device}")
    if device.type == "cuda":
        print(f"[train_static] CUDA device: {torch.cuda.get_device_name(device)}")
    pin_memory = device.type == "cuda"

    # ----------------------------------------------------------------- data --
    print(f"[train_static] Loading features from: {args.processed_path}")
    X, y, classes = load_dataset(args.processed_path)
    num_classes = len(classes)
    print(
        f"[train_static] Loaded {X.shape[0]} samples, "
        f"{X.shape[1]} features, {num_classes} classes."
    )

    train_loader, val_loader, X_val_t, y_val_np = make_loaders(
        X, y, args.val_split, args.batch_size, args.seed, pin_memory
    )

    # ---------------------------------------------------------------- model --
    model = StaticSignMLP(num_classes=num_classes, in_dim=config.STATIC_FEATURE_DIM).to(device)
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
            save_static_model(model, classes, config.STATIC_MODEL_PATH)
            marker = "  <- best (saved)"

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}{marker}"
        )

    print(f"[train_static] Best validation accuracy: {best_val_acc:.4f}")
    print(f"[train_static] Best model saved to: {config.STATIC_MODEL_PATH}")
    print(f"[train_static] Labels saved to:     {config.STATIC_LABELS_PATH}")

    # ----------------------------------------------------- classification report --
    # Report on the validation set using the FINAL model state. (This reflects
    # the end-of-training model; the best checkpoint is already saved to disk.)
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t.to(device))
        val_preds = val_logits.argmax(dim=1).cpu().numpy()

    print("\n[train_static] Validation classification report:")
    print(
        classification_report(
            y_val_np,
            val_preds,
            labels=list(range(num_classes)),
            target_names=classes,
            zero_division=0,
        )
    )

    # ------------------------------------------------------------- history plot --
    history_path = os.path.join(config.MODELS_DIR, "static_history.png")
    plot_history(history, history_path)
    print(f"[train_static] Training history plot saved to: {history_path}")


if __name__ == "__main__":
    main()
