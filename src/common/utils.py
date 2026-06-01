"""
utils.py
========

Small, dependency-light utility helpers shared across preprocessing, training,
inference and the Flask backend.

Everything here is intentionally generic so that the *domain* logic (landmark
extraction, model architectures) lives elsewhere.  Functions covered:

* :func:`set_seed`               -- make a run reproducible.
* :func:`save_json` / :func:`load_json` -- tiny JSON read/write helpers.
* :func:`save_labels`            -- persist an index->label list as JSON.
* :func:`get_logger`             -- a consistently-formatted stdlib logger.
* :func:`pad_or_truncate_sequence` -- fix a variable-length temporal sequence to
  a constant length (used by dynamic preprocessing AND dynamic inference, so the
  padding/truncation logic is defined exactly once).
* :func:`ensure_dir`             -- create a directory (and parents) if needed.
"""

import os
import json
import random
import logging

import numpy as np


def set_seed(seed):
    """Seed Python, NumPy and (if available) PyTorch RNGs for reproducibility.

    PyTorch is imported lazily so that scripts which only do data wrangling do
    not pay the import cost (and so this module stays importable even in a
    torch-free context).

    Parameters
    ----------
    seed : int
        The random seed to apply everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch not installed in this context -- that's fine, the rest is seeded.
        pass


def ensure_dir(path):
    """Ensure that ``path`` exists as a directory, creating parents as needed.

    If ``path`` looks like a file path (has a non-empty directory component and
    an extension) callers should pass its *directory*; to be forgiving we simply
    create whatever directory string is given.

    Parameters
    ----------
    path : str
        Directory path to create.  No error if it already exists.

    Returns
    -------
    str
        The same ``path``, for convenient chaining.
    """
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    """Serialize ``obj`` to ``path`` as pretty-printed UTF-8 JSON.

    The parent directory is created automatically if it does not exist.

    Parameters
    ----------
    obj : Any
        A JSON-serializable object (dict, list, str, number, bool, None).
    path : str
        Destination file path.
    """
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    """Load and return the JSON object stored at ``path``.

    Parameters
    ----------
    path : str
        Source file path.

    Returns
    -------
    Any
        The deserialized JSON object.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_labels(labels, path):
    """Persist a class-label list as a plain JSON array (index -> label).

    The backend and frontend read this file to map model output indices back to
    human-readable class strings.  We coerce every entry to ``str`` so the file
    is always a clean array of strings regardless of the input type.

    Parameters
    ----------
    labels : sequence
        Ordered labels where position ``i`` is the name of class ``i``.
    path : str
        Destination ``*_labels.json`` path.
    """
    labels_list = [str(label) for label in labels]
    save_json(labels_list, path)


def get_logger(name):
    """Return a configured :class:`logging.Logger` with a single stream handler.

    Repeated calls with the same ``name`` return the same logger and do NOT add
    duplicate handlers (a common logging pitfall).

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__`` of the calling module).

    Returns
    -------
    logging.Logger
        A logger that writes ``LEVEL [name] message`` lines to stdout at INFO.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Prevent messages from also propagating to the (possibly unconfigured)
        # root logger, which would duplicate output.
        logger.propagate = False
    return logger


def pad_or_truncate_sequence(seq, length, feature_dim):
    """Force a variable-length feature sequence to exactly ``length`` frames.

    Used by BOTH the WLASL preprocessing script and dynamic inference so the
    padding/truncation rule is defined in exactly one place:

    * If the sequence is **longer** than ``length`` it is truncated by keeping
      the first ``length`` frames.
    * If the sequence is **shorter** it is padded at the end with zero frames
      (each zero frame is ``feature_dim`` zeros, i.e. "no hands detected").
    * An empty sequence becomes an all-zero ``(length, feature_dim)`` array.

    Parameters
    ----------
    seq : array-like of shape (T, feature_dim)
        The input sequence (``T`` may be 0).
    length : int
        Target number of frames.
    feature_dim : int
        Per-frame feature dimension (e.g. ``config.DYNAMIC_FEATURE_DIM`` = 126).

    Returns
    -------
    numpy.ndarray, dtype float32, shape (length, feature_dim)
        The fixed-length sequence.
    """
    arr = np.asarray(seq, dtype=np.float32)

    # Normalize the shape of empty / degenerate inputs to (0, feature_dim).
    if arr.size == 0:
        arr = np.zeros((0, feature_dim), dtype=np.float32)
    elif arr.ndim == 1:
        # A single flat frame -> reshape to (1, feature_dim) when it fits.
        if arr.shape[0] == feature_dim:
            arr = arr.reshape(1, feature_dim)
        else:
            raise ValueError(
                "pad_or_truncate_sequence got a 1-D array of length %d that is "
                "not equal to feature_dim=%d" % (arr.shape[0], feature_dim)
            )
    elif arr.ndim != 2 or arr.shape[1] != feature_dim:
        raise ValueError(
            "pad_or_truncate_sequence expects shape (T, %d); got %r"
            % (feature_dim, arr.shape)
        )

    out = np.zeros((length, feature_dim), dtype=np.float32)
    n = min(arr.shape[0], length)
    if n > 0:
        out[:n] = arr[:n]
    return out
