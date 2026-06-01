"""
static_model.py
===============

Defines the :class:`StaticSignMLP` model used for STATIC sign recognition
(the ASL alphabet: A..Z, ``del``, ``nothing``, ``space``).

A static sign is recognised from a SINGLE frame, so the model input is the
flattened, *normalized* landmarks of ONE hand:

    STATIC_FEATURE_DIM = 21 landmarks * 3 coords (x, y, z) = 63 floats.

The module also provides symmetric save / load helpers that store everything
needed to rebuild the network (the architecture dimensions and the ordered
class list) so that inference code never has to hard-code those numbers.

Saved-model format (a single ``.pt`` file)::

    {
        "state_dict": <module.state_dict()>,
        "arch":       {"in_dim": int, "num_classes": int},
        "classes":    [<class string>, ...]   # index -> label
    }

Educational note
----------------
A plain multilayer perceptron (MLP) is sufficient here because each static
sign is fully described by a single set of normalized landmarks; there is no
temporal information to model. BatchNorm stabilises and speeds up training,
while Dropout regularises the relatively small 63-dimensional input space.
"""

import json
import os

import torch
import torch.nn as nn


class StaticSignMLP(nn.Module):
    """Feed-forward classifier for single-frame (static) hand signs.

    Architecture (exactly as required by the project contract)::

        Linear(63, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.3)
        Linear(256, 128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.3)
        Linear(128, num_classes)

    Parameters
    ----------
    num_classes : int
        Number of output classes (e.g. 29 for the ASL alphabet set).
    in_dim : int, optional
        Size of the flattened input feature vector. Defaults to 63
        (one hand: 21 landmarks * 3 coordinates).
    """

    def __init__(self, num_classes, in_dim=63):
        super().__init__()
        # Remember the dimensions so that save helpers can persist them and the
        # loader can rebuild an identical network without external knowledge.
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x):
        """Compute raw class scores (logits).

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, in_dim)``. The values are expected to be
            already *normalized* landmarks (see ``src/common/landmarks.py``).

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_classes)``. Apply ``softmax`` for
            probabilities; ``CrossEntropyLoss`` expects these raw logits.
        """
        return self.net(x)


def save_static_model(model, classes, path):
    """Serialise a :class:`StaticSignMLP` using the project's standard format.

    Besides writing the ``.pt`` checkpoint, this also writes a sibling JSON
    label file so that the Flask backend and the React frontend can read the
    class list without loading PyTorch. The JSON path is derived from
    ``config.STATIC_LABELS_PATH`` when ``path`` matches the configured model
    path; otherwise it is written next to ``path`` as ``<stem>_labels.json``.

    Parameters
    ----------
    model : StaticSignMLP
        The trained model to save.
    classes : list of str
        Ordered class labels (index -> label).
    path : str
        Destination ``.pt`` file path.
    """
    classes = list(classes)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "arch": {"in_dim": int(model.in_dim), "num_classes": int(model.num_classes)},
        "classes": classes,
    }
    torch.save(checkpoint, path)

    # Also emit the plain JSON label list for non-PyTorch consumers.
    labels_path = _labels_path_for(path)
    os.makedirs(os.path.dirname(os.path.abspath(labels_path)), exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as fh:
        json.dump(classes, fh, indent=2)


def load_static_model(path, map_location=None):
    """Rebuild a :class:`StaticSignMLP` from a saved checkpoint.

    The architecture is reconstructed purely from the stored ``"arch"`` dict,
    then the weights are loaded. The returned model is set to ``eval()`` mode
    because the loader is primarily used for inference.

    Parameters
    ----------
    path : str
        Path to the ``.pt`` checkpoint produced by :func:`save_static_model`.
    map_location : str or torch.device, optional
        Passed through to :func:`torch.load` (e.g. ``"cpu"`` to load a
        GPU-trained model on a CPU-only machine).

    Returns
    -------
    tuple
        ``(model, classes)`` where ``model`` is a ready-to-use
        :class:`StaticSignMLP` and ``classes`` is the ordered label list.
    """
    # weights_only=False: our checkpoint intentionally stores the arch dict and
    # class list alongside the tensors. Setting it explicitly silences the
    # deprecation warning and keeps loading working when torch flips the default.
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    arch = checkpoint["arch"]
    classes = list(checkpoint["classes"])

    model = StaticSignMLP(num_classes=arch["num_classes"], in_dim=arch["in_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    # ``map_location`` only controls where the checkpoint tensors are read; the
    # freshly built module is still on the CPU. Move it onto the requested
    # device so callers that feed CUDA tensors (e.g. StaticPredictor) don't hit
    # a cross-device mismatch at the forward pass.
    if map_location is not None:
        model.to(map_location)
    model.eval()
    return model, classes


def _labels_path_for(model_path):
    """Return the JSON labels path that pairs with a given model path.

    If the model path equals the configured ``STATIC_MODEL_PATH``, the
    configured ``STATIC_LABELS_PATH`` is used. Otherwise a labels file is
    placed next to the model as ``<stem>_labels.json``.
    """
    try:
        import config

        if os.path.abspath(model_path) == os.path.abspath(config.STATIC_MODEL_PATH):
            return config.STATIC_LABELS_PATH
    except Exception:
        # config may be unavailable in some isolated contexts; fall back below.
        pass

    stem, _ = os.path.splitext(model_path)
    return stem + "_labels.json"
