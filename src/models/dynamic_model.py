"""
dynamic_model.py
================

Defines the :class:`DynamicSignLSTM` model used for DYNAMIC (word-level) sign
recognition. A dynamic sign is a *gloss* performed over time, so a single
frame is not enough: the model consumes a SEQUENCE of frames.

Each frame is described by the flattened, *normalized* landmarks of up to TWO
hands:

    DYNAMIC_FEATURE_DIM = 2 hands * 21 landmarks * 3 coords = 126 floats.

The full input is therefore a tensor of shape ``(batch, T, 126)`` where ``T``
is the (padded / truncated) sequence length (``SEQUENCE_LENGTH = 30``).

Saved-model format (a single ``.pt`` file)::

    {
        "state_dict": <module.state_dict()>,
        "arch":       {"in_dim": int, "hidden": int,
                       "layers": int, "num_glosses": int},
        "classes":    [<gloss string>, ...]   # index -> label
    }

Educational note
----------------
An LSTM (Long Short-Term Memory recurrent network) is well suited to sign
glosses because the *order* and *motion* of the hands carry the meaning. We
read the whole sequence and use the hidden state at the LAST timestep as a
fixed-size summary of the motion, which a small classifier head then maps to a
gloss.
"""

import json
import os

import torch
import torch.nn as nn


class DynamicSignLSTM(nn.Module):
    """Recurrent classifier for multi-frame (dynamic) sign glosses.

    Architecture (exactly as required by the project contract)::

        LSTM(in_dim=126, hidden=128, num_layers=2,
             batch_first=True, dropout=0.3)
        -> take hidden state of the LAST timestep
        -> Linear(128, 128) -> ReLU -> Dropout(0.3)
        -> Linear(128, num_glosses)

    Parameters
    ----------
    num_glosses : int
        Number of output gloss classes.
    in_dim : int, optional
        Per-frame feature size. Defaults to 126 (two hands * 21 * 3).
    hidden : int, optional
        LSTM hidden size. Defaults to 128.
    layers : int, optional
        Number of stacked LSTM layers. Defaults to 2.
    """

    def __init__(self, num_glosses, in_dim=126, hidden=128, layers=2):
        super().__init__()
        # Persist dimensions so the network can be rebuilt from a checkpoint.
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.num_glosses = int(num_glosses)

        # ``dropout`` only applies *between* stacked LSTM layers, so it is only
        # meaningful when layers > 1 (PyTorch warns otherwise). With the
        # contract default of 2 layers it is active.
        self.lstm = nn.LSTM(
            input_size=self.in_dim,
            hidden_size=self.hidden,
            num_layers=self.layers,
            batch_first=True,
            dropout=0.3 if self.layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(self.hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_glosses),
        )

    def forward(self, x):
        """Compute gloss logits for a batch of sequences.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, T, in_dim)`` of *normalized* per-frame
            landmark features.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_glosses)``.
        """
        # ``output`` has shape (batch, T, hidden); we summarise the sequence by
        # the hidden representation at the final timestep.
        output, (h_n, c_n) = self.lstm(x)
        last_timestep = output[:, -1, :]  # (batch, hidden)
        return self.head(last_timestep)


def save_dynamic_model(model, classes, path):
    """Serialise a :class:`DynamicSignLSTM` using the standard format.

    Also writes a sibling JSON label list (gloss strings) for the backend /
    frontend, mirroring the behaviour of the static model helpers.

    Parameters
    ----------
    model : DynamicSignLSTM
        The trained model to save.
    classes : list of str
        Ordered gloss labels (index -> label).
    path : str
        Destination ``.pt`` file path.
    """
    classes = list(classes)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "arch": {
            "in_dim": int(model.in_dim),
            "hidden": int(model.hidden),
            "layers": int(model.layers),
            "num_glosses": int(model.num_glosses),
        },
        "classes": classes,
    }
    torch.save(checkpoint, path)

    labels_path = _labels_path_for(path)
    os.makedirs(os.path.dirname(os.path.abspath(labels_path)), exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as fh:
        json.dump(classes, fh, indent=2)


def load_dynamic_model(path, map_location=None):
    """Rebuild a :class:`DynamicSignLSTM` from a saved checkpoint.

    Parameters
    ----------
    path : str
        Path to the ``.pt`` checkpoint produced by :func:`save_dynamic_model`.
    map_location : str or torch.device, optional
        Passed through to :func:`torch.load`.

    Returns
    -------
    tuple
        ``(model, classes)`` ready for inference (model is in ``eval()`` mode).
    """
    # weights_only=False: the checkpoint stores the arch dict + gloss list with
    # the tensors. Explicit setting silences the deprecation warning and stays
    # working when torch flips the default to True.
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    arch = checkpoint["arch"]
    classes = list(checkpoint["classes"])

    model = DynamicSignLSTM(
        num_glosses=arch["num_glosses"],
        in_dim=arch["in_dim"],
        hidden=arch["hidden"],
        layers=arch["layers"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    # See load_static_model: move the module onto the requested device so CUDA
    # inference input matches the model's parameter device.
    if map_location is not None:
        model.to(map_location)
    model.eval()
    return model, classes


def _labels_path_for(model_path):
    """Return the JSON labels path that pairs with a given model path.

    If the model path equals the configured ``DYNAMIC_MODEL_PATH``, the
    configured ``DYNAMIC_LABELS_PATH`` is used. Otherwise a labels file is
    placed next to the model as ``<stem>_labels.json``.
    """
    try:
        import config

        if os.path.abspath(model_path) == os.path.abspath(config.DYNAMIC_MODEL_PATH):
            return config.DYNAMIC_LABELS_PATH
    except Exception:
        pass

    stem, _ = os.path.splitext(model_path)
    return stem + "_labels.json"
