"""
backend/model_manager.py
========================

Central model manager for the Real-Time Sign Language Interpreter backend.

This module owns the lifecycle of the two trained models used by the
project:

  * the STATIC model (an MLP that classifies a single video frame's
    hand landmarks into ASL alphabet letters / control tokens), and
  * the DYNAMIC model (an LSTM that classifies a short sequence of
    two-hand landmark frames into word-level glosses).

The actual neural networks, normalization, and inference logic live in
``src/inference`` (the ``StaticPredictor`` and ``DynamicPredictor``
classes).  ``ModelManager`` simply wraps those predictors so that the
Flask layer (``backend/app.py``) never has to care about how a model is
built, where the weights live, or whether a model happens to be missing.

Design goals
------------
1. **Graceful degradation.**  If a model file has not been trained yet
   (for example a fresh checkout where only the static model exists),
   the corresponding predictor simply becomes *unavailable*.  The other
   predictor, the health endpoint, and the rest of the API keep working.
   We surface availability through :meth:`ModelManager.status`.

2. **Single source of truth for normalization.**  The frontend and the
   webcam demo send *raw* (un-normalized) flattened landmarks.  All
   normalization happens inside the predictors (which call into
   ``src/common/landmarks.py``).  ``ModelManager`` therefore forwards the
   raw landmarks straight through and never touches the numbers.

3. **A stable, JSON-friendly contract.**  Every ``predict_*`` method
   returns a plain ``dict`` of the exact shape the Flask endpoints (and
   the React frontend) expect::

       {
         "prediction": <str>,
         "confidence": <float>,
         "top_k": [{"label": <str>, "confidence": <float>}, ... up to 3]
       }

Run / import conventions
------------------------
This file may be imported as part of the ``backend`` package
(``python -m backend.app``) or, in principle, executed directly for a
quick smoke test.  To make ``import config`` and
``from src.inference ... import ...`` resolve in both cases, we insert the
PROJECT ROOT (the parent directory of this ``backend`` folder) onto
``sys.path`` *before* importing any project modules.  The root is derived
from ``__file__`` so the code is not tied to any one machine.
"""

import os
import sys
import logging

# --------------------------------------------------------------------------- #
# sys.path bootstrap
# --------------------------------------------------------------------------- #
# This file lives at  <PROJECT_ROOT>/backend/model_manager.py
# so the project root is the *parent* of the directory containing this file.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Now that the project root is importable, pull in the shared configuration.
import config  # noqa: E402  (import after sys.path bootstrap, by design)


# Module-level logger.  The Flask app configures the root logging handler;
# here we just grab a named logger so messages are easy to attribute.
logger = logging.getLogger("backend.model_manager")


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class ModelUnavailable(Exception):
    """Raised when a prediction is requested for a model that is not loaded.

    The Flask layer catches this and translates it into an HTTP ``503
    Service Unavailable`` response, signalling that the server is healthy
    but that particular model has not been trained / loaded yet.
    """

    pass


# --------------------------------------------------------------------------- #
# Helpers for talking to the predictors
# --------------------------------------------------------------------------- #
# The predictor classes in ``src/inference`` are written by a separate part
# of the project.  To stay robust against small naming differences (and to
# document the expected contract), we resolve the prediction call through a
# short list of accepted method names.  The agreed contract is that each
# predictor exposes a method that accepts the raw landmarks/sequence and
# returns the standard result dict, but we accept a few common synonyms so a
# minor naming choice on the other side does not break the whole backend.
_STATIC_PREDICT_METHODS = ("predict", "predict_static", "predict_frame", "__call__")
_DYNAMIC_PREDICT_METHODS = ("predict", "predict_dynamic", "predict_sequence", "__call__")


def _resolve_predict_method(predictor, candidate_names):
    """Return the first callable attribute on *predictor* whose name appears
    in *candidate_names*.

    Parameters
    ----------
    predictor : object
        A predictor instance (``StaticPredictor`` or ``DynamicPredictor``).
    candidate_names : tuple[str, ...]
        Ordered method names to try.  The first one that exists and is
        callable wins.

    Returns
    -------
    callable
        The bound method to use for prediction.

    Raises
    ------
    AttributeError
        If none of the candidate names resolve to a callable.
    """
    for name in candidate_names:
        method = getattr(predictor, name, None)
        if callable(method):
            return method
    raise AttributeError(
        "Predictor {!r} exposes none of the expected prediction methods {}".format(
            type(predictor).__name__, candidate_names
        )
    )


def _coerce_result(raw):
    """Normalize whatever a predictor returns into the API result dict.

    The predictors are expected to return a dict already shaped like::

        {"prediction": str, "confidence": float, "top_k": [...]}

    But to be defensive we also accept a ``(label, confidence)`` tuple or a
    bare label string and synthesize the rest.  This keeps the backend
    working even if a predictor returns a slightly simpler structure, while
    still guaranteeing the exact contract to the frontend.

    Parameters
    ----------
    raw : dict | tuple | list | str
        The predictor's return value.

    Returns
    -------
    dict
        ``{"prediction": str, "confidence": float,
           "top_k": [{"label": str, "confidence": float}, ...]}``
    """
    # Case 1: already a dict -> sanitize / fill in missing pieces.
    if isinstance(raw, dict):
        prediction = raw.get("prediction")
        confidence = raw.get("confidence", 0.0)
        top_k = raw.get("top_k")

        # Build top_k from the head prediction if the predictor omitted it.
        if not top_k:
            top_k = [{"label": prediction, "confidence": float(confidence)}]

        # Coerce every entry to the exact {label, confidence(float)} shape and
        # cap at the top 3, matching the API contract.
        clean_top_k = []
        for entry in top_k[:3]:
            if isinstance(entry, dict):
                clean_top_k.append(
                    {
                        "label": entry.get("label"),
                        "confidence": float(entry.get("confidence", 0.0)),
                    }
                )
            elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
                clean_top_k.append(
                    {"label": entry[0], "confidence": float(entry[1])}
                )
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "top_k": clean_top_k,
        }

    # Case 2: (label, confidence) pair.
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        label, confidence = raw[0], float(raw[1])
        return {
            "prediction": label,
            "confidence": confidence,
            "top_k": [{"label": label, "confidence": confidence}],
        }

    # Case 3: bare label.
    return {
        "prediction": raw,
        "confidence": 1.0,
        "top_k": [{"label": raw, "confidence": 1.0}],
    }


# --------------------------------------------------------------------------- #
# ModelManager
# --------------------------------------------------------------------------- #
class ModelManager:
    """Loads and exposes the static and dynamic sign-language predictors.

    A single instance of this class is created by the Flask app at startup.
    Each predictor is loaded inside its own ``try/except`` so that a missing
    or corrupt model file disables only that one model and logs a warning,
    rather than crashing the whole backend.

    Attributes
    ----------
    static_predictor : object | None
        The loaded ``StaticPredictor`` instance, or ``None`` if unavailable.
    dynamic_predictor : object | None
        The loaded ``DynamicPredictor`` instance, or ``None`` if unavailable.
    """

    def __init__(self):
        # Predictor handles; remain ``None`` until successfully loaded.
        self.static_predictor = None
        self.dynamic_predictor = None

        # Cached, resolved prediction callables (avoids re-resolving on every
        # request).  Filled in alongside the predictors below.
        self._static_predict = None
        self._dynamic_predict = None

        # Human-readable device string ("cuda" / "cpu") for the health report.
        # config.get_device() returns a torch.device; str() gives e.g. "cuda".
        try:
            self._device_str = str(config.get_device())
        except Exception:  # pragma: no cover - extremely defensive fallback
            # If torch is unavailable for some reason we still want a string.
            self._device_str = getattr(config, "DEVICE", "cpu")

        logger.info("Initializing ModelManager (device=%s)...", self._device_str)

        # Load each predictor independently and defensively.
        self._load_static_predictor()
        self._load_dynamic_predictor()

        logger.info(
            "ModelManager ready -> static=%s, dynamic=%s",
            self.static_predictor is not None,
            self.dynamic_predictor is not None,
        )

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load_static_predictor(self):
        """Attempt to load the static (alphabet) predictor.

        Any failure (model file missing, bad weights, import error) is
        caught and logged as a warning; ``self.static_predictor`` simply
        stays ``None`` and the static endpoints will return ``503``.
        """
        try:
            # Imported lazily so that a problem importing the inference package
            # (e.g. torch not installed) does not prevent the dynamic model or
            # the health endpoint from working.
            from src.inference.static_predictor import StaticPredictor

            self.static_predictor = StaticPredictor()
            self._static_predict = _resolve_predict_method(
                self.static_predictor, _STATIC_PREDICT_METHODS
            )
            logger.info(
                "Static model loaded (%d classes).", len(self._safe_labels(self.static_predictor))
            )
        except FileNotFoundError as exc:
            logger.warning(
                "Static model file not found; static predictions disabled. (%s)", exc
            )
            self.static_predictor = None
            self._static_predict = None
        except Exception as exc:  # noqa: BLE001 - we genuinely want any failure
            logger.warning(
                "Could not load static predictor; static predictions disabled. (%s)",
                exc,
            )
            self.static_predictor = None
            self._static_predict = None

    def _load_dynamic_predictor(self):
        """Attempt to load the dynamic (word-level gloss) predictor.

        Same graceful-degradation contract as :meth:`_load_static_predictor`.
        """
        try:
            from src.inference.dynamic_predictor import DynamicPredictor

            self.dynamic_predictor = DynamicPredictor()
            self._dynamic_predict = _resolve_predict_method(
                self.dynamic_predictor, _DYNAMIC_PREDICT_METHODS
            )
            logger.info(
                "Dynamic model loaded (%d glosses).",
                len(self._safe_labels(self.dynamic_predictor)),
            )
        except FileNotFoundError as exc:
            logger.warning(
                "Dynamic model file not found; dynamic predictions disabled. (%s)",
                exc,
            )
            self.dynamic_predictor = None
            self._dynamic_predict = None
        except Exception as exc:  # noqa: BLE001 - we genuinely want any failure
            logger.warning(
                "Could not load dynamic predictor; dynamic predictions disabled. (%s)",
                exc,
            )
            self.dynamic_predictor = None
            self._dynamic_predict = None

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe_labels(predictor):
        """Best-effort extraction of a predictor's class/gloss label list.

        Predictors are expected to expose their ordered label list via a
        ``labels`` or ``classes`` attribute.  This helper tries the common
        names and always returns a list (empty if nothing is found), so the
        backend never crashes just trying to report labels.

        Parameters
        ----------
        predictor : object | None

        Returns
        -------
        list
            The ordered list of label strings, or ``[]``.
        """
        if predictor is None:
            return []
        for attr in ("labels", "classes", "class_names", "glosses"):
            value = getattr(predictor, attr, None)
            if value is not None:
                try:
                    return list(value)
                except TypeError:
                    # Not iterable for some reason; ignore and keep trying.
                    continue
        return []

    def static_available(self):
        """Return ``True`` if the static model is loaded and ready."""
        return self.static_predictor is not None

    def dynamic_available(self):
        """Return ``True`` if the dynamic model is loaded and ready."""
        return self.dynamic_predictor is not None

    # ------------------------------------------------------------------ #
    # Public API used by the Flask layer
    # ------------------------------------------------------------------ #
    def status(self):
        """Return a small dict describing the manager's health.

        Matches the ``/api/health`` contract::

            {"device": <str>, "models": {"static": <bool>, "dynamic": <bool>}}
        """
        return {
            "device": self._device_str,
            "models": {
                "static": self.static_available(),
                "dynamic": self.dynamic_available(),
            },
        }

    def labels(self):
        """Return the available class / gloss labels for both models.

        Matches the ``/api/labels`` contract::

            {"static": [...class strings...], "dynamic": [...gloss strings...]}

        An unavailable model contributes an empty list.
        """
        return {
            "static": self._safe_labels(self.static_predictor),
            "dynamic": self._safe_labels(self.dynamic_predictor),
        }

    def predict_static(self, landmarks):
        """Classify a single frame of *raw* one-hand landmarks.

        Parameters
        ----------
        landmarks : list[float] | numpy.ndarray
            A flat sequence of ``config.STATIC_FEATURE_DIM`` (63) raw,
            un-normalized landmark values: ``[x, y, z] * 21``.  Normalization
            happens inside the predictor, not here.

        Returns
        -------
        dict
            ``{"prediction", "confidence", "top_k"}`` as described in the
            module docstring.

        Raises
        ------
        ModelUnavailable
            If the static model has not been loaded.
        """
        if not self.static_available():
            raise ModelUnavailable("Static model is not loaded.")
        raw_result = self._static_predict(landmarks)
        return _coerce_result(raw_result)

    def predict_dynamic(self, sequence):
        """Classify a *raw* two-hand landmark sequence into a word gloss.

        Parameters
        ----------
        sequence : list[list[float]] | numpy.ndarray
            A variable-length sequence of frames, each frame being
            ``config.DYNAMIC_FEATURE_DIM`` (126) raw, un-normalized landmark
            values (two hands: left block then right block).  Padding /
            truncation to ``config.SEQUENCE_LENGTH`` and normalization are
            handled inside the predictor.

        Returns
        -------
        dict
            ``{"prediction", "confidence", "top_k"}``.

        Raises
        ------
        ModelUnavailable
            If the dynamic model has not been loaded.
        """
        if not self.dynamic_available():
            raise ModelUnavailable("Dynamic model is not loaded.")
        raw_result = self._dynamic_predict(sequence)
        return _coerce_result(raw_result)


# --------------------------------------------------------------------------- #
# Manual smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Running this module directly gives a quick, dependency-light check that
    # the manager can be constructed and reports sensible status/labels.  It
    # does NOT require a webcam and does not perform any real inference unless
    # the model files happen to exist.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    mgr = ModelManager()
    print("status:", mgr.status())
    labels = mgr.labels()
    print("static labels :", labels["static"][:10], "..." if len(labels["static"]) > 10 else "")
    print("dynamic labels:", labels["dynamic"][:10], "..." if len(labels["dynamic"]) > 10 else "")
