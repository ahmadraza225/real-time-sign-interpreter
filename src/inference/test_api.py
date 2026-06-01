"""
test_api.py
===========

Webcam-FREE smoke test for the Flask backend.

This script lets you verify the running backend from the camera-less desktop.
It uses :mod:`requests` to hit the documented endpoints and prints the
responses:

* ``GET  /api/health``           -- liveness + which models are loaded
* ``GET  /api/labels``           -- static class list + dynamic gloss list
* ``POST /api/predict/static``   -- classify a single 63-value landmark vector

Usage
-----
Start the backend first (in another terminal)::

    python -m backend.app

Then run this script from the project root::

    python -m src.inference.test_api
    python src/inference/test_api.py --url http://localhost:5000

The 63-value vector is generated DETERMINISTICALLY from a fixed NumPy seed, so
it is SYNTHETIC: the resulting prediction is meaningless as sign language, but a
``200`` response with a well-formed body proves the request/normalize/inference
pipeline works end-to-end.
"""

# ---------------------------------------------------------------------------
# sys.path bootstrap (see static_predictor.py for the rationale)
# ---------------------------------------------------------------------------
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Standard / third-party imports
# ---------------------------------------------------------------------------
import argparse
import json

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
import config

# Default backend base URL (matches the Flask API contract: localhost:5000).
DEFAULT_URL = "http://localhost:5000"

# Network timeout (seconds) so the script never hangs forever on a dead server.
REQUEST_TIMEOUT = 10


def _pretty(obj) -> str:
    """Return a nicely indented JSON string for printing API responses."""
    return json.dumps(obj, indent=2, ensure_ascii=False)


def make_sample_vector() -> list:
    """Build a deterministic SYNTHETIC 63-value RAW landmark vector.

    A fixed seed makes the request reproducible across runs.  The values do not
    correspond to any real hand pose, so the prediction is meaningless; this is
    purely to exercise the backend on a machine without a camera.

    Returns
    -------
    list of float
        :data:`config.STATIC_FEATURE_DIM` (63) Python floats, JSON-serializable.
    """
    rng = np.random.default_rng(seed=42)
    vec = rng.random(config.STATIC_FEATURE_DIM).astype(np.float32)
    # Convert to plain Python floats so ``json``/``requests`` can serialize them.
    return [float(v) for v in vec.tolist()]


def check_health(base_url: str) -> None:
    """GET /api/health and print the result."""
    url = base_url.rstrip("/") + "/api/health"
    print("\n[GET] {}".format(url))
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        print("  status: {}".format(resp.status_code))
        print(_indent(_pretty(resp.json())))
    except requests.exceptions.RequestException as exc:
        print("  ERROR: could not reach backend: {}".format(exc))
        print("  Is the server running?  Start it with: python -m backend.app")


def check_labels(base_url: str) -> None:
    """GET /api/labels and print the result."""
    url = base_url.rstrip("/") + "/api/labels"
    print("\n[GET] {}".format(url))
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        print("  status: {}".format(resp.status_code))
        data = resp.json()
        static_labels = data.get("static", [])
        dynamic_labels = data.get("dynamic", [])
        print("  static : {} classes -> {}".format(len(static_labels), static_labels))
        print("  dynamic: {} glosses -> {}".format(len(dynamic_labels), dynamic_labels))
    except requests.exceptions.RequestException as exc:
        print("  ERROR: could not reach backend: {}".format(exc))
    except ValueError:
        print("  ERROR: response was not valid JSON.")


def check_predict_static(base_url: str) -> None:
    """POST /api/predict/static with a synthetic vector and print the result."""
    url = base_url.rstrip("/") + "/api/predict/static"
    payload = {"landmarks": make_sample_vector()}
    print("\n[POST] {}".format(url))
    print("  body: {{\"landmarks\": [{} synthetic floats]}}".format(
        len(payload["landmarks"])
    ))
    print("  (synthetic input -> prediction is meaningless; this only proves "
          "the pipeline works)")
    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        print("  status: {}".format(resp.status_code))
        try:
            print(_indent(_pretty(resp.json())))
        except ValueError:
            print("  (non-JSON response): {}".format(resp.text))
    except requests.exceptions.RequestException as exc:
        print("  ERROR: could not reach backend: {}".format(exc))


def _indent(text: str, spaces: int = 4) -> str:
    """Indent every line of ``text`` by ``spaces`` for tidy nested printing."""
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Webcam-free smoke test for the Sign Interpreter Flask backend."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Base URL of the backend (default: {}).".format(DEFAULT_URL),
    )
    args = parser.parse_args()

    print("Sign Interpreter -- backend smoke test")
    print("=" * 50)
    print("Target backend: {}".format(args.url))

    check_health(args.url)
    check_labels(args.url)
    check_predict_static(args.url)

    print("\nDone.")


if __name__ == "__main__":
    _main()
