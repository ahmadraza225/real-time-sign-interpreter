"""
backend/app.py
==============

Flask backend for the Real-Time Sign Language Interpreter.

This is the HTTP (and optional WebSocket) layer that the React frontend and
the ``test_api.py`` client talk to.  It is intentionally thin: all of the
machine-learning logic lives behind a single :class:`ModelManager`
(``backend/model_manager.py``), and all landmark normalization lives in
``src/common/landmarks.py``.  The job of this file is purely:

  * accept and validate incoming JSON requests,
  * forward raw landmarks to the model manager,
  * translate the result (or a missing-model condition) into the exact
    HTTP responses described in the project's API contract, and
  * optionally expose the same static prediction over Socket.IO for
    low-latency streaming.

API contract (implemented exactly here)
----------------------------------------
GET  /api/health
        -> {"status": "ok",
            "device": <str>,
            "models": {"static": <bool>, "dynamic": <bool>}}

GET  /api/labels
        -> {"static": [...class strings...],
            "dynamic": [...gloss strings...]}

POST /api/predict/static
        body {"landmarks": [63 RAW floats]}
        -> {"prediction": <str>, "confidence": <float>,
            "top_k": [{"label": <str>, "confidence": <float>}, ... up to 3]}
        400 on invalid input ; 503 if the static model is not loaded.

POST /api/predict/dynamic
        body {"sequence": [[126 RAW floats], ...variable length]}
        -> {"prediction": <gloss>, "confidence": <float>, "top_k": [...]}
        The sequence is padded / truncated to config.SEQUENCE_LENGTH inside
        the predictor.  400 on invalid input ; 503 if the dynamic model is
        not loaded.

Optional Socket.IO
        client emits "landmarks" {landmarks: [63]}
        -> server emits "prediction" {...same shape as /api/predict/static...}

CORS
        Cross-origin requests are allowed from http://localhost:3000 (the
        Create-React-App dev server) and, for convenience during local
        development, from any origin ("*").

Running (PowerShell, from the project root c:/SignInterpreter)
        python -m backend.app
    or
        python backend/app.py

The server listens on 0.0.0.0:5000.  It needs NO webcam: it only ever
receives already-extracted landmark numbers from a client.  The React app
runs separately on port 3000 and calls these endpoints.
"""

import os
import sys
import logging

# --------------------------------------------------------------------------- #
# sys.path bootstrap (must run before importing project modules)
# --------------------------------------------------------------------------- #
# This file lives at  <PROJECT_ROOT>/backend/app.py
# so the project root is the parent of this file's directory.  Inserting it on
# sys.path makes both ``import config`` and ``from backend.model_manager import
# ...`` resolve whether we are launched as ``python -m backend.app`` or as
# ``python backend/app.py``.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402

from flask import Flask, request, jsonify  # noqa: E402
from flask_cors import CORS  # noqa: E402

# The model manager and its dedicated exception.  Imported via the package
# name so the same module object is shared no matter how the app is launched.
from backend.model_manager import ModelManager, ModelUnavailable  # noqa: E402


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backend.app")


# --------------------------------------------------------------------------- #
# Optional Socket.IO support
# --------------------------------------------------------------------------- #
# flask-socketio is optional.  If it (or its async dependencies) is not
# installed, the app must still run as a plain Flask/WSGI server.  We therefore
# wrap the import in try/except and fall back gracefully.
try:
    from flask_socketio import SocketIO  # type: ignore

    _SOCKETIO_AVAILABLE = True
except Exception as _exc:  # noqa: BLE001 - any import problem disables Socket.IO
    SocketIO = None  # type: ignore
    _SOCKETIO_AVAILABLE = False
    logger.warning(
        "flask-socketio not available; running without WebSocket support. (%s)",
        _exc,
    )


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #
def create_app():
    """Build and configure the Flask application and the ModelManager.

    Returns
    -------
    (flask.Flask, ModelManager, SocketIO | None)
        The configured app, the single shared model manager, and the
        Socket.IO server instance (or ``None`` if Socket.IO is unavailable).
    """
    app = Flask(__name__)

    # Allow the React dev server and, for local convenience, any origin.
    # Listing localhost:3000 explicitly documents the intended consumer; "*"
    # keeps things friction-free while developing on a single machine.
    CORS(
        app,
        resources={
            r"/api/*": {"origins": ["http://localhost:3000", "*"]},
        },
    )

    # Exactly one ModelManager for the whole process.  Constructing it loads
    # the models (or marks them unavailable) and logs what happened.
    manager = ModelManager()

    # ------------------------------------------------------------------ #
    # Input-validation helpers
    # ------------------------------------------------------------------ #
    def _bad_request(message):
        """Return a JSON 400 response with a helpful error message."""
        return jsonify({"error": message}), 400

    def _service_unavailable(message):
        """Return a JSON 503 response (model not loaded)."""
        return jsonify({"error": message}), 503

    def _is_number(value):
        """True if *value* is an int/float (but not a bool)."""
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _validate_flat_landmarks(values, expected_len):
        """Validate a flat list of numeric landmark values.

        Parameters
        ----------
        values : Any
            The candidate value from the request body.
        expected_len : int
            Required length (63 for static, 126 per dynamic frame).

        Returns
        -------
        str | None
            An error message if invalid, otherwise ``None``.
        """
        if not isinstance(values, (list, tuple)):
            return "'landmarks' must be a list of numbers."
        if len(values) != expected_len:
            return "Expected {} landmark values, got {}.".format(
                expected_len, len(values)
            )
        if not all(_is_number(v) for v in values):
            return "All landmark values must be numbers."
        return None

    # ------------------------------------------------------------------ #
    # Routes
    # ------------------------------------------------------------------ #
    @app.route("/api/health", methods=["GET"])
    def health():
        """Liveness + model-availability probe.

        Always returns 200 as long as the process is up; the ``models`` flags
        tell the client which predictions are currently possible.
        """
        status = manager.status()
        return jsonify(
            {
                "status": "ok",
                "device": status["device"],
                "models": status["models"],
            }
        )

    @app.route("/api/labels", methods=["GET"])
    def labels():
        """Return the ordered label lists for both models.

        An unavailable model yields an empty list, so the frontend can render
        whatever is currently supported without special-casing missing models.
        """
        return jsonify(manager.labels())

    @app.route("/api/predict/static", methods=["POST"])
    def predict_static():
        """Classify a single frame of one-hand landmarks (ASL alphabet).

        Request body::

            {"landmarks": [63 RAW floats]}

        The 63 values are raw (un-normalized) ``[x, y, z] * 21`` landmark
        coordinates; normalization happens inside the predictor.
        """
        data = request.get_json(silent=True)
        if data is None:
            return _bad_request("Request body must be valid JSON.")
        if "landmarks" not in data:
            return _bad_request("Missing required field 'landmarks'.")

        error = _validate_flat_landmarks(
            data["landmarks"], config.STATIC_FEATURE_DIM
        )
        if error is not None:
            return _bad_request(error)

        try:
            result = manager.predict_static(data["landmarks"])
        except ModelUnavailable as exc:
            return _service_unavailable(str(exc))
        except Exception as exc:  # noqa: BLE001 - never leak a stack trace as 500-less
            logger.exception("Static prediction failed.")
            return jsonify({"error": "Internal prediction error: {}".format(exc)}), 500

        return jsonify(result)

    @app.route("/api/predict/dynamic", methods=["POST"])
    def predict_dynamic():
        """Classify a variable-length two-hand landmark sequence (word gloss).

        Request body::

            {"sequence": [[126 RAW floats], ...variable length]}

        Each inner frame holds 126 raw landmark values (two hands).  The
        sequence is padded / truncated to ``config.SEQUENCE_LENGTH`` inside the
        predictor, so any non-empty length is accepted here.
        """
        data = request.get_json(silent=True)
        if data is None:
            return _bad_request("Request body must be valid JSON.")
        if "sequence" not in data:
            return _bad_request("Missing required field 'sequence'.")

        sequence = data["sequence"]
        if not isinstance(sequence, (list, tuple)):
            return _bad_request("'sequence' must be a list of frames.")
        if len(sequence) == 0:
            return _bad_request("'sequence' must contain at least one frame.")

        # Validate every frame: each must be a list of 126 numbers.
        for index, frame in enumerate(sequence):
            error = _validate_flat_landmarks(frame, config.DYNAMIC_FEATURE_DIM)
            if error is not None:
                return _bad_request("Frame {}: {}".format(index, error))

        try:
            result = manager.predict_dynamic(sequence)
        except ModelUnavailable as exc:
            return _service_unavailable(str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Dynamic prediction failed.")
            return jsonify({"error": "Internal prediction error: {}".format(exc)}), 500

        return jsonify(result)

    @app.route("/", methods=["GET"])
    def index():
        """Tiny landing page so visiting the root in a browser is informative.

        The real UI is the React app on port 3000; this just confirms the API
        is up and points the developer at the useful endpoints.
        """
        return jsonify(
            {
                "service": "Real-Time Sign Language Interpreter API",
                "endpoints": [
                    "GET  /api/health",
                    "GET  /api/labels",
                    "POST /api/predict/static",
                    "POST /api/predict/dynamic",
                ],
                "frontend": "React dev server runs separately on http://localhost:3000",
            }
        )

    # ------------------------------------------------------------------ #
    # Optional Socket.IO wiring
    # ------------------------------------------------------------------ #
    socketio = None
    if _SOCKETIO_AVAILABLE:
        # cors_allowed_origins mirrors the HTTP CORS policy above.
        socketio = SocketIO(app, cors_allowed_origins="*")

        @socketio.on("connect")
        def _on_connect():
            """Log new Socket.IO connections (useful while debugging)."""
            logger.info("Socket.IO client connected.")

        @socketio.on("disconnect")
        def _on_disconnect():
            """Log Socket.IO disconnections."""
            logger.info("Socket.IO client disconnected.")

        @socketio.on("landmarks")
        def _on_landmarks(payload):
            """Stream a single static prediction back over the socket.

            The client emits ``"landmarks"`` with ``{"landmarks": [63 floats]}``
            and the server responds by emitting ``"prediction"`` carrying the
            same shape as the ``/api/predict/static`` HTTP response, or an
            ``"error"`` event if the input is bad / the model is unavailable.
            """
            # Defensive payload handling: clients may send a dict or a bare list.
            if isinstance(payload, dict):
                values = payload.get("landmarks")
            else:
                values = payload

            error = _validate_flat_landmarks(values, config.STATIC_FEATURE_DIM)
            if error is not None:
                socketio.emit("error", {"error": error})
                return

            try:
                result = manager.predict_static(values)
            except ModelUnavailable as exc:
                socketio.emit("error", {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("Socket.IO static prediction failed.")
                socketio.emit("error", {"error": "Internal prediction error."})
                return

            socketio.emit("prediction", result)

    return app, manager, socketio


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    """Construct the app and run the development server.

    Uses ``socketio.run`` when Socket.IO is available (so WebSocket upgrades
    work), otherwise falls back to the plain Flask development server.  Either
    way the server binds to 0.0.0.0:5000.  ``debug`` is read from the
    ``FLASK_DEBUG`` environment variable (anything in {"1","true","yes","on"}).
    """
    app, manager, socketio = create_app()

    # Report what actually loaded so the operator sees it immediately.
    status = manager.status()
    logger.info(
        "Startup complete -> device=%s, static_model=%s, dynamic_model=%s",
        status["device"],
        status["models"]["static"],
        status["models"]["dynamic"],
    )
    if not status["models"]["static"]:
        logger.warning(
            "Static model not loaded: /api/predict/static will return 503 until "
            "models/static_mlp.pt exists (run training first)."
        )
    if not status["models"]["dynamic"]:
        logger.warning(
            "Dynamic model not loaded: /api/predict/dynamic will return 503 until "
            "models/dynamic_lstm.pt exists (run training first)."
        )

    debug_env = os.environ.get("FLASK_DEBUG", "").strip().lower()
    debug = debug_env in ("1", "true", "yes", "on")

    host = "0.0.0.0"
    port = 5000

    if socketio is not None:
        logger.info("Running with Socket.IO on %s:%d (debug=%s)", host, port, debug)
        # use_reloader is tied to debug; the reloader can double-load models, so
        # we keep it off unless explicitly debugging.
        #
        # ``allow_unsafe_werkzeug`` is required by newer flask-socketio releases
        # to permit serving over the Werkzeug dev server, but the keyword does
        # not exist on older versions (where it raises TypeError).  We try with
        # it first and transparently fall back without it for compatibility.
        try:
            socketio.run(
                app,
                host=host,
                port=port,
                debug=debug,
                use_reloader=debug,
                allow_unsafe_werkzeug=True,
            )
        except TypeError:
            socketio.run(
                app,
                host=host,
                port=port,
                debug=debug,
                use_reloader=debug,
            )
    else:
        logger.info(
            "Running with plain Flask server on %s:%d (debug=%s)", host, port, debug
        )
        app.run(host=host, port=port, debug=debug, use_reloader=debug)


if __name__ == "__main__":
    main()
