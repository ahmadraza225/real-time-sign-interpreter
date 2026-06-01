"""
webcam_demo.py
==============

Standalone, real-time ASL **alphabet** demo using a webcam + OpenCV.

    >>> python -m src.inference.webcam_demo
    >>> python src/inference/webcam_demo.py

WARNING -- THIS SCRIPT REQUIRES A PHYSICAL WEBCAM.
    The university desktop (RTX 4060 Ti) has NO camera, so DO NOT run this on
    the desktop -- ``cv2.VideoCapture(0)`` will simply fail to open.  Run it on
    a LAPTOP with a built-in/USB camera instead.  All training and preprocessing
    are deliberately webcam-free; live webcam capture lives only here (and in the
    React frontend).

What it does
------------
* Opens the default camera and runs MediaPipe Hands (video mode, one hand).
* Draws the detected hand landmarks on the frame.
* Feeds the RAW landmarks into :class:`StaticPredictor` (normalization happens
  inside the predictor -- the single source of truth) and overlays the predicted
  letter and its confidence.
* Debounces predictions: a letter is only *committed* to the sentence once it
  has been stable for ~12 consecutive frames, which prevents flicker from
  spamming the output.
* Builds a sentence: ``space`` -> " ", ``del`` -> backspace, ``nothing`` ->
  ignored, every other class -> that literal character.

Keys
----
* ``q`` -- quit
* ``c`` -- clear the built sentence
* ``b`` -- backspace (remove the last character)

This file has NO Flask dependency -- it is a pure, self-contained OpenCV app.
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
# Third-party imports
# ---------------------------------------------------------------------------
import cv2
import mediapipe as mp

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
import config
from src.common.landmarks import (
    make_hands_detector,
    extract_hand_landmarks_from_image,
)
from src.inference.static_predictor import StaticPredictor

# ---------------------------------------------------------------------------
# Tunable constants for the live demo
# ---------------------------------------------------------------------------
STABLE_FRAMES_REQUIRED = 12      # frames a letter must persist before committing
MIN_COMMIT_CONFIDENCE = 0.60     # ignore commits below this confidence
CAMERA_INDEX = 0                 # default webcam

# MediaPipe drawing helpers for nicely rendering the hand skeleton.
_mp_hands = mp.solutions.hands
_mp_drawing = mp.solutions.drawing_utils
_mp_drawing_styles = mp.solutions.drawing_styles


class SentenceBuilder:
    """Accumulate committed letters into a sentence, with debouncing.

    The builder watches the *current* per-frame prediction.  When the same
    label is seen for :data:`STABLE_FRAMES_REQUIRED` consecutive frames (and is
    confident enough) it is committed exactly once.  The counter then waits for
    the label to change before it will commit again, so holding a letter does
    not type it repeatedly.
    """

    def __init__(self, stable_frames: int = STABLE_FRAMES_REQUIRED,
                 min_confidence: float = MIN_COMMIT_CONFIDENCE):
        self.stable_frames = stable_frames
        self.min_confidence = min_confidence

        self.text = ""                 # the sentence built so far
        self._candidate = None         # label currently being counted
        self._count = 0                # how many consecutive frames seen
        self._last_committed = None    # last label actually committed

    def update(self, label, confidence) -> bool:
        """Feed one frame's prediction; return ``True`` if a commit happened.

        Parameters
        ----------
        label : str or None
            The predicted class this frame, or ``None`` when no hand was found.
        confidence : float
            Confidence for ``label`` (ignored when ``label`` is ``None``).
        """
        # No hand this frame: reset the candidate streak so a new gesture must
        # build up its stability from scratch, and allow re-committing later.
        if label is None:
            self._candidate = None
            self._count = 0
            self._last_committed = None
            return False

        # Track the consecutive-frame streak for the current label.
        if label == self._candidate:
            self._count += 1
        else:
            self._candidate = label
            self._count = 1

        # Commit once the streak is long enough, confident enough, and we are
        # not re-committing the very same label we just committed.
        if (
            self._count >= self.stable_frames
            and confidence >= self.min_confidence
            and label != self._last_committed
        ):
            self._commit(label)
            self._last_committed = label
            return True

        return False

    def _commit(self, label) -> None:
        """Apply a committed label to the sentence following the class rules."""
        if label == "space":
            self.text += " "
        elif label == "del":
            self.backspace()
        elif label == "nothing":
            # Explicit "no sign" class -> contribute nothing.
            return
        else:
            # A..Z (or any other literal class) -> append the character.
            self.text += label

    def backspace(self) -> None:
        """Remove the last character of the sentence (no-op if empty)."""
        self.text = self.text[:-1]

    def clear(self) -> None:
        """Reset the sentence and debounce state."""
        self.text = ""
        self._candidate = None
        self._count = 0
        self._last_committed = None


def _draw_overlay(frame, label, confidence, sentence, progress):
    """Render prediction, confidence, the built sentence and help text.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR image to draw onto (modified in place).
    label : str or None
        Current per-frame prediction, or ``None`` when no hand is detected.
    confidence : float
        Confidence for ``label``.
    sentence : str
        The sentence built so far.
    progress : float
        0..1 fraction of the way to committing the current candidate; drawn as
        a small progress bar so the user can see the debounce filling up.
    """
    h, w = frame.shape[:2]

    # --- Top banner: current prediction ---------------------------------
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), thickness=-1)
    if label is None:
        pred_text = "Prediction: (no hand)"
    else:
        pred_text = "Prediction: {}  ({:.0%})".format(label, confidence)
    cv2.putText(
        frame, pred_text, (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA,
    )

    # --- Debounce progress bar ------------------------------------------
    bar_x, bar_y, bar_w, bar_h = 10, 78, 250, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (80, 80, 80), thickness=1)
    fill = int(bar_w * max(0.0, min(1.0, progress)))
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                      (0, 200, 255), thickness=-1)

    # --- Bottom banner: the built sentence ------------------------------
    cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame, "Sentence:", (10, h - 52),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
    )
    # Show the tail of long sentences so the most recent text stays visible.
    shown = sentence[-40:] if len(sentence) > 40 else sentence
    cv2.putText(
        frame, shown, (10, h - 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
    )

    # --- Help text (top-right) ------------------------------------------
    cv2.putText(
        frame, "q:quit  c:clear  b:backspace", (w - 360, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA,
    )


def _main() -> None:
    """Run the live webcam alphabet demo."""
    print("Webcam ASL alphabet demo")
    print("-" * 40)
    print("NOTE: this requires a webcam. Do NOT run on the camera-less desktop.")

    # Load the trained static model up front so we fail fast if it is missing.
    try:
        predictor = StaticPredictor()
    except FileNotFoundError as exc:
        print("ERROR: {}".format(exc))
        return
    print("Loaded static model on device: {}".format(predictor.device))

    # Open the default camera.
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(
            "ERROR: could not open webcam at index {}. "
            "This machine may not have a camera.".format(CAMERA_INDEX)
        )
        return

    # MediaPipe Hands in video (streaming) mode, single hand.
    hands = make_hands_detector(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    builder = SentenceBuilder()
    window_name = "Sign Interpreter - Webcam Demo"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARNING: failed to read a frame; stopping.")
                break

            # Mirror the frame so it acts like a mirror (more natural to use).
            frame = cv2.flip(frame, 1)

            # MediaPipe wants RGB; OpenCV gives BGR.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            label = None
            confidence = 0.0

            if results.multi_hand_landmarks:
                # Draw the skeleton for the first detected hand.
                hand_landmarks = results.multi_hand_landmarks[0]
                _mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    _mp_hands.HAND_CONNECTIONS,
                    _mp_drawing_styles.get_default_hand_landmarks_style(),
                    _mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Extract RAW landmarks (the helper takes the FIRST hand) and
                # predict.  Normalization happens inside the predictor.
                raw = extract_hand_landmarks_from_image(frame, hands)
                if raw is not None:
                    result = predictor.predict(raw)
                    label = result["prediction"]
                    confidence = result["confidence"]

            # Update the debounced sentence builder.
            builder.update(label, confidence)

            # Compute debounce progress for the on-screen bar.
            progress = (
                min(1.0, builder._count / float(builder.stable_frames))
                if label is not None else 0.0
            )

            _draw_overlay(frame, label, confidence, builder.text, progress)
            cv2.imshow(window_name, frame)

            # Handle keys.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                builder.clear()
            elif key == ord("b"):
                builder.backspace()
    finally:
        # Always release resources, even on error / Ctrl-C.
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("\nFinal sentence: {!r}".format(builder.text))


if __name__ == "__main__":
    _main()
