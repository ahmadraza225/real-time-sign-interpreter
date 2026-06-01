"""
landmarks.py
============

SINGLE SOURCE OF TRUTH for hand-landmark extraction (MediaPipe) and for the
geometric normalization that turns RAW landmark coordinates into the
translation- and scale-invariant feature vectors the neural networks consume.

Design contract (shared across the whole project)
--------------------------------------------------
* MediaPipe Hands produces, per detected hand, 21 landmarks, each with an
  ``(x, y, z)`` coordinate.  Flattened that is ``21 * 3 = 63`` floats for one
  hand and ``126`` floats for two hands.
* The **frontend** (React + @mediapipe/hands) and the desktop
  ``webcam_demo.py`` always send **RAW** (un-normalized) flattened landmarks.
* Normalization is performed in exactly ONE place -- this module -- and is
  applied server-side (in the Flask predictors) and inside the training
  ``Dataset`` objects.  Therefore the models ALWAYS see NORMALIZED input and
  there is no risk of train/inference skew caused by two different
  normalization implementations.

Why normalize?
--------------
Raw MediaPipe coordinates depend on *where* the hand is in the image and on
*how big* it appears (distance to the camera).  Those nuisance factors carry no
information about *which sign* is being made.  We remove them by:

1. **Translation invariance** -- subtract the wrist landmark (index 0) from
   every landmark, so the wrist sits at the origin.
2. **Scale invariance** -- divide every coordinate by the largest Euclidean
   distance from the origin (i.e. the size of the hand), so a hand close to the
   camera and the same hand far away produce identical features.

All public functions return ``numpy.float32`` arrays so the data feeds cleanly
into PyTorch tensors.
"""

import numpy as np

# MediaPipe is only required for the *extraction* helpers.  Importing it lazily
# would complicate the module, so we import it at module load time; every script
# that uses these helpers already depends on mediapipe being installed.
import mediapipe as mp

# Pull the shared dimensional constants from the project-root config module so
# that there is no magic-number duplication anywhere in the codebase.
import config


# ---------------------------------------------------------------------------
# Normalization helpers (the geometric "single source of truth")
# ---------------------------------------------------------------------------
def normalize_hand(arr63):
    """Normalize a single hand's RAW 63-vector to be translation/scale invariant.

    Steps
    -----
    1. Reshape the flat ``(63,)`` vector into ``(21, 3)`` landmark points.
    2. Subtract the wrist point (landmark index 0) from every point so the
       wrist is at the origin -> **translation invariance**.
    3. Divide all coordinates by the maximum Euclidean distance of any point
       from the origin -> **scale invariance**.  If that maximum distance is
       zero (degenerate / all-zero input) we guard the division by using
       ``1.0`` instead, which leaves the (already zero) data untouched.
    4. Flatten back to a ``(63,)`` ``float32`` vector.

    Parameters
    ----------
    arr63 : array-like of shape (63,)
        RAW flattened landmarks ``[x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]``.

    Returns
    -------
    numpy.ndarray, dtype float32, shape (63,)
        The normalized landmark vector.
    """
    # Work on a float32 copy so we never mutate the caller's array.
    points = np.asarray(arr63, dtype=np.float32).reshape(config.NUM_HAND_LANDMARKS,
                                                         config.LANDMARK_DIMS)

    # 1) Translation invariance: move the wrist (point 0) to the origin.
    wrist = points[0].copy()
    points = points - wrist

    # 2) Scale invariance: divide by the size of the hand (max distance from
    #    the origin over all 21 points).  np.linalg.norm with axis=1 gives the
    #    Euclidean distance of each point; we take the largest.
    distances = np.linalg.norm(points, axis=1)
    max_distance = float(distances.max())
    if max_distance <= 0.0:
        # Degenerate hand (e.g. all zeros) -- avoid division by zero.
        max_distance = 1.0
    points = points / max_distance

    # 3) Flatten back to a 1-D feature vector.
    return points.reshape(-1).astype(np.float32)


def normalize_two_hands(arr126):
    """Normalize a RAW 126-vector (two hands) hand-by-hand.

    The 126-vector is laid out as ``[left_hand_63 | right_hand_63]``.  Each
    63-block is normalized **independently** with :func:`normalize_hand` so that
    the two hands keep their own translation/scale frames.  A hand that was not
    detected is encoded as an all-zero 63-block; such a block is left as zeros
    (we skip normalizing it, since there is nothing to normalize and we want the
    "missing hand" signal to survive as exact zeros).

    Parameters
    ----------
    arr126 : array-like of shape (126,)
        RAW flattened two-hand landmarks ``[left(63) | right(63)]``.

    Returns
    -------
    numpy.ndarray, dtype float32, shape (126,)
        The normalized two-hand feature vector, left block first, right second.
    """
    arr = np.asarray(arr126, dtype=np.float32).reshape(-1)

    left_block = arr[:config.STATIC_FEATURE_DIM]
    right_block = arr[config.STATIC_FEATURE_DIM:2 * config.STATIC_FEATURE_DIM]

    # Normalize each block independently, but leave all-zero (missing) hands as
    # zeros so the "no hand here" signal is preserved exactly.
    if np.any(left_block):
        left_norm = normalize_hand(left_block)
    else:
        left_norm = left_block.astype(np.float32)

    if np.any(right_block):
        right_norm = normalize_hand(right_block)
    else:
        right_norm = right_block.astype(np.float32)

    return np.concatenate([left_norm, right_norm]).astype(np.float32)


def normalize_sequence(seq_T_126):
    """Normalize every frame of a two-hand sequence.

    Parameters
    ----------
    seq_T_126 : array-like of shape (T, 126)
        A temporal sequence of RAW two-hand landmark vectors.

    Returns
    -------
    numpy.ndarray, dtype float32, shape (T, 126)
        The sequence with :func:`normalize_two_hands` applied to each frame.
    """
    seq = np.asarray(seq_T_126, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != config.DYNAMIC_FEATURE_DIM:
        raise ValueError(
            "normalize_sequence expects shape (T, %d), got %r"
            % (config.DYNAMIC_FEATURE_DIM, seq.shape)
        )

    normalized = np.empty_like(seq, dtype=np.float32)
    for t in range(seq.shape[0]):
        normalized[t] = normalize_two_hands(seq[t])
    return normalized.astype(np.float32)


# ---------------------------------------------------------------------------
# MediaPipe Hands helpers
# ---------------------------------------------------------------------------
def make_hands_detector(static_image_mode, max_num_hands,
                        min_detection_confidence=0.5):
    """Create and return a configured MediaPipe ``Hands`` detector.

    Parameters
    ----------
    static_image_mode : bool
        ``True`` for independent still images (no tracking between calls; used
        for the ASL alphabet image dataset).  ``False`` for video streams where
        MediaPipe tracks hands across frames (used for WLASL videos / webcam).
    max_num_hands : int
        Maximum number of hands to detect (1 for static alphabet, 2 for
        dynamic word signs).
    min_detection_confidence : float, optional
        Minimum confidence for the initial hand detection (default 0.5).

    Returns
    -------
    mediapipe.python.solutions.hands.Hands
        A ready-to-use detector.  Caller is responsible for calling
        ``.close()`` when finished (or using it as a context manager).
    """
    return mp.solutions.hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
    )


def extract_hand_landmarks_from_image(bgr_image, hands):
    """Extract RAW 63-vector landmarks for the FIRST detected hand in an image.

    Intended for single-hand static images (the ASL alphabet).  The input is a
    standard OpenCV BGR image; MediaPipe expects RGB, so we convert first.

    Parameters
    ----------
    bgr_image : numpy.ndarray, shape (H, W, 3), dtype uint8
        An OpenCV BGR image.
    hands : mediapipe Hands
        A detector created with :func:`make_hands_detector`
        (typically ``static_image_mode=True, max_num_hands=1``).

    Returns
    -------
    numpy.ndarray, dtype float32, shape (63,)
        RAW flattened landmarks ``[x, y, z] * 21`` (image-normalized
        coordinates straight from MediaPipe), OR
    None
        if no hand is detected in the image.
    """
    # MediaPipe works in RGB; OpenCV loads BGR.  Convert without an OpenCV
    # dependency by reversing the channel axis (cheaper than importing cv2 here
    # and equivalent to cv2.cvtColor(..., COLOR_BGR2RGB)).
    rgb_image = bgr_image[:, :, ::-1]

    # MediaPipe requires a contiguous array; the reversed view is not, so make
    # it contiguous explicitly.
    rgb_image = np.ascontiguousarray(rgb_image)

    results = hands.process(rgb_image)
    if not results.multi_hand_landmarks:
        return None

    # Take the FIRST detected hand.
    hand = results.multi_hand_landmarks[0]
    coords = np.empty(config.STATIC_FEATURE_DIM, dtype=np.float32)
    for i, lm in enumerate(hand.landmark):
        coords[i * 3 + 0] = lm.x
        coords[i * 3 + 1] = lm.y
        coords[i * 3 + 2] = lm.z
    return coords


def extract_two_hands_from_frame(rgb_frame, hands):
    """Extract a RAW 126-vector for two hands from an already-RGB frame.

    The two 63-blocks are ordered by MediaPipe's handedness label: **Left hand
    first, Right hand second**.  A hand that is not present in the frame is
    filled with a 63-length block of zeros (the "missing hand" encoding that the
    normalization layer preserves).

    Parameters
    ----------
    rgb_frame : numpy.ndarray, shape (H, W, 3), dtype uint8
        An RGB frame (callers that have BGR frames must convert first; video
        preprocessing converts each OpenCV frame before calling this).
    hands : mediapipe Hands
        A detector created with :func:`make_hands_detector`
        (typically ``static_image_mode=False, max_num_hands=2``).

    Returns
    -------
    numpy.ndarray, dtype float32, shape (126,)
        RAW flattened landmarks ``[left(63) | right(63)]``.
    """
    frame = np.ascontiguousarray(rgb_frame)
    results = hands.process(frame)

    # Start with both hands absent (all zeros).
    left = np.zeros(config.STATIC_FEATURE_DIM, dtype=np.float32)
    right = np.zeros(config.STATIC_FEATURE_DIM, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            # handedness.classification[0].label is "Left" or "Right".
            label = handedness.classification[0].label
            coords = np.empty(config.STATIC_FEATURE_DIM, dtype=np.float32)
            for i, lm in enumerate(hand_landmarks.landmark):
                coords[i * 3 + 0] = lm.x
                coords[i * 3 + 1] = lm.y
                coords[i * 3 + 2] = lm.z

            if label == "Left":
                left = coords
            else:  # "Right" (or anything non-Left) goes in the right block.
                right = coords

    return np.concatenate([left, right]).astype(np.float32)
