"""
preprocess_wlasl.py
===================

Convert the **word-level sign-language video** dataset (a restructured WLASL
set) into a ``.npz`` of fixed-length RAW two-hand landmark sequences suitable
for training the dynamic LSTM classifier.

Dataset layout (on disk)
------------------------
``config.WLASL_ROOT/<gloss>/<video_id>.mp4`` where the **folder name is the
gloss (the label)**.  There is NO JSON metadata file; labels are derived purely
from folder names.

Gloss selection
---------------
By default we pick the **top-K glosses ranked by number of videos**
(``K = config.WLASL_DEFAULT_NUM_GLOSSES``, default 20), because the dataset is
demo-scale with only a handful of clips per gloss.  An explicit comma-separated
``--glosses`` list overrides the automatic top-K selection.

Per-video processing
--------------------
1. Read every frame of the video with OpenCV.
2. Uniformly sample (or pad) the frames to ``config.SEQUENCE_LENGTH`` frames.
3. For each sampled frame run MediaPipe Hands
   (``static_image_mode=False``, ``max_num_hands=2``) and extract a RAW
   126-vector ``[left(63) | right(63)]``.
4. Assemble a ``(SEQUENCE_LENGTH, 126)`` array via
   :func:`pad_or_truncate_sequence` (guarantees exact length).

Output ``.npz``
---------------
* ``X``      -- float32 array of shape ``(N, 30, 126)`` of RAW landmarks,
* ``y``      -- int64 array of gloss indices,
* ``glosses``-- object array of gloss-name strings (index -> name).

Stored data is RAW; normalization happens later (training Dataset / inference),
so the project has a single normalization implementation.

This script NEVER touches a webcam -- it reads ``.mp4`` files from disk only.

Run from the project root, e.g.::

    python -m src.preprocessing.preprocess_wlasl
    python -m src.preprocessing.preprocess_wlasl --num-glosses 30
    python -m src.preprocessing.preprocess_wlasl --glosses hello,thanks,please
"""

# ---------------------------------------------------------------------------
# sys.path bootstrap: insert the PROJECT ROOT (two levels up) so that
# "import config" and "from src.common... import ..." resolve regardless of how
# this script is launched.  Computed from __file__, never hardcoded.
# ---------------------------------------------------------------------------
import os
import sys

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse

import numpy as np
import cv2
from tqdm import tqdm

import config
from src.common.landmarks import make_hands_detector, extract_two_hands_from_frame
from src.common.utils import get_logger, ensure_dir, pad_or_truncate_sequence

logger = get_logger("preprocess_wlasl")

# Video file extensions we will treat as clips.
_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def count_videos_per_gloss(wlasl_root):
    """Scan ``wlasl_root`` and count video files inside each gloss subfolder.

    Parameters
    ----------
    wlasl_root : str
        Path to the WLASL root directory whose subfolders are glosses.

    Returns
    -------
    dict[str, list[str]]
        Mapping ``gloss_name -> sorted list of absolute video paths``.
    """
    if not os.path.isdir(wlasl_root):
        raise FileNotFoundError(
            "WLASL root directory not found: %s\n"
            "Expected the layout datasets/archive/dataset/SL/<gloss>/<id>.mp4"
            % wlasl_root
        )

    gloss_to_videos = {}
    for gloss_name in sorted(os.listdir(wlasl_root)):
        gloss_dir = os.path.join(wlasl_root, gloss_name)
        if not os.path.isdir(gloss_dir):
            continue
        videos = [
            os.path.join(gloss_dir, name)
            for name in sorted(os.listdir(gloss_dir))
            if name.lower().endswith(_VIDEO_EXTENSIONS)
        ]
        if videos:
            gloss_to_videos[gloss_name] = videos
    return gloss_to_videos


def choose_glosses(gloss_to_videos, num_glosses, explicit_glosses):
    """Decide which glosses to include in the processed dataset.

    Parameters
    ----------
    gloss_to_videos : dict[str, list[str]]
        Output of :func:`count_videos_per_gloss`.
    num_glosses : int
        How many top glosses to keep when no explicit list is given.
    explicit_glosses : list[str] or None
        Explicit gloss names; when provided this overrides the top-K logic.

    Returns
    -------
    list[str]
        The chosen gloss names, ordered (alphabetically for an explicit list, or
        by descending video count then name for the automatic top-K).
    """
    if explicit_glosses:
        chosen = []
        for gloss in explicit_glosses:
            gloss = gloss.strip()
            if not gloss:
                continue
            if gloss not in gloss_to_videos:
                logger.warning(
                    "Requested gloss '%s' not found under WLASL_ROOT -- skipping.",
                    gloss,
                )
                continue
            chosen.append(gloss)
        if not chosen:
            raise RuntimeError(
                "None of the explicitly requested glosses were found on disk."
            )
        # Stable, predictable label ordering for explicit lists.
        return sorted(chosen)

    # Automatic top-K: sort by (descending video count, then gloss name).
    ranked = sorted(
        gloss_to_videos.keys(),
        key=lambda g: (-len(gloss_to_videos[g]), g),
    )
    return ranked[:num_glosses]


def sample_frame_indices(total_frames, target_length):
    """Choose ``target_length`` frame indices that uniformly cover the clip.

    * If the video has at least ``target_length`` frames we pick evenly spaced
      indices spanning the full clip.
    * If the video is shorter we use every frame it has; the resulting shorter
      sequence is zero-padded later by :func:`pad_or_truncate_sequence`.

    Parameters
    ----------
    total_frames : int
        Number of frames available in the video.
    target_length : int
        Desired number of sampled frames.

    Returns
    -------
    list[int]
        Sorted, unique frame indices to read.
    """
    if total_frames <= 0:
        return []
    if total_frames <= target_length:
        return list(range(total_frames))
    # Evenly spaced indices across [0, total_frames - 1].
    idx = np.linspace(0, total_frames - 1, num=target_length)
    return sorted(set(int(round(i)) for i in idx))


def read_video_frames(video_path):
    """Read all frames of a video as a list of BGR ``numpy`` arrays.

    Parameters
    ----------
    video_path : str
        Path to the video file.

    Returns
    -------
    list[numpy.ndarray]
        The decoded BGR frames (possibly empty if the video is unreadable).
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return frames
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def video_to_sequence(video_path, hands, target_length):
    """Convert one video into a RAW ``(target_length, 126)`` landmark sequence.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    hands : mediapipe Hands
        A detector created with ``static_image_mode=False, max_num_hands=2``.
    target_length : int
        Output sequence length (``config.SEQUENCE_LENGTH``).

    Returns
    -------
    numpy.ndarray, dtype float32, shape (target_length, 126), OR
    None
        ``None`` when the video could not be read at all.
    """
    frames = read_video_frames(video_path)
    if not frames:
        return None

    indices = sample_frame_indices(len(frames), target_length)
    if not indices:
        return None

    per_frame_features = []
    for i in indices:
        bgr = frames[i]
        # MediaPipe expects RGB; convert each sampled frame from OpenCV BGR.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        feat126 = extract_two_hands_from_frame(rgb, hands)
        per_frame_features.append(feat126)

    seq = np.stack(per_frame_features).astype(np.float32)  # (<=target_length, 126)
    # Guarantee an exact (target_length, 126) shape (pad short, truncate long).
    seq = pad_or_truncate_sequence(seq, target_length, config.DYNAMIC_FEATURE_DIM)
    return seq.astype(np.float32)


def preprocess(num_glosses, explicit_glosses, frames, output_path,
               max_videos_per_gloss):
    """Run the full WLASL preprocessing pipeline.

    Parameters
    ----------
    num_glosses : int
        Number of top glosses to keep (when no explicit list given).
    explicit_glosses : list[str] or None
        Explicit gloss names overriding the top-K logic.
    frames : int
        Target sequence length per video.
    output_path : str
        Destination ``.npz`` path.
    max_videos_per_gloss : int or None
        Cap on the number of videos used per gloss (``None`` / non-positive ->
        use all available videos).
    """
    gloss_to_videos = count_videos_per_gloss(config.WLASL_ROOT)
    logger.info("Scanned WLASL_ROOT: %d glosses contain at least one video.",
                len(gloss_to_videos))

    chosen = choose_glosses(gloss_to_videos, num_glosses, explicit_glosses)
    logger.info("Chosen %d gloss(es):", len(chosen))
    for gloss in chosen:
        logger.info("  %-20s %d video(s)", gloss, len(gloss_to_videos[gloss]))

    sequences = []          # list of (frames, 126) float32 arrays
    labels = []             # list of int gloss indices
    kept_per_gloss = {g: 0 for g in chosen}
    skipped_per_gloss = {g: 0 for g in chosen}

    # One streaming detector reused across all frames/videos.
    hands = make_hands_detector(
        static_image_mode=False,
        max_num_hands=config.MAX_HANDS_DYNAMIC,
        min_detection_confidence=0.5,
    )

    try:
        for gloss_idx, gloss in enumerate(chosen):
            videos = gloss_to_videos[gloss]
            if max_videos_per_gloss is not None and max_videos_per_gloss > 0:
                videos = videos[:max_videos_per_gloss]

            for video_path in tqdm(videos,
                                   desc="[%2d/%2d] %s" % (gloss_idx + 1,
                                                          len(chosen), gloss),
                                   unit="vid"):
                seq = video_to_sequence(video_path, hands, frames)
                if seq is None:
                    skipped_per_gloss[gloss] += 1
                    continue
                sequences.append(seq)
                labels.append(gloss_idx)
                kept_per_gloss[gloss] += 1
    finally:
        hands.close()

    if not sequences:
        raise RuntimeError(
            "No video sequences were extracted. Check WLASL_ROOT and that the "
            "video files are readable by OpenCV."
        )

    X = np.stack(sequences).astype(np.float32)        # (N, frames, 126)
    y = np.asarray(labels, dtype=np.int64)            # (N,)
    glosses_arr = np.array(chosen, dtype=object)      # (num_glosses,)

    ensure_dir(os.path.dirname(os.path.abspath(output_path)))
    np.savez(output_path, X=X, y=y, glosses=glosses_arr)

    # ----------------------- Summary report --------------------------------
    total_kept = int(sum(kept_per_gloss.values()))
    total_skipped = int(sum(skipped_per_gloss.values()))
    logger.info("=" * 60)
    logger.info("Saved RAW sequences to: %s", os.path.abspath(output_path))
    logger.info("X shape: %s (dtype=%s)", X.shape, X.dtype)
    logger.info("y shape: %s (dtype=%s)", y.shape, y.dtype)
    logger.info("Total kept: %d   Total skipped (unreadable): %d",
                total_kept, total_skipped)
    logger.info("-" * 60)
    logger.info("Per-gloss kept / skipped:")
    min_kept = None
    for gloss in chosen:
        logger.info("  %-20s kept=%-4d skipped=%-4d",
                    gloss, kept_per_gloss[gloss], skipped_per_gloss[gloss])
        if min_kept is None or kept_per_gloss[gloss] < min_kept:
            min_kept = kept_per_gloss[gloss]
    logger.info("=" * 60)
    logger.warning(
        "NOTE: This is a demo-scale dataset -- the smallest gloss has only %s "
        "kept sample(s). With so few samples per class the dynamic LSTM will "
        "easily overfit. For a stronger model: increase --num-glosses only if "
        "more data per gloss is available, gather more clips per gloss, and/or "
        "apply temporal/landmark augmentation during training.",
        min_kept,
    )


def parse_glosses_arg(value):
    """Parse the ``--glosses`` comma-separated argument into a clean list.

    Parameters
    ----------
    value : str or None
        Raw argument value, e.g. ``"hello, thanks ,please"``.

    Returns
    -------
    list[str] or None
        Cleaned gloss names, or ``None`` if no value was provided.
    """
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def build_arg_parser():
    """Construct the command-line argument parser for this script.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess word-level sign videos (WLASL) into a .npz of "
                    "fixed-length RAW two-hand landmark sequences (30x126)."
    )
    parser.add_argument(
        "--num-glosses", type=int, default=config.WLASL_DEFAULT_NUM_GLOSSES,
        help="Number of top glosses (by video count) to keep when no explicit "
             "--glosses list is given (default: %d)."
             % config.WLASL_DEFAULT_NUM_GLOSSES,
    )
    parser.add_argument(
        "--glosses", type=str, default=None,
        help="Optional explicit comma-separated gloss list (e.g. "
             "'hello,thanks,please'); overrides --num-glosses.",
    )
    parser.add_argument(
        "--frames", type=int, default=config.SEQUENCE_LENGTH,
        help="Target frames per sequence (pad/truncate) "
             "(default: config.SEQUENCE_LENGTH = %d)." % config.SEQUENCE_LENGTH,
    )
    parser.add_argument(
        "--output", type=str, default=config.WLASL_SEQ_NPZ,
        help="Output .npz path (default: config.WLASL_SEQ_NPZ).",
    )
    parser.add_argument(
        "--max-videos-per-gloss", type=int, default=0,
        help="Cap videos used per gloss (default: 0 = use all available).",
    )
    return parser


def main():
    """Parse arguments and run the preprocessing pipeline."""
    args = build_arg_parser().parse_args()
    explicit = parse_glosses_arg(args.glosses)

    logger.info("Configuration:")
    logger.info("  num_glosses          = %s", args.num_glosses)
    logger.info("  glosses (explicit)   = %s", explicit)
    logger.info("  frames               = %s", args.frames)
    logger.info("  output               = %s", args.output)
    logger.info("  max_videos_per_gloss = %s", args.max_videos_per_gloss)

    preprocess(
        num_glosses=args.num_glosses,
        explicit_glosses=explicit,
        frames=args.frames,
        output_path=args.output,
        max_videos_per_gloss=args.max_videos_per_gloss,
    )


if __name__ == "__main__":
    main()
