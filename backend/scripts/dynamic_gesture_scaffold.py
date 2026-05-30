"""Phase 2 scaffold for dynamic gesture recognition using sequence landmarks and LSTM."""

from dataclasses import dataclass


@dataclass
class SequenceSample:
    """Represents a single dynamic gesture sample with T timesteps and 63 feature dimensions."""

    label: str
    sequence_path: str


def expected_dataset_format() -> str:
    return """
    dataset/
      HELLO/
        sample_001.npy  # shape: (T, 63)
      THANK_YOU/
        sample_001.npy
    """.strip()


def next_steps() -> list[str]:
    return [
        "Collect temporal landmark sequences per gesture.",
        "Pad/trim sequences to a fixed timestep window.",
        "Train an LSTM classifier and export to backend/models/v2.",
        "Integrate sequence buffering and endpoint support in FastAPI.",
    ]
