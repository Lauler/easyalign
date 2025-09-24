import torch

from easyalign.vad.pyannote import VoiceActivitySegmentation
from easyalign.vad.pyannote import run_vad_pipeline as run_vad_pipeline_pyannote
from easyalign.vad.silero import run_vad_pipeline as run_vad_pipeline_silero
from easyalign.vad.utils import encode_metadata

DISPATCH_MODEL = {
    "pyannote": run_vad_pipeline_pyannote,
    "silero": run_vad_pipeline_silero,
}


def run_vad(
    audio_path: str,
    model,
    audio: torch.Tensor,
    audio_dir: str | None = None,
    chunk_size: int = 30,
    speeches: list = [],
    metadata: dict | None = None,
):
    """
    Run VAD on the given audio file.

    Args:
        audio_path: Relative uniquely identifying path to the audio file.
        model: The loaded VAD model.
        audio: The audio tensor.
        audio_dir: Directory where the audio files/dirs are located (if audio_path is relative).
        chunk_size: The maximum chunk size in seconds.
        speeches (list): Optional list of SpeechSegment objects to run VAD on specific
            segments of the audio.
        metadata (dict): Optional dictionary of additional file level metadata to include.
    """

    file_metadata = encode_metadata(
        audio_path=audio_path, audio_dir=audio_dir, speeches=speeches, metadata=metadata
    )

    if isinstance(model, VoiceActivitySegmentation):
        vad_pipeline = run_vad_pipeline_pyannote
    else:
        vad_pipeline = run_vad_pipeline_silero

    file_metadata = vad_pipeline(file_metadata, model=model, audio=audio, chunk_size=chunk_size)

    return file_metadata
