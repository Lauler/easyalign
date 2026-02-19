import logging
import os
import tempfile

import soundfile as sf
import torch
from tqdm import tqdm

from easyaligner.data.datamodel import AudioChunk, AudioMetadata, SpeechSegment
from easyaligner.utils import convert_audio_to_wav

logger = logging.getLogger(__name__)


def get_video_length(video_path):
    """
    Get the length of an audio file embedded in video containers using ffprobe.

    Parameters
    ----------
    video_path : str
        Path to the video file.

    Returns
    -------
    float
        Duration of the audio in seconds, or None if an error occurs.
    """
    import json
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        ffprobe_output = json.loads(result.stdout)
        duration = float(ffprobe_output["format"]["duration"])
        return duration
    except Exception:
        logger.error(f"Error getting audio length for {video_path}")


def encode_metadata(
    audio_path,
    audio_dir: str | None = None,
    sample_rate=16000,
    speeches: list[SpeechSegment] | None = None,
    metadata=None,
):
    """
    Create an AudioMetadata object for a given audio file.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    audio_dir : str, optional
        Directory containing the audio file.
    sample_rate : int, default 16000
        Sample rate of the audio.
    speeches : list of SpeechSegment, optional
        List of speech segments.
    metadata : dict, optional
        Additional metadata.

    Returns
    -------
    AudioMetadata
        The encoded audio metadata object.
    """
    full_audio_path = audio_path
    if audio_dir is not None:
        full_audio_path = os.path.join(audio_dir, audio_path)

    # Get length of audio file
    try:
        f = sf.SoundFile(full_audio_path)
        audio_length = len(f) / f.samplerate
    except Exception as e:
        audio_length = get_video_length(full_audio_path)
        if audio_length is None:
            logger.error(f"Could not get length of audio file {full_audio_path}. ")
            raise e

    audio_metadata = AudioMetadata(
        audio_path=audio_path,
        sample_rate=sample_rate,
        duration=audio_length,
        speeches=speeches,
        metadata=metadata,
    )

    return audio_metadata


def seconds_to_frames(seconds, sr=16000):
    """
    Convert seconds to number of frames.

    Parameters
    ----------
    seconds : float
        Time in seconds.
    sr : int, default 16000
        Sample rate.

    Returns
    -------
    int
        Number of frames.
    """
    return int(seconds * sr)


def encode_vad_segments(vad_segments):
    """
    Encode VAD segments into a list of AudioChunk objects.

    Parameters
    ----------
    vad_segments : list of dict
        List of dictionaries with 'start' and 'end' keys.

    Returns
    -------
    list of AudioChunk
        List of encoded audio chunks.
    """
    audio_segments = []
    for segment in vad_segments:
        audio_frames = seconds_to_frames(segment["end"] - segment["start"])
        audio_segment = AudioChunk(
            start=segment["start"], end=segment["end"], audio_frames=audio_frames
        )
        audio_segments.append(audio_segment)

    return audio_segments
