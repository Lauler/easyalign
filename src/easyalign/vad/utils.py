import os
import tempfile

import soundfile as sf
import torch
from tqdm import tqdm

from easyalign.data.datamodel import AudioChunk, AudioMetadata, SpeechSegment
from easyalign.utils import convert_audio_to_wav


def read_audio(audio_path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            convert_audio_to_wav(audio_path, os.path.join(tmpdirname, "tmp.wav"))
            audio, sr = sf.read(os.path.join(tmpdirname, "tmp.wav"))
        except Exception as e:
            print(f"Error reading audio file {audio_path}. {e}")
            return None, None
    return audio, sr


def get_video_length(video_path):
    """Get the length of an audio file embedded in video containers using ffprobe."""
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
        print(f"Error getting audio length for {video_path}")


def encode_metadata(
    audio_path,
    audio_dir: str | None = None,
    sample_rate=16000,
    speeches: list[SpeechSegment] | None = None,
    metadata=None,
):
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
            print(f"Could not get length of audio file {full_audio_path}. ")
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
    return int(seconds * sr)


def encode_vad_segments(vad_segments):
    audio_segments = []
    for segment in vad_segments:
        audio_frames = seconds_to_frames(segment["end"] - segment["start"])
        audio_segment = AudioChunk(
            start=segment["start"], end=segment["end"], audio_frames=audio_frames
        )
        audio_segments.append(audio_segment)

    return audio_segments
