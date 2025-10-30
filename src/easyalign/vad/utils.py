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
    f = sf.SoundFile(full_audio_path)
    audio_length = len(f) / f.samplerate

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
