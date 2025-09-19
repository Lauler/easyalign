import os
import tempfile

import soundfile as sf
import torch
from tqdm import tqdm

from easyalign.data.datamodel import AudioChunk, AudioMetadata, SpeechSegment
from easyalign.utils import convert_audio_to_wav
from easyalign.vad.pyannote import run_vad_pipeline as run_vad_pipeline_pyannote
from easyalign.vad.silero import run_vad_pipeline as run_vad_pipeline_silero

DISPATCH_MODEL = {
    "pyannote": run_vad_pipeline_pyannote,
    "silero": run_vad_pipeline_silero,
}


def read_audio(audio_path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            convert_audio_to_wav(audio_path, os.path.join(tmpdirname, "tmp.wav"))
            audio, sr = sf.read(os.path.join(tmpdirname, "tmp.wav"))
        except Exception as e:
            print(f"Error reading audio file {audio_path}. {e}")
            return None, None
    return audio, sr


def encode_metadata(audio_path, sample_rate=16000, speeches: list[SpeechSegment] = []):
    # Get length of audio file
    f = sf.SoundFile(audio_path)
    audio_length = len(f) / f.samplerate

    audio_metadata = AudioMetadata(
        audio_path=audio_path,
        sample_rate=sample_rate,
        duration=audio_length,
        speeches=speeches,
    )
    return audio_metadata


def encode_vad_segments(vad_segments):
    audio_segments = []
    for segment in vad_segments:
        audio_segment = AudioChunk(start=segment["start"], end=segment["end"])
        audio_segments.append(audio_segment)

    return audio_segments
