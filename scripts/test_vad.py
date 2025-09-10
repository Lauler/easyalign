import os
import tempfile

import soundfile as sf
import torch
from pyannote.audio import Model
from tqdm import tqdm

from easyalign.data.datamodel import AudioMetadata, AudioSegment, SpeechSegment
from easyalign.utils import convert_audio_to_wav
from easyalign.vad.pyannote import load_vad_model, merge_chunks


def read_audio(audio_path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            convert_audio_to_wav(audio_path, os.path.join(tmpdirname, "tmp.wav"))
            audio, sr = sf.read(os.path.join(tmpdirname, "tmp.wav"))
        except Exception as e:
            print(f"Error reading audio file {audio_path}. {e}")
            return None, None
    return audio, sr


vad_pipeline = load_vad_model()
audio_path = "data/audio_80.wav"

audio, sr = read_audio(audio_path)


vad_segments = vad_pipeline(
    {"waveform": torch.tensor(audio).unsqueeze(0).to(torch.float32), "sample_rate": sr}
)

vad_segments = merge_chunks(vad_segments, chunk_size=30)

AudioSegment(start=vad_segments[0]["start"], end=vad_segments[0]["end"])

vad_segments[0]


from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

model = load_silero_vad()
vad_silero = get_speech_timestamps(
    audio, model, max_speech_duration_s=30, return_seconds=True
)


def merge_chunks_silero(segments, chunk_size=30):
    current_start = segments[0]["start"]
    current_end = 0
    merged_segments = []
    subsegments = []

    for segment in segments:
        if (
            segment["end"] - current_start > chunk_size
            and current_end - current_start > 0
        ):
            merged_segments.append(
                {"start": current_start, "end": current_end, "segments": subsegments}
            )
            current_start = segment["start"]
            subsegments = []
        current_end = segment["start"]
        subsegments.append((segment["start"], segment["end"]))

    merged_segments.append(
        {"start": current_start, "end": segments[-1]["end"], "segments": subsegments}
    )
    return merged_segments


merge_chunks_silero(vad_silero, chunk_size=30)
vad_silero


vad_segments

AudioSegment(start=vad_silero[0]["start"], end=vad_silero[0]["end"])


def encode_vad_audiosegments(vad_segments, audio_path, sample_rate=16000):
    audio_segments = []
    for segment in vad_segments:
        audio_segment = AudioSegment(start=segment["start"], end=segment["end"])
        audio_segments.append(audio_segment)

    # Get length of audio file
    f = sf.SoundFile(audio_path)
    audio_length = len(f) / f.samplerate

    audio_metadata = AudioMetadata(
        audio_path=audio_path, sample_rate=sample_rate, duration=audio_length
    )
    return audio_metadata, audio_segments
