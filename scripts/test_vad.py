import os
import tempfile

import soundfile as sf
import torch
from pyannote.audio import Model
from tqdm import tqdm

from easyalign.data.datamodel import AudioChunk, AudioMetadata, SpeechSegment
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


def encode_metadata(
    audio_path, sample_rate=16000, speeches: list[SpeechSegment] = [], metadata=None
):
    # Get length of audio file
    f = sf.SoundFile(audio_path)
    audio_length = len(f) / f.samplerate

    audio_metadata = AudioMetadata(
        audio_path=audio_path,
        sample_rate=sample_rate,
        duration=audio_length,
        speeches=speeches,
        metadata=metadata,
    )
    return audio_metadata


def encode_vad_segments(vad_segments):
    audio_segments = []
    for segment in vad_segments:
        audio_segment = AudioChunk(start=segment["start"], end=segment["end"])
        audio_segments.append(audio_segment)

    return audio_segments


def run_vad_pipeline(metadata: AudioMetadata, vad_pipeline, chunk_size=30):
    audio, sr = read_audio(metadata.audio_path)
    if audio is None:
        return None

    if len(metadata.speeches) > 0:
        # Run VAD on each speech segment
        for speech in tqdm(metadata.speeches, desc="Running VAD on speeches"):
            speech_audio = audio[int(speech.start * sr) : int(speech.end * sr)]
            vad_segments = vad_pipeline(
                {
                    "waveform": torch.tensor(speech_audio).unsqueeze(0).to(torch.float32),
                    "sample_rate": sr,
                }
            )
            vad_segments = merge_chunks(vad_segments, chunk_size=chunk_size)
            # Add speech.start offset to each segment
            print(vad_segments)
            vad_segments = [
                {
                    "start": seg["start"] + speech.start,
                    "end": seg["end"] + speech.start,
                    "segments": seg["segments"],
                }
                for seg in vad_segments
            ]
            segments = encode_vad_segments(vad_segments)
            speech.chunks = segments
    else:
        # Run VAD on entire audio
        vad_segments = vad_pipeline(
            {
                "waveform": torch.tensor(audio).unsqueeze(0).to(torch.float32),
                "sample_rate": sr,
            }
        )

        vad_segments = merge_chunks(vad_segments, chunk_size=chunk_size)
        segments = encode_vad_segments(vad_segments)
        metadata.speeches.append(
            SpeechSegment(start=0, end=metadata.duration, text=None, chunks=segments)
        )

    return metadata


vad_pipeline = load_vad_model()
audio_path = "data/audio_80.wav"

audio, sr = read_audio(audio_path)
metadata = encode_metadata(
    audio_path,
    sample_rate=sr,
    speeches=[SpeechSegment(start=0, end=22), SpeechSegment(start=30, end=80)],
)
vad_segments = run_vad_pipeline(metadata, vad_pipeline, chunk_size=30)


from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

model = load_silero_vad()
