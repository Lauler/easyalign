import json
import logging
import multiprocessing as mp
import os
import tempfile

import msgspec
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Wav2Vec2Processor, WhisperProcessor

from easyalign.data.datamodel import AudioChunk, AudioMetadata, SpeechSegment
from easyalign.utils import convert_audio_to_wav

logger = logging.getLogger(__name__)


class VADAudioDataset(Dataset):
    """
    User can provide either

    1. a list of audio paths.
    2. a list of paths to metadata files that conform to the AudioMetadata datamodel,
        each containing a path to an audio file and optional speech segments to run VAD on.

    Args:
        audio_paths: List of paths to audio files.
        audio_dir: Directory containing audio files (if audio_paths are relative).
    """

    def __init__(
        self,
        audio_paths: list | None = None,
        audio_dir: str | None = None,
        sample_rate: int = 16000,
    ):
        self.audio_paths = audio_paths
        self.sample_rate = sample_rate
        self.audio_dir = audio_dir

        if audio_dir is not None:
            self.full_audio_paths = [os.path.join(audio_dir, file) for file in audio_paths]

    def read_audio(self, audio_path):
        with tempfile.TemporaryDirectory() as tmpdirname:
            try:
                convert_audio_to_wav(
                    input_file=audio_path,
                    output_file=os.path.join(tmpdirname, "tmp.wav"),
                    sample_rate=self.sample_rate,
                )
                audio, sr = sf.read(os.path.join(tmpdirname, "tmp.wav"))
            except Exception as e:
                logging.error(f"Error reading audio file {audio_path}. {e}")
                return None, None
        return audio, sr

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio, sr = self.read_audio(self.full_audio_paths[idx])

        return {
            "audio": audio,
            "sample_rate": sr,
            "audio_path": self.audio_paths[idx],  # original path
            "audio_dir": self.audio_dir,  # directory where audio is located
        }


def vad_collate_fn(batch):
    audios = [item["audio"] for item in batch]
    sample_rates = [item["sample_rate"] for item in batch]
    audio_paths = [item["audio_path"] for item in batch]
    audio_dirs = [item["audio_dir"] for item in batch]

    audios = torch.tensor(np.array(audios)).to(torch.float32)

    return {
        "audio": audios,
        "sample_rate": sample_rates,
        "audio_path": audio_paths,
        "audio_dir": audio_dirs,
    }


class AudioSliceDataset(Dataset):
    """
    AudioSliceDataset iterates over `chunk_size` sized slices of audio/features for a
    single audio file. AudioSliceDatasets are created by AudioFileDataset.

    This division between AudioFileDataset and AudioSliceDataset allows using nested
    DataLoaders, ensuring we can:

    1. Pre-load audio files and create wav2vec2 features in background processes with
        a DataLoader in the outer loop (AudioFileDataset).
    2. Load the wav2vec2 features of a given file for inference in background processes,
        using a separate DataLoader in the inner loop (AudioSliceDataset).
    """

    def __init__(self, features, metadata):
        self.features = features
        self.metadata = metadata  # Metadata, timestamps, etc.

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


class AudioFileDataset(Dataset):
    """
    Loads audio files and corresponding metadata files. Splits the audio into chunks
    according to metadata, and creates wav2vec2 features for each chunk. Returns an
    AudioSliceDataset object containing the features for each chunk, along with the
    metadata.

    Args:
        metadata: List of AudioMetadata objects or paths to JSON files.
        model_name: Model name to use for the processor
        audio_dir: Directory with audio files
        sr: Sample rate
        chunk_size: Chunk size in seconds to split audio into
    """

    def __init__(
        self,
        metadata: list[AudioMetadata] | list[str],
        model_name="KBLab/wav2vec2-large-voxrex-swedish",
        audio_dir="data",
        sample_rate=16000,  # sample rate
        chunk_size=30,  # seconds per chunk for wav2vec2
        use_vad=True,
    ):
        if isinstance(metadata[0], str):
            self.json_paths = metadata
            self.metadata = []
            decoder = msgspec.json.Decoder()
            for path in self.json_paths:
                with open(path, "r") as f:
                    data = f.read()
                    data = decoder.decode(data, type=AudioMetadata)
                    self.metadata.append(data)
        else:
            self.metadata = metadata

        self.audio_dir = audio_dir
        self.sr = sample_rate
        self.chunk_size = chunk_size
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.use_vad = use_vad

    def seconds_to_frames(self, seconds, sr=16000):
        return int(seconds * sr)

    def get_speech_metadata(self, metadata, sr=16000):
        for speech in metadata.speeches:
            start_frame = self.seconds_to_frames(speech.start, sr)
            end_frame = self.seconds_to_frames(speech.end, sr)
            speech.audio_frames = end_frame - start_frame
            yield speech.speech_id, start_frame, end_frame

    def speech_chunker(self, audio_path, metadata, sr=16000):
        audio, sr = self.read_audio(audio_path)
        for speech_id, start_frame, end_frame in self.get_speech_metadata(metadata, sr):
            yield speech_id, audio[start_frame:end_frame]

    def get_speech_features(self, audio_path, metadata):
        spectograms = []
        for speech_id, audio_speech in self.speech_chunker(audio_path, metadata):
            audio_speech = torch.tensor(audio_speech).unsqueeze(0)  # Add batch dimension
            # Chunk the audio according to `chunk_size`
            audio_chunks = torch.split(audio_speech, self.chunk_size * self.sr, dim=1)  # 30s
            for audio_chunk in audio_chunks:
                spectogram = self.processor(
                    audio_chunk, sampling_rate=self.sr, return_tensors="pt"
                ).input_values
                # Create tuple with spectogram and speech_id so we can link back to the speech
                spectograms.append((spectogram, speech_id))
        return spectograms

    def get_vad_features(self, audio_path, metadata, sr=16000):
        audio, sr = self.read_audio(audio_path)
        spectograms = []
        for speech in metadata.speeches:
            for vad_chunk in speech.chunks:
                start_frame = self.seconds_to_frames(vad_chunk.start, sr)
                end_frame = self.seconds_to_frames(vad_chunk.end, sr)
                vad_chunk.audio_frames = end_frame - start_frame
                audio_chunk = audio[start_frame:end_frame]
                audio_chunk = torch.tensor(audio_chunk).unsqueeze(0)  # Add batch dimension
                spectogram = self.processor(
                    audio_chunk, sampling_rate=sr, return_tensors="pt"
                ).input_values
                spectograms.append((spectogram, speech.speech_id))

        return spectograms

    def read_audio(self, audio_path):
        with tempfile.TemporaryDirectory() as tmpdirname:
            try:
                convert_audio_to_wav(audio_path, os.path.join(tmpdirname, "tmp.wav"))
                audio, sr = sf.read(os.path.join(tmpdirname, "tmp.wav"))
            except Exception as e:
                print(f"Error reading audio file {audio_path}. {e}")
                logging.error(f"Error reading audio file {audio_path}. {e}")
                os.makedirs("logs", exist_ok=True)
                with open("logs/error_audio_files.txt", "a") as f:
                    f.write(f"{audio_path}\n")
                return None, None
        return audio, sr

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx):
        metadata = self.metadata[idx]

        if self.audio_dir is not None:
            full_audio_path = os.path.join(self.audio_dir, metadata.audio_path)

        if self.use_vad:
            spectograms = self.get_vad_features(full_audio_path, metadata)
        else:
            spectograms = self.get_speech_features(full_audio_path, metadata)

        slice_dataset = AudioSliceDataset(spectograms, self.metadata)

        out_dict = {
            "dataset": slice_dataset,
            "audio_path": metadata.audio_path,
        }

        return out_dict
