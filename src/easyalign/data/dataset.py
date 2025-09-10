import json
import logging
import multiprocessing as mp
import os
import tempfile

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Wav2Vec2Processor, WhisperProcessor

from easyalign.utils import convert_audio_to_wav

logger = logging.getLogger(__name__)


class AudioSliceDataset(Dataset):
    """
    AudioSliceDataset iterates over slices of audio/features for a single audio file.
    AudioSliceDatasets are created by AudioFileDataset.

    This division between AudioFileDataset and AudioSliceDataset allows using nested
    DataLoaders, ensuring we can:

    1. Pre-load audio files and create wav2vec2 features in background processes with
        a DataLoader in the outer loop (AudioFileDataset).
    2. Load the wav2vec2 features of a given file for inference in background processes,
        using a separate DataLoader in the inner loop (AudioSliceDataset).
    """

    def __init__(self, features, sub_dict):
        self.features = features
        self.sub_dict = sub_dict  # Use this to access all metadata

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


class AudioFileDataset(Dataset):
    """
    Loads audio files and corresponding json metadata files. Splits the audio into chunks
    according to metadata, and creates wav2vec2 features for each chunk. Returns an
    AudioSliceDataset object containing the features for each chunk, along with the
    metadata.

    Args:
        json_paths (list): List of paths to json files
        model_name (str): Model name to use for the processor
        audio_dir (str): Directory with audio files
        sr (int): Sample rate
        chunk_size (int): Chunk size in seconds to split audio into
    """

    def __init__(
        self,
        audio_paths,
        json_paths,
        model_name="KBLab/wav2vec2-large-voxrex-swedish",
        audio_dir="data/audio/all",
        sr=16000,  # sample rate
        chunk_size=30,  # seconds per chunk for wav2vec2
    ):
        if audio_dir is not None:
            audio_paths = [os.path.join(audio_dir, file) for file in audio_paths]

        self.audio_paths = audio_paths
        self.sr = sr
        self.chunk_size = chunk_size
        self.json_paths = json_paths
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def seconds_to_frames(self, seconds, sr=16000):
        return int(seconds * sr)

    def get_speech_metadata(self, sub_dict):
        for speech in sub_dict["speeches"]:
            yield speech["speech_id"], speech["start_segment"], speech["end_segment"]

    def speech_chunker(self, audio_path, sub_dict, sr=16000):
        audio, sr = self.read_audio(audio_path)
        i = 0
        for speech_id, start, end in self.get_speech_metadata(sub_dict):
            start_frame = self.seconds_to_frames(start, sr)
            end_frame = self.seconds_to_frames(end, sr)
            sub_dict["speeches"][i]["audio_frames"] = end_frame - start_frame
            i += 1
            yield speech_id, audio[start_frame:end_frame]

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
        audio_path = self.audio_paths[idx]
        json_path = self.json_paths[idx]

        with open(json_path) as f:
            sub_dict = json.load(f)

        spectograms = []
        for speech_id, audio_speech in self.audio_chunker(audio_path, sub_dict):
            audio_speech = torch.tensor(audio_speech).unsqueeze(0)  # Add batch dimension
            audio_chunks = torch.split(audio_speech, self.chunk_size * self.sr, dim=1)  # 30s
            for audio_chunk in audio_chunks:
                spectogram = self.processor(
                    audio_chunk, sampling_rate=self.sr, return_tensors="pt"
                ).input_values
                # Create tuple with spectogram and speech_id so we can link back to the speech
                spectograms.append((spectogram, speech_id))

        mel_dataset = AudioSliceDataset(spectograms, sub_dict)

        out_dict = {
            "dataset": mel_dataset,
            "metadata": sub_dict["metadata"],
            "audio_path": audio_path,
            "json_path": json_path,
        }

        return out_dict
