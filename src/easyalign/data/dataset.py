import json
import multiprocessing as mp
import os

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Wav2Vec2Processor, WhisperProcessora

from easyalign.utils import convert_audio_to_wav


class AudioDataset(Dataset):
    """
    An audio file contains multiple candidate alignment segments.
    AudioDataset contains the preprocessed wav2vec2 features from all alignment segments
    for a given audio file. It returns them one at a time, so a DataLoader can be used
    to iterate over them.

    These AudioDataset objects are in turn returned by AudioFileChunkerDataset,
    which allows us to use nested DataLoaders to

    1. Pre-load audio files and create wav2vec2 features in background processes with
        a DataLoader in the outer loop (AudioFileChunkerDataset).
    2. Load the wav2vec2 features of a given file for inference in background processes,
        using a separate DataLoader in the inner loop (AudioDataset).
    """

    def __init__(self, features, sub_dict):
        self.features = features
        self.sub_dict = sub_dict  # Use this to access all metadata

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


class AlignmentChunkerDataset(AudioFileChunkerDataset):
    """
    Pytorch dataset that chunks audio according to start/end times of speech segments,
    and further chunks the speech segments to 30s chunks.

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
        # Inherit methods from AudioFileChunkerDataset
        super().__init__(json_paths=json_paths, audio_paths=audio_paths, model_name=model_name)
        if audio_dir is not None:
            audio_paths = [os.path.join(audio_dir, file) for file in audio_paths]

        self.audio_paths = audio_paths
        self.sr = sr
        self.chunk_size = chunk_size

    def seconds_to_frames(self, seconds, sr=16000):
        return int(seconds * sr)

    def json_chunks(self, sub_dict):
        for speech in sub_dict["speeches"]:
            yield speech["speech_id"], speech["start_segment"], speech["end_segment"]

    def audio_chunker(self, audio_path, sub_dict, sr=16000):
        audio, sr = self.read_audio(audio_path)
        i = 0
        for speech_id, start, end in self.json_chunks(sub_dict):
            start_frame = self.seconds_to_frames(start, sr)
            end_frame = self.seconds_to_frames(end, sr)
            sub_dict["speeches"][i]["audio_frames"] = end_frame - start_frame
            i += 1
            yield speech_id, audio[start_frame:end_frame]

    def check_if_aligned(self, sub_dict):
        """
        We include information about whether alignment has already been performed.
        Useful for skipping already aligned files.
        """
        is_aligned = "subs" in sub_dict
        return is_aligned

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

        mel_dataset = AudioDataset(spectograms, sub_dict)

        out_dict = {
            "dataset": mel_dataset,
            "metadata": sub_dict["metadata"],
            "audio_path": audio_path,
            "json_path": json_path,
        }

        return out_dict
