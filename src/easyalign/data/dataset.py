import logging
import os
import tempfile
from pathlib import Path

import msgspec
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor, WhisperProcessor

from easyalign.data.datamodel import AudioMetadata
from easyalign.utils import convert_audio_to_wav

logger = logging.getLogger(__name__)


class JSONMetadataDataset(Dataset):
    """
    Dataset for reading AudioMetadata JSON files.

    Args:
        json_paths: List of paths to JSON files.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> from easyalign.data.dataset import JSONMetadataDataset
        >>> json_files = list(Path("output/vad").rglob("*.json"))
        >>> dataset = JSONMetadataDataset(json_files)
        >>> loader = DataLoader(dataset, num_workers=4, prefetch_factor=2)
        >>> for metadata in loader:
        ...     print(metadata)
    """

    def __init__(self, json_paths: list[str | Path]):
        self.json_paths = [Path(p) for p in json_paths]
        self.decoder = msgspec.json.Decoder(type=AudioMetadata)

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx) -> AudioMetadata:
        json_path = self.json_paths[idx]
        with open(json_path, "r") as f:
            return self.decoder.decode(f.read())


class VADAudioDataset(Dataset):
    """
    Args:
        audio_paths: List of paths to audio files.
        audio_dir: Directory containing audio files (if `audio_paths` are relative).
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
        processor: The Wav2vec2Processor to use for feature extraction.
        audio_dir: Directory with audio files
        sample_rate: Sample rate to resample audio to. Default 16000.
        chunk_size: When VAD is not used, SpeechSegments are naively split into
            `chunk_size` sized chunks for feature extraction.
        use_vad: Whether to use VAD-based chunks (if available in metadata), or just
            naïvely split the audio of speech segments into `chunk_size` chunks.
    """

    def __init__(
        self,
        metadata: JSONMetadataDataset | list[AudioMetadata] | AudioMetadata,
        processor: Wav2Vec2Processor | WhisperProcessor,
        audio_dir="data",
        sample_rate=16000,  # sample rate
        chunk_size=30,  # seconds per chunk for wav2vec2
        alignment_strategy: str = "speech",
    ):
        if isinstance(metadata, AudioMetadata):
            metadata = [metadata]
        else:
            self.metadata = metadata

        self.audio_dir = audio_dir
        self.sr = sample_rate
        self.chunk_size = chunk_size
        self.processor = processor
        self.processor_attribute = (
            "input_values" if isinstance(processor, Wav2Vec2Processor) else "input_features"
        )
        self.alignment_strategy = alignment_strategy

    def read_audio(self, audio_path):
        with tempfile.TemporaryDirectory() as tmpdirname:
            try:
                convert_audio_to_wav(audio_path, os.path.join(tmpdirname, "tmp.wav"))
                audio, sr = sf.read(os.path.join(tmpdirname, "tmp.wav"))
            except Exception as e:
                print(f"Error reading audio file {audio_path}. \n\n {e}")
                logging.error(f"Error reading audio file {audio_path}. \n\n {e}")
                return None, None
        return audio, sr

    def seconds_to_frames(self, seconds, sr=16000):
        return int(seconds * sr)

    def get_speech_features(self, audio_path, metadata, sr=16000):
        """
        Extract features for each speech segment in the metadata. When VAD is not used,
        the speech segments are naively split into `chunk_size` sized chunks for wav2vec2
        inference.
        """
        audio, sr = self.read_audio(audio_path)
        features = []
        for speech in metadata.speeches:
            start_frame = self.seconds_to_frames(speech.start, sr)
            end_frame = self.seconds_to_frames(speech.end, sr)
            speech.audio_frames = end_frame - start_frame
            audio_speech = audio[start_frame:end_frame]
            audio_speech = torch.tensor(audio_speech).unsqueeze(0)  # Add batch dimension
            # Chunk the audio according to `chunk_size`
            audio_chunks = torch.split(audio_speech, self.chunk_size * self.sr, dim=1)  # 30s
            for audio_chunk in audio_chunks:
                inputs = self.processor(
                    audio_chunk,
                    sampling_rate=self.sr,
                    return_tensors="pt",
                )
                feature = getattr(inputs, self.processor_attribute)

                # Create tuple with feature and speech_id so we can link back to the speech
                features.append(
                    {"feature": feature, "start_time_global": -100, "speech_id": speech.speech_id}
                )
        return features

    def get_vad_features(self, audio_path, metadata, sr=16000):
        """
        Extract features for each VAD chunk in the metadata. To keep alignment timestamps
        in sync, we also return the global start time of each chunk.
        """
        audio, sr = self.read_audio(audio_path)
        features = []
        for speech in metadata.speeches:
            for i, vad_chunk in enumerate(speech.chunks):
                start_frame = self.seconds_to_frames(vad_chunk.start, sr)
                end_frame = self.seconds_to_frames(vad_chunk.end, sr)
                start_time_global = vad_chunk.start

                vad_chunk.audio_frames = end_frame - start_frame
                audio_chunk = audio[start_frame:end_frame]

                if isinstance(self.processor, Wav2Vec2Processor):
                    audio_chunk = torch.tensor(audio_chunk).unsqueeze(0)  # Add batch dimension

                inputs = self.processor(
                    audio_chunk,
                    sampling_rate=self.sr,
                    return_tensors="pt",
                )
                feature = getattr(inputs, self.processor_attribute)
                features.append(
                    {
                        "feature": feature,
                        "start_time_global": start_time_global,
                        "speech_id": speech.speech_id,
                    }
                )

        return features

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        metadata = self.metadata[idx]

        if self.audio_dir is not None:
            full_audio_path = os.path.join(self.audio_dir, metadata.audio_path)

        for i, speech in enumerate(metadata.speeches):
            if speech.speech_id is None:
                speech.speech_id = i  # Assign ID if missing

        if self.alignment_strategy == "chunk":
            features = self.get_vad_features(full_audio_path, metadata)
        else:
            features = self.get_speech_features(full_audio_path, metadata)

        slice_dataset = AudioSliceDataset(features, metadata)

        out_dict = {
            "dataset": slice_dataset,
            "audio_path": metadata.audio_path,
        }

        return out_dict
