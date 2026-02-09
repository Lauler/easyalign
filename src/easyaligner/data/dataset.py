import logging
import os
import tempfile
from pathlib import Path

import msgspec
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor, WhisperProcessor

from easyaligner.data.datamodel import AudioMetadata
from easyaligner.utils import convert_audio_to_wav, read_audio_segment

logger = logging.getLogger(__name__)


class JSONMetadataDataset(Dataset):
    """
    Dataset for reading AudioMetadata JSON files.

    Parameters
    ----------
    json_paths : list of str or list of Path
        List of paths to JSON files.

    Examples
    --------
    >>> from torch.utils.data import DataLoader
    >>> from easyaligner.data.dataset import JSONMetadataDataset
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
        logger.info(f"Loading metadata from {json_path}")
        with open(json_path, "r") as f:
            return self.decoder.decode(f.read())


class MsgpackMetadataDataset(Dataset):
    """
    Dataset for reading AudioMetadata Msgpack files.

    Parameters
    ----------
    msgpack_paths : list of str or list of Path
        List of paths to Msgpack files.
    """

    def __init__(self, msgpack_paths: list[str | Path]):
        self.msgpack_paths = [Path(p) for p in msgpack_paths]
        self.decoder = msgspec.msgpack.Decoder(type=AudioMetadata)

    def __len__(self):
        return len(self.msgpack_paths)

    def __getitem__(self, idx) -> AudioMetadata:
        msgpack_path = self.msgpack_paths[idx]
        logger.info(f"Loading metadata from {msgpack_path}")
        with open(msgpack_path, "rb") as f:
            return self.decoder.decode(f.read())


class VADAudioDataset(Dataset):
    """
    Dataset for VAD audio loading.

    Parameters
    ----------
    audio_paths : list of str, optional
        List of paths to audio files.
    audio_dir : str, optional
        Directory containing audio files (if `audio_paths` are relative).
    sample_rate : int, default 16000
        Sample rate.
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
            except Exception:
                logger.error(f"Failed to read audio file: {audio_path}", exc_info=True)
                return None, None
        return audio, sr

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        logger.info(f"Loading audio for VAD from {self.full_audio_paths[idx]}")
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

    Parameters
    ----------
    features : list
        List of audio features.
    metadata : AudioMetadata
        Metadata associated with the audio file.
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

    Parameters
    ----------
    metadata : JSONMetadataDataset or list of AudioMetadata or AudioMetadata
        List of AudioMetadata objects or paths to JSON files.
    processor : Wav2Vec2Processor or WhisperProcessor
        The Wav2vec2Processor to use for feature extraction.
    audio_dir : str, default "data"
        Directory with audio files
    sample_rate : int, default 16000
        Sample rate to resample audio to.
    chunk_size : int, default 30
        When VAD is not used, SpeechSegments are naively split into
        `chunk_size` sized chunks for feature extraction.
    alignment_strategy : str, default "speech"
        'speech' or 'chunk' - determines how chunks are defined.
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
            except Exception:
                logger.error(f"Failed to read audio file: {audio_path}", exc_info=True)
                return None, None
        return audio, sr

    def seconds_to_frames(self, seconds, sr=16000):
        return int(seconds * sr)

    def get_speech_features(self, audio_path, metadata, sr=16000):
        """
        Extract features for each speech segment in the metadata.

        When `alignment_strategy` is `speech`, the speech segments are split into `chunk_size`
        sized chunks for wav2vec2 inference.

        Parameters
        ----------
        audio_path : str
            Path to the audio file.
        metadata : AudioMetadata
            Metadata object.
        sr : int, default 16000
            Sample rate.

        Returns
        -------
        list of dict
            List of dictionaries containing extracted features and metadata for each chunk.
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

                # Create tuple with feature and speech_id so we can link back to the speech.
                # Insert dummy start_time_global for data collator compatibility with `get_vad_features`.
                features.append(
                    {"feature": feature, "start_time_global": -100, "speech_id": speech.speech_id}
                )
        return features

    def get_vad_features(self, audio_path, metadata, sr=16000):
        """
        Extract features for each VAD chunk in the metadata.

        The global start time of each chunk is also returned for debugging purposes.
        This method is used when `alignment_strategy` is set to `chunk`.

        Parameters
        ----------
        audio_path : str
            Path to the audio file.
        metadata : AudioMetadata
            Metadata object.
        sr : int, default 16000
            Sample rate.

        Returns
        -------
        list of dict
            List of dictionaries containing extracted features and metadata for each chunk.
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
                    # Add batch dimension
                    audio_chunk = torch.tensor(audio_chunk).unsqueeze(0)

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

        logger.info(f"Loading audio for alignment from {full_audio_path}")
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


class StreamingAudioSliceDataset(Dataset):
    """
    Dataset that lazily loads audio chunks on-demand using ffmpeg seek.

    Unlike AudioSliceDataset which holds all features in memory, this dataset
    stores only the chunk metadata and loads audio when __getitem__ is called.

    Parameters
    ----------
    audio_path : str or Path
        Path to the audio file.
    chunk_specs : list of dict
        List of dicts with 'start_sec', 'end_sec', 'speech_id' keys.
    processor : Wav2Vec2Processor or WhisperProcessor
        For feature extraction.
    sample_rate : int, default 16000
        Target sample rate.
    metadata : AudioMetadata, optional
        AudioMetadata object to pass through.
    """

    def __init__(
        self,
        audio_path: str | Path,
        chunk_specs: list[dict],
        processor: Wav2Vec2Processor | WhisperProcessor,
        sample_rate: int = 16000,
        metadata: AudioMetadata | None = None,
    ):
        self.audio_path = str(audio_path)
        self.chunk_specs = chunk_specs
        self.processor = processor
        self.sample_rate = sample_rate
        self.metadata = metadata
        self.processor_attribute = (
            "input_values" if isinstance(processor, Wav2Vec2Processor) else "input_features"
        )

    def __len__(self):
        return len(self.chunk_specs)

    def __getitem__(self, idx):
        spec = self.chunk_specs[idx]
        start_sec = spec["start_sec"]
        end_sec = spec["end_sec"]
        duration_sec = end_sec - start_sec

        # Read only this chunk from disk
        audio = read_audio_segment(
            audio_path=self.audio_path,
            start_sec=start_sec,
            duration_sec=duration_sec,
            sample_rate=self.sample_rate,
        )

        # Convert to tensor and add batch dimension for processor
        if isinstance(self.processor, Wav2Vec2Processor):
            audio = torch.tensor(audio).unsqueeze(0)

        # Extract features
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        feature = getattr(inputs, self.processor_attribute)

        return {
            "feature": feature,
            "start_time_global": start_sec,
            "speech_id": spec["speech_id"],
        }


class StreamingAudioFileDataset(Dataset):
    """
    Streaming version of AudioFileDataset that reads audio chunks on-demand.

    Instead of loading entire audio files and chunking in memory, this dataset
    returns a StreamingAudioSliceDataset that lazily loads each chunk via ffmpeg.

    Parameters
    ----------
    metadata : JSONMetadataDataset or list of AudioMetadata or AudioMetadata
        List of AudioMetadata objects, JSONMetadataDataset, or single AudioMetadata.
    processor : Wav2Vec2Processor or WhisperProcessor
        For feature extraction.
    audio_dir : str, default "data"
        Base directory for audio files.
    sample_rate : int, default 16000
        Target sample rate for resampling.
    chunk_size : int, default 30
        Maximum chunk size in seconds (for speech-based chunking).
    alignment_strategy : str, default "speech"
        'speech' or 'chunk' - determines how chunks are defined.
    """

    def __init__(
        self,
        metadata: JSONMetadataDataset | list[AudioMetadata] | AudioMetadata,
        processor: Wav2Vec2Processor | WhisperProcessor,
        audio_dir: str = "data",
        sample_rate: int = 16000,
        chunk_size: int = 30,
        alignment_strategy: str = "speech",
    ):
        if isinstance(metadata, AudioMetadata):
            self.metadata = [metadata]
        else:
            self.metadata = metadata

        self.audio_dir = audio_dir
        self.sr = sample_rate
        self.chunk_size = chunk_size
        self.processor = processor
        self.alignment_strategy = alignment_strategy

    def _get_speech_chunk_specs(self, metadata: AudioMetadata) -> list[dict]:
        """
        Build chunk specs from SpeechSegments, splitting into chunk_size pieces.

        This mirrors the behavior of AudioFileDataset.get_speech_features().

        Parameters
        ----------
        metadata : AudioMetadata
            Metadata object.

        Returns
        -------
        list of dict
            List of chunk specifications.
        """
        chunk_specs = []
        for speech in metadata.speeches:
            speech_start = speech.start
            speech_end = speech.end
            speech_duration = speech_end - speech_start

            # Calculate audio frames for the speech segment
            speech.audio_frames = int(speech_duration * self.sr)

            # Split into chunk_size sized pieces
            offset = 0.0
            while offset < speech_duration:
                chunk_start = speech_start + offset
                chunk_end = min(chunk_start + self.chunk_size, speech_end)

                chunk_specs.append(
                    {
                        "start_sec": chunk_start,
                        "end_sec": chunk_end,
                        "speech_id": speech.speech_id,
                    }
                )
                offset += self.chunk_size

        return chunk_specs

    def _get_vad_chunk_specs(self, metadata: AudioMetadata) -> list[dict]:
        """
        Build chunk specs from existing VAD chunks in metadata.

        This mirrors the behavior of AudioFileDataset.get_vad_features().

        Parameters
        ----------
        metadata : AudioMetadata
            Metadata object.

        Returns
        -------
        list of dict
            List of chunk specifications.
        """
        chunk_specs = []
        for speech in metadata.speeches:
            for vad_chunk in speech.chunks:
                start_sec = vad_chunk.start
                end_sec = vad_chunk.end

                # Calculate audio frames for the chunk
                vad_chunk.audio_frames = int((end_sec - start_sec) * self.sr)

                chunk_specs.append(
                    {
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "speech_id": speech.speech_id,
                    }
                )

        return chunk_specs

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        metadata = self.metadata[idx]
        audio_path = Path(self.audio_dir) / metadata.audio_path

        # Assign speech IDs if missing
        for i, speech in enumerate(metadata.speeches):
            if speech.speech_id is None:
                speech.speech_id = i

        logger.info(f"Creating streaming dataset for {audio_path}")

        # Build chunk specs based on alignment strategy
        if self.alignment_strategy == "chunk":
            chunk_specs = self._get_vad_chunk_specs(metadata)
        else:
            chunk_specs = self._get_speech_chunk_specs(metadata)

        # Return a streaming dataset for the inner dataloader
        slice_dataset = StreamingAudioSliceDataset(
            audio_path=audio_path,
            chunk_specs=chunk_specs,
            processor=self.processor,
            sample_rate=self.sr,
            metadata=metadata,
        )

        return {
            "dataset": slice_dataset,
            "audio_path": metadata.audio_path,
        }
