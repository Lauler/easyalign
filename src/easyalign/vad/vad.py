from pathlib import Path

import msgspec
import torch
from tqdm import tqdm

from easyalign.data.collators import vad_collate_fn
from easyalign.data.datamodel import SpeechSegment
from easyalign.data.dataset import VADAudioDataset
from easyalign.vad.pyannote import VoiceActivitySegmentation
from easyalign.vad.pyannote import run_vad_pipeline as run_vad_pipeline_pyannote
from easyalign.vad.silero import run_vad_pipeline as run_vad_pipeline_silero
from easyalign.vad.utils import encode_metadata


def run_vad(
    audio_path: str,
    model,
    audio: torch.Tensor,
    audio_dir: str | None = None,
    chunk_size: int = 30,
    speeches: list | None = None,
    metadata: dict | None = None,
):
    """
    Run VAD on the given audio file.

    Args:
        audio_path: Path to the audio file, that acts as a unique identifier.
        model: The loaded VAD model.
        audio: The audio tensor.
        audio_dir: Directory where the audio files/dirs are located (if audio_path is relative).
        chunk_size: The maximum length chunks VAD will create (seconds).
        speeches (list): Optional list of SpeechSegment objects to run VAD on specific
            segments of the audio.
        metadata (dict): Optional dictionary of additional file level metadata to include.
    """

    file_metadata = encode_metadata(
        audio_path=audio_path, audio_dir=audio_dir, speeches=speeches, metadata=metadata
    )

    if isinstance(model, VoiceActivitySegmentation):
        vad_pipeline = run_vad_pipeline_pyannote
    else:
        vad_pipeline = run_vad_pipeline_silero

    file_metadata = vad_pipeline(file_metadata, model=model, audio=audio, chunk_size=chunk_size)

    return file_metadata


def vad_pipeline(
    model,
    audio_paths: list,
    audio_dir: str,
    speeches: list | list[SpeechSegment] = [],
    chunk_size: int = 30,
    sample_rate: int = 16000,
    metadata: list[dict] | None = None,
    batch_size: int = 1,
    num_workers: int = 1,
    prefetch_factor: int = 2,
    save_json: bool = True,
    save_msgpack: bool = False,
    output_dir: str = "output/vad",
):
    """
    Run VAD on a list of audio files.

    Args:
        model: The loaded VAD model.
        audio_paths: List of paths to audio files.
        audio_dir: Directory where the audio files/dirs are located (if audio_paths are relative).
        speeches (list): Optional list of SpeechSegment objects to run VAD only on specific
            segments of the audio. Alignment can generally be improved if VAD/alignment is only
            performed on the segments of the audio that overlap with text transcripts.
        chunk_size: The maximum chunk size in seconds.
        sample_rate: The sample rate to resample the audio to before running VAD.
        metadata: Optional dictionary of additional file level metadata to include.
        batch_size: The batch size for the DataLoader.
        num_workers: The number of workers for the DataLoader.
        prefetch_factor: The prefetch factor for the DataLoader.
        save_json: Whether to save the VAD output as JSON files.
        json_dir: Directory to save the JSON files if save_json is True.
    """

    vad_dataset = VADAudioDataset(
        audio_paths=audio_paths, audio_dir=audio_dir, sample_rate=sample_rate
    )
    vad_dataloader = torch.utils.data.DataLoader(
        vad_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vad_collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    json_encoder = msgspec.json.Encoder()
    msgpack_encoder = msgspec.msgpack.Encoder()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []

    for i, audio_dict in enumerate(tqdm(vad_dataloader, desc="Running VAD on audio files")):
        audio = audio_dict["audio"][0]
        audio_path = audio_dict["audio_path"][0]
        vad_output = run_vad(
            audio_path=audio_path,
            audio_dir=audio_dir,
            model=model,
            audio=audio,
            chunk_size=chunk_size,
            speeches=speeches,
            metadata=metadata[i] if metadata is not None else None,
        )
        results.append(vad_output)

        if save_json:
            vad_msgspec = json_encoder.encode(vad_output)
            vad_msgspec = msgspec.json.format(vad_msgspec, indent=2)
            json_path = Path(output_dir) / (Path(audio_path).stem + ".json")
            with open(json_path, "wb") as f:
                f.write(vad_msgspec)

        if save_msgpack:
            vad_msgpack = msgpack_encoder.encode(vad_output)
            msgpack_path = Path(output_dir) / (Path(audio_path).stem + ".msgpack")
            with open(msgpack_path, "wb") as f:
                f.write(vad_msgpack)

    return results


def vad_pipeline_generator(
    model,
    audio_paths: list,
    audio_dir: str,
    speeches: list | list[SpeechSegment] = [],
    chunk_size: int = 30,
    sample_rate: int = 16000,
    metadata: list[dict] | None = None,
    batch_size: int = 1,
    num_workers: int = 1,
    prefetch_factor: int = 2,
    save_json: bool = True,
    save_msgpack: bool = False,
    output_dir: str = "output/vad",
):
    """
    Run VAD on a list of audio files.

    Args:
        model: The loaded VAD model.
        audio_paths: List of paths to audio files.
        audio_dir: Directory where the audio files/dirs are located (if audio_paths are relative).
        speeches (list): Optional list of SpeechSegment objects to run VAD only on specific
            segments of the audio. Alignment can generally be improved if VAD/alignment is only
            performed on the segments of the audio that overlap with text transcripts.
        chunk_size: The maximum chunk size in seconds.
        sample_rate: The sample rate to resample the audio to before running VAD.
        metadata: Optional dictionary of additional file level metadata to include.
        batch_size: The batch size for the DataLoader.
        num_workers: The number of workers for the DataLoader.
        prefetch_factor: The prefetch factor for the DataLoader.
        save_json: Whether to save the VAD output as JSON files.
        json_dir: Directory to save the JSON files if save_json is True.
    """

    vad_dataset = VADAudioDataset(
        audio_paths=audio_paths, audio_dir=audio_dir, sample_rate=sample_rate
    )
    vad_dataloader = torch.utils.data.DataLoader(
        vad_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vad_collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    json_encoder = msgspec.json.Encoder()
    msgpack_encoder = msgspec.msgpack.Encoder()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []

    for i, audio_dict in enumerate(tqdm(vad_dataloader, desc="Running VAD on audio files")):
        audio = audio_dict["audio"][0]
        audio_path = audio_dict["audio_path"][0]
        vad_output = run_vad(
            audio_path=audio_path,
            audio_dir=audio_dir,
            model=model,
            audio=audio,
            chunk_size=chunk_size,
            speeches=speeches,
            metadata=metadata[i] if metadata is not None else None,
        )
        results.append(vad_output)

        if save_json:
            vad_msgspec = json_encoder.encode(vad_output)
            vad_msgspec = msgspec.json.format(vad_msgspec, indent=2)
            json_path = Path(output_dir) / (Path(audio_path).stem + ".json")
            with open(json_path, "wb") as f:
                f.write(vad_msgspec)

        if save_msgpack:
            vad_msgpack = msgpack_encoder.encode(vad_output)
            msgpack_path = Path(output_dir) / (Path(audio_path).stem + ".msgpack")
            with open(msgpack_path, "wb") as f:
                f.write(vad_msgpack)

        yield vad_output
