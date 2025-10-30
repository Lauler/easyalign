from pathlib import Path

import msgspec
import numpy as np
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor

from easyalign.alignment.pytorch import segment_speech_probs
from easyalign.data.collators import alignment_collate_fn, audiofile_collate_fn, vad_collate_fn
from easyalign.data.datamodel import AudioMetadata, SpeechSegment
from easyalign.data.dataset import AudioFileDataset, JSONMetadataDataset, VADAudioDataset
from easyalign.data.utils import pad_probs
from easyalign.vad.vad import run_vad


def vad_pipeline_generator(
    model,
    audio_paths: list,
    audio_dir: str,
    speeches: list[list[SpeechSegment]] | None = None,
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
        speeches: Optional list of SpeechSegment objects to run VAD only on specific
            segments of the audio. Alignment can generally be improved if VAD/alignment is only
            performed on the segments of the audio that overlap with text transcripts.
        chunk_size: The maximum length chunks VAD will create (seconds).
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
            speeches=speeches[i] if speeches is not None else None,
            metadata=metadata[i] if metadata is not None else None,
        )
        results.append(vad_output)

        if save_json:
            vad_msgspec = json_encoder.encode(vad_output)
            vad_msgspec = msgspec.json.format(vad_msgspec, indent=2)
            json_path = (
                Path(output_dir) / Path(audio_path).parent / (Path(audio_path).stem + ".json")
            )
            with open(json_path, "wb") as f:
                f.write(vad_msgspec)

        if save_msgpack:
            vad_msgpack = msgpack_encoder.encode(vad_output)
            msgpack_path = (
                Path(output_dir) / Path(audio_path).parent / (Path(audio_path).stem + ".msgpack")
            )
            with open(msgpack_path, "wb") as f:
                f.write(vad_msgpack)

        yield vad_output


def vad_pipeline(
    model,
    audio_paths: list,
    audio_dir: str,
    speeches: list[list[SpeechSegment]] | None = None,
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
        audio_dir: Directory where the audio files/dirs are located (if `audio_paths` are relative).
        speeches (list): Optional list of SpeechSegment objects to run VAD only on specific
            segments of the audio. Alignment can generally be improved if VAD/alignment is only
            performed on the segments of the audio that overlap with text transcripts.
        chunk_size: The maximum length chunks VAD will create (seconds).
        sample_rate: The sample rate to resample the audio to before running VAD.
        metadata: Optional dictionary of additional file level metadata to include.
        batch_size: The batch size for the DataLoader.
        num_workers: The number of workers for the DataLoader.
        prefetch_factor: The prefetch factor for the DataLoader.
        save_json: Whether to save the VAD output as JSON files.
        save_msgpack: Whether to save the VAD output as Msgpack files.
        output_dir: Directory to save the JSON/Msgpack files if save_json/save_msgpack is True.
    """

    results = list(
        vad_pipeline_generator(
            model=model,
            audio_paths=audio_paths,
            audio_dir=audio_dir,
            speeches=speeches,
            chunk_size=chunk_size,
            sample_rate=sample_rate,
            metadata=metadata,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            save_json=save_json,
            save_msgpack=save_msgpack,
            output_dir=output_dir,
        )
    )

    return results


def emissions_pipeline_generator(
    model,
    processor: Wav2Vec2Processor,
    metadata: JSONMetadataDataset | list[AudioMetadata] | AudioMetadata,
    audio_dir: str,
    sample_rate: int = 16000,
    chunk_size: int = 30,
    use_vad: bool = True,
    batch_size_files: int = 1,
    num_workers_files: int = 1,
    prefetch_factor_files: int = 2,
    batch_size_features: int = 8,
    num_workers_features: int = 4,
    save_json: bool = True,
    save_msgpack: bool = False,
    save_emissions: bool = True,
    return_emissions: bool = False,
    output_dir: str = "output/emissions",
):
    """
    Run emissions extraction pipeline on the given audio files and save results to file. If `return_emissions`
    is True, function becomes a generator that yields tuples of (metadata, emissions) for each audio file.

    Args:
        model: The loaded ASR model.
        metadata: List of AudioMetadata objects or paths to JSON files.
        audio_dir: Directory with audio files
        sample_rate: Sample rate to resample audio to. Default 16000.
        chunk_size: When VAD is not used, SpeechSegments are naively split into
            `chunk_size` sized chunks for feature extraction.
        use_vad: Whether to use VAD-based chunks (if available in metadata), or just
            naïvely split the audio of speech segments into `chunk_size` chunks.
        batch_size_files: Batch size for the file DataLoader.
        num_workers_files: Number of workers for the file DataLoader.
        prefetch_factor_files: Prefetch factor for the file DataLoader.
        batch_size_features: Batch size for the feature DataLoader.
        num_workers_features: Number of workers for the feature DataLoader.
        save_json: Whether to save the emissions output as JSON files.
        save_msgpack: Whether to save the emissions output as Msgpack files.
        save_emissions: Whether to save the raw emissions as .npy files.
        return_emissions: Whether to return the emissions as a list of numpy arrays.
        output_dir: Directory to save the output files if saving is enabled.

    """
    file_dataset = AudioFileDataset(
        metadata=metadata,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        processor=processor,
        chunk_size=chunk_size,
        use_vad=use_vad,
    )

    file_dataloader = torch.utils.data.DataLoader(
        file_dataset,
        batch_size=batch_size_files,
        shuffle=False,
        collate_fn=audiofile_collate_fn,
        num_workers=num_workers_files,
        prefetch_factor=prefetch_factor_files,
    )

    for features in file_dataloader:
        slice_dataset = features[0]["dataset"]
        feature_dataloader = torch.utils.data.DataLoader(
            slice_dataset,
            batch_size=batch_size_features,
            shuffle=False,
            collate_fn=alignment_collate_fn,
            num_workers=num_workers_features,
        )

        probs_list = []
        speech_ids = []

        for batch in feature_dataloader:
            features = batch["features"].half().to("cuda")

            with torch.inference_mode():
                logits = model(features).logits

            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            probs = pad_probs(
                probs, chunk_size=file_dataset.chunk_size, sample_rate=file_dataset.sr
            )

            probs_list.append(probs)
            speech_ids.extend(batch["speech_ids"])

        metadata = slice_dataset.metadata
        metadata, emissions = save_emissions_and_metadata(
            metadata=metadata,
            probs_list=probs_list,
            speech_ids=speech_ids,
            save_json=save_json,
            save_msgpack=save_msgpack,
            save_emissions=save_emissions,
            return_emissions=return_emissions,
            output_dir=output_dir,
        )

        if return_emissions:
            yield metadata, emissions


def emissions_pipeline(
    model,
    processor: Wav2Vec2Processor,
    metadata: JSONMetadataDataset | list[AudioMetadata] | AudioMetadata,
    audio_dir: str,
    sample_rate: int = 16000,
    chunk_size: int = 30,
    use_vad: bool = True,
    batch_size_files: int = 1,
    num_workers_files: int = 1,
    prefetch_factor_files: int = 2,
    batch_size_features: int = 8,
    num_workers_features: int = 4,
    save_json: bool = True,
    save_msgpack: bool = False,
    save_emissions: bool = True,
    return_emissions: bool = False,
    output_dir: str = "output/emissions",
):
    """
    Run emissions extraction pipeline on the given audio files and save results to file.

    Args:
        model: The loaded ASR model.
        metadata: List of AudioMetadata objects or paths to JSON files.
        audio_dir: Directory with audio files
        sample_rate: Sample rate to resample audio to. Default 16000.
        chunk_size: When VAD is not used, SpeechSegments are naively split into
            `chunk_size` sized chunks for feature extraction.
        use_vad: Whether to use VAD-based chunks (if available in metadata), or just
            naïvely split the audio of speech segments into `chunk_size` chunks.
        batch_size_files: Batch size for the file DataLoader.
        num_workers_files: Number of workers for the file DataLoader.
        prefetch_factor_files: Prefetch factor for the file DataLoader.
        batch_size_features: Batch size for the feature DataLoader.
        num_workers_features: Number of workers for the feature DataLoader.
        save_json: Whether to save the emissions output as JSON files.
        save_msgpack: Whether to save the emissions output as Msgpack files.
        save_emissions: Whether to save the raw emissions as .npy files.
        return_emissions: Whether to return the emissions as a list of numpy arrays.
        output_dir: Directory to save the output files if saving is enabled.

    Returns:
        If `return_emissions` is True, returns a list of tuples (metadata, emissions)
        for each audio file. Otherwise, returns None.
    """

    emissions_output = list(
        emissions_pipeline_generator(
            model=model,
            processor=processor,
            metadata=metadata,
            audio_dir=audio_dir,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            use_vad=use_vad,
            batch_size_files=batch_size_files,
            num_workers_files=num_workers_files,
            prefetch_factor_files=prefetch_factor_files,
            batch_size_features=batch_size_features,
            num_workers_features=num_workers_features,
            save_json=save_json,
            save_msgpack=save_msgpack,
            save_emissions=save_emissions,
            return_emissions=return_emissions,
            output_dir=output_dir,
        )
    )
    return emissions_output


def save_metadata_json(metadata: AudioMetadata, output_dir: str = "output/emissions"):
    audio_path = metadata.audio_path
    json_encoder = msgspec.json.Encoder()
    json_msgspec = json_encoder.encode(metadata)
    json_msgspec = msgspec.json.format(json_msgspec, indent=2)
    json_path = Path(output_dir) / Path(audio_path).parent / (Path(audio_path).stem + ".json")
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "wb") as f:
        f.write(json_msgspec)


def save_metadata_msgpack(metadata: AudioMetadata, output_dir: str = "output/emissions"):
    audio_path = metadata.audio_path
    msgpack_encoder = msgspec.msgpack.Encoder()
    msgpack_msgspec = msgpack_encoder.encode(metadata)
    msgpack_path = (
        Path(output_dir) / Path(audio_path).parent / (Path(audio_path).stem + ".msgpack")
    )
    Path(msgpack_path).parent.mkdir(parents=True, exist_ok=True)
    with open(msgpack_path, "wb") as f:
        f.write(msgpack_msgspec)


def save_emissions_and_metadata(
    metadata: AudioMetadata,
    probs_list: list[np.ndarray],
    speech_ids: list[str],
    save_json: bool = True,
    save_msgpack: bool = False,
    save_emissions: bool = True,
    return_emissions: bool = False,
    output_dir: str = "output/emissions",
) -> tuple[AudioMetadata | None, list[np.ndarray] | None]:
    audio_path = metadata.audio_path
    base_path = Path(audio_path).parent / Path(audio_path).stem
    speech_index = 0

    if save_emissions:
        # Segment the probs according to speech_ids and save each speech's probs separately
        for speech_id, probs in segment_speech_probs(probs_list, speech_ids):
            probs_path = Path(output_dir) / base_path / f"{speech_id}.npy"
            Path(probs_path).parent.mkdir(parents=True, exist_ok=True)

            probs_base_path = Path(probs_path).relative_to(Path(output_dir))
            metadata.speeches[speech_index].probs_path = str(probs_base_path)
            speech_index += 1
            np.save(probs_path, probs)

    if save_json:
        save_metadata_json(metadata, output_dir=output_dir)
    if save_msgpack:
        save_metadata_msgpack(metadata, output_dir=output_dir)

    if return_emissions:
        emissions = []
        for _, probs in segment_speech_probs(probs_list, speech_ids):
            emissions.append(probs)

        return metadata, emissions

    return None, None


def save_alignments(
    metadata: AudioMetadata,
    alignments: list[dict],
    save_json: bool = True,
    save_msgpack: bool = False,
    output_dir: str = "output/alignments",
):
    audio_path = metadata.audio_path
    base_path = Path(audio_path).parent / Path(audio_path).stem

    json_encoder = msgspec.json.Encoder()
    msgpack_encoder = msgspec.msgpack.Encoder()

    for speech, alignment in zip(metadata.speeches, alignments):
        alignment_path = Path(output_dir) / base_path / f"{speech.speech_id}_alignment.json"
        Path(alignment_path).parent.mkdir(parents=True, exist_ok=True)

        alignment_base_path = Path(alignment_path).relative_to(Path(output_dir))
        speech.alignment_path = str(alignment_base_path)

        if save_json:
            alignment_msgspec = json_encoder.encode(alignment)
            alignment_msgspec = msgspec.json.format(alignment_msgspec, indent=2)
            with open(alignment_path, "wb") as f:
                f.write(alignment_msgspec)

        if save_msgpack:
            alignment_msgpack = msgpack_encoder.encode(alignment)
            # Replace .json with .msgpack
            msgpack_path = alignment_path.with_suffix(".msgpack")
            with open(msgpack_path, "wb") as f:
                f.write(alignment_msgpack)
