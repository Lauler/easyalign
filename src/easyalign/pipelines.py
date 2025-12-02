from pathlib import Path

import msgspec
import numpy as np
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor

from easyalign.alignment.pytorch import (
    align_chunks,
    align_speech,
)
from easyalign.alignment.utils import add_logits_to_metadata, get_output_logits_length
from easyalign.data.collators import alignment_collate_fn, audiofile_collate_fn, vad_collate_fn
from easyalign.data.datamodel import AudioMetadata, SpeechSegment
from easyalign.data.dataset import AudioFileDataset, JSONMetadataDataset, VADAudioDataset
from easyalign.data.utils import pad_probs
from easyalign.utils import save_emissions_and_metadata
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

        if not Path(audio_dir, audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

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
        speeches (list): Optional list of SpeechSegment objects to run VAD and alignment only on
            specific segments of the audio. Alignment can generally be improved if VAD/alignment is
            only performed on the segments of the audio that overlap with text transcripts.
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
    alignment_strategy: str = "speech",
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
        alignment_strategy: Strategy for aligning features to text. One of 'speech' or 'chunk'.
            If `speech`, audio is split into `chunk_size` sized chunks based on SpeechSegments
            If `chunk`, audio is taken from existing VAD chunks.
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
        alignment_strategy=alignment_strategy,
    )

    file_dataloader = torch.utils.data.DataLoader(
        file_dataset,
        batch_size=batch_size_files,
        shuffle=False,
        collate_fn=audiofile_collate_fn,
        num_workers=num_workers_files,
        prefetch_factor=prefetch_factor_files,
    )

    maximum_nr_logits = get_output_logits_length(
        audio_frames=int(file_dataset.chunk_size * file_dataset.sr),
        chunk_size=file_dataset.chunk_size,
        conv_kernel=model.config.conv_kernel,
        conv_stride=model.config.conv_stride,
        add_adapter=getattr(model.config, "add_adapter", False),
        num_adapter_layers=getattr(model.config, "num_adapter_layers", 0),
        adapter_stride=getattr(model.config, "adapter_stride", 2),
        sample_rate=file_dataset.sr,
    )

    for features in file_dataloader:
        slice_dataset = features[0]["dataset"]
        metadata = slice_dataset.metadata

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
            probs = pad_probs(probs, maximum_nr_logits=maximum_nr_logits)

            probs_list.append(probs)
            speech_ids.extend(batch["speech_ids"])

        # Count the number of non-padding output logits for each chunk and add to metadata
        metadata = add_logits_to_metadata(
            model=model,
            metadata=metadata,
            chunk_size=file_dataset.chunk_size,
            sample_rate=file_dataset.sr,
        )
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
    alignment_strategy: str = "speech",
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
        chunk_size: When `alignment_strategy` is set to `speech`, SpeechSegments are split into
            `chunk_size` sized chunks for feature extraction.
        alignment_strategy: Strategy for aligning features to text. One of 'speech' or 'chunk'.
            If `speech`, audio is split into `chunk_size` sized chunks based on SpeechSegments.
            If `chunk`, audio is taken from existing VAD chunks.
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
            alignment_strategy=alignment_strategy,
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


def alignment_pipeline_generator(
    dataloader: torch.utils.data.DataLoader,
    text_normalizer: callable,
    processor: Wav2Vec2Processor,
    tokenizer=None,
    emissions_dir: str = "output/emissions",
    output_dir: str = "output/alignments",
    alignment_strategy: str = "speech",
    start_wildcard: bool = False,
    end_wildcard: bool = False,
    blank_id: int = 0,
    word_boundary: str = "|",
    chunk_size: int = 30,
    save_json: bool = True,
    save_msgpack: bool = False,
    delete_emissions: bool = False,
    remove_wildcards: bool = True,
    add_leading_space: bool = False,
    device="cuda",
):
    """
    Perform alignment on speech segments or VAD chunks using emissions.

    Speech based alignment is typically used when aligning human transcriptions,
    while chunk based alignment is typically used to align the output of ASR models.

    Args:
        dataloader: DataLoader loading AudioMetadata objects from JSON or Msgpack files.
        text_normalizer: Function to normalize text according to regex rules.
        processor: Wav2Vec2Processor to preprocess the audio.
        tokenizer: Optional tokenizer for custom segmentation of text (e.g. sentence segmentation,
            or paragraph segmentation). The tokenizer should either i) be a PunktTokenizer from
            nltk, or ii) directly return a list of spans (start_char, end_char) when called on a
            string.
        emissions_dir: Directory where the emissions are stored.
        output_dir: Directory to save alignment outputs.
        alignment_strategy: Strategy for aligning features to text. One of 'speech' or 'chunk'.
            If `speech`, alignments are performed on SpeechSegments.
            If `chunk`, alignments are performed on VAD chunks.
        start_wildcard: Whether to add a wildcard token at the start of the segments.
        end_wildcard: Whether to add a wildcard token at the end of the segments.
        blank_id: ID of the blank token in the tokenizer.
        word_boundary: Token indicating word boundaries in the tokenizer.
        chunk_size: maximum chunk size in seconds.
        delete_emissions: Whether to delete the emissions files after alignment to save space.
        remove_wildcards: Whether to remove wildcard tokens from the final alignment.
        add_leading_space: Whether to add a leading space to the text segments (only used
            for speech based alignment when speech text is supplied as lists).
        device: Device to run the alignment on (e.g. "cuda" or "cpu").

    Yields:
        List of aligned segments with word-level timestamps for each audio file.
    """

    if alignment_strategy == "speech":
        align_func = align_speech
    elif alignment_strategy == "chunk":
        align_func = align_chunks

    mapping = align_func(
        dataloader=dataloader,
        text_normalizer=text_normalizer,
        processor=processor,
        tokenizer=tokenizer,
        emissions_dir=emissions_dir,
        output_dir=output_dir,
        start_wildcard=start_wildcard,
        end_wildcard=end_wildcard,
        blank_id=blank_id,
        word_boundary=word_boundary,
        chunk_size=chunk_size,
        save_json=save_json,
        save_msgpack=save_msgpack,
        delete_emissions=delete_emissions,
        remove_wildcards=remove_wildcards,
        device=device,
    )

    yield from mapping
