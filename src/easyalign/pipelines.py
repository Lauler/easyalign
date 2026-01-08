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
from easyalign.data import (
    AudioFileDataset,
    JSONMetadataDataset,
    StreamingAudioFileDataset,
    VADAudioDataset,
)
from easyalign.data.collators import (
    alignment_collate_fn,
    audiofile_collate_fn,
    metadata_collate_fn,
    vad_collate_fn,
)
from easyalign.data.datamodel import AudioMetadata, SpeechSegment
from easyalign.data.utils import pad_probs
from easyalign.text.normalization import default_text_normalize
from easyalign.utils import save_emissions_and_metadata, save_metadata_json, save_metadata_msgpack
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
    return_vad: bool = False,
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

    Yields:
        If `return_vad` is True, yields AudioMetadata objects for each audio file.
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
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "wb") as f:
                f.write(vad_msgspec)

        if save_msgpack:
            vad_msgpack = msgpack_encoder.encode(vad_output)
            msgpack_path = (
                Path(output_dir) / Path(audio_path).parent / (Path(audio_path).stem + ".msgpack")
            )
            msgpack_path.parent.mkdir(parents=True, exist_ok=True)
            with open(msgpack_path, "wb") as f:
                f.write(vad_msgpack)

        if return_vad:
            yield vad_output


def vad_pipeline(
    model,
    audio_paths: list,
    audio_dir: str | None = None,
    speeches: list[list[SpeechSegment]] | None = None,
    chunk_size: int = 30,
    sample_rate: int = 16000,
    metadata: list[dict] | None = None,
    batch_size: int = 1,
    num_workers: int = 1,
    prefetch_factor: int = 2,
    save_json: bool = True,
    save_msgpack: bool = False,
    return_vad: bool = False,
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
        return_vad: Whether to return the VAD output as a list of AudioMetadata objects.
        output_dir: Directory to save the JSON/Msgpack files if save_json/save_msgpack is True.

    Returns:
        If `return_vad` is True, returns a list of AudioMetadata objects for each audio file.
        Otherwise, returns `None`.
    """

    vad_generator = vad_pipeline_generator(
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
        return_vad=return_vad,
        output_dir=output_dir,
    )

    if return_vad:
        return list(vad_generator)
    else:
        # Consume the generator without returning anything, saving files to disk only
        for _ in vad_generator:
            pass

    return None


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
    streaming: bool = False,
    save_json: bool = True,
    save_msgpack: bool = False,
    save_emissions: bool = True,
    return_emissions: bool = False,
    output_dir: str = "output/emissions",
    device: str = "cuda",
):
    """
    Run emissions extraction pipeline on the given audio files and save results to file. If
    `return_emissions` is True, function becomes a generator that yields tuples of
    (metadata, emissions) for each audio file.

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
        streaming: Whether to use streaming audio files.
        save_json: Whether to save the emissions output as JSON files.
        save_msgpack: Whether to save the emissions output as Msgpack files.
        save_emissions: Whether to save the raw emissions as .npy files.
        return_emissions: Whether to return the emissions as a list of numpy arrays.
        output_dir: Directory to save the output files if saving is enabled.
        device: Device to run the model on (e.g. "cuda" or "cpu").

    Yields:
        If `return_emissions` is True, yields tuples of (metadata, emissions) for each audio file.
    """
    if streaming:
        DatasetClass = StreamingAudioFileDataset
    else:
        DatasetClass = AudioFileDataset

    file_dataset = DatasetClass(
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
            features = batch["features"].half().to(device)

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
    streaming: bool = False,
    save_json: bool = True,
    save_msgpack: bool = False,
    save_emissions: bool = True,
    return_emissions: bool = False,
    output_dir: str = "output/emissions",
    device: str = "cuda",
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
        streaming: Whether to use streaming audio files.
        save_json: Whether to save the emissions output as JSON files.
        save_msgpack: Whether to save the emissions output as Msgpack files.
        save_emissions: Whether to save the raw emissions as .npy files.
        return_emissions: Whether to return the emissions as a list of numpy arrays.
        output_dir: Directory to save the output files if saving is enabled.
        device: Device to run the model on (e.g. "cuda" or "cpu").

    Returns:
        If `return_emissions` is True, returns a list of tuples (metadata, emissions)
        for each audio file. Otherwise, returns None.
    """

    emissions_generator = emissions_pipeline_generator(
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
        streaming=streaming,
        save_json=save_json,
        save_msgpack=save_msgpack,
        save_emissions=save_emissions,
        return_emissions=return_emissions,
        output_dir=output_dir,
        device=device,
    )

    if return_emissions:
        return list(emissions_generator)
    else:
        # Consume the generator without returning anything, saving files only
        for _ in emissions_generator:
            pass

    return None


def alignment_pipeline_generator(
    dataloader: torch.utils.data.DataLoader,
    text_normalizer: callable,
    processor: Wav2Vec2Processor,
    tokenizer=None,
    emissions_dir: str = "output/emissions",
    alignment_strategy: str = "speech",
    start_wildcard: bool = False,
    end_wildcard: bool = False,
    blank_id: int = 0,
    word_boundary: str = "|",
    chunk_size: int = 30,
    ndigits: int = 5,
    indent: int = 2,
    save_json: bool = True,
    save_msgpack: bool = False,
    return_alignments: bool = False,
    delete_emissions: bool = False,
    remove_wildcards: bool = True,
    output_dir: str = "output/alignments",
    device: str = "cuda",
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
        alignment_strategy: Strategy for aligning features to text. One of 'speech' or 'chunk'.
            If `speech`, alignments are performed on SpeechSegments.
            If `chunk`, alignments are performed on VAD chunks.
        start_wildcard: Whether to add a wildcard token at the start of the segments.
        end_wildcard: Whether to add a wildcard token at the end of the segments.
        blank_id: ID of the blank token in the tokenizer.
        word_boundary: Token indicating word boundaries in the tokenizer.
        chunk_size: maximum chunk size in seconds.
        ndigits: Number of decimal digits to round the alignment times and scores to.
        indent: Indentation level for saved JSON files. `None` to disable pretty formatting.
        save_json: Whether to save alignment metadata in JSON format.
        save_msgpack: Whether to save alignment metadata in Msgpack format.
        return_alignments: Whether to yield the alignment mappings.
        delete_emissions: Whether to delete the emissions files after alignment to save space.
        remove_wildcards: Whether to remove wildcard tokens from the final alignment.
        add_leading_space: Whether to add a leading space to the text segments (only used
            for speech based alignment when speech text is supplied as lists).
        output_dir: Directory to save alignment outputs.
        device: Device to run the alignment on (e.g. "cuda" or "cpu").

    Yields:
        List of aligned speech segments for each audio file.
    """

    if alignment_strategy == "speech":
        align_func = align_speech
    elif alignment_strategy == "chunk":
        align_func = align_chunks

    for batch in tqdm(dataloader):
        for metadata in batch:
            alignment_mapping = align_func(
                metadata=metadata,
                text_normalizer=text_normalizer,
                processor=processor,
                tokenizer=tokenizer,
                emissions_dir=emissions_dir,
                start_wildcard=start_wildcard,
                end_wildcard=end_wildcard,
                blank_id=blank_id,
                word_boundary=word_boundary,
                chunk_size=chunk_size,
                ndigits=ndigits,
                delete_emissions=delete_emissions,
                remove_wildcards=remove_wildcards,
                device=device,
            )

            if save_json:
                save_metadata_json(metadata, output_dir=output_dir, indent=indent)

            if save_msgpack:
                save_metadata_msgpack(metadata, output_dir=output_dir)

            if return_alignments:
                yield alignment_mapping


def alignment_pipeline(
    dataloader: torch.utils.data.DataLoader,
    text_normalizer: callable,
    processor: Wav2Vec2Processor,
    tokenizer=None,
    alignment_strategy: str = "speech",
    start_wildcard: bool = False,
    end_wildcard: bool = False,
    blank_id: int = 0,
    word_boundary: str = "|",
    chunk_size: int = 30,
    ndigits: int = 5,
    indent: int = 2,
    save_json: bool = True,
    save_msgpack: bool = False,
    return_alignments: bool = False,
    delete_emissions: bool = False,
    remove_wildcards: bool = True,
    emissions_dir: str = "output/emissions",
    output_dir: str = "output/alignments",
    device: str = "cuda",
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
        alignment_strategy: Strategy for aligning features to text. One of 'speech' or 'chunk'.
            If `speech`, alignments are performed on SpeechSegments.
            If `chunk`, alignments are performed on VAD chunks.
        start_wildcard: Whether to add a wildcard token at the start of the segments.
        end_wildcard: Whether to add a wildcard token at the end of the segments.
        blank_id: ID of the blank token in the tokenizer.
        word_boundary: Token indicating word boundaries in the tokenizer.
        chunk_size: maximum chunk size in seconds.
        ndigits: Number of decimal digits to round the alignment times and scores to.
        indent: Indentation level for saved JSON files. `None` to disable pretty formatting.
        save_json: Whether to save alignment metadata in JSON format.
        save_msgpack: Whether to save alignment metadata in Msgpack format.
        return_alignments: Whether to return the alignment mappings.
        delete_emissions: Whether to delete the emissions files after alignment to save space.
        remove_wildcards: Whether to remove wildcard tokens from the final alignment.
        emissions_dir: Directory where the emissions are stored.
        output_dir: Directory to save alignment outputs.
        device: Device to run the alignment on (e.g. "cuda" or "cpu").

    Returns:
        If `return_alignments` is True, returns a list of alignment mappings for each audio file.
        Otherwise, returns `None`.
    """
    align_generator = alignment_pipeline_generator(
        dataloader=dataloader,
        text_normalizer=text_normalizer,
        processor=processor,
        tokenizer=tokenizer,
        emissions_dir=emissions_dir,
        output_dir=output_dir,
        alignment_strategy=alignment_strategy,
        start_wildcard=start_wildcard,
        end_wildcard=end_wildcard,
        blank_id=blank_id,
        word_boundary=word_boundary,
        chunk_size=chunk_size,
        ndigits=ndigits,
        indent=indent,
        save_json=save_json,
        save_msgpack=save_msgpack,
        return_alignments=return_alignments,
        delete_emissions=delete_emissions,
        remove_wildcards=remove_wildcards,
        device=device,
    )

    if return_alignments:
        return list(align_generator)
    else:
        # Consume the generator without returning anything, saving files to disk only
        for _ in align_generator:
            pass

    return None


def pipeline(
    vad_model,
    emissions_model,
    processor: Wav2Vec2Processor,
    audio_paths: list,
    audio_dir: str,
    speeches: list[list[SpeechSegment]] | None = None,
    sample_rate: int = 16000,
    chunk_size: int = 30,
    alignment_strategy: str = "speech",
    text_normalizer: callable = default_text_normalize,
    tokenizer=None,
    start_wildcard: bool = False,
    end_wildcard: bool = False,
    blank_id: int = 0,
    word_boundary: str = "|",
    indent: int = 2,
    ndigits: int = 5,
    batch_size_files: int = 1,
    num_workers_files: int = 2,
    prefetch_factor_files: int = 1,
    batch_size_features: int = 8,
    num_workers_features: int = 4,
    streaming: bool = False,
    save_json: bool = True,
    save_msgpack: bool = False,
    save_emissions: bool = True,
    return_alignments: bool = False,
    delete_emissions: bool = False,
    output_vad_dir: str = "output/vad",
    output_emissions_dir: str = "output/emissions",
    output_alignments_dir: str = "output/alignments",
    device="cuda",
):
    """
    Complete pipeline to run VAD, extract emissions, and perform alignment.

    Args:
        vad_model: The loaded VAD model.
        asr_model: The loaded ASR model.
        processor: Wav2Vec2Processor to preprocess the audio.
        audio_paths: List of paths to audio files (relative to `audio_dir`).
        audio_dir: Base directory with audio files relative to `audio_paths`.
        speeches: List of SpeechSegment objects to run VAD and alignment only on specific
            segments of the audio. If `alignment_strategy` is 'speech', the text needs to be
            supplied in the SpeechSegment objects. If `alignment_strategy` is 'chunk' and ASR
            transcriptions are used, there is no need to supply text in the SpeechSegment objects.
        sample_rate: Sample rate to resample audio to. Default 16000.
        chunk_size: When `alignment_strategy` is set to `speech`, SpeechSegments are split into
            `chunk_size` sized chunks for feature extraction.
        alignment_strategy: Strategy for aligning features to text. One of 'speech' or 'chunk'.
            If `speech`, audio is split into `chunk_size` sized chunks based on SpeechSegments.
            If `chunk`, VAD chunks are used as basis for feature extraction and alignment.
            NOTE: `chunk` currently only works with ASR. The individual VAD chunks won't
            contain the relevant text information for alignment.
        text_normalizer: Function to normalize text according to regex rules.
        tokenizer: Optional tokenizer for custom segmentation of text (e.g. sentence segmentation,
            or paragraph segmentation). The tokenizer should either i) be a PunktTokenizer from
            nltk, or ii) directly return a list of spans (start_char, end_char) when called on a
            string.
        batch_size_files: Batch size for the file DataLoader.
        num_workers_files: Number of workers for the file DataLoader.
        prefetch_factor_files: Prefetch factor for the file DataLoader.
        batch_size_features: Batch size for the feature DataLoader.
        num_workers_features: Number of workers for the feature DataLoader.
        streaming: Whether to use streaming loading of audio files.
        save_json: Whether to save the output files as JSON.
        save_msgpack: Whether to save the output files as Msgpack.
        save_emissions: Whether to save the raw emissions as .npy files.
        return_alignments: Whether to return the alignment mappings.
        delete_emissions: Whether to delete the emissions files after alignment to save space.
        output_vad_dir: Directory to save the VAD output files.
        output_emissions_dir: Directory to save the emissions output files.
        output_alignments_dir: Directory to save alignment output files.
        device: Device to run the alignment on (e.g. "cuda" or "cpu").

    Returns:
        If `return_alignments` is True, returns a list of alignment mappings for each audio file.
        Otherwise, returns `None` (the alignments are saved to disk only).
    """

    json_paths = [Path(p).with_suffix(".json") for p in audio_paths]

    # Step 1: Run VAD
    vad_pipeline(
        model=vad_model,
        audio_paths=audio_paths,
        audio_dir=audio_dir,
        speeches=speeches,
        chunk_size=chunk_size,
        sample_rate=sample_rate,
        batch_size=batch_size_files,
        num_workers=num_workers_files,
        prefetch_factor=prefetch_factor_files,
        save_json=save_json,
        save_msgpack=save_msgpack,
        return_vad=False,
        output_dir=output_vad_dir,
    )

    # Step 2: Extract Emissions
    json_dataset = JSONMetadataDataset(
        json_paths=[str(Path(output_vad_dir) / p) for p in json_paths]
    )

    emissions_pipeline(
        model=emissions_model,
        processor=processor,
        metadata=json_dataset,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        alignment_strategy=alignment_strategy,
        batch_size_files=batch_size_files,
        num_workers_files=num_workers_files,
        prefetch_factor_files=prefetch_factor_files,
        batch_size_features=batch_size_features,
        num_workers_features=num_workers_features,
        streaming=streaming,
        save_json=save_json,
        save_msgpack=save_msgpack,
        save_emissions=save_emissions,
        return_emissions=False,
        output_dir=output_emissions_dir,
    )

    # Step 3: Perform Alignment
    json_dataset = JSONMetadataDataset(
        json_paths=[str(Path(output_emissions_dir) / p) for p in json_paths]
    )
    json_dataloader = torch.utils.data.DataLoader(
        json_dataset,
        batch_size=batch_size_files,
        shuffle=False,
        collate_fn=metadata_collate_fn,
        num_workers=num_workers_files,
        prefetch_factor=prefetch_factor_files,
    )

    alignments = alignment_pipeline(
        dataloader=json_dataloader,
        text_normalizer=text_normalizer,
        processor=processor,
        tokenizer=tokenizer,
        emissions_dir=output_emissions_dir,
        output_dir=output_alignments_dir,
        alignment_strategy=alignment_strategy,
        start_wildcard=start_wildcard,
        end_wildcard=end_wildcard,
        blank_id=blank_id,
        word_boundary=word_boundary,
        chunk_size=chunk_size,
        ndigits=ndigits,
        indent=indent,
        save_json=save_json,
        save_msgpack=save_msgpack,
        return_alignments=return_alignments,
        delete_emissions=delete_emissions,
        remove_wildcards=True,
        device=device,
    )

    return alignments
