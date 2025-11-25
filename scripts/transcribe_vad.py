import logging
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
from nltk.tokenize import PunktTokenizer
from tqdm import tqdm
from transformers import (
    AutoModelForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from easyalign.alignment.pytorch import (
    align_pytorch,
    get_segment_alignment,
    get_word_spans,
    join_word_timestamps,
)
from easyalign.alignment.utils import get_output_logits_length
from easyalign.data.collators import (
    audiofile_collate_fn,
    metadata_collate_fn,
    transcribe_collate_fn,
)
from easyalign.data.datamodel import AlignmentSegment, AudioMetadata, SpeechSegment, WordSegment
from easyalign.data.dataset import AudioFileDataset, JSONMetadataDataset
from easyalign.pipelines import (
    emissions_pipeline,
    save_metadata_json,
    save_metadata_msgpack,
    vad_pipeline,
)
from easyalign.text.normalization import (
    SpanMapNormalizer,
    add_deletions_to_mapping,
    merge_multitoken_expressions,
)
from easyalign.text.tokenizer import load_tokenizer
from easyalign.vad.pyannote import load_vad_model

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def text_normalizer(text: str) -> str:
    normalizer = SpanMapNormalizer(text)
    normalizer.transform(r"\(.*?\)", "")  # Remove parentheses and their content
    normalizer.transform(r"\s[^\w\s]\s", " ")  # Remove punctuation between whitespace
    normalizer.transform(r"[^\w\s]", "")  # Remove punctuation and special characters
    normalizer.transform(r"\s+", " ")  # Normalize whitespace to a single space
    normalizer.transform(r"^\s+|\s+$", "")  # Strip leading and trailing whitespace
    normalizer.transform(r"\w+", lambda m: m.group().lower())

    mapping = normalizer.get_token_map()
    normalized_tokens = [item["normalized_token"] for item in mapping]
    return normalized_tokens, mapping


def encode_alignments(
    mapping: list[dict],
):
    alignment_segments = []

    for segment in mapping:
        segment_words = []
        word_scores = []

        for token in segment["tokens"]:
            segment_words.append(
                WordSegment(
                    text=token["text_span_full"],
                    start=token["start_time"],
                    end=token["end_time"],
                    score=token["score"],
                )
            )
            word_scores.append(token["score"])

        alignment_segment = AlignmentSegment(
            start=segment["start_segment"],
            end=segment["end_segment"],
            duration=segment["end_segment"] - segment["start_segment"],
            words=segment_words,
            text=segment["text_span_full"],
            score=np.mean(word_scores) if word_scores else None,
        )

        alignment_segments.append(alignment_segment)

    return alignment_segments


def align_chunks(
    dataloader: torch.utils.data.DataLoader,
    text_normalizer: callable,
    processor: Wav2Vec2Processor,
    tokenizer=None,
    emissions_dir: str = "output/emissions",
    output_dir: str = "output/alignments",
    start_wildcard: bool = False,
    end_wildcard: bool = False,
    blank_id: int = 0,
    word_boundary: str = "|",
    chunk_size: int = 30,
    save_json: bool = True,
    save_msgpack: bool = False,
    delete_emissions: bool = False,
    device="cuda",
):
    """
    Perform alignment on VAD chunks using wav2vec2 emissions. Chunk based alignment is typically
    used to align the output of ASR models such as Whisper.

    Args:
        dataloader: DataLoader providing AudioMetadata objects with speech segments and chunks.
        text_normalizer: Function to normalize text according to regex rules.
        processor: Wav2Vec2Processor to preprocess the audio.
        tokenizer: Optional tokenizer for custom segmentation of text (e.g. sentence segmentation,
            or paragraph segmentation). The tokenizer should either i) be a PunktTokenizer from nltk,
            or ii) directly return a list of spans (start_char, end_char) when called on a string.
        emissions_dir: Directory where the wav2vec2 emissions are stored.
        output_dir: Directory to save alignment outputs.
        start_wildcard: Whether to add a wildcard token at the start of the segments.
        end_wildcard: Whether to add a wildcard token at the end of the segments.
        blank_id: ID of the blank token in the tokenizer.
        word_boundary: Token indicating word boundaries in the tokenizer.
        chunk_size: maximum chunk size in seconds.
        delete_emissions: Whether to delete the emissions files after alignment to save space.
        device: Device to run the alignment on (e.g. "cuda" or "cpu").

    Returns:
        List of aligned segments with word-level timestamps.
    """
    chunk_mappings = []
    for batch in tqdm(dataloader):
        for metadata in batch:
            for speech in metadata.speeches:
                emissions_filepath = Path(emissions_dir) / speech.probs_path
                emissions = np.load(emissions_filepath)

                for i, chunk in enumerate(speech.chunks):
                    normalized_tokens, mapping = text_normalizer(chunk.text)
                    emissions_chunk = emissions[i]

                    tokens, scores = align_pytorch(
                        normalized_tokens=normalized_tokens,
                        processor=processor,
                        emissions=torch.tensor(emissions_chunk).to(device).unsqueeze(0),
                        start_wildcard=start_wildcard,
                        end_wildcard=end_wildcard,
                        device=device,
                    )

                    word_spans, mapping = get_word_spans(
                        tokens=tokens,
                        scores=scores,
                        mapping=mapping,
                        blank=blank_id,
                        start_wildcard=start_wildcard,
                        end_wildcard=end_wildcard,
                        word_boundary=word_boundary,
                        processor=processor,
                    )

                    mapping = join_word_timestamps(
                        word_spans=word_spans,
                        mapping=mapping,
                        speech=speech,
                        chunk_size=chunk_size,
                        start_segment=chunk.start,
                    )

                    mapping = merge_multitoken_expressions(mapping)
                    mapping = add_deletions_to_mapping(mapping, chunk.text)

                    mapping = get_segment_alignment(
                        mapping=mapping,
                        original_text=chunk.text,
                        tokenizer=tokenizer,
                        segment_spans=None,
                    )

                    chunk_mappings.extend(mapping)
                    speech.alignments.extend(encode_alignments(mapping))

            if save_json:
                save_metadata_json(metadata, output_dir=output_dir)

            if save_msgpack:
                save_metadata_msgpack(metadata, output_dir=output_dir)

    return chunk_mappings


if __name__ == "__main__":
    model = WhisperForConditionalGeneration.from_pretrained(
        "KBLab/kb-whisper-large", torch_dtype=torch.float16
    ).to("cuda")
    model_vad = load_vad_model()

    vad_outputs = vad_pipeline(
        model=model_vad,
        audio_paths=["audio_80.wav"],
        audio_dir="data",
        speeches=None,
        chunk_size=30,
        sample_rate=16000,
        metadata=None,
        batch_size=1,
        num_workers=1,
        prefetch_factor=2,
        save_json=True,
        save_msgpack=False,
        output_dir="output/vad",
    )

    json_dataset = JSONMetadataDataset(json_paths=list(Path("output/vad").rglob("*.json")))

    processor = WhisperProcessor.from_pretrained("kblab/kb-whisper-large")
    file_dataset = AudioFileDataset(
        metadata=json_dataset,
        processor=processor,
        sample_rate=16000,
        chunk_size=30,
        alignment_strategy="chunk",
    )

    file_dataloader = torch.utils.data.DataLoader(
        file_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=audiofile_collate_fn,
        num_workers=2,
        prefetch_factor=2,
    )

    for features in file_dataloader:
        slice_dataset = features[0]["dataset"]
        metadata = features[0]["dataset"].metadata
        transcription_texts = []

        feature_dataloader = torch.utils.data.DataLoader(
            slice_dataset,
            batch_size=4,
            num_workers=2,
            prefetch_factor=2,
            collate_fn=transcribe_collate_fn,
        )

        for batch in feature_dataloader:
            with torch.inference_mode():
                batch = batch["features"].to("cuda").half()
                predicted_ids = model.generate(
                    batch,
                    return_dict_in_generate=True,
                    task="transcribe",
                    language="sv",
                    output_scores=True,
                    max_length=250,
                )

                transcription = processor.batch_decode(
                    predicted_ids["sequences"], skip_special_tokens=True
                )

                transcription_texts.extend(transcription)

        for i, speech in enumerate(metadata.speeches):
            for j, chunk in enumerate(speech.chunks):
                chunk.text = transcription_texts[j]

        # Write final transcription to file with msgspec serialization
        output_path = Path("output/transcriptions") / Path(metadata.audio_path).with_suffix(
            ".json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_metadata_json(metadata, output_dir="output/transcriptions")

    # Align with wav2vec2
    model = (
        AutoModelForCTC.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish").to("cuda").half()
    )
    processor = Wav2Vec2Processor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish")
    json_dataset = JSONMetadataDataset(
        json_paths=list(Path("output/transcriptions").rglob("*.json"))
    )

    get_output_logits_length(
        audio_frames=719,
        chunk_size=30,
        conv_kernel=model.config.conv_kernel,
        conv_stride=model.config.conv_stride,
    )

    emissions_output = emissions_pipeline(
        model=model,
        processor=processor,
        metadata=json_dataset,
        audio_dir="data",
        sample_rate=16000,
        chunk_size=30,
        alignment_strategy="chunk",
        batch_size_files=1,
        num_workers_files=2,
        prefetch_factor_files=2,
        batch_size_features=4,
        num_workers_features=4,
        save_json=True,
        save_msgpack=False,
        save_emissions=True,
        return_emissions=True,
        output_dir="output/emissions",
    )

    json_dataset = JSONMetadataDataset(json_paths=list(Path("output/emissions").rglob("*.json")))
    audiometa_loader = torch.utils.data.DataLoader(
        json_dataset,
        batch_size=1,
        num_workers=4,
        prefetch_factor=2,
        collate_fn=metadata_collate_fn,
    )

    mapping = align_chunks(
        dataloader=audiometa_loader,
        text_normalizer=text_normalizer,
        processor=processor,
        tokenizer=None,
        emissions_dir="output/emissions",
        output_dir="output/alignments",
        start_wildcard=True,
        end_wildcard=True,
        blank_id=0,
        word_boundary="|",
        chunk_size=30,
        delete_emissions=False,
        device="cuda",
    )

    # Input random data to wav2vec2 model to verify it works
    test_input = torch.randn(1, int(720)).to("cuda").half()
    with torch.inference_mode():
        test_output = model(test_input).logits
        print(f"Test output shape: {test_output.shape}")
