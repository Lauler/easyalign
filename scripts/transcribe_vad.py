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
from easyalign.data.collators import audiofile_collate_fn, transcribe_collate_fn
from easyalign.data.datamodel import AudioMetadata, SpeechSegment
from easyalign.data.dataset import AudioFileDataset, JSONMetadataDataset
from easyalign.pipelines import emissions_pipeline, vad_pipeline
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


def align_speech(
    dataloader,
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
    delete_emissions: bool = False,
    add_leading_space: bool = False,
    device="cuda",
):
    mapping = []
    for batch in tqdm(dataloader):
        for metadata in batch:
            for speech in metadata.speeches:
                emissions_filepath = Path(emissions_dir) / speech.probs_path
                emissions = np.load(emissions_filepath)
                emissions = np.vstack(emissions)

                if len(speech.text) == 1:
                    original_text = speech.text[0]
                elif len(speech.text) > 1:
                    # If the user tokenized the text into multiple segments, concatenate them
                    # for alignment
                    if add_leading_space:
                        # Add leading space for all except the first segment
                        original_text = "".join(
                            [speech.text[0]] + [" " + t for t in speech.text[1:]]
                        )
                    else:
                        original_text = "".join(speech.text)
                else:
                    logger.warning(
                        (
                            f"No text found for speech id {speech.speech_id} in"
                            f"{metadata.audio_path}. Skipping alignment."
                        )
                    )
                    continue

                normalized_tokens, mapping = text_normalizer(original_text)
                tokens, scores = align_pytorch(
                    normalized_tokens=normalized_tokens,
                    processor=processor,
                    emissions=torch.tensor(emissions).to(device).unsqueeze(0),
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
                    start_segment=speech.start,
                )

                mapping = merge_multitoken_expressions(mapping)
                mapping = add_deletions_to_mapping(mapping, original_text)

                mapping = get_segment_alignment(
                    mapping=mapping,
                    original_text=original_text,
                    tokenizer=tokenizer,
                    segment_spans=speech.text_spans,
                )

                if delete_emissions:
                    Path(emissions_filepath.parent).unlink()

            # Add info to metadata and save

    return mapping


if __name__ == "__main__":
    model = WhisperForConditionalGeneration.from_pretrained(
        "KBLab/kb-whisper-large", torch_dtype=torch.float16
    ).to("cuda")
    model_vad = load_vad_model()

    vad_outputs = vad_pipeline(
        model=model_vad,
        audio_paths=["audio_mono_120.wav"],
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
        use_vad=True,
    )

    file_dataloader = torch.utils.data.DataLoader(
        file_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=audiofile_collate_fn,
        num_workers=2,
        prefetch_factor=2,
    )

    transcription_texts = []
    for features in file_dataloader:
        slice_dataset = features[0]["dataset"]
        metadata = features[0]["dataset"].metadata

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
