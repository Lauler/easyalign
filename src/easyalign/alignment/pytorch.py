import itertools
import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as F
from nltk.tokenize.punkt import PunktSentenceTokenizer
from torchaudio.functional import TokenSpan
from tqdm import tqdm
from transformers.models.wav2vec2.processing_wav2vec2 import Wav2Vec2Processor

from easyalign.alignment.utils import (
    get_output_logits_length,
)
from easyalign.data.datamodel import AlignmentSegment, SpeechSegment, WordSegment
from easyalign.text.normalization import add_deletions_to_mapping, merge_multitoken_expressions
from easyalign.utils import (
    save_metadata_json,
    save_metadata_msgpack,
)

logger = logging.getLogger(__name__)


def align_pytorch(
    normalized_tokens: list[str],
    processor: Wav2Vec2Processor,
    emissions: torch.Tensor,
    start_wildcard: bool,
    end_wildcard: bool,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align audio emissions with text transcripts.

    Args:
        normalized_tokens: List of normalized text that has been tokenized.
        processor: Wav2Vec2Processor instance for tokenization.
        emissions: Tensor of audio emissions (logits) with shape
            (batch, sequence (time), vocab_size).
        start_unknown: If True, adds a star wildcard token at the start of the transcript
            to allow better alignment if the audio starts with other irrelevant speech.
        end_unknown: If True, adds a star wildcard token at the end of the transcript.
        device: Device to run the alignment on (e.g., "cpu" or "cuda").

    Returns:
        alignments: Aligned indices of character tokens (their logit indices in the emissions).
        scores: Alignment scores (probabilities) for the tokens.
    """
    transcript = " ".join(normalized_tokens)
    transcript = transcript.replace("\n", " ").upper()

    if start_wildcard:
        transcript = "* " + transcript
    if end_wildcard:
        transcript = transcript + " *"

    # Further tokenization using the model's tokenizer (usually character-level)
    targets = processor.tokenizer(transcript, return_tensors="pt")["input_ids"]
    targets = targets.to(device)

    # Add star wildcard token to the end of the emissions
    if start_wildcard or end_wildcard:
        # batch, sequence (time), vocab_size
        star_dim = torch.zeros(
            (1, emissions.size(1), 1), device=emissions.device, dtype=emissions.dtype
        )
        emissions = torch.cat((emissions, star_dim), 2)  # Add wildcard star token to the emissions

    alignments, scores = F.forced_align(emissions, targets, blank=0)
    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    # scores = scores.exp()  # convert back to probability
    return alignments, scores


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
    ndigits: int = 5,
    indent: int = 2,
    save_json: bool = True,
    save_msgpack: bool = False,
    return_alignments: bool = False,
    delete_emissions: bool = False,
    remove_wildcards: bool = True,
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
            or paragraph segmentation). The tokenizer should either i) be a PunktTokenizer from
            nltk, or ii) directly return a list of spans (start_char, end_char) when called on a
            string.
        emissions_dir: Directory where the wav2vec2 emissions are stored.
        output_dir: Directory to save alignment outputs.
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
        device: Device to run the alignment on (e.g. "cuda" or "cpu").

    Returns:
        List of aligned segments with word-level timestamps.
    """
    for batch in tqdm(dataloader):
        for metadata in batch:
            chunk_mappings = []
            for speech in metadata.speeches:
                emissions_filepath = Path(emissions_dir) / speech.probs_path
                emissions = np.load(emissions_filepath)

                for i, chunk in enumerate(speech.chunks):
                    normalized_tokens, mapping = text_normalizer(chunk.text)
                    emissions_chunk = emissions[i]
                    emissions_chunk = emissions_chunk[: chunk.num_logits]

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

                    if remove_wildcards:
                        mapping = [m for m in mapping if m["normalized_tokens"] != "*"]

                    mapping = get_segment_alignment(
                        mapping=mapping,
                        original_text=chunk.text,
                        tokenizer=tokenizer,
                        segment_spans=None,
                    )

                    alignment_mapping = encode_alignments(mapping, ndigits=ndigits)

                    chunk_mappings.extend(alignment_mapping)
                    speech.alignments.extend(alignment_mapping)

                if delete_emissions:
                    Path(emissions_filepath).unlink()

            if save_json:
                save_metadata_json(metadata, output_dir=output_dir, indent=indent)

            if save_msgpack:
                save_metadata_msgpack(metadata, output_dir=output_dir)

            if return_alignments:
                yield chunk_mappings


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
    ndigits: int = 5,
    indent: int = 2,
    save_json: bool = True,
    save_msgpack: bool = False,
    return_alignments: bool = False,
    delete_emissions: bool = False,
    remove_wildcards: bool = True,
    device="cuda",
):
    mapping = []
    for batch in tqdm(dataloader):
        for metadata in batch:
            for speech in metadata.speeches:
                emissions_filepath = Path(emissions_dir) / speech.probs_path
                emissions = np.load(emissions_filepath)
                emissions = np.vstack(emissions)

                if speech.text:
                    original_text = speech.text
                else:
                    logger.warning(
                        (
                            f"No text found for speech id {speech.speech_id}. \n\n"
                            f"Skipping alignment for file: {metadata.audio_path}.\n"
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

                if remove_wildcards:
                    mapping = [m for m in mapping if m["normalized_tokens"] != "*"]

                mapping = get_segment_alignment(
                    mapping=mapping,
                    original_text=original_text,
                    tokenizer=tokenizer,
                    segment_spans=speech.text_spans,
                )

                alignment_mapping = encode_alignments(mapping, ndigits=ndigits)
                speech.alignments.extend(alignment_mapping)

                if delete_emissions:
                    Path(emissions_filepath).unlink()

            # Add info to metadata and save

            if save_json:
                save_metadata_json(metadata, output_dir=output_dir, indent=indent)

            if save_msgpack:
                save_metadata_msgpack(metadata, output_dir=output_dir)

            if return_alignments:
                yield alignment_mapping


def format_timestamp(timestamp):
    """
    Convert timestamp in seconds to "hh:mm:ss,ms" format.
    """
    hours = int(timestamp // 3600)
    minutes = int((timestamp % 3600) // 60)
    seconds = int(timestamp % 60)
    milliseconds = int((timestamp % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def unflatten(char_list: list[TokenSpan], word_lengths: list[int]) -> list[list[TokenSpan]]:
    """
    Unflatten a list of character output tokens (TokenSpans) from wav2vec2 into words
    (lists of TokenSpans) based on provided normalized word lengths.

    Args:
        char_list:
            A list of character tokens.
        lengths:
            A list of character lengths of the words (normalized tokens).
    """
    assert len(char_list) == sum(word_lengths)
    word_start = 0
    words = []
    for word_length in word_lengths:
        words.append(char_list[word_start : word_start + word_length])
        word_start += word_length
    return words


def get_word_timing(
    word_span: list[F._alignment.TokenSpan],
    frames_per_logit: float,
    start_segment: float = 0.0,
    sample_rate: int = 16000,
) -> tuple[float, float]:
    """
    Calculate the start and end time of a word span in the original audio file.

    Args:
        word_span: A list of TokenSpan objects that together represent the word span's
            timings in the aligned audio chunk.
        frames_per_logit: The number of audio frames per model output logit. This is the
            total number of frames in our audio chunk divided by the number of
            (non-padding) logit outputs for the chunk.
        start_segment: The start time of the speech segment in the original audio file.
            We offset the start/end time of the word span by this value (in
            case chunking/slicing of the audio was performed).
        sample_rate: The sample rate of the audio file, default 16000.

    """
    start = (word_span[0].start * frames_per_logit) / sample_rate + start_segment
    end = (word_span[-1].end * frames_per_logit) / sample_rate + start_segment

    score = sum(span.score * len(span) for span in word_span)
    length = sum(len(span) for span in word_span)  # Token utterances can last multiple frames
    score = score / length  # Normalize the score by the length of the word span

    return start, end, score


def get_word_spans(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    mapping: list[dict],
    blank: int = 0,
    start_wildcard: bool = True,
    end_wildcard: bool = True,
    word_boundary: str | None = "|",
    processor: Wav2Vec2Processor = None,
) -> tuple[list, list]:
    """
    Merge wav2vec2 token (character level) predictions and get their word spans.

    Args:
        tokens: Tokens predicted by the model.
        scores: Scores for each token.
        mapping: Token mapping information.
        blank: The token ID for the blank (padding) token.
        start_wildcard: Whether to add a start wildcard token, to better account for
            speech in the audio that is not covered by the text.
        end_wildcard: Whether to add an end wildcard token.
        word_boundary: The token used to indicate word boundaries. Usually, this is
            the "|" token. Sometimes, the model is trained without word boundary tokens
            (Pytorch native Wav2Vec2 models).
        processor: The Wav2Vec2Processor used for tokenization.

    Returns:
        A tuple containing (word_spans, updated_mapping).
    """
    if start_wildcard:
        mapping.insert(0, {"normalized_token": "*", "text": "*", "start_char": 0, "end_char": 0})
    if end_wildcard:
        mapping.append(
            {
                "normalized_token": "*",
                "text": "*",
                "start_char": mapping[-1]["end_char"] + 1,
                "end_char": mapping[-1]["end_char"] + 1,
            }
        )

    token_spans = F.merge_tokens(tokens, scores, blank=blank)

    if word_boundary:
        assert processor is not None, (
            "Wav2Vec2 Processor must be provided if word_boundary is specified."
        )
        # Find the token ID for the word boundary token
        word_boundary_id = processor.tokenizer.convert_tokens_to_ids(word_boundary)
        # Remove all TokenSpan where token==word_boundary_id
        token_spans = [s for s in token_spans if s.token != word_boundary_id]

    # Unflatten the token spans based on the normalized tokens' lengths
    word_spans = unflatten(token_spans, [len(token["normalized_token"]) for token in mapping])

    return word_spans, mapping


def get_segment_alignment(
    mapping: list[dict],
    original_text: str,
    tokenizer=None,
    segment_spans: list[tuple[int, int]] | None = None,
):
    """
    Get alignment timestamps for any arbitrary segmentation of the original text.
    By default, this function performs a sentence span tokenization if user does
    not provide custom segment spans.

    Args:
        mapping: A list of dictionaries containing the original text tokens that
            have been aligned with the audio, together with character indices and timestamps.
        original_text: The original unnormalized text that was aligned with the audio.
        tokenizer: A PunktSentenceTokenizer instance to tokenize the original text
            into sentences (if segment_spans are not provided). Alternatively, a callable
            function that takes the original text as input and returns a list of
            (start_char, end_char) tuples for each segment.
        segment_spans: Optional list of tuples containing the start and end character
            indices of custom segments in the original text.

    Returns:
        A list of dictionaries containing the start and end timestamps for each segment,
        along with the original text of the segment.
        dict keys:
            - "start_segment": Start timestamp of the segment.
            - "end_segment": End timestamp of the segment.
            - "text": The original text of the segment.
    """

    if not segment_spans:
        # If user does not provide segment spans, we use the tokenizer to get segment spans
        if isinstance(tokenizer, PunktSentenceTokenizer):
            # Use the PunktSentenceTokenizer to get sentence spans
            segment_spans = tokenizer.span_tokenize(original_text)
        elif callable(tokenizer):
            # Use a user supplied custom tokenizer to get custom (start_char, end_char) spans
            segment_spans = tokenizer(original_text)
        else:
            start_char = mapping[0]["start_char"]
            end_char = mapping[-1]["end_char"]
            segment_spans = [(start_char, end_char)]  # Single segment with entire text

    segment_mapping = []
    remaining_tokens = mapping.copy()
    previous_tokens = []  # List to keep track of tokens that have been removed from the mapping

    for span in segment_spans:
        start_segment_index = span[0]  # Character index in the original text
        end_segment_index = span[1]
        start_segment_time = None
        end_segment_time = None
        segment_tokens = []

        while remaining_tokens:
            token = remaining_tokens[0]
            segment_tokens.append(token)

            # Check if start_segment_index falls within this token's character range
            if (
                start_segment_time is None
                and start_segment_index >= token["start_char"]
                and start_segment_index < token["end_char_extended"]
            ):
                start_segment_time = assign_segment_time(
                    current_token=token,
                    token_list=remaining_tokens,
                    fallback_direction="next",
                )
                start_segment_extended_index = token["start_char"]

            # Check if end_segment_index falls within this token's character range
            if (
                end_segment_time is None
                and end_segment_index > token["start_char"]
                and end_segment_index <= token["end_char_extended"]
            ):
                end_segment_time = assign_segment_time(
                    current_token=token,
                    token_list=previous_tokens if previous_tokens else remaining_tokens,
                    fallback_direction="previous",
                )
                end_segment_extended_index = token["end_char_extended"]

            # If we have both timestamps, we can create the segment and break the loop
            if start_segment_time is not None and end_segment_time is not None:
                segment_mapping.append(
                    {
                        "start_segment": start_segment_time,
                        "end_segment": end_segment_time,
                        "text": original_text[start_segment_index:end_segment_index],
                        "text_span_full": original_text[
                            start_segment_extended_index:end_segment_extended_index
                        ],
                        "tokens": segment_tokens,
                    }
                )
                previous_tokens.append(remaining_tokens.pop(0))
                break

            previous_tokens.append(remaining_tokens.pop(0))

    return segment_mapping


def assign_segment_time(
    current_token: dict,
    token_list: list[dict],
    fallback_direction: str = "next",
):
    """
    Assign a timestamp for the segment based on the current token's metadata.

    If alignment timestamps are missing for the current token, we search for the
    closest available token that has a timestamp (either among future tokens in
    the `segment_mapping`, or the previous tokens in the `previous_removed` list).

    Args:
        current_token: The current token dictionary containing the token's metadata.
        token_list: A list of token alignments (dictionaries) that acts as fallback
            when the current token has no timestamp.
        fallback_direction: The direction to search for a timestamp ("next" or "previous"
            tokens) as a fallback when the current token has no timestamp.
    Returns:
        The start or end time of the segment. If no timestamp is found, returns None.
    """
    time = (
        current_token["start_time"] if fallback_direction == "next" else current_token["end_time"]
    )

    # We start searching from the first or last token in the list, depending on the direction.
    token_idx = 0 if fallback_direction == "next" else -1
    index_increment = 1 if fallback_direction == "next" else -1  # Move forward or backward

    # Loop is skipped if the current token already has a timestamp.
    while time is None and abs(token_idx) < len(token_list):
        try:
            time = (
                token_list[token_idx]["start_time"]
                if fallback_direction == "next"
                else token_list[token_idx]["end_time"]
            )
            token_idx += index_increment
        except IndexError:
            # If we reach the end of the list, return None
            return None

    return time


def join_word_timestamps(
    word_spans: list[list[F.TokenSpan]],
    mapping: list[dict],
    speech: SpeechSegment,
    chunk_size: int = 20,
    start_segment: float = 0.0,
) -> list[dict]:
    """
    Join word spans from the alignment with the normalized token mapping, adding timestamps
    to the mapping.

    Args:
        word_spans: List of lists of TokenSpan objects representing the aligned words.
        mapping: List of dictionaries containing the original normalized text tokens that
            have been aligned with the audio (together with character indices relative
            to the original text).
        audio_length: Length of the audio input in frames.
        chunk_size: Size of the audio chunks in seconds (when doing batched inference).
        start_segment: Start time of the audio segment inside the audio file (in seconds).

    Returns:
        An updated mapping with start and end times for each normalized token.
    """

    frames_per_logit = None
    if speech.audio_frames is None:
        # Chunks are aligned independently
        audio_frames = sum([chunk.audio_frames for chunk in speech.chunks])

        logit_lengths = []
        for chunk in speech.chunks:
            logit_length = get_output_logits_length(chunk.audio_frames, chunk_size=chunk_size)
            logit_lengths.append(logit_length)

        frames_per_logit = audio_frames / sum(logit_lengths)
    else:
        # Whole audio segment is aligned at once, and we chunk according to chunk_size
        audio_frames = speech.audio_frames

        frames_per_logit = audio_frames / get_output_logits_length(
            audio_frames, chunk_size=chunk_size
        )

    for aligned_token, normalized_token in zip(word_spans, mapping):
        start_time, end_time, score = get_word_timing(
            aligned_token, frames_per_logit, start_segment=start_segment
        )
        normalized_token["start_time"] = start_time
        normalized_token["end_time"] = end_time
        normalized_token["score"] = score

    return mapping


def encode_alignments(
    mapping: list[dict],
    ndigits: int = 5,
):
    def round_floats(obj, ndigits=ndigits):
        if isinstance(obj, float):
            return round(obj, ndigits)
        elif isinstance(obj, np.floating):
            return round(float(obj), ndigits)
        return obj

    alignment_segments = []

    for segment in mapping:
        segment_words = []
        word_scores = []

        for token in segment["tokens"]:
            segment_words.append(
                WordSegment(
                    text=token["text_span_full"],
                    start=round_floats(token["start_time"]),
                    end=round_floats(token["end_time"]),
                    score=round_floats(token["score"]),
                )
            )
            word_scores.append(token["score"])

        alignment_segment = AlignmentSegment(
            start=round_floats(segment["start_segment"]),
            end=round_floats(segment["end_segment"]),
            duration=round_floats(segment["end_segment"] - segment["start_segment"]),
            words=segment_words,
            text=segment["text_span_full"],
            score=round_floats(np.mean(word_scores)) if word_scores else None,
        )

        alignment_segments.append(alignment_segment)

    return alignment_segments
