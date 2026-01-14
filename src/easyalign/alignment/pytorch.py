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

    Parameters
    ----------
    normalized_tokens : list of str
        List of normalized text that has been tokenized.
    processor : Wav2Vec2Processor
        Wav2Vec2Processor instance for tokenization.
    emissions : torch.Tensor
        Tensor of audio emissions (logits) with shape
        (batch, sequence (time), vocab_size).
    start_wildcard : bool
        If True, adds a star wildcard token at the start of the transcript
        to allow better alignment if the audio starts with other irrelevant speech.
    end_wildcard : bool
        If True, adds a star wildcard token at the end of the transcript.
    device : str
        Device to run the alignment on (e.g., "cpu" or "cuda").

    Returns
    -------
    torch.Tensor
        Aligned indices of character tokens (their logit indices in the emissions).
    torch.Tensor
        Alignment scores (probabilities) for the tokens.
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
    metadata,
    text_normalizer_fn: callable,
    processor: Wav2Vec2Processor,
    tokenizer=None,
    emissions_dir: str = "output/emissions",
    start_wildcard: bool = False,
    end_wildcard: bool = False,
    blank_id: int = 0,
    word_boundary: str = "|",
    chunk_size: int = 30,
    ndigits: int = 5,
    delete_emissions: bool = False,
    remove_wildcards: bool = True,
    device="cuda",
) -> list:
    """
    Perform alignment on VAD chunks for a single AudioMetadata using wav2vec2 emissions.

    Chunk based alignment is typically used to align the output of ASR models such as Whisper.

    Parameters
    ----------
    metadata : AudioMetadata
        AudioMetadata object containing speech segments and chunks.
    text_normalizer_fn : callable
        Function to normalize text according to regex rules.
    processor : Wav2Vec2Processor
        Wav2Vec2Processor to preprocess the audio.
    tokenizer : object, optional
        Optional tokenizer for custom segmentation of text (e.g. sentence segmentation,
        or paragraph segmentation). The tokenizer should either i) be a PunktTokenizer from
        nltk, or ii) directly return a list of spans (start_char, end_char) when called on a
        string.
    emissions_dir : str, default "output/emissions"
        Directory where the wav2vec2 emissions are stored.
    start_wildcard : bool, default False
        Whether to add a wildcard token at the start of the segments.
    end_wildcard : bool, default False
        Whether to add a wildcard token at the end of the segments.
    blank_id : int, default 0
        ID of the blank token in the tokenizer.
    word_boundary : str, default "|"
        Token indicating word boundaries in the tokenizer.
    chunk_size : int, default 30
        maximum chunk size in seconds.
    ndigits : int, default 5
        Number of decimal digits to round the alignment times and scores to.
    delete_emissions : bool, default False
        Whether to delete the emissions files after alignment to save space.
    remove_wildcards : bool, default True
        Whether to remove wildcard tokens from the final alignment.
    device : str, default "cuda"
        Device to run the alignment on (e.g. "cuda" or "cpu").

    Returns
    -------
    list of AlignmentSegment
        List of aligned segments with word-level timestamps.
    """
    chunk_mappings = []
    for speech in metadata.speeches:
        emissions_filepath = Path(emissions_dir) / speech.probs_path
        emissions = np.load(emissions_filepath)

        for i, chunk in enumerate(speech.chunks):
            normalized_tokens, mapping = text_normalizer_fn(chunk.text)
            emissions_chunk = emissions[i]
            emissions_chunk = emissions_chunk[: chunk.num_logits]

            # Check CTC constraint before alignment
            can_align, T, L, R = is_alignable(
                normalized_tokens, processor, emissions_chunk.shape[0]
            )

            if not can_align:
                logger.warning(
                    f"Alignment infeasible for chunk {i} in file: {metadata.audio_path}, "
                    f"starting at: {chunk.start} and ending at: {chunk.end}.\n\n"
                    f"emission_frames (T={T}) < target_length (L={L}) + repeats (R={R})."
                    f"\n\n (Whisper probably hallucinated a too long text).\n\n"
                    f"Using fallback linear interpolation for timestamps.\n\n"
                )
                alignment_mapping = process_fallback_alignment(
                    mapping, chunk.start, chunk.end, chunk.text, tokenizer, None, ndigits
                )
                chunk_mappings.extend(alignment_mapping)
                speech.alignments.extend(alignment_mapping)
                continue

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

    return chunk_mappings


def align_speech(
    metadata,
    text_normalizer_fn: callable,
    processor: Wav2Vec2Processor,
    tokenizer=None,
    emissions_dir: str = "output/emissions",
    start_wildcard: bool = False,
    end_wildcard: bool = False,
    blank_id: int = 0,
    word_boundary: str = "|",
    chunk_size: int = 30,
    ndigits: int = 5,
    delete_emissions: bool = False,
    remove_wildcards: bool = True,
    device="cuda",
) -> list:
    speech_mappings = []
    for speech in metadata.speeches:
        emissions_filepath = Path(emissions_dir) / speech.probs_path
        emissions = np.load(emissions_filepath)
        emissions = np.vstack(emissions)

        if speech.text:
            original_text = speech.text
        else:
            logger.warning(
                (
                    f"No text found for speech id {speech.speech_id} \n\n"
                    f"Skipping alignment for file: {metadata.audio_path}.\n"
                )
            )
            continue

        normalized_tokens, mapping = text_normalizer_fn(original_text)

        # Check CTC constraint before alignment
        can_align, T, L, R = is_alignable(normalized_tokens, processor, emissions.shape[0])

        if not can_align:
            logger.warning(
                (
                    f"Alignment infeasible for speech {speech.speech_id} in file: "
                    f"{metadata.audio_path}, starting at: {speech.start} and ending at: "
                    f"{speech.end}.\n\n"
                    f"emission_frames (T={T}) < target_length (L={L}) + repeats (R={R})."
                    f"\n\n (Text is probably too long for the given audio).\n\n"
                    f"Using fallback linear interpolation for timestamps.\n\n"
                )
            )
            alignment_mapping = process_fallback_alignment(
                mapping,
                speech.start,
                speech.end,
                original_text,
                tokenizer,
                speech.text_spans,
                ndigits,
            )
            speech.alignments.extend(alignment_mapping)
            speech_mappings.extend(alignment_mapping)
            if delete_emissions:
                Path(emissions_filepath).unlink()
            continue

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
        speech_mappings.extend(alignment_mapping)

        if delete_emissions:
            Path(emissions_filepath).unlink()

    return speech_mappings


def count_target_repeats(targets: torch.Tensor) -> int:
    """
    Count consecutive repeated tokens in target sequence.

    This corresponds to R in PyTorch's CTC constraint: T >= L + R.

    Parameters
    ----------
    targets : torch.Tensor
        Target token indices.

    Returns
    -------
    int
        Number of consecutive repeats.
    """
    if targets.numel() <= 1:
        return 0
    targets_flat = targets.flatten()
    return int((targets_flat[1:] == targets_flat[:-1]).sum().item())


def is_alignable(
    normalized_tokens: list[str],
    processor: Wav2Vec2Processor,
    num_emission_frames: int,
) -> tuple[bool, int, int, int]:
    """
    Check if alignment is feasible given PyTorch's CTC constraint: T >= L + R.

    Parameters
    ----------
    normalized_tokens : list of str
        List of normalized text tokens.
    processor : Wav2Vec2Processor
        Wav2Vec2Processor instance for tokenization.
    num_emission_frames : int
        Number of emission frames (T) from wav2vec2.

    Returns
    -------
    tuple
        Tuple of (can_align, T, L, R) where:
            - can_align: Whether alignment is feasible
            - T: Number of emission frames
            - L: Target label length
            - R: Number of consecutive repeated tokens
    """
    transcript = " ".join(normalized_tokens).replace("\n", " ").upper()
    targets = processor.tokenizer(transcript, return_tensors="pt")["input_ids"]
    L = targets.size(1)
    R = count_target_repeats(targets)
    T = num_emission_frames
    return (T >= L + R, T, L, R)


def create_fallback_alignment(
    mapping: list[dict],
    chunk_start: float,
    chunk_end: float,
) -> list[dict]:
    """
    Create fallback alignment with linearly interpolated timestamps.

    Used when forced alignment is infeasible (e.g. due to Whisper hallucination).
    Adds start_time, end_time, and score to each token in the mapping.
    The mapping already contains: normalized_token, text, start_char, end_char.

    Parameters
    ----------
    mapping : list of dict
        Token mapping from text normalizer.
    chunk_start : float
        Start time of the chunk in seconds.
    chunk_end : float
        End time of the chunk in seconds.

    Returns
    -------
    list of dict
        Updated mapping with linearly interpolated timestamps and score=0.0.
    """
    n_tokens = len(mapping)
    if n_tokens == 0:
        return mapping

    duration = chunk_end - chunk_start
    step = duration / n_tokens

    for i, token in enumerate(mapping):
        token["start_time"] = chunk_start + i * step
        token["end_time"] = chunk_start + (i + 1) * step
        token["score"] = 0.0  # Zero score indicates fallback/low confidence

    return mapping


def process_fallback_alignment(
    mapping: list[dict],
    start: float,
    end: float,
    original_text: str,
    tokenizer,
    segment_spans: list | None,
    ndigits: int,
) -> list[AlignmentSegment]:
    """
    Process alignment using fallback linear interpolation.

    Used when forced alignment is infeasible (e.g. due to Whisper hallucination).
    Applies the full post-processing pipeline to the fallback timestamps.

    Parameters
    ----------
    mapping : list of dict
        Token mapping from text normalizer.
    start : float
        Start time of the segment/chunk.
    end : float
        End time of the segment/chunk.
    original_text : str
        The original text of the segment/chunk.
    tokenizer : object
         Optional tokenizer for custom segmentation.
    segment_spans : list, optional
        Optional list of segment spans.
    ndigits : int
        Number of decimal digits to round the timestamps.

    Returns
    -------
    list of AlignmentSegment
        List of aligned segments.
    """
    mapping = create_fallback_alignment(mapping, start, end)
    mapping = merge_multitoken_expressions(mapping)
    mapping = add_deletions_to_mapping(mapping, original_text)
    mapping = get_segment_alignment(
        mapping=mapping,
        original_text=original_text,
        tokenizer=tokenizer,
        segment_spans=segment_spans,
    )
    return encode_alignments(mapping, ndigits=ndigits)


def format_timestamp(timestamp):
    """
    Convert timestamp in seconds to "hh:mm:ss,ms" format.

    Parameters
    ----------
    timestamp : float
        Timestamp in seconds.

    Returns
    -------
    str
        Formatted timestamp.
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

    Parameters
    ----------
    char_list : list of TokenSpan
        A list of character tokens.
    word_lengths : list of int
        A list of character lengths of the words (normalized tokens).

    Returns
    -------
    list of list of TokenSpan
        List of words, where each word is a list of characters.
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

    Parameters
    ----------
    word_span : list of TokenSpan
        A list of TokenSpan objects that together represent the word span's
        timings in the aligned audio chunk.
    frames_per_logit : float
        The number of audio frames per model output logit. This is the
        total number of frames in our audio chunk divided by the number of
        (non-padding) logit outputs for the chunk.
    start_segment : float, default 0.0
        The start time of the speech segment in the original audio file.
        We offset the start/end time of the word span by this value (in
        case chunking/slicing of the audio was performed).
    sample_rate : int, default 16000
        The sample rate of the audio file, default 16000.

    Returns
    -------
    tuple
        Tuple containing (start_time, end_time, score).
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

    Parameters
    ----------
    tokens : torch.Tensor
        Tokens predicted by the model.
    scores : torch.Tensor
        Scores for each token.
    mapping : list of dict
        Token mapping information.
    blank : int, default 0
        The token ID for the blank (padding) token.
    start_wildcard : bool, default True
        Whether to add a start wildcard token, to better account for
        speech in the audio that is not covered by the text.
    end_wildcard : bool, default True
        Whether to add an end wildcard token.
    word_boundary : str, optional
        The token used to indicate word boundaries. Usually, this is
        the "|" token. Sometimes, the model is trained without word boundary tokens
        (Pytorch native Wav2Vec2 models).
    processor : Wav2Vec2Processor, optional
        The Wav2Vec2Processor used for tokenization.

    Returns
    -------
    tuple
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

    Parameters
    ----------
    mapping : list of dict
        A list of dictionaries containing the original text tokens that
        have been aligned with the audio, together with character indices and timestamps.
    original_text : str
        The original unnormalized text that was aligned with the audio.
    tokenizer : object, optional
        A PunktSentenceTokenizer instance to tokenize the original text
        into sentences (if segment_spans are not provided). Alternatively, a callable
        function that takes the original text as input and returns a list of
        (start_char, end_char) tuples for each segment.
    segment_spans : list of tuple, optional
        Optional list of tuples containing the start and end character
        indices of custom segments in the original text.

    Returns
    -------
    list of dict
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
            try:
                start_char = mapping[0]["start_char"]
                end_char = mapping[-1]["end_char"]
                segment_spans = [(start_char, end_char)]  # Single segment with entire text
            except IndexError:
                segment_spans = []  # No segments if mapping/text is empty

    segment_mapping = []
    remaining_tokens = mapping.copy()
    previous_tokens = []  # List to keep track of tokens that have been removed from the mapping

    for span in segment_spans:
        start_segment_index = span[0]  # Character index in the original text
        end_segment_index = span[1]
        start_segment_time = None
        end_segment_time = None
        segment_tokens = []

        if start_segment_index < remaining_tokens[0]["start_char"]:
            # Warn the user they probably need to strip leading whitespace from the original text
            logger.warning(
                (
                    "Segment indices start before the first token index. This may be due to "
                    "leading whitespace in the original text. Consider stripping leading/trailing "
                    "whitespace from the original text before creating SpeechSegment objects.\n",
                )
            )

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

    Parameters
    ----------
    current_token : dict
        The current token dictionary containing the token's metadata.
    token_list : list of dict
        A list of token alignments (dictionaries) that acts as fallback
        when the current token has no timestamp.
    fallback_direction : str, default "next"
        The direction to search for a timestamp ("next" or "previous"
        tokens) as a fallback when the current token has no timestamp.

    Returns
    -------
    float or None
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

    Parameters
    ----------
    word_spans : list of list of TokenSpan
        List of lists of TokenSpan objects representing the aligned words.
    mapping : list of dict
        List of dictionaries containing the original normalized text tokens that
        have been aligned with the audio (together with character indices relative
        to the original text).
    speech : SpeechSegment
        The speech segment.
    chunk_size : int, default 20
        Size of the audio chunks in seconds (when doing batched inference).
    start_segment : float, default 0.0
        Start time of the audio segment inside the audio file (in seconds).

    Returns
    -------
    list of dict
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
