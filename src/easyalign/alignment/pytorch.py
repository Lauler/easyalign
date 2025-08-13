import itertools
import logging

import numpy as np
import torch
import torchaudio.functional as F
from nltk.tokenize.punkt import PunktSentenceTokenizer
from torchaudio.functional import TokenSpan
from transformers.models.wav2vec2.processing_wav2vec2 import Wav2Vec2Processor

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
        emissions: Tensor of audio emissions (logits) with shape (batch, sequence (time), vocab_size).
        start_unknown: If True, adds a star wildcard token at the start of the transcript
            to allow better alignment if the audio starts with other irrelevant speech.
        end_unknown: If True, adds a star wildcard token at the end of the transcript.
        device: Device to run the alignment on (e.g., "cpu" or "cuda").

    Returns:
        alignments: Aligned token indices for the emissions.
        scores: Alignment scores (probabilities) for the tokens.
    """
    transcript = " ".join(normalized_tokens)
    transcript = transcript.replace("\n", " ").upper()

    if start_wildcard:
        transcript = "* " + transcript
    if end_wildcard:
        transcript = transcript + " *"

    targets = processor.tokenizer(transcript, return_tensors="pt")["input_ids"]
    targets = targets.to(device)

    # Add star wildcard token to the end of the emissions
    if start_wildcard or end_wildcard:
        # batch, sequence (time), vocab_size
        star_dim = torch.zeros(
            (1, emissions.size(1), 1), device=emissions.device, dtype=emissions.dtype
        )
        emissions = torch.cat((emissions, star_dim), 2)  # Add star token to the emissions

    alignments, scores = F.forced_align(emissions, targets, blank=0)
    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores


def format_timestamp(timestamp):
    """
    Convert timestamp in seconds to "hh:mm:ss,ms" format.
    """
    hours = int(timestamp // 3600)
    minutes = int((timestamp % 3600) // 60)
    seconds = int(timestamp % 60)
    milliseconds = int((timestamp % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def calculate_w2v_output_length(
    audio_frames: int,
    chunk_size: int,
    conv_stride: list[int] = [5, 2, 2, 2, 2, 2, 2],
    sample_rate: int = 16000,
    frames_first_logit: int = 400,
):
    """
    Calculate the number of output logits ("characters") from the wav2vec2 model based
    on the number of audio frames and audio chunk size.

    The wav2vec2-large model outputs one logit per 320 audio frames. The exception
    is the first logit, which is output after 400 audio frames (the model's minimum
    input length).

    We need to take into account the first logit, otherwise the alignment will slowly
    drift over time for long audio files (when chunking the audio for batched inference).

    Args:
        audio_frames:
            Number of audio frames in the audio file, or part of audio file to be aligned.
        chunk_size:
            Number of seconds to chunk the audio by for batched inference.
        conv_stride:
            The convolutional stride of the wav2vec2 model (see model.config.conv_stride).
            The product sum of the list is the number of audio frames per output logit.
            Defaults to the conv_stride of wav2vec2-large.
        sample_rate:
            The sample rate of the w2v processor, default 16000.
        frames_first_logit:
            First logit consists of more frames than the rest. Wav2vec2-large outputs
            the first logit after 400 frames.

    Returns:
        The number of logit outputs for the audio file.
    """
    frames_per_logit = np.prod(conv_stride)  # 320 for wav2vec2-large
    extra_frames = frames_first_logit - frames_per_logit

    frames_per_full_chunk = chunk_size * sample_rate  # total frames for chunk_size seconds
    n_full_chunks = (
        audio_frames // frames_per_full_chunk
    )  # This is 0 if length of audio is less than chunk_size

    # Calculate the number of logit outputs for a full size chunk
    logits_per_full_chunk = (frames_per_full_chunk - extra_frames) // frames_per_logit
    n_full_chunk_logits = n_full_chunks * logits_per_full_chunk

    # Calculate the number of logit outputs for the last chunk
    # (the remainder of the audio may be shorter than the `chunk_size`)
    n_last_chunk_frames = audio_frames % frames_per_full_chunk

    if n_last_chunk_frames >= frames_per_logit:
        # If the last chunk is long enough to produce at least one logit
        n_last_chunk_logits = (n_last_chunk_frames - extra_frames) // frames_per_logit
    elif (n_last_chunk_frames > 0) and (n_last_chunk_frames < frames_per_logit):
        # If the last chunk is shorter than frames_per_logit but longer than 0,
        # it will still produce one logit (the first logit).
        # Our data collator pads the last chunk up to 400 frames
        # if it happens to be shorter than the model's minimum input length
        n_last_chunk_logits = 1
    else:
        # If the last chunk is empty (0 frames), it will not produce any logits.
        n_last_chunk_logits = 0

    return n_full_chunk_logits + n_last_chunk_logits


def segment_speech_probs(probs_list: list[np.ndarray], speech_ids: list[str]):
    """
    Divide the accumulated probs of audio file into the speeches they belong to.
    (we can't assume that a batch maps to a single speech)

    Args:
        probs_list: List of np.ndarrays containing the probs
            with shape (batch_size, seq_len, vocab_size).
        speech_ids: List of speech ids that each chunk (observation)
            in the probs_list belongs to.
    """
    # Count the number of chunks per speech id
    speech_chunk_counts = [
        (key, sum(1 for i in group)) for key, group in itertools.groupby(speech_ids)
    ]
    # Create a list of indices where each speech chunk starts
    split_indices = list(itertools.accumulate([count for _, count in speech_chunk_counts]))[:-1]

    probs_in_speech = np.concatenate(probs_list, axis=0)
    probs_split = np.split(probs_in_speech, split_indices, axis=0)
    unique_speech_ids = dict.fromkeys(speech_ids).keys()  # Preserves order

    assert len(speech_chunk_counts) == len(probs_split) == len(set(unique_speech_ids))
    for speech_id, probs in zip(unique_speech_ids, probs_split):
        yield speech_id, probs


def add_timestamps_to_mapping(
    mapping: dict,
    tokens: torch.Tensor,
    scores: torch.Tensor,
    audio_length: int,
    start_segment: float = 0.0,
    chunk_size: int = 30,
) -> list[dict]:
    """
    Add the timestamps from aligned tokens to the original text tokens via the mapping.
    """

    token_spans = F.merge_tokens(tokens, scores, blank=0)
    # Remove all TokenSpan with token=4 (token 4 is "|", used for space)
    token_spans = [s for s in token_spans if s.token != 4]
    word_spans = unflatten(token_spans, [len(word) for word in mapping["normalized_tokens"]])
    ratio = audio_length / calculate_w2v_output_length(audio_length, chunk_size=chunk_size)

    for aligned_token, normalized_token in zip(word_spans, mapping["normalized_mapping"].items()):
        original_index = normalized_token[1]["normalized_word_index"]
        original_token = mapping["mapping"][original_index]
        start_time, end_time = get_word_timing(aligned_token, ratio, start_segment=start_segment)

        if not normalized_token[1]["is_multi_word"]:
            normalized_token[1]["start_time"] = start_time
            normalized_token[1]["end_time"] = end_time
            original_token["start_time"] = start_time
            original_token["end_time"] = end_time
        else:
            if normalized_token[1]["is_first_word"]:
                original_token["start_time"] = start_time
            if normalized_token[1]["is_last_word"]:
                original_token["end_time"] = end_time

            normalized_token[1]["start_time"] = start_time
            normalized_token[1]["end_time"] = end_time

    return mapping


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
    ratio: float,
    start_segment: float = 0.0,
    sample_rate: int = 16000,
) -> tuple[float, float]:
    """
    Calculate the start and end time of a word span in the original audio file.

    Args:
        word_span:
            A list of TokenSpan objects that together represent the word span's
            timings in the aligned audio chunk.
        ratio:
            The number of audio frames per model output logit. This is the
            total number of frames in our audio chunk divided by the number of
            (non-padding) logit outputs for the chunk.
        start_segment:
            The start time of the speech segment in the original audio file.
            We offset the start/end time of the word span by this value (in
            case chunking/slicing of the audio was performed).
        sample_rate:
            The sample rate of the audio file, default 16000.

    """
    start = (word_span[0].start * ratio) / sample_rate + start_segment
    end = (word_span[-1].end * ratio) / sample_rate + start_segment

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
    Get word spans from the token predictions and their scores.

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
        # Remove all TokenSpan with token=word_boundary_id
        token_spans = [s for s in token_spans if s.token != word_boundary_id]

    # Unflatten the token spans based on the normalized tokens' lengths
    word_spans = unflatten(token_spans, [len(token["normalized_token"]) for token in mapping])

    return word_spans, mapping


def get_segment_alignment(
    mapping: list[dict],
    original_text: str,
    tokenizer: PunktSentenceTokenizer | None = None,
    segment_spans: list[tuple[int, int]] | None = None,
):
    """
    Get alignment timestamps for any arbitrary segmentation of the original text.
    By default, this function performs a sentence span tokenization if user does
    not provide custom segment spans.

    Args:
        mapping: A list of dictionaries containing the original text tokens that
            have been aligned with the audio, together with character indices and timestamps.
        tokenizer: A PunktSentenceTokenizer instance to tokenize the original text
            into sentences (if segment_spans are not provided).
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
        # If user does not provide segment spans, we default to sentence spans
        segment_spans = tokenizer.span_tokenize(original_text)

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
                    direction="next",
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
                    direction="previous",
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
                break

            previous_tokens.append(remaining_tokens.pop(0))

    return segment_mapping


def assign_segment_time(
    current_token: dict,
    token_list: list[dict],
    direction: str = "next",
):
    """
    Assign a timestamp for the segment based on the current token's metadata.

    If alignment timestamps are missing for the current token, we search for the
    closest available token that has a timestamp (either among future tokens in
    the token_mapping, or the previous tokens in the previous_removed list).

    Args:
        current_token: The current token dictionary containing the token's metadata.
        token_list: A list of token alignment dictionaries to search in, such as
            `token_mapping` or `previous_removed`, depending on direction.
        direction: The direction to search for a timestamp ("next" or "previous").
    Returns:
        The start or end time of the segment. If no timestamp is found, returns None.
    """
    time = current_token["start_time"] if direction == "next" else current_token["end_time"]

    # We start searching from the first or last token in the list, depending on the direction.
    token_idx = 0 if direction == "next" else -1
    index_increment = 1 if direction == "next" else -1  # Move forward or backward in the list

    # Loop is skipped if the current token already has a timestamp.
    while time is None and abs(token_idx) < len(token_list):
        try:
            time = (
                token_list[token_idx]["start_time"]
                if direction == "next"
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
    audio_length: int,
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
    ratio = audio_length / calculate_w2v_output_length(audio_length, chunk_size=chunk_size)

    for aligned_token, normalized_token in zip(word_spans, mapping):
        start_time, end_time, score = get_word_timing(
            aligned_token, ratio, start_segment=start_segment
        )
        normalized_token["start_time"] = start_time
        normalized_token["end_time"] = end_time
        normalized_token["score"] = score

    return mapping
