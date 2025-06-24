import itertools
import logging

import numpy as np
import torch
import torchaudio.functional as F
from nltk.tokenize.punkt import PunktSentenceTokenizer
from transformers.models.wav2vec2.processing_wav2vec2 import Wav2Vec2Processor

logger = logging.getLogger(__name__)


def align_pytorch(
    transcripts: list[str],
    processor: Wav2Vec2Processor,
    emissions: torch.Tensor,
    start_unknown: bool,
    end_unknown: bool,
    device: str,
) -> tuple:
    transcript = " ".join(transcripts)
    transcript = transcript.replace("\n", " ").upper()

    if start_unknown:
        transcript = "* " + transcript
    if end_unknown:
        transcript = transcript + " *"

    targets = processor.tokenizer(transcript, return_tensors="pt")["input_ids"]
    targets = targets.to(device)

    # Add star wildcard token to the end of the emissions
    star_dim = torch.zeros(
        (1, emissions.size(1), 1), device=emissions.device, dtype=emissions.dtype
    )
    emissions = torch.cat((emissions, star_dim), 2)

    alignments, scores = F.forced_align(emissions, targets, blank=0)
    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    # scores = scores.exp()  # convert back to probability
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


def assign_segment_time(
    current_token: dict,
    token_list: list[dict],
    direction: str = "next",
):
    """
    If alignment timestamps are missing for the current token, we search for the
    closest available token that has a timestamp (either among future tokens in
    the token_mapping, or the previous tokens in the previous_removed list).

    Args:
        current_token: The current token dictionary containing the token's metadata.
        token_list: A list of token alignment dictionaries to search in, such as
            `token_mapping` or `previous_removed`, depending on direction.
        direction: The direction to search for a timestamp ("next" or "previous").
    """
    time = current_token["start_time"] if direction == "next" else current_token["end_time"]

    # We start searching from the first or last token in the list, depending on the direction.
    token_idx = 0 if direction == "next" else -1
    index_increment = 1 if direction == "next" else -1  # Move forward or backward in the list

    # Loop is skipped if the current token already has a timestamp.
    while time is None:
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


def get_segment_alignment(
    mapping: dict,
    tokenizer: PunktSentenceTokenizer,
    segment_spans: list[tuple[int, int]] = None,
):
    """
    Get the alignment for any arbitrary segment in the original text based on the
    provided mapping. By default, sentence spans are used if no segment spans
    are provided.

    Args:
        mapping: Dictionary containing the original text tokens that
            have been aligned with the audio.
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
        segment_spans = tokenizer.span_tokenize(mapping["original_text"])

    segment_mapping = []
    token_mapping = mapping["mapping"].copy()
    previous_removed = []

    for span in segment_spans:
        start_segment_index = span[0]  # Character index in the original text
        end_segment_index = span[1]
        while token_mapping:
            token = token_mapping[0]

            if start_segment_index in list(range(token["original_start"], token["original_end"])):
                start_segment_time = assign_segment_time(
                    current_token=token,
                    token_list=token_mapping,
                    direction="next",
                )

            if (end_segment_index - 1) in list(
                range(token["original_start"], token["original_end"])
            ):
                print(
                    f"start_segment_index: {start_segment_index}, end_segment_index: {end_segment_index}, token: {token}"
                )
                end_segment_time = assign_segment_time(
                    current_token=token,
                    token_list=previous_removed if previous_removed else token_mapping,
                    direction="previous",
                )

                # Once we have both the start and end timestamps, we can append to the
                # sentence mapping and break the loop
                segment_mapping.append(
                    {
                        "start_segment": start_segment_time,
                        "end_segment": end_segment_time,
                        "text": mapping["original_text"][start_segment_index:end_segment_index],
                    }
                )
                break

            previous_removed.append(token_mapping.pop(0))

    return segment_mapping

    # # return sentence_mapping
    # # Reconstruct original tokens from mapping
    # original_tokens = []
    # for transformation in mappings[0]["mapping"]:
    #     original_tokens.append(transformation["original_token"])


def unflatten(char_list, word_lengths):
    """
    Unflatten a list of character output tokens from wav2vec2 into words.

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
            A list of TokenSpan objects representing the word span timings in the
            aligned audio chunk.
        ratio:
            The number of audio frames per model output logit. This is the
            total number of frames in our audio chunk divided by the number of
            (non-padding) logit outputs for the chunk.
        start_segment:
            The start time of the speech segment in the original audio file.
        sample_rate:
            The sample rate of the audio file, default 16000.

    """
    start = (word_span[0].start * ratio) / sample_rate + start_segment
    end = (word_span[-1].end * ratio) / sample_rate + start_segment
    return start, end
