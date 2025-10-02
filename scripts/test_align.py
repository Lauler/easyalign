import os
import unicodedata

import torch
import torchaudio
from num2words import num2words
from transformers import AutoModelForCTC, Wav2Vec2Processor

from easyalign.alignment.pytorch import (
    align_pytorch,
    get_segment_alignment,
    get_word_spans,
    join_word_timestamps,
)
from easyalign.text.normalization import (
    SpanMapNormalizer,
    add_deletions_to_mapping,
    merge_multitoken_expressions,
)
from easyalign.text.tokenizer import load_tokenizer


def get_probs(audiofile: str, log_probs: bool = True) -> tuple:
    """
    Transcribe the audio file using wav2vec2 model.
    This function loads the audio file, splits it into chunks of 20 seconds,
    runs the model on each chunk, and returns the probabilities for each token in the audio.
    The probabilities are concatenated across all chunks.
    Returns the probabilities and the length of the audio input.

    Args:
        audiofile (str): Path to the audio file.
        log_probs (bool): Whether to return log probs (True) or probs (False).

    Returns:
        tuple: A tuple containing the (log) probabilities and the length of the audio input.
    """

    # Load the audio file
    audio_input, sr = torchaudio.load(audiofile)
    audio_input.to(device).half()  # Convert to half precision

    # Split the audio into chunks of chunk_size seconds
    chunk_size = 20
    audio_chunks = torch.split(audio_input, chunk_size * sr, dim=1)

    # Transcribe each audio chunk
    all_probs = []

    for audio_chunk in audio_chunks:
        # If audio chunk is shorter than chunk_size seconds, pad it to chunk_size seconds
        if audio_chunk.shape[1] < chunk_size * sr:
            padding = torch.zeros((1, chunk_size * sr - audio_chunk.shape[1]))
            audio_chunk = torch.cat([audio_chunk, padding], dim=1)
        input_values = (
            processor(audio_chunk, sampling_rate=16000, return_tensors="pt", padding="longest")
            .input_values.to(device)
            .squeeze(dim=0)
        )
        with torch.inference_mode():
            logits = model(input_values.half()).logits
            if log_probs:
                probs = torch.nn.functional.log_softmax(logits, dim=-1)  # Log probabilities
            else:
                probs = torch.nn.functional.softmax(logits, dim=-1)

        all_probs.append(probs)

    # Concatenate the probabilities
    align_probs = torch.cat(all_probs, dim=1)
    return align_probs, len(audio_input[0])


def replace_digits_till(match):
    """
    Replacement function for regex that converts a range of numbers into a Swedish phrase.
    """
    first_num = num2words(int(match.group(1)), lang="sv")
    second_num = num2words(int(match.group(2)), lang="sv")
    return f"{first_num} till {second_num}"


def replace_digits_comma(match):
    """
    Replacement function for regex that converts a number with a comma into a Swedish phrase.
    """
    first_num = num2words(int(match.group(1)), lang="sv")
    second_num = num2words(int(match.group(2)), lang="sv")
    return f"{first_num} komma {second_num}"


def swedish_regex_normalization(normalizer: SpanMapNormalizer) -> str:
    """
    Normalize Swedish text using regex patterns.
    """

    # Normalize the text with NFKC if needed
    normalizer.transform(r".", lambda m: unicodedata.normalize("NFKC", m.group(0)))
    normalizer.transform(
        r"(\d+)[\. ](\d+)", lambda m: num2words(m.group(1) + m.group(2), lang="sv")
    )
    normalizer.transform(r"\(.*?\)", "")  # Remove parentheses and their content
    normalizer.transform(r"(\d+)-(\d+)", lambda m: replace_digits_till(m))
    normalizer.transform(r"(\d+),(\d+)", lambda m: replace_digits_comma(m))
    normalizer.transform(r"(\d+)[\:\-\/](\d+)", lambda m: f"{m.group(1)} {m.group(2)}")
    normalizer.transform(r"(?<=\d )§", "paragrafen")
    normalizer.transform(r"§(?= \d)", "paragraf")
    normalizer.transform(r"\s[^\w\s]\s", " ")  # Remove punctuation between whitespace
    normalizer.transform(r"[^\w\s]", "")  # Remove punctuation and special characters
    normalizer.transform(r"\s+", " ")  # Normalize whitespace to a single space
    normalizer.transform(r"^\s+|\s+$", "")  # Strip leading and trailing whitespace
    normalizer.transform(r"(\d+)", lambda m: num2words(int(m.group(1)), lang="sv"))
    normalizer.transform(r"\w+", lambda m: m.group().lower())
    normalizer.transform(r"[éè]", "e")
    normalizer.transform(r"[áà]", "a")

    return normalizer


def remove_accents(text: str) -> str:
    """Remove accents from text while preserving Swedish diacritics (åäö)."""
    # NFD decomposes characters into base + combining marks
    nfd = unicodedata.normalize("NFD", text)
    # Filter out combining marks (category Mn) but preserve the base characters
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


if __name__ == "__main__":
    # This is just a placeholder to prevent the code from running on import.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCTC.from_pretrained(
        "KBLab/wav2vec2-large-voxrex-swedish", torch_dtype=torch.float16
    ).to(device)
    processor = Wav2Vec2Processor.from_pretrained(
        "KBLab/wav2vec2-large-voxrex-swedish", sample_rate=16000, return_tensors="pt"
    )

    probs, audio_length = get_probs("data/audio_mono_120.wav", log_probs=True)

    text = """Example text to align."""

    normalizer = SpanMapNormalizer(text)

    # Normalize the text with NFKC
    normalizer = swedish_regex_normalization(normalizer)

    normalized_text, mapping, original_text = (
        normalizer.current_text,
        normalizer.get_token_map(),
        normalizer.original_text,
    )

    normalized_tokens = [item["normalized_token"] for item in mapping]

    tokens, scores = align_pytorch(
        normalized_tokens=normalized_tokens,
        processor=processor,
        emissions=probs,
        start_wildcard=True,
        end_wildcard=True,
        device=device,
    )

    word_spans, mapping = get_word_spans(
        tokens=tokens,
        scores=scores,
        mapping=mapping,
        blank=0,
        start_wildcard=True,
        end_wildcard=True,
        word_boundary="|",
        processor=processor,
    )
    mapping = join_word_timestamps(
        word_spans=word_spans,
        mapping=mapping,
        audio_length=audio_length,
        chunk_size=20,
        start_segment=0,
    )

    mapping = merge_multitoken_expressions(mapping)
    final_map = add_deletions_to_mapping(mapping, original_text)

    tokenizer = load_tokenizer(language="swedish")
    new_abbreviations = {
        "d.v.s",
        "dvs",
        "fr.o.m",
        "kungl",
        "m.m",
        "milj",
        "o.s.v",
        "t.o.m",
        "milj.kr",
    }
    print("Current abbreviations:", tokenizer._params.abbrev_types)

    tokenizer._params.abbrev_types.update(new_abbreviations)
    print("Updated abbreviations:", tokenizer._params.abbrev_types)

    sentence_mapping = get_segment_alignment(
        mapping=final_map,
        original_text=original_text,
        tokenizer=tokenizer,
    )

    # Output to json file
    import json

    with open("data/normalized_mapping.json", "w", encoding="utf-8") as f:
        json.dump(sentence_mapping, f, ensure_ascii=False, indent=2)

    # Output the slices of audio corresponding to start_segment and end_segment
    audio_input, sr = torchaudio.load("data/audio_mono_120.wav")
    audio_slices = []
    os.makedirs("data/audio_slices", exist_ok=True)
    for i, segment in enumerate(sentence_mapping):
        start_segment = segment["start_segment"]
        end_segment = segment["end_segment"]
        # Convert from seconds to frames
        start_segment = int(start_segment * sr)
        end_segment = int(end_segment * sr)
        audio_slice = audio_input[:, start_segment:end_segment]
        # Output to data/audio_slices/
        torchaudio.save(
            f"data/audio_slices/{i}.wav",
            audio_slice,
            sr,
        )
