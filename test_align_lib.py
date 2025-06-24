import torch
import torchaudio
import torchaudio.functional as F
from transformers import AutoModelForCTC, Wav2Vec2Processor

from easyalign.alignment.pytorch import (
    add_timestamps_to_mapping,
    align_pytorch,
    get_segment_alignment,
)
from easyalign.text.normalization import (
    get_normalized_tokens,
    normalize_text_with_mapping,
)
from easyalign.tokenizer import load_tokenizer


def get_probs(audiofile: str, logits_only: bool = False) -> tuple:
    """
    Transcribe the audio file using wav2vec2 model.
    This function loads the audio file, splits it into chunks of 20 seconds,
    runs the model on each chunk, and returns the probabilities for each token in the audio.
    The probabilities are concatenated across all chunks.
    Returns the probabilities and the length of the audio input.

    Args:
        audiofile (str): Path to the audio file.
        logits_only (bool): Whether to return logits only or probabilities.

    Returns:
        tuple: A tuple containing the probabilities and the length of the audio input.
    """

    # Load the audio file
    audio_input, sr = torchaudio.load(audiofile)
    audio_input.to(device).half()  # Convert to half precision

    # Split the audio into chunks of 30 seconds
    chunk_size = 20
    audio_chunks = torch.split(audio_input, chunk_size * sr, dim=1)

    # Transcribe each audio chunk
    all_probs = []

    for audio_chunk in audio_chunks:
        # If audio chunk is shorter than 30 seconds, pad it to 30 seconds
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
            if logits_only:
                probs = logits
            else:
                probs = torch.nn.functional.softmax(logits, dim=-1)

        all_probs.append(probs)

    # Concatenate the probabilities
    align_probs = torch.cat(all_probs, dim=1)
    return align_probs, len(audio_input[0])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCTC.from_pretrained(
        "KBLab/wav2vec2-large-voxrex-swedish", torch_dtype=torch.float16
    ).to(device)
    processor = Wav2Vec2Processor.from_pretrained(
        "KBLab/wav2vec2-large-voxrex-swedish", sample_rate=16000, return_tensors="pt"
    )

    # Find way to add token to processor
    # processor.tokenizer.add_tokens(["*"])

    probs, audio_length = get_probs("data/audio_mono_120.wav", logits_only=False)

    text = """Testar text att linjera."""
    normalized_text, mapping, original_text = normalize_text_with_mapping(text)
    normalized_mapping, normalized_tokens = get_normalized_tokens(mapping)

    mapping = {
        "original_text": original_text,
        "mapping": mapping,
        "normalized_mapping": normalized_mapping,
        "normalized_tokens": normalized_tokens,
    }

    # Align with pytorch
    alignments = []
    alignment_scores = []

    tokens, scores = align_pytorch(
        transcripts=mapping["normalized_tokens"],
        processor=processor,
        emissions=probs,
        start_unknown=True,
        end_unknown=True,
        device=device,
    )
    alignments.append(tokens)
    alignment_scores.append(scores)

    mapping["normalized_tokens"].insert(0, "*")
    mapping["normalized_tokens"].append("*")

    mapping = add_timestamps_to_mapping(
        mapping=mapping,
        tokens=tokens,
        scores=scores,
        audio_length=audio_length,
        start_segment=0,
        chunk_size=20,
    )

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
        mapping=mapping,
        tokenizer=tokenizer,
    )

    # Reconstruct original tokens from mapping
    original_tokens = []
    for transformation in mapping["mapping"]:
        original_tokens.append(transformation["original_token"])

    sentence_mapping
