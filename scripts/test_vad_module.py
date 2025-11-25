import logging
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
from nltk.tokenize import PunktTokenizer
from tqdm import tqdm
from transformers import AutoModelForCTC, Wav2Vec2Processor

from easyalign.alignment.pytorch import (
    align_pytorch,
    get_segment_alignment,
    get_word_spans,
    join_word_timestamps,
)
from easyalign.data.collators import metadata_collate_fn
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

model = AutoModelForCTC.from_pretrained(
    "KBLab/wav2vec2-large-voxrex-swedish", torch_dtype=torch.float16
).to("cuda")
processor = Wav2Vec2Processor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish")
model_vad = load_vad_model()

text = """Statsminister Göran Persson (asdasfs) sa ju häromdagen att det kan dröja till efter 2006 innan euron införs i Sverige om det blir ett ja i omröstningen. 
Persson varnade för att förhandlingarna om till vilken kurs kronan ska knytas t.e. kan dra ut på tiden. Men professorn i nationalekonomi Harry Flam ser inte alls de problemen.
Jag tror inte att frågan om knytkursen kommer att vara något större problem faktiskt. Om Sverige säger ja till euron i folkomröstningen så lämnar vi ju kronan bakom oss."""

text = """
Statsminister Göran Persson sa ju häromdagen att det kan dröja till efter 2006 innan euron införs i Sverige om det blir ett ja i omröstningen. 
Persson varnade för att förhandlingarna om till vilken kurs kronan ska knytas till euron kan dra ut på tiden. Men professorn i nationalekonomi Harry Flam ser inte alls de problemen. 
Jag tror inte att frågan om knytkursen kommer vara något större problem faktiskt.
Om Sverige säger ja till euron i folkomröstningen lämnar vi kronan bakom oss. Men innan vi byter valuta måste vi veta exakt hur många kronor vi får för euron. 
Detta kallas för knytkurs. Knytkursen bestämmer om vi får 8, 9 eller 10 kronor per euro. 
Den frågan avgörs någon gång under denna höst i ett hemligt möte mellan euroländernas finansiella kommitté och den svenska regeringen.
Grupperna förhandlar om och bestämmer en centralkurs mellan kronan och euron. 
Denna kurs bestäms bl.a. av hur mycket en varukorg kostar i Sverige jämfört med EMU-länderna. 
Det är vid denna förhandling Göran Persson är rädd för att Sverige ska få ett skambud på kronan medan nationalekonomen Harry Flam tror att det kommer gå smidigt och inte vålla några större diskussioner. 
Nej, därför att EU tror jag kommer att vara väldigt angeläget om att underlätta Sveriges inträde. Det är
"""

text = """
Hallå. Förlåt. Kan du göra ljud någon annanstans? Titta. 
Det gråa som är mot djup savann, som ger kontrast och framhäver det ännu mera kontrast, fast med färg. 
Du menar elefanterna på savannen? Det är inte så enkelt. Jo. Vet du varför överklassen inte accepterar oss?
För att vi är från Malmö. Nej, nej, nej. Det är Gyllenhammar, Trolle och Ramel också. Jag vet inte vilka det är.
Precis! Det är för att vi anses vara okultiverade som inte ser mer än elefanter på savann. Vi måste se något annat, mer. Vad?
Det är djup, kombination, struktur. Snälla, Kennedy, kan vi ta det sen? Jag måste få iväg de här hyresavierna. 
Vi ska starta konstgalleri! Vi ska bjuda hit de största, finaste familjerna. Vi ska köpa konst. Vi ska titta på konst.
Vi ska sälja konst. Och vi ska hjälpa unga konstnärer. Ja, vi ska bjuda på god kurdisk mat också. Och vet du Nisha? 
Vi ska ha den dyraste, bästa konsten, hos familjen Mursa. Men snälla Kennedy, vad kan du om sånt? Jag kan lära mig.
Som när du skulle lära dig skriva din självbiografi? Ja, precis. Och hur tyckte du att dina två A4-sidor blev?
Jag tyckte att den första delen, den blev faktiskt bra, jag var inlevelse. Hade det varit på kurdiska, hade jag
skrivit tusen sidor, men nu det blev kompakt. Jag är inte så bra på att beskriva. 
"""

tokenizer = load_tokenizer(language="swedish")
sentence_list = tokenizer.tokenize(text)
span_list = list(tokenizer.span_tokenize(text))

speeches = [[SpeechSegment(speech_id=0, text=text, text_spans=span_list, start=None, end=None)]]

vad_outputs = vad_pipeline(
    model=model_vad,
    audio_paths=["audio_80.wav"],
    audio_dir="data",
    speeches=speeches,
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

emissions_output = emissions_pipeline(
    model=model,
    processor=processor,
    metadata=json_dataset,
    audio_dir="data",
    sample_rate=16000,
    chunk_size=30,
    alignment_strategy="speech",
    batch_size_files=1,
    num_workers_files=2,
    prefetch_factor_files=2,
    batch_size_features=8,
    num_workers_features=4,
    save_json=True,
    save_msgpack=False,
    save_emissions=True,
    return_emissions=False,
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


def format_speech_text(speech: SpeechSegment, add_leading_space: bool = False) -> str:
    print("Speech text segments:", speech.text)
    if len(speech.text) == 1:
        original_text = speech.text[0]
    elif len(speech.text) > 1:
        # If the user tokenized the text into multiple segments, concatenate them
        # for alignment
        if add_leading_space:
            # Add leading space for all except the first segment
            original_text = "".join([speech.text[0]] + [" " + t for t in speech.text[1:]])

            # Modify the text_spans accordingly
            offset = 1
            for i, text_span in enumerate(speech.text_spans):
                if i == 0:
                    continue
                start, end = text_span
                speech.text_spans[i] = (
                    start + offset - 1,
                    end + offset,
                )  # Add offset to both start and end
                offset += 1  # Increment offset for next segment
        else:
            original_text = "".join(speech.text)
    else:
        logger.warning(
            (
                f"No text found for speech id {speech.speech_id} in"
                f"{metadata.audio_path}. Skipping alignment."
            )
        )
        original_text = ""
    return original_text


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
    save_json: bool = True,
    save_msgpack: bool = False,
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

                original_text = format_speech_text(speech, add_leading_space=add_leading_space)

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

                speech.alignments.extend(encode_alignments(mapping))
                if delete_emissions:
                    Path(emissions_filepath.parent).unlink()

            # Add info to metadata and save

            if save_json:
                save_metadata_json(metadata, output_dir=output_dir)

            if save_msgpack:
                save_metadata_msgpack(metadata, output_dir=output_dir)

    return mapping


mapping = align_speech(
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
    add_leading_space=False,
    device="cuda",
)


def align_chunks(
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
    chunk_mappings = []
    for batch in tqdm(dataloader):
        for metadata in batch:
            for speech in metadata.speeches:
                emissions_filepath = Path(emissions_dir) / speech.probs_path
                emissions = np.load(emissions_filepath)
                print("Emissions shape:", emissions.shape)

                for i, chunk in enumerate(speech.chunks):
                    normalized_tokens, mapping = text_normalizer(original_text)
                    emissions_chunk = emissions[i]
                    print("Emissions chunk shape:", emissions_chunk.shape)

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
                    mapping = add_deletions_to_mapping(mapping, original_text)
                    mapping = get_segment_alignment(
                        mapping=mapping,
                        original_text=original_text,
                        tokenizer=tokenizer,
                        segment_spans=metadata.text_spans,
                    )

                    chunk_mappings.append(mapping)

    return chunk_mappings


chunk_mappings = align_chunks(
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
    add_leading_space=True,
    device="cuda",
)


emissions = np.load(Path("output/emissions") / "audio_mono_120/0.npy")
emissions.shape
emissions = np.vstack(emissions)

np.vstack(emissions).shape

for emission in emissions_output:
    metadata = emission[0]
    emissions = emission[1]
    speech = metadata.speeches[0]
    emissions = np.load(Path("output/emissions") / speech.probs_path)
    emissions = np.vstack(emissions)

    text = """
    Hallå. Förlåt. Kan du göra ljud någon annanstans? Titta. 
    Det gråa som är mot djup savann, som ger kontrast och framhäver det ännu mera kontrast, fast med färg. 
    Du menar elefanterna på savannen? Det är inte så enkelt. Jo. Vet du varför överklassen inte accepterar oss?
    För att vi är från Malmö. Nej, nej, nej. Det är Gyllenhammar, Trolle och Ramel också. Jag vet inte vilka det är.
    Precis! Det är för att vi anses vara okultiverade som inte ser mer än elefanter på savann. Vi måste se något annat, mer. Vad?
    Det är djup, kombination, struktur. Snälla, Kennedy, kan vi ta det sen? Jag måste få iväg de här hyresavierna. 
    Vi ska starta konstgalleri! Vi ska bjuda hit de största, finaste familjerna. Vi ska köpa konst. Vi ska titta på konst.
    Vi ska sälja konst. Och vi ska hjälpa unga konstnärer. Ja, vi ska bjuda på god kurdisk mat också. Och vet du Nisha? 
    Vi ska ha den dyraste, bästa konsten, hos familjen Mursa. Men snälla Kennedy, vad kan du om sånt? Jag kan lära mig.
    Som när du skulle lära dig skriva din självbiografi? Ja, precis. Och hur tyckte du att dina två A4-sidor blev?
    Jag tyckte att den första delen, den blev faktiskt bra, jag var inlevelse. Hade det varit på kurdiska, hade jag
    skrivit tusen sidor, men nu det blev kompakt. Jag är inte så bra på att beskriva. 
    """

    speech.text = text
    normalizer = SpanMapNormalizer(speech.text)
    normalizer.transform(r"\(.*?\)", "")  # Remove parentheses and their content
    normalizer.transform(r"\s[^\w\s]\s", " ")  # Remove punctuation between whitespace
    normalizer.transform(r"[^\w\s]", "")  # Remove punctuation and special characters
    normalizer.transform(r"\s+", " ")  # Normalize whitespace to a single space
    normalizer.transform(r"^\s+|\s+$", "")  # Strip leading and trailing whitespace
    normalizer.transform(r"\w+", lambda m: m.group().lower())


normalized_text, mapping, original_text = (
    normalizer.current_text,
    normalizer.get_token_map(),
    normalizer.original_text,
)

normalized_tokens = [item["normalized_token"] for item in mapping]

tokens, scores = align_pytorch(
    normalized_tokens=normalized_tokens,
    processor=processor,
    emissions=torch.tensor(emissions).to("cuda").unsqueeze(0),
    start_wildcard=True,
    end_wildcard=True,
    device="cuda",
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
#### TODO: Make start_segment apply to speech.start for each chunk when using VAD
mapping = join_word_timestamps(
    word_spans=word_spans,
    mapping=mapping,
    speech=speech,
    chunk_size=30,
    start_segment=speech.start,
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

text = """Statsminister Göran Persson (asdasfs) sa ju häromdagen att det kan dröja till efter 2006 innan euron införs i Sverige om det blir ett ja i omröstningen. 
Persson varnade för att förhandlingarna om till vilken kurs kronan ska knytas t.e. kan dra ut på tiden. Men professorn i nationalekonomi Harry Flam ser inte alls de problemen.
Jag tror inte att frågan om knytkursen kommer att vara något större problem faktiskt. Om Sverige säger ja till euron i folkomröstningen så lämnar vi ju kronan bakom oss."""


"""
Hallå. Förlåt. Kan du göra ljud någon annanstans? Titta. 
Det gråa som är mot djup savann, som ger kontrast och framhäver det ännu mera kontrast, fast med färg. 
Du menar elefanterna på savannen? Det är inte så enkelt. Jo. Vet du varför överklassen inte accepterar oss?
För att vi är från Malmö. Nej, nej, nej. Det är Gyllenhammar, Trolle och Ramel också. Jag vet inte vilka det är.
Precis! Det är för att vi anses vara okultiverade som inte ser mer än elefanter på savann. Vi måste se något annat, mer. Vad?
Det är djup, kombination, struktur. Snälla, Kennedy, kan vi ta det sen? Jag måste få iväg de här hyresavierna. 
Vi ska starta konstgalleri! Vi ska bjuda hit de största, finaste familjerna. Vi ska köpa konst. Vi ska titta på konst.
Vi ska sälja konst. Och vi ska hjälpa unga konstnärer. Ja, vi ska bjuda på god kurdisk mat också. Och vet du Nisha? 
Vi ska ha den dyraste, bästa konsten, hos familjen Mursa. Men snälla Kennedy, vad kan du om sånt? Jag kan lära mig.
Som när du skulle lära dig skriva din självbiografi? Ja, precis. Och hur tyckte du att dina två A4-sidor blev?
Jag tyckte att den första delen, den blev faktiskt bra, jag var inlevelse. Hade det varit på kurdiska, hade jag
skrivit tusen sidor, men nu det blev kompakt. Jag är inte så bra på att beskriva. 
"""

import torchaudio

sr = 16000
audio_input, sr = torchaudio.load(Path("data") / "audio_mono_120.wav")

# Output the slices of audio corresponding to start_segment and end_segment
# audio_input, sr = torchaudio.load(Path("data") / metadata.audio_path)
audio_slices = []
os.makedirs("data/audio_slices", exist_ok=True)
for i, segment in enumerate(mapping):
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
        sample_rate=sr,
    )


sr = 16000
# Output the VAD segments
for chunk in speech.chunks:
    start = chunk.start
    end = chunk.end

    start = int(start * sr)
    end = int(end * sr)
    audio_slice = audio_input[:, start:end]

    torchaudio.save(
        f"data/audio_slices/vad_{start}_{end}.wav",
        audio_slice,
        sample_rate=sr,
    )
