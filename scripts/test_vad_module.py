import multiprocessing as mp
import os
from pathlib import Path

import msgspec
import num2words
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCTC, Wav2Vec2Processor

from easyalign.alignment.pytorch import (
    align_pytorch,
    get_segment_alignment,
    get_word_spans,
    join_word_timestamps,
)
from easyalign.data.collators import metadata_collate_fn
from easyalign.data.datamodel import AudioMetadata
from easyalign.data.dataset import AudioFileDataset, JSONMetadataDataset
from easyalign.pipelines import emissions_pipeline, vad_pipeline
from easyalign.text.normalization import (
    SpanMapNormalizer,
    add_deletions_to_mapping,
    merge_multitoken_expressions,
)
from easyalign.text.tokenizer import load_tokenizer
from easyalign.vad.pyannote import load_vad_model

model = AutoModelForCTC.from_pretrained(
    "KBLab/wav2vec2-large-voxrex-swedish", torch_dtype=torch.float16
).to("cuda")
processor = Wav2Vec2Processor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish")
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

emissions_output = emissions_pipeline(
    model=model,
    processor=processor,
    metadata=json_dataset,
    audio_dir="data",
    sample_rate=16000,
    chunk_size=30,
    use_vad=True,
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

for batch in audiometa_loader:
    for metadata in batch:
        for speech in metadata.speeches:
            emissions = np.load(Path("output/emissions") / speech.probs_path)
            emissions = np.vstack(emissions)

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

# Output the slices of audio corresponding to start_segment and end_segment
audio_input, sr = torchaudio.load(Path("data") / metadata.audio_path)
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
