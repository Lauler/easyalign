import logging
from pathlib import Path

import torch
from nltk.tokenize import PunktTokenizer
from transformers import (
    AutoModelForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from easyalign.data.collators import (
    audiofile_collate_fn,
    metadata_collate_fn,
    transcribe_collate_fn,
)
from easyalign.data.datamodel import SpeechSegment
from easyalign.data.dataset import AudioFileDataset, JSONMetadataDataset
from easyalign.pipelines import pipeline
from easyalign.text.normalization import (
    SpanMapNormalizer,
)
from easyalign.text.tokenizer import load_tokenizer
from easyalign.utils import save_metadata_json
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

# Strip whitespace from beginning and end of text because nltk's span_tokenize
# behaves inconsistently with leading/trailing whitespace (retains them),
# as opposed to whitespace between sentences (does not retain them).
tokenizer = load_tokenizer(language="swedish")
text = text.strip()
span_list = list(tokenizer.span_tokenize(text))

speeches = [[SpeechSegment(speech_id=0, text=text, text_spans=span_list, start=None, end=None)]]


if __name__ == "__main__":
    model_vad = load_vad_model()
    model = (
        AutoModelForCTC.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish").to("cuda").half()
    )
    processor = Wav2Vec2Processor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish")

    pipeline(
        vad_model=model_vad,
        emissions_model=model,
        processor=processor,
        audio_paths=["audio/statsminister.wav"],
        audio_dir="data",
        speeches=speeches,
        alignment_strategy="speech",
        text_normalizer=text_normalizer,
        tokenizer=tokenizer,
        start_wildcard=True,
        end_wildcard=True,
        blank_id=processor.tokenizer.pad_token_id,
        word_boundary="|",
        output_vad_dir="output/vad",
        output_emissions_dir="output/emissions",
        output_alignments_dir="output/alignments",
        save_json=True,
        save_msgpack=False,
        return_alignments=False,
    )
