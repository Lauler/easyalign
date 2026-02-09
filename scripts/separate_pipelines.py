import logging
from pathlib import Path

import torch
from transformers import AutoModelForCTC, Wav2Vec2Processor

from easyaligner.data.collators import metadata_collate_fn
from easyaligner.data.datamodel import SpeechSegment
from easyaligner.data.dataset import JSONMetadataDataset
from easyaligner.pipelines import (
    align_speech,
    alignment_pipeline,
    emissions_pipeline,
    vad_pipeline,
)
from easyaligner.text.normalization import text_normalizer
from easyaligner.text.tokenizer import load_tokenizer
from easyaligner.vad.pyannote import load_vad_model

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

text = """Statsminister Göran Persson (asdasfs) sa ju häromdagen att det kan dröja till efter
 2006 innan euron införs i Sverige om det blir ett ja i omröstningen.
"""

tokenizer = load_tokenizer(language="swedish")
sentence_list = tokenizer.tokenize(text)
text = text.strip()
span_list = list(tokenizer.span_tokenize(text))

speeches = [[SpeechSegment(speech_id=0, text=text, text_spans=span_list, start=None, end=None)]]

vad_outputs = vad_pipeline(
    model=model_vad,
    audio_paths=["statsminister.wav"],
    audio_dir="data/sv",
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
    audio_dir="data/sv",
    sample_rate=16000,
    chunk_size=30,
    alignment_strategy="speech",
    batch_size_files=1,
    num_workers_files=2,
    prefetch_factor_files=2,
    batch_size_features=8,
    num_workers_features=4,
    streaming=True,
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

alignments = alignment_pipeline(
    dataloader=audiometa_loader,
    text_normalizer_fn=text_normalizer,
    processor=processor,
    tokenizer=None,
    emissions_dir="output/emissions",
    output_dir="output/alignments",
    alignment_strategy="speech",
    start_wildcard=True,
    end_wildcard=True,
    blank_id=0,
    word_boundary="|",
    chunk_size=30,
    ndigits=5,
    indent=2,
    save_json=True,
    save_msgpack=False,
    return_alignments=True,
    delete_emissions=False,
    remove_wildcards=True,
    device="cuda",
)


mapping = align_speech(
    dataloader=audiometa_loader,
    text_normalizer_fn=text_normalizer,
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
    device="cuda",
)
