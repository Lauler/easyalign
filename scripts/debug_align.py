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
from easyalign.data.datamodel import AudioMetadata, SpeechSegment
from easyalign.data.dataset import AudioFileDataset, JSONMetadataDataset
from easyalign.pipelines import emissions_pipeline, vad_pipeline
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

                if len(speech.text) == 1:
                    original_text = speech.text[0]
                elif len(speech.text) > 1:
                    # If the user tokenized the text into multiple segments, concatenate them
                    # for alignment
                    if add_leading_space:
                        # Add leading space for all except the first segment
                        original_text = "".join(
                            [speech.text[0]] + [" " + t for t in speech.text[1:]]
                        )
                    else:
                        original_text = "".join(speech.text)
                else:
                    logger.warning(
                        (
                            f"No text found for speech id {speech.speech_id} in"
                            f"{metadata.audio_path}. Skipping alignment."
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

                mapping = get_segment_alignment(
                    mapping=mapping,
                    original_text=original_text,
                    tokenizer=tokenizer,
                    segment_spans=speech.text_spans,
                )

                if delete_emissions:
                    Path(emissions_filepath.parent).unlink()

            # Add info to metadata and save

    return mapping


if __name__ == "__main__":
    model = AutoModelForCTC.from_pretrained(
        "KBLab/wav2vec2-large-voxrex-swedish", torch_dtype=torch.float16
    ).to("cuda")
    processor = Wav2Vec2Processor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish")
    model_vad = load_vad_model()

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

    speeches = [
        [SpeechSegment(speech_id=0, text=text, text_spans=span_list, start=None, end=None)]
    ]

    vad_outputs = vad_pipeline(
        model=model_vad,
        audio_paths=["audio_mono_120.wav"],
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
        use_vad=False,
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
