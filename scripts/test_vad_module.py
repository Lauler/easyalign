import os
from pathlib import Path

import msgspec
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCTC, Wav2Vec2Processor

from easyalign.alignment.pytorch import segment_speech_probs
from easyalign.data.collators import (
    alignment_collate_fn,
    audiofile_collate_fn,
)
from easyalign.data.datamodel import AudioMetadata
from easyalign.data.dataset import (
    AudioFileDataset,
)
from easyalign.data.utils import pad_probs
from easyalign.pipelines import vad_pipeline
from easyalign.vad.silero import load_vad_model

model = AutoModelForCTC.from_pretrained(
    "KBLab/wav2vec2-large-voxrex-swedish", torch_dtype=torch.float16
).to("cuda")
processor = Wav2Vec2Processor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish")
model_vad = load_vad_model()

vad_outputs = vad_pipeline(
    model=model_vad,
    audio_paths=["audio_80.wav"],
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


def emissions_pipeline(
    model,
    processor: Wav2Vec2Processor,
    metadata: list[AudioMetadata] | list[str] | AudioMetadata | str,
    audio_dir: str,
    sample_rate: int = 16000,
    chunk_size: int = 30,
    use_vad: bool = True,
    batch_size_files: int = 1,
    num_workers_files: int = 1,
    prefetch_factor_files: int = 2,
    batch_size_features: int = 8,
    num_workers_features: int = 4,
    save_json: bool = True,
    save_msgpack: bool = False,
    save_emissions: bool = True,
    return_emissions: bool = False,
    output_dir: str = "output/emissions",
):
    """
    Run emissions extraction pipeline on the given audio files and save results to file. If `return_emissions`
    is True, function becomes a generator that yields tuples of (metadata, emissions) for each audio file.

    Args:
        model: The loaded ASR model.
        metadata: List of AudioMetadata objects or paths to JSON files.
        audio_dir: Directory with audio files
        sample_rate: Sample rate to resample audio to. Default 16000.
        chunk_size: When VAD is not used, SpeechSegments are naively split into
            `chunk_size` sized chunks for feature extraction.
        use_vad: Whether to use VAD-based chunks (if available in metadata), or just
            naïvely split the audio of speech segments into `chunk_size` chunks.
        batch_size_files: Batch size for the file DataLoader.
        num_workers_files: Number of workers for the file DataLoader.
        prefetch_factor_files: Prefetch factor for the file DataLoader.
        batch_size_features: Batch size for the feature DataLoader.
        num_workers_features: Number of workers for the feature DataLoader.
        save_json: Whether to save the emissions output as JSON files.
        save_msgpack: Whether to save the emissions output as Msgpack files.
        save_emissions: Whether to save the raw emissions as .npy files.
        return_emissions: Whether to return the emissions as a list of numpy arrays.
        output_dir: Directory to save the output files if saving is enabled.

    """
    file_dataset = AudioFileDataset(
        metadata=metadata,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        processor=processor,
        chunk_size=chunk_size,
        use_vad=use_vad,
    )

    file_dataloader = torch.utils.data.DataLoader(
        file_dataset,
        batch_size=batch_size_files,
        shuffle=False,
        collate_fn=audiofile_collate_fn,
        num_workers=num_workers_files,
        prefetch_factor=prefetch_factor_files,
    )

    for features in file_dataloader:
        slice_dataset = features[0]["dataset"]
        feature_dataloader = torch.utils.data.DataLoader(
            slice_dataset,
            batch_size=batch_size_features,
            shuffle=False,
            collate_fn=alignment_collate_fn,
            num_workers=num_workers_features,
        )

        probs_list = []
        speech_ids = []

        for batch in feature_dataloader:
            features = batch["features"].half().to("cuda")

            with torch.inference_mode():
                logits = model(features).logits

            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            probs = pad_probs(
                probs, chunk_size=file_dataset.chunk_size, sample_rate=file_dataset.sr
            )

            probs_list.append(probs)
            speech_ids.extend(batch["speech_ids"])

        metadata = slice_dataset.metadata
        metadata, emissions = save_outputs(
            metadata=metadata,
            probs_list=probs_list,
            speech_ids=speech_ids,
            save_json=save_json,
            save_msgpack=save_msgpack,
            save_emissions=save_emissions,
            return_emissions=return_emissions,
            output_dir=output_dir,
        )

        if return_emissions:
            yield metadata, emissions


def save_outputs(
    metadata: AudioMetadata,
    probs_list: list[np.ndarray],
    speech_ids: list[str],
    save_json: bool = True,
    save_msgpack: bool = False,
    save_emissions: bool = True,
    return_emissions: bool = False,
    output_dir: str = "output/emissions",
):
    audio_path = metadata.audio_path
    base_path = Path(audio_path).parent / Path(audio_path).stem
    json_encoder = msgspec.json.Encoder()
    msgpack_encoder = msgspec.msgpack.Encoder()
    speech_index = 0

    if save_emissions:
        # Segment the probs according to speech_ids and save each speech's probs separately
        for speech_id, probs in segment_speech_probs(probs_list, speech_ids):
            probs_path = Path(output_dir) / base_path / f"{speech_id}.npy"
            Path(probs_path).parent.mkdir(parents=True, exist_ok=True)

            metadata.speeches[speech_index].probs_path = str(probs_path)
            speech_index += 1
            np.save(probs_path, probs)

    if save_json:
        json_msgspec = json_encoder.encode(metadata)
        json_msgspec = msgspec.json.format(json_msgspec, indent=2)
        json_path = Path(output_dir) / Path(audio_path).parent / (Path(audio_path).stem + ".json")
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "wb") as f:
            f.write(json_msgspec)

    if save_msgpack:
        msgpack_msgspec = msgpack_encoder.encode(metadata)
        msgpack_path = (
            Path(output_dir) / Path(audio_path).parent / (Path(audio_path).stem + ".msgpack")
        )
        Path(msgpack_path).parent.mkdir(parents=True, exist_ok=True)
        with open(msgpack_path, "wb") as f:
            f.write(msgpack_msgspec)

    if return_emissions:
        emissions = []
        print(probs_list)
        for _, probs in segment_speech_probs(probs_list, speech_ids):
            print(probs)
            emissions.append(probs)

        return metadata, emissions

    return None, None


emissions_output = list(
    emissions_pipeline(
        model=model,
        processor=processor,
        metadata=vad_outputs,
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
        return_emissions=True,
        output_dir="output/emissions",
    )
)


file_dataset = AudioFileDataset(
    metadata=vad_outputs,
    audio_dir="data",
    sample_rate=16000,
    model_name="KBLab/wav2vec2-large-voxrex-swedish",
    chunk_size=30,
    use_vad=True,
)

file_dataloader = torch.utils.data.DataLoader(
    file_dataset, batch_size=1, shuffle=False, collate_fn=audiofile_collate_fn, num_workers=2
)

for features in file_dataloader:
    slice_dataset = features[0]["dataset"]
    feature_dataloader = torch.utils.data.DataLoader(
        slice_dataset, batch_size=8, shuffle=False, collate_fn=alignment_collate_fn, num_workers=2
    )
    probs_list = []
    speech_ids = []

    for batch in feature_dataloader:
        features = batch["features"].half().to("cuda")

        with torch.inference_mode():
            logits = model(features).logits

        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
        probs = pad_probs(probs, chunk_size=file_dataset.chunk_size, sample_rate=file_dataset.sr)

        probs_list.append(probs)
        speech_ids.extend(batch["speech_ids"])


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
