import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCTC, Wav2Vec2Processor

from easyalign.alignment.pytorch import calculate_w2v_output_length
from easyalign.data.dataset import (
    AudioFileDataset,
    AudioSliceDataset,
    VADAudioDataset,
    alignment_collate_fn,
    audiofile_collate_fn,
    vad_collate_fn,
)
from easyalign.vad.silero import load_vad_model
from easyalign.vad.vad import run_vad

vad_dataset = VADAudioDataset(audio_paths=["audio_80.wav"], audio_dir="data", sample_rate=16000)
dataloader = torch.utils.data.DataLoader(
    vad_dataset, batch_size=1, shuffle=False, collate_fn=vad_collate_fn
)
audio_dict = next(iter(dataloader))

model = AutoModelForCTC.from_pretrained(
    "KBLab/wav2vec2-large-voxrex-swedish", torch_dtype=torch.float16
).to("cuda")
model_vad = load_vad_model()

audio = audio_dict["audio"][0]
res = run_vad(
    audio_path=audio_dict["audio_path"][0],
    audio_dir="data",
    model=model_vad,
    audio=audio,
    chunk_size=30,
)

file_dataset = AudioFileDataset(
    metadata=[res],
    audio_dir="data",
    sample_rate=16000,
    model_name="KBLab/wav2vec2-large-voxrex-swedish",
    chunk_size=30,
    use_vad=True,
)

file_dataloader = torch.utils.data.DataLoader(
    file_dataset, batch_size=1, shuffle=False, collate_fn=audiofile_collate_fn, num_workers=2
)


maximum_nr_logits = calculate_w2v_output_length(
    file_dataset.chunk_size * file_dataset.sr, chunk_size=file_dataset.chunk_size
)


def pad_probs(probs, chunk_size, sample_rate):
    nr_logits = calculate_w2v_output_length(chunk_size * sample_rate, chunk_size=chunk_size)
    probs = np.pad(
        array=probs,
        pad_width=(
            (0, 0),
            (0, nr_logits - probs.shape[1]),  # Add remaining logits as padding
            (0, 0),
        ),
        mode="constant",
    )
    return probs


for features in file_dataloader:
    slice_dataset = features[0]["dataset"]
    feature_dataloader = torch.utils.data.DataLoader(
        slice_dataset, batch_size=8, shuffle=False, collate_fn=alignment_collate_fn, num_workers=2
    )
    for batch in feature_dataloader:
        spectograms = batch["spectograms"].half().to("cuda")

        with torch.inference_mode():
            logits = model(spectograms).logits

        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()

        # Pad the second dimension up to the nr_logits that args.chunk_size * args.sample_rate yields.
        # Usually collate_fn takes care of this when batch contains at least 1 obs that is chunk_size long.
        # We need to handle the case when batch contains only 1 obs, or all obs are shorter than chunk_size.
        probs = np.pad(
            array=probs,
            pad_width=(
                (0, 0),
                (0, maximum_nr_logits - probs.shape[1]),  # Add remaining logits as padding
                (0, 0),
            ),
            mode="constant",
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
