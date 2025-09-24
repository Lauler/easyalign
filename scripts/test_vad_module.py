import os

import torch
from tqdm import tqdm

from easyalign.data.dataset import (
    AudioFileDataset,
    AudioSliceDataset,
    VADAudioDataset,
    vad_collate_fn,
)
from easyalign.vad.silero import load_vad_model
from easyalign.vad.utils import encode_metadata, encode_vad_segments
from easyalign.vad.vad import run_vad

vad_dataset = VADAudioDataset(
    audio_paths=["audio_mono_120.wav"], audio_dir="data", sample_rate=16000
)
dataloader = torch.utils.data.DataLoader(
    vad_dataset, batch_size=1, shuffle=False, collate_fn=vad_collate_fn
)
audio_dict = next(iter(dataloader))

model = load_vad_model()
audio = audio_dict["audio"][0]
res = run_vad(
    audio_path=audio_dict["audio_path"][0],
    audio_dir="data",
    model=model,
    audio=audio,
    chunk_size=30,
)

logit_dataset = AudioFileDataset(
    metadata=[res],
    audio_dir="data",
    sample_rate=16000,
    model_name="KBLab/wav2vec2-large-voxrex-swedish",
    chunk_size=30,
    use_vad=False,
)

slice_dataset = logit_dataset[0]
slice_dataset["dataset"][1]

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
