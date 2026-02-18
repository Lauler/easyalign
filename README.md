# Easier forced alignment with `easyaligner`

<div align="center"><img width="1020" height="340" alt="image" src="https://github.com/user-attachments/assets/a3589539-5c85-4ac1-a4a7-d5e801207faa" /></div>

`easyaligner` is a fast and memory efficient forced alignment pipeline for speech and text. Given a text transcript, `easyaligner` will help identify where each word or phrase was spoken in the audio. The library supports aligning both from ground-truth transcripts, as well as from ASR-generated transcripts (`easyaligner` acts as the backend that powers alignment in [`easywhisper`](https://github.com/kb-labb/easywhisper)). Some notable features of `easyaligner` include:

* **GPU accelerated forced alignment**. Uses [Pytorch's forced alignment API](https://docs.pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html) with a GPU based implementation of the Viterbi algorithm. Enables fast and memory-efficient forced alignment of long audio segments ([Pratap et al., 2024](https://jmlr.org/papers/volume25/23-1318/23-1318.pdf#page=8)). 
* **Flexible text normalization for improved alignment quality**. Users can supply custom regex-based text normalization functions to preprocess transcripts before alignment. A mapping from the original text to the normalized text is maintained internally. All of the applied normalizations and transformations are consequently **non-destructive and reversible after alignment**.  
* **Batch processing support for emission extraction**. `easyaligner` supports batched inference for wav2vec2-based models, keeping track of non-padded logits when doing alignment.   
* **Modular pipeline design**. The library has separate, independent, pipelines for VAD, emission extraction, and forced alignment. Users can run everything end-to-end, or run the separate stages individually. 

## Installation

### With GPU support (recommended)

```bash
pip install easyaligner --extra-index-url https://download.pytorch.org/whl/cu128
```

> [!TIP]  
> Remove `--extra-index-url` if you want a CPU-only installation.

### Using uv

When installing with [uv](https://docs.astral.sh/uv/), it will select the appropriate PyTorch version automatically (CPU for macOS, CUDA for Linux/Windows/ARM):

```bash
uv pip install easyaligner
```

## Usage

```python
from transformers import (
    AutoModelForCTC,
    Wav2Vec2Processor,
)

from easyaligner.data.datamodel import SpeechSegment
from easyaligner.pipelines import pipeline
from easyaligner.text.normalization import (
    SpanMapNormalizer,
    text_normalizer,
)
from easyaligner.text.tokenizer import load_tokenizer
from easyaligner.vad.pyannote import load_vad_model

text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
"""

tokenizer = load_tokenizer(language="swedish") # sentence tokenizer 
text = text.strip()
span_list = list(tokenizer.span_tokenize(text)) # start, end character indices for each sentence

speeches = [[SpeechSegment(speech_id=0, text=text, text_spans=span_list, start=None, end=None)]]

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
    text_normalizer_fn=text_normalizer,
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
```

## Outputs

By default, `easyaligner` saves the outputs of each stage of the pipeline (VAD, emission extraction, forced alignment) as JSON files in separate directories. The final aligned output can be found in `output/alignments`. The directory structure after running the full pipeline will look as follows:  

```
output
├── alignments
├── emissions
└── vad
```

The `output/emissions` directory will, in addition to the JSON files, also contain output emissions for each JSON file in `.npy` format.  

All intermediate files can safely be deleted, assuming there is no need to re-run the pipeline from a specific intermediate stage. 

## Logging and Error Handling

### Enabling Logging

To see progress and error messages one can add the following logging configuration at the start of a script:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    # handlers=[
    #     logging.FileHandler('easyaligner.log'), # Log to a file
    #     logging.StreamHandler()  # Also print to console
    # ]
)
```

### Error Handling

`easyaligner` pipelines use PyTorch DataLoaders for efficient parallel processing and prefetching of data. During processing, the library silently skips files that fail to load (corrupted audio, missing file, etc.). The errors are logged with full traceback, the pipeline however continues processing the remaining files. 

`easyaligner` leaves it up the user to decide how to handle failed files (retry, validate inputs, etc.).

> [!TIP]  
> Track which files failed after processing completes by comparing output:

```python
from pathlib import Path

# After pipeline completes, check which files produced output
output_files = list(Path("output/vad").rglob("*.json"))
output_stems = {f.stem for f in output_files}

# Find files that failed (no output produced)
failed = [p for p in audio_paths if Path(p).stem not in output_stems]
```

