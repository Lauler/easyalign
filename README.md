# Easier forced alignment with `easyaligner`

`easyaligner` is a fast and memory efficient forced alignment pipeline for aligning speech and text. It is designed with ease of use in mind, supporting alignment both from ground-truth transcripts, as well as from ASR-generated transcripts. `easyaligner` acts as the backend that powers alignment in [`easywhisper`](https://github.com/kb-labb/easywhisper). Some notable features of `easyaligner` include:

* Uses [Pytorch's forced alignment API](https://docs.pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html) with support for efficient GPU accelerated forced alignment. Enables aligning long audio segments fast and memory-efficiently ([Pratap et al., 2024](https://jmlr.org/papers/volume25/23-1318/23-1318.pdf)). 
* Supports **custom regex-based text normalization** functionality to preprocess transcripts before alignment, in order to improve alignment quality. Maintains a mapping from original to normalized text, meaning the **normalizations and transformations are non-destructive** and reversible after alignment.  
* Separates VAD, emission extraction (emissions are written to disk), and alignment into modular pipeline stages. Allows users to run everything end-to-end, or to run the separate stages individually (better flexibility for parallelization).

## Installation

### With GPU support (recommended)

```bash
pip install easyaligner --extra-index-url https://download.pytorch.org/whl/cu128
```

> [!TIP]  
> Remove `--extra-index-url` if you want CPU-only installation.

### Using uv

When installing with [uv](https://docs.astral.sh/uv/), it will select the appropriate PyTorch version automatically (CPU for macOS, CUDA for Linux/Windows/ARM):

```bash
uv pip install easyaligner
```

### For development

```bash
git clone https://github.com/kb-labb/easyaligner.git
cd easyaligner

pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
```

## Logging and Error Handling

### Enabling Logging

To see progress and error messages one can add the following logging configuration at the start of a script:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    # handlers=[
    #     logging.FileHandler('easyalign.log'), # Log to a file
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

