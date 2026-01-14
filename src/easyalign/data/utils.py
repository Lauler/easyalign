from pathlib import Path

import msgspec
import numpy as np

from easyalign.data.datamodel import AudioMetadata


def pad_probs(probs, maximum_nr_logits: int):
    """
    Pad `probs` to match the maximum number of logits based on chunk size and sample rate.

    `probs` has the shape (batch_size, nr_logits, vocab_size).

    Pytorch's forced alignment API expects tensors with matching nr_logits shape. This function
    pads the `nr_logits` dimension from the wav2vec2 output up to the number of logits that an
    input with `chunk_size * sample_rate` would produce (the maximum possible based on our
    current chunking strategy).

    Usually the collator handles the padding when the batch contains at least 1 obs that is
    `chunk_size` long. However, if the entire batch contains only observations shorter than
    `chunk_size`, `probs` needs to be padded accordingly.

    Parameters
    ----------
    probs : np.ndarray
        Array containing probabilities.
    maximum_nr_logits : int
        Maximum number of logits to pad to.

    Returns
    -------
    np.ndarray
        Padded probabilities.
    """
    assert probs.shape[1] <= maximum_nr_logits, (
        f"Number of logits in `probs` ({probs.shape[1]}) exceeds the maximum number of logits "
        f"({maximum_nr_logits}). Did you use a consistent `chunk_size` and `sample_rate` "
        "throughout your pipeline?"
    )
    probs = np.pad(
        array=probs,
        pad_width=(
            (0, 0),
            (0, maximum_nr_logits - probs.shape[1]),  # Add remaining logits as padding
            (0, 0),
        ),
        mode="constant",
    )
    return probs


def read_json(json_path: str | Path) -> AudioMetadata:
    """
    Convenience function to read a JSON file and parse it into an `AudioMetadata` object.

    For better performance, use `JSONMetadataDataset` in `easyalign.data.dataset`.

    Parameters
    ----------
    json_path : str or Path
        Path to the JSON file.

    Returns
    -------
    AudioMetadata
        Parsed AudioMetadata object.
    """
    with open(json_path, "r") as f:
        return msgspec.json.decode(f.read(), type=AudioMetadata)
