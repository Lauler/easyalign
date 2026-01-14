import itertools

import numpy as np
import torch

from easyalign.data.datamodel import AudioMetadata


def _calculate_receptive_field(
    conv_kernel: list[int],
    conv_stride: list[int],
) -> int:
    """
    Calculate the receptive field of the model based on convolutional kernel sizes and strides.

    Formula: RF_next = RF_prev + (kernel - 1) * accumulated_stride

    Parameters
    ----------
    conv_kernel : list of int
        List of kernel sizes.
    conv_stride : list of int
        List of strides.

    Returns
    -------
    int
        Receptive field size.
    """
    receptive_field = 1
    accumulated_stride = 1
    for kernel_size, stride in zip(conv_kernel, conv_stride):
        receptive_field += (kernel_size - 1) * accumulated_stride
        accumulated_stride *= stride

    return receptive_field


def _compute_logits(
    frames: int,
    receptive_field: int,
    conv_kernel: list[int] = [10, 3, 3, 3, 3, 2, 2],
    conv_stride: list[int] = [5, 2, 2, 2, 2, 2, 2],
    add_adapter: bool = False,
    num_adapter_layers: int = 0,
    adapter_stride: int = 2,
) -> int:
    """
    Compute the number of output logits for a given number of input frames.

    Mimics Hugging Face's `_get_feat_extract_output_lengths` logic. This implementation
    however always pads the input to at least the receptive field size. Our calculation
    will therefore always output 1 logit for inputs smaller than the receptive field.

    See: https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1073

    Parameters
    ----------
    frames : int
        Number of input frames.
    receptive_field : int
        Receptive field size.
    conv_kernel : list of int, default [10, 3, 3, 3, 3, 2, 2]
        List of kernel sizes.
    conv_stride : list of int, default [5, 2, 2, 2, 2, 2, 2]
        List of strides.
    add_adapter : bool, default False
        Whether an adapter is added.
    num_adapter_layers : int, default 0
        Number of adapter layers.
    adapter_stride : int, default 2
        Stride of the adapter.

    Returns
    -------
    int
        Number of output logits.
    """

    # Pad input frames to at least the receptive field (see `pad_` easyalign/data/collators.py)
    current_logits = max(frames, receptive_field)  # This is still in frames

    # CNN layers
    for kernel_size, stride in zip(conv_kernel, conv_stride):
        current_logits = torch.div(current_logits - kernel_size, stride, rounding_mode="floor") + 1

    if add_adapter:
        for _ in range(num_adapter_layers):
            current_logits = (
                torch.div(current_logits - 1, adapter_stride, rounding_mode="floor") + 1
            )

    return int(current_logits.item())


def get_output_logits_length(
    audio_frames: int,
    chunk_size: float,
    conv_kernel: list[int] = [10, 3, 3, 3, 3, 2, 2],
    conv_stride: list[int] = [5, 2, 2, 2, 2, 2, 2],
    add_adapter: bool = False,
    num_adapter_layers: int = 0,
    adapter_stride: int = 2,
    sample_rate: int = 16000,
) -> int:
    """
    Calculates the total number of output logits for a given audio length (in frames).

    Flexibly handles different models and configurations with varying convolutional kernel sizes,
    strides, numbers of adapter layers, as well as possible chunking of the audio input.

    Parameters
    ----------
    audio_frames : int
        Number of audio frames in the audio file, or part of audio file to be
        aligned.
    chunk_size : float
        Number of seconds the audio was chunked by for batched inference or VAD.
    conv_kernel : list of int, default [10, 3, 3, 3, 3, 2, 2]
        The convolutional kernel sizes of the emissions model
        (see `model.config.conv_kernel` for default values).
    conv_stride : list of int, default [5, 2, 2, 2, 2, 2, 2]
        The convolutional stride of the emissions model
        (see `model.config.conv_stride`).
    add_adapter : bool, default False
        Whether a convolutional network should be stacked on top of the
        wav2vec2 encoder.
    num_adapter_layers : int, default 0
        Number of adapter layers in the model
        (`model.config.num_adapter_layers`).
    adapter_stride : int, default 2
        The stride of each adapter layer (`model.config.adapter_stride`).
    sample_rate : int, default 16000
        The sample rate of the w2v processor, default 16000.

    Returns
    -------
    int
        Total number of output logits.
    """
    receptive_field = _calculate_receptive_field(conv_kernel, conv_stride)

    frames_per_chunk = int(chunk_size * sample_rate)
    num_full_chunks = audio_frames // frames_per_chunk
    remainder_frames = audio_frames % frames_per_chunk

    # Calculate logits for full chunks
    logits_per_full_chunk = _compute_logits(
        frames_per_chunk,
        receptive_field=receptive_field,
        add_adapter=add_adapter,
        num_adapter_layers=num_adapter_layers,
        adapter_stride=adapter_stride,
    )
    total_logits = num_full_chunks * logits_per_full_chunk

    # Add logits for the remainder frames
    if remainder_frames > 0:
        total_logits += _compute_logits(
            remainder_frames,
            receptive_field=receptive_field,
            num_adapter_layers=num_adapter_layers,
            adapter_stride=adapter_stride,
        )

    return total_logits


def add_logits_to_metadata(
    model, metadata: AudioMetadata, chunk_size: float, sample_rate: int = 16000
):
    """
    Adds the number of output logits to each chunk in the metadata based on the
    model configuration.

    Parameters
    ----------
    model : object
        The emissions model (e.g., Wav2Vec2ForCTC) used for alignment.
    metadata : AudioMetadata
        List of AudioMetadata objects containing SpeechSegments.
    chunk_size : float
        Number of seconds the audio was chunked by for batched inference or VAD.
    sample_rate : int, default 16000
        The sample rate of the w2v processor.

    Returns
    -------
    AudioMetadata
        Updated metadata object.
    """
    conv_kernel = model.config.conv_kernel
    conv_stride = model.config.conv_stride
    num_adapter_layers = getattr(model.config, "num_adapter_layers", 0)
    adapter_stride = getattr(model.config, "adapter_stride", 2)

    for speech in metadata.speeches:
        for chunk in speech.chunks:
            if chunk.audio_frames is not None:
                chunk.num_logits = get_output_logits_length(
                    audio_frames=chunk.audio_frames,
                    chunk_size=chunk_size,
                    conv_kernel=conv_kernel,
                    conv_stride=conv_stride,
                    num_adapter_layers=num_adapter_layers,
                    adapter_stride=adapter_stride,
                    sample_rate=sample_rate,
                )

    return metadata


def segment_speech_probs(probs_list: list[np.ndarray], speech_ids: list[str] | list[int]):
    """
    Divide the accumulated probs of audio file into the speeches they belong to.
    (we can't assume that a batch maps to a single speech)

    Parameters
    ----------
    probs_list : list of np.ndarray
        List of np.ndarrays containing the probs
        with shape (batch_size, seq_len, vocab_size).
    speech_ids : list of str or list of int
        List of speech ids that each chunk (observation)
        in the probs_list belongs to.

    Yields
    ------
    str or int
        Speech ID.
    np.ndarray
        Probabilities for the speech segment.
    """
    # Count the number of chunks per speech id
    speech_chunk_counts = [
        (key, sum(1 for i in group)) for key, group in itertools.groupby(speech_ids)
    ]
    # Create a list of indices where each speech chunk starts
    split_indices = list(itertools.accumulate([count for _, count in speech_chunk_counts]))[:-1]

    probs_in_speech = np.concatenate(probs_list, axis=0)
    probs_split = np.split(probs_in_speech, split_indices, axis=0)
    unique_speech_ids = dict.fromkeys(speech_ids).keys()  # Preserves order

    assert len(speech_chunk_counts) == len(probs_split) == len(set(unique_speech_ids))
    for speech_id, probs in zip(unique_speech_ids, probs_split):
        yield speech_id, probs
