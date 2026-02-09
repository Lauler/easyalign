import numpy as np
import torch


def pad_to_min_length(vec: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Pad a vector to a minimum length.

    Wav2Vec2 models require a minimum input length of 400 frames.
    We pad to 640 frames to be safe, so other models with larger receptive fields also work.

    Parameters
    ----------
    vec : np.ndarray or torch.Tensor
        Input vector.

    Returns
    -------
    torch.Tensor
        Padded vector.
    """
    audio_frames = torch.as_tensor(vec.shape[-1]).to(vec.device)
    if audio_frames < 640:
        vec = torch.nn.functional.pad(vec, (0, 640 - audio_frames))

    return vec


def alignment_collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for alignment.

    We need to pad the input_values to the longest sequence,
    since wav2vec2 doesn't do this by default.
    The individual elements in the batch are tuples: (feature, speech_id)

    Parameters
    ----------
    batch : list of dict
        Batch of features.

    Returns
    -------
    dict
        Collated batch.
    """
    # Remove None values
    speech_ids = []
    start_times = []
    input_lengths = []

    for b in batch:
        if b is not None:
            speech_ids.append(b["speech_id"])
            start_times.append(b["start_time_global"])
            input_lengths.append(b["feature"][0].shape[1])

    batch = [pad_to_min_length(b["feature"][0].squeeze(0)) for b in batch if b is not None]

    # Pad the input_values to the longest sequence
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)

    return {
        "features": batch,
        "start_times": start_times,
        "speech_ids": speech_ids,
        "input_lengths": input_lengths,
    }


def vad_collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for VAD.

    Parameters
    ----------
    batch : list of dict
        Batch of audio data.

    Returns
    -------
    dict
        Collated batch.
    """
    audios = [item["audio"] for item in batch]
    sample_rates = [item["sample_rate"] for item in batch]
    audio_paths = [item["audio_path"] for item in batch]
    audio_dirs = [item["audio_dir"] for item in batch]

    audios = torch.tensor(np.array(audios)).to(torch.float32)

    return {
        "audio": audios,
        "sample_rate": sample_rates,
        "audio_path": audio_paths,
        "audio_dir": audio_dirs,
    }


def transcribe_collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for transcription.

    Parameters
    ----------
    batch : list of dict
        Batch of features.

    Returns
    -------
    dict
        Collated batch.
    """
    # Remove None values
    speech_ids = [b["speech_id"] for b in batch if b is not None]
    start_times = [b["start_time_global"] for b in batch if b is not None]
    batch = [b["feature"] for b in batch if b is not None]

    # Concat, keep batch dimension
    batch = torch.cat(batch, dim=0)

    return {
        "features": batch,
        "start_times": start_times,
        "speech_ids": speech_ids,
    }


def audiofile_collate_fn(batch: list[dict]) -> list:
    """
    Collate function to allow dictionaries with Datasets in the batch.

    Parameters
    ----------
    batch : list of dict
        Batch of datasets.

    Returns
    -------
    list of dict
        Collated batch.
    """
    # Remove None values
    batch = [b for b in batch if b is not None]

    # Return None if batch is empty
    if len(batch) == 0:
        return None

    # Return the batch
    return batch


def metadata_collate_fn(batch: list) -> list:
    """
    Collate function to allow dictionaries with AudioMetadata objects in the batch.

    Parameters
    ----------
    batch : list
        Batch of metadata.

    Returns
    -------
    list
        Collated batch.
    """
    batch = [b for b in batch if b is not None]

    return batch
