import numpy as np
import torch


def pad_to_min_length(vec):
    audio_frames = torch.as_tensor(vec.shape[-1]).to(vec.device)
    if audio_frames < 640:
        vec = torch.nn.functional.pad(vec, (0, 640 - audio_frames))

    return vec


def alignment_collate_fn(batch):
    """
    We need to pad the input_values to the longest sequence,
    since wav2vec2 doesn't do this by default.
    The individual elements in the batch are tuples: (feature, speech_id)
    """
    # Remove None values
    speech_ids = [b[1] for b in batch if b is not None]
    batch = [pad_to_min_length(b[0][0].squeeze(0)) for b in batch if b is not None]

    # Pad the input_values to the longest sequence
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)

    return {
        "features": batch,
        "speech_ids": speech_ids,
    }


def vad_collate_fn(batch):
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


def audiofile_collate_fn(batch: dict) -> list:
    """
    Collate function to allow dictionaries with Datasets in the batch.
    """
    # Remove None values
    batch = [b for b in batch if b is not None]

    # Return None if batch is empty
    if len(batch) == 0:
        return None

    # Return the batch
    return batch
