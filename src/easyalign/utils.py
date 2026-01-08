import subprocess
from pathlib import Path

import msgspec
import numpy as np

from easyalign.alignment.utils import segment_speech_probs
from easyalign.data.datamodel import AudioMetadata


def convert_audio_to_wav(input_file, output_file, sample_rate=16000):
    # fmt: off
    command = [
        'ffmpeg',
        '-i', input_file,
        '-ar', str(sample_rate),  # Set the audio sample rate
        '-ac', '1',      # Set the number of audio channels to 1 (mono)
        '-c:a', 'pcm_s16le',
        '-loglevel', 'warning',
        '-hide_banner',
        '-nostats',
        '-nostdin',
        output_file
    ]
    # fmt: on
    subprocess.run(command)


def numpy_encoder(obj):
    """Custom encoder function for NumPy floats for msgspec."""
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def save_metadata_json(
    metadata: AudioMetadata, output_dir: str = "output/emissions", indent: int | None = 2
):
    audio_path = metadata.audio_path
    json_encoder = msgspec.json.Encoder(enc_hook=numpy_encoder)
    json_msgspec = json_encoder.encode(metadata)
    if indent is not None:
        json_msgspec = msgspec.json.format(json_msgspec, indent=indent)
    json_path = Path(output_dir) / Path(audio_path).parent / (Path(audio_path).stem + ".json")
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "wb") as f:
        f.write(json_msgspec)


def save_metadata_msgpack(metadata: AudioMetadata, output_dir: str = "output/emissions"):
    audio_path = metadata.audio_path
    msgpack_encoder = msgspec.msgpack.Encoder(enc_hook=numpy_encoder)
    msgpack_msgspec = msgpack_encoder.encode(metadata)
    msgpack_path = (
        Path(output_dir) / Path(audio_path).parent / (Path(audio_path).stem + ".msgpack")
    )
    Path(msgpack_path).parent.mkdir(parents=True, exist_ok=True)
    with open(msgpack_path, "wb") as f:
        f.write(msgpack_msgspec)


def save_emissions_and_metadata(
    metadata: AudioMetadata,
    probs_list: list[np.ndarray],
    speech_ids: list[str],
    indent: int | None = 2,
    save_json: bool = True,
    save_msgpack: bool = False,
    save_emissions: bool = True,
    return_emissions: bool = False,
    output_dir: str = "output/emissions",
) -> tuple[AudioMetadata | None, list[np.ndarray] | None]:
    audio_path = metadata.audio_path
    base_path = Path(audio_path).parent / Path(audio_path).stem
    speech_index = 0

    if save_emissions:
        # Segment the probs according to speech_ids and save each speech's probs separately
        for speech_id, probs in segment_speech_probs(probs_list, speech_ids):
            probs_path = Path(output_dir) / base_path / f"{speech_id}.npy"
            Path(probs_path).parent.mkdir(parents=True, exist_ok=True)

            probs_base_path = Path(probs_path).relative_to(Path(output_dir))
            metadata.speeches[speech_index].probs_path = str(probs_base_path)
            speech_index += 1
            np.save(probs_path, probs)

    if save_json:
        save_metadata_json(metadata, output_dir=output_dir, indent=indent)
    if save_msgpack:
        save_metadata_msgpack(metadata, output_dir=output_dir)

    if return_emissions:
        emissions = []
        for _, probs in segment_speech_probs(probs_list, speech_ids):
            emissions.append(probs)

        return metadata, emissions

    return None, None


def save_alignments(
    metadata: AudioMetadata,
    alignments: list[dict],
    save_json: bool = True,
    save_msgpack: bool = False,
    output_dir: str = "output/alignments",
    indent: int | None = 2,
):
    audio_path = metadata.audio_path
    base_path = Path(audio_path).parent / Path(audio_path).stem

    json_encoder = msgspec.json.Encoder(enc_hook=numpy_encoder)
    msgpack_encoder = msgspec.msgpack.Encoder(enc_hook=numpy_encoder)

    for speech, alignment in zip(metadata.speeches, alignments):
        alignment_path = Path(output_dir) / base_path / f"{speech.speech_id}_alignment.json"
        Path(alignment_path).parent.mkdir(parents=True, exist_ok=True)

        alignment_base_path = Path(alignment_path).relative_to(Path(output_dir))
        speech.alignment_path = str(alignment_base_path)

        if save_json:
            alignment_msgspec = json_encoder.encode(alignment)
            if indent is not None:
                alignment_msgspec = msgspec.json.format(alignment_msgspec, indent=indent)
            with open(alignment_path, "wb") as f:
                f.write(alignment_msgspec)

        if save_msgpack:
            alignment_msgpack = msgpack_encoder.encode(alignment)
            # Replace .json with .msgpack
            msgpack_path = alignment_path.with_suffix(".msgpack")
            with open(msgpack_path, "wb") as f:
                f.write(alignment_msgpack)
