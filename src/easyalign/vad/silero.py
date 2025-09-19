from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
from tqdm import tqdm

from easyalign.data.datamodel import AudioMetadata, SpeechSegment
from easyalign.vad.utils import encode_vad_segments


def load_vad_model(onnx=False, opset_version=16):
    return load_silero_vad(onnx=onnx, onnx_opset_version=opset_version)


def merge_chunks(segments, chunk_size=30):
    current_start = segments[0]["start"]
    current_end = 0
    merged_segments = []
    subsegments = []

    for segment in segments:
        if (
            segment["end"] - current_start > chunk_size
            and current_end - current_start > 0
        ):
            merged_segments.append(
                {"start": current_start, "end": current_end, "segments": subsegments}
            )
            current_start = segment["start"]
            subsegments = []
        current_end = segment["start"]
        subsegments.append((segment["start"], segment["end"]))

    merged_segments.append(
        {"start": current_start, "end": segments[-1]["end"], "segments": subsegments}
    )
    return merged_segments


def run_vad_pipeline(metadata: AudioMetadata, model, chunk_size=30):
    """
    Run VAD pipeline on the given audio metadata.

    Args:
        metadata (AudioMetadata): The audio metadata object.
        vad_pipeline: The VAD pipeline function.
        backend (str): The VAD backend to use ("silero" or "pyannote").
        chunk_size (int): The maximum chunk size in seconds.
    """

    audio, sr = read_audio(metadata.audio_path)
    if audio is None:
        return None

    if len(metadata.speeches) > 0:
        # Run VAD on each speech segment
        for speech in tqdm(metadata.speeches, desc="Running VAD on speeches"):
            speech_audio = audio[int(speech.start * sr) : int(speech.end * sr)]
            vad_segments = get_speech_timestamps(
                speech_audio,
                model,
                max_speech_duration_s=chunk_size,
                return_seconds=True,
            )
            vad_segments = merge_chunks(vad_segments, chunk_size=chunk_size)
            # Add speech.start offset to each segment
            print(vad_segments)
            vad_segments = [
                {
                    "start": seg["start"] + speech.start,
                    "end": seg["end"] + speech.start,
                    "segments": seg["segments"],
                }
                for seg in vad_segments
            ]
            segments = encode_vad_segments(vad_segments)
            speech.chunks = segments
    else:
        # Run VAD on entire audio
        vad_segments = get_speech_timestamps(
            audio,
            model,
            max_speech_duration_s=chunk_size,
            return_seconds=True,
        )

        vad_segments = merge_chunks(vad_segments, chunk_size=chunk_size)
        segments = encode_vad_segments(vad_segments)
        metadata.speeches.append(
            SpeechSegment(start=0, end=metadata.duration, text=None, chunks=segments)
        )

    return metadata
