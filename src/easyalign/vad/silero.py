from silero_vad import get_speech_timestamps, load_silero_vad


def load_vad_model(onnx=False, opset_version=16):
    return load_silero_vad(onnx=onnx, onnx_opset_version=opset_version)


def merge_chunks(segments, chunk_size=30):
    current_start = segments[0]["start"]
    current_end = 0
    merged_segments = []
    subsegments = []

    for segment in segments:
        if segment["end"] - current_start > chunk_size and current_end - current_start > 0:
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
