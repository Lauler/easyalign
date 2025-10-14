import msgspec


class WordSegment(msgspec.Struct):
    """
    Word-level alignment data.
    """

    text: str
    start: float  # in seconds
    end: float  # in seconds
    score: float | None = None  # Optional confidence score

    def to_dict(self):
        return {f: getattr(self, f) for f in self.__struct_fields__}


class AudioChunk(msgspec.Struct):
    """
    Segment of audio, usually created by VAD.
    """

    start: float  # in seconds
    end: float  # in seconds
    duration: float | None = None  # in seconds
    audio_frames: int | None = None  # Number of audio frames chunk spans

    def to_dict(self):
        return {f: getattr(self, f) for f in self.__struct_fields__}

    def calculate_duration(self):
        self.duration = self.end - self.start
        return self.duration

    def __post_init__(self):
        if self.duration is None:
            self.calculate_duration()


class AlignmentSegment(msgspec.Struct):
    """
    A segment of aligned audio and text.

    This can be sentence, paragraph, or any other unit of text.
    """

    start: float  # in seconds
    end: float  # in seconds
    text: str
    words = list[WordSegment]
    duration: float | None = None  # in seconds
    score: float | None = None  # Optional confidence score

    def to_dict(self):
        return {f: getattr(self, f) for f in self.__struct_fields__}

    def calculate_duration(self):
        self.duration = self.end - self.start
        return self.duration

    def __post_init__(self):
        if self.duration is None:
            self.calculate_duration()


class SpeechSegment(msgspec.Struct):
    """
    A speech slice composed of multiple aligned audio segments.

    May be a speech given by a single speaker, or a dialogue between multiple speakers.
    Whatever unit of abstraction the user prefers.

    If no SpeechSegment is defined, the entire audio is treated as a single speech.
    """

    start: float  # in seconds
    end: float  # in seconds
    text: list[str] | None = None  # Optional text transcription (manual, or created by ASR)
    # If `text_spans` is supplied, custom segments of the text will be aligned to audio.
    # Each tuple is (start_char, end_char) in the `text`.
    text_spans: list[tuple[int, int]] | None = None
    chunks: list[AudioChunk] = []  # Audio chunks from which we create w2v2 logits
    alignments: list[AlignmentSegment] = []  # Aligned text segments
    duration: float | None = None  # in seconds
    audio_frames: int | None = None  # Number of audio frames speech segment spans
    speech_id: str | int | None = None
    probs_path: str | None = None  # Path to saved wav2vec2 emissions/probs
    metadata: dict | None = None  # Extra metadata such as speaker name, etc.

    def to_dict(self):
        return {f: getattr(self, f) for f in self.__struct_fields__}

    def calculate_duration(self):
        self.duration = self.end - self.start
        return self.duration

    def __post_init__(self):
        if self.duration is None:
            self.calculate_duration()

        if self.text is not None and self.text_spans is None:
            if len(self.text) == 1:
                self.text_spans = [(0, len(self.text[0]))]
            else:
                self.text_spans = []
                current_begin = 0
                for t in self.text:
                    self.text_spans.append((current_begin, current_begin + len(t)))
                    current_begin += len(t)


class AudioMetadata(msgspec.Struct):
    """
    Data model for the metadata of an audio file.
    """

    audio_path: str
    sample_rate: int
    duration: float  # in seconds
    speeches: list[SpeechSegment] = []
    metadata: dict | None = None  # Optional extra metadata

    def to_dict(self):
        return {f: getattr(self, f) for f in self.__struct_fields__}
