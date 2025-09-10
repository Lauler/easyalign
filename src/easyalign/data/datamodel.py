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


class AudioSegment(msgspec.Struct):
    """
    Segment of audio with (optional) associated text transcription. Alternatively,
    a VAD segment with no text.
    """

    start: float  # in seconds
    end: float  # in seconds
    words: list[WordSegment] = []
    duration: float | None = None  # in seconds
    text: str | None = None  # Optional text transcription

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

    May be a speech given by a single speaker or a dialogue between multiple speakers.
    Whatever unit of abstraction the user prefers.

    If no SpeechSegment is defined, the entire audio is treated as a single speech.
    """

    start: float  # in seconds
    end: float  # in seconds
    text: str
    alignments: list[AudioSegment] = []
    duration: float | None = None  # in seconds
    speech_id: str | int | None = None
    metadata: dict | None = None  # Extra metadata such as speaker name, etc.

    def to_dict(self):
        return {f: getattr(self, f) for f in self.__struct_fields__}

    def calculate_duration(self):
        self.duration = self.end - self.start
        return self.duration

    def __post_init__(self):
        if self.duration is None:
            self.calculate_duration()


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
