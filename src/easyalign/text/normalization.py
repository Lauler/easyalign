import re
import unicodedata
from typing import Union

from easyalign.text.languages.sv import abbreviations, ocr_corrections, symbols


def text_normalizer(text: str) -> str:
    """
    Default text normalization function.

    Applies
        - Unicode normalization (NFKC)
        - Lowercasing
        - Normalization of whitespace
        - Remove parentheses and special characters

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list of str
        List of normalized tokens.
    list of dict
        Mapping between tokens and original text spans.
    """
    # Unicode normalization
    normalizer = SpanMapNormalizer(text)
    # # Remove parentheses, brackets, stars, and their content
    # normalizer.transform(r"\(.*?\)", "")
    # normalizer.transform(r"\[.*?\]", "")
    # normalizer.transform(r"\*.*?\*", "")

    # Unicode normalization on tokens and lowercasing
    normalizer.transform(r"\S+", lambda m: unicodedata.normalize("NFKC", m.group()))
    normalizer.transform(r"\S+", lambda m: m.group().lower())
    normalizer.transform(r"[^\w\s]", "")  # Remove punctuation and special characters
    normalizer.transform(r"\s+", " ")  # Normalize whitespace to a single space
    normalizer.transform(r"^\s+|\s+$", "")  # Strip leading and trailing whitespace

    mapping = normalizer.get_token_map()
    normalized_tokens = [item["normalized_token"] for item in mapping]
    return normalized_tokens, mapping


def format_symbols_abbreviations():
    """
    Formats abbreviations into dicts that include the pattern (abbreviation)
    and replacement (expansion).

    Follows the same logic as the user-supplied patterns in collect_regex_patterns.

    Returns
    -------
    list of dict
        List of abbreviation patterns.
    """
    abbreviation_patterns = []
    for abbreviation, expansion in abbreviations.items():
        abbreviation_patterns.append(
            {
                "pattern": re.escape(abbreviation),
                "replacement": expansion,
                "transformation_type": "substitution",
            }
        )

    for symbol, expansion in symbols.items():
        abbreviation_patterns.append(
            {
                "pattern": re.escape(symbol),
                "replacement": expansion,
                "transformation_type": "substitution",
            }
        )

    return abbreviation_patterns


class SpanMapNormalizer:
    def __init__(self, text: str):
        self.original_text = text
        self.current_text = text
        self.span_map = [(i, i + 1) for i in range(len(text))]

    def transform(self, pattern: str, replacement: Union[str, callable]):
        """
        Apply a regex transformation to the current text, while keeping track of the
        character span that every character in the new text maps to in the original text.

        In the example below, the 4 characters in the replacement "I am" map to the
        match pattern "I'm" at span (0, 3) of the original text.

        Example text: "I'm sorry"
        Example pattern: r"I'm"
        Example replacement: "I am"

        new_text: "I am sorry"
        new_span_map: [(0, 3), (0, 3), (0, 3), (0, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]

        Parameters
        ----------
        pattern : str
            The regex pattern to match.
        replacement : str or callable
            The replacement string or a function that takes a
            match object and returns a replacement string.
        """

        new_text, new_span_map, last_end = "", [], 0
        for match in re.finditer(pattern, self.current_text):
            start, end = match.span()
            new_text += self.current_text[last_end:start]
            new_span_map.extend(self.span_map[last_end:start])
            replacement_text = replacement(match) if callable(replacement) else replacement
            if start < end:
                source_span = (self.span_map[start][0], self.span_map[end - 1][1])
            else:
                # if zero-width match, use the start position
                source_pos = (
                    self.span_map[start][0]
                    if start < len(self.span_map)
                    else len(self.original_text)
                )
                source_span = (source_pos, source_pos)
            new_text += replacement_text
            new_span_map.extend([source_span] * len(replacement_text))
            last_end = end
        new_text += self.current_text[last_end:]
        new_span_map.extend(self.span_map[last_end:])
        self.current_text, self.span_map = new_text, new_span_map

    def get_token_map(self, tokenization_level="word") -> list[dict]:
        """
        Tokenize the current text and create a mapping of normalized tokens to the
        original text spans they were normalized from.

        Parameters
        ----------
        tokenization_level : str, default "word"
            Tokenization level ('word' or 'char').

        Returns
        -------
        list of dict
            Token mapping.
        """
        if tokenization_level == "word":
            # Match any sequence of non-whitespace characters
            tokenization_pattern = r"\S+"
        elif tokenization_level == "char":
            # Any non-whitespace character
            tokenization_pattern = r"\S"

        token_mapping = []
        for match in re.finditer(tokenization_pattern, self.current_text):
            norm_token = match.group(0)
            norm_start, norm_end = match.span()
            if not self.span_map or norm_start >= len(self.span_map):
                continue
            original_start = self.span_map[norm_start][0]
            original_end = self.span_map[norm_end - 1][1]
            original_text = self.original_text[original_start:original_end]
            token_mapping.append(
                {
                    "normalized_token": norm_token,
                    "text": original_text,
                    "start_char": original_start,
                    "end_char": original_end,
                }
            )
        return token_mapping


def merge_multitoken_expressions(timestamp_mapping: list[dict]) -> list[dict]:
    """
    Merge any multi-token expressions in the mapping.

    If multiple normalized tokens share (map to) the same original source span, this
    function will concatenate them into a single entry. The original text span
    will be assigned the start and end times of the first and last token, respectively,
    in the multi-token expression.

    Converts the input mapping from having one entry per normalized token
    to having one entry per "token" (span) in the original text.

    Parameters
    ----------
    timestamp_mapping : list of dict
        Normalized tokens, their original text, start and
        end character indices, and their timestamps.

    Returns
    -------
    list of dict
        A list of dictionaries with merged multi-token entries.
    """

    if not timestamp_mapping:
        return []

    merged_mapping, start_token_index = [], 0
    while start_token_index < len(timestamp_mapping):
        timestamp_token = timestamp_mapping[start_token_index]
        end_token_index = start_token_index + 1

        # Group multi-token expressions by their identical original source spans
        while (
            end_token_index < len(timestamp_mapping)
            and timestamp_mapping[end_token_index]["start_char"] == timestamp_token["start_char"]
            and timestamp_mapping[end_token_index]["end_char"] == timestamp_token["end_char"]
        ):
            end_token_index += 1

        # Subset the (possible) multi-token group
        multi_token = timestamp_mapping[start_token_index:end_token_index]
        combined_item = {
            "start_time": multi_token[0]["start_time"],
            "end_time": multi_token[-1]["end_time"],
            "text": timestamp_token["text"],  # text before deletions are included
            "start_char": timestamp_token["start_char"],
            "end_char": timestamp_token["end_char"],
            "score": sum([item["score"] for item in multi_token]) / len(multi_token),
            "normalized_tokens": " ".join(item["normalized_token"] for item in multi_token),
        }
        merged_mapping.append(combined_item)
        start_token_index = end_token_index
    return merged_mapping


def add_deletions_to_mapping(merged_map: list[dict], original_text: str) -> list[dict]:
    """
    Takes a mapping of original text and normalized tokens with their timestamps
    and (re)inserts previously deleted text spans into the mapping. This allows
    us to reconstruct the original text (with punctuation and other deletions).

    Deleted text spans are assigned the start and end times of the previous
    adjacent token in the mapping.

    Parameters
    ----------
    merged_map : list of dict
        A list of dictionaries with the mapping of normalized tokens
        to their original text spans and timestamps.
    original_text : str
        The original text from which the mapping was created.

    Returns
    -------
    list of dict
        A list of updated dictionaries with an entry that includes deleted
        text spans (e.g. punctuation). Allows reconstructing the original text.
    """
    if not merged_map:
        return []
    final_map = []
    for i, item in enumerate(merged_map):
        new_item = item.copy()
        # The new end is the start of the next event's original span.
        if i + 1 < len(merged_map):
            new_end = merged_map[i + 1]["start_char"]
        else:  # For the last event, the new end is the end of the original text.
            new_end = len(original_text)

        new_item["end_char_extended"] = new_end
        new_item["text_span_full"] = original_text[new_item["start_char"] : new_end]
        final_map.append(new_item)
    return final_map
