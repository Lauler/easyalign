import re
import unicodedata
from collections import OrderedDict

import num2words
import numpy as np

from easyalign.text.languages.sv import abbreviations, ocr_corrections, symbols


def format_symbols_abbreviations():
    """
    Formats abbreviations into dicts that include the pattern (abbreviation)
    and replacement (expansion).

    Follows the same logic as the user-supplied patterns in collect_regex_patterns.
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


def collect_regex_patterns(user_patterns=None):
    """
    Collects regex patterns for text normalization and substitution.

    Args:
        user_patterns (list of dict): User-supplied regex patterns with keys "pattern",
        "replacement", and "transformation_type".

    Returns:
        list of dict: Collected regex patterns with default and user-supplied patterns.
    """

    patterns = []

    # Include abbreviations
    patterns.extend(format_symbols_abbreviations())

    patterns.extend(
        [
            # Capture pattern groups of type '(digits) kap. (digits) §'. For example "4 kap. 7 §".
            # Replace the numbers with ordinals: "fjärde kapitlet sjunde paragrafen"
            {
                "pattern": r"(\d+) kap\. (\d+) \§",
                "replacement": lambda m: (
                    f"{num2words(int(m.group(1)), lang='sv', ordinal=True)} kapitlet "
                    f"{num2words(int(m.group(2)), lang='sv', ordinal=True)} paragrafen"
                ),
                "transformation_type": "substitution",
            },
            {
                "pattern": r"(\d+)[\. ](\d+)",
                "replacement": lambda m: f"{num2words(m.group(1) + m.group(2), lang='sv')}",
                "transformation_type": "substitution",
            },
            # Remove everyting within parentheses including the parentheses
            {"pattern": r"\(.*?\)", "replacement": "", "transformation_type": "deletion"},
            # Replace : or / between digits with whitespace and num2words the digits
            {
                "pattern": r"(\d+):(\d+)",
                "replacement": lambda m: (
                    f"{num2words(int(m.group(1)), lang='sv')} "
                    f"{num2words(int(m.group(2)), lang='sv')}"
                ),
                "transformation_type": "substitution",
            },
            # Replace - between digits with " till " and num2words the digits
            {
                "pattern": r"(\d+)-(\d+)",
                "replacement": lambda m: (
                    f"{num2words(int(m.group(1)), lang='sv')} till "
                    f"{num2words(int(m.group(2)), lang='sv')}"
                ),
                "transformation_type": "substitution",
            },
            # Replace , between digits with " komma " and num2words the digits
            {
                "pattern": r"(\d+),(\d+)",
                "replacement": lambda m: (
                    f"{num2words(int(m.group(1)), lang='sv')} komma "
                    f"{num2words(int(m.group(2)), lang='sv')}"
                ),
                "transformation_type": "substitution",
            },
            {
                "pattern": r"(\d+)[\.\,\:\-\/](\d+)",
                "replacement": lambda m: f"{m.group(1)} {m.group(2)}",
                "transformation_type": "substitution",
            },
            # Replace § with 'paragrafen' if preceded by a number
            {
                "pattern": r"(?<=\d )§",
                "replacement": r"paragrafen",
                "transformation_type": "substitution",
            },
            # Replace § with 'paragraf' if succeeded by a number
            {
                "pattern": r"§(?= \d)",
                "replacement": r"paragraf",
                "transformation_type": "substitution",
            },
            # Remove punctuation between whitespace
            {"pattern": r"\s[^\w\s]\s", "replacement": " ", "transformation_type": "substitution"},
            # Remove punctuation
            {"pattern": r"[^\w\s]", "replacement": "", "transformation_type": "deletion"},
            # Remove multiple spaces (more than one) with a single space
            {"pattern": r"\s{2,}", "replacement": " ", "transformation_type": "substitution"},
            # Strip leading and trailing whitespace
            {"pattern": r"^\s+|\s+$", "replacement": "", "transformation_type": "deletion"},
            # Replace digits with words
            {
                "pattern": r"(\d+)",
                "replacement": lambda m: num2words(int(m.group(1)), lang="sv"),
                "transformation_type": "substitution",
            },
            # Tokenize the rest of the text into words
            {
                "pattern": r"\w+",
                "replacement": lambda m: m.group(),
                "transformation_type": "substitution",  # Not really a substitution, but we need to record the transformation
            },
        ]
    )

    # Include user-supplied patterns
    if user_patterns:
        patterns.extend(user_patterns)

    return patterns


def record_transformation(mapping, original_text, start, end, transformation_type, replacement):
    """
    Records a transformation in the mapping with additional context for debugging.

    Args:
        mapping (list of dicts): The list that stores transformation records.
        original_text (str): The original text being normalized.
        start (int): The start index of the original text span.
        end (int): The end index of the original text span.
        transformation_type (str): The type of transformation ('substitution', 'deletion', 'insertion').
        replacement (str): The replacement text (empty string for deletions).
    """
    original_span = original_text[start:end] if start is not None and end is not None else ""
    transformation_record = {
        "original_start": start,
        "original_end": end,
        "transformation_type": transformation_type,
        "replacement": replacement,
        "normalized_start": None,  # To be filled in during the apply_transformations step
        "normalized_end": None,  # To be filled in during the apply_transformations step
        "original_token": original_span,
        "normalized_token": (
            replacement.lower()
            if transformation_type == "substitution" and replacement != " "
            else None
        ),
        "start_time": None,
        "end_time": None,
        "index": None,
    }
    mapping.append(transformation_record)


def apply_transformations(text, mapping):
    """
    Applies recorded transformations to the text and updates the mapping with normalized positions.

    Args:
        text (str): The original text.
        mapping (list of dicts): The list of transformations.

    Returns:
        str: The transformed (normalized) text.
    """

    text_length = len(text)
    modified = np.zeros(text_length, dtype=bool)  # Track modified characters using a boolean mask

    offset = 0
    normalized_text = text

    # Sort transformations by their original start position to ensure correct application order
    mapping.sort(key=lambda x: x["original_start"])

    for i, transformation in enumerate(mapping):
        original_start = transformation["original_start"]
        original_end = transformation["original_end"]

        if modified[original_start:original_end].any():
            # Skip this transformation if it overlaps with a previous transformation
            continue
        else:
            # Mark the characters as modified
            modified[original_start:original_end] = True

        replacement = transformation["replacement"]

        # Calculate the adjusted start and end positions based on the current offset
        adjusted_start = original_start + offset
        adjusted_end = original_end + offset

        # Apply the transformation
        normalized_text = (
            normalized_text[:adjusted_start] + replacement + normalized_text[adjusted_end:]
        )

        # Update the normalized spans in the transformation record
        transformation["normalized_start"] = adjusted_start
        transformation["normalized_end"] = adjusted_start + len(replacement)

        # Update the offset for the next transformation
        offset += len(replacement) - (original_end - original_start)

        transformation["index"] = i

    return normalized_text


def normalize_text_with_mapping(text, user_patterns=None, combine_regexes=False):
    """
    Normalize speech text transcript while keeping track of transformations.

    Args:
        text (str): The original text to normalize.
        user_patterns (list of dicts, optional): User-supplied regex patterns, replacements, and type of transformation.

    Returns:
        tuple: Normalized text and list of mappings from original to new text positions.
    """
    mapping = []

    # Correct some OCR-errors before normalization
    for key, value in ocr_corrections.items():
        text = re.sub(key, value, text)

    # Collect all regex patterns for substitutions and deletions
    transformations = collect_regex_patterns(user_patterns)

    # Track already matched character spans using a boolean mask
    modified_chars = np.zeros(len(text), dtype=bool)

    # Record transformations for each pattern match
    for pattern_dict in transformations:
        pattern = pattern_dict["pattern"]
        transformation_type = pattern_dict["transformation_type"]

        for match in re.finditer(pattern, text.lower()):
            start, end = match.span()
            if modified_chars[start:end].any():
                # Skip this match if it overlaps with a previous match
                continue
            else:
                # Mark the characters as "to be modified"
                modified_chars[start:end] = True

            # If pattern_dict["replacement"] is a lambda function, call it to get the replacement string
            # Otherwise, use the replacement string
            if callable(pattern_dict["replacement"]):
                replacement = pattern_dict["replacement"](match)
            else:
                replacement = pattern_dict["replacement"]

            record_transformation(mapping, text, start, end, transformation_type, replacement)

    text = unicodedata.normalize("NFKC", text)

    # Apply the recorded transformations to the text
    normalized_text = apply_transformations(text, mapping)

    return normalized_text, mapping, text


def get_normalized_tokens(mapping, casing="lower"):
    """
    We previously created a mapping from the original text to the normalized text.
    Since the normalized text may contain multi-word tokens, we need to further tokenize the
    normalized text and maintain a mapping from the normalized tokens to the original text.

    E.g. expanded abbreviations such as "bl.a." will be normalized to the token "bland annat".
    This function will split the normalized token into two tokens, "bland" and "annat", while
    maintaining the original token's start and end times and its index in the original mapping.
    """

    normalized_mapping = OrderedDict()
    normalized_tokens = []

    offset = 0  # Multi-word offset, used to adjust the index in the normalized_mapping
    for i, record in enumerate(mapping):
        if record["transformation_type"] == "substitution" and record["replacement"] != " ":
            normalized_token = (
                record["normalized_token"]
                if casing == "lower"
                else record["normalized_token"].upper()  # Swedish wav2vec2 has uppercase tokens
            )

            if " " in normalized_token:
                # Multi-word token, split it into individual words
                multi_tokens = normalized_token.split(" ")
                for multi_word_index, token in enumerate(multi_tokens):
                    normalized_tokens.append(token)
                    offset += 1 if multi_word_index >= 1 else 0
                    normalized_mapping[i + offset] = {
                        "token": token,
                        "start_time": record["start_time"],  # Empty for now
                        "end_time": record["end_time"],  # Empty for now
                        "normalized_word_index": i,
                        "is_multi_word": True,
                        "is_first_word": multi_word_index == 0,
                        "is_last_word": multi_word_index == len(multi_tokens) - 1,
                    }
            else:
                normalized_mapping[i + offset] = {
                    "token": normalized_token,
                    "start_time": record["start_time"],  # Empty for now
                    "end_time": record["end_time"],  # Empty for now
                    "normalized_word_index": i,
                    "is_multi_word": False,
                }
                normalized_tokens.append(normalized_token)

    return normalized_mapping, normalized_tokens
