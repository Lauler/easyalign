import nltk
from nltk.tokenize.punkt import PunktTokenizer


def load_tokenizer(language: str = "swedish") -> PunktTokenizer:
    """
    Loads a PunktTokenizer for the specified language that can be used to sentence tokenize text.

    Parameters
    ----------
    language : str, default "swedish"
        Language to use for the tokenizer, e.g. "swedish", "english".

    Returns
    -------
    PunktTokenizer
        Loaded tokenizer.
    """
    try:
        tokenizer = PunktTokenizer(lang=language)
    except LookupError:
        nltk.download("punkt_tab")
        tokenizer = PunktTokenizer(lang=language)

    return tokenizer
