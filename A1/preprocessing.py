"""Text preprocessing: tokenization, case folding, stemming, stopwords."""

import re

import Stemmer

# Split on anything that is NOT alphanumeric or accented letter.
# Covers ASCII, Latin Extended (Czech diacritics), Cyrillic.
_TOKEN_RE = re.compile(r"[a-zA-Z0-9\u00C0-\u024F\u0100-\u017F\u0400-\u04FF]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text by splitting on whitespace/punctuation sequences.

    Works for both baseline (run-0) and advanced runs.
    Extracts sequences of alphanumeric and accented characters.
    """
    return _TOKEN_RE.findall(text)


def case_fold(tokens: list[str]) -> list[str]:
    return [t.lower() for t in tokens]


_stemmers: dict[str, Stemmer.Stemmer] = {}


def stem(tokens: list[str], lang: str) -> list[str]:
    """Apply Snowball stemming. lang should be 'english' or 'czech'."""
    if lang not in _stemmers:
        # Map short codes to Snowball language names
        lang_map = {"en": "english", "cs": "czech", "english": "english", "czech": "czech"}
        _stemmers[lang] = Stemmer.Stemmer(lang_map.get(lang, lang))
    return _stemmers[lang].stemWords(tokens)


def build_stopset(
    df: dict[str, int], num_docs: int, threshold: float
) -> frozenset[str]:
    """Build stopword set from document frequency statistics.

    Removes terms appearing in more than threshold fraction of documents.
    """
    cutoff = threshold * num_docs
    return frozenset(term for term, freq in df.items() if freq > cutoff)
