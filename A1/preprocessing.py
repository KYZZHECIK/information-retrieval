"""Text preprocessing: tokenization, case folding, stemming, stopwords."""

import re
import sys
from pathlib import Path

import Stemmer as PyStemmer

# Split on anything that is NOT alphanumeric or accented letter.
# Covers ASCII, Latin Extended-A/B (Czech diacritics), Cyrillic.
_TOKEN_RE = re.compile(r"[a-zA-Z0-9\u00C0-\u024F\u0100-\u017F\u0400-\u04FF]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text by splitting on whitespace/punctuation sequences.

    Extracts sequences of alphanumeric and accented characters.
    """
    return _TOKEN_RE.findall(text)


def case_fold(tokens: list[str]) -> list[str]:
    return [t.lower() for t in tokens]


# ── English stemmer (Snowball via PyStemmer) ───────────────────────────

_en_stemmer: PyStemmer.Stemmer | None = None


def _get_en_stemmer() -> PyStemmer.Stemmer:
    global _en_stemmer
    if _en_stemmer is None:
        _en_stemmer = PyStemmer.Stemmer("english")
    return _en_stemmer


# ── Czech lemmatizer (MorphoDiTa) ─────────────────────────────────────

_cs_morpho = None
_cs_morpho_loaded = False

MODELS_DIR = Path(__file__).resolve().parent / "models"
CS_MODEL_DICT = "czech-morfflex2.0-pdtc1.0-220710.dict"
CS_MODEL_TAGGER = "czech-morfflex2.0-pdtc1.0-220710.tagger"

# Which Czech lemmatization mode to use: "dict" (default) or "tagger" (contextual)
_cs_mode: str = "dict"


def set_czech_mode(mode: str):
    """Set Czech lemmatization mode: 'dict', 'tagger', or 'fallback'."""
    global _cs_mode
    _cs_mode = mode


# ── Dictionary-based lemmatization (default) ──────────────────────────

_cs_morpho = None
_cs_morpho_loaded = False


def _get_cs_morpho():
    """Load Czech MorphoDiTa morphological dictionary (lazy, once)."""
    global _cs_morpho, _cs_morpho_loaded
    if _cs_morpho_loaded:
        return _cs_morpho

    _cs_morpho_loaded = True
    dict_path = MODELS_DIR / CS_MODEL_DICT
    if not dict_path.exists():
        print(
            f"WARNING: Czech MorphoDiTa dict not found at {dict_path}. "
            f"Run 'uv run python -m A1.download_models' to download it. "
            f"Falling back to light Czech stemmer.",
            file=sys.stderr,
        )
        return None

    from ufal.morphodita import Morpho

    _cs_morpho = Morpho.load(str(dict_path))
    if _cs_morpho is None:
        print("WARNING: Failed to load MorphoDiTa dict.", file=sys.stderr)
    else:
        print(f"Loaded Czech MorphoDiTa dict from {dict_path}", file=sys.stderr)
    return _cs_morpho


# ── Tagger-based lemmatization (contextual) ───────────────────────────

_cs_tagger = None
_cs_tagger_loaded = False


def _get_cs_tagger():
    """Load Czech MorphoDiTa tagger model (lazy, once)."""
    global _cs_tagger, _cs_tagger_loaded
    if _cs_tagger_loaded:
        return _cs_tagger

    _cs_tagger_loaded = True
    tagger_path = MODELS_DIR / CS_MODEL_TAGGER
    if not tagger_path.exists():
        print(
            f"WARNING: Czech MorphoDiTa tagger not found at {tagger_path}. "
            f"Run 'uv run python -m A1.download_models --tagger' to download it. "
            f"Falling back to dict-based lemmatization.",
            file=sys.stderr,
        )
        return None

    from ufal.morphodita import Tagger

    _cs_tagger = Tagger.load(str(tagger_path))
    if _cs_tagger is None:
        print("WARNING: Failed to load MorphoDiTa tagger.", file=sys.stderr)
    else:
        print(f"Loaded Czech MorphoDiTa tagger from {tagger_path}", file=sys.stderr)
    return _cs_tagger


def _clean_lemma(raw_lemma: str) -> str:
    """Strip MorphoDiTa technical suffixes from lemma."""
    return raw_lemma.split("_")[0].split("`")[0]


# ── Lemmatization caches ──────────────────────────────────────────────

_cs_dict_cache: dict[str, str] = {}
_cs_tagger_cache: dict[tuple, tuple] = {}


def _lemmatize_czech_dict(tokens: list[str]) -> list[str]:
    """Lemmatize Czech tokens using MorphoDiTa dictionary (no context)."""
    from ufal.morphodita import TaggedLemmas

    morpho = _get_cs_morpho()
    if morpho is None:
        return _stem_czech_fallback(tokens)

    lemmas_out = TaggedLemmas()
    result = []
    for token in tokens:
        if token in _cs_dict_cache:
            result.append(_cs_dict_cache[token])
            continue

        rc = morpho.analyze(token, morpho.GUESSER, lemmas_out)
        if rc >= 0 and len(lemmas_out) > 0:
            lemma = _clean_lemma(lemmas_out[0].lemma) or token
        else:
            lemma = token

        _cs_dict_cache[token] = lemma
        result.append(lemma)
    return result


def _lemmatize_czech_tagger(tokens: list[str]) -> list[str]:
    """Lemmatize Czech tokens using MorphoDiTa tagger (context-aware).

    The tagger processes a sentence (list of tokens) and uses POS context
    to disambiguate and select the correct lemma.
    """
    from ufal.morphodita import Forms, TaggedLemmas, TokenRanges

    tagger = _get_cs_tagger()
    if tagger is None:
        return _lemmatize_czech_dict(tokens)

    # Check cache (full token sequence as key)
    key = tuple(tokens)
    if key in _cs_tagger_cache:
        return list(_cs_tagger_cache[key])

    # Feed tokens to the tagger as a pre-tokenized "sentence"
    forms = Forms()
    for t in tokens:
        forms.push_back(t)

    lemmas_out = TaggedLemmas()
    tagger.tag(forms, lemmas_out)

    result = []
    for i, tl in enumerate(lemmas_out):
        lemma = _clean_lemma(tl.lemma)
        result.append(lemma if lemma else tokens[i] if i < len(tokens) else "")

    # Cache result
    if len(tokens) <= 20:  # only cache short sequences (query-length)
        _cs_tagger_cache[key] = tuple(result)

    return result


def _lemmatize_czech(tokens: list[str]) -> list[str]:
    """Dispatch to the configured Czech lemmatization method."""
    if _cs_mode == "tagger":
        return _lemmatize_czech_tagger(tokens)
    elif _cs_mode == "dict":
        return _lemmatize_czech_dict(tokens)
    else:
        return _stem_czech_fallback(tokens)


# ── Fallback light Czech stemmer ───────────────────────────────────────

_CS_SUFFIXES = [
    # Longest suffixes first for greedy matching
    "ového", "ovému", "ových", "ovými",
    "ského", "skému", "ských", "skými",
    "ování", "ovány", "ováni", "ována",
    "ností", "nosti",
    "ičkách", "ičkami", "ičkám",
    "ečkám", "ečkách", "ečkami",
    "áčkům", "áčkách", "áčkami",
    "átkem", "átkům", "átků",
    "ěním", "ením", "ateli",
    "ního", "ními", "ních", "nímu",
    "ovou", "ový", "ová", "ové",
    "skou", "ský", "ská", "ské",
    "ách", "ích", "ého", "ému", "ými",
    "ové", "ovi", "ami", "ání", "ění", "ení", "ost",
    "ním", "ou", "em", "ům", "mi", "ám",
    "ní", "ný", "ná", "né",
    "ěl", "al", "il",
    "ě", "ů", "í", "é", "á", "ý",
    "e", "i", "y", "a", "o", "u",
]


def _stem_czech_fallback(tokens: list[str]) -> list[str]:
    """Light Czech stemmer as fallback when MorphoDiTa is unavailable."""
    result = []
    for word in tokens:
        if len(word) <= 3:
            result.append(word)
            continue
        stemmed = word
        for suffix in _CS_SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                stemmed = word[: -len(suffix)]
                break
        result.append(stemmed)
    return result


# ── Unified stem interface ─────────────────────────────────────────────


def stem(tokens: list[str], lang: str) -> list[str]:
    """Apply stemming/lemmatization.

    English: Snowball stemmer (PyStemmer).
    Czech: MorphoDiTa lemmatizer (with light stemmer fallback).
    """
    lang_key = {"en": "en", "cs": "cs", "english": "en", "czech": "cs"}.get(lang, lang)

    if lang_key == "en":
        return _get_en_stemmer().stemWords(tokens)
    elif lang_key == "cs":
        return _lemmatize_czech(tokens)
    else:
        return tokens


def build_stopset(
    df: dict[str, int], num_docs: int, threshold: float
) -> frozenset[str]:
    """Build stopword set from document frequency statistics.

    Removes terms appearing in more than threshold fraction of documents.
    """
    cutoff = threshold * num_docs
    return frozenset(term for term, freq in df.items() if freq > cutoff)
