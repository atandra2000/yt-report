"""Language resources: a maintainable stopword source, and enough detection to use it.

The failure this module exists to prevent is a report whose "Products, Models &
Tools" list reads `Aber`, `Sie`, `Dinge`, `Universums`. Those are ordinary
German words that the capitalisation test cannot reject, because German
capitalises every noun as a spelling rule rather than as a signal of a proper
name — so `Anthropic` and `Aber` are indistinguishable to a regex.

A blacklist of the words that happened to appear in one run does not fix it: the
next German video has different nouns, and the filter that was supposed to hold
the line has been taught one example. So the words live in
`resources/stopwords/<lang>.txt` beside the code, and the list is deliberately
short — this is **one signal**, not the classifier. Corroboration (an entity is
seen more than once, or in speech and on screen) and category vocabulary do the
rest. A missing noun costs one false candidate; a missing file costs a language.

Detection is split in two, because the two cases are not the same problem:

* **Scripts that mark their own language.** Devanagari, CJK, Cyrillic, Arabic and
  Hangul ranges are unambiguous — no Latin-script language uses them, and no other
  script-using language shares them for our purposes. One codepoint range is a
  complete answer.
* **Latin scripts, where the capital is not a signal at all.** Spanish, French,
  Portuguese, Italian, German and English must be told apart by vocabulary, so
  they are scored by how many of their own function words appear. It is a
  heuristic and is recorded as one: the caller gets the languages it consulted and
  the score behind each, and a low-confidence guess widens the filter to every
  loaded Latin language rather than guessing narrowly and missing.

The widened fallback is the important part. A filter that consults only the
detected language is one misdetection away from emitting `Aber` again; a filter
that consults every Latin language when the detection is weak is merely less
precise, and precision is what the corroboration rule downstream already
re-supplies.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

#: `resources/stopwords/`, resolved from this file rather than from the working
#: directory — the scripts are run from anywhere, and a resource path that
#: depended on the caller's cwd would work in the self-test and fail in the CLI.
RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources" / "stopwords"

#: Every language a stopword list ships for. Also the answer to "which
#: languages are supported", so adding a file is the whole installation step.
SUPPORTED_LANGUAGES = ("en", "de", "es", "fr", "pt", "it", "hi")

#: Languages written in the Latin script, which are the ones detection has to
#: guess between. Everything else is settled by a codepoint range.
_LATIN_LANGUAGES = ("en", "de", "es", "fr", "pt", "it")

#: Non-Latin script → language. Ranges rather than characters, because Devanagari
#: and CJK are the same "does this look foreign" test at very different scales.
_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("hi", 0x0900, 0x097F),   # Devanagari
    ("ja", 0x3040, 0x30FF),   # kana
    ("zh", 0x4E00, 0x9FFF),   # CJK unified ideographs
    ("ko", 0xAC00, 0xD7AF),   # Hangul syllables
    ("ru", 0x0400, 0x04FF),   # Cyrillic
    ("ar", 0x0600, 0x06FF),   # Arabic
)

#: Below this many function-word hits the Latin-script detection is not worth
#: acting on, and every loaded Latin language is consulted instead. Measured
#: against the shipped lists: a genuine 40-word English passage scores well over
#: ten, and a two-word code sample scores one or two.
_MIN_LATIN_HITS = 3

#: Words are split on anything that is not a letter or a digit in any script.
#: `\w` under `re.UNICODE` already does this, and folding the case of a
#: Devanagari token is a no-op rather than an error.
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)

#: German inflection, longest first, stripped before the lookup. The resource
#: file holds lemmas; the transcript holds inflected forms. `Universums` is not
#: `universum`, and a filter that compares them literally misses the exact word
#: the failure is named after — so the token is reduced to its lemma by peeling
#: at most two suffixes.
#:
#: Bounded to two peels on purpose. A German noun takes a great many endings
#: (`Arbeitswelten`, `Universitäten`), and a general stemmer would either be
#: wrong or be a parser. Two peels covers the case and case-plural forms that
#: make up ordinary spoken German; a longer word that slips through costs one
#: false candidate, and the corroboration rule downstream re-supplies the
#: precision this gives up.
GERMAN_SUFFIXES = ("ungen", "heiten", "schaften", "ern", "ung", "keit", "lich",
                   "isch", "en", "er", "es", "em", "ern", "e", "n", "s", "t")


def _german_lemma(word: str) -> set[str]:
    """The word itself plus up to two suffix-stripped forms."""
    out = {word}
    current = word
    for _ in range(2):
        for suf in GERMAN_SUFFIXES:
            if len(current) - len(suf) >= 3 and current.endswith(suf):
                current = current[: -len(suf)]
                out.add(current)
                break
        else:
            break
    return out


@lru_cache(maxsize=None)
def stopwords(language: str) -> frozenset[str]:
    """The stopword set for one language, or empty if it ships none.

    Comments and blank lines are dropped, so a resource file can explain itself
    — the reason these files carry a header at all is that the next person to add
    a language should not have to reverse-engineer why a word is in the list.
    """
    path = RESOURCE_DIR / f"{language}.txt"
    if not path.is_file():
        return frozenset()
    out: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        word = raw.split("#", 1)[0].strip().lower()
        if word:
            out.add(word)
    return frozenset(out)


def tokens(text: str) -> list[str]:
    """Lower-cased word tokens, in order. Empty for empty input, never raises."""
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN.finditer(unicodedata.normalize("NFC", text))]


def script_languages(text: str) -> list[str]:
    """Every non-Latin language whose script appears, in resource order.

    More than one is possible and correct: a video with Hindi captions and an
    English title has both, and a filter that took only the first would miss the
    other half of the document.
    """
    if not text:
        return []
    found: list[str] = []
    for name, lo, hi in _SCRIPT_RANGES:
        if any(lo <= ord(ch) <= hi for ch in text):
            found.append(name)
    return found


def latin_scores(text: str) -> dict[str, int]:
    """Function-word hits per Latin-script language. A ranking, not a verdict."""
    toks = set(tokens(text))
    if not toks:
        return {}
    return {lang: len(toks & stopwords(lang)) for lang in _LATIN_LANGUAGES}


def detect_languages(text: str) -> tuple[list[str], dict[str, object]]:
    """Which languages to filter with, and the score behind each.

    Returns `(languages, detail)` where `detail` records what the decision was
    made from, so a caller writing an artifact can say *why* a word was dropped
    rather than only that it was. `["de", "en", "es", ...]` with
    `{"basis": "weak-latin", "scores": {...}}` means the detection was too thin
    to choose and every Latin language was consulted — the honest outcome, and the
    one that cannot reintroduce `Aber` on a video it failed to identify.

    `detail["morphology"]` is true only when German was identified
    *confidently*. Stripping a suffix to find a word's lemma is a second guess,
    and two guesses compound: under the wide fallback, peeling `Sonnet` down to
    `sonne` ("sun") deletes a model name from an English sentence because of a
    German word. So the peel runs only where the language is actually known, and
    the fallback compares forms literally.
    """
    scripts = script_languages(text)
    scores = latin_scores(text)
    detail: dict[str, object] = {"script_languages": scripts, "scores": scores,
                                 "morphology": False}
    if scripts:
        detail["basis"] = "script"
        return scripts, detail
    best = max(scores, key=lambda k: scores[k]) if scores else None
    if best and scores[best] >= _MIN_LATIN_HITS:
        detail["basis"] = "function-words"
        detail["chosen"] = best
        detail["morphology"] = best == "de"
        return [best], detail
    detail["basis"] = "weak-latin" if scores else "empty"
    return list(_LATIN_LANGUAGES), detail


def is_ordinary(word: str, languages: list[str] | tuple[str, ...],
                morphology: bool = True) -> bool:
    """Is this word ordinary vocabulary in *any* of the given languages?

    Any, not all. A name that is an ordinary word in one language the transcript
    touches is still a name in the other — "Marc" is a stopword nowhere, but
    "Anita" is ordinary English and a real person's name everywhere else — and the
    question this is asked is "is there any language here in which this is just a
    word?", not "is this a word in every language here".

    German is compared by lemma as well as by form when `morphology` is set;
    see `_german_lemma` and `detect_languages` for when that is on.
    """
    if not word:
        return True
    w = word.strip().lower()
    if not w:
        return True
    for lang in languages:
        words = stopwords(lang)
        if not words:
            continue
        if w in words:
            return True
        if morphology and lang == "de" and (_german_lemma(w) & words):
            return True
    return False


def stopword_hits(word: str, languages: list[str] | tuple[str, ...]) -> list[str]:
    """Which of the consulted languages list this word. For a recorded reason."""
    w = word.strip().lower()
    if not w:
        return []
    out = []
    for lang in languages:
        words = stopwords(lang)
        if not words:
            continue
        if w in words or (lang == "de" and (_german_lemma(w) & words)):
            out.append(lang)
    return out
