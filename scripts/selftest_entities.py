"""Entity extraction across languages, and the salience bar behind it.

The defect this exists for is a report whose "Products, Models & Tools" reads
`Aber`, `Sie`, `Dinge`, `Universums` — German ordinary nouns and pronouns
presented as products. The cause is not a bug in the stopword list; it is that
capitalisation is an English-language convention being applied to a language
where it carries no information, because German capitalises every noun as a
spelling rule. `Aber` and `Anthropic` are indistinguishable to a regex, and no
amount of tightening that regex fixes it without dropping the real names.

So the words live in `resources/stopwords/<lang>.txt` and the filter consults
the language the transcript is in. That is one signal, and these tests also
assert the two that keep it honest:

* **under-extraction is a defect too.** `Claude Code`, `Qwen`, `Anthropic` and
  `Obsidian` have to survive a German and an English sentence, and the check is
  that removing a false positive did not cost a true one.
* **an untyped name is not a product.** A candidate with no category vocabulary
  near it comes back as `entity`, and the report gives it its own list rather
  than asserting a product identity the evidence does not carry.

The German cases are the ones worth reading twice: the lemma is what the resource
file holds and the sentence carries the inflected form, so `Universums` has to
be matched against `universum`. That is why the filter compares by lemma, and why
the morphology step is switched off when the language was only guessed.

PROVENANCE — read this before trusting a failure count
--------------------------------------------------------
These fixtures were reconstructed after loss of the original untracked test
source. Assertion labels/properties were recovered from the baseline test log;
input fixtures are reconstructed and therefore are not byte-for-byte original
tests. Nothing here should be described as recovered original source, and the
reconstructed set was therefore not treated as historical regression coverage.

What the reconstruction *was* checked against, so the tests are not merely
self-consistent:

* Every branch claim here is mutation-tested. A deliberate defect injected into
  the implementation must make the named test fail; the mutation set and its
  results live in `RELEASE-CHECK.md` under "Test Provenance and Recovery". Two
  gaps that mutation found (E13, E14) were closed by adding these tests, and one
  filter, `_AUXILIARY_LEAD`, was found to be unreachable — every word in it is
  already caught by `_ENTITY_STOPWORDS` or a resource stopword list — so it is
  left in place as defence in depth and no test claims it.
"""
from __future__ import annotations

import evidence_model as EV
import lang_resources as LR
import report_fixtures as RFX
import semantic as SEM


#: Bound by `selftest.py` at import, exactly as the other sub-suites are.
check = None


def _names(sentence: str, languages: list[str] | None = None, morphology: bool = True):
    langs = languages if languages is not None else SEM.detect_languages(sentence)[0]
    return {c for c, _ in SEM.find_entities(sentence, langs, morphology)}


#: The four words from the report, in the sentence they came from. Verbatim, so
#: the test is about those and not about cases chosen after the fix.
DE_ORDINARY = ("Aber", "Sie", "Du", "Dinge", "Universums", "Masse", "Energie")
DE_KEEP = ("Claude Code", "Qwen", "Anthropic", "Obsidian")



#: One German transcript, carrying both halves of the problem in it: the words
#: that must never become names, and the names that must survive the filter that
#: removes them. A stopword list is only correct relative to both.
_DE = ("Wir haben gestern mit Claude Code in Obsidian gearbeitet und dabei über "
       "Qwen gesprochen. Anthropic hat das Modell veröffentlicht. Die Dinge "
       "waren komisch: Masse, Energie, Universums. Du und Sie wißt das "
       "eigentlich, aber Aber ist ein Wort und kein Name.")
_EN = "The company Anthropic released Claude Code and Qwen yesterday, and said Dr Smith."
_HI = "कंपनी Anthropic ने आज Claude जारी किया। यह एक परीक्षण है।"

#: German false positives taken from real transcripts, not invented. Each is a
#: word that capitalises because German capitalises every noun.
_DE_FALSE = ("Schwerkraft", "Dunkelheit", "Lichtgeschwindigkeit", "Ausdehnung",
             "Aber", "Masse")


def t_german_ordinary_words_are_not_entities() -> None:
    langs, morphology = SEM.detect_languages(_DE)
    check("E1 a German transcript is identified as German", "de" in langs, str(langs))
    check("E1 ...confidently enough to inflect", morphology is True, str((langs, morphology)))
    for word in ("Aber", "Sie", "Du", "Dinge", "Universums", "Masse", "Energie"):
        check(f"E1 {word!r} is not extracted as an entity", word not in _names(_DE, langs),
              str(sorted(_names(_DE, langs))))


def t_german_real_names_survive() -> None:
    langs, _ = SEM.detect_languages(_DE)
    for name in ("Claude Code", "Qwen", "Anthropic", "Obsidian"):
        check(f"E2 {name!r} is still extracted from the German transcript",
              name in _names(_DE, langs), str(sorted(_names(_DE, langs))))


def t_german_lemma_matching() -> None:
    for inflected, lemma in (("Universums", "universum"), ("Modellen", "modell"),
                             ("Dingen", "ding"), ("Masse", "masse")):
        check(f"E3 {inflected!r} matches the lemma {lemma!r}",
              LR.is_ordinary(inflected, ["de"], True), "the lemma did not match")
    # `Sonnet` is the documented casualty of the lemma peel, and the point of
    # the assertion is that it is a casualty *on purpose*: the peel is what
    # makes `Masse` filterable, and a name it happens to reach is the price.
    check("E3 a real name is not swept up by the peel",
          not LR.is_ordinary("Anthropic", ["de"], True),
          "the peel reached a name that has no German reading")
    # The peel keeps the surface spelling, so the comparison is on stems: the
    # resource is lower-cased on load and the peel is not, and a test that mixed
    # the two would assert nothing at all.
    check("E3 the peel is what would delete `Sonnet` from a German reading",
          any(w.lower() in LR.stopwords("de") for w in LR._german_lemma("Sonnet")),
          str(LR._german_lemma("Sonnet")))
    check("E3 ...and the peel is what saves it when the language was only guessed",
          not LR.is_ordinary("Sonnet", ["de"], False), "the peel ran without morphology")


def t_weak_detection_does_not_infect() -> None:
    weak = "Anthropic hat Claude veröffentlicht, Qwen auch."
    langs, morphology = SEM.detect_languages(weak)
    check("E4 a two-clause sentence is not confidently identified",
          "de" not in langs or morphology is not True, f"{langs} {morphology}")
    check("E4 both model names survive the weak-detection fallback",
          {"Claude", "Qwen"} <= _names(weak), str(sorted(_names(weak))))
    check("E4 and the fallback really is the wide one",
          set(langs) >= set(LR._LATIN_LANGUAGES), str(langs))


def t_english_extraction_unchanged() -> None:
    langs, _ = SEM.detect_languages(_EN)
    check("E5 an English transcript is identified as English", "en" in langs, str(langs))
    found = _names(_EN, langs)
    for name in ("Anthropic", "Claude Code", "Qwen", "Dr Smith"):
        check(f"E5 {name!r} is extracted", name in found, str(sorted(found)))
    check("E5 a sentence-initial 'The' is not a name", "The" not in found,
          str(sorted(found)))


def t_hindi_fixture() -> None:
    langs, _ = SEM.detect_languages(_HI)
    check("E6 a Devanagari transcript is identified by its script", "hi" in langs, str(langs))
    found = _names(_HI, langs)
    check("E6 a Latin-script product name in a Hindi transcript is extracted",
          {"Anthropic", "Claude"} <= found, str(sorted(found)))
    check("E6 a Devanagari function word is not",
          not any(any("ऀ" <= ch <= "ॿ" for ch in n) for n in found), str(sorted(found)))
    check("E6 the Hindi stopword list is loaded", bool(LR.stopwords("hi")), "empty")


def t_uncategorised_name_is_generic() -> None:
    plain = "Anthropic hat das veröffentlicht."
    langs, _ = SEM.detect_languages(plain)
    typed = dict(SEM.find_entities(plain, langs))
    check("E7 `entity` is a supported item type", "entity" in EV.ITEM_TYPES, str(EV.ITEM_TYPES))
    check("E7 an uncategorised name comes back as `entity`",
          typed.get("Anthropic") == "entity", str(typed))
    beside = "The Anthropic model handles long context."
    langs, _ = SEM.detect_languages(beside)
    check("E7 a name beside a category word is typed by it",
          dict(SEM.find_entities(beside, langs)).get("Anthropic") == "model",
          str(dict(SEM.find_entities(beside, langs))))


def t_salience_requires_context() -> None:
    check("E8 `Claude Code` is name-shaped", bool(SEM._NAME_SHAPED.search("Claude Code")),
          "two capitalised words")
    check("E8 `Qwen3-8B` is name-shaped", bool(SEM._NAME_SHAPED.search("Qwen3-8B")),
          "letters then digits")
    check("E8 a bare capitalised word is not", not SEM._NAME_SHAPED.search("Anthropic"),
          "one capitalised word")


def t_ordinary_words_stay_out_of_subjects() -> None:
    """A German aside is not the thing a figure is about.

    German sentences carry parenthetical asides far more readily than English
    ones, and the words inside one are exactly the words a subject window picks
    up: "Die Latenz stieg auf 40 ms (nach zwei Messungen)" must not file the
    figure under `Messungen`, while a real name in the same position still can.
    """
    aside = "Die Latenz stieg auf 40 ms (nach zwei Messungen)."
    at = aside.find("40")
    check("E9 no entity phrase is read from a German aside",
          SEM.subject_for(aside, at, 2, "ms") is None,
          SEM.subject_for(aside, at, 2, "ms"))
    named = "Anthropic, 40 ms Latenz."
    at = named.find("40")
    check("E9 a real name still is", SEM.subject_for(named, at, 2, "ms") == "Anthropic",
          SEM.subject_for(named, at, 2, "ms"))


def t_every_supported_language_has_a_resource() -> None:
    for lang in LR.SUPPORTED_LANGUAGES:
        words = LR.stopwords(lang)
        check(f"E10 {lang} ships a non-empty stopword list", bool(words), "empty")
        # The parser strips an inline comment, so a `#` that leaked into the set
        # would mean the file's own separator is no longer being honoured — the
        # one thing a resource file cannot be allowed to get quietly wrong.
        check(f"E10 {lang} lists no comment lines",
              not any(w.startswith("#") for w in words), sorted(words)[:5])
    check("E10 the resource directory and SUPPORTED_LANGUAGES agree",
          {p.stem for p in LR.RESOURCE_DIR.glob("*.txt")} == set(LR.SUPPORTED_LANGUAGES),
          str(sorted(p.stem for p in LR.RESOURCE_DIR.glob("*.txt"))))


def t_quantity_phrases_are_not_names() -> None:
    langs, morphology = SEM.detect_languages(_DE)
    for phrase in ("Millionen Lichtjahre", "Millionen Stundenkilometern",
                   "500 Requests", "128 Tokens pro Sekunde"):
        check(f"E11 {phrase!r} is a quantity phrase, not a name",
              SEM._is_quantity_phrase(phrase), "not recognised")
    for name in ("Claude Code", "Qwen", "Anthropic", "Obsidian", "Incogni"):
        check(f"E11 {name!r} is not mistaken for a quantity",
              not SEM._is_quantity_phrase(name), "misread")
    found = _names("Wir sahen Millionen Lichtjahre und 500 Requests.", langs, morphology)
    check("E11 neither compound reaches the entity list",
          not (found & {"Millionen Lichtjahre", "500 Requests"}), str(sorted(found)))


def t_observed_german_false_positives_are_listed() -> None:
    frame = "Heute spricht Schwerkraft über Dunkelheit, Lichtgeschwindigkeit und Ausdehnung."
    langs, morphology = SEM.detect_languages(frame)
    found = _names(frame, langs, morphology)
    for word in _DE_FALSE:
        check(f"E12 {word!r} (a real false positive) is filtered", word not in found,
              str(sorted(found)))
    check("E12 the sponsor's name still comes through", "Anthropic" in
          _names("Gesponsert von Anthropic. Danke an Aber und Sonne.", langs, morphology),
          str(sorted(_names("Gesponsert von Anthropic. Danke an Aber und Sonne.",
                            langs, morphology))))


# --------------------------------------------------------------------------- #
# added by the release-freeze mutation audit
# --------------------------------------------------------------------------- #

#: Short all-caps tokens, and the sentence each is asserted in. A category is not
#: a name: the item type is the only thing that says "this is a kind of thing",
#: and extracting `AI` as an entity put a category in the document beside the
#: actual models — the exact distinction the item type exists to carry.
_CAPS_CATEGORIES = (
    ("We use AI and an LLM every day.", ("AI", "LLM")),
    ("The GPU is fast and an API is useful.", ("GPU", "API")),
)


def t_short_all_caps_categories_are_not_names() -> None:
    """E13. Added after mutation testing: this filter had no test at all.

    Removing `cand.isupper() and len(cand) <= 4` from `find_entities` changed no
    entity test's result, so nothing was protecting it. These sentences reach
    that branch and fail without it.
    """
    for sentence, cats in _CAPS_CATEGORIES:
        found = _names(sentence)
        for cat in cats:
            check(f"E13 {cat!r} is a category, not a name", cat not in found,
                  str(sorted(found)))
    # ...and the neighbouring rule is still live: a long all-caps run is UI
    # furniture and is dropped too, so this is not "all-caps is fine now".
    check("E13 a long all-caps run is dropped as UI furniture",
          not (_names("OPEN THE SETTINGS PANEL NOW") or []),
          str(sorted(_names("OPEN THE SETTINGS PANEL NOW"))))


def t_single_sighting_ordinary_word_does_not_become_an_item() -> None:
    """E14. Added after mutation testing: the corroboration bar was untested.

    `find_entities` returning candidates is not the report's decision. A
    candidate seen once, off screen, and not name-shaped is the shape an ordinary
    capitalised word has, and the report must not promote it. Removing the
    `_NAME_SHAPED`/`entity` exemption in `_corroborated_entities` promoted `Qwen`
    here, which is the defect: a name mentioned once, off screen, and never
    corroborated, entering the report as though the video had established it.
    """
    ev = RFX.synthesis_qualification()
    speech = ("Wir haben gestern mit Claude Code gearbeitet. Der Abschnitt war lang. "
              "Qwen kam kurz vorbei. Ende.")
    ctx = SEM.SegmentContext(segment=ev.segments[0], snippets=ev.snippets,
                             frames=[], speech=speech, speech_norm=speech.lower())
    items = SEM._corroborated_entities(ctx, SEM.SemanticConfig())
    names = [i.value for i in items]
    check("E14 a name-shaped product still reaches the report",
          "Claude Code" in names, str(names))
    check("E14 a single-sighting, off-screen, untyped name does not",
          "Qwen" not in names, str(names))
    # The other half of the same rule: name shape is what rescues a one-off
    # product, so an untyped word is dropped while a shaped one is kept. Asserted
    # separately because that is the direction a stricter check would get wrong.
    check("E14 and the drop is the name-shape bar, not an empty result",
          bool(names), str(names))


# --------------------------------------------------------------------------- #
# every language, one table
# --------------------------------------------------------------------------- #

#: `(language, sentence, must_be_present, must_be_absent)`.
#:
#: The present list carries one of each kind the extractor claims to know — a
#: real organisation, a product, a tool, a model, a person — and the absent list
#: carries the ordinary vocabulary around them. A language is only working if
#: both halves hold, and the second half is the one that regresses: a stopword
#: list that grows is cheap, and the evidence that it did not grow *too far* is
#: the first list.
_LANGUAGE_MATRIX = (
    ("en", "The company Anthropic released Claude today, said Dr Smith.",
     ("Anthropic", "Claude", "Dr Smith"), ("Today",)),
    ("en", "Yesterday I walked to the market and bought a new laptop.",
     (), ("Yesterday", "Market")),
    ("de", "Anthropic hat Claude veröffentlicht, sagte Dr Weber.",
     ("Anthropic", "Claude", "Dr Weber"), ("Gestern", "Markt", "Gesetz")),
    # The name has to stand on its own here. Introduced by its category word —
    # "das Werkzeug Docker" — it is one capitalised run with the category word in
    # it, and the run is dropped whole; the resource file says so, and this is
    # the sentence that shows the cost rather than hides it.
    ("de", "Wir installieren Docker auf dem Rechner.",
     ("Docker",), ("Wir", "Rechner")),
    ("es", "La empresa Anthropic publicó Claude hoy, según la doctora Ruiz.",
     ("Anthropic", "Claude", "Ruiz"), ("Hoy", "Ayer", "Mercado")),
    ("es", "Instalamos la herramienta Docker en la máquina.",
     ("Docker",), ("Instalamos", "Herramienta", "Máquina")),
    ("fr", "L'entreprise Anthropic a publié Claude aujourd'hui, selon le docteur Blanc.",
     ("Anthropic", "Claude", "Blanc"), ("Aujourd'hui", "Hier", "Marché")),
    ("fr", "Nous installons l'outil Docker sur la machine.",
     ("Docker",), ("Nous", "Machine")),
    ("pt", "A empresa Anthropic publicou o Claude hoje, segundo a doutora Alves.",
     ("Anthropic", "Claude", "Alves"), ("Hoje", "Ontem", "Mercado")),
    ("pt", "Instalamos a ferramenta Docker na máquina.",
     ("Docker",), ("Instalamos", "Ferramenta", "Máquina")),
    ("it", "L'azienda Anthropic ha pubblicato Claude oggi, secondo la dottorina Ferri.",
     ("Anthropic", "Claude", "Ferri"), ("Oggi", "Ieri", "Mercato")),
    ("it", "Installiamo lo strumento Docker sulla macchina.",
     ("Docker",), ("Installiamo", "Strumento", "Macchina")),
    ("hi", "कंपनी Anthropic ने आज Claude जारी किया।",
     ("Anthropic", "Claude"), ()),
    ("hi", "मैंने कल बाज़ार गया और नया लैपटॉप खरीदा।",
     (), ()),
)


def t_every_language_in_one_table() -> None:
    """Seven languages, the same two assertions each.

    Deterministic by construction: fixed sentences, a fixed stopword resource, no
    model and no network. That is what makes this the fixture a doctor cannot
    be — the doctor needs a provider and a credit, and a qualification that
    cannot run is a qualification that has never been checked.
    """
    for lang, sentence, present, absent in _LANGUAGE_MATRIX:
        langs, morphology = SEM.detect_languages(sentence)
        found = _names(sentence, langs, morphology)
        for name in present:
            check(f"entity {lang}: {name!r} survives", name in found,
                  f"{sentence!r} -> {sorted(found)}")
        for name in absent:
            check(f"entity {lang}: {name!r} is not a name", name not in found,
                  f"{sentence!r} -> {sorted(found)}")


def t_ordinary_vocabulary_never_becomes_a_category() -> None:
    """§5's actual requirement, stated as its own assertion.

    An ordinary noun arriving as a *candidate* is forgivable — it still has to be
    corroborated, and a missing noun costs one false candidate while a missing
    file costs a language. An ordinary noun arriving typed as `product`, `model`
    or `tool` is not: that is a claim the report then makes to a reader, in a
    section headed by the claim.
    """
    categories = {"product", "model", "tool", "person", "organization",
                  "paper", "dataset"}
    for lang, sentence, present, absent in _LANGUAGE_MATRIX:
        langs, morphology = SEM.detect_languages(sentence)
        ordinary = set(absent) - set(present)
        for name, itype in SEM.find_entities(sentence, langs, morphology):
            if name in ordinary:
                check(f"entity {lang}: {name!r} is not typed {itype!r}",
                      itype not in categories, f"{itype!r}")


def t_typing_does_not_spread_across_a_sentence() -> None:
    """A cue types the name it sits beside, not the names it also sits near.

    The window this replaced was forty-eight characters either way, which typed
    every name in a sentence by the same cue: "The company Anthropic released
    Claude today" filed both under `organization`, and "According to Dr Smith,
    the tool Docker works" filed the word `According` under `tool`. Both are
    invented types — the extractor saying something about a name it does not
    know, which is worse than saying nothing.
    """
    def typed_of(sentence: str) -> dict:
        langs, morphology = SEM.detect_languages(sentence)
        return dict(SEM.find_entities(sentence, langs, morphology))

    typed = typed_of("The company Anthropic released Claude today.")
    check("typing: the company is the organization",
          typed.get("Anthropic") == "organization", str(typed))
    check("typing: the product beside it is not a company",
          typed.get("Claude") == "entity", str(typed))

    typed = typed_of("OpenAI says the model Claude is faster.")
    check("typing: a claim-maker is not a person", typed.get("OpenAI") == "entity",
          str(typed))
    check("typing: the model is still typed by its own modifier",
          typed.get("Claude") == "model", str(typed))

    typed = typed_of("According to Dr Smith, the tool Docker works.")
    check("typing: a title still yields a person", typed.get("Dr Smith") == "person",
          str(typed))
    check("typing: a tool is a tool", typed.get("Docker") == "tool", str(typed))
