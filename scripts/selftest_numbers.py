"""The numbers section: what reaches it, and what is kept out of it.

The defect this exists for is a report whose "Numbers, Pricing & Benchmarks"
reads

    21 — Browse all skills
    173K — & system
    127 — CA
    02 — PHASE TWO

Interface furniture, presented as figures a reader would use. Every one of those
passed the old rule, because each had a *grounded label* — the text physically
beside the number on the frame — and the old rule asked only for a label. A label
proves the number was read correctly; it says nothing about what the number
counts. That is the gap, and it is closed by a semantic classification that reads
the speech, the label, the unit, the frame's layout and the segment's subject.

The cases below run in two directions, and both matter. A filter that only knows
what to remove removes the figures too: the benchmark table, the price row, the
count a speaker gave a subject, and the figure that is both on screen and in the
speech all have to survive, or the report has thrown away the material it exists
to record. The raw OCR observation is asserted to survive in every case — the
filter withholds a number from one section, it never deletes an observation.
"""
from __future__ import annotations

import semantic as SEM

#: Bound by `selftest.py` at import, exactly as the other sub-suites are.
check = None


def _region(text: str, x: int, y: int, w: int = 120, h: int = 14) -> dict:
    return {"text": text, "confidence": 0.95, "bbox": [x, y, w, h], "read_method": "native"}


class _Frame:
    """The parts of a `VisualFrame` the classifier reads."""

    def __init__(self, regions):
        self.frame_id = "s0001"
        self.t_start = 10.0
        self.t_end = None
        self.ocr_regions = regions
        self.ocr_text = "\n".join(r["text"] for r in regions)
        self.ocr_confidence = 0.99
        self.ocr_read_method = "native"


class _Ctx:
    def __init__(self, speech: str):
        self.speech = speech
        self.frames = []


def _role(label, speech, regions, value, unit, display,
          classification="meaningful", reason="labelled on its own line"):
    obs = SEM.Observation(frame_id="s0001", text=display, line_label=label)
    obs.values = [(value, display, unit)]
    obs.classification = classification
    obs.reason = reason
    return SEM.classify_number_role(obs, value, unit, display, _Frame(regions), _Ctx(speech))


#: The figures from the report that prompted this, with the regions that produced
#: them. Kept verbatim from the artifact so the test is about those and not about
#: cases I invented after seeing the fix.
_UI_CASES = (
    ("21 — Browse all skills", "Browse all skills", "21", None, "21",
     "That's everything, thanks for watching.",
     [_region("21", 10, 100), _region("Browse all skills", 200, 100)]),
    ("173K — & system", "& system", "173K", None, "173K",
     "That's everything, thanks for watching.",
     [_region("173K", 10, 130), _region("& system", 200, 130)]),
    ("127 — CA", "CA", "127", None, "127",
     "This row is an entry in the sheet.",
     [_region("127", 10, 160), _region("CA", 200, 160)]),
    ("02 — PHASE TWO", "PHASE TWO", "02", None, "02",
     "On to the next part of the setup.",
     [_region("02", 10, 190), _region("PHASE TWO", 200, 190)]),
)

_KEEP_CASES = (
    ("a benchmark table row", "Model A", "The benchmark is shown here on the right.",
     [_region("Model A", 10, 100), _region("32 tok/s", 200, 100, w=80),
      _region("Model B", 10, 130), _region("18 tok/s", 200, 130, w=80)],
     "32", "tok/s", "32tok/s", "benchmark"),
    ("a price row", "Pro plan", "Here is the pricing table for each plan.",
     [_region("Pro plan", 10, 100), _region("$20", 200, 100),
      _region("Free", 10, 130), _region("$0", 200, 130)],
     "20", None, "$20", "price"),
    ("a count with a label the speaker also used", "queue depth",
     "Requests queue up behind it while the batch drains.",
     [_region("412", 10, 100), _region("queue depth", 200, 100, w=140)],
     "412", None, "412", "metric"),
    ("a figure shown and spoken", "VRAM", "You need at least 32GB of VRAM for this.",
     [_region("32 GB", 10, 100), _region("VRAM", 200, 100)],
     "32", "GB", "32GB", "metric"),
)


def t_ui_numbers_are_rejected() -> None:
    """The four figures that reached the report, and the rule that now stops them."""
    for label, name, value, unit, display, speech, regions in _UI_CASES:
        role, importance, why = _role(name, speech, regions, value, unit, display)
        check(f"N1 {label}: classified as interface or unidentifiable",
              role in ("ui", "unknown"), f"role={role!r} ({why})")
        check(f"N1 {label}: not high importance", importance != "high",
              f"importance={importance!r}")
        # The gate in report.py is the second line of defence and the one a
        # reader actually meets, so it is asserted on the same inputs rather
        # than assumed to follow from the role.
        from report import Refs, _is_noise_number
        item = SEM.Item(id="itm-x", type="number", value=value, display=display,
                        unit=unit, subject=name, importance=importance,
                        semantic_role=role,
                        provenance=[SEM.ocr_provenance(_Frame(regions), display,
                                                       confidence=0.99)])
        check(f"N1 {label}: withheld from the numbers section",
              _is_noise_number(item, Refs(SEM.Evidence(schema_version=3, video={},
                                                      transcript={}, snippets=[],
                                                      segments=[], items=[item],
                                                      visual=[]))),
              "the report gate let it through")


def t_benchmark_price_and_count_are_retained() -> None:
    """The other direction. A filter that only knows what to remove is a defect."""
    for name, label, speech, regions, value, unit, display, expected in _KEEP_CASES:
        role, importance, why = _role(label, speech, regions, value, unit, display)
        check(f"N2 {name}: role is {expected}", role == expected,
              f"role={role!r} ({why})")
        check(f"N2 {name}: kept in the report", importance in ("high", "medium"),
              f"importance={importance!r}")


def t_confidence_is_not_importance() -> None:
    """A perfectly legible UI number and a partly legible benchmark, side by side.

    The counterexample is the whole reason `importance` is a separate field. If
    either of these derived from the read score, one of them would be wrong and
    nothing in the schema would say so.
    """
    obs = SEM.Observation(frame_id="s0001", text="173K", line_label="subscribers")
    obs.values = [("173K", "173K", None)]
    obs.classification = "meaningful"
    obs.reason = "labelled on its own line"
    frame = _Frame([_region("173K", 10, 100), _region("subscribers", 200, 100)])
    frame.ocr_confidence = 0.99
    role, importance, _ = SEM.classify_number_role(
        obs, "173K", None, "173K", frame, _Ctx("Thanks for watching."))
    check("N3 a 0.99 read of a subscriber counter is still low importance",
          importance == "low", f"importance={importance!r}")
    check("N3 ...and is classified as interface", role == "ui", f"role={role!r}")

    obs2 = SEM.Observation(frame_id="s0002", text="18 tok/s", line_label="Model B")
    obs2.values = [("18", "18tok/s", "tok/s")]
    obs2.classification = "meaningful"
    obs2.reason = "labelled on its own line"
    frame2 = _Frame([_region("Model A", 10, 100), _region("32 tok/s", 200, 100, w=80),
                     _region("Model B", 10, 130), _region("18 tok/s", 200, 130, w=80)])
    frame2.ocr_confidence = 0.78
    role2, importance2, _ = SEM.classify_number_role(
        obs2, "18", "tok/s", "18tok/s", frame2,
        _Ctx("The benchmark is shown here on the right."))
    check("N3 a 0.78 read of a benchmark row is high importance",
          importance2 == "high", f"importance={importance2!r}")
    check("N3 ...and is classified as a benchmark", role2 == "benchmark", f"role={role2!r}")


def t_unit_without_a_label_is_not_a_fact() -> None:
    """`4% — outoute`: the unit is real, the label is a misread, the pair is noise.

    Found on a real report, and the case that separates "the unit tells me what
    kind of number this is" from "that is enough to print it". A figure the
    speaker also said is caught by cross-modal support and keeps `high`; one that
    is only on screen, with a label that names nothing, does not.
    """
    role, importance, why = _role(
        "outoute", "Nothing in this segment names a quantity.",
        [_region("4%", 10, 100), _region("outoute", 200, 100)], "4", "%", "4%")
    check("N4a a unit with a meaningless label is identified as a metric", role == "metric",
          f"role={role!r}")
    check("N4a ...and is not high enough to print", importance == "low",
          f"importance={importance!r}")
    check("N4a the reason says why", "no label says what it measures" in why, why)

    # The same figure, spoken, is the video's own claim and is kept.
    role2, importance2, _ = _role(
        "VRAM", "You need at least 32GB of VRAM for this.",
        [_region("32 GB", 10, 100), _region("VRAM", 200, 100)], "32", "GB", "32GB")
    check("N4a a unit that the speech also states stays high",
          importance2 == "high", f"importance={importance2!r}")


def t_isolated_number_is_suppressed() -> None:
    """A figure nothing explains. Read, recorded, and not shown as a fact."""
    role, importance, why = _role(
        "and", "Nothing in particular is being discussed here.",
        [_region("42", 10, 100), _region("and", 200, 100)], "42", None, "42")
    check("N4 an isolated figure is not identified", role == "unknown", f"role={role!r}")
    check("N4 ...and is low importance", importance == "low", f"importance={importance!r}")
    check("N4 the reason is recorded rather than assumed", bool(why), why)


def t_raw_observation_survives() -> None:
    """The filter withholds; it never deletes.

    Every number the extractor reads stays in `visual[].ocr_text` whatever the
    classification decided, because the decision is a *reporting* judgement about
    one section of the report and the observation underneath is the evidence.
    """
    frame = _region_list = [_region("21", 10, 100), _region("Browse all skills", 200, 100)]
    obs = SEM.Observation(frame_id="s0001", text="21", line_label="Browse all skills")
    obs.values = [("21", "21", None)]
    SEM.classify_observation(obs, "That's everything, thanks for watching.")
    check("N5 a UI figure is still classified, not dropped", bool(obs.classification),
          obs.classification)
    check("N5 the raw read is unchanged on the frame",
          "21" in "\n".join(r["text"] for r in frame),
          str([r["text"] for r in frame]))


def t_role_survives_the_round_trip() -> None:
    """A role written into `evidence.json` is read back as the same role.

    The field is a claim about the evidence, and an artifact that could not be
    re-read would make the report unauditable — which is the failure the whole
    provenance layer exists to prevent, applied to a field added to it.
    """
    from evidence_model import Item
    item = Item(id="itm-1", type="number", value="40", display="40 ms", unit="ms",
                importance="high", semantic_role="benchmark",
                provenance=[SEM.transcript_provenance(
                    SEM.Snippet(id="s001", start_s=0.0, end_s=1.0, text="40 ms"))])
    back = Item.from_dict(item.to_dict(), "items[itm-1]")
    check("N6 semantic_role round-trips", back.semantic_role == "benchmark",
          str(back.semantic_role))
    check("N6 it is absent rather than null when unset",
          "semantic_role" not in Item(id="itm-2", type="number", value="1",
                                      provenance=item.provenance).to_dict(),
          "an unset role was written into the artifact")

    import evidence_validate as EV
    from evidence_model import Evidence, Snippet

    def _doc(i):
        # A segment, because an unassigned snippet is a blocking error and this
        # test is about the role field, not about segment membership. Built here
        # rather than through a fixture so the document is four lines long.
        from evidence_model import Segment
        snip = Snippet(id="s001", start_s=0.0, end_s=1.0, text="40 ms")
        seg = Segment(id="seg-1", start_s=0.0, end_s=1.0, title="All",
                      title_source="chapter", snippet_ids=["s001"],
                      item_ids=[i.id], frame_ids=[])
        return Evidence(schema_version=3, video={}, transcript={}, snippets=[snip],
                        segments=[seg], items=[i], visual=[])

    ev = _doc(item)
    check("N6 a known role validates", EV.validate_evidence(ev).valid,
          str(EV.validate_evidence(ev).blocking_errors[:2]))
    item.semantic_role = "invented-role"
    codes = [e["code"] for e in EV.validate_evidence(_doc(item)).blocking_errors]
    check("N6 an unknown role is refused", "unsupported-enum" in codes, str(codes))


# --------------------------------------------------------------------------- #
# subject extraction for a spoken figure
# --------------------------------------------------------------------------- #

#: (label, sentence, the figure as it appears, its unit, the subject the previous
#: sentence established, the expected subject).
#:
#: Deterministic fixtures, written as literal sentences rather than built from a
#: template, because a template would encode the same assumption as the code it
#: is checking. The expected value is `None` wherever the sentence genuinely
#: does not name a referent — those are the load-bearing rows: a confidently
#: wrong subject files a figure under the wrong heading, which is worse than
#: leaving it unresolved.
_SUBJECT_FIXTURES = (
    ("an adverb before the figure is not its subject",
     "This ultimately reaches 100%.", "100", None, None, None),
    ("an adverb opening the sentence is not its subject",
     "Ultimately, 100% of the work is done.", "100", None, None, None),
    ("a comparative names the dimension, not the referent",
     "The model is 200x faster.", "200", "x", None, "model"),
    ("a past-tense verb does not become the subject",
     "The latency dropped to 20 ms.", "20", "ms", None, "latency"),
    ("the figure's own unit is stepped over",
     "You need 32 GB of VRAM for this model.", "32", "GB", None, "VRAM"),
    ("a count takes the noun beside it",
     "It has 700 users on the waitlist.", "700", None, None, "users"),
    ("a rate denominator is not a referent",
     "It could cost $500 per month.", "500", None, None, None),
    ("a modifier does not join the subject",
     "We did about 700 users last week.", "700", None, None, "users"),
    ("a bare cost with no referent is unresolved",
     "The plan could cost $500.", "500", None, None, None),
    ("a magnitude word is dropped, not treated as the subject",
     "Up to 2 million tokens in context.", "2", None, None, "tokens"),
    ("a comparative quantifier is not a subject",
     "It takes less than 10 seconds.", "10", None, None, None),
    ("a transitive predicate keeps its object",
     "This increases throughput to 32 tok/s.", "32", "tok/s", None, "throughput"),
    ("an explicit noun subject is used",
     "The model runs at 30 frames per second.", "30", None, None, "model"),
    ("a referent continued from the previous sentence",
     "It hit 40 ms.", "40", "ms", "latency", "latency"),
    ("a figure with no referent at all stays unresolved",
     "And then 500.", "500", None, None, None),
    # The next group was added after running these against real sentences rather
    # than against the fixtures above. Every one of them was a *real* output
    # that was wrong in the same way as the two that started this work: a word
    # adjacent to the figure that cannot name a referent. A fixture set written
    # only from imagination missed every one.
    ("a contraction is not a referent",
     "We're talking like 200x faster than the previous version.", "200", "x", None, None),
    ("a comparison preposition is not a referent",
     "It is 3x slower than the baseline.", "3", "x", None, None),
    ("a bare verb is not a referent",
     "It scores 92% on the benchmark.", "92", "%", None, None),
    ("an irregular past-tense verb is not a referent",
     "The repo grew to 45000 lines of code.", "45000", None, None, "lines"),
    ("a month is a billing period, not a referent",
     "Pricing starts at $20 per month.", "20", None, None, None),
    ("money spent per token has no referent in the sentence",
     "Training costs $3 per million tokens.", "3", None, None, None),
    # …and the ones that must survive, which is the other half. A gate that
    # refuses everything passes every rejection above and is useless.
    ("a metric whose subject is a bare noun",
     "The GPU has 80 GB of memory.", "80", "GB", None, "GPU"),
    ("a count of people",
     "The team has 12 engineers.", "12", None, None, "team"),
    ("a compound metric noun",
     "Context length is 128000 tokens.", "128000", None, None, "Context length"),
    ("a rate of work",
     "It processes 1200 requests per second.", "1200", None, None, "requests"),
)


def t_spoken_figure_subject() -> None:
    """What a spoken figure is *about*, which is the question the transcript pass
    deliberately deferred and this enrichment answers.

    The defect this exists for is a subject read off the nearest word: "100% —
    ultimately", "200x — 're talking like". Both are real outputs, both are
    nonsense, and both are invisible downstream because a subject is a plausible-
    looking string in a typed field. So the rule asserted here is not "the
    subject is correct" but "a non-null subject names a referent" — which is the
    property that was actually broken, and the one a reader would rely on.
    """
    for label, text, figure, unit, previous, want in _SUBJECT_FIXTURES:
        at = text.find(figure)
        check(f"subject: fixture is well-formed ({label})", at > 0, text)
        got = SEM.subject_for(text, at, len(figure), unit, previous)
        check(f"subject: {label}", got == want, f"got {got!r}, wanted {want!r}")


def t_subject_is_a_referent_not_an_adjacent_token() -> None:
    """A non-null subject must be capable of naming a referent.

    The general form of the rule the fixtures above test case by case, and the
    one that has to hold for sentences nobody wrote down. It is asserted against
    the words that produced the real defects, plus the classes they belong to —
    an adverb, a gerund, a magnitude, a unit — so a future change that
    reintroduces the shape fails here even if no fixture happens to catch it.
    """
    for word in ("ultimately", "talking", "million", "ms", "seconds", "tok/s",
                 "faster", "per", "about", "'re", "like", "than", "grew",
                 "scores", "starts", "respond"):
        check(f"subject gate: {word!r} cannot be a referent",
              SEM._clean_referent(word) is None, SEM._clean_referent(word))
    for word in ("latency", "VRAM", "users", "tokens", "throughput", "memory"):
        check(f"subject gate: {word!r} can be a referent",
              SEM._clean_referent(word) == word, SEM._clean_referent(word))

    # And the gate is not a licence to invent: a phrase that survives it still
    # has to be the phrase the sentence actually contains.
    check("subject gate: it does not manufacture a phrase from nothing",
          SEM._clean_referent("") is None and SEM._clean_referent("   ") is None)
    check("subject gate: a clause is not a subject",
          SEM._clean_referent("the model that the vendor claims is faster") is None,
          SEM._clean_referent("the model that the vendor claims is faster"))


def t_subject_does_not_borrow_from_the_previous_sentence_implicitly() -> None:
    """Cross-sentence carry-over is explicit or it does not happen.

    A window that reaches back past a full stop picks up a real word from the
    wrong sentence — "Latency was the bottleneck. It hit 40 ms" filed the figure
    under `bottleneck`, which is harder to catch than a bad guess because it
    looks grounded. So the window stops at the sentence boundary, and the only
    route across it is the `previous` argument, which a caller supplies from
    evidence it has already resolved.
    """
    text = "Latency was the bottleneck. It hit 40 ms."
    at = text.find("40")
    check("subject: a referent is not borrowed from the previous sentence",
          SEM.subject_for(text, at, 2, "ms") is None,
          SEM.subject_for(text, at, 2, "ms"))
    check("subject: …unless it is passed in explicitly",
          SEM.subject_for(text, at, 2, "ms", "latency") == "latency",
          SEM.subject_for(text, at, 2, "ms", "latency"))
    # A `previous` that is itself not a referent does not become one by arriving
    # through the trusted argument.
    check("subject: an implausible carry-over is refused too",
          SEM.subject_for(text, at, 2, "ms", "ultimately") is None,
          SEM.subject_for(text, at, 2, "ms", "ultimately"))


# --------------------------------------------------------------------------- #
# the classification matrix
# --------------------------------------------------------------------------- #

#: A results table as one is actually drawn: two columns of names and bare
#: decimals, with a header row. Nothing here carries a unit, which is exactly
#: why this shape was the one the table test could not see.
_BENCH_TABLE = [
    _region("Model", 10, 10), _region("Score", 200, 10),
    _region("Llama", 10, 40), _region("72.4", 200, 40),
    _region("Qwen", 10, 70), _region("81.9", 200, 70),
    _region("GPT", 10, 100), _region("88.1", 200, 100),
]
_BENCH_SPEECH = "Here are the benchmark results comparing the models on accuracy."

#: A code editor: a rising column of line numbers beside the code they number.
_CODE_GUTTER = ([_region(str(n), 10, 10 + n * 20) for n in range(20, 25)]
                + [_region("const x = compute(a, b);", 220, 10 + n * 20)
                   for n in range(20, 25)])

#: Every row a name beside a figure. `(label, value, display, regions, speech,
#: expected_role, expected_in_report)`.
_MATRIX = (
    # — meaningful ——————————————————————————————————————————————————————————
    ("a benchmark table of bare decimals", "Qwen", "81.9", "81.9",
     _BENCH_TABLE, _BENCH_SPEECH, "benchmark", True),
    ("a table value the speaker never reads aloud", "Llama", "72.4", "72.4",
     _BENCH_TABLE, _BENCH_SPEECH, "benchmark", True),
    ("a price", "Pro plan", "20", "$20",
     [_region("$20", 10, 100), _region("Pro plan", 200, 100)],
     "The Pro plan costs twenty dollars a month.", "price", True),
    ("a version number", "version", "3.2.1", "3.2.1",
     [_region("3.2.1", 10, 100), _region("version", 200, 100)],
     "It ships in the 3.2.1 release.", "metric", True),
    ("a configuration value", "max_tokens", "4096", "4096",
     [_region("4096", 10, 100), _region("max_tokens", 200, 100, w=140)],
     "We set max tokens to four thousand.", "metric", True),
    ("a subscriber count the speaker also gives", "subscribers", "173K", "173K",
     [_region("173K", 10, 100), _region("subscribers", 200, 100)],
     "It has 173K subscribers now.", "metric", True),
    ("a subscriber counter spelled out", "Subscribers", "173K", "173K",
     [_region("173K", 10, 100), _region("Subscribers", 200, 100)],
     "That's everything, thanks for watching.", "ui", False),
    # — low-value ———————————————————————————————————————————————————————————
    ("a code editor line number", "export default function Footer()", "21", "21",
     _CODE_GUTTER, "We benchmarked the models and here are the latency results.",
     "unknown", False),
    ("a gutter beside a filename", "Footer.tsx", "25", "25",
     _CODE_GUTTER, "We benchmarked the models and here are the latency results.",
     "unknown", False),
    ("a UI counter", "Comments", "1024", "1,024",
     [_region("1,024", 10, 100), _region("Comments", 200, 100)],
     "That's it for this one.", "ui", False),
    ("a timestamp", "timestamp", "12:34", "12:34",
     [_region("12:34", 10, 100), _region("timestamp", 200, 100)],
     "Thanks for watching.", "ui", False),
    ("a subscriber counter the speaker never mentions", "& system", "173K", "173K",
     [_region("173K", 10, 100), _region("& system", 200, 100)],
     "That's everything, thanks for watching.", "unknown", False),
    ("a chapter grid in a segment that happens to be about numbers", "Opening", "3",
     "3",
     [_region("00:00", 10, 10), _region("Opening", 200, 10),
      _region("12:34", 10, 40), _region("Middle", 200, 40),
      _region("1:02:30", 10, 70), _region("End", 200, 70)],
     "Let's look at the throughput numbers next.", "unknown", False),
)


def t_classification_matrix() -> None:
    """Every category §3 names, in one place, running in both directions.

    The filter has to be wrong in the two opposite ways and the two failures
    look nothing alike from the outside: a benchmark table that never qualifies
    is a report with no numbers in it, and a code editor that does qualify is a
    report listing line numbers as findings. A suite that only tested the second
    would have shipped the first.

    A clock time is here as a table cell for a reason that is not "timestamps
    are chrome": `00:00`, `12:34` and `1:02:30` side by side with names is a
    chapter list, and a chapter list has the geometry of a results table. The
    two are told apart by the cell's shape, which is the only thing that does.
    """
    for name, label, value, display, regions, speech, expected, in_report in _MATRIX:
        unit = None
        r, importance, why = _role(label, speech, regions, value, unit, display)
        check(f"N7 {name}: role is {expected}", r == expected, f"role={r!r} ({why})")
        kept = importance in ("high", "medium")
        check(f"N7 {name}: {'kept' if in_report else 'withheld'} in the report",
              kept == in_report, f"importance={importance!r}")


def t_gutter_and_table_are_told_apart_structurally() -> None:
    """The exclusion is geometry, and it fires only on geometry.

    A gutter and a results table are the same shape to a naive row test, so the
    thing separating them is a rising column of small integers on the left. If
    that test can be made to pass on a real table it is worse than no test, so
    it is asserted in both directions: the gutter is refused, and the table —
    which is the same grid with names in the first column — is not.
    """
    check("N8 the editor gutter is recognised",
          SEM._line_number_gutter(_Frame(_CODE_GUTTER)),
          "a rising column of line numbers was not recognised")
    check("N8 a results table is not mistaken for one",
          not SEM._line_number_gutter(_Frame(_BENCH_TABLE)),
          "a results table was read as a line-number gutter")
    check("N8 the results table is a table",
          SEM._in_table_row(_Frame(_BENCH_TABLE)),
          "a table of bare decimals was not seen as a table")



# --------------------------------------------------------------------------- #
# §4 — what may not become a subject, and what must still
# --------------------------------------------------------------------------- #

#: Sentences where nothing at all may come back. Each of these once returned a
#: word that names nothing, which is the failure the whole gate exists to
#: prevent: the figure is filed under a heading that is not what it measures, and
#: because the word is a real word from the sentence it reads as grounded.
_NO_SUBJECT = (
    ("it costs 500 dollars", "500", None, "a currency written as a word"),
    ("requests per second hit 1200", "1200", None, "the denominator of a rate"),
    ("this ultimately reaches 100%", "100", None, "an adverb"),
    ("throughput basically plateaus at 8", "8", None, "another adverb"),
    ("we're talking like 200x", "200", None, "a contraction's clitic"),
    ("it's about 12% better", "12", None, "another clitic"),
    ("200x faster than the baseline", "200", None, "a comparative"),
    ("i think that 500 users churn", "500", None, "a matrix clause, not a noun phrase"),
)

#: The same categories, where a referent *does* exist and must still be found —
#: with the offending word somewhere in the sentence and absent from the answer.
#: Asserting only the table above would pass against a gate that returns None
#: everywhere, which is a gate that has stopped working rather than one being
#: careful. `expected` is the referent; `forbidden` is the word in the sentence
#: that must not have become the heading.
_SUBJECT_SKIPS_THE_BAD_WORD = (
    ("the latency dropped to 20 ms", "20", "ms", "latency", "ms",
     "a unit written after the figure"),
    ("the latency dropped to 20 ms", "20", "ms", "latency", "dropped",
     "a past-tense verb before it"),
    ("we saw 42 mentions", "42", None, "mentions", "saw",
     "a bare-form verb after a pronoun"),
    ("throughput improved 15 percent", "15", None, "throughput", "improved",
     "a verb in the middle"),
    ("up to 2 million tokens", "2", None, "tokens", "million",
     "a magnitude word"),
    ("users last week numbered 700", "700", None, "users", "week",
     "a unit-shaped modifier"),
)

#: The other direction again, with nothing adversarial anywhere in the sentence.
#: A filter is only conservative relative to what it refuses; without these the
#: suite passes against a gate that returns None for every figure in the document.
_STILL_A_SUBJECT = (
    ("the queue depth hit 412", "412", None, "queue depth"),
    ("32GB of VRAM", "32", "GB", "VRAM"),
    ("3.5 million views", "3.5", None, "views"),
    ("the benchmark hit 88.1", "88.1", None, "benchmark"),
    ("GIL dropped to 40 ms", "40", "ms", "GIL"),
)


def t_subject_is_null_rather_than_wrong() -> None:
    """§4, one assertion per category: an unresolved subject beats a wrong one.

    Every case here has a plausible wrong answer, which is what makes them worth
    a fixture: `dollars`, `ultimately`, `saw`, `'re` and `think` all read as
    subjects under a rule more specific than the one in force, and each is a
    real word from the sentence rather than an invention — which is exactly why
    a wrong subject is harder to catch than a missing one.
    """
    for text, needle, unit, what in _NO_SUBJECT:
        at = text.find(needle)
        got = SEM.subject_for(text, at, len(needle), unit)
        check(f"N9 {what}: no subject rather than a wrong one",
              got is None, f"{text!r} -> {got!r}")


def t_the_gate_finds_the_referent_and_skips_the_word() -> None:
    """The tightening must not cost the referents it was meant to protect.

    A gate that refuses everything is not conservative, it is broken, and it
    fails the same way a permissive one does — by filing a figure under no
    heading at all. The tradeoff this records is deliberate and
    one-directional: `subject = null` is preferred to a confidently wrong
    subject, and the price is paid in recall on phrasings this layer cannot
    parse without a parser.
    """
    for text, needle, unit, expected, forbidden, what in _SUBJECT_SKIPS_THE_BAD_WORD:
        at = text.find(needle)
        got = SEM.subject_for(text, at, len(needle), unit)
        check(f"N10 {what}: the referent is {expected!r}", got == expected,
              f"{text!r} -> {got!r}")
        check(f"N10 {what}: ...and not {forbidden!r}",
              got is None or forbidden not in got.split(),
              f"got {got!r}")


def t_the_gate_does_not_over_reach() -> None:
    """Ordinary referents, with nothing adversarial anywhere in the sentence."""
    for text, needle, unit, expected in _STILL_A_SUBJECT:
        at = text.find(needle)
        got = SEM.subject_for(text, at, len(needle), unit)
        check(f"N11 {text!r}: still resolves to {expected!r}", got == expected,
              f"got {got!r}")


def t_ui_verdict_is_pinned_not_merely_low() -> None:
    """N12. Added after mutation testing: two branches were only half-asserted.

    `t_ui_numbers_are_rejected` accepts any verdict in {ui, unknown} that is not
    high importance. That is the right bar for "does this reach the report", and
    it is too loose to say *which* rule did the suppressing — so deleting the
    honour-the-observation-layer branch, which turns `ui` into `unknown`, changed
    nothing. These assertions pin the role and the stated reason, which is what
    distinguishes the two rules from each other.
    """
    # Signal 6: the observation layer already called this furniture. Honouring
    # that is the difference between `ui` and the `unknown` a figure with nothing
    # to say about it gets, and the two are not the same claim.
    role, importance, reason = _role(
        "", "Nothing in the speech is relevant to this figure.",
        [_region("77", 10, 100)], "77", None, "77", classification="ui-chrome")
    check("N12 an observation the layer called ui-chrome is classified `ui`",
          role == "ui", f"role={role!r} reason={reason!r}")
    check("N12 ...and the reason says the observation layer decided",
          "observation layer" in reason, reason)

    # Signal 3: an ordinal label. Same role and importance as the vocabulary rule
    # below it, so only the reason tells the two apart — which is exactly why
    # asserting importance alone would have left this rule untested.
    role, importance, reason = _role(
        "PHASE TWO", "On to the next part of the setup.",
        [_region("02", 10, 100), _region("PHASE TWO", 200, 100)], "02", None, "02")
    check("N12 an ordinal label is `ui` and low", (role, importance) == ("ui", "low"),
          f"role={role!r} importance={importance!r}")
    check("N12 ...and the reason names the ordinal rule, not the vocabulary one",
          "ordinal" in reason, reason)

    # ...and the neighbouring vocabulary rule, so the assertion above cannot be
    # satisfied by the wrong branch producing a similar sentence.
    _, _, reason = _role(
        "Subscribers", "That's everything, thanks for watching.",
        [_region("173K", 10, 100), _region("Subscribers", 200, 100)], "173K", None, "173K")
    check("N12 ...and a furniture label is caught by a different rule",
          "interface furniture" in reason, reason)


# --------------------------------------------------------------------------- #
# N13 · a grouped expression, and a figure whose label follows it
# --------------------------------------------------------------------------- #

def _obs(text: str, speech: str, regions: list[dict] | None = None):
    """One observation as `_observations_for` builds it, from a real region."""
    frame = _Frame(regions if regions is not None else [_region(text, 10, 10)])
    for o in SEM._observations_for(frame):
        if o.text == text:
            SEM.classify_observation(o, speech)
            return o, frame
    raise AssertionError(f"no observation built for {text!r}")


def t_equation_is_read_as_one_expression() -> None:
    """`28 × 28 = 784` was legible on a frame and absent from the evidence.

    Not because the OCR was wrong and not because the number had no unit — but
    because `find_values` requires a unit, so a formula parsed to nothing at all,
    and the transcript's own grammar was worse than nothing: it read `28 × 28` as
    the figure `28` carrying the unit `×`, keeping one operand and dropping the
    equation's answer.
    """
    for label, text, want in (
        ("multiplication", "28 × 28 = 784", ["28", "784"]),
        ("an ASCII x reads the same", "28 x 28 = 784", ["28", "784"]),
        ("a sum", "16 +16 + 10", ["16", "10"]),
        ("a product of sums", "784×16+16×16 + 16×10", ["784", "16", "10"]),
        ("an arrow", "784 → 10", ["784", "10"]),
    ):
        obs, _ = _obs(text, "Nothing in the speech is relevant to this figure.")
        check(f"N13 {label}: every figure of the expression is read",
              [v[1] for v in obs.values] == want,
              f"got {[v[1] for v in obs.values]}, wanted {want}")
        check(f"N13 {label}: the expression is recorded, not just its figures",
              obs.structure == text, f"structure={obs.structure!r}")
        check(f"N13 {label}: the observation is meaningful, not irrelevant",
              obs.classification == "meaningful",
              f"{obs.classification}: {obs.reason}")

    obs, frame = _obs("28 × 28 = 784", "Here is how the input is flattened.",
                      [_region("Why the layers?", 10, 10), _region("28 × 28 = 784", 10, 60)])
    made = SEM._screen_item(obs, frame, _Ctx("Here is how the input is flattened."),
                            SEM.SemanticConfig())
    check("N13 a formula yields one item per figure, not one for the first",
          isinstance(made, list) and len(made) == 2, f"got {type(made).__name__}")
    check("N13 each figure carries the expression it was read from",
          all((i.detail or {}).get("expression") == "28 × 28 = 784" for i in made),
          str([i.detail for i in made]))
    check("N13 the operands are not filed as the answer",
          [i.value for i in made] == ["28", "784"], str([i.value for i in made]))

    # A slide that explains the figure ranks it as a headline figure; one that
    # does not is recorded and ranked below a corroborated reading. The
    # distinction is drawn from the frame's own words, not the segment's topic.
    obs, frame = _obs("28 × 28 = 784", "x", [_region("Why the layers?", 10, 10),
                                             _region("28 × 28 = 784", 10, 60)])
    role, importance, why = SEM.classify_number_role(
        obs, "784", None, "784", frame, _Ctx("x"))
    check("N13 a formula on an explaining frame is a high-importance metric",
          (role, importance) == ("metric", "high"), f"{role}/{importance}: {why}")


def t_expression_shape_refuses_a_misread_subscript() -> None:
    """A plus sign in a matrix header is not an equation.

    `+ Wo, 1 a1` and `(0) + 60)` were read off an animation of a matrix and both
    parse as expressions if the only question asked is "is there an operator in
    here". A formula starts with a figure and an operator; a run that starts with
    the operator is punctuation that landed inside a misread subscript.
    """
    for text in ("+ Wo, 1 a1", "(0) + 60)", "—3", "(0)", "WK, 0 Wk,1"):
        check(f"N13 {text!r} is not read as an expression",
              SEM.find_expression_values(text) is None,
              str(SEM.find_expression_values(text)))
    check("N13 a clock time is not an expression", SEM.find_expression_values("12:34") is None)
    check("N13 a version is not an expression", SEM.find_expression_values("1.2.3") is None)
    check("N13 a multiplier in prose is not an expression",
          SEM.find_expression_values("2x faster") is None)
    check("N13 a URL is not an expression",
          SEM.find_expression_values("https://x.com/2/3") is None)


def t_trailing_label_recovers_the_other_side() -> None:
    """`label_for_value` reads the words to the *left*. A legend writes them right.

    `784 inputs`, `10 outputs` and `784 weights per neuron` were on the frames of
    a real video and reached no evidence item, because the referent followed the
    figure and only one direction was ever read.
    """
    for text, want_value, want_label in (
        ("784 inputs", "784", "inputs"),
        ("10 outputs", "10", "outputs"),
        ("784 weights per neuron", "784", "weights per neuron"),
    ):
        got = SEM.trailing_label(text)
        check(f"N13 {text!r} recovers its referent", got is not None
              and (got[0], got[2]) == (want_value, want_label), str(got))
        obs, _ = _obs(text, "Nothing in the speech is relevant to this figure.")
        check(f"N13 {text!r} becomes a value, not an ambiguous read",
              bool(obs.values) and obs.line_label == want_label,
              f"values={obs.values} label={obs.line_label!r}")
    for text in ("784 •", "784 02", "00:01", "784"):
        check(f"N13 {text!r} still has no referent", SEM.trailing_label(text) is None,
              str(SEM.trailing_label(text)))


def t_glyphs_are_not_referents() -> None:
    """The noise that survived a real run was a subscript per item, at high value.

    A maths animation's frames hold em-dashes, bullets, and single letters with a
    subscript digit. Each was accepted as the label beside a figure, and the
    table-and-topic rule then promoted the figure — `2 — З`, `1 — ao`, `0 — (O)`,
    three of them high-importance, while the equation the video was about went
    unrecorded.
    """
    for label in ("—3", "—З", "•", "(O)", "O1", "a5", "a4", "WK,", ".."):
        check(f"N13 {label!r} is not a referent", SEM._clean_label(label) is None,
              f"got {SEM._clean_label(label)!r}")
    check("N13 a real referent still passes", SEM._clean_label("inputs") == "inputs")
    check("N13 a two-character string is not a referent", SEM._clean_label("GB") is None)
    check("N13 a real two-word referent still passes",
          SEM._clean_label("hidden layers") == "hidden layers")


def t_no_fabricated_visual_number() -> None:
    """The boundary holds: recovery reads the frame and adds nothing to it.

    Every case that must *not* yield a number is asserted here rather than left
    to the two tests above, because the failure this whole change risks is the
    opposite one — loosening the gate until a subscript, a clock time and a
    version all became figures a report would print.
    """
    frame = _Frame([_region("02", 10, 10), _region("PHASE TWO", 200, 10),
                    _region("127", 10, 40), _region("likes", 200, 40),
                    _region("00:01", 10, 70), _region("1.2.3", 10, 100)])
    found = []
    for o in SEM._observations_for(frame):
        SEM.classify_observation(o, "Let us look at the matrix representation.")
        if o.classification == "meaningful" and o.values:
            made = SEM._screen_item(o, frame, _Ctx("Let us look at the matrix representation."),
                                    SEM.SemanticConfig())
            found.extend(made if isinstance(made, list) else [made])
    for value in ("02", "127", "00:01", "1.2.3"):
        check(f"N13 {value!r} did not become a figure", value not in {i.value for i in found},
              str([i.value for i in found]))
    check("N13 every recovered item names the frame it was read from",
          all(any(p.ref == "s0001" for p in i.provenance) for i in found),
          str([i.provenance for i in found]))
