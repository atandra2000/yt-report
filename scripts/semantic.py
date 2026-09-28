"""Semantic multimodal evidence extraction: what the numbers *mean*.

The OCR pass recovered screen-only text. It is mechanical by design, and what it
produces is a list of things it saw: `0.09`, `700`, `32GB`, `0.0%`, `0x`. None of
those is a fact yet. `0.0%` beside a channel's subscriber counter is UI chrome;
`32GB` next to "you need at least 32GB of VRAM" is the video's own claim, stated
twice; `0x` alone is a glyph. The difference is not visible in the string — it is
in the relationship between the string, the sentence being spoken, and the frame
it sits in. That relationship is what this module computes.

Three separations the shape exists to hold, because collapsing any of them is a
specific, observed failure:

* **OCR confidence is not semantic confidence.** The engine's score ranks its own
  detections. Measured on this pipeline's frames, most detections came back at
  1.0 and the misreads sat *inside* that top group — a UI label read as
  "SUBSCRIDERS" at maximum confidence. A high read score may raise what an item
  is allowed to assert; it may never become the item's confidence. The two are
  computed from different inputs and never copied between them.
* **Confidence is not importance.** A model name on screen may be read at full
  confidence and be worth little; one price the speaker returns to three times is
  worth a lot on a frame that read at 0.7. `importance` is a separate field with
  a separate rule.
* **A raw observation is not an item.** Every OCR string stays in
  `visual[].ocr_text` and `ocr_regions` regardless of what this module concludes.
  Each classification records its reason, so "the extractor decided this was
  chrome" is auditable and reversible — a wrong decision then costs a derived
  item, never the observation underneath it.

Runs **per segment**, over that segment's snippets and the frames aligned to
them, so the working set is bounded by segment length rather than video length.
The model boundary is a seam, not an implementation detail: a segment is the unit
a reader would check a claim against, and it is the unit that fits in a context
window on a long video.

The deterministic extractor below is the tested, offline, default path. A model
backend is a drop-in behind `SemanticModel`; the invariants — provenance present,
confidence in range, no item from an empty read — are enforced by the validator
and hold whichever produced the item.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from cache import JsonCache, content_key
import lang_resources
from evidence_model import (
    SCHEMA_VERSION, Conflict, Evidence, Item, Segment, Snippet, VisualFrame,
)
from evidence_provenance import (
    ocr_provenance, transcript_provenance, vlm_provenance,
)
from ocr import MEDIUM_CONFIDENCE, normalize_text

#: What this module produces from a segment. Bump it when a change here would
#: make a re-run differ — ids, item shapes, classification, the subject rules.
#: Imported from `versions` so every version is readable in one file, and
#: re-exported here because this is the name the rest of the pipeline and its
#: tests use. The history of what each bump was for lives beside the constant.
from versions import EXTRACTION_VERSION  # noqa: E402

# --------------------------------------------------------------------------- #
# observation classes
# --------------------------------------------------------------------------- #

#: The seven ways a piece of visible text can fail to be evidence. Recorded per
#: observation, not used to filter: every class below still leaves its observation
#: in `visual[]`, and only `meaningful` is a precondition for an item.
OBSERVATION_CLASSES = (
    "meaningful",      # carries information about the video's subject
    "ui-chrome",       # counters, nav, timestamps, badges — present on every frame
    "caption",         # burned-in subtitles duplicating the speech track
    "decorative",      # headings, watermarks, section furniture
    "unreadable",      # the read was too unsure to act on
    "irrelevant",      # a legible number with no referent we can establish
    "ambiguous",       # several readings fit; not resolved either way
    "vlm-refused",     # a model reading was discarded, and the reason is recorded
)

#: Words that mark a region as interface furniture rather than content. Kept as
#: a small, inspectable list and recorded in `notes` whenever it fires, so the
#: judgement is visible rather than buried in a score. It is a *hint that raises
#: suspicion*, never a filter: a region on this list that the transcript supports
#: still becomes evidence, because a spoken number genuinely does appear next to
#: interface text on a screen recording.
CHROME_HINTS = frozenset({
    "subscribe", "subscribers", "subscribed", "notifications", "show more",
    "show less", "read more", "settings", "share", "copy link", "download",
    "search", "menu", "home", "profile", "dashboard", "overview", "inbox",
    "comments", "likes", "views", "views na", "log in", "sign in", "edit",
    "follow", "sort by", "filter", "next", "previous", "back", "save",
    "loading", "error", "settings", "help", "more", "less", "open",
})

#: Section furniture: a heading names a topic but asserts nothing about it.
DECORATIVE_HINTS = frozenset({
    "overview", "introduction", "summary", "conclusion", "takeaways",
    "agenda", "contents", "thank you", "subscribe", "credits", "chapter",
    "demo", "example", "notes", "references", "resources", "links", "further reading",
})

#: A UI percentage that is structurally a placeholder rather than a measurement.
#: Recognised structurally (a value of exactly zero beside a delta-looking
#: neighbour), not as a banned string — `$0.00` on a pricing page is a real price.
_ZERO_PLACEHOLDER = re.compile(r"^[-+•·]?\s*0\.0+%")


# --------------------------------------------------------------------------- #
# observation classification
# --------------------------------------------------------------------------- #

@dataclass
class Observation:
    """One piece of visible text, with everything decided about it so far."""

    frame_id: str
    text: str
    bbox: tuple[float, float, float, float] | None = None
    ocr_confidence: float | None = None
    line_label: str | None = None   # text to the left on the same visual line
    classification: str = "ambiguous"
    reason: str = ""
    values: list[tuple[str, str, str | None]] = field(default_factory=list)
    # (value, display, unit)
    #: The expression these values were read out of, verbatim (`28 × 28 = 784`).
    #: Set only when the frame showed them as one grouped run; it is what the
    #: item records so the relationship a reader saw is not lost, and it is never
    #: parsed into an equation this pipeline could not itself verify.
    structure: str | None = None

    def to_dict(self) -> dict:
        d = {"frame_id": self.frame_id, "text": self.text[:120],
             "classification": self.classification, "reason": self.reason}
        if self.bbox:
            d["bbox"] = [round(v, 1) for v in self.bbox]
        if self.ocr_confidence is not None:
            d["ocr_confidence"] = self.ocr_confidence
        if self.line_label:
            d["line_label"] = self.line_label
        if self.structure:
            d["structure"] = self.structure
        return d


def _y_center(bbox: Sequence) -> float:
    _, y, _, h = bbox
    return y + h / 2.0


def _y_overlap(a: Sequence, b: Sequence) -> float:
    """Fraction of the shorter box's height that the two boxes share.

    Decides "these two regions are on the same visual line". Geometry rather
    than adjacency, because adjacency in a flat region list is reading order of
    the engine's detection pass, which is not a claim about the layout.
    """
    _, ay, _, ah = a
    _, by, _, bh = b
    lo = max(ay, by)
    hi = min(ay + ah, by + bh)
    if hi <= lo:
        return 0.0
    return (hi - lo) / max(1e-6, min(ah, bh))


def line_groups(regions: list[dict]) -> list[list[dict]]:
    """Group regions into visual lines by vertical overlap.

    This is the one piece of layout the extractor reconstructs, and it is what
    turns `700` into `700 users`. Without it, a value is a bare glyph whose
    referent is a guess; with it, the label is the text physically beside the
    number on the same row, which is a reading of the frame rather than a
    pattern applied to a string.
    """
    if not regions:
        return []
    ordered = sorted(regions, key=lambda r: (_y_center(r["bbox"]), r["bbox"][0]))
    groups: list[list[dict]] = []
    for r in ordered:
        placed = False
        for g in groups:
            if any(_y_overlap(r["bbox"], m["bbox"]) >= 0.5 for m in g):
                g.append(r)
                placed = True
                break
        if not placed:
            groups.append([r])
    for g in groups:
        g.sort(key=lambda r: r["bbox"][0])
    return groups


def _looks_chrome(text: str) -> bool:
    t = normalize_text(text).lower()
    if not t:
        return True
    return any(w in t.split() for w in CHROME_HINTS) or t in CHROME_HINTS


def _looks_decorative(text: str) -> bool:
    t = normalize_text(text).lower()
    return t in DECORATIVE_HINTS


def classify_observation(obs: Observation, speech: str) -> None:
    """Decide what an observation is, in place, and say why.

    Order matters and is the whole design. The three questions are asked in
    increasing order of how much the *context* has to be trusted:

    1. Is the read usable at all? (the engine's own band)
    2. Does the speech already say this? (cross-modal — a hit makes it
       corroboration rather than a stray glyph)
    3. Failing that, does the fragment look like interface or furniture?

    Chrome is checked last on purpose. A number on screen *and* in the speech is
    the video's own claim shown twice, and demoting it because it happened to sit
    beside a "Subscribe" button would discard the single most reliable kind of
    screen evidence there is.
    """
    text = normalize_text(obs.text)
    if not text:
        obs.classification, obs.reason = "unreadable", "empty read"
        return
    if obs.ocr_confidence is None:
        obs.classification, obs.reason = "unreadable", "read carried no confidence"
        return
    if obs.ocr_confidence < MEDIUM_CONFIDENCE:
        obs.classification = "unreadable"
        obs.reason = f"read at {obs.ocr_confidence}, below the {MEDIUM_CONFIDENCE} usable band"
        return

    in_speech = _in_speech(text, speech)
    if obs.values and value_in_speech(obs.values, speech):
        # A figure counts as spoken when the number and its unit both appear,
        # however the caption happened to space them. Not as a substring: the
        # on-screen `0.09` inside a larger figure must not match.
        in_speech = True
    has_value = bool(obs.values)
    chrome = _looks_chrome(text)
    decorative = _looks_decorative(text)

    if not has_value and not in_speech and chrome:
        obs.classification, obs.reason = "ui-chrome", "interface vocabulary with no spoken support"
        return
    if not has_value and not in_speech and decorative:
        obs.classification, obs.reason = "decorative", "section furniture, asserts nothing"
        return
    if in_speech and not has_value:
        # The frame is showing the caption track. Not a separate fact; it tells us
        # the frame is on-topic, which the alignment already records.
        obs.classification, obs.reason = "caption", "duplicates the spoken transcript"
        return
    if not obs.values:
        obs.classification, obs.reason = "ambiguous", "no value and no spoken support to act on"
        return

    if _ZERO_PLACEHOLDER.match(text.strip()):
        obs.classification = "ui-chrome"
        obs.reason = "zero placeholder beside a metric, not a measured result"
        return
    if in_speech:
        obs.classification = "meaningful"
        obs.reason = "value also present in the aligned speech"
        return
    if obs.line_label:
        obs.classification = "meaningful"
        obs.reason = f"labelled on its own line as {obs.line_label!r}"
        return
    if obs.structure:
        # Grouped on the frame: the operator between them is the evidence, and
        # the fact that a reader can see `28 × 28 = 784` as one thing is the only
        # support a unitless figure on an equation slide ever has.
        obs.classification = "meaningful"
        obs.reason = (f"part of the expression {obs.structure!r} as displayed")
        return
    if any(u in (obs.text or "") for u in ("http://", "https://", "www.")):
        obs.classification = "meaningful"
        obs.reason = "URL, self-identifying without a label"
        return
    if chrome:
        obs.classification, obs.reason = "ui-chrome", "interface vocabulary, value unlabelled"
        return
    obs.classification = "irrelevant"
    obs.reason = "value with no label on its line and no spoken support"


def _in_speech(text: str, speech: str) -> bool:
    """Does this exact fragment appear in what is being said nearby?

    Normalised on both sides, which cannot equate `$500` with `$5,000` or `7B`
    with `70B` — the normalisation collapses whitespace and case only.
    """
    if not text or not speech:
        return False
    hay = normalize_text(speech)
    needle = normalize_text(text)
    return needle in hay


# --------------------------------------------------------------------------- #
# number relevance
# --------------------------------------------------------------------------- #

#: Vocabulary that makes a label a *measurement*. A label naming one of these is
#: what turns `700` into `700 users`, and its absence is why an unlabelled
#: figure has no referent worth printing.
#:
#: The test is whether the word answers "what is being counted or measured?" —
#: not whether it appears in a technical video. `user`, `latency` and `price` do;
#: `model` does not, because "model" is a category of thing, not a quantity, and
#: treating it as one is how "127 — MODEL" from a comparison table's header row
#: became a figure with a subject.
MEASURE_VOCAB = frozenset({
    "users", "user", "members", "customers", "subscribers", "downloads", "installs",
    "requests", "queries", "tokens", "params", "parameters", "context", "batch",
    "size", "length", "duration", "latency", "throughput", "speed", "rate",
    "fps", "resolution", "bitrate", "accuracy", "precision", "recall", "score",
    "benchmark", "baseline", "memory", "ram", "vram", "disk", "storage", "bandwidth",
    "power", "watt", "temperature", "weight", "height", "width", "depth",
    "price", "cost", "fee", "plan", "subscription", "license", "budget",
    "seconds", "second", "minutes", "minute", "hours", "hour", "days", "day",
    "weeks", "week", "months", "month", "years", "year", "version", "release",
    "count", "total", "capacity", "limit", "quota", "budget", "coverage",
    "gb", "mb", "tb", "kb", "ms", "usd", "eur", "gbps", "mbps", "tflops",
    "percent", "percentage", "ratio", "share", "growth", "margin", "error",
})

#: Vocabulary that only ever describes the *platform interface* a figure is
#: printed on. Deliberately narrow: a word earns a place here only if it names
#: furniture and nothing a measurement could also be called. `queue`, `duration`,
#: `position` and `level` are absent for exactly that reason — a benchmark run has
#: a duration and a server has a queue, and dropping `412 — queue depth` because
#: the word is on a list is the same class of error as dropping it because the
#: string is on a list. Those cases are decided by the table-and-topic rule and by
#: the no-support branch instead, which are evidence rather than vocabulary.
#:
#: `resolution` is absent for the same reason and the opposite way: it names a
#: real property of a video file. The chrome sense — `1080p` on a player — is
#: caught by `quality`, `settings` and `auto` sitting beside it.
UI_LABEL_VOCAB = frozenset({
    "chapter", "episode", "playlist", "sidebar", "toolbar", "menu", "button",
    "comment", "comments", "reply", "replies", "like", "likes", "dislike",
    "share", "shares", "view", "views", "subscriber", "subscribers", "follow",
    "followers", "notification", "notifications", "bookmark", "saved",
    "progress", "timestamp", "volume", "quality", "auto", "hd", "sd", "4k",
    "settings", "options", "sort", "filter", "search", "next", "previous",
    "loading", "subscribed", "unsubscribe", "tab", "panel", "window", "screen",
    "of", "total", "phase", "part", "step", "steps",
    # Present in the observation layer's `CHROME_HINTS` and missing here, which
    # made a figure labelled `2026 — inbox` reach the numbers table while a
    # figure labelled `173K — subscribers` did not. Two lists naming the same
    # interface is how they drift; this is the larger of the two.
    "inbox", "overview", "dashboard", "read", "login", "signin", "profile",
    "activity", "replies", "history", "drafts", "trash", "archive", "spam",
})

#: A label that is itself an ordinal or a section marker rather than a name.
#: `02 — PHASE TWO` and `21 — Browse all skills` both land here: the number is
#: the *index* of a thing, and the thing is the report's content, not a figure.
_ORDINAL_LABEL = re.compile(
    r"^(?:phase|chapter|part|step|stage|level|section|episode|track|slide|"
    r"page|apartment)\s+\w+$|^\d+$", re.IGNORECASE)

#: A label that is a table header or a two-letter column code, not a referent.
_CELL_LABEL = re.compile(r"^[A-Z]{1,3}$|^\W+$")

#: Vocabulary that makes a *segment* a segment about measurements. This is a
#: statement about the segment's topic, not about the number — which is the
#: difference between this and a blacklist. "A 2×2 grid of 21 — Browse all
#: skills" is also a labelled table, and the only thing that separates it from a
#: benchmark table is what the speaker is doing while the table is on screen.
BENCHMARK_TOPIC = frozenset({
    "benchmark", "benchmarks", "chart", "charts", "graph", "graphs", "plot",
    "axis", "axes", "bar", "bars", "results", "ranking", "leaderboard",
    "comparison", "comparisons", "compare", "compares", "compared",
    "comparing", "faster", "slower", "throughput", "latency", "accuracy",
    "score", "scores", "measured", "measuring", "evaluation", "eval",
})

#: A wider topic set that still means "this segment is showing measurements",
#: without committing to a comparison.
MEASUREMENT_TOPIC = BENCHMARK_TOPIC | frozenset({
    "price", "prices", "pricing", "cost", "costs", "costing", "subscription",
    "plan", "plans", "tier", "tiers", "budget", "spec", "specs",
    "specification", "specifications", "configuration", "configurations",
    "limits", "capacity", "throughput", "measured", "measurement", "measurements",
    "metrics", "usage", "capacity", "scaling", "quantized", "quantised",
})


def _topic_vocabulary(speech: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (speech or "").lower()))


#: A cell that is a clock time — `12:34`, `1:02:30`. A timestamp is a position
#: in the video, not a measurement of anything, and a grid of them is a chapter
#: list or a player scrubber rather than a results table. Recognised from its
#: shape, because that is what distinguishes it: a ratio, a version and a clock
#: time are the same characters in a different arrangement.
_CLOCK_TIME = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")

#: A cell holding nothing but a figure: `72.4`, `1,024`, `3.2.1`. This is the
#: shape a real results table is made of, and `find_values` deliberately refuses
#: a bare figure — on a screen it is more often a list index or a column
#: position, and the *label* is what vouches for it. Which is right in
#: isolation and wrong in a table: inside a table the geometry already says
#: "this cell is a value", and requiring a unit as well meant no table of bare
#: numbers was ever recognised as a table at all.
_BARE_CELL = re.compile(r"^[\d][\d.,:]*$")

#: A cell holding a short whole number — the only thing a gutter is made of.
_GUTTER_CELL = re.compile(r"^\d{1,4}$")


def _is_value_cell(text: str) -> bool:
    """Is this one region a cell that holds a figure rather than a name?"""
    t = (text or "").strip()
    if not t or _CLOCK_TIME.match(t):
        return False
    return bool(find_values(t)) or bool(_BARE_CELL.match(t))


def _in_table_row(frame: "VisualFrame") -> bool:
    """Does this frame lay its text out as a table of label/value pairs?

    Read from geometry, not from the string: two or more visual lines, each
    holding exactly one region that parses as a figure and at least one that
    does not. That is what a results table is, and it is the same evidence a
    reader uses when they look at the frame and say "that is a table".

    Geometry rather than a pattern because a list of end-screen cards is also a
    grid of a number and a label, and the two are only distinguishable by what
    the segment is about — which is why `classify_number_role` requires the topic
    signal before this can make anything a benchmark.
    """
    regions = list(frame.ocr_regions or [])
    if len(regions) < 4:
        return False
    rows = 0
    for group in line_groups(regions):
        values = [r for r in group if _is_value_cell(r.get("text") or "")]
        labels = [r for r in group if r not in values]
        if values and labels:
            rows += 1
    return rows >= 2


def _line_number_gutter(frame: "VisualFrame") -> bool:
    """Is this frame a code editor showing line numbers down its left edge?

    Read from the shape of the column, not from the words beside it. A gutter is
    the leftmost region on each of several lines, every one of them a small
    whole number, running upward in near-consecutive steps, with prose or code
    in the same rows. That is a structure — no filename, language or extension
    needs naming, which is the point: a list of extensions is a list of the
    wrong ones by the time anyone adds a build tool.

    The near-consecutive step is what separates it from a genuine first column
    of a table. A results table's first column holds *names*, and the one place
    a table's numbers sit in a rising run is a sequence, which is not a
    measurement either.
    """
    groups = [g for g in line_groups(list(frame.ocr_regions or []))
              if len(g) >= 2]
    if len(groups) < 4:
        return False
    column: list[int] = []
    for group in groups:
        left = min(group, key=lambda r: (r.get("bbox") or [0])[0])
        m = _GUTTER_CELL.match((left.get("text") or "").strip())
        if not m:
            return False
        column.append(int(m.group(0)))
    steps = [b - a for a, b in zip(column, column[1:])]
    return bool(steps) and all(0 < s <= 3 for s in steps)


def _unit_role(value: str, unit: str | None, display: str) -> str | None:
    """The role the *form* of a figure asserts, when the form is decisive.

    Money is a price; a duration, a size or a rate is a metric. This is not a
    blacklist of bad strings — it is a reading of what the writer of the number
    committed to by writing a currency mark or a unit at all. A bare `0.09` with
    no unit and no label commits to nothing, which is why it falls through to
    `unknown` rather than being guessed into a metric.
    """
    text = f"{display or value}{unit or ''}".strip().lower()
    if any(c in text for c in "$€£₹"):
        return "price"
    if unit and re.search(r"\b(?:gb|mb|tb|kb|ms|us|hz|w|kw|ghz|mhz|ghz|k|v|a)\b", unit.lower()):
        return "metric"
    if re.search(r"%|percent|percentage|ratio|\d+\s*x\b|fps|hz", text):
        return "metric"
    return None


def classify_number_role(obs: "Observation", value: str, unit: str | None,
                         display: str, frame: "VisualFrame",
                         ctx: "SegmentContext") -> tuple[str, str, str]:
    """What a figure is *for*, and how much a reader needs it. `(role, importance, reason)`.

    The gate between an OCR string and a report-visible number. It reads six
    independent signals rather than a list of banned strings, because a banned
    string only knows the numbers that were already wrong:

    1. **Cross-modal support.** The value and its unit also appear in the aligned
       speech. This is the strongest support there is — the figure is the video's
       own claim, shown twice — and it outranks everything below it.
    2. **Label semantics.** Does the label name a *measure* (`users`, `latency`,
       `$`), interface furniture (`chapter`, `views`), or neither?
    3. **Label shape.** An ordinal, a table header, or a two-letter column code
       is furniture whatever vocabulary it happens to contain.
    4. **Unit shape.** Currency, duration, size, rate.
    5. **Table structure plus segment topic.** A labelled row in a table, in a
       segment whose speech is about measurements, is a benchmark. Both halves
       are required: a grid of numbers with labels beside them is also what an
       end screen of suggested videos looks like, and the speaker's subject is
       the only thing that tells the two apart.
    6. **Segment topic and claim relationship.** A figure a claim in this
       segment is *about* is central by construction.
    7. **The frame's own classification.** A value the observation layer already
       called interface furniture is not re-litigated here.

    Importance is never derived from `ocr_confidence`. A subscriber counter read
    at 0.99 is `ui`; a benchmark bar read at 0.78 is `benchmark`. Those are the
    two cases that make a confidence-derived importance look broken to anyone who
    compares them side by side, so the field is computed from context only and
    the reason string says which signal decided it.
    """
    label = (obs.line_label or "").strip()
    cross_modal = value_in_speech(obs.values, ctx.speech)
    why: list[str] = []

    # (1) The video said this number. Nothing below can demote it: a spoken
    # figure that happens to sit beside a "Subscribe" button is still the point.
    if cross_modal:
        role = _unit_role(value, unit, display) or (
            "price" if _looks_money(label) else "metric" if _measure_in(label) else "count")
        return role, "high", (f"the value and its unit are also in the aligned speech, "
                              f"and the label names a {'measure' if role != 'price' else 'price'}")

    # (6) The observation layer already decided this is furniture. Honouring it
    # here rather than re-deciding is why the two layers cannot disagree.
    if obs.classification == "ui-chrome":
        return "ui", "low", f"the observation layer classified the text as interface ({obs.reason})"

    # (8) A grouped expression. The frame wrote the figures as one structure, so
    # the structure is the referent: there is no label to read and no unit to
    # fall back on, and a unitless figure with both is a figure a reader can use.
    # `high` only where the frame is a slide that explains itself — a caption
    # track and a code editor both produce grouped digits, and prose beside the
    # expression is what tells those apart.
    if obs.structure:
        explained = _frame_explains(frame)
        return "metric", ("high" if explained else "medium"), (
            f"one of {len(obs.values)} figures in the expression {obs.structure!r}, which the "
            f"frame displays as a single {'annotated' if explained else 'unannotated'} structure")

    # (3) Label shape. Checked before label vocabulary because "PHASE TWO" and
    # "MODEL" both contain a word that appears in a technical sentence, and both
    # are furniture — a word-shape test is what they fail and a vocabulary test is
    # what they pass.
    if label and (_ORDINAL_LABEL.match(label) or _CELL_LABEL.match(label)):
        return "ui", "low", f"the label {label!r} is an ordinal or a table cell, not a referent"

    # (2) Label vocabulary.
    if label and _ui_vocabulary(label):
        return "ui", "low", f"the label {label!r} names interface furniture, not a quantity"
    if label and _measure_in(label):
        role = _unit_role(value, unit, display) or "metric"
        return role, "medium", f"the label {label!r} names a quantity that can be measured"

    # (4) Unit shape with no label to explain it.
    role = _unit_role(value, unit, display)
    if role is not None:
        # A unit says *what kind* of number this is and not *what about*, and
        # `4% — outoute` is the shape that proves it: the `%` is real, the label
        # is a misread, and together they produce a figure a reader cannot use.
        # Low, not medium — the role is still recorded, because it is the honest
        # description of the reading, but a role is not a licence to print it.
        # A figure the speaker also said is caught by the cross-modal branch
        # above and keeps its `high`, which is where this case belongs.
        return role, "low", (f"the unit makes this a {role}, but no label says what it "
                             "measures, so nothing here says what it counts")

    # (5) Table structure plus segment topic — reached only when nothing above
    # explained the figure. The pair is required, never either alone: a grid of
    # numbers with labels beside them is what an end screen of suggested videos
    # also looks like, and the speaker's subject is the only thing separating the
    # two. But the topic is a property of the *segment*, so this has to be the
    # last resort rather than an early one: a dashboard column of unrelated
    # percentages sitting in a segment that happens to discuss numbers would
    # otherwise have every one of them promoted to a high-importance benchmark.
    # What this catches is the case nothing else can — a figure with no unit and
    # no measure-word in its label, sitting in a real table, in a segment about
    # measurements. That is the ordinary shape of a benchmark table: `81.9` beside
    # `Qwen`, a decimal and a name, and nothing else.
    #
    # The gutter exclusion is the one shape that is a table and is not one: a
    # code editor is a grid of a number and prose, and a screen-recording of a
    # benchmark discussion is a segment about benchmarks. Recognising the gutter
    # structurally is what keeps the two apart.
    topic = _topic_vocabulary(ctx.speech)
    if (label and (topic & MEASUREMENT_TOPIC) and _in_table_row(frame)
            and not _line_number_gutter(frame)):
        money = {"price", "prices", "pricing", "cost", "costs", "subscription",
                 "plan", "plans", "tier", "tiers", "budget"}
        if _unit_role(value, unit, display) == "price" or (topic & money):
            role = "price"
        elif topic & BENCHMARK_TOPIC:
            role = "benchmark"
        else:
            role = "metric"
        return role, "high", f"a labelled row in a table, in a segment about {role}s"

    if unit:
        return "unknown", "low", f"unit {unit!r} present but unrecognised and unlabelled"

    # (6) A claim in this segment, if the figure belongs to one.
    if _figure_in_claim(ctx.speech, value, unit):
        return "metric", "high", ("a claim in this segment is about this figure, so it is "
                                  "the measurement the claim rests on")

    return "unknown", "low", ("read, but nothing in the frame, the label or the speech says "
                              "what it counts")


def _frame_explains(frame: "VisualFrame") -> bool:
    """Does the frame carry prose of its own, beside the figures?

    A slide that writes `28 × 28 = 784` also writes something about it; a caption
    track and a terminal both group digits without explaining them. Read from the
    frame's own regions — three or more words, and not the caption the transcript
    already holds — so the difference is one a reader can see rather than one this
    layer infers from the segment's topic.
    """
    for r in (frame.ocr_regions or []):
        text = (r.get("text") or "").strip()
        if len(text.split()) >= 3 and not find_values(text):
            return True
    return False


def _looks_money(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t.split() for w in ("price", "pricing", "cost", "fee", "plan",
                                         "subscription", "per", "month", "year", "seat"))
def _measure_in(text: str) -> bool:
    t = (text or "").lower()
    words = set(re.findall(r"[a-z]+", t))
    return bool(words & MEASURE_VOCAB) or "%" in t or any(c in t for c in "$€£₹")


def _ui_vocabulary(text: str) -> bool:
    words = set(re.findall(r"[a-z]+", (text or "").lower()))
    # Furniture wins a tie, and the ties are real: `subscribers`, `views`,
    # `duration` and `resolution` are words a measurement can legitimately use.
    # A YouTube channel has subscribers and a benchmark run has a duration, and
    # the only thing that tells them apart is the furniture the number is printed
    # on — so the word that names the interface decides, and the word that names
    # the quantity does not rescue it.
    return bool(words & UI_LABEL_VOCAB)


def _figure_in_claim(speech: str, value: str, unit: str | None) -> bool:
    """Does a claim in this segment carry this same figure?

    A claim is a sentence that states something about a subject, and a figure
    inside one is the evidence for it. That relationship — not the string, not
    the frame, not the confidence — is what makes a number central.
    """
    for sent in sentences(speech or ""):
        found = find_claim(sent)
        if not found:
            continue
        if any(v == value and u == unit for v, _, u in find_values(sent)):
            return True
    return False


_CURRENCY = "$€£"


def value_pattern(value: str, unit: str | None) -> re.Pattern | None:
    """A pattern matching a figure the way a transcript and a screen disagree.

    Captions and ASR render the same quantity differently — `32GB`, `32 GB`,
    "32 gigs" — so cross-modal matching cannot be a string comparison or the one
    corroboration case the whole layer exists for never fires. It also cannot be
    a whitespace-stripped comparison either: that equates `1 2GB` with `12GB`, so
    the tolerance is placed *between* the digits and the unit and nowhere else.
    Digits and punctuation stay exact.
    """
    if not value:
        return None
    v = re.escape(value)
    if not unit:
        return re.compile(rf"(?<![\w.]){v}(?![\w])", re.IGNORECASE)
    u = re.escape(unit)
    if unit[0] in _CURRENCY:
        return re.compile(rf"(?<![\w.]){u}\s*{v}(?![\w])", re.IGNORECASE)
    return re.compile(rf"(?<![\w.]){v}\s*{u}(?![\w])", re.IGNORECASE)


def value_in_speech(values: list[tuple[str, str, str | None]], speech: str) -> bool:
    """True when any figure this observation holds is also being spoken."""
    if not speech:
        return False
    for value, _display, unit in values:
        rx = value_pattern(value, unit)
        if rx is not None and rx.search(speech):
            return True
    return False


# --------------------------------------------------------------------------- #
# values
# --------------------------------------------------------------------------- #

# Reused from the transcript extractor rather than re-declared: a hedge word, a
# unit list and a currency handling that exist in two places are two answers to
# "is this a figure, and is it hedged". The semantic layer is the only place that
# decides what a figure *means*, so parsing one is still its job — but the parse
# must agree with the one that produced the transcript items.
from build_evidence import NUM_RE, URL_RE, hedge_before  # noqa: E402

_LABEL_BEFORE_VALUE = re.compile(
    r"^(?P<label>[A-Za-z][A-Za-z0-9 ._/+-]{1,48}?)\s*[:=]?\s*(?=[$€£]?\d)")

#: Auxiliaries and prepositions that end a label phrase. "The card has 32 GB" is
#: about the card; taking the label as "The card has" would file every figure in
#: the document under a sentence fragment instead of the thing it measures.
_LABEL_TAIL = re.compile(
    r"\s+(?:has|have|had|is|are|was|were|be|shows?|show|shows|costs?|costs|"
    r"needs?|need|holds?|hold|capacity|of|for|with|at|only|just|up|down)$", re.IGNORECASE)
_LABEL_HEAD = re.compile(r"^(?:the|a|an|this|that|its|their|his|her|your|our|my)\s+", re.IGNORECASE)

#: A lone figure, used only where a label has already vouched for it.
_BARE_NUMBER = re.compile(r"(?<![\w.$€£])\d[\d,]*(?![\w])")

#: An operator that makes a run of digits one visual *expression* rather than a
#: list of figures. Deliberately excludes a bare `-`: a date, a version and a
#: range all use it, and a minus sign is the one operator whose absence in a
#: maths frame is more common than its presence. `x` counts only between two
#: digits, so "2x faster" stays a multiplier and "28 x 28" becomes an operator.
_EXPR_OP = re.compile(r"[×✕✖*+=→⨯]|<-|->|/|(?<=\d)\s?[xX]\s?(?=\d)")

#: A formula starts with a figure and an operator, not with an operator. `+ Wo, 1 a1`
#: and `(0) + 60)` are matrix headers an OCR pass read with a plus in them, and a
#: rule that only asked "is there an operator somewhere" promoted both. The shape
#: test is what separates a formula the frame laid out from punctuation that
#: landed in the middle of a misread subscript.
_EXPR_SHAPE = re.compile(
    r"^[$€£]?\s*\d[\d.,]*\s*(?:[×✕✖*+=→⨯/]|\s?[xX]\s?(?=\d))")

#: `1.2.3`, `v2.1` — the same characters as an expression, the opposite meaning.
_EXPR_VERSION = re.compile(r"^v?\d+(?:\.\d+){1,3}$")

#: A figure with its referent *after* it: `784 inputs`, `784 weights per neuron`,
#: `10 outputs`. A caption track and a diagram legend both write it this way, and
#: `label_for_value` — which reads the words to the left — recovered none of it,
#: which is how the node counts on a network diagram went unrecorded while the
#: digits a reader would call noise were recorded with an em-dash for a subject.
_TRAILING_LABEL = re.compile(
    r"^(?P<value>[$€£]?\d[\d,.]*)\s+(?P<label>[^\W\d_][\w.\-/]*(?:\s+[\w.\-/]+){0,4})\s*$",
    re.UNICODE)

#: A matrix subscript read as a word: `O1`, `a4`, `W1`.
_SUBSCRIPT_LABEL = re.compile(r"[A-Za-z]\d{1,3}")

#: Text that is only figures and the punctuation around them. Two such regions
#: side by side are two values, not a label and a value: a version strip reading
#: `2021 2022 2023` otherwise produces one item per year, each "labelled" by the
#: year before it, and a row of counters reading `3,789 408 218` produces three
#: items whose subjects are each other.
_NUMERIC_ONLY = re.compile(r"^[\d\s.,:;%$€£/\\()\[\]+*#-]*$")

#: How long a region may be before it stops being a labelled row and becomes
#: prose. Measured on the shipped artifacts, table rows read as 2–4 words and
#: headlines as 8+; the gap between those is where the two stop being confused.
_LABEL_MAX_WORDS = 5

#: A definition's value is a term, not a sentence. 90 characters is roughly a
#: noun phrase; anything longer is the clause that followed the "is".
_DEFINITION_MAX_VALUE = 90

#: Subjects that cannot define anything: a definition is about a *thing*, and
#: "This is going to be all about showing you…" defines no thing at all.
_DEFINITION_NO_SUBJECT = frozenset({
    "this", "that", "it", "these", "those", "they", "we", "you", "i", "he", "she",
    "here", "there", "everything", "anything", "something", "nothing", "one",
})


def find_values(text: str) -> list[tuple[str, str, str | None]]:
    """(value, display, unit) triples in one piece of visible text.

    A bare number with no unit and no currency is refused here, exactly as the
    transcript pass refuses it, and for the same reason: on a screen it is far
    more often a list index, a column position or a year than a measurement, and
    `700` next to `users` is recovered by the *label*, not by loosening the unit
    list. A labelled bare number is recovered by `label_for_value` below instead.
    """
    out: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()
    for m in NUM_RE.finditer(text):
        raw, unit, cur = m.group("value"), m.group("unit"), m.group("cur")
        if not unit and not cur:
            continue
        display = f"{cur}{raw}" if cur else f"{raw}{unit}"
        key = re.sub(r"\s+", " ", display)
        if key in seen:
            continue
        seen.add(key)
        out.append((raw.replace(",", ""), display, cur or unit))
    for url in URL_RE.findall(text):
        if url in seen:
            continue
        seen.add(url)
        out.append((url, url, None))
    return out


def find_expression_values(text: str) -> list[tuple[str, str, str | None]] | None:
    """The figures inside one visual expression, or `None` if this is not one.

    `28 × 28 = 784` is a single region holding three figures that the reader reads
    as one relationship, and `find_values` refuses all three: no unit, no
    currency, so nothing survives the parse. That refusal is correct in prose and
    wrong here, because on a frame the grouping is *visible* — one region, one
    baseline, one operator between the digits. The structure is the evidence, and
    a layer that only recognises unit-bearing figures cannot see it.

    Nothing is inferred beyond what is written. The figures are returned as they
    are read, in the order they appear; whether `28 × 28` really is `784` is the
    speaker's or the slide's claim, and this only reports that the three sit in
    one expression. Repeated operands collapse to one value — `28 × 28` is two
    occurrences of one figure, and a document that lists it twice is a document
    reporting a count of two where the screen shows one number twice — while the
    expression itself is carried verbatim on each item, so the repetition and the
    position are both still there. A clock time, a version and a lone figure are
    all refused: each is a run of digits that is not an expression.
    """
    t = (text or "").strip()
    if not t or _CLOCK_TIME.match(t) or _EXPR_VERSION.match(t):
        return None
    if not _EXPR_OP.search(t) or not _EXPR_SHAPE.match(t):
        return None
    raw_values = [m.group(0) for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", t)]
    if len(raw_values) < 2:
        return None
    out: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()
    for raw in raw_values:
        key = raw.replace(",", "")
        if key in seen:
            continue
        seen.add(key)
        out.append((key, raw, None))
    return out


def trailing_label(text: str) -> tuple[str, str, str | None] | None:
    """`(value, display, subject)` for a figure whose label follows it.

    The mirror of `label_for_value`, and it exists for the same reason that one
    does: a referent is a referent whichever side of the figure the writer put
    it on, and reading only the left-hand side is what made a legend and a
    diagram caption unreadable to this layer. The label is passed through the
    same `_clean_label` gate, so `784 02` and `10 •` are still refused.
    """
    m = _TRAILING_LABEL.match((text or "").strip())
    if not m:
        return None
    label = _clean_label(m.group("label"))
    if not label:
        return None
    display = m.group("value")
    return (display.replace(",", ""), display, label)


def _clean_label(raw: str) -> str | None:
    """Turn a candidate label into a subject, or reject it.

    One function for both sources of a label — the text before a figure inside a
    region, and the region to its left on the same row — because they were
    diverging, and the divergence showed up in the output: a figure read from
    its own region came back labelled "All", while the same figure read from the
    neighbouring region came back labelled "All for". The difference is an
    artefact of which path ran, and an artefact that changes a document's
    subjects is not one to leave in.

    Rejects, in order: nothing, figures-only (two values are not a label),
    interface punctuation, a trailing verb or preposition, a leading determiner
    left over after that, anything over three words, and any word that is on its
    own a discourse marker rather than a name.
    """
    label = (raw or "").strip(" .:;,;-•|")
    if not label or _NUMERIC_ONLY.match(label):
        return None
    if label[0] in ">•@/·*-=+|":
        return None
    # A glyph is not a name. `—3`, `—З` and `(0)` sit on the same visual line as
    # a figure in a maths animation and are what a subscript or a bullet reads
    # as; accepting one as the figure's referent produced items whose subject was
    # an em-dash and a misread digit, filed at high importance because a
    # grid-shaped frame plus a numeric segment satisfied the table-and-topic rule.
    if not re.search(r"[A-Za-z]", label):
        return None
    # One letter wrapped in anything is still one letter: `(O)` on a frame of
    # node labels is the node, not the name of what the figure beside it counts.
    if len(re.sub(r"[^A-Za-z]", "", label)) < 2:
        return None
    # A single letter with a digit is a matrix subscript, not a caption: `O1`,
    # `W1,0`, `a4`. The same rule as above, in the shape the subscripts take.
    if _SUBSCRIPT_LABEL.fullmatch(label):
        return None
    label = _LABEL_TAIL.sub("", label).strip(" .:-")
    label = _LABEL_HEAD.sub("", label).strip()
    if not label or len(label) < 3 or len(label.split()) > 3:
        return None
    if NUM_RE.fullmatch(label) or _NUMERIC_ONLY.match(label):
        return None
    if label.lower() in _ENTITY_STOPWORDS or label.lower() in _QUANTIFIER_TAIL:
        return None
    return label


def label_for_value(text: str) -> str | None:
    """The thing a figure is about, read from the words immediately before it.

    This is the path that recovers `700` as `700 users`, and `32 GB` as
    `The card`. It is a *parse of a labelled region*, not a filter: it runs after
    the observation has been judged meaningful, and a label only sets `subject` —
    it never, on its own, promotes something into evidence. A reader who
    disagrees with one of these labels can disagree with the `subject` field and
    nothing else in the item moves.
    """
    # A region long enough to be a sentence is prose, not a labelled row. Without
    # this, "GPT 6 Sol & Luna Are Here (And 50% CHEAPER!)" yields the label "GPT"
    # for the 50%, which is a fragment of a headline rather than a caption for
    # anything, and the item is then filed under a word that names nothing.
    if len(text.split()) > _LABEL_MAX_WORDS:
        return None
    # Matched against the original, not the normalised form: `normalize_text`
    # lowercases, and a label that comes back as "model a" has lost the two
    # characters that distinguish Model A from Model B.
    m = _LABEL_BEFORE_VALUE.match(text.strip())
    if not m:
        return None
    return _clean_label(m.group("label"))


# --------------------------------------------------------------------------- #
# hedging
# --------------------------------------------------------------------------- #

def hedge_for(text: str, at: int) -> str | None:
    """The hedge governing the character at `at`, or None.

    Delegates to the transcript extractor's detector so a hedge is detected the
    same way in both layers. Preserving it is the point: "could cost $500" and
    "costs $500" are different claims, and flattening the first into the second
    produces an item that passes every provenance check in the document and is
    still false.
    """
    return hedge_before(text, at)


# --------------------------------------------------------------------------- #
# semantic confidence
# --------------------------------------------------------------------------- #

@dataclass
class ConfidencePolicy:
    """How an extraction's own confidence is computed, and from what.

    Kept apart from `ocr.py`'s bands on purpose. The read band answers "how sure
    is the engine that these are the characters"; this answers "how much is the
    *interpretation* worth", which is a different question with different inputs:
    how many independent modalities back it, whether the value is labelled, and
    whether the speaker hedged. An item's confidence is never set to a frame's
    `ocr_confidence`, and the two are never averaged.

    The cross-modal term is a bounded bonus, not an addition of two scores. Two
    modalities agreeing genuinely does raise confidence — they are independent
    reads of the same fact — but "add 0.9 + 0.9" produces 1.8 and then gets
    clamped to 1.0, which is the same as saying full confidence, and would make
    every corroborated item look as certain as a metadata field.
    """

    base_transcript: float = 0.90
    base_labelled_screen: float = 0.78
    base_unlabelled_screen: float = 0.55
    #: Added when a second, independent modality backs the same value. Capped, so
    #: corroboration is a real but bounded effect.
    cross_modal_bonus: float = 0.08
    #: Subtracted when the source hedged. A hedge is not a less certain figure, it
    #: is a different claim about the figure, so this is applied to the
    #: confidence AND the item keeps `status: "hedged"`.
    hedge_penalty: float = 0.15
    #: Confidence of a model-derived observation, before corroboration. Set below
    #: every text-derived band: a caption is a description of pixels, and a
    #: description that is wrong is wrong in ways no score can bound.
    base_vlm: float = 0.60
    ceiling: float = 0.95

    def apply(self, base: float, *, cross_modal: bool = False, hedged: bool = False) -> float:
        c = base
        if cross_modal:
            c += self.cross_modal_bonus
        if hedged:
            c -= self.hedge_penalty
        return round(max(0.0, min(self.ceiling, c)), 4)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "base_transcript", "base_labelled_screen", "base_unlabelled_screen",
            "cross_modal_bonus", "hedge_penalty", "base_vlm", "ceiling")}


#: Importance is a third axis again — not how sure, but how much it matters to a
#: reader. Stated as a rule so it can be disagreed with, and never derived from
#: confidence: a fully-confident channel name is still not worth much.
IMPORTANCE_RULES = {
    "high": "a value the speaker asserted, a labelled screen value, or a claim",
    "medium": "an unlabelled screen value with a self-identifying form, or a "
              "spoken figure whose subject was not established",
    "low": "metadata and interface text retained for completeness",
}


@dataclass
class SemanticConfig:
    """What the extractor is allowed to do. Defaults are the shipped policy."""

    #: A step sequence needs at least this many candidate steps in one segment.
    #: The floor is what keeps "just run the tests" from becoming a one-step
    #: procedure — a procedure with one step is a sentence.
    min_steps_in_segment: int = 2
    #: Cap on items emitted from one segment, so a dense dashboard cannot
    #: dominate the document. The cap truncates in confidence order and the
    #: count of dropped candidates is reported, never silently applied.
    max_items_per_segment: int = 80
    enable_comparisons: bool = True
    enable_steps: bool = True
    enable_entities: bool = True
    enable_vlm: bool = True
    policy: ConfidencePolicy = field(default_factory=ConfidencePolicy)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("min_steps_in_segment", "max_items_per_segment", "enable_comparisons",
              "enable_steps", "enable_entities", "enable_vlm")}
        d["policy"] = self.policy.to_dict()
        return d


# --------------------------------------------------------------------------- #
# stable identity
# --------------------------------------------------------------------------- #

def _norm_value(value: str) -> str:
    """Identity form of a value.

    Numbers compare by value, so `32`, `32.0` and `32.00` are one item; strings
    compare after whitespace collapse only, because case and punctuation are part
    of a model name or a URL. Deliberately *not* a fuzzy match: "GPT-4" and
    "GPT-4o" are different models, and a deduplicator that merges them deletes the
    distinction the report exists to preserve.
    """
    v = re.sub(r"\s+", " ", value.strip())
    try:
        f = float(v.replace(",", ""))
    except ValueError:
        return v
    return repr(int(f)) if f.is_integer() else repr(f)


def item_identity(it: Item) -> tuple:
    """Semantic identity: what makes two items the same claim.

    Type, value, unit, qualifier and status. Provenance is excluded — two reads
    of the same claim are one item with two citations, which is the cross-modal
    case the schema is shaped for — and so is `subject`, because a subject is
    *derived*, differently on each side: "VRAM" from a screen row, "memory" from
    the sentence that mentions the card. Folding it into the key would leave the
    same figure filed twice under two labels, and an evidence count that cannot
    be reproduced is a count nobody should trust. `subjects_compatible` is the
    guard instead: it is checked before a merge, so "32 GB of VRAM" and "32 GB of
    memory" still stay apart.

    Status stays in the key. "could cost $500" and "costs $500" are the same
    number and different claims, and merging them would perform exactly the
    flattening the hedge field exists to prevent.
    """
    return (it.type, _norm_value(it.value),
            re.sub(r"\s+", " ", (it.unit or "").lower()),
            re.sub(r"\s+", " ", (it.qualifier or "").lower()),
            it.status)


def subjects_compatible(a: str | None, b: str | None) -> bool:
    """Can these two labels plausibly name the same thing?

    Word-bounded containment, either way, plus the case where one side has no
    label at all — an unlabelled screen reading is compatible with any subject and
    takes the spoken one, which is the whole point of the cross-modal merge.
    Containment rather than equality because the two sides genuinely differ in
    length: a UI row says `waitlist`, the sentence says `waitlist signups`, and
    requiring them to match exactly would never merge a corroborated figure.
    """
    if not a or not b:
        return True
    la = re.sub(r"\s+", " ", a.strip().lower())
    lb = re.sub(r"\s+", " ", b.strip().lower())
    if la == lb:
        return True
    # Tested against the *other* label, not against both concatenated: a search
    # over "Model A Model B" always finds one of them, which is how two distinct
    # benchmark rows came to be treated as one subject.
    return bool(re.search(rf"(?<![\w]){re.escape(la)}(?![\w])", lb)
                or re.search(rf"(?<![\w]){re.escape(lb)}(?![\w])", la))


def item_id_for(it: Item) -> str:
    """A stable id derived from semantic identity and subject.

    Hashed rather than sequential so that re-processing the same evidence yields
    the same ids and therefore does not duplicate, while a genuinely new claim
    still gets a fresh one. The `sem-` prefix keeps these distinguishable from the
    mechanical items the earlier pass emits, which is useful when reading the file
    and harmless to the validator.

    The subject is in the hash even though it is not in `item_identity`, because
    two figures that share a value and a unit but are kept apart by incompatible
    subjects — a document that says 32 GB of VRAM and 32 GB of disk — are two
    items, and hashing identity alone would mint the same id for both.
    """
    h = content_json_key((item_identity(it), re.sub(r"\s+", " ", (it.subject or "").lower())))
    return f"sem-{h[:10]}"


def content_json_key(obj: object) -> str:
    return content_key(json.dumps(obj, sort_keys=True, default=str))


# --------------------------------------------------------------------------- #
# segment context
# --------------------------------------------------------------------------- #

@dataclass
class SegmentContext:
    """Everything the extractor may look at for one segment.

    Bounded on purpose. The stage is specified segment-by-segment precisely so
    that the amount of text a decision is made from stays proportional to a topic
    rather than to the video: a 40-minute video must not produce one context
    window, and a decision made from 40 minutes of text is not one a reader could
    check against the segment it is filed under.
    """

    segment: Segment
    snippets: list[Snippet]
    frames: list[VisualFrame]
    speech: str
    speech_norm: str
    by_snippet_frames: dict[str, list[VisualFrame]] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.segment.title


def build_contexts(ev: Evidence, alignments: list, frames_by_id: dict[str, VisualFrame],
                   snippets_by_id: dict[str, Snippet]) -> list[SegmentContext]:
    """One context per segment, assembled from what already exists.

    `by_snippet_frames` comes from the alignment rather than from a time lookup
    here, so the frames a segment is allowed to reason about are exactly the ones
    the aligner paired — one definition of "relevant", in one place.

    One addition, and it is not a second definition. A frame the aligner did not
    pair — typically because a snippet already had its three nearest frames — is
    still inside some segment's time range, and if a model was asked about it, its
    answer has to be readable by something. Without this, a selected frame is paid
    for, recorded in full, and then dropped by every consumer, and the run reports
    a model call whose result no item in the document came from.
    """
    frames_of_snippet: dict[str, list[VisualFrame]] = {}
    for a in alignments:
        for fid in a.frame_ids:
            f = frames_by_id.get(fid)
            if f is not None:
                frames_of_snippet.setdefault(a.snippet_id, []).append(f)

    out: list[SegmentContext] = []
    for seg in ev.segments:
        sns = [snippets_by_id[s] for s in seg.snippet_ids if s in snippets_by_id]
        frames: list[VisualFrame] = []
        for s in seg.snippet_ids:
            for f in frames_of_snippet.get(s, []):
                if f.frame_id not in {x.frame_id for x in frames}:
                    frames.append(f)
        for fid in seg.frame_ids:
            f = frames_by_id.get(fid)
            if f is not None and f.frame_id not in {x.frame_id for x in frames}:
                frames.append(f)
        speech = " ".join(s.text for s in sns)
        out.append(SegmentContext(segment=seg, snippets=sns, frames=frames, speech=speech,
                                  speech_norm=normalize_text(speech).lower(),
                                  by_snippet_frames=frames_of_snippet))
    return out


def _observations_for(frame: VisualFrame) -> list[Observation]:
    """Read one frame's regions into observations, with their line labels.

    A frame that was read but produced no regions still yields **one** observation
    from the whole text. The regions are a layout, and a read without them is
    still a read — dropping the frame would lose a corroborating sighting of a
    value another frame located precisely. A frame with no text at all yields
    nothing, which is the structural reason a semantic item cannot be built from
    an empty read: there is nothing to be a claim about, and the validator's
    `ocr-without-text` check says so if something tries.
    """
    regions = [r for r in (frame.ocr_regions or [])
               if isinstance(r, dict) and (r.get("text") or "").strip()]
    if not regions:
        if not (frame.ocr_text or "").strip():
            return []
        obs = Observation(frame_id=frame.frame_id, text=frame.ocr_text,
                          ocr_confidence=frame.ocr_confidence)
        obs.values = find_values(obs.text)
        if obs.values and len(obs.values) == 1:
            obs.line_label = label_for_value(obs.text)
        return [obs]

    out: list[Observation] = []
    for group in line_groups(regions):
        for i, r in enumerate(group):
            text = (r.get("text") or "").strip()
            obs = Observation(frame_id=frame.frame_id, text=text,
                              bbox=tuple(r["bbox"]) if r.get("bbox") else None,
                              ocr_confidence=r.get("confidence", frame.ocr_confidence))
            obs.values = find_values(text)
            if obs.values:
                obs.line_label = label_for_value(text)
            # Structure is read before the unit parse, not after it, because the
            # unit parse is wrong on an expression: `28 × 28` reads as the figure
            # `28` carrying the unit `×`, which drops the second operand and the
            # result. A region the frame drew as one expression is one expression,
            # whatever the transcript's number grammar makes of its operators.
            if not URL_RE.search(text):
                expr = find_expression_values(text)
                if expr:
                    obs.values, obs.structure = expr, text
            if not obs.values:
                # A figure whose referent is written after it. Recovered at the
                # boundary so every later layer sees it through the same `values`
                # field it already reads.
                trailing = trailing_label(text)
                if trailing:
                    obs.values = [trailing]
                    obs.line_label = trailing[2]
            if i > 0:
                # The neighbour's own text, not its normalised form: a row label
                # like `Model A` is a name, and lowercasing it here is how a
                # benchmark table ends up with two rows both called "model a".
                # It must also not be figures-only — a row of counters is two
                # values side by side, and treating the left one as the right
                # one's caption is how "408" ends up as the subject of "218".
                prev = (group[i - 1].get("text") or "").strip()
                if prev and not find_values(prev) and not URL_RE.search(prev):
                    obs.line_label = obs.line_label or _clean_label(prev)
            if not obs.values and obs.line_label:
                # A bare figure that a label vouches for. `700` beside `users` is
                # a measurement; `700` on its own is a glyph, and the refusal in
                # `find_values` is what keeps the second from becoming the first
                # by accident. The label is the difference, so the label unlocks it.
                bare = _BARE_NUMBER.search(normalize_text(text))
                if bare and not URL_RE.search(text):
                    raw = bare.group(0)
                    obs.values = [(raw.replace(",", ""), raw, None)]
            out.append(obs)
    return out


# --------------------------------------------------------------------------- #
# subjects, claims, procedures
# --------------------------------------------------------------------------- #

#: Words that end a subject phrase. The subject of a figure is what sits between
#: the last of these and the figure itself — "at least 32GB of VRAM" gives
#: "VRAM", not "at least 32GB of".
_SUBJECT_BOUNDARY = re.compile(
    r"\b(?:of|for|with|at|to|in|on|is|are|was|were|be|been|and|or|but|from|by|"
    r"that|which|than|less|more|fewer|about|around|roughly|approximately|"
    r"up|only|just|per|"
    r"as|it|you|we|i|they|this|that|these|those|my|your|our|their|its|"
    r"have|has|had|do|does|did|can|could|will|would|should|may|might)\b",
    re.IGNORECASE)
_DETERMINER = re.compile(
    r"^(?:the|a|an|at least|about|around|roughly|approximately|only|just|"
    r"up to|over|under|more than|less than|some|around)\s+", re.IGNORECASE)

#: Fragments a boundary split can leave behind. "at least 32GB" splits at "at"
#: and leaves "least", which is a quantifier tail rather than a subject, and
#: filing a memory figure under "least" is worse than filing it under nothing.
_QUANTIFIER_TAIL = frozenset({"least", "most", "to", "than", "over", "under",
                              "about", "around", "roughly", "approximately", "up"})

#: Predicates that make a sentence a *claim about something* rather than a
#: sentence that happens to contain a digit. Deliberately short: a longer list
#: would turn every caption fragment into a claim.
_CLAIM_PREDICATES = re.compile(
    r"\b(?:reduces?|reduced|improves?|improved|increases?|decreases?|drops?|"
    r"costs?|costs?\s+about|saves?|saves?\s+on|runs?|took|takes?|processes?|"
    r"achieves?|achievable|gets?|hit|hits?|achieve|achieves|gains?|gained|"
    r"supports?|handles?|requires?|required|needs?|needed|delivers?|means?|"
    r"outperforms?|outperform|scales?|trains?|trained|inference|training|"
    r"performs?|performance)\b", re.IGNORECASE)

_COMPARISON_MARKERS = (
    " vs ", " vs. ", " versus ", " compared to ", " compared with ",
    " instead of ", " rather than ", " faster than ", " cheaper than ",
    " quicker than ", " better than ", " more than ", " less than ",
    " outperform ", " outperforms ", " ahead of ", " beats ",
)

_DIMENSIONS = {
    "latency": "latency", "speed": "speed", "faster": "speed", "throughput": "throughput",
    "cost": "cost", "price": "cost", "cheaper": "cost", "pricing": "cost",
    "accuracy": "accuracy", "quality": "quality", "benchmark": "benchmark",
    "memory": "memory", "vram": "memory", "ram": "memory", "size": "size",
    "tokens": "context", "context": "context", "performance": "performance",
    "score": "score", "length": "length", "weights": "size",
}

#: Base-form verbs that open an instruction.
#:
#: Discourse openers are deliberately absent. `so`, `now`, `and` and `first`
#: begin a large share of all spoken sentences, so admitting them turned
#: "So, let's get started" and "Now, this is all happening inside my browser"
#: into numbered steps — eighty of them in one twenty-minute video, none of them
#: a step. A step has to start with something you do to a thing.
_STEP_LEAD = re.compile(
    r"^(?:install|run|open|click|go to|navigate|set|add|create|download|configure|"
    r"type|press|choose|select|paste|execute|build|deploy|import|export|update|"
    r"upgrade|write|edit|copy|move|delete|start|stop|check|verify|rename|"
    r"enable|disable|install|connect|close|save|send|load|uninstall|"
    r"make sure|you'?ll need to|you need to|we need to|you'?ll want to|"
    r"you can start by|the first thing to do is)\b", re.IGNORECASE)

#: A sequence marker counts as one only when an action verb follows it. "First
#: the visual layer" introduces a topic; "First install the package" instructs.
#: The lookahead embeds the same alternation the lead rule uses, wrapped as a
#: group — spliced in raw it would be a top-level `|`, which does not compile.
_STEP_SEQUENCE = re.compile(
    r"^(?:first|firstly|next|then|finally|lastly|after that|second|third)\b[,: ]+"
    r"(?=(?:install|run|open|click|go to|navigate|set|add|create|download|"
    r"configure|type|press|choose|select|paste|execute|build|deploy|import|"
    r"export|update|upgrade|write|edit|copy|move|delete|start|stop|check|"
    r"verify|rename|enable|disable|connect|close|save|send|load|make sure)\b)",
    re.IGNORECASE)

#: A procedure is announced before it is given. Requiring the announcement is
#: what separates a workflow the speaker is walking through from a talk that
#: happens to contain imperatives — and it is a phrase a reader can check.
_PROCEDURE_FRAMING = re.compile(
    r"\b(?:the steps?|here'?s how|here is how|to get started|to set (?:it|this) up|"
    r"the setup|the process|step by step|install(?:ation)? steps|the workflow|"
    r"the first thing (?:to do|you need)|here'?s what you need|"
    r"go ahead and|let'?s (?:set|build|install|wire|hook|run) )\b", re.IGNORECASE)

_DEFINITION_RE = re.compile(
    r"^(?P<subject>[A-Za-z][\w ./+-]{1,48}?)\s+"
    r"(?:is|are|means|refers to|stands for|basically is|essentially is)\s+"
    r"(?P<value>.{3,160})$", re.IGNORECASE)

_ENTITY_STOPWORDS = frozenset({
    "the", "this", "that", "these", "those", "there", "then", "they", "we", "you",
    "it", "he", "she", "his", "her", "their", "our", "your", "my", "but", "and",
    "so", "now", "if", "when", "what", "why", "how", "here", "also", "just",
    "like", "ok", "okay", "yeah", "right", "well", "actually", "basically",
    "anyway", "first", "second", "third", "next", "last", "one", "two", "three",
    "i", "a", "an", "as", "at", "by", "for", "from", "in", "of", "on", "or", "to",
    "up", "with", "is", "are", "was", "were", "be", "do", "does", "can", "could",
    "will", "would", "should", "have", "has", "had", "not", "no", "yes", "ok",
    "every", "some", "any", "all", "more", "most", "much", "many", "very",
    "really", "thing", "things", "stuff", "guy", "guys", "people", "video",
    "channel", "comment", "comments", "link", "links", "part", "parts",
    # Discourse markers that open a sentence and are capitalised for exactly
    # that reason. This list, not a "not at the start of a sentence" rule, is
    # what keeps them out — because a real product name is just as often
    # sentence-initial ("Opus handles long context") as a filler word is, and
    # dropping every initial candidate loses half the names in a transcript.
    "however", "moreover", "furthermore", "therefore", "thus", "hence",
    "meanwhile", "nevertheless", "nonetheless", "overall", "basically",
    "honestly", "frankly", "obviously", "clearly", "sure", "yes", "no",
    "yeah", "hey", "hi", "hello", "thanks", "thank", "please", "let",
    "let\'s", "lets", "going", "gonna", "wanna", "kinda", "sort", "maybe",
    "probably", "definitely", "certainly", "absolutely", "exactly", "only",
    "even", "still", "already", "again", "once", "twice", "never", "always",
    "often", "sometimes", "usually", "always", "never", "very", "quite",
})

#: A candidate that opens with one of these is a sentence, not a name.
_AUXILIARY_LEAD = frozenset({
    "could", "would", "should", "can", "will", "shall", "may", "might", "must",
    "do", "does", "did", "is", "are", "was", "were", "has", "have", "had", "am",
    "let", "lets", "going", "gonna", "wanna", "got", "get", "if", "when", "what",
})

#: A Romance verb in the first person plural, which is regular enough to be a
#: morphology test rather than a vocabulary. Spanish and Portuguese end the form
#: in `-amos`, Italian in `-iamo`; English has no such form, and French's `-ons`
#: never reaches here because the verb is never capitalised in running prose.
#:
#: Only ever applied to a candidate that *opens the sentence*. Mid-sentence the
#: capital is not a signal anyway, so the form is simply not seen; at the start of
#: a sentence it is the only capital the verb will ever have, and
#: `Instalamos la herramienta Docker` filed the video's tooling under a verb.
#: The cost is the same one every morphological test in this module pays, and it
#: is paid in the safe direction: a name that ends in one of these endings is
#: dropped, not a name that happens to look like one.
_ROMANCE_1PL = re.compile(r"(?:amos|iamo)$", re.IGNORECASE)


def _opens_the_sentence(sentence: str, at: int) -> bool:
    return not sentence[:at].strip(" \t\n\r\"'“”«»([{-—…·•")


#: How many words either side of a name may carry the word that types it. Two
#: is enough for every construction the vocabulary was written against — "the
#: model Claude", "OpenAI, the company, says", "According to Dr Smith" — and
#: small enough that a cue describing a *different* name cannot reach.
_TYPE_CUE_WORDS = 2


def _adjacent(sentence: str, start: int, end: int) -> str:
    """The name plus the words immediately around it, and nothing further.

    A wider window types every name in a sentence by the same cue, which is how
    "The company Anthropic released Claude today" filed both of them under
    `organization` and "According to Dr Smith, the tool Docker works" filed the
    word `According` under `tool`. The cue describes the name it sits beside, so
    the window has to be the neighbourhood rather than the sentence.
    """
    left = " ".join(sentence[:start].split()[-_TYPE_CUE_WORDS:])
    right = " ".join(sentence[end:].split()[:_TYPE_CUE_WORDS])
    return f"{left} {sentence[start:end]} {right}"

_ENTITY_CATEGORY = (
    ("model", re.compile(r"\bmodels?\b|\bmodel\b", re.IGNORECASE)),
    ("tool", re.compile(r"\btools?\b|\blibrar(?:y|ies)\b|\bframeworks?\b|\bpackages?\b|"
                        r"\bSDKs?\b|\bCLIs?\b|\bplugins?\b|\bextensions?\b", re.IGNORECASE)),
    ("paper", re.compile(r"\bpapers?\b|\bet al\.?\b|\barxiv\b|\bpreprint\b", re.IGNORECASE)),
    ("dataset", re.compile(r"\bdatasets?\b|\bcorpus\b|\bcorpora\b|\bbenchmark suite\b", re.IGNORECASE)),
    # Attributive only. A finite verb after a name is a predicate *about* it —
    # "OpenAI says the model works" is a company making a claim, and reading
    # the verb as the description files every speaker in a video under `person`.
    # What a title or "according to" gives is a description; what "says" gives
    # is a role in the sentence. The title forms carry the same signal without
    # the ambiguity, and an uncertain person stays `entity`, which is the honest
    # reading and the one the report prefers.
    ("person", re.compile(r"\baccording to\b|\b(?:dr|prof|mr|ms|mme|herr|frau)\.?\s",
                          re.IGNORECASE)),
    ("organization", re.compile(r"\bcompan(?:y|ies)\b|\bannounc(?:ed|es)\b|\bfrom\b|"
                                r"\bfounded\b|\bteam\b|\borg\b|\buniversity\b|\blab\b", re.IGNORECASE)),
)

_CAPITALISED = re.compile(r"\b([A-Z][a-zA-Z0-9]*(?:[ .+-][A-Z][a-zA-Z0-9]*)*)\b")

#: When two sentences type the same name differently, the more specific reading
#: wins. "a paper on X" beside a later bare mention of X should not downgrade a
#: paper to a product, and a bare mention should not upgrade one either.
_TYPE_RANK = {"person": 0, "organization": 1, "paper": 2, "dataset": 3,
              "model": 4, "tool": 5, "product": 6, "entity": 7}

#: A candidate whose *shape* says "this is a name", independent of what
#: language it is in and of whether a category word happens to sit next to it.
#:
#: Two forms qualify: two or more capitalised words (`Claude Code`, `New York`),
#: and a single token that mixes cases or carries digits and a separator
#: (`Qwen3-8B`, `GPT-4`, `Llama-3`). A single plain capitalised word does not —
#: `Aber`, `Masse` and `Anthropic` are all one capitalised word, and only two of
#: them are names.
_NAME_SHAPED = re.compile(
    r"(?:[A-Z][\w.'+-]*[ .+-])+[A-Z][\w.'+-]*"   # two or more capitalised words
    r"|^[A-Za-z][A-Za-z]*\d[\w.-]*[A-Za-z0-9]$"    # letters then digits: Qwen3, Llama3
    r"|^[A-Za-z]+[-.][A-Za-z0-9]"                 # a hyphen or dot inside: GPT-4
    r"|^[A-Z][a-z]+[A-Z]")                        # internal capitals: iPhone, macOS


def detect_languages(text: str) -> tuple[list[str], bool]:
    """Which languages to filter ordinary words in, and whether to inflect.

    A thin re-export of `lang_resources.detect_languages` so callers inside this
    module do not have to know the module exists, and so the import surface stays
    one name rather than two. The second value is the morphology flag: German
    suffix-stripping is only safe where German was actually identified.
    """
    langs, detail = lang_resources.detect_languages(text)
    return langs, bool(detail.get("morphology"))


def subject_before(text: str, at: int,
                   languages: list[str] | None = None) -> str | None:
    """The phrase a figure is about, read from the words just before it.

    Returns None rather than a guess when the phrase is longer than five words,
    because a long span is a clause, not a subject, and turning a clause into a
    subject is how `importance` ends up describing the wrong thing.
    """
    window = text[max(0, at - 60):at]
    # A subject belongs to its own sentence. Without this the window reaches back
    # across a full stop, and "Latency was the bottleneck. It hit 40 ms" files
    # the figure under `bottleneck` — a real word from the wrong sentence, which
    # is harder to catch than a bad guess because it looks grounded. Carrying a
    # referent across the boundary is `subject_for`'s `previous` argument, and it
    # is explicit for that reason.
    cut = max(window.rfind("."), window.rfind("!"), window.rfind("?"))
    if cut != -1:
        window = window[cut + 1:]
    if not window.strip():
        return None
    # Scanned right to left, taking the first part that yields a subject. The
    # last part is often empty — "reduces inference cost by 3x" splits at "by"
    # and leaves nothing after it, and the subject is the phrase before that
    # preposition rather than nothing at all.
    #
    # The separators are kept alongside the parts, because the separator is the
    # evidence for one rule: a single word sitting directly after a personal
    # pronoun is that pronoun's predicate. "We saw 42 mentions" splits at `we`
    # and offers `saw`, and no morphology test can refuse it — `saw` is not an
    # -ed or -ing form, and irregular verbs are the reason a vocabulary exists at
    # all. The sentence's own shape answers it instead, and the shape is a fact
    # about English rather than about any one video: a pronoun takes a verb.
    parts = _SUBJECT_BOUNDARY.split(window)
    seps = list(_SUBJECT_BOUNDARY.finditer(window))
    for idx in range(len(parts) - 1, -1, -1):
        part = parts[idx]
        tail = _DETERMINER.sub("", part).strip(" ,.;:—-")
        if not tail or tail.lower() in _QUANTIFIER_TAIL:
            continue
        if (idx > 0 and len(tail.split()) == 1
                and _PERSONAL_PRONOUN.fullmatch(seps[idx - 1].group(0).strip())):
            continue
        # A subject is a noun phrase, so it ends where the predicate starts —
        # but a predicate is also routinely the *last* word of the span, and
        # then the noun is everything before it ("latency dropped"). Which of
        # the two halves is the referent is decided by which one survives
        # `_clean_referent`, so both are tried rather than guessing: a
        # pre-predicate cut keeps the object ("increases throughput"), a
        # post-predicate cut keeps the subject ("latency dropped"). Trying both
        # is what lets one rule serve transitive and intransitive verbs.
        head = _clean_referent(tail, languages=languages)
        if head:
            return head
        cut = _PREDICATE_CUT.search(tail)
        if cut:
            for candidate in (tail[cut.start():], tail[:cut.start()]):
                cleaned = _clean_referent(
                    _LEADING_PREDICATE.sub("", candidate).strip(" ,.;:-"),
                    languages=languages)
                if cleaned:
                    return cleaned
            return None
        # Then the phrase must survive `_clean_referent`, which is what stops
        # "ultimately reaches" and "latency dropped" from being filed as
        # subjects. A part that fails is skipped, not repaired: the scan
        # continues leftwards and takes the next candidate, or gives up.
        cleaned = _clean_referent(tail, languages=languages)
        if cleaned:
            return cleaned
    return None


def subject_after(text: str, at: int, length: int, unit: str | None = None,
                 languages: list[str] | None = None) -> str | None:
    """The phrase a figure is about, read from the words just after it.

    English puts the quantity first about as often as the noun — "at least
    32GB of VRAM", "500 users on the waitlist" — and a subject recovered only
    from the left misses every one of those, which is precisely the class of
    figure this layer exists to explain. Leading prepositions are skipped rather
    than treated as the start of the subject, so "32GB of VRAM" yields `VRAM`
    and not `of`.

    A *rate* preposition is the one exception: the noun after "per" is the
    interval a figure is quoted per, not the thing measured, so `$500 per
    month` yields nothing rather than `month`.
    """
    window = text[at + length:at + length + 60]
    out: list[str] = []
    for raw in window.split():
        word = raw.strip(" ,.;:—-")
        if not word:
            break
        if _SUBJECT_BOUNDARY.fullmatch(word) or word.lower() in _COMPARISON_PREPOSITIONS:
            if not out:
                if word.lower() in _RATE_PREPOSITIONS:
                    return None   # the denominator of a rate is not a referent
                if word.lower() in _COMPARISON_PREPOSITIONS:
                    return None   # "200x faster than X" is about the left side
                continue         # a leading preposition introduces the subject
            break
        if NUM_RE.search(word) or not re.search(r"[A-Za-z]{2}", word):
            break
        # The figure's own unit sits between the digits and the noun whenever the
        # referent follows: "20 ms latency", "32 GB of VRAM". It measures, so it
        # is stepped over rather than read as the subject — which is what made
        # every metric file itself under `ms`.
        if not out and _is_unit_word(word):
            continue
        out.append(word)
        if len(out) >= 3:
            break
    phrase = " ".join(out).strip()
    if not phrase or phrase.lower() in _QUANTIFIER_TAIL:
        return None
    return _clean_referent(phrase, unit, languages)


def subject_for(text: str, at: int, length: int, unit: str | None = None,
                previous: str | None = None,
                languages: list[str] | None = None) -> str | None:
    """The subject of a figure, preferring the left of it.

    "You need at least 32GB of VRAM" has nothing usable on the left and a clear
    answer on the right, so both sides are tried. Returns None when neither does,
    which is the honest outcome for a bare figure in a sentence that does not say
    what it is.

    `previous` is a subject already established for the sentence before this one,
    used only when the current sentence says nothing itself. It is a *continuation*
    signal, not a guess: the caller passes a subject it resolved from evidence,
    and it is applied only after both sides of this sentence have come up empty.
    """
    langs = languages if languages is not None else detect_languages(text)[0]
    found = (subject_before(text, at, langs)
             or subject_after(text, at, length, unit, langs))
    if found:
        return found
    return _clean_carry(previous) if previous else None


def sentences(speech: str) -> list[str]:
    """Split into sentences, keeping each one whole.

    Whole, because a claim's value is the sentence that carried it and a
    truncated one is a quotation that never happened. Splits on terminal
    punctuation followed by a capital, which is crude but never merges two
    sentences — over-splitting costs a fragment, under-splitting costs a
    fabricated claim spanning two.
    """
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9$\"'])|\n+", speech)
    return [p.strip() for p in parts if p and p.strip()]


#: Claims that assert something without stating a figure. "This model is the
#: fastest" is the most common form of the claim a chart settles or contradicts,
#: and a claim layer that only fires on a stated number cannot see it — so a
#: frame reading it as false had nothing to disagree with, and the disagreement
#: was silently dropped. Added for the cross-modal case, not to extract more.
_FIGURELESS_CLAIMS = re.compile(
    r"\b(?:fastest|slowest|best|worst|highest|lowest|cheapest|most|least|"
    r"largest|smallest|outperforms|beats|wins|leads)\b", re.IGNORECASE)

_COPULA = re.compile(r"\b(?:is|are|was|were)\b", re.IGNORECASE)


def _figureless_subject(sentence: str) -> str | None:
    """The noun phrase before the copula, or None.

    Deliberately narrow. A demonstrative subject ("this is going to be about
    ...") is a clause, not a thing, and turning one into a subject is how
    `subject` ends up describing the wrong thing — the same refusal
    `subject_before` makes for the same reason.
    """
    m = _COPULA.search(sentence)
    head = sentence[:m.start()] if m else ""
    head = _DETERMINER.sub("", head).strip(" ,.;:—-")
    if not head or len(head.split()) > 4 or len(head) > 48:
        return None
    if head.lower() in _DEFINITION_NO_SUBJECT:
        return None
    return head


def find_claim(sentence: str) -> tuple[str, str | None] | None:
    """(subject, metric-display) when a sentence asserts something checkable.

    `display` is None for a claim that states no figure. The claim still exists
    — it is the sentence's own assertion, and a frame is exactly the thing that
    can support or contradict it — but it has no linked metric, and the loop
    that builds one skips it rather than inventing a value to attach.
    """
    if not _CLAIM_PREDICATES.search(sentence):
        if not _FIGURELESS_CLAIMS.search(sentence):
            return None
        subject = _figureless_subject(sentence)
        return (subject, None) if subject else None
    for m in NUM_RE.finditer(sentence):
        subj = subject_before(sentence, m.start())
        if not subj:
            continue
        raw, unit, cur = m.group("value"), m.group("unit"), m.group("cur")
        if not unit and not cur:
            continue
        return subj, (f"{cur}{raw}" if cur else f"{raw}{unit}")
    return None


def find_comparison(sentence: str) -> tuple[str, str, list[dict]] | None:
    """A two-subject comparison, with dimensions only where one is named.

    Returns None unless an explicit contrast marker separates two named subjects
    *and* puts a figure on each side of it. Inferred comparisons — two products
    mentioned in one sentence — are how a comparison gets manufactured out of
    adjacency, and a comparison with an invented subject is worse than no
    comparison, because it reads as a measured result.

    The marker is chosen, not the first one found. A sentence often carries two
    ("A is faster than B, at 120 ms versus 180 ms") and only one of them is the
    split that separates the two figures; taking the first puts both numbers on
    the same side and loses the comparison entirely.
    """
    low = sentence.lower()
    best_score: tuple[int, int] | None = None
    best: tuple[str, str, str, str] | None = None
    for marker in _COMPARISON_MARKERS:
        start = 0
        while (i := low.find(marker, start)) != -1:
            start = i + len(marker)
            left = sentence[:i].strip(" ,.;:")
            right = sentence[start:].strip(" ,.;:")
            a, b = _last_entity_phrase(left), _first_entity_phrase(right)
            if not a or not b or a.lower() == b.lower():
                continue
            na, nb = len(find_values(left)), len(find_values(right))
            # A figure on each side is not required: "Postgres versus MySQL" is a
            # real comparison with no numbers, and dropping it would lose a claim
            # the speaker actually made. It only breaks ties between candidate
            # markers, preferring the split that separates the evidence.
            score = (min(na, nb), na + nb)
            if best_score is None or score > best_score:
                best_score, best = score, (a, b, left, right)
    if best is None:
        return None
    a, b, left, right = best

    dimensions: list[dict] = []
    seen: set[str] = set()
    for word in re.findall(r"[A-Za-z]+", low):
        dim = _DIMENSIONS.get(word)
        if not dim or dim in seen:
            continue
        seen.add(dim)
        # Only values on the correct side of the split are attributed to that
        # subject. A number anywhere in the sentence is not evidence about either
        # one, and moving one across is precisely the "read a comparator's figure
        # onto the subject model" error.
        va, vb = _value_for(left, dim), _value_for(right, dim)
        if va is None and vb is None:
            continue
        row: dict = {"dimension": dim}
        if va is not None:
            row["a"] = va
        if vb is not None:
            row["b"] = vb
        dimensions.append(row)
    return a, b, dimensions


_DIMENSION_TOKEN = re.compile(r"[A-Za-z]+")


def _value_for(text: str, dim: str) -> str | None:
    """The figure on one side of a comparison that belongs to a named dimension.

    The dimension word is often named once, on one side only — "Opus has 120 ms
    latency versus Sonnet at 180 ms" — so requiring it on both sides would lose
    the second figure. Where it appears, the nearest figure is taken; where it
    does not, a side holding exactly one figure is unambiguous, and a side
    holding two is not, so nothing is claimed for it.
    """
    if not text:
        return None
    for m in _DIMENSION_TOKEN.finditer(text):
        if _DIMENSIONS.get(m.group(0).lower()) != dim:
            continue
        nums = NUM_RE.findall(text[max(0, m.start() - 40):m.end() + 40])
        if nums:
            return _display(nums[0])
        break
    nums = NUM_RE.findall(text)
    return _display(nums[0]) if len(nums) == 1 else None


def _display(match) -> str:
    """Render a figure the way the frame showed it.

    Accepts either a match object or the `(cur, value, unit)` tuple that
    `NUM_RE.findall` yields — both spellings are in use here, and the tuple form
    is positional, so its indices are named here rather than at each call site.
    """
    if hasattr(match, "group"):
        cur, raw, unit = match.group("cur"), match.group("value"), match.group("unit")
    else:
        cur, raw, unit = match[0], match[1], match[2]
    return f"{cur}{raw}" if cur else f"{raw}{unit or ''}"


def _entity_phrases(text: str, languages: list[str] | None = None,
                    morphology: bool = True) -> list[str]:
    """Capitalised names in a fragment, determiner stripped.

    Capitalised only, and that is load-bearing: the fallback of taking the first
    or last two words regardless turns "more than 500 users" into a comparison
    between the sentence and the number 500. A name is written with a capital;
    a quantity is not. When the two modalities disagree on capitalisation the
    comparison is simply not asserted, which is the weaker but true claim.

    The language stopword lists apply here too, for the same reason they apply
    in `find_entities`: this function's output becomes a *subject* attached to a
    figure, and "Aber" as the thing a benchmark is about is the same defect one
    layer earlier.
    """
    if languages is None:
        languages, morphology = detect_languages(text)
    langs = languages
    out = []
    for cand in _CAPITALISED.findall(text):
        cand = _LABEL_HEAD.sub("", cand.strip(" .+-")).strip()
        if len(cand) <= 1 or cand.lower() in _ENTITY_STOPWORDS:
            continue
        if any(lang_resources.is_ordinary(w, langs, morphology)
               for w in lang_resources.tokens(cand)):
            continue
        out.append(cand)
    return out


def _last_entity_phrase(text: str) -> str | None:
    found = _entity_phrases(text)
    return found[-1] if found else None


def _first_entity_phrase(text: str) -> str | None:
    found = _entity_phrases(text)
    return found[0] if found else None


def find_definition(sentence: str) -> tuple[str, str] | None:
    m = _DEFINITION_RE.match(sentence.strip())
    if not m:
        return None
    subject = m.group("subject").strip()
    if subject.lower() in _DEFINITION_NO_SUBJECT or len(subject.split()) > 5:
        return None
    value = re.split(r"\s+(?:and|but|which|because|so|then)\b|[,;]", m.group("value"), 1)[0]
    value = value.strip().rstrip(".")
    if len(value) < 3 or len(subject) > 48:
        return None
    # A value longer than a noun phrase is the rest of the sentence, and a gloss
    # that long says nothing a reader could check. Measured, not guessed: on a
    # real transcript every genuine definition here was well under this.
    if len(value) > _DEFINITION_MAX_VALUE:
        return None
    return subject, value


#: A candidate carrying a word that names a *unit* is a measurement written out
#: rather than a thing with a name — "Millionen Lichtjahre" is a distance, and
#: "3x schneller" is a ratio. Reusing the numbers layer's unit vocabulary is the
#: whole trick: the same words that make a number meaningful make a name
#: implausible, and a second list would drift from the first.

#: A word that cannot be the thing a figure is about, whatever its position.
#:
#: These are the words that were being read as subjects because they sit next to
#: the number: an adverb in "ultimately reaches 100%", a gerund in "we're
#: talking like 200x", a magnitude in "up to 2 million tokens". None of them
#: names a referent, and a subject that is not a referent is worse than no
#: subject — it files the figure under the wrong heading, which is the failure
#: this whole pass exists to remove.
#:
#: The `-ly` rule and the closed list are the general mechanism; the individual
#: words are the *English* instances of "grammatically incapable of being a
#: noun", which is a property of the language rather than of any one video.
_ADVERB_LIKE = re.compile(r"^[a-z]{3,}ly$", re.IGNORECASE)
_ADVERB_WORDS = frozenset("""
also just only even still already almost nearly roughly about around over under
ultimately eventually typically generally essentially basically actually really
truly simply merely mostly largely partly somewhat fairly rather quite then
together apart aside instead meanwhile however therefore thus hence
""".split())

#: Magnitude words. "2 million tokens" is two million *of tokens* — `million`
#: scales the figure and names nothing, so it is dropped from the phrase rather
#: than allowed to head it.
_MAGNITUDE_WORDS = frozenset("""
thousand million billion trillion quadrillion k m b bn
""".split())

#: Spellings that are a unit and never a referent. Read to the right of a
#: figure, "20 ms" and "32 GB of VRAM" both begin with the unit; returning it as
#: the subject files every metric under its own unit name. The figure's own
#: `unit` field is checked first where the caller has one, so this list only has
#: to cover the spellings a caller reaches without it.
_UNIT_WORDS = frozenset("""
ms ns us µs s sec secs second seconds min mins minute minutes hr hrs hour hours
day days week weeks month months year years gb tb mb kb pb gbps mbps tbps
hz khz mhz ghz w kw mw v kv a ma ma c f rpm fps tok/s tokens/s words/s req/s
requests/s rps qps usd eur gbp us/% percent x × ‰
""".split())

#: A preposition that introduces a *rate* rather than a referent. "per month"
#: names the interval a figure is quoted per; `month` is the denominator of a
#: rate, not the thing being measured, so it is not a subject.
_RATE_PREPOSITIONS = frozenset({"per"})

#: Words this module has already decided name a referent: the measurement
#: vocabulary the numbers layer reasons about, and the singular forms of the six
#: entity categories. The language filter below consults the stopword resources,
#: and those are tuned for *entity* extraction — `model` and `team` are in
#: `en.txt` because a heading may call a thing by its category, and a subject
#: window reading them as ordinary words refused every figure the report exists to
#: file. Two gates in one module disagreeing about a word is the defect, so the
#: word wins here.
_REFERENT_VOCAB = MEASURE_VOCAB | frozenset({
    "model", "models", "tool", "tools", "library", "framework", "package",
    "sdk", "cli", "plugin", "extension", "paper", "dataset", "corpus",
    "benchmark", "company", "team", "organization", "organisation",
    "university", "lab", "person", "speaker", "author", "product", "service",
    "toolchain", "framework",
})

#: A personal pronoun. In a subject window these are already boundaries, so the
#: word on their right is the clause's predicate.
_PERSONAL_PRONOUN = re.compile(r"i|we|you|he|she|they|it", re.IGNORECASE)

#: Currency written as a word. `_CURRENCY` catches the symbol, and the symbol is
#: nearly always stripped into the figure's own `unit` before the subject is
#: read, so without this list "it costs 500 dollars" filed the figure under
#: `dollars` — and the docstring above claimed exactly the opposite. A short
#: closed list, because a currency *is* a closed set: there is no general
#: English morphology that recognises one, and inventing one would be a
#: dictionary in all but name.
_CURRENCY_WORDS = frozenset("""
dollar dollars cent cents euro euros pound pounds penny pence yen rupees rupee
rupees franc francs rucible reales reais zloty złoty koruna forint dirham
shekel riyal baht won taka ringgit rupiah bahtc
""".split())

#: Words that modify the noun after them rather than naming anything. "700 users
#: last week" is about users, and the phrase stops at the modifier — otherwise
#: the subject comes out as "users last week" and every reader has to work out
#: which part is the thing.
_MODIFIER_WORDS = frozenset("""
last next past previous every each per annual monthly weekly daily hourly
average averaged total combined overall maximum minimum peak
""".split())

#: A bare comparative or superlative. "200x faster" names the *dimension* being
#: compared, not the thing compared — the referent is on the other side of the
#: comparison and has to come from the left or from an earlier sentence. A word
#: ending in `-er`/`-est` is only accepted as a subject when the measure
#: vocabulary vouches for it (`user`, `number`), which keeps `faster` out without
#: also throwing out `users`.
_COMPARATIVE = re.compile(r"^[a-z]{3,}(?:er|est)$", re.IGNORECASE)

#: Prepositions that introduce a *comparison* rather than a referent. "200x
#: faster than the previous version" — the words after `like` and `than` are the
#: thing being compared against, which is not what the figure is about. A
#: boundary word already splits the phrase here; these are listed separately
#: because a *leading* boundary is skipped rather than treated as a break, so
#: "like 200x" used to read `like` as the subject.
_COMPARISON_PREPOSITIONS = frozenset({"like", "than", "versus", "vs"})

#: A verb in the `-s` form, or an irregular past tense. This is the one genuinely
#: risky English shape — far too many nouns end in `s` to exclude them all, and
#: an irregular past (`grew`, `fell`) carries no marker at all — so the list is
#: short, closed, and holds only words that are a verb in essentially every
#: sentence they appear in. A word that is sometimes a noun and sometimes a verb
#: (`report`, `model`, `cost`, `plan`) is deliberately absent: excluding those
#: would throw out real referents, and a missing subject is the recoverable
#: outcome here, a wrong one is not.
#:
#: This is the same mechanism as `_CLAIM_PREDICATES` above and for the same
#: reason — English verbhood is not recoverable from spelling, so a closed
#: vocabulary is the honest tool and the alternative (a parser) is not this
#: layer's job. The cost is that a verb nobody listed can still be read as a
#: subject, which is why every one of them is a word whose *noun* sense is rare.
#: Bare forms a talk about numbers actually produces in the slot after a figure,
#: which is the one place the morphology test cannot reach: "500 users churn" and
#: "8 nodes ship" are both a noun and its verb, and the phrase is a noun phrase
#: only up to the first of the two. Completing the closed list the gate already
#: consults, rather than adding a general English verb heuristic — a currency is
#: a closed set and a verb is not, so the list stays closed and short.
_VERB_FORMS = frozenset("""
respond responds happen happens remain remains seem seems appear appears
start starts began begin begins grow grows grew fall falls fell rise rises rose
run runs ran take takes took give gives gave come comes came show shows showed
find finds tell tells say says said become becomes became build builds built
buy buys bought cut cuts draw draws drive drives feel feels hold holds
keep keeps knew know knows leave leaves lost meet meets meant mean means
pay pays put puts read reads sell sells spend spends stand stands
understand understands win wins write writes exceed exceeds deliver delivers
generate generates consume consumes produce produces obtain obtains
expect expects suggest suggests indicate indicates demonstrate demonstrates
enable enables add adds remove removes create creates update updates
install installs download downloads upload uploads accept accepts
reject rejects render renders store stores load loads
score scores scored measure measures measured reach reaches reached
exceed exceeds exceeded require requires required include includes included
provide provides provided allow allows allowed support supports supported
handle handles handled process processes processed train trains trained
achieve achieves achieved scale scales scaled perform performs performed
churn jump jumps double doubles triple triples climb climbs soar soars
surge surges ship ships serve serves earn earns charge charges quote quotes
multiply multiplies divide divides stream streams deploy deploys monitor monitors
""".split())

#: A contraction. "we're talking like 200x" splits on `we` and leaves `'re
#: talking like`, and the clitic is not a word — it is the tail of one. Every
#: English contraction is this shape, so the test is structural rather than a
#: list of the hundred ways to make one.
_CONTRACTION = re.compile(r"^['’][a-z]+$|^[a-z]+['’]", re.IGNORECASE)

#: A verb form. `-ed` and `-ing` are the two English inflectional endings a noun
#: essentially never takes, which makes them a general test rather than a list:
#: "latency dropped" yields `latency`, and "talking" is never a subject. The
#: irregular and `-s` forms are not here — too many nouns end in `s` for that to
#: be safe — so they still rely on the predicate vocabulary below.
#:
#: `search`, not `match`: the pattern is anchored only at the end, so the test is
#: "does this word *end* in -ed/-ing". `match` would anchor at the start as well
#: and silently never fire, which is the shape of validator defect that looks
#: like a rule which rarely fires.
_VERB_FORM = re.compile(r"(?:ed|ing)$", re.IGNORECASE)

#: Predicates that do not inflect for tense in a way the morphology above can see.
#: The same list the phrase cut already used, now consulted per word so a
#: sentence-final form like "reaches" is recognised as a predicate too. Written
#: as a vocabulary rather than as two compiled patterns because three rules
#: consume it and a copy per rule would drift.
_PREDICATE_VOCAB = frozenset("""
is are was were be been being has have had do does did
cost costs reach reaches hit hits run runs take takes get gets
need needs require requires support supports handle handles
drop drops reduce reduces increase increases scale scales
train trains process processes achieve achieves talk talks
""".split())

#: The same vocabulary as one alternation, for the two patterns that need to
#: search a span rather than test a single token. Joined explicitly because a
#: newline inside a raw pattern is a literal character, and a vocabulary written
#: as readable lines would then match a newline rather than a predicate.
_PREDICATE_ALT = "|".join(sorted(_PREDICATE_VOCAB))

#: Whole-word match for a single predicate token.
_SUBJECT_PREDICATE = re.compile(rf"^(?:{_PREDICATE_ALT})$", re.IGNORECASE)

#: Where a noun phrase ends, as a search rather than a whole-word match. Two
#: shapes are recognised: a listed predicate, and any `-ed`/`-ing` form, which
#: covers the tense and participle variants no list can enumerate ("latency
#: dropped", "context growing") without a parser.
_PREDICATE_CUT = re.compile(
    rf"\b(?:{_PREDICATE_ALT}|[A-Za-z]{{3,}}(?:ed|ing))\b", re.IGNORECASE)

#: Leading predicate words dropped after a cut, so "increases throughput" is read
#: from the verb rather than rejected for starting with one.
_LEADING_PREDICATE = re.compile(
    rf"^(?:{_PREDICATE_ALT}|[A-Za-z]{{3,}}(?:ed|ing))\b[\s,]*", re.IGNORECASE)


def _is_unit_word(word: str) -> bool:
    """A spelling that measures rather than names. Currency included, because
    "500 dollars" is a figure and "dollars" is its unit, not its subject."""
    w = word.strip(".,;:!?").lower()
    return w in _UNIT_WORDS or w in _CURRENCY_WORDS or w in _CURRENCY


def _clean_referent(phrase: str, unit: str | None = None,
                    languages: list[str] | None = None) -> str | None:
    """The noun phrase inside `phrase`, or None when there is not one.

    The single gate every candidate passes before it is allowed to become a
    subject. It is deliberately a *filter* rather than a parser: a phrase is
    accepted only if every word in it is capable of naming a referent, so the
    failure mode is a missing subject rather than a wrong one. That is the whole
    design — a confidently incorrect subject files a figure under the wrong
    heading, and an unresolved one is simply not filed.

    Magnitude words are dropped rather than rejected, because "2 million
    tokens" has a real referent and only its scale word is in the way. A unit is
    different: it is *the figure's own* spelling, and "the latency dropped to
    20 ms" filed under `ms` is the same class of error as filing it under
    `ultimately`. So a unit is rejected wherever it appears, including when the
    caller passed it as the item's own `unit` — a figure's unit is precisely the
    word least likely to be its subject. `unit` is consulted for the opposite
    purpose: to add a spelling the list does not carry (`GiB`) to the set of
    words that cannot be a referent.
    """
    words = [w for w in phrase.split() if w]
    if not words or len(words) > 5:
        return None
    units = set(_UNIT_WORDS) | {u for u in re.split(r"[/·]", unit or "") if u.strip()}
    kept: list[str] = []
    for w in words:
        bare = w.strip(".,;:!?").lower()
        if not bare or not re.search(r"[A-Za-z]{2}", w):
            return None
        if _is_unit_word(w) or bare in units or bare in _CURRENCY:
            return None       # a unit measures, it never names
        if bare in _MAGNITUDE_WORDS:
            continue          # scales the figure, names nothing
        if _ADVERB_LIKE.match(bare) or bare in _ADVERB_WORDS:
            return None       # "ultimately reaches 100%"
        if _VERB_FORM.search(bare) or _SUBJECT_PREDICATE.match(bare) \
                or bare in _VERB_FORMS:
            return None       # "latency dropped", "talking", "to respond"
        if _SUBJECT_BOUNDARY.fullmatch(bare) or bare in _COMPARISON_PREPOSITIONS:
            return None       # a preposition or auxiliary inside the phrase
        if _CONTRACTION.match(w.strip(" ,.;:—-")):
            return None       # "'re" is the tail of a word, not a referent
        if bare in _MODIFIER_WORDS:
            break             # "users last week" is about users
        if (languages and w.lower() not in _REFERENT_VOCAB
                and lang_resources.is_ordinary(w, languages)):
            # The gate around this loop is an English vocabulary; the shipped
            # resources already answer the general question in every language
            # the report supports. Without this, "Die Latenz stieg auf" and
            # "Wir sahen" are subjects in German — a determiner, a clause and a
            # preposition filed as the thing a figure is about, which is the one
            # failure the whole pass exists to prevent.
            #
            # After the modifier break, not before: a modifier ends the phrase
            # rather than tainting it, so "users last week" is still `users` even
            # though `last` is an ordinary English word.
            return None
        kept.append(w)
    if not kept:
        return None
    head = kept[-1].strip(".,;:!?").lower()
    if _COMPARATIVE.match(head) and head not in MEASURE_VOCAB:
        # A lone comparative names the dimension, not the referent.
        return None
    out = _DETERMINER.sub("", " ".join(kept)).strip(" ,.;:-")
    if not out or len(out) > 48 or NUM_RE.search(out):
        return None
    return out


def _clean_carry(phrase: str) -> str | None:
    """The same gate, for a subject handed in from the previous sentence."""
    return _clean_referent(phrase)


_UNIT_SUFFIX = re.compile(r"(?:jahre|jahre|sekunden|minuten|stunden| kilometern|"
                          r"kilometre|metern|meter|bytes|bit|bits)$", re.IGNORECASE)


def _is_quantity_phrase(candidate: str) -> bool:
    """Is this candidate a quantity written in words rather than a name?

    Two conditions, either sufficient. A token from `MEASURE_VOCAB` — the units
    and quantities the numbers layer already knows — or a compound whose tail is
    a unit by suffix (`Lichtjahre`, `Stundenkilometern`). The first is a lookup
    against a list that has to stay current; the second is a shape, and it is
    what catches the German compounds (`Galaxienhaufen`, `Millionen
    Stundenkilometern`) that no list of this size would cover.
    """
    toks = lang_resources.tokens(candidate)
    if any(t in MEASURE_VOCAB for t in toks):
        return True
    words = candidate.split()
    return len(words) >= 2 and bool(_UNIT_SUFFIX.search(words[-1]))


def find_entities(sentence: str, languages: list[str] | None = None,
                  morphology: bool = True) -> list[tuple[str, str]]:
    """Corroboration-free candidates from one sentence. Not yet entities.

    Returning candidates rather than items is the point: an entity needs to be
    seen more than once, or once in speech and once on screen, before it is
    anything. A capitalised word is a capitalised word until something else in
    the document agrees with it, and an extractor that promotes on the first
    sighting turns every sentence-initial "However" into a person.

    `languages` narrows which stopword lists are consulted. Omit it and the
    sentence is asked what language it is in, which is right for a one-off call
    and wasteful for a whole segment — `_corroborated_entities` detects once over
    the segment's speech and passes the answer down.

    Two filters run here and they are not the same filter:

    * **Capitalisation**, which is the English-language convention and only
      works where that convention holds. German capitalises every noun, so
      `Aber` and `Masse` arrive looking exactly like `Anthropic` and `Masse` is
      not a name. That gap is closed by the language stopword lists, not by
      tightening the capital test — a stricter test would drop the real names,
      which are just as often sentence-initial.
    * **Name shape**, which is a weak positive: two capitalised words, or a token
      with digits or an internal capital, is name-shaped in almost any language.
      It rescues `Qwen3-8B` and `Claude Code` from a sentence that happened to
      carry no category word.

    An uncategorised candidate comes back as `entity`, not as `product`. The
    extractor genuinely does not know, and the report says so rather than
    asserting a product identity no evidence supports.
    """
    if languages is None:
        languages, morphology = detect_languages(sentence)
    langs = languages
    out: list[tuple[str, str]] = []
    for m in _CAPITALISED.finditer(sentence):
        # A determiner is part of the phrase the engine matched, never part of
        # the name: "The Opus" and "Opus" are one entity, and splitting them
        # leaves every introduced name with a single sighting and no entity.
        cand = _LABEL_HEAD.sub("", m.group(1).strip(" .+-")).strip()
        if len(cand) < 2 or cand.lower() in _ENTITY_STOPWORDS:
            continue
        # Ordinary vocabulary in any language the transcript touches. Checked on
        # the whole candidate as well as its first word, because German writes
        # "Diese Methode" and the head noun is the one that matters.
        if any(lang_resources.is_ordinary(w, langs, morphology)
               for w in lang_resources.tokens(cand)):
            continue
        if _is_quantity_phrase(cand):
            # "Millionen Lichtjahre", "Wiederkehrende Muster" — a phrase
            # describing a quantity or a property. The unit vocabulary already
            # exists for the numbers layer, and it is the same signal here: a
            # candidate carrying a word that names a *unit of measure* is a
            # measurement written out, not a thing with a name. General, and it
            # needs no word of its own.
            continue
        if len(cand.split()) > 4:
            continue
        if cand.isupper() and len(cand) > 4:
            continue  # UI furniture in caps
        if cand.split()[0].lower() in _AUXILIARY_LEAD:
            # "Could I" and "Would you" capitalise at the start of a sentence and
            # look exactly like a name to a capitalisation test. Two sightings
            # of a question is still a question.
            continue
        if (_opens_the_sentence(sentence, m.start())
                and _ROMANCE_1PL.search(cand.split()[-1])):
            # A first-person-plural verb opening the sentence. Same argument as
            # the auxiliary above, decided by inflection rather than by a list,
            # because in these three languages the inflection *is* the regular
            # part of the grammar.
            continue
        if cand.isupper() and len(cand) <= 4:
            # A short all-caps token is a category — AI, LLM, GPU, API — not a
            # name. Extracting one as an entity put "AI" in the document beside
            # the actual models, which is exactly the distinction the item type
            # is supposed to carry.
            continue
        itype = "entity"
        near = _adjacent(sentence, m.start(), m.end())
        for name, rx in _ENTITY_CATEGORY:
            if rx.search(near):
                itype = name
                break
        out.append((cand, itype))
    return out


def find_steps(sentences_: list[str]) -> list[str]:
    """Instructional sentences, in order, or nothing.

    Two conditions, both required. The sentence must open with an action — a
    discourse opener is not one. And the segment must have *announced* a
    procedure, because a workflow the speaker walks through is introduced as one
    and a talk that merely contains imperatives is not. Without the second
    condition a single "run the command" in an hour of talking becomes a
    one-step procedure, and the number of steps stops tracking the material.
    """
    if not any(_PROCEDURE_FRAMING.search(s) for s in sentences_):
        return []
    return [s for s in sentences_
            if _STEP_LEAD.match(s.strip()) or _STEP_SEQUENCE.match(s.strip())]


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

@dataclass
class SemanticResult:
    """What the stage produced. `observations` is kept so a run can be audited
    on the decisions it made about the frames it did *not* turn into evidence."""

    items: list[Item] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"items": [i.to_dict() for i in self.items],
                "conflicts": [c.to_dict() for c in self.conflicts],
                "observations": self.observations,
                "metrics": self.metrics}


def _vlm_by_frame(payload: dict | None) -> dict[str, dict]:
    if not payload:
        return {}
    return {str(f.get("frame_id")): f for f in payload.get("frames", []) if f.get("frame_id")}


#: The kinds of entity a model observation may assert, mapped onto the schema's
#: vocabulary. A kind the model did not state is not guessed into one: "a logo
#: that might be a product" is a different claim from "Product X", and typing the
#: first as the second is the fabrication this whole layer is built to catch.
_VLM_ENTITY_TYPES = {
    "product": "product", "model": "model", "tool": "tool",
    "organization": "organization", "person": "person",
}

_DIGITS = re.compile(r"\d+(?:\.\d+)?")


def _digits(text: str) -> set[str]:
    """The numeric runs in a string, as written.

    Compared instead of the text itself, because the OCR that read a frame
    routinely garbles the characters around a number — `32.` for `32`, `18,` for
    `18` — and requiring an exact match would reject the readings this pass
    exists to make. What it will not tolerate is a figure whose digits the frame's
    own read never contained at all.
    """
    return set(_DIGITS.findall(text or ""))


def _corroborated(text: str, hay: str) -> bool:
    """True when `text` appears literally in `hay`, whitespace ignored.

    Literal, not fuzzy. A near-match is how "Qwen3.8" becomes a citation for
    "Qwen3.6", and the whole reason OCR and VLM readings are kept apart is that
    one of them is wrong in a way no score reveals.
    """
    n, h = normalize_text(text).lower(), normalize_text(hay).lower()
    return bool(n) and (n in h or n.replace(" ", "") in h.replace(" ", ""))


def _vlm_label_is_grounded(label: str | None, frame: VisualFrame, ctx: SegmentContext) -> bool:
    """May this row/column/series label be attached to a value?

    Only if the frame's own OCR read, or the speech, actually contains it. This
    is the check that makes the table case safe rather than dangerous: OCR
    returns `100 32 180 18` with nothing attaching them to anything, and the
    model's job is to say which row each belongs to. But a label that appears
    nowhere in the frame and nowhere in the speech is a label the model supplied
    from the conversation — and "the transcript said Product X somewhere in this
    segment, so the 0.09 on screen is Product X's price" is precisely the
    inference the design forbids.
    """
    if not label:
        return False
    return _corroborated(label, frame.ocr_text or "") or _corroborated(label, ctx.speech)


def _vlm_items(frame: VisualFrame, rec: dict, ctx: SegmentContext, cfg: SemanticConfig
               ) -> tuple[list[Item], list[dict], list[dict]]:
    """Structured visual evidence from one frame's observation.

    Returns `(items, claim_links, decisions)`. The claim links are the
    corroborations and the disagreements, held back for the claim loop: a spoken
    claim and the frame that supports or contradicts it are two records about one
    statement, and building them apart and joining them afterwards is what keeps
    the two modalities' confidences separate instead of averaged into one number.
    The decisions are what was *refused* and why, which belong in the observation
    log — an extractor that silently discards a model reading is
    indistinguishable from one that never received it.

    Every rule here is a refusal. The model is not trusted to supply a label that
    the frame does not show, a value whose digits the frame's read never
    contained, or an entity no second modality names. What survives is a smaller
    document than the model returned, and that is the intended outcome: a
    fabricated table cell inherits authority in the claims table, the summary
    and the conclusion, and a dropped one costs the reader a figure they can go
    and look at.
    """
    from vlm import VisionObservation  # noqa: PLC0415 — one reader of the shape

    obs = VisionObservation.from_dict(rec, frame.frame_id)
    if obs.status != "ok":
        return [], [], []
    model_conf = obs.confidence if obs.confidence is not None else cfg.policy.base_vlm
    ctype = obs.content_type
    ocr_read = frame.ocr_text or ""
    speech = ctx.speech
    items: list[Item] = []
    links: list[dict] = []
    decisions: list[dict] = []
    notes: list[str] = []

    def prov(evidence_text: str, cross_modal: bool = False) -> list:
        return [vlm_provenance(frame, evidence_text,
                               confidence=cfg.policy.apply(model_conf, cross_modal=cross_modal))]

    # -- values, with the row and column they were seen under -----------------
    for n in obs.numbers:
        value = str(n.get("value") or "").strip()
        if not value:
            continue
        if n.get("legible") is False:
            # The model says it inferred this one. An inferred value is exactly
            # the "A = 73.4" that the phase brief forbids, so it is recorded as
            # an unreadable finding rather than promoted to a number.
            items.append(Item(
                id="", type="visual_finding", value=value[:200], subject=n.get("row") or n.get("label"),
                verbatim=n.get("context") or obs.summary,
                provenance=prov(value), confidence=cfg.policy.base_vlm,
                status="unreadable", importance="low",
                detail={"kind": "uncertainty", "content_type": ctype, "value": value},
                notes=f"the model marked {value!r} as inferred rather than read in {frame.frame_id}"))
            continue
        seen = _digits(value)
        if seen and not (seen & _digits(ocr_read)) and not (seen & _digits(speech)):
            notes.append(f"dropped {value!r}: its digits appear in neither the frame's read nor the speech")
            continue
        row = (n.get("row") or "").strip() or None
        column = (n.get("column") or "").strip() or None
        label = (n.get("label") or "").strip() or None
        subject = next((x for x in (row, label, column) if x), None)
        grounded = _vlm_label_is_grounded(subject, frame, ctx)
        if subject and not grounded:
            notes.append(f"dropped the label {subject!r} for {value!r}: it is not in the frame's read "
                         f"or the speech, so attaching it would be the model's inference")
            subject = None
        unit = (n.get("unit") or "").strip() or None
        if not subject and not unit:
            # A bare figure with nothing to attach it to. Emitting it puts a
            # number in the document whose referent nobody can supply, and every
            # reader then has to decide independently whether to trust it.
            notes.append(f"dropped {value!r}: no grounded label and no unit")
            continue
        in_speech = _corroborated(value, speech)
        # The one signal OCR never has: a model looked at the picture and can say
        # what kind of frame this is. A figure read off a chart is a benchmark by
        # what is on the frame, whatever the number happens to be — which is why
        # this cannot be recovered from the label or the unit alone.
        # The content type first, and it outranks the unit. A bar on a chart is
        # a benchmark whatever its unit says — `40 ms` read off a latency chart
        # is the vendor's result, not a metric the viewer measured — and a unit
        # the model itself supplied is the weakest of the three signals here.
        role = {"chart": "benchmark"}.get(ctype) or _unit_role(value, unit, f"{value}{unit or ''}")
        if role is None:
            role = {"table": "count", "diagram": "metric"}.get(ctype)
        if role is None:
            role = "metric" if unit else "count" if subject else "unknown"
        items.append(Item(
            id="", type="number", value=value, unit=unit, subject=subject,
            display=f"{value}{unit}" if unit else value,
            verbatim=n.get("context") or obs.summary,
            provenance=prov(value, cross_modal=in_speech),
            confidence=cfg.policy.apply(cfg.policy.base_vlm, cross_modal=in_speech),
            status="asserted", importance="high" if subject else "medium",
            semantic_role=role,
            detail={"row": row, "column": column, "content_type": ctype},
            notes=(f"model-derived from {frame.frame_id} by {obs.backend or 'a model'}"
                   + (f" ({ctype})" if ctype != "unknown" else "")
                   + f"; role {role} from the frame's own content type"
                   + "; not a reading of the pixels and not verified")))

    # -- entities, only with a second modality naming them -------------------
    for e in obs.entities:
        text = str(e.get("text") or "").strip()
        itype = _VLM_ENTITY_TYPES.get(str(e.get("kind") or "").strip().lower())
        if not text or not itype:
            continue
        backing = []
        if _corroborated(text, ocr_read):
            backing.append(ocr_provenance(frame, text,
                                         confidence=float(frame.ocr_confidence or 0.0),
                                         read_method=frame.ocr_read_method or "native"))
        snip = _snippet_containing(ctx, text)
        if snip is not None:
            backing.append(transcript_provenance(snip))
        if not backing:
            # A logo recognised from appearance, or a name the model supplied
            # from the surrounding conversation. Both are claims a logo alone
            # cannot carry, and the corroboration requirement is what stops a
            # guess at a product's identity entering the document as a fact.
            notes.append(f"dropped the entity {text!r}: neither the frame's read nor the speech names it")
            continue
        corroborated = len(backing) > 1
        items.append(Item(
            id="", type=itype, value=text, verbatim=text,
            provenance=backing + prov(text, cross_modal=corroborated),
            confidence=cfg.policy.apply(cfg.policy.base_vlm, cross_modal=corroborated),
            status="asserted", importance="medium",
            detail={"content_type": ctype, "kind": "entity"},
            notes=(f"named by a model reading of {frame.frame_id}"
                   + ("; corroborated by a second modality" if corroborated
                      else "; named by the frame's own read only"))))

    # -- diagram edges and chart orderings, with both ends named --------------
    for r in obs.relationships:
        src = str(r.get("source") or "").strip()
        dst = str(r.get("target") or "").strip()
        if not src or not dst:
            continue
        label = (r.get("label") or "").strip() or None
        items.append(Item(
            id="", type="relationship", value=f"{src} → {dst}", subject=src,
            qualifier=label,
            verbatim=obs.summary,
            provenance=prov(f"{src} → {dst}"),
            confidence=cfg.policy.base_vlm, status="asserted",
            importance="high" if r.get("certain") is not False else "medium",
            detail={"from": src, "to": dst, "label": label,
                    "kind": (r.get("kind") or "").strip() or None,
                    "certain": r.get("certain"), "content_type": ctype},
            notes=(f"connection seen in {frame.frame_id} ({ctype}); model-derived, not verified"
                   + ("" if r.get("certain") is not False else "; the model marked the direction uncertain"))))

    # -- what was being done on screen ---------------------------------------
    for a in obs.actions:
        desc = str(a.get("description") or "").strip()
        if not desc:
            continue
        items.append(Item(
            id="", type="visual_finding", value=desc[:300], subject=(a.get("target") or None),
            verbatim=obs.summary, provenance=prov(desc),
            confidence=cfg.policy.base_vlm, status="asserted", importance="medium",
            detail={"kind": "action", "content_type": ctype, "target": a.get("target")},
            notes=f"operation visible in {frame.frame_id}; a visible state, not an inferred interaction"))

    # -- the parts that could not be read. Kept, not dropped. ---------------
    for u in obs.uncertainties:
        text = str(u.get("text") or "").strip()
        if not text:
            continue
        items.append(Item(
            id="", type="visual_finding", value=text[:300], verbatim=obs.summary,
            provenance=prov(text), confidence=cfg.policy.base_vlm,
            status="unreadable", importance="low",
            detail={"kind": "uncertainty", "content_type": ctype,
                    "reason": u.get("reason")},
            notes=f"could not be read in {frame.frame_id}; recorded rather than guessed"))

    # -- whether the frame supports what was said ----------------------------
    for c in obs.claims_supported:
        claim = str(c.get("claim") or "").strip()
        support = str(c.get("support") or "").strip()
        if not claim or support not in ("supported", "not_supported", "unclear"):
            continue
        links.append({"claim": claim, "support": support, "reason": c.get("reason"),
                      "frame": frame, "content_type": ctype, "confidence": c.get("confidence")})

    if notes:
        decisions.append({"frame_id": frame.frame_id, "classification": "vlm-refused",
                          "reason": "; ".join(notes), "content_type": ctype,
                          "backend": obs.backend, "model": obs.model})
    return items, links, decisions



def _screen_item(obs: Observation, frame: VisualFrame, ctx: SegmentContext,
                 cfg: SemanticConfig) -> Item | list[Item] | None:
    """Turn one meaningful screen observation into an item, or into its figures.

    `None` for a value with no label and no corroboration, which is the case the
    `irrelevant` class exists for. The alternative — emitting it and marking it
    low-importance — puts a number in the document whose referent nobody can
    supply, and every downstream reader then has to decide independently whether
    to trust it.

    A *list* when the observation is a grouped expression: `28 × 28 = 784` is
    three figures and one relationship, and keeping only the first would drop
    the answer while keeping the question. Each figure is filed as its own item
    — deduplication and ranking downstream are per-figure, and merging them
    would make `28` and `784` one thing — and each carries the expression
    verbatim in its `detail`, so the relationship the reader saw is preserved
    without this layer asserting an equation it cannot itself check.
    """
    if not obs.values:
        return None
    if len(obs.values) > 1 and obs.structure:
        total = len(re.findall(r"\d[\d,]*(?:\.\d+)?", obs.structure or ""))
        return [_expression_item(obs, v[0], frame, ctx, cfg, n, total)
                for n, v in enumerate(obs.values, 1)]

    value, display, unit = obs.values[0]
    subject = obs.line_label
    # Typed by what the value IS, not by whether it has a unit. A bare figure
    # recovered from a labelled row has no unit and is a number; typing it as a
    # link because a unit was absent put "46 COMMENTS" in the document as a URL.
    is_url = bool(URL_RE.fullmatch(value.strip())) or value.lower().startswith(("http://", "https://", "www."))
    itype = "link" if is_url else "number"
    if itype == "number" and not subject and unit is None:
        # A bare figure on screen is only meaningful with a subject; the label is
        # the only honest source of one. A URL is exempt — it names itself, and
        # refusing an unlabelled link would drop exactly the screen-only resource
        # this pipeline is built to recover.
        return None
    labelled = bool(subject)
    conf = cfg.policy.apply(
        cfg.policy.base_labelled_screen if labelled else cfg.policy.base_unlabelled_screen,
        cross_modal=value_in_speech(obs.values, ctx.speech))
    role, importance, why = classify_number_role(obs, value, unit, display, frame, ctx)
    return Item(
        id="", type=itype, value=value, display=display, unit=unit, subject=subject,
        verbatim=obs.text,
        provenance=[ocr_provenance(frame, obs.text,
                                   confidence=float(frame.ocr_confidence or 0.0),
                                   read_method=frame.ocr_read_method or "native")],
        confidence=conf,
        status="asserted",
        importance=importance,
        semantic_role=role,
        notes=(f"read from {frame.frame_id} ({obs.classification}: {obs.reason}); "
               f"role {role}: {why}"),
    )


def _expression_item(obs: Observation, value: str, frame: VisualFrame,
                     ctx: SegmentContext, cfg: SemanticConfig, position: int,
                     total: int) -> Item:
    """One figure of a grouped expression, carrying the expression with it."""
    display, unit = value, None
    role, importance, why = classify_number_role(obs, value, unit, display, frame, ctx)
    conf = cfg.policy.apply(
        cfg.policy.base_unlabelled_screen,
        cross_modal=value_in_speech(obs.values, ctx.speech))
    return Item(
        id="", type="number", value=value, display=display, unit=None,
        subject=None, verbatim=obs.text,
        provenance=[ocr_provenance(frame, obs.text,
                                   confidence=float(frame.ocr_confidence or 0.0),
                                   read_method=frame.ocr_read_method or "native")],
        confidence=conf, status="asserted", importance=importance, semantic_role=role,
        detail={"expression": obs.structure, "position": f"{position}/{total}"},
        notes=(f"read from {frame.frame_id} ({obs.classification}: {obs.reason}); "
               f"role {role}: {why}"),
    )


def _extract_segment(ctx: SegmentContext, ev: Evidence, frames_by_id: dict[str, VisualFrame],
                     snippets_by_id: dict[str, Snippet], vlm: dict[str, dict],
                     cfg: SemanticConfig) -> tuple[list[Item], list[dict]]:
    """Everything derived from one segment. The unit of both work and context."""
    items: list[Item] = []
    obs_out: list[dict] = []

    for frame in ctx.frames:
        for obs in _observations_for(frame):
            classify_observation(obs, ctx.speech)
            obs_out.append(obs.to_dict())
            if obs.classification == "meaningful":
                made = _screen_item(obs, frame, ctx, cfg)
                if isinstance(made, list):
                    items.extend(made)
                elif made is not None:
                    items.append(made)

    # -- VLM observations. Model-derived, capped, and never merged into a read. --
    #
    # Structured records become typed items with their structure intact; a
    # caption-only record — an older artifact, or a provider that could not be
    # made to call the schema — still yields its numbers through the text path,
    # because dropping a legible figure is a worse failure than a weakly
    # described one. Either way the provenance is `vlm` and the notes say so.
    claim_links: list[dict] = []
    if cfg.enable_vlm:
        for frame in ctx.frames:
            rec = vlm.get(frame.frame_id)
            if not rec or rec.get("status") != "ok":
                continue
            summary = (rec.get("summary") or rec.get("caption") or "").strip()
            if not summary:
                continue
            v_items, links, decisions = _vlm_items(frame, rec, ctx, cfg)
            items.extend(v_items)
            obs_out.extend(decisions)
            claim_links.extend(links)
            if not rec.get("numbers") and not rec.get("relationships") and not rec.get("actions"):
                model_conf = rec.get("confidence")
                base = cfg.policy.apply(cfg.policy.base_vlm, cross_modal=_in_speech(summary, ctx.speech))
                for value, display, unit in find_values(summary) or [("", summary, None)]:
                    items.append(Item(
                        id="", type="quote" if unit is None else "number",
                        value=value or summary[:200], display=display if unit is not None else None,
                        unit=unit, verbatim=summary,
                        provenance=[vlm_provenance(frame, summary,
                                                   confidence=float(model_conf or cfg.policy.base_vlm))],
                        confidence=base, status="asserted", importance="medium",
                        semantic_role=("metric" if unit is not None else "unknown"),
                        notes=(f"model-derived from {frame.frame_id} by "
                               f"{rec.get('model') or rec.get('backend')}; "
                               "not a reading of the pixels and not verified")))

    # -- claims and their metrics, kept as separate linked items. --------------
    for sent in sentences(ctx.speech):
        found = find_claim(sent)
        if not found:
            continue
        subject, display = found
        snip = _snippet_containing(ctx, sent)
        if snip is None:
            continue
        prov = [transcript_provenance(snip)]
        claim = Item(id="", type="claim", value=sent[:400], subject=subject, verbatim=sent,
                     provenance=prov, confidence=cfg.policy.apply(cfg.policy.base_transcript),
                     status="asserted", importance="high",
                     notes=("the source's own wording; see the linked claim_metric for the figure"
                            if display else
                            "the source's own wording; no figure was stated, so a frame is the "
                            "only thing that can settle it"))
        # Cross-modal corroboration is a *second provenance entry*, not a
        # higher number. The claim keeps the confidence of what was said and the
        # frame keeps the confidence of what was seen, and a reader can see both
        # and discount either. Averaging them would produce a single figure that
        # describes neither modality and cannot be checked against either.
        for link in claim_links:
            if not _corroborated(link["claim"], sent):
                continue
            if link["support"] == "supported":
                claim.provenance.append(vlm_provenance(
                    link["frame"], link["claim"],
                    confidence=cfg.policy.apply(link["confidence"] or cfg.policy.base_vlm,
                                                cross_modal=True)))
                claim.notes += f"; corroborated by {link['frame'].frame_id}"
            elif link["support"] == "not_supported":
                # The frame is read as contradicting what was said. Both records
                # are kept and a conflict is raised below — deciding which is
                # right is a judgement about the video, not a fact the document
                # can recover, and an extractor that makes the call has already
                # lost the ability to be checked.
                claim.status = "conflicting"
                claim.notes += (f"; the frame at {link['frame'].frame_id} does not appear to support it")
                items.append(Item(
                    id="", type="visual_finding", value=link["claim"][:300], subject=subject,
                    verbatim=claim.verbatim,
                    provenance=[vlm_provenance(link["frame"], link["claim"],
                                               confidence=cfg.policy.apply(
                                                   link["confidence"] or cfg.policy.base_vlm))],
                    confidence=cfg.policy.base_vlm, status="conflicting", importance="high",
                    detail={"kind": "structure", "content_type": link["content_type"]},
                    notes=(f"the model read {link['frame'].frame_id} as not supporting this claim"
                           + (f": {link['reason']}" if link.get("reason") else "")
                           + "; neither record has been corrected against the other")))
            else:
                # "unclear" is not "false". The model saying it cannot tell is
                # recorded as a finding and no conflict is raised, because a
                # conflict is a claim that two sources disagree and here nobody
                # has said they do.
                items.append(Item(
                    id="", type="visual_finding", value=link["claim"][:300], subject=subject,
                    verbatim=claim.verbatim,
                    provenance=[vlm_provenance(link["frame"], link["claim"],
                                               confidence=cfg.policy.apply(
                                                   link["confidence"] or cfg.policy.base_vlm))],
                    confidence=cfg.policy.base_vlm, status="asserted", importance="low",
                    detail={"kind": "uncertainty", "content_type": link["content_type"]},
                    notes=(f"the model could not tell whether {link['frame'].frame_id} supports this "
                           "claim; recorded as unresolved, not as disagreement")))
        claim.id = item_id_for(claim)
        # The disagreement records are linked now that the claim has an id; the
        # conflict itself is raised in `extract`, once every item does.
        for extra in items:
            if extra.type == "visual_finding" and not (extra.detail or {}).get("claim_id"):
                extra.detail["claim_id"] = claim.id
        items.append(claim)
        m = NUM_RE.search(sent)
        metric_value = m.group("value").replace(",", "") if m else display
        if not metric_value:
            # A claim that states no figure has no metric. Inventing one — even a
            # bare `0` — would put a number in the document that nobody said and
            # no frame showed, and a claim_metric with no figure behind it is the
            # shape a fabricated measurement takes.
            continue
        metric = Item(
            id="", type="claim_metric", value=metric_value,
            display=display, unit=(m.group("cur") or m.group("unit")) if m else None,
            subject=subject, verbatim=sent,
            provenance=list(prov), confidence=cfg.policy.apply(cfg.policy.base_transcript),
            status="asserted", importance="high",
            detail={"claim_id": claim.id},
            notes="figure supporting the linked claim; related but not identical evidence")
        metric.id = item_id_for(metric)
        items.append(metric)

    # -- comparisons ----------------------------------------------------------
    if cfg.enable_comparisons:
        for sent in sentences(ctx.speech):
            found = find_comparison(sent)
            if not found:
                continue
            a, b, dimensions = found
            snip = _snippet_containing(ctx, sent)
            if snip is None:
                continue
            note = "contrast stated by the source"
            if not dimensions:
                note += "; no dimension named, so no value is attributed to either subject"
            it = Item(
                id="", type="comparison", value=f"{a} vs {b}", subject=a, verbatim=sent,
                detail={"subject_a": a, "subject_b": b, "dimensions": dimensions},
                provenance=[transcript_provenance(snip)],
                confidence=cfg.policy.apply(cfg.policy.base_transcript),
                status="asserted", importance="high", notes=note)
            it.id = item_id_for(it)
            items.append(it)

    # -- definitions ----------------------------------------------------------
    for sent in sentences(ctx.speech):
        found = find_definition(sent)
        if not found:
            continue
        subject, value = found
        snip = _snippet_containing(ctx, sent)
        if snip is None:
            continue
        it = Item(id="", type="definition", value=value[:300], subject=subject, verbatim=sent,
                  provenance=[transcript_provenance(snip)],
                  confidence=cfg.policy.apply(cfg.policy.base_transcript),
                  status="asserted", importance="medium", notes="gloss stated in passing")
        it.id = item_id_for(it)
        items.append(it)

    # -- procedures, only when there is a run of them -------------------------
    if cfg.enable_steps:
        steps = find_steps(sentences(ctx.speech))
        if len(steps) >= cfg.min_steps_in_segment:
            for n, s in enumerate(steps, start=1):
                snip = _snippet_containing(ctx, s)
                if snip is None:
                    continue
                it = Item(id="", type="step", value=s[:300], verbatim=s,
                          detail={"index": n, "group": ctx.segment.id},
                          provenance=[transcript_provenance(snip)],
                          confidence=cfg.policy.apply(cfg.policy.base_transcript),
                          status="asserted", importance="high",
                          notes=f"step {n} of {len(steps)} in this segment")
                it.id = item_id_for(it)
                items.append(it)

    # -- entities, only once corroborated -------------------------------------
    if cfg.enable_entities:
        items.extend(_corroborated_entities(ctx, cfg))

    for it in items:
        if not it.id:
            it.id = item_id_for(it)
    return items, obs_out


def _snippet_containing(ctx: SegmentContext, text: str) -> Snippet | None:
    """The snippet a derived sentence came from, by normalised containment.

    Matching on the snippet rather than on the segment is what keeps a claim's
    citation pointing at the exact fragment that carried it. When the sentence
    spans two snippets the earliest containing snippet is used, so the citation
    names where the statement began.
    """
    needle = normalize_text(text).lower()
    for s in ctx.snippets:
        hay = normalize_text(s.text).lower()
        if needle and (needle in hay or hay in needle):
            return s
    return None


def _corroborated_entities(ctx: SegmentContext, cfg: SemanticConfig) -> list[Item]:
    """Entities seen more than once, or once in speech and once on screen.

    The corroboration rule is the whole safeguard against over-extraction: one
    sighting of a capitalised word is a capitalised word. Two independent
    sightings is a name the document actually uses.
    """
    # Corroboration counts the NAME, not the name-and-type pair. The two
    # sentences that introduce one product rarely use the same word to describe
    # it — "the Opus model" and "Opus handles" type differently — and keying on
    # the pair would leave every real entity at one sighting and drop all of them.
    #
    # The language is detected once over the whole segment rather than per
    # sentence. A two-word German sentence carries too little vocabulary to be
    # identified on its own, so per-sentence detection would fall back to
    # "every Latin language" for most of the segment anyway and do the same work
    # more slowly.
    langs, morphology = detect_languages(ctx.speech)
    counts: dict[str, int] = {}
    typed: dict[str, str] = {}
    spelling: dict[str, str] = {}
    for sent in sentences(ctx.speech):
        for name, itype in find_entities(sent, langs, morphology):
            key = name.lower()
            counts[key] = counts.get(key, 0) + 1
            spelling.setdefault(key, name)
            if key not in typed or _TYPE_RANK.get(itype, 99) < _TYPE_RANK.get(typed[key], 99):
                typed[key] = itype
    on_screen: dict[str, list[VisualFrame]] = {}
    for frame in ctx.frames:
        for region in (frame.ocr_regions or []):
            rtext = (region.get("text") or "").strip()
            if len(rtext) < 2 or len(rtext.split()) > 4:
                continue
            key = _LABEL_HEAD.sub("", rtext).strip().lower()
            if key in counts:
                on_screen.setdefault(key, []).append(frame)

    out: list[Item] = []
    for key, n in sorted(counts.items()):
        itype = typed[key]
        name = spelling[key]
        frames = on_screen.get(key, [])
        if n < 2 and not frames:
            # One sighting, off screen, and a single capitalised word — the shape
            # an ordinary noun has in every language. A name-shaped token
            # (`Claude Code`, `Qwen3-8B`) is exempted, because a product
            # introduced once and never repeated is common and a word the
            # capitalisation rule merely caught is not.
            if not (_NAME_SHAPED.search(name) and itype == "entity"):
                continue
        snip = _snippet_containing(ctx, name)
        if snip is None:
            snip = ctx.snippets[0] if ctx.snippets else None
        if snip is None:
            continue
        display_name = spelling[key]
        prov = [transcript_provenance(snip)]
        if frames:
            prov.extend(ocr_provenance(f, name, confidence=float(f.ocr_confidence or 0.0),
                                       read_method=f.ocr_read_method or "native")
                        for f in frames[:3] if name.lower() in (f.ocr_text or "").lower())
        cross = bool(frames)
        it = Item(
            id="", type=itype, value=display_name, subject=display_name, verbatim=snip.text[:200],
            provenance=[p for p in prov if p.ocr_text is not None or p.source == "transcript"],
            confidence=cfg.policy.apply(cfg.policy.base_transcript, cross_modal=cross),
            status="asserted",
            importance="high" if itype in ("model", "tool", "paper", "dataset") else "medium",
            notes=(f"named {n}× in this segment"
                   + (f" and on {len(frames)} frame(s)" if frames else "")
                   + ("; type not established — no surrounding vocabulary named one"
                      if itype == "entity"
                      else "; typed by surrounding vocabulary, not disambiguated")))
        it.id = item_id_for(it)
        out.append(it)
    return out


# --------------------------------------------------------------------------- #
# dedup and conflicts
# --------------------------------------------------------------------------- #

def merge_items(existing: list[Item], incoming: list[Item]) -> tuple[list[Item], int]:
    """Fold `incoming` into `existing` on semantic identity. Returns the merged
    list and the number of items that were folded in.

    Conservative on purpose. Identity is (type, value, unit, subject, qualifier,
    status) — not a text similarity — so `GPT-4` and `GPT-4o` stay apart and
    `$500` and `$5,000` stay apart, while a value read from four frames and the
    same value spoken once become one item carrying five citations. The two
    sides of that trade are the same failure in opposite directions: a duplicate
    inflates the evidence count, and an over-eager merge deletes a distinction
    the report exists to preserve.
    """
    by_identity: dict[tuple, list[Item]] = {}
    for i in existing:
        by_identity.setdefault(item_identity(i), []).append(i)
    out = list(existing)
    merged = 0
    for it in incoming:
        key = item_identity(it)
        target = next((c for c in by_identity.get(key, [])
                       if subjects_compatible(c.subject, it.subject)), None)
        if target is None:
            by_identity.setdefault(key, []).append(it)
            out.append(it)
            continue
        have = {(p.source, p.ref) for p in target.provenance}
        for p in it.provenance:
            if (p.source, p.ref) not in have:
                target.provenance.append(p)
                have.add((p.source, p.ref))
        # The weaker read governs, for the same reason the OCR pass takes the
        # minimum: an item backed by one sure read and one doubtful one is as
        # doubtful as the doubtful one allows.
        target.confidence = round(min(target.confidence, it.confidence), 4)
        if target.importance is None:
            target.importance = it.importance
        elif it.importance == "high":
            target.importance = "high"
        if target.subject is None:
            target.subject = it.subject
        if target.detail is None and it.detail is not None:
            # The richer structure wins, and it usually arrives second. OCR reads
            # a benchmark table as four unlabelled numbers and files them by line;
            # a model reading of the same frame says which cell each was in. When
            # the two merge into one item — correctly, they are the same fact seen
            # twice — the cell has to survive the merge. Dropping it leaves a
            # document that has the value and has lost the one thing the visual
            # pass was called for.
            target.detail = dict(it.detail)
        merged += 1
    return out, merged


def _adopt_subjects(screen_items: list[Item], spoken: list[Item]) -> list[Item]:
    """Give an unlabelled screen value the subject the speech gives it.

    A frame showing `32 GB` beside no label and a sentence saying "you need at
    least 32GB of VRAM" is one fact read twice, not two facts. Without this the
    two items differ only by a `subject` one of them does not have, the identity
    tuple separates them, and the document shows the same figure twice — the
    exact duplication that makes an evidence count unreadable.

    Adoption needs a **unique** match on value and unit. With two candidates the
    frame is compatible with both, and picking either would be the guess this
    layer exists to avoid; the item keeps no subject and stays a screen-only
    reading, which is the weaker but truthful claim.
    """
    by_value: dict[tuple[str, str], list[Item]] = {}
    for it in spoken:
        if it.subject and it.type in ("number", "claim_metric"):
            by_value.setdefault((_norm_value(it.value), (it.unit or "").lower()), []).append(it)
    for it in screen_items:
        if it.subject or it.type not in ("number", "link"):
            continue
        matches = by_value.get((_norm_value(it.value), (it.unit or "").lower()), [])
        if len(matches) == 1:
            it.subject = matches[0].subject
            it.notes = f"{it.notes}; subject taken from {matches[0].id} (the spoken form)"
    return screen_items


def detect_conflicts(items: list[Item], ev: Evidence) -> list[Conflict]:
    """Cross-modal disagreements, recorded and left open.

    The correspondence required is textual, not inferred. Two figures are
    compared only when they share a unit, sit in the same segment, come from
    different modalities, differ in value, **and** one side's label appears
    verbatim in the other's own text. That last condition is what makes this
    high-precision: it is checkable from the document, and it is the difference
    between "the screen said 700 where the speaker said 500 about the same
    thing" and "a video mentions several memory sizes".

    Keying on the two `subject` fields instead would almost never fire, because
    the modalities name the same quantity differently — a UI row says `VRAM` and
    the sentence says `the card`. Making the detector match on labels rather than
    on the thing both labels point at is the same mistake in miniature as the
    nearest-frame alignment this stage replaced.

    Nothing here resolves. Which source to believe is a judgement about the
    video, not a fact recoverable from the document, and a resolver that picked
    one would make the artifact unable to show that the disagreement existed.
    """
    by_item = {i.id: i for i in items}
    text_of = _item_text_map(ev)
    seg_of = {iid: seg.id for seg in ev.segments for iid in seg.item_ids}

    spoken = [i for i in items if i.type in ("number", "claim_metric")
              and any(p.source == "transcript" for p in i.provenance)]
    shown = [i for i in items if i.type == "number"
             and any(p.source in ("ocr", "vlm") for p in i.provenance)]

    conflicts: list[Conflict] = []
    seen: set[frozenset] = set()
    for s in spoken:
        for d in shown:
            if (s.unit or "").lower() != (d.unit or "").lower() or not s.unit:
                continue
            if _norm_value(s.value) == _norm_value(d.value):
                continue
            if seg_of.get(s.id) != seg_of.get(d.id) or not seg_of.get(s.id):
                continue
            if not _names_each_other(s, d, text_of):
                continue
            pair = frozenset((s.id, d.id))
            if pair in seen:
                continue
            seen.add(pair)
            both = [s, d]
            ids = sorted({i.id for i in both})
            conflicts.append(Conflict(
                item_ids=ids, kind="value-mismatch",
                statement=(
                    f"{s.display or s.value}"
                    + (f" {s.unit}" if s.unit else "")
                    + " is spoken while the frame shows "
                    + f"{d.display or d.value}"
                    + (f" {d.unit}" if d.unit else "")
                    + (" for " + str(d.subject) if d.subject else "")
                    + " — the two disagree and neither has been corrected against the other"),
                resolution="unresolved"))
            for i in both:
                if by_item.get(i.id) is not None:
                    i.status = "conflicting"
    return conflicts


def attach_items_to_segments(ev: Evidence, items: list[Item], alignments: list) -> None:
    """File every new item under the segment it belongs to.

    The build stage could only file what it knew about — the transcript items,
    which are reachable from the snippet that carried them. A screen-only item
    has no snippet, so it reaches its segment through the frame the aligner
    paired. Without this a segment's `item_ids` lists only half of what was
    extracted from it, and a reader following the segment to its evidence finds
    nothing on screen — which is the whole point of the list.
    """
    seg_of_snippet = {sid: seg.id for seg in ev.segments for sid in seg.snippet_ids}
    seg_of_frame: dict[str, str] = {}
    for a in alignments:
        for fid in a.frame_ids:
            if a.segment_id:
                seg_of_frame.setdefault(fid, a.segment_id)
    by_id = {seg.id: seg for seg in ev.segments}
    for it in items:
        seg_id = next((seg_of_snippet.get(p.ref) for p in it.provenance
                       if p.source == "transcript" and p.ref in seg_of_snippet), None)
        if not seg_id:
            seg_id = next((seg_of_frame.get(p.ref) for p in it.provenance
                           if p.source in ("ocr", "vlm") and p.ref in seg_of_frame), None)
        seg = by_id.get(seg_id) if seg_id else None
        if seg is not None and it.id not in seg.item_ids:
            seg.item_ids.append(it.id)


def _item_text_map(ev: Evidence) -> dict[str, str]:
    """item id → the text its provenance points at, for the label cross-check."""
    snippets = {s.id: s.text for s in ev.snippets}
    frames = {v.frame_id: (v.ocr_text or v.vlm_caption or "") for v in ev.visual}
    out: dict[str, str] = {}
    for it in ev.items:
        parts = [snippets.get(p.ref, "") if p.source == "transcript"
                 else frames.get(p.ref, "") for p in it.provenance]
        out[it.id] = normalize_text(" ".join(parts)).lower()
    return out


def _names_each_other(spoken: Item, shown: Item, text_of: dict[str, str]) -> bool:
    """Does one side's label literally appear in the other side's own text?

    The only correspondence check that is decidable here. Either direction
    counts — a screen row labelled `VRAM` and a sentence that says `VRAM` is the
    ordinary case, and its mirror (a sentence naming `the card`, a row labelled
    `card`) is the same fact read the other way round.
    """
    st, dt = text_of.get(spoken.id, ""), text_of.get(shown.id, "")
    for a, b in ((shown.subject, st), (spoken.subject, dt)):
        if a and len(a) >= 3 and re.sub(r"\s+", " ", a.strip().lower()) in b:
            return True
    return False


def vlm_disagreements(items: list[Item]) -> list[Conflict]:
    """Turn the model's "this frame does not support that claim" into a conflict.

    Built here, after every item has an id, rather than where the disagreement
    is noticed — a conflict names items, and an item with no id yet is a pointer
    to nothing. `unclear` never reaches this function: a model saying it cannot
    tell has not asserted a disagreement, and promoting that to a conflict is the
    "do not turn unclear into false" rule in the one place it is cheapest to keep.
    """
    by_id = {i.id: i for i in items}
    out: list[Conflict] = []
    for i in items:
        if i.type != "visual_finding" or i.status != "conflicting":
            continue
        cid = str((i.detail or {}).get("claim_id") or "")
        claim = by_id.get(cid)
        if claim is None:
            continue
        fid = next((p.ref for p in i.provenance if p.source == "vlm"), None)
        out.append(Conflict(
            item_ids=sorted({claim.id, i.id}), kind="source-disagreement",
            statement=(f"the claim at {claim.id} is read as unsupported by the frame at {fid}"
                       f" ({i.notes}) — the two disagree and neither has been corrected "
                       "against the other"),
            resolution="unresolved"))
    return out


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def _segment_cache_key(ctx: SegmentContext, cfg: SemanticConfig, vlm: dict) -> str:
    """Identity of the extraction *question* for one segment.

    The segment's own text, the frames aligned to it with their reads, any model
    captions, the config, the extraction version and the schema version. Change
    any of them and the entry misses — which is the only thing that makes a cache
    safe to keep across a change to what the extractor asks.

    `t_start`/`t_end` are in the frame payload for the same reason the reads are,
    and were missing here for a while: the extraction *emits* a provenance
    timestamp copied from the frame, so the timestamp is part of the answer, and
    a key that omits it serves an answer computed against a different frame time.
    Measured on a real video: ten items citing `t_start=266.0` for a frame
    `frames.json` placed at 266.76, every one of them from a cache entry built
    before the frame was re-measured, and the evidence gate refusing the document
    as a result.
    """
    payload = {
        "segment": ctx.segment.to_dict(),
        "speech": ctx.speech,
        "frames": [{"frame_id": f.frame_id, "text": f.ocr_text,
                    "regions": f.ocr_regions, "conf": f.ocr_confidence,
                    "t_start": f.t_start, "t_end": f.t_end}
                   for f in ctx.frames],
        "vlm": {fid: vlm[fid] for fid in sorted(vlm) if any(f.frame_id == fid for f in ctx.frames)},
        "config": cfg.to_dict(),
    }
    return content_key(json.dumps(payload, sort_keys=True, default=str),
                       EXTRACTION_VERSION, SCHEMA_VERSION, "deterministic-v1")


def extract(ev: Evidence, alignments: list, vlm_payload: dict | None = None,
            cfg: SemanticConfig | None = None, use_cache: bool = True) -> SemanticResult:
    """Run semantic extraction across every segment, then reconcile.

    The document is not mutated: the caller decides what to keep. `alignments` is
    the aligner's output, passed in rather than recomputed, so there is exactly
    one definition of which frames are relevant to which sentence.
    """
    cfg = cfg or SemanticConfig()
    started = time.perf_counter()
    frames_by_id = {f.frame_id: f for f in ev.visual}
    snippets_by_id = {s.id: s for s in ev.snippets}
    vlm = _vlm_by_frame(vlm_payload)
    contexts = build_contexts(ev, alignments, frames_by_id, snippets_by_id)

    cache = JsonCache("semantic") if use_cache else None
    produced: list[Item] = []
    observations: list[dict] = []
    calls = misses = 0
    truncated = 0

    for ctx in contexts:
        def _run(c: SegmentContext = ctx) -> dict:
            its, obs = _extract_segment(c, ev, frames_by_id, snippets_by_id, vlm, cfg)
            if len(its) > cfg.max_items_per_segment:
                # Truncate in confidence order and say so. A silent cap would let
                # a dense dashboard quietly drop the whole tail of a segment.
                its.sort(key=lambda i: -i.confidence)
                dropped = len(its) - cfg.max_items_per_segment
                nonlocal truncated
                truncated += dropped
                its = its[:cfg.max_items_per_segment]
                obs = obs + [{"segment_id": c.segment.id, "classification": "ambiguous",
                              "reason": f"{dropped} candidate item(s) dropped by the per-segment cap",
                              "text": "", "frame_id": None}]
            return {"items": [i.to_dict() for i in its], "observations": obs}

        if cache is not None:
            key = _segment_cache_key(ctx, cfg, vlm)
            payload, hit = cache.get_or_call(key, _run)
        else:
            payload, hit = _run(), False
        calls += 1
        misses += 0 if hit else 1
        observations.extend(payload["observations"])
        produced.extend(Item.from_dict(d, "semantic.item") for d in payload["items"])

    # 1. enrich what the transcript pass already found, 2. add what only the
    # visual layer knows. Enrichment first so a spoken figure is never shadowed
    # by a screen reading of the same value — they merge into one item anyway,
    # but the spoken provenance is the creator's claim and is listed first.
    items = list(ev.items)
    items = _enrich_transcript_items(items, contexts, frames_by_id, vlm, cfg)
    produced = _adopt_subjects(produced, items)
    items, folded = merge_items(items, produced)
    attach_items_to_segments(ev, items, alignments)
    conflicts = detect_conflicts(items, ev)
    conflicts.extend(vlm_disagreements(items))

    classes: dict[str, int] = {}
    for o in observations:
        classes[o.get("classification", "?")] = classes.get(o.get("classification", "?"), 0) + 1
    srcs: dict[str, int] = {}
    for i in items:
        for p in i.provenance:
            srcs[p.source] = srcs.get(p.source, 0) + 1

    result = SemanticResult(
        items=items, conflicts=conflicts, observations=observations,
        metrics={
            "segments_processed": len(contexts),
            "observations_classified": len(observations),
            "observations_by_class": dict(sorted(classes.items())),
            "items_total": len(items),
            "items_by_type": _count(items, lambda i: i.type),
            "items_by_importance": _count(items, lambda i: i.importance or "unassessed"),
            "items_by_source": dict(sorted(srcs.items())),
            "screen_only_items": sum(
                1 for i in items
                if any(p.source in ("ocr", "vlm") for p in i.provenance)
                and not any(p.source == "transcript" for p in i.provenance)),
            "cross_modal_items": sum(
                1 for i in items
                if any(p.source == "transcript" for p in i.provenance)
                and any(p.source in ("ocr", "vlm") for p in i.provenance)),
            "items_folded_by_dedup": folded,
            "items_dropped_by_cap": truncated,
            "conflicts_detected": len(conflicts),
            "model_calls": 0,
            "extraction_calls": calls,
            "cache_hits": calls - misses,
            "cache_hit_rate": round((calls - misses) / calls, 4) if calls else None,
            "wall_s": round(time.perf_counter() - started, 3),
            "config": cfg.to_dict(),
        })
    return result


def _count(items: list[Item], key) -> dict:
    out: dict[str, int] = {}
    for i in items:
        k = key(i)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def _enrich_transcript_items(items: list[Item], contexts: list[SegmentContext],
                             frames_by_id: dict[str, VisualFrame], vlm: dict,
                             cfg: SemanticConfig) -> list[Item]:
    """Add a subject, an importance, and cross-modal provenance to spoken figures.

    The figures themselves were produced by the transcript pass, which owns the
    question "what was literally said". What this stage adds is the question the
    pass deliberately deferred: what the figure is *about*, and whether the screen
    was showing the same thing at the time. A spoken `32GB` and an on-screen
    `32GB` become one item with two citations — which is what "corroborated" has
    to mean in an artifact, as opposed to two items that happen to match.
    """
    speech_of: dict[str, str] = {}
    frames_of: dict[str, list[VisualFrame]] = {}
    for ctx in contexts:
        for s in ctx.snippets:
            speech_of[s.id] = ctx.speech
            frames_of[s.id] = ctx.by_snippet_frames.get(s.id, [])

    for it in items:
        if it.type not in ("number",):
            continue
        spoken = [p for p in it.provenance if p.source == "transcript"]
        if not spoken:
            continue
        snip_text = it.verbatim or ""
        if snip_text and it.subject is None:
            needle = it.display or it.value or ""
            at = snip_text.find(needle)
            it.subject = subject_for(snip_text, at, len(needle)) if at > 0 else None
        frames = frames_of.get(spoken[0].ref, [])
        rx = value_pattern(it.value, it.unit)
        supporting = [f for f in frames
                      if rx is not None and rx.search(f.ocr_text or "")
                      and f.ocr_confidence is not None
                      and f.ocr_confidence >= cfg.policy.base_unlabelled_screen]
        have = {(p.source, p.ref) for p in it.provenance}
        for f in supporting[:5]:
            if ("ocr", f.frame_id) in have:
                continue
            # Cite the literal substring the frame shows, not the transcript's
            # rendering of it. A citation that quotes text the pixels did not
            # display is the failure `ocr-text-not-in-frame` exists to catch, and
            # it would fire on `32GB` cited against a frame that read `32 GB`.
            found = rx.search(f.ocr_text or "")
            it.provenance.append(ocr_provenance(
                f, found.group(0) if found else (it.display or it.value),
                confidence=float(f.ocr_confidence or 0.0),
                read_method=f.ocr_read_method or "native"))
        cross = bool(supporting)
        it.confidence = cfg.policy.apply(
            cfg.policy.base_transcript, cross_modal=cross, hedged=(it.status == "hedged"))
        if it.importance is None:
            it.importance = "high" if it.subject else "medium"
        if it.semantic_role is None:
            # A spoken figure is never interface furniture: nobody narrates a
            # progress bar. The role is read from the shape of what was said —
            # money is a price, a unit is a metric, and a bare figure the speaker
            # gave a subject to is a count. The report-visible gate downstream
            # keys on this, so a spoken figure must be classifiable or it would
            # be filtered by the same rule that removes UI noise.
            said = f"{it.subject or ''} {it.verbatim or ''}"
            role = _unit_role(it.value, it.unit, it.display or "")
            if role is None and _looks_money(said):
                role = "price"
            elif role is None and it.unit:
                role = "metric"
            elif role is None and it.subject:
                role = "count"
            it.semantic_role = role or "unknown"
        if cross:
            existing = it.notes
            extra = f"also shown on {', '.join(f.frame_id for f in supporting[:3])}"
            it.notes = f"{existing}; {extra}" if existing else extra
    return items






