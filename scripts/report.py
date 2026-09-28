"""`evidence.json` → `REPORT.md`. A renderer, not a writer.

The product used to end at a prompt: a model read a transcript and composed prose
with nothing to check it against, so every fact in the output was an assertion and
a fabricated figure looked exactly like a verified one. This module is the other
end of that. It takes a validated `Evidence` and produces a document in which
**every sentence is assembled out of fields that are already in the evidence** —
an item's own `value`, `display`, `qualifier`, `verbatim`, `notes`, and the
timestamps its provenance resolves to — and nothing else.

That constraint is the whole design, and it is why this is not a summariser:

* **It cannot introduce a fact.** A number reaches the report because an item
  carries it; a name because an item carries it; a timestamp because a snippet or
  a frame has one. There is no free-text slot, so there is nothing for a
  hallucination to enter through. "No unsupported claims" is therefore a property
  of the code path rather than a check run afterwards — `report_validate.py`
  still checks it, because a property nobody tests is a property nobody has.
* **It preserves source distinctions** because provenance decides the wording:
  a transcript item reads as the creator's statement, an `ocr` item as text on
  screen, a `vlm` item as a model's reading of a frame. None of the three can be
  silently rewritten into another, and the tag is rendered inline.
* **It abstains rather than fills.** A section with no evidence is not rendered
  at all. A hedged figure keeps its hedge. An unreadable chart stays unreadable.
  A conflict stays open, and the report says so in neutral language instead of
  picking a winner.

Length is derived from the evidence, not chosen. `derive_budget` reads the
duration, the segment count, the item count, how many items are marked
important, how dense the items are per segment, and how much visual evidence
there is — so an eight-minute talking head does not become four thousand words
and a dense forty-five-minute lecture is not truncated to fit a target set for
the first one.

Anti-repetition is structural. Every item is assigned exactly one *home* in the
document (`_assign_homes`), and a section renders the items filed with it. An
item that becomes a Key Takeaway is therefore not also a paragraph in Detailed
Notes: cross-reference is allowed as a pointer, restatement as a second sentence
is not. That is what the duplicate-8-gram metric in `report_validate.py` is
measuring, and it is low here because it was designed out rather than tuned out.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from evidence_model import Evidence
from vlm import MAX_MODEL_CONFIDENCE

# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #

#: Where a fact came from, in the words a reader needs. The evidence document
#: distinguishes five provenance sources; a report that collapses them into one
#: undifferentiated voice has thrown away the distinction the whole layer exists
#: to keep, so the tag is rendered next to every time-anchored fact.
SOURCE_TAGS = {
    "transcript": "narration",
    "ocr": "on-screen text",
    "frame": "frame",
    "vlm": "model reading",
    "metadata": "metadata",
}

#: Item types that are *narrative*: what was said, asserted, defined or claimed.
#: These are the only items eligible for a sentence in Detailed Notes, and the
#: only ones a takeaway may abstract.
NARRATIVE_TYPES = frozenset({"claim", "topic", "definition", "claim_metric"})

ENTITY_TYPES = frozenset({"person", "organization"})
THING_TYPES = frozenset({"product", "model", "tool", "paper", "dataset"})
#: A named thing the extractor could not type further. It gets its own list
#: rather than joining `THING_TYPES` because "Obsidian" under the heading
#: *Products, Models & Tools* asserts something the evidence does not — that it is
#: a product or a model — while "Obsidian, named twice; type not established"
#: asserts only what was actually seen.
GENERIC_TYPES = frozenset({"entity"})
VISUAL_TYPES = frozenset({"relationship", "visual_finding"})

#: The lower bound on the tag vocabulary. A "creator" is the channel's speaker;
#: "narration" is the evidence source, and the report states the second because
#: the first is an inference the document does not license.

_NUM_TOKEN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
_HEDGE_RE = re.compile(r"hedge[^']*'([^']+)'", re.I)


# --------------------------------------------------------------------------- #
# budget
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Budget:
    """An evidence-derived ceiling, not a target picked in advance.

    `words` is the number the report is measured against; the per-section caps
    are what actually keeps a single dense segment from eating the document. Both
    come from the same derived `weight`, so a video with more evidence gets a
    longer report and a video with the same evidence arranged differently does
    not.
    """

    level: str
    words: int
    max_takeaways: int
    max_items_per_segment: int
    max_entities: int
    max_numbers: int
    max_quotes: int
    max_steps: int
    max_comparisons: int
    max_links: int
    floor: int
    ceiling: int

    def to_dict(self) -> dict:
        return {"level": self.level, "target_words": self.words,
                "max_takeaways": self.max_takeaways,
                "max_items_per_segment": self.max_items_per_segment,
                "max_entities": self.max_entities, "max_numbers": self.max_numbers,
                "max_quotes": self.max_quotes, "max_steps": self.max_steps,
                "max_comparisons": self.max_comparisons, "max_links": self.max_links,
                "floor_words": self.floor, "ceiling_words": self.ceiling}


#: guidance ranges. They bound the *shape* of a report; the evidence decides
#: where inside them a particular video lands.
_LEVELS = {
    "brief":     {"scale": 0.45, "floor": 300,  "ceiling": 700,
                  "takeaways": 4, "segment": 3, "entities": 6, "numbers": 8,
                  "quotes": 1, "steps": 5, "comparisons": 3, "links": 5},
    "balanced":  {"scale": 1.00, "floor": 800,  "ceiling": 1800,
                  "takeaways": 6, "segment": 6, "entities": 10, "numbers": 20,
                  "quotes": 3, "steps": 10, "comparisons": 6, "links": 10},
    "deep":      {"scale": 1.80, "floor": 1800, "ceiling": 3500,
                  "takeaways": 8, "segment": 12, "entities": 18, "numbers": 40,
                  "quotes": 5, "steps": 20, "comparisons": 10, "links": 20},
}

DETAIL_LEVELS = tuple(_LEVELS)


def derive_budget(ev: Evidence, level: str = "balanced") -> Budget:
    """Size the document from what the evidence actually contains.

    The inputs are the ones that change how much there is to say: runtime (a
    long video has more segments), how many segments, how many items, how many of
    those the extractor marked important, how dense the items are inside a
    segment (a segment with one item and a segment with forty are not the same
    thing to report), and how much visual evidence exists — which is the input
    most likely to be missing, and therefore the one whose absence should most
    visibly shorten the report.
    """
    spec = _LEVELS.get((level or "balanced").lower(), _LEVELS["balanced"])
    items = ev.items
    n_items = len(items)
    n_seg = max(1, len(ev.segments))
    n_vis = sum(1 for i in items if i.type in VISUAL_TYPES)
    n_frames_read = sum(1 for f in ev.visual if f.vlm_status == "ok" or f.ocr_status == "ok")
    n_important = sum(1 for i in items if i.importance == "high")
    density = n_items / n_seg

    weight = (0.6 * n_items + 4.0 * len(ev.segments) + 14.0 * n_vis
              + 2.0 * n_frames_read + 1.5 * n_important + 6.0 * density)
    # The floor is guidance about what a *real* video of this size tends to
    # produce, not a minimum to pad up to. A 90-second fixture and a video with
    # three extractable facts both get what they have; inflating either to hit a
    # word count is the padding this module exists to remove.
    words = max(1, min(spec["ceiling"], int(120 + weight * spec["scale"])))
    return Budget(
        level=spec is _LEVELS["balanced"] and "balanced" or level,
        words=words,
        max_takeaways=spec["takeaways"], max_items_per_segment=spec["segment"],
        max_entities=spec["entities"], max_numbers=spec["numbers"],
        max_quotes=spec["quotes"], max_steps=spec["steps"],
        max_comparisons=spec["comparisons"], max_links=spec["links"],
        floor=spec["floor"], ceiling=spec["ceiling"])


# --------------------------------------------------------------------------- #
# provenance rendering
# --------------------------------------------------------------------------- #

def _fmt_t(t: float | None) -> str | None:
    if t is None:
        return None
    t = float(t)
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


class Refs:
    """Turns an item's provenance into the compact markers a reader sees.

    The internal model keeps a full `Provenance` per item — source, ref,
    confidence, read method, the literal OCR text. None of that belongs in a
    report. What does is a timestamp and the name of the modality it came from,
    and `report_validate.py` checks the timestamps resolve, so the marker is a
    pointer rather than a decoration.
    """

    def __init__(self, ev: Evidence):
        self._snips = {s.id: s for s in ev.snippets}
        self._frames = {f.frame_id: f for f in ev.visual}
        self.times: set[float] = set()
        for s in ev.snippets:
            self.times.add(float(s.start_s))
        for f in ev.visual:
            if f.t_start is not None:
                self.times.add(float(f.t_start))
        for sg in ev.segments:
            if sg.start_s is not None:
                self.times.add(float(sg.start_s))

    def time_of(self, source: str, ref: str) -> float | None:
        if source == "transcript":
            s = self._snips.get(ref)
            return float(s.start_s) if s else None
        if source in ("frame", "ocr", "vlm"):
            f = self._frames.get(ref)
            return float(f.t_start) if f is not None and f.t_start is not None else None
        return None

    def markers(self, item) -> list[str]:
        """`["[04:31 · narration]"]` — one per distinct (time, source) pair.

        Deduplicated because an item corroborated by three modalities at the same
        instant is one fact, and three identical timestamps next to it read as
        three separate findings.
        """
        out: list[str] = []
        seen: set[tuple] = set()
        for p in item.provenance or []:
            tag = SOURCE_TAGS.get(p.source, p.source)
            t = self.time_of(p.source, p.ref)
            if t is None:
                out.append(f"[{tag}]")
                if ("meta", tag) not in seen:
                    seen.add(("meta", tag))
                continue
            key = (round(t, 1), tag)
            if key in seen:
                continue
            seen.add(key)
            out.append(f"[{_fmt_t(t)} · {tag}]")
        return out

    def source_tags(self, item) -> list[str]:
        return [SOURCE_TAGS.get(p.source, p.source) for p in (item.provenance or [])]

    def grounded_label(self, item) -> str | None:
        """The item's `subject`, but only if the surface it came from contains it.

        The extractor's own rule is that a figure with no grounded label is not
        given a subject. This is that rule enforced again at the point of
        rendering, and it is not redundant: a subject is set at extraction time
        from whatever was nearest, and "automatically" is a perfectly good word
        from the sentence "$1 is granted automatically" that says nothing about
        what the dollar is for.

        The test is containment, not plausibility. A label that appears in the
        frame's own read is what the screen said, however garbled the screen's
        OCR made it — second-guessing that would be the report inventing a
        judgement the evidence layer deliberately did not make. A label that
        appears nowhere on the surface the figure was read from is not a label.
        """
        subject = (item.subject or "").strip()
        if not subject:
            return None
        visual: list[str] = []
        spoken: list[str] = []
        for pr in item.provenance or []:
            if pr.source in ("frame", "ocr", "vlm"):
                f = self._frames.get(pr.ref)
                if f is not None:
                    visual.append(f.ocr_text or "")
                    visual.append(f.vlm_caption or "")
            elif pr.source == "transcript":
                s = self._snips.get(pr.ref)
                if s is not None:
                    spoken.append(s.text)
        # When the figure is also on screen, the label has to be on the screen.
        # "Every sign up automatically gets $1" grounds the word "automatically"
        # in the speech, and grounding it there is exactly the mistake: the
        # sentence says the credit is automatic, not that the dollar is the
        # automatic thing. Speech is the fallback surface only when there is no
        # visual one at all.
        surfaces = [s for s in visual if s] or [s for s in spoken if s]
        if not surfaces:
            return None
        if any(_label_grounded(subject, s) for s in surfaces):
            return subject
        return None

    def ref_text(self, item, limit: int = 3) -> str:
        m = self.markers(item)
        return " ".join(m[:limit])


# --------------------------------------------------------------------------- #
# blocks and sections
# --------------------------------------------------------------------------- #

@dataclass
class Block:
    """One rendered line, and everything it rests on.

    `evidence_ids` is the binding: every string in `text` came out of one of
    those items, and `report_validate.py` re-derives that claim by checking the
    rendered tokens against the evidence document. A block with no evidence ids
    is a template sentence — allowed only where it states a fact about the report
    itself, which is declared in the report's own meta block rather than invented.
    """

    text: str
    evidence_ids: list[str] = field(default_factory=list)
    refs: str = ""
    kind: str = "para"

    def to_dict(self) -> dict:
        d: dict = {"text": self.text, "kind": self.kind}
        if self.evidence_ids:
            d["evidence_ids"] = list(self.evidence_ids)
        if self.refs:
            d["refs"] = self.refs
        return d


@dataclass
class Section:
    key: str
    title: str
    level: int = 2
    blocks: list[Block] = field(default_factory=list)
    subsections: list["Section"] = field(default_factory=list)
    note: str | None = None
    #: "prose" joins the blocks into paragraphs; "list" renders one bullet each.
    #: The distinction is not cosmetic: Detailed Notes is read as a narrative and a
    #: bullet per claim turns it into an index, while Extracted Details *is* an
    #: index and prose would pad it.
    style: str = "list"

    @property
    def evidence_ids(self) -> list[str]:
        out: list[str] = []
        for b in self.blocks:
            out.extend(b.evidence_ids)
        for s in self.subsections:
            out.extend(s.evidence_ids)
        seen: set[str] = set()
        return [i for i in out if not (i in seen or seen.add(i))]

    def word_count(self) -> int:
        n = len(self.title.split())
        for b in self.blocks:
            n += len(b.text.split())
        for s in self.subsections:
            n += s.word_count()
        return n

    def to_dict(self) -> dict:
        d: dict = {"key": self.key, "title": self.title, "level": self.level,
                   "style": self.style}
        if self.blocks:
            d["blocks"] = [b.to_dict() for b in self.blocks]
        if self.subsections:
            d["subsections"] = [s.to_dict() for s in self.subsections]
        if self.note:
            d["note"] = self.note
        if self.evidence_ids:
            d["evidence_ids"] = self.evidence_ids
        return d


@dataclass
class Report:
    title: str
    subtitle: str = ""
    #: One line under the title, only when the run was degraded or failed. Kept
    #: out of the section list so it cannot be indexed, duplicated or dropped by
    #: any of the section logic.
    banner: str = ""
    mode: str = "FULL"
    mode_reason: str = ""
    sections: list[Section] = field(default_factory=list)
    budget: Budget | None = None
    meta: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def evidence_ids(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for s in self.sections:
            for i in s.evidence_ids:
                if not (i in seen or seen.add(i)):
                    out.append(i)
        return out

    def word_count(self) -> int:
        n = len(self.title.split()) + len(self.subtitle.split())
        for s in self.sections:
            n += s.word_count()
        return n

    def to_dict(self) -> dict:
        return {
            "schema": "yt-report/report/1",
            "title": self.title, "subtitle": self.subtitle, "banner": self.banner,
            "mode": self.mode, "mode_reason": self.mode_reason,
            "budget": self.budget.to_dict() if self.budget else None,
            "meta": self.meta,
            "word_count": self.word_count(),
            "sections": [s.to_dict() for s in self.sections],
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# home assignment — the anti-repetition mechanism
# --------------------------------------------------------------------------- #

def _rank(item) -> tuple:
    """Ordering for "which of these matters most". Importance first, then how
    many independent sources back it, then confidence, then id — so the ordering
    is total and reproducible rather than dependent on document order."""
    corroboration = len({p.source for p in (item.provenance or [])})
    return ({"high": 0, "medium": 1, "low": 2}.get(item.importance, 1),
            -corroboration, -(item.confidence or 0.0), item.id)


def _norm_key(text: str) -> str:
    """Case- and punctuation-insensitive identity for a piece of evidence text.

    Used only to collapse *the same* finding recorded more than once — a step
    said ten times in a video produces ten `step` items that are one fact. It is
    deliberately conservative: anything that differs by a word is kept, because a
    near-duplicate is more often two real facts than one fact stated twice.
    """
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _dedupe(items: list) -> tuple[list, int]:
    """Drop repeats of an identical statement, keeping the best-ranked copy.

    Returns the survivors and how many were dropped, so the count is reported
    rather than applied silently — a filter that hides its own effect is a filter
    nobody can audit.
    """
    seen: dict[str, object] = {}
    for it in sorted(items, key=_rank):
        seen.setdefault(_norm_key(it.value), it)
    kept = list(seen.values())
    kept.sort(key=_rank)
    return kept, len(items) - len(kept)


def _is_noise_number(item, refs) -> bool:
    """Should this figure be kept out of "Numbers, Pricing & Benchmarks"?

    Two gates, and the first is the one that matters.

    **The semantic gate.** `semantic_role` is the extractor's own reading of what
    the figure is *for*, computed from the speech, the label, the unit, the frame
    and the segment topic — never from how legible the digits were. A `ui` role
    is interface furniture and does not belong in a section of figures a reader is
    meant to use; an `unknown` role means nothing in the evidence says what the
    number counts, and a number whose referent nobody can supply is not a fact.
    Both are dropped here and stay in `evidence.json` and `visual[]` underneath,
    which is the whole point: the raw observation is never deleted, only withheld
    from the one section a reader treats as authoritative.

    **The structural gate**, which is the old rule and is kept because it catches
    what the semantic one cannot: a figure with no unit, read off a frame, and
    carrying no label saying what it is. A document produced before
    `semantic_role` existed has no role, and this still works on it.

    The two are combined rather than substituted because they fail differently.
    The structural gate is precise and has a false-negative problem — `21 —
    Browse all skills` has a grounded label, so it passes, and it is exactly the
    UI noise this is meant to remove. The semantic gate is what closes that, and
    it closes it because `Browse all skills` names no quantity and no claim in
    the segment is about `21`.
    """
    if item.semantic_role in ("ui", "unknown"):
        return True
    if item.semantic_role in ("benchmark", "price", "metric", "configuration", "count"):
        return False
    # No role recorded: an older document, or a figure from a source this
    # extractor did not classify. Fall through to the structural rule, which is
    # the conservative answer for a field that is not there.
    if item.unit:
        return False
    if refs.grounded_label(item):
        return False
    return bool(item.provenance) and all(p.source == "ocr" for p in item.provenance)


#: Why a piece of evidence is in the report at all. A taxonomy, not a blacklist:
#: a promotion phrase is excluded because of *what kind of thing it is*, and the
#: same word in a technical claim ("we discount the activation by 2x") is not
#: excluded, because the classification reads the whole sentence rather than
#: matching a term. A list of banned words loses to the next sentence that
#: deserves one.
_PROMOTION = (
    "subscribe", "subscription", "patreon", "sponsor", "sponsorship", "merch",
    "newsletter", "discord", "merch", "tip jar", "buy me a coffee", "patrons",
    "hit the bell", "ring the bell", "smash that", "like and subscribe",
    "link in the description", "links in the description", "in the description below",
    "use my link", "discount code", "coupon code", "promo code", "check out my",
    "support the channel", "support the show", "become a member", "membership",
    "sign up", "giveaway", "follow me on", "follow me on twitter", "@ me",
)
_BOILERPLATE = (
    "welcome back", "thanks for watching", "thank you for watching", "see you next",
    "see you in the next", "don't forget to subscribe", "if you enjoyed",
    "intro music", "outro music", "end screen", "let's get started", "let's get into it",
    "before we begin", "let's dive in", "let's dive right in", "without further ado",
    "so much for", "that's all for today", "goodbye", "bye for now",
)
#: Openers that mark a sentence as an example rather than as the claim itself.
_EXAMPLE = ("for example", "for instance", "imagine", "let's say", "say you",
            "suppose", "take a", "think of it as", "as an example")
#: Closers that mark a sentence as the point the video arrived at.
_CONCLUSION = ("so the point", "the point is", "in short", "ultimately", "that's why",
               "which is why", "in the end", "the takeaway", "which means that",
               "that's the", "put simply", "in other words", "the upshot")
#: Background: true, and not what a reader opening the report came for.
_CONTEXT = ("historically", "in the 19", "in the 20", "back in", "years ago",
            "originally", "used to be", "at the time", "the history of")
#: Sequence words. A sentence that orders actions is a procedure whatever the
#: extractor typed it as, and the extractor's own `step` type is not the only
#: shape a procedure arrives in.
_PROCEDURE = ("then run", "then restart", "step 1", "step one", "first you",
              "next you", "finally,", "after that,", "and then ")

#: What a takeaway is worth, best first. A promotion scores nothing because it is
#: excluded, not because it ranks low: ranking it low would still let it through
#: when the pool was short, which is the case the live report hit.
_ROLE_ORDER = {
    "core_idea": 0, "conclusion": 1, "benchmark": 2, "example": 3,
    "procedure": 4, "supporting_detail": 5, "context": 6,
}
#: Never a Key Takeaway, whatever the rank would say.
_NOT_TAKEAWAY = frozenset({"promotion", "boilerplate", "question"})


def classify_evidence_role(item) -> str:
    """What kind of thing this evidence item is, from its own words.

    Ten classes, decided in one pass over the text the speaker actually said. The
    order is the order of how much the text itself settles it:

    1. **promotion / boilerplate** — the channel, not the subject. These are
       excluded rather than ranked, because a real report made a "subscribe and
       support the channel on Patreon" line its third Key Takeaway: the pool was
       short, and a ranked-but-low promotion fills a short pool.
    2. **procedure** — an imperative, the extractor's own `step` type.
    3. **question** — a sentence that ends in a question mark and answers none.
    4. **example / conclusion / context** — the sentence's own opener or closer.
    5. **core_idea** — a claim the speaker made and a reader would repeat.
    6. **supporting_detail** — the default, and a real answer rather than a
       failure: most of a transcript is detail, and filing it as "nothing" is how
       a good video's notes get thrown away with its subscribe button.
    """
    text = f"{item.verbatim or ''} {item.value or ''}".lower()
    if any(p in text for p in _PROMOTION):
        return "promotion"
    if any(b in text for b in _BOILERPLATE):
        return "boilerplate"
    if item.type == "step":
        return "procedure"
    body = (item.value or item.verbatim or "").strip()
    if body.endswith("?") and not any(
            a in text for a in ("is the answer", "the answer is", "because")):
        return "question"
    if any(e in text for e in _EXAMPLE):
        return "example"
    if any(c in text for c in _CONCLUSION):
        return "conclusion"
    if any(c in text for c in _CONTEXT):
        return "context"
    if any(p in text for p in _PROCEDURE):
        return "procedure"
    if item.importance == "high" or item.type == "topic":
        return "core_idea"
    return "supporting_detail"


def _takeaway_order(item) -> tuple:
    """Sort key for the takeaway pool: role first, the extractor's rank second."""
    role = classify_evidence_role(item)
    return (_ROLE_ORDER.get(role, 9), _rank(item))


def _is_thin_narrative(item) -> bool:
    """A one- or two-word gloss, or a value that is not a statement at all.

    The extractor produces `definition` items for every copula it can find, and a
    one-word result ("gravel", "partially") tells a reader nothing the speaker's
    own sentence does not already say better. The speaker's sentence is still in
    the document as the item's `verbatim`; dropping the item drops the
    *restatement* of it, not the evidence.
    """
    value = (item.value or "").strip()
    if not value:
        return True
    if not re.search(r"[A-Za-z]", value):
        return True   # a bare figure belongs to the numbers section
    if len(value.split()) < 2 and item.importance != "high":
        return True
    return False


def _assign_homes(ev: Evidence, budget: Budget, refs) -> dict[str, list]:
    """Decide which section owns which item. Runs once; everything else follows.

    Ownership is the mechanism the old report lacked. Two sections rendering the
    same item as the same sentence is the repetition that made the previous
    format unreadable, and a prompt cannot prevent it because a prompt has no
    memory of what it already wrote. Here an item is filed exactly once:

    * numbers, comparisons, steps, quotes, links, entities and visual structures
      go to their own sections and are *referenced* elsewhere, never restated;
    * narrative items are split — the strongest go to Key Takeaways, the rest to
      Detailed Notes — so a takeaway is a pointer to material the report does not
      repeat at length;
    * a conflict is rendered once, in Source & Reliability, and its members are
      marked so the sections that hold them can point rather than explain.

    Unused evidence is a normal outcome. Low-value OCR observations are the
    documented case: a report that renders every observation is a report nobody
    reads, and the metric that would reward it (`evidence_items_used`) is
    explicitly not a target.
    """
    buckets: dict[str, list] = {k: [] for k in
                                 ("takeaway", "narrative", "entity", "thing",
                                  "generic", "number", "comparison", "step",
                                  "quote", "link", "conflict")}
    visual: list = []
    for it in ev.items:
        if it.status == "unreadable" and it.type not in VISUAL_TYPES:
            continue
        if it.type == "topic" and all(p.source == "metadata" for p in (it.provenance or [])):
            # The uploader's title is a label on the video, not something the video
            # discussed. It is already the report's H1, and repeating it as a
            # finding is the cheapest kind of padding there is.
            continue
        if it.type in NARRATIVE_TYPES:
            if not _is_thin_narrative(it):
                buckets["narrative"].append(it)
        elif it.type in ENTITY_TYPES:
            buckets["entity"].append(it)
        elif it.type in THING_TYPES:
            buckets["thing"].append(it)
        elif it.type in GENERIC_TYPES:
            buckets["generic"].append(it)
        elif it.type == "number":
            # A metadata-sourced number is a property of the video file, not a
            # figure the video discusses. The duration belongs in the header; in
            # the numbers table it reads as though the speaker quoted it.
            if not _is_noise_number(it, refs) and any(p.source != "metadata"
                                                for p in (it.provenance or [])):
                buckets["number"].append(it)
        elif it.type == "comparison":
            buckets["comparison"].append(it)
        elif it.type == "step":
            buckets["step"].append(it)
        elif it.type == "quote":
            buckets["quote"].append(it)
        elif it.type in ("link", "resource"):
            buckets["link"].append(it)
        elif it.type in VISUAL_TYPES:
            visual.append(it)
        # `claim_metric` is a pointer to a claim, not a fact of its own; it is
        # rendered with the claim it supports rather than as a line.

    for key in ("entity", "thing", "number", "comparison", "step", "quote", "link"):
        buckets[key], _ = _dedupe(buckets[key])
    for v in buckets.values():
        v.sort(key=_rank)
    visual, _ = _dedupe(visual)
    buckets["visual"] = visual

    conflict_members = {i for c in ev.conflicts for i in c.item_ids}
    for key in ("number", "comparison"):
        buckets[key] = [i for i in buckets[key] if i.id not in conflict_members]

    # Split the narrative pool. Takeaways are the strongest items across the
    # whole document; everything left stays in Detailed Notes. Capping the
    # takeaway count at roughly a third of the pool is what stops a
    # claim-heavy video from having its narrative section emptied out to
    # populate a bullet list.
    #
    # The pool is ordered by *what the item is* before it is cut, and promotion,
    # boilerplate and unanswered questions are dropped rather than ranked: a
    # ranked promotion still fills a short pool, which is exactly how a channel
    # plug became a Key Takeaway in a real report. They stay in Detailed Notes,
    # where a reader who wants the whole transcript still finds them.
    pool = sorted(buckets["narrative"], key=_takeaway_order)
    keep = [i for i in pool if classify_evidence_role(i) not in _NOT_TAKEAWAY]
    n_take = min(budget.max_takeaways, max(1, len(keep) // 3)) if keep else 0
    chosen = keep[:n_take]
    taken = {id(i) for i in chosen}
    buckets["takeaway"] = chosen
    buckets["narrative"] = [i for i in buckets["narrative"] if id(i) not in taken]
    return buckets


# --------------------------------------------------------------------------- #
# item → sentence
# --------------------------------------------------------------------------- #

def _role_text(item) -> str | None:
    """What the source said about an entity, or None.

    `qualifier` only. Never `notes` — in this model `notes` is the *extractor's*
    commentary ("named this segment; typed surrounding vocabulary not
    disambiguated"), and putting it in the report would make the report narrate
    its own extraction, which is both a second voice the reader cannot source and
    the single largest source of repeated text in the document: the same
    acceptance rationale appears next to every entity the pass accepted. A
    filtering list of known boilerplate phrases was tried first and is the wrong
    shape — it loses to the next rationale the extractor invents.
    """
    q = (item.qualifier or "").strip()
    if item.type in GENERIC_TYPES:
        # An untyped name has no qualifier to quote, and the list it appears
        # under is already titled "Other Named Things" — so what the row adds is
        # the fact that the type was *not* established, which is the honest
        # content of the row and is otherwise left implicit.
        return "type not established"
    return _fragment(q) or None


def _sentence(text: str) -> str:
    """Evidence text, turned into a sentence. No rewriting beyond punctuation.

    Capitalising the first character and terminating the last is the whole of
    it. Paraphrase is where a renderer starts becoming a writer, and a writer
    here is a fabrication risk with a nicer font.
    """
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return ""
    if not t[0].isupper():
        t = t[0].upper() + t[1:]
    if t[-1] not in ".!?":
        t += "."
    return t


def _fragment(text: str) -> str:
    """An inline clause — a subject, a qualifier — with no capitalisation forced.

    `_sentence` is right for a statement that starts a line and wrong for a phrase
    that sits after an em dash, where capitalising turns "retriever context" into
    something that reads as the name of a thing.
    """
    return re.sub(r"\s+", " ", (text or "").strip()).rstrip(".")


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def _hedge_word(item) -> str | None:
    m = _HEDGE_RE.search(item.notes or "")
    return m.group(1) if m else None


def _number_block(item, refs: str, label: str | None) -> Block:
    """One figure, with everything that qualifies it still attached.

    The rules this exists to hold: the exact value and unit survive; the subject
    the figure is about survives; a hedge survives as a hedge ("could cost $500"
    never renders as "$500"); an unreadable value renders as unreadable rather
    than as an omission that a reader would fill in.
    """
    display = item.display or item.value
    if item.status == "unreadable":
        text = f"**not legible** — the value could not be read from {_fragment(item.subject) or 'the frame'}"
        return Block(text, [item.id], refs, kind="number")
    context: list[str] = []
    if label:
        context.append(_fragment(label))
    elif item.subject:
        # The referent is stated nowhere the figure was read. Saying so is
        # better than printing a word that looks like a label and is not, and
        # better than dropping the figure, which is a real reading.
        context.append("referent not labelled on the source")
    if item.qualifier:
        context.append(item.qualifier.strip().rstrip("."))
    hedge = _hedge_word(item)
    if item.status == "hedged":
        lead = f"given as a possibility ({hedge})" if hedge else "given as a possibility"
        context.insert(0, lead)
    tail = (" — " + "; ".join(context)) if context else ""
    return Block(f"**{display}**{tail}", [item.id], refs, kind="number")


def _narrative_block(item, refs: str) -> Block:
    """A claim, gloss or topic, said once, in the source's own words.

    `verbatim` is preferred over the extracted `value`, and the reason is that
    the two are not the same sentence: the extractor splits a spoken clause into
    a `subject` and a `value` ("But Google" / "is preparing to launch a
    prototype satellite"), and recombining them either way loses the connector
    that made it a sentence. `verbatim` is the whole sentence as it was said and
    is on the same provenance, so using it is strictly better grounded, not
    merely better written.

    It is used only when the value really is part of it — so a value the
    extractor assembled from elsewhere is never presented as something the
    speaker said in those words.
    """
    value = _fragment(item.value)
    verbatim = (item.verbatim or "").strip()
    if verbatim and value and value.lower() in verbatim.lower() and len(verbatim) <= 320:
        return Block(_sentence(verbatim), [item.id], refs, kind="narrative")
    if item.type == "definition" and item.subject and value:
        return Block(_sentence(f"{_fragment(item.subject)}: {value}"), [item.id], refs,
                     kind="narrative")
    return Block(_sentence(value), [item.id], refs, kind="narrative")


def _visual_block(item, refs: str) -> Block:
    """A model-derived reading of a frame, tagged so it is never read as a fact
    about the pixels. `uncertainties` do not come here — they are legibility
    findings about the run, and they belong in Source & Reliability."""
    d = item.detail or {}
    kind = str(d.get("kind") or "")
    if item.type == "relationship":
        src, dst = d.get("from"), d.get("to")
        label = d.get("label")
        arrow = f"{src} → {dst}"
        text = f"The frame shows {arrow}" + (f" ({label})" if label else "")
        if d.get("certain") is False:
            text += "; the direction of this connection was read as uncertain"
        return Block(_sentence(text), [item.id], refs, kind="visual")
    if kind == "action":
        target = d.get("target")
        text = f"On screen: {item.value}" + (f" ({target})" if target else "")
        return Block(_sentence(text), [item.id], refs, kind="visual")
    if kind == "entity":
        return Block(_sentence(f"Named on screen: {item.value}"), [item.id], refs, kind="visual")
    return Block(_sentence(item.value), [item.id], refs, kind="visual")


def _comparison_block(item, refs: str) -> Block:
    d = item.detail or {}
    a, b = d.get("subject_a") or "", d.get("subject_b") or ""
    dims = d.get("dimensions") or []
    if not a or not b:
        return Block(_sentence(f"Comparison mentioned: {item.value}"), [item.id], refs,
                     kind="comparison")
    parts: list[str] = []
    for row in dims:
        if not isinstance(row, dict):
            continue
        name = row.get("dimension") or "figure"
        va, vb = row.get("a"), row.get("b")
        if va and vb:
            parts.append(f"{name}: {a} {va}, {b} {vb}")
        elif va:
            parts.append(f"{name}: {a} {va} (no figure recorded for {b})")
        elif vb:
            parts.append(f"{name}: {b} {vb} (no figure recorded for {a})")
    head = f"**{a} vs {b}**"
    body = (" — " + "; ".join(parts)) if parts else \
        " — compared, with no figure attributed to either side"
    return Block(head + body, [item.id], refs, kind="comparison")


def _quote_block(item, refs: str) -> Block:
    """A quotation, only when a read actually backs it.

    Spoken words are verbatim by construction. Text read off a frame is not: it
    is verbatim only when the read carried verification, and a low-confidence OCR
    string rendered as a quotation is a fabricated quote wearing a source tag.
    """
    proven = any(p.source == "transcript" or p.verified_verbatim
                 for p in (item.provenance or []))
    if not proven:
        return Block("", [], "", kind="rejected-quote")
    return Block(f"> {_sentence(item.verbatim or item.value)}", [item.id], refs, kind="quote")


def _normalise_url(url: str) -> str:
    """Deterministic, lossless cleanup only.

    Trailing sentence punctuation is stripped and the scheme+host lowercased —
    both are things a reader would do anyway and neither changes where the link
    points. Path case, query order, `http` vs `https` and fragments are left
    exactly as they were: normalising those can turn a working link into a dead
    one, and a report that silently rewrites a URL is worse than one that repeats
    a messy one.
    """
    u = url.strip().rstrip(".,;:)]}'\"")
    m = re.match(r"(https?://)([^/]+)(/.*)?$", u, re.I)
    if not m:
        return u
    return f"{m.group(1).lower()}{m.group(2).lower()}{m.group(3) or ''}"


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #

def _segment_label(ev: Evidence, seg, items: list) -> str:
    """A heading for a span of the video, from material that is already there.

    A chapter title is uploader metadata and is used as-is when it exists, but
    its provenance is recorded in the timeline entry rather than presented as a
    verified section name. With no usable title the label comes from the segment's
    own strongest item, and if the segment holds nothing renderable it is not
    given a heading at all.
    """
    if seg.title and seg.title.strip():
        return seg.title.strip()
    for it in items:
        text = (it.subject or it.display or it.value or "").strip()
        if text:
            return text[:70].rstrip() + ("…" if len(text) > 70 else "")
    return f"Segment {seg.id}"


def evidence_counts(ev: Evidence, buckets: dict) -> dict:
    """The figures the report states about itself.

    One function so the Executive Summary and `REPORT.json` cannot disagree about
    how many figures there are, and so every one of them can be declared as a
    report figure — which is what lets `report_validate.py` tell a counted
    statement from an invented number instead of exempting numbers wholesale.
    """
    return {
        "figure": len(buckets["number"]),
        "comparison": len(buckets["comparison"]),
        "procedure": len({(i.detail or {}).get("group") for i in buckets["step"]}),
        "visual observation": len(buckets["visual"]),
    }


def _exec_summary(ev: Evidence, buckets: dict, refs: Refs, budget: Budget) -> Section:
    """A compressed view of the evidence, introducing nothing in it.

    Every clause is a count, a name or a heading already present in the
    document. There is no interpretive sentence here and no "the video argues
    that…" — a summary that draws a conclusion is a second reading of the same
    material, and two readings of one video disagree exactly where the reader
    most needs them to agree.
    """
    blocks: list[Block] = []
    video = ev.video or {}
    topics = [i for i in ev.items if i.type == "topic"]
    segs = [s for s in ev.segments if (s.title or "").strip()]
    named: list = []
    seen_name: set[str] = set()
    for it in buckets["entity"] + buckets["thing"]:
        key = _norm_key(it.value)
        if not key or key in seen_name:
            continue
        seen_name.add(key)
        named.append(it)
    named = named[:budget.max_entities]
    counts = evidence_counts(ev, buckets)
    have = ", ".join(f"{v} {k}" if v == 1 else f"{v} {k}s"
                     for k, v in counts.items() if v)

    first = _sentence(topics[0].value) if topics else (
        _sentence(video.get("title") or "") if video.get("title") else "")
    if first:
        blocks.append(Block(first, [topics[0].id] if topics else [],
                            refs.ref_text(topics[0]) if topics else "", kind="summary"))

    # The three things a summary is for, and which the previous version had
    # none of: what the video is centrally about, the two to four concepts that
    # carry it, and what it concluded. Every clause is a word an evidence item
    # already holds, cited to that item — this assembles, it does not write, and
    # a reader who disagrees with a clause can open the item behind it.
    core = _core_ideas(buckets["narrative"] + buckets["takeaway"], 4)
    if core:
        said = "; ".join(_core_clause(i) for i in core)
        blocks.append(Block(f"Core: {said}.",
                            [i.id for i in core],
                            " ".join(refs.ref_text(i) for i in core), kind="summary"))
    closing = [i for i in (buckets["narrative"] + buckets["takeaway"])
               if classify_evidence_role(i) == "conclusion"]
    if closing:
        lead = closing[-1]
        blocks.append(Block(f"Conclusion: {_fragment(lead.value or lead.verbatim)}.",
                            [lead.id], refs.ref_text(lead), kind="summary"))

    if segs:
        labels = "; ".join(s.title.strip() for s in segs[:8] if s.title.strip())
        if labels:
            blocks.append(Block(f"Structure: {labels}.", kind="summary"))
    if named:
        who = "; ".join(i.value for i in named[:12])
        blocks.append(Block(f"Named: {who}.", [i.id for i in named[:12]], kind="summary"))
    if have:
        blocks.append(Block(f"Extracted: {have}.", kind="summary"))
    if ev.conflicts:
        blocks.append(Block(
            f"{_plural(len(ev.conflicts), 'unresolved discrepancy', 'unresolved discrepancies')} "
            f"between sources {'is' if len(ev.conflicts) == 1 else 'are'} preserved rather than "
            f"reconciled.", kind="summary"))
    return Section("executive_summary", "Executive Summary", blocks=blocks)


def _core_clause(item) -> str:
    """One clause of the central line: the subject and the value, without the echo.

    The extractor splits "Then the sum is largest when…" into a subject and a
    value that both contain the same words, so printing both reads as a
    stutter. The value is the sentence the speaker said and the subject is the
    head of it, so when one contains the other only the longer one is printed —
    and nothing is invented to fill the gap.
    """
    value, subject = _fragment(item.value), _fragment(item.subject)
    if not subject or not value:
        return value or subject
    if subject.lower() in value.lower() or value.lower() in subject.lower():
        return value if len(value) >= len(subject) else subject
    return f"{subject} — {value}"


def _core_ideas(items: list, limit: int) -> list:
    """The few claims that are the video's subject, rather than its first topic.

    Ranked by the extractor's own importance and the item's role, then
    deduplicated on the words that carry the meaning, so a video that states its
    central claim four times yields one line rather than four. Detail, examples
    and context are never core: they are the material a core claim is made of,
    and listing them is how a summary ends up describing a video's texture
    instead of its argument.
    """
    ranked = sorted((i for i in items
                     if classify_evidence_role(i) in ("core_idea", "conclusion", "benchmark")
                     and (i.value or i.subject)),
                    key=_takeaway_order)
    out: list = []
    seen: set[str] = set()
    for i in ranked:
        key = _norm_key(i.value or i.subject or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(i)
        if len(out) >= limit:
            break
    return out


def _takeaways(buckets: dict, refs: Refs) -> Section:
    blocks: list[Block] = []
    for it in buckets["takeaway"]:
        blocks.append(_narrative_block(it, refs.ref_text(it)))
    return Section("key_takeaways", "Key Takeaways", blocks=blocks)


def _detailed_notes(ev: Evidence, buckets: dict, refs: Refs, budget: Budget) -> Section:
    """The primary narrative, organised by segment.

    Each subsection holds the narrative items filed to that segment plus the
    visual findings in it, and nothing else — the figures, comparisons, steps and
    quotes live in their own sections, which is what makes this a narrative
    rather than a second copy of the document. Segments with nothing renderable
    produce no subsection, and a section with no subsections is not rendered.
    """
    narrative = {i.id: i for i in buckets["narrative"]}
    by_seg: dict[str, list] = {}
    for seg in ev.segments:
        for iid in seg.item_ids:
            if iid in narrative:
                by_seg.setdefault(seg.id, []).append(narrative[iid])
    vis_by_seg: dict[str, list] = {}
    num_by_seg: dict[str, list] = {}
    for seg in ev.segments:
        members = set(seg.item_ids)
        got = [i for i in buckets["visual"] if i.id in members]
        if got:
            vis_by_seg[seg.id] = got
        nums = [i for i in buckets["number"] if i.id in members]
        if nums:
            num_by_seg[seg.id] = nums

    subs: list[Section] = []
    for seg in ev.segments:
        items = by_seg.get(seg.id, [])[:budget.max_items_per_segment]
        vis = vis_by_seg.get(seg.id, [])[:4]
        # A segment with nothing but figures gets no narrative subsection: the
        # Numbers section already holds it, and a second copy under the segment's
        # heading is the repetition the whole home-assignment exists to prevent.
        if not items and not vis:
            continue
        blocks: list[Block] = [_narrative_block(i, refs.ref_text(i)) for i in items]
        blocks = [b for b in blocks if b.text]
        if items:
            # The figures this segment discussed, in one clause. The number's
            # primary home is the index below; this is the "supporting details"
            # a narrative section owes its reader, and it is bounded so a dense
            # segment cannot reprint the whole numbers table.
            # Only *measured* quantities get a clause in the narrative. A unitless
            # read off a slide — `02`, `7 (Sundays)`, `18 (Terminals)` — is
            # almost always ordinal furniture that OCR read faithfully and that
            # means nothing as a sentence, and a narrative that lists them reads
            # as noise even though every one of them is honestly sourced. The
            # index still carries all of them, with their markers, because a read
            # that happened is a fact about the screen.
            nums = [n for n in num_by_seg.get(seg.id, []) if n.unit][:4]
            if nums:
                # A hedged figure has to carry its hedge wherever it is restated.
                # The numbers table does ("given as a possibility (up to)"), and
                # this line is the same fact said a second time in the same
                # document — so printing `128k` bare here while the table prints
                # it hedged makes the report state two different strengths for
                # one reading, and the flat one is the one in the narrative.
                # Found by the report gate's `hedge-dropped` check on a real
                # video, not by reading the output.
                clauses = "; ".join(
                    f"{_hedge_word(n) + ' ' if _hedge_word(n) else ''}"
                    f"{n.display or n.value}"
                    + (f" ({_fragment(lbl)})"
                       if (lbl := refs.grounded_label(n)) else "")
                    for n in nums)
                blocks.append(Block(_sentence(f"Figures given here: {clauses}"),
                                    [n.id for n in nums],
                                    " ".join(refs.markers(nums[0])[:1]), kind="support"))
        blocks.extend(_visual_block(i, refs.ref_text(i)) for i in vis)
        # The key *is* the segment id. `_timeline` selects the sections it covers
        # by this value, and a decorated key here silently renders an empty
        # timeline on every input rather than raising.
        subs.append(Section(seg.id, _segment_label(ev, seg, items or vis),
                            level=3, blocks=blocks, style="prose"))
    return Section("detailed_notes", "Detailed Notes", subsections=subs)


def _supporting_items(ev: Evidence, name: str) -> list:
    """Items whose own text mentions `name` — the "relevant evidence" pointer.

    A product name on its own tells a reader nothing about why it is in the
    report. The figure it was measured by does, and that figure is already in the
    document, so the pointer is a lookup rather than a new claim.
    """
    low = name.lower()
    if len(low) < 3:
        return []
    out = []
    for i in ev.items:
        if i.type in (THING_TYPES | ENTITY_TYPES | GENERIC_TYPES):
            continue
        if not any(p.source in ("transcript", "ocr", "vlm", "frame")
                   for p in (i.provenance or [])):
            # A pointer to something with no timestamp points nowhere. A channel
            # name citing the metadata field it came from is not "evidence" to a
            # reader looking for where a claim was made.
            continue
        hay = " ".join(str(x) for x in (i.subject, i.value, i.display, i.qualifier) if x)
        if low in hay.lower():
            out.append(i)
    out.sort(key=_rank)
    return out


def _entity_section(ev: Evidence, items: list, refs: Refs, title: str,
                    budget: Budget) -> Section:
    blocks: list[Block] = []
    seen: set[str] = set()
    for it in items[:budget.max_entities]:
        key = it.value.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        role = _role_text(it)
        support = _supporting_items(ev, it.value)
        bits = [role] if role else []
        own = set(refs.markers(it))
        for sup in support[:1]:
            sup_ref = refs.markers(sup)
            # A supporter cited at the same moment as the entity adds a second
            # identical timestamp next to the first, which reads as two
            # independent findings where there is one.
            if sup_ref and sup_ref[0] in own:
                continue
            bits.append(f"see {_supporter_phrase(sup)} {sup_ref[0]}".strip() if sup_ref
                        else f"see {_supporter_phrase(sup)}")
        text = f"**{it.value}**" + (" — " + "; ".join(bits) if bits else "")
        blocks.append(Block(text, [it.id] + [s.id for s in support[:1]],
                            refs.ref_text(it, 1), kind="entity"))
    return Section("extracted_details", "Extracted Details",
                   subsections=[Section("entities", title, level=3, blocks=blocks)])


def _supporter_phrase(item) -> str:
    if item.type == "number":
        return f"the {item.display or item.value} figure"
    if item.type == "comparison":
        return f"the {item.value} comparison"
    if item.type in NARRATIVE_TYPES:
        return "a statement in the narration"
    return f"a {item.type}"


def _things_section(ev: Evidence, items: list, refs: Refs, budget: Budget) -> Section:
    return _entity_section(ev, items, refs, "Products, Models & Tools", budget)


def _numbers_section(items: list, refs: Refs, budget: Budget) -> Section:
    blocks = [_number_block(i, refs.ref_text(i), refs.grounded_label(i))
              for i in items[:budget.max_numbers]]
    return Section("numbers", "Numbers, Pricing & Benchmarks", level=3, blocks=blocks)


def _steps_section(items: list, refs: Refs, budget: Budget) -> Section:
    """Actual procedures only. A single imperative sentence is not a procedure,
    and the extractor's own floor is what guarantees that — so anything arriving
    here has already passed it, and the report does not re-decide."""
    groups: dict[str, list] = {}
    for it in items[:budget.max_steps]:
        groups.setdefault(str((it.detail or {}).get("group") or ""), []).append(it)
    blocks: list[Block] = []
    for _, run in groups.items():
        run.sort(key=lambda i: ((i.detail or {}).get("index") or 0, i.id))
        for i in run:
            blocks.append(_narrative_block(i, refs.ref_text(i)))
    return Section("steps", "Steps & Workflows", level=3, blocks=blocks)


def _comparisons_section(items: list, refs: Refs, budget: Budget) -> Section:
    blocks = [_comparison_block(i, refs.ref_text(i)) for i in items[:budget.max_comparisons]]
    return Section("comparisons", "Comparisons", level=3, blocks=blocks)


def _links_section(ev: Evidence, items: list, refs: Refs, budget: Budget) -> Section:
    """URLs that exist in the evidence, deduplicated. Nothing else is a link.

    The video's own address is excluded: it is in the header, and repeating it in
    a resources list reads as though it were something the video pointed at.
    """
    own = _normalise_url(str(ev.video.get("url") or ""))
    blocks: list[Block] = []
    seen: set[str] = set()
    for it in items:
        url = _normalise_url(str(it.display or it.value))
        if not url.startswith("http") or url == own or url in seen:
            continue
        seen.add(url)
        if len(seen) > budget.max_links:
            break
        blocks.append(Block(f"<{url}>", [it.id], refs.ref_text(it, 1), kind="link"))
    return Section("links", "Resources & Links", level=3, blocks=blocks)


def _quotes_section(items: list, refs: Refs, budget: Budget) -> Section:
    blocks = [_quote_block(i, refs.ref_text(i)) for i in items]
    blocks = [b for b in blocks if b.text][:budget.max_quotes]
    return Section("quotes", "Important Quotes", level=3, blocks=blocks)


def _timeline(ev: Evidence, used_segments: set[str]) -> Section:
    """A compact label per section the report actually covers.

    Built from real segment starts. An unmeasured segment is skipped rather than
    rendered as `00:00`, which is a real timestamp belonging to a different
    moment, and a timeline whose entries are restated summaries is a second copy
    of the document in the wrong shape.
    """
    blocks: list[Block] = []
    for seg in ev.segments:
        if seg.id not in used_segments or seg.start_s is None:
            continue
        label = (seg.title or "").strip() or f"segment {seg.id}"
        blocks.append(Block(f"{_fmt_t(seg.start_s)} — {label}", kind="timeline"))
    return Section("timeline", "Timeline", blocks=blocks)


def _conflict_template(statement: str) -> str:
    """The shape of a conflict statement, with its figures removed.

    Three conflicts can be the same disagreement seen pairwise — A against B, A
    against C, B against C — and rendering each in full prints the same
    twenty-word sentence three times differing only in the digits. Grouping on
    the template keeps every member and every timestamp while saying the shape
    once, which is both shorter and easier to check: a reader can see that one
    claim disagrees with two different sources in a single line.
    """
    return re.sub(r"-?\d[\d,]*(?:\.\d+)?", " ", statement or "").strip()


def _group_conflicts(ev: Evidence) -> list:
    """(conflict, kind, groups) — conflicts clustered by statement shape.

    Yields the representative conflict alongside the group so its statement can
    be rendered once; every member of every pair in the group is still reported.
    """
    by_id = {i.id: i for i in ev.items}
    order: list = []
    bucket: dict = {}
    for c in ev.conflicts:
        members = [by_id[i] for i in c.item_ids if i in by_id]
        if not members:
            continue
        key = (c.kind, _conflict_template(c.statement))
        if key not in bucket:
            bucket[key] = (c, [])
            order.append(key)
        bucket[key][1].append(members)
    return [(bucket[k][0], k[0], bucket[k][1]) for k in order]


#: Words too ordinary to identify a referent. A "label" made only of these is a
#: word that happened to be nearby, not a name for what the figure measures.
_LABEL_STOPWORDS = frozenset("""
about above after again against along also although always among another around asked
been before being between both came come could does doing done down during each even
ever every from further give given goes going have here however into just keep kept
less like made make many more most much must never only other over said same should
since some such than that their them then there these they thing things this those
through time under until upon very want well went were what when where which while
will with within without would your they
""".split())


def _content_words(label: str) -> list[str]:
    """The words in a label that could actually identify a referent.

    Length and case carry the weight. Three-letter lowercase words are almost
    always function words or a mis-read glyph, and an all-caps word is an
    acronym — `TPU`, `GPU`, `FP8` — which is exactly the kind of short label
    that is a perfectly good referent and would otherwise be discarded.
    """
    out: list[str] = []
    for w in re.findall(r"[A-Za-z0-9]+", label or ""):
        low = w.lower()
        if w.isupper() and len(w) >= 2:
            out.append(low)
        elif len(w) >= 4 and low not in _LABEL_STOPWORDS:
            out.append(low)
    return out


def _label_grounded(label: str, surface: str) -> bool:
    """Is this label actually named on this surface?

    Overlap on content words, not literal containment. A subject is a *rephrasing*
    of what the source said — "retrieval latency on the small index" for a
    sentence reading "Retrieval latency was 120 ms on the small index" — so
    demanding the whole string appear verbatim rejects every good label. What it
    must not accept is a label whose only content word is a word of the sentence
    that has nothing to do with the figure, which is what "automatically" next to
    "$1" is.
    """
    words = _content_words(label)
    if not words:
        return False
    low = (surface or "").lower()
    return any(w in low for w in words)


def _degradation(ev: Evidence) -> tuple[str, str]:
    """What this run could and could not see, in the words a reader needs.

    A transcript-only report presented as though a visual analysis had run is the
    specific dishonesty this state exists to prevent, so it is computed from the
    artifact's own fields and stated at the top of the document rather than left
    for the reader to infer from an absence.
    """
    has_snippets = bool(ev.snippets)
    asr = str((ev.transcript or {}).get("language_source") or "")
    frames = ev.visual
    ocr_read = any(f.ocr_status == "ok" for f in frames)
    ocr_unavailable = bool(frames) and not any(
        f.ocr_status in ("ok", "no_text", "repeat") for f in frames)
    vlm_ok = any(f.vlm_status == "ok" for f in frames)
    vlm_asked = any(f.vlm_status in ("ok", "no_visual") for f in frames)
    parts: list[str] = []
    if not has_snippets:
        return "FAILED", "no usable transcript; nothing in this report is sourced from speech"
    if vlm_ok and ocr_read:
        return "FULL", "transcript, on-screen text and a model reading of selected frames"
    parts.append("transcript" + (" (from local ASR)" if asr == "asr-detected" else ""))
    if ocr_read:
        parts.append("on-screen text")
    if vlm_asked:
        parts.append("a model reading of selected frames")
    reason = " + ".join(p for p in parts if p) or "transcript"
    if not ocr_read:
        reason += "; no frame was read for on-screen text, so nothing here comes from the visuals"
    elif not vlm_ok:
        reason += "; no model reading of the frames, so diagrams, tables and charts are not interpreted"
    if ocr_unavailable and ocr_read:
        reason += "; some frames were not read"
    return "DEGRADED", reason


def _reliability(ev: Evidence, buckets: dict, mode: str, reason: str,
                 refs: Refs, budget: Budget | None = None) -> Section:
    """Provenance, degradation and open conflicts, stated plainly.

    Conflicts are reproduced here and nowhere else. The report does not choose a
    winner between two sources, does not average them, and does not quietly
    prefer the one with higher confidence: resolving a contradiction is a
    judgement about what to believe, and a report that makes it has stopped being
    a record of what the video said.

    Every count this section states is about figures the report actually carries,
    or is paired with the count it withheld. A report that announced eleven
    figures and rendered none is the failure `check_figures` now blocks on: a
    section of figures is what a reader trusts, and a count that does not match
    it is the document asserting evidence it does not contain.
    """
    # The mode itself is stated once, under the title. Repeating it here would be
    # the document's own definition of the repetition it is meant to avoid.
    blocks: list[Block] = []
    sources: dict[str, int] = {}
    for it in ev.items:
        for p in it.provenance or []:
            sources[p.source] = sources.get(p.source, 0) + 1
    if sources:
        parts = ", ".join(f"{SOURCE_TAGS.get(k, k)}: {v}" for k, v in sorted(sources.items()))
        blocks.append(Block(f"Evidence by source — {parts}.", kind="meta"))

    by_id = {i.id: i for i in ev.items}
    for c, kind, groups in _group_conflicts(ev):
        first = True
        for members in groups:
            ids = [m.id for m in members]
            pair = " vs ".join(
                f"{m.display or m.value} ({refs.ref_text(m, 1) or 'no timestamp'})"
                for m in members)
            if first:
                # The extractor's own statement names the subject and both
                # figures; it is the most informative sentence available and it
                # is repeated verbatim for every pair of the same shape, so it is
                # used once and the rest are listed compactly. The markers go
                # with it, because a conflict a reader cannot trace is the one
                # kind of claim that most needs checking.
                text = ("**Unresolved conflict (" + kind + ").** "
                        + _fragment(c.statement or pair)
                        + f" Sources: {pair}. Not resolved here.")
                first = False
            else:
                text = "**Same disagreement, other pair.** " + pair + ". Not resolved here."
            blocks.append(Block(text, ids, "", kind="conflict"))

    visual_unreadable = [i for i in buckets["visual"]
                         if str((i.detail or {}).get("kind")) == "uncertainty"]
    unreadable = [i for i in ev.items if i.status == "unreadable"
                  and i not in visual_unreadable]
    if unreadable:
        got = "; ".join(f"{i.display or i.value}"
                        + (f" ({i.subject})" if i.subject else "")
                        for i in unreadable[:8])
        blocks.append(Block(
            f"**Not legible.** {_plural(len(unreadable), 'recorded value')} could not be read "
            f"from the frame and {'is' if len(unreadable) == 1 else 'are'} not stated: "
            f"{got}.", kind="meta"))

    if visual_unreadable:
        blocks.append(Block(
            f"**Visual content not interpreted.** {_plural(len(visual_unreadable), 'part')} "
            f"of a frame could not be read: "
            + "; ".join(i.value for i in visual_unreadable[:5]) + ".", kind="meta"))

    rendered = rendered_figures(ev, buckets, budget)
    detected = [i for i in ev.items if i.type == "number"]
    withheld = [i for i in detected if i not in rendered]
    single = [i for i in rendered if {p.source for p in (i.provenance or [])} == {"ocr"}]
    if single:
        # Not a filter — a statement of weight. These strings were on a frame and
        # the read is the only evidence for them; a garbled OCR of a dense slide
        # produces faithful garbage, and a reader looking at a price table needs
        # to know which rows that applies to before trusting the row.
        blocks.append(Block(
            f"**Figures from a single read.** {len(single)} of {len(rendered)} figures "
            f"rest on one text read of one frame with nothing else naming them. They are "
            f"reported as the screen was read; the labels are the engine's and may be "
            f"garbled.", kind="meta"))
    if withheld:
        # The other half of the same invariant: a figure that was read, is in
        # evidence.json, and is not in the report is a fact the report knows about
        # and did not print. Saying so is the difference between a threshold and
        # a hole, and it is stated with the count so the two sides of the
        # invariant can be added back to the number detected.
        blocks.append(Block(
            f"**{_plural(len(withheld), 'figure')} withheld.** {len(detected)} figures were "
            f"detected but omitted because they failed the report visibility threshold — "
            f"interface furniture, or no label, unit or speech says what they count. They "
            f"remain in the evidence file and the frame reads; nothing was deleted.",
            kind="meta"))

    vlm_items = [i for i in ev.items if any(p.source == "vlm" for p in (i.provenance or []))]
    if vlm_items:
        blocks.append(Block(
            f"**Model-derived content.** {_plural(len(vlm_items), 'item')} "
            f"{'comes' if len(vlm_items) == 1 else 'come'} from a model's reading of a frame, "
            f"not from a reading of its characters. It is capped at {MAX_MODEL_CONFIDENCE} "
            f"confidence, below every text-read band, and is not verbatim.", kind="meta"))
    return Section("reliability", "Source & Reliability", blocks=blocks)


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #

def single_read_figures(ev: Evidence) -> list:
    """Figures whose only evidence is one text read of one frame.

    Named here and called from two places — the reliability note that states how
    many there are, and the declared report figures the validator checks those
    counts against — because a counted statement that is not declared is exactly
    the invented number this pipeline exists to prevent. It caught one the first
    time it was written.
    """
    return [i for i in ev.items if i.type == "number"
            and {p.source for p in (i.provenance or [])} == {"ocr"}]


def rendered_figures(ev: Evidence, buckets: dict, budget: Budget | None = None) -> list:
    """The figures the report actually prints.

    `buckets["number"]` is the pool that survived the noise gate; the section
    then truncates it to the budget, so the number of figures a reader can look
    up is the truncated length and not the pool. Both the reliability note and
    `report_validate.check_figures` read this one function, so the two cannot be
    computed differently — which is the whole defect this replaces: the note
    counted every `number` item in the document while the section printed none of
    them, and nothing compared the two.
    """
    pool = buckets.get("number") or []
    cap = budget.max_numbers if budget is not None else len(pool)
    return list(pool[:cap])


def _provenance_counts(ev: Evidence) -> dict:
    """Provenance entries per source. A statement about the document, not about
    the video, so it belongs with the report figures rather than in the prose
    sections that carry the video's own claims."""
    out: dict[str, int] = {}
    for it in ev.items:
        for p in it.provenance or []:
            out[p.source] = out.get(p.source, 0) + 1
    return out


def generate(ev: Evidence, detail: str = "balanced") -> Report:
    """Validated evidence in, a report out. Deterministic: same input, same bytes.

    Empty sections are dropped at the end rather than skipped at each call site,
    because a section becomes empty for a reason that is only visible after its
    siblings have been filled — a segment whose items all became takeaways, an
    entity list that deduped to nothing. Rendering first and pruning second means
    the pruning sees the real document.
    """
    budget = derive_budget(ev, detail)
    refs = Refs(ev)
    buckets = _assign_homes(ev, budget, refs)
    mode, reason = _degradation(ev)

    title = str(ev.video.get("title") or "Untitled video")
    bits: list[str] = []
    if ev.video.get("channel"):
        bits.append(str(ev.video["channel"]))
    if ev.video.get("upload_date"):
        bits.append(str(ev.video["upload_date"]))
    if ev.video.get("duration_s"):
        d = ev.video["duration_s"]
        bits.append(f"{int(d) // 60}:{int(d) % 60:02d}")

    details_subs: list[Section] = []
    ent = _entity_section(ev, buckets["entity"], refs, "People & Organizations", budget)
    if ent.subsections[0].blocks:
        details_subs.append(ent.subsections[0])
    th = _things_section(ev, buckets["thing"], refs, budget)
    if th.subsections[0].blocks:
        details_subs.append(th.subsections[0])
    gen = _entity_section(ev, buckets["generic"], refs, "Other Named Things", budget)
    if gen.subsections[0].blocks:
        details_subs.append(gen.subsections[0])
    for sec in (_numbers_section(buckets["number"], refs, budget),
                _steps_section(buckets["step"], refs, budget),
                _comparisons_section(buckets["comparison"], refs, budget),
                _links_section(ev, buckets["link"], refs, budget),
                _quotes_section(buckets["quote"], refs, budget)):
        if sec.blocks:
            details_subs.append(sec)

    notes = _detailed_notes(ev, buckets, refs, budget)
    used_segments = {s.key for s in notes.subsections}

    sections = [
        _exec_summary(ev, buckets, refs, budget),
        _takeaways(buckets, refs),
        notes,
    ]
    if details_subs:
        sections.append(Section("extracted_details", "Extracted Details",
                                subsections=details_subs))
    sections.append(_timeline(ev, used_segments))
    sections.append(_reliability(ev, buckets, mode, reason, refs, budget))

    kept = [s for s in sections if s.blocks or s.subsections]
    # One statement, in the first thing a reader sees, and nowhere else. A second
    # copy further down is a paragraph saying the same thing twice — the defect
    # the duplicate-8-gram check exists to catch, committed deliberately by the
    # renderer that was built to avoid it.
    banner = f"**{mode}.** {reason}." if mode != "FULL" else ""

    report = Report(title=title, subtitle=" · ".join(bits), mode=mode, mode_reason=reason,
                    sections=kept, budget=budget, banner=banner)
    report.meta = {
        "video_id": ev.video.get("id"),
        "channel": ev.video.get("channel"),
        "duration_s": ev.video.get("duration_s"),
        "schema_version": ev.schema_version,
        "detail_level": budget.level,
        # Figures about the report itself. Declared here so `report_validate.py`
        # can tell a counted statement ("71 frames were sampled") from an
        # invented one, without a blanket exemption from the number check.
        "report_figures": {
            "evidence_items": len(ev.items),
            "segments": len(ev.segments),
            "frames_sampled": len(ev.visual),
            "frames_read": sum(1 for f in ev.visual if f.ocr_status == "ok"),
            "frames_model_read": sum(1 for f in ev.visual if f.vlm_status == "ok"),
            "conflicts": len(ev.conflicts),
            "snippets": len(ev.snippets),
            **{f"count_{k.replace(' ', '_')}": v
               for k, v in evidence_counts(ev, buckets).items()},
            "figures_single_read": len(single_read_figures(ev)),
            "figures_total": sum(1 for i in ev.items if i.type == "number"),
            # The three counts the figure/reliability invariant is stated in. They
            # are declared so `check_figures` can tell a counted statement from an
            # invented one, and so a reader of `report.json` can add them up.
            "figures_rendered": len(rendered_figures(ev, buckets, budget)),
            "figures_withheld": (sum(1 for i in ev.items if i.type == "number")
                                 - len(rendered_figures(ev, buckets, budget))),
            **{f"provenance_{k}": v for k, v in _provenance_counts(ev).items()},
            # The ceiling `vlm.py` puts on a model-reported observation. Stated in
            # the report and declared here, so "0.70" is a policy figure a checker
            # can resolve rather than a number the prose invented.
            "model_confidence_cap": MAX_MODEL_CONFIDENCE,
        },
    }
    return report


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def _period(text: str) -> str:
    """Terminate a block that needs it. `text` usually already ends in a full
    stop, and adding another is the sort of small carelessness that makes a
    generated document read as machine output."""
    return text if text[-1:] in ".!?:;" else text + "."


def _render_block(b: Block) -> str:
    if b.kind == "quote":
        return b.text
    return f"{_period(b.text)} {b.refs}".rstrip()


def _render_prose(blocks: list[Block]) -> str:
    """One paragraph per subsection, with each claim's marker inline.

    A bullet per claim is an index wearing a heading, and Detailed Notes is the
    one section a reader reads straight through. Provenance stays inline so the
    paragraph remains checkable sentence by sentence.
    """
    parts = [f"{_period(b.text)} {b.refs}".rstrip() for b in blocks if b.text]
    return " ".join(parts)


def render_markdown(report: Report) -> str:
    """`REPORT.md`. The rendering is a separate pass from the planning so that
    the structure can be validated as data before a reader ever sees it."""
    out: list[str] = [f"# {report.title}", ""]
    if report.subtitle:
        out += [f"*{report.subtitle}*", ""]
    if report.banner:
        out += [f"> {report.banner}", ""]
    for s in report.sections:
        out += _render_section(s)
    return "\n".join(out).rstrip() + "\n"


def _render_section(s: Section) -> list[str]:
    if not s.blocks and not s.subsections:
        return []
    out = ["#" * s.level + " " + s.title, ""]
    if s.note:
        out += [f"_{s.note}_", ""]
    live = [b for b in s.blocks if b.text]
    if live and s.style == "prose":
        out += [_render_prose(live), ""]
    elif live:
        for b in live:
            body = _render_block(b)
            out.append(body if body.startswith(">") else f"- {body}")
        out.append("")
    for sub in s.subsections:
        if sub.blocks or sub.subsections:
            out += _render_section(sub)
    return out


def report_json(report: Report) -> dict:
    """`REPORT.json` — the report's *structure*, referencing evidence ids.

    Not a second source of truth: it holds no evidence text, only the ids of the
    items each section rendered, so the two documents cannot disagree about what
    was said. It exists so a later check can ask "which evidence backs this
    paragraph" without re-parsing markdown.
    """
    doc = report.to_dict()
    doc["evidence_index"] = report.evidence_ids
    return doc


def write_report(report: Report, artifact_dir: str | Path) -> tuple[Path, Path]:
    """Write `REPORT.md` and `REPORT.json`, atomically and in that order.

    `REPORT.md` first because it is the artifact a human opens; the JSON index is
    the derived one. A half-written report that parses as a valid short document
    is worse than no report, so both go through a temporary file and a rename.
    """
    d = Path(artifact_dir)
    md, js = d / "REPORT.md", d / "REPORT.json"
    for path, text in ((md, render_markdown(report)),
                       (js, json.dumps(report_json(report), indent=2, ensure_ascii=False) + "\n")):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    return md, js
