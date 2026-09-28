"""Typed data model for `evidence.json` — the canonical, provenance-aware record.

Everything downstream (validation, then report synthesis) reads this file and
nothing else. The model exists because the report used to be the only
representation of what a video said, so every fact in it was prose with no way
back to the snippet or frame it came from — a fabricated figure and a verified
one were indistinguishable in the artifact itself.

Three rules the shape exists to hold:

* **Provenance is mandatory.** An item with no resolvable source is an assertion,
  not a fact. The validator treats that as a blocking error, not a warning.
* **A missing value is null, never zero.** An unmeasured frame and a frame at
  t=0 are identical in a number field, and only one of them is real.
* **Hedging is data.** "could cost $500" and "costs $500" are different items, so
  epistemic status is a field rather than something buried in prose.

Extensible without a schema change: a new provenance source (`vlm`) or item type
is a vocabulary addition, and a source type only has to teach the validator how to
resolve its `ref`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# 2 changed the shape of one record: `alignment[]` went from a single `frame_id`
# to a ranked `frame_ids` list plus the span it was derived from, so a snippet
# can carry several frames and "none of them were relevant" can be said outright
# rather than by omitting the field. Bumped rather than absorbed because a
# schema-1 alignment read as schema 2 is a schema-1 `frame_id` silently becoming
# an empty frame list — a document claiming no visual correspondence where it
# had one.
# 3 added the visual layer: two item types (`relationship`, `visual_finding`)
# and a structured `vlm_observation` on each visual frame. Bumped because an
# older reader would not reject the new item types — it would drop the table
# cells, the diagram edges and the recorded uncertainties, and keep writing a
# document that looks complete. A silent narrowing of what the artifact says is
# the one degradation a reader cannot detect, so the version moves and an old
# reader is told rather than left to guess.
SCHEMA_VERSION = 3

ITEM_TYPES = frozenset({
    "topic", "person", "organization", "product", "model", "tool", "paper",
    "dataset", "number", "claim", "comparison", "step", "quote", "resource",
    "link", "definition", "claim_metric",
    # Added by the visual pass, for the two shapes OCR cannot express. A
    # `relationship` is a diagram edge or a chart ordering with both ends named;
    # a `visual_finding` is a structured visual observation that is not a value —
    # an action being demonstrated, or a part of the frame that could not be
    # read. Both are here because the alternative is encoding structure as
    # English in `value`, which survives as prose and is unrecoverable.
    "relationship", "visual_finding",
    # A named thing the extractor could not type further. Added because
    # `product` was doing two jobs: the product it had evidence for, and every
    # capitalised word it had nothing to say about. A reader cannot tell those
    # apart, and "Obsidian — product" asserts something the evidence does not.
    # `entity` is the honest residue, and it renders beside the typed lists
    # rather than inside them.
    "entity",
})

#: What a number is *for*, which is a third axis again — not how sure the read
#: was, and not how much it matters. A subscriber counter is read at 0.99 and is
#: `ui`; a benchmark bar read at 0.78 is `benchmark`. The report's numbers
#: section is gated on this, which is why it has to be a field of its own rather
#: than a shape inferred downstream: an inference made after the evidence was
#: written cannot be recorded with its reason.
SEMANTIC_ROLES = frozenset({
    "benchmark",   # a measured result, a score, a comparison figure
    "price",       # money: a subscription, a licence, a cost
    "metric",      # a quantity with a named unit: ms, GB, %, requests/s
    "configuration",  # a setting a reader would set: model id, context size
    "count",       # a count of things, with the things named
    "ui",          # interface furniture: progress, counters, chapter numbers
    "unknown",     # read, but nothing in the evidence says what it is
})

# `asserted` is what the source says outright. The rest record that it did not:
# `hedged` exists because a hedge flattened into an assertion is a fabrication
# that passes every provenance check, and `conflicting` is set on the items
# named by an unresolved conflict rather than on the conflict itself.
STATUSES = frozenset({"asserted", "hedged", "retracted", "unreadable", "conflicting"})

# `vlm` is declared now, before any VLM pass exists, so that adding one later is
# not a schema change and so a model-derived item can never be silently
# indistinguishable from a creator statement.
SOURCES = frozenset({"transcript", "frame", "ocr", "metadata", "vlm"})

# `upscale-2x` is not a reading method of different pixels — it is the same pixels
# rendered larger for the engine. It is in the enum so a reader can see that a
# citation came from a re-read, rather than assuming the frame was read at native
# size and treating the extra legibility as source detail.
READ_METHODS = frozenset({"native", "crop-3x", "crop-5x", "upscale-2x", "unknown"})

#: An OCR result's own status. Kept here rather than in `ocr.py` because the
#: validator must be able to check a document without importing the subsystem that
#: produced it — a status vocabulary that lives in the producer cannot be enforced
#: by a checker that only sees the artifact. `repeat` says an observation was
#: already recorded on an earlier frame and this frame is a pointer to it, which
#: is a statement about the index rather than about the pixels.
OCR_STATUSES = frozenset({"ok", "no_text", "repeat", "unavailable", "failed", "unreadable",
                          "skipped"})

#: A VLM pass's own outcome, for the same reason `OCR_STATUSES` lives here: the
#: validator has to be able to tell "a model looked and found nothing" from "no
#: model ran", and a status vocabulary owned by the producer cannot be enforced
#: by a checker that only sees the artifact. `malformed` is separate from
#: `failed` because they mean different things to a later run: a failed call is
#: worth retrying, a malformed one means the model answered in a shape this pass
#: cannot read, and retrying the same frame with the same prompt will do it again.
VLM_STATUSES = frozenset({"ok", "no_visual", "unavailable", "failed", "malformed", "skipped"})

#: How a snippet came to be paired with a frame. Recorded rather than inferred,
#: because "the nearest frame in time" and "the frame that shows what was just
#: said" are different claims and a reader cannot tell them apart from a score.
ALIGNMENT_METHODS = frozenset({
    "overlap", "time-nearest", "bounded-window", "ocr-assisted", "segment-context",
    "no-relevant-frame",
})

#: The confidence a pairing gets when the honest answer is "no frame here
#: corresponds to this". It is a constant rather than a computed score so that a
#: reader can see at a glance that this is the absence branch, not a weak match
#: that happened to land near zero.
NO_FRAME_CONFIDENCE = 0.1

TITLE_SOURCES = frozenset({"chapter", "model", "topic"})
#: How a frame came to exist. `refinement` was missing from this set and every
#: run that sampled one was refused by the evidence gate — 52 blocking errors on
#: an 18-minute video, all of them the same one: the sampler names its third
#: layer `refinement` and the schema had never heard of it. The four names that
#: were available are all lies for that frame: a refinement frame is not a cut
#: (`scene`), it is not on the coverage grid (`uniform`), it was not chosen
#: because OCR found something in it (`ocr_hit`), and nobody asked for it by id
#: (`queried`). Calling it `unknown` would have made the gate pass by discarding
#: the distinction the sampler draws on purpose — which frames are baseline
#: coverage and which are the extra ones the video or the speaker earned.
FRAME_KINDS = frozenset({"uniform", "refinement", "scene", "ocr_hit", "queried", "unknown"})

CONFLICT_KINDS = frozenset({"value-mismatch", "source-disagreement", "reversal"})

RESOLUTIONS = frozenset({"unresolved", "resolved"})

#: How much an item matters to a reader, which is not the same question as how
#: sure it is. See `Item.importance`.
IMPORTANCE = frozenset({"high", "medium", "low"})

METADATA_FIELDS = frozenset({
    "id", "url", "title", "channel", "upload_date", "duration_s", "description",
    "tags", "thumbnail", "chapters", "sponsors",
})


class EvidenceError(ValueError):
    """Raised when a document cannot be read as an Evidence at all.

    Distinct from a validation failure: a malformed document has no structure to
    report findings against, so it is one blocking error, not twenty.
    """


def _req(doc: dict, key: str, where: str):
    if key not in doc:
        raise EvidenceError(f"{where}: missing required field {key!r}")
    return doc[key]


@dataclass
class Provenance:
    """One resolvable pointer from an item back to the material that supports it.

    `ref` resolves against exactly one namespace per `source`: a snippet id, a
    frame id, or a metadata field name. It is never a free-text pointer, because
    a pointer that cannot be checked is the failure mode this whole layer exists
    to remove.
    """

    source: str
    ref: str
    confidence: float = 1.0
    t_start: float | None = None
    t_end: float | None = None
    # Frame/OCR/VLM only. `ocr_text` is the literal text that was read; it is the
    # only thing that may back `verified_verbatim`, and it is absent when no read
    # happened rather than filled with a guess.
    ocr_text: str | None = None
    read_method: str | None = None
    verified_verbatim: bool = False

    def to_dict(self) -> dict:
        d: dict = {"source": self.source, "ref": self.ref, "confidence": self.confidence}
        for k in ("t_start", "t_end", "ocr_text", "read_method"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.verified_verbatim:
            d["verified_verbatim"] = True
        return d

    @classmethod
    def from_dict(cls, doc: dict, where: str) -> "Provenance":
        if not isinstance(doc, dict):
            raise EvidenceError(f"{where}: provenance entry must be an object")
        return cls(
            source=_req(doc, "source", where),
            ref=_req(doc, "ref", where),
            confidence=doc.get("confidence", 1.0),
            t_start=doc.get("t_start"),
            t_end=doc.get("t_end"),
            ocr_text=doc.get("ocr_text"),
            read_method=doc.get("read_method"),
            verified_verbatim=bool(doc.get("verified_verbatim", False)),
        )


@dataclass
class Snippet:
    """A transcript fragment with a real, measured span.

    Snippets are the transcript's unit of provenance: a citation points at the
    exact fragment that carried a claim, not at the file.
    """

    id: str
    start_s: float
    end_s: float
    text: str
    segment_id: str | None = None

    def to_dict(self) -> dict:
        d = {"id": self.id, "start_s": self.start_s, "end_s": self.end_s, "text": self.text}
        if self.segment_id:
            d["segment_id"] = self.segment_id
        return d

    @classmethod
    def from_dict(cls, doc: dict, where: str) -> "Snippet":
        if not isinstance(doc, dict):
            raise EvidenceError(f"{where}: snippet must be an object")
        return cls(
            id=str(_req(doc, "id", where)),
            start_s=_req(doc, "start_s", where),
            end_s=_req(doc, "end_s", where),
            text=doc.get("text", ""),
            segment_id=doc.get("segment_id"),
        )


@dataclass
class Segment:
    """A contiguous span of the video, addressed by time rather than by order.

    `title_source` is kept separate from `title` because chapter titles are
    uploader metadata, not verified: on one 20-minute video 2 of 7 chapter
    titles did not describe what played under them, and a segment that hid that
    distinction would carry the error into every downstream reader.

    `frame_ids` is the segment's own view of the visual layer: the frames
    aligned to any snippet inside it. It is a separate list rather than something
    a reader derives, because "which frames illustrate this section" is a
    question the segment should answer in one hop.
    """

    id: str
    start_s: float
    end_s: float
    title: str
    title_source: str = "chapter"
    snippet_ids: list[str] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)
    frame_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "start_s": self.start_s, "end_s": self.end_s,
            "title": self.title, "title_source": self.title_source,
            "snippet_ids": list(self.snippet_ids), "item_ids": list(self.item_ids),
            "frame_ids": list(self.frame_ids),
        }

    @classmethod
    def from_dict(cls, doc: dict, where: str) -> "Segment":
        if not isinstance(doc, dict):
            raise EvidenceError(f"{where}: segment must be an object")
        return cls(
            id=str(_req(doc, "id", where)),
            start_s=_req(doc, "start_s", where),
            end_s=_req(doc, "end_s", where),
            title=doc.get("title", ""),
            title_source=doc.get("title_source", "chapter"),
            snippet_ids=list(doc.get("snippet_ids") or []),
            item_ids=list(doc.get("item_ids") or []),
            frame_ids=list(doc.get("frame_ids") or []),
        )


@dataclass
class Item:
    """One fact, carried at the strength the source stated it.

    Only the fields the item type needs are set; a `number` fills `value`/`unit`
    and a `link` fills `value`, and neither invents the other's keys. A claim
    that cannot be typed precisely keeps `verbatim` plus `notes` instead of
    being squeezed into a wrong field.

    `detail` is the one escape from the flat shape, and it exists for the two
    types that are genuinely relational: a `comparison` has subjects and
    per-dimension values, and a `step` has a position in a sequence. Encoding
    either as a prose `value` would be exactly the "generic claim string" the
    extraction contract forbids — the structure would survive as English and be
    unrecoverable. It is optional, typed by the validator, and carries nothing
    the item's own provenance does not already back.

    `importance` is deliberately not derived from `confidence`. They answer
    different questions and the corpus contains the counterexample: a channel
    name read off a frame at full confidence is worth almost nothing to a
    reader, while one price the speaker returns to three times is worth a great
    deal from a frame that read at 0.7. Deriving one from the other would make
    every confidently-read piece of interface furniture look load-bearing.
    """

    id: str
    type: str
    value: str
    provenance: list[Provenance]
    display: str | None = None
    unit: str | None = None
    subject: str | None = None
    qualifier: str | None = None
    verbatim: str | None = None
    confidence: float = 1.0
    status: str = "asserted"
    notes: str | None = None
    detail: dict | None = None
    importance: str | None = None
    semantic_role: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"id": self.id, "type": self.type, "value": self.value}
        for k in ("display", "unit", "subject", "qualifier", "verbatim", "notes",
                  "importance", "semantic_role"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.detail is not None:
            d["detail"] = self.detail
        d["provenance"] = [p.to_dict() for p in self.provenance]
        d["confidence"] = self.confidence
        d["status"] = self.status
        return d

    @classmethod
    def from_dict(cls, doc: dict, where: str) -> "Item":
        if not isinstance(doc, dict):
            raise EvidenceError(f"{where}: item must be an object")
        # An absent `provenance` is not a schema violation — it is an item with
        # no source, which the validator reports precisely as `missing-provenance`.
        # Treating the two as the same error would bury the finding under a
        # generic "invalid schema" for a document whose shape is otherwise fine.
        prov = doc.get("provenance", [])
        if not isinstance(prov, list):
            raise EvidenceError(f"{where}: item 'provenance' must be a list")
        detail = doc.get("detail")
        if detail is not None and not isinstance(detail, dict):
            raise EvidenceError(f"{where}: item 'detail' must be an object")
        return cls(
            id=str(_req(doc, "id", where)),
            type=_req(doc, "type", where),
            value=_req(doc, "value", where),
            provenance=[Provenance.from_dict(p, f"{where}.provenance[{i}]") for i, p in enumerate(prov)],
            display=doc.get("display"),
            unit=doc.get("unit"),
            subject=doc.get("subject"),
            qualifier=doc.get("qualifier"),
            verbatim=doc.get("verbatim"),
            confidence=doc.get("confidence", 1.0),
            status=doc.get("status", "asserted"),
            notes=doc.get("notes"),
            detail=detail,
            importance=doc.get("importance"),
            semantic_role=doc.get("semantic_role"),
        )


@dataclass
class VisualFrame:
    """A sampled frame and everything actually known about it.

    `t_start` is null when the sampler's timestamp could not be read. That is
    the F7 case: an unmeasured frame used to render as `00:00:00`, which reads
    as "the very first second of the video" and silently misplaces the evidence.
    `ocr_text` is null until an OCR pass exists — an unanalysed frame says so
    rather than carrying an empty string that reads as "nothing was on it".

    The `ocr_*` block is a read, recorded with enough detail to be checked later:
    the backend that produced it, how sure it was, and where on the frame each run
    was found. `ocr_status` is what keeps "nothing legible here" (a fact about the
    frame) distinguishable from "OCR did not run here" (a fact about the run) — a
    null text with no status is ambiguous, and an ambiguous null is the state this
    layer exists to end.

    The `vlm_*` block is the same contract one layer up, and it is separate rather
    than folded into `ocr_*` on purpose: a model's description of a frame is not a
    reading of its pixels, and merging the two would let a caption's confidence
    and a character's confidence sit in the same field, where they would be
    averaged into a number that describes neither.

    `vlm_observation` holds the model's structured record whole — its content
    type, the numbers with their row and column, the relationship endpoints, the
    uncertainties — rather than a field per structure. The flat scalars beside it
    exist for the checks that have to be answerable from one line of the document;
    the record exists so that a table reading is not flattened into a sentence
    and then unrecoverable. `vlm_caption` is a copy of the record's `summary` and
    the validator enforces that they agree, because two independently editable
    copies of what a model said is how the containment check stops meaning
    anything.
    """

    frame_id: str
    file: str
    kind: str = "unknown"
    t_start: float | None = None
    t_end: float | None = None
    width: int | None = None
    height: int | None = None
    upscale: bool = False
    ocr_text: str | None = None
    caption: str | None = None
    entities: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    ocr_confidence: float | None = None
    ocr_status: str | None = None
    ocr_backend: str | None = None
    ocr_read_method: str | None = None
    ocr_regions: list[dict] = field(default_factory=list)
    vlm_caption: str | None = None
    vlm_status: str | None = None
    vlm_backend: str | None = None
    vlm_confidence: float | None = None
    vlm_observation: dict | None = None

    def to_dict(self) -> dict:
        d: dict = {"frame_id": self.frame_id, "file": self.file, "kind": self.kind}
        for k in ("t_start", "t_end", "width", "height", "ocr_text", "caption",
                  "ocr_confidence", "ocr_status", "ocr_backend", "ocr_read_method",
                  "vlm_caption", "vlm_status", "vlm_backend", "vlm_confidence"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        d["upscale"] = self.upscale
        if self.ocr_regions:
            d["ocr_regions"] = list(self.ocr_regions)
        if self.entities:
            d["entities"] = list(self.entities)
        if self.figures:
            d["figures"] = list(self.figures)
        if self.vlm_observation is not None:
            d["vlm_observation"] = self.vlm_observation
        return d

    @classmethod
    def from_dict(cls, doc: dict, where: str) -> "VisualFrame":
        if not isinstance(doc, dict):
            raise EvidenceError(f"{where}: visual entry must be an object")
        obs = doc.get("vlm_observation")
        if obs is not None and not isinstance(obs, dict):
            raise EvidenceError(f"{where}: 'vlm_observation' must be an object")
        return cls(
            frame_id=str(_req(doc, "frame_id", where)),
            file=doc.get("file", ""),
            kind=doc.get("kind", "unknown"),
            t_start=doc.get("t_start"),
            t_end=doc.get("t_end"),
            width=doc.get("width"),
            height=doc.get("height"),
            upscale=bool(doc.get("upscale", False)),
            ocr_text=doc.get("ocr_text"),
            caption=doc.get("caption"),
            entities=list(doc.get("entities") or []),
            figures=list(doc.get("figures") or []),
            ocr_confidence=doc.get("ocr_confidence"),
            ocr_status=doc.get("ocr_status"),
            ocr_backend=doc.get("ocr_backend"),
            ocr_read_method=doc.get("ocr_read_method"),
            ocr_regions=list(doc.get("ocr_regions") or []),
            vlm_caption=doc.get("vlm_caption"),
            vlm_status=doc.get("vlm_status"),
            vlm_backend=doc.get("vlm_backend"),
            vlm_confidence=doc.get("vlm_confidence"),
            vlm_observation=obs,
        )


@dataclass
class Alignment:
    """One snippet's pairing with the frames that might illustrate it.

    A list, not a single id, because a spoken sentence and its on-screen
    counterpart are routinely spread across a slide held for ten seconds and the
    two cuts either side of it. Picking one and discarding the rest loses the
    frame that actually carried the figure.

    The empty case is a first-class record, not an omission. `frame_ids: []` with
    `method: "no-relevant-frame"` says the aligner looked in a bounded window and
    found nothing worth pairing — which is a claim, checkable against the window
    config, and one a missing record cannot make. Omitting it would make "we
    found no frame" and "we never looked" the same artifact.

    `method` is what separates a pairing that overlaps the speech from one that
    merely sits near it, so a reader never has to infer which from a score.
    """

    snippet_id: str
    frame_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    method: str = "time-nearest"
    start_s: float | None = None
    end_s: float | None = None
    segment_id: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"snippet_id": self.snippet_id, "frame_ids": list(self.frame_ids),
                   "confidence": self.confidence, "method": self.method}
        for k in ("start_s", "end_s", "segment_id", "notes"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, doc: dict, where: str) -> "Alignment":
        if not isinstance(doc, dict):
            raise EvidenceError(f"{where}: alignment entry must be an object")
        # A schema-1 record's `frame_id` is read back as a one-element list. The
        # default would otherwise swallow it and the document would go on to
        # assert, in the `no-relevant-frame` branch, that this snippet had no
        # visual correspondence at all.
        frames = doc.get("frame_ids")
        if frames is None and doc.get("frame_id") is not None:
            frames = [doc["frame_id"]]
        if frames is None:
            frames = []
        if not isinstance(frames, list):
            raise EvidenceError(f"{where}: alignment 'frame_ids' must be a list")
        return cls(
            snippet_id=str(_req(doc, "snippet_id", where)),
            frame_ids=[str(f) for f in frames],
            confidence=doc.get("confidence", 0.0),
            method=doc.get("method", "time-nearest"),
            start_s=doc.get("start_s"),
            end_s=doc.get("end_s"),
            segment_id=doc.get("segment_id"),
            notes=doc.get("notes"),
        )


@dataclass
class Conflict:
    """Two items that disagree, recorded and left open.

    `resolution` defaults to `unresolved` and nothing in this layer ever sets it
    otherwise: resolving a contradiction is a judgement about which source to
    believe, and an extractor that makes that call has already lost the ability
    to be checked.
    """

    item_ids: list[str]
    kind: str
    statement: str
    resolution: str = "unresolved"

    @property
    def cid(self) -> str:
        """A stable, human-checkable name for *this* conflict.

        Derived from the content rather than from a position, because a position
        shifts the moment an unrelated conflict is added and a validator whose
        message names the wrong conflict is worse than one that names none. A
        reader can paste this into `evidence.json` and find the same object, which
        is what makes a validation failure actionable rather than merely true.
        """
        return f"{self.kind}:" + "+".join(sorted(self.item_ids))

    def to_dict(self) -> dict:
        return {"item_ids": list(self.item_ids), "kind": self.kind,
                "statement": self.statement, "resolution": self.resolution}

    @classmethod
    def from_dict(cls, doc: dict, where: str) -> "Conflict":
        if not isinstance(doc, dict):
            raise EvidenceError(f"{where}: conflict must be an object")
        ids = doc.get("item_ids")
        if not isinstance(ids, list) or not ids:
            raise EvidenceError(f"{where}: conflict needs a non-empty 'item_ids'")
        return cls(item_ids=[str(i) for i in ids], kind=_req(doc, "kind", where),
                   statement=doc.get("statement", ""),
                   resolution=doc.get("resolution", "unresolved"))


@dataclass
class Evidence:
    schema_version: int
    video: dict
    transcript: dict
    segments: list[Segment] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)
    visual: list[VisualFrame] = field(default_factory=list)
    alignment: list[Alignment] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    # Snippets live under transcript rather than as a top-level array: they are
    # the transcript's own content, and this way the snippet_count in the header
    # and the list that backs it cannot drift apart as two independent fields.
    snippets: list[Snippet] = field(default_factory=list)
    # Every piece of visible text the semantic stage *considered*, with the
    # classification it reached and the reason. Not the raw reads — those are in
    # `visual[].ocr_text` and `ocr_regions` and are never summarised away — but
    # the decisions, so that "the extractor called this interface chrome" is
    # checkable and reversible. Without it, an over-eager filter is
    # indistinguishable from a frame that genuinely held nothing worth having.
    observations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "video": self.video,
            "transcript": self.transcript,
            "segments": [s.to_dict() for s in self.segments],
            "items": [i.to_dict() for i in self.items],
            "visual": [v.to_dict() for v in self.visual],
            "alignment": [a.to_dict() for a in self.alignment],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "observations": list(self.observations),
            "validation": self.validation,
        }

    def transcript_dict(self) -> dict:
        """transcript block plus its snippets — the on-disk shape."""
        d = dict(self.transcript)
        d["snippets"] = [s.to_dict() for s in self.snippets]
        return d

    @classmethod
    def from_dict(cls, doc: dict) -> "Evidence":
        if not isinstance(doc, dict):
            raise EvidenceError("evidence document must be a JSON object")
        if "schema_version" not in doc:
            raise EvidenceError("evidence: missing required field 'schema_version'")
        for key, typ in (("video", dict), ("transcript", dict)):
            if not isinstance(doc.get(key), typ):
                raise EvidenceError(f"evidence: {key!r} must be an object")
        tdoc = dict(doc["transcript"])
        snippets = [Snippet.from_dict(s, f"transcript.snippets[{i}]")
                    for i, s in enumerate(tdoc.pop("snippets", []) or [])]
        for key in ("segments", "items", "visual", "alignment", "conflicts", "observations"):
            if not isinstance(doc.get(key, []), list):
                raise EvidenceError(f"evidence: {key!r} must be an array")
        return cls(
            schema_version=doc["schema_version"],
            video=doc["video"],
            transcript=tdoc,
            segments=[Segment.from_dict(s, f"segments[{i}]") for i, s in enumerate(doc.get("segments", []))],
            items=[Item.from_dict(s, f"items[{i}]") for i, s in enumerate(doc.get("items", []))],
            visual=[VisualFrame.from_dict(s, f"visual[{i}]") for i, s in enumerate(doc.get("visual", []))],
            alignment=[Alignment.from_dict(s, f"alignment[{i}]") for i, s in enumerate(doc.get("alignment", []))],
            conflicts=[Conflict.from_dict(s, f"conflicts[{i}]") for i, s in enumerate(doc.get("conflicts", []))],
            validation=doc.get("validation", {}) or {},
            snippets=snippets,
            observations=list(doc.get("observations", [])),
        )


def dumps(evidence: Evidence) -> str:
    """Serialise with the snippets inlined under `transcript`.

    `sort_keys` is off deliberately: field order here is the reading order a
    human checks against the audit, and stability comes from the construction
    code always filling the same keys in the same order, not from sorting.
    """
    doc = evidence.to_dict()
    doc["transcript"] = evidence.transcript_dict()
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def load_evidence(path: str | Path) -> Evidence:
    """Read and structurally parse an `evidence.json`.

    Structural only: whether the document is *coherent* (references resolve,
    provenance exists) is the validator's job, so a caller that only wants the
    data does not have to opt into the full rule set.
    """
    p = Path(path)
    try:
        doc = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read {p}: {exc}") from exc
    return Evidence.from_dict(doc)


def save_evidence(evidence: Evidence, path: str | Path) -> None:
    """Write atomically — a half-written evidence.json must never be readable as
    a valid one, since its absence blocks synthesis and its presence admits it."""
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(dumps(evidence))
    tmp.replace(p)
