"""Validator for `evidence.json` — the gate synthesis has to pass.

Design rule, taken from `verify_report.py`'s own docstring and kept here: a
checker stays high-precision. Every check below identifies a defect that is
*mechanically* decidable from the document — a reference that does not resolve,
a confidence outside [0,1], a timestamp past the end of the video. It never
judges whether a fact is true, whether a summary is fair, or whether an item is
worth including. Those are not decidable here and pretending otherwise is what
makes a checker cry wolf and get ignored.

Blocking vs warning is the load-bearing distinction:

* **Blocking** — the document cannot be trusted as a description of the source:
  a dangling pointer, missing provenance, an impossible timestamp, a duplicated
  id. Synthesis must not run on it, because every fact downstream inherits the
  defect and the artifact looks finished.
* **Warning** — the document is coherent but incomplete or degraded: frames with
  no measured timestamp, a coverage figure below 1.0. Recorded, surfaced, and
  allowed through, because a video genuinely can have no ASR metadata or no
  readable frames, and blocking on that would mean never producing a report for
  exactly the videos that need one.

An anti-vacuity check is part of the test strategy, not an option: a validator
that passes a document citing a frame which does not exist is worse than none,
because it is trusted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from evidence_model import (
    ALIGNMENT_METHODS, CONFLICT_KINDS, FRAME_KINDS, IMPORTANCE, ITEM_TYPES,
    SEMANTIC_ROLES,
    METADATA_FIELDS, NO_FRAME_CONFIDENCE, OCR_STATUSES, READ_METHODS, RESOLUTIONS,
    SCHEMA_VERSION, SOURCES, STATUSES, TITLE_SOURCES, VLM_STATUSES,
    Evidence, EvidenceError, load_evidence,
)
from ocr import HIGH_CONFIDENCE, normalize_text
from vlm import MAX_MODEL_CONFIDENCE

# Frame timestamps come from ffmpeg showinfo, so a cited frame's own span and
# the frame index can differ by the rounding in the sampler's grid, but never by
# anything larger. This is the tolerance for "consistent", not for "close".
TIME_EPS = 0.5

# A transcript routinely ends a second or two past the video's reported duration:
# the last caption cue carries a fixed duration, and the platform's duration is
# itself rounded. Blocking on that would reject a good transcript for the
# platform's arithmetic. A fabricated timestamp is off by far more than this, so
# a small tail excess is recorded and surfaced rather than treated as a defect.
TAIL_EPS = 5.0


@dataclass
class ValidationResult:
    valid: bool
    blocking_errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    provenance_coverage: float = 1.0
    transcript_loss_s: float = 0.0
    unmeasured_frames: list[str] = field(default_factory=list)
    dangling_refs: list[dict] = field(default_factory=list)
    ungrounded_items: list[str] = field(default_factory=list)
    ocr_status: str | None = None
    vlm_status: str | None = None
    aligned_frames: int = 0
    unaligned_snippets: int = 0

    def to_dict(self) -> dict:
        d = {
            "valid": self.valid,
            "blocking_errors": self.blocking_errors,
            "warnings": self.warnings,
            "provenance_coverage": self.provenance_coverage,
            "transcript_loss_s": self.transcript_loss_s,
            "unmeasured_frames": self.unmeasured_frames,
            "dangling_refs": self.dangling_refs,
            "ungrounded_items": self.ungrounded_items,
            "aligned_frames": self.aligned_frames,
            "unaligned_snippets": self.unaligned_snippets,
            "blocking": bool(self.blocking_errors),
        }
        if self.ocr_status is not None:
            d["ocr_status"] = self.ocr_status
        if self.vlm_status is not None:
            d["vlm_status"] = self.vlm_status
        return d

    def error_codes(self) -> list[str]:
        return [e["code"] for e in self.blocking_errors]


def _err(code: str, where: str, msg: str, **extra) -> dict:
    d = {"code": code, "where": where, "message": msg}
    d.update(extra)
    return d


def _warn(code: str, where: str, msg: str, **extra) -> dict:
    d = {"code": code, "where": where, "message": msg}
    d.update(extra)
    return d


def _check_confidence(value, where: str, errors: list[dict]) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(_err("invalid-confidence", where, f"confidence must be a number, got {value!r}"))
    elif not (0.0 <= float(value) <= 1.0):
        errors.append(_err("invalid-confidence", where, f"confidence {value} outside [0,1]"))


def _check_ocr_region(region, where: str, frame, errors: list[dict]) -> None:
    """A region must be a real detection: text, a confidence, and a box that
    lies on the frame.

    The bounds check is the point. A box is a claim about where on the image the
    words were, and a box that runs past the frame edge is a claim about a
    different image than the one cited. Both are decidable here, so neither is a
    judgement call.
    """
    if not isinstance(region, dict):
        errors.append(_err("invalid-ocr-region", where, f"region must be an object, got {region!r}"))
        return
    if not (region.get("text") or "").strip():
        errors.append(_err("invalid-ocr-region", where, "region carries no text"))
    _check_confidence(region.get("confidence"), f"{where}.confidence", errors)
    bbox = region.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        errors.append(_err("invalid-bbox", f"{where}.bbox",
                          f"bbox must be [x, y, w, h] of four numbers, got {bbox!r}"))
        return
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bbox):
        errors.append(_err("invalid-bbox", f"{where}.bbox", f"bbox holds a non-number: {bbox!r}"))
        return
    x, y, w, h = (float(v) for v in bbox)
    if w <= 0 or h <= 0:
        errors.append(_err("invalid-bbox", f"{where}.bbox", f"bbox must have positive extent, got {bbox}"))
    if x < 0 or y < 0:
        errors.append(_err("invalid-bbox", f"{where}.bbox", f"bbox starts off-frame, got {bbox}"))
    if frame is not None and frame.width and frame.height and (x + w > frame.width + 1 or y + h > frame.height + 1):
        errors.append(_err("invalid-bbox", f"{where}.bbox",
                          f"bbox {bbox} runs past the {frame.width}x{frame.height} frame"))
    rm = region.get("read_method")
    if rm is not None and rm not in READ_METHODS:
        errors.append(_err("unsupported-enum", f"{where}.read_method", f"{rm!r} not in {sorted(READ_METHODS)}"))


def _ocr_text_supported(frame, text: str) -> bool:
    """True when `text` is literally present in the frame's own stored read.

    Compared on the normalised forms so a line break or a case difference does
    not fail a genuine match, and on nothing else — normalisation cannot equate
    `$500` with `$5,000`, which is the merge this check exists to prevent.
    """
    hay = normalize_text(frame.ocr_text or "")
    needle = normalize_text(text)
    if not needle:
        return False
    if needle in hay:
        return True
    # A citation may name one region of a frame read in several lines; a box of
    # text assembled across a line break is still a literal read of it.
    return needle.replace(" ", "") in hay.replace(" ", "")


def _vlm_text_supported(frame, text: str) -> bool:
    """True when `text` is part of the observation actually stored for the frame.

    The haystack is the whole structured record, not the caption, and the reason
    is the same shape as the check above: a table cell's value is recorded where
    it was seen — `numbers[].value`, with its row and column beside it — and
    never in the prose summary. A citation of that value is the legitimate case;
    a citation of anything else is a model observation the document attributes to
    a frame that never recorded it.

    The frame's own OCR read is deliberately *not* part of the haystack. A model
    that repeats what OCR already said has still recorded it in its own record,
    and letting the OCR text stand in for the model's words would make this check
    pass for a caption the model never produced.
    """
    obs = frame.vlm_observation
    haystack = obs if isinstance(obs, dict) else {"summary": frame.vlm_caption}
    from vlm import VisionObservation  # noqa: PLC0415 — one reader of the shape

    hay = normalize_text(VisionObservation.from_dict(haystack, frame.frame_id).searchable_text())
    needle = normalize_text(text)
    if not needle:
        return False
    return needle in hay or needle.replace(" ", "") in hay.replace(" ", "")


def _check_item_detail(item, where: str, errors: list[dict], warnings: list[dict]) -> None:
    """The two relational types must actually carry their relation.

    A `comparison` whose detail is missing its subjects, or whose dimension row
    supplies a value for neither side, is a claim of comparison with nothing
    compared. Checking the shape here is what stops the escape hatch from becoming
    the place where an unverified number goes to stop being checked.
    """
    d = item.detail
    if d is None:
        if item.type == "comparison":
            errors.append(_err("comparison-detail-missing", where,
                               "a comparison must name its two subjects in 'detail'; "
                               "'value' prose is not a structure a reader can check"))
        elif item.type == "step" and item.subject is None and d is None:
            warnings.append(_warn("step-unindexed", where,
                                  "a step with no index in 'detail' cannot be placed in its sequence"))
        return
    if item.type == "comparison":
        for key in ("subject_a", "subject_b"):
            if not str(d.get(key) or "").strip():
                errors.append(_err("comparison-incomplete", f"{where}.detail",
                                   f"comparison has no {key}"))
        dims = d.get("dimensions")
        if not isinstance(dims, list):
            errors.append(_err("comparison-incomplete", f"{where}.detail.dimensions",
                               f"dimensions must be a list, got {dims!r}"))
            return
        for i, row in enumerate(dims):
            dw = f"{where}.detail.dimensions[{i}]"
            if not isinstance(row, dict):
                errors.append(_err("comparison-incomplete", dw, f"must be an object, got {row!r}"))
                continue
            if not str(row.get("dimension") or "").strip():
                errors.append(_err("comparison-incomplete", dw, "dimension row names no dimension"))
            if not any(str(row.get(k) or "").strip() for k in ("a", "b")):
                # The failure this prevents is the one in the audit: a comparator's
                # figure read onto the subject model. A row with no value on
                # either side is an empty row that reads as a measurement.
                errors.append(_err("comparison-empty-dimension", dw,
                                   "dimension row carries no value for either subject; an absent "
                                   "value is 'not measured', not a row of nils"))
    elif item.type == "claim_metric":
        # Links to a claim and nothing else; the claim's own checks cover whether
        # the link resolves.
        if "claim_id" not in d:
            errors.append(_err("metric-unlinked", f"{where}.detail",
                               "a claim_metric must name the claim it supports"))
    elif item.type == "step":
        idx = d.get("index")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 1:
            errors.append(_err("step-unindexed", f"{where}.detail.index",
                               f"step index must be a positive integer, got {idx!r}"))
    elif item.type == "relationship":
        # A relationship with only one end named is half a diagram, and half a
        # diagram reads as a complete one: "User flows into the retriever" with
        # no target is a sentence that could have been written about anything.
        for key in ("from", "to"):
            if not str(d.get(key) or "").strip():
                errors.append(_err("relationship-incomplete", f"{where}.detail",
                                   f"relationship has no {key!r}; both endpoints are the relation"))
        if d.get("certain") is False:
            warnings.append(_warn("relationship-uncertain", f"{where}.detail",
                                  "the model marked this connection's direction as uncertain; it is "
                                  "recorded as seen, not as established"))
    elif item.type == "visual_finding":
        kind = str(d.get("kind") or "").strip()
        if kind not in ("action", "uncertainty", "entity", "structure"):
            errors.append(_err("finding-kind-invalid", f"{where}.detail.kind",
                               f"visual finding kind must be action, uncertainty, entity or "
                               f"structure, got {kind!r}"))
        if not str(d.get("content_type") or "").strip():
            errors.append(_err("finding-content-type-missing", f"{where}.detail.content_type",
                               "a visual finding must name the content type it was seen in"))
    else:
        warnings.append(_warn("detail-unexpected", where,
                              f"'detail' on a {item.type} item is not read by any check; "
                              "structure that nothing validates is prose with extra punctuation"))


def _check_vlm_observation(frame, where: str, errors: list[dict], warnings: list[dict]) -> None:
    """The stored observation must be internally consistent with the frame.

    Three things, all decidable from the document. The content type is
    model-derived, so the only claim being made about it is that it is one this
    schema knows how to read. The caption must be the record's own summary,
    because the two are two copies of the same sentence and a reader who finds
    they disagree cannot tell which one the model said. And a `malformed` or
    `failed` status must not be carrying structures, because an unusable
    response is recorded as the attempt it was rather than as an observation.
    """
    from vlm import CONTENT_TYPES, VisionObservation  # noqa: PLC0415 — one reader of the shape

    obs = frame.vlm_observation
    ctype = obs.get("content_type")
    if ctype is not None and ctype not in CONTENT_TYPES:
        errors.append(_err("unsupported-enum", f"{where}.vlm_observation.content_type",
                           f"{ctype!r} not in {sorted(CONTENT_TYPES)}"))
    parsed = VisionObservation.from_dict(obs, frame.frame_id)
    if parsed.status == "malformed" and obs.get("status") != "malformed":
        errors.append(_err("vlm-malformed-not-marked", f"{where}.vlm_observation",
                           "the stored record violates its own schema but is not marked "
                           "'malformed'; an unusable response must be recorded as the attempt it was"))
    if frame.vlm_caption and normalize_text(frame.vlm_caption) != normalize_text(parsed.summary or ""):
        errors.append(_err("vlm-caption-diverges", where,
                           "vlm_caption is not the summary of the stored observation; two copies of "
                           "the model's words that disagree cannot both be what it said"))


def _check_confidence_copying(item, where: str, warnings: list[dict]) -> None:
    """Warn when an item's confidence is its read's score, unchanged.

    Not blocking, because a correct extractor could land on the same number. But
    the failure it names is specific and has happened: an item's confidence
    silently becoming the OCR engine's score means a frame read at 1.0 produces
    an item certain enough to assert, and the engine's errors sit at the *top* of
    its scale. The two numbers answer different questions, so an exact match
    between them is a signal to look, not a defect on its own.
    """
    reads = [p for p in item.provenance if p.source in ("ocr", "vlm")]
    if len(item.provenance) != 1 or not reads:
        return
    p = reads[0]
    if isinstance(item.confidence, (int, float)) and not isinstance(item.confidence, bool) \
            and isinstance(p.confidence, (int, float)) and not isinstance(p.confidence, bool) \
            and abs(float(item.confidence) - float(p.confidence)) < 1e-9:
        warnings.append(_warn("confidence-copies-ocr", where,
                              f"item confidence {item.confidence} is exactly the {p.source} read's "
                              "score; semantic confidence and read confidence answer different "
                              "questions and must be computed separately"))


def _check_observations(evidence: Evidence, frame_ids: set[str],
                        errors: list[dict], warnings: list[dict]) -> None:
    """Every recorded decision must point at a frame that exists.

    The observation log is the audit trail for what the extractor *declined* to
    make evidence. A decision recorded against a frame that is not in the index
    is a decision about nothing, and it means the log cannot be trusted to say
    which observations were considered.
    """
    from semantic import OBSERVATION_CLASSES  # noqa: PLC0415 — the producer's vocabulary

    for i, o in enumerate(evidence.observations):
        where = f"observations[{i}]"
        if not isinstance(o, dict):
            errors.append(_err("invalid-observation", where, f"must be an object, got {o!r}"))
            continue
        cls = o.get("classification")
        if cls is not None and cls not in OBSERVATION_CLASSES:
            errors.append(_err("unsupported-enum", f"{where}.classification",
                               f"{cls!r} not in {sorted(OBSERVATION_CLASSES)}"))
        fid = o.get("frame_id")
        if fid is not None and fid not in frame_ids:
            errors.append(_err("dangling-frame-ref", f"{where}.frame_id",
                               f"frame {fid!r} does not exist", ref=fid))
        if not str(o.get("reason") or "").strip():
            # A classification with no stated reason is a filter, and a filter
            # nobody can audit is the thing this layer replaced.
            warnings.append(_warn("observation-no-reason", where,
                                  "observation carries a classification but no reason; "
                                  "the decision cannot be reviewed or reversed"))


def validate_evidence(evidence: Evidence | dict) -> ValidationResult:
    """Run every invariant. Returns the full result — a caller decides whether to
    act on `valid`; nothing here raises, so a report of *all* the defects comes
    back in one pass rather than one per run."""
    res = ValidationResult(valid=True)
    E, W = res.blocking_errors, res.warnings

    if not isinstance(evidence, Evidence):
        try:
            evidence = Evidence.from_dict(evidence)
        except EvidenceError as exc:
            E.append(_err("invalid-schema", "evidence", str(exc)))
            res.valid = False
            return res

    if evidence.schema_version != SCHEMA_VERSION:
        E.append(_err("unsupported-schema-version", "schema_version",
                      f"expected {SCHEMA_VERSION}, got {evidence.schema_version!r}"))

    video = evidence.video
    duration = video.get("duration_s")
    if duration is None:
        W.append(_warn("unknown-duration", "video.duration_s",
                       "no duration; timestamp bounds cannot be checked"))
    elif isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        E.append(_err("invalid-duration", "video.duration_s", f"duration must be >= 0, got {duration!r}"))

    def within_duration(t, where: str, label: str) -> None:
        if t is None or duration is None:
            return
        if isinstance(t, bool) or not isinstance(t, (int, float)):
            E.append(_err("invalid-timestamp", where, f"{label} must be a number, got {t!r}"))
        elif t < 0:
            E.append(_err("impossible-timestamp", where, f"{label} is negative ({t})"))
        elif t > duration + TAIL_EPS:
            E.append(_err("timestamp-beyond-duration", where,
                          f"{label} {t} exceeds video duration {duration}"))
        elif t > duration + TIME_EPS:
            W.append(_warn("timestamp-past-duration-tail", where,
                           f"{label} {t} runs {round(t - duration, 2)}s past the reported duration "
                           f"{duration}; usually a trailing caption cue, but it is a real overrun"))

    # ---- id registries. Built first so every later check can resolve refs. ----
    snippet_ids = {s.id for s in evidence.snippets}
    frame_ids = {v.frame_id for v in evidence.visual}
    item_ids = {i.id for i in evidence.items}
    segment_ids = {seg.id for seg in evidence.segments}

    def dupes(ids: list[str], code: str, where: str) -> set[str]:
        seen, dup = set(), set()
        for i in ids:
            (dup if i in seen else seen).add(i)
        for i in sorted(dup):
            E.append(_err(code, where, f"duplicate id {i!r}"))
        return dup

    dupes([s.id for s in evidence.snippets], "duplicate-id", "transcript.snippets")
    dupes([v.frame_id for v in evidence.visual], "duplicate-id", "visual")
    dupes([i.id for i in evidence.items], "duplicate-id", "items")
    dupes([seg.id for seg in evidence.segments], "duplicate-id", "segments")

    if len(evidence.snippets) != len(snippet_ids):
        E.append(_err("duplicate-id", "transcript.snippets", "duplicate snippet ids"))
    if len(evidence.visual) != len(frame_ids):
        E.append(_err("duplicate-id", "visual", "duplicate frame ids"))
    if len(evidence.items) != len(item_ids):
        E.append(_err("duplicate-id", "items", "duplicate item ids"))
    if len(evidence.segments) != len(segment_ids):
        E.append(_err("duplicate-id", "segments", "duplicate segment ids"))

    # ---- snippets: the transcript's own bookkeeping must agree with its header.
    declared_count = evidence.transcript.get("snippet_count")
    if declared_count is None:
        W.append(_warn("missing-snippet-count", "transcript.snippet_count",
                       "no snippet_count declared; coverage cannot be cross-checked"))
    elif declared_count != len(evidence.snippets):
        E.append(_err("snippet-count-mismatch", "transcript.snippet_count",
                      f"declares {declared_count}, carries {len(evidence.snippets)}"))
    for s in evidence.snippets:
        within_duration(s.start_s, f"snippets[{s.id}].start_s", "snippet start")
        within_duration(s.end_s, f"snippets[{s.id}].end_s", "snippet end")
        if isinstance(s.start_s, (int, float)) and isinstance(s.end_s, (int, float)) and s.end_s < s.start_s:
            E.append(_err("impossible-timestamp", f"snippets[{s.id}]", "snippet ends before it starts"))

    # ---- segments: every snippet assigned to exactly one, and no loss.
    assigned: dict[str, list[str]] = {}
    for seg in evidence.segments:
        if seg.title_source not in TITLE_SOURCES:
            E.append(_err("unsupported-enum", f"segments[{seg.id}].title_source",
                          f"{seg.title_source!r} not in {sorted(TITLE_SOURCES)}"))
        if seg.end_s is not None and seg.end_s < seg.start_s:
            E.append(_err("impossible-timestamp", f"segments[{seg.id}]", "segment ends before it starts"))
        within_duration(seg.start_s, f"segments[{seg.id}].start_s", "segment start")
        within_duration(seg.end_s, f"segments[{seg.id}].end_s", "segment end")
        for sid in seg.snippet_ids:
            assigned.setdefault(sid, []).append(seg.id)
            if sid not in snippet_ids:
                d = {"kind": "snippet", "ref": sid, "from": f"segments[{seg.id}]"}
                res.dangling_refs.append(d)
                E.append(_err("dangling-snippet-ref", f"segments[{seg.id}].snippet_ids",
                              f"snippet {sid!r} does not exist", ref=sid))
        for iid in seg.item_ids:
            if iid not in item_ids:
                res.dangling_refs.append({"kind": "item", "ref": iid, "from": f"segments[{seg.id}]"})
                E.append(_err("dangling-item-ref", f"segments[{seg.id}].item_ids",
                              f"item {iid!r} does not exist", ref=iid))

    for sid, segs in sorted(assigned.items()):
        if len(segs) > 1:
            E.append(_err("snippet-duplicated", f"segments[{segs[0]}]",
                          f"snippet {sid!r} assigned to {len(segs)} segments: {', '.join(segs)}", ref=sid))

    # Transcript loss is measured in seconds, not in snippets, because a lost
    # 2-minute block and a lost 2-second fragment are not the same failure and a
    # count cannot tell them apart.
    by_id = {s.id: s for s in evidence.snippets}
    lost = [s for s in evidence.snippets if not assigned.get(s.id)]
    res.transcript_loss_s = round(
        sum(max(0.0, (s.end_s or s.start_s) - s.start_s) for s in lost
            if isinstance(s.start_s, (int, float))
            and isinstance(s.end_s if s.end_s is not None else s.start_s, (int, float))), 3)
    if lost:
        E.append(_err("transcript-unassigned", "segments",
                      f"{len(lost)} snippet(s) belong to no segment ({res.transcript_loss_s}s of speech)",
                      snippets=[s.id for s in lost[:20]]))

    # A snippet that claims a segment must be claimed by that same segment.
    for s in evidence.snippets:
        if s.segment_id and assigned.get(s.id) and s.segment_id not in assigned[s.id]:
            E.append(_err("segment-disagreement", f"snippets[{s.id}]",
                          f"claims segment {s.segment_id!r}, listed under {assigned[s.id]}"))

    # ---- visual: a frame is either measured or explicitly unmeasured. F7.
    for v in evidence.visual:
        where = f"visual[{v.frame_id}]"
        if v.kind not in FRAME_KINDS:
            E.append(_err("unsupported-enum", f"{where}.kind", f"{v.kind!r} not in {sorted(FRAME_KINDS)}"))
        if v.t_start is None:
            res.unmeasured_frames.append(v.frame_id)
        else:
            within_duration(v.t_start, f"{where}.t_start", "frame start")
        within_duration(v.t_end, f"{where}.t_end", "frame end")
        if v.t_start is not None and v.t_end is not None and v.t_end < v.t_start:
            E.append(_err("impossible-timestamp", where, "frame ends before it starts"))
    declared_unmeasured = evidence.validation.get("unmeasured_frames")
    if declared_unmeasured is not None and sorted(declared_unmeasured) != sorted(res.unmeasured_frames):
        E.append(_err("unmeasured-frame-disagreement", "validation.unmeasured_frames",
                      f"declares {sorted(declared_unmeasured)}, measured set is {sorted(res.unmeasured_frames)}"))
    if res.unmeasured_frames:
        W.append(_warn("unmeasured-frames", "visual",
                       f"{len(res.unmeasured_frames)} frame(s) have no measured timestamp; "
                       "the run is degraded and no timing claim may be made about them",
                       frames=res.unmeasured_frames[:20]))

    # ---- visual OCR. A read is only checkable if it says what produced it.
    #
    # These are the rules that make an OCR-backed number a fact rather than a
    # plausible string. The expensive one is `ocr-text-not-in-frame`: it compares
    # the cited text against the frame's OWN stored read, which is the only
    # mechanically decidable way to catch a figure that was typed into an item
    # and attributed to pixels that never showed it.
    frames_by_id = {v.frame_id: v for v in evidence.visual}
    for v in evidence.visual:
        where = f"visual[{v.frame_id}]"
        read = v.ocr_text is not None
        if v.ocr_status is not None and v.ocr_status not in OCR_STATUSES:
            E.append(_err("unsupported-enum", f"{where}.ocr_status",
                          f"{v.ocr_status!r} not in {sorted(OCR_STATUSES)}"))
        if read and v.ocr_status is None:
            E.append(_err("ocr-missing-status", where,
                          "ocr_text is present with no ocr_status; a read that cannot say "
                          "whether it happened is indistinguishable from a fabricated one"))
        if not read and v.ocr_status in ("ok",):
            E.append(_err("ocr-status-contradiction", where,
                          "ocr_status='ok' with no ocr_text; the frame was not read"))
        if read and v.ocr_status == "no_text":
            # `no_text` says the frame was read and held nothing legible. Text
            # alongside it is a contradiction in one direction only: the other
            # direction — `no_text` with no text — is the ordinary, correct state
            # for a talking head, and treating it as a defect would fail every
            # frame the engine correctly found empty.
            E.append(_err("ocr-status-contradiction", where,
                          "ocr_status='no_text' but ocr_text is present"))
        if read and not (v.ocr_backend or "").strip():
            E.append(_err("ocr-origin-missing", where,
                          "ocr_text is present with no ocr_backend; the read cannot be attributed"))
        if v.ocr_read_method is not None and v.ocr_read_method not in READ_METHODS:
            E.append(_err("unsupported-enum", f"{where}.ocr_read_method",
                          f"{v.ocr_read_method!r} not in {sorted(READ_METHODS)}"))
        if v.ocr_confidence is not None:
            _check_confidence(v.ocr_confidence, f"{where}.ocr_confidence", E)
        if read and v.ocr_confidence is None:
            E.append(_err("ocr-missing-confidence", where,
                          "ocr_text is present with no ocr_confidence"))
        if v.width is not None and v.height is not None:
            if v.width <= 0 or v.height <= 0:
                E.append(_err("invalid-frame-size", where,
                              f"frame is {v.width}x{v.height}; a frame has positive extent"))
        for j, rg in enumerate(v.ocr_regions or []):
            _check_ocr_region(rg, f"{where}.ocr_regions[{j}]", v, E)

    # The run's OCR outcome, derived rather than trusted: a document cannot
    # declare itself healthy. Worst-first, because a run where 40 frames read and
    # 2 failed is not an `ok` run. `repeat` is absent from the list — it says an
    # observation was already recorded elsewhere, not how well anything read.
    for s in ("unavailable", "failed", "unreadable", "skipped", "no_text", "ok"):
        if any(v.ocr_status == s for v in evidence.visual):
            res.ocr_status = s
            break
    declared = evidence.validation.get("ocr_status")
    if declared is not None and declared not in OCR_STATUSES:
        E.append(_err("unsupported-enum", "validation.ocr_status",
                      f"{declared!r} not in {sorted(OCR_STATUSES)}"))
    elif declared is not None and res.ocr_status is not None and declared != res.ocr_status:
        E.append(_err("ocr-status-disagreement", "validation.ocr_status",
                      f"declares {declared!r}, the frames' own reads say {res.ocr_status!r}"))

    degraded = [v for v in evidence.visual
                if v.ocr_status in ("unavailable", "failed", "unreadable", "skipped")]
    if degraded:
        W.append(_warn("ocr-degraded", "visual",
                       f"{len(degraded)} frame(s) were not read by OCR "
                       f"({', '.join(sorted({v.ocr_status for v in degraded}))}); no visual text may be "
                       "claimed for them"))


    # ---- items and their provenance. The core of the layer.
    covered = 0
    for it in evidence.items:
        where = f"items[{it.id}]"
        if it.type not in ITEM_TYPES:
            E.append(_err("unsupported-enum", f"{where}.type", f"{it.type!r} not a supported item type"))
        if it.status not in STATUSES:
            E.append(_err("unsupported-enum", f"{where}.status", f"{it.status!r} not in {sorted(STATUSES)}"))
        if it.importance is not None and it.importance not in IMPORTANCE:
            E.append(_err("unsupported-enum", f"{where}.importance",
                          f"{it.importance!r} not in {sorted(IMPORTANCE)}"))
        if it.semantic_role is not None and it.semantic_role not in SEMANTIC_ROLES:
            E.append(_err("unsupported-enum", f"{where}.semantic_role",
                          f"{it.semantic_role!r} not in {sorted(SEMANTIC_ROLES)}"))
        _check_confidence(it.confidence, f"{where}.confidence", E)
        _check_item_detail(it, where, E, W)

        if not it.provenance:
            res.ungrounded_items.append(it.id)
            E.append(_err("missing-provenance", where,
                          "item has no provenance entry; every item must resolve to a source"))
            continue
        covered += 1

        for j, p in enumerate(it.provenance):
            pw = f"{where}.provenance[{j}]"
            if p.source not in SOURCES:
                E.append(_err("unsupported-enum", f"{pw}.source", f"{p.source!r} not in {sorted(SOURCES)}"))
            if p.read_method is not None and p.read_method not in READ_METHODS:
                E.append(_err("unsupported-enum", f"{pw}.read_method", f"{p.read_method!r} not in {sorted(READ_METHODS)}"))
            _check_confidence(p.confidence, f"{pw}.confidence", E)
            within_duration(p.t_start, f"{pw}.t_start", "provenance start")
            within_duration(p.t_end, f"{pw}.t_end", "provenance end")

            if p.source == "transcript":
                if p.ref not in snippet_ids:
                    res.dangling_refs.append({"kind": "snippet", "ref": p.ref, "from": pw})
                    E.append(_err("dangling-transcript-ref", pw, f"snippet {p.ref!r} does not exist", ref=p.ref))
            elif p.source in ("frame", "ocr", "vlm"):
                fid = p.ref
                if fid not in frame_ids:
                    res.dangling_refs.append({"kind": "frame", "ref": fid, "from": pw})
                    E.append(_err("dangling-frame-ref", pw, f"frame {fid!r} does not exist", ref=fid))
                else:
                    # A cited frame's span must be the frame index's own span. A
                    # citation that points at the right frame with the wrong
                    # timestamp is a second pointer that drifted, and nothing
                    # downstream reads the index to catch it.
                    vf = frames_by_id[fid]
                    if vf.t_start is not None and p.t_start is not None and abs(p.t_start - vf.t_start) > TIME_EPS:
                        E.append(_err("frame-timestamp-mismatch", pw,
                                      f"cites t_start={p.t_start} but {fid} is at {vf.t_start}"))
                    if vf.t_start is None and p.t_start is not None:
                        E.append(_err("frame-timestamp-fabricated", pw,
                                      f"{fid} is unmeasured but the citation carries t_start={p.t_start}"))
                if p.verified_verbatim and not (p.ocr_text or "").strip():
                    E.append(_err("unbacked-verification", pw,
                                  "verified_verbatim=true with no ocr_text; no read backs this claim"))
                if p.verified_verbatim and p.source == "vlm":
                    # A model describing a frame has not read its pixels, so
                    # `verified_verbatim` — which means "the source says exactly
                    # this" — is not available to this source at all.
                    E.append(_err("model-verbatim", pw,
                                  "a vlm observation cannot be verified_verbatim; a description "
                                  "of a frame is not a reading of it"))
                if p.source == "vlm":
                    # The rule the separation exists to make enforceable: a model
                    # citation must name a frame a model actually described, and
                    # the text it carries must be part of what was recorded. The
                    # second half is checked against the whole structured
                    # observation rather than the caption alone, because a value
                    # read out of a table cell is recorded in `numbers` and never
                    # appears in the summary — and it is exactly that value, which
                    # OCR could not attach to its row, that this layer exists to
                    # recover. Checking the caption alone would have rejected the
                    # one thing the pass is for.
                    vf = frames_by_id.get(fid)
                    if vf is not None:
                        if vf.vlm_status != "ok" or not (vf.vlm_caption or "").strip():
                            E.append(_err("vlm-without-result", pw,
                                          f"cites {fid} as a model observation, but that frame "
                                          f"carries none (vlm_status {vf.vlm_status!r})"))
                        elif p.ocr_text and not _vlm_text_supported(vf, p.ocr_text or ""):
                            E.append(_err("vlm-text-not-in-frame", pw,
                                          f"cited text is not part of the observation stored "
                                          f"for {fid}"))
                if p.source == "ocr":
                    # An OCR citation is a claim about a read of the pixels, so it
                    # has to name one. These three are the failure the whole layer
                    # was built for: a value attributed to a frame whose stored
                    # read does not contain it, a verification asserted on a
                    # reading the engine was unsure of, and an item manufactured
                    # out of a frame that yielded nothing at all.
                    vf = frames_by_id.get(fid)
                    if vf is not None:
                        if not (vf.ocr_text or "").strip():
                            E.append(_err("ocr-without-text", pw,
                                          f"cites {fid} as an OCR read, but that frame carries no text "
                                          f"(status {vf.ocr_status!r})"))
                        elif not _ocr_text_supported(vf, p.ocr_text or ""):
                            E.append(_err("ocr-text-not-in-frame", pw,
                                          f"cited text {p.ocr_text!r} does not appear in the read stored "
                                          f"for {fid}"))
                    if (p.verified_verbatim
                            and isinstance(p.confidence, (int, float))
                            and not isinstance(p.confidence, bool)
                            and p.confidence < HIGH_CONFIDENCE):
                        E.append(_err("low-confidence-verbatim", pw,
                                      f"verified_verbatim at confidence {p.confidence}, below the "
                                      f"{HIGH_CONFIDENCE} policy floor; a low-confidence read is an "
                                      "observation, not a confirmation"))
            elif p.source == "metadata":
                if p.ref not in METADATA_FIELDS:
                    E.append(_err("dangling-metadata-ref", pw,
                                  f"{p.ref!r} is not a metadata field of the schema", ref=p.ref))

    if evidence.items:
        res.provenance_coverage = round(covered / len(evidence.items), 4)
        if res.provenance_coverage < 1.0:
            res.ungrounded_items = [i.id for i in evidence.items if not i.provenance]
            W.append(_warn("provenance-coverage", "items",
                           f"{res.provenance_coverage:.0%} of items carry provenance"))

    for it in evidence.items:
        _check_confidence_copying(it, f"items[{it.id}]", W)

    # A `claim_metric` that names a claim which is not in the document is a
    # relationship pointing at nothing, and a reader following the link finds no
    # claim. Same for the reverse: a claim with no metric is not a defect, since
    # most claims are qualitative.
    for it in evidence.items:
        # Only a `claim_metric` links to a claim. A comparison's `detail` carries
        # subjects and dimensions, and reading a missing `claim_id` off it as a
        # dangling reference would fail every structured comparison in the
        # document for the crime of not being one.
        if it.detail and "claim_id" in it.detail and str(it.detail.get("claim_id") or "") not in item_ids:
            res.dangling_refs.append({"kind": "item", "ref": it.detail.get("claim_id"),
                                      "from": f"items[{it.id}].detail.claim_id"})
            E.append(_err("dangling-item-ref", f"items[{it.id}].detail.claim_id",
                          f"links to claim {it.detail.get('claim_id')!r}, which does not exist",
                          ref=it.detail.get("claim_id")))

    _check_observations(evidence, frame_ids, E, W)

    # ---- alignment.
    #
    # The checks here are the ones a reader of `alignment.json` would otherwise
    # have to take on trust. `frame-timestamp-mismatch` already exists for
    # citations, and it applies here for the same reason: a record that points at
    # a frame is a pointer, and a pointer that names the right frame with the
    # wrong time is still wrong in a way nothing downstream reads the index to
    # catch.
    seen_snippets: set[str] = set()
    for a in evidence.alignment:
        where = f"alignment[{a.snippet_id}]"
        _check_confidence(a.confidence, f"{where}.confidence", E)
        if a.method not in ALIGNMENT_METHODS:
            E.append(_err("unsupported-enum", f"{where}.method",
                          f"{a.method!r} not in {sorted(ALIGNMENT_METHODS)}"))
        if a.snippet_id in seen_snippets:
            E.append(_err("duplicate-id", where, f"snippet {a.snippet_id!r} is aligned twice"))
        seen_snippets.add(a.snippet_id)
        if a.snippet_id not in snippet_ids:
            res.dangling_refs.append({"kind": "snippet", "ref": a.snippet_id, "from": where})
            E.append(_err("dangling-transcript-ref", where, f"snippet {a.snippet_id!r} does not exist"))
        # The empty pairing and the "no relevant frame" method are the same claim
        # stated twice. Either alone is fine; they cannot disagree.
        if not a.frame_ids:
            res.unaligned_snippets += 1
            if a.method != "no-relevant-frame":
                E.append(_err("alignment-method-contradiction", where,
                              f"no frames paired but method is {a.method!r}; the empty "
                              "result must be declared as 'no-relevant-frame'"))
            if a.confidence != NO_FRAME_CONFIDENCE:
                E.append(_err("alignment-empty-confidence", where,
                              f"an empty pairing must carry the {NO_FRAME_CONFIDENCE} sentinel, "
                              f"not {a.confidence}; a computed score on a non-result reads as a weak match"))
        elif a.method == "no-relevant-frame":
            E.append(_err("alignment-method-contradiction", where,
                          f"method is 'no-relevant-frame' but {len(a.frame_ids)} frame(s) are paired"))
        for fid in a.frame_ids:
            if fid not in frame_ids:
                res.dangling_refs.append({"kind": "frame", "ref": fid, "from": where})
                E.append(_err("dangling-frame-ref", where, f"frame {fid!r} does not exist"))
                continue
            res.aligned_frames += 1
            if len(a.frame_ids) != len(set(a.frame_ids)):
                E.append(_err("duplicate-id", where, f"frame {fid!r} is listed twice in one pairing"))
        # The record's own span must be the snippet's span. A pairing whose
        # `start_s` was typed in rather than carried leaves a record that looks
        # self-contained and cannot be re-derived from the transcript.
        sn = by_id.get(a.snippet_id)
        if sn is not None and a.start_s is not None and abs(a.start_s - sn.start_s) > TIME_EPS:
            E.append(_err("alignment-timestamp-mismatch", where,
                          f"alignment start {a.start_s} does not match {a.snippet_id}'s own "
                          f"start {sn.start_s}"))
        if a.segment_id and a.segment_id not in segment_ids:
            res.dangling_refs.append({"kind": "segment", "ref": a.segment_id, "from": where})
            E.append(_err("dangling-segment-ref", where, f"segment {a.segment_id!r} does not exist"))

    # A segment's frame set must be the union of its snippets' pairings. Held
    # here rather than trusted, because the two are written by different passes
    # and a segment that lists a frame nothing aligns to is a claim about the
    # video that no record supports.
    for seg in evidence.segments:
        for fid in seg.frame_ids:
            if fid not in frame_ids:
                res.dangling_refs.append({"kind": "frame", "ref": fid, "from": f"segments[{seg.id}].frame_ids"})
                E.append(_err("dangling-frame-ref", f"segments[{seg.id}].frame_ids",
                              f"frame {fid!r} does not exist", ref=fid))
        expected = {fid for a in evidence.alignment
                    if a.segment_id == seg.id for fid in a.frame_ids}
        for fid in seg.frame_ids:
            if fid not in expected:
                W.append(_warn("segment-frame-unaligned", f"segments[{seg.id}].frame_ids",
                               f"frame {fid!r} is listed on the segment but no snippet in it "
                               "aligns to that frame"))

    # ---- VLM provenance. A `vlm` entry is a claim that a model looked.
    #
    # This is the check the whole separation rests on. If a document can carry a
    # `vlm` citation on a frame nothing ever analysed, then `vlm` is not a source
    # — it is a free-text field that makes a model observation look like a read,
    # and every downstream reader has to take it on trust.
    for v in evidence.visual:
        where = f"visual[{v.frame_id}]"
        if v.vlm_status is not None and v.vlm_status not in VLM_STATUSES:
            E.append(_err("unsupported-enum", f"{where}.vlm_status",
                          f"{v.vlm_status!r} not in {sorted(VLM_STATUSES)}"))
        if v.vlm_caption is not None and v.vlm_status is None:
            E.append(_err("vlm-missing-status", where,
                          "vlm_caption is present with no vlm_status; a model observation that "
                          "cannot say whether a model produced it is indistinguishable from a fabricated one"))
        if not v.vlm_caption and v.vlm_status == "ok":
            E.append(_err("vlm-status-contradiction", where,
                          "vlm_status='ok' with no vlm_caption; no model described this frame"))
        if v.vlm_caption and not (v.vlm_backend or "").strip():
            E.append(_err("vlm-origin-missing", where,
                          "vlm_caption is present with no vlm_backend; the observation cannot be attributed"))
        if v.vlm_confidence is not None:
            _check_confidence(v.vlm_confidence, f"{where}.vlm_confidence", E)
            if v.vlm_confidence > MAX_MODEL_CONFIDENCE:
                # The cap is a policy about what a description is worth, and it
                # has to hold against a hand-edited document as well as against
                # the producer. A model observation at 0.95 reads as a
                # measurement, and no check downstream would question it.
                E.append(_err("vlm-confidence-over-cap", f"{where}.vlm_confidence",
                              f"{v.vlm_confidence} exceeds the model-observation ceiling of "
                              f"{MAX_MODEL_CONFIDENCE}; a description of a frame is not a reading of it"))
        if v.vlm_observation is not None:
            _check_vlm_observation(v, where, E, W)
        elif v.vlm_status == "ok":
            # Not blocking: a schema-2 document carries a caption and no
            # structured record, and refusing to validate it would reject
            # artifacts that are correct for the schema they were written under.
            W.append(_warn("vlm-observation-absent", where,
                           "the model observation is a caption with no structured record; the "
                           "values, relationships and uncertainties it recorded cannot be read back"))
    for s in ("unavailable", "failed", "skipped", "no_visual", "ok"):
        if any(v.vlm_status == s for v in evidence.visual):
            res.vlm_status = s
            break
    no_vlm = [v.frame_id for v in evidence.visual if v.vlm_status is None]
    if no_vlm:
        W.append(_warn("vlm-not-run", "visual",
                       f"{len(no_vlm)} frame(s) carry no vlm_status; no visual *interpretation* "
                       "may be claimed for them, whatever their OCR read says"))

    # ---- conflicts: recorded, never silently resolved.
    for c in evidence.conflicts:
        where = f"conflicts[{c.kind}]"
        if c.kind not in CONFLICT_KINDS:
            E.append(_err("unsupported-enum", f"{where}.kind", f"{c.kind!r} not in {sorted(CONFLICT_KINDS)}"))
        if c.resolution not in RESOLUTIONS:
            E.append(_err("unsupported-enum", f"{where}.resolution", f"{c.resolution!r} not in {sorted(RESOLUTIONS)}"))
        for iid in c.item_ids:
            if iid not in item_ids:
                res.dangling_refs.append({"kind": "item", "ref": iid, "from": where})
                E.append(_err("dangling-item-ref", where, f"conflict names unknown item {iid!r}", ref=iid))
        if c.resolution == "resolved":
            W.append(_warn("conflict-resolved", where,
                           "a conflict marked resolved must name who resolved it in the report"))

    # The two lists must agree in both directions. An item flagged `conflicting`
    # that no conflict names is a claim of disagreement with nothing to disagree
    # with, and a conflict naming items that are not flagged leaves the reader to
    # discover which side of the contradiction each value is on.
    conflicted = {i.id for i in evidence.items if i.status == "conflicting"}
    named = {iid for c in evidence.conflicts for iid in c.item_ids}
    for iid in sorted(conflicted - named):
        E.append(_err("conflict-status-orphan", f"items[{iid}]",
                      "status is 'conflicting' but no conflict names this item", ref=iid))
    for iid in sorted(named - conflicted):
        W.append(_warn("conflict-status-missing", f"items[{iid}]",
                       "named by a conflict but its status is not 'conflicting'; a reader "
                       "cannot tell which side of the disagreement this value is on"))

    res.valid = not E
    return res


def validate_file(path: str | Path) -> tuple[Evidence | None, ValidationResult]:
    """Validate an on-disk `evidence.json`. A structurally unreadable file is a
    blocking `invalid-schema` result rather than an exception, so the caller has
    one uniform path for "do not synthesise"."""
    try:
        ev = load_evidence(path)
    except EvidenceError as exc:
        return None, ValidationResult(valid=False, blocking_errors=[
            _err("invalid-schema", "evidence", str(exc))], provenance_coverage=0.0)
    return ev, validate_evidence(ev)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Validate an evidence.json. Exit 0 valid, 1 blocking failures, 2 unreadable.")
    ap.add_argument("evidence", help="path to evidence.json, or an artifact directory containing one")
    ap.add_argument("--json", action="store_true", help="emit the full validation result as JSON")
    args = ap.parse_args(argv)

    p = Path(args.evidence)
    if p.is_dir():
        p = p / "evidence.json"
    ev, res = validate_file(p)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(f"{'VALID' if res.valid else 'INVALID'}  {p}")
        for e in res.blocking_errors:
            print(f"  BLOCK  {e['code']}  {e['where']}: {e['message']}")
        for w in res.warnings:
            print(f"  warn   {w['code']}  {w['where']}: {w['message']}")
        print(f"  provenance_coverage={res.provenance_coverage}  transcript_loss_s={res.transcript_loss_s}  "
              f"unmeasured_frames={len(res.unmeasured_frames)}")
        print(f"  aligned_frames={res.aligned_frames}  unaligned_snippets={res.unaligned_snippets}"
              + (f"  vlm_status={res.vlm_status}" if res.vlm_status else ""))
    return 0 if res.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
