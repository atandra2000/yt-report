"""Transcript ↔ frame alignment: which frames, if any, illustrate which sentence.

The pipeline's hardest alignment mistake is not missing a pairing — it is
inventing one. The nearest frame in time is a fact about the clock, and on a
talking-head video the nearest frame to any sentence is a shot of a person
talking. Presenting that as "the frame that shows this" gives a document where
every claim has a visual citation and roughly none of them are true.

So the scoring is deliberately asymmetric. Temporal proximity decides *whether*
a frame is a candidate at all and dominates the ordering; the visual signals
reorder candidates that are already close. That ordering is the whole design:

* An OCR read is worth a bounded, small boost — **not** enough to pull in a
  frame that is 20 seconds away, only enough to break a tie between frames 1.2 s
  and 1.8 s away. The reason is measurement, not modesty: the read's own
  confidence is a rank, not a correctness measure, and on the measured
  distribution for this engine the errors cluster at the *top* of that scale.
  A boost sized on the score's magnitude would therefore amplify exactly the
  frames where it is least calibrated.
* Segment context is a second, later pass, because it is the only signal that
  needs the whole segment to compute: a slide held across four sentences is
  picked by all four, and that agreement is evidence no single sentence can
  supply on its own.
* Nothing clears a floor → `frame_ids: []`, `method: "no-relevant-frame"`, and a
  fixed low confidence. The empty result is a finding, and it is recorded as
  one.

Frames with no measured timestamp are never candidates. Their time is unknown,
so any pairing would be a claim about a frame whose position is unknown — the
F7 rule, and the reason `alignment.json` counts them as excluded rather than
quietly dropping them.

Deterministic and offline: same inputs, same records, byte for byte. There is no
model here, and there should not be — a stage whose output cannot be re-derived
is not a stage a validator can check.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from evidence_model import (
    NO_FRAME_CONFIDENCE, Alignment, Evidence, Segment, Snippet, VisualFrame,
)

#: A pairing is never certain. The strongest honest claim this stage can make is
#: "these frames fall inside the speech and carry a usable read", and capping at
#: 0.95 keeps the top of the scale meaning that rather than meaning "proved".
MAX_PAIRING_CONFIDENCE = 0.95


@dataclass
class AlignmentConfig:
    """Every number the aligner is allowed to use, in one place.

    Exposed rather than hard-coded because the window is the single assumption
    most likely to be wrong for a given video: a rapid slide deck wants a tight
    window, a slow demo wants a wide one. A reader who cannot see the window
    cannot tell a genuine absence of visual evidence from a window that was too
    small to find any.
    """

    #: Half-width of the candidate window around a snippet's span, in seconds.
    #: Not the whole video: a bounded window is what keeps this O(n log n) and,
    #: more importantly, what keeps an unrelated frame three minutes away from
    #: ever being a candidate.
    window_s: float = 4.0
    #: How many candidates a single snippet may retain. More than one because a
    #: slide is often cut across the sentence that discusses it.
    max_frames: int = 3
    #: Below this, the answer is "no relevant frame" rather than a weak pairing.
    min_confidence: float = 0.35
    #: Ceiling on what an OCR read can add. Sized to reorder near-ties (0.62 vs
    #: 0.44 for a 1.2 s / 1.8 s gap) and not to rescue a real temporal gap.
    ocr_boost: float = 0.22
    #: Ceiling on what agreement across a segment's snippets can add. Smaller
    #: again: it is a popularity signal, and popularity is weak evidence.
    segment_boost: float = 0.10
    #: A read below the OCR medium band contributes nothing. A low-confidence
    #: read of `32 6B` is one character from a different memory size.
    ocr_min_confidence: float = 0.50

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("window_s", "max_frames", "min_confidence", "ocr_boost",
                 "segment_boost", "ocr_min_confidence")}


@dataclass
class _Candidate:
    """One frame under consideration for one snippet, with its score broken out.

    The components are kept, not summed away, so `alignment.json` can show the
    working and a reader can disagree with one term instead of with a number
    whose derivation they cannot see.
    """

    frame: VisualFrame
    distance_s: float
    overlaps: bool
    temporal: float
    ocr: float
    segment: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return min(MAX_PAIRING_CONFIDENCE, self.temporal + self.ocr + self.segment)


def _temporal(score_base: float, distance_s: float, window_s: float) -> float:
    """Linear decay from `score_base` at the span to 0 at the window edge.

    Linear rather than exponential because the window is already the stated
    horizon: a curve that decayed faster would only add a hidden second
    threshold, and one that decayed slower would make the edge a cliff.
    """
    if window_s <= 0:
        return score_base
    return score_base * max(0.0, 1.0 - distance_s / window_s)


def _usable_read(frame: VisualFrame, cfg: AlignmentConfig) -> bool:
    return bool((frame.ocr_text or "").strip()) and (
        frame.ocr_confidence is not None and frame.ocr_confidence >= cfg.ocr_min_confidence)


def _span_distance(t: float, start: float, end: float) -> tuple[float, bool]:
    """Distance from `t` to the span [start, end], and whether it falls inside."""
    if t < start:
        return start - t, False
    if t > end:
        return t - end, False
    return 0.0, True


class _FrameIndex:
    """Frames sorted by time, for bounded-window lookup by bisect.

    A per-snippet scan of every frame would be O(snippets × frames) and, worse,
    would make "near" mean "least far of everything" — which is precisely the
    failure of a global nearest-frame pairing. The window is enforced here, by
    construction, so no code path can accidentally widen it.
    """

    def __init__(self, frames: list[VisualFrame]):
        self.measured = sorted((f for f in frames if f.t_start is not None),
                               key=lambda f: f.t_start)  # type: ignore[arg-type]
        self.unmeasured = [f for f in frames if f.t_start is None]
        self._times = [f.t_start for f in self.measured]  # type: ignore[misc]
        self._by_id = {f.frame_id: f for f in frames}

    def in_window(self, lo: float, hi: float) -> list[VisualFrame]:
        i = bisect.bisect_left(self._times, lo)
        out: list[VisualFrame] = []
        while i < len(self._times) and self._times[i] <= hi:  # type: ignore[operator]
            out.append(self.measured[i])
            i += 1
        return out

    def by_id(self, fid: str) -> VisualFrame | None:
        return self._by_id.get(fid)


def _distance_stats(ev: Evidence, records: list[Alignment]) -> dict:
    """How far each winning frame sat from the sentence it was paired with.

    Reported so that a future change to `window_s` or to the frame set is
    measurable rather than argued about: the number that says "the extra
    alignments are real" is not the count of alignments, it is the distribution
    of the distances behind them.
    """
    by_id = {s.id: s for s in ev.snippets}
    frames = {f.frame_id: f for f in ev.visual}
    out: list[float] = []
    for r in records:
        if not r.frame_ids:
            continue
        sn = by_id.get(r.snippet_id)
        top = frames.get(r.frame_ids[0])
        if sn is None or top is None or top.t_start is None:
            continue
        dist, _inside = _span_distance(top.t_start, sn.start_s, sn.end_s or sn.start_s)
        out.append(round(dist, 2))
    if not out:
        return {"count": 0}
    ordered = sorted(out)
    return {
        "count": len(out),
        "median_s": ordered[len(ordered) // 2],
        "p95_s": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max_s": ordered[-1],
        "beyond_2x_window": sum(1 for d in out if d > 2 * AlignmentConfig().window_s),
    }


def classify_unaligned(ev: Evidence, records: list[Alignment],
                       cfg: AlignmentConfig | None = None) -> dict:
    """Why each `no-relevant-frame` snippet got no frame, in four honest buckets.

    A high `no_relevant_frame` count is not a defect on its own: most of a talk is
    speech over a frame that shows nothing, and pairing those would trade real
    restraint for a number that reads better. It is also not automatically
    healthy, because the same count is what a too-narrow window or a too-strict
    floor would produce. The two are indistinguishable from the count alone, so
    this separates them, offline, from the same window arithmetic the aligner
    used:

    * `speech_only` — no frame in the window, or every frame there is blank. The
      restraint is correct.
    * `visual_candidate_missed` — nothing inside the window, but a frame with
      content within twice the window. The window, not the floor, is what
      excluded it.
    * `alignment_missed` — a frame with content *was* in the window and scored
      below `min_confidence`. The floor is what excluded it.
    * `unknown` — anything the other three do not honestly cover.

    This measures; it decides nothing. `min_confidence` and `window_s` are not
    read for anything but recomputing the score that was already computed, and
    no alignment record is changed here.
    """
    cfg = cfg or AlignmentConfig()
    index = _FrameIndex(ev.visual)
    snip = {s.id: s for s in ev.snippets}
    counts = {"speech_only": 0, "visual_candidate_missed": 0,
              "alignment_missed": 0, "unknown": 0}

    for r in records:
        if r.frame_ids:
            continue
        sn = snip.get(r.snippet_id)
        if sn is None:
            counts["unknown"] += 1
            continue
        lo, hi = sn.start_s - cfg.window_s, (sn.end_s or sn.start_s) + cfg.window_s
        near = index.in_window(lo, hi)
        with_content = [f for f in near if _has_content(f)]
        if with_content:
            best = max(_score_frame(f, sn, cfg) for f in with_content)
            counts["alignment_missed" if best < cfg.min_confidence else "unknown"] += 1
            continue
        if near:
            counts["speech_only"] += 1
            continue
        wider = [f for f in index.in_window(sn.start_s - 2 * cfg.window_s,
                                            (sn.end_s or sn.start_s) + 2 * cfg.window_s)
                 if _has_content(f)]
        counts["visual_candidate_missed" if wider else "speech_only"] += 1
    return counts


def _has_content(frame: VisualFrame) -> bool:
    """Anything a reader could be shown: readable characters or a model note."""
    return bool((frame.ocr_text or "").strip() or (frame.vlm_caption or "").strip())


def _score_frame(frame: VisualFrame, sn: Snippet, cfg: AlignmentConfig) -> float:
    dist, inside = _span_distance(frame.t_start, sn.start_s, sn.end_s or sn.start_s)
    temporal = 0.62 if inside else _temporal(0.62, dist, cfg.window_s)
    ocr = cfg.ocr_boost * float(frame.ocr_confidence) if _usable_read(frame, cfg) else 0.0
    return temporal + ocr


def align_evidence(ev: Evidence, cfg: AlignmentConfig | None = None) -> tuple[list[Alignment], dict]:
    """Align every snippet. Returns the records and the run's measurements.

    Three passes, in this order, because each depends on the previous one:
    candidate scoring (needs only the span), segment agreement (needs every
    snippet's winner in that segment), then final selection and method naming
    (needs the boosts to know what decided the ranking).
    """
    cfg = cfg or AlignmentConfig()
    index = _FrameIndex(ev.visual)
    snip_by_id = {s.id: s for s in ev.snippets}
    seg_of = {sid: seg.id for seg in ev.segments for sid in seg.snippet_ids}

    # -- pass 1: candidates and their base scores ----------------------------
    candidates: dict[str, list[_Candidate]] = {}
    considered = 0
    for sn in ev.snippets:
        lo, hi = sn.start_s - cfg.window_s, (sn.end_s or sn.start_s) + cfg.window_s
        rows: list[_Candidate] = []
        for f in index.in_window(lo, hi):
            dist, inside = _span_distance(f.t_start, sn.start_s, sn.end_s or sn.start_s)  # type: ignore[arg-type]
            temporal = 0.62 if inside else _temporal(0.62, dist, cfg.window_s)
            ocr = cfg.ocr_boost * float(f.ocr_confidence) if _usable_read(f, cfg) else 0.0
            rows.append(_Candidate(frame=f, distance_s=dist, overlaps=inside,
                                   temporal=temporal, ocr=ocr,
                                   notes=[("overlaps the spoken span" if inside else
                                           f"{dist:.2f}s from the span")] +
                                         ([f"usable read at {f.ocr_confidence}"] if ocr else [])))
        candidates[sn.id] = rows
        considered += len(rows)

    # -- pass 2: segment agreement ------------------------------------------
    # A frame that wins for several snippets in one segment is the frame the
    # segment is *about*, and the spoken sentences cannot all be describing a
    # different slide each. Capped, and applied after the base scores so it can
    # only reorder near-ties.
    winners: dict[str, list[str]] = {}
    for sid, rows in candidates.items():
        if not rows:
            continue
        top = max(rows, key=lambda r: r.score)
        if top.score >= cfg.min_confidence:
            seg = seg_of.get(sid)
            if seg:
                winners.setdefault(seg, []).append(top.frame.frame_id)
    shared = {fid for fids in winners.values() if len(fids) > 1 for fid in set(fids)}
    for rows in candidates.values():
        for r in rows:
            if r.frame.frame_id in shared and not r.overlaps:
                r.segment = cfg.segment_boost
                r.notes.append("illustrates other snippets in this segment")

    # -- pass 3: select, name the method ------------------------------------
    records: list[Alignment] = []
    for sn in ev.snippets:
        rows = candidates[sn.id]
        by_time = sorted(rows, key=lambda r: (r.distance_s, not r.overlaps))
        ranked = sorted(rows, key=lambda r: -r.score)
        keep = [r for r in ranked if r.score >= cfg.min_confidence][:cfg.max_frames]

        if not keep:
            # The honest branch. Recorded rather than omitted, and the confidence
            # is the fixed sentinel so it cannot be mistaken for a weak match.
            records.append(Alignment(
                snippet_id=sn.id, frame_ids=[], confidence=NO_FRAME_CONFIDENCE,
                method="no-relevant-frame", start_s=sn.start_s, end_s=sn.end_s,
                segment_id=seg_of.get(sn.id),
                notes=(f"no frame within ±{cfg.window_s}s reached the "
                       f"{cfg.min_confidence} floor"
                       + (f"; {len(rows)} candidate(s) considered" if rows else
                          "; no frame fell in the window"))))
            continue

        top = keep[0]
        top_by_time = by_time[0].frame.frame_id if by_time else None
        if top.segment > 0:
            method = "segment-context"
        elif top.frame.frame_id != top_by_time and top.ocr > 0:
            method = "ocr-assisted"
        elif top.overlaps:
            method = "overlap"
        elif top.frame.frame_id != top_by_time:
            method = "bounded-window"
        else:
            method = "time-nearest"

        records.append(Alignment(
            snippet_id=sn.id, frame_ids=[r.frame.frame_id for r in keep],
            confidence=round(keep[0].score, 4), method=method,
            start_s=sn.start_s, end_s=sn.end_s, segment_id=seg_of.get(sn.id),
            notes="; ".join(keep[0].notes)))

    metrics = {
        "snippets_processed": len(ev.snippets),
        "candidate_frames_considered": considered,
        "aligned_snippets": sum(1 for r in records if r.frame_ids),
        "no_relevant_frame": sum(1 for r in records if not r.frame_ids),
        "aligned_frame_links": sum(len(r.frame_ids) for r in records),
        "distinct_frames_used": len({f for r in records for f in r.frame_ids}),
        "frames_total": len(ev.visual),
        "frames_unmeasured_excluded": len(index.unmeasured),
        "methods": {m: sum(1 for r in records if r.method == m)
                    for m in sorted({r.method for r in records})},
        # How far the frames that did align actually were. A denser frame set
        # makes more alignments possible, and the failure this guards against is
        # that the extra ones are all to distant frames — a report that looks
        # better covered while every claim it attaches to a frame is attached to
        # the wrong moment. `beyond_window` counts alignments whose frame is
        # further from the snippet than the window allows, which is zero by
        # construction; what it catches is a *window change* in a later commit,
        # not a scoring change today.
        "alignment_distance_s": _distance_stats(ev, records),
        # Why the unaligned ones are unaligned. Measured on every run, read by
        # nobody in the pipeline: it is the number that says whether the window
        # and the floor are costing recall, and it is only trustworthy if it is
        # collected the same way every time rather than on demand.
        "no_frame_reasons": classify_unaligned(ev, records, cfg),
    }
    return records, metrics


def frames_to_snippets(records: list[Alignment], snippets: list[Snippet]) -> dict[str, list[str]]:
    """The inverse view: which snippets a frame illustrates.

    Derived from the same records rather than computed separately, so the two
    directions cannot disagree — a second, independent reverse alignment is a
    second answer to a question the forward one already answered. Snippets are
    listed in transcript order, so the first entry is where the frame first came
    up rather than an arbitrary one.
    """
    order = {s.id: i for i, s in enumerate(snippets)}
    out: dict[str, list[str]] = {}
    for r in records:
        for fid in r.frame_ids:
            out.setdefault(fid, []).append(r.snippet_id)
    for sids in out.values():
        sids.sort(key=lambda s: order.get(s, 0))
    return out


def segment_frame_ids(records: list[Alignment], segments: list[Segment]) -> dict[str, list[str]]:
    """Each segment's frame set: every frame paired to any snippet inside it."""
    out: dict[str, list[str]] = {s.id: [] for s in segments}
    for r in records:
        seg = r.segment_id
        if not seg or seg not in out:
            continue
        for fid in r.frame_ids:
            if fid not in out[seg]:
                out[seg].append(fid)
    return out


def to_artifact(records: list[Alignment], cfg: AlignmentConfig, metrics: dict,
                candidates: list[dict] | None = None) -> dict:
    """The standalone `alignment.json`.

    A diagnostic view of records that are ALSO in `evidence.json`, not a second
    store: `evidence.json` stays canonical, and this file exists so the candidate
    table is inspectable without reading the whole evidence document. The two are
    written in the same call, from the same objects.
    """
    return {
        "version": 1,
        "config": cfg.to_dict(),
        "metrics": metrics,
        "records": [r.to_dict() for r in records],
        "candidates": candidates or [],
    }


def candidate_table(records: list[Alignment], ev: Evidence, cfg: AlignmentConfig) -> list[dict]:
    """Re-derive the per-candidate working for the artifact.

    Cheap enough to recompute (it is the same bisect pass) and it means the
    artifact's working cannot drift from the records that were selected, which
    is the failure mode of storing the two at different moments.
    """
    index = _FrameIndex(ev.visual)
    rows: list[dict] = []
    for r in records:
        sn = next((s for s in ev.snippets if s.id == r.snippet_id), None)
        if sn is None:
            continue
        lo, hi = sn.start_s - cfg.window_s, (sn.end_s or sn.start_s) + cfg.window_s
        picked = set(r.frame_ids)
        for f in index.in_window(lo, hi):
            dist, inside = _span_distance(f.t_start, sn.start_s, sn.end_s or sn.start_s)  # type: ignore[arg-type]
            temporal = 0.62 if inside else _temporal(0.62, dist, cfg.window_s)
            ocr = cfg.ocr_boost * float(f.ocr_confidence) if _usable_read(f, cfg) else 0.0
            rows.append({
                "snippet_id": sn.id, "frame_id": f.frame_id,
                "distance_s": round(dist, 3), "overlaps": inside,
                "temporal": round(temporal, 4), "ocr_boost_applied": round(ocr, 4),
                "score": round(min(MAX_PAIRING_CONFIDENCE, temporal + ocr), 4),
                "selected": f.frame_id in picked,
            })
    return rows
