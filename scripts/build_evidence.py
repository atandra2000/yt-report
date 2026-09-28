"""Build `evidence.json` from what the fetcher already produced.

This is the extract step, and it is deliberately small. It does not summarise,
does not judge, and does not decide what matters — it turns artifacts that
already exist on disk into typed, provenance-carrying records and nothing more.
The two extraction rules that are real, rather than approximations of a model,
are:

* **Metadata is evidence.** Title, channel, URL, description links, duration and
  chapter boundaries are things the platform states, so they become items citing
  the metadata field they came from — the same status as anything else, with the
  source named.
* **A number is evidence when it is spoken.** Numeric literals are lifted from
  transcript snippets with the snippet's own verbatim text attached, so a figure
  can be checked against the sentence that carried it.

**What this module deliberately does NOT do is turn a number seen on screen into
an item.** It used to, mechanically, with the same pattern the transcript pass
uses; that pass has moved to `semantic.py`, which is the only stage with enough
context to know whether `0.0%` is a benchmark result or a subscriber counter's
delta. One owner per decision, and the mechanical version is the one that
produced a document where every legible glyph looked equally like a fact.

Hedging is detected, not normalised: a figure introduced by "could", "about",
"up to" and friends is recorded as `hedged`, because a hedge flattened into an
assertion is a fabrication that passes every provenance check in the document.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from evidence_model import (
    SCHEMA_VERSION, Evidence, Item, Segment, Snippet, VisualFrame,
)
from evidence_provenance import (
    frame_provenance, metadata_provenance, transcript_provenance,
)
from ocr import MAX_REGIONS
from versions import SAMPLING_VERSION

# Units kept tight on purpose. A greedy unit list invents units ("32nd", "4x4").
# The per-second rates are in for a domain reason, not a general one: a table of
# `18 tok/s` is the single most common measurement in this material, and a bare
# `18` with its unit refused is a row the pipeline drops. Currency is split out
# because it reads BEFORE its digits ("$500"), and a cost is the case hedging
# matters most on — "could cost $500" is the example the whole status field
# exists for.
_CURRENCY = r"[$€£]"
_UNITS = (
    r"USD|EUR|GBP|"
    r"%|tok/s|tokens/s|words/s|req/s|requests/s|rps|qps|fps|GB/s|TB/s|"
    r"GB|TB|MB|KB|TFLOPS|PFLOPS|FLOPS|TOPS|GHz|MHz|ms|ns|µs|"
    r"hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?|years?|"
    r"x|×|k|K|M|B|bn|billion|million|thousand|trillion"
)
NUM_RE = re.compile(
    rf"(?<![\w.])(?:(?P<cur>{_CURRENCY})\s*)?"
    rf"(?P<value>\d{{1,3}}(?:,\d{{3}})+|\d+(?:\.\d+)?)\s*(?P<unit>{_UNITS})?(?![\w])",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")

# A hedge in the clause before a figure changes what the figure asserts. Scoped
# to a short lookback so "could" four sentences earlier does not taint it.
_HEDGES = (
    "could", "may ", "might", "maybe", "perhaps", "possibly", "probably",
    "approximately", "roughly", "about", "around", "nearly", "almost", "up to",
    "at least", "at most", "an estimated", "estimate of", "estimated", "i think",
    "we think", "should be", "would be", "expected to", "or so", "give or take",
    "if i recall", "if memory serves", "don'T quote me", "don't quote me",
)
_LOOKBACK = 90


def hedge_before(text: str, at: int) -> str | None:
    """Return the hedge word governing the number at `at`, or None."""
    window = text[max(0, at - _LOOKBACK):at].lower()
    for h in _HEDGES:
        if h in window:
            return h.strip()
    return None


def build_snippets(raw: list[dict]) -> list[Snippet]:
    """Assign stable ids by position. Position is the only identity a transcript
    fragment has, and it is stable across runs of the same fetch, which is what
    keeps a regenerated evidence.json from churning every citation."""
    out = []
    for i, s in enumerate(raw, start=1):
        start = float(s["start"])
        end = start + float(s.get("dur") or 0.0)
        out.append(Snippet(id=f"s{i:03d}", start_s=round(start, 3), end_s=round(end, 3),
                           text=(s.get("text") or "").strip()))
    return out


def _transcript_header(lang_used: str, used_asr: bool, snippets: list[Snippet], word_count: int) -> dict:
    """Split the fetcher's single `lang_used` string into the header's fields.

    The fetcher packs provenance into one string ("en (ASR: whisper
    large-v3-turbo)"), which is unreadable as data and loses the distinction
    between auto-generated and human captions. Splitting it here is what lets a
    reader tell a caption track from a local transcription.
    """
    lang = (lang_used or "").split(" (")[0].strip()
    auto = "(auto)" in (lang_used or "")
    if not lang or lang.lower() in ("none", "unknown"):
        # The fetcher could not say which language it read, and a header whose
        # `language` is null is a hole in the provenance: a reader cannot tell a
        # video with no transcript from a video whose language nobody recorded.
        # The words themselves still say, and the detector that gates entity
        # extraction already answers the question for the same text.
        from lang_resources import detect_languages
        found, _morph = detect_languages(" ".join(s.text for s in snippets[:80]))
        if found:
            lang = found[0]
            auto = True   # detected, not declared by the caption track
    if used_asr or "ASR:" in (lang_used or ""):
        source = "asr-detected"
        model = lang_used.split("ASR:", 1)[1].strip().rstrip(")") if "ASR:" in (lang_used or "") else None
        backend = "mlx-whisper" if model else None
    else:
        source = "captions"
        model = backend = None
    return {
        "language": lang or None,
        "language_source": source,
        "is_generated": auto or used_asr,
        "asr_backend": backend,
        "asr_model": model,
        "word_count": word_count,
        "snippet_count": len(snippets),
    }


def _video_block(meta: dict, duration: float | None) -> dict:
    chapters = []
    for c in meta.get("chapters") or []:
        try:
            start = float(c.get("start_time"))
        except (TypeError, ValueError):
            continue  # a chapter without a usable start cannot bound a segment
        chapters.append({
            "start_s": start,
            "end_s": float(c["end_time"]) if c.get("end_time") is not None else None,
            "title": c.get("title") or "",
            "verified_against_transcript": False,  # unverified: uploader metadata, see SKILL.md 2
        })
    return {
        "id": meta.get("id"),
        "url": meta.get("webpage_url"),
        "title": meta.get("title"),
        "channel": meta.get("channel"),
        "upload_date": meta.get("upload_date"),
        "duration_s": duration,
        "description": meta.get("description") or "",
        "tags": list(meta.get("tags") or []),
        "thumbnail": meta.get("thumbnail"),
        "chapters": chapters,
        "sponsors": [],  # no sponsor pass exists; an empty list says so, an inferred one would not
    }


def _metadata_items(video: dict) -> list[Item]:
    """Items the platform states outright. Each cites the exact field, so a
    reader can check any of them against the same document."""
    items: list[Item] = []
    n = 0

    def add(itype: str, value, display: str | None, ref: str, notes: str | None = None) -> None:
        nonlocal n
        n += 1
        items.append(Item(
            id=f"itm-{n:03d}", type=itype, value=str(value), display=display,
            provenance=[metadata_provenance(ref)], status="asserted",
            confidence=1.0, notes=notes,
        ))

    if video.get("title"):
        add("topic", video["title"], None, "title", "uploader-supplied title; not a verified summary")
    if video.get("channel"):
        add("organization", video["channel"], None, "channel")
    if video.get("url"):
        add("link", video["url"], video["url"], "url")
    if video.get("upload_date"):
        add("claim", video["upload_date"], None, "upload_date",
            "upload date, not a filming date")
    if video.get("duration_s"):
        # int where it is whole: "300.0s" is a float artifact, and a duration
        # written with a spurious decimal is one more thing to read past.
        secs = video["duration_s"]
        secs = int(secs) if float(secs).is_integer() else secs
        add("number", secs, f"{secs}s", "duration_s")
    seen: set[str] = set()
    for url in URL_RE.findall(video.get("description") or ""):
        if url in seen:
            continue
        seen.add(url)
        add("link", url, url, "description", "uploader-supplied description link")
    return items


def _transcript_items(snippets: list[Snippet], taken_ids: set[str]) -> list[Item]:
    """Numeric literals, each carrying the sentence that spoke it."""
    items: list[Item] = []
    n = 0
    for sn in snippets:
        text = sn.text
        for m in NUM_RE.finditer(text):
            raw = m.group("value")
            unit = m.group("unit")
            cur = m.group("cur")
            # A bare number with no unit is far more often an ordinal or a year
            # than a measurement. Carrying it would bury the real figures; the
            # schema accepts unit=None, but the extractor does not guess.
            if not unit and not cur:
                continue
            value = raw.replace(",", "")
            key = f"{value}{cur or unit}|{sn.id}".lower()
            if key in taken_ids:
                continue
            taken_ids.add(key)
            n += 1
            hedge = hedge_before(text, m.start())
            # Currency reads before its digits ("$500"), everything else after.
            display = f"{cur}{raw}" if cur else f"{raw}{unit}"
            items.append(Item(
                id=f"itm-{n + 1000:04d}",  # 1000+ keeps transcript items clear of metadata items
                type="number", value=value, unit=cur or unit, display=display,
                verbatim=text, status="hedged" if hedge else "asserted",
                provenance=[transcript_provenance(sn)],
                confidence=0.9,
                notes=f"spoken as a hedge ({hedge!r}); hedge preserved" if hedge else None,
            ))
    return items


def _visual(frames_index: list[dict], visual: dict | None = None) -> list[VisualFrame]:
    """Represent frames, carrying whatever an OCR pass actually read.

    A frame with no measured timestamp keeps `t_start=None` and lands in
    `unmeasured_frames` — the OCR pass never supplies one, because a provider
    sees a file, not a clock.

    Frames whose read is recorded as a repeat of an earlier frame keep no text of
    their own and name the frame that carries it. That is the only compaction
    here: nothing is merged, and every frame that showed a value still points at
    the read that saw it.

    A `vlm` block on the same record is restored the same way — the model's
    observation, its status and its structured record travel with the frame, so a
    re-run over an existing artifact directory can still cite what a model said
    about a frame even when no model is configured this time.
    """
    reads = {f["frame_id"]: f for f in (visual or {}).get("frames", [])}
    out = []
    for f in frames_index:
        fid = f.get("frame_id") or Path(f.get("file", "")).stem
        t = f.get("t_start")
        r = reads.get(fid) or {}
        res = r.get("result") or {}
        text = r.get("text", res.get("text"))
        status = res.get("status")
        if r.get("same_as"):
            # A repeat: the read belongs to the frame named here.
            text, status = None, "repeat"
        seen = r.get("vlm") if isinstance(r.get("vlm"), dict) else {}
        out.append(VisualFrame(
            frame_id=fid,
            file=f.get("file") or f"frames/{fid}.jpg",
            kind=f.get("kind") or ("uniform" if fid.startswith("u") else "scene" if fid.startswith("s") else "unknown"),
            t_start=round(float(t), 3) if isinstance(t, (int, float)) else None,
            width=res.get("width") or f.get("width"),
            height=res.get("height") or f.get("height"),
            upscale=bool(f.get("upscale", False)),
            ocr_text=text if (text or "").strip() else None,
            ocr_confidence=res.get("confidence"),
            ocr_status=status,
            ocr_backend=res.get("backend"),
            ocr_read_method=res.get("read_method"),
            ocr_regions=list(res.get("regions") or [])[:MAX_REGIONS],
            vlm_caption=seen.get("summary"),
            vlm_status=seen.get("status"),
            vlm_backend=seen.get("backend"),
            vlm_confidence=seen.get("confidence"),
            vlm_observation=seen or None,
        ))
    return out


def _segments(raw_segs: list[dict], snippets: list[Snippet], item_ids_by_snippet: dict[str, list[str]],
              title_source: str) -> list[Segment]:
    """Rebuild segments over snippet ids, preserving the fetcher's routing.

    Coverage is computed here, not assumed: the count of routed snippets is
    compared against the transcript's own total so a routing bug surfaces as a
    failed validation rather than a silently shorter report.
    """
    out = []
    for i, seg in enumerate(raw_segs, start=1):
        ids = []
        for s in seg.get("snippets", []):
            for sn in snippets:
                if abs(sn.start_s - round(float(s["start"]), 3)) < 0.0015 and sn.text == (s.get("text") or "").strip():
                    ids.append(sn.id)
                    break
        items: list[str] = []
        for sid in ids:
            items.extend(item_ids_by_snippet.get(sid, []))
        out.append(Segment(
            id=f"seg-{i:02d}",
            start_s=round(float(seg["start"]), 3),
            end_s=round(float(seg["end_s"]), 3) if seg.get("end_s") is not None else None,
            title=seg.get("title") or "",
            title_source=title_source,
            snippet_ids=ids,
            item_ids=items,
        ))
    return out


def build_evidence(meta: dict, raw_snippets: list[dict], raw_segs: list[dict],
                   frames_index: list[dict] | None = None,
                   lang_used: str = "", used_asr: bool = False,
                   visual: dict | None = None) -> Evidence:
    """The extract step. Pure: same inputs, same document, byte for byte."""
    snippets = build_snippets(raw_snippets)
    duration = meta.get("duration")
    video = _video_block(meta, float(duration) if isinstance(duration, (int, float)) else None)

    text_words = sum(len(s.text.split()) for s in snippets)
    ev = Evidence(
        schema_version=SCHEMA_VERSION,
        video=video,
        transcript=_transcript_header(lang_used, used_asr, snippets, text_words),
        snippets=snippets,
    )

    ev.visual = _visual(frames_index or [], visual)

    items = _metadata_items(video)
    t_items = _transcript_items(snippets, set())
    items.extend(t_items)

    by_snippet: dict[str, list[str]] = {}
    for it in t_items:
        for p in it.provenance:
            if p.source == "transcript":
                by_snippet.setdefault(p.ref, []).append(it.id)
    ev.items = items

    ev.segments = _segments(raw_segs, snippets, by_snippet,
                            "chapter" if meta.get("chapters_available") else "topic")

    # Snippets carry their segment back-reference, so coverage is checkable from
    # either direction and a mismatch between the two lists is detectable.
    for seg in ev.segments:
        for sid in seg.snippet_ids:
            sn = next((s for s in ev.snippets if s.id == sid), None)
            if sn:
                sn.segment_id = seg.id
    return ev


_FRAMES_MD_RE = re.compile(r"^- `(?P<name>[^`]+\.jpg)` \[(?P<ts>[^\]]+)\]$")


def _frames_from_manifest(md: Path) -> dict[str, float]:
    """Recover measured times from a `FRAMES.md` written by an older run.

    `00:00:00` is deliberately read as UNMEASURED, not as zero. That formatter
    rendered a missing timestamp as zero before this layer existed, so a literal
    zero in a legacy manifest is ambiguous between "the first second of the
    video" and "the sampler could not tell" — and importing it as a fact would
    put a frame at t=0 that may not be there. Second resolution is the best
    available and is kept as such.
    """
    times: dict[str, float] = {}
    try:
        lines = md.read_text().splitlines()
    except OSError:
        return times
    for line in lines:
        m = _FRAMES_MD_RE.match(line.strip())
        if not m:
            continue
        hh, mm, ss = m.group("ts").split(":")
        t = int(hh) * 3600 + int(mm) * 60 + float(ss)
        times[m.group("name")] = None if t == 0 else t  # type: ignore[assignment]
    return times


class _StaleFrameSet(RuntimeError):
    """The frames on disk were sampled by a policy this build does not run."""


def load_frames_index(artifact_dir: str | Path) -> list[dict]:
    """Read `frames.json` if the run wrote one, else fall back to `FRAMES.md`,
    else to the bare `frames/` directory with no timestamps. A frame discovered
    here but never measured is still recorded — as unmeasured, which is a fact.

    A frame set written under a different sampling policy is refused rather than
    read. The frames are real images with real timestamps, so nothing downstream
    can tell that they are the wrong ones — the evidence layer would build a
    perfectly valid document about a video sampled by a policy that no longer
    runs, and every count in it would be a measurement of the old sampler. The
    remedy is in the message: re-run `yt_fetch.py`, which costs one decode.
    """
    d = Path(artifact_dir)
    fj = d / "frames.json"
    if fj.exists():
        try:
            data = json.loads(fj.read_text())
            if isinstance(data, dict):
                written = data.get("sampling_version")
                if written is not None and int(written) != SAMPLING_VERSION:
                    raise _StaleFrameSet(
                        f"frames.json was written by sampling policy v{written}; this "
                        f"build samples under v{SAMPLING_VERSION}. The frames on disk are "
                        f"the wrong set and every count built on them would describe the "
                        f"old policy — re-run yt_fetch.py to re-sample")
                return data.get("frames", [])
            return data
        except (OSError, json.JSONDecodeError):
            return []
    frames = d / "frames"
    if not frames.is_dir():
        return []
    legacy = _frames_from_manifest(d / "FRAMES.md")
    return [{"frame_id": p.stem, "file": f"frames/{p.name}", "t_start": legacy.get(p.name)}
            for p in sorted(frames.glob("*.jpg"))]


def write_evidence(ev: Evidence, artifact_dir: str | Path) -> Path:
    """Write `evidence.json`, then `validation.json`. Validation is written even
    when blocking, because "the run failed and here is exactly why" is the
    output a later stage needs; what it must not do is proceed."""
    from evidence_model import save_evidence
    from evidence_validate import validate_evidence

    d = Path(artifact_dir)
    ev.validation = validate_evidence(ev).to_dict()
    save_evidence(ev, d / "evidence.json")
    (d / "validation.json").write_text(json.dumps(ev.validation, indent=2) + "\n")
    return d / "evidence.json"


def run_pipeline(meta: dict, raw_snippets: list[dict], raw_segs: list[dict],
                 artifact_dir: str | Path, frames_index: list[dict] | None = None,
                 lang_used: str = "", used_asr: bool = False,
                 visual: dict | None = None, vlm_provider=None,
                 align_cfg=None, semantic_cfg=None, use_cache: bool = True,
                 max_vlm_frames: int | None = None, vlm_observations: dict | None = None,
                 ) -> tuple[Path, dict]:
    """Build, align, interpret, extract, validate — in that order, in one pass.

    The order is the dependency graph, not a preference. Alignment runs before
    the VLM pass because the VLM's *frame selection* is defined in terms of what
    aligned to speech — "text on screen that the speaker is not describing" is
    only knowable once the aligner has said what was described. The VLM pass runs
    before semantic extraction because its observations are one of the inputs.
    The validator runs last because it has nothing to check until there is a
    document. `alignment.json` is written from the same records that go into the
    evidence document, so the diagnostic view and the canonical one cannot
    disagree.
    """
    from align import AlignmentConfig, align_evidence, candidate_table, segment_frame_ids, to_artifact
    from semantic import SemanticConfig, extract
    from vlm import DEFAULT_MAX_VLM_FRAMES, apply_observations, build_context, get_provider, merge_into_visual, run_vlm

    d = Path(artifact_dir)
    acfg = align_cfg or AlignmentConfig()
    scfg = semantic_cfg or SemanticConfig()
    budget = DEFAULT_MAX_VLM_FRAMES if max_vlm_frames is None else max_vlm_frames

    ev = build_evidence(meta, raw_snippets, raw_segs, frames_index,
                        lang_used, used_asr, visual)

    alignments, align_metrics = align_evidence(ev, acfg)
    ev.alignment = alignments
    for seg_id, fids in segment_frame_ids(alignments, ev.segments).items():
        seg = next((s for s in ev.segments if s.id == seg_id), None)
        if seg is not None:
            seg.frame_ids = fids

    provider = vlm_provider or get_provider("none")
    if vlm_observations is not None and provider.name == "unavailable":
        # A caller with a fixture map and no provider name still gets the map,
        # rather than having it silently dropped because the default is `none`.
        provider = get_provider("fake", observations=vlm_observations)
    # The context is built here rather than inside the pass because it needs the
    # evidence document — the video title, the segment, the snippets — and the
    # pass deliberately knows nothing but frames and what it is handed.
    contexts = {f.frame_id: build_context(ev, f, alignments) for f in ev.visual}
    vlm_payload = run_vlm(d, ev.visual, alignments, provider, max_frames=budget,
                          context_by_frame=contexts)
    apply_observations(ev.visual, vlm_payload)
    _attach_analysed_frames(ev, vlm_payload)
    _persist_vlm(d, visual, vlm_payload)

    result = extract(ev, alignments, vlm_payload, scfg, use_cache=use_cache)
    ev.items = result.items
    ev.conflicts = result.conflicts
    ev.observations = result.observations
    _file_items_into_segments(ev)

    art = to_artifact(alignments, acfg, align_metrics, candidate_table(alignments, ev, acfg))
    art["semantic"] = result.metrics
    art["vlm"] = vlm_payload.get("summary")
    tmp = d / "alignment.json.tmp"
    tmp.write_text(json.dumps(art, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(d / "alignment.json")

    write_evidence(ev, d)
    metrics = {"alignment": align_metrics, "semantic": result.metrics,
               "vlm": vlm_payload.get("summary")}
    return d / "evidence.json", metrics


def _file_items_into_segments(ev: Evidence) -> None:
    """Attach every item to the segment its own provenance points at.

    The fetcher's segmentation gives transcript items a segment, because at that
    point every item came from a snippet and the snippet knew where it was. The
    semantic stage adds items that do not: a diagram edge has no snippet, only a
    frame, and a frame is a moment in the video that falls inside exactly one
    segment. Without this step those items belong to no segment, and an item that
    belongs to no segment is an item no consumer can find — which is why the
    report's narrative, its timeline and its visual sections would all be
    structurally incapable of showing the one thing the VLM pass was run for.

    Resolution is by the provenance entry, never by topic similarity: a snippet
    names its segment, a frame has a measured time, and time membership is a fact
    about the video rather than a judgement about relevance. An item whose
    provenance resolves to nothing is left unfiled rather than guessed at.
    """
    by_id = {sg.id: sg for sg in ev.segments}
    seg_of_snippet = {s.id: s.segment_id for s in ev.snippets if s.segment_id}
    span = [(sg.start_s, sg.end_s, sg.id) for sg in ev.segments if sg.start_s is not None]

    def segment_for(t: float | None) -> str | None:
        if t is None:
            return None
        for start, end, sid in span:
            if start <= t <= (end if end is not None else t):
                return sid
        return None

    frames = {f.frame_id: f for f in ev.visual}
    for it in ev.items:
        if any(it.id in sg.item_ids for sg in ev.segments):
            continue
        target: str | None = None
        for p in it.provenance or []:
            if p.source == "transcript":
                target = seg_of_snippet.get(p.ref)
            elif p.source in ("frame", "ocr", "vlm"):
                f = frames.get(p.ref)
                target = segment_for(f.t_start if f is not None else p.t_start)
            if target:
                break
        if target and target in by_id:
            by_id[target].item_ids.append(it.id)


def _attach_analysed_frames(ev: Evidence, payload: dict) -> None:
    """File every analysed frame under the segment whose time range contains it.

    The aligner pairs a snippet with the few frames nearest to it, so a dense
    screen frame can be analysed, cost a call, and belong to no segment — and a
    frame in no segment is a frame no consumer reads, so its observation would be
    paid for and then discarded without a trace. Time membership is not a second
    opinion about relevance: `Segment` already defines itself by span, and this
    only lets a frame the model was asked about be filed under the topic that was
    playing when it was on screen.

    A frame the aligner did pair is already listed and is left alone, so the
    aligner's choice still wins wherever there was one.
    """
    analysed = {str(f.get("frame_id")) for f in payload.get("frames", [])
                if f.get("status") == "ok"}
    if not analysed:
        return
    by_id = {f.frame_id: f for f in ev.visual}
    for seg in ev.segments:
        listed = set(seg.frame_ids)
        for fid in sorted(analysed):
            if fid in listed:
                continue
            f = by_id.get(fid)
            if f is None or f.t_start is None or seg.start_s is None:
                continue
            end = seg.end_s if seg.end_s is not None else f.t_start
            if seg.start_s <= f.t_start <= end:
                seg.frame_ids.append(fid)
                listed.add(fid)


def _persist_vlm(artifact_dir: Path, visual: dict | None, payload: dict) -> None:
    """Fold the model record into `visual.json` — the one canonical visual file.

    The alternative, a `vlm.json`, is a second copy of what a model said about
    the same frames, and a reader who finds the two disagreeing has no way to
    tell which one the model said. `evidence.json` remains canonical for what is
    *claimed*; `visual.json` is the visual layer's own record, and the pass's
    observations belong there.

    A run with no OCR produced no `visual.json` to extend, so the payload is
    written as one. That keeps the artifact set identical whether or not the OCR
    stage ran, which is the property `--no-ocr` exists to preserve.
    """
    from ocr import load_visual, write_visual
    from vlm import merge_into_visual

    existing = visual if visual is not None else load_visual(artifact_dir)
    if existing is None:
        existing = {"version": 1, "backend": None, "summary": None, "frames": []}
        ids = {str(f.get("frame_id")) for f in payload.get("frames", [])}
        existing["frames"] = [{"frame_id": fid, "vlm": None} for fid in sorted(ids)]
        for entry in existing["frames"]:
            rec = next((r for r in payload.get("frames", [])
                        if str(r.get("frame_id")) == entry["frame_id"]), None)
            entry["vlm"] = rec
        existing["vlm_summary"] = payload.get("summary")
    else:
        merge_into_visual(existing, payload)
    write_visual(artifact_dir, existing)


def load_artifact_inputs(artifact_dir: str | Path) -> dict:
    """Read the artifacts a run already produced, for the extract stage.

    Every one of these is optional and its absence is reported rather than
    assumed: an artifact directory from before a stage existed must still run,
    and must record that the stage had nothing to work from instead of
    pretending the input was empty and clean.
    """
    d = Path(artifact_dir)
    out: dict = {"meta": {}, "snippets": [], "segments": [], "visual": None,
                 "vlm": None, "frames": [], "notes": []}

    meta_p = d / "meta.json"
    if meta_p.exists():
        try:
            out["meta"] = json.loads(meta_p.read_text())
        except (OSError, json.JSONDecodeError):
            out["notes"].append("meta.json unreadable; metadata items omitted")

    tj = d / "transcript.json"
    if tj.exists():
        try:
            data = json.loads(tj.read_text())
            out["snippets"] = data.get("snippets") or []
            out["segments"] = data.get("segments") or []
            # The keys `yt_fetch` writes. They are `language` and `asr`, not
            # `lang_used` and `used_asr`: reading names the writer does not use
            # returns empty for both, and a document then says its transcript
            # came from caption tracks when it came from a local transcription —
            # with no `asr_backend`, no `asr_model`, and a `language` that falls
            # back to whatever the text happens to look like. Found by the ASR
            # round-trip test, which is the only thing that reads this file back.
            out["lang_used"] = data.get("language", "")
            out["used_asr"] = bool(data.get("asr"))
            if out["segments"] and not any("snippets" in s for s in out["segments"]):
                # Written by a fetch that recorded only each segment's start
                # times. Segment membership cannot be re-derived from those, and
                # proceeding would assign every snippet to nothing — so say which
                # file is stale rather than reporting 432 unassigned snippets.
                out["notes"].append(
                    "transcript.json holds only segment start times (an older format); "
                    "segment membership cannot be rebuilt from it — re-run yt_fetch.py")
        except (OSError, json.JSONDecodeError):
            out["notes"].append("transcript.json unreadable; no snippets")
    else:
        out["notes"].append("transcript.json absent (pre-dates the fetch change); "
                            "re-run yt_fetch.py to get snippet timings")

    out["frames"] = load_frames_index(d)
    from ocr import load_visual
    out["visual"] = load_visual(d)
    # The model's own record arrives inside `visual.json`, one `vlm` key per
    # frame, so `_visual` can restore it onto the frames and a re-run costs no
    # calls. There is no separate `vlm.json` to go stale independently of the
    # visual layer it describes.
    out["notes"].append("model observations are read from the 'vlm' block of visual.json")
    return out


def main(argv: list[str] | None = None) -> int:
    """Extract stage entry point: artifact directory in, validated evidence and
    a validated report out.

    Reads only what a previous stage wrote. It cannot see a frame it did not
    read, and it re-runs on an existing directory for no download cost, which is
    what makes adding a stage to an already-fetched video free.
    """
    import argparse

    from align import AlignmentConfig
    from semantic import SemanticConfig
    from vlm import DEFAULT_MAX_VLM_FRAMES, VLMConfig

    ap = argparse.ArgumentParser(
        description="Build evidence.json and REPORT.md from an artifact directory: "
                    "build, align, extract, validate, report, validate the report.")
    ap.add_argument("artifact_dir", help="directory holding meta.json / transcript.json / frames/")
    ap.add_argument("--window", type=float, default=None,
                    help="alignment candidate half-window in seconds (default 4.0)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="frames retained per snippet (default 3)")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached segment extractions")
    ap.add_argument("--vlm-provider", default=None, metavar="NAME",
                    help="anthropic | openai | openrouter | ollama | lmstudio | llamacpp | "
                         "custom | none. Overrides VLM_PROVIDER. 'none' (the default) records "
                         "that no model ran and invents nothing")
    ap.add_argument("--no-vlm", action="store_true",
                    help="skip the visual pass entirely; same as --vlm-provider none")
    ap.add_argument("--vlm-model", default=None, help="model id (overrides VLM_MODEL)")
    ap.add_argument("--vlm-base-url", default=None, metavar="URL",
                    help="OpenAI-compatible base URL, e.g. for a local server or a gateway "
                         "(overrides VLM_BASE_URL)")
    ap.add_argument("--max-vlm-frames", type=int, default=None,
                    help=f"hard cap on model calls for this run (default {DEFAULT_MAX_VLM_FRAMES}); "
                         "0 sends no frame at all")
    ap.add_argument("--vlm-timeout", type=float, default=None,
                    help="per-call timeout in seconds (overrides VLM_TIMEOUT, default 90)")
    ap.add_argument("--detail", default="balanced", choices=["brief", "balanced", "deep"],
                    help="report length budget; the evidence decides where inside the range a "
                         "particular video lands (default balanced)")
    ap.add_argument("--synth-provider", default=None,
                    help="OpenAI-compatible model to write the report with: openai | "
                         "openrouter | ollama | lmstudio | llamacpp | custom | none "
                         "(overrides SYNTH_PROVIDER). Off by default — a model call "
                         "costs money, so it is a decision, not a fallback")
    ap.add_argument("--synth-model", default=None,
                    help="model id for the synthesis writer (overrides SYNTH_MODEL)")
    ap.add_argument("--synth-base-url", default=None, metavar="URL",
                    help="base URL for the synthesis writer (overrides SYNTH_BASE_URL)")
    ap.add_argument("--output-language", default=None,
                    choices=["auto", "en", "es", "fr", "de", "pt", "it", "nl", "ja", "zh", "hi", "ko", "ru", "ar"],
                    help="language for the rendered prose; evidence stays in its source "
                         "language and identifiers are never translated (default auto, "
                         "which follows the video)")
    ap.add_argument("--no-critic", action="store_true",
                    help="skip the second-pass critique; the writer's first draft is "
                         "validated as-is")
    ap.add_argument("--synthesis-only", action="store_true",
                    help="regenerate REPORT.md from the evidence already on disk; no "
                         "download, no OCR, no VLM pass")
    ap.add_argument("--no-report", action="store_true",
                    help="build and validate evidence only; write no REPORT.md")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    d = Path(args.artifact_dir).expanduser()
    if not d.is_dir():
        print(f"not a directory: {d}", file=os.sys.stderr)
        return 2

    data = load_artifact_inputs(d)
    for n in data["notes"]:
        print(f"  note: {n}", file=os.sys.stderr)
    if not data["snippets"]:
        print("no transcript snippets; run yt_fetch.py first", file=os.sys.stderr)
        return 3

    acfg = AlignmentConfig()
    if args.window is not None:
        acfg.window_s = args.window
    if args.max_frames is not None:
        acfg.max_frames = args.max_frames

    cfg = VLMConfig.from_env(
        provider="none" if args.no_vlm else args.vlm_provider,
        model=args.vlm_model, base_url=args.vlm_base_url,
        timeout=args.vlm_timeout, max_frames=args.max_vlm_frames)
    from vlm import provider_from_config
    provider = provider_from_config(cfg)
    if cfg.provider not in ("none", "") and not provider.available:
        # Said once, here, and again in the artifact. A run that asked for a
        # model and silently got none would produce a document that looks
        # complete and is missing the one layer it was asked for.
        print(f"  note: VLM unavailable — {getattr(provider, 'reason', 'unknown')}", file=os.sys.stderr)

    if args.synthesis_only:
        # Report from the evidence already on disk. The flag promises no
        # download, no OCR and no VLM pass, so it must not run the extraction
        # pipeline at all — routing it through `run_pipeline` would rebuild
        # evidence.json from raw inputs, which is both slower and a different
        # operation from the one that was asked for.
        from synthesis import SynthConfig
        synth = SynthConfig.from_env(
            provider=args.synth_provider, model=args.synth_model,
            base_url=args.synth_base_url, output_language=args.output_language,
            detail=args.detail, critic=not args.no_critic)
        return write_and_validate_report(d, args.detail, quiet=args.quiet, synth=synth)

    path, metrics = run_pipeline(
        data["meta"], data["snippets"], data["segments"], d, data["frames"],
        data.get("lang_used", ""), data.get("used_asr", False),
        data["visual"], provider, acfg, SemanticConfig(),
        use_cache=not args.no_cache,
        max_vlm_frames=DEFAULT_MAX_VLM_FRAMES if args.max_vlm_frames is None else args.max_vlm_frames)

    if not args.quiet:
        print(f"{path}")
        print(json.dumps(metrics, indent=2)[:4000])
    res = json.loads((d / "validation.json").read_text())
    print(f"{'VALID' if res.get('valid') else 'INVALID'}  {d / 'evidence.json'}")
    for e in res.get("blocking_errors", []):
        print(f"  BLOCK  {e['code']}  {e['where']}: {e['message']}")
    if not res.get("valid"):
        # The report is only allowed to be built from evidence that passed. A
        # report rendered from a document the validator rejected would inherit
        # every one of its defects and add a layer of formatting over the top.
        print("evidence is invalid; no report written", file=os.sys.stderr)
        return 1
    if args.no_report:
        return 0

    # Synthesis is off unless asked for. A model call costs money, so running
    # one is a decision the user makes, never a fallback that happens because a
    # key was in the environment — the same rule the VLM pass follows.
    synth = None
    if args.synth_provider or args.synthesis_only:
        from synthesis import SynthConfig
        synth = SynthConfig.from_env(
            provider=args.synth_provider, model=args.synth_model,
            base_url=args.synth_base_url, output_language=args.output_language,
            detail=args.detail, critic=not args.no_critic)

    rc = write_and_validate_report(d, args.detail, quiet=args.quiet, synth=synth)
    return 0 if rc == 0 else 1


def write_and_validate_report(artifact_dir: str | Path, detail: str = "balanced",
                              quiet: bool = False, synth: "SynthConfig | None" = None) -> int:
    """Generate `REPORT.md` from the validated evidence, then check it.

    Two gates, in order, and the second is the one that can refuse to write: the
    report is rendered in memory, validated, and only then committed to disk. A
    report file that exists is a report someone will believe, and a report that
    failed its grounding check must never be one of them.
    """
    from report import generate, render_markdown
    from report_validate import validate_report
    from synthesis import InvalidEvidence, SynthConfig, load_valid_evidence, synthesize

    d = Path(artifact_dir)
    try:
        ev = load_valid_evidence(d / "evidence.json")
    except InvalidEvidence as exc:
        print(f"BLOCK  {exc}", file=os.sys.stderr)
        return 1

    # The model-backed path, when one is configured. It never raises and never
    # weakens a check: on any failure it returns the deterministic report and a
    # reason, and the same validator below judges whichever came back.
    outcome = synthesize(ev, synth) if synth is not None else None
    if outcome is None:
        report, mode, reason = generate(ev, detail), "deterministic", ""
    else:
        report, mode, reason = outcome.report, outcome.mode, outcome.fallback_reason
    markdown = render_markdown(report)
    result = validate_report(report, ev, markdown, authored_by=mode)
    result_doc = result.to_dict()
    result_doc["mode"] = mode
    if reason:
        result_doc["fallback_reason"] = reason
    if outcome is not None:
        result_doc["synthesis"] = outcome.to_dict()
    if outcome is not None and outcome.language_note:
        # Said out loud as well as written down. A run that was asked for a
        # language and produced English has not translated anything, and a reader
        # who sees the report in English has no other way to know.
        print(f"  NOTE  {outcome.language_note}", file=os.sys.stderr)
    (d / "report_validation.json").write_text(json.dumps(result_doc, indent=2) + "\n")
    for w in result.warnings:
        print(f"  WARN  {w['code']}  {w['where']}: {w['message']}", file=os.sys.stderr)
    if not result.valid:
        for e in result.blocking_errors[:25]:
            print(f"  BLOCK  {e['code']}  {e['where']}: {e['message']}", file=os.sys.stderr)
        n = len(result.blocking_errors)
        if n > 25:
            print(f"  … and {n - 25} more (see report_validation.json)", file=os.sys.stderr)
        print(f"REPORT.md not written: {n} grounding failure(s)", file=os.sys.stderr)
        return 1

    from report import write_report
    md, js = write_report(report, d)
    if not quiet:
        m = result.metrics
        if reason:
            # A degraded run says so, in the place the reader will look. A
            # silently-downgraded report is indistinguishable from a grounded one.
            print(f"  synthesis unavailable — {reason}")
        print(f"{'VALID' if result.valid else 'INVALID'}  {md}  [{mode}]")
        print(f"  {m['report_word_count']} words · {m['detail_level']} · "
              f"provenance {m['provenance_coverage']:.0%} · "
              f"{m['evidence_items_used']}/{m['evidence_items_total']} evidence items used · "
              f"repetition {m['repetition_ratio']:.1%} · {m['section_count']} sections")
        print(f"  {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
