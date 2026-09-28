"""The report's own gate. Every claim in `REPORT.md` has to survive a question.

`evidence_validate.py` asks whether the evidence document is coherent. This asks
the harder question about the next artifact: whether the document rendered from
it is *grounded* — whether every number, every URL, every timestamp and every
visual statement in the prose can be traced back to something the evidence
document actually says, and whether the report is the size its own evidence
justifies.

The split with the generator matters. `report.py` is written so that it *cannot*
introduce an unsupported fact — there is no free-text slot — and this module does
not take that on trust. A renderer that is correct today and edited carelessly
tomorrow is exactly how a check earns its keep, and a check that only ever runs
on the generator's own happy path proves nothing.

**High precision, low recall**, the rule this repository holds for every checker.
A rule that cries wolf is worse than no rule, because it trains the reader to
skip the output. So each check below is written to fire on a defect that is
*identifiable without judgement*, and the genuinely ambiguous findings — an
unused evidence item, a section that would have been longer — are reported as
metrics rather than as failures. The one judgement call in here, the proper-noun
check, is deliberately narrow: it looks at capitalised words that are not
sentence-initial and asks whether the item they are attributed to contains them.

`python3 report_validate.py <artifact-dir>` prints the metrics and exits non-zero
on a blocking error.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from evidence_model import Evidence
from report import Block, Report, _normalise_url, render_markdown

#: A marker that does not resolve to a real moment in the video is a fabricated
#: citation, which is the single most damaging defect a report can carry: it
#: looks exactly like a correct one.
#:
#: A `mm:ss` marker denotes the *interval* [mm:ss, mm:ss+1), not a point, because
#: that is what a second-resolution rendering of a measured time is. So the test
#: is whether any measured moment falls inside it — not whether it sits within a
#: tolerance of one. A tolerance test here fails on every correct marker whose
#: measured time is 762.96 s, which is a real moment rendered as `12:42`.
def _in_second(t: int, c: float) -> bool:
    return int(c) == t

_MARKER = re.compile(r"\[(\d{1,2}:\d{2})(?:\s*·\s*([a-z\- ]+))?\]")
#: A bracket that is citation-shaped but is not a well-formed marker: digits,
#: colons, dots and dashes only, three to twelve characters. `_MARKER` cannot
#: match one of these — it only matches `\d{1,2}:\d{2}` — so without this the
#: `timestamp-unparsable` branch below was unreachable and a malformed citation
#: like `[04:5]` or `[0:4:31]` sat in a rendered report with nothing looking at
#: it. It looks exactly like the valid markers around it, which is the whole
#: reason a fabricated one is dangerous.
_MARKER_LOOSE = re.compile(r"\[([0-9][0-9:.\-]{1,11})(?:\s*·\s*([a-z\- ]+))?\]")
_LIST_ORDINAL = re.compile(r"^\s*(?:[-*]\s+)?\d+[.)]\s+")
_TIMELINE_LINE = re.compile(r"^\s*(?:[-*]\s*)?\d{1,2}:\d{2}\s+[—-]\s")
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_URL = re.compile(r"https?://[^\s<>\"')\]]+")
_PROPER = re.compile(r"\b([A-Z][A-Za-z0-9]{2,})\b")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'’\-]{2,}")

#: The words a renderer needs in order to write a sentence. Every one of them is
#: connective tissue, not content: a proper noun drawn from this set would mean
#: the report has invented a capitalised common noun, which is not a real failure
#: mode worth a check.
_CONNECTIVES = frozenset("""
the a an and or but if then than that this these those it its they them their there here
for from with without within into onto over under about after before between during while
when where which who whom whose what why how not no nor none both either each every some any
one two three four five six seven eight nine ten more most less least other another such
however therefore thus hence although though because since while as at by in of on to is are
was were be been being has have had do does did can could may might must shall should will
would its it's source narration metadata quoted value figure compared named structure extracted
coverage full degraded failed segment segments step steps item items the—""".split())

#: Words the renderer's own templates contribute. A capitalised word in a block
#: is a fabrication unless it is in the evidence the block cites *or* in this
#: set — and this set exists because the renderer does have fixed prose of its
#: own ("Unresolved conflict", "Not legible"). It is deliberately small and
#: spelled out rather than derived: a set that regenerated itself from the source
#: would absorb every word the renderer ever learns to write, which is the same
#: as no check at all. Adding a template word here is a deliberate widening.
TEMPLATE_PROSE = frozenset({
    "Unresolved", "Conflict", "Not", "Legible", "Visual", "Content", "Interpreted",
    "Model", "Derived", "Evidence", "Source", "Reliability", "Extracted", "Details",
    "Named", "Structure", "Coverage", "Full", "Degraded", "Failed", "Reported",
    "Reading", "Frame", "Narration", "Metadata", "On", "Screen", "Step", "Steps",
    "Timeline", "Executive", "Summary", "Key", "Takeaways", "Detailed", "Notes",
    "People", "Organizations", "Products", "Models", "Tools", "Numbers", "Pricing",
    "Benchmarks", "Comparisons", "Resources", "Links", "Important", "Quotes",
    "Wraps", "Traceable",
})

#: Section kinds whose blocks are factual and therefore must be traceable. A
#: template line in the reliability section is not.
_FACTUAL_KINDS = frozenset({"number", "quote", "comparison", "visual", "narrative",
                            "link", "entity", "timeline", "conflict", "support"})

#: Lower-case words the renderer's own templates contribute, which no evidence
#: document will ever contain. Spelled out rather than derived, for the same
#: reason `TEMPLATE_PROSE` is: a set that regenerated itself from the source
#: would absorb every word the renderer ever learns to write, which is the same
#: as no check at all.
#:
#: Measured, not guessed. `acceptance/template_audit.py` reports the content
#: words a real report emits that appear in no evidence document, across the
#: fifteen fixtures and every artifact on disk; this is that set. Fifteen
#: fixtures were not enough on their own — a real video added four more on the
#: first run — which is the whole reason the audit exists and the reason the
#: set is dated rather than presented as complete. `shows`/`figures` are table
#: headers, `given`/`possibility` the hedge phrasing, `referent` the unlabelled-
#: figure aside, `statement` the supporter phrase for a narration item, `side`
#: the no-figure-attributed comparison aside, `value-mismatch` a conflict *kind*
#: slug printed verbatim. Adding a word here is a deliberate widening.
TEMPLATE_WORDS = frozenset({
    "shows", "shown", "figures", "given", "frame", "frames", "conflict",
    "conflicts", "resolved", "unresolved", "sources", "source", "value-mismatch",
    "possibility", "referent", "side", "statement",
    # `_supporter_phrase` in report.py, which renders an entity as "see the
    # 40 ms figure" when another item names it. The word was missing, so every
    # entity with a corroborating item produced a report the validator then
    # refused to write — a renderer/validator disagreement about the renderer's
    # own vocabulary, which is the only kind that can block a correct report.
    # None of the 20 audited documents happened to contain a supported entity,
    # which is why the audit did not surface it; that is a statement about the
    # corpus, not about the code path.
    "see",
})


@dataclass
class ReportValidation:
    metrics: dict = field(default_factory=dict)
    blocking_errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.blocking_errors

    def to_dict(self) -> dict:
        return {"valid": self.valid, "metrics": self.metrics,
                "blocking_errors": self.blocking_errors, "warnings": self.warnings}


def _err(code: str, where: str, msg: str) -> dict:
    return {"code": code, "where": where, "message": msg}


def _warn(code: str, where: str, msg: str) -> dict:
    return {"code": code, "where": where, "message": msg}


# --------------------------------------------------------------------------- #
# the haystack a claim is allowed to come from
# --------------------------------------------------------------------------- #

def evidence_text(ev: Evidence) -> str:
    """Every string the evidence document contains, in one blob.

    Deliberately generous — it includes notes, captions and verbatim text, not
    just the "official" value fields. The question this answers is "did the report
    say something no source in the document says?", and a word lifted from a
    provenance note is a word the document says.
    """
    parts: list[str] = []
    parts.append(str(ev.video.get("title") or ""))
    parts.append(str(ev.video.get("channel") or ""))
    parts.append(str(ev.video.get("description") or ""))
    parts.append(" ".join(str(t) for t in (ev.video.get("tags") or [])))
    for s in ev.snippets:
        parts.append(s.text)
    for s in ev.segments:
        parts.append(s.title)
    for i in ev.items:
        for f in ("value", "display", "unit", "subject", "qualifier", "verbatim", "notes"):
            v = getattr(i, f, None)
            if v:
                parts.append(str(v))
        if i.detail:
            parts.append(" ".join(str(v) for v in i.detail.values() if isinstance(v, (str, int, float))))
    for f in ev.visual:
        parts.append(f.ocr_text or "")
        parts.append(f.caption or "")
        parts.append(f.vlm_caption or "")
    for c in ev.conflicts:
        parts.append(c.statement)
    return "\n".join(p for p in parts if p)


def evidence_numbers(ev: Evidence) -> set[str]:
    """Every numeric literal anywhere in the evidence, in both comma styles.

    A figure may appear as `1,067` in one place and `1067` in another; matching
    on the normalised form is what lets `1,067` in a report be checked against
    `1067` in an item without the check passing on a *different* number that
    happens to share its digits.
    """
    raw = evidence_text(ev)
    out: set[str] = set()
    for m in _NUM.finditer(raw):
        tok = m.group(0).rstrip(".")
        out.add(tok)
        out.add(tok.replace(",", ""))
    return out


def evidence_urls(ev: Evidence) -> set[str]:
    raw = evidence_text(ev) + "\n" + str(ev.video.get("url") or "")
    return {_normalise_url(u.rstrip("/")) for u in _URL.findall(raw)}


# --------------------------------------------------------------------------- #
# the checks
# --------------------------------------------------------------------------- #

def _all_blocks(report: Report) -> list[tuple[str, Block]]:
    out: list[tuple[str, Block]] = []

    def walk(section, path: str):
        for b in section.blocks:
            out.append((path, b))
        for sub in section.subsections:
            walk(sub, f"{path}/{sub.key}")

    for s in report.sections:
        walk(s, s.key)
    return out


def _strip_markers(text: str) -> str:
    return _MARKER.sub(" ", text)


def check_timestamps(report: Report, ev: Evidence, md: str,
                     errors: list[dict]) -> tuple[int, int]:
    """Every `[mm:ss]` in the report must land on a real moment.

    The candidate times are the measured ones — snippet starts, frame timestamps,
    segment starts. A marker matching none of them is a citation to a moment that
    does not exist, and it is the failure a reader is least able to detect: it
    looks exactly like the markers around it.
    """
    times: set[float] = set()
    for s in ev.snippets:
        times.add(round(float(s.start_s), 2))
    for f in ev.visual:
        if f.t_start is not None:
            times.add(round(float(f.t_start), 2))
    for s in ev.segments:
        if s.start_s is not None:
            times.add(round(float(s.start_s), 2))

    total = resolved = 0
    seen: set[str] = set()
    for m in _MARKER.finditer(md):
        total += 1
        seen.add(m.group(0))
        mm, _, ss = m.group(1).partition(":")
        try:
            t = int(mm) * 60 + int(ss)
        except ValueError:
            errors.append(_err("timestamp-unparsable", m.group(0), f"cannot read {m.group(1)!r}"))
            continue
        if any(_in_second(t, c) for c in times):
            resolved += 1
        else:
            errors.append(_err("timestamp-fabricated", m.group(0),
                               f"{m.group(1)} does not resolve to any measured snippet, "
                               f"frame or segment in the evidence"))
    # Anything citation-shaped that `_MARKER` could not read at all. Counted in
    # the total, because a marker the reader will read as a citation is one the
    # report is asserting with.
    for m in _MARKER_LOOSE.finditer(md):
        if m.group(0) in seen:
            continue
        seen.add(m.group(0))
        total += 1
        errors.append(_err("timestamp-unparsable", m.group(0),
                           f"{m.group(1)!r} is not a well-formed mm:ss marker"))
    return total, resolved


def check_numbers(report: Report, ev: Evidence, md: str,
                  errors: list[dict]) -> tuple[int, int]:
    """Every figure in the prose must be a figure the evidence contains.

    Markers, list ordinals and the header's runtime are excluded — they are
    formatting, not claims. What remains is checked twice, because one check is
    not enough:

    * against every numeric literal in the document, plus the report's own
      declared `report_figures`, which exist so that a counted statement
      ("71 frames were sampled") is distinguishable from an invented one rather
      than exempted by a blanket rule; and
    * against **the items the block actually cites**. The document-wide test is
      set membership, so a figure changed from 20 to 27 passes whenever 27
      happens to appear *somewhere else in the same video* — which, on a report
      carrying ninety figures, is most of the time. Measured on a real video:
      `20` corrupted to `27` was accepted by the document-wide rule and refused
      only once the figure was bound to the item the sentence cites. That is the
      same rule `synth_validate` applies to model prose, for the same reason.

    A block that cites nothing falls back to the document-wide set: there is no
    citation to be bound to, and the weaker test is all that is left.

    What comes back is a count of *numeric facts*. Timestamps are counted
    separately, by `check_timestamps`: they are positions in a video, not
    quantities, and a real report scored 44/44 on "every number is supported"
    when twenty-four of the forty-four were `[HH:MM]` markers that could not
    have been unsupported.
    """
    allowed = evidence_numbers(ev)
    allowed |= {str(v) for v in (report.meta.get("report_figures") or {}).values()}

    def _permitted(token: str, pool: set[str]) -> bool:
        return token in pool or token.replace(",", "") in pool

    for line in md.splitlines():
        if line.startswith("*") and line.endswith("*"):
            continue  # the header's title/channel/runtime line
        if _TIMELINE_LINE.match(line):
            continue  # `00:00 — Introduction`: a section label, not a figure
        line = _LIST_ORDINAL.sub("", line)
        line = _strip_markers(line)
        for m in _NUM.finditer(line):
            tok = m.group(0).rstrip(".")
            if _permitted(tok, allowed):
                continue
            errors.append(_err("number-unsupported", tok,
                               f"{tok!r} appears in the report but in no evidence item, "
                               f"snippet, frame read or declared report figure"))

    by_id = {i.id: i for i in ev.items}
    snippets = {s.id: s for s in ev.snippets}
    checked = 0
    for path, b in _all_blocks(report):
        if b.kind in ("link", "meta", "summary", "timeline") or not b.evidence_ids:
            continue
        cited = {i for i in b.evidence_ids if i in by_id}
        if not cited:
            continue
        checked += 1
        own: set[str] = set()
        for i in cited:
            item = by_id[i]
            own |= evidence_numbers_for(item)
            # A narrative block's words come from the snippet its items were
            # extracted from, and that snippet carries figures the *item* does
            # not: "takes in 784 numbers as an input" is a definition item
            # whose value is "just a function". Binding the rule to the item
            # alone flagged it, and the figure is as grounded as the sentence
            # quoting it. So the block's own provenance is the boundary, which
            # is the same boundary `check_provenance` already enforces.
            for p in (item.provenance or []):
                if p.source == "transcript" and p.ref in snippets:
                    own |= evidence_numbers_for(snippets[p.ref])

        declared = {str(v) for v in (report.meta.get("report_figures") or {}).values()}
        for tok in _NUM.findall(_strip_markers(b.text)):
            tok = tok.rstrip(".")
            if tok in declared or tok.replace(",", "") in declared:
                continue  # a counted statement about the run, owned by no item
            if not _permitted(tok, own):
                errors.append(_err(
                    "number-misattributed", tok,
                    f"{tok!r} is in the report but not in the evidence this block cites "
                    f"({', '.join(sorted(cited))})"))
    total = len(_NUM.findall(_strip_markers(md)))
    return total, total - len([e for e in errors if e["code"].startswith("number-")])


def evidence_numbers_for(item) -> set[str]:
    """The numeric literals one cited thing carries, in the same normalisation
    `evidence_numbers` uses so a figure written `1,200` matches `1200`.

    Takes an `Item` or a `Snippet`: both are things a block's prose can be a
    reading of, and the positional rule has to look at whichever the block's
    provenance actually names.
    """
    fields = (getattr(item, "text", None), getattr(item, "verbatim", None),
              getattr(item, "value", None), getattr(item, "display", None),
              getattr(item, "unit", None), getattr(item, "subject", None),
              getattr(item, "detail", None))
    out: set[str] = set()
    for value in fields:
        for tok in _NUM.findall(str(value or "")):
            out.add(tok.rstrip("."))
    return out


#: The three shapes a figure count takes in Source & Reliability. Read, not
#: guessed: the renderer writes all three sentences and the validator parses all
#: three, so a change to either side fails here rather than quietly decoupling
#: them. The withheld and detected counts are matched separately because they are
#: two different numbers in the same sentence, and reading the wrong one makes
#: the report look self-contradictory when it is not.
_ANNOUNCED = re.compile(r"(\d+)\s+of\s+(\d+)\s+figures\s+rest on one text read")
_WITHHELD = re.compile(r"(\d+)\s+figures?\s+withheld")
_DETECTED = re.compile(r"(\d+)\s+figures?\s+were detected but omitted")


def check_figures(report: Report, ev: Evidence, blocks: list, errors: list[dict]) -> None:
    """The report may not advertise figures its visible sections do not contain.

    A real report stated "10 of 11 figures rest on one text read" in Source &
    Reliability and had no Numbers section at all: the count was taken over every
    `number` item in `evidence.json` while the section printed the items that
    survived the noise gate, which on that video was none of them. Both numbers
    were true and the sentence between them was false.

    Three things have to agree, and this is the check that makes them:

    1. the figures actually rendered, counted here from the blocks themselves
       rather than from the renderer's own count, so renderer drift is caught;
    2. the total the reliability note announces, which must be that number;
    3. the withheld count, which must be `detected − rendered` — and when a
       figure was detected and withheld, the note has to say so in those words.

    "Rendered" means *printed as a figure*: a block of kind `number` or
    `comparison`. A figure quoted inside a sentence is evidence the report uses,
    but it is not an entry in the section a count like "N of M figures" is
    describing, and counting it would let the note describe a list the reader
    never sees. The kinds excluded are the ones that state no evidence of their
    own — the header, the timeline, the resource list and the conflict notices.
    """
    by_id = {i.id: i for i in ev.items}
    rendered = {eid for _path, b in blocks
                if b.kind in ("number", "comparison")
                for eid in b.evidence_ids
                if eid in by_id and by_id[eid].type == "number"}
    detected = sum(1 for i in ev.items if i.type == "number")
    withheld = detected - len(rendered)

    rel = " ".join(b.text for path, b in blocks
                   if path.startswith("reliability") and b.kind != "conflict")
    announced = _ANNOUNCED.search(rel)
    stated = _WITHHELD.search(rel)
    detected_stated = _DETECTED.search(rel)
    declares = "figures_rendered" in (report.meta.get("report_figures") or {})

    if announced and int(announced.group(2)) != len(rendered):
        errors.append(_err(
            "figures-announced-not-rendered", "reliability",
            f"Source & Reliability reports {announced.group(2)} figures, but "
            f"{len(rendered)} are rendered in the report's visible sections"))
    if withheld and not stated and (announced or declares):
        errors.append(_err(
            "figures-withheld-unreported", "reliability",
            f"{withheld} figure(s) were detected and are not rendered, and the report does "
            f"not say they were omitted because they failed the report visibility threshold"))
    if stated and int(stated.group(1)) != withheld:
        errors.append(_err(
            "figure-count-mismatch", "reliability",
            f"Source & Reliability says {stated.group(1)} figure(s) were withheld, but "
            f"{detected} were detected and {len(rendered)} are rendered, so {withheld} were"))
    if detected_stated and int(detected_stated.group(1)) != detected:
        errors.append(_err(
            "figure-count-mismatch", "reliability",
            f"Source & Reliability says {detected_stated.group(1)} figure(s) were detected, "
            f"but the evidence holds {detected}"))
    declared = report.meta.get("report_figures") or {}
    for key, got in (("figures_rendered", len(rendered)),
                     ("figures_withheld", withheld),
                     ("figures_total", detected)):
        if key in declared and int(declared[key]) != got:
            errors.append(_err(
                "figure-count-mismatch", "meta",
                f"report.json declares {key}={declared[key]}, but the report "
                f"{'renders' if key == 'figures_rendered' else 'has'} {got}"))


def check_urls(ev: Evidence, md: str, errors: list[dict]) -> tuple[int, int]:
    allowed = evidence_urls(ev)
    seen: set[str] = set()
    for u in _URL.findall(md):
        n = _normalise_url(u.rstrip("/"))
        seen.add(n)
        if n not in allowed:
            errors.append(_err("url-unsupported", u,
                               "this URL does not appear anywhere in the evidence document"))
    return len(seen), sum(1 for u in seen if u in allowed)


def check_provenance(report: Report, ev: Evidence, blocks, errors: list[dict]) -> tuple[int, int]:
    """Every factual block must say where it came from, and the ids must resolve.

    Two failures are kept apart on purpose. A block naming an id that is not in
    the document is a broken pointer and is blocking. A block with a resolvable
    id but no rendered marker is a coverage gap — the reader cannot check it —
    and is blocking too, because an untraceable figure is the defect this whole
    project exists to prevent. Metadata-sourced facts are exempt: a title has no
    timestamp to point at, and inventing one would be worse.
    """
    by_id = {i.id: i for i in ev.items}
    total = with_marker = 0
    for path, b in blocks:
        if b.kind not in _FACTUAL_KINDS or not b.text:
            continue
        total += 1
        if b.evidence_ids:
            missing = [i for i in b.evidence_ids if i not in by_id]
            if missing:
                errors.append(_err("evidence-id-unresolved", path,
                                   f"block cites {', '.join(missing)}, which is not in the evidence"))
        elif b.kind != "timeline":
            errors.append(_err("claim-unsupported", path,
                               f"factual block with no evidence: {b.text[:80]!r}"))
            continue
        if _MARKER.search(b.refs or "") or _MARKER.search(b.text) or not b.evidence_ids:
            with_marker += 1
            continue
        anchored = any(
            p.source in ("transcript", "frame", "ocr", "vlm")
            for i in b.evidence_ids if i in by_id
            for p in (by_id[i].provenance or []))
        if not anchored:
            with_marker += 1  # metadata-only: a title or channel has no timestamp
        else:
            errors.append(_err("provenance-missing", path,
                               f"time-anchored block rendered without a source marker: {b.text[:80]!r}"))
    return total, with_marker


def check_claims(blocks, haystack_lower: set[str], errors: list[dict],
                 authored_by: str = "deterministic") -> int:
    """A factual block may only assert what the evidence document contains.

    Two surfaces. The first is the proper noun: a name that appears in no source
    is a fabrication, and the renderer has no free-text slot, so a real one means
    a template became editable somewhere. The second is the plain content word —
    "the kernel scheduler was rewritten to cut tail latency" carries no proper
    noun, no URL and no figure, and passes every other check in this file while
    asserting four things the evidence never said.

    Both are tested against the *whole* evidence document rather than the
    block's own items, for the same reason `check_numbers` and `check_urls` are:
    the question is "did the report say something no source in the document
    says?", and a stricter per-block rule would cry wolf on a sentence that
    draws two items together — which is what a narrative section is for.

    The content-word half is the deterministic renderer's rule and only the
    renderer's. Model-written prose is checked for exactly this by
    `synth_validate`, against a function-word list chosen for running prose
    rather than a fixed template vocabulary; running this half over it as well
    fires on ordinary connectives the model chose and the evidence never said,
    which is a false positive, not a grounding failure. The proper-noun half
    runs over both.
    """
    checked = 0
    content_words_applied = authored_by == "deterministic"
    for path, b in blocks:
        if not b.evidence_ids or b.kind in ("timeline", "link", "meta", "summary", "conflict"):
            # A conflict block is headed by fixed prose and named by its members;
            # its figures and URLs are still checked, over the whole document, in
            # `check_numbers` and `check_urls`.
            continue
        checked += 1
        text = _strip_markers(b.text)
        for sent in re.split(r"(?<=[.!?])\s+", text):
            # The proper-noun scan skips the sentence's first character, because
            # a capitalised word opening a sentence is not evidence of a name.
            # The content-word scan must not: it has no such exemption, and
            # truncating the first word there turns "The figures" into "igures".
            body = sent[1:] if sent and sent[0].isupper() else sent
            for m in _PROPER.finditer(body):
                word = m.group(1)
                if word in TEMPLATE_PROSE or word.lower() in _CONNECTIVES:
                    continue
                if word.lower() not in haystack_lower:
                    errors.append(_err("claim-unsupported", path,
                                       f"{word!r} is not present in the evidence this block cites"))
            if not content_words_applied:
                continue
            for word in _content_words(sent):
                if word not in haystack_lower:
                    errors.append(_err("claim-unsupported", path,
                                       f"{word!r} is not present in the evidence document"))
    return checked


def _content_words(text: str) -> set[str]:
    """The words a sentence *asserts*: not connectives, not the renderer's own
    prose, not the marker syntax. Lower-cased, because the evidence haystack is.
    """
    out = set()
    for w in _WORD.findall(text):
        low = w.lower()
        if low in _CONNECTIVES or low in TEMPLATE_WORDS or low in {p.lower() for p in TEMPLATE_PROSE}:
            continue
        out.add(low)
    return out


def check_hedges(by_id: dict, blocks, errors: list[dict]) -> int:
    """A hedged item must be stated with its hedge.

    "could cost $500" rendered as "costs $500" passes every other check in this
    file and is still false in the way that matters to a reader deciding
    whether to act on it. The synthesis gate has always enforced this for model
    prose; the deterministic gate is where a *hand-edited* `REPORT.md` would
    come through, and until now nothing here would notice.
    """
    from synth_validate import _has_hedge  # noqa: PLC0415 — no import cycle at module load

    checked = 0
    for path, b in blocks:
        if not b.evidence_ids or b.kind in ("timeline", "link", "meta", "summary"):
            continue
        hedged = [by_id[i] for i in b.evidence_ids
                  if i in by_id and (by_id[i].status == "hedged"
                                     or (by_id[i].qualifier or "").strip())]
        if not hedged:
            continue
        checked += 1
        if not _has_hedge(b.text):
            errors.append(_err("hedge-dropped", path,
                               "cites hedged evidence but states it without a hedge: "
                               + ", ".join(f"{i.id}={i.display or i.value!r}" for i in hedged[:4])))
    return checked


def check_duplicate_homes(report: Report, blocks, errors: list[dict]) -> int:
    """One section, once. One item, one narrative home.

    "Each evidence item has one home" is the reason the duplicate-8-gram metric
    reads low without being tuned, and it was asserted by a test rather than
    enforced by the gate. A section rendered twice — the failure mode when a
    renderer grows a second pass over the same material — is a fact the reader
    will believe twice, and nothing here used to see it.

    Two distinct shapes, because they are two distinct bugs: the *same section*
    emitted twice, and one item given two *different* narrative homes. Only
    prose sections count for the second. An item legitimately appears in the
    narrative and again in the numbers table, because a table is an index a
    reader scans, not a second telling of the same thing.
    """
    seen_keys: set[str] = set()
    for s in report.sections:
        if s.key in seen_keys:
            errors.append(_err("section-duplicated", s.key,
                               f"section {s.key!r} was rendered more than once"))
        seen_keys.add(s.key)

    prose = {"key_takeaways", "detailed_notes"}
    homes: dict[str, set] = {}
    for path, b in blocks:
        if b.kind in ("summary", "meta", "timeline"):
            continue
        section_key = path.split("/", 1)[0]
        if section_key not in prose:
            continue
        for i in b.evidence_ids:
            homes.setdefault(i, set()).add(section_key)
    duplicated = {i: sorted(v) for i, v in homes.items() if len(v) > 1}
    for item_id, where in sorted(duplicated.items()):
        errors.append(_err("item-duplicated", item_id,
                           f"rendered as narrative in more than one section: {', '.join(where)}"))
    return len(duplicated)


def check_visual(by_id: dict, blocks, errors: list[dict]) -> int:
    """A visual statement must rest on visual evidence, not on speech.

    The failure is the report narrating a chart it never saw: the diagram
    sentence is composed from the narration, and the reader has no way to tell it
    from one the model actually read off the pixels.
    """
    total = 0
    for path, b in blocks:
        if b.kind != "visual":
            continue
        total += 1
        ok = any(
            p.source in ("vlm", "frame", "ocr")
            for i in b.evidence_ids
            for p in (by_id[i].provenance or []) if i in by_id)
        if not ok:
            errors.append(_err("visual-unsupported", path,
                               f"a visual statement is not backed by any frame evidence: {b.text[:80]!r}"))
    return total


def check_conflicts(report: Report, ev: Evidence, blocks, md: str,
                    errors: list[dict], warnings: list[dict]) -> int:
    """Every recorded conflict must still be visible, and none may be resolved.

    The report is allowed to say a conflict exists, to quote both sides, and to
    decline. It is not allowed to drop one, and it is not allowed to leave only
    one side — a report that shows the disputed value without its counterpart has
    resolved the conflict by omission, which is the same failure as resolving it
    by judgement and much harder to notice.
    """
    rendered = {frozenset(b.evidence_ids) for _, b in blocks if b.kind == "conflict"}
    n = len(ev.conflicts)
    for c in ev.conflicts:
        if not any(frozenset(c.item_ids) <= s for s in rendered):
            errors.append(_err("conflict-dropped", c.kind,
                               f"an unresolved conflict was not carried into the report: {c.statement[:100]}"))
    if n and "not resolved" not in md and "unresolved" not in md.lower():
        warnings.append(_warn("conflict-language", "reliability",
                              "conflicts are present but the report does not say they are unresolved"))
    return n


def empty_headings(md: str) -> list[str]:
    """Headings with nothing under them.

    A heading with nothing under it is a claim that the video had something
    there; rendering one is how a report tells a reader to go looking for content
    that does not exist. Split out from the check so the markdown scan is one
    judgement with one name — the test that guards this asserts the same
    predicate the gate uses, rather than a re-implementation that can drift.
    """
    lines = md.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            continue
        level = len(line) - len(line.lstrip("#"))
        for nxt in lines[i + 1:]:
            if nxt.startswith("#"):
                nxt_level = len(nxt) - len(nxt.lstrip("#"))
                if nxt_level <= level:
                    # The section really did end here with nothing under it.
                    out.append(line.strip())
                break
            if nxt.strip():
                break
    return out


def check_empty_sections(report: Report, md: str, errors: list[dict]) -> int:
    empty = empty_headings(md)
    for line in empty:
        errors.append(_err("section-empty", line,
                           "heading has no content beneath it"))
    return len(empty)


def duplicate_ngrams(text: str, n: int = 8) -> tuple[int, int]:
    """Repeated 8-grams, and how many there were in total.

    A blunt measure, and deliberately so: it cannot tell a legitimate repeated
    phrase from a padded one, and it is not asked to. It is a tripwire on the
    failure mode this renderer was redesigned to remove — the same paragraph
    restated under three headings — and a metric that tries to be clever about
    meaning would be a metric that can be argued with.
    """
    words = [w.lower() for w in _WORD.findall(_strip_markers(text))]
    if len(words) < n:
        return 0, 0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(grams) - len(set(grams)), len(grams)


#: Above this share of repeated 8-grams the report is padded rather than
#: grounded. Below it, warned about; three times it, refused.
REPEAT_WARN = 0.06
REPEAT_BLOCK = 0.18

#: How far past its own ceiling a report may run before the budget gate fires.
#: A little over is a legitimate judgement call; a lot over means the report
#: stopped reading its evidence.
OVERRUN_BLOCK = 1.5


def validate_report(report: Report, ev: Evidence, markdown: str | None = None,
                    *, authored_by: str = "deterministic") -> ReportValidation:
    """Every check, every metric, one verdict. Blocking errors mean do not ship.

    `authored_by` names who wrote the prose, because one check is only sound for
    one of them: the content-word rule assumes a fixed template vocabulary, which
    is true of the renderer and false of a model. Everything else applies to both.
    """
    md = markdown if markdown is not None else render_markdown(report)
    errors: list[dict] = []
    warnings: list[dict] = []
    blocks = _all_blocks(report)
    hay = {w.lower() for w in _WORD.findall(evidence_text(ev))}
    by_id = {i.id: i for i in ev.items}

    ts_total, ts_ok = check_timestamps(report, ev, md, errors)
    num_total, num_ok = check_numbers(report, ev, md, errors)
    url_total, url_ok = check_urls(ev, md, errors)
    prov_total, prov_ok = check_provenance(report, ev, blocks, errors)
    claims_checked = check_claims(blocks, hay, errors, authored_by=authored_by)
    visual_total = check_visual(by_id, blocks, errors)
    conflicts = check_conflicts(report, ev, blocks, md, errors, warnings)
    hedges = check_hedges(by_id, blocks, errors)
    duplicated = check_duplicate_homes(report, blocks, errors)
    check_figures(report, ev, blocks, errors)
    check_empty_sections(report, md, errors)

    used = {i for _, b in blocks for i in b.evidence_ids if i in by_id}
    words = report.word_count()
    transcript_words = int((ev.transcript or {}).get("word_count") or 0)
    dups, grams = duplicate_ngrams(md)
    ratio = round(dups / grams, 4) if grams else 0.0
    if ratio > REPEAT_WARN:
        (errors if ratio > REPEAT_BLOCK else warnings).append(
            _err("repetition-excessive" if ratio > REPEAT_BLOCK else "repetition-high",
                 "report",
                 f"{dups} of {grams} 8-grams repeat ({ratio:.1%}); the same content is being "
                 f"rendered more than once"))

    budget = report.budget
    if budget is not None and words > budget.ceiling * OVERRUN_BLOCK:
        errors.append(_err("budget-exceeded", "report",
                           f"{words} words against a {budget.level} ceiling of {budget.ceiling}; "
                           f"the report is not sized from its evidence"))
    elif budget is not None and words > budget.ceiling:
        warnings.append(_warn("budget-overrun", "report",
                              f"{words} words against a {budget.level} ceiling of {budget.ceiling}"))

    metrics = {
        "report_word_count": words,
        "detail_level": budget.level if budget else None,
        "target_words": budget.words if budget else None,
        "ceiling_words": budget.ceiling if budget else None,
        "transcript_word_count": transcript_words,
        "compression_ratio": round(words / transcript_words, 4) if transcript_words else None,
        "evidence_items_total": len(ev.items),
        "evidence_items_used": len(used),
        "evidence_items_unused": len(ev.items) - len(used),
        "evidence_utilisation": round(len(used) / len(ev.items), 4) if ev.items else None,
        "provenance_coverage": round(prov_ok / prov_total, 4) if prov_total else 1.0,
        "provenance_markers": ts_total,
        "timestamps_resolved": ts_ok,
        "duplicate_8grams": dups,
        "ngrams": grams,
        "repetition_ratio": ratio,
        "unsupported_claims": len([e for e in errors if e["code"] == "claim-unsupported"]),
        "unsupported_numbers": len([e for e in errors if e["code"] == "number-unsupported"]),
        "unsupported_urls": len([e for e in errors if e["code"] == "url-unsupported"]),
        "unsupported_visual_statements": len([e for e in errors if e["code"] == "visual-unsupported"]),
        "fabricated_timestamps": len([e for e in errors if e["code"] == "timestamp-fabricated"]),
        "dropped_hedges": len([e for e in errors if e["code"] == "hedge-dropped"]),
        "hedged_blocks_checked": hedges,
        "items_with_two_homes": duplicated,
        "claim_blocks_checked": claims_checked,
        # Numeric *facts* only. The old `numbers_in_report` counted `[HH:MM]`
        # markers, so a report whose every "figure" was a citation scored 44/44
        # supported and measured nothing; timestamps are counted on their own line
        # below and the two are added up only as `numeric_tokens_total`.
        "numeric_facts_in_report": num_total,
        "numeric_facts_supported": num_ok,
        "timestamp_markers": ts_total,
        "numeric_tokens_total": num_total + ts_total,
        "urls_in_report": url_total,
        "urls_supported": url_ok,
        "visual_statements": visual_total,
        "section_count": sum(1 for s in report.sections),
        "blocks_rendered": len(blocks),
        "timeline_coverage": round(
            sum(1 for _, b in blocks if b.kind == "timeline")
            / max(1, len(ev.segments)), 4),
        "conflicts_in_evidence": conflicts,
        "mode": report.mode,
        "authored_by": authored_by,
    }
    return ReportValidation(metrics=metrics, blocking_errors=errors, warnings=warnings)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from synthesis import InvalidEvidence, load_valid_evidence

    ap = argparse.ArgumentParser(
        description="Validate REPORT.md against the evidence it was generated from.")
    ap.add_argument("artifact_dir")
    ap.add_argument("--detail", default="balanced", help="detail level the report was built at")
    args = ap.parse_args(argv)

    d = Path(args.artifact_dir)
    try:
        ev = load_valid_evidence(d / "evidence.json")
    except InvalidEvidence as exc:
        print(f"BLOCK  evidence is not valid, so the report cannot be checked: {exc}")
        return 1
    md_path = d / "REPORT.md"
    if not md_path.exists():
        print(f"FAIL  no REPORT.md in {d}")
        return 1
    from report import generate
    report = generate(ev, args.detail)
    res = validate_report(report, ev, md_path.read_text())
    print(json.dumps(res.metrics, indent=2))
    for w in res.warnings:
        print(f"  WARN  {w['code']}  {w['where']}: {w['message']}")
    for e in res.blocking_errors[:25]:
        print(f"  BLOCK {e['code']}  {e['where']}: {e['message']}")
    if len(res.blocking_errors) > 25:
        print(f"  … and {len(res.blocking_errors) - 25} more")
    print("VALID" if res.valid else f"INVALID ({len(res.blocking_errors)} blocking)")
    return 0 if res.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
