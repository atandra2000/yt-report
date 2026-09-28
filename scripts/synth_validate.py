"""What a synthesized sentence is allowed to say, checked against the evidence.

The validator in `report_validate.py` guards a *deterministic* renderer: it knows
every sentence was assembled from items, so it checks that the assembly did not
invent anything. A synthesized report breaks that assumption — the sentences are
new — so it needs checks that a renderer cannot need, and this is where they
live.

The difference is what counts as evidence. The deterministic renderer's sentence
came from an item, so "is it in the evidence" is nearly a formality. A
synthesized sentence is a *claim about* items, and every one of these checks
exists because a fluent, confident sentence can pass a naive check while being
false in the specific way that matters:

* **Unsupported content words.** Not "does the sentence mention something not in
  the evidence" — a connective is not a fabrication. Rather: the content words
  (the nouns and numbers that carry the meaning) must be findable in the cited
  items. "Qwen3-8B" cited to an item about Llama is a fabrication, and it is
  invisible to a reader because the sentence is otherwise well-formed.
* **Hedges.** An item marked `hedged`, or carrying a qualifier, must be cited by
  a sentence that hedges. "could cost $500" → "costs $500" passes every check in
  this repo and is still false. This is the single most valuable check here.
* **Conflicts.** Two levels, kept apart on purpose. A *block* may discuss one side
  of an unresolved conflict, but may not state a disputed item as settled. The
  *report* must then represent every conflict it references completely — both
  sides present somewhere. Requiring each paragraph to carry both sides instead
  is what made a correct report unrepresentable; see
  `check_conflict_completeness`.
* **Numbers.** Every figure must be a figure the cited items contain, with its
  unit. The deterministic renderer can only emit an item's own number, so this is
  a formality there and the real check here.
* **Visual statements.** A sentence about a chart must rest on a visual item.
  Derived from speech, it is a sentence about something nobody looked at.
* **Source kind.** A claim, an observation and an inference are different
  statements about different things, and `source_kind` is where the model says
  which it is making.

Every check reports a code and a location, and every one of them is *blocking*.
The design intent is that a draft which fails is not repaired into something
that passes — the writer gets one revision, and if that still fails, the
deterministic renderer runs instead. Weakening the standard to accommodate a
model is the failure this whole layer exists to prevent.
"""
from __future__ import annotations

import re
import unicodedata

from evidence_model import Evidence, Item

#: Function words, connectives and ordinary verbs. These are the words a
#: sentence is *made of* rather than the words it *asserts*, and requiring them
#: to appear in the evidence would make every well-formed sentence fail. The list
#: is closed and ordinary on purpose: adding a word here weakens the check, and
#: the temptation to add "significantly" or "substantially" is exactly the
#: failure — a hedge the model introduced and the check would wave through.
_FUNCTION_WORDS = frozenset("""
a an the this that these those and or but nor so yet for if then than as
is are was were be been being am do does did done have has had having
it its it's they them their there here what which who whom whose when where
why how not no nor none any some all both each every other another such
same own very too also just only more most less least much many few
can could may might must shall should will would about into over under
again further once because while during before after above below between
through against among within without across per via
""".split())

#: Hedge markers. A sentence citing a hedged item must contain one. Kept
#: explicit rather than inferred from item status alone, because "the video does
#: not specify a price" is a hedge that no single word marks.
_HEDGES = frozenset("""
may might could can would should possibly perhaps probably likely unlikely
about around approximately roughly nearly almost over under up to at most
at least as much as more than less than estimate estimated appear appears
seem seems suggest suggests potentially apparently roughly circa an estimated
""".split())

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_URL = re.compile(r"https?://[^\s<>\"')\]]+")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
#: A capitalised word, which is how a proper noun surfaces in running prose.
_PROPER = re.compile(r"\b([A-Z][A-Za-z0-9]{2,})\b")

#: Connective tissue rather than content. Reused from the report validator so
#: "is", "the" and "video" are treated the same way in both layers — one word
#: list for the whole repo is easier to keep honest than two that drift.
_CONNECTIVES = frozenset("""
the a an and or but if then than that this these those it its they them their there here
for from with without within into onto over under about after before between during while
when where which who whom whose what why how not no nor none both either each every some any
one two three four five six seven eight nine ten more most less least other another such
however therefore thus hence although though because since while as at by in of on to is are
was were be been being has have had do does did can could may might must shall should will
would source narration metadata quoted value figure compared named structure extracted
presenter author speaker creator video frame chart table diagram code terminal screen
coverage full degraded failed segment segments step steps item items video videos presenter
presenter he she they it this that the—the
""".split())

#: Words the report's own headings contribute. A capitalised word that is one of
#: these is the renderer's, not the model's. Kept explicit rather than derived
#: from the source: a set that regenerated itself would absorb every word the
#: renderer ever learns to write, which is the same as no check at all.
_TEMPLATE_PROSE = frozenset({
    "Unresolved", "Conflict", "Not", "Legible", "Visual", "Content", "Interpreted",
    "Model", "Derived", "Evidence", "Source", "Reliability", "Extracted", "Details",
    "Named", "Structure", "Coverage", "Full", "Degraded", "Failed", "Reported",
    "Reading", "Frame", "Narration", "Metadata", "Step", "Steps", "Timeline",
    "Executive", "Summary", "Key", "Takeaways", "Detailed", "Notes", "People",
    "Organizations", "Products", "Models", "Tools", "Numbers", "Pricing",
    "Benchmarks", "Comparisons", "Resources", "Links", "Important", "Quotes",
    "Figure", "Table", "Chart", "Diagram", "Code", "Terminal", "Screen",
})

#: Words that mark a sentence as describing something rather than asserting it.
#: Words that mark a sentence as *reporting* a limitation rather than asserting
#: past one. "Unresolved" belongs here for a specific reason: a conflict
#: disclosed neutrally is the correct rendering, and a check that only accepted
#: hedges would push the writer toward hedging language that the evidence does
#: not actually call for — overcorrecting a fidelity problem into a new one.
_UNCERTAINTY = frozenset("""
unclear illegible ambiguous unreadable uncertain uncertainty indeterminate unknown
unresolved disputed discrepancy disagrees conflicting contradicts mismatch
partially partially-visible truncated cropped blurry low-resolution
not fully cannot not entirely appears
""".split())


class Finding:
    """One problem, located. `code` is stable so a test can assert on it.

    `severity` is the field that decides whether a finding blocks. It is explicit
    rather than inferred from the code string, because "advisory" as a magic code
    value is the kind of thing that survives three refactors and then quietly
    stops meaning anything.
    """

    __slots__ = ("code", "where", "message", "unit", "severity")

    def __init__(self, code: str, where: str, message: str, unit: str = "",
                 severity: str = "blocking"):
        self.code = code
        self.where = where
        self.message = message
        self.unit = unit
        self.severity = severity

    @property
    def blocking(self) -> bool:
        return self.severity == "blocking"

    def to_dict(self) -> dict:
        return {"code": self.code, "where": self.where, "message": self.message,
                "unit": self.unit, "severity": self.severity}


class DraftValidation:
    def __init__(self):
        self.findings: list[Finding] = []
        self.checks_run: dict[str, int] = {}

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.blocking]

    @property
    def valid(self) -> bool:
        return not self.blocking

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def count(self, check: str) -> None:
        self.checks_run[check] = self.checks_run.get(check, 0) + 1

    def to_dict(self) -> dict:
        return {"valid": self.valid, "checks": self.checks_run,
                "blocking_errors": [f.to_dict() for f in self.blocking],
                "findings": [f.to_dict() for f in self.findings]}


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #

def _words(text: str) -> list[str]:
    """Lowercase word tokens, with accents folded so a comparison is not a
    spelling argument."""
    folded = unicodedata.normalize("NFKD", text.lower())
    return _WORD.findall(folded)


def _content_words(text: str) -> set[str]:
    """The words in a sentence that carry its meaning.

    Function words and short connectives are dropped because they belong to the
    sentence, not to the claim. What remains is what the sentence asserts, and
    that is what has to be findable in the evidence.
    """
    return {w for w in _words(text) if w not in _FUNCTION_WORDS and len(w) > 2}


def _has_hedge(text: str) -> bool:
    low = text.lower()
    return any(re.search(rf"\b{re.escape(h)}\b", low) for h in _HEDGES)


def _mentions_uncertainty(text: str) -> bool:
    low = text.lower()
    return any(re.search(rf"\b{re.escape(u)}\b", low) for u in _UNCERTAINTY)


def _normalise_number(n: str) -> str:
    return n.replace(",", "")


# --------------------------------------------------------------------------- #
# the checks
# --------------------------------------------------------------------------- #

def validate_draft_grounding(doc: dict, ev: Evidence) -> DraftValidation:
    """Every check, over every unit. Blocking throughout."""
    from synth_model import units, validate_draft

    res = DraftValidation()

    for problem in validate_draft(doc):
        res.add(Finding("synthesis-malformed", "draft", problem))

    by_id = {i.id: i for i in ev.items}
    conflict_sides = _conflict_sides(ev)
    # Acronyms the draft defines anywhere. Computed once, up front, because a
    # term introduced in the summary is available to every section after it.
    doc_acronyms = _defined_acronyms(" ".join(
        u.get("text") or "" for _, u in units(doc)))

    for where, unit in units(doc):
        text = unit.get("text") or ""
        ids = [i for i in (unit.get("evidence_ids") or []) if isinstance(i, str)]
        cited = [by_id[i] for i in ids if i in by_id]

        _check_ids_resolve(res, where, text, ids, by_id, cited)
        if not cited:
            # Without a resolvable citation there is nothing left to check the
            # text against, and every other check would be vacuous.
            continue
        _check_content_grounded(res, where, text, cited, doc_acronyms)
        _check_numbers(res, where, text, cited)
        _check_urls(res, where, text, cited)
        _check_hedges(res, where, text, cited)
        _check_conflicts(res, where, text, cited, conflict_sides)
        _check_visual(res, where, text, cited)
        _check_source_kind(res, where, unit, cited)

    # The report-level half, run once over the whole draft rather than per
    # paragraph. See `check_conflict_completeness` for why the two are separate
    # rules and why the per-paragraph one it replaces was wrong.
    for finding in check_conflict_completeness(doc, ev):
        res.count("conflicts-complete")
        res.add(finding)

    return res


def _check_ids_resolve(res, where, text, ids, by_id, cited) -> None:
    res.count("evidence-ids-resolve")
    unknown = [i for i in ids if i not in by_id]
    for i in unknown:
        res.add(Finding("evidence-id-unknown", where,
                        f"cites {i!r}, which is not an item in evidence.json", text))
    if ids and not cited:
        res.add(Finding("evidence-id-unresolvable", where,
                        "cites only ids that do not resolve; nothing supports this sentence",
                        text))


def _defined_acronyms(sentence: str) -> set[str]:
    """Acronyms this sentence introduces, as `Long Form Name (ABC)`.

    Only the explicit parenthetical form counts. A bare capitalised token is not
    treated as self-grounding, because "the GPU" written next to no expansion is
    exactly the kind of thing this check exists to catch — the point is to allow
    the writer to *define* a term, not to let capitalisation excuse anything.
    """
    out: set[str] = set()
    for m in re.finditer(r"\(([A-Z][A-Za-z0-9&/.-]{1,9})\)", sentence):
        out.add(m.group(1))
    return out


def _check_content_grounded(res, where, text, cited, doc_acronyms=frozenset()) -> None:
    """Proper nouns and technical identifiers must come from the cited evidence.

    Deliberately narrow, and the narrowness is the design. A synthesized
    sentence is *new prose*, so it is entitled to ordinary vocabulary — the
    model has to be allowed to write "sits", "covers" and "comparison" or it
    cannot write prose at all. Demanding every content word appear in the
    evidence would reject every well-formed sentence and leave the check with
    nothing but false positives.

    What is not allowed is inventing a *thing*: a name, a product, a model, a
    standard. So the surface checked is capitalised words that are not
    sentence-initial, plus identifier-shaped tokens. That is exactly the
    "Qwen3-8B attached to an item about Llama" fabrication, and it is invisible
    to a reader because the sentence is otherwise well-formed.

    It cannot catch a wrong *relationship* between two correctly-grounded words
    — no mechanical check can — which is why the hedge, conflict and source-kind
    checks below carry as much of the weight as this one.
    """
    res.count("entities-grounded")
    haystack = " ".join(_item_ground_text(i) for i in cited)
    hay_lower = haystack.lower()
    for sent in re.split(r"(?<=[.!?])\s+", text):
        body = sent[1:] if sent and sent[0].isupper() else sent
        # An acronym the document has defined is grounded by that definition.
        # "Retrieval Augmented Generation (RAG)" is fine in the sentence that
        # introduces it, and equally fine in the sentence after — a reader does
        # not re-read a term every paragraph. Found by running a real model: the
        # first draft defined RAG correctly and was then rejected in a later
        # section for using the acronym two paragraphs after defining it.
        defined = _defined_acronyms(sent) | set(doc_acronyms)
        for m in _PROPER.finditer(body):
            word = m.group(1)
            if word in _TEMPLATE_PROSE or word.lower() in _CONNECTIVES:
                continue
            if word in defined:
                continue
            if word.lower() not in hay_lower:
                res.add(Finding("claim-unsupported", where,
                                f"{word!r} is not present in the evidence this "
                                f"sentence cites", text))
                break


def _check_numbers(res, where, text, cited) -> None:
    """Every figure in the sentence must be a figure the cited items contain."""
    res.count("numbers-grounded")
    allowed = set()
    for item in cited:
        for n in _NUMBER.findall(_item_ground_text(item)):
            allowed.add(_normalise_number(n))
    for n in _NUMBER.findall(text):
        if _normalise_number(n) not in allowed:
            res.add(Finding("number-unsupported", where,
                            f"figure {n!r} does not appear in the cited evidence", text))


def _check_urls(res, where, text, cited) -> None:
    res.count("urls-grounded")
    allowed = set()
    for item in cited:
        allowed |= set(_URL.findall(_item_ground_text(item)))
    for u in _URL.findall(text):
        if u not in allowed:
            res.add(Finding("url-unsupported", where,
                            f"URL {u!r} does not appear in the cited evidence", text))


def _check_hedges(res, where, text, cited) -> None:
    """A hedged item must be cited by a hedged sentence.

    "could cost $500" rendered as "costs $500" is a fabrication that passes every
    other check in this repo: the number is real, the item is real, the citation
    is real. Only the epistemic status was lost, and that is the part the source
    actually hedged on.
    """
    res.count("hedges-preserved")
    hedged = [i for i in cited if i.status == "hedged" or (i.qualifier or "").strip()]
    if hedged and not _has_hedge(text):
        res.add(Finding("hedge-dropped", where,
                        "cites hedged evidence but states it without a hedge: "
                        + ", ".join(f"{i.id}={i.display or i.value!r}" for i in hedged[:4]),
                        text))
    # The inverse is also a defect: a confidently-worded sentence resting only
    # on an `unreadable` or `conflicting` item is asserting past its source.
    weak = [i for i in cited if i.status in ("unreadable", "conflicting")]
    if weak and not _has_hedge(text) and not _mentions_uncertainty(text):
        res.add(Finding("uncertainty-dropped", where,
                        "cites evidence recorded as "
                        + "/".join(sorted({i.status for i in weak}))
                        + " but states it flatly", text))


def _check_conflicts(res, where, text, cited, conflict_sides) -> None:
    """Retained as the *block-level* half of conflict validation.

    A block may discuss one side of an unresolved conflict; that is allowed, and
    `check_conflict_completeness` is what decides whether the report as a whole
    still discloses the disagreement. What is not allowed here is a block that
    states a disputed item *flatly* — the one-sided block must still be marked as
    disputed rather than asserted, which is the same requirement the hedge check
    places on a `hedged` item. Without this the completeness check would accept a
    report whose only mention of the dispute is a confident sentence.
    """
    res.count("conflicts-marked")
    cited_ids = {i.id for i in cited}
    disputed = {i.id for i in cited
                if any(i.id in sides for sides in conflict_sides)}
    if disputed and not (_has_hedge(text) or _mentions_uncertainty(text)):
        res.add(Finding("conflict-side-asserted", where,
                        "states a disputed item without marking it as disputed ("
                        + ", ".join(sorted(disputed)) + "); the conflict itself must "
                        "be disclosed somewhere in the report", text))


def check_conflict_completeness(doc: dict, ev: Evidence) -> list[Finding]:
    """The report-level invariant: every unresolved conflict is completely
    represented before the report is written.

    The unit of judgment is the whole draft, not one paragraph. The rule this
    replaces was *"every paragraph mentioning a member must cite every member"*,
    and that rule is strictly wrong: it made a correct report unrepresentable. A
    report that explains one side in Detailed Notes and discloses the full
    disagreement in Source & Reliability is the report we want, and the old rule
    refused it — which is how a video whose sources disagree could never produce a
    model-written report at all, and the pipeline fell back to the deterministic
    renderer on precisely the videos that most needed a writer.

    Two things are being kept apart deliberately, because conflating them is what
    produced the defect:

    * **Block-level grounding** — a paragraph still cannot cite evidence it does
      not have, and cannot state a disputed item as settled
      (`_check_conflicts`, above). Unchanged, and not weakened.
    * **Report-level completeness** — a conflict the draft *references at all*
      must have every side it names present somewhere in the draft. This function.

    Three failure modes, and they are different failures:

    * the draft cites one side and never the other, anywhere — the disagreement
      is resolved by omission, and nothing in the report says so;
    * the draft cites a conflict member but neither side is ever fully disclosed
      — same failure, arriving by a different route;
    * a conflict names an item that is not in the evidence — the completeness
      check would pass vacuously on a set it can never satisfy, so it is refused
      here rather than left to look like a clean report.

    A conflict the draft never mentions is not a failure. The report is not
    obliged to discuss every dispute in the video, and the deterministic conflict
    section is what guarantees the ones that matter are disclosed anyway.
    """
    from synth_model import units

    cited: set[str] = {i for _, u in units(doc)
                       for i in (u.get("evidence_ids") or []) if isinstance(i, str)}
    known = {i.id for i in ev.items}
    findings: list[Finding] = []
    for c in ev.conflicts:
        if c.resolution == "resolved":
            continue
        sides = set(c.item_ids)
        dangling = sorted(sides - known)
        if dangling:
            findings.append(Finding(
                "conflict-dangling", f"conflicts[{c.cid}]",
                f"conflict {c.cid!r} names evidence that does not exist ("
                + ", ".join(dangling) + "); it can never be fully disclosed"))
            continue
        overlap = cited & sides
        if not overlap:
            # The report says nothing about this disagreement. Its own choice —
            # the deterministic conflict section discloses it regardless.
            continue
        missing = sorted(sides - cited)
        if missing:
            findings.append(Finding(
                "conflict-incomplete", f"conflicts[{c.cid}]",
                f"the report cites {', '.join(sorted(overlap))} from conflict "
                f"{c.cid!r} but never {', '.join(missing)}; the disagreement is "
                f"resolved by omission"))
    return findings


def _check_visual(res, where, text, cited) -> None:
    """A sentence about something seen must rest on something seen.

    Derived from speech instead, it is a claim about a frame nobody opened, and
    the reader has no way to tell it from one a model actually read.
    """
    res.count("visual-grounded")
    if not _describes_visual(text):
        return
    if not any(p.source in ("vlm", "frame", "ocr") for i in cited for p in i.provenance):
        res.add(Finding("visual-unsupported", where,
                        "describes something visual but no cited item comes from a frame, "
                        "an OCR read or a model", text))
    # A visual claim resting only on a low-confidence OCR read is a guess. It is
    # allowed only if the sentence says so.
    only_weak_ocr = cited and all(
        p.source in ("ocr", "frame") and (p.confidence or 0) < 0.7
        for i in cited for p in i.provenance)
    if only_weak_ocr and not _mentions_uncertainty(text):
        res.add(Finding("visual-uncertain", where,
                        "rests only on a low-confidence visual read but states no "
                        "uncertainty", text))


#: Words that name a *visual artifact*, not words that merely co-occur with one.
#: The first draft of this list included "window", "line" and "bar", and the
#: result was a false positive on "context window" — a technical term with
#: nothing to do with a screen. A cue list has to be made of nouns a reader would
#: agree are pictures; a common noun that merely happens to be used in UI
#: vocabulary will fire on ordinary technical prose, and a check that cries wolf
#: is worse than none because it teaches the writer to ignore it.
_VISUAL_CUES = ("chart", "graph", "plot", "table", "diagram", "slide", "screenshot",
                "terminal", "spreadsheet", "dashboard", "editor", "axis", "legend",
                "code listing", "code sample")


def _describes_visual(text: str) -> bool:
    low = text.lower()
    return any(re.search(rf"\b{re.escape(c)}\b", low) for c in _VISUAL_CUES)


def _check_source_kind(res, where, unit, cited) -> None:
    """`source_kind` must match what the cited evidence actually is.

    A sentence over only `vlm` provenance that calls itself a `claim` is
    mislabelled in the direction that matters — it invites a reader to treat a
    model's reading of pixels as something the speaker said.
    """
    res.count("source-kind")
    kind = unit.get("source_kind")
    if kind is None:
        return
    sources = {p.source for i in cited for p in i.provenance}
    if kind == "observation" and not (sources & {"vlm", "frame", "ocr"}):
        res.add(Finding("source-kind-mismatch", where,
                        "labelled 'observation' but every cited item is from speech", unit_text(unit)))
    if kind == "claim" and sources and not (sources & {"transcript", "metadata"}):
        res.add(Finding("source-kind-mismatch", where,
                        "labelled 'claim' but no cited item is a spoken or metadata statement",
                        unit_text(unit)))


def unit_text(unit: dict) -> str:
    return unit.get("text") or ""


def _item_ground_text(item: Item) -> str:
    """Every string an item carries. The haystack a sentence may draw from."""
    parts = [item.value or "", item.display or "", item.subject or "",
             item.qualifier or "", item.verbatim or "", item.notes or ""]
    if item.detail:
        parts += [str(v) for v in item.detail.values() if not isinstance(v, (list, dict))]
        for v in item.detail.values():
            if isinstance(v, list):
                parts += [str(x) for x in v if not isinstance(x, (list, dict))]
            elif isinstance(v, dict):
                parts += [str(x) for x in v.values() if not isinstance(x, (list, dict))]
    for p in item.provenance:
        if p.ocr_text:
            parts.append(p.ocr_text)
    return " ".join(parts)


def _conflict_sides(ev: Evidence) -> list[set[str]]:
    """Each unresolved conflict as the set of item ids that disagree.

    `resolved` conflicts are excluded on purpose: once a document records a
    conflict as settled, requiring both sides in every sentence would be a
    constraint on a question the evidence has already answered.
    """
    return [set(c.item_ids) for c in ev.conflicts if c.resolution != "resolved"]


def check_omissions(doc: dict, planned_ids) -> list[Finding]:
    """Evidence the plan said mattered, and the draft never cited.

    Advisory, not blocking. A summary that omits a low-priority item is the
    system working as intended — the whole point of a plan is that not everything
    gets said — so this reports rather than refuses. It exists so that "the
    report is thin" has a measurable cause rather than being a feeling.
    """
    from synth_model import units

    cited = {i for _, u in units(doc)
             for i in (u.get("evidence_ids") or []) if isinstance(i, str)}
    return [Finding("omission", "draft",
                    f"planned evidence {eid} was never cited", severity="advisory")
            for eid in sorted(set(planned_ids) - cited)]
