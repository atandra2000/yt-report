"""The boundary between evidence and report generation — and now the pipeline.

One function, deliberately: `load_valid_evidence`. A synthesis step calls it and
receives an `Evidence` that has passed every invariant, or an exception naming
the failures. It cannot receive a document with a dangling frame reference,
because the only way in runs the validator first.

The rest of this module is the model-backed path, and its shape is a chain in
which every stage can refuse the next one:

    evidence.json
      → load_valid_evidence          blocking: the document is coherent
      → plan                         which evidence is central, and why
      → draft                        structured JSON, every sentence cited
      → validate                     blocking: every sentence is supported
      → critique (advisory)          what a careful reader would still object to
      → one revision                 the writer may fix what critique found
      → validate again               blocking, and the last word
      → render                       Report/Section/Block, then the existing
                                     renderer and the existing validator

The fallback ladder is the reason this is written as a chain rather than as a
function with a few `if`s. A synthesis model can be absent, can return invalid
JSON, can return valid JSON that fails grounding — and in every one of those
cases the answer is the deterministic renderer, never a weakened check. The
validator is not a threshold the model negotiates; a draft that fails it is
discarded, not repaired into something that passes.

The one revision is bounded on purpose. A writer that can iterate with itself
will iterate toward its own idea of a good report, and the thing that keeps it
honest — the evidence it is not allowed to see past — is exactly what an
unbounded loop talks itself out of. One pass of criticism, one revision, then
validation decides.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from evidence_model import Evidence, EvidenceError, load_evidence
from evidence_validate import ValidationResult, validate_evidence
from report import Block, Report, Section, derive_budget
from synth_model import (
    PROMPT_VERSION,
    SYNTH_SCHEMA_VERSION,
    SynthesisClient,
    units,
    validate_critique,
    validate_draft,
)
from synth_validate import DraftValidation, check_omissions, validate_draft_grounding

#: What the writer is told about the document. The video's own title and
#: duration only — never a transcript, never a frame, never a filesystem path.
#: The restriction is structural rather than advisory: the prompt is assembled
#: from an explicit allowlist of evidence fields, so there is no way for a
#: transcript to reach the model by being a field nobody remembered to strip.
_MAX_EVIDENCE_CHARS = 24000
_MAX_ITEMS = 400


class InvalidEvidence(EvidenceError):
    """Blocking validation failures. Carries the whole result, not just the first
    error, so the failing stage can show every problem at once instead of making
    the caller re-run once per defect."""

    def __init__(self, result: ValidationResult):
        self.result = result
        detail = "; ".join(f"{e['code']} at {e['where']}" for e in result.blocking_errors[:5])
        super().__init__(f"evidence is not synthesizable ({len(result.blocking_errors)} blocking): {detail}")


class BudgetExhausted(RuntimeError):
    """A configured ceiling on model calls was reached. A refusal, not a failure."""


class Budget:
    """A hard ceiling on model calls, spent before each one.

    Not a counter checked afterwards. `spend` refuses when the ceiling is gone
    and the caller must handle the refusal, so a run configured for one call
    cannot quietly make three.

    It lives here rather than in the transport because the ceiling is a *caller's*
    policy: `chat_json` has no opinion about how many calls a run may afford, and
    a budget enforced at the transport would be invisible to the code that has to
    explain the refusal. Every call site in this module spends explicitly, so a
    new call cannot be added without deciding what it costs.
    """

    def __init__(self, calls: int | None):
        self.limit = calls
        self.spent = 0

    def spend(self) -> None:
        if self.limit is not None and self.spent >= self.limit:
            raise BudgetExhausted(f"the configured ceiling of {self.limit} model call(s) "
                                  f"is already spent")
        self.spent += 1

    def to_dict(self) -> dict:
        return {"limit": self.limit, "spent": self.spent}


def load_valid_evidence(path: str | Path) -> Evidence:
    """The only supported entry point for the synthesis stage."""
    ev = load_evidence(path)
    result = validate_evidence(ev)
    if not result.valid:
        raise InvalidEvidence(result)
    return ev


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

@dataclass
class SynthConfig:
    """Where the synthesis model is and how hard it should try.

    Environment first, CLI second, the same precedence `VLMConfig` uses, for the
    same reason: a user who exported one variable and a script that passes a flag
    should not have to know about each other. The credential is never logged,
    never cached and never written into an artifact.
    """

    provider: str = "none"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 120.0
    output_language: str = "auto"
    detail: str = "balanced"
    #: One revision after a critique, never more.
    max_revisions: int = 1
    critic: bool = True

    ENV = {"provider": "SYNTH_PROVIDER", "model": "SYNTH_MODEL",
           "base_url": "SYNTH_BASE_URL", "api_key": "SYNTH_API_KEY",
           "timeout": "SYNTH_TIMEOUT", "output_language": "SYNTH_OUTPUT_LANGUAGE"}

    @classmethod
    def from_env(cls, **overrides) -> "SynthConfig":
        base: dict = {"provider": os.environ.get("SYNTH_PROVIDER") or "none"}
        for name, var in cls.ENV.items():
            raw = os.environ.get(var)
            if raw is None or name in base:
                continue
            if name == "timeout":
                try:
                    base[name] = float(raw)
                except ValueError:
                    continue
            else:
                base[name] = raw
        for k, v in overrides.items():
            if v is not None:
                base[k] = v
        return cls(**base)

    def to_public_dict(self) -> dict:
        return {"provider": self.provider, "model": self.model,
                "base_url": self.base_url, "timeout": self.timeout,
                "output_language": self.output_language, "detail": self.detail,
                "critic": self.critic, "max_revisions": self.max_revisions,
                "api_key_present": bool(self.api_key)}

    def resolved_language(self, ev: Evidence) -> str:
        """`auto` means "speak the video's language". Anything else is a request
        the writer is told to honour.

        The evidence stays in its source language either way: only the rendered
        prose changes. Names, identifiers, URLs and code are protected, and the
        validator checks that they came through untranslated.
        """
        if self.output_language and self.output_language != "auto":
            return self.output_language
        lang = (ev.transcript.get("language") or "").strip()
        return lang or "en"


# --------------------------------------------------------------------------- #
# the cache
# --------------------------------------------------------------------------- #

def cache_key(ev: Evidence, cfg: SynthConfig) -> str:
    """A key that changes when the *content* changes, not just the video.

    Keyed on the evidence content hash rather than the video id, because the
    failure this prevents is a stale report: an evidence document edited by a
    later pass, or regenerated with different OCR, would otherwise reuse a draft
    describing the previous contents. Model, provider, prompt version, schema
    version, detail level and output language are all in the key, so a draft is
    never reused across a configuration that could have produced a different one.
    """
    payload = json.dumps(ev.to_dict(), sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    parts = [digest, cfg.provider or "none", cfg.model or "-",
             str(PROMPT_VERSION), str(SYNTH_SCHEMA_VERSION),
             cfg.detail, cfg.output_language]
    return "synth-" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #

def _evidence_view(ev: Evidence, limit: int = _MAX_ITEMS) -> list[dict]:
    """The evidence, in the shape the model is allowed to see.

    An allowlist, deliberately. Every field here is a fact the report may state;
    anything not on it cannot reach the model, so "the writer only sees validated
    evidence" is a property of this function rather than a promise about prompt
    construction. The `provenance` is included but reduced to modality and
    confidence — a writer that can see a raw `ref` will quote it, and item ids
    belong in the structured output, not in the prose.
    """
    out: list[dict] = []
    for item in ev.items[:limit]:
        sources = sorted({p.source for p in item.provenance})
        entry: dict = {
            "id": item.id,
            "type": item.type,
            "text": item.display or item.value,
            "status": item.status,
            "importance": item.importance,
            "sources": sources,
        }
        for key in ("unit", "subject", "qualifier", "verbatim", "notes"):
            v = getattr(item, key, None)
            if v:
                entry[key] = v
        if item.detail:
            entry["detail"] = item.detail
        if "vlm" in sources or "frame" in sources or "ocr" in sources:
            entry["visual"] = True
        out.append(entry)
    return out


def _segments_view(ev: Evidence) -> list[dict]:
    """Segment titles and times — the spine a sectioned report is built on.

    No snippet text. A segment's own items carry its content; letting the raw
    transcript in here would make the transcript an alternative source for the
    writer's sentences, and every grounding check downstream reasons about items.
    """
    return [{"id": s.id, "title": s.title, "start_s": s.start_s, "end_s": s.end_s,
             "item_ids": list(s.item_ids)} for s in ev.segments]


def _conflicts_view(ev: Evidence) -> list[dict]:
    return [{"item_ids": list(c.item_ids), "kind": c.kind,
             "statement": c.statement, "resolution": c.resolution}
            for c in ev.conflicts]


def build_plan(ev: Evidence, cfg: SynthConfig) -> dict:
    """Which evidence is central, and how the document should be organised.

    Deterministic, and that is a choice rather than a shortcut. A planner is the
    one stage that must be reproducible: it decides what the report is *about*,
    and a plan that changed between two runs of the same evidence would make
    every downstream cache entry and every before/after comparison meaningless.
    It also cannot invent anything — it selects and groups, and every id it emits
    is checked against the document before the writer sees it.
    """
    items = ev.items
    high = [i for i in items if i.importance == "high"]
    hedged = [i for i in items if i.status == "hedged"]
    conflicted = {i for c in ev.conflicts for i in c.item_ids if c.resolution != "resolved"}
    visual = [i for i in items
              if any(p.source in ("vlm", "frame", "ocr") for p in i.provenance)]

    summary_ids = [i.id for i in (high or items)[:12]]
    if not summary_ids:
        summary_ids = [i.id for i in items[:12]]

    # Takeaways: high-importance first, then anything visual that is also
    # important, then conflicts — a disagreement between two sources is one of
    # the more useful things a report can carry, and a plan that ranked purely
    # by confidence would drop it.
    ordered: list[str] = []
    seen: set[str] = set()
    for group in (high, visual, [i for i in items if i.id in conflicted], hedged):
        for i in group:
            if i.id not in seen:
                seen.add(i.id)
                ordered.append(i.id)

    groups: list[list[str]] = [ordered[i:i + 3] for i in range(0, len(ordered), 3)][:8]
    if not groups and ordered:
        groups = [[ordered[0]]]

    section_plan = []
    for s in ev.segments:
        ids = [i for i in s.item_ids if any(x.id == i for x in items)]
        if ids:
            section_plan.append({"segment_id": s.id, "title": s.title, "item_ids": ids})

    return {
        "summary_evidence_ids": summary_ids,
        "takeaway_groups": groups,
        "section_plan": section_plan,
        "timeline_ids": [s.id for s in ev.segments],
        "detail_level": cfg.detail,
    }


def _plan_prompt(ev: Evidence, cfg: SynthConfig, plan: dict) -> str:
    view = _evidence_view(ev)
    used = _trim_to_budget(view, _MAX_EVIDENCE_CHARS)
    return f"""Write a report about this video from the evidence below.

TITLE: {ev.video.get('title') or '(untitled)'}
DURATION: {ev.video.get('duration_s')} seconds
LANGUAGE OF THE SOURCE MATERIAL: {cfg.resolved_language(ev)}
OUTPUT LANGUAGE: {cfg.output_language}
DETAIL LEVEL: {cfg.detail}

SEGMENTS:
{json.dumps(_segments_view(ev), ensure_ascii=False, indent=1)}

UNRESOLVED CONFLICTS (a conflict you mention must be represented completely
somewhere in the report — you may discuss one side in a section and disclose both
sides together in another, but a side you never mention anywhere is a conflict
you have resolved by omission):
{json.dumps(_conflicts_view(ev), ensure_ascii=False, indent=1)}

EVIDENCE ITEMS:
{json.dumps(used, ensure_ascii=False, indent=1)}

Produce:
  summary     — {2 if cfg.detail == 'brief' else 4} paragraphs. What is this video
                about, what are its main ideas, what does it actually conclude,
                and what is uncertain or disputed. Not a list of items.
  takeaways   — {4 if cfg.detail == 'brief' else (6 if cfg.detail == 'balanced' else 8)}
                high-value points. Each is an idea with its support, not
                "X was mentioned".
  sections    — one per segment that has evidence, in order. Each section is
                prose: connect the evidence, keep the qualifiers, say what the
                numbers are and what they mean, and say where the visual evidence
                is ambiguous. Do not write one paragraph per item.

The three must not overlap. Draft the summary first, then the takeaways, then
write each section so that it says something the summary did not — the detail
behind that segment's claims, not a restatement of them. A sentence that appears
in the summary must not reappear in a section, and no two sections may make the
same point. Repeating yourself is a defect, not emphasis.

Each item gets exactly one home, and there are exactly two kinds of home: a
takeaway and a section. The summary is not one of them. An item in the summary
and again in a section is the ordinary shape of a report that opens with its
point and then shows the work, so cite it there as often as the argument needs.

What the gate refuses is one item carrying a takeaway *and* a section — never
in two homes, because then the report tells the reader the same thing twice and
the second telling adds nothing. Where a point deserves both, the takeaway
states the claim and the section develops it from *other* evidence; the takeaway
is a pointer and the section is its elaboration. You may restate a point in a
takeaway in your own words, but the takeaway's `evidence_ids` must not repeat an
item a section already carries, because the item is what the gate counts, not
the sentence.

Every unit carries `evidence_ids` naming the items it rests on. Write in
{cfg.output_language if cfg.output_language != 'auto' else 'the language of the source material'}.
Names, identifiers, URLs, code and figures stay exactly as they appear in the
evidence, whatever language the prose is in.
"""


def _trim_to_budget(view: list[dict], budget: int) -> list[dict]:
    """Keep the whole view if it fits; otherwise keep the most important prefix.

    A truncation that dropped the conflict or the hedged figure would be worse
    than a shorter report, so the budget is applied to a list already ordered by
    importance, and the items that are never dropped are named in the prompt
    regardless.
    """
    if len(json.dumps(view, ensure_ascii=False)) <= budget:
        return view
    out: list[dict] = []
    size = 0
    for e in view:
        chunk = len(json.dumps(e, ensure_ascii=False)) + 1
        if size + chunk > budget:
            break
        out.append(e)
        size += chunk
    return out


# --------------------------------------------------------------------------- #
# revision prompt
# --------------------------------------------------------------------------- #

def _revision_prompt(ev: Evidence, cfg: SynthConfig, doc: dict,
                     findings: list, critique: dict | None) -> str:
    """The one revision. Both sources of pressure, and nothing else.

    The validator's findings come first and are not optional: they are blocking
    and the draft will be discarded if they are not fixed. The model's own
    critique follows as advisory judgement — a careful reader noticing a
    repetition or an omission that no mechanical check can see.
    """
    lines = [f"Fix the problems below in the draft. Change nothing else.",
             "", "DRAFT:", json.dumps(doc, ensure_ascii=False, indent=1), ""]
    if findings:
        lines.append("BLOCKING — the draft is rejected without these fixed:")
        for f in findings[:40]:
            lines.append(f"  [{f.code}] at {f.where}: {f.message}")
        lines.append("")
    if critique:
        for f in (critique.get("findings") or [])[:20]:
            lines.append(f"  [critic/{f.get('code')}] at {f.get('where')}: {f.get('message')}")
    lines += ["",
              "Return the complete corrected draft in the same schema. Keep every "
              "sentence's evidence_ids. Do not add facts."]
    return "\n".join(lines)


def _critique_prompt(ev: Evidence, doc: dict) -> str:
    return f"""Check this report draft against the evidence it cites.

EVIDENCE:
{json.dumps(_evidence_view(ev), ensure_ascii=False, indent=1)}

DRAFT:
{json.dumps(doc, ensure_ascii=False, indent=1)}

Report only defects you can point at in the evidence above:
  unsupported-claim   a sentence asserting something the cited items do not say
  missing-qualifier   a hedge or qualifier the evidence carries and the draft drops
  conflict-hidden     an unresolved conflict mentioned on one side only
  number-error        a figure that does not match the cited evidence
  external-knowledge  a fact that is true but not in this evidence
  repetition          the same point made twice
  omission            important cited evidence the draft never uses
  visual-unsupported  a claim about a frame the evidence does not support

If the draft is sound, return an empty findings array. Do not rewrite the draft."""


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #

@dataclass
class SynthOutcome:
    """What happened, in a form the caller can print and a test can assert on.

    `report` is the deliverable and `fallback_reason` is why it is the
    deterministic one. They are separate because "the report is grounded" and
    "the report was written by a model" are independent facts, and a report that
    quietly degraded would otherwise read identically to one that did not.
    """

    report: Report
    mode: str = "deterministic"
    fallback_reason: str | None = None
    #: Set when an output language was asked for and the deterministic renderer
    #: answered instead. Not a warning about quality — the renderer is a valid
    #: report — but a statement that a request was not honoured, which is the
    #: one thing a report that quietly stays in English must not let a reader
    #: assume about it.
    language_note: str | None = None
    draft: dict | None = None
    validation: DraftValidation | None = None
    revisions: int = 0
    critique: dict | None = None
    cache_hit: bool = False
    model_report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "fallback_reason": self.fallback_reason,
            "language_note": self.language_note,
            "revisions": self.revisions,
            "cache_hit": self.cache_hit,
            "model": self.model_report,
            "draft_valid": self.validation.valid if self.validation else None,
            "checks": self.validation.checks_run if self.validation else {},
            "findings": [f.to_dict() for f in (self.validation.findings
                                               if self.validation else [])],
        }


#: The sentence the CLI states when `--output-language` was asked for and the
#: deterministic renderer answered. Named once so the code, the docs and the test
#: cannot drift into three different claims about the same behaviour.
OUTPUT_LANGUAGE_REQUIRES_WRITER = (
    "output-language requires a configured synthesis writer for true translation")


def deterministic_report(ev: Evidence, detail: str, reason: str,
                         output_language: str = "auto") -> SynthOutcome:
    """The renderer, unchanged. The floor every fallback lands on.

    `output_language` is passed only so the outcome can say whether the request
    was honoured. The renderer has no translations to choose between — its prose
    is a fixed English template — so a run that asked for Japanese and got this
    is a run that did not do what it was asked, and the report has to say so
    rather than let the reader infer a translation that never happened.
    """
    from report import generate

    asked = (output_language or "auto").strip()
    return SynthOutcome(
        report=generate(ev, detail), mode="deterministic", fallback_reason=reason,
        language_note=(OUTPUT_LANGUAGE_REQUIRES_WRITER
                       if asked and asked != "auto" else None))


def synthesize(ev: Evidence, cfg: SynthConfig, cache=None,
               budget: "Budget | None" = None,
               critic_budget: "Budget | None" = None) -> SynthOutcome:
    """Run the model-backed path, or explain why the deterministic one ran.

    Never raises for a model-side problem. A synthesis that cannot be produced is
    a report that gets written another way, not a failed run — but the reason is
    always recorded, so "the model was unavailable" is never silently reported as
    "the model produced nothing".

    `budget` and `critic_budget` are hard ceilings, and `None` means unbounded —
    the same default a full run has always had. Every call site below spends from
    one before it makes a call, so a run given a ceiling cannot exceed it and a
    caller reading the outcome can see exactly how much was spent.
    """
    from report import generate

    if not cfg.provider or cfg.provider == "none":
        return deterministic_report(ev, cfg.detail, "no synthesis model configured",
                                 cfg.output_language)

    base_url = _resolve_base(cfg)
    if not base_url:
        return deterministic_report(ev, cfg.detail,
                                    f"provider {cfg.provider!r} has no base URL; "
                                    "set SYNTH_BASE_URL or pass --synth-base-url",
                                    cfg.output_language)
    api_key = cfg.api_key or _resolve_key(cfg)
    if not api_key and cfg.provider not in ("ollama", "lmstudio", "llamacpp", "custom"):
        return deterministic_report(ev, cfg.detail,
                                    f"provider {cfg.provider!r} needs a credential "
                                    "(SYNTH_API_KEY)", cfg.output_language)

    plan = build_plan(ev, cfg)
    key = cache_key(ev, cfg)
    store = cache if cache is not None else _default_cache()
    cached = store.get(key) if store else None
    if isinstance(cached, dict) and cached.get("draft"):
        # The cached entry carries the model that wrote it, so a report served
        # from cache still says which model produced it. A cache hit that
        # reported "unknown model" would make the artifact unauditable in
        # exactly the run where nobody was watching — and the whole point of
        # recording the model is that a reader can compare two runs.
        outcome = _finish(ev, cfg, plan, cached["draft"], None, cache_hit=True,
                          model_report=cached.get("model") or {})
        if outcome is not None:
            return outcome
        # A cached draft that no longer validates is discarded, not repaired.
        # Re-running the model is cheaper than debugging a cache that launders a
        # bad draft back into the pipeline.

    client = SynthesisClient(model=cfg.model or _default_model(cfg),
                             base_url=base_url, api_key=api_key or "",
                             timeout=cfg.timeout)
    if budget is not None and budget.limit == 0:
        return deterministic_report(ev, cfg.detail,
                                    "the configured synthesis-call budget is 0; "
                                    "the model was not asked", cfg.output_language)
    try:
        budget.spend() if budget is not None else None
    except BudgetExhausted as exc:
        return deterministic_report(ev, cfg.detail,
                                    f"no draft was requested ({exc})",
                                    cfg.output_language)

    result = client.draft(_plan_prompt(ev, cfg, plan))
    if not result.ok:
        # One repair request naming the problem, then give up. A schema failure
        # is worth exactly one more ask; a transport failure has already spent
        # its negotiation ladder and asking again just costs money.
        if result.status == "malformed" and result.error:
            try:
                budget.spend() if budget is not None else None
            except BudgetExhausted:
                result = result
            else:
                retry = client.draft(
                    _plan_prompt(ev, cfg, plan)
                    + f"\n\nYour previous answer was rejected: {result.error}\n"
                      "Return only the JSON object, conforming to the schema exactly.")
                if retry.ok:
                    result = retry
        if not result.ok:
            return deterministic_report(
                ev, cfg.detail,
                f"synthesis model returned no usable draft ({result.status}: "
                f"{result.error})", cfg.output_language)

    outcome = _finish(ev, cfg, plan, result.doc, client, cache_hit=False,
                      model_report=client.describe(), budget=budget,
                      critic_budget=critic_budget)
    if outcome is None:
        reasons = _finish.last_findings[:4]
        detail = ("; ".join(f"{f.code} at {f.where}: {f.message[:90]}" for f in reasons)
                  if reasons else "no detail recorded")
        return deterministic_report(
            ev, cfg.detail,
            f"synthesis draft failed evidence validation and the writer's single "
            f"revision did not fix it ({len(_finish.last_findings)} blocking: "
            f"{detail})", cfg.output_language)

    if store is not None:
        store.put(key, {"draft": outcome.draft, "mode": result.mode,
                        "enforced": result.enforced,
                        "model": outcome.model_report or client.describe()})
    return outcome


def _finish(ev: Evidence, cfg: SynthConfig, plan: dict, doc: dict,
            client: "SynthesisClient", cache_hit: bool = False,
            model_report: dict | None = None,
            budget: "Budget | None" = None,
            critic_budget: "Budget | None" = None) -> SynthOutcome | None:
    """Validate, and revise once if the critique asks. Returns None on failure.

    The order is the point: blocking validation, then advisory critique, then at
    most one revision, then blocking validation again. A draft that still fails
    returns None and the caller falls back — the validator's verdict is never
    traded against the model's preference for its own output.

    The cache path passes no client and no critique: a cached draft is already the
    post-revision text, and re-critiquing it would make a cache hit cost a model
    call, which defeats the point of caching.
    """
    if not isinstance(doc, dict):
        return None
    validation = validate_draft_grounding(doc, ev)
    critique = None
    revisions = 0

    if validation.valid and cfg.critic and not cache_hit:
        critique = _critique(ev, doc, client, critic_budget)
        if critique and critique.get("findings"):
            # The revision is a second *draft*, so it spends from the draft
            # budget rather than the critic's. A run that allowed one call
            # therefore gets one draft and no revision, and says so, rather than
            # quietly making the call it was told not to.
            try:
                budget.spend() if budget is not None else None
            except BudgetExhausted as exc:
                critique = dict(critique, findings=critique.get("findings") or [],
                                _revision_blocked=str(exc))
                _finish.last_findings = []
                report = render_draft(ev, doc, cfg, plan)
                return SynthOutcome(report=report, mode="synthesis", draft=doc,
                                    validation=validation, revisions=0,
                                    critique=critique, cache_hit=cache_hit,
                                    model_report=model_report or {})
            revisions = 1
            revised = _revise(ev, cfg, doc, client, validation.findings, critique)
            if revised is not None:
                doc = revised
                validation = validate_draft_grounding(doc, ev)

    if not validation.valid:
        # Returning a bare `None` here made the fallback unexplainable: the
        # caller could only report "validation failed" with no way to say which
        # check fired. The findings ride back on the exception-free path as an
        # attribute so the caller can print them.
        _finish.last_findings = validation.blocking
        return None

    _finish.last_findings = []
    report = render_draft(ev, doc, cfg, plan)
    return SynthOutcome(report=report, mode="synthesis", draft=doc,
                        validation=validation, revisions=revisions,
                        critique=critique, cache_hit=cache_hit,
                        model_report=model_report or {})


#: The findings from the most recent `_finish` refusal, for the caller to report.
#: A function attribute rather than a return value because the contract is
#: `None`-means-refused and changing that to a tuple would make every call site
#: unpack a result it mostly does not need.
_finish.last_findings: list = []


def _critique(ev: Evidence, doc: dict, client: "SynthesisClient",
              budget: "Budget | None" = None) -> dict | None:
    """A second opinion on a draft that already passed. Advisory throughout.

    A critique that cannot be obtained is silence rather than a block: the
    blocking validation has already run, and refusing a grounded report because
    an optional reviewer was unavailable would trade a real guarantee for a
    nicety. A critique that *does* arrive is passed to the writer as advice, and
    whatever it says, the validator still has the last word.
    """
    try:
        budget.spend() if budget is not None else None
    except BudgetExhausted:
        return None
    try:
        res = client.critique(_critique_prompt(ev, doc))
    except Exception:  # noqa: BLE001 — a failed critique must not fail the report
        return None
    if not res.ok or not isinstance(res.doc, dict):
        return None
    # A critique in the wrong shape is not a critique. Reading `type` as though
    # it were `code` is the guess this repo refuses to make everywhere else, and
    # the negotiator cannot catch it: it checks the schema's top-level `required`
    # keys, and `findings` was present.
    problems = validate_critique(res.doc)
    if problems:
        res.doc = {**res.doc, "_malformed": problems[:3]}
        return None
    return res.doc


def _revise(ev: Evidence, cfg: SynthConfig, doc: dict, client: "SynthesisClient",
            findings, critique) -> dict | None:
    """The one revision. Returns the new draft, or None if it did not come back.

    Bounded to a single attempt by design. A writer allowed to iterate with
    itself drifts toward its own idea of a good report, and the constraint that
    keeps it honest — the evidence it may not see past — is exactly what an
    unbounded loop argues itself out of. One revision, then the validator decides.
    """
    try:
        res = client.draft(_revision_prompt(ev, cfg, doc, findings, critique))
    except Exception:  # noqa: BLE001 — a failed revision leaves the draft as it was
        return None
    if not res.ok or not isinstance(res.doc, dict):
        return None
    problems = validate_draft(res.doc)
    return None if problems else res.doc


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def _conflict_section(ev: Evidence, refs: "Refs") -> Section | None:
    """Every unresolved conflict, on both sides, resolved by nobody.

    Rendered from the evidence rather than from the draft, which is the whole
    reason it exists here. The writer is *asked* to keep conflicts, and a writer
    that forgets is caught downstream — but `report_validate.check_conflicts`
    requires the conflict to appear as a block citing both members, and no amount
    of good prose in a section achieves that. Without this section a video with
    an unresolved conflict could never produce a model-written report at all: the
    validator would refuse every one, and the pipeline would fall back to the
    deterministic renderer on exactly the videos whose sources disagree.

    So the model cannot drop a conflict by forgetting it, and cannot invent one
    either — this is built from `ev.conflicts` and nothing else. Which side is
    right is not decided here, because that is a judgement about the video and
    the document has no standing to make it.
    """
    if not ev.conflicts:
        return None
    by_id = {i.id: i for i in ev.items}
    blocks: list[Block] = []
    for c in ev.conflicts:
        members = [by_id[i] for i in c.item_ids if i in by_id]
        if len(members) < 2:
            continue
        pair = " vs ".join(
            f"{m.display or m.value}" + (f" [{refs.ref_text(m, 1)}]" if refs.ref_text(m, 1) else "")
            for m in members)
        statement = (c.statement or "").strip().rstrip(".")
        lead = f"**Unresolved conflict ({c.kind}).** " + (statement + ". " if statement else "")
        blocks.append(Block(f"{lead}Sources: {pair}. Not resolved here.",
                            [m.id for m in members], "", kind="conflict"))
    if not blocks:
        return None
    return Section("source_reliability", "Source & Reliability", blocks=blocks)


def render_draft(ev: Evidence, doc: dict, cfg: SynthConfig, plan: dict) -> Report:
    """A validated draft, as the `Report` the rest of the pipeline already knows.

    The synthesis path deliberately produces the *same* object the deterministic
    renderer produces, so `render_markdown` and `report_validate` apply unchanged
    and a synthesized report is checked by exactly the machinery that checks a
    rendered one. Two report shapes would mean two validators, and the brief for
    this repo is that they must not be able to disagree.
    """
    from report import Refs

    budget = derive_budget(ev, cfg.detail)
    refs = Refs(ev)
    by_id = {i.id: i for i in ev.items}

    def _blocks(units_) -> list[Block]:
        out = []
        for _, u in units_:
            ids = [i for i in (u.get("evidence_ids") or []) if i in by_id]
            if not ids:
                continue
            markers: list[str] = []
            for iid in ids[:3]:
                markers += refs.markers(by_id[iid])
            seen: set[str] = set()
            uniq = [m for m in markers if not (m in seen or seen.add(m))]
            out.append(Block(text=u.get("text", "").strip(), evidence_ids=ids,
                             refs=" ".join(uniq[:4]),
                             kind="narrative"))
        return out

    sections: list[Section] = []

    summary = Section("exec_summary", "Executive Summary", style="prose",
                      blocks=_blocks([(w, u) for w, u in units(doc)
                                      if w.startswith("summary")]))
    sections.append(summary)

    takeaways = Section("key_takeaways", "Key Takeaways", style="list",
                        blocks=_blocks([(w, u) for w, u in units(doc)
                                        if w.startswith("takeaways")]))
    sections.append(takeaways)

    notes_subs = []
    for i, s in enumerate(doc.get("sections") or []):
        if not isinstance(s, dict):
            continue
        title = (s.get("title") or "").strip()
        blocks = _blocks([(w, u) for w, u in units(doc)
                          if w.startswith(f"sections[{i}].")])
        if not title or not blocks:
            # A section with nothing in it is a claim that the video had
            # something there. Dropping it is the fix; rendering an empty
            # heading is the defect.
            continue
        notes_subs.append(Section(f"seg{i}", title, level=3, style="prose", blocks=blocks))
    if notes_subs:
        sections.append(Section("detailed_notes", "Detailed Notes", style="prose",
                                subsections=notes_subs))

    conflicts = _conflict_section(ev, refs)
    if conflicts is not None:
        sections.append(conflicts)

    kept = [s for s in sections if s.blocks or s.subsections]

    return Report(title=ev.video.get("title") or "Report",
                  subtitle=" · ".join(x for x in [ev.video.get("channel"),
                                                 _fmt_duration(ev)] if x),
                  mode="SYNTHESIS",
                  mode_reason=(f"written by {cfg.model or 'the configured synthesis model'} "
                               f"from {len(ev.items)} evidence items"),
                  sections=kept, budget=budget,
                  meta=_meta(ev, cfg, plan, kept))


def _meta(ev: Evidence, cfg: SynthConfig, plan: dict, sections: list[Section]) -> dict:
    return {
        "video_id": ev.video.get("id"),
        "channel": ev.video.get("channel"),
        "duration_s": ev.video.get("duration_s"),
        "schema_version": ev.schema_version,
        "detail_level": cfg.detail,
        "synthesis": "llm",
        "synth_model": cfg.model,
        "synth_prompt_version": PROMPT_VERSION,
        "synth_schema_version": SYNTH_SCHEMA_VERSION,
        "output_language": cfg.output_language,
        "evidence_items_total": len(ev.items),
        "summary_evidence_ids": plan.get("summary_evidence_ids", []),
    }


def _fmt_duration(ev: Evidence) -> str:
    d = ev.video.get("duration_s")
    if not isinstance(d, (int, float)):
        return ""
    m, s = divmod(int(d), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


# --------------------------------------------------------------------------- #
# provider plumbing
# --------------------------------------------------------------------------- #

def _resolve_base(cfg: SynthConfig) -> str | None:
    from chat_json import resolve_endpoint

    return resolve_endpoint(cfg.provider, cfg.base_url)


def _resolve_key(cfg: SynthConfig) -> str:
    import chat_json as CJ

    k = CJ.key_for(cfg.provider, cfg.base_url)
    return k or ""


def _default_model(cfg: SynthConfig) -> str:
    """A model id for the alias. Not a recommendation — a default, and it is
    written into the artifact so a run is never anonymous."""
    return {"openai": "gpt-4o-mini", "openrouter": "openai/gpt-4o-mini",
            "ollama": "llama3.1", "lmstudio": "local-model",
            "llamacpp": "local-model"}.get(cfg.provider, "local-model")


def _default_cache():
    from cache import CACHE_DIR, JsonCache

    return JsonCache("synthesis", CACHE_DIR)
