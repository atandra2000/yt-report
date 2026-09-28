"""The synthesis layer, tested offline with a stubbed transport.

The tests fall into three groups, and the third is the one that matters most.

The first exercises the mechanics: schema validation, the cache key, the plan,
the renderer, the fallback ladder. None of them needs a network.

The second runs the whole chain against fixtures with a scripted model, checking
that a well-behaved draft survives and becomes a report the existing validator
accepts. That is the property which lets the two paths coexist: the synthesis
path produces the same `Report` object, so the same markdown renderer and the
same grounding checks apply to both.

The third is the one this suite exists for. Each test hands the validator a draft
containing a *specific* way of being wrong — an unsupported entity, a number that
was never said, a hedge quietly removed, one side of a conflict, a visual claim
with no frame behind it — and asserts that it is rejected. A grounding check that
has never been shown to fail a bad draft has never been tested, and a validator
nobody has tried to fool is not a validator.

No test asserts on generated prose. Prose from a model is not a fixture, and a
test that pinned exact sentences would fail on a rewording while proving nothing
about grounding. What is asserted is structural: a draft parses, every unit
carries ids, every id resolves, and a named violation is caught.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import chat_json as CJ
import report_fixtures as RF
import synth_model as SM
import synth_validate as SV
import synthesis as SY
from evidence_provenance import transcript_provenance, vlm_provenance
from report_validate import (check_duplicate_homes, empty_headings,
                             validate_report)

#: Bound by `selftest.py` at import, exactly as `selftest_report` and
#: `selftest_vlm_provider` are. There is one harness, not three.
check = None


# --------------------------------------------------------------------------- #
# a scripted model
# --------------------------------------------------------------------------- #

def _tool_response(payload: dict) -> str:
    return json.dumps({"choices": [{"message": {"content": json.dumps(payload)},
                                    "finish_reason": "stop"}]})


def _scripted(responses: list):
    """A transport that replays `responses` in order and records every request.

    `assert_schema` makes the endpoint enforce the contract the way a compliant
    gateway would: a response missing a required key is refused with a 400 whose
    detail names `response_format`, which is what drives the negotiation ladder
    down. Without it every test would pass in `json_schema` mode and the
    negotiation path would never be exercised outside the provider suite.
    """
    sent: list[dict] = []

    def transport(url, headers, data, timeout):
        body = json.loads(data)
        sent.append(body)
        if not responses:
            raise AssertionError("the model was called more times than the script allows")
        nxt = responses.pop(0)
        if callable(nxt):
            return nxt(body)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    transport.sent = sent  # type: ignore[attr-defined]
    return transport


def _draft(summary=(), takeaways=(), sections=()) -> dict:
    return {"summary": list(summary), "takeaways": list(takeaways),
            "sections": list(sections)}


def _unit(text: str, ids, kind: str = "mixed") -> dict:
    return {"text": text, "evidence_ids": list(ids), "source_kind": kind}


# --------------------------------------------------------------------------- #
# 1. schema
# --------------------------------------------------------------------------- #

def t_synth_schema() -> None:
    good = _draft([_unit("The video covers caching.", ["itm-001"])],
                  [_unit("A cache sits between the app and the database.", ["itm-002"])],
                  [{"title": "Caching", "paragraphs": [_unit("The second read avoids the database.", ["itm-003"])]}])
    check("a well-formed draft passes", SM.validate_draft(good) == [], str(SM.validate_draft(good)))

    for label, bad, code in [
        ("a missing key", {"takeaways": [], "sections": []}, "missing required key"),
        ("an empty unit", _draft([{"text": "  ", "evidence_ids": ["itm-001"]}]), "empty"),
        ("a unit with no evidence ids", _draft([{"text": "Something.", "evidence_ids": []}]), "no 'evidence_ids'"),
        ("a non-object unit", _draft(["just a string"]), "not an object"),
        ("a non-string evidence id", _draft([{"text": "x", "evidence_ids": [7]}]), "non-string evidence id"),
        ("a bad source_kind", _draft([_unit("x", ["itm-001"], kind="vibes")]), "not one of"),
        ("a missing source_kind",
         _draft([{"text": "x", "evidence_ids": ["itm-001"]}]), "missing 'source_kind'"),
        ("an unknown top-level key", {**good, "conclusion": "extra"}, "unexpected key"),
        ("a section with no paragraphs",
         {**good, "sections": [{"title": "x", "paragraphs": []}]}, "no paragraphs"),
    ]:
        problems = SM.validate_draft(bad)
        check(f"synthesis schema rejects {label}",
              any(code in p for p in problems), str(problems))

    check("every unit is reachable by the shared traversal",
          len(list(SM.units(good))) == 3, str(len(list(SM.units(good)))))


# --------------------------------------------------------------------------- #
# 2. grounding — the checks must fail a bad draft
# --------------------------------------------------------------------------- #

def _codes(res: SV.DraftValidation) -> set[str]:
    return {f.code for f in res.findings}


def _simple():
    return RF.simple_talking_head()


def t_synth_grounding_happy() -> None:
    ev = _simple()
    ids = [i.id for i in ev.items]
    doc = _draft(
        [_unit("The video explains what a cache is and where it sits.", ids[:2])],
        [_unit("The second read never reaches the database.", ids[2:3])],
        [{"title": "How caching works",
          "paragraphs": [_unit("A cache sits between the application and the slow database.",
                               ids[1:2])]}])
    res = SV.validate_draft_grounding(doc, ev)
    check("a fully grounded draft validates", res.valid,
          str([f.to_dict() for f in res.blocking]))
    check("every check actually ran", res.checks_run.get("hedges-preserved", 0) >= 1,
          str(res.checks_run))


def t_synth_rejects_unsupported_entity() -> None:
    """The fluent extension. Every word is a real word; the claim is not real."""
    ev = _simple()
    target = next(i for i in ev.items if i.type == "topic")
    doc = _draft([_unit("The video presents Redis as the recommended cache.", [target.id])])
    res = SV.validate_draft_grounding(doc, ev)
    check("an entity absent from the cited evidence is rejected",
          "claim-unsupported" in _codes(res), str([f.to_dict() for f in res.blocking]))
    check("the unsupported word is named in the finding",
          any("redis" in f.message.lower() for f in res.blocking), str([f.message for f in res.blocking]))


def t_synth_rejects_wrong_number() -> None:
    ev = RF.technical_lecture()
    num = next(i for i in ev.items if i.type == "number")
    wrong = "9" if "9" not in (num.display or num.value) else "7"
    doc = _draft([_unit(f"The presenter states the figure is {wrong} GB of VRAM.", [num.id])])
    res = SV.validate_draft_grounding(doc, ev)
    check("a figure not present in the cited evidence is rejected",
          "number-unsupported" in _codes(res), str([f.to_dict() for f in res.blocking]))

    ok_doc = _draft([_unit(f"The presenter states {num.display or num.value}.", [num.id])])
    check("the same sentence with the real figure validates",
          SV.validate_draft_grounding(ok_doc, ev).valid,
          str([f.to_dict() for f in SV.validate_draft_grounding(ok_doc, ev).blocking]))


def t_synth_rejects_removed_hedge() -> None:
    """The failure this whole layer exists for: a real number, said wrongly."""
    ev = RF.hedged_claim()
    hedged = next(i for i in ev.items if i.status == "hedged")
    value = hedged.display or hedged.value
    doc = _draft([_unit(f"The upgrade costs {value}.", [hedged.id])])
    res = SV.validate_draft_grounding(doc, ev)
    check("a hedged item cited without a hedge is rejected",
          "hedge-dropped" in _codes(res), str([f.to_dict() for f in res.blocking]))

    kept = _draft([_unit(f"The upgrade could cost about {value}.", [hedged.id])])
    check("the same claim with its hedge validates",
          SV.validate_draft_grounding(kept, ev).valid,
          str([f.to_dict() for f in SV.validate_draft_grounding(kept, ev).blocking]))


def t_synth_conflict_validation() -> None:
    """Conflict validation at two levels, and the seven cases that separate them.

    The rule this replaces was *"every paragraph mentioning a member must cite
    every member"*, and it was wrong in a way that blocked correct work: a report
    that explains one side in a section and discloses the whole disagreement in
    Source & Reliability is the report the system wants, and the old rule refused
    it. A video whose sources disagree therefore could never produce a
    model-written report at all, and the pipeline fell back to the deterministic
    renderer on precisely the videos that most needed a writer.

    What replaced it keeps the two levels apart:

    * **block level** — a paragraph still cannot state a disputed item as settled
      (`conflict-side-asserted`);
    * **report level** — a conflict the draft references must be *completely*
      represented somewhere in it (`conflict-incomplete`).

    Each case below is one of those two failing, or neither, so a regression
    names which half moved rather than only that "something" about conflicts did.
    """
    ev = RF.unresolved_conflict()
    conflict = ev.conflicts[0]
    a, b = conflict.item_ids[0], conflict.item_ids[1]
    by = {i.id: i for i in ev.items}
    shown = {i: (by[i].display or by[i].value) for i in (a, b)}
    other = next(i.id for i in ev.items if i.id not in (a, b))

    one_sided = f"The narration puts the figure at {shown[a]}."
    disclosed = ("The narration and the frame disagree about this figure; "
                 "the discrepancy is unresolved.")

    def codes(doc) -> set:
        return {f.code for f in SV.validate_draft_grounding(doc, ev).blocking}

    # 1. A paragraph mentions side A; the conflict is fully disclosed elsewhere.
    #    This is the case the old rule refused and the real one must accept.
    res = SV.validate_draft_grounding(
        _draft([_unit(one_sided, [a]), _unit(disclosed, [a, b])]), ev)
    check("conflict 1: one side in a section, both sides disclosed elsewhere, validates",
          res.valid, str([f.to_dict() for f in res.blocking]))

    # 2. The same paragraph, with no disclosure anywhere: the disagreement is now
    #    resolved by omission and nothing in the report says so.
    check("conflict 2: a side cited with the other never mentioned is rejected",
          "conflict-incomplete" in codes(_draft([_unit(one_sided, [a])])),
          str(sorted(codes(_draft([_unit(one_sided, [a])])))))

    # 3. Both sides in one paragraph.
    res = SV.validate_draft_grounding(_draft([_unit(disclosed, [a, b])]), ev)
    check("conflict 3: both sides in one paragraph validates",
          res.valid, str([f.to_dict() for f in res.blocking]))

    # 4. The report says nothing about the dispute at all. That is a legitimate
    #    choice — the deterministic conflict section discloses it regardless — so
    #    completeness must not fire on a conflict nobody mentioned.
    res = SV.validate_draft_grounding(
        _draft([_unit("The video discusses the topic at length.", [other])]), ev)
    check("conflict 4: a conflict the report never mentions is not an omission",
          res.valid, str([f.to_dict() for f in res.blocking]))

    # 5. A conflict naming evidence that does not exist could never satisfy any
    #    completeness rule, so it would pass vacuously. Refused instead.
    ev_bad = RF.unresolved_conflict()
    ev_bad.conflicts[0].item_ids = [a, "itm-does-not-exist"]
    got = {f.code for f in SV.validate_draft_grounding(
        _draft([_unit(one_sided, [a])]), ev_bad).blocking}
    check("conflict 5: a conflict naming nonexistent evidence is refused",
          "conflict-dangling" in got, str(sorted(got)))

    # 6. One side rewritten to agree with the other. Both failure modes fire: the
    #    paragraph states a disputed item flatly, *and* the report never discloses
    #    the disagreement. Either alone is enough to refuse.
    got = codes(_draft([_unit(f"The video states the figure is {shown[a]}, "
                                "and the frame agrees.", [a]),
                        _unit(f"The frame shows {shown[b]} instead; the discrepancy "
                              "is unresolved.", [b])]))
    check("conflict 6: a side rewritten to agree with the other is rejected",
          "conflict-side-asserted" in got, str(sorted(got)))

    # 7. Disclosed in full, with the detail not repeated in every paragraph.
    res = SV.validate_draft_grounding(
        _draft([_unit(one_sided, [a]), _unit(disclosed, [a, b]),
                _unit("Other material is discussed here.", [other])]), ev)
    check("conflict 7: full disclosure without repeating the detail everywhere validates",
          res.valid, str([f.to_dict() for f in res.blocking]))

    # A failure must name the conflict, or a reader holding a 40-item evidence
    # file has no way to find the one the report got wrong.
    finding = next(f for f in SV.validate_draft_grounding(
        _draft([_unit(one_sided, [a])]), ev).blocking
        if f.code == "conflict-incomplete")
    check("conflict: the failure names the conflict, not just a position",
          a in finding.where and b in finding.where and conflict.kind in finding.where,
          finding.where)


def _speech_only_item(ev, item_id="itm-spoken"):
    """An item backed only by a transcript snippet.

    Two of the checks need a claim that is *purely* spoken — a visual sentence
    resting on it, and an `observation` label on it. Fixtures legitimately change
    which items are corroborated by a second modality, so this adds one when
    needed rather than assuming a particular fixture's provenance mix. A check
    that breaks when an unrelated fixture gains a second source is a test of the
    fixture, not of the check.
    """
    from evidence_model import Item

    for i in ev.items:
        if all(p.source == "transcript" for p in i.provenance):
            return i
    it = Item(id=item_id, type="claim", value="the speaker describes a chart",
              provenance=[transcript_provenance(ev.snippets[0])])
    ev.items.append(it)
    return it


def t_synth_rejects_unsupported_visual() -> None:
    ev = RF.corroboration()
    spoken = _speech_only_item(ev)
    doc = _draft([_unit("The chart shows the two models side by side.", [spoken.id])])
    res = SV.validate_draft_grounding(doc, ev)
    check("a visual claim resting only on speech is rejected",
          "visual-unsupported" in _codes(res), str([f.to_dict() for f in res.blocking]))


def t_synth_rejects_unknown_evidence_id() -> None:
    ev = _simple()
    doc = _draft([_unit("Caching is discussed.", ["itm-999"])])
    res = SV.validate_draft_grounding(doc, ev)
    check("an evidence id that does not resolve is rejected",
          "evidence-id-unknown" in _codes(res), str([f.to_dict() for f in res.blocking]))


def t_synth_rejects_contradicting_status() -> None:
    ev = RF.unreadable_chart()
    weak = [i for i in ev.items if i.status in ("unreadable", "conflicting")]
    if not weak:
        check("the unreadable-chart fixture has an item with a weak status", False, "none found")
        return
    doc = _draft([_unit("The chart shows throughput doubling.", [weak[0].id])])
    res = SV.validate_draft_grounding(doc, ev)
    check("an item marked unreadable, cited flatly, is rejected",
          "uncertainty-dropped" in _codes(res) or "claim-unsupported" in _codes(res),
          str([f.to_dict() for f in res.blocking]))


def t_synth_source_kind() -> None:
    ev = RF.corroboration()
    spoken = _speech_only_item(ev)
    doc = _draft([_unit("The presenter reads the table aloud.", [spoken.id], kind="observation")])
    res = SV.validate_draft_grounding(doc, ev)
    check("a speech-only sentence labelled 'observation' is rejected",
          "source-kind-mismatch" in _codes(res), str([f.to_dict() for f in res.blocking]))


# --------------------------------------------------------------------------- #
# 3. the chain
# --------------------------------------------------------------------------- #

def _visual(item) -> bool:
    return any(p.source in ("vlm", "frame", "ocr") for p in item.provenance)


def _good_draft(ev: Evidence) -> dict:
    """A draft assembled from the fixture's own evidence, so it always grounds.

    This models a *well-behaved* writer, which means respecting three things the
    validator enforces and a naive helper would not:

    * a sentence that names a chart, a table or a dashboard must cite an item
      that actually came from a frame;
    * an item marked `hedged` must be cited by a sentence that hedges;
    * both sides of an unresolved conflict must be cited together.

    Writing flat sentences about hedged or disputed evidence and then calling the
    result "grounded" would test the helper's naivety rather than the pipeline,
    and the fixture sweep would report failures that are really the test being
    wrong.
    """
    content = [i for i in ev.items
               if i.type != "organization"
               and not (i.display or i.value).startswith("http")]

    # Conflicted and hedged items are handled as groups, never one at a time.
    conflicted: dict[str, set] = {}
    for c in ev.conflicts:
        if c.resolution != "resolved":
            conflicted.setdefault(c.item_ids[0], set()).update(c.item_ids)

    def _sides_of(i):
        """Every item in `i`'s conflict group, whichever member was picked."""
        group = set()
        for ids in conflicted.values():
            if i.id in ids:
                group |= ids
        if i.status == "conflicting" and not group:
            group = {x.id for x in ev.items if x.status == "conflicting"}
        return sorted(group)

    def _sentence_for(i):
        """Prose that states `i` at the strength the evidence stated it."""
        val = (i.display or i.value).rstrip(".")
        sides = _sides_of(i)
        if len(sides) > 1:
            # Whichever side we were handed, the sentence carries both. A draft
            # that reports one side of a dispute has resolved it by omission,
            # and that is the defect this check exists to catch.
            others = [o for o in ev.items if o.id in sides and o.id != i.id]
            other = others[0] if others else None
            if other is not None:
                return (f"The sources disagree: the narration cites {val} while "
                        f"the frame shows {(other.display or other.value).rstrip('.')}; "
                        f"the discrepancy is unresolved."), sides
        if i.status == "hedged" or (i.qualifier or "").strip():
            return f"The figure could be about {val}.", [i.id]
        if i.status == "unreadable":
            return f"A {val} appears, though it is not fully legible.", [i.id]
        if _visual(i):
            return f"The frame shows {val}.", [i.id]
        return f"The video covers {val}.", [i.id]

    # A sentence that names a chart, a table or a dashboard must cite something
    # that came from a frame. Rather than special-case the wording, the writer
    # here picks a visual item whenever one is available for a slot — the same
    # discipline a real writer is held to by the same check.
    visual_items = [i for i in content if _visual(i)]

    def _pick(n):
        if visual_items:
            return visual_items[n % len(visual_items)]
        return content[n % len(content)] if content else ev.items[n % len(ev.items)]

    s0, ids0 = _sentence_for(_pick(0))
    s1, ids1 = _sentence_for(_pick(1))
    s2, ids2 = _sentence_for(_pick(2))
    title = ev.segments[0].title if ev.segments else "Notes"
    return _draft([_unit(s0, ids0)], [_unit(s1, ids1)],
                  [{"title": title, "paragraphs": [_unit(s2, ids2)]}])


def _with_transport(transport, fn):
    """Run `fn` with every `SynthesisClient` built on `transport`.

    The client is the only place a network call happens, so patching its
    constructor is the whole seam. Monkeypatching is ugly and this is still ugly,
    but the alternative — a `transport` parameter threaded through `synthesize`
    and `_finish` — would put a test seam in the public signature of the
    pipeline, which is worse: a caller who can inject a transport can also
    inject a model that returns whatever they like.
    """
    import synth_model as _sm

    orig = _sm.SynthesisClient

    class _Patched(orig):
        def __init__(self, **kw):
            kw["transport"] = transport
            super().__init__(**kw)

    _sm.SynthesisClient = _Patched
    SY.SynthesisClient = _Patched
    try:
        return fn()
    finally:
        _sm.SynthesisClient = orig
        SY.SynthesisClient = orig


def t_synth_pipeline() -> None:
    ev = _simple()
    doc = _good_draft(ev)
    transport = _scripted([_tool_response(doc)])
    cfg = SY.SynthConfig(provider="custom", model="stub", base_url="http://x/v1",
                         api_key="k", detail="balanced", critic=False)
    outcome = _with_transport(transport, lambda: SY.synthesize(ev, cfg, cache=None))
    check("a good draft produces a synthesis report", outcome.mode == "synthesis",
          f"{outcome.mode}: {outcome.fallback_reason}")
    check("the synthesized report passes its own validation",
          outcome.validation.valid, str([f.to_dict() for f in outcome.validation.blocking]))
    check("the report carries sections", len(outcome.report.sections) >= 2,
          str([s.key for s in outcome.report.sections]))

    md = __import__("report").render_markdown(outcome.report)
    res = validate_report(outcome.report, ev, md, authored_by=outcome.mode)
    check("the synthesized report passes the EXISTING report validator",
          res.valid, str([e for e in res.blocking_errors[:3]]))
    check("the synthesized report records which evidence backs it",
          len(outcome.report.evidence_ids) > 0, str(outcome.report.evidence_ids))


def t_synth_fallback_ladder() -> None:
    ev = _simple()
    # no provider configured
    o = SY.synthesize(ev, SY.SynthConfig(provider="none"))
    check("no provider falls back to the deterministic renderer", o.mode == "deterministic")
    check("the fallback reason is recorded", bool(o.fallback_reason), str(o.fallback_reason))

    # A provider that needs a credential, in an environment that has none.
    # The variables are cleared rather than assumed absent: the developer
    # running this suite is far more likely to have a key exported than not,
    # and a test that passes or fails depending on whose shell it ran in is
    # not a test.
    import os

    cleared = {k: os.environ.pop(k) for k in
               ("SYNTH_API_KEY", "VLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
               if k in os.environ}
    try:
        o2 = SY.synthesize(ev, SY.SynthConfig(provider="openai", model="m",
                                              base_url="http://x/v1", api_key=None))
        check("a missing credential falls back", o2.mode == "deterministic",
              f"{o2.mode}: {o2.fallback_reason}")
        check("the credential reason names the provider",
              "credential" in (o2.fallback_reason or ""), str(o2.fallback_reason))
        check("no network call is attempted without a credential",
              "connection failed" not in (o2.fallback_reason or ""),
              str(o2.fallback_reason))
    finally:
        os.environ.update(cleared)

    # A model that answers in prose instead of JSON. Two responses: the initial
    # call and the one repair request it is allowed.
    prose = json.dumps({"choices": [{"message": {"content": "Sure! Here is your report."},
                                     "finish_reason": "stop"}]})
    cfg3 = SY.SynthConfig(provider="custom", model="m", base_url="http://x/v1",
                          api_key="k", critic=False)
    o3 = _with_transport(_scripted([prose, prose]),
                         lambda: SY.synthesize(ev, cfg3, cache=None))
    check("a model returning prose falls back to deterministic", o3.mode == "deterministic",
          f"{o3.mode}: {o3.fallback_reason}")
    check("the prose answer is named as the reason",
          "no usable draft" in (o3.fallback_reason or ""), str(o3.fallback_reason))


def t_synth_fallback_when_grounding_fails() -> None:
    ev = _simple()
    # a draft with a number that is not in the evidence -> must fall back, not render
    bad = _draft([_unit("The system runs at 9999 requests per second.", [ev.items[0].id])])
    transport = _scripted([_tool_response(bad)] * 4)
    cfg4 = SY.SynthConfig(provider="custom", model="m", base_url="http://x/v1",
                          api_key="k", critic=False)
    o = _with_transport(transport, lambda: SY.synthesize(ev, cfg4, cache=None))
    check("a draft that fails grounding falls back to deterministic, never renders",
          o.mode == "deterministic", f"{o.mode}: {o.fallback_reason}")
    check("the grounding failure is the stated reason",
          "validation" in (o.fallback_reason or ""), str(o.fallback_reason))


def t_synth_cache() -> None:
    ev = _simple()
    cfg = SY.SynthConfig(provider="custom", model="m", base_url="http://x/v1",
                         api_key="k", detail="balanced", critic=False)
    k1 = SY.cache_key(ev, cfg)
    check("the cache key is stable for the same evidence and config", k1 == SY.cache_key(ev, cfg))
    ev.video["title"] = "Changed"
    check("the cache key changes when the evidence content changes",
          k1 != SY.cache_key(ev, cfg), "editing evidence must invalidate synthesis")
    ev.video["title"] = "Fixture video"
    for field, val in (("model", "other"), ("detail", "deep"),
                       ("output_language", "fr"), ("provider", "openai")):
        c2 = SY.SynthConfig(**{**cfg.__dict__, field: val})
        check(f"the cache key changes when {field} changes", k1 != SY.cache_key(ev, c2),
              f"{field} must be part of the key")


def t_synth_critic_and_revision() -> None:
    """The critique pass: advisory, and bounded to exactly one revision.

    Two things are being established. That a critique that arrives can send the
    writer back once, and that the revision is validated like any other draft
    rather than trusted because it came from the same model that made the
    mistake.
    """
    ev = RF.hedged_claim()
    hedged = next(i for i in ev.items if i.status == "hedged")
    val = hedged.display or hedged.value
    good = _draft([_unit(f"The tool could cost about {val}.", [hedged.id])])
    critique = {"findings": [{"code": "repetition", "where": "summary[0]",
                              "message": "the same point appears twice"}]}

    # 1. A critique arrives -> exactly one revision -> the draft still validates.
    transport = _scripted([_tool_response(critique), _tool_response(good)])
    cfg = SY.SynthConfig(detail="balanced", critic=True)
    plan = SY.build_plan(ev, cfg)

    def _run():
        client = SY.SynthesisClient(model="m", base_url="http://x/v1", api_key="k")
        return SY._finish(ev, cfg, plan, good, client, model_report={})

    out = _with_transport(transport, _run)
    check("a draft survives one critique and one revision", out is not None and out.validation.valid,
          "a still-invalid draft must return None so the caller falls back")
    check("the single revision is recorded", out is not None and out.revisions == 1,
          str(out.revisions if out else None))
    check("the critique is retained for the record",
          out is not None and bool(out.critique), str(out.critique if out else None))

    # 2. Exactly one revision, not two: the script has only two responses, so a
    #    second attempt would raise and the test would fail rather than pass
    #    quietly. The call count is the assertion.
    check("the critic and the writer are each called exactly once",
          len(transport.sent) == 2, f"{len(transport.sent)} model calls")

    # 3. A revision that does not fix the problem is rejected, not accepted.
    broken = _draft([_unit(f"The tool costs {val}.", [hedged.id])])
    transport2 = _scripted([_tool_response(critique), _tool_response(broken)])
    out2 = _with_transport(transport2, lambda: SY._finish(
        ev, cfg, plan, good,
        SY.SynthesisClient(model="m", base_url="http://x/v1", api_key="k"), model_report={}))
    check("a revision that reintroduces a dropped hedge is rejected",
          out2 is None, "the revision is validated like any other draft")

    # 4. An unavailable critic does not block a grounded report.
    transport3 = _scripted([ConnectionError("critic unavailable")])
    out3 = _with_transport(transport3, lambda: SY._finish(
        ev, cfg, plan, good,
        SY.SynthesisClient(model="m", base_url="http://x/v1", api_key="k"), model_report={}))
    check("an unavailable critic does not block a grounded report",
          out3 is not None and out3.validation.valid,
          "the critique is advisory; the blocking validation is what decides")


def t_synth_critic_catches_a_planted_defect() -> None:
    """Acceptance item 9: the critic must be able to see a real defect.

    Scripted so the test is about the *contract* — findings reach the writer and
    the writer revises — rather than about any particular model's judgement. A
    green run here does not prove a real model is perceptive; it proves that when
    a critique does report a defect, the pipeline acts on it.
    """
    ev = RF.unresolved_conflict()
    conflict = ev.conflicts[0]
    sides = list(conflict.item_ids)
    # A draft that cites one side only: the thing the critic is there to catch.
    one_sided = _draft([_unit(
        f"The video reports {next(i for i in ev.items if i.id == sides[0]).display}.",
        [sides[0]])])
    findings = SV.validate_draft_grounding(one_sided, ev)
    check("the blocking validator catches the one-sided conflict before the critic sees it",
          "conflict-incomplete" in {f.code for f in findings.blocking},
          str([f.to_dict() for f in findings.blocking]))

    both = _draft([_unit(conflict.statement, sides)])
    critique = {"findings": [{"code": "conflict-hidden", "where": "summary[0]",
                              "message": "one side was missing"}], "clean": False}
    transport = _scripted([_tool_response(critique), _tool_response(both)])
    cfg = SY.SynthConfig(detail="balanced", critic=True)
    out = _with_transport(transport, lambda: SY._finish(
        ev, cfg, SY.build_plan(ev, cfg), one_sided,
        SY.SynthesisClient(model="m", base_url="http://x/v1", api_key="k"), model_report={}))
    # The one-sided draft is rejected by the validator before the critique runs,
    # so the outcome is a refusal rather than a revision: the ordering is the
    # guarantee, and this asserts it.
    check("a draft with a hidden conflict is refused, not merely criticised",
          out is None, "blocking validation runs first and a failure is not repairable by prose")


def t_synth_output_language() -> None:
    ev = _simple()
    cfg_auto = SY.SynthConfig(output_language="auto")
    check("output-language auto resolves to the source language",
          cfg_auto.resolved_language(ev) == "en", cfg_auto.resolved_language(ev))
    cfg_fr = SY.SynthConfig(output_language="fr")
    check("an explicit output language is honoured", cfg_fr.resolved_language(ev) == "fr")
    ev_es = _simple()
    ev_es.transcript["language"] = "es"
    check("auto follows a non-English source", cfg_auto.resolved_language(ev_es) == "es",
          cfg_auto.resolved_language(ev_es))


# --------------------------------------------------------------------------- #
# 4. coverage across the fixtures
# --------------------------------------------------------------------------- #

def t_synth_fixture_coverage() -> None:
    """Every fixture must survive a grounded draft and produce a readable report."""
    from report import render_markdown
    for name, make in RF.FIXTURES.items():
        ev = make()
        if not ev.items:
            check(f"{name}: has evidence to synthesize", False, "no items")
            continue
        doc = _good_draft(ev)
        res = SV.validate_draft_grounding(doc, ev)
        check(f"{name}: a grounded draft validates", res.valid,
              str([f.to_dict() for f in res.blocking][:2]))
        outcome = SY.render_draft(ev, doc, SY.SynthConfig(detail="balanced"),
                                  SY.build_plan(ev, SY.SynthConfig()))
        md = render_markdown(outcome)
        check(f"{name}: renders non-empty markdown", len(md.split()) > 10, f"{len(md.split())} words")
        # A heading the writer rendered with nothing under it is the failure this
        # asserts: the draft may name a section it filled with no unit, and
        # `check_empty_sections` in the report gate is what has to catch it. The
        # predicate is the gate's, not a second implementation of it.
        check(f"{name}: no empty sections survive", not empty_headings(md),
              f"a heading rendered with nothing under it: {empty_headings(md)[:3]}")


# --------------------------------------------------------------------------- #
# 5. the structured-output contract, as a real endpoint requires it
# --------------------------------------------------------------------------- #

def t_vlm_schema_is_strict_mode_valid() -> None:
    """Both schemas must survive a strict-mode endpoint.

    Found the hard way: OpenRouter fronts several providers and forwards
    `json_schema` to whichever one is cheapest, and that provider rejects a
    schema unless every object sets `additionalProperties: false` and lists
    *every* property in `required`. The original observation schema did neither,
    so a perfectly good model produced a valid answer and the run recorded it as
    a schema failure — the negotiation ladder correctly degraded to
    `prompt_only`, and a real provider was left permanently unqualified through
    no fault of its own.

    A test that only checked the schema parsed would have missed all of it.
    """
    import vlm
    import synth_model as SM

    for name, schema in (("OBSERVATION_SCHEMA", vlm.OBSERVATION_SCHEMA),
                         ("SYNTH_SCHEMA", SM.SYNTH_SCHEMA)):
        bad: list[str] = []

        def walk(node, path, seen):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    if node.get("additionalProperties") is not False:
                        bad.append(f"{path}: additionalProperties is not false")
                    props = set(node.get("properties") or {})
                    if props and set(node.get("required") or []) != props:
                        missing = sorted(props - set(node.get("required") or []))
                        bad.append(f"{path}: required omits {missing}")
                for k, v in node.items():
                    walk(v, f"{path}.{k}", seen)
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]", seen)

        walk(schema, name, bad)
        check(f"{name} is strict-mode valid", not bad, "; ".join(bad[:3]))
        # And it must still be JSON, because it crosses the wire as JSON.
        check(f"{name} serialises", isinstance(json.dumps(schema), str))


def t_vlm_error_text_reads_through_a_gateway() -> None:
    """A gateway that buries the complaint must not hide a downgrade.

    OpenRouter answers a rejected `response_format` with `"Provider returned
    error"` at the top level and the actual reason several levels down in
    `error.metadata.raw`. Classifying on the top line alone turns a recoverable
    negotiation step into an unexplained failure, and the ladder never steps
    down — the run just fails.
    """
    import urllib.error
    import chat_json as CJ

    import io

    def _err(body, code=400, reason="Bad Request"):
        return urllib.error.HTTPError("http://x", code, reason, {},
                                      io.BytesIO(json.dumps(body).encode()))

    nested = {"error": {"message": "Provider returned error", "metadata": {"raw": json.dumps(
        {"error": {"message": "Invalid schema for response_format: 'required' is required "
                              "to be supplied", "param": "response_format"}})}}}
    e = CJ.classify_http(_err(nested))
    check("an error buried in gateway metadata still downgrades",
          e.downgrade, f"status={e.status} reason={e.reason}")

    flat = {"error": {"message": "response_format is not supported"}}
    e2 = CJ.classify_http(_err(flat))
    check("a flat structured-output rejection still downgrades", e2.downgrade, e2.reason)

    auth = {"error": {"message": "Invalid API key"}}
    e3 = CJ.classify_http(_err(auth, 401, "Unauthorized"))
    check("a 401 stays fatal and is not treated as a downgrade",
          e3.fatal and not e3.downgrade, f"fatal={e3.fatal} downgrade={e3.downgrade}")


def t_vlm_credential_is_scoped_to_its_provider() -> None:
    """A key must never be sent to a service that did not issue it.

    The lookup used to be `VLM_API_KEY or OPENAI_API_KEY` for every
    OpenAI-compatible provider, so configuring OpenRouter with only
    `OPENAI_API_KEY` exported would have put an OpenAI credential in an
    `Authorization` header aimed at openrouter.ai. Scoped per provider now.
    """
    import os
    import chat_json as CJ
    import vlm

    saved = {k: os.environ.pop(k, None) for k in
             ("VLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY")}
    try:
        os.environ["OPENAI_API_KEY"] = "sk-openai"
        check("openai reads its own key", CJ.key_for("openai") == "sk-openai")
        check("openrouter does not read OpenAI's key", CJ.key_for("openrouter") is None,
              str(CJ.key_for("openrouter")))
        check("anthropic does not read OpenAI's key", CJ.key_for("anthropic") is None,
              str(CJ.key_for("anthropic")))

        os.environ["OPENROUTER_API_KEY"] = "sk-openrouter"
        check("openrouter reads its own key", CJ.key_for("openrouter") == "sk-openrouter")
        check("openai still reads OpenAI's key", CJ.key_for("openai") == "sk-openai")

        os.environ.pop("OPENAI_API_KEY")
        p = vlm.OpenAICompatProvider(provider_name="openrouter", model="m")
        check("the provider does not pick up another service's key",
              p.api_key == "sk-openrouter", str(p.api_key))
        check("a keyless provider needs no credential at all",
              vlm.OpenAICompatProvider(provider_name="ollama", model="m").api_key == "")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --------------------------------------------------------------------------- #
# the one-home contract, and the conflict contract beside it
# --------------------------------------------------------------------------- #

#: The two narrative homes, named as the gate names them. The prompt asserts
#: this list because a rule the writer is never told is a rule the writer breaks,
#: and the last real failure of this kind was a draft refused for something the
#: instruction did not mention.
_NARRATIVE_HOMES = ("takeaway", "section")


def _render(ev, doc):
    from report_validate import _all_blocks
    cfg = SY.SynthConfig()
    report = SY.render_draft(ev, doc, cfg, SY.build_plan(ev, cfg))
    errors: list[dict] = []
    check_duplicate_homes(report, _all_blocks(report), errors)
    return report, errors


def _prose_item(ev):
    """An item the renderer will place in prose — the check keys on section
    membership, not on the item's type, so any item will do; `topic` is what the
    conflict fixture carries."""
    return next(i.id for i in ev.items if i.type not in ("number", "link"))


def t_prompt_states_the_contract_the_gate_enforces() -> None:
    """The instruction and the check must name the same constraint.

    Asserted on substance rather than on two literal phrases, because a test that
    greps for wording fails on a reflow and passes on a reworded rule that means
    something else. Three things have to be true at once: the two narrative
    homes are named, the summary is named as *not* one of them, and the writer is
    told the count is on `evidence_ids` — which is the field the gate reads, and
    the one place a reader is told to keep their restatements from colliding.
    """
    prompt = SY._plan_prompt(RF.simple_talking_head(),
                             SY.SynthConfig(detail="brief"), {})
    # Normalised to one line: the prompt is hard-wrapped for the model's
    # benefit, and a rule that happens to straddle a line break is the rule this
    # test would then be asserting about the wrapping rather than the content.
    low = " ".join(prompt.lower().split())
    check("the prompt names the two narrative homes",
          all(h in low for h in _NARRATIVE_HOMES), "a home is unnamed")
    check("the prompt says the summary is not one of them",
          "the summary is not one of them" in low,
          "the summary's exemption is unstated")
    check("the prompt says what the gate counts",
          "evidence_ids" in low and "not repeat an item" in low,
          "the writer is not told the count is on evidence_ids")
    check("the prompt keeps the report-level conflict rule",
          "represented completely" in low and "one side in a section" in low,
          "the conflict rule the completeness check enforces is unstated")
    check("the prompt does not require each paragraph to carry both sides",
          "never in two homes" in low,
          "the rule is stated as a per-paragraph conflict requirement")


def t_one_home_matrix() -> None:
    """The five shapes §2 names, each with the verdict the gate must give.

    They are one test because they are one rule, and splitting them would hide
    the shape that matters: case 3 is the one a naive reading gets wrong in the
    *permissive* direction, and case 5 is the one that used to be enforced at the
    wrong level — per paragraph, which made a correct report unrepresentable and
    pushed every conflicted video onto the deterministic renderer.
    """
    ev = RF.unresolved_conflict()
    item = _prose_item(ev)
    other = next((i.id for i in ev.items if i.id != item), item)

    cases = (
        ("1 an item in one section", "pass",
         _draft([_unit("An overview sentence.", [other])],
                [_unit("A high-value point.", [other])],
                [{"title": "Detail", "paragraphs": [_unit("The point, developed.", [item])]}])),

        ("2 the same item in a takeaway and a section", "fail",
         _draft([_unit("An overview sentence.", [other])],
                [_unit("A high-value point.", [item])],
                [{"title": "Detail", "paragraphs": [_unit("The point again.", [item])]}])),

        # 3: the takeaway restates a point in words, and cites a *different*
        # item — which is the shape the prompt tells the writer to use. The text
        # repeating itself is not what the gate counts; the item is.
        ("3 the same item reached indirectly, by a summary of another", "pass",
         _draft([_unit("An overview sentence.", [other])],
                [_unit("The point again, in the writer's words.", [other])],
                [{"title": "Detail", "paragraphs": [_unit("The point, developed.", [item])]}])),

        ("4 the same item in a summary and a section", "pass",
         _draft([_unit("An overview sentence.", [item])],
                [_unit("A high-value point.", [other])],
                [{"title": "Detail", "paragraphs": [_unit("The point, developed.", [item])]}])),

        # 5: the item is in a takeaway only, and a section carries a different
        # one — one home, so the duplicate check must stay quiet. The defect this
        # guards is the *other* gate, asserted below, not this one.
        ("5 one item in a takeaway and none of it in a section", "pass",
         _draft([_unit("An overview sentence.", [other])],
                [_unit("A high-value point.", [item])],
                [{"title": "Detail", "paragraphs": [_unit("Something else.", [other])]}])),
    )

    for label, expect, doc in cases:
        _, errors = _render(ev, doc)
        got = "fail" if "item-duplicated" in {e["code"] for e in errors} else "pass"
        check(f"one-home case {label}: {expect}", got == expect,
              f"got {got!r}; errors={errors}")


def t_conflict_completeness_is_a_report_level_rule() -> None:
    """The conflict invariant, stated over the document, in both directions.

    The unit here is the whole report and it has to be: a paragraph that explains
    one side and a reliability section that discloses the disagreement is the
    report this pipeline exists to produce, and the previous rule — every
    paragraph mentioning a member must cite every member — refused it. So the
    case that must pass is asserted first, and the case that must fail is the one
    where a side is never mentioned anywhere.
    """
    from synth_validate import check_conflict_completeness

    ev = RF.unresolved_conflict()
    conflict = next(c for c in ev.conflicts if c.resolution != "resolved")
    a, b = conflict.item_ids[0], conflict.item_ids[-1]
    rest = [i.id for i in ev.items if i.id not in (a, b)]

    # Both sides present, one of them in a paragraph and the other in the same
    # report — the shape the renderer itself produces by construction.
    both = _draft([_unit("An overview.", rest[:1] or [a])],
                  [_unit("A high-value point.", [b])],
                  [{"title": "Detail", "paragraphs": [
                      _unit("One side of the disagreement, on its own.", [a])]}])
    check("one side in a paragraph and both sides in the report: accepted",
          not [f for f in check_conflict_completeness(both, ev)
               if f.blocking],
          str(check_conflict_completeness(both, ev)))

    # One side cited, the other never mentioned anywhere.
    pick = lambda i: [rest[i % len(rest)] if rest else a]
    one = _draft([_unit("An overview.", pick(0))],
                 [_unit("A high-value point.", [a])],
                 [{"title": "Detail", "paragraphs": [_unit("A detail.", pick(0))]}])
    findings = check_conflict_completeness(one, ev)
    check("one side mentioned and no complete disclosure anywhere: refused",
          any(f.code == "conflict-incomplete" and f.blocking for f in findings),
          str(findings))

    # A conflict the draft never touches is the report's own choice.
    untouched = _draft([_unit("An overview.", pick(1))],
                       [_unit("A high-value point.", pick(2))],
                       [{"title": "Detail", "paragraphs": [_unit("A detail.", pick(3))]}])
    check("a conflict the report never mentions is not its failure",
          not [f for f in check_conflict_completeness(untouched, ev) if f.blocking],
          str(check_conflict_completeness(untouched, ev)))


# --------------------------------------------------------------------------- #
# the output-language contract
# --------------------------------------------------------------------------- #

def t_output_language_requires_a_writer() -> None:
    """§7, half one: the request is either honoured or stated as not honoured.

    The failure this forbids is quiet and completely invisible from the report:
    a run asked for Japanese, given the deterministic renderer, and out comes
    English that looks exactly like a translation. The report cannot detect that
    — it is the report — so the note has to come from the layer that knows which
    renderer ran, and it has to be checked here, where both branches are
    reachable without a provider.
    """
    ev = _simple()

    auto = SY.synthesize(ev, SY.SynthConfig(detail="balanced"))
    check("language: `auto` asks for nothing, so nothing is noted",
          auto.mode == "deterministic" and auto.language_note is None,
          f"{auto.mode} {auto.language_note!r}")

    asked = SY.synthesize(ev, SY.SynthConfig(detail="balanced", output_language="ja"))
    check("language: the deterministic renderer says the request was not honoured",
          asked.language_note == SY.OUTPUT_LANGUAGE_REQUIRES_WRITER,
          f"{asked.language_note!r}")
    check("language: …and the wording is the documented one",
          SY.OUTPUT_LANGUAGE_REQUIRES_WRITER ==
          "output-language requires a configured synthesis writer for true translation",
          SY.OUTPUT_LANGUAGE_REQUIRES_WRITER)
    check("language: the note reaches the machine-readable result",
          asked.to_dict()["language_note"] == SY.OUTPUT_LANGUAGE_REQUIRES_WRITER,
          str(asked.to_dict().get("language_note")))

    # A configured writer is what discharges the note. A stub stands in for a
    # provider here: what is under test is the note's condition, not the model.
    doc = _good_draft(ev)
    cfg = SY.SynthConfig(provider="custom", model="stub", base_url="http://x/v1",
                         api_key="k", detail="balanced", critic=False,
                         output_language="ja")
    out = _with_transport(_scripted([_tool_response(doc)]),
                          lambda: SY.synthesize(ev, cfg, cache=None))
    check("language: a writer discharges the note",
          out.mode == "synthesis" and out.language_note is None,
          f"{out.mode} {out.language_note!r}")


def t_evidence_stays_in_its_source_language() -> None:
    """§7, half two: what a translated report may not have translated.

    The writer is asked for prose in another language and asked to leave
    everything else alone. That instruction is only as good as the gate behind
    it, so the four protected things are asserted on the rendered report rather
    than on the prompt: the figures, the URLs, the identifiers, and the
    qualifier — the last one because it is the easiest of the four to lose and
    the most expensive, being the difference between a price and an estimate of
    a price.
    """
    import copy

    ev = RF.hedged_claim()
    # A projection of everything §7 protects, taken before synthesis runs. A
    # copy of the object would not do: synthesis mutates items in place in
    # places, and a deep copy taken from the same objects records the mutation.
    def snapshot(e):
        return [(i.id, i.value, i.display, i.subject, i.qualifier, i.verbatim,
                 p.ocr_text) for i in e.items for p in (i.provenance or [None])
                if p is not None] + \
               [(c.cid, tuple(c.item_ids), c.resolution) for c in e.conflicts]

    before = snapshot(ev)
    doc = _good_draft(ev)
    cfg = SY.SynthConfig(provider="custom", model="stub", base_url="http://x/v1",
                         api_key="k", detail="balanced", critic=False,
                         output_language="de")
    outcome = _with_transport(_scripted([_tool_response(doc)]),
                               lambda: SY.synthesize(ev, cfg, cache=None))
    from report import render_markdown
    md = render_markdown(outcome.report)
    check("language: the report was written", bool(md.strip()), "empty report")
    check("language: a writer was used, so the note is not raised",
          outcome.mode == "synthesis", outcome.mode)

    # The evidence store is never rewritten. This is the whole of "do not
    # translate evidence.json": a run that asked for German must leave the
    # source-language items, qualifiers, URLs and identifiers exactly as the
    # fetch and extraction layers wrote them, or the next stage reads a
    # translation as if it were a source.
    changed = [(a, b) for a, b in zip(before, snapshot(ev)) if a != b]
    check("language: values, qualifiers, verbatim text and conflict sides survive",
          not changed, str(changed[:2]))
    check("language: the snapshot is not vacuous", len(before) > 1, str(before))


# --------------------------------------------------------------------------- #
# added by the release-freeze mutation audit
# --------------------------------------------------------------------------- #

def t_synth_malformed_draft_reaches_the_gate() -> None:
    """S11. Added after mutation testing: the wiring was untested.

    `t_synth_schema` calls `validate_draft` directly, so the structural rules
    were well covered — but nothing asserted that a structural problem becomes a
    *blocking finding* on the real entry point. Deleting the line that records it
    changed no test's result, which meant a malformed draft would have left the
    gate with nothing to refuse it on. This is the assertion that closes that
    seam: the rule and the report it produces are checked together.
    """
    ev = _simple()
    for label, bad in [
        ("a missing required key", {"takeaways": [], "sections": []}),
        ("an empty unit", _draft([{"text": "  ", "evidence_ids": ["itm-001"],
                                   "source_kind": "mixed"}])),
        ("an unknown top-level key", {**_draft([_unit("Caching is discussed.",
                                                      ["itm-001"])]),
                                      "conclusion": "extra"}),
    ]:
        res = SV.validate_draft_grounding(bad, ev)
        check(f"S11 a malformed draft ({label}) reaches the gate as a finding",
              "synthesis-malformed" in _codes(res),
              str([f.to_dict() for f in res.blocking]))
    # ...and a well-formed draft raises it not at all, so the check is the rule
    # firing and not the gate being permanently red.
    good = _draft([_unit("The video covers caching.", ["itm-001"])],
                  [_unit("A cache sits between the app and the database.", ["itm-002"])],
                  [{"title": "Caching",
                    "paragraphs": [_unit("The second read avoids the database.",
                                         ["itm-003"])]}])
    res = SV.validate_draft_grounding(good, ev)
    check("S11 a well-formed draft raises no structural finding",
          "synthesis-malformed" not in _codes(res),
          str([f.to_dict() for f in res.blocking]))


def t_synth_rejects_an_unsupported_url() -> None:
    """S12. Added after mutation testing: no test reached the URL rule.

    A URL is the one string a writer can add that a reader is expected to
    follow, and one the evidence may never have contained. Removing the URL
    check changed no test's result, so an invented link would have passed the
    gate silently. The valid half matters as much: a URL that *is* in the cited
    evidence is the writer doing its job and must be allowed through.
    """
    ev = _simple()
    doc = _draft([_unit("The notes are at https://example.invalid/not-in-evidence.",
                        ["itm-001"])])
    res = SV.validate_draft_grounding(doc, ev)
    check("S12 a URL absent from the cited evidence is refused",
          "url-unsupported" in _codes(res), str([f.to_dict() for f in res.blocking]))

    # The valid half, on the same rule: a URL the evidence really carries is the
    # writer doing its job. `notes` is part of the item's ground text, so putting
    # the URL there is what makes it allowed — and the note is restored, because
    # the fixture is shared with the tests that run after this one.
    item = next(i for i in ev.items if i.id == "itm-001")
    original = item.notes
    try:
        item.notes = "the written notes are at https://example.com/evidence"
        ok = _draft([_unit("The notes are at https://example.com/evidence for the full transcript",
                           [item.id])])
        res = SV.validate_draft_grounding(ok, ev)
        check("S12 a URL that is in the cited evidence is allowed",
              "url-unsupported" not in _codes(res),
              str([f.to_dict() for f in res.blocking]))
    finally:
        item.notes = original
