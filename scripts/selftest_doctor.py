"""The doctors, the budgets, and the four ways a provider call can go wrong.

Everything here runs against a scripted transport. There is no network and no
credential, which is not a limitation here — it is the reason these run on every
commit. A doctor that has only ever been exercised against a live provider is
exercised on the days a key happens to be valid, and a qualification command
whose own logic is untested cannot be used to qualify anything.

The cases, in the order they matter:

* **Error taxonomy.** `provider unavailable`, `authenticated but out of credits`,
  `model unavailable`, `cannot follow schema`, `malformed JSON` and `valid
  synthesis` are six different answers with six different fixes. The test asserts
  each lands on its own verdict, because the failure mode of collapsing them is
  a user who reads "synthesis failed" and changes the model, and the model was
  never the problem.
* **Refusal, not coercion.** A model that answers in prose where a schema was
  sent is NOT QUALIFIED. The test asserts the payload is not repaired, not
  re-keyed, and not accepted.
* **The VLM path reaches evidence.** A conformant observation from a chart, a
  table, a diagram and a screen has to arrive in `evidence.json` and then in a
  synthesized sentence, or the VLM pass is a cost with no output.
* **The critic's four cases** — clean, changed number, removed hedge, unsupported
  sentence — with the clean one first, because everything after it is meaningless
  if the critic reports a defect in a sound draft.
* **Budgets.** A ceiling that is reported rather than enforced produces a bill
  and a report saying the limit was respected. Only one of those is true.
"""
from __future__ import annotations

import json
import ssl
import urllib.error

import chat_json as CJ
import critic_doctor as CD
import report_fixtures as RF
import synth_doctor as SD
import synthesis as SY
import vlm as VL


#: Bound by `selftest.py` at import, exactly as the other sub-suites are.
check = None


# --------------------------------------------------------------------------- #
# scripted endpoints
# --------------------------------------------------------------------------- #

def _http_error(code: int, body: str) -> urllib.error.HTTPError:
    import io
    return urllib.error.HTTPError("http://x/v1/chat/completions", code, "err", {},
                                  io.BytesIO(body.encode("utf-8")))


def _answer(payload: dict) -> str:
    return json.dumps({"choices": [{"message": {"content": json.dumps(payload)},
                                    "finish_reason": "stop"}]})


def _good_draft(ev) -> dict:
    """A draft that passes every check, so the doctor is measuring the transport."""
    claim = next(i for i in ev.items if i.type == "claim")
    num = next(i for i in ev.items if i.type == "number")
    visual = next((i for i in ev.items if i.type == "visual_finding"), None)
    ids = [claim.id] + ([visual.id] if visual else [])
    return {
        "summary": [{"text": "The video puts a benchmark claim to the viewer and "
                             "qualifies it as the vendor's own reported result.",
                     "evidence_ids": ids, "source_kind": "claim"}],
        "takeaways": [{"text": f"The stated latency figure is {num.display or num.value}.",
                       "evidence_ids": [num.id], "source_kind": "claim"}],
        "sections": [{"title": "What the video claims",
                      "paragraphs": [{"text": "The speaker presents the comparison as the "
                                              "vendor's benchmark rather than an independent "
                                              "measurement.",
                                      "evidence_ids": ids, "source_kind": "claim"}]}],
    }


def _findings(codes: list[dict]) -> dict:
    return {"findings": codes}


def _scripted(script):
    """A transport replaying `script` in order, recording what it was asked."""
    sent: list[dict] = []

    def transport(url, headers, data, timeout):
        body = json.loads(data.decode("utf-8"))
        sent.append(body)
        step = script[min(len(sent) - 1, len(script) - 1)]
        if isinstance(step, Exception):
            raise step
        return step

    transport.sent = sent
    return transport


def _with_transport(transport, fn):
    """Run `fn` with every `SynthesisClient` built on `transport`.

    Copied from `selftest_synthesis` rather than generalised, for the reason that
    file gives: threading a `transport` parameter through `synthesize` would put a
    test seam in the public signature of the pipeline, and a caller who can
    inject a transport can also inject a model that returns whatever they like.
    """
    import synth_model as _sm
    orig = _sm.SynthesisClient

    class _Patched(orig):
        def __init__(self, **kw):
            kw["transport"] = transport
            super().__init__(**kw)

    _sm.SynthesisClient = _Patched
    # `synthesis.py` imported the name directly, so patching the defining module
    # alone leaves the pipeline building the real client. Both have to be patched
    # for the seam to hold, which is exactly why the transport is a constructor
    # argument on the client rather than a module-level function.
    SY.SynthesisClient = _Patched
    try:
        return fn()
    finally:
        _sm.SynthesisClient = orig
        SY.SynthesisClient = orig


def _cfg(**kw) -> SY.SynthConfig:
    base = dict(provider="openai", model="test-model", base_url="http://x/v1",
                api_key="k", detail="brief", output_language="en", critic=False)
    base.update(kw)
    return SY.SynthConfig(**base)


# --------------------------------------------------------------------------- #
# error taxonomy
# --------------------------------------------------------------------------- #

def t_http_error_taxonomy() -> None:
    """Six different failures, six different statuses.

    Asserted on `classify_http` directly because that is the one function every
    provider call funnels through; a test at the doctor level would pass for the
    wrong reason whenever the doctor happened to classify a message correctly.
    """
    cases = (
        ("unreachable provider", 401, '{"error":{"message":"invalid api key"}}',
         "unavailable"),
        ("out of credits (429)", 429,
         '{"error":{"message":"You exceeded your current quota"}}', "out_of_credits"),
        ("out of credits (402)", 402, '{"error":{"message":"Payment required"}}',
         "out_of_credits"),
        ("out of credits (400)", 400,
         '{"error":{"message":"insufficient_quota"}}', "out_of_credits"),
        ("model unavailable", 404, '{"error":{"message":"model not found"}}', "failed"),
        ("schema not supported", 400,
         '{"error":{"message":"Unrecognized request argument: response_format"}}', "failed"),
    )
    for name, code, body, expected in cases:
        err = CJ.classify_http(_http_error(code, body))
        check(f"D1 {name} → {expected}", err.status == expected,
              f"got {err.status!r}: {err.reason}")
    billing = CJ.classify_http(_http_error(429, '{"error":{"message":"quota exceeded"}}'))
    rate = CJ.classify_http(_http_error(429, '{"error":{"message":"slow down"}}'))
    check("D1 billing and rate limiting are not the same status",
          billing.status != rate.status, f"{billing.status} vs {rate.status}")
    check("D1 an exhausted balance is fatal — waiting does not help",
          billing.fatal and not rate.fatal, f"{billing.fatal} / {rate.fatal}")


def t_synthesis_doctor_distinguishes_failures() -> None:
    """Each failure lands on its own step, and none of them is `qualified`."""
    ev = RF.synthesis_qualification()
    scenarios = (
        ("out_of_credits", CJ.ChatError(
            "out_of_credits", "HTTP 429: the account cannot pay", fatal=True),
         "account_paid"),
        ("unauthenticated", CJ.ChatError(
            "unavailable", "HTTP 401: authentication failed", fatal=True), "authenticated"),
        ("model missing", CJ.ChatError(
            "failed", "HTTP 404: model not found", fatal=True), "model_exists"),
        ("schema refused", CJ.ChatError(
            "failed", "HTTP 400: endpoint rejected the structured-output parameter",
            downgrade=True), "model_completes"),
        ("malformed JSON", CJ.ChatError("malformed", "assistant text was not a JSON object"),
         "structured_enforced"),
    )
    for name, err, step in scenarios:
        def boom(url, headers, data, timeout, _err=err):
            raise _err
        code, rep = SD.run_synthesis_doctor(ev, _cfg(), transport=boom)
        check(f"D2 {name}: not qualified", not rep.qualified, str(rep.to_dict()["checks"]))
        check(f"D2 {name}: fails at {step}", rep.checks[step] is not None
              and not rep.checks[step].ok, str(rep.checks[step]))
        check(f"D2 {name}: exactly one call was attempted", rep.extra["calls"]["spent"] == 1,
              str(rep.extra["calls"]))

    def prose(url, headers, data, timeout):
        # A 200 that answers in prose where a schema was sent. The most
        # important case, because it is the one that looks like success.
        return json.dumps({"choices": [{"message": {"content":
            "Certainly! Here is a summary of the video."}, "finish_reason": "stop"}]})
    code, rep = SD.run_synthesis_doctor(ev, _cfg(), transport=prose)
    check("D2 a 200 answering in prose is NOT qualified", not rep.qualified,
          str(rep.to_dict()["checks"]))
    check("D2 ...and the reason names the schema, not the transport",
          "schema" in (rep.error or "").lower(), rep.error)
    check("D2 ...and the prose was not coerced into a draft",
          rep.checks["draft_valid"].ok is False, str(rep.checks["draft_valid"]))

    def good(url, headers, data, timeout):
        return _answer(_good_draft(ev))
    code, rep = SD.run_synthesis_doctor(ev, _cfg(), transport=good)
    check("D2 a conformant draft qualifies", rep.qualified, str(rep.to_dict()["checks"]))
    check("D2 ...with exit 0", code == 0, str(code))
    check("D2 ...and it is still a step-by-step record, not a single boolean",
          len(rep.to_dict()["checks"]) == len(SD.SYNTH_CHECKS), str(len(rep.to_dict()["checks"])))


def t_doctor_does_not_claim_success_on_http_200() -> None:
    """A schema-valid shape carrying an ungrounded sentence is not a pass either.

    The doctor walks ten steps and the ninth is grounding. A provider that
    returns HTTP 200, valid JSON, the right keys, and a sentence nothing supports
    has passed eight of ten, and the report it would have produced would have
    been refused downstream anyway.
    """
    ev = RF.synthesis_qualification()
    draft = _good_draft(ev)
    draft["takeaways"].append({
        "text": "It is said by Anthropic to be the fastest configuration in use today.",
        "evidence_ids": [draft["takeaways"][0]["evidence_ids"][0]],
        "source_kind": "inference"})

    def answer(url, headers, data, timeout):
        return _answer(draft)
    code, rep = SD.run_synthesis_doctor(ev, _cfg(), transport=answer)
    check("D3 an ungrounded draft is not qualified", not rep.qualified,
          str(rep.to_dict()["checks"]))
    check("D3 it is caught by the grounding step, not the schema step",
          not rep.checks["draft_grounded"].ok and rep.checks["structured_enforced"].ok,
          str(rep.checks["draft_grounded"]))

    # The known boundary, asserted so it cannot be forgotten. `_check_content_
    # grounded` checks fabricated *things* — names, products, standards — because
    # a synthesized sentence is entitled to ordinary vocabulary. An unsupported
    # *judgement* in ordinary words is therefore not caught by the blocking gate.
    # That is the critic's job, which is why critic-doctor exists and why case D
    # plants exactly this.
    import synth_validate as SV
    cited = draft["takeaways"][0]["evidence_ids"]

    clean = _good_draft(ev)

    def _with_extra(sentence: str) -> dict:
        # From the *clean* draft, not from `draft` — which already carries the
        # fabricated name, and would smuggle that finding into every case.
        d = json.loads(json.dumps(clean))
        d["takeaways"].append({"text": sentence, "evidence_ids": list(cited),
                               "source_kind": "inference"})
        return d

    v_name = SV.validate_draft_grounding(
        _with_extra("It is said by Anthropic to be the fastest configuration in use today."),
        ev)
    check("D3 a fabricated NAME is caught by the blocking gate", not v_name.valid,
          "the name case slipped through")
    v_judgement = SV.validate_draft_grounding(
        _with_extra("This is widely considered the fastest configuration anyone uses."),
        ev)
    check("D3 an unsupported JUDGEMENT in ordinary words is NOT caught (known gap)",
          v_judgement.valid, str([f.code for f in v_judgement.blocking]))


def t_config_failures_make_no_call() -> None:
    """Exit 3, and a transport that would raise if anything reached it."""
    ev = RF.synthesis_qualification()

    def never(url, headers, data, timeout):
        raise AssertionError("a call was made with no provider configured")
    code, rep = SD.run_synthesis_doctor(ev, SY.SynthConfig(provider="none"), transport=never)
    check("D4 no provider is exit 3", code == 3, str(code))
    check("D4 no call was made", rep.extra["calls"]["spent"] == 0, str(rep.extra["calls"]))
    check("D4 nothing downstream is claimed", not rep.qualified, str(rep.to_dict()["checks"]))

    code, rep = SD.run_synthesis_doctor(ev, _cfg(provider="custom", base_url=None,
                                                 api_key="k"), transport=never)
    check("D4 a provider with no base URL is exit 3", code == 3, str(code))

    code, rep = SD.run_synthesis_doctor(ev, _cfg(provider="openai", api_key=""),
                                        transport=never)
    check("D4 a missing credential is exit 3", code == 3, str(code))


# --------------------------------------------------------------------------- #
# the critic's four cases
# --------------------------------------------------------------------------- #

def t_critique_shape_is_validated() -> None:
    """A critique in the wrong shape is not a critique.

    Found on a live run. The negotiator reports `enforced: true` because the
    schema's top-level `required` key was present, and the model answered
    `{type, evidence_id, description}` where the schema asks for
    `{code, where, message}`. Every finding then scored as `[None] None: None`
    and the doctor reported 0/4 with four malformed rows — a real pair of finds
    turned into a total failure, and a different vocabulary turned into an
    objection.
    """
    import synth_model as SM
    good = {"findings": [{"code": "number-error", "where": "takeaways[0]",
                          "message": "90 ms does not match 40 ms"}]}
    check("D5a a conforming critique validates", not SM.validate_critique(good),
          str(SM.validate_critique(good)))
    check("D5a an empty findings array is valid",
          not SM.validate_critique({"findings": []}),
          str(SM.validate_critique({"findings": []})))

    for doc, why in (
        ({"findings": [{"type": "number-error", "evidence_id": "itm-003",
                        "description": "wrong figure"}]},
         "the vocabulary the model actually used"),
        ({"findings": [{"code": "number-error", "where": "t"}]}, "a finding missing a field"),
        ({"findings": [{"code": "", "where": "t", "message": "m"}]}, "an empty field"),
        ({"findings": "none"}, "findings that are not an array"),
        ({}, "no findings key at all"),
        ({"findings": [], "surprise": 1}, "an undeclared top-level key"),
    ):
        problems = SM.validate_critique(doc)
        check(f"D5a rejected: {why}", bool(problems), "accepted")

    # And the pipeline treats an unreadable critique as silence rather than as a
    # block, which is the whole reason this is a warning-shaped failure.
    ev = RF.synthesis_qualification()
    cfg = _cfg(critic=True)
    bad = _answer({"findings": [{"type": "x", "description": "y"}]})
    seen = {}
    out = _with_transport(_scripted([_answer(CD._base_draft(ev)), bad]), lambda: SY.synthesize(
        ev, cfg, cache=_DictCache(), budget=SY.Budget(2), critic_budget=SY.Budget(1)))
    seen["mode"] = out.mode
    check("D5a a malformed critique does not block a grounded draft",
          out.mode == "synthesis", f"{out.mode}: {out.fallback_reason}")
    check("D5a ...and it is not passed on as advice", out.critique is None,
          str(out.critique)[:120])


def t_critic_cases_have_known_answers() -> None:
    """The cases are real before any model is asked about them.

    Asserted against the *validator*, not against a model: the clean case must
    pass every deterministic check, and each mutation must be the specific
    defect its name says. A baseline that is already defective would make every
    verdict after it a measurement of the baseline.
    """
    import report as REP
    import synth_model as SM
    import synth_validate as SV
    from report_validate import validate_report
    ev = RF.synthesis_qualification()
    cfg = _cfg()

    clean, _ = (CD._base_draft(ev), "")
    check("D5 the clean case is schema-valid", not SM.validate_draft(clean),
          str(SM.validate_draft(clean)))
    check("D5 the clean case is grounded", SV.validate_draft_grounding(clean, ev).valid,
          str([f.code for f in SV.validate_draft_grounding(clean, ev).blocking]))
    rep = SY.render_draft(ev, clean, cfg, SY.build_plan(ev, cfg))
    res = validate_report(rep, ev, REP.render_markdown(rep), authored_by="synthesis")
    check("D5 the clean case passes the report validator", res.valid,
          str([e["code"] for e in res.blocking_errors[:3]]))

    changed, note = CD._mutate_changed_number(CD._base_draft(ev), ev)
    codes = {f.code for f in SV.validate_draft_grounding(changed, ev).blocking}
    check("D5 case B plants a number error", "number-unsupported" in codes,
          f"{codes} — {note}")

    hedged, note = CD._mutate_removed_qualifier(CD._base_draft(ev), ev)
    codes = {f.code for f in SV.validate_draft_grounding(hedged, ev).blocking}
    check("D5 case C plants a dropped hedge", "hedge-dropped" in codes,
          f"{codes} — {note}")

    unsupported, note = CD._mutate_unsupported_sentence(CD._base_draft(ev), ev)
    ids = {i for _w, u in SM.units(unsupported) for i in u["evidence_ids"]}
    check("D5 case D adds exactly one unit, and it is the invented one",
          sum(1 for _ in SM.units(unsupported)) == sum(1 for _ in SM.units(clean)) + 1
          and "widely considered" in unsupported["takeaways"][-1]["text"], note)
    check("D5 case D is the one the deterministic checks do NOT catch",
          SV.validate_draft_grounding(unsupported, ev).valid,
          "if the validator catches it, case D is re-testing the validator")


def t_critic_doctor_scores_each_case() -> None:
    """The four verdicts, driven by a scripted critic that has known answers."""
    ev = RF.synthesis_qualification()
    # A, B, C, D: the first three return nothing, the last returns a finding.
    # That makes exactly one case fail, which is what proves the scoring is
    # per-case rather than a single "did it return JSON" check.
    answers = [
        _answer(_findings([])),                       # A clean → no findings
        _answer(_findings([{"code": "number-error", "where": "takeaways[0]",
                            "message": "90 ms does not match 40 ms"}])),
        _answer(_findings([{"code": "missing-qualifier", "where": "summary[0]",
                            "message": "the hedge is gone"}])),
        _answer(_findings([])),                       # D missed on purpose
    ]
    code, rep = CD.run_critic_doctor(ev, _cfg(), transport=_scripted(answers),
                                     budget=SY.Budget(4))
    cases = {c["case"]: c for c in rep.extra["cases"]}
    check("D6 four cases ran", len(cases) == 4, str(sorted(cases)))
    check("D6 four calls were spent", rep.extra["calls"]["spent"] == 4,
          str(rep.extra["calls"]))
    check("D6 every case records that it was answered",
          all(c["answered"] for c in cases.values()), str(cases))
    check("D6 the clean case passes when the critic is quiet", cases["clean"]["passed"],
          str(cases["clean"]))
    check("D6 a found number error passes", cases["changed_number"]["passed"],
          str(cases["changed_number"]))
    check("D6 a found missing qualifier passes", cases["removed_qualifier"]["passed"],
          str(cases["removed_qualifier"]))
    check("D6 a missed unsupported sentence fails the run", not cases["unsupported_sentence"]["passed"],
          str(cases["unsupported_sentence"]))
    check("D6 the run exits non-zero when any case is wrong", code == 1, str(code))
    check("D6 the verdict is not qualified either", not rep.qualified, str(rep.qualified))

    # A critic that never answers is not a critic that found nothing. Four
    # billing refusals scored as "observed: none" would be reported as a quiet
    # critic plus three misses — five claims about the model, none of them true.
    class _Dead:
        def __init__(self, **kw):
            kw["transport"] = lambda *a: (_ for _ in ()).throw(
                __import__("urllib.error", fromlist=["x"]).URLError("no route"))
    import synth_model as _sm
    _orig = _sm.SynthesisClient
    _sm.SynthesisClient = _Dead
    try:
        code, rep = CD.run_critic_doctor(ev, _cfg(), budget=SY.Budget(4))
    finally:
        _sm.SynthesisClient = _orig
    dead = {c["case"]: c for c in rep.extra["cases"]}
    check("D6 a critic that never answered fails every case, clean included",
          not any(dead[k]["passed"] for k in dead), str(dead))
    check("D6 ...and is recorded as 'no answer', not as 'none'",
          all(dead[k]["observed"] == "no answer" for k in dead), str(dead))
    check("D6 ...and the transport step fails too", not rep.qualified, str(rep.qualified))

    # A critic that reports a defect in a sound draft is the opposite failure and
    # must be caught by exactly the same machinery.
    noisy = [_answer(_findings([{"code": "unsupported-claim", "where": "summary[0]",
                                 "message": "this is fine, but here is a complaint"}]))] * 4
    code, rep = CD.run_critic_doctor(ev, _cfg(), transport=_scripted(noisy),
                                     budget=SY.Budget(4))
    cases = {c["case"]: c for c in rep.extra["cases"]}
    check("D6 a false positive on the clean case fails the run",
          not cases["clean"]["passed"] and code == 1, f"{code} {cases['clean']}")


def t_critic_budget_never_exceeded() -> None:
    """A lowered budget skips cases, and skipped cases are not passes."""
    ev = RF.synthesis_qualification()
    answers = [_answer(_findings([]))] * 4
    transport = _scripted(answers)
    code, rep = CD.run_critic_doctor(ev, _cfg(), transport=transport, budget=SY.Budget(2))
    cases = {c["case"]: c for c in rep.extra["cases"]}
    check("D7 the budget is not exceeded", rep.extra["calls"]["spent"] == 2,
          str(rep.extra["calls"]))
    check("D7 the transport saw exactly two calls", len(transport.sent) == 2,
          str(len(transport.sent)))
    check("D7 the un-run cases say so", not cases["removed_qualifier"]["ran"]
          and "not run" in cases["removed_qualifier"]["note"],
          str(cases["removed_qualifier"]))
    check("D7 an incomplete run exits non-zero", code == 1, str(code))
    check("D7 ...and is not reported as qualified", not rep.qualified, str(rep.qualified))


# --------------------------------------------------------------------------- #
# budgets in the synthesis path
# --------------------------------------------------------------------------- #

def t_budget_is_enforced_not_reported() -> None:
    ev = RF.synthesis_qualification()
    answers = [_answer(_good_draft(ev))]
    transport = _scripted(answers)

    out = _with_transport(transport, lambda: SY.synthesize(
        ev, _cfg(critic=False), cache=_DictCache(),
        budget=SY.Budget(1), critic_budget=SY.Budget(0)))
    check("D8 one allowed call produced a synthesis report", out.mode == "synthesis",
          f"{out.mode}: {out.fallback_reason}")
    check("D8 ...and exactly one call was made", len(transport.sent) == 1,
          str(len(transport.sent)))

    out = _with_transport(transport, lambda: SY.synthesize(
        ev, _cfg(critic=False), cache=_DictCache(),
        budget=SY.Budget(0), critic_budget=SY.Budget(0)))
    check("D8 a zero budget asks the model nothing", out.mode == "deterministic",
          out.mode)
    check("D8 ...and says why rather than degrading silently",
          "budget" in (out.fallback_reason or "").lower(), out.fallback_reason)
    check("D8 ...and the endpoint was never contacted", len(transport.sent) == 1,
          str(len(transport.sent)))


def t_failed_response_is_not_cached_as_success() -> None:
    """A refusal must not leave a draft behind for the next run to find.

    The cache is keyed on the evidence, so a bad answer stored under that key is
    served to every later run of the same video — including one with working
    credentials. `synthesize` only writes on success, and this asserts it by
    checking both halves: a good draft *is* written (so the check is not passing
    because nothing is ever cached) and a failing run writes nothing.
    """
    ev = RF.synthesis_qualification()

    store = _DictCache()
    transport = _scripted([_answer(CD._base_draft(ev))])
    out = _with_transport(transport, lambda: SY.synthesize(
        ev, _cfg(critic=False), cache=store,
        budget=SY.Budget(1), critic_budget=SY.Budget(0)))
    check("D9 a good draft is cached", out.mode == "synthesis",
          f"{out.mode}: {out.fallback_reason}")
    check("D9 ...and the cache holds it", bool(store.puts), str(store.puts))

    store2 = _DictCache()
    out2 = _with_transport(_failing_transport(), lambda: SY.synthesize(
        ev, _cfg(critic=False), cache=store2,
        budget=SY.Budget(1), critic_budget=SY.Budget(0)))
    check("D9 a failing run falls back", out2.mode == "deterministic", out2.mode)
    check("D9 ...and writes nothing to the cache", not store2.puts, str(store2.puts))
    check("D9 ...so a later run cannot be served the failure",
          store2.get(store2.last_key) is None, str(sorted(store2.puts)))


class _DictCache:
    """The smallest object `synthesize` treats as a cache."""

    def __init__(self):
        self.store: dict = {}
        self.puts: list = []
        self.last_key = None

    def get(self, key):
        self.last_key = key
        return self.store.get(key)

    def put(self, key, value):
        self.last_key = key
        self.puts.append(key)
        self.store[key] = value


def _failing_transport():
    def transport(url, headers, data, timeout):
        raise urllib.error.URLError("connection refused")
    return transport


# --------------------------------------------------------------------------- #
# the VLM path reaches the evidence
# --------------------------------------------------------------------------- #

def t_vlm_observation_reaches_evidence_and_report() -> None:
    """A conformant reading has to arrive, or the VLM pass is a cost with no output.

    Driven by a fake provider rather than a real one, because the property under
    test is the *plumbing* — the observation is recorded, becomes an item, and
    reaches a synthesized sentence — and that is identical whatever produced the
    pixels.
    """
    import semantic as SEM
    import vlm as V
    ev = RF.synthesis_qualification()
    frame = ev.visual[0]
    obs = V.VisionObservation.from_dict({
        "content_type": "table",
        "summary": "A two-row comparison table of p99 latency.",
        "confidence": 0.65,
        "numbers": [{"value": "40", "unit": "ms", "context": "p99 latency",
                     "row": "vector database", "column": "p99"}],
    }, frame.frame_id)
    check("D10 a conformant observation re-parses", obs.status == "ok", str(obs.status))

    doc = {"frames": [{"frame_id": frame.frame_id, "ocr_text": frame.ocr_text}]}
    merged = V.merge_into_visual(doc, {"frames": [{"frame_id": frame.frame_id,
                                                   **obs.to_dict()}],
                                       "summary": obs.summary})
    check("D10 the observation merges into the visual artifact",
          (merged["frames"][0].get("vlm") or {}).get("summary") == obs.summary,
          str(merged["frames"][0].get("vlm"))[:120])

    ev.visual[0].vlm_observation = obs.to_dict()
    items = SEM._vlm_items(ev.visual[0], obs.to_dict(), _one_ctx(ev), SEM.SemanticConfig())
    nums = [i for i in items[0] if i.type == "number"]
    check("D10 a figure in the observation becomes a number item", bool(nums),
          str([(i.type, i.value) for i in items[0]]))
    check("D10 ...carrying the role the frame's content type implies",
          bool(nums) and nums[0].semantic_role in ("metric", "count", "benchmark"),
          str([i.semantic_role for i in nums]))
    chart = dict(obs.to_dict(), content_type="chart")
    chart_items = SEM._vlm_items(ev.visual[0], chart, _one_ctx(ev), SEM.SemanticConfig())
    chart_nums = [i for i in chart_items[0] if i.type == "number"]
    check("D10 a figure read off a CHART is a benchmark, whatever the label says",
          bool(chart_nums) and chart_nums[0].semantic_role == "benchmark",
          str([i.semantic_role for i in chart_nums]))
    check("D10 ...and citing the frame", bool(nums) and any(
        p.source == "vlm" for p in nums[0].provenance), str(nums[0].provenance if nums else []))

    # And it must survive into what the writer is allowed to see.
    ev.items = list(ev.items) + [i for i in items[0] if i.type == "number"]
    for i in ev.items[-1:]:
        i.id = "itm-vlm-1"
    view = SY._evidence_view(ev)
    check("D10 the writer's evidence view marks it as visual",
          any(e.get("visual") for e in view if e["id"] == "itm-vlm-1"),
          str([e for e in view if e["id"] == "itm-vlm-1"]))


def _one_ctx(ev):
    import semantic as SEM
    ctx = SEM.SegmentContext(segment=ev.segments[0], snippets=ev.snippets,
                             frames=ev.visual, speech=ev.snippets[0].text,
                             speech_norm=ev.snippets[0].text.lower())
    return ctx


def t_non_conforming_vlm_output_is_rejected() -> None:
    """The schema is not loosened to make a provider look qualified.

    A payload with a field the schema does not define is refused at the parse,
    and the frame records the refusal rather than an empty observation that
    reads as "nothing to see here".
    """
    import vlm as V
    ev = RF.synthesis_qualification()
    frame_id = ev.visual[0].frame_id

    # The confidence ceiling is applied when the reading is turned into a record,
    # not when it is parsed. Asserted at the layer that owns the guarantee: a
    # model that returns 1.0 must not produce a document carrying 1.0, because a
    # description at full confidence reads as a measurement.
    obs = V.VisionObservation.from_dict(
        {"content_type": "table", "summary": "x", "confidence": 0.99}, frame_id)
    check("D11 a model confidence above the ceiling is clamped, not carried",
          obs.confidence is not None and obs.confidence <= V.MAX_MODEL_CONFIDENCE,
          str(obs.confidence))
    check("D11 ...and the cap is the documented one",
          obs.confidence == V.MAX_MODEL_CONFIDENCE, str(obs.confidence))

    # `from_dict` fills optional fields rather than refusing them, because an
    # older artifact written by an earlier schema version is still readable. The
    # refusal it does make is for a payload that is not an observation at all,
    # and that refusal has to be visible as a status rather than an exception —
    # an exception here takes down the whole pass for one bad frame.
    for payload, why in ((None, "nothing at all"),
                         ("a string, not an object", "a non-object payload")):
        try:
            bad = V.VisionObservation.from_dict(payload, frame_id)
            refused = bad.status != "ok"
        except Exception:
            refused = False
        check(f"D11 {why} is refused as a status, not a crash", refused, "accepted")

    # And the schema itself is strict, so a compliant endpoint rejects the extra
    # field before it ever reaches the parser. Asserted on the schema rather than
    # on a round trip, because the round trip is not where that guarantee lives.
    schema_props = V.OBSERVATION_SCHEMA["properties"]
    check("D11 the observation schema forbids undeclared fields",
          V.OBSERVATION_SCHEMA.get("additionalProperties") is False,
          str(V.OBSERVATION_SCHEMA.get("additionalProperties")))
    check("D11 ...and covers every required key",
          set(V.OBSERVATION_SCHEMA.get("required", ())) <= set(schema_props),
          str(sorted(set(V.OBSERVATION_SCHEMA.get("required", ())) - set(schema_props))))


# --------------------------------------------------------------------------- #
# the release check: the four states, the budget, and the matrix
# --------------------------------------------------------------------------- #

def _report(provider="p", model="m", *, qualified=True, answered=True,
            stage="vision", error=None, checks=("reachable",)):
    """A `DoctorReport` shaped like a doctor's, built here rather than by one.

    The point of the fixture is the two axes that matter and that no real
    provider gives cheaply: whether the chain held, and whether an answer ever
    came back. A real run produces one combination or the other by accident; the
    four states are only testable if both are settable independently.
    """
    import vlm_doctor as VD
    r = VD.DoctorReport({"provider": provider, "model": model}, checks)
    for name in checks:
        r.set(name, qualified, "ok" if qualified else "broke here")
    if not qualified:
        r.stage = stage
    r.error = error
    r.latency_s = 0.01 if answered else None
    if answered:
        r.observation = {"content_type": "table", "summary": "s"}
    return r


def t_release_check_four_states() -> None:
    """PASS, FAIL, UNTESTED and SKIPPED, each produced by the real classifier.

    The two middle ones are the whole reason this module exists and they are
    adjacent: both mean "not qualified", and only one of them means "we tried".
    Asserted here as four independent constructions so the rule is pinned from
    both sides — a classifier that returned UNTESTED for everything would pass a
    suite that only ever built failures.
    """
    import release_check as RC

    check("R1 a chain that held and answered is PASS",
          RC.classify(_report(qualified=True, answered=True)) == RC.PASS)
    check("R1 a chain that broke after an answer is FAIL",
          RC.classify(_report(qualified=False, answered=True)) == RC.FAIL)
    check("R1 a chain that never got an answer is UNTESTED, not FAIL",
          RC.classify(_report(qualified=False, answered=False)) == RC.UNTESTED,
          "an unreached provider was recorded as a broken one")
    check("R1 a step that was not requested is SKIPPED",
          RC.classify(_report(), requested=False) == RC.SKIPPED)
    check("R1 a missing report is SKIPPED rather than a silent pass",
          RC.classify(None) == RC.SKIPPED)
    check("R1 a failure recorded with no error text is still FAIL, not UNTESTED",
          RC.classify(_report(qualified=False, answered=True, error=None)) == RC.FAIL)


def t_release_check_matrix() -> None:
    """The matrix, and the status rule, in the two directions that matter.

    A row's `status` is the sentence a maintainer reads, so the two mistakes are
    worth naming: `QUALIFIED` for a provider that was never exercised, and
    `UNQUALIFIED` for one whose optional model layer failed while the report was
    still produced. The first overclaims; the second underclaims and sends
    someone looking for a pipeline bug that is not there.
    """
    import release_check as RC

    matrix = RC.qualification_matrix([
        RC.Step("vlm", RC.PASS, provider="openrouter", model="m",
                checks=[{"check": "vision_capable", "ok": True},
                        {"check": "schema_valid", "ok": True}]),
        RC.Step("synthesis", RC.PASS, provider="openrouter", model="m"),
        RC.Step("critic", RC.PASS, provider="openrouter", model="m"),
    ])
    check("R2 the matrix has one row per provider and model", len(matrix) == 1, str(matrix))
    check("R2 an exercised provider is QUALIFIED", matrix[0]["status"] == "QUALIFIED",
          str(matrix[0]))

    matrix = RC.qualification_matrix([
        RC.Step("vlm", RC.UNTESTED, provider="openrouter", model="m",
                checks=[{"check": "vision_capable", "ok": False}]),
        RC.Step("synthesis", RC.UNTESTED, provider="openrouter", model="m"),
        RC.Step("critic", RC.UNTESTED, provider="openrouter", model="m"),
    ])
    check("R2 an unexercised provider is CONDITIONAL, never QUALIFIED",
          matrix[0]["status"] == "CONDITIONAL", str(matrix[0]))
    check("R2 an unreached provider is not FAIL in a capability column",
          matrix[0]["vision"] == RC.UNTESTED, str(matrix[0]))

    matrix = RC.qualification_matrix([
        RC.Step("vlm", RC.FAIL, provider="openrouter", model="m",
                checks=[{"check": "vision_capable", "ok": False}]),
        RC.Step("synthesis", RC.PASS, provider="openrouter", model="m"),
        RC.Step("critic", RC.PASS, provider="openrouter", model="m"),
    ])
    check("R2 a failed optional layer beside a working one is DEGRADED",
          matrix[0]["status"] == "DEGRADED", str(matrix[0]))
    check("R2 …and the working capability still reads PASS",
          matrix[0]["synthesis"] == RC.PASS, str(matrix[0]))

    matrix = RC.qualification_matrix([
        RC.Step("vlm", RC.FAIL, provider="openrouter", model="m",
                checks=[{"check": "vision_capable", "ok": False}]),
        RC.Step("synthesis", RC.SKIPPED, provider="openrouter", model="m"),
        RC.Step("critic", RC.SKIPPED, provider="openrouter", model="m"),
    ])
    check("R2 a failure with nothing exercised beside it is UNQUALIFIED",
          matrix[0]["status"] == "UNQUALIFIED", str(matrix[0]))


def t_release_check_budget_is_a_ceiling() -> None:
    """The ceilings bound the run, and the report says what was spent.

    Reported spend rather than reported plan: a summary that restates the limits
    it was given is not a measurement, and a run that quietly exceeds a ceiling
    is worse than one that was given no ceiling at all.
    """
    import release_check as RC

    spent_frames: list[int] = []

    def fake_run(cfg, frame=None):
        spent_frames.append(cfg.max_frames)
        return 0, _report(qualified=True, answered=True)

    vlm_cfg = VL.VLMConfig.from_env(provider="fake", model="m", max_frames=7)
    syn_cfg = SY.SynthConfig(provider="p", model="m", base_url="http://x/v1", api_key="k")
    orig_vlm, orig_syn, orig_cri = RC.VD.run_doctor, RC.SD.run_synthesis_doctor, RC.CD.run_critic_doctor
    RC.VD.run_doctor = fake_run
    RC.SD.run_synthesis_doctor = lambda *a, **k: (0, _report(qualified=True, answered=True))
    RC.CD.run_critic_doctor = lambda *a, **k: (0, _report(qualified=True, answered=True))
    try:
        result = RC.run_release_check(vlm_cfg=vlm_cfg, synth_cfg=syn_cfg,
                                      max_vlm_frames=2, max_synthesis_calls=1,
                                      max_critic_calls=1)
    finally:
        RC.VD.run_doctor, RC.SD.run_synthesis_doctor, RC.CD.run_critic_doctor = (
            orig_vlm, orig_syn, orig_cri)

    check("R3 the frame ceiling is the one passed, not the one in the config",
          result["budget"]["max_vlm_frames"] == 2, str(result["budget"]))
    check("R3 the reported spend is inside the ceiling",
          result["spend"]["frames"] <= result["budget"]["max_vlm_frames"],
          str(result["spend"]))
    check("R3 a step that spent nothing is reported as having spent nothing",
          result["spend"]["calls"] <= result["budget"]["max_synthesis_calls"]
          + result["budget"]["max_critic_calls"], str(result["spend"]))
    check("R3 the ceilings are all recorded", set(result["budget"]) == {
        "max_vlm_frames", "max_synthesis_calls", "max_critic_calls",
        "max_revision_calls"}, str(result["budget"]))
    check("R3 the architecture versions travel with the result",
          {"pipeline_version", "schema_version", "extraction_version",
           "synth_prompt_version"} <= set(result["architecture"]), str(result["architecture"]))
    check("R3 a negative ceiling is refused rather than clamped",
          _raises(lambda: RC.run_release_check(
              vlm_cfg=vlm_cfg, synth_cfg=syn_cfg, max_revision_calls=-1)),
          "a ceiling below zero was accepted")


def _raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    except Exception:  # noqa: BLE001 — a different failure is still a failure
        return False
    return False


def t_release_check_never_prints_a_credential() -> None:
    """No key, in the rendered text or in the machine-readable result.

    Checked on a run that was *given* one, because a result that cannot contain a
    secret because none was supplied proves nothing about the code that formats
    it. The key is passed through the config the way a real run would carry it.
    """
    import release_check as RC

    syn_cfg = SY.SynthConfig(provider="openrouter", model="m",
                            base_url="http://x/v1", api_key="sk-do-not-print-me")
    result = RC.run_release_check(
        vlm_cfg=VL.VLMConfig.from_env(provider="none"),
        synth_cfg=syn_cfg, max_synthesis_calls=0, max_critic_calls=0)
    payload = json.dumps(result, default=str) + RC.render(result)
    check("R4 the configured key appears nowhere in the release result",
          "sk-do-not-print-me" not in payload, "a credential reached the output")


def t_release_check_without_providers_calls_nothing() -> None:
    """The state a maintainer actually gets when there is no credit.

    Every step is SKIPPED or UNTESTED, nothing is called, and the overall verdict
    is CONDITIONAL. This is the run that must not be read as a failure of the
    code, and it is the run that most easily could be: a harness that reported
    "not qualified" here would be indistinguishable from one reporting a broken
    provider, and the first thing anyone would do is go and debug the pipeline.
    """
    import release_check as RC

    result = RC.run_release_check(
        vlm_cfg=VL.VLMConfig.from_env(provider="none"),
        synth_cfg=SY.SynthConfig(), max_synthesis_calls=1, max_critic_calls=1)
    states = {s["step"]: s["state"] for s in result["steps"]}
    check("R5 every step is reported, not omitted", set(states) == set(RC.STEPS), str(states))
    check("R5 nothing was called", result["spend"]["calls"] == 0, str(result["spend"]))
    check("R5 no provider means SKIPPED, not FAIL",
          all(v in (RC.SKIPPED, RC.UNTESTED) for v in states.values()), str(states))
    check("R5 the overall verdict is CONDITIONAL", result["status"] == "CONDITIONAL",
          result["status"])
    check("R5 the result says it is not production-ready",
          "CONDITIONAL" in RC.render(result), "the footer lost its verdict")


def t_vlm_doctor_survives_an_unclassified_exception() -> None:
    """An exception the provider did not classify is a verdict, not a traceback.

    The provider turns the *expected* egress failures — refused credential,
    missing model, rate limit — into an observation carrying a reason, and the
    doctor walks that. Everything outside that vocabulary used to escape as a
    traceback, and `release_check` calls this function in-process, so one of
    them took the whole five-step matrix down and the reader got a stack where
    the answer should have been.

    Three things are asserted, and the third is the one that is easy to get
    wrong. The unclassified escape is a *verdict*; it is `UNTESTED` rather than
    `FAIL`, because nothing was disproven and nothing was shown; and the two
    expected classifications still classify the same way they did, which is
    asserted in the same run so a fix that made everything `FAILED` would fail
    here rather than pass.
    """
    import release_check as RC
    import vlm_doctor as VD

    class _Escapes:
        available, structured_mode = True, None

        def analyze(self, *a, **kw):
            raise ssl.SSLError("handshake failure")

    class _Refused:
        """The classification the provider already does, unchanged."""
        available, structured_mode = True, None

        def analyze(self, path, stem, ctx):
            return VL.VisionObservation(frame_id=stem, status="unavailable",
                                        notes="HTTP 401 invalid api key",
                                        error="HTTP 401")

    cfg = VL.VLMConfig(provider="custom", model="m", base_url="https://x/v1", api_key="k")
    orig = VD.provider_from_config
    try:
        VD.provider_from_config = lambda _cfg: _Escapes()
        code, rep = VD.run_doctor(cfg)
    finally:
        VD.provider_from_config = orig

    check("D12 an unclassified provider exception does not escape the doctor",
          isinstance(rep, VD.DoctorReport) and code == 3, str(code))
    checks = {c["check"]: c for c in rep.to_dict()["checks"]}
    check("D12 ...it is recorded as an unreachable endpoint, with the exception named",
          checks["reachable"]["ok"] is False
          and "SSLError" in checks["reachable"]["detail"], str(checks["reachable"]))
    check("D12 ...the steps after the break are unreachable, not passed",
          checks["vision_capable"]["ok"] is False
          and checks["vision_capable"]["detail"] == "unreachable"
          and checks["test_observation_valid"]["detail"] == "unreachable",
          str(checks))
    check("D12 ...and the run is not qualified", rep.qualified is False, str(rep.qualified))
    check("D12 ...it is UNTESTED, not FAILED: no answer came back",
          RC.classify(rep) == RC.UNTESTED,
          "an infrastructure exception was filed as a broken provider")

    # The distinction the fix had to preserve: the provider's own classification
    # still lands where it always did, and is still told apart from the escape.
    try:
        VD.provider_from_config = lambda _cfg: _Refused()
        code, ref = VD.run_doctor(cfg)
    finally:
        VD.provider_from_config = orig
    ref_checks = {c["check"]: c for c in ref.to_dict()["checks"]}
    check("D12 a refused credential still breaks at authentication, not at transport",
          ref_checks["reachable"]["ok"] is True
          and ref_checks["authenticated"]["ok"] is False
          and ref.stage == "auth", f"{ref.stage} {ref_checks['authenticated']}")
    # Both land on UNTESTED at the release level, and that is not a collapse:
    # a refused credential is a configuration nobody gave the run, and the
    # module's own rule is that it must never be filed as a broken provider.
    # What tells them apart is the chain — the stage, the detail, the exception
    # named — which is where a reader looks to fix either one.
    check("D12 ...and a refused credential is still UNTESTED, as it always was",
          RC.classify(ref) == RC.UNTESTED, RC.classify(ref))
    check("D12 ...with the two told apart by the chain they left behind",
          ref.error != rep.error, f"{ref.error!r} vs {rep.error!r}")

    # And the call that motivated it: the same escape, reached the way
    # `release-check` reaches it, must not take the matrix with it.
    try:
        VD.provider_from_config = lambda _cfg: _Escapes()
        result = RC.run_release_check(
            vlm_cfg=cfg, synth_cfg=SY.SynthConfig(),
            max_synthesis_calls=0, max_critic_calls=0)
    finally:
        VD.provider_from_config = orig
    states = {s["step"]: s["state"] for s in result["steps"]}
    check("D12 release-check survives the doctor raising, in-process",
          states["vlm"] in (RC.UNTESTED, RC.FAIL), str(states))
    check("D12 ...every step is still reported", set(states) == set(RC.STEPS), str(states))
    check("D12 ...and the verdict stays CONDITIONAL rather than UNQUALIFIED",
          result["status"] == "CONDITIONAL", result["status"])


def t_max_vlm_frames_reaches_the_doctor() -> None:
    """The flag controls the cap, up to the documented maximum, and no further.

    `--max-vlm-frames 20` was clamped to a hard-coded 2 and returned a
    CONDITIONAL run, which reads as a provider that cannot be trusted with more
    than two frames rather than as a flag that had been disconnected from its
    own argument. The clamp is asserted gone in both directions — a request
    above 2 reaches the doctor, and a request above the maximum does not.

    The propagation is checked at the only place it can be: the config the
    doctor is handed, because that is where `vlm.py` reads the cap from. A test
    that only read the reported budget would have passed against the defect.
    """
    import release_check as RC

    check("R6 the documented maximum is the VLM pass's own cap, not a second number",
          RC.MAX_VLM_FRAMES_CEILING == VL.DEFAULT_MAX_VLM_FRAMES,
          f"{RC.MAX_VLM_FRAMES_CEILING} vs {VL.DEFAULT_MAX_VLM_FRAMES}")

    seen: list[int] = []

    def fake_run(cfg, frame=None):
        seen.append(cfg.max_frames)
        return 0, _report(qualified=True, answered=True)

    vlm_cfg = VL.VLMConfig.from_env(provider="fake", model="m", max_frames=2)
    syn_cfg = SY.SynthConfig()
    orig_vlm = RC.VD.run_doctor

    def _run(requested: int) -> dict:
        seen.clear()
        RC.VD.run_doctor = fake_run
        try:
            return RC.run_release_check(vlm_cfg=vlm_cfg, synth_cfg=syn_cfg,
                                        max_vlm_frames=requested,
                                        max_synthesis_calls=0, max_critic_calls=0)
        finally:
            RC.VD.run_doctor = orig_vlm

    r = _run(7)
    check("R6 a requested cap above 2 reaches the doctor",
          seen and set(seen) == {7}, f"the doctor was handed {seen}, not 7")
    check("R6 ...and is the number the result reports", r["budget"]["max_vlm_frames"] == 7,
          str(r["budget"]))

    _run(2)
    check("R6 the default of 2 still means 2", set(seen) == {2}, str(seen))

    r = _run(RC.MAX_VLM_FRAMES_CEILING)
    check("R6 the documented maximum is honoured exactly",
          set(seen) == {RC.MAX_VLM_FRAMES_CEILING}, str(seen))

    r = _run(20)
    check("R6 a request above the maximum is reduced to it, not ignored",
          set(seen) == {RC.MAX_VLM_FRAMES_CEILING}, str(seen))
    check("R6 ...and the reduction is visible in the reported budget",
          r["budget"]["max_vlm_frames"] == RC.MAX_VLM_FRAMES_CEILING, str(r["budget"]))
    check("R6 ...every doctor call in the run is held to it, not just the first",
          len(seen) >= 2 and all(v == RC.MAX_VLM_FRAMES_CEILING for v in seen), str(seen))

    check("R6 a request for no frames resolves to the one the doctor sends",
          RC.resolve_vlm_frame_ceiling(0) == 1,
          str(RC.resolve_vlm_frame_ceiling(0)))
    check("R6 a negative ceiling is refused rather than clamped",
          _raises(lambda: RC.resolve_vlm_frame_ceiling(-1)),
          "a ceiling below zero was accepted")
    check("R6 ...and refused on the run too",
          _raises(lambda: _run(-1)), "a ceiling below zero was accepted")

    # One ceiling for the run, not one per step. Two steps send frames; each
    # was handed the whole ceiling, so the total reported was up to twice the
    # number printed beside it, and every individual line stayed true.
    check("R6 the doctor charges the one frame it actually sends",
          _run(7)["spend"]["frames"] == 1, "the doctor's spend is not its one probe")
    for requested in (1, 2, 7, RC.MAX_VLM_FRAMES_CEILING):
        r = _run(requested)
        check(f"R6 the whole run stays inside the ceiling of {requested}",
              r["spend"]["frames"] <= r["budget"]["max_vlm_frames"], str(r["spend"]))


def t_connectivity_and_end_to_end_reach_the_verdict() -> None:
    """A step that fails has to move the top-level answer.

    `connectivity` and `end-to-end` were run and printed and could fail without
    the verdict moving: a full live run rejected by the report validator, with
    every capability column reading PASS, still exited 0 and published
    QUALIFIED. A row whose failure changes nothing is a row nobody reads.

    The other half is the half that is easy to break while fixing it. A missing
    credential and a live run nobody asked for are `UNTESTED` and `SKIPPED`,
    and they must leave the verdict exactly where it was — otherwise the fix
    turns "we do not know" into "this provider is broken", which is the one
    substitution this module exists to prevent.
    """
    import release_check as RC

    def steps(conn: str, e2e: str, cap: str = RC.PASS) -> list:
        return [RC.Step("connectivity", conn), RC.Step("vlm", cap),
                RC.Step("synthesis", cap), RC.Step("critic", cap),
                RC.Step("end-to-end", e2e)]

    qualified = RC.qualification_matrix([RC.Step("vlm", RC.PASS, provider="p", model="m"),
                                         RC.Step("synthesis", RC.PASS, provider="p", model="m"),
                                         RC.Step("critic", RC.PASS, provider="p", model="m")])
    check("R7 a clean matrix with no steps is QUALIFIED",
          RC.overall_status(qualified, []) == "QUALIFIED", str(qualified))

    check("R7 a failing connectivity takes a clean run to UNQUALIFIED",
          RC.overall_status(qualified, steps(RC.FAIL, RC.PASS)) == "UNQUALIFIED")
    check("R7 a failing end-to-end does the same",
          RC.overall_status(qualified, steps(RC.PASS, RC.FAIL)) == "UNQUALIFIED")
    check("R7 an unreached connectivity leaves the verdict alone",
          RC.overall_status(qualified, steps(RC.UNTESTED, RC.PASS)) == "QUALIFIED")
    check("R7 a skipped live run leaves it alone too",
          RC.overall_status(qualified, steps(RC.PASS, RC.SKIPPED)) == "QUALIFIED")
    check("R7 an unreached end-to-end is not a failure",
          RC.overall_status(qualified, steps(RC.PASS, RC.UNTESTED)) == "QUALIFIED")

    # And the same rule when nothing was exercised: the two steps may be
    # anything and the verdict stays CONDITIONAL, which exits 0.
    for conn, e2e in ((RC.UNTESTED, RC.SKIPPED), (RC.SKIPPED, RC.UNTESTED)):
        check(f"R7 nothing exercised stays CONDITIONAL with connectivity={conn} "
              f"end-to-end={e2e}",
              RC.overall_status([], steps(conn, e2e, RC.UNTESTED)) == "CONDITIONAL")

    # A matrix that already found a failure is not re-counted through the steps.
    degraded = RC.qualification_matrix([RC.Step("vlm", RC.FAIL, provider="p", model="m"),
                                        RC.Step("synthesis", RC.PASS, provider="p", model="m"),
                                        RC.Step("critic", RC.PASS, provider="p", model="m")])
    check("R7 the matrix's own verdict is not second-guessed by the steps",
          RC.overall_status(degraded, steps(RC.PASS, RC.PASS, RC.FAIL)) == "DEGRADED")

    # Through the real run, with the three doctors stubbed to answer and hold.
    # The claim under test is the published `status`, not a helper's return.
    import eval_live

    def _with_e2e(status: str) -> dict:
        orig_vlm = RC.VD.run_doctor
        orig_syn = RC.SD.run_synthesis_doctor
        orig_cri = RC.CD.run_critic_doctor
        orig_live = eval_live.main
        RC.VD.run_doctor = lambda *a, **k: (0, _report(qualified=True, answered=True))
        RC.SD.run_synthesis_doctor = lambda *a, **k: (0, _report(qualified=True, answered=True))
        RC.CD.run_critic_doctor = lambda *a, **k: (0, _report(qualified=True, answered=True))
        eval_live.main = lambda argv, result_out=None: (
            result_out.append({"status": status}) if result_out is not None else None) or 0
        try:
            return RC.run_release_check(
                vlm_cfg=VL.VLMConfig(provider="fake", model="m"),
                synth_cfg=SY.SynthConfig(provider="p", model="m"),
                url="https://youtu.be/x", max_vlm_frames=4,
                max_synthesis_calls=1, max_critic_calls=1)
        finally:
            (RC.VD.run_doctor, RC.SD.run_synthesis_doctor, RC.CD.run_critic_doctor,
             eval_live.main) = (orig_vlm, orig_syn, orig_cri, orig_live)

    ok = _with_e2e("ok")
    check("R7 a live run that produced a report is a PASS step",
          {s["step"]: s["state"] for s in ok["steps"]}["end-to-end"] == RC.PASS,
          str(ok["steps"]))
    rejected = _with_e2e("report-rejected")
    check("R7 a live run the validator rejected publishes UNQUALIFIED",
          rejected["status"] == "UNQUALIFIED", rejected["status"])
    unfetchable = _with_e2e("failed")
    check("R7 a live run that never started is UNTESTED, and the run says so",
          {s["step"]: s["state"] for s in unfetchable["steps"]}["end-to-end"] == RC.UNTESTED,
          str(unfetchable["steps"]))
    check("R7 ...and does not become a failure of the provider",
          unfetchable["status"] != "UNQUALIFIED", unfetchable["status"])


def t_critic_schema_gate_is_one_verdict() -> None:
    """`structured_enforced OK` printed directly above `findings[0] missing code`.

    A live evaluation produced both, and they cannot describe the same response
    as valid. The cause: the malformed case was recorded as *answered* — which it
    was, the endpoint returned it — and `structured_enforced` was then computed
    from "answered", so a response the validator had already rejected counted as
    evidence that the validator had passed. `cases_matched` read the findings and
    disagreed. The fix is not a better message; it is that the chain has one
    verdict per response and the later step is unreachable when the earlier one
    fails.
    """
    ev = RF.synthesis_qualification()
    # The exact response the live run produced: the schema's top-level
    # `findings` key is present, so the negotiator calls it conformant, and every
    # entry inside it is in the wrong vocabulary.
    live = {"findings": [{"type": "number-error", "evidence_id": "itm-003",
                          "description": "the figure does not match the evidence"}]}
    answers = [_answer(live) for _ in range(4)]
    code, rep = CD.run_critic_doctor(ev, _cfg(), transport=_scripted(answers),
                                     budget=SY.Budget(4))
    checks = {c["check"]: c for c in rep.to_dict()["checks"]}
    cases = {c["case"]: c for c in rep.extra["cases"]}

    check("D7 the run is not qualified", not rep.qualified, str(rep.qualified))
    check("D7 structured_enforced is FAIL, not OK",
          checks["structured_enforced"]["ok"] is False, str(checks["structured_enforced"]))
    check("D7 the reason is the schema failure, not a transport one",
          "schema" in checks["structured_enforced"]["detail"],
          checks["structured_enforced"]["detail"])
    check("D7 every case is marked schema-invalid",
          all(c["schema_valid"] is False for c in cases.values()),
          str({k: v["schema_valid"] for k, v in cases.items()}))
    check("D7 the cases are still recorded as answered — it did reply",
          all(c["answered"] for c in cases.values()),
          str({k: v["answered"] for k, v in cases.items()}))
    check("D7 cases_matched is unreachable, not a second verdict about the same answers",
          checks["cases_matched"]["detail"] == "unreachable",
          str(checks["cases_matched"]))
    check("D7 the report says the answers were unreadable",
          "not in the schema" in (rep.error or ""), str(rep.error))
    check("D7 a malformed run exits 1, not 0", code == 1, str(code))

    # The chain one case earlier, for the same reason: an endpoint that never
    # answered cannot pass the schema gate either, and must not be counted.
    class _Dead:
        def __init__(self, **kw):
            kw["transport"] = lambda *a: (_ for _ in ()).throw(
                __import__("urllib.error", fromlist=["x"]).URLError("no route"))
    import synth_model as _sm
    _orig = _sm.SynthesisClient
    _sm.SynthesisClient = _Dead
    try:
        _c, rep = CD.run_critic_doctor(ev, _cfg(), budget=SY.Budget(4))
    finally:
        _sm.SynthesisClient = _orig
    checks = {c["check"]: c for c in rep.to_dict()["checks"]}
    check("D7 a critic that never answers does not pass the schema gate",
          checks["structured_enforced"]["ok"] is False, str(checks["structured_enforced"]))

    # And the gate is reachable and true when every response is genuinely valid:
    # a gate that cannot pass is not a gate.
    good = [_answer(_findings([])),
            _answer(_findings([{"code": "number-error", "where": "takeaways[0]",
                                "message": "90 ms does not match 40 ms"}])),
            _answer(_findings([{"code": "missing-qualifier", "where": "summary[0]",
                                "message": "the hedge is gone"}])),
            _answer(_findings([{"code": "unsupported-sentence", "where": "takeaways[1]",
                                "message": "not in the evidence"}]))]
    code, rep = CD.run_critic_doctor(ev, _cfg(), transport=_scripted(good),
                                     budget=SY.Budget(4))
    checks = {c["check"]: c for c in rep.to_dict()["checks"]}
    check("D7 a well-formed critic passes the schema gate",
          checks["structured_enforced"]["ok"] is True, str(checks["structured_enforced"]))
    check("D7 ...and the case measurement is then reachable and true",
          checks["cases_matched"]["ok"] is True, str(checks["cases_matched"]))
    check("D7 ...and the run qualifies", rep.qualified and code == 0, f"{code} {rep.qualified}")
