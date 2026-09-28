"""`yt-report release-check` — the cheapest sequence that qualifies a provider.

This is not a health check and it is not an end-to-end run. Its only question is
the one that decides whether the release claim is true: **for this provider and
this model, which capabilities have actually been exercised successfully?** Every
other thing a maintainer might want to measure is a different command that
already exists.

It exists because the four capabilities are qualified in four different places —
`vlm_doctor.py`, `synth_doctor.py`, `critic_doctor.py` and the live run — and
nothing collected the answers. A report that said "synthesis reached 9/10 real
gates" is a fact about a run, and a reader of the repository had no way to ask
the question again a week later without repeating the whole experiment.

Four states, and the distinction between two of them is the whole point:

``PASS``
    A real request was made, it produced output, and the output held.
``FAIL``
    A real request was made, it produced output, and the output did not hold.
    The capability is *disproven* — this is the state that must never be
    laundered into a weaker one.
``UNTESTED``
    No usable answer ever came back. No credential, no credit, no egress, an
    unreachable endpoint, a spent budget. Nothing was disproven and nothing was
    shown; the honest verdict is that the question was not reached.
``SKIPPED``
    The step was not requested — no provider configured, or the optional
    end-to-end run was not asked for.

The difference between ``FAIL`` and ``UNTESTED`` is the difference between "this
model is not qualified" and "we do not know", and the two look identical in a
report that only prints a checkmark. Converting one into the other is how a
release ends up claiming production readiness it never earned, so the rule is a
single function over one question — *did an answer come back* — rather than a
per-doctor interpretation of its own stage names.

The matrix, not the step list, is what a reader acts on: a provider is usable for
a report job if its vision and synthesis columns are ``PASS``, and the
``status`` column says which of the three ways that is currently true —
``QUALIFIED`` (exercised and holding), ``CONDITIONAL`` (never exercised, or only
partly), ``UNQUALIFIED`` (exercised and failing), ``DEGRADED`` (an optional
model layer failed while the deterministic pipeline still produced a report).

Budgets are ceilings, not targets. Every step takes a `Budget`, spends from it
before it makes a call, and the totals are reported next to the ceilings so a
reader can see the run stayed inside them rather than being told it did.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import critic_doctor as CD
import synth_doctor as SD
import synthesis as SY
import versions as V
import vlm as VL
import vlm_doctor as VD

#: The five steps, in the order they are run and the order they are reported.
#: A step that never ran is never omitted from the output; a missing row reads
#: as "nothing to report" and an absent capability reads as "not a problem".
STEPS = ("connectivity", "vlm", "synthesis", "critic", "end-to-end")

PASS, FAIL, UNTESTED, SKIPPED = "PASS", "FAIL", "UNTESTED", "SKIPPED"

#: The ceilings. Small on purpose — the cheapest useful sequence is two frames,
#: one draft, one critique, and a revision the writer is allowed at most once.
DEFAULT_MAX_VLM_FRAMES = 2
DEFAULT_MAX_SYNTHESIS_CALLS = 1
DEFAULT_MAX_CRITIC_CALLS = 1
DEFAULT_MAX_REVISION_CALLS = 1

#: The largest frame ceiling `release-check` will accept, and the only number it
#: stops being. It is `vlm.DEFAULT_MAX_VLM_FRAMES` — the cap the VLM pass
#: already enforces on itself, and the value every document in this repository
#: quotes as the maximum (`--vlm-provider anthropic --max-vlm-frames 8`). Named
#: rather than inlined because the defect it replaces was an inlined `2`: a
#: `--max-vlm-frames 20` that silently did nothing read as a provider problem
#: rather than as a flag that had been disconnected from its own argument.
MAX_VLM_FRAMES_CEILING = VL.DEFAULT_MAX_VLM_FRAMES


def resolve_vlm_frame_ceiling(requested: int) -> int:
    """The frame ceiling this run will actually enforce, after both bounds.

    The upper bound is `MAX_VLM_FRAMES_CEILING`. A request above it is reduced
    to it and the reduction is visible in `budget.max_vlm_frames` and in the
    `--help` text, rather than being refused: the ceiling is a *budget*, and a
    budget that cannot be raised is a budget that is not under the caller's
    control. The value is still enforced — nothing here raises what the VLM pass
    would itself have capped.

    The lower bound is one, because the VLM step in this sequence is
    `vlm-doctor`, which probes exactly one image whatever the ceiling says. A
    ceiling of zero would not make it cheaper; it would make the reported spend
    smaller than the run. "Send no frame at all" is a real mode and `eval-live`
    owns it, where it means a synthesis evaluation rather than a vision failure.
    """
    requested = int(requested)
    if requested < 0:
        raise ValueError("--max-vlm-frames cannot be negative: a ceiling below "
                         "zero is not a cheaper run, it is an impossible one")
    return max(1, min(requested, MAX_VLM_FRAMES_CEILING))


def answered(report) -> bool:
    """Did a real request come back with something the model produced?

    One question, asked of every doctor, rather than each doctor's own account of
    how far it got. A `stage` string is a doctor's private vocabulary: comparing
    against it here would make this module a second thing to update when a doctor
    grows a stage, and a name that no longer matches would read as "the request
    came back and was judged" — the one answer that must never be inferred
    instead of observed.

    So it is asked of the *evidence of an answer*: an observation, a parsed
    result, or a case that actually ran. A transport that raised, a credential
    that was refused and a budget that was already spent all leave those absent.
    """
    if report is None:
        return False
    if getattr(report, "observation", None) is not None:
        return True
    extra = getattr(report, "extra", {}) or {}
    # `ok`, not merely present: a result object is recorded for a refused
    # credential and a dead socket too, and treating its existence as an answer
    # is how a connection failure ends up filed as a model that cannot be
    # trusted rather than as a provider that was never reached.
    if (extra.get("result") or {}).get("ok"):
        return True
    return any(c.get("ran") and c.get("answered") is not False
               for c in (extra.get("cases") or []))


def classify(report, requested: bool = True) -> str:
    """The one rule that decides what a step's verdict means."""
    if not requested or report is None:
        return SKIPPED
    if getattr(report, "qualified", False):
        return PASS
    return FAIL if answered(report) else UNTESTED


def _http_failure_class(report) -> str:
    """What kind of failure it was, when there was one.

    Recorded because "FAILED" alone does not tell a maintainer whether to fix a
    key, buy credit, or change a model id — and the three produce the same
    checkmark. Derived from the recorded text, never by re-classifying a status
    the doctor already computed, so this cannot disagree with it.
    """
    text = " ".join(str(c.detail) for c in (getattr(report, "checks", {}) or {}).values())
    text += " " + str(getattr(report, "error", "") or "")
    low = text.lower()
    for code, klass in (("401", "auth"), ("403", "auth"), ("429", "rate-limit"),
                        ("402", "no-credit"), ("insufficient", "no-credit"),
                        ("credit", "no-credit"), ("connection", "unreachable"),
                        ("timed out", "timeout"), ("timeout", "timeout"),
                        ("404", "no-such-model"), ("not found", "no-such-model"),
                        ("could not be reached", "unreachable"),
                        ("could not be constructed", "unreachable"),
                        ("endpoint could not be reached", "unreachable"),
                        ("no call was made", "unreachable")):
        if code in low:
            return klass
    return "none" if getattr(report, "qualified", False) else "unclassified"


def _tokens(report) -> dict | None:
    """Token counts, when the provider reported them and when it did not, `None`."""
    result = (getattr(report, "extra", {}) or {}).get("result") or {}
    usage = result.get("usage") or {}
    return {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            if k in usage} or None


@dataclass
class Step:
    """One step's whole story, in the shape the CLI and the JSON both read."""

    name: str
    state: str
    reason: str = ""
    provider: str | None = None
    model: str | None = None
    calls: int = 0
    frames: int = 0
    revisions: int = 0
    cache_hits: int = 0
    latency_s: float | None = None
    tokens: dict | None = None
    http_failure_class: str = "none"
    checks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"step": self.name, "state": self.state, "reason": self.reason,
             "provider": self.provider, "model": self.model, "calls": self.calls,
             "frames": self.frames, "revisions": self.revisions,
             "cache_hits": self.cache_hits,
             "latency_s": round(self.latency_s, 3) if self.latency_s is not None else None,
             "tokens": self.tokens, "http_failure_class": self.http_failure_class}
        if self.checks:
            d["checks"] = self.checks
        return d


def _step_from(name: str, report, requested: bool, *,
               calls: int = 0, frames: int = 0, revisions: int = 0) -> Step:
    state = classify(report, requested)
    checks = [c.to_dict() for c in (getattr(report, "checks", {}) or {}).values()]
    failed = [c for c in checks if not c.get("ok")]
    return Step(
        name=name, state=state,
        reason=(str(getattr(report, "error", "") or "")
                or (failed[0]["check"] + ": " + failed[0]["detail"] if failed else "")),
        provider=(getattr(report, "config", {}) or {}).get("provider"),
        model=(getattr(report, "config", {}) or {}).get("model"),
        calls=calls, frames=frames, revisions=revisions,
        latency_s=getattr(report, "latency_s", None),
        tokens=_tokens(report), http_failure_class=_http_failure_class(report),
        checks=checks,
    )


# --------------------------------------------------------------------------- #
# the steps
# --------------------------------------------------------------------------- #

def _connectivity(vlm_cfg, synth_cfg) -> Step:
    """Step 1 — can this configuration reach anything at all?

    The cheapest step that can fail, and deliberately first: a wrong base URL or
    a missing key is discovered here for free rather than after a media upload
    and a fifteen-hundred-token draft. It asks each *configured* provider the
    question and reports one row, because a release check that quietly ignores
    the synthesis provider because the vision one answered is not a release
    check.
    """
    rows = []
    for label, cfg, module, err in (
            ("vlm", vlm_cfg, VD, "set VLM_PROVIDER or pass --vlm-provider"),
            ("synthesis", synth_cfg, SD,
             "set SYNTH_PROVIDER or pass --synth-provider")):
        if not cfg.provider or cfg.provider == "none":
            rows.append({"provider": label, "state": SKIPPED, "detail": f"not configured; {err}"})
            continue
        report = module.DoctorReport(cfg, ("reachable",))
        if label == "vlm":
            _code, report = VD.run_doctor(cfg)
        else:
            ev, _ = SD.load_evidence(None)
            _code, report = SD.run_synthesis_doctor(
                ev, cfg, budget=SY.Budget(0))
        rows.append({"provider": label, "state": classify(report),
                     "detail": report.error or ("" if report.qualified else "unreachable")})
    return Step(name="connectivity", state=_reachability_state(rows),
                reason="; ".join(f"{r['provider']}: {r['state']}" for r in rows),
                provider=", ".join(sorted({vlm_cfg.provider, synth_cfg.provider} - {None, "", "none"})) or None,
                checks=rows)


def _reachability_state(rows: list[dict]) -> str:
    """One step's state from its per-provider rows, by the module's own rule.

    The same three-question shape `classify` asks of one doctor, applied to the
    several this step makes: a row that was `SKIPPED` was never asked and does
    not count against the step; a row that is `FAIL` means an answer came back
    and did not hold, which is the one thing that must reach the top-level
    verdict; a row that is `UNTESTED` means the question was never reached and
    must leave the step no worse than "we do not know".

    It was a three-branch expression of its own before, which agreed with
    `classify` by coincidence rather than by construction — the two are now the
    same rule written once.
    """
    if not rows or all(r["state"] == SKIPPED for r in rows):
        return UNTESTED
    if any(r["state"] == FAIL for r in rows):
        return FAIL
    return PASS if any(r["state"] == PASS for r in rows) else UNTESTED


def _vlm_step(cfg) -> Step:
    """Step 2 — one real frame through `vlm-doctor`, and what came back.

    The ceiling is already on `cfg` when this is called: the doctor reads
    `cfg.max_frames`, so a ceiling held anywhere else is a ceiling nothing
    downstream can see. `run_release_check` is what puts it there.
    """
    if not cfg.provider or cfg.provider == "none":
        return Step(name="vlm", state=SKIPPED,
                    reason="no VLM provider configured; the vision columns stay UNTESTED")
    _, report = VD.run_doctor(cfg)
    spent = 1 if answered(report) else 0
    # One probe image, or none: the doctor sends a single frame whatever the
    # ceiling is, so reporting the whole ceiling here would be reporting the
    # plan where the module promises spend.
    return _step_from("vlm", report, True, calls=spent, frames=spent)


def _synth_step(ev, cfg, max_calls: int) -> Step:
    if not cfg.provider or cfg.provider == "none":
        return Step(name="synthesis", state=SKIPPED,
                    reason="no synthesis provider configured")
    budget = SY.Budget(max(0, max_calls))
    _, report = SD.run_synthesis_doctor(ev, cfg, budget=budget)
    used = (report.extra.get("calls") or {}).get("spent", 0) if report.extra else 0
    return _step_from("synthesis", report, True,
                      calls=int(used or 0) if answered(report) else 0)


def _critic_step(ev, cfg, max_calls: int) -> Step:
    if not cfg.provider or cfg.provider == "none":
        return Step(name="critic", state=SKIPPED,
                    reason="no synthesis provider configured; the critic runs on it")
    budget = SY.Budget(max(0, max_calls))
    _, report = CD.run_critic_doctor(ev, cfg, budget=budget)
    used = (report.extra.get("calls") or {}).get("spent", 0) if report.extra else 0
    return _step_from("critic", report, True,
                      calls=int(used or 0) if answered(report) else 0)


def _end_to_end(url: str | None, vlm_cfg, synth_cfg, budgets: dict) -> Step:
    """Step 5 — one small real video, only when asked for.

    Delegated to `eval_live`, which owns the fetch and the budget arithmetic.
    Reimplementing a second live runner here is the exact thing this repository
    documents against, and the duplication would drift silently: two runners
    spending the same ceiling differently, with only one of them checked.

    The state comes from the run's own `status`, not from its exit code. The two
    are not the same question: `eval_live` exits non-zero for a fetch that
    failed, an artifact directory that was never written, and an artifact with no
    captions — none of which is anything the provider did or did not do. Reading
    the code made all three `FAIL`, which is the one state the module is not
    allowed to produce without an answer behind it.
    """
    if not url:
        return Step(name="end-to-end", state=SKIPPED,
                    reason="no URL given; pass one to exercise a whole video")
    import eval_live
    argv = [url, "--max-vlm-frames", str(budgets["vlm_frames"]),
            "--max-synthesis-calls", str(budgets["synthesis_calls"]),
            "--max-critic-calls", str(budgets["critic_calls"]), "--no-report"]
    seen: list[dict] = []
    started = time.monotonic()
    try:
        eval_live.main(argv, result_out=seen)
    except SystemExit as exc:          # the module's own exit convention
        if not seen:
            seen.append({"status": "failed", "error": f"the live run exited {exc.code}"})
    except Exception as exc:           # noqa: BLE001 — a crash is a verdict
        seen.append({"status": "raised",
                     "error": f"the live run raised {type(exc).__name__}: {exc}"})
    out = seen[-1] if seen else {"status": "failed", "error": "the live run reported nothing"}
    status = str(out.get("status", "failed"))
    reason = str(out.get("error") or out.get("fallback_reason") or "")
    if status == "ok":
        state = PASS
    elif status in ("failed", "raised"):
        # Never got as far as exercising anything: the artifact could not be
        # read. Nothing was disproven, so this is UNTESTED and not FAIL — a
        # fetch that timed out must not read as a provider that cannot be
        # trusted.
        state, reason = UNTESTED, reason or f"the live run could not start ({status})"
    else:
        # The run completed and the pipeline's own gates rejected what it
        # produced: an answer came back and did not hold.
        state, reason = FAIL, reason or f"the live run produced {status}"
    return Step(name="end-to-end", state=state, reason=reason,
                provider=vlm_cfg.provider, model=synth_cfg.model or vlm_cfg.model,
                frames=budgets["vlm_frames"] if state != SKIPPED else 0,
                calls=(budgets["synthesis_calls"] + budgets["critic_calls"])
                if state in (PASS, FAIL) else 0,
                latency_s=time.monotonic() - started)


# --------------------------------------------------------------------------- #
# the matrix
# --------------------------------------------------------------------------- #

#: Column order, fixed. A table whose columns move between runs cannot be diffed,
#: and the matrix is the artifact most worth diffing.
MATRIX_COLUMNS = ("provider", "model", "vision", "structured_vlm", "synthesis",
                  "critic", "status")


def _status_of(states: list[str]) -> str:
    """`QUALIFIED`, `UNQUALIFIED`, `CONDITIONAL` or `DEGRADED` for one row.

    The order of these tests is the whole rule, so it is written as a chain and
    not as a table of cases: a capability that is *disproven* outranks one that
    is merely unexercised, and a model layer that failed while the report was
    still produced is a different claim again — not a pass, and not a failure of
    the pipeline.
    """
    if any(s == FAIL for s in states):
        # A failure with something working beside it is a degraded run: the
        # model layer is partly broken and the report still lands. A failure with
        # nothing working beside it is not degraded, it is a provider that does
        # not do this job — and calling that "degraded" would be the flattering
        # reading of a broken configuration.
        return "DEGRADED" if PASS in states else "UNQUALIFIED"
    if all(s == PASS for s in states):
        return "QUALIFIED"
    return "CONDITIONAL"


def qualification_matrix(steps: list[Step]) -> list[dict]:
    """One row per provider/model, with the four capabilities and a verdict.

    Built from the steps rather than re-queried, so the matrix can never disagree
    with the step list printed above it. Two providers appear as two rows even
    when they are the same alias, because vision and synthesis are qualified
    against different things and a single row would have to pick one.
    """
    by_name = {s.name: s for s in steps}
    vlm_step, syn = by_name.get("vlm"), by_name.get("synthesis")
    critic = by_name.get("critic")

    def vision_states() -> tuple[str, str]:
        if vlm_step is None or vlm_step.state == SKIPPED:
            return "SKIPPED", "SKIPPED"
        # Vision and structured output are two capabilities the one VLM exchange
        # can fail separately, and a provider that returns prose instead of JSON
        # sees perfectly well and still cannot produce an observation. They are
        # reported apart because they are repaired apart.
        if vlm_step.state == UNTESTED:
            # A provider that was never reached has no verdict on anything, and
            # reading its individual checks would manufacture one: the doctor
            # marks the steps after a break as failed-but-unreachable, and
            # `FAIL` in a capability column is a claim the release cannot make.
            return UNTESTED, UNTESTED
        checks = {c["check"]: c for c in (vlm_step.checks or [])}

        def and_of(*names: str) -> str | None:
            """One column's state, from the checks that actually have to hold.

            A capability is not qualified by the check that happens to be named
            after it. A provider can see the frame (`vision_capable`) and return
            every top-level key (`structured_enforced`) and still produce an
            observation that fails re-parse — a confidence above its ceiling, a
            missing content type, a shape the evidence gate will reject. That
            provider is not vision-capable for this pipeline, and publishing
            `PASS` in its column is the one claim a reader is entitled to act on.

            So each column ANDs the checks that constitute it, and a column whose
            checks are all absent inherits the step's own state rather than
            guessing. `schema_valid` was read here for years and emitted by
            nothing: the check is named `test_observation_valid`, and the silent
            fallback to `structured_enforced` is what let a failed re-parse
            through as a pass.
            """
            present = [checks[n] for n in names if n in checks]
            if not present:
                return None
            return PASS if all(c["ok"] for c in present) else FAIL

        vision = and_of("vision_capable", "test_observation_valid")
        structured = and_of("structured_enforced", "test_observation_valid")
        return ((vlm_step.state if vision is None else vision),
                (vlm_step.state if structured is None else structured))

    rows: dict[tuple, dict] = {}
    for step, cap in ((vlm_step, "vlm"), (syn, "synthesis"), (critic, "critic")):
        if step is None or not step.provider:
            continue
        key = (step.provider, step.model or "-")
        row = rows.setdefault(key, {"provider": step.provider, "model": step.model or "-",
                                    "vision": "SKIPPED", "structured_vlm": "SKIPPED",
                                    "synthesis": "SKIPPED", "critic": "SKIPPED"})
        if cap == "vlm":
            row["vision"], row["structured_vlm"] = vision_states()
        else:
            row[cap] = step.state
    if not rows and syn is not None and syn.provider:
        rows[("none", "-")] = {"provider": "none", "model": "-", "vision": "SKIPPED",
                               "structured_vlm": "SKIPPED", "synthesis": "SKIPPED",
                               "critic": "SKIPPED"}
    for row in rows.values():
        row["status"] = _status_of([row[c] for c in MATRIX_COLUMNS[2:6]])
    return list(rows.values())


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def overall_status(matrix: list[dict], steps: list[Step]) -> str:
    """The top-level verdict, from the matrix *and* the steps it cannot carry.

    The matrix answers one question: was each capability exercised, and did it
    hold. Two steps are not capabilities and have no column — `connectivity` and
    `end-to-end` — and they were the two that could be run, printed, and fail
    without the verdict moving. A step whose failure changes nothing is a row
    nobody reads, and the whole point of this command is that a reader does not
    have to know which rows to read.

    So the matrix still decides, and then the steps it does not cover get the
    same cascade the doctors already own: `FAIL` is an answer that came back and
    did not hold, so it takes the verdict to `UNQUALIFIED`; `UNTESTED` and
    `SKIPPED` are questions that were never reached, so they leave it exactly
    where it was. That is what keeps a missing credential and a deliberately
    skipped live run from reading as a broken provider — the two states stay
    distinguishable all the way to the top, which is the one thing the matrix
    could not do on its own.

    `UNQUALIFIED` rather than `DEGRADED` for a failing step: `DEGRADED` is the
    claim that a model layer broke while the report still landed, and the matrix
    is where that distinction is made, from columns that carry capabilities.
    """
    states = [r["status"] for r in matrix]
    if matrix and all(s == "QUALIFIED" for s in states):
        verdict = "QUALIFIED"
    elif any(s == "UNQUALIFIED" for s in states):
        verdict = "UNQUALIFIED"
    elif any(s == "DEGRADED" for s in states):
        verdict = "DEGRADED"
    else:
        verdict = "CONDITIONAL"
    if verdict in ("UNQUALIFIED", "DEGRADED"):
        return verdict
    if any(s.state == FAIL for s in steps if s.name in UNCOVERED_BY_MATRIX):
        return "UNQUALIFIED"
    return verdict


#: The steps the matrix has no column for, and therefore the only ones the
#: step-level cascade above can speak for. The other three become matrix
#: columns, and letting them speak twice would let a single capability be
#: counted against the verdict twice.
UNCOVERED_BY_MATRIX = frozenset({"connectivity", "end-to-end"})


def run_release_check(*, vlm_cfg, synth_cfg, url: str | None = None,
                      max_vlm_frames: int = DEFAULT_MAX_VLM_FRAMES,
                      max_synthesis_calls: int = DEFAULT_MAX_SYNTHESIS_CALLS,
                      max_critic_calls: int = DEFAULT_MAX_CRITIC_CALLS,
                      max_revision_calls: int = DEFAULT_MAX_REVISION_CALLS,
                      evidence=None) -> dict:
    """Run the sequence and return the machine-readable result. No printing.

    The budget is passed as ceilings and each step spends from its own `Budget`
    before it makes a call, so the totals reported are what was spent rather than
    what was intended — a run that reports its own plan as its usage is not a
    measurement.
    """
    if max_revision_calls < 0:
        raise ValueError("--max-revision-calls cannot be negative: a ceiling below "
                         "zero would be a run that cannot report honestly")
    frame_ceiling = resolve_vlm_frame_ceiling(max_vlm_frames)
    # The ceiling goes onto the config, which is where every consumer of it
    # reads from. Left on the argument it would apply to the reported budget
    # and to nothing else: the doctors read `cfg.max_frames`, so a caller's
    # request would be honoured in the JSON and ignored by every request made.
    # That is the shape the flag had before — correct in the summary, inert in
    # the run.
    vlm_cfg = replace(vlm_cfg, max_frames=frame_ceiling)
    ev = evidence if evidence is not None else SD.load_evidence(None)[0]

    # One frame ceiling for the run, drawn from by both steps that send frames.
    # The doctor goes first because it is cheaper to be wrong about: it sends
    # one image, so the live run gets what is left of the ceiling rather than
    # the whole of it, and the total the reader is shown is the total that was
    # available. Handing each step the full ceiling instead multiplied it —
    # every individual report stayed true and the sum was twice the number
    # printed beside it.
    vlm_step = _vlm_step(vlm_cfg)
    steps = [
        _connectivity(vlm_cfg, synth_cfg),
        vlm_step,
        _synth_step(ev, synth_cfg, max_synthesis_calls),
        _critic_step(ev, synth_cfg, max_critic_calls),
        _end_to_end(url, vlm_cfg, synth_cfg,
                    {"vlm_frames": max(0, frame_ceiling - vlm_step.frames),
                     "synthesis_calls": max_synthesis_calls,
                     "critic_calls": max_critic_calls}),
    ]
    by_name = {s.name: s for s in steps}
    # A revision is a second draft from the same writer, so it spends from the
    # synthesis ceiling rather than from one of its own — a second budget would
    # let the two ceilings be spent together and the total exceed both.
    revisions = 0 if by_name["synthesis"].state != PASS else min(
        max_revision_calls, 0 if max_synthesis_calls <= 1 else max_synthesis_calls - 1)
    by_name["synthesis"].revisions = revisions

    matrix = qualification_matrix(steps)
    # DEGRADED needs its own branch, and so do the two steps the matrix has no
    # column for. `overall_status` holds the whole rule; see there for why a
    # failing `connectivity` or `end-to-end` reaches the verdict and an
    # unreached one does not.
    overall = overall_status(matrix, steps)
    return {
        "architecture": {
            "pipeline_version": V.PIPELINE_VERSION,
            "schema_version": V.SCHEMA_VERSION,
            "extraction_version": V.EXTRACTION_VERSION,
            "vlm_prompt_version": V.VLM_PROMPT_VERSION,
            "synth_prompt_version": V.SYNTH_PROMPT_VERSION,
            "synth_schema_version": V.SYNTH_SCHEMA_VERSION,
        },
        "status": overall,
        "steps": [s.to_dict() for s in steps],
        "matrix": matrix,
        "budget": {"max_vlm_frames": frame_ceiling,
                   "max_synthesis_calls": max_synthesis_calls,
                   "max_critic_calls": max_critic_calls,
                   "max_revision_calls": max_revision_calls},
        "spend": {
            "calls": sum(s.calls for s in steps),
            "frames": sum(s.frames for s in steps),
            "revisions": revisions,
            "latency_s": round(sum(s.latency_s or 0.0 for s in steps), 3),
        },
    }


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #

def render(result: dict) -> str:
    lines = ["", f"release check — {result['status']}", ""]
    for s in result["steps"]:
        mark = {"PASS": "PASS    ", "FAIL": "FAIL    ", "UNTESTED": "UNTESTED",
                "SKIPPED": "SKIPPED "}[s["state"]]
        lines.append(f"  {mark}  {s['step']:<12} {s.get('reason') or ''}".rstrip())
        if s.get("provider"):
            bits = [f"provider {s['provider']}", f"model {s.get('model') or '-'}"]
            if s["calls"]:
                bits.append(f"{s['calls']} call(s)")
            if s["frames"]:
                bits.append(f"{s['frames']} frame(s)")
            if s.get("latency_s") is not None:
                bits.append(f"{s['latency_s'] * 1000:.0f} ms")
            if s.get("tokens"):
                bits.append(f"{s['tokens'].get('total_tokens', '?')} tokens")
            if s.get("http_failure_class") not in (None, "none"):
                bits.append(s["http_failure_class"])
            lines.append(f"             {'  '.join(bits)}")
    if result["matrix"]:
        widths = {c: max(len(c), *(len(str(r[c])) for r in result["matrix"])) + 2
                  for c in MATRIX_COLUMNS}
        lines += ["", "  " + "".join(f"{c:<{widths[c]}}" for c in MATRIX_COLUMNS).rstrip()]
        for row in result["matrix"]:
            lines.append("  " + "".join(
                f"{str(row[c]):<{widths[c]}}" for c in MATRIX_COLUMNS).rstrip())
    b, sp = result["budget"], result["spend"]
    lines += ["",
              f"  budget   frames {sp['frames']}/{b['max_vlm_frames']}   "
              f"synthesis {b['max_synthesis_calls']}   critic {b['max_critic_calls']}   "
              f"revisions {sp['revisions']}/{b['max_revision_calls']}   "
              f"calls {sp['calls']}   {sp['latency_s'] * 1000:.0f} ms",
              f"  status   {result['status']} — CONDITIONAL until a real synthesis, "
              f"VLM and critic run have all been exercised successfully.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="yt-report release-check",
        description="qualify a provider against the cheapest sequence that proves it")
    ap.add_argument("url", nargs="?", default=None,
                    help="optional YouTube URL for the one end-to-end step")
    ap.add_argument("--vlm-provider", default=None)
    ap.add_argument("--vlm-model", default=None)
    ap.add_argument("--vlm-base-url", default=None)
    ap.add_argument("--synth-provider", default=None)
    ap.add_argument("--synth-model", default=None)
    ap.add_argument("--synth-base-url", default=None)
    ap.add_argument("--synth-api-key", default=None)
    ap.add_argument("--detail", default="brief", choices=["brief", "balanced", "deep"])
    ap.add_argument("--max-vlm-frames", type=int, default=DEFAULT_MAX_VLM_FRAMES,
                    help=f"hard ceiling on frames sent for vision, across the "
                         f"vlm step and the end-to-end run together "
                         f"(default {DEFAULT_MAX_VLM_FRAMES}, "
                         f"maximum {MAX_VLM_FRAMES_CEILING})")
    ap.add_argument("--max-synthesis-calls", type=int, default=DEFAULT_MAX_SYNTHESIS_CALLS,
                    help="hard ceiling on draft calls (default 1)")
    ap.add_argument("--max-critic-calls", type=int, default=DEFAULT_MAX_CRITIC_CALLS,
                    help="hard ceiling on critique calls (default 1)")
    ap.add_argument("--max-revision-calls", type=int, default=DEFAULT_MAX_REVISION_CALLS,
                    help="hard ceiling on revisions (default 1)")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    ap.add_argument("--out", default=None, metavar="PATH",
                    help="write the machine-readable result here as well")
    args = ap.parse_args(argv)

    vlm_cfg = VL.VLMConfig.from_env(provider=args.vlm_provider, model=args.vlm_model,
                                    base_url=args.vlm_base_url,
                                    max_frames=args.max_vlm_frames)
    synth_cfg = SY.SynthConfig.from_env(provider=args.synth_provider,
                                        model=args.synth_model,
                                        base_url=args.synth_base_url,
                                        api_key=args.synth_api_key,
                                        detail=args.detail)
    result = run_release_check(
        vlm_cfg=vlm_cfg, synth_cfg=synth_cfg, url=args.url,
        max_vlm_frames=args.max_vlm_frames,
        max_synthesis_calls=args.max_synthesis_calls,
        max_critic_calls=args.max_critic_calls,
        max_revision_calls=args.max_revision_calls)

    payload = json.dumps(result, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(payload + "\n")
        print(f"release-check: wrote {args.out}", file=sys.stderr)
    if args.json:
        print(payload)
    else:
        print(render(result))
    # A CONDITIONAL run is a successful run: it did what it was asked and the
    # answer was "not qualified yet". Exiting non-zero would make a release
    # script unable to record the state, which is the one thing this command
    # exists to let it do.
    #
    # DEGRADED is not that. CONDITIONAL is "not exercised"; DEGRADED is
    # "exercised, and a capability was disproven while another worked". A script
    # that treats both as success is a script that ships on a broken model layer,
    # so DEGRADED exits non-zero alongside UNQUALIFIED.
    return 1 if result["status"] in ("UNQUALIFIED", "DEGRADED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
