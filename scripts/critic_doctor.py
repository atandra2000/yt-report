"""`critic-doctor` — can this model find a planted defect, and stay quiet otherwise?

The synthesis doctor asks whether a model can write a report. This asks the
complementary question, and it is the harder one to answer honestly: a critic
that returns `{"findings": []}` for everything is trivially safe and useless,
and a critic that returns forty findings for everything is the same failure
wearing the opposite costume. There is no single document that separates those
two. What separates them is a *set* of documents with known answers — one that is
clean, and three with exactly one defect planted in each.

    A. clean                the draft is grounded, hedged and consistent.
                            Expected: no findings. Anything it reports here is a
                            false positive, and that is the whole measurement.
    B. changed number       `40 ms` becomes `90 ms` in the prose while the
                            evidence still says 40.      Expected: number-error.
    C. removed qualifier    a `hedged` item's "could" is dropped. A hedge removed
                            is a fabrication that passes every structural check.
                            Expected: missing-qualifier.
    D. unsupported sentence a sentence asserting something the cited items do not
                            say.                        Expected: unsupported-claim.

The four are four calls, which is why the budget here is four and not one: the
question is a set of answers, and one call answers one of them. The budget is a
hard ceiling that reports which cases did not run — a skipped case is printed as
skipped, never as a pass, because a critic that was never asked has not been shown
to be quiet.

The validator remains the final authority throughout. Nothing here weakens a
check, and a critic that objects to a report the validator accepted does not get
to un-write it: the writer may revise once on the critic's advice and the
validator then decides. This command only measures whether the advice is worth
listening to.

    uv run scripts/critic_doctor.py                     # built-in tiny fixture
    uv run scripts/critic_doctor.py <artifact-dir>      # that video's evidence
    uv run scripts/critic_doctor.py --json

Exit 0 when every case that ran matched its expected answer. 1 when any ran case
did not. 3 when no call was made.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import synth_model as SM
import synthesis as SY
from report_validate import validate_report
from synth_doctor import Budget, BudgetExhausted
from vlm_doctor import DoctorCheck, DoctorReport

#: The chain this walks. It is shorter than the synthesis doctor's because the
#: transport questions are the same ones that doctor already answers — a critic
#: that cannot be reached is not a defect in the critic.
CRITIC_CHECKS = (
    "reachable",
    "authenticated",
    "account_paid",
    "model_exists",
    "model_completes",
    "structured_enforced",
    "critic_reachable",
    # The measurement itself, as a step rather than as the absence of an error.
    # `qualified` is computed from this chain, so without it a critic that
    # answered every call and matched none of the expected answers would read as
    # QUALIFIED — a run that reached the endpoint and proved nothing.
    "cases_matched",
)

#: One call per case. Not a budget the user sets by default: the number of cases
#: *is* the number of calls, and a default lower than that would silently skip
#: the measurement it exists to make.
DEFAULT_CALL_BUDGET = 4

#: Response budget for one critique. The client's own default is 4000, sized for
#: a full critique of a full report; a four-case probe asks four short questions
#: and should not need a fifth of the ceiling. Same reasoning as the synthesis
#: doctor's `--max-tokens`: a probe that is too expensive to run is a probe that
#: reports the provider's billing rather than the critic's quality.
DEFAULT_MAX_TOKENS = 1200

#: A finding's `code` is free text — the critic chooses it — so a case accepts any
#: code carrying one of these words. Narrow enough to be a real match, loose
#: enough not to fail a model that wrote `number-mismatch` for `number-error`.
_EXPECTED = {
    "changed_number": ("number", "figure", "mismatch", "incorrect"),
    "removed_qualifier": ("qualifier", "hedge", "hedg", "certainty", "modal"),
    "unsupported_sentence": ("unsupported", "not stated", "no support", "external",
                             "fabricat", "not in the evidence"),
}


# --------------------------------------------------------------------------- #
# the four cases
# --------------------------------------------------------------------------- #

def _base_draft(ev) -> dict:
    """A draft that is small, valid, and grounded — the thing the critic judges.

    Built from the fixture's own evidence rather than written free-hand, so every
    sentence in it is one the validator accepts, and asserted clean by
    `validate_draft_grounding` at import time by the self-test. That assertion is
    the point: a case whose *baseline* is defective measures the critic's
    tolerance of defects rather than its ability to find one, and every verdict
    after it would be about the baseline.

    Three of its constraints come from the validator rather than from taste, and
    each is worth naming because they are the rules the mutations then break:

    * a conflict the draft references must be represented in full somewhere, and
      a block may not state a disputed item as settled;
    * a sentence describing something visual must cite an item from a frame; and
    * a number in prose is resolved against the cited items, so `p99` reads as the
      figure `99` unless a cited item literally contains it.
    """
    claim = next(i for i in ev.items if i.type == "claim")
    num = next(i for i in ev.items if i.type == "number")
    hedged = next((i for i in ev.items if i.status == "hedged"), None)
    visual = next((i for i in ev.items if i.type == "visual_finding"), None)

    # Both sides of the conflict, in one block. A draft that cites one side and
    # discloses the other elsewhere is equally acceptable; what this must not
    # have is a half-represented conflict or a block stating a disputed item as
    # settled.
    paired = [claim.id] + ([visual.id] if visual else [])
    hedged_ids = [hedged.id] if hedged else []

    figure = num.display or num.value
    return {
        "summary": [{
            "text": ("The video puts a benchmark claim to the viewer and qualifies it: it "
                     "is presented as the vendor's own reported result, and the frame "
                     "shows the two figures without any methodology behind them."),
            "evidence_ids": paired + hedged_ids,
            "source_kind": "mixed",
        }],
        "takeaways": [{
            "text": f"The stated latency figure is {figure}.",
            "evidence_ids": [num.id],
            "source_kind": "claim",
        }],
        "sections": [{
            "title": "What the video claims",
            "paragraphs": [{
                "text": ("The speaker describes the figure as the vendor's own benchmark "
                         "rather than as an independently measured result, and says the "
                         "rest of the comparison could not be read."),
                "evidence_ids": paired + hedged_ids,
                "source_kind": "claim",
            }],
        }],
    }


def _mutate_changed_number(draft: dict, ev) -> tuple[dict, str]:
    """Change a figure in the prose. The evidence is untouched; the sentence lies."""
    d = copy.deepcopy(draft)
    num = next(i for i in ev.items if i.type == "number")
    original = num.display or num.value
    for _where, unit in SM.units(d):
        if original in unit["text"]:
            unit["text"] = unit["text"].replace(original, "90 ms", 1)
            return d, f"the figure {original} was changed to 90 ms in the takeaway"
    return copy.deepcopy(draft), "no figure found in the draft to change"


def _mutate_removed_qualifier(draft: dict, ev) -> tuple[dict, str]:
    """Drop a hedge. Every word that remains is in the evidence; the claim is not."""
    d = copy.deepcopy(draft)
    hedged = next((i for i in ev.items if i.status == "hedged"), None)
    if hedged is None:
        return d, "the evidence has no hedged item to qualify"
    flat = ("is a benchmark result the vendor confirms and the on-screen table is the "
            "independent verification of it")
    for _where, unit in SM.units(d):
        if "qualifies it" in unit["text"]:
            unit["text"] = unit["text"].replace(
                "puts a benchmark claim to the viewer and qualifies it: it is presented as "
                "the vendor's own reported result, and the frame shows the two figures "
                "without any methodology behind them", flat)
            return d, ("the summary's hedge ('qualifies it', 'presented as the vendor's own "
                       "reported result') was replaced with a flat assertion")
    return d, "the hedged sentence was not found in the draft"


def _mutate_unsupported_sentence(draft: dict, ev) -> tuple[dict, str]:
    """Add a sentence no cited item supports, and cite the item it does not come from."""
    d = copy.deepcopy(draft)
    num = next(i for i in ev.items if i.type == "number")
    d["takeaways"].append({
        "text": ("This is widely considered the fastest configuration available today, "
                 "and it is the one most teams adopt."),
        "evidence_ids": [num.id],
        "source_kind": "inference",
    })
    return d, ("a takeaway asserting a market-wide judgement was added, citing an item "
               "that says nothing of the kind")


#: `(key, title, mutate, expects_findings)`. The order is A, B, C, D from the
#: brief, and the clean case is first on purpose: everything after it is
#: meaningless if the critic reports a defect in a sound draft.
CASES = (
    ("clean", "A. clean draft", None, False),
    ("changed_number", "B. a figure changed in the prose", _mutate_changed_number, True),
    ("removed_qualifier", "C. a hedge removed", _mutate_removed_qualifier, True),
    ("unsupported_sentence", "D. an unsupported sentence", _mutate_unsupported_sentence, True),
)


def _matches(case_key: str, findings: list[dict]) -> bool:
    if case_key == "clean":
        return not findings
    words = _EXPECTED[case_key]
    return any(any(w in str(f.get("code", "")).lower() for w in words) for f in findings)


def run_critic_doctor(ev, cfg: SY.SynthConfig, transport=None,
                      budget: Budget | None = None,
                      max_tokens: int = DEFAULT_MAX_TOKENS) -> tuple[int, DoctorReport]:
    """Four known-answer cases. `(exit_code, report)`."""
    budget = budget or Budget(DEFAULT_CALL_BUDGET)
    report = DoctorReport(cfg, CRITIC_CHECKS)
    report.extra = {"source": "critic_doctor", "max_tokens": max_tokens,
                    "calls": budget.to_dict(), "cases": []}

    if not cfg.provider or cfg.provider == "none":
        report.failed("reachable", "no synthesis model configured; set SYNTH_PROVIDER "
                                   "or pass --synth-provider", stage="config")
        report.error = ("nothing was called. The critic runs on the synthesis model, so "
                        "there is no separate provider to configure.")
        return 3, report
    base_url = SY._resolve_base(cfg)
    if not base_url:
        report.failed("reachable", f"provider {cfg.provider!r} has no base URL",
                      stage="config")
        return 3, report
    api_key = cfg.api_key or SY._resolve_key(cfg)
    if not api_key and cfg.provider not in ("ollama", "lmstudio", "llamacpp", "custom"):
        report.failed("authenticated", "the provider needs a credential (SYNTH_API_KEY)",
                      stage="config")
        return 3, report

    model = cfg.model or SY._default_model(cfg)
    client = SM.SynthesisClient(model=model, base_url=base_url, api_key=api_key or "",
                                timeout=cfg.timeout, transport=transport)
    base_draft = _base_draft(ev)
    results: list[tuple[str, str, str, list[dict], float, bool, bool, str, bool]] = []

    for key, title, mutate, expects_findings in CASES:
        doc = base_draft if mutate is None else mutate(base_draft, ev)[0]
        note = "" if mutate is None else mutate(base_draft, ev)[1]
        try:
            budget.spend()
        except BudgetExhausted:
            # Printed as `not run`, never as a pass. An unasked question has no
            # answer, and reporting it as a clean result is the exact failure
            # this command exists to prevent.
            results.append((key, title, "not run — the call budget was spent", [], 0.0,
                            False, False, "not run", False))
            continue
        started = time.monotonic()
        try:
            res = client.critique(SY._critique_prompt(ev, doc), max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001 — any escape is a verdict
            results.append((key, title, f"the call raised {type(exc).__name__}: {exc}",
                            [], time.monotonic() - started, False, False, "no answer", False))
            continue
        elapsed = time.monotonic() - started
        if not res.ok:
            # The critic did not answer. That is not the same as the critic
            # answering "there is nothing wrong", and scoring it as though it
            # were is the one conflation this command exists to prevent: four
            # billing refusals would otherwise be reported as a quiet critic on a
            # sound draft, plus three misses — five findings about the model,
            # none of them true.
            results.append((key, title,
                            (res.error or f"the critic returned {res.status}")[:200],
                            [], elapsed, False, False, "no answer", False))
            continue
        # The shape is checked before the content is scored. A finding without a
        # `code` cannot be matched to an expectation, and counting it as a
        # finding would turn "the critic used a different vocabulary" into "the
        # critic objected" — which is how a live run reported two genuine finds
        # as four `[None] None: None` rows and a 0/4 score.
        #
        # `answered` records that the endpoint returned a critique at all; the
        # schema verdict is carried separately and is the only thing
        # `structured_enforced` reads. The two used to be one flag, which is how
        # one run printed `structured_enforced OK` directly above
        # `findings[0] missing code`: a response the validator had already
        # rejected was counted as evidence that the validator had passed.
        problems = SM.validate_critique(res.doc)
        if problems:
            results.append((key, title,
                            "the critique was not in the schema: " + "; ".join(problems[:2]),
                            [], elapsed, False, True, "malformed", False))
            continue
        findings = [f for f in (res.doc.get("findings") or []) if isinstance(f, dict)]
        detail = note or "the draft is grounded, hedged and consistent"
        # Two separate facts, kept as two: `res.ok` is whether the critic
        # answered at all, `_matches` is whether the answer was the right one.
        # Collapsing them made a critic that returned valid JSON saying nothing
        # score the same as one that found the planted defect.
        results.append((key, title, detail, findings, elapsed,
                        _matches(key, findings), True,
                        "findings" if findings else "none", True))

    skipped = "not run — the call budget was spent"
    report.extra["calls"] = budget.to_dict()
    report.extra["cases"] = [
        {"case": k, "title": t, "note": n, "findings": f, "latency_s": round(l, 3),
         "expected": "findings" if k in _EXPECTED else "none",
         "observed": observed, "answered": answered_, "schema_valid": schema_ok,
         "passed": p, "ran": n != skipped}
        for k, t, n, f, l, p, answered_, observed, schema_ok in results]

    ran = [r for r in results if r[2] != skipped]
    # `results` rows are
    # (key, title, note, findings, latency, passed, answered, observed, schema_valid).
    if not ran:
        report.failed("reachable", "no case was run", stage="budget")
        report.error = "the call budget was spent before any case ran"
        return 3, report

    # The transport verdicts, taken from the first call that reached the endpoint.
    answered = [r for r in ran if r[6]]
    first_ok = bool(answered)
    report.set("reachable", True, f"{base_url} answered")
    if first_ok:
        report.set("authenticated", True, "the credential was accepted")
        report.set("account_paid", True, "the call was billed")
        report.set("model_exists", True, f"{model} resolved")
        report.set("model_completes", len(answered) == len(ran),
                   f"the model returned a critique for {len(answered)}/{len(ran)} case(s)")
    else:
        report.failed("authenticated", "no case returned a critique", stage="call")
        report.error = "the critic never returned a usable answer"
    # `structured_enforced` answers one question — did every response satisfy the
    # *complete* critique schema, nested findings included — and `validate_critique`
    # is the only thing that answers it. A case that failed it contributes
    # `malformed`, not a pass, and the cases after it are unreachable rather than
    # wrong: a finding that could not be read cannot be matched to an expectation,
    # and reporting that as a miss would be a second, different claim about the
    # same response.
    malformed = [r for r in ran if not r[8]]
    if malformed:
        report.failed(
            "structured_enforced",
            f"{len(malformed)} of {len(ran)} critique(s) failed the schema: "
            + "; ".join(str(r[2]) for r in malformed[:2])[:200],
            stage="schema")
    else:
        report.set("structured_enforced", True,
                   "every critique response satisfied the full critique schema, findings included")
    report.set("critic_reachable", first_ok,
               f"the model answered {len(answered)}/{len(ran)} case(s)")
    matched = sum(1 for r in ran if r[5])
    complete = len(ran) == len(CASES)
    if malformed:
        report.error = ("the critic's answers were not in the schema, so whether they "
                        "matched the expected findings is unknown")
        return 1, report
    report.set("cases_matched", matched == len(ran) and complete,
               f"{matched}/{len(CASES)} case(s) matched the expected answer"
               + ("" if complete else f"; only {len(ran)} ran"))

    passed = [r for r in ran if r[5]]
    if len(passed) != len(ran):
        wrong = [r[0] for r in ran if not r[5]]
        report.error = ("the critic did not match the expected answer for: "
                        + ", ".join(wrong))
        return 1, report
    if len(ran) < len(CASES):
        report.error = (f"{len(CASES) - len(ran)} case(s) did not run within the budget "
                        "and prove nothing")
        return 1, report
    return 0, report


def render(report: DoctorReport) -> str:
    """The chain, then one line per case. The case table is the measurement."""
    out = [report.render(observation_title="critic").rstrip("\n")]
    for c in report.extra.get("cases", []):
        mark = "ok  " if c["passed"] else ("SKIP" if not c["ran"] else "FAIL")
        out.append(f"    {c['case']:<20} {mark}  expected {c['expected']}, "
                   f"observed {c['observed']}"
                   + (f"  ·  {c['note']}" if c.get("note") else ""))
        for f in c["findings"][:3]:
            out.append(f"        [{f.get('code')}] {f.get('where')}: {f.get('message')}")
    calls = report.extra.get("calls", {})
    out.append(f"    {'calls':<20}       {calls.get('spent')}/{calls.get('limit')}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="critic-doctor",
        description="Check whether the configured model finds a planted defect and "
                    "stays quiet about a sound draft. Four known-answer cases.")
    ap.add_argument("artifact_dir", nargs="?", default=None,
                    help="a directory holding a validated evidence.json; omit to use the "
                         "built-in qualification fixture")
    ap.add_argument("--synth-provider", default=None)
    ap.add_argument("--synth-model", default=None)
    ap.add_argument("--synth-base-url", default=None)
    ap.add_argument("--synth-api-key", default=None)
    ap.add_argument("--synth-timeout", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help=f"response budget per critique, in tokens (default "
                         f"{DEFAULT_MAX_TOKENS}; the client's own default is 4000, "
                         "which is a full critique of a full report)")
    ap.add_argument("--max-critic-calls", type=int, default=DEFAULT_CALL_BUDGET,
                    help=f"hard ceiling on model calls (default {DEFAULT_CALL_BUDGET}, one "
                         "per case; a lower value skips cases and they are reported as "
                         "not run rather than as passing)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    cfg = SY.SynthConfig.from_env(provider=args.synth_provider, model=args.synth_model,
                                   base_url=args.synth_base_url, api_key=args.synth_api_key,
                                   timeout=args.synth_timeout, critic=True)
    from synth_doctor import load_evidence
    try:
        ev, source = load_evidence(args.artifact_dir)
    except (FileNotFoundError, SY.InvalidEvidence) as exc:
        print(f"critic-doctor: {exc}", file=sys.stderr)
        return 3

    code, report = run_critic_doctor(ev, cfg, budget=Budget(args.max_critic_calls),
                                     max_tokens=args.max_tokens)
    report.extra["evidence_source"] = source
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        sys.stdout.write(render(report))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
