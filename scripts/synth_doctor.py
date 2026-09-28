"""`synthesis-doctor` and `critic-doctor` — can this model actually write this report?

`vlm-doctor` exists because one HTTP 200 could not tell you which of seven
things had gone wrong. The same is true here, and the failure list is longer:
a synthesis provider can be unreachable, unauthenticated, **authenticated but out
of credits**, pointed at a model that does not exist, answering in prose where a
schema was requested, returning something that is not JSON at all, returning JSON
that is not the report, or returning a report whose sentences are not supported by
the evidence they cite. Seven different problems, seven different fixes, and every
one of them arrives as an exception with a stack trace and no indication of which
it was. Collapsing them into "synthesis failed" is why real qualification kept
looking like a provider problem when it was a schema problem.

So this walks a chain and reports every step separately — the same shape as
`vlm_doctor`, and deliberately the same *code*:

    reachable               the endpoint answered at all
    authenticated           the credential was accepted
    account_paid            the account could pay for the call (a 429 that means
                            billing is a different failure from a 429 that means
                            slow down, and they must never share a verdict)
    model_exists            the named model resolved
    model_completes         the call returned prose or JSON rather than an error
    structured_requested    a schema constraint was sent and accepted
    structured_enforced     the returned payload actually carried every required key
    draft_valid             the payload parsed into a schema-valid draft
    draft_grounded          every unit cites resolvable evidence and survives the
                            blocking grounding checks
    report_validated        the rendered report passes `report_validate`

The last three are the ones that get conflated. A gateway that accepts
`response_format: json_schema`, returns HTTP 200, and answers in prose has
succeeded at six of the ten steps and produced nothing usable.

`qualified` is true only when all ten hold. HTTP 200 is never a verdict.

    uv run scripts/synth_doctor.py                       # fixture evidence, one call
    uv run scripts/synth_doctor.py <artifact-dir>         # that video's evidence
    uv run scripts/synth_doctor.py --json                 # machine-readable

    uv run scripts/critic_doctor.py <artifact-dir>        # four known-answer cases

Exit 0 only for a qualified provider, 1 for a reachable-but-unqualified one, and
3 for a configuration that never reached a call.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import chat_json as CJ
import report_fixtures as RF
import synth_model as SM
import synth_validate as SV
import synthesis as SY
from evidence_model import Evidence
from report_validate import validate_report
from synthesis import Budget, BudgetExhausted
from vlm_doctor import DoctorCheck, DoctorReport

#: The chain, in diagnostic order. Nothing after a false step can be true, which
#: is what makes a top-to-bottom read of the output a diagnosis.
SYNTH_CHECKS = (
    "reachable",
    "authenticated",
    "account_paid",
    "model_exists",
    "model_completes",
    "structured_requested",
    "structured_enforced",
    "draft_valid",
    "draft_grounded",
    "report_validated",
)

#: The one call this is allowed to make. A provider smoke test that processed a
#: whole video would cost a video's worth of tokens to answer a question a
#: forty-line document answers just as well — and would fail for a dozen reasons
#: that have nothing to do with the synthesis model.
MAX_CALLS = 1

#: The response budget for a *probe*. The writer's own default is 8000 tokens,
#: which is right for a report and wrong here: this asks for a brief report over
#: a forty-line document, and asking a credit-limited account for four times what
#: that needs turns an affordable check into an `out_of_credits` verdict that
#: says nothing about the model. The negotiator still raises the ceiling on its
#: own if the answer is truncated, and that retry is reported like any other
#: call, so a small default cannot hide a model that needs the room.
DEFAULT_MAX_TOKENS = 1500


def load_evidence(artifact_dir: str | Path | None) -> tuple[Evidence, str]:
    """The document under test, and where it came from.

    A validated `evidence.json` when a directory is given; the built-in
    qualification fixture otherwise. The fixture is not a convenience — it is
    the only way to run this command with no video, and it is the same document
    every run, which is what makes two runs comparable.
    """
    if artifact_dir is None:
        return RF.synthesis_qualification(), "built-in qualification fixture"
    d = Path(artifact_dir).expanduser()
    path = d if d.is_file() else d / "evidence.json"
    if not path.is_file():
        raise FileNotFoundError(f"no evidence.json at {path}")
    ev = SY.load_valid_evidence(path)
    return ev, str(path)


def run_synthesis_doctor(ev: Evidence, cfg: SY.SynthConfig,
                         transport=None, budget: Budget | None = None,
                         max_tokens: int = DEFAULT_MAX_TOKENS) -> tuple[int, DoctorReport]:
    """One real call, walked step by step. `(exit_code, report)`."""
    budget = budget or Budget(MAX_CALLS)
    report = DoctorReport(cfg, SYNTH_CHECKS)
    report.extra = {"source": "synth_doctor",
                    "prompt_version": SM.PROMPT_VERSION,
                    "synth_schema_version": SM.SYNTH_SCHEMA_VERSION,
                    "evidence_items": len(ev.items),
                    "max_tokens": max_tokens,
                    "calls": budget.to_dict()}

    # ---- configuration. Nothing below this is knowable, so nothing is claimed.
    if not cfg.provider or cfg.provider == "none":
        report.failed("reachable", "no synthesis model configured; set SYNTH_PROVIDER "
                                   "or pass --synth-provider", stage="config")
        report.error = ("nothing was called. Set SYNTH_PROVIDER to one of: openai, "
                        "openrouter, ollama, lmstudio, llamacpp, custom")
        return 3, report
    base_url = SY._resolve_base(cfg)
    if not base_url:
        report.failed("reachable", f"provider {cfg.provider!r} has no base URL; "
                                   "set SYNTH_BASE_URL or pass --synth-base-url",
                      stage="config")
        report.error = "the provider alias resolves to no endpoint, so no call was possible"
        return 3, report
    api_key = cfg.api_key or SY._resolve_key(cfg)
    keyless = cfg.provider in ("ollama", "lmstudio", "llamacpp", "custom")
    if not api_key and not keyless:
        report.failed("authenticated", f"provider {cfg.provider!r} needs a credential "
                                       "(SYNTH_API_KEY) and none is set", stage="config")
        report.error = "no call was made: the provider was never given a credential"
        return 3, report

    model = cfg.model or SY._default_model(cfg)
    client = SM.SynthesisClient(model=model, base_url=base_url, api_key=api_key or "",
                                timeout=cfg.timeout, transport=transport)
    prompt = SY._plan_prompt(ev, cfg, SY.build_plan(ev, cfg))

    # ---- the one call.
    try:
        budget.spend()
    except BudgetExhausted as exc:
        report.failed("reachable", str(exc), stage="budget")
        report.error = "the call budget was already spent; no call was made"
        return 3, report

    started = time.monotonic()
    try:
        result = client.draft(prompt, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 — any escape is itself a verdict
        report.latency_s = time.monotonic() - started
        report.failed("reachable", f"the call raised {type(exc).__name__}: {exc}",
                      stage="transport")
        report.error = "the endpoint could not be reached"
        return 3, report
    report.latency_s = time.monotonic() - started
    report.extra["calls"] = budget.to_dict()
    report.extra["result"] = result.to_dict()

    # ---- walk the chain from the one exchange. Every verdict below is derived
    # from this same call rather than from a separate probe, so a health check
    # can never disagree with the request the pipeline would actually make.
    _classify_result(report, result, client, model, base_url)
    if not report.checks.get("model_completes") or not report.checks["model_completes"].ok:
        return 1, report

    # ---- structure, parse, ground, render. Each is a separate verdict because
    # they fail separately and a reader needs to know which one did.
    if not report.checks.get("structured_enforced") or not report.checks["structured_enforced"].ok:
        return 1, report

    problems = SM.validate_draft(result.doc) if isinstance(result.doc, dict) else ["no draft"]
    report.set("draft_valid", not problems,
               "every required key present and well-formed" if not problems
               else "; ".join(problems[:3]))
    if problems:
        report.error = ("the model answered in the right shape of JSON and the wrong "
                        "shape of draft. It stays unqualified; the schema is not loosened "
                        "to accommodate it.")
        return 1, report

    validation = SV.validate_draft_grounding(result.doc, ev)
    detail = ("every unit cites resolvable evidence; numbers, URLs and conflicts check out"
              if validation.valid
              else f"{len(validation.blocking)} blocking: "
                   + "; ".join(f"{f.code} at {f.where}" for f in validation.blocking[:3]))
    report.set("draft_grounded", validation.valid, detail)
    if not validation.valid:
        report.error = ("the draft parsed and then failed the blocking grounding checks. "
                        "A model that will not hold to the evidence stays unqualified.")
        return 1, report

    try:
        rendered = SY.render_draft(ev, result.doc, cfg, SY.build_plan(ev, cfg))
        from report import render_markdown
        result_report = validate_report(rendered, ev, render_markdown(rendered),
                                        authored_by="synthesis")
    except Exception as exc:  # noqa: BLE001 — a render failure is a verdict too
        report.failed("report_validated", f"the draft could not be rendered: {exc}",
                      stage="render")
        report.error = "the draft is valid and grounded but does not render into a report"
        return 1, report
    report.set("report_validated", result_report.valid,
               f"grounding validator passed · {result_report.metrics.get('report_word_count')} words"
               if result_report.valid
               else "; ".join(f"{e['code']} at {e['where']}" for e in result_report.blocking_errors[:3]))
    report.observation = {
        "content_type": "synthesis",
        "confidence": None,
        "summary": _first_text(result.doc),
        "counts": {"summary": len(result.doc.get("summary") or []),
                   "takeaways": len(result.doc.get("takeaways") or []),
                   "sections": len(result.doc.get("sections") or []),
                   "units": sum(1 for _ in SM.units(result.doc))},
    }
    if not result_report.valid:
        report.error = "the report validator refused the rendered report"
        return 1, report
    return 0, report


def _first_text(doc) -> str:
    if not isinstance(doc, dict):
        return ""
    for _where, unit in SM.units(doc):
        if isinstance(unit, dict) and unit.get("text"):
            return str(unit["text"])[:400]
    return ""


def _classify_result(report: DoctorReport, result, client, model: str, base_url: str) -> None:
    """Turn one failed result into the right pair of verdicts.

    Split out because the mapping from a transport error to a position in the
    chain is the whole point of this command, and burying it in the middle of the
    flow is how it ends up wrong for a status nobody thought of.
    """
    report.set("reachable", True, f"{base_url} answered")
    reason = (result.error or "").lower()
    status = result.status

    if status == "out_of_credits":
        report.failed("account_paid", result.error or "the account cannot pay for this call",
                      stage="billing")
        report.error = ("the credential was accepted and the account cannot pay. This is "
                        "not a rate limit and waiting will not help.")
        return
    report.set("account_paid", True, "the call was billed")

    if status == "unavailable" or "401" in reason or "403" in reason or "authentication" in reason:
        report.failed("authenticated", result.error or "the credential was refused",
                      stage="auth")
        report.error = "the endpoint answered and refused the credential"
        return
    report.set("authenticated", True, "the credential was accepted")

    if "404" in reason or "not found" in reason or "does not exist" in reason:
        report.failed("model_exists", result.error or f"the endpoint does not resolve {model!r}",
                      stage="model")
        report.error = f"the endpoint does not resolve model {model!r}"
        return
    report.set("model_exists", True, f"{model} resolved")

    if status in ("unavailable", "failed") and "response_format" in reason:
        # The endpoint refused the structured-output parameter specifically. The
        # negotiator has already walked its ladder; arriving here means every
        # mode was refused, which is a capability the model does not have here.
        #
        # `model_completes` is FALSE here, and that is the substantive change:
        # the check means "the model returned prose or JSON rather than an
        # error", and this is a 4xx with zero choices and zero message content.
        # It used to be recorded as a pass on the strength of the endpoint
        # having replied at all — which is the transport answering, not the model
        # completing, and the two are exactly what the status hierarchy exists
        # to keep apart. The capability finding is unchanged and still reported:
        # the endpoint cannot be trusted to this contract, and `report.error`
        # says so.
        report.set("model_completes", False,
                   "the endpoint answered with an error and emitted no content")
        report.set("structured_requested", False,
                   "the endpoint rejected the schema constraint itself")
        report.failed("structured_enforced", result.error or "the endpoint rejected the schema",
                      stage="structured")
        report.error = ("the endpoint rejected the structured-output parameter on every "
                        "mode. It answers text but not this contract.")
        return
    if status == "failed":
        report.failed("model_completes", result.error or "the call failed", stage="call")
        report.error = "the model returned no usable content"
        return

    # Past here the model completed: something came back that was not a transport
    # or account failure. Whether it is the *right* thing is the schema's verdict,
    # and keeping the two apart is what lets a reader tell "the model cannot do
    # this" from "the model would not do this".
    report.set("model_completes", True, f"{model} returned {result.status}")

    if status == "malformed":
        mode = result.mode or "none"
        report.set("structured_requested", bool(result.mode), f"constraint sent: {mode}")
        report.failed("structured_enforced", result.error or "the answer was not a JSON object",
                      stage="schema")
        report.error = ("the model answered, but not in the schema. It is not coerced into "
                        "one — a repaired guess is how a fabricated table reaches a report.")
        return

    mode = result.mode
    report.set("structured_requested", bool(mode),
               f"constraint sent: {mode}" if mode
               else "no structured-output constraint was accepted or sent")
    if not result.enforced:
        report.failed("structured_enforced",
                      f"the endpoint accepted {mode or 'the request'} but the answer did "
                      f"not follow the schema", stage="schema")
        report.error = ("structured output is not enforced on this endpoint, so nothing "
                        "downstream can be trusted to the contract")
        return
    report.set("structured_enforced", True, f"the {mode} response carried every required key")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _synth_config(args) -> SY.SynthConfig:
    return SY.SynthConfig.from_env(provider=args.synth_provider, model=args.synth_model,
                                   base_url=args.synth_base_url, api_key=args.synth_api_key,
                                   timeout=args.synth_timeout, detail=args.detail,
                                   output_language=args.output_language, critic=False)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="synthesis-doctor",
        description="Check whether the configured model can actually write a report "
                    "from evidence: one call, every step reported separately.")
    ap.add_argument("artifact_dir", nargs="?", default=None,
                    help="a directory holding a validated evidence.json; omit to use the "
                         "built-in qualification fixture")
    ap.add_argument("--synth-provider", default=None)
    ap.add_argument("--synth-model", default=None)
    ap.add_argument("--synth-base-url", default=None)
    ap.add_argument("--synth-api-key", default=None)
    ap.add_argument("--synth-timeout", type=float, default=None)
    ap.add_argument("--detail", default="brief", choices=["brief", "balanced", "deep"],
                    help="the report length the model is asked for (default brief, "
                         "because this is a qualification probe and not a report)")
    ap.add_argument("--output-language", default="en",
                    help="language for the drafted prose (default en, so a provider that "
                         "silently answers in another language is visible)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help=f"response budget for the probe, in tokens (default "
                         f"{DEFAULT_MAX_TOKENS}; the writer's own default is 8000, "
                         "which is a report's budget and not a probe's)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)

    cfg = _synth_config(args)
    try:
        ev, source = load_evidence(args.artifact_dir)
    except (FileNotFoundError, SY.InvalidEvidence) as exc:
        print(f"synthesis-doctor: {exc}", file=sys.stderr)
        return 3

    code, report = run_synthesis_doctor(ev, cfg, max_tokens=args.max_tokens)
    report.extra["evidence_source"] = source
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        sys.stdout.write(report.render(observation_title="draft"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
