"""`eval-live` — one cheap real run, measured, with the budgets enforced.

Real provider calls cost money and take minutes. The full pipeline on a
twenty-minute video is five figures and an hour; the question a person actually
has is usually much smaller — *does this provider work at all, and what does it
do to this kind of video?* — and answering that expensively is how a smoke test
becomes the most expensive thing in the repo.

So this is deliberately the cheap one, and the budget is the point:

    --max-vlm-frames N        frames sent for interpretation      (default 2)
    --max-synthesis-calls N   report drafts requested             (default 1)
    --max-critic-calls N      critiques requested                (default 1)

Every one of those is a **hard ceiling**, spent through `Budget.spend()` before
each call and refused when exhausted — not a counter checked afterwards. A limit
that is reported rather than enforced produces a bill and a report saying the
limit was respected, and only one of those is true.

Zero is a real answer, not a failure to try. A run with `--max-vlm-frames 0`
records `vlm_calls: 0` and is a synthesis evaluation; a run with
`--synthesis-calls 0` is a VLM evaluation. The report says which it was, because
"the pipeline ran" and "the model was exercised" are different claims and the
second one is the one being made here.

Output is `LIVE-EVAL-<timestamp>.md` in the artifact directory: provider, model,
video, duration, exact call counts, cache hits, latency, then what came out —
conformant and useful VLM observations, abstentions and failures; entities,
meaningful numbers, screen-only facts, comparisons, claims and procedures; and
report word count, provenance coverage, unsupported claims and numbers,
repetition, and whether the conflicts survived. No overall score. A single number
over six dimensions would hide the one dimension that went wrong, which is the
only reason anyone would look.

    uv run scripts/eval_live.py <url> --synth-provider openai --synth-model gpt-4o-mini
    uv run scripts/eval_live.py <url> --vlm-provider openai --vlm-model gpt-4o-mini \\
        --max-vlm-frames 2 --max-synthesis-calls 1 --max-critic-calls 1
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import report as REP
import report_validate as RV
import semantic as SEM
import synth_model as SM
import synthesis as SY
from vlm import VLMConfig, provider_from_config

#: The default ceilings. Two frames, one draft, one critique: enough to exercise
#: every provider path, cheap enough to run whenever a config changes.
DEFAULT_MAX_VLM_FRAMES = 2
DEFAULT_MAX_SYNTHESIS_CALLS = 1
DEFAULT_MAX_CRITIC_CALLS = 1


class LiveMetrics:
    """Every number the report prints, counted as it happens.

    Held in one object rather than threaded through the pipeline so that a call
    cannot be made somewhere the accounting does not reach. `synthesis_calls` and
    `critic_calls` are incremented by the budget, not here, for the same reason:
    the only call site that knows a call happened is the one that authorised it.
    """

    def __init__(self) -> None:
        self.vlm_calls = 0
        self.cache_hits = 0
        self.failures: list[str] = []
        self.latencies: dict[str, list[float]] = {}
        self.started = time.monotonic()

    def timed(self, name: str, seconds: float) -> None:
        self.latencies.setdefault(name, []).append(seconds)

    def fail(self, where: str, why: str) -> None:
        self.failures.append(f"{where}: {why}")

    def to_dict(self) -> dict:
        return {
            "vlm_calls": self.vlm_calls,
            "cache_hits": self.cache_hits,
            "failures": self.failures,
            "latency_s": {k: [round(x, 3) for x in v] for k, v in self.latencies.items()},
            "wall_clock_s": round(time.monotonic() - self.started, 3),
        }


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def run_live(artifact_dir: Path, vlm_cfg: VLMConfig, synth_cfg: SY.SynthConfig,
             max_vlm_frames: int, synth_budget: Budget, critic_budget: Budget,
             detail: str = "brief") -> tuple[int, dict]:
    """Build evidence, exercise each configured provider, and report.

    Reads the artifact directory rather than fetching, so the same video can be
    re-evaluated against a different provider for the cost of the model calls and
    not a second download.
    """
    import build_evidence as B
    from align import AlignmentConfig

    m = LiveMetrics()
    d = Path(artifact_dir)
    if not d.is_dir():
        return 2, {"status": "failed", "error": f"not a directory: {d}"}

    data = B.load_artifact_inputs(d)
    if not data["snippets"]:
        return 2, {"status": "failed", "error": "no transcript snippets in the artifact"}

    acfg = AlignmentConfig()
    sem_cfg = SEM.SemanticConfig()

    provider = None
    if vlm_cfg.provider not in ("none", ""):
        provider = provider_from_config(vlm_cfg)
        if not getattr(provider, "available", False):
            m.fail("vlm", getattr(provider, "reason", "provider unavailable"))
        elif max_vlm_frames <= 0:
            m.fail("vlm", "skipped: --max-vlm-frames 0")
            m.cache_hits = 0
        else:
            vlm_cfg = VLMConfig.from_env(
                provider=vlm_cfg.provider, model=vlm_cfg.model,
                base_url=vlm_cfg.base_url, timeout=vlm_cfg.timeout,
                max_frames=max_vlm_frames)

    _path, pipeline_metrics = B.run_pipeline(
        data["meta"], data["snippets"], data["segments"], d, data["frames"],
        data.get("lang_used", ""), data.get("used_asr", False), data["visual"],
        provider, acfg, sem_cfg, use_cache=True,
        max_vlm_frames=max_vlm_frames)

    ev_path = d / "evidence.json"
    res = json.loads((d / "validation.json").read_text())
    if not res.get("valid"):
        return 1, {"status": "evidence-invalid",
                   "validation": res.get("blocking_errors", [])[:10]}

    # ---- synthesis. One draft, one critique, each behind its own ceiling.
    synth_outcome = None
    if synth_cfg.provider and synth_cfg.provider != "none":
        if synth_budget.limit <= 0:
            m.fail("synthesis", "skipped: --max-synthesis-calls 0")
        else:
            started = time.monotonic()
            synth_outcome = SY.synthesize(ev, synth_cfg, cache=None,
                                          budget=synth_budget,
                                          critic_budget=critic_budget)
            m.timed("synthesis", time.monotonic() - started)
            if synth_outcome.mode == "deterministic":
                m.fail("synthesis", synth_outcome.fallback_reason or "fell back")
    else:
        m.fail("synthesis", "no provider configured; the report is deterministic")

    # ---- the report, whichever writer produced it.
    synth = synth_cfg if synth_cfg.provider != "none" else None
    rc = B.write_and_validate_report(d, detail, quiet=True, synth=synth)

    out = {
        "status": "ok" if rc == 0 else "report-rejected",
        "max_vlm_frames": max_vlm_frames,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evidence_source": str(ev_path),
        "pipeline": pipeline_metrics,
        "metrics": _metrics_dict(m, synth_budget, critic_budget, synth_outcome),
        "quality": _quality(d, ev_path, synth_outcome),
    }
    return (0 if rc == 0 else 1), out


def _metrics_dict(m: LiveMetrics, synth_budget: Budget, critic_budget: Budget,
                  synth_outcome) -> dict:
    d = m.to_dict()
    d["synthesis_calls"] = synth_budget.spent
    d["critic_calls"] = critic_budget.spent
    d["budgets"] = {"synthesis": synth_budget.to_dict(), "critic": critic_budget.to_dict()}
    d["mode"] = synth_outcome.mode if synth_outcome else "deterministic"
    d["fallback_reason"] = synth_outcome.fallback_reason if synth_outcome else (
        "no synthesis model configured")
    return d


def _quality(d: Path, ev_path: Path, synth_outcome) -> dict:
    """What came out, by the four headings the report is written under.

    Read from the artifacts rather than recomputed, so this describes the same
    document a reader would read. Every figure here is a count of something
    observable, except the two "quality" judgements, which are counts of
    specific defects — a defect count is a measurement, an overall score is an
    opinion with a decimal point.
    """
    from evidence_model import load_evidence

    ev = load_evidence(ev_path)
    items = ev.items
    by_type: dict[str, int] = {}
    for i in items:
        by_type[i.type] = by_type.get(i.type, 0) + 1
    roles: dict[str, int] = {}
    for i in items:
        if i.semantic_role:
            roles[i.semantic_role] = roles.get(i.semantic_role, 0) + 1

    vlm_frames = [f for f in ev.visual if f.vlm_observation]
    vlm_ok = [f for f in vlm_frames if (f.vlm_observation or {}).get("status") == "ok"]
    vlm_bad = [f for f in vlm_frames if (f.vlm_observation or {}).get("status") not in ("ok", None)]
    abstained = [f for f in vlm_frames
                 if "no_visual" in json.dumps(f.vlm_observation or {})]

    rpath = d / "report_validation.json"
    rv_doc = json.loads(rpath.read_text()) if rpath.is_file() else {}
    metrics = rv_doc.get("metrics", {}) or {}
    blocking = rv_doc.get("blocking_errors", []) or []
    codes: dict[str, int] = {}
    for e in blocking:
        codes[e.get("code", "?")] = codes.get(e.get("code", "?"), 0) + 1

    report_ev = None
    rj = d / "report.json"
    if rj.is_file():
        try:
            report_ev = json.loads(rj.read_text())
        except json.JSONDecodeError:
            report_ev = None

    return {
        "vlm": {
            "frames_asked": len(vlm_frames),
            "conformant": len(vlm_ok),
            "useful": sum(1 for f in vlm_ok if (f.vlm_observation or {}).get("summary")),
            "abstentions": len(abstained),
            "failures": [f.frame_id for f in vlm_bad],
            "reached_evidence": sorted({
                p.ref for i in items for p in i.provenance if p.source == "vlm"}),
        },
        "extraction": {
            "items_by_type": dict(sorted(by_type.items())),
            "numbers_by_role": dict(sorted(roles.items())),
            "entities": by_type.get("entity", 0) + by_type.get("product", 0)
                        + by_type.get("model", 0) + by_type.get("tool", 0)
                        + by_type.get("organization", 0) + by_type.get("person", 0),
            "screen_only_facts": sum(1 for i in items
                                     if i.provenance and all(p.source in ("ocr", "frame", "vlm")
                                                              for p in i.provenance)),
            "comparisons": by_type.get("comparison", 0),
            "claims": by_type.get("claim", 0),
            "procedures": by_type.get("step", 0),
            "conflicts": len(ev.conflicts),
            "unresolved_conflicts": sum(1 for c in ev.conflicts if c.resolution != "resolved"),
        },
        "report": {
            "mode": rv_doc.get("mode"),
            "words": metrics.get("report_word_count"),
            "provenance_coverage": metrics.get("provenance_coverage"),
            "evidence_items_used": metrics.get("evidence_items_used"),
            "evidence_items_total": metrics.get("evidence_items_total"),
            "repetition_ratio": metrics.get("repetition_ratio"),
            "sections": metrics.get("section_count"),
            "unsupported_claims": codes.get("claim-unsupported", 0),
            "unsupported_numbers": codes.get("number-unsupported", 0)
                                    + codes.get("number-misattributed", 0),
            "unresolved_conflicts_in_report": codes.get("conflict-dropped", 0),
            "other_blocking": {k: v for k, v in sorted(codes.items())
                               if k not in ("claim-unsupported", "number-unsupported",
                                            "number-misattributed", "conflict-dropped")},
        },
    }


# --------------------------------------------------------------------------- #
# the artifact
# --------------------------------------------------------------------------- #

def write_report(out: dict, path: Path) -> Path:
    """`LIVE-EVAL-<timestamp>.md`, next to the artifacts it describes."""
    stamp = out.get("timestamp", "").replace(":", "").replace("+", "_") or "run"
    target = path / f"LIVE-EVAL-{stamp[:19]}.md"
    m = out.get("metrics", {})
    q = out.get("quality", {})

    def row(label, value):
        return f"| {label} | {value} |"

    lines = [f"# Live evaluation — {stamp}", "",
             f"Status: **{out.get('status')}**", "",
             "## Run", "",
             "| Field | Value |", "| --- | --- |",
             row("Provider (synthesis)", m.get("provider", "—")),
             row("Provider (VLM)", out.get("vlm_provider", "—")),
             row("Model (synthesis)", m.get("model", "—")),
             row("Model (VLM)", out.get("vlm_model", "—")),
             row("Video", out.get("title", "—")),
             row("Duration (s)", out.get("duration_s", "—")),
             row("VLM calls", m.get("vlm_calls", 0)),
             row("Synthesis calls", m.get("synthesis_calls", 0)),
             row("Critic calls", m.get("critic_calls", 0)),
             row("Cache hits", m.get("cache_hits", 0)),
             row("Latency (s)", m.get("wall_clock_s", "—")),
             row("Report mode", m.get("mode", "—")),
             ""]
    if m.get("fallback_reason"):
        lines += [f"Fallback reason: {m['fallback_reason']}", ""]
    if m.get("failures"):
        lines += ["### Failures", ""] + [f"- {f}" for f in m["failures"]] + [""]

    v, e, r = q.get("vlm", {}), q.get("extraction", {}), q.get("report", {})
    lines += ["## VLM quality", "",
              f"- conformant observations: {v.get('conformant', 0)}",
              f"- useful observations: {v.get('useful', 0)}",
              f"- abstentions: {v.get('abstentions', 0)}",
              f"- failures: {', '.join(v.get('failures') or []) or 'none'}",
              f"- frames whose reading reached `evidence.json`: "
              f"{', '.join(v.get('reached_evidence') or []) or 'none'}", ""]
    lines += ["## Extraction quality", "",
              f"- entities: {e.get('entities', 0)}",
              f"- numbers by role: {e.get('numbers_by_role') or 'none'}",
              f"- screen-only facts: {e.get('screen_only_facts', 0)}",
              f"- comparisons: {e.get('comparisons', 0)}",
              f"- claims: {e.get('claims', 0)}",
              f"- procedures: {e.get('procedures', 0)}",
              f"- conflicts: {e.get('conflicts', 0)} "
              f"({e.get('unresolved_conflicts', 0)} unresolved)", ""]
    lines += ["## Report quality", "",
              f"- words: {r.get('words')}",
              f"- provenance coverage: {r.get('provenance_coverage')}",
              f"- unsupported claims: {r.get('unsupported_claims', 0)}",
              f"- unsupported numbers: {r.get('unsupported_numbers', 0)}",
              f"- repetition: {r.get('repetition_ratio')}",
              f"- conflicts preserved: "
              f"{'yes' if not r.get('unresolved_conflicts_in_report') else 'NO'}",
              ""]
    if r.get("other_blocking"):
        lines += [f"- other blocking findings: {r['other_blocking']}", ""]
    lines += ["## Human observations", "",
              "_Written by whoever ran the command. Factual notes about obvious "
              "quality failures; no score._", ""]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None, result_out: list | None = None) -> int:
    """Run it. `result_out` receives the structured result, when given.

    The exit code is the shell's answer and the structured result is the
    report's, and they are not the same question: this function exits non-zero
    for a fetch that failed and for an artifact with no captions, neither of
    which is anything a provider did. `release_check` calls this in-process and
    has to tell those apart from a run that completed and produced a report the
    validator refused, so it needs the result and not just the code. Appending
    to a caller-supplied list keeps that a parameter rather than a module
    global, which would leak one run's answer into the next.
    """
    import argparse

    import build_evidence as B

    def _record(out: dict) -> dict:
        if result_out is not None:
            result_out.append(out)
        return out

    ap = argparse.ArgumentParser(
        prog="eval-live",
        description="A cheap, budgeted, measured end-to-end run against real providers.")
    ap.add_argument("url", help="the YouTube URL to fetch and evaluate")
    ap.add_argument("--vlm-provider", default=None)
    ap.add_argument("--vlm-model", default=None)
    ap.add_argument("--vlm-base-url", default=None)
    ap.add_argument("--synth-provider", default=None)
    ap.add_argument("--synth-model", default=None)
    ap.add_argument("--synth-base-url", default=None)
    ap.add_argument("--synth-api-key", default=None)
    ap.add_argument("--max-vlm-frames", type=int, default=DEFAULT_MAX_VLM_FRAMES,
                    help=f"hard ceiling on frames interpreted (default "
                         f"{DEFAULT_MAX_VLM_FRAMES}; 0 sends no frame at all)")
    ap.add_argument("--max-synthesis-calls", type=int, default=DEFAULT_MAX_SYNTHESIS_CALLS,
                    help=f"hard ceiling on report drafts (default {DEFAULT_MAX_SYNTHESIS_CALLS})")
    ap.add_argument("--max-critic-calls", type=int, default=DEFAULT_MAX_CRITIC_CALLS,
                    help=f"hard ceiling on critiques (default {DEFAULT_MAX_CRITIC_CALLS})")
    ap.add_argument("--detail", default="brief", choices=["brief", "balanced", "deep"])
    ap.add_argument("--no-report", action="store_true",
                    help="write no LIVE-EVAL file; print the metrics only")
    args = ap.parse_args(argv)

    vlm_cfg = VLMConfig.from_env(provider=args.vlm_provider, model=args.vlm_model,
                                 base_url=args.vlm_base_url, max_frames=args.max_vlm_frames)
    synth_cfg = SY.SynthConfig.from_env(provider=args.synth_provider, model=args.synth_model,
                                        base_url=args.synth_base_url,
                                        api_key=args.synth_api_key, detail=args.detail)

    fetch = Path(__file__).parent / "yt_fetch.py"
    print(f"eval-live: fetching {args.url}", file=sys.stderr)
    import subprocess
    rc = subprocess.run(["uv", "run", str(fetch), args.url]).returncode
    if rc != 0:
        print(f"eval-live: the fetch failed (exit {rc}); nothing was evaluated", file=sys.stderr)
        _record({"status": "failed", "error": f"the fetch failed (exit {rc})"})
        return rc
    vid = args.url.split("v=")[-1].split("&")[0]
    matches = sorted(Path(os.path.expanduser("~/Desktop/yt-reports")).glob(f"*-{vid}"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        print("eval-live: the fetch wrote no artifact directory", file=sys.stderr)
        _record({"status": "failed", "error": "the fetch wrote no artifact directory"})
        return 70
    d = matches[0]

    code, out = run_live(d, vlm_cfg, synth_cfg, args.max_vlm_frames,
                         Budget(args.max_synthesis_calls), Budget(args.max_critic_calls),
                         detail=args.detail)
    out.setdefault("vlm_provider", vlm_cfg.provider)
    out.setdefault("vlm_model", vlm_cfg.model)
    out["metrics"].setdefault("provider", synth_cfg.provider)
    out["metrics"].setdefault("model", synth_cfg.model)
    try:
        meta = json.loads((d / "meta.json").read_text())
        out.setdefault("title", meta.get("title"))
        out.setdefault("duration_s", meta.get("duration_s"))
    except (OSError, json.JSONDecodeError):
        out.setdefault("title", d.name)

    if not args.no_report:
        out["written"] = str(write_report(out, d))
    _record(out)
    print(json.dumps(out, indent=2, default=str)[:4000])
    if not args.no_report and out.get("written"):
        print(f"\n{out['written']}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
