#!/usr/bin/env python3
"""Collect the acceptance metrics for a set of real video runs.

One row per artifact directory. Everything here is read off the artifacts a run
actually left on disk — the point of the exercise is to report what the
pipeline produced, so nothing is recomputed and nothing is assumed. A field
whose source is missing is emitted as `null` with a note, never as a zero: a
zero would read as "measured, and the answer was none", which is a different
claim from "not measured".

    uv run --script <...> acceptance/metrics.py <label>=<dir> [...]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "acceptance" / "logs"

#: The artifacts a completed run is supposed to leave. A missing one is a
#: finding, not a formatting problem, so it is checked and reported.
REQUIRED = ["meta.json", "transcript.json", "transcript.md", "frames.json", "FRAMES.md",
            "visual.json", "alignment.json", "evidence.json", "validation.json",
            "REPORT.json", "REPORT.md"]


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 — absence and unreadability both mean "not measured"
        return None


def _stages(label: str) -> dict:
    return _load(LOGS / f"{label}.stages.json") or {}


def _visual_counts(visual: dict | None) -> tuple[int, int, int, int, int]:
    """(frames, ocr_read, ocr_unavailable, vlm_selected, vlm_ok)"""
    if not visual:
        return (0, 0, 0, 0, 0)
    frames = visual.get("frames") or []
    ocr_read = sum(1 for f in frames if (f.get("result") or {}).get("status") == "ok"
                   or f.get("text"))
    ocr_unavailable = sum(1 for f in frames
                          if (f.get("result") or {}).get("status") in ("unavailable", "failed"))
    vlm = [(f.get("vlm") or {}) for f in frames]
    selected = sum(1 for v in vlm if v.get("status") not in (None, "skipped"))
    ok = sum(1 for v in vlm if v.get("status") in ("ok", "no_visual"))
    return (len(frames), ocr_read, ocr_unavailable, selected, ok)


def _item_modes(evidence: dict | None) -> dict:
    """Evidence by how it was read. The three that matter for §8: model-derived,
    OCR-only, and corroborated by two modalities — the last being the one a
    transcript-only report can never have."""
    if not evidence:
        return {"visual": 0, "vlm": 0, "ocr_only": 0, "cross_modal": 0, "transcript_only": 0}
    out = Counter()
    for item in evidence.get("items") or []:
        srcs = {p.get("source") for p in (item.get("provenance") or [])}
        if "vlm" in srcs:
            out["vlm"] += 1
        elif "ocr" in srcs or "frame" in srcs:
            out["ocr_only"] += 1
        else:
            out["transcript_only"] += 1
        if srcs & {"transcript", "metadata"} and srcs & {"ocr", "frame", "vlm"}:
            out["cross_modal"] += 1
        elif srcs & {"ocr", "frame", "vlm"}:
            out["visual"] += 1
    return {k: out[k] for k in ("visual", "vlm", "ocr_only", "cross_modal", "transcript_only")}


def collect(label: str, d: Path) -> dict:
    missing = [f for f in REQUIRED if not (d / f).exists()]
    has_frames_dir = (d / "frames").is_dir()
    meta = _load(d / "meta.json")
    ev = _load(d / "evidence.json")
    rv = _load(d / "report_validation.json")
    vis = _load(d / "visual.json")
    val = _load(d / "validation.json")
    stages = _stages(label)
    m = (rv or {}).get("metrics") or {}
    syn = (rv or {}).get("synthesis") or {}
    vs = (vis or {}).get("vlm_summary") or {}
    frames, ocr_read, ocr_unavail, vlm_sel, vlm_ok = _visual_counts(vis)
    modes = _item_modes(ev)

    report_words = m.get("report_word_count")
    tw = m.get("transcript_word_count")
    return {
        "label": label,
        "dir": str(d),
        "artifacts_missing": missing,
        "frames_dir": has_frames_dir,
        "frame_files": len(list((d / "frames").glob("*"))) if has_frames_dir else 0,
        "duration_s": (meta or {}).get("duration_s") or (meta or {}).get("duration"),
        "transcript_words": tw,
        "report_words": report_words,
        "compression_ratio": (round(report_words / tw, 4)
                              if report_words and tw else m.get("compression_ratio")),
        "segments": len((ev or {}).get("segments") or []),
        "items_total": m.get("evidence_items_total") or len((ev or {}).get("items") or []),
        "items_used": m.get("evidence_items_used"),
        "provenance_coverage": m.get("provenance_coverage"),
        "unsupported_claims": m.get("unsupported_claims"),
        "unsupported_numbers": m.get("unsupported_numbers"),
        "unsupported_urls": m.get("unsupported_urls"),
        "unsupported_visual": m.get("unsupported_visual_statements"),
        "fabricated_timestamps": m.get("fabricated_timestamps"),
        "dropped_hedges": m.get("dropped_hedges"),
        "visual_items": modes["visual"],
        "vlm_items": modes["vlm"],
        "ocr_only_items": modes["ocr_only"],
        "cross_modal_items": modes["cross_modal"],
        "conflicts_unresolved": m.get("conflicts_in_evidence"),
        "duplicate_8grams": m.get("duplicate_8grams"),
        "repetition_ratio": m.get("repetition_ratio"),
        "report_valid": (rv or {}).get("valid"),
        "report_mode": (rv or {}).get("mode"),
        "authored_by": m.get("authored_by"),
        "synthesis_mode": syn.get("mode") or None,
        "synthesis_fallback_reason": (syn.get("fallback_reason") or None),
        "synthesis_revisions": syn.get("revisions"),
        "synthesis_cache_hit": syn.get("cache_hit"),
        "critic_findings": len((syn.get("critique") or {}).get("findings") or []) or None,
        "critic_called": bool(syn.get("critique")),
        "frames_total": vs.get("frames_total", frames),
        "frames_ocr_read": ocr_read,
        "frames_ocr_unavailable": ocr_unavail,
        "vlm_model": vs.get("model"),
        "vlm_selected": vs.get("frames_selected", vlm_sel),
        "vlm_calls": vs.get("vlm_calls"),
        "vlm_calls_ok": vs.get("calls_ok", vlm_ok),
        "vlm_calls_failed": vs.get("calls_failed"),
        "vlm_cache_hits": vs.get("cache_hits"),
        "vlm_latency_s": round(vs["vlm_wall_s"], 1) if vs.get("vlm_wall_s") else None,
        "evidence_valid": (val or {}).get("valid"),
        "evidence_error_count": len((val or {}).get("blocking_errors") or []) or 0,
        "stage_seconds": {k: v.get("seconds") for k, v in stages.items() if isinstance(v, dict)},
        "stage_exit": stages.get("_exit"),
    }


def main(argv: list[str]) -> int:
    rows = []
    for arg in argv:
        label, _, path = arg.partition("=")
        rows.append(collect(label, Path(path).expanduser()))
    out = ROOT / "acceptance" / "metrics.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {out}")
    for r in rows:
        missing = (" MISSING:" + ",".join(r["artifacts_missing"])) if r["artifacts_missing"] else ""
        print(f"  {r['label']:18s} valid={r['report_valid']} words={r['report_words']} "
              f"items={r['items_total']}/{r['items_used']} vlm_calls={r['vlm_calls']} "
              f"ok={r['vlm_calls_ok']} synth={r['synthesis_mode']}{missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
