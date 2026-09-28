#!/usr/bin/env python3
"""What the report gate's content-word rule costs on real evidence.

The rule is "a content word in a factual block must appear somewhere in the
evidence document". That is a strong rule and it has exactly one way to be
wrong: the renderer's own fixed prose. Fifteen synthetic fixtures did not
surface those words; a real video did, immediately.

So this measures the false-positive rate over every real artifact on disk, and
prints the words responsible. The output is the input to `TEMPLATE_WORDS` — the
deliberate widening the module documents — and running it again after a widening
is how "the check passed on the fixtures" stops being the standard.

    uv run --with yt-dlp --with pillow --with numpy python3 acceptance/template_audit.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import report as R  # noqa: E402
import report_fixtures as FX  # noqa: E402
import report_validate as RV  # noqa: E402


def _load_evidence(d: Path):
    from evidence_model import Evidence
    try:
        return Evidence.from_dict(json.loads((d / "evidence.json").read_text()))
    except Exception as exc:  # noqa: BLE001
        print(f"  skip {d.name}: {type(exc).__name__}: {exc}")
        return None


def main() -> int:
    offenders: dict[str, set] = defaultdict(set)
    checked = 0

    print("fixtures:")
    for name in FX.FIXTURES:
        ev = FX.FIXTURES[name]()
        md = R.render_markdown(R.generate(ev, "balanced"))
        res = RV.validate_report(R.generate(ev, "balanced"), ev, md)
        checked += 1
        for e in res.blocking_errors:
            if e["code"] == "claim-unsupported":
                word = e["message"].split("'")[1]
                offenders[word].add(f"fixture:{name}")
                print(f"  FALSE POSITIVE {name}: {e['message'][:90]}")

    print("real artifacts:")
    root = Path.home() / "Desktop" / "yt-reports"
    for d in sorted(root.iterdir()) if root.exists() else []:
        if not (d / "evidence.json").exists():
            continue
        ev = _load_evidence(d)
        if ev is None:
            continue
        try:
            rep = R.generate(ev, "balanced")
            md = R.render_markdown(rep)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {d.name}: generation failed: {type(exc).__name__}: {exc}")
            continue
        res = RV.validate_report(rep, ev, md)
        checked += 1
        n = 0
        for e in res.blocking_errors:
            if e["code"] == "claim-unsupported":
                word = e["message"].split("'")[1]
                offenders[word].add(d.name)
                n += 1
        others = sorted({e["code"] for e in res.blocking_errors if e["code"] != "claim-unsupported"})
        print(f"  {d.name[:52]:52s} claim-unsupported={n} other={others or 'none'}")

    print(f"\nTEMPLATE_WORDS currently holds {len(RV.TEMPLATE_WORDS)} words: "
          f"{', '.join(sorted(RV.TEMPLATE_WORDS))}")
    if not offenders:
        print(f"\nno false positives across {checked} documents")
        return 0
    print(f"\n{len(offenders)} word(s) are renderer prose the set does not yet cover:")
    for word, where in sorted(offenders.items()):
        print(f"  {word!r:20s} in {len(where)} document(s): {sorted(where)[:3]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
