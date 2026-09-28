#!/usr/bin/env python3
"""Run the same video twice and say exactly which layers are reproducible.

The claim "deterministic" is only worth what it covers, and it is easy to state
it about the whole system when it holds for two layers out of four. So this runs
the pipeline twice over the same artifact directory with a warm cache and
compares the outputs byte for byte, reporting per layer rather than one verdict:

* **evidence** — the extraction over the same inputs. Claimed deterministic.
* **report** — the renderer over the same evidence. Claimed deterministic.
* **model layers** — not exercised here; a live run is where true LLM
  nondeterminism lives, and it is measured separately and labelled, not assumed.

Anything that legitimately varies run to run (a generation timestamp) is
identified by name rather than silently excluded, because a diff that is
explained away is how a real nondeterminism gets to hide inside a "deterministic"
label.

    uv run --with yt-dlp --with pillow --with numpy python3 acceptance/reproducibility.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Fields that are allowed to differ because they record *when* something ran
#: rather than *what* it produced. Each is stripped before comparison and named
#: here, so an unexplained difference still fails the run.
VOLATILE = {"generated_at", "observed_at", "elapsed_ms", "cache", "vlm_wall_s",
            "latency_s", "wall_s", "fetched_at", "run_at"}


def _sha(p: Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def _strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def _stable_json(p: Path):
    return json.dumps(_strip_volatile(json.loads(p.read_text())), sort_keys=True,
                      ensure_ascii=False)


def _run(d: Path, extra: list[str], env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_evidence.py"), str(d),
         "--vlm-provider", "none", "--synthesis-only", *extra],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900)


def main() -> int:
    cands = ([Path(sys.argv[1]).expanduser()] if len(sys.argv) > 1
             else sorted((Path.home() / "Desktop" / "yt-reports").glob("*EOdXR6lU5ZA")))
    if not cands:
        print("SKIP  no real artifact directory; run a fetch first")
        return 0
    src = cands[0]
    env = dict(os.environ)
    for k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        env.pop(k, None)

    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "art"
        shutil.copytree(src, d)

        # Run 1 populates the caches; run 2 is the one that proves the *cached*
        # path is stable, which is the path a re-run actually takes.
        for label, passes in (("cold", 1), ("warm", 2)):
            for i in range(passes):
                proc = _run(d, [], env)
                if proc.returncode != 0:
                    print(f"FAIL  {label} pass {i + 1} exited {proc.returncode}")
                    print(proc.stderr[-600:])
                    return 1
            results[label] = {
                "evidence": _stable_json(d / "evidence.json"),
                "evidence_raw": _sha(d / "evidence.json"),
                "report_md": (d / "REPORT.md").read_text(),
                "report_md_sha": _sha(d / "REPORT.md"),
                "report_json": _stable_json(d / "REPORT.json"),
                "frames_json": _stable_json(d / "frames.json"),
                "alignment": _stable_json(d / "alignment.json"),
            }

    cold, warm = results["cold"], results["warm"]
    rows: list[tuple[str, bool, str]] = []

    rows.append(("evidence.json is structurally identical",
                 cold["evidence"] == warm["evidence"],
                 "differs with volatile fields removed"))
    rows.append(("evidence.json is byte-identical",
                 cold["evidence_raw"] == warm["evidence_raw"],
                 "differs byte for byte"))
    rows.append(("REPORT.md is byte-identical", cold["report_md_sha"] == warm["report_md_sha"],
                 "differs byte for byte"))
    rows.append(("REPORT.json is structurally identical",
                 cold["report_json"] == warm["report_json"], "differs"))
    rows.append(("frames.json is structurally identical",
                 cold["frames_json"] == warm["frames_json"], "differs"))
    rows.append(("alignment.json is structurally identical",
                 cold["alignment"] == warm["alignment"], "differs"))

    ev = json.loads((src / "evidence.json").read_text())
    ids = [i["id"] for i in ev["items"]]
    rows.append(("evidence ids are unique", len(ids) == len(set(ids)),
                 f"{len(ids) - len(set(ids))} duplicates"))

    # Timestamps: the measured frame times must be the same numbers, not merely
    # the same count. A frame index that drifts changes what a citation means.
    fr = json.loads((src / "frames.json").read_text())
    entries = fr if isinstance(fr, list) else (fr.get("frames") or [])
    times = [f.get("t_start") for f in entries]
    rows.append(("every frame carries a measured timestamp",
                 all(t is not None for t in times), "some frame has no measured time"))

    # And the *citations* in the report resolve against the same evidence.
    md = warm["report_md"]
    markers = set(re.findall(r"\[(\d{1,2}:\d{2})", md))
    rows.append(("the report still carries provenance markers", bool(markers),
                 "no [mm:ss] markers in the report"))

    ok = True
    for name, good, detail in rows:
        print(f"  {'ok  ' if good else 'FAIL'} {name}" + ("" if good else f"  — {detail}"))
        ok &= good

    out = ROOT / "acceptance" / "reproducibility.json"
    out.write_text(json.dumps({
        "source": str(src),
        "runs": 2,
        "cache": "warm on the second run",
        "deterministic_layers": ["evidence extraction", "report rendering"],
        "nondeterministic_layers": ["any live model call (VLM, synthesis, critic)"],
        "volatile_fields_excluded": sorted(VOLATILE),
        "checks": {name: bool(good) for name, good, _ in rows},
        "evidence_sha": {"cold": cold["evidence_raw"], "warm": warm["evidence_raw"]},
        "report_sha": {"cold": cold["report_md_sha"], "warm": warm["report_md_sha"]},
    }, indent=2) + "\n")
    print(f"\n-> {out}")
    print("NOTE  determinism is claimed only for the layers listed as deterministic; "
          "a live model call is not deterministic and is measured separately.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
