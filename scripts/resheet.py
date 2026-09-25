#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["yt-dlp>=2025.1.1", "youtube-transcript-api>=1.0", "pillow>=10.0", "numpy>=1.26"]
# ///
"""Rebuild contact sheets from frames already on disk — no download, no ASR.

    uv run resheet.py /path/to/video-dir [...]

Use when a fetch produced frames but the sheets are missing or empty, which
happened once when a malformed ffmpeg filter string made sheet generation fail
silently. Reads the measured timestamps back out of FRAMES.md, so the rebuilt
sheets are ordered and labelled exactly as the original run would have.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

COLS = 4
MAX_ROWS = 7          # past this a sheet is scaled down until tiles stop being readable
PER_SHEET = COLS * MAX_ROWS


def load_times(frames_md: Path) -> dict[str, float]:
    times: dict[str, float] = {}
    if not frames_md.exists():
        return times
    for line in frames_md.read_text().splitlines():
        m = re.match(r"- `(.+?\.jpg)` \[(\d\d):(\d\d):(\d\d)\]", line)
        if m:
            times[m.group(1)] = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
    return times


def resheet(outdir: Path) -> dict:
    frames_dir = outdir / "frames"
    frames = list(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
    if not frames:
        return {"dir": str(outdir), "status": "no_frames"}

    times = load_times(outdir / "FRAMES.md")
    ordered = sorted(frames, key=lambda p: times.get(p.name, 0.0))
    for old in outdir.glob("contact-sheet*.jpg"):
        old.unlink()

    sheets, failures = [], []
    for i in range(0, len(ordered), PER_SHEET):
        chunk = ordered[i : i + PER_SHEET]
        rows = max(1, -(-len(chunk) // COLS))
        name = "contact-sheet.jpg" if len(ordered) <= PER_SHEET else f"contact-sheet-{i // PER_SHEET + 1}.jpg"
        dest = outdir / name
        inputs: list[str] = []
        for f in chunk:
            inputs += ["-i", str(f)]
        graph = (
            "".join(f"[{j}:v]scale=384:-2[v{j}];" for j in range(len(chunk)))
            + "".join(f"[v{j}]" for j in range(len(chunk)))
            + f"concat=n={len(chunk)}:v=1:a=0,tile={COLS}x{rows}"
        )
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph,
             "-frames:v", "1", str(dest)],
            check=False, capture_output=True, text=True,
        )
        (sheets if dest.exists() else failures).append(
            dest.name if dest.exists() else f"chunk {i // PER_SHEET + 1}: {proc.stderr[:160]}"
        )

    result = {"dir": str(outdir), "frames": len(ordered), "sheets": sheets}
    if failures:
        result["FAILED"] = failures
    return result


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    import json
    for arg in sys.argv[1:]:
        print(json.dumps(resheet(Path(arg)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
