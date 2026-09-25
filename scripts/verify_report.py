#!/usr/bin/env python3
"""Mechanical checks on a finished REPORT.md against its own artifact directory.

Deliberately high-precision and low-recall. A checker that cries wolf is worse than
no checker, because it trains people to ignore it — so this flags only defects that
can be identified without judgement, and reports the ambiguous one as INFO.

  FAIL  a citation points at a frame that does not exist in frames/
  FAIL  a forbidden provenance label is used in a source position
  INFO  a frame is both cited and listed among those not opened (usually the
        coverage list itself; only a defect if the report genuinely means both)

What this CANNOT do is the important half. It cannot tell you that a real frame
carries content it does not, that a value is right but pinned to the wrong frame, or
that an absence claim is false. Reports in this project have passed every check here
and still fabricated heavily. A clean run means "no mechanical defect found", never
"verified".

    python3 verify_report.py <artifact-dir>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FRAME = r"[us]\d{4}"

# A citation is a frame ID in a source cell or an explicit frame reference. A bare ID
# in a comma list is a coverage list, not a citation — matching those too is what made
# an earlier version of this script flag every honest "frames I did not open" list.
CITED = re.compile(
    rf"Screen\s*\(\s*(?:frame\s+)?`?({FRAME})"   # Screen (u0007, verbatim) / Screen (frame u0007…)
    rf"|frames?\s+`({FRAME})`"                   # frame `u0007`
    rf"|`({FRAME})\.jpg`"                        # `u0007.jpg`
    rf"|\|\s*({FRAME})\s*\|",                    # a bare ID as a whole table cell
    re.I,
)
ANY_FRAME = re.compile(FRAME)
UNOPENED = re.compile(r"did not open|didn't open|not opened|never opened|unopened", re.I)

# A forbidden label only counts when it is being USED as a source. The same words
# inside a disclaimer ("no 'Screen (sheet)' is used here") are the correct behaviour.
FORBIDDEN = [
    (re.compile(r"Screen\s*\(\s*sheet\s*\)", re.I), "'Screen (sheet)' used as a source"),
    (re.compile(r"\|\s*`?Contact sheet`?\s*\|", re.I), "'Contact sheet' used as a source-column value"),
    (re.compile(r"\|\s*`?Sheet`?\s*\|", re.I), "'Sheet' used as a source-column value"),
]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    d = Path(sys.argv[1]).resolve()
    report = d / "REPORT.md"
    if not report.exists():
        print(f"FAIL  no REPORT.md in {d}")
        return 1
    lines = report.read_text(encoding="utf-8", errors="replace").splitlines()
    frames_dir = d / "frames"
    on_disk = {p.stem for p in frames_dir.glob("*.jpg")} if frames_dir.is_dir() else set()
    failures: list[str] = []
    notes: list[str] = []

    cited: set[str] = set()
    for line in lines:
        for m in CITED.finditer(line):
            cited |= {g for g in m.groups() if g}

    unopened: set[str] = set()
    for line in lines:
        if UNOPENED.search(line):
            unopened |= set(ANY_FRAME.findall(line))

    # Skip lines that disclaim the labels, so a report may say it does not use them.
    label_scan = "\n".join(l for l in lines if not UNOPENED.search(l) and not re.search(
        r"\b(no|not|never|without|none|neither|nor)\b", l, re.I))

    print(f"  {len(cited)} cited · {len(on_disk)} on disk · {len(unopened)} named as unopened\n")

    missing = sorted(cited - on_disk) if on_disk else []
    if missing:
        failures.append(f"cites frame(s) that do not exist: {', '.join(missing)}")
    print(f"  {'ok  ' if not missing else 'FAIL'} every cited frame exists")

    hits = [msg for rx, msg in FORBIDDEN if rx.search(label_scan)]
    if hits:
        failures.extend(hits)
    print(f"  {'ok  ' if not hits else 'FAIL'} no forbidden provenance label used as a source")

    both = sorted(cited & unopened)
    if both:
        notes.append(f"{len(both)} frame(s) appear both as a citation and in an "
                     f"unopened list ({', '.join(both[:6])}{'…' if len(both) > 6 else ''}). "
                     f"Usually the coverage list; check it is not a real contradiction.")
    print(f"  {'ok  ' if not both else 'info'} cited ∩ unopened checked")

    for n in notes:
        print(f"\n  INFO  {n}")

    # Human-numbered citations ("frame 7", "Frame u0007") are not mechanically
    # checkable: the N in "frame 7" is the agent's own index, and it does not map to
    # a filename (u0007 and s0007 are different pictures). A report that cites this
    # way cannot be audited by this script at all, which is worth saying out loud
    # rather than reporting a clean "0 cited".
    human = re.findall(r"[Ff]rames?\s+(\d+)\b", "\n".join(lines))
    if human and not cited:
        notes.append(f"{len(human)} citation(s) use the human form 'frame N' rather than a "
                     f"filename, so this script could not check ANY of them. Use the "
                     f"`u0007` / `s0013` filenames — a human index does not map to a file.")
        print(f"  WARN  {len(human)} 'frame N' citations are not mechanically checkable")

    print()
    if failures:
        print(f"FAILED {len(failures)} mechanical check(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("No mechanical defect found. This is NOT a verification of content —")
    print("open the frames, and check adjacent frames in a sequence especially.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
