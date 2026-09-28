#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10.0"]
# ///

"""Tile every sampled frame into labelled contact sheets.

Two problems it exists to solve, both of which silently corrupt a report:

  * the skill's own contact-sheet*.jpg tiles are NOT in filename order, so a
    frame ID cannot be inferred from a tile's position on a sheet;
  * a multi-file image read can return fewer images than requested with no
    indication of the drop, which shifts every later result by one when
    mapped positionally.

Labelling each tile with its own filename removes both, and doubles as
full-coverage triage: one pass shows every frame that exists.

Usage:
    uv run label_sheet.py <artifact-dir> [--out DIR] [--cols 2] [--width 620]
                          [--per-sheet 12] [--only u0012,s0017]
"""
import argparse
import os
import sys

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Pillow required: uv run --with pillow label_sheet.py ...")

LABEL_H = 46


def frames(framedir):
    return sorted(f for f in os.listdir(framedir) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def tile(path, width):
    im = Image.open(path).convert("RGB")
    h = int(im.height * width / im.width)
    im = im.resize((width, h))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, width, LABEL_H], fill="black")
    d.text((8, LABEL_H // 2 - 6), os.path.splitext(os.path.basename(path))[0], fill="yellow")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", help="report artifact dir (containing frames/)")
    ap.add_argument("--out", default=None, help="output dir (default: <artifact>/labelled)")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--width", type=int, default=620, help="tile width px")
    ap.add_argument("--per-sheet", type=int, default=12, help="tiles per sheet")
    ap.add_argument("--only", default=None, help="comma-separated frame ids, e.g. u0012,s0017")
    a = ap.parse_args()

    framedir = os.path.join(a.artifact, "frames")
    if not os.path.isdir(framedir):
        sys.exit(f"no frames/ under {a.artifact}")
    out = a.out or os.path.join(a.artifact, "labelled")
    os.makedirs(out, exist_ok=True)

    names = frames(framedir)
    if a.only:
        want = {w.strip() for w in a.only.split(",") if w.strip()}
        names = [n for n in names if os.path.splitext(n)[0] in want]
    if not names:
        sys.exit("no frames selected")

    tiles = [tile(os.path.join(framedir, n), a.width) for n in names]
    th = max(t.height for t in tiles)
    per = max(1, a.per_sheet)
    made = []
    for i in range(0, len(tiles), per):
        chunk = tiles[i:i + per]
        rows = (len(chunk) + a.cols - 1) // a.cols
        sheet = Image.new("RGB", (a.cols * a.width, rows * th), "black")
        for k, t in enumerate(chunk):
            sheet.paste(t, ((k % a.cols) * a.width, (k // a.cols) * th))
        p = os.path.join(out, f"labelled-{i // per + 1}.jpg")
        sheet.save(p, quality=92)
        made.append(p)

    print(f"{len(tiles)} frames -> {len(made)} sheet(s) in {out}")
    for p in made:
        print(p)
        start = made.index(p) * per
        print("  " + ", ".join(os.path.splitext(n)[0] for n in names[start:start + per]))


if __name__ == "__main__":
    main()
