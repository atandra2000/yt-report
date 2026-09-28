#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10.0", "numpy>=1.26"]
# ///
"""Render a code-editor and a settings-pane video, for exercising the sampler.

WHY THIS EXISTS, AND WHAT IT IS NOT.

The `screen_ui` and `code` branches of `classify_visual_profile` had only ever
been exercised from constructed OCR strings — a list of tuples handed straight
to `ocr_profile`. That tests the *rule* and nothing about the *world*: it never
runs ffmpeg, never decodes a frame, never asks the Vision backend to read
anything, and so cannot tell whether the pipeline as a whole produces the
counts the rule reads.

No real screen recording or code-editor video exists in this repository, and
this script deliberately does not download one. So the honest claim is the
narrow one: **this is a rendered fixture, and passing it does not mean the
profiles are real-world validated.** It establishes that the whole path —
render, encode, scene-probe, text-probe through real Apple Vision OCR,
classify, plan, extract, dedupe — reaches the `code` and `screen_ui` branches
and that the numbers it reports are self-consistent. The one thing it cannot
establish is that a real editor window looks like this, and only a real video
can settle that.

Run it directly to write the two clips:

    uv run visual_fixture.py /tmp/fixtures

Each clip is deterministic: same pixels every run, so a sampler result can be
compared across changes instead of only against itself.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

W, H = 1280, 720

CODE_LINES = [
    ("ln", "from dataclasses import dataclass, field"),
    ("kw", "@dataclass"),
    ("kw", "class FrameBudget:"),
    ("op", "    max_frames: int = 300"),
    ("op", "    step_s: float = 12.0"),
    ("op", "    intervals: list = field(default_factory=list)"),
    ("bl", ""),
    ("kw", "    def plan(self, duration: float) -> list:"),
    ("op", "        count = min(self.max_frames,"),
    ("op", "                       max(1, round(duration / self.step_s)))"),
    ("op", "        step = duration / max(1, count - 1)"),
    ("kw", "        return [round(i * step, 3) for i in range(count)]"),
    ("bl", ""),
    ("cm", "# a near-duplicate run still keeps its last frame"),
    ("op", "    def dedupe(self, frames, threshold=3.0):"),
    ("op", "        kept, last = [], None"),
    ("kw", "        for frame in frames:"),
    ("op", "            thumb = self._thumb(frame)"),
    ("op", "            if thumb is None or last is None:"),
    ("op", "                kept.append(frame)"),
    ("cm", "            elif self._delta(thumb, last) >= threshold:"),
    ("op", "                kept.append(frame)"),
    ("cm", "            # else: a duplicate, and the tail is kept"),
    ("op", "        return kept"),
    ("bl", ""),
    ("kw", "if __name__ == \"__main__\":"),
    ("op", "    budget = FrameBudget(max_frames=64)"),
    ("op", "    print(len(budget.plan(1120.0)))"),
]

UI_LINES = [
    "System Preferences",
    "Appearance          Light",
    "Notifications       3 apps",
    "Sound               Alerts",
    "Focus               Do Not Disturb",
    "Screen Time         2h 15m today",
    "Privacy & Security  FileVault On",
    "Network             Wi-Fi Connected",
    "Bluetooth           4 devices",
    "General             About This Mac",
    "Desktop & Dock      6 items",
    "Displays            2560 x 1440",
    "Keyboard            87 keys",
    "Trackpad            Three-finger drag",
    "Printers & Scanners None",
    "Extensions          3 enabled",
    "Users & Groups      2 accounts",
    "Passwords           41 items",
    "Internet Accounts   4 accounts",
    "Game Center         On",
    "Wallet & Apple Pay  3 cards",
    "Siri & Spotlight    2 services",
    "Privacy & Security  Firewall On",
    "Login Items         12 items",
]


def _base(dark: bool) -> "Image.Image":  # noqa: F821
    from PIL import Image, ImageDraw
    bg = (30, 32, 38) if dark else (246, 247, 249)
    fg = (222, 226, 233) if dark else (38, 40, 46)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    # A sidebar, so the frame has the columnar structure a real app window has
    # and a 16x16 reduction sees something that is not a flat field.
    side = (40, 43, 51) if dark else (233, 236, 240)
    d.rectangle([0, 0, 280, H], fill=side)
    return img


def _font(size: int):
    from PIL import ImageFont
    for path in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Courier.dfont",
                 "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def code_frame(t: float):
    """An editor window. The content advances a line every few seconds, so the
    sampler has real changes to find rather than a still image."""
    from PIL import Image, ImageDraw
    img = _base(True)
    d = ImageDraw.Draw(img)
    f = _font(19)
    d.rectangle([280, 0, W, 40], fill=(52, 56, 66))
    d.text((300, 12), "sampler.py — yt-report", font=_font(17), fill=(170, 176, 188))
    d.rectangle([280, 40, 330, H], fill=(36, 38, 45))
    # Scroll: an offset that grows with time, so the window is never static.
    offset = int(t // 3) % 6
    y = 60
    for i, (_kind, text) in enumerate(CODE_LINES):
        if y > H - 20:
            break
        d.text((292, y), str(i + 1), font=_font(14), fill=(88, 94, 108))
        if text:
            d.text((344, y), text, font=f, fill=(214, 219, 228))
        y += 28
    d.rectangle([344, 60 + offset * 28, 344 + 260, 60 + offset * 28 + 22],
                outline=(120, 170, 255), width=2)
    return img


def ui_frame(t: float):
    """A settings pane whose selection moves — a real UI changes without the
    picture being visually different at 16x16, which is exactly the case the
    deduper has to be careful about."""
    from PIL import Image, ImageDraw
    img = _base(False)
    d = ImageDraw.Draw(img)
    f = _font(20)
    d.text((32, 40), "Settings", font=_font(30), fill=(20, 22, 28))
    d.rectangle([32, 92, 248, 96], fill=(214, 218, 226))
    selected = int(t // 4) % len(UI_LINES)
    y = 118
    for i, line in enumerate(UI_LINES):
        if y > H - 20:
            break
        if i == selected:
            d.rounded_rectangle([24, y - 6, 256, y + 26], radius=6, fill=(10, 132, 255))
            d.text((36, y), line, font=f, fill=(255, 255, 255))
        else:
            d.text((36, y), line, font=f, fill=(44, 46, 54))
        y += 34
    d.text((300, 60), "Every setting on one page", font=_font(24), fill=(20, 22, 28))
    d.text((300, 100), "Scroll to see the rest of the panel.", font=f, fill=(90, 94, 104))
    return img


def _encode(out: Path, seconds: int, fps: int, frame_fn) -> Path:
    """Render the frames and let ffmpeg make the clip. Written to a temp
    directory and removed after, so a failed run leaves no half-written video
    where a later run would pick it up."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="yt-visual-fixture-") as tmp:
        # ffmpeg's own printf pattern, so the frame files are written with the
        # same names the encoder will ask for rather than the two being kept in
        # step by hand.
        src = str(Path(tmp) / "f%04d.png")
        n = seconds * fps
        for i in range(n):
            frame_fn(i / fps).save(src % (i + 1))
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(src),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
            check=True, capture_output=True)
    return out


def build(dest: str | Path, seconds: int = 40, fps: int = 12) -> dict[str, Path]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    return {
        "code": _encode(dest / "code-editor.mp4", seconds, fps, code_frame),
        "screen_ui": _encode(dest / "settings-pane.mp4", seconds, fps, ui_frame),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    dest = Path(argv[0]) if argv else Path("/tmp/yt-visual-fixtures")
    clips = build(dest)
    for name, path in clips.items():
        print(f"{name}: {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
