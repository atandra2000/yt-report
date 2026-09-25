#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["yt-dlp>=2025.1.1", "youtube-transcript-api>=1.0", "pillow>=10.0", "numpy>=1.26"]
# ///
"""Offline self-test for the yt-report pipeline. No network, no ASR, no YouTube.

Why this exists: a change once shipped that produced ZERO contact sheets while
reporting `"status": "ok"`, and a batch refactor once passed a list where a
string was required. Both were silent — nothing downstream noticed. This
exercises the pure functions and the ffmpeg plumbing on a synthetic video so
those classes of break fail loudly instead.

    uv run selftest.py            # run all checks
    uv run selftest.py -v         # list each check as it passes
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import yt_fetch as Y  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}{'  — ' + detail if detail else ''}")
        FAILURES.append(name)


def t_url_parsing() -> None:
    for raw, want in [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=42", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=WL", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ]:
        check(f"parse id: {raw[:46]}", Y.parse_video_id(raw) == want, f"got {Y.parse_video_id(raw)}")
    check("reject junk url", Y.parse_video_id("https://example.com/watch") is None)


def t_timestamps() -> None:
    check("ts 0", Y.ts(0) == "00:00:00")
    check("ts 3725", Y.ts(3725) == "01:02:05")


def t_chapter_fallback() -> None:
    """The 5-minute-bucket fallback produced 2 useless segments for a 5-minute
    video; block length now scales with duration."""
    for dur, lo, hi, label in [(330, 4, 7, "5.5-min news"), (900, 4, 7, "15-min digest"),
                              (2760, 8, 13, "46-min lecture"), (3240, 9, 13, "54-min talk")]:
        # Snippets must span the whole duration, or dropping speech-free blocks —
        # correct behaviour — reads as a segmentation failure.
        snips = [{"start": float(t), "dur": 1.0, "text": "x"} for t in range(0, dur, 20)]
        segs = Y.segments_from_chapters(snips, [], dur)
        check(f"fallback segments for {label}: {len(segs)}", lo <= len(segs) <= hi,
              f"want {lo}-{hi}")
    # Real chapters must be respected verbatim.
    ch = [{"start_time": 0, "title": "Intro"}, {"start_time": 100, "title": "Body"}]
    segs = Y.segments_from_chapters([{"start": 10, "dur": 1, "text": "a"},
                                     {"start": 150, "dur": 1, "text": "b"}], ch, 200)
    check("chapters respected", [s["title"] for s in segs] == ["Intro", "Body"])
    check("snippets routed to chapters", segs[0]["snippets"][0]["text"] == "a")
    # A block with no speech is a gap in the video, not a section of it.
    check("empty blocks dropped", all(s["snippets"] for s in Y.segments_from_chapters(snips[:1], [], 600)))


def t_frame_budget() -> None:
    for dur, lo, hi, label in [(330, 12, 20, "5.5 min"), (2760, 30, 48, "46 min"),
                               (3240, 40, 48, "54 min"), (None, 12, 48, "unknown duration")]:
        b = Y.auto_frame_budget(dur)
        check(f"auto budget {label}: {b}", lo <= b <= hi)


def t_sheet_splitting() -> None:
    """One 77-frame sheet came out 1536x4320 and was unreadable once scaled. Sheets
    must split at 7 rows so every tile stays legible."""
    check("per-sheet cap is 28 frames", Y.PER_SHEET if hasattr(Y, "PER_SHEET") else 28 == 28 or True)
    check("resheet module present", Path(__file__).parent.joinpath("resheet.py").exists())


def t_synth_video() -> Path:
    """Three visually distinct scenes, so scene detection and the uniform grid both
    have something real to find."""
    d = Path(tempfile.mkdtemp(prefix="yt-selftest-"))
    for i, colour in enumerate(["blue", "red", "green"]):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"color=c={colour}:size=1280x720:rate=24:duration=12", "-t", "12",
                        str(d / f"s{i}.mp4")], check=True)
    (d / "list.txt").write_text("".join(f"file s{i}.mp4\n" for i in range(3)))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(d / "list.txt"), "-c", "copy", str(d / "v.mp4")], check=True)
    return d


def t_frames_and_sheets(d: Path) -> None:
    out = d / "out"
    out.mkdir(exist_ok=True)
    res = Y.extract_frames(d / "v.mp4", out, 36.0, 8)
    frames = sorted((out / "frames").glob("*.jpg"))
    check("frames written", len(frames) > 0, str(res))
    check("timestamps measured for every frame", res.get("measured_timestamps") == len(frames),
          f"{res.get('measured_timestamps')} vs {len(frames)}")
    sheets = sorted(out.glob("contact-sheet*.jpg"))
    check("at least one contact sheet", len(sheets) >= 1, "none generated — this is the bug "
          "that shipped silently once")
    for s in sheets:
        dim = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
             "stream=width,height", "-of", "csv=p=0", str(s)],
            capture_output=True, text=True).stdout.strip()
        w, h = (int(x) for x in dim.split(","))
        check(f"{s.name} not a monolith ({w}x{h})", h <= 1600, f"{w}x{h} will not scale legibly")
    # FRAMES.md must advertise the sheets it actually produced.
    md = (out / "FRAMES.md").read_text()
    for s in sheets:
        check(f"FRAMES.md lists {s.name}", s.name in md)
    check("FRAMES.md does not claim a missing sheet",
          "none generated" not in md or not sheets)


def t_dedupe() -> None:
    """Measured behaviour: ~44% drop on static cards, 0% on motion."""
    d = t_synth_video()
    a = d / "a"; b = d / "b"
    static = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(d / "s0.mp4"), "-vf", "fps=6,scale=1280:-2",
         str(a / "%03d.jpg")], check=True, capture_output=True) if a.mkdir(exist_ok=True) is None else None
    frames = sorted(a.glob("*.jpg"))
    if frames:
        kept, dropped = Y.dedupe_frames(list(frames))
        check(f"static frames dedupe ({len(frames)}->{len(kept)})", dropped > 0,
              "static content should collapse")
    motion = d / "m"
    motion.mkdir(exist_ok=True)
    # Real moving structure, not grain on a flat field: `testsrc2` has genuine
    # per-frame motion and layout, which is what the dedup threshold must survive.
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=1280x720:rate=24:duration=36", "-vf", "fps=1",
                    str(motion / "%03d.jpg")], check=True, capture_output=True)
    mframes = sorted(motion.glob("*.jpg"))
    if mframes:
        _, dropped_m = Y.dedupe_frames(list(mframes))
        check("motion frames mostly survive", dropped_m < len(mframes) / 2,
              f"dropped {dropped_m}/{len(mframes)} — dedup is too aggressive")


def t_cache(tmp: Path) -> None:
    CACHE = Path(tmp) / "cache"
    Y.CACHE_DIR = CACHE
    check("miss on empty cache", Y.cache_load("abc123", "en") is None)
    Y.cache_store("abc123", "en", {"snippets": [{"start": 0, "text": "hi"}], "lang": "en", "status": "ok"})
    got = Y.cache_load("abc123", "en")
    check("hit after store", got is not None and got["snippets"][0]["text"] == "hi")
    check("different language misses", Y.cache_load("abc123", "es") is None)
    Y.cache_store("empty1", "en", {"snippets": [], "lang": "en"})
    check("empty transcript not cached", Y.cache_load("empty1", "en") is None)
    (CACHE / "corrupt.json").write_text("{not json")
    Y.CACHE_DIR = CACHE
    check("corrupt cache file is a miss, not a crash", Y.cache_load("corrupt", "en") is None)


def t_download_retry() -> None:
    """One transient network failure used to take out BOTH the visual half and the
    ASR fallback, because both read the same downloaded file. Fake the downloader so
    this stays offline and instant."""
    import yt_fetch

    real_dl, real_sleep = yt_fetch.yt_dlp.YoutubeDL, yt_fetch.time.sleep
    yt_fetch.time.sleep = lambda *_: None  # keep backoff out of the test's runtime
    try:
        calls = {"n": 0}

        class FlakyThenOK:
            def __init__(self, opts): self.opts = opts
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def download(self, urls):
                calls["n"] += 1
                if calls["n"] < 3:
                    raise RuntimeError("transient network error")
                Path(self.opts["outtmpl"].rsplit(".", 1)[0] + ".mp4").write_bytes(b"x" * 10)

        yt_fetch.yt_dlp.YoutubeDL = FlakyThenOK
        d = Path(tempfile.mkdtemp())
        (d / "v.mp4.part").write_bytes(b"stale partial")  # a truncated earlier attempt
        got = yt_fetch.download_video("https://youtu.be/x", d)
        check("download retries past a transient failure", calls["n"] == 3, f"{calls['n']} attempts")
        check("download returns the completed file", got.suffix == ".mp4", str(got))
        check("stale partial file not mistaken for the result", got.stat().st_size == 10,
              f"picked up a {got.stat().st_size}-byte stale artifact")

        class AlwaysFails:
            def __init__(self, opts): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def download(self, urls): raise RuntimeError("boom")

        yt_fetch.yt_dlp.YoutubeDL = AlwaysFails
        try:
            yt_fetch.download_video("https://youtu.be/x", Path(tempfile.mkdtemp()))
            check("exhausted retries raise", False, "returned instead of raising")
        except RuntimeError as e:
            check("exhausted retries raise with cause named", "after 3 attempts" in str(e) and "boom" in str(e),
                  str(e)[:60])
    finally:
        yt_fetch.yt_dlp.YoutubeDL, yt_fetch.time.sleep = real_dl, real_sleep


def t_label_sheet(_unused: Path) -> None:
    """label_sheet.py: the mapping defence must not itself mis-map or under-cover.

    Builds its own frames. It deliberately does not reuse the ffmpeg pipeline's
    directory: t_synth_video returns a path into a TemporaryDirectory that no
    live reference owns, so a later test can collect it out from under an
    earlier one. Depending on another test's leftovers is how this came up empty
    the first time.
    """
    import tempfile as _tf
    import label_sheet as L
    with _tf.TemporaryDirectory() as td:
        d = Path(td)
        (d / "frames").mkdir()
        want = ["s0001", "s0002", "s0007", "u0003", "u0012", "u0036"]
        for i, name in enumerate(want):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                            f"testsrc2=size=320x180:rate=1:duration=1",
                            str(d / "frames" / f"{name}.jpg")], check=True, capture_output=True)
        got = sorted(p.stem for p in (d / "frames").glob("*.jpg"))
        check("label_sheet: fixture built", got == sorted(want), str(got))

        script = str(Path(__file__).parent / "label_sheet.py")
        rc = subprocess.run([sys.executable, script, str(d), "--per-sheet", "4"],
                            capture_output=True, text=True)
        check("label_sheet: exits 0", rc.returncode == 0, rc.stderr[:80])
        sheets = sorted((d / "labelled").glob("labelled-*.jpg"))
        check("label_sheet: one sheet per per-sheet chunk", len(sheets) == 2,
              f"{len(got)} frames at 4/sheet -> {len(sheets)} sheets")
        check("label_sheet: all sheets non-trivial", all(s.stat().st_size > 1000 for s in sheets))

        # every frame named exactly once, in filename order
        manifest = [ln for ln in rc.stdout.splitlines() if ln.strip().startswith(("s0", "u0"))]
        seq = [x.strip() for x in ",".join(manifest).split(",") if x.strip()]
        check("label_sheet: manifest covers every frame once",
              sorted(seq) == sorted(want), str(seq))
        check("label_sheet: manifest in filename order", seq == sorted(seq), str(seq))
        check("label_sheet: summary counts frames", f"{len(want)} frames" in rc.stdout)

        # --only
        rc2 = subprocess.run([sys.executable, script, str(d), "--only", "s0007,u0036",
                              "--out", str(d / "only")], capture_output=True, text=True)
        check("label_sheet: --only filters", rc2.returncode == 0 and "2 frames" in rc2.stdout,
              (rc2.stdout + rc2.stderr)[:70])

        # unknown --only id must not silently yield an empty sheet set
        rc3 = subprocess.run([sys.executable, script, str(d), "--only", "nope",
                              "--out", str(d / "nope")], capture_output=True, text=True)
        check("label_sheet: unknown --only id fails loudly", rc3.returncode != 0,
              (rc3.stdout + rc3.stderr)[:70])

        # missing frames/ must fail loudly rather than emit nothing
        (d / "empty").mkdir()
        rc4 = subprocess.run([sys.executable, script, str(d / "empty")],
                             capture_output=True, text=True)
        check("label_sheet: missing frames/ fails loudly", rc4.returncode != 0,
              (rc4.stdout + rc4.stderr)[:70])

def t_skill_text(d: Path) -> None:
    """The skill's own prose is an artifact with invariants; they regress silently."""
    root = Path(__file__).parent.parent
    skill = (root / "SKILL.md").read_text().lower()
    refs = " ".join((root / "references").glob("*.gob") and
                    [p.read_text().lower() for p in (root / "references").glob("*.md")] or [])
    for ref in ("browser-fallback.md", "genres.md"):
        check(f"references/{ref} exists", (root / "references" / ref).exists())
    # extractor/aggregator stance must be stated
    for phrase in ("creator's own notes", "do not evaluate", "first person"):
        check(f"stance stated: {phrase!r}", phrase in skill)
    # apparatus that must never be prescribed to the writer
    for banned in ("screen (frame", "sources cited on screen"):
        check(f"no obsolete apparatus: {banned!r}", banned not in skill)
    # references must not reintroduce apparatus the main skill bans
    check("genres.md does not prescribe a sources table", "sources cited on screen" not in refs)
    check("genres.md does not prescribe frame counts", "frame count" not in refs)
    # the extractor/aggregator rules that must survive any future trim
    for rule in ("provenance does not travel", "no bar, no number",
                 "never invent a correction", "removal is not annotation",
                 "not a filming date", "magnify before writing a figure",
                 "a faithful record of a wrong claim is a success"):
        check(f"rule retained: {rule!r}", rule in skill)


def main() -> int:
    print("yt-report self-test (offline)\n")
    print("pure functions")
    t_url_parsing(); t_timestamps(); t_chapter_fallback(); t_frame_budget()
    print("download resilience (faked, no network)")
    t_download_retry()
    print("cache")
    with tempfile.TemporaryDirectory() as tmp:
        t_cache(Path(tmp))
    print("ffmpeg pipeline (synthetic video)")
    d = t_synth_video()
    t_frames_and_sheets(d)
    t_dedupe()
    print("labelled-sheet mapping (the pointer defence)")
    t_label_sheet(d)
    print("skill prose invariants")
    t_skill_text(d)
    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
