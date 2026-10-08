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
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import align as A  # noqa: E402
import build_evidence as B  # noqa: E402
import evidence_model as M  # noqa: E402
import evidence_provenance as P  # noqa: E402
import evidence_validate as E  # noqa: E402
import ocr as O  # noqa: E402
import report as REP  # noqa: E402
import report_fixtures as RFX  # noqa: E402
import report_validate as RV  # noqa: E402
import semantic as SEM  # noqa: E402
import selftest_cache as SC  # noqa: E402
import selftest_doctor as SD  # noqa: E402
import selftest_entities as SE  # noqa: E402
import selftest_numbers as SN  # noqa: E402
import selftest_ocr_perf as SOP  # noqa: E402
import selftest_grounding as SG  # noqa: E402
import selftest_report as SR  # noqa: E402
import selftest_synthesis as SS  # noqa: E402
import selftest_vlm_provider as SP  # noqa: E402
import synthesis as S  # noqa: E402
import vlm as V  # noqa: E402
import yt_fetch as Y  # noqa: E402

from evidence_model import SCHEMA_VERSION

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    # The detail is coerced because a harness that raises while reporting a
    # failure loses the failure and every check after it — the one moment the
    # output is actually needed is the one moment it must not be typed.
    if not isinstance(detail, str):
        detail = json.dumps(detail, default=str)[:400]
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}{'  — ' + detail if detail else ''}")
        FAILURES.append(name)


# The two new suites report through this same function, so a failure in either is
# printed in the same voice and lands in the same FAILURES list. Binding the name
# is the whole integration: there is no second harness to keep in step, and no
# way for one suite's failures to be counted while the other suite's are not.
SR.check = SP.check = SS.check = SG.check = SC.check = check
SD.check = SE.check = SN.check = SOP.check = check


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


def t_multilingual_segmentation() -> None:
    """A German video came out as `Part 1` … `Part 5` with caption blobs inside.

    Two separate failures, and they are separable: the uploader's chapter titles
    were `Part N`, which name nothing; and the caption track arrives as
    multi-sentence blocks whose timestamps are the only structure they have.
    Neither is a language problem, and neither needs a language model to fix.
    """
    # The shape auto-generated German captions actually arrive in: one blob per
    # few seconds of speech, several sentences long, with the only structure
    # being where the blob starts and how long it runs.
    german = [
        {"start": 0.0, "dur": 30.0, "text":
         "Die einsamsten Orte, die es gibt. Voids sind nicht einfach leer, "
         "sondern sie sind merkwürdige, finstere Welten. Millionen Lichtjahre "
         "von uns entfernt. Fast vollkommen frei von Galaxien, Sternen oder "
         "irgendwelchem Licht. Und sie sind nicht einfach unheimlich, sie sind "
         "auf eine Weise unheimlich, die uns fast nichts bedeutet."},
        {"start": 30.0, "dur": 30.0, "text":
         "Im Zentrum eines Void sitzt ein supermassives Schwarzes Loch. "
         "Und doch wissen wir, wie dunkel es wirklich ist. Wir können das Licht "
         "nicht sehen, nur die Wirkung, die es auf die Materie ringsum hat. "
         "Je näher der Ereignishorizont, desto stärker dieser Effekt."},
        {"start": 60.0, "dur": 30.0, "text":
         "Und desto schneller bewegt sich das Licht. Das ist der ganze Grund, "
         "warum Astrophysiker diese Objekte so gerne vermessen. Sie sind "
         "Laboratorien, die die Allgemeine Relativitätstheorie direkt vor der "
         "eigenen Nase prüfen lassen."},
    ]
    split = Y.split_long_snippets(german)
    check("F1 a multi-sentence caption blob is split into sentences",
          len(split) > len(german), f"{len(german)} -> {len(split)}")
    check("F1 every piece is a sentence, not a fragment of one",
          all(x["text"][-1] in ".!?" for x in split), str([x["text"][-6:] for x in split]))
    check("F1 no text is duplicated or lost",
          " ".join(x["text"] for x in split).split() == " ".join(x["text"] for x in german).split())
    def _in_some_blob(t: float) -> bool:
        return any(b["start"] <= t < b["start"] + b["dur"] for b in german)
    check("F1 every piece is timestamped inside the blob it came from",
          all(_in_some_blob(x["start"]) for x in split), str([x["start"] for x in split]))
    check("F1 the first piece of each blob keeps that blob's own start",
          {b["start"] for b in german} <= {x["start"] for x in split},
          str(sorted({x["start"] for x in split})))
    check("F1 the pieces of one blob are ordered and non-overlapping",
          all(b["start"] >= a["start"] for a, b in zip(split, split[1:])), str(split))
    check("F1 a blob already under the cap is untouched",
          len(Y.split_long_snippets([{"start": 1.0, "dur": 2.0, "text": "Ein kurzer Satz."}])) == 1)

    segs = Y.segments_from_chapters(split, [], 90.0)
    titles = [s["title"] for s in segs]
    check("F1 a chapter title of 'Part N' is never produced",
          not any(t.startswith("Part ") for t in titles), str(titles))
    check("F1 every section is titled from the transcript, in its own language",
          all(t and t[0].isupper() and t == t.strip() for t in titles), str(titles))
    check("F1 the titles are German, taken from the speech",
          any("Orte" in t or "Licht" in t or "Objekte" in t for t in titles), str(titles))
    check("F1 every snippet is in exactly one section",
          sorted(i for s in segs for i in [x["start"] for x in s["snippets"]])
          == sorted(x["start"] for x in split),
          f"{sum(len(s['snippets']) for s in segs)} of {len(split)} placed")

    # Uploader chapters still win: a creator's own sectioning is information the
    # transcript-boundary detector does not have.
    segs = Y.segments_from_chapters(split, [{"start_time": 0.0, "title": "Introduction"},
                                            {"start_time": 30.0, "title": "The problem"}], 90.0)
    check("F1 real uploader chapters still win",
          [s["title"] for s in segs] == ["Introduction", "The problem"],
          str([s["title"] for s in segs]))


def t_asr_evidence_round_trip(d: Path) -> None:
    """The ASR path has never been exercised in the real environment.

    Not a real transcription — that costs a model download and minutes of CPU,
    and this test is about the *contract* around it. Captions unavailable, ASR
    invoked, the transcript marked as ASR, the detected language preserved, the
    provenance timestamps intact, and the whole evidence/report pipeline able to
    consume the result. Each of those is a step that can fail silently on its
    own, which is why a single end-to-end run checks all of them.
    """
    import evidence_model as M

    spoken = [
        {"start": 0.0, "dur": 6.0, "text": "Die Latenz liegt bei 100 ms, gemessen im Rechenzentrum."},
        {"start": 6.0, "dur": 6.0, "text": "Der Cache senkt sie auf 40 ms bei gleicher Last."},
    ]
    called: dict[str, object] = {}

    def _no_captions(video_id, lang):
        raise RuntimeError("no en captions on this video (available: de). "
                           "Use --whisper to transcribe the en audio locally instead.")

    def _asr(video, lang):
        called["invoked"] = True
        called["lang"] = lang
        return (" ".join(s["text"] for s in spoken), spoken,
                f"{lang} (ASR: whisper large-v3-turbo)")

    saved = {k: getattr(Y, k) for k in
             ("fetch_meta", "fetch_transcript", "whisper_transcript", "download_video",
              "extract_frames", "cache_load", "cache_store")}
    argv = sys.argv
    try:
        Y.fetch_meta = lambda url: {"id": "asrfixture001", "title": "Latenz",
                                    "channel": "Fixture", "duration": 12.0,
                                    "url": "https://youtu.be/asrfixture001",
                                    "chapters": [], "chapters_available": False}
        Y.fetch_transcript = _no_captions
        Y.whisper_transcript = _asr
        Y.download_video = lambda url, tmp: Path("fake.mp4")
        Y.extract_frames = lambda *a, **k: {"status": "skipped"}
        Y.cache_load = lambda *a, **k: None
        Y.cache_store = lambda *a, **k: None
        import argparse
        out = Y.run_one("https://youtu.be/asrfixture001",
                        argparse.Namespace(url=["https://youtu.be/asrfixture001"], lang="en",
                                           no_asr=False, frames=-1, no_cache=True, no_ocr=True,
                                           vlm="none", vlm_model=None, max_vlm_frames=0,
                                           ocr_backend="auto"), d)
    finally:
        for k, v in saved.items():
            setattr(Y, k, v)
        sys.argv = argv

    # The evidence stage is a separate entry point reading only what the fetch
    # stage wrote, so it is driven the same way a second process would drive it.
    import build_evidence as B
    check("A1 the evidence stage accepts the ASR artifact",
          B.main([str(d), "--no-vlm", "--no-cache", "--quiet", "--detail", "brief"]) == 0)

    check("A1 ASR was invoked when the captions were unavailable",
          called.get("invoked") is True, str(called))
    check("A1 the run is marked as ASR, not as captions",
          out.get("asr") is True and out.get("status") == "ok", json.dumps(out.get("transcript_note")))
    check("A1 the reason the captions failed is preserved",
          "no en captions" in (out.get("transcript_note") or ""), str(out.get("transcript_note")))

    ev = M.load_evidence(d / "evidence.json")
    t = ev.transcript
    check("A1 the transcript header records the ASR backend and model",
          t.get("asr_backend") == "mlx-whisper" and "large-v3-turbo" in (t.get("asr_model") or ""),
          json.dumps(t))
    check("A1 the language the fetcher used is preserved",
          t.get("language") == "en", json.dumps(t))
    check("A1 an ASR transcript is marked as not a human caption track",
          t.get("is_generated") is True, json.dumps(t))
    check("A1 the provenance timestamps survived the ASR path",
          [(s.start_s, s.end_s) for s in ev.snippets] == [(0.0, 6.0), (6.0, 12.0)],
          str([(s.start_s, s.end_s) for s in ev.snippets]))
    spoken_numbers = [i for i in ev.items
                      if i.type == "number" and any(p.source == "transcript" for p in i.provenance)]
    check("A1 the numbers the ASR heard became evidence items",
          {i.value for i in spoken_numbers} == {"100", "40"},
          str([i.value for i in spoken_numbers]))
    check("A1 every spoken number carries a timestamped transcript provenance",
          spoken_numbers and all(any(p.source == "transcript" and p.t_start is not None
                                     and p.t_end is not None for p in i.provenance)
                                for i in spoken_numbers),
          str([(i.value, [p.to_dict() for p in i.provenance]) for i in spoken_numbers]))

    from evidence_validate import validate_evidence
    res = validate_evidence(ev)
    check("A1 the ASR transcript produces valid evidence", res.valid,
          str([e for e in (res.blocking_errors or [])][:3]))

    import report as R
    import report_validate as RV
    rep = R.generate(ev)
    rep_res = RV.validate_report(rep, ev, R.render_markdown(rep))
    check("A1 the report layer consumes the ASR transcript", rep_res.valid,
          str([e["code"] for e in rep_res.blocking_errors[:3]]))
    check("A1 the figures the ASR heard are in the report",
          rep_res.metrics["numeric_facts_in_report"] > 0, json.dumps(rep_res.metrics))

    # And a language the fetcher could not name is detected from the words
    # rather than left null, which is what the German artifact carried.
    hdr = B._transcript_header("none (no_transcript)", True, [
        M.Snippet(id=f"s00{i}", start_s=i * 6.0, end_s=i * 6.0 + 6.0, text=t) for i, t in enumerate(
            ["Die Latenz liegt bei 100 Millisekunden, gemessen im Rechenzentrum.",
             "Der Cache senkt sie auf 40 Millisekunden bei gleicher Last.",
             "Diese Zahl stammt aus einer Messung von letzter Woche."])], 18)
    check("A1 a language the fetcher could not name is detected, not null",
          hdr["language"] == "de", json.dumps(hdr))


def t_alignment_misses_are_measured() -> None:
    """65% of snippets on a real video got no frame. Is that restraint or loss?

    It is not the threshold, and it is not mostly the floor. On a video sampled
    at roughly one frame every thirty seconds, a +/-4s window contains a frame
    about 3% of the time: the aligner is being starved, not being strict. That is
    a fact about the frame budget, and it has to be measurable before anyone
    changes a number in response to it — so the four buckets are computed on
    every run and asserted to be exhaustive.
    """
    import align as AL
    ev = _build([(0.0, 3.0, "The first thing to say is this."), (3.0, 3.0, "And then the second.")],
                [_frame_rec("s0001", 1.5, [_region("a table of results", 10, 10)]),
                 _frame_rec("s0002", 400.0, [_region("a different slide", 10, 10)])])
    recs, metrics = AL.align_evidence(ev)
    reasons = metrics["no_frame_reasons"]
    check("F2 every unaligned snippet is classified into one of the four buckets",
          set(reasons) == {"speech_only", "visual_candidate_missed",
                           "alignment_missed", "unknown"}, str(reasons))
    check("F2 the buckets account for every unaligned snippet",
          sum(reasons.values()) == metrics["no_relevant_frame"],
          f"{reasons} vs {metrics['no_relevant_frame']}")
    check("F2 a snippet with a frame in its window is aligned",
          metrics["aligned_snippets"] >= 1, json.dumps(metrics["methods"]))

    # A snippet whose only nearby frame is blank is genuinely speech-only, and
    # a snippet with nothing sampled near it is a sampling gap. The two must not
    # be reported as the same thing, which is the whole reason for the buckets.
    blank = _frame_rec("s0003", 6.0, [])
    blank["ocr_text"] = ""
    blank["ocr_confidence"] = None
    ev2 = _build([(0.0, 3.0, "Speech over nothing."), (100.0, 3.0, "Speech elsewhere.")],
                 [blank, _frame_rec("s0004", 108.0, [_region("just outside the window", 10, 10)])])
    _r2, m2 = AL.align_evidence(ev2)
    check("F2 a blank frame in the window counts as speech-only",
          m2["no_frame_reasons"]["speech_only"] >= 1, json.dumps(m2["no_frame_reasons"]))
    check("F2 a frame with content just outside the window is a missed candidate",
          m2["no_frame_reasons"]["visual_candidate_missed"] >= 1,
          json.dumps(m2["no_frame_reasons"]))


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
    """The budget is now a ceiling, not an answer.

    It used to be `duration / 30` clamped to 48 — one number doing two jobs:
    deciding how densely to sample *and* capping how many frames to write. On an
    18-minute maths animation that produced one frame every 30.3 seconds, so a
    ±4s alignment window contained a sampled frame 3% of the time, and the 67
    unaligned snippets that were never sampled were indistinguishable from the
    117 that were genuinely speech-only. The ceiling is now sized on the
    *densest* profile, so a caller guessing the content profile wrong cannot cap
    the run below what the video needs, and the profile decides how much of the
    ceiling to spend.
    """
    finest = min(Y.VISUAL_PROFILES.values())
    for dur, label in [(330, "5.5 min"), (1120, "18.7 min"), (2760, "46 min"),
                       (3240, "54 min"), (None, "unknown duration")]:
        b = Y.auto_frame_budget(dur)
        want = round((dur or 600) / finest)
        check(f"auto budget {label}: {b} is the dense-profile ceiling",
              b == max(12, min(Y.MAX_AUTO_FRAMES, want)), f"got {b}, wanted {want}")
        check(f"auto budget {label}: a talking head is still affordable under it",
              b >= round((dur or 600) / Y.VISUAL_PROFILES["talking_head"]),
              f"{b} is fewer than the sparse profile needs")
    check("auto budget is clamped at the top", Y.auto_frame_budget(100000) == Y.MAX_AUTO_FRAMES)
    check("auto budget has a floor for a tiny clip", Y.auto_frame_budget(5) >= 12)


def t_sheet_splitting() -> None:
    """One 77-frame sheet came out 1536x4320 and was unreadable once scaled. Sheets
    must split at 7 rows so every tile stays legible."""
    import resheet as RS  # noqa: PLC0415
    check("per-sheet cap is 28 frames", RS.PER_SHEET == 28, str(RS.PER_SHEET))
    # A cap that is defined but not applied is the defect the split exists for:
    # 28 is 4 columns by 7 rows, and 7 rows is what keeps a tile legible when
    # the sheet is scaled down. Assert the shape, not just the product.
    check("per-sheet cap is 4 columns by 7 rows", (RS.COLS, RS.MAX_ROWS) == (4, 7),
          f"{RS.COLS}x{RS.MAX_ROWS}")
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
    refs = " ".join([p.read_text().lower() for p in (root / "references").glob("*.md")])
    for ref in ("browser-fallback.md", "genres.md"):
        check(f"references/{ref} exists", (root / "references" / ref).exists())
    # stance that survives the contract rewrite: report the material, never evaluate it
    check("stance stated: 'do not evaluate'", "do not evaluate" in skill)
    # evidence.json is now the canonical record; the skill must say so, or a writer
    # will keep reading the transcript and inventing its own citations
    check("evidence.json named as the canonical record", "evidence.json" in skill)
    # The report is generated. If the skill ever tells a reader to compose one by
    # hand, the renderer stops being the thing that guarantees grounding and the
    # old fabrication class comes straight back.
    check("report is generated, not hand-written",
          "you do not write `report.md`" in skill and "report.md`" in skill)
    check("report.md named as an output artifact", "report.md" in skill)
    check("report.json named as the structure index", "report.json" in skill)
    check("report validation named as the second gate", "report_validate.py" in skill)
    # The old contract. These are the exact phrases that produced a first-person
    # report whose checker had been disarmed by the voice rule that forbade
    # provenance; they must not come back.
    # A banned phrase quoted inside a prohibition is not a contract — it is the
    # rule working. So the scan skips any line carrying a negation, which is the
    # the same discipline the retired `verify_report.py` applied to forbidden
    # provenance labels,
    # and which observation 0020 exists because getting it wrong fabricated a
    # whole report. What must not exist is the phrase being *instructed*.
    negated = re.compile(r"\b(never|not|no|without|forbid\w*|banned|rather than|instead of)\b")
    for banned in ("first person, their voice", "the creator's own notes",
                   "i found", "i noticed", "i break down", "i try"):
        asserted = [ln for ln in skill.splitlines()
                    if banned in ln and not negated.search(ln)]
        check(f"no first-person contract: {banned!r}", not asserted,
              " | ".join(asserted)[:160])
    check("neutral third person is stated", "neutral third person" in skill)
    # Provider configuration has to be documented, or the feature is unreachable.
    for token in ("vlm_provider", "vlm_model", "vlm_base_url", "vlm_api_key",
                  "vlm_timeout", "vlm_max_frames", "--no-vlm", "openrouter",
                  "ollama", "--detail"):
        check(f"vlm/CLI documented: {token}", token in skill.lower())
    check("vlm/smoke test documented", "vlm.py" in skill)
    check("vlm/off by default is stated", "off by default" in skill)
    # Degraded modes are a product requirement, not a nicety.
    check("degraded modes documented", "degraded" in skill and "failed" in skill)
    # Length is evidence-derived, not a fixed number.
    check("detail levels documented", all(d in skill for d in ("brief", "balanced", "deep")))
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
                 "a faithful record of a wrong claim is a success",
                 # The number-relevance and entity rules. Each of these was a
                 # real report defect, and a rule that is not written down
                 # cannot survive the next trim of the file.
                 "a figure has to be about something",
                 "importance` is not `confidence",
                 "table structure alone is not enough",
                 "a capital letter is not a name",
                 "a measurement written in words is not a name",
                 "an untyped name is not a product",
                 # The four qualification states, and the rule that an HTTP 200
                 # is not one of them.
                 "http 200 is never a verdict",
                 "untested", "degraded", "qualified"):
        check(f"rule retained: {rule!r}", rule.lower() in skill.lower())
    # The new commands have to be reachable from the skill's own front door, or
    # a reader who is told about them has to guess where they live.
    for cmd in ("vlm-doctor", "synthesis-doctor", "critic-doctor", "eval-live",
                "release-check"):
        check(f"command documented: {cmd}", cmd in skill)
    # And the stopword resources are part of the contract, not an implementation
    # detail, so their absence has to fail rather than silently disable a filter.
    resources = root / "resources" / "stopwords"
    langs = sorted(p_.stem for p_ in resources.glob("*.txt"))
    check("stopword resources ship for every supported language",
          langs == ["de", "en", "es", "fr", "hi", "it", "pt"], str(langs))


# --------------------------------------------------------------------------- #
# evidence layer
# --------------------------------------------------------------------------- #

def _ev_dict() -> dict:
    """A minimal, valid evidence document used as the mutation base.

    Written as a literal rather than assembled through the builder so a test
    failure points at the validator, not at the builder that produced the input.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "video": {"id": "dQw4w9WgXcQ", "url": "https://youtu.be/dQw4w9WgXcQ", "title": "T",
                  "channel": "C", "upload_date": "20260915", "duration_s": 100.0,
                  "description": "", "tags": [], "thumbnail": None,
                  "chapters": [], "sponsors": []},
        "transcript": {"language": "en", "language_source": "captions", "is_generated": False,
                       "asr_backend": None, "asr_model": None, "word_count": 2, "snippet_count": 1,
                       "snippets": [{"id": "s001", "start_s": 1.0, "end_s": 3.0, "text": "a 32GB card"}]},
        "segments": [{"id": "seg-01", "start_s": 0.0, "end_s": 100.0, "title": "Intro",
                      "title_source": "chapter", "snippet_ids": ["s001"], "item_ids": ["itm-01"]}],
        "items": [{"id": "itm-01", "type": "number", "value": "32", "unit": "GB", "display": "32GB",
                   "verbatim": "a 32GB card", "status": "asserted", "confidence": 0.9,
                   "provenance": [{"source": "transcript", "ref": "s001", "t_start": 1.0,
                                   "t_end": 3.0, "confidence": 0.97}]}],
        "visual": [{"frame_id": "u0012", "file": "frames/u0012.jpg", "kind": "uniform",
                    "t_start": 50.0, "upscale": False}],
        "alignment": [], "conflicts": [], "validation": {},
    }


def _mutate(fn):
    doc = _ev_dict()
    fn(doc)
    return doc


def _codes(doc: dict) -> list[str]:
    return E.validate_evidence(doc).error_codes()


def t_evidence_valid() -> None:
    r = E.validate_evidence(_ev_dict())
    check("valid evidence passes", r.valid, str(r.blocking_errors))
    check("valid evidence has full coverage", r.provenance_coverage == 1.0)
    check("valid evidence has no transcript loss", r.transcript_loss_s == 0)
    check("valid evidence has no unmeasured frames", r.unmeasured_frames == [])


def t_evidence_malformed() -> None:
    for name, doc in [
        ("not an object", []),
        ("missing schema_version", {k: v for k, v in _ev_dict().items() if k != "schema_version"}),
        ("wrong schema_version", _mutate(lambda d: d.__setitem__("schema_version", 99))),
        ("video not an object", _mutate(lambda d: d.__setitem__("video", "nope"))),
        ("items not an array", _mutate(lambda d: d.__setitem__("items", {}))),
    ]:
        r = E.validate_evidence(doc)
        check(f"malformed rejected: {name}", not r.valid, "accepted a malformed document")
    check("unparseable file is one blocking error, not a crash",
          E.validate_file("/nonexistent/evidence.json")[1].error_codes() == ["invalid-schema"])


def t_evidence_provenance() -> None:
    check("missing provenance fails", "missing-provenance" in _codes(_mutate(lambda d: d["items"][0].pop("provenance"))))
    check("empty provenance fails", "missing-provenance" in _codes(_mutate(lambda d: d["items"][0].__setitem__("provenance", []))))
    check("dangling transcript ref fails",
          "dangling-transcript-ref" in _codes(_mutate(lambda d: d["items"][0]["provenance"][0].__setitem__("ref", "s999"))))
    check("dangling frame ref fails", "dangling-frame-ref" in _codes(
        _mutate(lambda d: d["items"][0]["provenance"].append(
            {"source": "frame", "ref": "u9999", "confidence": 0.9}))))
    check("dangling metadata ref fails", "dangling-metadata-ref" in _codes(
        _mutate(lambda d: d["items"][0]["provenance"].append(
            {"source": "metadata", "ref": "nonexistent_field"}))))
    check("dangling refs are reported", any(
        E.validate_evidence(_mutate(lambda d: d["items"][0]["provenance"].append(
            {"source": "frame", "ref": "u9999"}))).dangling_refs))


def t_evidence_confidence_and_time() -> None:
    for bad in (1.5, -0.1, 2, "high", True):
        check(f"invalid confidence fails: {bad!r}", "invalid-confidence" in _codes(
            _mutate(lambda d, b=bad: d["items"][0].__setitem__("confidence", b))))
        check(f"invalid provenance confidence fails: {bad!r}", "invalid-confidence" in _codes(
            _mutate(lambda d, b=bad: d["items"][0]["provenance"][0].__setitem__("confidence", b))))
    check("negative timestamp fails", "impossible-timestamp" in _codes(
        _mutate(lambda d: d["transcript"]["snippets"][0].__setitem__("start_s", -1))))
    check("string timestamp fails", "invalid-timestamp" in _codes(
        _mutate(lambda d: d["transcript"]["snippets"][0].__setitem__("start_s", "12"))))
    check("timestamp beyond duration fails", "timestamp-beyond-duration" in _codes(
        _mutate(lambda d: d["transcript"]["snippets"][0].__setitem__("start_s", 500.0))))
    check("negative duration fails", "invalid-duration" in _codes(
        _mutate(lambda d: d["video"].__setitem__("duration_s", -5))))
    check("frame timestamp must match the frame index", "frame-timestamp-mismatch" in _codes(
        _mutate(lambda d: d["items"][0]["provenance"].append(
            {"source": "frame", "ref": "u0012", "t_start": 77.0}))))


def t_evidence_ids() -> None:
    check("duplicate item ids fail", "duplicate-id" in _codes(_mutate(
        lambda d: d["items"].append(dict(d["items"][0])))))
    check("duplicate segment ids fail", "duplicate-id" in _codes(_mutate(
        lambda d: d["segments"].append(dict(d["segments"][0])))))
    check("duplicate snippet ids fail", "duplicate-id" in _codes(_mutate(
        lambda d: d["transcript"]["snippets"].append(dict(d["transcript"]["snippets"][0])))))
    check("duplicate frame ids fail", "duplicate-id" in _codes(_mutate(
        lambda d: d["visual"].append(dict(d["visual"][0])))))
    check("conflict naming a missing item fails", "dangling-item-ref" in _codes(_mutate(
        lambda d: d["conflicts"].append(
            {"item_ids": ["itm-01", "itm-99"], "kind": "value-mismatch",
             "statement": "x", "resolution": "unresolved"}))))
    check("segment naming a missing item fails", "dangling-item-ref" in _codes(_mutate(
        lambda d: d["segments"][0].__setitem__("item_ids", ["itm-77"]))))
    check("alignment naming a missing frame fails", "dangling-frame-ref" in _codes(_mutate(
        lambda d: d["alignment"].append(
            {"snippet_id": "s001", "frame_id": "u0001", "confidence": 0.5, "method": "time-nearest"}))))
    check("snippet count mismatch fails", "snippet-count-mismatch" in _codes(_mutate(
        lambda d: d["transcript"].__setitem__("snippet_count", 5))))


def t_evidence_enums() -> None:
    for field, value, doc_path in [
        ("type", "vibe", ("items", 0, "type")),
        ("status", "vibes", ("items", 0, "status")),
    ]:
        check(f"unsupported item {field} fails", "unsupported-enum" in _codes(
            _mutate(lambda d, p=doc_path, v=value: d[p[0]][p[1]].__setitem__(p[2], v))))
    check("unsupported provenance source fails", "unsupported-enum" in _codes(
        _mutate(lambda d: d["items"][0]["provenance"][0].__setitem__("source", "vibes"))))
    check("unsupported segment title_source fails", "unsupported-enum" in _codes(
        _mutate(lambda d: d["segments"][0].__setitem__("title_source", "vibes"))))
    check("unsupported frame kind fails", "unsupported-enum" in _codes(
        _mutate(lambda d: d["visual"][0].__setitem__("kind", "vibes"))))
    check("unsupported conflict kind fails", "unsupported-enum" in _codes(
        _mutate(lambda d: d["conflicts"].append(
            {"item_ids": ["itm-01"], "kind": "vibes", "statement": "x", "resolution": "unresolved"}))))
    check("unsupported conflict resolution fails", "unsupported-enum" in _codes(
        _mutate(lambda d: d["conflicts"].append(
            {"item_ids": ["itm-01"], "kind": "value-mismatch", "statement": "x", "resolution": "vibes"}))))


def t_evidence_verified_verbatim() -> None:
    check("verified_verbatim without a read fails", "unbacked-verification" in _codes(
        _mutate(lambda d: d["items"][0]["provenance"].append(
            {"source": "frame", "ref": "u0012", "t_start": 50.0, "verified_verbatim": True}))))
    ok = _mutate(lambda d: (d["visual"][0].update(
        {"ocr_text": "a 32GB card", "ocr_status": "ok", "ocr_backend": "t",
         "ocr_confidence": 0.95, "width": 1280, "height": 720,
         "ocr_regions": [{"text": "a 32GB card", "confidence": 0.95, "bbox": [1, 2, 30, 40]}]}),
        d["items"][0]["provenance"].append(
            {"source": "ocr", "ref": "u0012", "t_start": 50.0, "ocr_text": "32GB",
             "read_method": "native", "confidence": 0.95, "verified_verbatim": True})))
    check("verified_verbatim with a real read passes", E.validate_evidence(ok).valid,
          str(E.validate_evidence(ok).blocking_errors))
    # The frame's own stored read is the only thing that can back a figure, so a
    # citation naming a different one is a fabrication dressed as a citation.
    def _cite_a_figure_the_frame_never_showed(d: dict) -> None:
        d["visual"][0].update({"ocr_text": "a 64GB card"})
        d["items"][0]["provenance"].append(
            {"source": "ocr", "ref": "u0012", "t_start": 50.0, "ocr_text": "32GB",
             "read_method": "native", "confidence": 0.95, "verified_verbatim": True})

    check("a citation naming a figure the frame never showed fails",
          "ocr-text-not-in-frame" in _codes(_mutate(_cite_a_figure_the_frame_never_showed)))
    check("ocr_provenance refuses empty text", _raises(
        lambda: P.ocr_provenance(E.VisualFrame(frame_id="u1", file="f"), "  ")))


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


def t_evidence_segments() -> None:
    lost = _mutate(lambda d: d["segments"][0].__setitem__("snippet_ids", []))
    r = E.validate_evidence(lost)
    check("unassigned snippet blocks", not r.valid and "transcript-unassigned" in r.error_codes())
    check("transcript loss is measured in seconds", r.transcript_loss_s == 2.0, str(r.transcript_loss_s))
    dup = _mutate(lambda d: d["segments"].append(
        dict(d["segments"][0], id="seg-02", title="Copy")))
    check("snippet in two segments blocks", "snippet-duplicated" in E.validate_evidence(dup).error_codes())
    # coverage identity: routed snippets == total, no loss and no duplication
    ev = E.Evidence.from_dict(_ev_dict())
    ev.snippets[0].segment_id = "seg-01"
    routed = [s for seg in ev.segments for s in seg.snippet_ids]
    check("segment coverage equals total transcript coverage",
          sorted(routed) == sorted(s.id for s in ev.snippets), f"{routed}")
    check("back-reference agrees with the segment list", ev.snippets[0].segment_id == "seg-01")
    disagree = _mutate(lambda d: d["transcript"]["snippets"][0].__setitem__("segment_id", "seg-09"))
    check("contradictory back-reference blocks", "segment-disagreement" in E.validate_evidence(disagree).error_codes())


def t_evidence_unmeasured() -> None:
    doc = _mutate(lambda d: d["visual"].append(
        {"frame_id": "s0003", "file": "frames/s0003.jpg", "kind": "scene", "upscale": False}))
    r = E.validate_evidence(doc)
    check("unmeasured frame does not block on its own", r.valid, str(r.blocking_errors))
    check("unmeasured frame is listed, not zeroed", r.unmeasured_frames == ["s0003"], str(r.unmeasured_frames))
    check("unmeasured frame carries a null timestamp",
          E.Evidence.from_dict(doc).visual[1].t_start is None)
    check("unmeasured frame raises a warning",
          any(w["code"] == "unmeasured-frames" for w in r.warnings))
    check("a declared unmeasured set must match reality", "unmeasured-frame-disagreement" in _codes(
        _mutate(lambda d: d["validation"].__setitem__("unmeasured_frames", ["s0003"]))))


def t_evidence_shapes() -> None:
    """transcript-only, metadata-only, and mixed must all be legal. An evidence
    layer that only validates a fully-loaded run is a layer that never runs on
    the partial runs, which is most of them."""
    t_only = _ev_dict()
    t_only["visual"] = []
    check("transcript-only evidence passes", E.validate_evidence(t_only).valid)
    m_only = _ev_dict()
    m_only["transcript"] = {"language": None, "language_source": "none", "is_generated": False,
                            "asr_backend": None, "asr_model": None, "word_count": 0, "snippet_count": 0,
                            "snippets": []}
    m_only["segments"] = []
    m_only["items"] = [{"id": "itm-01", "type": "topic", "value": "T", "status": "asserted",
                        "confidence": 1.0, "provenance": [{"source": "metadata", "ref": "title"}]}]
    check("metadata-only evidence passes", E.validate_evidence(m_only).valid,
          str(E.validate_evidence(m_only).blocking_errors))
    mixed = _ev_dict()
    mixed["items"][0]["provenance"].append(
        {"source": "frame", "ref": "u0012", "t_start": 50.0, "ocr_text": "32GB",
         "read_method": "native", "confidence": 0.94, "verified_verbatim": True})
    check("mixed transcript + frame provenance passes", E.validate_evidence(mixed).valid,
          str(E.validate_evidence(mixed).blocking_errors))


def t_evidence_conflicts() -> None:
    doc = _mutate(lambda d: (
        d["items"].append({"id": "itm-02", "type": "number", "value": "64", "unit": "GB",
                           "status": "conflicting", "confidence": 0.8,
                           "provenance": [{"source": "metadata", "ref": "tags"}]}),
        d["conflicts"].append({"item_ids": ["itm-01", "itm-02"], "kind": "value-mismatch",
                               "statement": "32GB here, 64GB there", "resolution": "unresolved"}),
    ))
    r = E.validate_evidence(doc)
    check("an unresolved conflict is valid, not an error", r.valid, str(r.blocking_errors))
    check("conflicting items keep their own status",
          {i.status for i in E.Evidence.from_dict(doc).items} == {"asserted", "conflicting"})


def t_evidence_anti_vacuity() -> None:
    """A validator that passes a broken document is worse than no validator,
    because it is trusted. These two fixtures are the ones a permissive
    implementation lets through."""
    broken_frame = _mutate(lambda d: d["items"][0]["provenance"].append(
        {"source": "frame", "ref": "u9999", "t_start": 10.0, "confidence": 0.9,
         "ocr_text": "32GB", "verified_verbatim": True}))
    r = E.validate_evidence(broken_frame)
    check("ANTI-VACUITY: frame ref u9999 fails", not r.valid and "dangling-frame-ref" in r.error_codes(),
          "validator accepted a citation to a frame that does not exist")
    broken_prov = _mutate(lambda d: d["items"][0].__setitem__("provenance", []))
    r2 = E.validate_evidence(broken_prov)
    check("ANTI-VACUITY: item with no provenance fails",
          not r2.valid and "missing-provenance" in r2.error_codes())
    check("ANTI-VACUITY: coverage reports the gap",
          r2.provenance_coverage == 0.0 and r2.ungrounded_items == ["itm-01"])
    # and the gate the synthesis stage uses must refuse both
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "evidence.json"
        p.write_text(json.dumps(broken_frame))
        check("synthesis gate refuses dangling frame", _raises(lambda: S.load_valid_evidence(p)))
        p.write_text(json.dumps(_ev_dict()))
        check("synthesis gate admits valid evidence",
              S.load_valid_evidence(p).schema_version == SCHEMA_VERSION)


def t_evidence_build() -> None:
    """The builder, on the two sources it can honestly read today."""
    meta = {"id": "vid1", "title": "How to train a model", "channel": "Chan",
            "webpage_url": "https://youtu.be/vid1", "upload_date": "20260915", "duration": 300,
            "description": "Paper: https://example.com/p", "tags": ["ml"],
            "chapters": [{"start_time": 0, "title": "Intro", "end_time": 60}],
            "chapters_available": True}
    snippets = [
        {"start": 1.0, "dur": 4.0, "text": "You need 32GB of fast VRAM for this."},
        {"start": 70.0, "dur": 5.0, "text": "Training could cost $5000 per run, I think."},
        {"start": 200.0, "dur": 3.0, "text": "That took 4 hours on 8 GPUs."},
    ]
    segs = Y.segments_from_chapters(snippets, meta["chapters"], meta["duration"])
    ev = B.build_evidence(meta, snippets, segs,
                          frames_index=[{"frame_id": "u0012", "file": "frames/u0012.jpg",
                                         "kind": "uniform", "t_start": 68.5},
                                        {"frame_id": "s0003", "file": "frames/s0003.jpg",
                                         "kind": "scene", "t_start": None}],
                          lang_used="en (ASR: whisper large-v3-turbo)", used_asr=True)
    r = E.validate_evidence(ev)
    check("built evidence validates", r.valid, str(r.blocking_errors))
    check("built evidence covers the whole transcript", r.transcript_loss_s == 0.0, str(r.transcript_loss_s))
    check("built evidence has full provenance coverage", r.provenance_coverage == 1.0)
    check("unmeasured frame is recorded, not zeroed", ev.visual[1].t_start is None)
    check("ASR transcript is marked generated", ev.transcript["is_generated"] and ev.transcript["asr_model"])
    check("language split out of the fetcher's packed string", ev.transcript["language"] == "en")
    check("asr is not silently a creator statement", ev.transcript["language_source"] == "asr-detected")

    nums = {i.display: i for i in ev.items if i.type == "number"}
    check("spoken number extracted with its unit", "32GB" in nums, str(list(nums)))
    check("number cites the exact snippet", nums["32GB"].provenance[0].ref == "s001")
    check("number carries the verbatim sentence", "fast VRAM" in nums["32GB"].verbatim)
    check("currency figure extracted", "$5000" in nums, str(list(nums)))
    check("a 'could cost' figure stays hedged",
          nums["$5000"].status == "hedged" and "could" in (nums["$5000"].notes or ""),
          f"status={nums['$5000'].status} notes={nums['$5000'].notes}")
    check("an unhedged figure is asserted",
          nums["32GB"].status == "asserted" and nums["32GB"].notes is None)
    check("duration recorded as a number item", any(
        i.type == "number" and i.value == "300" for i in ev.items))
    check("description link captured as a link item", any(
        i.type == "link" and "example.com" in i.value for i in ev.items))
    check("link cites the description field", any(
        i.type == "link" and i.provenance[0].ref == "description" for i in ev.items))
    check("no item is asserted without a source", all(i.provenance for i in ev.items))
    check("no visual content is claimed", all(v.ocr_text is None and not v.figures for v in ev.visual))
    check("segment titles record that chapters are unverified",
          all(ch["verified_against_transcript"] is False for ch in ev.video["chapters"]))
    check("items are wired to their segment",
          any(it.id in seg.item_ids for seg in ev.segments for it in ev.items if it.type == "number"))

    # determinism: same inputs, same document — otherwise every re-run churns
    # every citation in every downstream artifact
    again = B.build_evidence(meta, snippets, segs, frames_index=[
        {"frame_id": "u0012", "file": "frames/u0012.jpg", "kind": "uniform", "t_start": 68.5},
        {"frame_id": "s0003", "file": "frames/s0003.jpg", "kind": "scene", "t_start": None}],
        lang_used="en (ASR: whisper large-v3-turbo)", used_asr=True)
    check("build is deterministic", M.dumps(again) == M.dumps(ev))
    check("evidence.json is valid JSON, not Python NaN", "Infinity" not in M.dumps(ev)
          and "NaN" not in M.dumps(ev))

    with tempfile.TemporaryDirectory() as td:
        p = B.write_evidence(ev, Path(td))
        check("write_evidence emits evidence.json", p.exists())
        check("write_evidence emits validation.json", (Path(td) / "validation.json").exists())
        check("round-trips through disk", M.dumps(M.load_evidence(p)) == M.dumps(ev))
        check("written document is valid when re-read", S.load_valid_evidence(p).schema_version == SCHEMA_VERSION)


def t_evidence_real_shapes() -> None:
    """Two behaviours found by running the builder against real fetched
    artifacts, both of which a synthetic fixture would have hidden."""
    # A caption track's last cue routinely ends a second or two past the
    # platform's reported duration. That is the platform's arithmetic, not a
    # fabricated timestamp, and blocking on it would reject a good transcript.
    tail = _ev_dict()
    tail["transcript"]["snippets"][0]["end_s"] = 102.0
    r = E.validate_evidence(tail)
    check("small tail overrun warns, does not block", r.valid, str(r.blocking_errors))
    check("tail overrun is recorded as a warning",
          any(w["code"] == "timestamp-past-duration-tail" for w in r.warnings))
    far = _ev_dict()
    far["transcript"]["snippets"][0]["end_s"] = 400.0
    check("a large overrun still blocks", "timestamp-beyond-duration" in E.validate_evidence(far).error_codes())

    # A legacy FRAMES.md rendered a missing timestamp as 00:00:00, so a literal
    # zero in one is ambiguous and must not be imported as a fact.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "frames").mkdir()
        for name in ("u0001", "s0002", "u0003"):
            (d / "frames" / f"{name}.jpg").write_bytes(b"x")
        (d / "FRAMES.md").write_text(
            "# Frames\n\n- `u0001.jpg` [00:00:00]\n- `s0002.jpg` [00:12:34]\n- `u0003.jpg` [00:07:05]\n")
        idx = B.load_frames_index(d)
        got = {f["frame_id"]: f["t_start"] for f in idx}
        check("legacy manifest: real times recovered", got.get("s0002") == 754.0 and got.get("u0003") == 425.0, str(got))
        check("legacy manifest: ambiguous zero stays unmeasured", got.get("u0001") is None, str(got))
        check("every frame on disk is represented", len(idx) == 3)

    # A fresh run writes frames.json, which is what later runs cite.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "out"
        d.mkdir()
        res = Y.extract_frames(t_synth_video() / "v.mp4", d, 36.0, 8)
        check("extract_frames writes frames.json", (d / "frames.json").exists(), str(res))
        idx = B.load_frames_index(d)
        check("frames.json carries measured times", all(f.get("t_start") is not None for f in idx), str(idx[:2]))
        check("frames.json frame ids match the files on disk",
              {f["frame_id"] for f in idx} == {p.stem for p in (d / "frames").glob("*.jpg")})
        check("a fully measured run marks nothing unmeasured", "unmeasured" not in (d / "FRAMES.md").read_text())


# --------------------------------------------------------------------------- #
# OCR layer
# --------------------------------------------------------------------------- #

#: Fixtures: the text rendered on each, and the substrings that must come back
#: exactly.
#:
#: The required substrings are the evidence-bearing parts — a figure, a price, a
#: URL, an identifier — because those are what a citation would rest on. The
#: surrounding punctuation is the engine's business: it renders `json.loads(` as
#: `json. loads (` and small text as `secor` for `second`, and a check that
#: demanded the original punctuation would be asserting that one particular
#: engine was used rather than that the content was read. Those character errors
#: are real and are reported as a limitation, not designed away.
OCR_FIXTURES: dict[str, tuple[list[str], str]] = {
    "slide_title": (["Qwen3.8-32B", "Release Notes"], "Qwen3.8-32B Release Notes"),
    "paragraph": (["32 GB", "VRAM"], "Deploying the model requires 32 GB of VRAM on a single card."),
    "terminal": (["$ pip install transformers"], "$ pip install transformers"),
    "source_code": (["def", "load", "json", "read_text"], "def load(p): return json.loads(p.read_text())"),
    "numeric": (["142.7 K"], "Peak throughput 142.7 K tokens per second"),
    "price": (["$5,000"], "List price $5,000 per seat"),
    "url": (["https://github.com/QwenLM/Qwen3"], "https://github.com/QwenLM/Qwen3"),
    "table": (["32 GB", "0.20", "64 GB", "2.0"], "Model A   32 GB  0.20\nModel B   64 GB  2.0"),
}

SKIPPED: list[str] = []


def skip(name: str, why: str) -> None:
    """Record a check that could not run.

    Distinct from `check(..., True)`: a skipped check proves nothing, and the
    summary says so. A test suite that reports an OCR check as passing on a
    machine with no OCR backend has told the reader something false.
    """
    print(f"  skip {name} — {why}")
    SKIPPED.append(name)


def _norm(s: str) -> str:
    """Compare on substance: case and spacing are the engine's business, the
    characters are ours."""
    return re.sub(r"\s+", " ", s or "").strip().casefold()


def _ocr_fixtures(d: Path) -> Path:
    """Render the fixture set: eight legible cases and one that must not be read.

    The unreadable fixture is blurred past legibility, not merely small. A
    backend's failure mode on unreadable input is invention, so the only
    meaningful assertion is that it returns nothing — an empty string would be
    just as much a fabrication.
    """
    from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: PLC0415

    out = d / "frames"
    out.mkdir(parents=True, exist_ok=True)
    sans = "/System/Library/Fonts/Supplemental/Arial.ttf"
    mono = "/System/Library/Fonts/Menlo.ttc"
    big = ImageFont.truetype(sans, 64)
    small = ImageFont.truetype(sans, 26)
    code = ImageFont.truetype(mono, 28)

    def canvas():
        im = Image.new("RGB", (1280, 720), "white")
        return im, ImageDraw.Draw(im)

    for name, (_, text) in OCR_FIXTURES.items():
        im, dr = canvas()
        font = code if name in ("terminal", "source_code") else (
            small if name in ("paragraph", "table") else big)
        dr.multiline_text((70, 90), text, font=font, fill="black", spacing=18)
        im.save(out / f"{name}.png")

    # Unreadable: the same words, blurred past the point of reading them.
    im, dr = canvas()
    dr.text((70, 90), "SECRET $9,999 SKU-4471", font=big, fill="black")
    im.filter(ImageFilter.GaussianBlur(16)).save(out / "unreadable.png")
    return out


def t_ocr_fixtures(d: Path) -> None:
    """Read every fixture and check it against what was rendered on it."""
    provider = O.get_provider()
    if not provider.available:
        for name in OCR_FIXTURES:
            skip(f"ocr fixture: {name}", "no OCR backend on this machine")
        skip("unreadable fixture is not hallucinated", "no OCR backend on this machine")
        return

    frames = _ocr_fixtures(d)
    results = O.extract_many_cached(provider, sorted(str(p) for p in frames.glob("*.png")))

    for name, (must, _) in OCR_FIXTURES.items():
        r = results[f"{frames}/{name}.png"]
        check(f"fixture {name}: read at all", r.text is not None, f"status={r.status}")
        if r.text:
            missing = [m for m in must if _norm(m) not in _norm(r.text)]
            check(f"fixture {name}: evidence text exact", not missing,
                  f"missing {missing!r} from {r.text!r}")
        check(f"fixture {name}: confidence recorded in [0,1]",
              r.confidence is not None and 0.0 <= r.confidence <= 1.0, str(r.confidence))
        check(f"fixture {name}: backend named", bool(r.backend), r.backend)
        bad = [rg.bbox for rg in r.regions
               if len(rg.bbox) != 4 or rg.bbox[2] <= 0 or rg.bbox[3] <= 0
               or rg.bbox[0] < 0 or rg.bbox[1] < 0
               or rg.bbox[0] + rg.bbox[2] > 1281 or rg.bbox[1] + rg.bbox[3] > 721]
        check(f"fixture {name}: every box lies on the frame", not bad, str(bad[:2]))

    unread = results[f"{frames}/unreadable.png"]
    # The assertion is that nothing legible was produced, not that the string is
    # empty: an invented "SECRET $9,999" would be a fabrication dressed as a read.
    check("unreadable fixture yields no text, not a guess",
          unread.text is None or not _norm("9,999") in _norm(unread.text or ""),
          f"engine returned {unread.text!r}")
    check("unreadable fixture is not marked verbatim-eligible",
          unread.confidence is None or unread.confidence < O.HIGH_CONFIDENCE,
          str(unread.confidence))


def t_ocr_result_invariants() -> None:
    check("status 'ok' with no text is refused",
          _raises(lambda: O.OCRResult(text=None, confidence=None, regions=[], status="ok",
                                      backend="x")))
    check("confidence outside [0,1] is refused",
          _raises(lambda: O.OCRResult(text="x", confidence=1.5, regions=[], status="ok",
                                      backend="x")))
    check("an empty read is representable",
          O.OCRResult(text=None, confidence=None, regions=[], status="no_text",
                      backend="x").status == "no_text")
    check("an unread frame and a low-confidence one are different states",
          O.OCRResult(text=None, confidence=None, regions=[], status="no_text",
                      backend="x").confidence_band == "none"
          and O.OCRResult(text="x", confidence=0.3, regions=[], status="ok",
                          backend="x").confidence_band == "low")
    r = O.OCRResult(text="32 GB", confidence=0.95, regions=[], status="ok", backend="x")
    check("a result round-trips through its dict form",
          O.OCRResult.from_dict(r.to_dict()).text == "32 GB")


def t_ocr_normalisation() -> None:
    """Dedup may not make values equal that the pixels kept apart.

    These four pairs are the ones a greedy normaliser merges and a reader then
    cannot tell apart. They are the reason `normalize_text` touches whitespace
    and case and nothing else.
    """
    for a, b in (("$500", "$5,000"), ("7B", "70B"), ("0.20", "2.0"), ("32 GB", "64 GB")):
        check(f"dedup keeps {a!r} and {b!r} distinct", O.normalize_text(a) != O.normalize_text(b))
    check("dedup does fold case and spacing",
          O.normalize_text("  Qwen3.8  Release\n") == O.normalize_text("qwen3.8 release"))


def t_ocr_dedup() -> None:
    """Repeated text is recorded once and traced to every frame that showed it."""
    def r(text, conf=0.95):
        return O.OCRResult(text=text, confidence=conf, regions=[],
                           status="ok" if text else "no_text", backend="t")

    frames = [{"frame_id": f"u{i:04d}", "file": f"frames/u{i:04d}.jpg", "t_start": float(i)}
              for i in range(1, 6)]
    results = {f"u{i:04d}": r("Qwen3.8" if i <= 3 else f"value {i}") for i in range(1, 6)}
    obs = O.group_observations(frames, results)
    check("the first occurrence carries the text", obs[0].result is not None and obs[0].result.text == "Qwen3.8")
    check("adjacent repeats reference the first frame",
          [o.same_as for o in obs[1:3]] == ["u0001", "u0001"], str([o.same_as for o in obs]))
    check("a repeat carries no text of its own", obs[1].result is None)
    check("a changed frame is not folded into the run", obs[3].same_as is None and obs[3].result.text == "value 4")
    check("each frame keeps its own timestamp",
          [o.t_start for o in obs] == [1.0, 2.0, 3.0, 4.0, 5.0], str([o.t_start for o in obs]))
    check("a frame with no text breaks the run",
          O.group_observations([{"frame_id": "a", "t_start": 1.0}, {"frame_id": "b", "t_start": 2.0},
                                {"frame_id": "c", "t_start": 3.0}],
                               {"a": r("same"), "b": r(None, 0), "c": r("same")})[2].same_as is None)
    # The same two values one character apart must never reach the same group.
    paired = O.group_observations(
        [{"frame_id": "a", "t_start": 1.0}, {"frame_id": "b", "t_start": 2.0},
         {"frame_id": "c", "t_start": 3.0}, {"frame_id": "d", "t_start": 4.0}],
        {"a": r("$500"), "b": r("$5,000"), "c": r("7B"), "d": r("70B")})
    check("near-identical values are not merged into one observation",
          all(o.result is not None for o in paired), str([o.same_as for o in paired]))


def t_ocr_cache(d: Path) -> None:
    """A cache that re-reads unchanged frames is not a cache; one that serves a
    stale read after a config change is worse than none."""
    calls: list[list[str]] = []

    class Counting(O.OCRProvider):
        name = "counting"

        def __init__(self) -> None:
            self.version = "v1"
            self.languages = ["en-US"]
            self.min_text_height = 0.0

        @property
        def available(self) -> bool:
            return True

        def extract(self, image_path):
            return O.OCRResult(text="read", confidence=0.9, regions=[], status="ok",
                               backend=self.name)

        def extract_many(self, image_paths):
            calls.append(list(image_paths))
            return {p: self.extract(p) for p in image_paths}

    orig_cache = O.CACHE_DIR
    O.CACHE_DIR = d / "cache"
    try:
        (d / "a.jpg").write_bytes(b"alpha")
        (d / "b.jpg").write_bytes(b"beta")
        p = Counting()
        paths = [str(d / "a.jpg"), str(d / "b.jpg")]
        first = O.extract_many_cached(p, paths)
        check("a cold cache reads every frame", len(calls) == 1 and len(calls[0]) == 2)
        check("a cold read is not marked cached", not any(r.cached for r in first.values()))
        O.extract_many_cached(p, paths)
        check("a warm cache reads nothing", len(calls) == 1, f"{len(calls)} backend calls")
        check("a warm read is marked cached", all(r.cached for r in
                                                  O.extract_many_cached(p, paths).values()))
        # Changing the backend must not serve the old backend's answer.
        p2 = Counting()
        p2.version = "v2"
        O.extract_many_cached(p2, paths)
        check("a different backend version invalidates the cache", len(calls) == 2, f"{len(calls)} calls")
        p2.languages = ["fr-FR"]
        O.extract_many_cached(p2, paths)
        check("a different language config invalidates the cache", len(calls) == 3, f"{len(calls)} calls")
        # Content, not path: the same bytes elsewhere is the same read.
        (d / "copy-of-a.jpg").write_bytes(b"alpha")
        O.extract_many_cached(p2, [str(d / "copy-of-a.jpg")])
        check("the cache is keyed on content, not on the path", len(calls) == 3, f"{len(calls)} calls")
        (d / "a.jpg").write_bytes(b"alpha-changed")
        O.extract_many_cached(p2, [str(d / "a.jpg")])
        check("edited pixels invalidate their own entry", len(calls) == 4, f"{len(calls)} calls")
        # A corrupt entry is a miss, not a crash.
        for cf in O.CACHE_DIR.rglob("*.json"):
            cf.write_text("{not json")
        O.extract_many_cached(p2, paths)
        check("a corrupt cache entry is a miss, not a crash", len(calls) == 5, f"{len(calls)} calls")
    finally:
        O.CACHE_DIR = orig_cache


def _ocr_doc(**visual_over) -> dict:
    """A valid document whose single frame carries an OCR read.

    Written as a literal so a validator test failure points at the validator and
    not at the builder that produced the input.
    """
    doc = _ev_dict()
    doc["visual"][0].update({"width": 1280, "height": 720, "ocr_text": "Price $5,000 per seat",
                             "ocr_confidence": 0.95, "ocr_status": "ok",
                             "ocr_backend": "apple-vision", "ocr_read_method": "native",
                             "ocr_regions": [{"text": "Price $5,000 per seat", "confidence": 0.95,
                                              "bbox": [100, 200, 300, 40], "read_method": "native"}]})
    doc["items"].append({"id": "itm-02", "type": "number", "value": "5000", "unit": "$",
                         "display": "$5,000", "status": "asserted", "confidence": 0.95,
                         "verbatim": "$5,000",
                         "provenance": [{"source": "ocr", "ref": "u0012", "t_start": 50.0,
                                         "t_end": 50.0, "ocr_text": "$5,000", "read_method": "native",
                                         "confidence": 0.95, "verified_verbatim": True}]})
    for k, v in visual_over.items():
        doc["visual"][0][k] = v
    return doc


def _ocr_mutate(fn):
    """Mutate the OCR document rather than the base one: these rules are about a
    citation that has a read behind it, and the base document has none."""
    doc = _ocr_doc()
    fn(doc)
    return doc


def t_ocr_evidence() -> None:
    """The builder, on a read it can honestly carry."""
    provider = O.UnavailableProvider("test")
    visual = {"version": 1, "backend": "apple-vision", "frames": [
        {"frame_id": "u0001", "t_start": 10.0, "text": "The card has 32 GB",
         "result": {"status": "ok", "text": "The card has 32 GB", "confidence": 0.96,
                    "backend": "apple-vision", "read_method": "native", "width": 1280, "height": 720,
                    "regions": [{"text": "The card has 32 GB", "confidence": 0.96,
                                 "bbox": [10, 20, 300, 40], "read_method": "native"}]}},
        {"frame_id": "u0002", "t_start": 20.0, "text": "The card has 32 GB",
         "result": {"status": "ok", "text": "The card has 32 GB", "confidence": 0.93,
                    "backend": "apple-vision", "read_method": "native", "width": 1280, "height": 720,
                    "regions": []}},
        {"frame_id": "u0003", "t_start": 30.0, "same_as": "u0001"},
        {"frame_id": "u0004", "t_start": None,
         "result": {"status": "unavailable", "text": None, "confidence": None,
                    "backend": "unavailable", "notes": "no backend"}},
    ]}
    meta = {"id": "v", "title": "T", "channel": "C", "webpage_url": "https://youtu.be/v",
            "upload_date": "20260915", "duration": 60, "description": "", "tags": [],
            "chapters": [{"start_time": 0, "title": "All", "end_time": 60}], "chapters_available": True}
    snippets = [{"start": 1.0, "dur": 4.0, "text": "The table here shows the memory."}]
    segs = Y.segments_from_chapters(snippets, meta["chapters"], 60)
    frames = [{"frame_id": f"u{i:04d}", "file": f"frames/u{i:04d}.jpg", "t_start": t}
              for i, t in ((1, 10.0), (2, 20.0), (3, 30.0), (4, None))]
    ev = B.build_evidence(meta, snippets, segs, frames_index=frames, lang_used="en", visual=visual)
    r = E.validate_evidence(ev)
    check("an OCR-backed document validates", r.valid, str(r.blocking_errors[:3]))
    check("the read lands in the visual representation",
          ev.visual[0].ocr_text == "The card has 32 GB" and ev.visual[0].ocr_status == "ok")
    check("the read records its backend", ev.visual[0].ocr_backend == "apple-vision")
    check("the read records its confidence", ev.visual[0].ocr_confidence == 0.96)
    check("a repeated frame names the frame that carries the text",
          ev.visual[2].ocr_text is None and ev.visual[2].ocr_status == "repeat")
    check("an unmeasured frame is still unmeasured", ev.visual[3].t_start is None)
    check("an unavailable read is recorded, not dropped", ev.visual[3].ocr_status == "unavailable")
    check("the derived run status is the worst frame's",
          r.ocr_status == "unavailable", str(r.ocr_status))
    check("a degraded pass is surfaced as a warning",
          any(w["code"] == "ocr-degraded" for w in r.warnings))

    # A number on screen is NOT an item yet. Turning one into evidence needs to
    # know what it refers to, and that is the semantic stage's decision, not this
    # one's. Asserting the absence here is what stops a mechanical pass quietly
    # growing back a second opinion about what is meaningful.
    check("a screen-only number is not an item at the build stage",
          not [i for i in ev.items if any(p.source == "ocr" for p in i.provenance)],
          str([i.id for i in ev.items if any(p.source == "ocr" for p in i.provenance)]))

    # A low-confidence read stays an observation and never a verification.
    low = B.build_evidence(meta, snippets, segs, frames_index=frames, lang_used="en",
                           visual={"frames": [{"frame_id": "u0001", "t_start": 10.0,
                                               "text": "The card has 32 GB",
                                               "result": {"status": "ok", "text": "The card has 32 GB",
                                                          "confidence": 0.55, "backend": "apple-vision",
                                                          "read_method": "native", "width": 1280, "height": 720,
                                                          "regions": []}}]})
    check("a medium-confidence read is stored on the frame",
          low.visual[0].ocr_confidence == 0.55 and not low.items[0].provenance[0].source == "ocr")
    check("a document with a low-confidence read is still valid",
          E.validate_evidence(low).valid, str(E.validate_evidence(low).blocking_errors[:2]))


def t_ocr_validation() -> None:
    """One check per rule. A validator nobody has tried to fool is not a check."""
    check("a valid OCR document passes", E.validate_evidence(_ocr_doc()).valid,
          str(E.validate_evidence(_ocr_doc()).blocking_errors))

    # 1. provenance must reference an existing frame
    check("1 ocr provenance must name a real frame", "dangling-frame-ref" in E.validate_evidence(
        _ocr_mutate(lambda d: d["items"][1]["provenance"][0].__setitem__("ref", "u9999"))).error_codes())
    # 2. timestamp must match the frame index
    check("2 ocr timestamp must match the index", "frame-timestamp-mismatch" in E.validate_evidence(
        _ocr_mutate(lambda d: d["items"][1]["provenance"][0].__setitem__("t_start", 77.0))).error_codes())
    check("2b an unmeasured frame cannot be given a timestamp",
          "frame-timestamp-fabricated" in E.validate_evidence(_ocr_mutate(
              lambda d: (d["visual"][0].__setitem__("t_start", None),
                         d["items"][1]["provenance"][0].__setitem__("t_start", 12.0)))).error_codes())
    # 3. verified_verbatim needs a read
    check("3 verified_verbatim without a read fails", "unbacked-verification" in E.validate_evidence(
        _ocr_mutate(lambda d: d["items"][1]["provenance"][0].__setitem__("ocr_text", "  "))).error_codes())
    # 4. an empty read must not back an item
    check("4 a frame with no text cannot back an item", "ocr-without-text" in E.validate_evidence(
        _ocr_mutate(lambda d: d["visual"][0].__setitem__("ocr_text", None))).error_codes())
    # 5. boxes must be valid
    for bad, label in (([1, 2, 3], "wrong arity"), ([1, 2, "x", 4], "non-numeric"),
                       ([0, 0, 0, 40], "zero width"), ([-5, 0, 100, 40], "off the left edge"),
                       ([1000, 600, 400, 300], "past the frame edge")):
        check(f"5 bbox rejected: {label}", "invalid-bbox" in E.validate_evidence(
            _ocr_mutate(lambda d, b=bad: d["visual"][0]["ocr_regions"][0].__setitem__("bbox", b))).error_codes())
    # 6. confidence in [0,1]
    for bad in (1.5, -0.1, "high"):
        check(f"6 ocr confidence rejected: {bad!r}", "invalid-confidence" in E.validate_evidence(
            _ocr_mutate(lambda d, b=bad: (d["visual"][0].__setitem__("ocr_confidence", b),
                                      d["items"][1]["provenance"][0].__setitem__("confidence", b)))).error_codes())
    # 7. a read must identify its origin
    check("7 a read with no backend fails", "ocr-origin-missing" in E.validate_evidence(
        _ocr_mutate(lambda d: d["visual"][0].__setitem__("ocr_backend", ""))).error_codes())
    check("7 a read with no status fails", "ocr-missing-status" in E.validate_evidence(
        _ocr_mutate(lambda d: d["visual"][0].pop("ocr_status"))).error_codes())
    check("7 an unsupported status fails", "unsupported-enum" in E.validate_evidence(
        _ocr_mutate(lambda d: d["visual"][0].__setitem__("ocr_status", "vibes"))).error_codes())
    # 8. no fabricated text — the check the whole layer exists for
    check("8 text the frame never showed is rejected", "ocr-text-not-in-frame" in E.validate_evidence(
        _ocr_mutate(lambda d: (d["items"][1]["provenance"][0].__setitem__("ocr_text", "$4,999"),
                           d["items"][1].__setitem__("display", "$4,999")))).error_codes())
    check("8 the same figure in another frame is still rejected", "ocr-text-not-in-frame" in
          E.validate_evidence(_ocr_mutate(lambda d: d["items"][1]["provenance"][0].__setitem__(
              "ocr_text", "32 GB"))).error_codes())
    # 9. round trip
    doc = M.Evidence.from_dict(_ocr_doc())
    check("9 an ocr read survives a save/load round trip",
          M.dumps(M.Evidence.from_dict(json.loads(M.dumps(doc)))) == M.dumps(doc))
    reloaded = json.loads(M.dumps(doc))["visual"][0]
    check("9 the round trip keeps regions, status and backend",
          reloaded["ocr_regions"] and reloaded["ocr_status"] == "ok" and reloaded["ocr_backend"] == "apple-vision")
    # 10. deliberately corrupted provenance
    for label, fn in (
        ("citation points at a frame that does not exist", lambda d: d["items"][1]["provenance"][0].__setitem__("ref", "ghost")),
        ("citation names a read the frame never made", lambda d: d["items"][1]["provenance"][0].__setitem__("ocr_text", "injected")),
        ("verification asserted on a doubtful read", lambda d: d["items"][1]["provenance"][0].__setitem__("confidence", 0.4)),
        ("frame text swapped for another frame's", lambda d: d["visual"][0].__setitem__("ocr_text", "64 GB")),
    ):
        check(f"10 corrupted provenance rejected: {label}", not E.validate_evidence(
            _ocr_mutate(fn)).valid)

    # Low-confidence verification is refused by the constructor as well as the
    # validator: a caller that builds provenance directly must not be able to
    # launder a doubtful read into a fact.
    f = M.VisualFrame(frame_id="u1", file="f", t_start=1.0, ocr_text="32 GB",
                      ocr_confidence=0.55, ocr_status="ok", ocr_backend="t")
    check("the constructor refuses to verify a low-confidence read",
          not P.ocr_provenance(f, "32 GB", confidence=0.55).verified_verbatim)
    check("the constructor verifies a high-confidence read",
          P.ocr_provenance(f, "32 GB", confidence=0.95).verified_verbatim)
    check("the constructor still refuses empty text", _raises(lambda: P.ocr_provenance(f, "")))


def t_ocr_failure_modes() -> None:
    """A missing backend, a missing frame and an empty directory are all ordinary
    outcomes, and none of them may crash a run or look like a clean result."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "frames").mkdir()
        (d / "frames" / "u0001.jpg").write_bytes(b"not a real image")
        frames = B.load_frames_index(d)
        payload = O.run_ocr(d, frames, O.UnavailableProvider("no backend here"))
        check("an unavailable backend produces an artifact, not a crash",
              payload["summary"]["status"] == "unavailable" and len(payload["frames"]) == 1)
        check("an unavailable backend claims no text",
              payload["frames"][0]["result"]["text"] is None)
        check("an unavailable backend says why",
              "no backend" in (payload["summary"]["notes"] or ""), str(payload["summary"]["notes"]))

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        payload = O.run_ocr(d, [{"frame_id": "u0001", "file": "frames/u0001.jpg", "t_start": 1.0}],
                            O.UnavailableProvider("x"))
        check("a frame the index names but the disk lacks is 'unreadable', not 'no text'",
              payload["frames"][0]["result"]["status"] == "unreadable")

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "frames").mkdir()
        rc = subprocess.run([sys.executable, str(Path(__file__).parent / "ocr.py"), str(d)],
                            capture_output=True, text=True)
        check("ocr on a directory with no frames exits non-zero", rc.returncode == 1, rc.stderr[:120])

    # --no-ocr must leave the same artifact set, not a smaller one.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        frames = [{"frame_id": "u0001", "file": "frames/u0001.jpg", "t_start": 1.0}]
        p = O.write_visual(d, O.skipped_payload(frames, "--no-ocr"))
        check("--no-ocr still writes visual.json", p.exists())
        doc = json.loads(p.read_text())
        check("--no-ocr records why", doc["summary"]["status"] == "skipped"
              and doc["summary"]["reason"] == "--no-ocr", str(doc["summary"]))
        check("--no-ocr claims no text on any frame",
              all((f.get("result") or {}).get("text") is None for f in doc["frames"]))
        ev = B.build_evidence({"id": "v", "title": "T", "channel": "C",
                               "webpage_url": "https://youtu.be/v", "upload_date": "20260915",
                               "duration": 10, "description": "", "tags": [],
                               "chapters": [{"start_time": 0, "title": "All", "end_time": 10}],
                               "chapters_available": True},
                              [{"start": 1.0, "dur": 2.0, "text": "hello world"}],
                              Y.segments_from_chapters(
                                  [{"start": 1.0, "dur": 2.0, "text": "hello world"}],
                                  [{"start_time": 0, "title": "All", "end_time": 10}], 10),
                              frames_index=frames, lang_used="en", visual=doc)
        r = E.validate_evidence(ev)
        check("a skipped pass validates and is surfaced, not hidden",
              r.valid and r.ocr_status == "skipped"
              and any(w["code"] == "ocr-degraded" for w in r.warnings),
              f"valid={r.valid} status={r.ocr_status}")

    rc = subprocess.run([sys.executable, str(Path(__file__).parent / "yt_fetch.py"), "--help"],
                        capture_output=True, text=True)
    check("--no-ocr is documented in --help", "--no-ocr" in rc.stdout, rc.stdout[:200])
    check("--ocr-backend is documented in --help", "--ocr-backend" in rc.stdout)


# --------------------------------------------------------------------------- #
# alignment + semantic extraction
# --------------------------------------------------------------------------- #

def _region(text: str, x: int, y: int, w: int = 120, h: int = 14, conf: float = 0.95) -> dict:
    return {"text": text, "confidence": conf, "bbox": [x, y, w, h], "read_method": "native"}


def _frame_rec(fid: str, t: float | None, regions: list[dict] | None = None,
               text: str | None = None, conf: float = 0.95,
               status: str = "ok", backend: str = "apple-vision") -> dict:
    """One `visual.json` frame entry.

    A read with regions derives its text from them, so the stored read and the
    regions cannot disagree — the same invariant the real producer holds, and one
    a fixture that hand-wrote both would quietly violate.
    """
    body = text if text is not None else "\n".join(r["text"] for r in (regions or []))
    return {"frame_id": fid, "t_start": t, "text": body or None,
            "result": {"status": status, "text": body or None,
                       "confidence": conf if body else None, "backend": backend,
                       "read_method": "native", "width": 1280, "height": 720,
                       "regions": regions or []}}


def _build(snippets: list[tuple[float, float, str]], frames: list[dict],
           duration: float = 60.0) -> "M.Evidence":
    """A complete `Evidence` from a transcript and a set of frame reads.

    Every test in this section starts here rather than from a stored artifact:
    the cases are about what the extractor *decides*, and a decision can only be
    checked against inputs small enough to read in full.
    """
    meta = {"id": "v", "title": "T", "channel": "C", "webpage_url": "https://youtu.be/v",
            "upload_date": "20260925", "duration": duration, "description": "", "tags": [],
            "chapters": [{"start_time": 0, "title": "All", "end_time": duration}],
            "chapters_available": True}
    raw = [{"start": s, "dur": d, "text": t} for s, d, t in snippets]
    segs = Y.segments_from_chapters(raw, meta["chapters"], duration)
    index = [{"frame_id": f["frame_id"], "file": f"frames/{f['frame_id']}.jpg",
              "t_start": f["t_start"]} for f in frames]
    return B.build_evidence(meta, raw, segs, frames_index=index, lang_used="en",
                            visual={"frames": frames})


def _aligned(ev: "M.Evidence", cfg: "A.AlignmentConfig" | None = None):
    """Align, publish the records onto the document, and return the metrics."""
    acfg = cfg or A.AlignmentConfig()
    recs, metrics = A.align_evidence(ev, acfg)
    ev.alignment = recs
    for seg_id, fids in A.segment_frame_ids(recs, ev.segments).items():
        seg = next(s for s in ev.segments if s.id == seg_id)
        seg.frame_ids = fids
    return recs, metrics, acfg


def _extracted(ev: "M.Evidence", recs, cfg: "SEM.SemanticConfig" | None = None) -> "SEM.SemanticResult":
    res = SEM.extract(ev, recs, None, cfg or SEM.SemanticConfig(), use_cache=False)
    ev.items, ev.conflicts, ev.observations = res.items, res.conflicts, res.observations
    return res


def _record(recs, snippet_id: str):
    return next(r for r in recs if r.snippet_id == snippet_id)


def _items_of(ev, type_: str, value: str | None = None):
    """Items of a type, excluding metadata.

    Metadata items are the platform's own statements and are never what these
    tests are about; leaving them in makes "no number was extracted" fail on the
    video's duration.
    """
    return [i for i in ev.items if i.type == type_ and (value is None or i.value == value)
            and any(p.source != "metadata" for p in i.provenance)]


def t_alignment() -> None:
    """Alignment: exact, nearby, absent, and OCR-assisted. Deterministic, offline."""
    # 1. exact temporal alignment — a frame inside the spoken span.
    ev = _build([(10.0, 2.0, "The table here shows the memory.")],
                [_frame_rec("s0001", 11.0, [_region("Model A 32 GB", 100, 100)])])
    recs, metrics, _ = _aligned(ev)
    first_recs = [x.to_dict() for x in recs]
    r = _record(recs, "s001")
    check("A1 a frame inside the spoken span is paired", r.frame_ids == ["s0001"], str(r.frame_ids))
    check("A1 the method is recorded as an overlap, not a guess", r.method == "overlap", r.method)
    check("A1 the record carries the span it was derived from",
          r.start_s == 10.0 and r.end_s == 12.0, f"{r.start_s}..{r.end_s}")
    check("A1 the pairing is bounded above the OCR band",
          r.confidence <= 0.95 and r.confidence > 0.5, str(r.confidence))

    # 2. nearby-frame alignment — just outside the span, inside the window, and
    # with nothing legible on it, so the score is purely temporal.
    ev = _build([(10.0, 2.0, "Something is on the screen now.")],
                [_frame_rec("s0001", 13.0, [], text="", status="no_text")])
    recs2, _, _ = _aligned(ev)
    r = _record(recs2, "s001")
    check("A2 a frame just past the span is still paired", r.frame_ids == ["s0001"], str(r))
    check("A2 its method is the nearest-in-time one, and says so",
          r.method == "time-nearest", r.method)
    check("A2 the score falls off with distance", 0.3 < r.confidence < 0.62, str(r.confidence))

    # 3. no relevant frame — the honest branch, recorded as a record.
    ev = _build([(10.0, 2.0, "Nothing is being shown here.")],
                [_frame_rec("s0001", 30.0, [_region("unrelated", 10, 10)])])
    recs, metrics, _ = _aligned(ev)
    r = _record(recs, "s001")
    check("A3 a frame outside the window is not paired", r.frame_ids == [], str(r.frame_ids))
    check("A3 the empty pairing names its method", r.method == "no-relevant-frame", r.method)
    check("A3 the empty pairing carries the sentinel, not a computed score",
          r.confidence == M.NO_FRAME_CONFIDENCE, str(r.confidence))
    check("A3 the count of unpaired snippets is reported", metrics["no_relevant_frame"] == 1,
          str(metrics["no_relevant_frame"]))
    check("A3 a frame no snippet can reach is counted, not hidden",
          metrics["distinct_frames_used"] == 0, str(metrics))

    # 4. OCR-assisted — a farther frame with a usable read beats a nearer blank one.
    ev = _build([(10.0, 2.0, "Now the benchmark.")],
                [_frame_rec("s0001", 10.5, [dict(_region("", 0, 0), text="")], text="", status="no_text"),
                 _frame_rec("s0002", 12.8, [_region("Model B 18 tok/s", 100, 100)])])
    ev.visual[0].ocr_text, ev.visual[0].ocr_status, ev.visual[0].ocr_confidence = None, "no_text", None
    recs, _, _ = _aligned(ev)
    r = _record(recs, "s001")
    check("A4 a usable read ranks above a nearer blank frame", r.frame_ids[0] == "s0002", str(r))
    check("A4 the blank frame is kept as a weaker candidate rather than dropped",
          "s0001" in r.frame_ids, str(r.frame_ids))
    check("A4 the method records that the read decided it", r.method == "ocr-assisted", r.method)
    check("A4 the reason is written down, not just scored", "read at" in (r.notes or ""), str(r.notes))

    # The window is a real bound, not a suggestion.
    ev = _build([(10.0, 2.0, "A.")], [_frame_rec("s0001", 13.9, [_region("x", 1, 1)])])
    recs, _, _ = _aligned(ev, A.AlignmentConfig(window_s=1.0))
    check("A4b a narrower window excludes what a wider one would pair",
          _record(recs, "s001").method == "no-relevant-frame", str(_record(recs, "s001")))

    # 5. an unmeasured frame is never a candidate — its time is unknown.
    ev = _build([(10.0, 2.0, "Look.")], [_frame_rec("s0001", None, [_region("x", 1, 1)])])
    recs, metrics, _ = _aligned(ev)
    check("A5 an unmeasured frame is excluded from alignment",
          _record(recs, "s001").frame_ids == [], str(_record(recs, "s001")))
    check("A5 and the exclusion is counted", metrics["frames_unmeasured_excluded"] == 1, str(metrics))

    # 6. several frames may support one snippet, and a segment collects them.
    ev = _build([(10.0, 2.0, "A."), (12.5, 2.0, "B.")],
                [_frame_rec("s0001", 10.4, [_region("row one", 10, 10)]),
                 _frame_rec("s0002", 11.0, [_region("row two", 10, 40)]),
                 _frame_rec("s0003", 14.0, [_region("row three", 10, 10)])])
    recs, _, _ = _aligned(ev)
    check("A6 a snippet may keep more than one frame", len(_record(recs, "s001").frame_ids) > 1,
          str(_record(recs, "s001")))
    check("A6 the segment's frame set is the union of its snippets'",
          set(ev.segments[0].frame_ids) == {f for r in recs for f in r.frame_ids},
          str(ev.segments[0].frame_ids))

    # 7. the inverse view is derived from the same records, never recomputed.
    inv = A.frames_to_snippets(recs, ev.snippets)
    check("A7 the reverse index agrees with the forward one",
          {f for r in recs for f in r.frame_ids} == set(inv), str(inv))

    # 8. determinism: the same input, the same records. Ids are positional, so a
    #    record that moves on a re-run would silently re-point every citation.
    ev2 = _build([(10.0, 2.0, "The table here shows the memory.")],
                 [_frame_rec("s0001", 11.0, [_region("Model A 32 GB", 100, 100)])])
    recs2, _, _ = _aligned(ev2)
    check("A8 alignment is deterministic",
          first_recs == [r.to_dict() for r in recs2])


def t_alignment_validation() -> None:
    """The validator must be able to fail on a corrupted alignment, not just pass one."""
    def aligned_doc():
        ev = _build([(10.0, 2.0, "Shown here.")],
                    [_frame_rec("s0001", 11.0, [_region("x", 1, 1)])])
        _aligned(ev)
        return ev

    r = aligned_doc().alignment[0].to_dict()

    def with_align(mutate):
        d = dict(r)
        mutate(d)
        return E.validate_evidence({"schema_version": SCHEMA_VERSION, "video": {"duration_s": 60},
                                    "transcript": {"snippet_count": 1},
                                    "alignment": [d]})

    check("A-V1 a dangling frame ref fails", "dangling-frame-ref" in with_align(
        lambda d: d.__setitem__("frame_ids", ["u9999"])).error_codes())
    check("A-V2 a dangling snippet ref fails", "dangling-transcript-ref" in E.validate_evidence(
        {"schema_version": SCHEMA_VERSION, "video": {"duration_s": 60},
         "transcript": {"snippet_count": 0}, "alignment": [{**r, "snippet_id": "s999"}]}).error_codes())
    check("A-V3 an unknown method fails", "unsupported-enum" in with_align(
        lambda d: d.__setitem__("method", "vibes")).error_codes())
    check("A-V4 an empty pairing claiming a method fails", "alignment-method-contradiction" in with_align(
        lambda d: (d.__setitem__("frame_ids", []), d.__setitem__("confidence", 0.4))).error_codes())
    check("A-V5 a pairing claiming no-relevant-frame but listing frames fails",
          "alignment-method-contradiction" in with_align(
              lambda d: d.__setitem__("method", "no-relevant-frame")).error_codes())
    check("A-V6 a computed score on an empty pairing fails", "alignment-empty-confidence" in with_align(
        lambda d: (d.__setitem__("frame_ids", []), d.__setitem__("method", "no-relevant-frame"),
                   d.__setitem__("confidence", 0.44))).error_codes())
    check("A-V7 an alignment timestamp that disagrees with the snippet fails",
          "alignment-timestamp-mismatch" in E.validate_evidence(
              {"schema_version": SCHEMA_VERSION, "video": {"duration_s": 60},
               "transcript": {"snippet_count": 1, "snippets": [
                   {"id": "s001", "start_s": 10.0, "end_s": 12.0, "text": "x"}]},
               "alignment": [{**r, "start_s": 25.0}]}).error_codes())
    check("A-V8 a confidence outside [0,1] fails", "invalid-confidence" in with_align(
        lambda d: d.__setitem__("confidence", 1.4)).error_codes())
    check("A-V9 the good alignment itself passes",
          E.validate_evidence({"schema_version": SCHEMA_VERSION, "video": {"duration_s": 60},
                               "transcript": {"snippet_count": 1, "snippets": [
                                   {"id": "s001", "start_s": 10.0, "end_s": 12.0, "text": "x"}]},
                               "segments": [{"id": "seg-01", "start_s": 0, "end_s": 60,
                                             "title": "A", "snippet_ids": ["s001"],
                                             "item_ids": [], "frame_ids": ["s0001"]}],
                               "visual": [{"frame_id": "s0001", "file": "f", "t_start": 11.0}],
                               "alignment": [r]}).valid,
          str(E.validate_evidence({"schema_version": SCHEMA_VERSION, "video": {"duration_s": 60},
                                   "transcript": {"snippet_count": 1, "snippets": [
                                       {"id": "s001", "start_s": 10.0, "end_s": 12.0, "text": "x"}]},
                                   "segments": [{"id": "seg-01", "start_s": 0, "end_s": 60,
                                                 "title": "A", "snippet_ids": ["s001"],
                                                 "item_ids": [], "frame_ids": ["s0001"]}],
                                   "visual": [{"frame_id": "s0001", "file": "f", "t_start": 11.0}],
                                   "alignment": [r]}).blocking_errors))


def t_semantic_numbers() -> None:
    """What a number means, which is the question the OCR pass could not answer."""
    # 5. transcript-only number: spoken, nothing on screen, still evidence.
    ev = _build([(10.0, 3.0, "You need at least 32GB of VRAM for this model.")],
                [_frame_rec("s0001", 11.0, [], text="", status="no_text")])
    recs, _, _ = _aligned(ev)
    _extracted(ev, recs)
    n = _items_of(ev, "number", "32")
    check("S5 a spoken figure becomes an item", len(n) == 1, str(_items_of(ev, "number")))
    check("S5 it cites the transcript only",
          [p.source for p in n[0].provenance] == ["transcript"], str(n[0].provenance))
    check("S5 the subject is recovered from the sentence", n[0].subject == "VRAM", str(n[0].subject))
    check("S5 the unit is carried, not guessed", n[0].unit == "GB", str(n[0].unit))
    check("S5 the sentence that carried it is attached", "VRAM" in (n[0].verbatim or ""))

    # 6. OCR-only number with no semantic context: seen, classified, not evidence.
    ev = _build([(10.0, 3.0, "Anyway, moving on to the next part.")],
                [_frame_rec("s0001", 11.0, [_region("0.0%", 10, 10), _region("0x", 40, 10)])])
    recs, _, _ = _aligned(ev)
    res = _extracted(ev, recs)
    check("S6 a bare number with no label and no speech is not an item",
          not _items_of(ev, "number"), str([i.id for i in ev.items if i.type == "number"]))
    classes = {o.get("classification") for o in res.observations}
    check("S6 the observation is still recorded", len(res.observations) >= 2, str(res.observations))
    check("S6 and it is classified, not dropped", "irrelevant" in classes or "ui-chrome" in classes,
          str(classes))
    check("S6 the raw read survives on the frame",
          "0.0%" in (ev.visual[0].ocr_text or ""), str(ev.visual[0].ocr_text))
    check("S6 the reason is recorded for the decision",
          all(o.get("reason") for o in res.observations if o.get("frame_id")),
          str([o.get("reason") for o in res.observations]))

    # 7. transcript + OCR: one item, two citations, from two independent reads.
    ev = _build([(10.0, 3.0, "This model wants 32GB of VRAM to run.")],
                [_frame_rec("s0001", 11.0, [_region("32 GB", 200, 100, conf=1.0)], conf=1.0)])
    recs, _, _ = _aligned(ev)
    res = _extracted(ev, recs)
    n = _items_of(ev, "number", "32")
    check("S7 a figure on both channels is ONE item, not two", len(n) == 1, str(len(n)))
    srcs = sorted(p.source for p in n[0].provenance)
    check("S7 it carries both provenances", srcs == ["ocr", "transcript"], str(srcs))
    check("S7 corroboration is counted as cross-modal", res.metrics["cross_modal_items"] >= 1,
          str(res.metrics["cross_modal_items"]))
    check("S7 the two readings are both recorded, not one overwritten",
          {p.source for p in n[0].provenance} == {"transcript", "ocr"}, str(n[0].provenance))
    check("S7 the item cites the literal text the frame showed",
          any("32 GB" == (p.ocr_text or "") for p in n[0].provenance if p.source == "ocr"),
          str([p.ocr_text for p in n[0].provenance]))
    check("S7 the document validates", E.validate_evidence(ev).valid,
          str(E.validate_evidence(ev).blocking_errors[:3]))

    # 18. OCR confidence 1.0, semantic confidence lower. Never the same number.
    read = ev.visual[0].ocr_confidence
    check("S18 the frame's read confidence is untouched at 1.0", read == 1.0, str(read))
    check("S18 the item's confidence is not the read's confidence",
          n[0].confidence != read, f"item={n[0].confidence} read={read}")
    check("S18 and it is lower than a perfect reading would imply",
          n[0].confidence < read, str(n[0].confidence))
    check("S18 the two numbers are within their own ranges",
          0.0 <= n[0].confidence <= 1.0 and 0.0 <= read <= 1.0)

    # 9. hedging survives. "could cost $500" is not "costs $500".
    ev = _build([(10.0, 3.0, "This could cost around $500 per month for a team.")],
                [_frame_rec("s0001", 11.0, [], text="", status="no_text")])
    recs, _, _ = _aligned(ev)
    _extracted(ev, recs)
    h = _items_of(ev, "number", "500")
    check("S9 a hedged figure is extracted", len(h) == 1, str(_items_of(ev, "number")))
    check("S9 its status stays hedged, not asserted", h[0].status == "hedged", h[0].status)
    check("S9 the hedge is named in the notes", "hedge" in (h[0].notes or ""), str(h[0].notes))
    check("S9 the value is not rounded into a certainty",
          h[0].display == "$500" and "around" not in h[0].display, str(h[0].display))
    check("S9 hedging costs confidence as well as changing the status",
          h[0].confidence < SEM.ConfidencePolicy().base_transcript, str(h[0].confidence))

    # 8. a spoken figure and a different displayed one: a conflict, left open.
    ev = _build([(10.0, 3.0, "The card ships with 32GB of VRAM as standard.")],
                [_frame_rec("s0001", 11.0, [_region("VRAM", 10, 10, w=80),
                                             _region("64 GB", 200, 11, w=70)])])
    recs, _, _ = _aligned(ev)
    res = _extracted(ev, recs)
    check("S8 a disagreement is detected", len(res.conflicts) == 1, str(res.conflicts))
    c = res.conflicts[0]
    check("S8 it is recorded as a value mismatch", c.kind == "value-mismatch", c.kind)
    check("S8 it is left unresolved — nothing auto-corrects a source", c.resolution == "unresolved",
          c.resolution)
    check("S8 both sides are named", len(c.item_ids) >= 2, str(c.item_ids))
    check("S8 and both items are marked as in conflict",
          all(i.status == "conflicting" for i in ev.items if i.id in c.item_ids),
          str([(i.id, i.status) for i in ev.items if i.id in c.item_ids]))
    check("S8 the two values are still separately readable",
          {"32", "64"} <= {i.value for i in ev.items}, str([i.value for i in ev.items]))
    check("S8 neither value was rewritten to match the other",
          not [i for i in ev.items if i.value == "64" and any(p.source == "transcript" for p in i.provenance)],
          str([(i.value, [p.source for p in i.provenance]) for i in ev.items]))
    check("S8 the document is still valid with an open conflict", E.validate_evidence(ev).valid,
          str(E.validate_evidence(ev).blocking_errors[:3]))


def t_semantic_screen() -> None:
    """Screen-only information: the case the pipeline exists to recover."""
    # 15. a screen-only benchmark table, read by layout rather than by pattern.
    ev = _build([(10.0, 4.0, "The benchmark is shown here on the right.")],
                [_frame_rec("s0001", 11.0, [
                    _region("Model A", 10, 100, w=90), _region("32 tok/s", 200, 100, w=80),
                    _region("Model B", 10, 130, w=90), _region("18 tok/s", 200, 130, w=80)])])
    recs, _, _ = _aligned(ev)
    res = _extracted(ev, recs)
    nums = {i.subject: i.display for i in _items_of(ev, "number")}
    check("S15 each row's figure gets the label beside it",
          nums == {"Model A": "32tok/s", "Model B": "18tok/s"}, str(nums))
    check("S15 a labelled screen value is high importance",
          all(i.importance == "high" for i in _items_of(ev, "number")),
          str([i.importance for i in _items_of(ev, "number")]))
    check("S15 it cites the frame, with the row text it came from",
          all(any(p.source == "ocr" for p in i.provenance) for i in _items_of(ev, "number")),
          str([i.provenance for i in _items_of(ev, "number")]))
    check("S15 the frame's own read is preserved for the record",
          "Model A" in ev.visual[0].ocr_text and "32 tok/s" in ev.visual[0].ocr_text,
          str(ev.visual[0].ocr_text))
    check("S15 the label keeps the frame's own casing",
          set(nums) == {"Model A", "Model B"}, str(set(nums)))
    check("S15 the document validates", E.validate_evidence(ev).valid,
          str(E.validate_evidence(ev).blocking_errors[:3]))

    # A column with no value stays empty. "No bar, no number", in data form.
    ev = _build([(10.0, 4.0, "The benchmark is shown here.")],
                [_frame_rec("s0001", 11.0, [
                    _region("Model A", 10, 100, w=90), _region("32 tok/s", 200, 100, w=80),
                    _region("Model B", 10, 130, w=90), _region("N/A", 200, 130, w=60)])])
    recs, _, _ = _aligned(ev)
    _extracted(ev, recs)
    check("S15b an absent value is not manufactured as a number",
          not [i for i in _items_of(ev, "number") if i.subject == "Model B"],
          str([(i.subject, i.value) for i in _items_of(ev, "number")]))

    # 14. a screen-only URL: self-identifying, so it needs no label.
    ev = _build([(10.0, 4.0, "Links are in the description if you want them.")],
                [_frame_rec("s0001", 11.0, [_region("https://example.com/bench", 10, 10, w=300)])])
    recs, _, _ = _aligned(ev)
    _extracted(ev, recs)
    links = _items_of(ev, "link")
    check("S14 a URL on screen becomes a link item",
          [i.value for i in links] == ["https://example.com/bench"], str([i.value for i in links]))
    check("S14 it cites the frame it was read from",
          links and links[0].provenance[0].source == "ocr", str(links[0].provenance if links else None))

    # 19. several frames showing one value: one item, many citations.
    ev = _build([(10.0, 4.0, "The price is $0.09 per message.")],
                [_frame_rec("s0001", 10.5, [_region("price per message", 10, 100, w=180),
                                             _region("$0.09", 210, 100, w=70)]),
                 _frame_rec("s0002", 11.0, [_region("price per message", 10, 100, w=180),
                                             _region("$0.09", 210, 100, w=70)])])
    recs, _, _ = _aligned(ev)
    _extracted(ev, recs)
    prices = [i for i in ev.items if i.value == "0.09"]
    check("S19 the same value on two frames is one item", len(prices) == 1, str(len(prices)))
    check("S19 with both frames cited",
          {p.ref for p in prices[0].provenance if p.source == "ocr"} == {"s0001", "s0002"},
          str([p.ref for p in prices[0].provenance]))
    check("S19 the item's confidence is the weaker of the two reads, not the stronger",
          prices[0].confidence <= min(p.confidence for p in prices[0].provenance if p.source == "ocr"),
          str(prices[0].confidence))

    # 20. one frame, several unrelated figures: each keeps its own label.
    ev = _build([(10.0, 4.0, "Look at the numbers on the screen right now.")],
                [_frame_rec("s0001", 11.0, [
                    _region("latency", 10, 100, w=80), _region("120 ms", 200, 100, w=70),
                    _region("memory", 10, 130, w=80), _region("32 GB", 200, 130, w=70),
                    _region("throughput", 10, 160, w=90), _region("18 tok/s", 210, 160, w=80)])])
    recs, _, _ = _aligned(ev)
    _extracted(ev, recs)
    by_subject = {i.subject: i.display for i in _items_of(ev, "number")}
    check("S20 three unrelated figures on one frame stay three items",
          len(by_subject) == 3, str(by_subject))
    check("S20 each keeps its own label, not the frame's first one",
          by_subject == {"latency": "120ms", "memory": "32GB", "throughput": "18tok/s"},
          str(by_subject))

    # 13. a UI percentage is chrome, not a benchmark result.
    ev = _build([(10.0, 4.0, "Let us look at the actual results now.")],
                [_frame_rec("s0001", 11.0, [
                    _region("Subscribers", 10, 10, w=100), _region("- 0.0%", 200, 10, w=70),
                    _region("Settings", 10, 40, w=100), _region("Show more", 10, 70, w=100)])])
    recs, _, _ = _aligned(ev)
    res = _extracted(ev, recs)
    check("S13 a zero delta beside a counter is chrome, not a result",
          not [i for i in ev.items if i.type == "number" and i.display == "0.0%"],
          str([(i.type, i.display) for i in ev.items]))
    check("S13 it is classified as chrome with a stated reason",
          any(o["classification"] == "ui-chrome" and o.get("reason") for o in res.observations),
          str([o for o in res.observations if o.get("frame_id")]))
    check("S13 the underlying read is still on the frame",
          "0.0%" in (ev.visual[0].ocr_text or ""))

    # A burned-in caption is not a second fact.
    ev = _build([(10.0, 4.0, "The throughput numbers are on the slide.")],
                [_frame_rec("s0001", 11.0, [_region("The throughput numbers are on the slide.", 10, 200, w=600)])])
    recs, _, _ = _aligned(ev)
    res = _extracted(ev, recs)
    check("S13b a caption duplicating the speech is classified as such",
          any(o["classification"] == "caption" for o in res.observations),
          str([o["classification"] for o in res.observations]))


def t_semantic_structure() -> None:
    """Comparisons, procedures, entities — the structured item types."""
    # 10. a comparison carries its subjects and only the dimensions it can name.
    ev = _build([(10.0, 4.0, "Opus has 120 ms latency versus Sonnet at 180 ms.")],
                [_frame_rec("s0001", 11.0, [], text="", status="no_text")])
    recs, _, _ = _aligned(ev)
    _extracted(ev, recs)
    comps = _items_of(ev, "comparison")
    check("S10 a stated comparison becomes a comparison item", len(comps) == 1,
          str([i.type for i in ev.items]))
    if comps:
        d = comps[0].detail or {}
        check("S10 it names both subjects", bool(d.get("subject_a")) and bool(d.get("subject_b")), str(d))
        dims = {r["dimension"]: (r.get("a"), r.get("b")) for r in d.get("dimensions", [])}
        check("S10 the dimension it named carries both values",
              dims.get("latency") == ("120ms", "180ms"), str(dims))
        check("S10 it is not a generic claim string", comps[0].value != "Opus versus Sonnet", comps[0].value)
        check("S10 the sentence that stated it is attached",
              "Opus" in (comps[0].verbatim or ""), str(comps[0].verbatim))
        check("S10 the document validates with a structured comparison",
              E.validate_evidence(ev).valid, str(E.validate_evidence(ev).blocking_errors[:3]))

    # A contrast with no dimension named carries no values, rather than guessing.
    ev = _build([(10.0, 4.0, "We picked Postgres versus MySQL for the job.")],
                [_frame_rec("s0001", 11.0, [], text="", status="no_text")])
    recs, _, _ = _aligned(ev)
    _extracted(ev, recs)
    comps = _items_of(ev, "comparison")
    check("S10b a comparison with no named dimension invents no value",
          comps and comps[0].detail.get("dimensions") == [], str(comps[0].detail if comps else None))
    check("S10b and says so in its notes", comps and "no dimension named" in (comps[0].notes or ""),
          str(comps[0].notes if comps else None))

    # 11. a procedure, only when there is a run of instructions.
    ev = _build([(10.0, 2.0, "Here is how the setup works."),
                 (11.0, 2.0, "First install the package."), (13.0, 2.0, "Run the command."),
                 (15.0, 2.0, "Configure the endpoint."), (17.0, 2.0, "That is the whole setup.")],
                [_frame_rec("s0001", 11.5, [], text="", status="no_text")])
    recs, _, _ = _aligned(ev)
    _extracted(ev, recs)
    steps = _items_of(ev, "step")
    check("S11 a run of instructions becomes steps", len(steps) == 3, str(len(steps)))
    check("S11 they are ordered", [s.detail["index"] for s in steps] == [1, 2, 3],
          str([s.detail.get("index") for s in steps]))
    check("S11 they name the group they belong to",
          all(s.detail.get("group") == ev.segments[0].id for s in steps),
          str([s.detail.get("group") for s in steps]))
    check("S11 a trailing remark is not promoted to a step",
          not [s for s in steps if "whole setup" in s.value], str([s.value for s in steps]))
    check("S11 the announcement itself is not a step",
          not [s for s in steps if "here is how" in s.value.lower()], str([s.value for s in steps]))

    # The same run of imperatives in a talk that never announced a procedure is
    # not one. Without this gate a passing mention of "run the command" became a
    # numbered step, and the count of steps stopped tracking the material.
    ev_ann = _build([(10.0, 2.0, "So, let's get started."), (12.0, 2.0, "Now this is all in the browser."),
                     (14.0, 2.0, "And today I am going to show you how it works."),
                     (16.0, 2.0, "First the visual layer, then the memory.")],
                    [_frame_rec("s0001", 11.0, [], text="", status="no_text")])
    recs_a, _, _ = _aligned(ev_ann)
    _extracted(ev_ann, recs_a)
    check("S11c imperatives in a talk that announced no procedure are not steps",
          not _items_of(ev_ann, "step"), str([s.value for s in _items_of(ev_ann, "step")]))

    # One imperative sentence is a sentence, not a procedure.
    ev = _build([(10.0, 2.0, "Here is how the setup works."), (11.0, 2.0, "Run the command twice."),
                 (13.0, 2.0, "That is all it takes.")],
                [_frame_rec("s0001", 11.0, [], text="", status="no_text")])
    recs, _, _ = _aligned(ev)
    _extracted(ev, recs)
    check("S11b a single instruction is not a procedure", not _items_of(ev, "step"),
          str([s.value for s in _items_of(ev, "step")]))

    # 12. entities, only once corroborated.
    ev = _build([(10.0, 3.0, "The Opus model is the one I use. Opus handles long context well."),
                 (14.0, 3.0, "However the docs were thin. Anyway it worked.")],
                [_frame_rec("s0001", 11.0, [], text="", status="no_text")])
    recs, _, _ = _aligned(ev)
    _extracted(ev, recs)
    ents = [i for i in ev.items if i.type in ("model", "product", "tool", "organization")
            and any(p.source != "metadata" for p in i.provenance)]
    check("S12 a name used twice becomes an entity", [i.value for i in ents] == ["Opus"],
          str([(i.type, i.value) for i in ents]))
    check("S12 it is typed and says the typing was inferred",
          ents and "not disambiguated" in (ents[0].notes or ""), str(ents[0].notes if ents else None))
    check("S12 a sentence-initial capitalised word is not an entity",
          not [i for i in ents if i.value in ("However", "Anyway", "The")],
          str([i.value for i in ents]))

    # 11c. a claim and its figure are separate, linked items.
    ev = _build([(10.0, 4.0, "The approach reduces inference cost by 3x per request.")],
                [_frame_rec("s0001", 11.0, [], text="", status="no_text")])
    recs, _, _ = _aligned(ev)
    _extracted(ev, recs)
    claims = _items_of(ev, "claim")
    metrics = _items_of(ev, "claim_metric")
    check("S11c the claim and its figure are two items", len(claims) == 1 and len(metrics) == 1,
          str([i.type for i in ev.items]))
    check("S11c the metric links back to the claim",
          metrics and metrics[0].detail.get("claim_id") == claims[0].id,
          str(metrics[0].detail if metrics else None))
    check("S11c and the link resolves", E.validate_evidence(ev).valid,
          str(E.validate_evidence(ev).blocking_errors[:3]))

    # Deduplication is on semantic identity, and re-running does not duplicate.
    ev = _build([(10.0, 4.0, "It costs 5GB of disk space for the index.")],
                [_frame_rec("s0001", 10.5, [_region("5 GB", 200, 100)]),
                 _frame_rec("s0002", 11.0, [_region("5 GB", 200, 100)])])
    recs, _, _ = _aligned(ev)
    first = _extracted(ev, recs)
    again = _extracted(ev, recs)
    check("S-D1 re-extracting the same evidence produces the same items",
          [i.id for i in first.items] == [i.id for i in again.items], "ids changed on a re-run")
    check("S-D2 and the same count", len(first.items) == len(again.items),
          f"{len(first.items)} vs {len(again.items)}")
    # A hedged and an asserted form of one figure are different claims.
    check("S-D3 a hedge is part of identity, so the two forms never merge",
          SEM.item_identity(SEM.Item(id="", type="number", value="500", unit="$",
                                     provenance=[], status="hedged"))
          != SEM.item_identity(SEM.Item(id="", type="number", value="500", unit="$",
                                        provenance=[], status="asserted")))
    check("S-D4 distinct models with similar names are not merged",
          not SEM.subjects_compatible("Model A", "Model B"))
    check("S-D5 a label and the phrase naming it are compatible",
          SEM.subjects_compatible("waitlist", "waitlist signups"))


def t_vlm_and_provenance() -> None:
    """The VLM abstraction, frame selection, and the checks that keep it honest."""
    frames = [M.VisualFrame(frame_id=f"f{i}", file=f"frames/f{i}.jpg", t_start=float(i * 10),
                            ocr_text="x" if i % 2 else None, ocr_confidence=0.9 if i % 2 else None,
                            ocr_status="ok" if i % 2 else "no_text",
                            ocr_regions=[{"text": f"r{j}", "confidence": 0.9, "bbox": [j, j, 5, 5]}
                                         for j in range(40 if i == 3 else 2)])
            for i in range(1, 6)]
    align_records = [M.Alignment(snippet_id="s001", frame_ids=["f1"], confidence=0.6,
                                 method="overlap")]
    chosen = V.select_frames(frames, align_records, max_frames=3)
    check("V1 a dense-text frame is selected for interpretation", "f3" in chosen, str(chosen))
    check("V2 a blank frame is never sent to a model", "f2" not in chosen and "f4" not in chosen,
          str(chosen))
    check("V3 the selection is bounded", len(chosen) <= 3, str(chosen))
    check("V4 the selection is deterministic",
          chosen == V.select_frames(frames, align_records, max_frames=3))

    unavail = V.get_provider("none")
    obs = unavail.analyze("/nonexistent.jpg", "f1", {})
    check("V5 with no backend the pass says so rather than staying silent",
          obs.status == "unavailable" and obs.caption is None, str(obs.to_dict()))
    check("V6 a model observation cannot exceed the model band",
          V.VisionObservation.from_dict({"frame_id": "f1", "confidence": 0.99}).confidence
          <= V.MAX_MODEL_CONFIDENCE)

    # 17. VLM provenance on a frame no model looked at must FAIL.
    ev = _build([(10.0, 3.0, "The chart here shows the results.")],
                [_frame_rec("s0001", 11.0, [_region("a bar", 10, 10)])])
    recs, _, _ = _aligned(ev)
    _extracted(ev, recs)
    check("S17 the clean document has no vlm provenance",
          not [p for i in ev.items for p in i.provenance if p.source == "vlm"],
          str([p.source for i in ev.items for p in i.provenance]))

    forged = M.Item(id="x1", type="quote", value="a bar chart",
                    provenance=[P.vlm_provenance(ev.visual[0], "a bar chart")])
    ev.items.append(forged)
    codes = E.validate_evidence(ev).error_codes()
    check("S17 vlm provenance with no vlm result FAILS", "vlm-without-result" in codes, str(codes))
    check("S17 and the document is not synthesizable", not E.validate_evidence(ev).valid)

    # With a real result recorded, the same citation is fine.
    ev2 = _build([(10.0, 3.0, "The chart here shows the results.")],
                 [_frame_rec("s0001", 11.0, [_region("a bar", 10, 10)])])
    ev2.visual[0].vlm_status = "ok"
    ev2.visual[0].vlm_caption = "A bar chart comparing two models."
    ev2.visual[0].vlm_backend = "fake-vlm"
    ev2.visual[0].vlm_confidence = 0.7
    ev2.items.append(M.Item(id="x1", type="quote", value="a bar chart",
                            provenance=[P.vlm_provenance(ev2.visual[0], "a bar chart")]))
    check("S17b the same citation passes once a result exists",
          E.validate_evidence(ev2).valid, str(E.validate_evidence(ev2).blocking_errors[:3]))
    check("S17c a model observation is never marked verified_verbatim",
          "model-verbatim" in E.validate_evidence(
              {"schema_version": SCHEMA_VERSION, "video": {"duration_s": 60},
               "transcript": {"snippet_count": 0}, "visual": [
                   {"frame_id": "s0001", "file": "f", "t_start": 11.0, "vlm_status": "ok",
                    "vlm_caption": "A bar chart", "vlm_backend": "fake"}],
               "items": [{"id": "i", "type": "quote", "value": "v",
                          "provenance": [{"source": "vlm", "ref": "s0001",
                                          "ocr_text": "A bar chart", "verified_verbatim": True}]}]
               }).error_codes())

    # 16. a semantic item with no provenance at all must FAIL.
    ev3 = _build([(10.0, 3.0, "It costs 5GB of disk space.")],
                 [_frame_rec("s0001", 11.0, [], text="", status="no_text")])
    recs3, _, _ = _aligned(ev3)
    _extracted(ev3, recs3)
    ev3.items.append(M.Item(id="orphan", type="number", value="999", unit="GB",
                            provenance=[]))
    check("S16 a semantic item with missing provenance FAILS",
          "missing-provenance" in E.validate_evidence(ev3).error_codes(),
          str(E.validate_evidence(ev3).error_codes()))

    # The separation the whole layer rests on, asserted directly.
    ev4 = _build([(10.0, 3.0, "It costs 5GB of disk space for the index.")],
                 [_frame_rec("s0001", 10.5, [_region("5 GB", 200, 100)], conf=1.0)])
    recs4, _, _ = _aligned(ev4)
    _extracted(ev4, recs4)
    item = next(i for i in ev4.items if i.value == "5")
    check("S18b a perfect read does not produce a certain item",
          item.confidence < 1.0, str(item.confidence))
    check("S18c the read's own confidence is still 1.0 on the frame",
          ev4.visual[0].ocr_confidence == 1.0, str(ev4.visual[0].ocr_confidence))
    check("S18d the two fields are separate in the record",
          "ocr_confidence" not in item.to_dict() and "confidence" in item.to_dict(),
          str(sorted(item.to_dict())))

    # Metrics: the run must be able to account for its own work.
    res = _extracted(ev4, recs4)
    for key in ("segments_processed", "observations_classified", "items_total",
                "items_by_type", "cross_modal_items", "conflicts_detected",
                "extraction_calls", "cache_hit_rate", "wall_s", "config"):
        check(f"S-M the run reports {key}", key in res.metrics, str(sorted(res.metrics)))
    check("S-M every observation records a reason",
          all(o.get("reason") for o in ev4.observations if o.get("frame_id")),
          str([o for o in ev4.observations if not o.get("reason")]))

    # The cache key must move whenever the extraction contract does. It did not,
    # once, and the symptom was a document full of duplicate-id failures with no
    # mention of a cache anywhere in the output.
    ctx = SEM.build_contexts(ev4, recs4, {f.frame_id: f for f in ev4.visual},
                             {s.id: s for s in ev4.snippets})[0]
    key_now = SEM._segment_cache_key(ctx, SEM.SemanticConfig(), {})
    original = SEM.EXTRACTION_VERSION
    try:
        SEM.EXTRACTION_VERSION = original + 1
        check("S-C1 bumping the extraction version invalidates every cached segment",
              SEM._segment_cache_key(ctx, SEM.SemanticConfig(), {}) != key_now)
    finally:
        SEM.EXTRACTION_VERSION = original
    check("S-C2 the key is stable for the same inputs",
          SEM._segment_cache_key(ctx, SEM.SemanticConfig(), {}) == key_now)
    changed = SEM.build_contexts(_build([(10.0, 4.0, "It costs 5GB of disk space for the index.")],
                                        [_frame_rec("s0001", 10.5, [_region("5 GB", 200, 100)])]),
                                 recs4, {f.frame_id: f for f in ev4.visual},
                                 {s.id: s for s in ev4.snippets})[0]
    check("S-C3 different content gives a different key",
          SEM._segment_cache_key(changed, SEM.SemanticConfig(), {}) != key_now)


# --------------------------------------------------------------------------- #
# vision-language pass: offline, deterministic, no network
# --------------------------------------------------------------------------- #

def _png(width: int = 64, height: int = 48, seed: int = 0) -> bytes:
    """A real, minimal PNG with deterministic content.

    The frames have to exist on disk and have to hash differently from one
    another — the cache key is over the file's bytes, so two "fixtures" that were
    the same bytes would be the same question, and a cache-hit test over them
    would prove nothing. Drawing them here rather than checking in binaries keeps
    every byte of the suite reviewable as code. The pixels are a gradient: the
    fake provider answers from a fixture map and never looks at them, which is
    exactly what makes it a test double rather than a claim about vision.
    """
    raw = b"".join(
        b"\x00" + bytes(((x * 7 + y * 11 + seed * 31) % 256) for x in range(width))
        for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


#: One entry per visual kind the phase brief names, written in the shape the
#: schema requires — which is what makes these a test of the contract rather
#: than of the fake. Each carries the transcript it is read against, so the
#: grounding rules have something real to check against.
VLM_FIXTURES: dict[str, dict] = {}


VLM_FIXTURES["chart"] = {
    "content_type": "chart", "confidence": 0.66,
    "summary": "A bar chart of tokens per second for two models; the vertical axis is cut off.",
    "visual_elements": [
        {"kind": "title", "label": "Inference Speed"},
        {"kind": "axis_label", "label": "tokens per second"},
        {"kind": "series", "label": "Qwen"},
        {"kind": "series", "label": "Llama"},
    ],
    # No `numbers`: the axis is illegible, so there is nothing to read a value
    # from. An ordering is what survives, and it is a real observation.
    "numbers": [],
    "relationships": [{"from": "Qwen", "to": "Llama", "kind": "ordering", "certain": True}],
    "uncertainties": [{"text": "the y-axis tick labels are cropped out of frame",
                       "reason": "the chart is cut off at the bottom"}],
}


VLM_FIXTURES["table"] = {
    "content_type": "table", "confidence": 0.7,
    "summary": "A three-column benchmark table comparing two models on latency and throughput.",
    "visual_elements": [{"kind": "column_header", "label": "Latency"},
                        {"kind": "column_header", "label": "Throughput"}],
    "text_relationships": [{"text": "Model", "role": "row_label"}],
    # The whole point: OCR returns four numbers with nothing attaching them to
    # anything. Row and column travel with each value or the reading is lost.
    "numbers": [
        {"value": "100", "unit": "ms", "row": "Qwen", "column": "Latency", "legible": True},
        {"value": "32", "unit": "tok/s", "row": "Qwen", "column": "Throughput", "legible": True},
        {"value": "180", "unit": "ms", "row": "Llama", "column": "Latency", "legible": True},
        {"value": "18", "unit": "tok/s", "row": "Llama", "column": "Throughput", "legible": True},
    ],
    "relationships": [], "uncertainties": [],
}


VLM_FIXTURES["diagram"] = {
    "content_type": "diagram", "confidence": 0.62,
    "summary": "A retrieval pipeline drawn as four stacked boxes joined by downward arrows.",
    "visual_elements": [{"kind": "block", "label": "User"}, {"kind": "block", "label": "API"},
                        {"kind": "block", "label": "Retriever"}, {"kind": "block", "label": "LLM"}],
    "numbers": [],
    # OCR reads four words and no arrows. These edges are the whole reading.
    "relationships": [
        {"from": "User", "to": "API", "kind": "arrow", "certain": True},
        {"from": "API", "to": "Retriever", "kind": "arrow", "certain": True},
        {"from": "Retriever", "to": "LLM", "kind": "arrow", "certain": True},
    ],
    "uncertainties": [],
}


VLM_FIXTURES["terminal"] = {
    "content_type": "terminal", "confidence": 0.64,
    "summary": "A terminal running the test suite; three tests failed.",
    "visual_elements": [{"kind": "command", "label": "uv run pytest"},
                        {"kind": "output", "label": "3 failed"}],
    "text_relationships": [{"text": "3 failed", "role": "value"}],
    "numbers": [{"value": "3", "unit": None, "label": "failed", "context": "3 failed", "legible": True}],
    "actions": [{"description": "running the test suite", "target": "terminal"}],
    "uncertainties": [],
}


VLM_FIXTURES["code"] = {
    "content_type": "code", "confidence": 0.6,
    "summary": "A Python editor showing the top of a function in load.py.",
    "visual_elements": [{"kind": "language", "label": "Python"},
                        {"kind": "filename", "label": "load.py"}],
    "numbers": [], "relationships": [],
    # The visible line is truncated on screen. Saying so is the correct answer;
    # completing it would be the fabrication this whole pass is built to avoid.
    "uncertainties": [{"text": "the function body is cut off below the visible line",
                       "reason": "only the first two lines are on screen"}],
}


VLM_FIXTURES["browser"] = {
    "content_type": "browser", "confidence": 0.68,
    "summary": "A local settings page with the model selector set to Qwen3-8B.",
    "visual_elements": [{"kind": "address_bar", "label": "localhost:3000/settings"},
                        {"kind": "control", "label": "Model"}],
    "text_relationships": [{"text": "Qwen3-8B", "role": "value", "target": "Model"}],
    # Named because the frame's own read contains the string. A name the read
    # does not carry would be refused further down.
    "entities": [{"text": "Qwen3-8B", "kind": "model", "confidence": 0.7}],
    "actions": [{"description": "changing the selected model", "target": "Model"}],
    "uncertainties": [],
}


VLM_FIXTURES["product"] = {
    "content_type": "product_screen", "confidence": 0.65,
    "summary": "A dashboard headed Acme with a reading of 42 active sessions.",
    "visual_elements": [{"kind": "branding", "label": "Acme"}],
    "text_relationships": [{"text": "Acme", "role": "title"}],
    # The first is corroborated by the frame's read; the second is the model
    # recognising a logo from appearance. The second must not become an item.
    "entities": [{"text": "Acme", "kind": "product", "confidence": 0.8},
                 {"text": "Nebula Cloud", "kind": "product", "confidence": 0.55}],
    "numbers": [{"value": "42", "unit": None, "label": "active sessions", "legible": True}],
    "uncertainties": [],
}


VLM_FIXTURES["mixed"] = {
    "content_type": "mixed", "confidence": 0.6,
    "summary": "A slide pairing a throughput figure with a small request-flow diagram.",
    "visual_elements": [{"kind": "title", "label": "Throughput"}],
    "numbers": [{"value": "32", "unit": "tok/s", "row": "Qwen", "column": "Throughput", "legible": True}],
    "relationships": [{"from": "Client", "to": "Queue", "kind": "arrow", "certain": True}],
    "uncertainties": [],
}


VLM_FIXTURES["unreadable"] = {
    "content_type": "chart", "confidence": 0.3,
    "summary": "A dense chart whose labels and axes cannot be read at this resolution.",
    "visual_elements": [], "numbers": [], "relationships": [],
    "uncertainties": [{"text": "the axis labels and the legend are not legible",
                       "reason": "the frame is too small to read the glyphs"}],
}


VLM_FIXTURES["ambiguous"] = {
    "content_type": "diagram", "confidence": 0.4,
    "summary": "Two boxes joined by a line whose direction cannot be determined.",
    "visual_elements": [{"kind": "block", "label": "Ingest"}, {"kind": "block", "label": "Store"}],
    "numbers": [],
    # Recorded, with the direction explicitly not claimed.
    "relationships": [{"from": "Ingest", "to": "Store", "kind": "arrow", "certain": False}],
    "uncertainties": [{"text": "the connector has no visible arrowhead"}],
}


#: What OCR reads on each fixture frame. The screen text is the ground the
#: grounding rules check against, so it is written out rather than implied: a
#: fixture whose OCR text was a placeholder would pass every refusal rule
#: vacuously and prove nothing.
VLM_SCREEN_TEXT = {
    "chart": "Inference Speed\ntokens per second\nQwen\nLlama",
    "table": "Model  Latency  Throughput\nQwen  100 ms  32 tok/s\nLlama  180 ms  18 tok/s",
    "diagram": "User\nAPI\nRetriever\nLLM",
    "terminal": "$ uv run pytest\ncollected 42 items\n3 failed",
    "code": "def load(path):\n    return json.load(open(path",
    "browser": "localhost:3000/settings\nModel\nQwen3-8B\nSave",
    "product": "Acme\nOverview\n42 active sessions",
    "mixed": "Throughput\nQwen 32 tok/s\nClient -> Queue",
    "unreadable": "l l l l l l\nIIII IIII",
    "ambiguous": "Ingest\nStore",
}

#: What the speaker says while each fixture is on screen. A segment needs speech
#: for a context window to exist, and the conflict and corroboration cases need
#: something to agree or disagree with.
VLM_SPEECH = {
    "chart": "Here is how the two models compare on raw speed.",
    "table": "So Qwen is the faster one here, and it also does more tokens per second.",
    "diagram": "The request goes from the user through the API into the retriever and then the model.",
    "terminal": "Let me run the tests and see what fails.",
    "code": "This is the loader that reads the file.",
    "browser": "I am switching the model over to Qwen3-8B right now.",
    "product": "That is the Acme dashboard, with 42 active sessions.",
    "mixed": "Throughput is 32 tokens per second, and the client queues into it.",
    "unreadable": "The numbers on this one are quite small.",
    "ambiguous": "These two are connected, I think.",
}

#: The expected properties, per fixture. Not a count — a count is what the
#: extractor produces and cannot be the thing that proves it worked. These are
#: the properties a reader of the report would rely on.
VLM_EXPECTED = {
    "chart": {"no_numbers": True, "relationship": ("Qwen", "Llama"), "uncertainty": True},
    "table": {"numbers": {("Qwen", "Latency"): "100", ("Qwen", "Throughput"): "32",
                          ("Llama", "Latency"): "180", ("Llama", "Throughput"): "18"}},
    "diagram": {"relationships": [("User", "API"), ("API", "Retriever"), ("Retriever", "LLM")]},
    "terminal": {"action": True},
    "code": {"uncertainty": True, "no_numbers": True},
    "browser": {"entity": "Qwen3-8B", "action": True},
    "product": {"entity": "Acme", "refused_entity": "Nebula Cloud"},
    "mixed": {"number": "32", "relationship": ("Client", "Queue")},
    "unreadable": {"no_numbers": True, "uncertainty": True},
    "ambiguous": {"relationship": ("Ingest", "Store"), "uncertain_relationship": True},
}


def _vlm_frames(d: Path) -> list[M.VisualFrame]:
    """The ten fixture frames, with the OCR read each one's rules check against."""
    out = []
    for i, kind in enumerate(VLM_FIXTURES):
        text = VLM_SCREEN_TEXT[kind]
        out.append(M.VisualFrame(
            frame_id=kind, file=f"frames/{kind}.png", t_start=10.0 + i,
            ocr_text=text, ocr_confidence=0.95, ocr_status="ok",
            ocr_backend="fixture",
            ocr_regions=[{"text": ln, "confidence": 0.95, "bbox": [10, 10 + 20 * j, 300, 14]}
                         for j, ln in enumerate(text.splitlines())]))
    return out


def _vlm_provider() -> V.FakeProvider:
    return V.FakeProvider(VLM_FIXTURES, model="fixture-v1")


def t_vlm_observation_schema() -> None:
    """The schema is a boundary. Every field of it is checked here."""
    ok = V.VisionObservation.from_dict({
        "frame_id": "f1", "summary": "a table", "content_type": "table", "confidence": 0.7,
        "numbers": [{"value": "32", "row": "Qwen", "column": "Throughput"}],
        "relationships": [{"from": "a", "to": "b"}],
    })
    check("W1 a well-formed response is accepted",
          ok.status == "ok" and ok.content_type == "table" and len(ok.numbers) == 1,
          ok.to_dict())
    check("W2 a relationship round-trips under names that are not keywords",
          {"from": "a", "to": "b"} in ok.to_dict()["relationships"], str(ok.to_dict()))

    for bad, label in (
        ({"frame_id": "f1", "summary": "x", "content_type": "hologram", "confidence": 0.5},
         "W3 an invented content type"),
        ({"frame_id": "f1", "summary": "x", "content_type": "chart", "confidence": 0.5,
          "numbers": [{"row": "no value here"}]}, "W4 a number with no value"),
        ({"frame_id": "f1", "summary": "x", "content_type": "chart", "confidence": 0.5,
          "relationships": [{"from": "a"}]}, "W5 a relationship with one end"),
        ({"frame_id": "f1", "summary": "x", "content_type": "chart", "confidence": 0.5,
          "claims_supported": [{"claim": "c", "support": "probably"}]}, "W6 a support verdict that is not one of the three"),
    ):
        got = V.VisionObservation.from_dict(bad)
        check(f"{label} is refused rather than absorbed",
              got.status == "malformed" and bool(got.notes), f"{got.status} {got.notes!r}")

    for raw, label in ((2.5, "above the cap"), (-1, "negative"), (float("nan"), "not a number"),
                       ("high", "a string")):
        got = V.VisionObservation.from_dict(
            {"frame_id": "f1", "summary": "x", "content_type": "chart", "confidence": raw})
        check(f"W7 a confidence {label} never lands in the document",
              got.confidence is None or 0.0 <= got.confidence <= V.MAX_MODEL_CONFIDENCE,
              str(got.confidence))

    notdict = V.VisionObservation.from_dict("I am a markdown table, not JSON", "f1")
    check("W8 a response that is not an object is recorded, not raised",
          notdict.status == "malformed" and notdict.frame_id == "f1", notdict.to_dict())

    partial = V.VisionObservation.from_dict(
        {"frame_id": "f1", "summary": "x", "content_type": "chart", "confidence": 0.5,
         "numbers": [{"value": "5", "unit": "%"}, "bare string", 42]})
    check("W9 a partly valid response keeps what parsed and says what did not",
          partial.status == "malformed" and len(partial.numbers) == 2 and partial.notes,
          f"{partial.status} {len(partial.numbers)} {partial.notes!r}")


def _vlm_run_one(d: Path, kind: str, provider=None, cache=None) -> tuple[M.Evidence, dict]:
    """One fixture, through the whole path, from frames on disk to a document.

    Each stage is the production one: the real selector, the real context
    builder, the real runner, the real evidence builders and the real validator.
    Only the model is a double. A suite that tested the extraction logic against
    hand-assembled payloads would pass while the pipeline stayed broken, which is
    the failure this whole section exists to catch.
    """
    ev = _build([(10.0, 3.0, VLM_SPEECH[kind])], [_frame_rec(kind, 11.0, [
        _region(ln, 10, 10 + 20 * j) for j, ln in enumerate(VLM_SCREEN_TEXT[kind].splitlines())])])
    # `_build` names every frame `.jpg`; these fixtures are drawn as PNGs, and a
    # fixture that lied about its own file would exercise the wrong media-type
    # path and hide a real mismatch between what is written and what is sent.
    ev.visual[0].file = f"frames/{kind}.png"
    (d / ev.visual[0].file).parent.mkdir(parents=True, exist_ok=True)
    (d / ev.visual[0].file).write_bytes(_png(seed=list(VLM_FIXTURES).index(kind)))
    recs, _, _ = _aligned(ev)
    ctxs = {f.frame_id: V.build_context(ev, f, recs) for f in ev.visual}
    payload = V.run_vlm(d, ev.visual, recs, provider or _vlm_provider(),
                        max_frames=8, context_by_frame=ctxs, cache=cache)
    V.apply_observations(ev.visual, payload)
    res = SEM.extract(ev, recs, payload, SEM.SemanticConfig(), use_cache=False)
    ev.items, ev.conflicts, ev.observations = res.items, res.conflicts, res.observations
    return ev, payload


def _vlm_findings(ev) -> list[M.Item]:
    return [i for i in ev.items if i.type == "visual_finding"]


def t_vlm_fixture_flows(d: Path) -> None:
    """Ten visual kinds, end to end, each against what it should and should not yield.

    The per-fixture expectations are properties — a row label preserved, a
    direction recorded, an illegible axis yielding no value at all. Counting items
    would be measuring the extractor's output volume, which is the one number it
    can inflate without being caught.
    """
    from cache import JsonCache

    frames = _vlm_frames(d)
    check("W10 every fixture frame is a real file with distinct bytes",
          len({_png(seed=i) for i in range(len(frames))}) == len(frames))

    for kind, expect in VLM_EXPECTED.items():
        cache = JsonCache(f"vlm-{kind}", root=d / "cache")
        ev, payload = _vlm_run_one(d, kind, cache=cache)
        res = E.validate_evidence(ev)
        vlm_items = [i for i in ev.items if any(p.source == "vlm" for p in i.provenance)]
        check(f"W11 {kind}: the document validates", res.valid,
              str(res.blocking_errors[:2]))
        check(f"W12 {kind}: the observation is recorded on the frame",
              ev.visual[0].vlm_status == "ok" and ev.visual[0].vlm_observation
              and ev.visual[0].vlm_observation["content_type"] == VLM_FIXTURES[kind]["content_type"],
              str(ev.visual[0].vlm_status))
        check(f"W13 {kind}: the caption and the record agree",
              (ev.visual[0].vlm_caption or "") == VLM_FIXTURES[kind]["summary"])
        check(f"W14 {kind}: the run accounts for its own work",
              payload["summary"]["vlm_calls"] == 1 and payload["summary"]["calls_ok"] == 1,
              str(payload["summary"]))

        if expect.get("no_numbers"):
            check(f"W15 {kind}: no value is invented from an illegible frame",
                  not [i for i in vlm_items if i.type == "number"],
                  str([i.value for i in vlm_items if i.type == "number"]))
        if "numbers" in expect:
            # Keyed on the cell, not the row: two values in one row are exactly
            # the case this pass exists to get right, and a row-keyed dict would
            # have collapsed them and reported success.
            got = {((i.detail or {}).get("row"), (i.detail or {}).get("column")): i.value
                   for i in vlm_items if i.type == "number"}
            check(f"W16 {kind}: every value keeps the row and column it was seen under",
                  got == expect["numbers"], f"got {got}")
        if "number" in expect:
            check(f"W17 {kind}: the legible value is recovered",
                  any(i.type == "number" and i.value == expect["number"] for i in vlm_items),
                  str([i.value for i in vlm_items]))
        if "relationship" in expect:
            a, b = expect["relationship"]
            check(f"W18 {kind}: the connection survives with both ends",
                  any(i.type == "relationship" and (i.detail or {}).get("from") == a
                      and (i.detail or {}).get("to") == b for i in vlm_items),
                  str([(i.detail or {}).get("from") for i in vlm_items
                       if i.type == "relationship"]))
        if "relationships" in expect:
            got = [((i.detail or {}).get("from"), (i.detail or {}).get("to"))
                   for i in vlm_items if i.type == "relationship"]
            check(f"W19 {kind}: every arrow is recorded in order", got == expect["relationships"],
                  f"got {got}")
        if expect.get("uncertain_relationship"):
            check(f"W20 {kind}: an ambiguous direction is marked, not resolved",
                  any(i.type == "relationship" and (i.detail or {}).get("certain") is False
                      for i in vlm_items))
            check(f"W21 {kind}: and the uncertainty is surfaced to a reader",
                  any(w["code"] == "relationship-uncertain" for w in res.warnings),
                  str([w["code"] for w in res.warnings]))
        if "entity" in expect:
            check(f"W22 {kind}: a corroborated entity becomes an item",
                  any(i.value == expect["entity"] and i.type in ("model", "product", "tool")
                      for i in vlm_items), str([i.value for i in vlm_items]))
        if "refused_entity" in expect:
            check(f"W23 {kind}: a name only the model saw is refused",
                  not [i for i in vlm_items if i.value == expect["refused_entity"]],
                  str([i.value for i in vlm_items]))
            check(f"W24 {kind}: and the refusal is recorded with a reason",
                  any(o.get("classification") == "vlm-refused" and o.get("reason")
                      and expect["refused_entity"] in o["reason"] for o in ev.observations),
                  str([o.get("reason") for o in ev.observations])[:2])
        if expect.get("action"):
            check(f"W25 {kind}: the operation being demonstrated is recorded",
                  any((i.detail or {}).get("kind") == "action" for i in _vlm_findings(ev)))
        if expect.get("uncertainty"):
            check(f"W26 {kind}: what could not be read is kept, not dropped",
                  any((i.detail or {}).get("kind") == "uncertainty" for i in _vlm_findings(ev)))

    # The one that matters most, stated on its own: a chart with a cropped axis
    # yields an ordering and an uncertainty, and not one number.
    ev, _ = _vlm_run_one(d, "chart", cache=JsonCache("vlm-chart2", root=d / "cache"))
    numbers = [i for i in ev.items if i.type == "number" and any(p.source == "vlm" for p in i.provenance)]
    check("W27 a cropped axis produces no value at all", not numbers,
          str([i.value for i in numbers]))
    check("W28 and the ordering is what survives instead",
          any(i.type == "relationship" and (i.detail or {}).get("kind") == "ordering"
              for i in ev.items))


class _StubProvider(V.AnthropicProvider):
    """The real provider with its HTTP call replaced, so the failure paths run offline.

    Subclassing rather than inventing a third provider is what makes these tests
    about the production code: the status classification, the fatal latch, the
    request assembly and the response parsing are all the shipped ones, with only
    the socket replaced.
    """

    def __init__(self, label: str, exc: Exception | None = None, payload: dict | None = None):
        # Each stub declares its own model identity. Sharing one would make them
        # share a cache entry — which is the cache working correctly, since the
        # same provider, model, prompt, schema, frame and context genuinely are
        # the same question — and the failure cases would then be answered out of
        # a success another case had stored.
        super().__init__(model=f"stub-{label}", api_key="test-key-not-a-secret")
        self.exc = exc
        self.payload = payload
        self.posts = 0

    def _post(self, body: dict) -> dict:
        self.posts += 1
        if self.exc is not None:
            raise self.exc
        return self.payload or {}


def _tool_response(input_: dict) -> dict:
    return {"stop_reason": "tool_use",
            "content": [{"type": "tool_use", "name": V.TOOL_NAME, "input": input_}]}


def t_vlm_failures(d: Path) -> None:
    """Every way this pass can fail, and what it is required to leave behind.

    A provider that is absent, one that is configured but cannot authenticate,
    one that times out, one that is rate limited, one that answers in prose, and
    one that raises — six states that look identical from outside and must not be
    recorded identically, because "no model ran" and "a model ran and found
    nothing" and "the call failed" are three different facts about the artifact.
    """
    (d / "frames").mkdir(parents=True, exist_ok=True)
    (d / "frames" / "chart.png").write_bytes(_png(seed=99))
    frames = [M.VisualFrame(frame_id="chart", file="frames/chart.png", t_start=10.0,
                            ocr_text=VLM_SCREEN_TEXT["chart"], ocr_status="ok")]
    recs = [M.Alignment(snippet_id="s001", frame_ids=["chart"], confidence=0.6, method="overlap")]

    for name, provider, want in (
        ("none requested", V.get_provider("none"), "unavailable"),
        ("no credentials", V.get_provider("anthropic", api_key=""), "unavailable"),
        ("fake with no fixtures", V.get_provider("fake"), "unavailable"),
        ("a typo in the name", V.get_provider("gpt-vision"), "unavailable"),
    ):
        p = V.run_vlm(d, frames, recs, provider)
        check(f"W30 {name}: every frame says the pass did not run",
              all(f["status"] == want for f in p["frames"]), str(p["frames"][0]["status"]))
        check(f"W31 {name}: and the reason is on the record, not just absent",
              bool(p["summary"]["notes"]), str(p["summary"]))
        check(f"W32 {name}: no call was counted against a provider that made none",
              p["summary"]["vlm_calls"] == 0, str(p["summary"]["vlm_calls"]))

    good = _tool_response({"frame_id": "chart", "summary": "a chart",
                           "content_type": "chart", "confidence": 0.6})
    for label, exc, want in (
        ("a timeout", V._ProviderError("failed", "timeout after 90s"), "failed"),
        ("a rate limit", V._ProviderError("failed", "HTTP 429 rate limited"), "failed"),
        ("an authentication failure", V._ProviderError("unavailable", "HTTP 401", fatal=True),
         "unavailable"),
        ("a response that is not JSON", V._ProviderError("malformed", "not JSON"), "malformed"),
        ("an exception from the transport", RuntimeError("connection reset"), "failed"),
    ):
        stub = _StubProvider(label=label.split()[0], exc=exc)
        p = V.run_vlm(d, frames, recs, stub)
        rec = p["frames"][0]
        check(f"W33 {label} is recorded as {want}", rec["status"] == want, str(rec))
        check(f"W34 {label} is counted as a failed call",
              p["summary"]["calls_failed"] == 1, str(p["summary"]))
        check(f"W35 {label} is not stored as if it were an observation",
              V.JsonCache(f"vlm-fail-{want}", root=d / "cache").get(
                  V._cache_key(d / "frames" / "chart.png", "chart", {}, stub)) is None)

    stub = _StubProvider(label="prose", payload={"stop_reason": "end_turn",
                                 "content": [{"type": "text", "text": "It is a chart."}]})
    p = V.run_vlm(d, frames, recs, stub)
    check("W36 a model that answers in prose instead of calling the schema is refused",
          p["frames"][0]["status"] == "malformed", str(p["frames"][0]))

    stub = _StubProvider(label="good", payload=good)
    p = V.run_vlm(d, frames, recs, stub)
    check("W37 a well-formed response succeeds and says which model produced it",
          p["frames"][0]["status"] == "ok" and p["frames"][0]["model"] == "stub-good",
          str(p["frames"][0]))
    check("W38 the provider describes itself into the artifact",
          p["provider"]["structured_output"] is True and p["provider"]["model"] == "stub-good",
          str(p["provider"]))
    check("W39 the key never appears anywhere in the artifact",
          "api_key" not in json.dumps(p) and "x-api-key" not in json.dumps(p))

    # What would actually go over the wire, asserted without sending it. The
    # schema-checkable observation is the load-bearing part: a model left free
    # to answer in prose returns prose, and prose is not checkable against the
    # frame. Forced tool_choice used to supply that guarantee, but the forced
    # shape returns a 400 on current Claude models, so the call is now steered
    # from _INSTRUCTION under tool_choice "auto".
    sent: dict = {}

    def capture(body: dict) -> dict:
        sent.update(body)
        return good

    wire = _StubProvider(label="wire", payload=good)
    wire._post = capture  # type: ignore[method-assign]
    (d / "frames" / "chart.png").write_bytes(_png(seed=7))
    V.run_vlm(d, frames, recs, wire, cache=V.JsonCache("vlm-wire", root=d / "cache"),
              context_by_frame={"chart": {"video_title": "T", "timestamp": "00:10.0",
                                          "nearby_transcript": "here it is",
                                          "ocr_text": VLM_SCREEN_TEXT["chart"]}})
    check("W45 the request carries the schema and steers the call rather than forcing it",
          sent.get("tool_choice") == {"type": "auto"}
          and sent["tools"][0]["input_schema"] == V.OBSERVATION_SCHEMA
          and "calling the provided tool exactly once" in sent["system"],
          str(sent.get("tool_choice")))
    check("W46 exactly one image is sent, base64, with the type the file actually is",
          [b["type"] for b in sent["messages"][0]["content"]] == ["image", "text"]
          and sent["messages"][0]["content"][0]["source"]["media_type"] == "image/png",
          str(sent["messages"][0]["content"][0]["source"].get("media_type")))
    check("W47 the context reaches the model labelled as context, not as frame content",
          "CONTEXT, not content" in sent["messages"][0]["content"][1]["text"])
    check("W48 and the key is not in the request body",
          "test-key-not-a-secret" not in json.dumps(sent))

    # A fatal failure stops the pass rather than costing one attempt per frame.
    many = [M.VisualFrame(frame_id=f"f{i}", file=f"frames/f{i}.png", t_start=float(i),
                          ocr_text="Latency 100 ms", ocr_status="ok") for i in range(4)]
    for i, f in enumerate(many):
        (d / f.file).write_bytes(_png(seed=200 + i))
    stub = _StubProvider(label="auth", exc=V._ProviderError("unavailable", "HTTP 401", fatal=True))
    V.run_vlm(d, many, [], stub, max_frames=4)
    check("W40 an authentication failure is not retried once per frame", stub.posts == 1,
          f"{stub.posts} calls for 4 frames")

    # A frame nothing analysed cannot carry a model citation.
    ev = _build([(10.0, 3.0, "Here is the chart.")],
                [_frame_rec("s0001", 11.0, [_region("Inference Speed", 10, 10)])])
    ev.items.append(M.Item(id="forged", type="quote", value="a bar chart",
                           provenance=[P.vlm_provenance(ev.visual[0], "a bar chart")]))
    codes = E.validate_evidence(ev).error_codes()
    check("W41 a model citation on an unanalysed frame is refused", "vlm-without-result" in codes,
          str(codes))

    # A citation whose text the stored observation does not contain.
    ev2, _ = _vlm_run_one(d, "chart", cache=V.JsonCache("vlm-forge", root=d / "cache"))
    ev2.items.append(M.Item(id="forged2", type="quote", value="the y axis reads 0 to 100",
                            provenance=[P.vlm_provenance(ev2.visual[0], "the y axis reads 0 to 100")]))
    codes = E.validate_evidence(ev2).error_codes()
    check("W42 a model citation the observation never recorded is refused",
          "vlm-text-not-in-frame" in codes, str(codes))

    # A model confidence above the ceiling, as a hand-edited document would carry.
    ev3, _ = _vlm_run_one(d, "chart", cache=V.JsonCache("vlm-cap", root=d / "cache"))
    ev3.visual[0].vlm_confidence = 0.95
    codes = E.validate_evidence(ev3).error_codes()
    check("W43 a model observation above the confidence ceiling is refused",
          "vlm-confidence-over-cap" in codes, str(codes))

    # The frame's own OCR read is not the model's: a caption repeating OCR is not
    # a caption the model produced.
    ev4, _ = _vlm_run_one(d, "product", cache=V.JsonCache("vlm-ocr", root=d / "cache"))
    ev4.items.append(M.Item(id="forged3", type="quote", value="Acme Overview",
                            provenance=[P.vlm_provenance(ev4.visual[0], "Acme Overview")]))
    check("W44 text the frame's OCR read but not the model's is still not the model's",
          "vlm-text-not-in-frame" in E.validate_evidence(ev4).error_codes())


def t_vlm_cache(d: Path) -> None:
    """The key is the question. Every part of it has to move when the answer changes.

    A cache entry that survives a changed model, a changed prompt or a changed
    schema does not fail — it answers a question nobody asked, confidently, and
    the caller cannot tell. So each of those is moved here and checked.
    """
    from cache import JsonCache

    cache = JsonCache("vlm-cache", root=d / "cache")
    frames = [M.VisualFrame(frame_id="chart", file="frames/chart.png", t_start=10.0,
                            ocr_text=VLM_SCREEN_TEXT["chart"], ocr_status="ok")]
    path = d / "frames" / "chart.png"

    base_key = V._cache_key(path, "chart", {"nearby_transcript": "a"}, _vlm_provider())
    check("W50 the key is stable for the same question", base_key ==
          V._cache_key(path, "chart", {"nearby_transcript": "a"}, _vlm_provider()))

    p1 = V.run_vlm(d, frames, [], _vlm_provider(), context_by_frame={"chart": {"nearby_transcript": "a"}},
                   cache=cache)
    provider2 = _vlm_provider()
    p2 = V.run_vlm(d, frames, [], provider2, context_by_frame={"chart": {"nearby_transcript": "a"}},
                   cache=cache)
    check("W51 a second run is served from the cache and calls nobody",
          len(provider2.calls) == 0 and p2["summary"]["cache_hits"] == 1
          and p2["summary"]["vlm_calls"] == 1, str(p2["summary"]))
    check("W52 the cached answer is the answer, not a degraded one",
          {k: v for k, v in p2["frames"][0].items() if k != "cache"}
          == {k: v for k, v in p1["frames"][0].items() if k != "cache"},
          str(p2["frames"][0])[:120])

    other_provider = _vlm_provider()
    other_provider.name = "another-vendor"
    for label, prov, fid, ctx in (
        ("the model", V.FakeProvider(VLM_FIXTURES, model="fixture-v2"), "chart",
         {"nearby_transcript": "a"}),
        ("the provider", other_provider, "chart", {"nearby_transcript": "a"}),
        ("the context", _vlm_provider(), "chart", {"nearby_transcript": "something else"}),
        ("the frame", _vlm_provider(), "other", {"nearby_transcript": "a"}),
    ):
        key = V._cache_key(path, fid, ctx, prov)
        check(f"W53 changing {label} changes the cache key", key != base_key, key)

    original = V.PROMPT_VERSION
    try:
        V.PROMPT_VERSION = original + 1
        check("W54 changing the prompt version invalidates every cached analysis",
              V._cache_key(path, "chart", {"nearby_transcript": "a"}, _vlm_provider()) != base_key)
    finally:
        V.PROMPT_VERSION = original

    # A failure is not an observation and must not become a stored one.
    stub = _StubProvider(label="retry", exc=V._ProviderError("failed", "HTTP 429"))
    V.run_vlm(d, frames, [], stub, cache=cache)
    check("W55 a rate-limited call is not cached",
          cache.get(V._cache_key(path, "chart", {}, stub)) is None)
    stub.payload = _tool_response({"frame_id": "chart", "summary": "a chart",
                                   "content_type": "chart", "confidence": 0.5})
    stub.exc = None
    p3 = V.run_vlm(d, frames, [], stub, cache=cache)
    check("W56 and the same frame is retried successfully on the next run",
          p3["frames"][0]["status"] == "ok", str(p3["frames"][0]))
    check("W57 the retried answer is now cached",
          cache.get(V._cache_key(path, "chart", {}, stub)) is not None)


def t_vlm_selection(d: Path) -> None:
    """The budget, the suppression, and the determinism. The three promises.

    A selector that sends every frame of a screen recording is not wrong about
    any individual frame; it is wrong about the bill. So the cap is asserted
    against a frame set much larger than it, and the talking-head suppression is
    asserted against frames that would otherwise qualify on a stray region.
    """
    (d / "frames").mkdir(parents=True, exist_ok=True)
    many = []
    for i in range(40):
        dense = i % 4 == 0
        many.append(M.VisualFrame(
            frame_id=f"f{i:02d}", file=f"frames/f{i:02d}.png", t_start=float(i),
            ocr_text=("Latency 100 ms Throughput 32 tok/s Model" if dense
                      else ("x" if i % 7 == 0 else None)),
            ocr_status="ok" if dense or i % 7 == 0 else "no_text",
            ocr_regions=[{"text": f"r{j}", "confidence": 0.9, "bbox": [j, j, 5, 5]}
                         for j in range(40 if dense else 1)]))
        (d / many[-1].file).write_bytes(_png(seed=300 + i))

    chosen = V.select_frames(many, [], max_frames=5)
    check("W60 the budget is hard, on a frame set far larger than it",
          len(chosen) == 5, str(len(chosen)))
    scored = V.score_frames(many, [])
    check("W61 the frames that did not fit are counted, not silently dropped",
          len(scored) > len(chosen), f"{len(scored)} eligible, {len(chosen)} sent")
    p = V.run_vlm(d, many, [], _vlm_provider(), max_frames=5, cache=V.JsonCache("vlm-sel", root=d / "cache"))
    check("W62 the run reports what it considered, selected and paid for",
          p["summary"]["candidates_considered"] == len(scored)
          and p["summary"]["frames_selected"] == 5 and p["summary"]["max_vlm_frames"] == 5,
          str(p["summary"]))
    check("W63 the cost is reported as unknown rather than guessed",
          p["summary"]["estimated_cost_usd"] is not None
          and p["provider"]["cost_per_image_usd"] == 0.0, str(p["summary"]))

    talking = [M.VisualFrame(frame_id="t0", file="frames/t0.png", t_start=1.0,
                             ocr_text=None, ocr_status="no_text",
                             ocr_regions=[{"text": "", "confidence": 0.0, "bbox": [0, 0, 1, 1]}] * 2)]
    check("W64 a talking head with no text is never sent", V.select_frames(talking, []) == [],
          str(V.select_frames(talking, [])))
    check("W65 selection is deterministic for the same inputs",
          chosen == V.select_frames(many, [], max_frames=5))
    check("W66 a budget of zero sends nothing", V.select_frames(many, [], max_frames=0) == [])
    p0 = V.run_vlm(d, many, [], _vlm_provider(), max_frames=0)
    check("W67 and the run still produces a complete, honest artifact",
          p0["summary"]["vlm_calls"] == 0
          and all(f["status"] == "skipped" for f in p0["frames"]), str(p0["summary"]))

    ctx = V.build_context(
        _build([(10.0, 3.0, "Here is the chart.")], [_frame_rec("f", 11.0, [_region("x", 1, 1)])]),
        _build([(10.0, 3.0, "Here is the chart.")], [_frame_rec("f", 11.0, [_region("x", 1, 1)])]).visual[0],
        [M.Alignment(snippet_id="s001", frame_ids=["f"], confidence=0.6, method="overlap")])
    check("W68 the context is bounded, and says what it is",
          len(json.dumps(ctx)) < 6000 and "CONTEXT, not content" in V._context_text(ctx),
          str(len(json.dumps(ctx))))


def _claim_run(d: Path, speech: str, screen: str, observation: dict,
               cache_name: str) -> M.Evidence:
    """A document where a spoken claim and one frame meet, for the support cases."""
    (d / "frames").mkdir(parents=True, exist_ok=True)
    ev = _build([(10.0, 4.0, speech)], [_frame_rec("s0001", 11.0, [_region(screen, 10, 10)])])
    ev.visual[0].file = "frames/s0001.png"
    (d / ev.visual[0].file).write_bytes(_png(seed=400))
    recs, _, _ = _aligned(ev)
    ctxs = {f.frame_id: V.build_context(ev, f, recs) for f in ev.visual}
    payload = V.run_vlm(d, ev.visual, recs,
                        V.FakeProvider({"s0001": observation}, model="fixture-v1"),
                        context_by_frame=ctxs, cache=V.JsonCache(cache_name, root=d / "cache"))
    V.apply_observations(ev.visual, payload)
    res = SEM.extract(ev, recs, payload, SEM.SemanticConfig(), use_cache=False)
    ev.items, ev.conflicts, ev.observations = res.items, res.conflicts, res.observations
    return ev


def t_vlm_crossmodal(d: Path) -> None:
    """What the frame does to what was said: support, disagreement, and silence.

    Three outcomes, and the third is the one that is easiest to get wrong. A
    model saying `unclear` has not asserted a disagreement, and a pipeline that
    promotes it to one has manufactured a conflict — the same move as turning a
    hedge into an assertion, in a place where a reader would take it as evidence
    about the video rather than about the extractor.
    """
    claim = "This model is the fastest one here."

    ev = _claim_run(d, claim, "Qwen 32 tok/s\nLlama 18 tok/s", {
        "content_type": "chart", "confidence": 0.65,
        "summary": "A bar chart in which Qwen's bar is taller than Llama's.",
        "relationships": [{"from": "Qwen", "to": "Llama", "kind": "ordering", "certain": True}],
        "claims_supported": [{"claim": claim, "support": "supported",
                              "reason": "the ordering on screen matches the claim",
                              "confidence": 0.6}],
    }, "vlm-support")
    res = E.validate_evidence(ev)
    check("W70 a supported claim keeps both modalities as separate provenance",
          any(i.type == "claim" and len([p for p in i.provenance if p.source == "vlm"]) == 1
              and len([p for p in i.provenance if p.source == "transcript"]) == 1
              for i in ev.items), str([(i.type, [p.source for p in i.provenance]) for i in ev.items]))
    backed = next(i for i in ev.items if i.type == "claim"
                  and any(p.source == "vlm" for p in i.provenance))
    check("W71 and the two confidences are not averaged into one",
          backed.confidence < 0.95
          and all(p.confidence <= V.MAX_MODEL_CONFIDENCE for p in backed.provenance
                  if p.source == "vlm"), str(backed.confidence))
    check("W72 the document with corroboration validates", res.valid, str(res.blocking_errors[:2]))
    check("W73 and no conflict is raised where the modalities agree", not ev.conflicts,
          str([c.statement for c in ev.conflicts]))

    ev = _claim_run(d, claim, "Qwen 12 tok/s\nLlama 40 tok/s", {
        "content_type": "chart", "confidence": 0.6,
        "summary": "A bar chart in which Llama's bar is taller than Qwen's.",
        "relationships": [{"from": "Llama", "to": "Qwen", "kind": "ordering", "certain": True}],
        "claims_supported": [{"claim": claim, "support": "not_supported",
                              "reason": "the taller bar belongs to the other model",
                              "confidence": 0.55}],
    }, "vlm-disagree")
    check("W74 a frame that reads as contradicting the speaker raises a conflict",
          any(c.kind == "source-disagreement" for c in ev.conflicts),
          str([c.kind for c in ev.conflicts]))
    check("W75 and both records survive — neither is corrected against the other",
          any(i.status == "conflicting" and i.type == "claim" for i in ev.items)
          and any(i.status == "conflicting" and i.type == "visual_finding" for i in ev.items))
    check("W76 the conflict is left unresolved", all(c.resolution == "unresolved" for c in ev.conflicts))
    check("W77 the spoken sentence is not rewritten",
          any(i.type == "claim" and i.verbatim == claim for i in ev.items))
    check("W78 a document recording a disagreement still validates",
          E.validate_evidence(ev).valid, str(E.validate_evidence(ev).blocking_errors[:2]))

    ev = _claim_run(d, claim, "Qwen 32 tok/s", {
        "content_type": "chart", "confidence": 0.4,
        "summary": "A chart whose legend is not readable.",
        "uncertainties": [{"text": "the legend cannot be read"}],
        "claims_supported": [{"claim": claim, "support": "unclear",
                              "reason": "the legend is illegible", "confidence": 0.3}],
    }, "vlm-unclear")
    check("W79 'unclear' is recorded as unresolved, not as disagreement",
          not ev.conflicts, str([c.statement for c in ev.conflicts]))
    check("W80 and it is kept as a finding rather than dropped",
          any((i.detail or {}).get("kind") == "uncertainty" for i in ev.items),
          str([i.type for i in ev.items]))

    # A value whose digits are in neither the frame nor the speech is refused
    # even when the model attaches a plausible label to it.
    ev = _claim_run(d, "Pricing is under a dollar.", "Plan Pro $0.09", {
        "content_type": "table", "confidence": 0.7,
        "summary": "A pricing table with a Pro row.",
        "numbers": [{"value": "0.09", "unit": "$", "row": "Plan Pro", "column": "Price",
                     "legible": True}],
    }, "vlm-price")
    priced = [i for i in ev.items if i.type == "number" and i.value == "0.09"]
    check("W81 a price on screen is recovered with the row it was under",
          len(priced) == 1 and priced[0].subject == "Plan Pro",
          str([(i.value, i.subject) for i in priced]))

    ev = _claim_run(d, "We did about 700 users last week.", "counter 700", {
        "content_type": "application_ui", "confidence": 0.6,
        "summary": "A subscriber counter reading 700.",
        "numbers": [{"value": "700", "unit": None, "label": "Product X subscribers",
                     "legible": True}],
    }, "vlm-antipattern")
    attributed = [i for i in ev.items if i.type == "number" and i.value == "700"
                  and i.subject == "Product X subscribers"]
    check("W82 a figure the model attributed to something the frame does not show is refused",
          not attributed, str([(i.value, i.subject) for i in ev.items if i.type == "number"]))
    check("W83 and the refusal is on the record with its reason",
          any(o.get("classification") == "vlm-refused" and "Product X" in o.get("reason", "")
              for o in ev.observations), str([o.get("reason") for o in ev.observations])[:2])
    check("W84 a bare figure with no grounded label is not given a subject",
          not [i for i in ev.items if i.type == "number" and i.subject
               and "Product X" in i.subject])

    # A frame the aligner never paired is still inside the segment's time range.
    # If it is not filed there, a model call buys an observation no consumer can
    # read — paid for, recorded in full, and dropped without a trace.
    (d / "frames").mkdir(parents=True, exist_ok=True)
    for kind in ("chart", "diagram"):
        (d / "frames" / f"{kind}.png").write_bytes(_png(seed=list(VLM_FIXTURES).index(kind)))
    ev = _build([(10.0, 4.0, "Here is the pipeline diagram.")], [])
    ev.visual = [f for f in _vlm_frames(d) if f.frame_id in ("chart", "diagram")]
    ev.visual[0].file, ev.visual[1].file = "frames/chart.png", "frames/diagram.png"
    for f in ev.visual:
        f.t_start = 11.5  # inside the segment, but paired with no snippet
    recs: list = []      # no alignments at all: the frame is reachable only by time
    payload = V.run_vlm(d, ev.visual, recs, _vlm_provider(),
                        context_by_frame={f.frame_id: V.build_context(ev, f, recs)
                                          for f in ev.visual},
                        cache=V.JsonCache("vlm-reach", root=d / "cache"))
    V.apply_observations(ev.visual, payload)
    B._attach_analysed_frames(ev, payload)
    res2 = SEM.extract(ev, recs, payload, SEM.SemanticConfig(), use_cache=False)
    ev.items, ev.conflicts, ev.observations = res2.items, res2.conflicts, res2.observations
    check("W88 an analysed frame is filed under the segment that was playing",
          set(ev.segments[0].frame_ids) == {"chart", "diagram"}, str(ev.segments[0].frame_ids))
    check("W89 and its edges reach the document",
          any(i.type == "relationship" and (i.detail or {}).get("to") == "API"
              for i in ev.items), str([i.type for i in ev.items]))

    # Numbers that disagree across modalities keep both and become a conflict.
    ev = _claim_run(d, "It costs $75 a month, which is the expensive plan.",
                    "Plan 8 dollars a month", {
        "content_type": "table", "confidence": 0.7,
        "summary": "A pricing table.",
        "numbers": [{"value": "8", "unit": "dollars", "row": "Plan", "column": "Month",
                     "legible": True}],
    }, "vlm-value-conflict")
    check("W85 both figures survive, neither corrected against the other",
          any(i.value == "75" for i in ev.items) and any(i.value == "8" for i in ev.items),
          str([(i.type, i.value) for i in ev.items]))
    check("W86 the transcript's figure keeps its own provenance",
          any(i.value == "75" and any(p.source == "transcript" for p in i.provenance)
              for i in ev.items), str([(i.value, [p.source for p in i.provenance])
                                       for i in ev.items]))
    check("W87 and the frame's figure is recorded with its own",
          any(i.type == "number" and i.value == "8"
              and any(p.source == "vlm" for p in i.provenance) for i in ev.items))


# --------------------------------------------------------------------------- #
# S · the frame sampling policy
# --------------------------------------------------------------------------- #

#: The four synthetic sources: a static card, a differently-coloured static card
#: (a cut is a change between them), a moving test pattern, and a high-contrast
#: grid that stands in for a code editor or a UI.
_CARD, _OTHER, _MOVING = "color=c=navy", "color=c=darkgreen", "testsrc2"


def _synth(d: Path, name: str, clips: list[tuple[str, float]]) -> Path:
    """Concatenate lavfi clips into one video, offline, no network.

    Each clip is a *complete* source description rather than a name this helper
    expands. The sources disagree about their own option names — `color` takes
    `size`/`rate` and `testsrc2` takes `s`/`r`, and mixing them is a parse error
    rather than a source — so the one thing that cannot be composed here is the
    part that differs. The length comes from `-t`, because `testsrc2` is an
    endless source and rejects a `duration` option outright.
    """
    for i, (src, length) in enumerate(clips):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", src,
             "-t", str(length), "-pix_fmt", "yuv420p", str(d / f"{name}{i}.mp4")], check=True)
    listing = "".join(f"file {name}{i}.mp4\n" for i in range(len(clips)))
    (d / f"{name}.txt").write_text(listing)
    out = d / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(d / f"{name}.txt"), "-c", "copy", str(out)], check=True)
    return out


#: The synthetic sources. Two flat cards whose boundary is a hard cut, and a
#: moving pattern that changes without ever producing one.
_CARD = "color=c=navy:size=1280x720:rate=12"
_OTHER = "color=c=darkgreen:size=1280x720:rate=12"
_THIRD = "color=c=maroon:size=1280x720:rate=12"
_MOVING = "testsrc2=s=1280x720:r=12"


def t_sampling_profiles() -> None:
    """Four shapes, four densities — and the profile says why."""
    cases = [
        ("a flat card with no cuts is a talking head", [], "talking_head"),
        ("a handful of cuts over fifteen minutes is still a talking head",
         [float(300 * i) for i in range(1, 5)], "talking_head"),
        ("a slide changing once a minute is a lecture",
         [float(60 * i) for i in range(1, 15)], "lecture"),
        ("a cut every four seconds is dynamic",
         [float(4 * i) for i in range(1, 225)], "dynamic"),
        ("a run of changes that never reaches a hard cut is a lecture",
         [float(200 + 11 * i) for i in range(40)], "lecture"),
    ]
    for label, scene_times, want in cases:
        got = Y.classify_visual_profile(scene_times, 900.0)
        check(f"S1 {label}", got.name == want, f"got {got.name} ({got.reason})")
        check(f"S1 {label}: the reason says which measurement decided it",
              bool(got.reason), "an unexplained profile is not a decision")


#: A screen that reads as an editor: a lot of text, most of it code-shaped.
_CODE_SCREEN = ("  const handleClick = (e) => {\n    setOpen(!open);\n  };\n"
                "  useEffect(() => { fetch(url) }, [url]);\n  return <Panel />;\n")
#: A screen that reads as an application: chrome, labels, no code.
_UI_SCREEN = ("Settings   Profile   Billing   Log out\n"
              "Storage    42 GB of 100 GB used\n"
              "Backups    Last run 3 days ago    Enable now\n"
              "Members    12 seats    Renew subscription    Manage\n")


def t_sampling_text_probe_profiles() -> None:
    """What is *written* separates the profiles a cut count cannot.

    A screen recording, a code editor and a slide deck all look identical to a
    scene score: none of them cuts. Reading a dozen small frames before the plan
    is made is what puts them apart, and the ordering matters — a montage that
    also carries captions is still a montage, so cuts stay the primary signal and
    the text only refines the quiet cases.
    """
    probe = lambda body, n=16: [(float(i), body) for i in range(n)]
    cases = [
        ("a quiet video with nothing written on it is a talking head",
         [], probe(""), "talking_head"),
        ("a quiet video full of code is code", [], probe(_CODE_SCREEN), "code"),
        ("a quiet video full of application chrome is screen_ui",
         [], probe(_UI_SCREEN), "screen_ui"),
        ("a montage is dynamic whatever it says",
         [float(4 * i) for i in range(1, 225)], probe(_CODE_SCREEN), "dynamic"),
        ("a slide deck changing once a minute is a lecture",
         [float(60 * i) for i in range(1, 15)], probe(_CODE_SCREEN), "lecture"),
    ]
    for label, cuts, reads, want in cases:
        got = Y.classify_visual_profile(cuts, 900.0, Y.ocr_profile(reads))
        check(f"S1b {label}", got.name == want, f"got {got.name} ({got.reason})")
        check(f"S1b {label}: the reason names the measurement that decided it",
              bool(got.reason), "an unexplained profile is not a decision")

    # The read is a refinement, never a licence to skip the video. Frames that
    # read nothing still count as frames the sampler looked at.
    empty = Y.ocr_profile([(0.0, ""), (1.0, "")])
    check("S1b a frame that read nothing is not a frame that was not read",
          empty.frames == 2 and empty.text_frames == 0, str(empty.to_dict()))
    check("S1b an unreadable video leaves the cut decision standing",
          Y.classify_visual_profile([1.0, 2.0], 600.0, empty).name == "talking_head",
          Y.classify_visual_profile([1.0, 2.0], 600.0, empty).name)

    # Density is the only thing a profile decides, and a denser profile samples
    # finer — which is the whole reason the read happens before the plan.
    steps = {name: Y.VISUAL_PROFILES[name] for name in
             ("talking_head", "lecture", "screen_ui", "code")}
    check("S1b denser profiles sample finer, in order",
          steps["talking_head"] > steps["lecture"] > steps["screen_ui"] > steps["code"],
          str(steps))
    check("S1b every profile's step lands in the range the policy targets",
          all(4.0 <= s <= 40.0 for s in steps.values()), str(steps))

    # A ratio over three lines is not evidence. A test pattern burns a timecode
    # into its corner, two of those three lines match a code shape, and the
    # classifier read that as a terminal and sampled the card at 6s.
    thin = Y.ocr_profile([(float(i), "00:00:12.34") for i in range(16)])
    got = Y.classify_visual_profile([], 600.0, thin)
    check("S1b too little text to classify on falls back to the cut count",
          got.name == "talking_head", f"got {got.name} ({got.reason})")
    check("S1b the fallback says why it fell back", "too few" in got.reason, got.reason)
    few = Y.ocr_profile([(float(i), "x = 1") for i in range(16)])
    check("S1b one code-shaped line is not a code editor",
          Y.classify_visual_profile([], 600.0, few).name == "talking_head",
          Y.classify_visual_profile([], 600.0, few).name)


def t_sampling_coverage_and_budget() -> None:
    """The two invariants the whole policy rests on, checked without decoding.

    * **Complete coverage.** The baseline grid spans [0, duration] inclusive, so
      the last frame is the end of the video. The exclusive version looked
      covered — a frame 24s into a 36s video is inside the video — and left the
      entire final scene with no frame at all.
    * **A hard budget.** The layers are sized to sum to the budget, so the total
      is bounded by construction rather than trimmed afterwards. The old sampler
      passed the budget to each ffmpeg pass separately and a run could write
      twice what it was asked for.
    """
    for profile in Y.VISUAL_PROFILES:
        prof = Y.VisualProfile(profile, Y.VISUAL_PROFILES[profile], "test")
        for dur in (60.0, 600.0, 1120.0, 3600.0):
            for budget in (0, 1, 2, 5, 40, 160, 1000):
                plan = Y.plan_sampling(dur, budget, prof)
                ts = plan.times()
                check(f"S2 {profile}/{dur:.0f}s/{budget}: within the budget",
                      len(ts) <= budget or budget == 0, f"{len(ts)} > {budget}")
                if not ts or budget == 0:
                    continue
                check(f"S2 {profile}/{dur:.0f}s/{budget}: ordered and unique",
                      all(b > a for a, b in zip(ts, ts[1:])), str(ts[:8]))
                if plan.baseline_count == 1:
                    # One frame cannot be at both ends. It goes in the middle,
                    # where the content is, rather than at 0s where the title
                    # card usually is.
                    check(f"S2 {profile}/{dur:.0f}s/{budget}: a lone frame is mid-video",
                          abs(ts[0] - dur / 2) < 0.01, str(ts))
                    continue
                check(f"S2 {profile}/{dur:.0f}s/{budget}: starts at the beginning",
                      ts[0] <= 0.001, f"first frame at {ts[0]}")
                check(f"S2 {profile}/{dur:.0f}s/{budget}: reaches the end",
                      abs(ts[-1] - dur) < 0.01, f"last frame at {ts[-1]} of {dur}")
                grid = [round(i * plan.baseline_step, 3)
                        for i in range(plan.baseline_count)]
                check(f"S2 {profile}/{dur:.0f}s/{budget}: the baseline is the whole video",
                      all(any(abs(t - g) < 0.01 for g in grid) for t in ts),
                      "a refinement frame escaped the baseline grid")


def t_sampling_metrics_are_reported() -> None:
    """Every number the policy claims to produce, and the ones that keep it honest.

    A metric nobody reads is a comment. These are the ones a run is compared on
    before and after a change: how many frames each layer asked for, how they
    were spread, and how many of the video's own sections a frame landed in.
    The last is the one that catches the failure the frame count cannot — three
    frames in the first ten seconds of a lecture and none in the last five
    minutes is the same number as three spread evenly.
    """
    d = Path(tempfile.mkdtemp(prefix="yt-metrics-"))
    try:
        v = _synth(d, "deck", [(_CARD, 30.0), (_OTHER, 30.0), (_THIRD, 30.0)])
        segs = [{"start": 0.0, "end_s": 30.0, "title": "A"},
                {"start": 30.0, "end_s": 60.0, "title": "B"},
                {"start": 60.0, "end_s": 90.0, "title": "C"}]
        res = Y.extract_frames(v, d / "out", 90.0, Y.auto_frame_budget(90.0), segments=segs)
        for key in ("duration_s", "uniform_candidates", "scene_candidates",
                    "refinement_candidates", "final_frames", "frames_per_minute",
                    "coverage_start_s", "coverage_end_s", "largest_temporal_gap_s",
                    "median_temporal_gap_s", "p95_temporal_gap_s",
                    "segments_with_visual_coverage", "visual_density_profile",
                    "text_probe"):
            check(f"S1c the run reports {key}", key in res, str(sorted(res))[:300])
        check("S1c the final frame count is reported as itself",
              res["final_frames"] == res["frames"], str(res["frames"]))
        check("S1c the three candidate counts are separable",
              res["uniform_candidates"] > 0
              and res["scene_candidates"] + res["refinement_candidates"]
              == res["refinement_intervals"],
              f"uniform={res['uniform_candidates']} scene={res['scene_candidates']} "
              f"refine={res['refinement_candidates']} intervals={res['refinement_intervals']}")
        check("S1c refinement never outnumbers the baseline",
              res["refinement_candidates"] <= res["uniform_candidates"],
              f"uniform={res['uniform_candidates']} refine={res['refinement_candidates']}")
        check("S1c section coverage is counted, not assumed",
              0 <= res["segments_with_visual_coverage"] <= len(segs),
              str(res["segments_with_visual_coverage"]))
        check("S1c the text probe reports its own evidence",
              isinstance(res["text_probe"], dict) and "reason" in res["text_probe"],
              str(res["text_probe"]))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_sampling_kinds_survive_the_evidence_gate() -> None:
    """Every kind the sampler writes must be one the evidence schema accepts.

    The two halves of the pipeline named the same concept in two vocabularies
    and nothing compared them. `yt_fetch` writes `refinement` into `frames.json`
    and `evidence_model` had never heard of it, so every run that sampled a
    refinement frame was refused by the evidence gate — 52 identical blocking
    errors on an 18-minute video — and the suite was green, because no test fed
    a sampled frame set through a full validation.

    The assertion is a subset check, not a list: it holds for whatever the
    sampler grows next, and it fails at the moment someone adds a layer rather
    than months later on a real video.
    """
    from evidence_model import FRAME_KINDS  # noqa: PLC0415
    written = set(Y.FRAME_KIND_BY_PREFIX.values()) | {"unknown"}
    check("S1d every kind the sampler writes is one the schema accepts",
          written <= set(FRAME_KINDS), f"unknown to the gate: {sorted(written - set(FRAME_KINDS))}")
    check("S1d the layers stay distinguishable in the artifact",
          {"uniform", "refinement"} <= written, str(sorted(written)))


def t_sampling_is_deterministic() -> None:
    """The same inputs must give the same plan, byte for byte.

    A sampler that varies run to run makes every report non-reproducible and
    every cache meaningless, and the variation is invisible until two runs of the
    same video are diffed — which nobody does.
    """
    prof = Y.classify_visual_profile([12.0, 40.0, 91.5, 300.0], 600.0)
    hints = Y.hint_intervals([{"start": 30.0, "dur": 4.0, "text": "as you can see on this screen"}],
                             [{"start": 0.0, "end_s": 300.0, "title": "A"}])
    first = Y.plan_sampling(600.0, 80, prof, hints, Y._change_intervals([12.0, 40.0])).times()
    for _ in range(3):
        again = Y.plan_sampling(600.0, 80, prof, hints, Y._change_intervals([12.0, 40.0])).times()
        check("S3 the plan is identical across runs", again == first, str(again[:6]))


    prof = Y.classify_visual_profile([12.0, 40.0, 91.5, 300.0], 600.0)
    hints = Y.hint_intervals([{"start": 30.0, "dur": 4.0, "text": "as you can see on this screen"}],
                             [{"start": 0.0, "end_s": 300.0, "title": "A"}])
    first = Y.plan_sampling(600.0, 80, prof, hints, Y._change_intervals([12.0, 40.0])).times()
    for _ in range(3):
        again = Y.plan_sampling(600.0, 80, prof, hints, Y._change_intervals([12.0, 40.0])).times()
        check("S3 the plan is identical across runs", again == first, str(again[:6]))

    # Determinism is only half the contract. The other half is that a *config*
    # change actually changes the answer, because a sampler whose output ignored
    # its own settings would pass every check above and be wrong every time.
    # `SAMPLING_VERSION` is the stamp that makes the difference visible to a
    # cached artifact; these are the inputs it stands for.
    #
    # Note what a bigger budget does and does not do. It never buys a finer
    # *baseline* — the grid is a function of duration and profile alone, which is
    # the whole reason no amount of cutting elsewhere can move coverage — so the
    # test is that the answer never gets coarser and the grid never moves, not
    # that it gets denser.
    big_plan = Y.plan_sampling(600.0, 160, prof, hints, Y._change_intervals([12.0, 40.0]))
    bigger = big_plan.times()
    check("S3 a larger budget never samples less", len(bigger) >= len(first),
          f"{len(first)} -> {len(bigger)}")
    check("S3 a larger budget does not move the coverage grid",
          big_plan.baseline_step == Y.plan_sampling(600.0, 80, prof).baseline_step,
          f"{big_plan.baseline_step} vs {Y.plan_sampling(600.0, 80, prof).baseline_step}")
    check("S3 the budget is still a ceiling after the change", len(bigger) <= 160,
          str(len(bigger)))
    finer = Y.VisualProfile("screen_ui", Y.VISUAL_PROFILES["screen_ui"], "test")
    check("S3 a denser profile asks for a denser sample",
          len(Y.plan_sampling(600.0, 80, finer).times())
          > len(Y.plan_sampling(600.0, 80, prof).times()),
          f"{prof.step_s}s vs {finer.step_s}s")
    check("S3 the stamped version is what a cached artifact is checked against",
          "SAMPLING_VERSION" in (Path(Y.__file__).parent / "versions.py").read_text(),
          "the stamp is not in the versions file")


def t_sampling_does_not_starve_the_tail() -> None:
    """A montage-heavy opening must not spend the budget.

    The old sampler gave the scene filter and the uniform grid the same budget
    each, so a title montage could take everything and the tail got whatever was
    left. The regression here is the one that matters: however violent the
    opening, the last quarter of the video is still sampled.
    """
    prof = Y.classify_visual_profile([float(4 * i) for i in range(1, 60)], 900.0)
    changes = Y._change_intervals([float(4 * i) for i in range(1, 60)])
    plan = Y.plan_sampling(900.0, 60, prof, [], changes)
    ts = plan.times()
    tail = [t for t in ts if t >= 675.0]
    check("S4 the tail is sampled despite a montage at the front", len(tail) >= 3,
          f"{len(tail)} frame(s) in the last quarter: {ts[-4:]}")
    check("S4 the refinement budget went where the changes were",
          sum(1 for t in ts if t < 250.0) > sum(1 for t in ts if t >= 675.0),
          "the opening did not get the extra frames")
    # A hint is not a licence to sample the whole video.
    everywhere = [{"start": 0.0, "end": 900.0, "weight": 5, "reason": "hints at everything"}]
    plan2 = Y.plan_sampling(900.0, 60, prof, everywhere)
    check("S4 an interval covering the whole video is refused as a hint",
          len(plan2.intervals) == 0, str(plan2.intervals))


def t_sampling_on_synthetic_videos() -> None:
    """The shapes, decoded for real, through the actual sampler.

    Everything above tests the plan. This runs `extract_frames` and checks the
    frames that came out, because a plan nothing reads is a comment.

    The assertions are deliberately about things that hold on *any* content —
    the budget, determinism, the profile, the plan's coverage. How many frames
    survive is not asserted against a number, because on a synthetic flat card
    every sample is the same image and deduplication is right to keep one: the
    interesting question is whether the sampler *looked*, and `planned_frames`
    beside `frames` is the answer.
    """
    d = Path(tempfile.mkdtemp(prefix="yt-sampling-"))
    try:
        v1 = _synth(d, "head", [(_CARD, 120.0)])
        r1 = Y.extract_frames(v1, d / "o1", 120.0, Y.auto_frame_budget(120.0))
        check("S5 a flat card with no cuts is a talking head",
              r1["visual_density_profile"] == "talking_head", str(r1["profile_reason"]))
        check("S5 a talking head stays sparse",
              r1["uniform_step_s"] >= 20.0, str(r1["uniform_step_s"]))
        check("S5 a talking head respects its budget",
              r1["frames"] <= Y.auto_frame_budget(120.0), str(r1["frames"]))
        check("S5 the plan covered the whole video even though dedupe kept one frame",
              r1["planned_coverage_end_s"] == 120.0, str(r1["planned_coverage_end_s"]))
        check("S5 the plan asked for more than the flat video could justify",
              r1["planned_frames"] > 1, str(r1["planned_frames"]))

        v2 = _synth(d, "deck", [(_CARD, 20.0), (_OTHER, 20.0), (_THIRD, 20.0),
                                (_CARD, 20.0), (_OTHER, 20.0), (_THIRD, 20.0)])
        r2 = Y.extract_frames(v2, d / "o2", 120.0, Y.auto_frame_budget(120.0))
        check("S5 a slide deck is not a talking head",
              r2["visual_density_profile"] == "lecture", str(r2["profile_reason"]))
        check("S5 a slide deck is sampled finer than a talking head",
              r2["uniform_step_s"] < r1["uniform_step_s"],
              f"{r2['uniform_step_s']} vs {r1['uniform_step_s']}")
        check("S5 the cuts were found", r2["hard_cuts_detected"] >= 4,
              str(r2["hard_cuts_detected"]))
        check("S5 the slide changes became refinement intervals",
              r2["change_intervals"] >= 1, str(r2["change_intervals"]))
        check("S5 the plan reached the end of a 120s video",
              r2["planned_coverage_end_s"] == 120.0, str(r2["planned_coverage_end_s"]))

        v3 = _synth(d, "motion", [(_MOVING, 90.0)])
        r3 = Y.extract_frames(v3, d / "o3", 90.0, Y.auto_frame_budget(90.0))
        check("S5 a moving pattern is sampled and its budget respected",
              r3["frames"] > 1 and r3["frames"] <= Y.auto_frame_budget(90.0),
              f"{r3['frames']} frame(s) of a {Y.auto_frame_budget(90.0)} budget")
        check("S5 motion is not mistaken for a hard cut",
              r3["visual_density_profile"] in ("talking_head", "lecture", "dynamic"),
              str(r3["profile_reason"]))

        # A late burst: nothing for two minutes, then twenty fast changes.
        v4 = _synth(d, "late", [(_CARD, 100.0)] + [(_OTHER, 2.0)] * 20)
        r4 = Y.extract_frames(v4, d / "o4", 140.0, Y.auto_frame_budget(140.0))
        check("S5 a late visual burst is still covered",
              r4["planned_coverage_end_s"] == 140.0, str(r4["planned_coverage_end_s"]))
        check("S5 a burst at the end does not move the opening",
              r4["coverage_start_s"] <= 0.5, str(r4["coverage_start_s"]))

        a = Y.extract_frames(v2, d / "o2b", 120.0, Y.auto_frame_budget(120.0))
        # Timings are excluded because they are the one thing a run genuinely
        # cannot reproduce; everything else has to match, and a new timing
        # landing in this dict fails the test rather than being quietly excused
        # by a wildcard. `probe_frames` and `probe_backend` are deliberately NOT
        # here: how many frames the probe read and which backend read them are
        # both deterministic, and excusing them would let a probe that silently
        # stopped reading pass this test.
        timings = ("elapsed_s", "probe_s", "probe_extract_s", "probe_ocr_s",
                   "probe_ocr_s_per_frame", "contact_sheets")
        check("S5 two runs of one video agree on every reported number",
              {k: v for k, v in a.items() if k not in timings}
              == {k: v for k, v in r2.items() if k not in timings},
              "the sampler is not deterministic")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_sampling_budget_is_whole_process() -> None:
    """`--frames N` is N frames, not N per pass.

    The old code passed `budget` to every ffmpeg invocation, so a run could
    write twice what it was asked for and nothing said so. This checks the
    ceiling holds on a real video with the refinement pass actually firing.
    """
    d = Path(tempfile.mkdtemp(prefix="yt-budget-"))
    try:
        v = _synth(d, "mixed", [(_CARD, 40.0), (_THIRD, 40.0), (_OTHER, 40.0)])
        snips = [{"start": float(t), "dur": 3.0, "text": "as you can see this diagram"}
                 for t in range(40, 80, 3)]
        segs = [{"start": 0.0, "end_s": 60.0, "title": "A"},
                {"start": 60.0, "end_s": 120.0, "title": "B"}]
        for budget in (4, 9, 17):
            res = Y.extract_frames(v, d / f"out{budget}", 120.0, budget,
                                   snippets=snips, segments=segs)
            check(f"S6 budget {budget} is a ceiling on the whole process",
                  res["frames"] <= budget, f"wrote {res['frames']}")
            check(f"S6 budget {budget}: the plan respected it too",
                  res["planned_frames"] <= budget, str(res["planned_frames"]))
            check(f"S6 budget {budget}: the run recorded the policy version",
                  res["sampling_version"] == Y.SAMPLING_VERSION, str(res["sampling_version"]))
            check(f"S6 budget {budget}: the plan still reached the end",
                  res["planned_coverage_end_s"] == 120.0,
                  f"last planned frame at {res['planned_coverage_end_s']}")
            check(f"S6 budget {budget}: the tail frame is on disk",
                  any(res["coverage_end_s"] is not None and res["coverage_end_s"] >= 110.0
                      for _ in [0]) or res["near_duplicates_dropped"] > 0,
                  f"last frame at {res['coverage_end_s']}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_stale_frame_set_is_refused() -> None:
    """A frame set from another sampling policy must not be read back.

    The frames are real images with real timestamps, so every layer above this
    one would build a perfectly valid document about a video sampled by a policy
    that no longer runs — and every count in it would be a measurement of the
    old sampler. Nothing downstream can detect it; only the stamp can.
    """
    import build_evidence as B
    d = Path(tempfile.mkdtemp(prefix="yt-stale-"))
    try:
        frames = [{"frame_id": "u0001", "file": "frames/u0001.jpg", "kind": "uniform",
                   "t_start": 12.0}]
        (d / "frames.json").write_text(json.dumps(
            {"sampling_version": Y.SAMPLING_VERSION - 1, "frames": frames}))
        try:
            B.load_frames_index(d)
            check("S7 a frame set from another policy is refused", False, "it was read")
        except Exception as exc:
            check("S7 a frame set from another policy is refused", True)
            check("S7 the refusal says what to do about it",
                  "yt_fetch.py" in str(exc), str(exc)[:160])

        (d / "frames.json").write_text(json.dumps(
            {"sampling_version": Y.SAMPLING_VERSION, "frames": frames}))
        got = B.load_frames_index(d)
        check("S7 the current policy's frame set reads normally", len(got) == 1, str(got))
        check("S7 the policy version is a cache key, not a comment",
              Y.SAMPLING_VERSION != 1 and "SAMPLING_VERSION" in
              (Path(Y.__file__).parent / "versions.py").read_text(),
              "a version that never moves invalidates nothing")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    print("yt-report self-test (offline)\n")
    print("pure functions")
    t_url_parsing(); t_timestamps(); t_chapter_fallback(); t_frame_budget()
    t_sampling_profiles()
    t_sampling_text_probe_profiles()
    t_sampling_metrics_are_reported()
    t_sampling_kinds_survive_the_evidence_gate()
    t_sampling_coverage_and_budget()
    t_sampling_is_deterministic()
    t_sampling_does_not_starve_the_tail()
    t_sampling_budget_is_whole_process()
    t_stale_frame_set_is_refused()
    t_multilingual_segmentation()
    with tempfile.TemporaryDirectory() as tmp:
        t_asr_evidence_round_trip(Path(tmp))
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
    t_sampling_on_synthetic_videos()
    print("evidence layer (schema, provenance, validation)")
    t_evidence_valid(); t_evidence_malformed(); t_evidence_provenance()
    t_evidence_confidence_and_time(); t_evidence_ids(); t_evidence_enums()
    t_evidence_verified_verbatim(); t_evidence_segments(); t_evidence_unmeasured()
    t_evidence_shapes(); t_evidence_conflicts(); t_evidence_anti_vacuity()
    t_evidence_build(); t_evidence_real_shapes()
    print("ocr layer (fixtures, provenance, validation rules)")
    with tempfile.TemporaryDirectory() as tmp:
        t_ocr_fixtures(Path(tmp))
    t_ocr_result_invariants()
    t_ocr_normalisation()
    t_ocr_dedup()
    tmpfile = Path(tempfile.mkdtemp())
    try:
        t_ocr_cache(tmpfile)
    finally:
        shutil.rmtree(tmpfile, ignore_errors=True)
    t_ocr_evidence()
    t_ocr_validation()
    t_ocr_failure_modes()
    print("ocr cost and dedupe coverage (attributable timings, no lost timeline)")
    SOP.t_one_call_per_uncached_frame()
    SOP.t_cache_hit_avoids_the_provider()
    SOP.t_identical_bytes_are_one_cache_entry()
    SOP.t_rereads_are_batched()
    SOP.t_ocr_cache_key_boundaries()
    SOP.t_sampling_policy_is_not_part_of_the_ocr_key()
    SOP.t_timing_fields_are_complete()
    SOP.t_probe_is_batched_and_disposable()
    SOP.t_probe_metrics_reach_the_sampling_report()
    SOP.t_dedupe_keeps_the_end_of_a_run()
    SOP.t_dedupe_keeps_run_tail_is_optional()
    SOP.t_dedupe_coverage_accounting()
    SOP.t_coverage_ignores_unmeasured_frames()
    SOP.t_profile_branches_from_synthetic_reads()
    SOP.t_probe_below_the_char_floor_does_not_decide()
    SOP.t_expression_evidence_is_preserved()
    SOP.t_unitless_number_does_not_invent_a_subject()
    SOP.t_probe_results_are_not_final_evidence()
    print("alignment")
    t_alignment()
    t_alignment_validation()
    t_alignment_misses_are_measured()
    print("semantic extraction")
    t_semantic_numbers()
    t_semantic_screen()
    t_semantic_structure()
    t_vlm_and_provenance()
    print("vision-language pass (offline fixtures, no network)")
    t_vlm_observation_schema()
    with tempfile.TemporaryDirectory() as tmp:
        vd = Path(tmp)
        t_vlm_fixture_flows(vd)
        t_vlm_failures(vd)
        t_vlm_cache(vd)
        t_vlm_selection(vd)
        t_vlm_crossmodal(vd)
    print("vlm providers (configuration, negotiation, every refusal — no network)")
    SP.t_vlm_config()
    SP.t_vlm_provider_selection()
    with tempfile.TemporaryDirectory() as tmp:
        vtmp = Path(tmp)
        SP.t_vlm_structured_negotiation(vtmp)
        SP.t_vlm_response_validation(vtmp)
        SP.t_vlm_transport_failures(vtmp)
        SP.t_vlm_media_rejection_is_not_a_mode_rejection()
        SP.t_vlm_secret_hygiene(vtmp)
        SP.t_vlm_smoke_test(vtmp)
    print("report layer (generation, grounding, golden structure)")
    SR.t_report_fixtures()
    SR.t_report_shape()
    SR.t_report_degradation()
    SR.t_report_uncertainty()
    SR.t_report_noise()
    SR.t_report_repetition()
    SR.t_report_corruption()
    SR.t_report_determinism()
    SR.t_report_budget()
    SR.t_report_golden()
    SR.t_report_provenance_tags()
    SR.t_report_first_person()
    SR.t_figure_reliability_consistency()
    SR.t_numeric_metrics_exclude_timestamps()
    SR.t_takeaways_are_insights()
    SR.t_exec_summary_states_the_central_argument()
    SR.t_conflict_renders_both_sides()
    with tempfile.TemporaryDirectory() as tmp:
        SR.t_report_writes(Path(tmp))
    print("cache contract (a key that changes when the answer would)")
    with tempfile.TemporaryDirectory() as tmp:
        SC.t_vlm_key_dimensions(Path(tmp))
    SC.t_synth_key_dimensions()
    SC.t_failures_are_not_cached_as_success()
    SC.t_secrets_never_reach_a_key()
    SC.t_corrupt_entry_is_a_miss()
    SC.t_ordering_matters()
    SC.t_semantic_key_covers_frame_time()
    SC.t_extraction_version_invalidates_the_cache()
    SC.t_cache_matrix_is_complete()
    SC.t_versions_live_in_one_place()
    with tempfile.TemporaryDirectory() as tmp:
        SC.t_context_ordering_is_normalised(Path(tmp))
    print("grounding gate (fourteen deliberate corruptions)")
    SG.t_unsupported_factual_claim()
    SG.t_unsupported_number()
    SG.t_unsupported_url()
    SG.t_unsupported_visual_statement()
    SG.t_invalid_timestamp()
    SG.t_nonexistent_evidence_id()
    SG.t_dropped_unresolved_conflict()
    SG.t_removed_hedge()
    SG.t_duplicated_factual_section()
    SG.t_excessive_repetition()
    SG.t_fabricated_entity()
    SG.t_unsupported_comparison()
    SG.t_unsupported_procedural_step()
    SG.t_malformed_synthesis_json()
    SG.t_gate_blocks_the_write()
    SG.t_evidence_gate_stops_before_the_report()
    SG.t_clean_evidence_is_not_repaired()
    print("synthesis layer (schema, grounding, pipeline, fallbacks — no network)")
    SS.t_synth_schema()
    SS.t_synth_grounding_happy()
    SS.t_synth_rejects_unsupported_entity()
    SS.t_synth_rejects_wrong_number()
    SS.t_synth_rejects_removed_hedge()
    SS.t_synth_conflict_validation()
    SS.t_synth_rejects_unsupported_visual()
    SS.t_synth_rejects_unknown_evidence_id()
    SS.t_synth_rejects_contradicting_status()
    SS.t_synth_source_kind()
    SS.t_synth_pipeline()
    SS.t_synth_fallback_ladder()
    SS.t_synth_fallback_when_grounding_fails()
    SS.t_synth_cache()
    SS.t_synth_critic_and_revision()
    SS.t_synth_output_language()
    SS.t_prompt_states_the_contract_the_gate_enforces()
    SS.t_one_home_matrix()
    SS.t_conflict_completeness_is_a_report_level_rule()
    SS.t_output_language_requires_a_writer()
    SS.t_evidence_stays_in_its_source_language()
    SS.t_synth_malformed_draft_reaches_the_gate()
    SS.t_synth_rejects_an_unsupported_url()
    SS.t_synth_fixture_coverage()
    SS.t_vlm_schema_is_strict_mode_valid()
    SS.t_vlm_error_text_reads_through_a_gateway()
    SS.t_vlm_credential_is_scoped_to_its_provider()
    print("number relevance (what reaches the numbers section, and what does not)")
    SN.t_ui_numbers_are_rejected()
    SN.t_benchmark_price_and_count_are_retained()
    SN.t_confidence_is_not_importance()
    SN.t_unit_without_a_label_is_not_a_fact()
    SN.t_isolated_number_is_suppressed()
    SN.t_raw_observation_survives()
    SN.t_role_survives_the_round_trip()
    SN.t_spoken_figure_subject()
    SN.t_subject_is_a_referent_not_an_adjacent_token()
    SN.t_subject_does_not_borrow_from_the_previous_sentence_implicitly()
    SN.t_classification_matrix()
    SN.t_gutter_and_table_are_told_apart_structurally()
    SN.t_subject_is_null_rather_than_wrong()
    SN.t_the_gate_finds_the_referent_and_skips_the_word()
    SN.t_the_gate_does_not_over_reach()
    SN.t_ui_verdict_is_pinned_not_merely_low()
    SN.t_equation_is_read_as_one_expression()
    SN.t_expression_shape_refuses_a_misread_subscript()
    SN.t_trailing_label_recovers_the_other_side()
    SN.t_glyphs_are_not_referents()
    SN.t_no_fabricated_visual_number()
    print("entities (seven languages, salience, and the generic type)")
    SE.t_german_ordinary_words_are_not_entities()
    SE.t_german_real_names_survive()
    SE.t_german_lemma_matching()
    SE.t_weak_detection_does_not_infect()
    SE.t_english_extraction_unchanged()
    SE.t_hindi_fixture()
    SE.t_uncategorised_name_is_generic()
    SE.t_salience_requires_context()
    SE.t_ordinary_words_stay_out_of_subjects()
    SE.t_every_supported_language_has_a_resource()
    SE.t_quantity_phrases_are_not_names()
    SE.t_every_language_in_one_table()
    SE.t_ordinary_vocabulary_never_becomes_a_category()
    SE.t_typing_does_not_spread_across_a_sentence()
    SE.t_observed_german_false_positives_are_listed()
    SE.t_short_all_caps_categories_are_not_names()
    SE.t_single_sighting_ordinary_word_does_not_become_an_item()
    print("doctors, budgets, and the VLM path (no network, no credential)")
    SD.t_http_error_taxonomy()
    SD.t_synthesis_doctor_distinguishes_failures()
    SD.t_doctor_does_not_claim_success_on_http_200()
    SD.t_config_failures_make_no_call()
    SD.t_critique_shape_is_validated()
    SD.t_critic_cases_have_known_answers()
    SD.t_critic_doctor_scores_each_case()
    SD.t_critic_schema_gate_is_one_verdict()
    SD.t_critic_budget_never_exceeded()
    SD.t_budget_is_enforced_not_reported()
    SD.t_failed_response_is_not_cached_as_success()
    SD.t_vlm_observation_reaches_evidence_and_report()
    SD.t_non_conforming_vlm_output_is_rejected()
    SD.t_release_check_four_states()
    SD.t_release_check_matrix()
    SD.t_release_check_budget_is_a_ceiling()
    SD.t_release_check_never_prints_a_credential()
    SD.t_release_check_without_providers_calls_nothing()
    SD.t_vlm_doctor_survives_an_unclassified_exception()
    SD.t_max_vlm_frames_reaches_the_doctor()
    SD.t_connectivity_and_end_to_end_reach_the_verdict()
    print("skill prose invariants")
    t_skill_text(d)
    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        if SKIPPED:
            print(f"skipped {len(SKIPPED)} (proved nothing): {', '.join(SKIPPED)}")
        return 1
    if SKIPPED:
        print(f"all checks passed — {len(SKIPPED)} skipped, which proves nothing: {', '.join(SKIPPED)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
