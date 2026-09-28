#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["youtube-transcript-api>=1.0", "yt-dlp>=2025.1.1", "pillow>=10.0", "numpy>=1.26"]
# ///
"""Fetch everything text-extractable from a YouTube video into a folder.

    uv run yt_fetch.py <url> [--out DIR] [--whisper] [--lang en]

Writes DIR/transcript.md (timestamped), DIR/meta.json, DIR/URL.txt.
Exit 2 with a "no transcript" note when captions are unavailable and
--whisper was not passed (the caller then decides whether to pay for ASR).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import os
import tempfile
import time
from pathlib import Path

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript

from versions import SAMPLING_VERSION


def parse_video_id(url_or_id: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/|/shorts/)([0-9A-Za-z_-]{11})", url_or_id)
    return m.group(1) if m else (url_or_id if re.fullmatch(r"[0-9A-Za-z_-]{11}", url_or_id) else None)


def ts(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _snippets(fetched) -> list[dict]:
    return [{"start": s.start, "dur": s.duration, "text": s.text.replace("\n", " ")} for s in fetched]


def fetch_transcript(video_id: str, lang: str) -> tuple[str, list[dict], str]:
    """Return (text, snippets, lang_used). English (or --lang) tracks only —
    never a silent fallback to another language, which would read as English
    while being anything but. Manual captions beat auto within the language."""
    api = YouTubeTranscriptApi()
    try:
        available = list(api.list(video_id))  # one-shot iterable: materialize before filtering twice
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"no caption tracks: {e}") from e
    tracks = [t for t in available if t.language_code == lang or t.language_code.startswith(lang + "-")]
    if not tracks:
        others = ", ".join(sorted({t.language_code for t in available})) or "none"
        raise RuntimeError(
            f"no {lang} captions on this video (available: {others}). "
            f"Use --whisper to transcribe the {lang} audio locally instead."
        )
    tracks.sort(key=lambda t: (t.is_generated, t.language_code))  # human first, then en over en-US
    last_err: Exception | None = None
    for track in tracks:
        try:
            fetched = api.fetch(video_id, languages=[track.language_code])
        except Exception as e:  # noqa: BLE001, PERF203 — one bad track must not kill the loop
            last_err = e
            continue
        tag = "auto" if track.is_generated else "human"
        snippets = _snippets(fetched)
        return " ".join(s["text"] for s in snippets), snippets, f"{track.language_code} ({tag})"
    raise RuntimeError(f"{len(tracks)} {lang} caption track(s) listed but none could be fetched: {str(last_err)[:200]}")


def download_video(url: str, tmpdir: Path) -> Path:
    """One 720p-with-audio download, reused by both ASR and frame extraction.

    Frames alone need only the video track and ASR only the audio track, but
    downloading twice costs the user the whole file twice — the 720p file is small
    enough that one fetch serving both is the cheaper default.
    """
    # Cap resolution: 720p is legible enough for slide text and keeps a 1-hour
    # download to a few hundred MB, versus ~2 GB at 1080p for no extra signal.
    fmt = "bestvideo[height<=720]+bestaudio/best[height<=720]/best"
    opts = {
        "quiet": True, "no_warnings": True, "noprogress": True, "format": fmt,
        "merge_output_format": "mp4",
        # No extension in the template plus a forced merge format: otherwise yt-dlp
        # writes .webm beside a .mp4 template and every downstream ffmpeg call
        # silently gets a path that does not exist.
        "outtmpl": str(tmpdir / "v.%(ext)s"),
    }
    # Transient network failures are common enough to be the single most likely
    # cause of a run producing no frames AND no ASR — one blip takes out both
    # halves at once, because both read this one file. Retry it, and clear any
    # partial output between attempts so a truncated download can never be
    # mistaken for a complete one.
    last: Exception | None = None
    for attempt in range(3):
        if attempt:
            time.sleep(2 * attempt)
        for stale in tmpdir.iterdir():
            if stale.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".part", ".ytdl"):
                stale.unlink(missing_ok=True)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as exc:  # noqa: BLE001 - yt-dlp raises a wide range; retry all
            last = exc
            continue
        videos = [p for p in tmpdir.iterdir() if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")]
        if videos:
            return max(videos, key=lambda p: p.stat().st_size)
        last = RuntimeError("download produced no video file")
    raise RuntimeError(f"download failed after 3 attempts: {last}")


def whisper_transcript(video: Path, lang: str) -> tuple[str, list[dict], str]:
    """Transcribe the audio track locally with mlx-whisper. No API, nothing uploaded."""
    import mlx_whisper  # noqa: PLC0415  (optional dep, only imported on the ASR path)

    audio = video.with_suffix(".wav")
    # 16 kHz mono is what whisper was trained on; anything else is resampled by it anyway.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(audio)],
        check=True, capture_output=True,
    )
    # large-v3-turbo: ~4x faster than large-v3, near-identical WER on clean speech.
    out = mlx_whisper.transcribe(str(audio), path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language=lang)
    snippets = [{"start": s["start"], "dur": s["end"] - s["start"], "text": s["text"].strip()} for s in out["segments"]]
    audio.unlink(missing_ok=True)
    return " ".join(s["text"] for s in snippets), snippets, f"{lang} (ASR: whisper large-v3-turbo)"


def fetch_meta(url: str) -> dict:
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noprogress": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "id": info.get("id"),
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "channel_url": info.get("channel_url") or info.get("uploader_url"),
        "upload_date": info.get("upload_date"),
        "duration": info.get("duration"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "description": info.get("description") or "",
        "tags": info.get("tags") or [],
        "categories": info.get("categories") or [],
        "chapters": info.get("chapters") or [],
        "chapters_available": bool(info.get("chapters")),
        "webpage_url": info.get("webpage_url") or url,
        "thumbnail": info.get("thumbnail"),
    }


def segments_from_chapters(snippets: list[dict], chapters: list[dict], duration: float | None) -> list[dict]:
    """Anchor transcript lines to chapters; otherwise cut where the talk turns.

    Two fallbacks used to sit here and both were worse than the thing they
    replaced. Fixed 5-minute buckets gave a 5.5-minute roundup two sections.
    Scaled blocks fixed that and left a German video cut into `Part 1` … `Part 5`
    — five sections that name nothing, so a reader who opens the report to learn
    what the video is about is told it is about "Part 3". A block has no idea
    where a topic ends; the transcript does.

    So the no-chapters path cuts at *topic boundaries*: the snippet transitions
    where the vocabulary on either side of the gap stops overlapping, which is
    the same signal a reader uses when they say "he moves on here", and which
    needs no language-specific model to work in German, Spanish or Hindi. Each
    section is then titled from its own first sentence. Uploader chapters still
    win when they exist, because a creator's own sectioning is information this
    stage does not have.
    """
    if chapters:
        bounds = [(c["start_time"], c["title"]) for c in chapters]
    else:
        total = duration or (snippets[-1]["start"] + 5 if snippets else 300)
        target = int(max(4, min(12, round((total / 60) / 3))))
        cuts = _topic_boundaries(snippets, target)
        # The last entry is the end of the transcript, not a section start.
        bounds = [(start, _segment_title(snippets, start, _next_cut(cuts, n)) or f"Part {n + 1}")
                  for n, start in enumerate(cuts[:-1])]
    out = []
    for i, (start, title) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else float("inf")
        body = [s for s in snippets if start <= s["start"] < end]
        # end_s is the segment's own upper bound, clamped to something real: a
        # segment running to infinity cannot be checked against a video duration,
        # and `float("inf")` is not valid JSON — it would serialise as `Infinity`
        # and be unreadable by any strict parser. The last segment ends where the
        # speech does, or at the video duration, whichever is known.
        tail = end
        if tail == float("inf"):
            tail = (snippets[-1]["start"] + snippets[-1].get("dur", 0.0)) if snippets else start
            if duration:
                tail = min(tail, float(duration))
        out.append({"start": start, "end_s": tail, "title": title,
                    "chars": sum(len(s["text"]) for s in body), "snippets": body})
    # A block that captured no speech is a gap in the video, not a section of it.
    return [s for s in out if s["snippets"]]


#: A caption blob longer than this is cut into sentences before anything else
#: looks at it. Auto-generated tracks — the German one that produced the
#: `Part 1`…`Part 5` report especially — arrive as multi-sentence blocks of
#: arbitrary length, and a segment built from them is a wall of text that no
#: downstream rule can cut, because by then the timestamps are gone.
BLOB_CHARS = 320

#: Sentence terminators across the languages this pipeline reads. `。` and `؟` are
#: there because a CJK or Arabic caption is one long run without `.` and would
#: otherwise be a single unit for the whole video.
_SENTENCE_END = re.compile(r"(?<=[.!?…؟。])[\s ]+")


def _next_cut(cuts: list[float], n: int) -> float:
    """The start of the section after `cuts[n]`, or the end of the transcript."""
    return cuts[n + 1] if n + 1 < len(cuts) else float("inf")


def _vocab(text: str) -> set[str]:
    return {w for w in re.findall(r"[^\W\d_]{3,}", (text or "").lower())}


def _topic_boundaries(snippets: list[dict], target: int) -> list[float]:
    """Snippet starts where the talk most likely changes subject.

    Scored, not thresholded against a fixed number: each candidate is a change
    in the words on either side of it, and the strongest `target − 1` are taken
    with a minimum gap so the sections stay readable. `0` is always a boundary
    because the transcript starts there. No text is dropped and none is
    duplicated — this returns times, and the caller re-binds every snippet.
    """
    if len(snippets) < 3:
        if not snippets:
            return []
        # A short transcript has no interior to cut on, but it still has an end:
        # the sentinel is what makes the last section reach the end of the
        # speech, and dropping it leaves the caller with no sections at all and
        # every snippet belonging to none.
        return [float(snippets[0]["start"]),
                float(snippets[-1]["start"] + (snippets[-1].get("dur") or 0.0)) + 1e-6]
    window = max(2, min(6, len(snippets) // max(2, target)))
    scored: list[tuple[float, float]] = []
    for i in range(window, len(snippets) - window):
        before, after = _vocab(" ".join(s["text"] for s in snippets[i - window:i])), \
            _vocab(" ".join(s["text"] for s in snippets[i:i + window]))
        overlap = len(before & after) / max(1, len(before | after))
        scored.append((overlap, float(snippets[i]["start"])))
    scored.sort()
    span = (snippets[-1]["start"] + (snippets[-1].get("dur") or 0.0)) - snippets[0]["start"]
    min_gap = max(30.0, span / (target * 1.5))
    cuts = [float(snippets[0]["start"])]
    for _score, start in scored:
        if len(cuts) >= target:
            break
        if all(abs(start - c) >= min_gap for c in cuts):
            cuts.append(start)
    cuts.append(float(snippets[-1]["start"] + (snippets[-1].get("dur") or 0.0)) + 1e-6)
    return sorted(set(cuts))


def _segment_title(snippets: list[dict], start: float, end: float) -> str:
    """The section's own first sentence, as its title.

    Taken from the transcript rather than composed, so a German section is titled
    in German and every word in it is a word the video said. Cut at the first
    sentence boundary and then at the word cap, so a title stays a label.
    """
    body = " ".join(s["text"] for s in snippets if start <= s["start"] < end).strip()
    if not body:
        return ""
    head = _SENTENCE_END.split(body)[0].strip()
    words = head.split()
    return (" ".join(words[:11]) + ("…" if len(words) > 11 else "")) if words else ""


def split_long_snippets(snippets: list[dict], cap: int = BLOB_CHARS) -> list[dict]:
    """Cut caption blobs into sentence-level units, keeping every timestamp.

    The timestamps are interpolated across the blob in proportion to character
    offset, so a split is a claim about *when* something was said that the
    caption track supports: the words are in order, the times are ordered, and
    the pieces tile the original interval exactly. That is what makes coverage
    checkable — the parts start where the previous part ended and end where the
    original ended, so `sum(dur)` is unchanged and no text is duplicated or
    lost. A blob already under the cap is returned untouched.
    """
    out: list[dict] = []
    for s in snippets:
        text = (s.get("text") or "").strip()
        if len(text) <= cap:
            out.append(s)
            continue
        pieces = _SENTENCE_END.split(text)
        # A caption track that never terminates a sentence is one unit; fall
        # back to clause boundaries so it is still usable rather than a wall.
        if len(pieces) == 1:
            pieces = re.split(r"(?<=[,;:])\s+", text)
        offset, total = 0, len(text) or 1
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            at = offset / total
            offset += len(piece) + 1
            start = float(s["start"]) + at * float(s.get("dur") or 0.0)
            out.append({"start": round(start, 3),
                        "dur": round(float(s.get("dur") or 0.0) / max(1, len(pieces)), 3),
                        "text": piece})
    return out


def _frame_times(stderr: str | None, budget: int) -> list[float]:
    """Presentation timestamps, in seconds, of the frames written by the same
    ffmpeg filter chain. Parsed from the extraction pass's own showinfo output, so
    a cited frame carries a real timestamp instead of an estimate derived from its
    position in a sorted list. The alternative — a second pass that only decodes to
    print timestamps — costs a full extra decode of the video per sampler."""
    return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", stderr or "")[:budget]]


#: Mean absolute difference between two 16x16 grey thumbnails below which the
#: video is judged *unchanged* between two samples. The same measure is used by
#: `dedupe_frames` (to drop redundant frames) and by `change_intervals` (to find
#: where the picture changed) — one measure, two questions, so they cannot drift
#: apart and disagree about what "the same frame" means.
DEDUPE_THRESHOLD = 3.0

#: What a 16x16 grey reduction can and cannot see. Measured over the 126 frames
#: of a real 18-minute lecture, comparing every pair of frames within 16s of each
#: other: of the 218 pairs whose on-screen text differs, a 16x16 grey delta under
#: 3.0 misses 29 of them. Raising the resolution to 64x64 does not help (it
#: misses 27), and neither does a contrast or edge-energy map — both amplify
#: compression noise and score *worse* (28 and 94 misses respectively). Text
#: occupies a handful of pixels in each cell and a 16-cell average absorbs it.
#:
#: So the threshold is left where it is and the loss is bounded structurally
#: instead: `dedupe_frames` never removes the LAST frame of a run, so a stretch
#: that holds one image for a minute keeps the image at both ends of it. The
#: failure this prevents is not "a frame looks similar" — it is "the end of the
#: stretch, which is where the new text appeared, is the one that got deleted".
DEDUPE_UNSEEN_TEXT_RATE = (29, 218)


def dedupe_frames(frames: list[Path], threshold: float = DEDUPE_THRESHOLD,
                  keep_run_tail: bool = True) -> tuple[list[Path], int]:
    """Drop near-identical frames, keeping the first of each run and its last.

    A video that holds a full-text card or a static slide for forty seconds
    yields several identical samples; they consume the contact sheet — the one
    image the reader triages from — without adding information. Comparing each
    frame to the last KEPT frame at 16x16 grey collapses those runs.

    Measured: on a static text-card video this drops ~44% of frames with no loss
    of distinct content; on a motion video (talking head, animation) it drops
    nothing, because genuinely moving frames differ by more than the threshold.
    That asymmetry is the point — it is a no-op where it should be.

    `keep_run_tail` retains the last frame of every run of two or more
    duplicates, and it is the fix for a measured defect rather than a tuning
    preference. Keeping only the FIRST frame of a run means the frame at the end
    of a static stretch — the one where the presenter has finished speaking and
    has drawn the next thing on the slide — is always the one discarded. On a
    real 18-minute lecture that produced a 48.7-second hole in an 18.7-minute
    timeline where the plan had promised 12.2 seconds: four consecutive baseline
    steps, all deleted, and the change that ended the stretch never sampled.
    Keeping the tail halves such a hole without keeping a single interior
    duplicate, and costs no extra decode.

    Thumbnails are taken once up front rather than lazily. The run-length rule
    needs to know what comes *after* a candidate, and re-reading each image to
    find out would decode the whole set twice for a comparison that is cheaper
    than the decoding: 16x16 greys for a full video's frames are a few hundred
    kilobytes.
    """
    thumbs = [_thumb(f) for f in frames]
    n = len(frames)
    kept: list[Path] = []
    last = None
    i = 0
    while i < n:
        arr = thumbs[i]
        if arr is None or last is None or _delta(arr, last) >= threshold:
            # Unreadable, or genuinely different from the last kept frame.
            kept.append(frames[i])
            if arr is not None:
                last = arr
            i += 1
            continue
        # A duplicate. Find where the run of identical pictures ends.
        j = i
        while (j + 1 < n and thumbs[j + 1] is not None
               and _delta(thumbs[j + 1], last) < threshold):
            j += 1
        if keep_run_tail and j > i:
            kept.append(frames[j])  # the end of the stretch, not its middle
        for k in range(i, j + 1):
            if k == j and keep_run_tail and j > i:
                continue
            frames[k].unlink(missing_ok=True)
        i = j + 1
    return kept, len(frames) - len(kept)


#: Mean absolute difference between two 16x16 grey thumbnails above which the
#: video is judged to have *changed* between two baseline samples. The same
#: measure `dedupe_frames` already computes, reused rather than a second one.
CHANGE_THRESHOLD = 8.0

def _thumb(f: Path):
    """A 16x16 grey reduction of one frame, or None when it cannot be read."""
    try:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        return np.asarray(Image.open(f).convert("L").resize((16, 16)), dtype=np.float32)
    except Exception:  # noqa: BLE001 — an unreadable frame is kept, not lost
        return None


def _delta(a, b) -> float:
    """Mean absolute difference between two thumbnails, 0-255."""
    import numpy as np  # noqa: PLC0415
    return float(np.abs(a - b).mean())


def change_intervals(frames: list[Path], times: dict[str, float],
                     threshold: float = CHANGE_THRESHOLD, pad: float = 4.0) -> list[dict]:
    """Where the picture actually changed between two baseline samples.

    Measured on the frames the baseline pass already wrote, using the same 16x16
    grey reduction `dedupe_frames` uses — so this costs no extra decode and
    answers the question the scene filter cannot.

    A scene score compares a frame to the *previous frame*. At 12-30 fps that
    difference is tiny: on a synthetic test pattern moving continuously, one
    frame in twenty scored above 0.02 over twenty seconds, and a hard cut scored
    0.37. So a scene filter finds cuts and nothing else, and a screen recording
    or a maths animation — where everything moves slowly and continuously — looks
    to it like a static slide. Comparing samples ten seconds apart measures the
    thing that matters: does this interval contain something a reader has not
    already been shown.
    """
    out: list[dict] = []
    prev_thumb = prev_time = None
    for f in frames:
        t = times.get(f.name)
        thumb = _thumb(f)
        if thumb is None or t is None:
            continue
        if prev_thumb is not None and _delta(thumb, prev_thumb) >= threshold:
            out.append({"start": max(0.0, prev_time - pad), "end": t + pad,
                        "weight": 1, "reason": "the picture changed here"})
        prev_thumb, prev_time = thumb, t
    return out


# --------------------------------------------------------------------------- #
# frame sampling policy
# --------------------------------------------------------------------------- #
#
# One budget, spent in three layers, planned before a single frame is decoded:
#
#   uniform baseline  → every part of the video is covered, whatever else happens
#   hint refinement   → intervals the speaker pointed at
#   change refinement → intervals the video itself showed as changing
#
# The baseline is computed first and is never reduced by anything the other two
# layers want. That ordering is the whole fix: the previous sampler let a scene
# filter and a uniform grid each draw the full budget independently, so a
# montage-heavy opening could take everything and a fixed grid took whatever it
# had left, and on a real maths animation the grid landed one frame every 30
# seconds — a window in which a sampled frame exists 3% of the time.

#: The profiles the sampler can tell apart, and the uniform spacing each one
#: gets. Starting heuristics, not measurements: the point of the table is that
#: the numbers are in one place and can be argued with.
#:
#: **Five rows, and two of them are separated by what is *written*, not by what
#: was cut.** A scene score compares a frame to the *previous* frame, so at
#: 12-30 fps a sliding window and a mouse cursor score near zero while a hard cut
#: scores 0.37; on a synthetic pattern moving continuously, one frame in twenty
#: cleared 0.02 over twenty seconds. A screen recording, a code editor and a
#: slide deck are therefore indistinguishable to it — all three read as "static".
#: The classes are separated by the words on the screen, which is `OcrProfile`'s
#: job, and the reason it is affordable is that it reads a dozen small probe
#: frames rather than the whole grid: a coarse read is enough to tell a terminal
#: from a presenter, and the fine read still happens once, on the frames the plan
#: actually chose.
#:
#: What a *talking head* and a *montage* do separate on is exactly the axis the
#: scene score measures, which is why cuts remain the primary signal and OCR
#: only ever refines it.
VISUAL_PROFILES = {
    "talking_head": 30.0,
    "lecture": 12.0,
    "screen_ui": 8.0,
    "code": 6.0,
    "dynamic": 15.0,
}

#: How many small frames the pre-sampling OCR read looks at. Enough to tell a
#: screen recording from a talking head, few enough that the read costs a
#: fraction of the main pass. Fixed, not scaled with duration: the answer this
#: read exists to give is a *coarse* profile, and a probe whose own spacing
#: changes with the video is one more thing that has to be deterministic.
OCR_PROBE_FRAMES = 16

#: The width the probe frames are reduced to. Small on purpose — this read only
#: has to recover "is there a lot of text, and is it code", and a 480px frame
#: holds roughly what a slide's body text needs at a fraction of the OCR time.
OCR_PROBE_WIDTH = 480

#: How much text the probe has to have found before it is allowed to decide the
#: profile. A *ratio* computed over four lines is not evidence of anything: on a
#: synthetic test pattern the burned-in timecode is three lines and two of them
#: match a code shape, which classified a moving test card as source code and
#: sampled it at 6s. Roughly a screen's worth of characters is the floor below
#: which the sampler falls back to the cut count and says so.
OCR_MIN_CHARS = 200

#: And how many code-shaped lines, over the whole probe, before the video is
#: called `code`. A code editor fills its window; a slide with one formula does
#: not. One line of `x = 1` is a maths animation, not a terminal.
OCR_MIN_CODE_LINES = 8

#: The most frames a duration-derived budget will ever ask for. Beyond this the
#: contact sheet stops being something a person reads and the OCR pass starts
#: costing more than the report is worth. It is a ceiling on the ceiling: a
#: caller passing `--frames` explicitly can still ask for more.
MAX_AUTO_FRAMES = 300

#: Scene score above which ffmpeg calls a frame a cut. Lower than the 0.30 the
#: sampler used before, because the old value found transitions and nothing
#: else; a slide advancing is a change worth a second look even though no human
#: would call it a scene.
SCENE_PROBE = 0.10

#: The layer a frame came from, by the prefix its filename carries. Every value
#: here is written into `frames.json` and read back as `visual[].kind`, so the
#: set has to be a subset of `evidence_model.FRAME_KINDS` or every run that
#: samples a refinement frame is refused by the evidence gate. It named
#: `refinement` for two policy versions before the schema learned the word, and
#: the suite did not catch it because no synthetic clip in it had a refinement
#: frame that survived to a validated document. The explicit tail frame (`z`) is
#: deliberately absent: it has no name of its own and falls through to
#: `unknown`, which is the honest answer for a frame the layers do not claim.
FRAME_KIND_BY_PREFIX = {"u": "uniform", "r": "refinement", "s": "scene"}

#: How the total budget is divided before any frame is written. The baseline
#: share is fixed, so refinement can never eat into coverage; the other two
#: split what is left, and the transcript hints win the larger share because
#: "as you can see" is the speaker telling us where to look.
BUDGET_SPLIT = {"baseline": 0.55, "hint": 0.25, "change": 0.20}

#: Phrases where a speaker is pointing at something. A hint and nothing more:
#: "look at this" is often followed by nothing but a talking head, and the
#: baseline layer is what stops a hint from turning that into a claim.
#: Deliberately short. A long keyword list is a maintenance liability and a
#: false-positive source, and it would be the only content signal in the
#: sampler — one signal among the scene and change layers, not the decision.
VISUAL_HINT_PHRASES = (
    # English
    "as you can see", "as you can see here", "on this screen", "this diagram",
    "the table", "this table", "the chart", "this chart", "shown here",
    "as displayed", "look at", "right here", "on the left", "on the right",
    "up here", "down here", "this is what", "here you can see", "this slide",
    # German — the second real artifact is German, and an English-only list
    # would have found nothing in it.
    "schauen sie", "hier sieht man", "auf diesem bildschirm", "diese tabelle",
    "diese grafik", "das diagramm", "zu sehen", "links hier", "rechts hier",
    "diese folie", "wie sie hier sehen",
)
_HINT_RE = re.compile("|".join(re.escape(p) for p in VISUAL_HINT_PHRASES), re.IGNORECASE)


#: What a *line* of on-screen text looks like when it is code rather than prose.
#: Shapes, not words: the vocabulary of programming is large, multilingual and
#: versioned, so a keyword list would be wrong everywhere it was not maintained,
#: while these shapes hold across every language and every language's libraries.
#: An opening bracket, an operator between two operands, a `::` or an arrow —
#: none of them occurs in a caption or a slide bullet.
_CODE_SHAPE = re.compile(
    r"""^\s*(?:\$|>|#|//|\.\.|[A-Za-z_][\w.]*\s*[:=]\s|[A-Za-z_]\w*\s*\(|\w+\s*[-+*/%]=\s)"""
    r"""|[{};]\s*$|[<>=!+\-*/%|&^]\s*[{<(\w]|\(\s*\)|=>|::|->|</|/>|\w+\.\w+\(|\w+\[\d+\]""")
#: A UI label: one to three short words, no sentence punctuation, not code. What
#: a button, a tab, a menu row or a field label is, and what a sentence is not.
_UI_SHAPE = re.compile(r"^[A-Za-z0-9][\w .&/+%#()-]{0,28}$")


def _line_shares(texts: list[str]) -> list[bool]:
    """How many of these lines read as code rather than as prose or a label."""
    return [bool(_CODE_SHAPE.search(t)) for t in texts]


class OcrProfile:
    """What the video *says* on screen, read coarsely before the plan is made.

    The only thing it decides is density. It never decides meaning: a screen
    recording is not thereby a source of figures, and the semantic layer remains
    the only thing allowed to conclude that. Every field is a count, and the
    `reason` records which one decided, so a profile can be argued with rather
    than merely trusted.
    """

    __slots__ = ("frames", "text_frames", "chars", "code_lines", "ui_lines", "reason")

    def __init__(self, frames: int, text_frames: int, chars: int,
                 code_lines: int, ui_lines: int, reason: str = "no signal from a coarse read"):
        self.frames, self.text_frames = frames, text_frames
        self.chars, self.code_lines, self.ui_lines = chars, code_lines, ui_lines
        self.reason = reason

    @property
    def text_frame_ratio(self) -> float:
        return (self.text_frames / self.frames) if self.frames else 0.0

    @property
    def chars_per_frame(self) -> float:
        return (self.chars / self.frames) if self.frames else 0.0

    @property
    def code_line_ratio(self) -> float:
        total = self.code_lines + self.ui_lines
        return (self.code_lines / total) if total else 0.0

    def to_dict(self) -> dict:
        return {"frames_read": self.frames, "frames_with_text": self.text_frames,
                "chars": self.chars, "chars_per_frame": round(self.chars_per_frame, 1),
                "code_lines": self.code_lines, "ui_lines": self.ui_lines,
                "code_line_ratio": round(self.code_line_ratio, 3), "reason": self.reason}

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"OcrProfile({self.text_frames}/{self.frames} frames with text, "
                f"{self.code_line_ratio:.0%} code)")


def ocr_profile(reads: list[tuple[float, str]]) -> OcrProfile:
    """Summarise a coarse pre-sampling read into the three counts that matter.

    `reads` is `(timestamp, text)` per probe frame, in time order. A frame that
    read nothing counts towards `frames` and not towards `text_frames`: "the
    sampler looked and there was nothing written" and "the sampler never looked"
    are different facts, and only the first one is a claim about the video.
    """
    lines: list[str] = []
    text_frames = chars = 0
    for _t, text in reads:
        body = (text or "").strip()
        if not body:
            continue
        text_frames += 1
        chars += len(body)
        lines.extend(ln.strip() for ln in body.splitlines() if ln.strip())
    code = sum(_line_shares(lines))
    ui = sum(1 for ln in lines
             if not _CODE_SHAPE.search(ln) and _UI_SHAPE.match(ln) and len(ln.split()) <= 3)
    return OcrProfile(len(reads), text_frames, chars, code, ui)


class VisualProfile:
    """What kind of video this is, as far as scene statistics can tell.

    Deliberately carries no meaning beyond density. It says how often to look,
    never what a frame contains — a screen recording is not thereby a source of
    figures, and the semantic layer is the only thing allowed to decide that.
    """

    __slots__ = ("name", "step_s", "reason")

    def __init__(self, name: str, step_s: float, reason: str):
        self.name, self.step_s, self.reason = name, step_s, reason

    def to_dict(self) -> dict:
        return {"name": self.name, "step_s": self.step_s, "reason": self.reason}

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"VisualProfile({self.name!r}, {self.step_s}s)"


def classify_visual_profile(scene_times: list[float], duration: float | None,
                            ocr: "OcrProfile | None" = None) -> VisualProfile:
    """Read a video's visual rhythm from its scene-change times and its text.

    Five measurements, each of which separates two cases the others cannot:

    * **how often** it cuts — a talking head almost never does, an edited
      montage constantly;
    * **how regularly** — evenly spaced cuts are a slide deck advancing;
    * **how sustained** — a long stretch of dense changes with quiet either side
      is a screen recording, which is the one profile that wants a *uniform*
      fine grid rather than scene hits, because the interesting frames are the
      ones between the changes;
    * **how much of the video changes at all** — a static card for forty seconds
      is not a slide deck however many times its title animates in;
    * **what is written on it** — this is the only measurement that separates a
      terminal from a slide deck, and both of them look identical to a scene
      score because neither one cuts.

    Order matters, and it is cuts first. A montage that also happens to carry
    captions is still a montage, and reading its text as the deciding signal
    would demote the one profile that is unambiguously right. `ocr` is a
    refinement of the quiet cases, never an override of a loud one.

    Short videos are read by count rather than rate: a 30-second clip with three
    cuts is not a montage, and dividing by the minute makes it look like one.
    """
    dur = float(duration or 0.0)
    hits = sorted(float(t) for t in scene_times)
    n = len(hits)
    per_min = (n / dur * 60.0) if dur > 0 else 0.0

    def made(name: str, why: str) -> VisualProfile:
        return VisualProfile(name, VISUAL_PROFILES[name], why)

    if dur and dur < 90:
        name = "dynamic" if n >= 4 else ("lecture" if n >= 1 else "talking_head")
        return made(name, f"{n} scene change(s) in {dur:.0f}s, read by count not rate")

    if per_min >= 4:
        return made("dynamic", f"{per_min:.0f} hard cuts a minute")
    if per_min >= 0.5:
        return made("lecture",
                    f"{per_min:.1f} hard cuts a minute, spread through the video")

    # Quiet by every cut measure. What the screen *says* is the only thing left
    # that can tell a screen recording from a presenter, so it decides here —
    # but only once it has read enough to be worth deciding on. Below the floor
    # it is not a weak signal, it is no signal, and treating a ratio over three
    # burned-in lines of a test pattern as a reading is how a moving test card
    # gets classified as source code.
    if ocr and ocr.frames and ocr.chars >= OCR_MIN_CHARS:
        if ocr.code_line_ratio >= 0.25 and ocr.code_lines >= OCR_MIN_CODE_LINES:
            return made("code", f"{ocr.code_lines} of {ocr.code_lines + ocr.ui_lines} read "
                                f"lines are code-shaped, across {ocr.chars} characters")
        if ocr.text_frame_ratio >= 0.6 and ocr.chars_per_frame >= 120:
            return made("screen_ui", f"{ocr.text_frames} of {ocr.frames} probe frames carry "
                                     f"text, {ocr.chars_per_frame:.0f} characters each, and the "
                                     f"video does not cut")
        if ocr.text_frame_ratio >= 0.6:
            return made("lecture", f"{ocr.text_frames} of {ocr.frames} probe frames carry "
                                   f"text and the video does not cut")
    elif ocr and ocr.frames and ocr.chars:
        return made("talking_head", f"{n} hard cut(s) in {dur:.0f}s, and the text probe read "
                                   f"only {ocr.chars} characters — too few to classify on")
    return made("talking_head", f"{n} hard cut(s) in {dur:.0f}s"
                             + ("" if not ocr or not ocr.text_frames
                                else f", and little written on it ({ocr.text_frames} of "
                                     f"{ocr.frames} probe frames)"))


def hint_intervals(snippets: list[dict], segments: list[dict] | None = None) -> list[dict]:
    """Intervals the speaker pointed at, with the weight of the pointing.

    A hint is not evidence and nothing downstream treats it as one — these
    intervals only decide where an extra frame might be worth decoding. What
    comes back is `{start, end, weight, reason}` with the transcript span as the
    interval, so a hint is never wider than the sentence that produced it.
    """
    out: list[dict] = []
    for s in snippets or []:
        text = (s.get("text") or "")
        hits = len(_HINT_RE.findall(text))
        if not hits:
            continue
        start = float(s.get("start", 0.0))
        end = float(s.get("end", start + (s.get("dur") or 0.0)))
        out.append({"start": start, "end": end, "weight": hits,
                    "reason": "the speaker points at something here"})
    for seg in segments or []:
        # One frame per segment, so a section the chapter list names is never
        # skipped by a video whose changes all landed in one other section.
        out.append({"start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end_s", 0.0)),
                    "weight": 1, "reason": "segment coverage"})
    out.sort(key=lambda i: (i["start"], -i["weight"]))
    return out


def _merge_intervals(intervals: list[dict], max_count: int) -> list[dict]:
    """Collapse overlapping intervals, strongest first, into at most `max_count`.

    The cap is what keeps the refinement pass's ffmpeg `select` expression a
    readable length, and dropping the weakest intervals rather than the last is
    what keeps a title montage from filling the list and leaving nothing for the
    rest of the video.
    """
    kept: list[dict] = []
    for iv in sorted(intervals, key=lambda i: (-i["weight"], i["start"])):
        if len(kept) >= max_count:
            break
        lo = float(iv["start"])
        hi = float(iv.get("end", lo))
        # Only intervals of the same kind merge. Two segments that happen to be
        # adjacent are two coverage requests, and merging them produces one
        # interval spanning the whole video — which is not a hint, it is the
        # absence of a hint, and it would quietly turn adaptive sampling back
        # into a uniformly finer grid.
        for k in kept:
            if k["reason"] == iv["reason"] and lo <= k["hi"] and hi >= k["lo"]:
                k["lo"], k["hi"] = min(k["lo"], lo), max(k["hi"], hi)
                k["weight"] += iv["weight"]
                break
        else:
            kept.append({"lo": lo, "hi": hi, "weight": iv["weight"], "reason": iv["reason"]})
    return sorted(kept, key=lambda k: k["lo"])


class SamplingPlan:
    """The frames a run will decode, decided before any of them are.

    Two properties the whole design rests on, both checkable here and both
    asserted in the tests: the total is bounded by the budget *by
    construction* — the layers are sized to sum to it, not trimmed afterwards —
    and the baseline is computed from the duration and the profile alone, so no
    amount of scene changes or spoken hints can move it.
    """

    __slots__ = ("baseline_step", "refine_step", "intervals", "budget", "profile",
                 "baseline_count", "hint_count", "change_count")

    def __init__(self, profile: VisualProfile, duration: float, budget: int,
                 intervals: list[dict]):
        self.profile = profile
        self.budget = max(0, int(budget))
        dur = max(1.0, float(duration))
        n_base = min(self.budget, max(1, round(dur / max(1.0, profile.step_s))))
        n_hint = min(self.budget - n_base, int(self.budget * BUDGET_SPLIT["hint"]))
        n_change = min(self.budget - n_base - n_hint,
                       int(self.budget * BUDGET_SPLIT["change"]))
        self.baseline_count, self.hint_count, self.change_count = n_base, n_hint, n_change
        # The grid spans [0, duration] *inclusive*, so the last baseline frame is
        # the end of the video and not one step short of it. The exclusive
        # version looked covered — a frame 24s into a 36s video is inside the
        # video — and left the entire final scene with no frame at all, which is
        # the exact defect the tail-coverage requirement exists to prevent.
        self.baseline_step = dur / (n_base - 1) if n_base > 1 else dur
        self.refine_step = max(2.0, self.baseline_step / 2.0)
        self.intervals = intervals

    def times(self) -> list[float]:
        """Every timestamp this plan will produce, in order, deduplicated.

        One list rather than three, because the merge is where a budget bug
        would live. Baseline first and on its own grid; refinement at twice the
        density inside the chosen intervals; the two are unioned at 0.5s
        tolerance, so a refinement frame that lands on a baseline frame is one
        frame and not two.
        """
        if self.baseline_count == 1:
            # One frame for the whole video goes in the middle, where the most
            # content is, rather than at 0s where the title card usually is.
            out: list[float] = [round(self.baseline_step / 2, 3)]
        else:
            out = [round(i * self.baseline_step, 3) for i in range(self.baseline_count)]
        room = self.budget - len(out)
        picked = 0
        step = self.refine_step
        for k in self.intervals:
            if picked >= room:
                break
            t = k["lo"] + step / 2
            while t < k["hi"] and picked < room:
                if all(abs(t - x) > step / 2 for x in out):
                    out.append(round(t, 3))
                    picked += 1
                t += step
        return sorted(out)

    def metrics(self, times: list[float], duration: float) -> dict:
        gaps = [b - a for a, b in zip(times, times[1:])] or [0.0]
        # The three candidate counts are reported separately because the whole
        # design is that no layer spends the whole budget on itself. A run where
        # `refinement_candidates` is the largest of the three is a run whose
        # coverage came from whatever the speaker happened to point at.
        scene_iv = sum(1 for k in self.intervals if k["reason"].startswith("the video changes"))
        return {
            "sampling_version": SAMPLING_VERSION,
            "duration_s": round(float(duration), 1),
            "budget": self.budget,
            "visual_density_profile": self.profile.name,
            "profile_reason": self.profile.reason,
            "uniform_step_s": round(self.baseline_step, 2),
            "refine_step_s": round(self.refine_step, 2),
            "uniform_candidates": self.baseline_count,
            "scene_candidates": scene_iv,
            "refinement_candidates": len(self.intervals) - scene_iv,
            "refinement_intervals": len(self.intervals),
            "planned_frames": len(times),
            "frames_per_minute": round(len(times) / (duration / 60.0), 2) if duration else None,
            "coverage_start_s": round(times[0], 2) if times else None,
            "coverage_end_s": round(times[-1], 2) if times else None,
            "largest_temporal_gap_s": round(max(gaps), 2),
            "median_temporal_gap_s": round(sorted(gaps)[len(gaps) // 2], 2),
            "p95_temporal_gap_s": round(sorted(gaps)[min(len(gaps) - 1, int(len(gaps) * 0.95))], 2),
            "budget_respected": len(times) <= self.budget,
        }


def plan_sampling(duration: float | None, budget: int, profile: VisualProfile,
                  hints: list[dict] | None = None,
                  change_intervals: list[dict] | None = None) -> SamplingPlan:
    """The plan, from everything the sampler knows before decoding anything.

    `hints` are the transcript's pointing and the per-segment coverage floor;
    `change_intervals` are where the video showed it changing. Both are optional
    and the plan is valid with neither — that is the case where a caller has
    only a duration, and it must still cover the whole video.
    """
    dur = float(duration or 0.0)
    every = list(hints or []) + list(change_intervals or [])
    # An interval longer than a quarter of the video is not a place someone
    # pointed at. Keeping it would spend the whole refinement reserve on a
    # single mask covering the entire timeline, which is the old uniform grid
    # with a smaller step and none of the adaptivity.
    every = [iv for iv in every
             if not dur or (float(iv.get("end", iv["start"])) - float(iv["start"])) <= 0.25 * dur]
    return SamplingPlan(profile, dur, budget, _merge_intervals(every, 24))


def _ocr_probe(video: Path, dur: float, backend: str = "auto") -> tuple[OcrProfile, dict]:
    """A coarse read of what the video says, taken before the plan is made.

    `OCR_PROBE_FRAMES` small frames on their own uniform grid, read once, thrown
    away. It exists for one measurement — a screen recording and a presenter look
    identical to a scene score, and only the words on the screen tell them apart —
    and it deliberately does *not* use the sampler's grid: the baseline grid is
    sized by the profile, so a probe taken on it would be reading a grid this
    function has not decided yet.

    Returns the profile *and* what it cost, because a policy that adds a pass has
    to be able to say what that pass bought and what it cost. The split matters:
    `probe_extract_s` is ffmpeg decoding sixteen small frames and `probe_ocr_s` is
    the read, and the two fail for unrelated reasons.

    The read is batched through one provider call. Sixteen per-frame calls paid
    sixteen process spawns and sixteen framework loads to do the work of one,
    which is the same mistake the main pass already stopped making.

    The probe is deliberately NOT cached and its frames are written to a
    temporary directory that is deleted on the way out. It is a classification
    input, not evidence: a probe result that reached the cache could be read back
    as a reading of a frame that the final pass never sampled, and the two must
    not be confused.

    Never raises and never blocks. An OCR backend that is missing, slow or broken
    returns an empty profile, which is a real answer ("no evidence either way")
    and leaves the classifier exactly where it was before this probe existed. The
    alternative — failing the run, or guessing the profile when the read failed —
    would put a sampling decision on a signal the run does not have.
    """
    empty = ocr_profile([])
    zero = {"probe_frames": 0, "probe_extract_s": 0.0, "probe_ocr_s": 0.0,
            "probe_ocr_s_per_frame": None, "probe_backend": None}
    if dur <= 0 or OCR_PROBE_FRAMES <= 0 or backend == "unavailable":
        return empty, zero
    try:
        import ocr as O  # noqa: PLC0415  (visual text layer, only needed on this path)
    except Exception:  # noqa: BLE001 — a sampler that cannot import a reader still samples
        return empty, zero
    try:
        provider = O.get_provider(backend)
        if not getattr(provider, "available", False):
            return empty, {**zero, "probe_backend": getattr(provider, "name", None)}
        step = dur / (OCR_PROBE_FRAMES + 1)
        extract_started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="yt-ocrprobe-") as tmp:
            dest = Path(tmp)
            proc = subprocess.run(
                ["ffmpeg", "-v", "info", "-y", "-i", str(video), "-vf",
                 f"fps=1/{step:.3f},scale={OCR_PROBE_WIDTH}:-2,showinfo",
                 "-frames:v", str(OCR_PROBE_FRAMES), str(dest / "p%03d.jpg")],
                check=False, capture_output=True, text=True,
            )
            extract_s = round(time.monotonic() - extract_started, 2)
            times = _frame_times(proc.stderr, OCR_PROBE_FRAMES)
            paths = [str(p) for p in sorted(dest.glob("p*.jpg"))]
            ocr_started = time.monotonic()
            batch = provider.extract_many(paths) if paths else {}
            ocr_s = round(time.monotonic() - ocr_started, 2)
        reads: list[tuple[float, str]] = []
        for i, path in enumerate(paths):
            res = batch.get(path)
            reads.append((times[i] if i < len(times) else float(i) * step,
                          getattr(res, "text", None) or ""))
        return ocr_profile(reads), {
            "probe_frames": len(paths),
            "probe_extract_s": extract_s,
            "probe_ocr_s": ocr_s,
            "probe_ocr_s_per_frame": round(ocr_s / len(paths), 3) if paths else None,
            "probe_backend": getattr(provider, "name", None),
        }
    except Exception:  # noqa: BLE001 — see above: no signal, not a failed run
        return empty, zero


def extract_frames(video: Path, outdir: Path, duration: float | None, budget: int,
                   snippets: list[dict] | None = None,
                   segments: list[dict] | None = None,
                   ocr_backend: str = "auto") -> dict:
    """Sample `video` into outdir/frames plus a contact sheet.

    Four ffmpeg passes, in this order, and the order is the design:

    1. **scene probe** — one cheap low-resolution decode that reports where the
       video changes. It produces no frame and nothing downstream cites it; it
       exists so the change intervals are known *before* any budget is committed.
    2. **text probe** — a dozen small frames, read once and discarded, so a
       screen recording and a presenter can be told apart. See `_ocr_probe`.
    3. **uniform baseline** — the coverage guarantee. Its spacing comes from the
       duration and the profile and from nothing else, so no amount of scene
       changes elsewhere in the video can move it.
    4. **refinement** — one frame every `refine_step` seconds, restricted to the
       intervals the plan chose.

    Each pass is capped by the number of frames the plan allocated to it, not by
    the run's budget, so the total is bounded by construction. The old sampler
    passed the budget to every pass independently, which is how a run could
    write twice what it was asked for. The two probes are the exception and are
    bounded by fixed constants instead, because they write no frame anyone
    cites.
    """
    frames = outdir / "frames"
    for old in frames.glob("*.jpg"):
        old.unlink()
    frames.mkdir(parents=True, exist_ok=True)

    dur = float(duration or 0.0)
    started = time.monotonic()

    # -- pass 1: the scene probe. Low resolution, no output, one decode. -------
    probe = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(video), "-vf",
         f"scale=160:-2,select='gt(scene,{SCENE_PROBE})',showinfo",
         "-fps_mode", "vfr", "-f", "null", "-"],
        check=False, capture_output=True, text=True,
    )
    cut_times = _frame_times(probe.stderr, 100000)
    # -- pass 2: the text probe. A dozen small frames, read once. -------------
    probe_started = time.monotonic()
    ocr, probe_timings = _ocr_probe(video, dur, ocr_backend)
    probe_s = round(time.monotonic() - probe_started, 2)
    profile = classify_visual_profile(cut_times, dur, ocr)
    hints = hint_intervals(snippets or [], segments or [])

    times: dict[str, float] = {}
    # -fps_mode vfr is required: ffmpeg 7+ removed -vsync. Frames are stored at
    # the video's native 720p width, NOT downscaled: a 768px frame makes on-screen
    # chart values unreadable, and an unreadable number is one the reader will
    # guess at — this produced a fabricated figure in a real report.
    # A provisional plan, built from the profile alone. Its baseline count and
    # step cannot change once the change intervals are known — they are a
    # function of the duration, the profile and the budget, and nothing else —
    # so the grid the reader relies on for coverage is decided before any
    # evidence about the video's content is taken into account. That ordering is
    # the guarantee; this line is where it is spent.
    provisional = plan_sampling(dur, budget, profile, hints)
    baseline = _run_pass(video, frames, "u",
                         f"fps=1/{provisional.baseline_step:.2f},scale=1280:-2",
                         max(1, provisional.baseline_count), times)

    # The tail, explicitly. `fps=1/step` emits at multiples of the step that are
    # strictly inside the video, so a grid whose last point *is* the duration
    # never produces it — the final scene of a 36s video sampled every 12s came
    # back with nothing after 24s. One seek into the last couple of seconds
    # costs a fraction of a decode and makes the guarantee unconditional.
    if dur > 0 and baseline:
        times.update(_tail_frame(video, frames, dur))

    # The change layer, measured on the baseline frames rather than guessed.
    # This is the signal that separates a slide deck from a screen recording,
    # and it is free: the images are already on disk and the comparison is the
    # one `dedupe_frames` already makes.
    changes = (change_intervals(baseline, times) if baseline else [])
    plan = plan_sampling(dur, budget, profile,
                         hints, changes + _change_intervals(cut_times))
    assert plan.baseline_step == provisional.baseline_step, \
        "the baseline grid moved once the change intervals were known"
    planned = plan.times()
    if plan.intervals and plan.budget - plan.baseline_count > 0:
        _run_pass(video, frames, "r", _refine_filter(plan),
                  plan.budget - plan.baseline_count, times)

    have = sorted(frames.glob("*.jpg"), key=lambda p: times.get(p.name, 0.0))
    # The budget is a ceiling on the whole process, not on each pass, so it is
    # enforced here as well as in the plan. Thinning drops the frames furthest
    # from the dense end of the timeline rather than the tail, so the ceiling
    # cannot cost coverage at the end of the video.
    if budget > 0 and len(have) > budget:
        have = _thin_to_budget(have, times, budget)
    if not have:
        return {"status": "no_frames", "reason": "ffmpeg produced no frames"}
    pre_dedupe_times = [times[f.name] for f in have if f.name in times]
    have, dropped = dedupe_frames(have)
    if not have:  # dedup is too aggressive for this video; keep the originals
        have = sorted(frames.glob("*.jpg"), key=lambda p: times.get(p.name, 0.0))
    # What deduplication cost the timeline. Computed here, while both the frame
    # list and the timestamps are in hand, so the number that reaches
    # `frames.json` and `meta.json` is the one measured from the same frames the
    # report is written about.
    coverage = dedupe_coverage(pre_dedupe_times,
                               [times[f.name] for f in have if f.name in times])

    # Contact sheet(s) built from the frame files themselves, so the sheet and the
    # individual frames are guaranteed to be the same images.
    #
    # Split at MAX_SHEET_ROWS. A single 77-frame sheet comes out 1536x4320; once
    # an image that tall is scaled to fit a reader, each 384px tile lands near
    # 170px and the slide text is gone — the sheet stops being able to triage, and
    # triage is the sheet's only job. Capping rows per sheet keeps every tile at
    # full size. Measured on a 46-minute lecture: one 20-row sheet was unusable,
    # the same 77 frames across 7-row sheets is readable.
    cols = 4
    max_rows = 7
    per_sheet = cols * max_rows
    sheets: list[Path] = []
    ordered = sorted(have, key=lambda p: times.get(p.name, 0.0))
    for i in range(0, len(ordered), per_sheet):
        chunk = ordered[i : i + per_sheet]
        rows = max(1, -(-len(chunk) // cols))
        dest = outdir / (f"contact-sheet.jpg" if len(ordered) <= per_sheet else f"contact-sheet-{i // per_sheet + 1}.jpg")
        # -pattern_type glob on a single file is not portable; feed the chunk explicitly.
        inputs: list[str] = []
        for f in chunk:
            inputs += ["-i", str(f)]
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", *inputs,
             "-filter_complex", f"{''.join(f'[{j}:v]scale=384:-2[v{j}];' for j in range(len(chunk)))}"
                                f"{''.join(f'[v{j}]' for j in range(len(chunk)))}"
                                f"concat=n={len(chunk)}:v=1:a=0,tile={cols}x{rows}",
             "-frames:v", "1", str(dest)],
            check=False, capture_output=True, text=True,
        )
        if dest.exists():
            sheets.append(dest)
        else:
            # Never let a failed ffmpeg pass look like "there were no frames to
            # show" — that reads as a clean result and silently loses triage.
            print(json.dumps({"warning": "contact sheet generation failed",
                              "chunk": f"{i + 1}-{i + len(chunk)}",
                              "ffmpeg": proc.stderr[:200]}), file=sys.stderr)
    index = [
        "# Frames", "",
        f"{len(have)} frames from a {dur / 60:.0f}-minute video, sampled as a "
        f"{plan.profile.name} profile: a uniform grid (~{plan.baseline_step:.0f}s apart) plus",
        f"refinement inside {len(plan.intervals)} interval(s) the transcript or the video pointed at,"
        f" then near-duplicates removed"
        + (f" ({dropped} dropped as visually identical)." if dropped else "."),
        "",
        "**Contact sheets:** " + ", ".join(f"`{s.name}`" for s in sheets) if sheets else "**Contact sheet: none generated**",
        "",
        "Read the sheets to triage, then open individual frames at native resolution — a sheet tile is",
        "never legible enough to transcribe from. Timestamps are measured from the video.", "",
    ]
    for f in ordered:
        # A frame whose timestamp could not be read prints as [unmeasured], never
        # as 00:00:00: a zero reads as "the first second of the video" and moves
        # the evidence to a place it was never seen.
        t = times.get(f.name)
        index.append(f"- `{f.name}` [{ts(t) if t is not None else 'unmeasured'}]")
    (outdir / "FRAMES.md").write_text("\n".join(index))

    # The machine-readable form of the same index. FRAMES.md is for triage; this
    # is what the evidence layer cites, because a citation has to resolve to a
    # measured timestamp rather than to a formatted string someone re-parsed.
    #
    # Stamped with the sampling policy. A frame set produced under another policy
    # is not the same evidence, and reading it back as though it were is the
    # silent version of the bug this policy fixes: the report would describe a
    # sampler that is no longer the one that ran.
    kinds = FRAME_KIND_BY_PREFIX
    frame_index = [
        {
            "frame_id": f.stem, "file": f"frames/{f.name}",
            "kind": kinds.get(f.name[0], "unknown"),
            "t_start": times.get(f.name),
        }
        for f in ordered
    ]
    (outdir / "frames.json").write_text(json.dumps({
        "sampling_version": SAMPLING_VERSION,
        "policy": {**plan.metrics(planned, dur),
                   # What the plan asked for is not what is on disk: near-duplicates
                   # were dropped, and the number that came out is the one a later
                   # run is compared on. `meta.json` carries the timings.
                   "final_frames": len(have),
                   "near_duplicates_dropped": dropped,
                   "coverage": coverage,
                   "text_probe": ocr.to_dict()},
        "frames": frame_index,
    }, indent=2))
    measured = sorted(t for t in (times.get(f.name) for f in have) if t is not None)
    return {
        "status": "ok",
        "frames": len(have),
        # The number the candidate counts are about. `planned_frames` is what the
        # plan asked for and `frames` is what survived deduplication; this is the
        # same as `frames` under its name in the metric table, spelled out because
        # a run compared across two policies needs the *final* count beside the
        # plan's, and `planned_frames` alone invites that comparison to be made
        # against the wrong number.
        "final_frames": len(have),
        "measured_timestamps": sum(1 for f in have if f.name in times),
        "near_duplicates_dropped": dropped,
        "contact_sheets": [s.name for s in sheets],
        "elapsed_s": round(time.monotonic() - started, 2),
        **plan.metrics(measured, dur),
        "frame_gaps_s": _gap_stats(measured),
        "coverage": coverage,
        "hard_cuts_detected": len(cut_times),
        "text_probe": ocr.to_dict(),
        # Timed separately because it is the one cost this policy adds that the
        # one before it did not, and a policy that buys better coverage has to
        # be able to say what it cost. One extra low-resolution decode plus
        # sixteen small reads, against a whole-video decode per sampled frame
        # layer below. `probe_extract_s` and `probe_ocr_s` split the two halves
        # so a probe that gets slower can be attributed to the decode or to the
        # read, which have nothing to do with each other.
        "probe_s": probe_s,
        **probe_timings,
        "hint_intervals": len(hints),
        "change_intervals": len(changes),
        "segments_with_visual_coverage": _segments_covered(segments or [], times, have),
        # The plan's coverage beside the surviving frames' coverage. On a video
        # that holds one image for two minutes, every sample is identical and
        # deduplication keeps one of them: the frames then cover a third of the
        # video and the report is right, but `coverage_end_s` alone reads as
        # "the sampler never looked at the end", which is the opposite of what
        # happened. Two numbers, never one.
        "planned_coverage_end_s": planned[-1] if planned else None,
        "planned_frames": len(planned),
    }


def _run_pass(video: Path, frames: Path, prefix: str, vf: str, cap: int,
              times: dict[str, float]) -> list[Path]:
    """One ffmpeg sampling pass, its frames and their measured times.

    `-frames:v` is the number of frames *this* layer was allocated, never the
    run's budget. The old code passed the budget to every pass, so a run could
    write twice what it was asked for and the total was only ever known after
    the fact.
    """
    if cap <= 0:
        return []
    proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-y", "-i", str(video), "-vf", f"{vf},showinfo",
         "-fps_mode", "vfr", "-frames:v", str(cap), str(frames / f"{prefix}%04d.jpg")],
        check=False, capture_output=True, text=True,
    )
    out: list[Path] = []
    for i, t in enumerate(_frame_times(proc.stderr, cap), start=1):
        f = frames / f"{prefix}{i:04d}.jpg"
        if f.exists():
            times[f.name] = t
            out.append(f)
    return out


def _tail_frame(video: Path, frames: Path, dur: float, window: float = 2.0) -> dict:
    """One frame from the last couple of seconds, so the tail is never empty.

    Seeked rather than filtered: the frame's real position is the seek offset
    plus whatever `showinfo` reports, so the timestamp is measured and not
    assumed to be the end of the video. A claim of "the end" that is really
    "two seconds before the end" is still within the tolerance a reader would
    accept, and is a fact rather than a rounding.
    """
    back = max(0.2, min(window, dur * 0.02))
    dest = frames / "z0001.jpg"
    proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-y", "-ss", f"{dur - back:.3f}", "-i", str(video),
         "-vf", "scale=1280:-2,showinfo", "-frames:v", "1", str(dest)],
        check=False, capture_output=True, text=True,
    )
    if not dest.exists():
        return {}
    shown = _frame_times(proc.stderr, 1)
    return {"z0001.jpg": round(dur - back + (shown[0] if shown else 0.0), 3)}


def _segments_covered(segments: list[dict], times: dict[str, float],
                      have: list[Path]) -> int:
    """How many of the video's own sections a surviving frame landed in.

    A count of frames says nothing about *where* they landed: three frames in the
    first ten seconds of a lecture and none in the last five minutes is a worse
    sample than three spread evenly, and the frame count reads the same either
    way. This is the number that says so.

    Measured on the frames that survived deduplication, not on the plan, for the
    same reason `coverage_end_s` sits beside `planned_coverage_end_s`: a section
    whose every sample was a duplicate of the one before it has been looked at and
    legitimately collapsed to nothing, and that is not the same as a section the
    sampler never reached.
    """
    if not segments:
        return 0
    stamps = [times.get(f.name) for f in have]
    covered = 0
    for seg in segments:
        lo = float(seg.get("start", 0.0))
        hi = float(seg.get("end_s", 0.0))
        if hi <= lo:
            continue
        if any(t is not None and lo <= t < hi for t in stamps):
            covered += 1
    return covered


def _gap_stats(measured: list[float]) -> dict:
    """The coverage numbers, so a thin tail is visible without reading frames."""
    gaps = [b - a for a, b in zip(measured, measured[1:])] or [0.0]
    ordered = sorted(gaps)
    return {
        "largest_temporal_gap_s": round(max(gaps), 2),
        "median_temporal_gap_s": round(ordered[len(ordered) // 2], 2),
        "p95_temporal_gap_s": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 2),
    }


def dedupe_coverage(before: list[float], after: list[float]) -> dict:
    """What deduplication cost the timeline, stated as coverage.

    `before` and `after` are the same timeline sampled at different densities:
    `before` is every frame handed to the deduper, `after` is what survived.
    Reporting one number cannot distinguish "the sampler never looked here" from
    "the sampler looked and the picture did not change" — and those are opposite
    findings, because the first is a coverage defect and the second is the
    deduper working. So both gaps are published, and the difference between them
    is the coverage deduplication is responsible for.

    A dropped frame is counted as gone. It is not in `after`, it is not counted
    toward coverage, and it is not available to alignment: a downstream stage
    that treats a removed frame as if it still exists is the silent version of
    this defect, and the only defence is that the number here never pretended
    otherwise.

    `dedupe_collapsed_intervals` counts maximal runs of consecutive drops, which
    is the shape that hurts: one isolated drop costs one frame, one collapsed
    interval of ten costs ten and a hole in the timeline.
    """
    before = sorted(t for t in before if t is not None)
    after = sorted(t for t in after if t is not None)
    kept = set(after)
    runs, run_len, run_start = [], 0, None
    for t in before:
        if t in kept:
            if run_len:
                runs.append((run_start, t, run_len))
            run_len, run_start = 0, None
        else:
            if not run_len:
                run_start = t
            run_len += 1
    if run_len:
        runs.append((run_start, before[-1] if before else None, run_len))
    widest = max((b - a for a, b, _ in runs if a is not None and b is not None), default=0.0)
    return {
        "frames_before_dedupe": len(before),
        "frames_after_dedupe": len(after),
        "dedupe_collapsed_intervals": len(runs),
        "dedupe_widest_collapsed_s": round(widest, 2),
        "largest_gap_before_dedupe_s": _gap_stats(before)["largest_temporal_gap_s"],
        "largest_gap_after_dedupe_s": _gap_stats(after)["largest_temporal_gap_s"],
        "p95_gap_before_dedupe_s": _gap_stats(before)["p95_temporal_gap_s"],
        "p95_gap_after_dedupe_s": _gap_stats(after)["p95_temporal_gap_s"],
    }


def _change_intervals(scene_times: list[float], pad: float = 3.0) -> list[dict]:
    """A frame either side of every place the video changed.

    One interval per cut rather than one frame per cut, and padded, so a slide
    that advances is sampled at the boundary instead of only after it — the frame
    a reader would have seen is the one on the far side of the change.
    """
    return [{"start": max(0.0, float(t) - pad), "end": float(t) + pad,
             "weight": 1, "reason": "the video changes here"}
            for t in scene_times]


def _thin_to_budget(have: list[Path], times: dict[str, float], budget: int) -> list[Path]:
    """Cut an over-budget set back to the budget without losing the tail.

    Keeps the first and last frame, then the frames with the most neighbours in
    the densest part of the timeline. Dropping the *last* N instead would be
    simpler and would recreate the original defect: a budget that silently
    un-covers the end of the video.
    """
    if budget <= 0 or len(have) <= budget:
        return have
    keep = {have[0].name, have[-1].name}
    ranked = sorted(have, key=lambda p: -sum(
        1 for q in have if abs(times.get(p.name, 0) - times.get(q.name, 0)) <= 4.0))
    for p in ranked:
        if len(keep) >= budget:
            break
        keep.add(p.name)
    return [p for p in have if p.name in keep][:budget]


def _refine_filter(plan: SamplingPlan) -> str:
    """The ffmpeg expression for the refinement pass.

    `select` with a mask over the planned intervals, at one frame per
    `refine_step` seconds *inside* them and nowhere else. The `isnan(prev_selected_t)`
    idiom is ffmpeg's own "first frame or far enough from the last one"; without
    the mask it would sample the whole video, and with the mask alone it would
    sample every frame inside them.
    """
    mask = "+".join(f"between(t,{k['lo']:.2f},{k['hi']:.2f})" for k in plan.intervals)
    step = max(2.0, plan.refine_step)
    return (f"select='(isnan(prev_selected_t)+gte(t-prev_selected_t,{step:.2f}))*"
            f"({mask})',scale=1280:-2")


CACHE_DIR = Path(os.environ.get("YT_REPORT_CACHE", Path.home() / ".cache" / "yt-report"))


def cache_load(video_id: str, lang: str) -> dict | None:
    """Reuse a previous run's transcript instead of re-transcribing.

    Local speech-to-text is the slow part of this pipeline — roughly realtime on
    Apple silicon, so a 50-minute talk costs most of an hour. Re-running the same
    URL to change the frame budget or rebuild a report should not pay that twice.
    The cache is keyed on video id + language, since that is exactly what the
    transcript depends on.
    """
    f = CACHE_DIR / f"{video_id}.{lang}.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not data.get("snippets"):
        return None
    return data


def cache_store(video_id: str, lang: str, data: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_DIR / f"{video_id}.{lang}.tmp"
        tmp.write_text(json.dumps(data))
        tmp.replace(CACHE_DIR / f"{video_id}.{lang}.json")  # atomic: no half-written cache
    except OSError:
        pass  # a cache miss next run is not worth failing the run for


def auto_frame_budget(duration_s: float | None) -> int:
    """The ceiling the profile then spends.

    Sized on the *densest* profile, not the average one, because a ceiling that
    is computed per video from a guess about its content is the old defect with
    a new name: guess wrong about a screen recording and the run is capped below
    what the video needs. The profile then decides how much of the ceiling to
    use, which is a decision it can be wrong about without capping anything.

    The old value was one per 30s, clamped to 48, which is why a maths animation
    got one frame every 30.3s and a +/-4s window contained a frame 3% of the
    time. 48 is also gone: it was a hard ceiling on a 60-minute video that
    wanted 500, and the ceiling is the only thing that can stop a run writing
    frames nobody will look at.
    """
    d = duration_s or 600
    finest = min(VISUAL_PROFILES.values())
    return int(max(12, min(MAX_AUTO_FRAMES, round(d / finest))))




def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="+", help="one or more video URLs (see --batch)")
    ap.add_argument("--out", default=None, help="output dir; with several URLs, the parent dir")
    ap.add_argument("--lang", default="en", help="transcript language; other languages are never used silently")
    ap.add_argument("--no-asr", action="store_true",
                    help="do not fall back to local speech-to-text when captions are missing or blocked")
    ap.add_argument("--frames", type=int, default=0,
                    help="max visual frames to sample; 0 = auto by duration, -1 disables")
    ap.add_argument("--batch", action="store_true", help="process every URL given, then print a summary table")
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write the transcript cache")
    ap.add_argument("--no-ocr", action="store_true",
                    help="skip the visual text pass; visual.json still records that OCR did not run")
    # The full provider vocabulary, not a two-name subset. A narrower list here
    # than in `build_evidence.py` is a second door to the same subsystem with a
    # smaller vocabulary, and the two drift apart silently: the flag would
    # refuse `--vlm openrouter` while the documented command accepted it.
    ap.add_argument("--vlm", default="none",
                    choices=["none", "anthropic", "openai", "openrouter", "ollama",
                             "lmstudio", "llamacpp", "custom"],
                    help="run the full evidence pipeline and have this vision-language backend "
                         "interpret the selected frames; 'none' (default) builds evidence from the "
                         "transcript and frames only")
    ap.add_argument("--vlm-model", default=None,
                    help="model id for --vlm anthropic (default: the backend's own)")
    ap.add_argument("--max-vlm-frames", type=int, default=None,
                    help="hard cap on model calls for this run (default 8); 0 sends no frame")
    ap.add_argument("--ocr-backend", default="auto", choices=["auto", "apple-vision"],
                    help="OCR backend; 'auto' uses the first available")
    args = ap.parse_args()

    if len(args.url) > 1 and not args.batch:
        print(f"{len(args.url)} URLs given — pass --batch to process them all", file=sys.stderr)
        return 1

    results = []
    for url in args.url:
        results.append(run_one(url, args, Path(args.out).expanduser() if args.out else None))
    if args.batch:
        print(json.dumps([{k: v for k, v in r.items() if k != "dir_full"} for r in results], indent=2))
    return max((0 if r["status"] == "ok" else r["exit"]) for r in results)


def run_one(url: str, args, out_override: Path | None) -> dict:
    vid = parse_video_id(url)
    if not vid:
        print("could not parse a video id from the URL", file=sys.stderr)
        return {"url": url, "status": "bad_url", "exit": 1}

    meta = fetch_meta(url)
    slug = re.sub(r"[^a-z0-9]+", "-", (meta.get("title") or vid).lower()).strip("-")[:70] or vid
    outdir = (out_override / f"{slug}-{vid}") if (out_override and len(args.url) > 1) \
        else (out_override or Path.home() / "Desktop" / "yt-reports" / f"{slug}-{vid}")
    outdir.mkdir(parents=True, exist_ok=True)

    frames_budget = auto_frame_budget(meta.get("duration")) if args.frames == 0 else max(0, args.frames)

    used_asr = False
    visual: dict = {"status": "skipped"}
    status, transcript_error = "ok", None
    cached = None if args.no_cache else cache_load(vid, args.lang)
    if cached:
        text, snippets, lang_used = " ".join(s["text"] for s in cached["snippets"]), cached["snippets"], cached["lang"]
        status = cached.get("status", "ok")
    else:
        try:
            text, snippets, lang_used = fetch_transcript(vid, args.lang)
        except Exception as e:  # noqa: BLE001
            transcript_error = str(e)[:300]
            status = "rate_limited" if "blocking requests from your IP" in transcript_error else "no_transcript"
            text, snippets, lang_used = "", [], f"none ({status})"

    # Frames are extracted from the same 720p download ASR needs, so a video that
    # needs both is downloaded once. Captions being available does not skip it —
    # the video is still needed for the visuals.
    with tempfile.TemporaryDirectory() as tmpdir:
        video: Path | None = None
        if not snippets or frames_budget > 0:
            try:
                video = download_video(url, Path(tmpdir))
            except Exception as e:  # noqa: BLE001
                transcript_error = (transcript_error + f" | download failed: {e}")[:300] if transcript_error \
                    else f"download failed: {str(e)[:200]}"

        # Most videos carry no usable captions, and YouTube blocks the caption
        # endpoint for a plain IP within minutes, so local ASR is the default
        # fallback rather than an opt-in flag.
        if not snippets and not args.no_asr and video is not None:
            asr_reason = {"rate_limited": "captions endpoint was rate-limited for this IP",
                          "no_transcript": f"this video has no {args.lang} captions"}.get(status, status)
            try:
                text, snippets, lang_used = whisper_transcript(video, args.lang)
                used_asr, status, transcript_error = True, "ok", asr_reason
            except Exception as e:  # noqa: BLE001
                transcript_error = f"ASR failed: {str(e)[:200]}"
        elif not snippets:
            transcript_error = (transcript_error or "") + " | drop --no-asr to enable local ASR"

        # Cut caption blobs into sentence-level units before anything reads their
        # timing. Done here, once, so the cache, the segments, the provenance
        # timestamps and the artifacts on disk are all built from the same units —
        # splitting later would leave a snippet id pointing at a range of the
        # video that no longer corresponds to the text under it.
        snippets = split_long_snippets(snippets)
        segs = segments_from_chapters(snippets, meta["chapters"], meta.get("duration")) if snippets else []

        if frames_budget > 0:
            if video is None:
                visual = {"status": "failed", "error": "no video downloaded"}
            else:
                try:
                    visual = extract_frames(video, outdir, meta.get("duration"), frames_budget,
                                            snippets=snippets, segments=segs,
                                            # `--no-ocr` says do not read the frames, and a
                                            # probe read is a read. Turning the flag off here
                                            # too is what keeps it an answer to the question
                                            # the user asked rather than to the one they did
                                            # not.
                                            ocr_backend="unavailable" if args.no_ocr
                                            else args.ocr_backend)
                except Exception as e:  # noqa: BLE001
                    visual = {"status": "failed", "error": str(e)[:200]}

    (outdir / "URL.txt").write_text(meta.get("webpage_url") or url + "\n")
    # Persist the visual outcome too, not just the transcript one. FRAMES.md records a
    # successful extraction, but when extraction FAILS nothing on disk said so — the
    # failure existed only on stdout, so a later reader of these artifacts could not
    # tell "no frames because there is nothing to see" from "no frames because the
    # download died". An artifact that cannot describe its own failure is how a report
    # ends up asserting screen content that was never captured.
    (outdir / "meta.json").write_text(
        json.dumps({**meta, "transcript_lang": lang_used, "asr": used_asr,
                    "snippet_count": len(snippets), "visual": visual,
                    "transcript_error": transcript_error}, indent=2)
    )

    lines = [f"# {meta.get('title')}", "", f"Source: {meta.get('webpage_url')}", ""]
    if used_asr:
        lines += [f"> Transcript below is local speech-to-text (mlx-whisper large-v3-turbo), used because "
                  f"{transcript_error or 'captions were unavailable'}.", ""]
    if not snippets:
        lines += [f"> **No {args.lang} transcript available** ({status}): {transcript_error}", ""]
    lines += ["## Full transcript (timestamped)", ""]
    for seg in segs:
        if not seg["snippets"]:
            continue
        lines += [f"### [{ts(seg['start'])}] {seg['title']}", ""]
        lines.append(" ".join(s["text"] for s in seg["snippets"]).strip())
        lines.append("")
    (outdir / "transcript.md").write_text("\n".join(lines))

    # The snippets themselves, not just the rendered prose. transcript.md is for
    # a reader; the evidence layer cites fragments by id, and re-deriving them
    # from the markdown would mean re-splitting text that was already split once
    # and losing every boundary it had.
    #
    # Each segment carries its own snippet objects, not just their start times.
    # Storing starts alone made the file lossy in a way nothing reported: the
    # extract stage re-derives segment membership by pairing a segment's snippet
    # back to the transcript's, and with only timestamps on disk that pairing has
    # no text to match on, so every snippet came back unassigned and the whole
    # run failed on a document that looked complete.
    (outdir / "transcript.json").write_text(json.dumps({
        "language": lang_used, "asr": used_asr, "snippets": snippets,
        "segments": [{"start": s["start"], "end_s": s["end_s"], "title": s["title"],
                      "snippets": s["snippets"]} for s in segs],
    }, indent=2))

    # Visual text. Runs after the frames exist and before the evidence layer,
    # because the evidence layer cites what OCR read. A backend that is missing
    # or broken must not cost the transcript: the failure is recorded in
    # visual.json as a status and the rest of the run continues.
    ocr_status: dict = {"status": "skipped", "reason": "frames disabled"}
    visual_payload: dict | None = None
    if frames_budget > 0 and not args.no_ocr:
        try:
            import build_evidence as B  # noqa: PLC0415  (evidence layer, only needed on this path)
            import ocr as O  # noqa: PLC0415  (visual text layer, only needed on this path)

            visual_payload = O.run_ocr(outdir, B.load_frames_index(outdir),
                                       O.get_provider(args.ocr_backend))
            O.write_visual(outdir, visual_payload)
            ocr_status = visual_payload["summary"]
        except Exception as e:  # noqa: BLE001
            ocr_status = {"status": "failed", "error": str(e)[:200]}
    elif args.no_ocr:
        # Written, not omitted: an artifact that is missing cannot say whether the
        # pass was turned off or never existed, and only one of those is a choice
        # the reader can act on.
        try:
            import build_evidence as B  # noqa: PLC0415
            import ocr as O  # noqa: PLC0415

            skipped = O.skipped_payload(B.load_frames_index(outdir), "--no-ocr")
            O.write_visual(outdir, skipped)
            ocr_status = skipped["summary"]
        except Exception as e:  # noqa: BLE001
            ocr_status = {"status": "skipped", "reason": "--no-ocr", "error": str(e)[:200]}

    # Evidence extraction. A transcript is prose until it carries provenance; this
    # is the step that gives every number a snippet to be checked against. A
    # blocking validation failure is written to validation.json and reported, but
    # it does not abort the fetch: the frames and transcript on disk are still
    # real, and deleting them would be the destructive response. What it does do
    # is set a non-zero exit, so no synthesis step can run on it.
    #
    # `--vlm` is what promotes this from the transcript-and-metadata document to
    # the full pipeline: alignment, then the vision pass, then semantic
    # extraction. The default path is unchanged and still writes the same file,
    # because a run nobody asked to spend model calls on should not quietly make
    # them, and because the cheaper document is the right one when no visual
    # interpretation is wanted.
    evidence_status: dict = {"status": "skipped"}
    vlm_status: dict | None = None
    try:
        import build_evidence as B  # noqa: PLC0415  (evidence layer, only needed on this path)

        if args.vlm != "none":
            from vlm import DEFAULT_MAX_VLM_FRAMES, get_provider  # noqa: PLC0415

            provider = get_provider(args.vlm, model=args.vlm_model)
            if not provider.available:
                vlm_status = {"status": "unavailable",
                              "reason": getattr(provider, "reason", "unknown")}
                print(f"  note: VLM unavailable — {vlm_status['reason']}", file=sys.stderr)
            _path, _metrics = B.run_pipeline(
                meta, snippets, segs, outdir,
                frames_index=B.load_frames_index(outdir),
                lang_used=lang_used, used_asr=used_asr, visual=visual_payload,
                vlm_provider=provider,
                max_vlm_frames=(DEFAULT_MAX_VLM_FRAMES if args.max_vlm_frames is None
                                else args.max_vlm_frames))
            vlm_status = _metrics.get("vlm")
        from evidence_model import load_evidence  # noqa: PLC0415

        # Read back from disk rather than from the object the pipeline held.
        # The document a later stage will read is the one that counts, and a
        # producer that validates a richer in-memory object than it wrote has
        # validated nothing.
        ev = load_evidence(outdir / "evidence.json")
        result = ev.validation
        evidence_status = {
            "status": "ok" if result["valid"] else "invalid",
            "items": len(ev.items), "segments": len(ev.segments),
            "snippets": len(ev.snippets), "frames": len(ev.visual),
            "blocking_errors": len(result["blocking_errors"]),
            "provenance_coverage": result["provenance_coverage"],
            "ocr_status": result.get("ocr_status"),
        }
    except Exception as e:  # noqa: BLE001
        evidence_status = {"status": "failed", "error": str(e)[:200]}

    if not args.no_cache and snippets:
        cache_store(vid, args.lang, {"snippets": snippets, "lang": lang_used, "status": status})

    summary = {
        "status": status,
        "dir": str(outdir),
        "url": url,
        "video_id": vid,
        "title": meta.get("title"),
        "duration_min": round((meta.get("duration") or 0) / 60, 1),
        "chapters": len(segs),
        "chapters_source": ("ytdlp" if meta.get("chapters_available") else "inferred") if segs else "none",
        "transcript_lang": lang_used,
        "asr": used_asr,
        "cached": bool(cached),
        "words": len(text.split()),
        "transcript_note": transcript_error,
        "visual": visual,
        "ocr": ocr_status,
        "vlm": vlm_status or {"status": "skipped", "reason": f"--vlm {args.vlm}"},
        "evidence": evidence_status,
    }
    print(json.dumps(summary, indent=2))
    # Exit 4: the fetch succeeded and the evidence layer rejected what it built.
    # A separate code because it is a different kind of failure — the artifacts
    # are good, the evidence is not — and a caller must be able to tell "fetch
    # failed, retry" from "do not synthesise this".
    exit_code = {"rate_limited": 3, "no_transcript": 2}.get(status, 0)
    if evidence_status["status"] == "invalid":
        exit_code = 4
    return {**summary, "exit": exit_code}


if __name__ == "__main__":
    raise SystemExit(main())
