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
    """Anchor transcript lines to chapters; otherwise cut even blocks sized to the video.

    The fallback used to be fixed 5-minute buckets, which is defensible for a
    40-minute talk and useless for anything else — a 5.5-minute news roundup came
    out as two segments. Block length now scales with duration so the segment
    count lands in roughly the 4–14 range that reads as sectioned without
    fragmenting, and blocks are still aligned to round minutes so timestamps
    look like times rather than offsets.
    """
    if chapters:
        bounds = [(c["start_time"], c["title"]) for c in chapters]
    else:
        total = duration or (snippets[-1]["start"] + 5 if snippets else 300)
        # Pick a target segment COUNT and derive the block from it. Fixed block
        # lengths are the wrong model: a 60s block gave a 5-minute news video a
        # usable 6 sections and a 15-minute digest a fragmenting 15. Roughly one
        # section per 3 minutes, clamped, keeps every length readable.
        target = int(max(4, min(12, round((total / 60) / 3))))
        block = max(30, int(total / target))
        bounds = [(float(t), f"Part {t // block + 1}") for t in range(0, int(total), block)]
    out = []
    for i, (start, title) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else float("inf")
        body = [s for s in snippets if start <= s["start"] < end]
        out.append({"start": start, "title": title, "chars": sum(len(s["text"]) for s in body), "snippets": body})
    # A block that captured no speech is a gap in the video, not a section of it.
    return [s for s in out if s["snippets"]]


def _frame_times(stderr: str | None, budget: int) -> list[float]:
    """Presentation timestamps, in seconds, of the frames written by the same
    ffmpeg filter chain. Parsed from the extraction pass's own showinfo output, so
    a cited frame carries a real timestamp instead of an estimate derived from its
    position in a sorted list. The alternative — a second pass that only decodes to
    print timestamps — costs a full extra decode of the video per sampler."""
    return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", stderr or "")[:budget]]


def dedupe_frames(frames: list[Path], threshold: float = 3.0) -> tuple[list[Path], int]:
    """Drop near-identical frames, keeping the first of each run.

    A video that holds a full-text card or a static slide for forty seconds
    yields several identical samples; they consume the contact sheet — the one
    image the reader triages from — without adding information. Comparing each
    frame to the last KEPT frame at 16x16 grey collapses those runs.

    Measured: on a static text-card video this drops ~44% of frames with no loss
    of distinct content; on a motion video (talking head, animation) it drops
    nothing, because genuinely moving frames differ by more than the threshold.
    That asymmetry is the point — it is a no-op where it should be.
    """
    try:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return frames, 0
    kept: list[Path] = []
    last = None
    for f in frames:
        try:
            arr = np.asarray(Image.open(f).convert("L").resize((16, 16)), dtype=np.float32)
        except Exception:  # noqa: BLE001, PERF203 — an unreadable frame is kept, not lost
            kept.append(f)
            continue
        if last is not None and float(np.abs(arr - last).mean()) < threshold:
            f.unlink(missing_ok=True)
            continue
        kept.append(f)
        last = arr
    return kept, len(frames) - len(kept)


def extract_frames(video: Path, outdir: Path, duration: float | None, budget: int) -> dict:
    """Sample `video` into outdir/frames plus a contact sheet. Two samplers run and
    merge: a uniform grid (guarantees the whole video is covered) and scene-change
    detection (catches fast transitions a sparse grid would step over)."""
    frames = outdir / "frames"
    for old in frames.glob("*.jpg"):
        old.unlink()
    frames.mkdir(parents=True, exist_ok=True)

    dur = duration or 0.0
    step = max(5.0, dur / budget) if dur else 30.0
    # -fps_mode vfr is required: ffmpeg 7+ removed -vsync.
    runs = [
        # Store frames at the video's native 720p width, NOT downscaled. A 768px
        # frame makes on-screen chart values unreadable, and an unreadable number
        # is one the reader will guess at — this bit produced a fabricated figure
        # in a report before the frames were extracted at full width.
        ("u", f"fps=1/{step:.2f},scale=1280:-2"),          # uniform grid: the coverage guarantee
        ("s", "select='gt(scene,0.30)',scale=1280:-2"),    # scene changes: transitions a grid steps over
    ]
    times: dict[str, float] = {}
    for prefix, vf in runs:
        proc = subprocess.run(
            ["ffmpeg", "-v", "info", "-y", "-i", str(video), "-vf", f"{vf},showinfo",
             "-fps_mode", "vfr", "-frames:v", str(budget), str(frames / f"{prefix}%04d.jpg")],
            check=False, capture_output=True, text=True,
        )
        for i, t in enumerate(_frame_times(proc.stderr, budget), start=1):
            f = frames / f"{prefix}{i:04d}.jpg"
            if f.exists():
                times[f.name] = t
    have = sorted(frames.glob("*.jpg"), key=lambda p: times.get(p.name, 0.0))
    if not have:
        return {"status": "no_frames", "reason": "ffmpeg produced no frames"}
    have, dropped = dedupe_frames(have)
    if not have:  # dedup is too aggressive for this video; keep the originals
        have = sorted(frames.glob("*.jpg"), key=lambda p: times.get(p.name, 0.0))

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
        f"{len(have)} frames from a {dur / 60:.0f}-minute video: a uniform grid (~{step:.0f}s apart) merged",
        "with scene-change detection, then near-duplicates removed"
        + (f" ({dropped} dropped as visually identical)." if dropped else "."),
        "",
        "**Contact sheets:** " + ", ".join(f"`{s.name}`" for s in sheets) if sheets else "**Contact sheet: none generated**",
        "",
        "Read the sheets to triage, then open individual frames at native resolution — a sheet tile is",
        "never legible enough to transcribe from. Timestamps are measured from the video.", "",
    ]
    for f in ordered:
        index.append(f"- `{f.name}` [{ts(times.get(f.name, 0.0))}]")
    (outdir / "FRAMES.md").write_text("\n".join(index))
    return {
        "status": "ok",
        "frames": len(have),
        "measured_timestamps": sum(1 for f in have if f.name in times),
        "near_duplicates_dropped": dropped,
        "uniform_step_s": round(step, 1),
        "contact_sheets": [s.name for s in sheets],
    }


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
    """Roughly one frame per 30s, clamped, with more frames for longer videos.

    A flat default of 20 gives one frame every 138s on an hour-long talk — fine
    for a speaker, useless for a slide deck. One per 30s keeps a slide talk
    covered without burying the contact sheet.
    """
    d = duration_s or 600
    return int(max(12, min(48, round(d / 30))))


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

        segs = segments_from_chapters(snippets, meta["chapters"], meta.get("duration")) if snippets else []

        if frames_budget > 0:
            if video is None:
                visual = {"status": "failed", "error": "no video downloaded"}
            else:
                try:
                    visual = extract_frames(video, outdir, meta.get("duration"), frames_budget)
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
    }
    print(json.dumps(summary, indent=2))
    return {**summary, "exit": {"rate_limited": 3, "no_transcript": 2}.get(status, 0)}


if __name__ == "__main__":
    raise SystemExit(main())
