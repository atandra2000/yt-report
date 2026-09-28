"""OCR as an evidence-producing subsystem: read the pixels, keep the provenance.

The frame index said only "this frame exists". This turns it into "this frame
contains these observable pieces of text", each read traceable to a frame file,
a timestamp the sampler measured, and the backend that produced it.

Four rules the shape exists to hold:

* **Never fabricate.** An unreadable frame yields `ocr_text=None` and an explicit
  status. There is no path from "no text found" to a guessed string, because a
  guessed figure is exactly the defect this layer exists to remove.
* **Never upscale and claim more.** Vision is handed the extracted frame. Where a
  second pass re-reads the same pixels at a larger input size, boxes are mapped
  back to native pixel coordinates and `read_method` records it — a bigger
  rendering of the same pixels is not more information.
* **A read is not a fact.** `verified_verbatim` is a claim about the read, and a
  low-confidence read cannot make it. The constructor enforces that rather than
  trusting the caller.
* **Provenance points at the source object.** A provider receives an image path
  and returns a result; it never sees a timestamp. Timestamps come from the frame
  index, so an OCR pass cannot invent or repair one.

The provider boundary is one method wide because that is all a backend is: a path
in, a result out. `UnavailableProvider` is a real implementation, not a stub —
OCR being absent degrades a run, and a run that silently behaved as though OCR had
happened is the failure this module must not have.
"""

#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10.0"]
# ///
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

CACHE_DIR = Path(os.environ.get("YT_REPORT_CACHE", Path.home() / ".cache" / "yt-report"))
HELPER_SRC = Path(__file__).parent / "ocr_vision.swift"
HELPER_BIN = CACHE_DIR / "ocr" / "ocr_vision"

# --- confidence policy ------------------------------------------------------
#
# Thresholds are set from what the backend actually returns, not from a habit.
# Measured over 122 shipped frames from two videos: 91% of detections came back
# at 1.0, 5.7% at 0.5, 2.4% at 0.3 (Vision reports a coarse confidence, not a
# continuous one). So 0.80 separates "the engine is sure" from "the engine is
# unsure" with a wide margin on either side, and the 0.5 cluster lands in
# `medium`, not in `high` — which is what keeps a 0.5 read from becoming
# `verified_verbatim`.
HIGH_CONFIDENCE = 0.80
MEDIUM_CONFIDENCE = 0.50

#: Statuses a result may carry. `no_text` is a real answer (the frame was read
#: and held nothing legible); `unavailable` and `failed` are answers about the
#: pass rather than the frame, and a caller must be able to tell them apart. The
#: canonical vocabulary lives in `evidence_model`, which is what the validator
#: checks against; this tuple is the local guard on construction.
OCR_STATUSES = ("ok", "no_text", "repeat", "unavailable", "failed", "unreadable", "skipped")

#: read_method values that mean "these pixels were rendered larger for the
#: engine's benefit", which the provenance carries so a reader never mistakes an
#: upscaled re-read for extra source detail.
UPSCALED_READ_METHOD = "upscale-2x"
NATIVE_READ_METHOD = "native"

#: The engine re-reads at 2x only for frames whose mean confidence is at or below
#: this. On a 51-frame screen-recording set this fired on 8 frames and lifted
#: mean confidence 0.784 -> 0.881 with ~17% more characters recovered; applying
#: it to every frame moved nothing measurable and cost a second pass over all of
#: them, so it is gated rather than unconditional.
REREAD_BELOW = 0.90
_UPSCALE = 2

#: Boxes per frame kept in `visual.json`. A dashboard frame can produce hundreds
#: of detections; the frame's own `ocr_text` is the joined text, and regions
#: exist for later evidence extraction, not to be a full re-transcription.
MAX_REGIONS = 400

_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Whitespace- and case-normalised form used only for dedup comparison.

    Deliberately narrow. It collapses whitespace and case and nothing else, so
    `$500` and `$5,000`, `7B` and `70B`, `0.20` and `2.0`, `32 GB` and `64 GB`
    can never compare equal: those pairs differ in characters this function does
    not touch. A more aggressive normaliser is how a dedup pass turns two
    different prices into one.
    """
    return _WS.sub(" ", (s or "").strip()).casefold()


@dataclass
class OCRRegion:
    """One detected text run: what was read, how sure, and where.

    `bbox` is `[x, y, w, h]` in pixels of the ORIGINAL frame, top-left origin.
    It is not in the model's canonical `Provenance` because a box is a property
    of a detection, not of a citation: a citation to the frame does not need to
    name where on the frame the words were, and making it do so would let a
    citation carry a geometry that nothing validates against the image.
    """

    text: str
    confidence: float
    bbox: list[float]
    read_method: str = NATIVE_READ_METHOD

    def to_dict(self) -> dict:
        return {"text": self.text, "confidence": round(float(self.confidence), 4),
                "bbox": [round(float(v), 2) for v in self.bbox], "read_method": self.read_method}

    @classmethod
    def from_dict(cls, d: dict) -> "OCRRegion":
        return cls(text=d.get("text", ""), confidence=float(d.get("confidence", 0.0)),
                   bbox=list(d.get("bbox") or []), read_method=d.get("read_method", NATIVE_READ_METHOD))

    @property
    def confidence_band(self) -> str:
        if self.confidence >= HIGH_CONFIDENCE:
            return "high"
        if self.confidence >= MEDIUM_CONFIDENCE:
            return "medium"
        return "low"


@dataclass
class OCRResult:
    """What one backend said about one frame.

    `text` is the joined detection text; `None` means nothing was read, and
    `status` says why. A result with an empty `text` and status `ok` cannot be
    constructed — the pair is contradictory, and the contradiction is caught here
    rather than by a validator three files away.
    """

    text: str | None
    confidence: float | None
    regions: list[OCRRegion]
    status: str
    backend: str
    backend_version: str = "unknown"
    width: int | None = None
    height: int | None = None
    read_method: str = NATIVE_READ_METHOD
    elapsed_ms: float = 0.0
    cached: bool = False
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.status not in OCR_STATUSES:
            raise ValueError(f"unknown OCR status {self.status!r}")
        if self.status == "ok" and not (self.text or "").strip():
            raise ValueError("OCRResult status='ok' with no text is contradictory; use 'no_text'")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"OCR confidence {self.confidence} outside [0,1]")

    @property
    def confidence_band(self) -> str:
        """`none` for a frame that was never read, so an unread frame is not
        reported as a low-confidence one — those are different failures."""
        if self.confidence is None:
            return "none"
        if self.confidence >= HIGH_CONFIDENCE:
            return "high"
        if self.confidence >= MEDIUM_CONFIDENCE:
            return "medium"
        return "low"

    def to_dict(self) -> dict:
        d: dict = {"status": self.status, "backend": self.backend,
                   "backend_version": self.backend_version, "text": self.text,
                   "confidence": (round(float(self.confidence), 4)
                                  if self.confidence is not None else None),
                   "confidence_band": self.confidence_band,
                   "read_method": self.read_method, "regions": [r.to_dict() for r in self.regions]}
        if self.width:
            d["width"] = self.width
        if self.height:
            d["height"] = self.height
        if self.cached:
            d["cached"] = True
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "OCRResult":
        return cls(
            text=d.get("text"), confidence=d.get("confidence"),
            regions=[OCRRegion.from_dict(r) for r in d.get("regions") or []],
            status=d.get("status", "unavailable"), backend=d.get("backend", "unknown"),
            backend_version=d.get("backend_version", "unknown"),
            width=d.get("width"), height=d.get("height"),
            read_method=d.get("read_method", NATIVE_READ_METHOD),
            cached=bool(d.get("cached", False)), notes=d.get("notes"),
        )


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #

class OCRProvider:
    """A backend that turns an image file into an `OCRResult`.

    Deliberately minimal. A provider knows about pixels and nothing else — no
    frame ids, no timestamps, no evidence document — because the one thing a
    text extractor must not do is decide what a number means or where it sits in
    the video. That judgement belongs to the evidence layer, which can be
    validated.
    """

    name = "abstract"

    @property
    def available(self) -> bool:
        raise NotImplementedError

    def extract(self, image_path: str | Path) -> OCRResult:
        raise NotImplementedError

    def extract_many(self, image_paths: list[str]) -> dict[str, OCRResult]:
        """Read several frames. The default is the naive loop, and a backend that
        can amortise start-up across a batch overrides it. Both are correct; the
        batch form exists because a per-frame process would pay framework load on
        every frame to do microseconds of work."""
        return {str(p): self.extract(p) for p in image_paths}


class UnavailableProvider(OCRProvider):
    """The provider used when no backend is installed.

    It exists so "OCR did not run" is a value in the artifact rather than an
    absence of one. A frame read by this provider carries
    `status="unavailable"` and `ocr_text=None`, and the run records that OCR was
    degraded — the transcript and the rest of the evidence are still written.
    """

    name = "unavailable"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    @property
    def available(self) -> bool:
        return False

    def extract(self, image_path: str | Path) -> OCRResult:
        return OCRResult(text=None, confidence=None, regions=[], status="unavailable",
                         backend=self.name, notes=self.reason[:200])


class VisionProvider(OCRProvider):
    """macOS Vision, driven through a compiled Swift helper.

    Vision is the backend because it ships with the OS: no model download, no
    dependency to pin, no per-run install, and it is fast enough to read a frame
    in tens of milliseconds. The cost is that it exists only on macOS, which is
    why this is behind the provider boundary rather than called directly.

    The helper is compiled on first use and cached next to the OCR cache, keyed
    on the source hash so editing the Swift invalidates the binary. Compilation
    failing is an unavailable backend, not a crash.
    """

    name = "apple-vision"

    def __init__(self, languages: list[str] | None = None, min_text_height: float = 0.0) -> None:
        self.languages = languages or ["en-US"]
        self.min_text_height = min_text_height
        self._version = "unknown"
        self._build_error: str | None = None
        #: Where the last `extract_many` spent its wall time, in seconds. Reset at
        #: the start of every batch, because a run that merges two batches into
        #: one number cannot say which of them was slow — and "OCR took 631s" is
        #: exactly the kind of figure that is unanswerable without this. The keys
        #: are the same ones `run_ocr` publishes, so a reader compares one table
        #: rather than two vocabularies.
        self.timings: dict = {}

    @property
    def available(self) -> bool:
        return self._ensure_binary() is not None

    @property
    def version(self) -> str:
        self._ensure_binary()
        return self._version

    def _ensure_binary(self) -> Path | None:
        """Compile the helper if needed. Returns None when it cannot be built."""
        started = time.perf_counter()
        if not shutil.which("swiftc") or sys.platform != "darwin":
            self._build_error = "swiftc/macOS not available"
            return None
        key = hashlib.sha256(HELPER_SRC.read_bytes()).hexdigest()[:16]
        binpath = CACHE_DIR / "ocr" / f"ocr_vision-{key}"
        if binpath.exists():
            self._version = binpath.name
            self.timings["vision_helper_compile_s"] = 0.0
            self.timings["vision_helper_startup_s"] = round(time.perf_counter() - started, 4)
            return binpath
        try:
            binpath.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(["swiftc", "-O", "-o", str(binpath), str(HELPER_SRC)],
                                  capture_output=True, text=True, timeout=300)
            if proc.returncode != 0 or not binpath.exists():
                self._build_error = f"swiftc failed: {(proc.stderr or '')[-200:]}"
                return None
        except (OSError, subprocess.SubprocessError) as exc:
            self._build_error = str(exc)[:200]
            return None
        self._version = binpath.name
        self.timings["vision_helper_compile_s"] = round(time.perf_counter() - started, 4)
        return binpath

    def _run(self, image_paths: list[str]) -> dict:
        """One helper process for a batch of frames, timed where it is spent.

        `vision_request_s` is the whole round trip: process spawn, framework
        load, recognition, and JSON back. It is deliberately not split further,
        because the helper is one black box and a split that cannot be measured
        honestly is a number nobody can defend. What IS separable is the cost
        that is *not* recognition — the compile, above, and the per-batch spawn
        in `vision_helper_startup_s` — and those are the two that a batching or
        caching mistake would inflate.
        """
        ensure_started = time.perf_counter()
        binary = self._ensure_binary()
        if binary is None:
            raise RuntimeError(f"vision helper unavailable: {self._build_error}")
        startup_s = time.perf_counter() - ensure_started
        job = {"images": image_paths, "languages": self.languages,
               "min_height": self.min_text_height, "language_correction": True}
        request_started = time.perf_counter()
        proc = subprocess.run([str(binary)], input=json.dumps(job),
                              capture_output=True, text=True, timeout=600)
        request_s = time.perf_counter() - request_started
        if proc.returncode != 0:
            raise RuntimeError(f"ocr_vision exit {proc.returncode}: {(proc.stderr or '')[:200]}")
        self.timings["vision_helper_startup_s"] = round(
            self.timings.get("vision_helper_startup_s", 0.0) + startup_s, 4)
        self.timings["vision_request_s"] = round(
            self.timings.get("vision_request_s", 0.0) + request_s, 4)
        self.timings["vision_requests"] = self.timings.get("vision_requests", 0) + 1
        return json.loads(proc.stdout)["results"]

    @staticmethod
    def _regions(raw: list[dict], scale: float, read_method: str,
                 width: int | None = None, height: int | None = None) -> list[OCRRegion]:
        """Map helper boxes back to native frame pixels.

        `scale` is the factor the image was rendered at for the engine. A box
        from an upscaled re-read describes the same pixels as a smaller box on
        the original, so dividing keeps the geometry honest: a region never
        claims to cover pixels the source frame does not have.

        Boxes are clamped to the frame because the engine's normalised float
        arithmetic lands a hair outside the edge on text flush against it
        (`x = -1.07` on a real detection). An unclamped value is a region
        claiming to cover pixels the frame does not have, and it would teach a
        reader to ignore a bounds check that is otherwise worth having.
        """
        out = []
        for b in raw:
            x, y, w, h = (v / scale for v in (b.get("bbox") or [0, 0, 0, 0]))
            if width and height:
                x, y = max(0.0, x), max(0.0, y)
                w, h = min(w, width - x), min(h, height - y)
            if w <= 0 or h <= 0:
                continue  # a box clamped to nothing is not a detection
            out.append(OCRRegion(text=(b.get("text") or "").strip(),
                                 confidence=float(b.get("confidence", 0.0)),
                                 bbox=[x, y, w, h],
                                 read_method=read_method))
        return out

    @staticmethod
    def _mean_confidence(regions: list[OCRRegion]) -> float | None:
        if not regions:
            return None
        # Mean over regions, not a minimum: one mis-read sidebar label should not
        # declare an otherwise clean slide unreadable. The low tail is still
        # visible per-region, and the frame band is what gates verbatim claims.
        return round(sum(r.confidence for r in regions) / len(regions), 4)

    def extract(self, image_path: str | Path) -> OCRResult:
        return self._extract_many([str(image_path)])[str(image_path)]

    def extract_many(self, image_paths: list[str]) -> dict[str, OCRResult]:
        return self._extract_many(image_paths)

    def _extract_many(self, image_paths: list[str]) -> dict[str, OCRResult]:
        """One process for the whole batch — and one for the re-reads.

        Per-image processes would pay framework load per frame; a 70-frame video
        would pay it 70 times for work that is microseconds each.

        The re-read used to be the same mistake a second time, and it was the
        dominant cost on a real 126-frame lecture: 53 frames fell under
        `REREAD_BELOW` and each spawned its own helper process, so 53 of the 54
        spawns in the whole pass were re-reads and 19.3 of its 31.6 seconds went
        into them. Batching them is the same fix, applied to the second loop
        that had been left un-batched.
        """
        started = time.perf_counter()
        # Reset per batch, but keep the compile cost: `available` compiles on a
        # cold cache before this runs, and dropping that number would make the
        # one-time build cost invisible in the one run that pays it.
        self.timings = {"vision_rereads": 0, "vision_reread_s": 0.0,
                        "vision_helper_compile_s": self.timings.get("vision_helper_compile_s", 0.0)}
        raw = self._run(image_paths)
        first: dict[str, tuple[list[OCRRegion], dict]] = {}
        reread_paths: list[str] = []
        for path in image_paths:
            r = raw.get(path) or {}
            status = r.get("status", "failed")
            w, h = r.get("width"), r.get("height")
            regions = self._regions(r.get("boxes") or [], 1.0, NATIVE_READ_METHOD, w, h)
            first[path] = (regions, r)
            if status == "ok" and regions and (self._mean_confidence(regions) or 1.0) < REREAD_BELOW:
                reread_paths.append(path)

        reread_started = time.perf_counter()
        rereads = self._reread(reread_paths) if reread_paths else {}
        self.timings["vision_rereads"] = len(reread_paths)
        self.timings["vision_reread_s"] = round(time.perf_counter() - reread_started, 4)

        out: dict[str, OCRResult] = {}
        for path in image_paths:
            regions, r = first[path]
            method = NATIVE_READ_METHOD
            reread = rereads.get(path)
            if reread is not None and reread.regions and len(reread.regions) >= len(regions):
                # Keep the re-read only when it recovered at least as much text.
                # A re-read that returns LESS is a worse read of the same pixels,
                # and swapping it in would lose real detections.
                regions, method = reread.regions, UPSCALED_READ_METHOD
            elapsed = round((time.perf_counter() - started) * 1000 / max(1, len(image_paths)), 2)
            text = "\n".join(rg.text for rg in regions if rg.text).strip() or None
            out[path] = OCRResult(
                text=text, confidence=self._mean_confidence(regions), regions=regions,
                status="ok" if text else "no_text", backend=self.name,
                backend_version=self._version, width=r.get("width"), height=r.get("height"),
                read_method=method, elapsed_ms=elapsed)
        return out

    def _reread(self, image_paths: list[str]) -> dict[str, OCRResult]:
        """Re-read several frames at 2x in one pass. Best-effort throughout.

        This is a rendering of the SAME pixels. The regions are divided back to
        native coordinates and marked `upscale-2x`, so nothing downstream can
        mistake a larger rendering for detail the frame never had.

        Every frame that cannot be re-read is simply absent from the result, and
        the caller keeps its first read — the same all-or-nothing-per-frame
        behaviour the single-frame version had. A failure anywhere in the batch
        (no Pillow, a full disk, a helper crash) degrades to "no re-reads at
        all" rather than to a partial one, because a re-read set that is
        quietly shorter than the set that asked for it is a result nobody can
        reproduce.
        """
        if not image_paths:
            return {}
        try:
            from PIL import Image  # noqa: PLC0415 — optional, only for the re-read
        except ImportError:
            return {}
        tmpdir = HELPER_BIN.parent
        staged: dict[str, Path] = {}
        try:
            tmpdir.mkdir(parents=True, exist_ok=True)
            for image_path in image_paths:
                digest = hashlib.sha256(str(image_path).encode()).hexdigest()[:12]
                tmp = tmpdir / f"reread-{digest}.png"
                with Image.open(image_path) as im:
                    native = im.size
                    im.convert("RGB").resize((native[0] * _UPSCALE, native[1] * _UPSCALE),
                                             Image.LANCZOS).save(tmp)
                staged[image_path] = tmp
            raw = self._run([str(p) for p in staged.values()])
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
            return {}
        finally:
            for tmp in staged.values():
                tmp.unlink(missing_ok=True)

        out: dict[str, OCRResult] = {}
        for image_path, tmp in staged.items():
            r = raw.get(str(tmp)) or {}
            if r.get("status") != "ok":
                continue
            with Image.open(image_path) as im:
                native = im.size
            # Clamp against the NATIVE size: the boxes have just been divided
            # back down by _UPSCALE, and the helper reported the size of the
            # 2x rendering it actually read.
            regions = self._regions(r.get("boxes") or [], _UPSCALE, UPSCALED_READ_METHOD,
                                    native[0], native[1])
            if not regions:
                continue
            out[image_path] = OCRResult(
                text="\n".join(rg.text for rg in regions if rg.text).strip() or None,
                confidence=self._mean_confidence(regions), regions=regions,
                status="ok", backend=self.name, backend_version=self._version,
                read_method=UPSCALED_READ_METHOD)
        return out


def get_provider(prefer: str = "auto") -> OCRProvider:
    """Pick a backend. `prefer` names one; `auto` takes the first that works.

    Falling back rather than failing is the point: a run on a machine without
    Vision must still produce transcript, frames and evidence, with OCR recorded
    as unavailable.
    """
    order = ([prefer] if prefer and prefer != "auto" else ["apple-vision"])
    for name in order:
        if name == "apple-vision":
            p = VisionProvider()
            if p.available:
                return p
    return UnavailableProvider("no OCR backend available (apple-vision requires macOS and swiftc)")


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #

def _config_key(provider: OCRProvider) -> str:
    """Everything about the backend that changes the answer.

    Included: backend identity and version, languages, and the read threshold.
    Excluded: file paths, timestamps, frame ids. A cache that invalidated on
    those would miss on every re-run and be worse than no cache.
    """
    parts = [provider.name, getattr(provider, "version", "unknown"),
             ",".join(getattr(provider, "languages", []) or []),
             str(getattr(provider, "min_text_height", 0.0)),
             f"reread<{REREAD_BELOW}x{_UPSCALE}", str(HIGH_CONFIDENCE), str(MEDIUM_CONFIDENCE)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _cache_file(image_path: Path, provider: OCRProvider) -> Path:
    """Cache key = content hash + config key.

    Content, not path: re-extracting the same video into a new directory
    re-OCRs nothing, and editing a frame in place invalidates exactly that frame.
    """
    digest = hashlib.sha256()
    with image_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return CACHE_DIR / "ocr" / f"{digest.hexdigest()[:24]}-{_config_key(provider)}.json"


def extract_many_cached(provider: OCRProvider, image_paths: list[str],
                        stats: dict | None = None) -> dict[str, OCRResult]:
    """Read every path, answering from cache where the content and config allow.

    A cache miss costs one OCR call and is the normal path on a first run; the
    cost this avoids is re-reading every frame whenever a report is regenerated,
    which is the common case and the one that made OCR feel unaffordable. Misses
    are batched through the provider so the cache never turns one read per frame
    into one process per frame.

    `stats`, when given, is filled in place with the accounting a caller needs to
    tell a cold run from a warm one: `cache_hits`, `cache_misses` and
    `ocr_call_s` (wall time actually spent inside the provider). Hits and misses
    are counted here rather than derived by the caller, because a hit rate
    computed from the cache's own opinion of itself is not a measurement — the
    same rule `JsonCache.get_or_call` follows by returning the flag.
    """
    if not provider.available:
        if stats is not None:
            stats.update({"cache_hits": 0, "cache_misses": len(image_paths), "ocr_call_s": 0.0})
        return {p: provider.extract(p) for p in image_paths}

    out: dict[str, OCRResult] = {}
    misses: list[str] = []
    cache_files: dict[str, Path] = {}
    for p in image_paths:
        cf = _cache_file(Path(p), provider)
        cache_files[p] = cf
        if not cf.exists():
            misses.append(p)
            continue
        try:
            hit = OCRResult.from_dict(json.loads(cf.read_text()))
            hit.cached = True
            out[p] = hit
        except (OSError, ValueError, json.JSONDecodeError):
            # A corrupt entry is a miss, never a crash — and it is left on disk
            # only until this read overwrites it.
            misses.append(p)

    if stats is not None:
        stats["cache_hits"] = len(out)
        stats["cache_misses"] = len(misses)
    if misses:
        call_started = time.perf_counter()
        try:
            fresh = provider.extract_many(misses)
        except Exception as exc:  # noqa: BLE001 — a backend failure degrades, never aborts
            fresh = {p: OCRResult(text=None, confidence=None, regions=[], status="failed",
                                  backend=provider.name, notes=str(exc)[:200]) for p in misses}
        if stats is not None:
            stats["ocr_call_s"] = round(time.perf_counter() - call_started, 3)
        for p, res in fresh.items():
            out[p] = res
            _store(cache_files[p], res)
    elif stats is not None:
        stats["ocr_call_s"] = 0.0
    return out


def _store(cache_file: Path, result: OCRResult) -> None:
    """Write one cache entry. Failure to cache costs time next run, not the run."""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(result.to_dict()))
        tmp.replace(cache_file)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #

@dataclass
class OCRObservation:
    """One frame's read, or a reference to an identical read already recorded.

    Adjacent frames of a held slide carry the same text, and writing it once per
    frame is what makes an OCR artifact unreadable. Grouping keeps the text once
    and every frame that showed it, so a reader can still answer "which frames
    showed 32 GB" exactly.

    `text` is set on the group's first occurrence only. A later frame with the
    same text sets `same_as` and carries no text of its own — the observation is
    the group's, the frame's own entry says which group it belongs to.
    """

    frame_id: str
    t_start: float | None = None
    result: OCRResult | None = None
    same_as: str | None = None
    repeat: bool = False

    def to_dict(self) -> dict:
        d: dict = {"frame_id": self.frame_id, "t_start": self.t_start}
        if self.result is not None:
            d["result"] = self.result.to_dict()
            d["text"] = self.result.text
        if self.same_as:
            d["same_as"] = self.same_as
        if self.repeat:
            d["repeat"] = True
        return d


def group_observations(frames: list[dict], results: dict[str, OCRResult]) -> list[OCRObservation]:
    """Build per-frame observations, grouping identical consecutive text.

    `results` is keyed by frame id, not by path: the grouping is a statement
    about frames in the index, and a path-keyed map would make it silently
    produce nothing for any caller that did not happen to set one.

    Grouping is on the normalised text and nothing else, and only for frames
    adjacent in the index: a slide that returns at 04:00 and 12:00 with different
    slides between them is not the same continuous observation, and pretending
    otherwise would claim a persistence the frames do not show. Normalisation
    cannot equate `$500` with `$5,000` or `7B` with `70B` — see
    `normalize_text`.
    """
    out: list[OCRObservation] = []
    last_key: str | None = None
    last_id: str | None = None
    for f in frames:
        fid = str(f.get("frame_id") or Path(f.get("file", "")).stem)
        res = results.get(fid)
        t = f.get("t_start")
        t_start = round(float(t), 3) if isinstance(t, (int, float)) else None
        key = normalize_text(res.text) if (res and res.text) else None
        if key and key == last_key and last_id:
            out.append(OCRObservation(frame_id=fid, t_start=t_start, same_as=last_id))
        else:
            out.append(OCRObservation(frame_id=fid, t_start=t_start, result=res))
            if key:
                last_key, last_id = key, fid
            else:
                last_key, last_id = None, None
    return out


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

def ocr_timings(provider: OCRProvider, frames_total: int, wall_s: float,
                stats: dict) -> dict:
    """The cost of one OCR pass, split so that a slow run can be attributed.

    Every number here answers "where did the time go", and the split is the
    point: `final_ocr_s` on its own cannot distinguish recognition from subprocess
    churn from a cache that stopped working, and those three have three
    different fixes. `ocr_call_s` is wall time inside the provider, so anything
    outside it — hashing every frame to build a cache key, reading and writing
    the cache, grouping, summarising — is `python_overhead_s`. A run whose
    `python_overhead_s` is a large share of the total is a run whose cost is the
    cache, not the recogniser.

    The probe pass reports the same field names under a `probe_` prefix in the
    sampling metrics, so the two passes are comparable without either being
    renamed to fit the other.
    """
    call_s = float(stats.get("ocr_call_s", 0.0))
    backend_timings = getattr(provider, "timings", {}) or {}
    return {
        "final_frames": frames_total,
        "final_ocr_s": round(wall_s, 3),
        "final_ocr_s_per_frame": round(wall_s / frames_total, 4) if frames_total else None,
        "ocr_cache_hits": int(stats.get("cache_hits", 0)),
        "ocr_cache_misses": int(stats.get("cache_misses", 0)),
        "ocr_backend_call_s": round(call_s, 3),
        "vision_helper_compile_s": round(float(backend_timings.get("vision_helper_compile_s", 0.0)), 3),
        "vision_helper_startup_s": round(float(backend_timings.get("vision_helper_startup_s", 0.0)), 3),
        "vision_request_s": round(float(backend_timings.get("vision_request_s", 0.0)), 3),
        "vision_requests": int(backend_timings.get("vision_requests", 0)),
        "vision_rereads": int(backend_timings.get("vision_rereads", 0)),
        "vision_reread_s": round(float(backend_timings.get("vision_reread_s", 0.0)), 3),
        "python_overhead_s": round(max(0.0, wall_s - call_s), 3),
    }


def run_ocr(artifact_dir: str | Path, frames: list[dict], provider: OCRProvider | None = None,
            progress: bool = False) -> dict:
    """OCR every frame in `frames`, returning the artifact payload.

    Writes nothing and raises nothing for a backend failure: `visual.json`'s
    `summary.status` is the machine-readable answer, and the caller decides what
    a degraded run means.

    Every frame is read exactly once. Cached reads are not reads; a frame that
    misses is batched into a single backend call; and the low-confidence re-read
    is the only second pass, and it is counted in `vision_rereads` so a run that
    is quietly re-reading everything is visible rather than merely slow.
    """
    d = Path(artifact_dir)
    provider = provider or get_provider()
    started = time.perf_counter()
    results: dict[str, OCRResult] = {}
    missing: list[str] = []
    for f in frames:
        fid = str(f.get("frame_id") or Path(f.get("file", "")).stem)
        abs_path = str(d / (f.get("file") or f"frames/{fid}.jpg"))
        f["abs_path"] = abs_path
        # Checked before the backend is consulted: a file that is not on disk is
        # a fact about the run that no backend is needed to establish, and
        # reporting it as "unavailable" would blame the pass for a missing frame.
        if not Path(abs_path).exists():
            missing.append(abs_path)

    stats: dict = {}
    if provider.available:
        results.update(extract_many_cached(provider, [f["abs_path"] for f in frames
                                                      if f["abs_path"] not in missing], stats))
    else:
        results.update({p: provider.extract(p) for p in (f["abs_path"] for f in frames)})
    for p in missing:
        # A frame the index names but the disk does not hold. `unreadable` says
        # the frame is gone, not that nothing legible was on it — the difference
        # matters because one is a fact about the frame and the other a fact
        # about the pixels.
        results[p] = OCRResult(text=None, confidence=None, regions=[], status="unreadable",
                               backend=provider.name, notes="frame file missing")
    if progress:
        for f in frames:
            print(f"  ocr {Path(f['abs_path']).name}: {results[f['abs_path']].status}",
                  file=os.sys.stderr)

    observations = group_observations(frames, {f["frame_id"]: results.get(f["abs_path"])
                                               for f in frames})
    wall_s = round(time.perf_counter() - started, 3)
    summary = summarize(observations, provider, wall_s)
    summary["timings"] = ocr_timings(provider, len(frames), wall_s, stats)
    return {
        "version": 1,
        "backend": provider.name,
        "backend_version": getattr(provider, "version", "unknown"),
        "config_key": _config_key(provider) if provider.available else None,
        "summary": summary,
        "frames": [o.to_dict() for o in observations],
    }


def summarize(observations: list[OCRObservation], provider: OCRProvider, wall_s: float) -> dict:
    """Counts a reviewer can check, not a claim that OCR worked.

    `text_chars` and `frames_with_text` are what distinguish a real read from a
    backend returning strings that mean nothing; `low_confidence_frames` is where
    a human has to look. `verbatim_eligible` is the count that may carry
    `verified_verbatim`, and it is deliberately smaller than the number of frames
    that produced text.
    """
    read = [o for o in observations if o.result is not None]
    with_text = [o for o in read if o.result.text]
    confs = [o.result.confidence for o in with_text if o.result.confidence is not None]
    return {
        "status": ("ok" if provider.available and read else
                   ("unavailable" if not provider.available else "empty")),
        "backend": provider.name,
        "frames_total": len(observations),
        "frames_read": len(read),
        "frames_with_text": len(with_text),
        "frames_no_text": len([o for o in read if not o.result.text and o.result.status == "no_text"]),
        "frames_repeated": len([o for o in observations if o.same_as]),
        "low_confidence_frames": len([c for c in confs if c < HIGH_CONFIDENCE]),
        "medium_confidence_frames": len([c for c in confs if MEDIUM_CONFIDENCE <= c < HIGH_CONFIDENCE]),
        "verbatim_eligible": len([c for c in confs if c >= HIGH_CONFIDENCE]),
        "text_chars": sum(len(o.result.text) for o in with_text),
        "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
        "ocr_wall_s": wall_s,
        "notes": provider.reason if isinstance(provider, UnavailableProvider) else None,
    }


def skipped_payload(frames: list[dict], reason: str) -> dict:
    """The artifact for a run where OCR was deliberately not attempted.

    Written rather than omitted, so `--no-ocr` leaves the same set of files with
    the same shape. A missing `visual.json` is indistinguishable from a pipeline
    that never had an OCR stage; this one says the pass was turned off and by
    what argument, which is the only thing a later reader can act on.
    """
    return {
        "version": 1,
        "backend": None,
        "backend_version": None,
        "config_key": None,
        "summary": {"status": "skipped", "reason": reason, "frames_total": len(frames),
                    "frames_read": 0, "frames_with_text": 0, "text_chars": 0},
        "frames": [{"frame_id": str(f.get("frame_id") or Path(f.get("file", "")).stem),
                    "t_start": f.get("t_start"),
                    "result": {"status": "skipped", "backend": None, "text": None,
                               "confidence": None, "regions": []}} for f in frames],
    }


def write_visual(artifact_dir: str | Path, payload: dict) -> Path:
    p = Path(artifact_dir) / "visual.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(p)
    return p


def load_visual(artifact_dir: str | Path) -> dict | None:
    p = Path(artifact_dir) / "visual.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main(argv: list[str] | None = None) -> int:
    """Run the pass over an artifact directory that already has frames.

    Separate from `yt_fetch.py` because the expensive half of a run — the ASR and
    the frame extraction — should not have to be repeated to add a text layer to
    frames already on disk. Exit 2 when no backend is available, so a caller can
    tell "OCR degraded" from "OCR ran".
    """
    import argparse

    ap = argparse.ArgumentParser(description="OCR the frames in an artifact directory into visual.json")
    ap.add_argument("artifact_dir", help="directory holding frames/ and frames.json")
    ap.add_argument("--backend", default="auto", choices=["auto", "apple-vision"])
    ap.add_argument("--quiet", action="store_true", help="suppress per-frame progress")
    args = ap.parse_args(argv)

    import build_evidence as B  # noqa: PLC0415 — only needed to read the frame index

    d = Path(args.artifact_dir).expanduser()
    frames = B.load_frames_index(d)
    if not frames:
        print(f"no frames found in {d}", file=os.sys.stderr)
        return 1
    provider = get_provider(args.backend)
    payload = run_ocr(d, frames, provider, progress=not args.quiet)
    write_visual(d, payload)
    print(json.dumps(payload["summary"], indent=2))
    return 0 if provider.available else 2


if __name__ == "__main__":
    raise SystemExit(main())
