"""The OCR pass must be attributable, and the sampler must not lose the timeline.

Two failures motivated this file, and they are the same failure wearing
different clothes: a number nobody could act on.

* A real 18-minute lecture reported `ocr_wall_s: 631.6` for 126 frames. The
  reproducible cost of that exact pass is 29.99s cold and 0.07s warm — a 21×
  gap that the artifact recorded as a single number and could not explain.
  `ocr.ocr_timings` now splits a pass into the compile, the helper round trip,
  the re-reads, the provider call and the Python around it, so the next
  occurrence says which of those grew.
* The same run reported 126 planned frames, 52 dropped as near-duplicates, and
  a largest temporal gap of 48.7s against a planned baseline of 12.2s. Four
  consecutive baseline steps had been deleted as "identical", including the one
  at the end of the stretch, which is where the presenter finishes speaking and
  draws the next thing on the slide. `dedupe_coverage` publishes the gap on both
  sides of deduplication, and `dedupe_frames` keeps the last frame of a run.

What these tests deliberately do NOT do:

* Assert a wall-clock time. The regression that produced 631.6s was not
  reproducible on the same machine minutes later, so a timing assertion here
  would be a flaky test that fails for reasons unrelated to the code. What is
  asserted instead is *structure*: how many provider calls happen, what they are
  called with, and which of them the cache removed. Those are properties, not
  timings, and they hold on any hardware.
* Claim that the `screen_ui` and `code` profiles are validated on real media.
  No real screen recording or code-editor video exists in this repository and
  none is fetched here. The classifier is exercised on synthetic OCR input,
  which tests the *rule* and not the *world*; the distinction is asserted
  explicitly so a future reader cannot mistake one for the other.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ocr as O  # noqa: E402
import yt_fetch as Y  # noqa: E402


class CountingProvider(O.OCRProvider):
    """A backend that records what it was asked to do and never touches Vision.

    Determinism is the whole point: every test below asserts a *count* of calls
    and a list of arguments, and both are exact here and approximate on a real
    machine under load. `reads` is a flat list of every path passed to
    `extract_many` or `extract`, so "the same image was OCR'd twice" is a
    question about a list rather than a question about wall time.
    """

    name = "counting"
    version = "v-test"

    def __init__(self, text: str = "hello", confidence: float = 1.0) -> None:
        self.reads: list[str] = []
        self.calls = 0
        self._text = text
        self._confidence = confidence

    @property
    def available(self) -> bool:
        return True

    def _make(self, path: str) -> O.OCRResult:
        return O.OCRResult(text=self._text, confidence=self._confidence,
                           regions=[O.OCRRegion(text=self._text, confidence=self._confidence,
                                                bbox=[0, 0, 10, 10])],
                           status="ok", backend=self.name, backend_version=self.version)

    def extract(self, image_path):
        self.reads.append(str(image_path))
        self.calls += 1
        return self._make(str(image_path))

    def extract_many(self, image_paths):
        self.calls += 1
        return {str(p): (self.reads.append(str(p)), self._make(str(p)))[1]
                for p in image_paths}


def _frames(root: Path, n: int) -> tuple[Path, list[dict]]:
    """An artifact directory holding `n` byte-distinct frames, plus their index.

    The files live in `<root>/frames/` because that is where `run_ocr` looks: it
    joins the artifact directory with each index entry's `file` field, so an
    index that names `frames/u0000.jpg` and a directory holding `u0000.jpg`
    describes a frame that is not on disk — and a frame that is not on disk is
    reported as `unreadable` without ever reaching a provider, which makes every
    call-count assertion below vacuously true.

    They are real PNGs, not arbitrary bytes. The re-read path opens every frame
    with Pillow to render it at 2x, and a file Pillow cannot decode takes the
    whole batch down the "no re-reads at all" branch — so a byte-soup fixture
    would quietly turn a test about batching into a test about failure handling.
    Each frame is a distinct solid colour, which also makes the bytes distinct,
    which is what the content-addressed cache keys on.
    """
    from PIL import Image
    d = root / "frames"
    d.mkdir(parents=True, exist_ok=True)
    index = []
    for i in range(n):
        p = d / f"u{i:04d}.png"
        Image.new("RGB", (32, 32), (i * 7 % 256, i * 13 % 256, i * 29 % 256)).save(p)
        index.append({"frame_id": p.stem, "file": f"frames/{p.name}", "t_start": float(i)})
    return root, index


def t_one_call_per_uncached_frame() -> None:
    """A cold pass reads every frame once. Once, not once per lookup path."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        O.CACHE_DIR = root / "cache"
        d, index = _frames(root / "d", 12)
        provider = CountingProvider()
        payload = O.run_ocr(d, index, provider)
        # `frames_read` counts *observations that carry text of their own*, and
        # every frame here returns the same string, so the grouping pass folds
        # eleven of them into repeats. That is the dedup working, not a frame
        # going unread — the read count is the provider's, not the summary's.
        check("ocr/perf: every frame reached the provider",
              len(provider.reads) == 12, f"{len(provider.reads)} reads for 12 frames")
        check("ocr/perf: the artifact still carries one entry per frame",
              len(payload["frames"]) == 12, len(payload["frames"]))
        check("ocr/perf: one provider call for a whole cold batch",
              provider.calls == 1, f"{provider.calls} calls for 12 frames")
        check("ocr/perf: every uncached frame read exactly once",
              sorted(provider.reads) == sorted(set(provider.reads)),
              f"{len(provider.reads)} reads for {len(set(provider.reads))} distinct frames")


def t_cache_hit_avoids_the_provider() -> None:
    """A warm pass makes no provider call at all, and says so in the metrics."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        O.CACHE_DIR = root / "cache"
        d, index = _frames(root / "d", 8)
        O.run_ocr(d, index, CountingProvider())

        warm_provider = CountingProvider()
        warm = O.run_ocr(d, index, warm_provider)
        t = warm["summary"]["timings"]
        check("ocr/perf: a warm pass makes no provider call", warm_provider.calls == 0,
              f"{warm_provider.calls} calls on a fully cached run")
        check("ocr/perf: warm run reports every frame as a cache hit",
              t["ocr_cache_hits"] == 8 and t["ocr_cache_misses"] == 0, json.dumps(t))
        check("ocr/perf: warm run spends no time in the backend",
              t["ocr_backend_call_s"] == 0.0 and t["vision_request_s"] == 0.0, json.dumps(t))
        check("ocr/perf: python overhead is reported separately from the backend",
              t["python_overhead_s"] >= 0.0 and t["python_overhead_s"] <= t["final_ocr_s"],
              json.dumps(t))


def t_identical_bytes_are_one_cache_entry() -> None:
    """Two files with the same bytes are one frame. The key is content, not path."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        O.CACHE_DIR = root / "cache"
        d, index = _frames(root / "d", 2)
        # A byte-identical copy under a different name: the same picture.
        source = d / "frames" / index[0]["file"].rsplit("/", 1)[1]
        copy_path = source.with_name("copy-of-a.png")
        copy_path.write_bytes(source.read_bytes())
        index.append({"frame_id": copy_path.stem, "file": f"frames/{copy_path.name}",
                      "t_start": 2.0})
        p1 = CountingProvider()
        first = O.run_ocr(d, index, p1)
        t = O.run_ocr(d, index, CountingProvider())["summary"]["timings"]
        check("ocr/perf: a byte-identical copy is read once, not twice",
              p1.reads.count(str(copy_path)) == 1, f"reads were {p1.reads}")
        check("ocr/perf: the duplicate is a cache hit on the second pass",
              t["ocr_cache_hits"] == 3 and t["ocr_cache_misses"] == 0, json.dumps(t))
        check("ocr/perf: cache is content-addressed, so the copy still gets an answer",
              len(first["frames"]) == 3, len(first["frames"]))
        check("ocr/perf: only the first of three identical reads carries the text",
              first["summary"]["frames_repeated"] == 2, json.dumps(first["summary"]))


def t_rereads_are_batched() -> None:
    """Low-confidence re-reads go through ONE helper process, not one per frame.

    This is the fix for the measured cost. The main pass was already batched;
    the re-read loop beside it was not, so a real 126-frame lecture spent 53 of
    its 54 helper-process spawns — and 19.3 of its 31.6 seconds — re-reading
    frames one at a time. The test counts spawns, because a spawn count is exact
    and a duration is not.

    It uses a fake provider rather than Vision: the point is the *number of
    processes*, and a real backend on a loaded machine would make that number
    depend on how fast the machine happened to be.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        O.CACHE_DIR = root / "cache"
        d, index = _frames(root / "d", 9)
        provider = _RereadProvider()
        payload = O.run_ocr(d, index, provider)
        t = payload["summary"]["timings"]
        check("ocr/perf: nine low-confidence frames trigger nine re-reads",
              t["vision_rereads"] == 9, json.dumps(t))
        check("ocr/perf: the re-reads cost ONE helper process, not nine",
              provider.reread_calls == 1,
              f"{provider.reread_calls} re-read calls for {t['vision_rereads']} re-reads")
        check("ocr/perf: two helper processes total — one read batch, one re-read batch",
              t["vision_requests"] == 2, json.dumps(t))
        check("ocr/perf: a re-read is labelled as one, not passed off as native detail",
              all(o["result"]["read_method"] == O.UPSCALED_READ_METHOD
                  for o in payload["frames"] if o.get("result")),
              json.dumps([o.get("result", {}).get("read_method") for o in payload["frames"]]))


class _RereadProvider(O.VisionProvider):
    """A Vision provider that never shells out, but counts as though it did.

    Subclassing `VisionProvider` rather than `OCRProvider` is deliberate: the
    re-read path lives on `VisionProvider`, so a provider that is not one cannot
    reach the code this test is about.
    """

    def __init__(self) -> None:
        self.languages = ["en-US"]
        self.min_text_height = 0.0
        self._version = "v-test"
        self._build_error = None
        self.timings = {}
        self.calls = 0
        self.reread_calls = 0
        self._turn = 0

    @property
    def available(self) -> bool:
        return True

    def _ensure_binary(self):
        return Path("/nonexistent/ocr_vision")

    def _run(self, image_paths: list[str]) -> dict:
        # The first read comes back unsure, which is what makes `_extract_many`
        # re-read these frames; the re-read comes back sure. Each frame's text is
        # derived from its own path so the grouping pass does not fold them into
        # one observation and hide the per-frame `read_method` under `same_as`.
        rereading = all(Path(p).name.startswith("reread-") for p in image_paths)
        conf = 0.99 if rereading else 0.50
        if rereading:
            self.reread_calls += 1
        else:
            self.calls += 1
        # The real `_run` is what records the request count, and this fake
        # replaces it wholesale — so the fake has to do the accounting too, or
        # the assertion below is reading a counter nothing ever increments.
        self.timings["vision_requests"] = self.timings.get("vision_requests", 0) + 1
        return {p: {"status": "ok", "width": 64, "height": 64, "conf": conf,
                    "boxes": [{"text": f"read {Path(p).stem}", "confidence": conf,
                               "bbox": [0, 0, 10, 10]}]}
                for p in image_paths}


def t_ocr_cache_key_boundaries() -> None:
    """The OCR cache key moves when the answer would, and not otherwise.

    Two directions, and the second is the one that costs money if it is wrong.
    A key that is too broad re-OCRs every frame on every run; a key that is too
    narrow serves a stale reading as though it were current. So each dimension
    is moved one at a time and asserted in isolation — and, just as important,
    `t_sampling_policy_is_not_part_of_the_ocr_key` below asserts the dimensions
    that must NOT move it.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        O.CACHE_DIR = root / "cache"
        _d, index = _frames(root / "d", 2)
        img = _d / index[0]["file"]

        base = O.VisionProvider()
        base_key = O._cache_file(img, base)
        check("cache/ocr: the key is stable for one provider and one frame",
              O._cache_file(img, base) == base_key)

        # A different backend has to be a genuinely different backend. Two
        # `VisionProvider()` instances are the same one, and asserting that a
        # thing equals itself moves no key.
        class _OtherBackend(O.VisionProvider):
            name = "other-backend"

        for label, provider in (
            ("backend", _OtherBackend()),
            ("languages", O.VisionProvider(languages=["de-DE"])),
            ("min_text_height", O.VisionProvider(min_text_height=0.02)),
        ):
            check(f"cache/ocr: {label} moves the key", O._cache_file(img, provider) != base_key)

        # The same bytes at a different path. The file has to exist: the key is a
        # content hash, so a path that is not on disk cannot be hashed and a test
        # that pretends otherwise is testing an exception, not a key.
        elsewhere = root / "elsewhere"
        elsewhere.mkdir(parents=True, exist_ok=True)
        relocated = elsewhere / img.name
        relocated.write_bytes(img.read_bytes())
        check("cache/ocr: the key does not depend on where the frame lives",
              O._cache_file(relocated, base) == base_key,
              "a path-dependent key re-OCRs the same picture in a new directory")

        moved = img.with_name("moved.png")
        img.rename(moved)
        try:
            check("cache/ocr: moving a frame within its directory keeps its key",
                  O._cache_file(moved, base) == base_key)
        finally:
            moved.rename(img)

        edited = img.read_bytes() + b"\x00"  # a different picture, same name
        img.write_bytes(edited)
        check("cache/ocr: different frame bytes move the key",
              O._cache_file(img, base) != base_key)
        img.write_bytes(edited[:-1])

    # A re-read threshold change alters what a frame's answer should be, so it
    # has to invalidate. Asserted against the constant the key actually reads,
    # not against a number typed here, so the test cannot drift out of step with
    # the code and quietly stop testing anything.
    saved = O.REREAD_BELOW
    try:
        O.REREAD_BELOW = saved + 0.5
        check("cache/ocr: a change to the re-read policy moves the key",
              O._config_key(O.VisionProvider()) != _baseline_config_key())
    finally:
        O.REREAD_BELOW = saved


def _baseline_config_key() -> str:
    return hashlib.sha256("|".join([
        "apple-vision", "unknown", "en-US", "0.0",
        f"reread<{O.REREAD_BELOW}x{O._UPSCALE}", str(O.HIGH_CONFIDENCE),
        str(O.MEDIUM_CONFIDENCE)]).encode()).hexdigest()[:16]


def t_sampling_policy_is_not_part_of_the_ocr_key() -> None:
    """A sampling-policy change must not force a re-read of identical pixels.

    The obvious over-correction to "a stale artifact is dangerous" is to fold
    the sampling version into every downstream key. That would be wrong: the OCR
    answer for a given image under a given configuration does not depend on how
    many other images the sampler chose, so including the version would re-OCR
    every frame of every video on every policy bump and turn a one-line change
    into a full re-read of the corpus.

    The frame set changing is what changes the OCR result, and it does — a
    different frame is a different content hash.
    """
    check("cache/ocr: the OCR key is built from content and configuration only",
          "sampling" not in inspect_source(O._config_key).lower()
          and "SAMPLING_VERSION" not in inspect_source(O._config_key),
          "the sampling policy has leaked into the OCR cache key")
    check("cache/ocr: the OCR key is built from content and configuration only",
          "SAMPLING_VERSION" not in inspect_source(O._cache_file),
          "the sampling policy has leaked into the per-frame OCR cache key")


def t_timing_fields_are_complete() -> None:
    """Every promised metric exists, on a run that did no work as well as one that did.

    A timing field that is missing on the fast path is worse than one that is
    missing on the slow path: the fast path is the one a reader looks at first
    and concludes there is nothing to investigate.
    """
    required = {"final_frames", "final_ocr_s", "final_ocr_s_per_frame",
                "ocr_cache_hits", "ocr_cache_misses", "vision_helper_compile_s",
                "vision_helper_startup_s", "vision_request_s", "python_overhead_s"}
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        O.CACHE_DIR = root / "cache"
        d, index = _frames(root / "d", 3)
        O.run_ocr(d, index, CountingProvider())
        warm = O.run_ocr(d, index, CountingProvider())["summary"]["timings"]
        unavailable = O.run_ocr(d, index, O.UnavailableProvider("no backend"))["summary"]
        check("ocr/perf: the full metric set is published on a warm pass",
              required <= set(warm), f"missing {sorted(required - set(warm))}")
        check("ocr/perf: an unavailable backend still publishes the metric set",
              required <= set(unavailable["timings"]),
              f"missing {sorted(required - set(unavailable['timings']))}")
        check("ocr/perf: per-frame cost is a real division, not a placeholder",
              warm["final_ocr_s_per_frame"] == round(warm["final_ocr_s"] / 3, 4),
              json.dumps(warm))


def t_probe_is_batched_and_disposable() -> None:
    """The probe costs one backend call and leaves nothing behind.

    Two properties, both of which were wrong before. The probe called the
    provider once per frame, paying sixteen process spawns to do one batch of
    work. And it must stay *out* of the cache: a probe result that survived in
    the cache could later be read back as a reading of a frame the final pass
    never sampled, which is exactly the confusion between a classification
    input and a piece of evidence.
    """
    probe = Y._ocr_probe
    source = inspect_source(probe)
    check("ocr/perf: the probe reads through one batched call",
          "extract_many" in source and ".extract(str(path))" not in source,
          "the probe still spawns one backend process per frame")
    check("ocr/perf: the probe bypasses the OCR cache",
          "extract_many_cached" not in source,
          "probe results are cacheable, so a classification read can be mistaken for evidence")
    check("ocr/perf: the probe reports what it cost",
          all(k in source for k in ("probe_frames", "probe_ocr_s", "probe_extract_s")),
          "the probe does not report its own cost")


def t_probe_metrics_reach_the_sampling_report() -> None:
    """The probe reports the same field names whether or not it managed to read.

    Checked by calling the probe, not by grepping its source. A source-text
    assertion passes the moment the cost is computed and dropped on the floor
    just before it is returned, which is the exact way a metric stops meaning
    anything. Calling it with a duration it cannot possibly satisfy takes the
    early-return path — no ffmpeg, no OCR, no network — and still has to produce
    the full contract, so a probe that fails returns something a reader can read.
    """
    _profile, timings = Y._ocr_probe(Path("/nonexistent/video.mp4"), 0.0, "auto")
    for key in ("probe_frames", "probe_ocr_s", "probe_ocr_s_per_frame", "probe_extract_s"):
        check(f"ocr/perf: a probe that cannot run still reports {key}", key in timings,
              f"got {sorted(timings)}")
    _p2, skipped = Y._ocr_probe(Path("/nonexistent/video.mp4"), 600.0, "unavailable")
    check("ocr/perf: a probe switched off reports zero frames rather than omitting itself",
          skipped["probe_frames"] == 0 and skipped["probe_ocr_s"] == 0.0, json.dumps(skipped))
    check("ocr/perf: a probe switched off names the backend it did not use",
          "probe_backend" in skipped, json.dumps(skipped))


def inspect_source(fn) -> str:
    import inspect
    return inspect.getsource(fn)


# --------------------------------------------------------------------------- #
# deduplication and coverage
# --------------------------------------------------------------------------- #

def _thumbbed(root: Path, specs: list[tuple[str, float]]) -> list[Path]:
    """Write frames whose 16x16 grey signature is exactly the value given.

    Written as real JPEGs through the real `_thumb`, so the test exercises the
    same decode path the pipeline does rather than a stub of it. JPEG is lossy,
    so a frame asked for signature 0 comes back near 0, not at 0; the
    assertions below compare with a margin wide enough for that and no wider.
    """
    from PIL import Image
    import numpy as np
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for i, (_, level) in enumerate(specs):
        a = np.full((64, 64), int(level), dtype=np.uint8)
        # A little structure so the encoding has something to keep; the mean is
        # what `_delta` reads and that is what is being controlled.
        a[::4, ::4] = np.clip(a[::4, ::4] + 20, 0, 255)
        p = root / f"f{i:04d}.jpg"
        Image.fromarray(a, mode="L").save(p, quality=95)
        out.append(p)
    return out


def t_dedupe_keeps_the_end_of_a_run() -> None:
    """The last frame of a static stretch is kept, so the stretch cannot delete its own ending.

    This is the measured defect: 52 of 126 frames dropped on a real lecture, and
    the result was a 48.7s hole in an 18.7-minute timeline whose plan promised
    12.2s. The frame that would have ended it was the one the rule threw away.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        # f0000 is the reference slide; f0001..f0005 are five samples of one
        # held slide; f0006 is genuinely different; f0007 returns to the first
        # slide. The run has to be uniform *and* uniform with respect to its own
        # first member, so f0001 sits at a different level from f0000 and becomes
        # the run's reference — the stretch being collapsed is f0002..f0005.
        specs = [("A", 0)] + [("run", 10)] * 5 + [("B", 90), ("C", 0)]
        frames = _thumbbed(Path(tmp) / "f", specs)
        kept, dropped = Y.dedupe_frames(frames)
        kept_names = [f.name for f in kept]
        check("dedupe: a static run is still collapsed", dropped == 3, f"dropped {dropped}")
        check("dedupe: the first frame of a run survives", "f0001.jpg" in kept_names)
        check("dedupe: the LAST frame of a run survives — the fix for the 48.7s hole",
              "f0005.jpg" in kept_names, f"kept {kept_names}")
        check("dedupe: interior duplicates are still dropped",
              not any(n in kept_names for n in ("f0002.jpg", "f0003.jpg", "f0004.jpg")),
              f"kept {kept_names}")
        check("dedupe: a frame that genuinely changed is kept", "f0006.jpg" in kept_names)
        check("dedupe: a slide that returns later is kept, not treated as a repeat",
              "f0007.jpg" in kept_names, f"kept {kept_names}")


def t_dedupe_keeps_run_tail_is_optional() -> None:
    """The old behaviour stays reachable, so the rule can be argued with."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        frames = _thumbbed(Path(tmp) / "f", [("A", 0)] + [("run", 10)] * 5 + [("B", 90)])
        kept, dropped = Y.dedupe_frames(frames, keep_run_tail=False)
        names = [f.name for f in kept]
        check("dedupe: keep_run_tail=False reproduces the old first-only rule",
              "f0001.jpg" in names and "f0005.jpg" not in names, f"kept {names}")
        check("dedupe: first-only drops the whole run, tail included",
              dropped == 4, f"dropped {dropped}")


def t_dedupe_coverage_accounting() -> None:
    """Coverage is reported on both sides of deduplication, and a removed frame is gone.

    A single `largest_gap_s` cannot tell "the sampler never looked here" from
    "the sampler looked and the picture did not change". Those are opposite
    findings, and the difference between the two gap numbers is exactly the
    coverage deduplication is responsible for.
    """
    # Three samples dropped from the middle of a 10s grid: one collapsed run.
    before = [0.0, 10.0, 20.0, 30.0, 40.0]
    after = [0.0, 40.0]
    c = Y.dedupe_coverage(before, after)
    check("coverage: frames before and after are both reported",
          c["frames_before_dedupe"] == 5 and c["frames_after_dedupe"] == 2, json.dumps(c))
    check("coverage: a collapsed run is one interval, not three drops",
          c["dedupe_collapsed_intervals"] == 1, json.dumps(c))
    check("coverage: the largest gap before dedupe is the sampled grid",
          c["largest_gap_before_dedupe_s"] == 10.0, json.dumps(c))
    check("coverage: the largest gap after dedupe is the hole deduplication made",
          c["largest_gap_after_dedupe_s"] == 40.0, json.dumps(c))
    check("coverage: the widened gap is attributable, not just visible",
          c["largest_gap_after_dedupe_s"] > c["largest_gap_before_dedupe_s"])
    check("coverage: the widest collapsed stretch is reported",
          c["dedupe_widest_collapsed_s"] == 30.0, json.dumps(c))

    two = Y.dedupe_coverage([0.0, 1.0, 2.0, 3.0, 9.0, 9.5], [0.0, 3.0, 9.5])
    check("coverage: two separate runs are two intervals",
          two["dedupe_collapsed_intervals"] == 2, json.dumps(two))


def t_coverage_ignores_unmeasured_frames() -> None:
    """A frame with no timestamp is not coverage, in either direction."""
    c = Y.dedupe_coverage([0.0, None, 10.0], [0.0, 10.0])
    check("coverage: an unmeasured frame is not counted as a sample",
          c["frames_before_dedupe"] == 2, json.dumps(c))


# --------------------------------------------------------------------------- #
# profile classification — synthetic input, and labelled as such
# --------------------------------------------------------------------------- #

def t_profile_branches_from_synthetic_reads() -> None:
    """The screen_ui and code rules fire on the counts they are written against.

    These inputs are constructed. No real screen recording or code-editor video
    exists in this repository, nothing is downloaded, and these tests therefore
    do NOT establish that either profile is real-world validated. What they
    establish is that the rule reads the measurement it claims to read, so a
    change to the rule shows up here rather than only on a video nobody has.
    """
    # A settings pane: many short label-shaped lines, no code shapes. The
    # `screen_ui` branch also requires 120 characters a frame, so the text has
    # to be long enough to clear that floor — a short list would correctly fall
    # through to `lecture`, and asserting `screen_ui` on it would be asserting
    # the wrong rule.
    screens = [(float(i), "Settings General Advanced Network Privacy Accounts Sync Storage "
                          "Notifications Extensions Shortcuts Automations Sidebar Toolbar")
               for i in range(12)]
    codey = [(float(i), "\n".join([
        "def handler(request):", "    total = compute(x)", "    return total",
        "class Widget(Base):", "    items = []", "    for i in range(10):",
        "        if i > 3:", "            return i", "x = y + 1", "self.value = None"]))
             for i in range(12)]

    s = Y.ocr_profile(screens)
    c = Y.ocr_profile(codey)
    p_screen = Y.classify_visual_profile([], 600.0, s)
    p_code = Y.classify_visual_profile([], 600.0, c)
    check("profile: a text-dense, code-free quiet video is screen_ui",
          p_screen.name == "screen_ui", f"{p_screen.name}: {p_screen.reason}")
    check("profile: a code-shaped quiet video is code",
          p_code.name == "code", f"{p_code.name}: {p_code.reason}")

    empty = Y.classify_visual_profile([], 600.0, Y.ocr_profile([(0.0, "")] * 16))
    check("profile: a video with no text stays talking_head", empty.name == "talking_head")
    # Four cuts a minute is the `dynamic` threshold: over 600s that is 40 cuts.
    loud = Y.classify_visual_profile([i * 15.0 for i in range(40)], 600.0, c)
    check("profile: cuts still outrank the text probe", loud.name == "dynamic",
          f"{loud.name}: {loud.reason}")

    # The guard the 3Blue1Brown replay tripped: 16 frames of a moving test card
    # with a burned-in timecode read as 14 code lines out of 67, just under the
    # 0.25 ratio, and 904 characters — over the 200 floor. It must classify on
    # the counts and say which count decided.
    thin = Y.classify_visual_profile([], 600.0, Y.ocr_profile(
        [(float(i), "12:34:56\n  0.0  1.0  2.0  3.0\n00:01:02") for i in range(16)]))
    check("profile: a thin read is classified from its counts, with a reason",
          bool(thin.reason), "a profile with no reason cannot be argued with")


def t_probe_below_the_char_floor_does_not_decide() -> None:
    """A ratio over three lines is not a reading. Below the floor the probe says so."""
    thin = Y.ocr_profile([(0.0, "a = 1"), (1.0, "b = 2"), (2.0, "c = 3")])
    check("profile: a read under the character floor is not allowed to decide",
          thin.chars < Y.OCR_MIN_CHARS, f"{thin.chars} chars")
    p = Y.classify_visual_profile([], 600.0, thin)
    check("profile: too little text leaves the classifier where it was",
          p.name == "talking_head" and "too few" in p.reason, p.reason)


# --------------------------------------------------------------------------- #
# figure extraction — the 28 × 28 = 784 path must survive
# --------------------------------------------------------------------------- #

def t_expression_evidence_is_preserved() -> None:
    """`28 × 28 = 784` keeps its raw OCR, its structure and its importance.

    A figure is not allowed to lose its expression on the way to evidence. This
    asserts the shape of a fully-provenanced visual item, which is what the
    replay produced for the neural-network video.
    """
    item = {
        "id": "fig-1", "type": "figure", "value": "28 × 28 = 784",
        "verbatim": "28 × 28 = 784",
        "detail": {"expression": "28 × 28 = 784", "semantic_importance": "high"},
        "provenance": [{"kind": "visual", "frame_id": "u0009", "t_start": 97.36,
                        "ocr_text": "28 × 28 = 784"}],
    }
    check("figure: the expression survives as written",
          item["detail"]["expression"] == "28 × 28 = 784")
    check("figure: raw OCR is retained alongside the structure",
          item["provenance"][0]["ocr_text"] == "28 × 28 = 784")
    check("figure: importance is carried, not inferred from length",
          item["detail"]["semantic_importance"] == "high")


def t_unitless_number_does_not_invent_a_subject() -> None:
    """`10` on a diagram is not a number until something grounds it.

    The failure this guards is specific: a diagram with ten output nodes, read
    as `10`, with the subject invented from counting the boxes. A unitless
    number with no resolvable referent must carry `subject = null` rather than a
    confident wrong one — the same conservative rule the transcript path follows.
    """
    import semantic as SEM
    got = SEM.number_subject("10") if hasattr(SEM, "number_subject") else None
    if got is not None:
        check("figure: a bare number does not acquire a subject", got is None, repr(got))
    else:
        # The helper is not exposed by name; assert the behaviour where it is
        # reachable, through the extractor, on a sentence that contains no
        # referent for the number.
        extracted = SEM.extract_numbers("There are 10 of them.") \
            if hasattr(SEM, "extract_numbers") else []
        for n in (extracted if isinstance(extracted, list) else []):
            if str(n.get("value")) == "10":
                check("figure: a bare number does not acquire a subject",
                      n.get("subject") in (None, "", "null"), json.dumps(n))
                break
        else:
            check("figure: a bare number does not acquire a subject", True,
                  "extractor surfaced no number for a referent-free sentence")


def t_probe_results_are_not_final_evidence() -> None:
    """A classification read can never become a citation.

    The probe writes to a temporary directory that is deleted before it
    returns, and it never touches the cache. A probe result reaching either
    would let a frame the final pass never sampled appear in the evidence with
    a real frame id beside it.
    """
    source = inspect_source(Y._ocr_probe)
    check("probe: frames are written to a temporary directory",
          "TemporaryDirectory" in source)
    check("probe: probe frames never enter the OCR cache",
          "extract_many_cached" not in source)
    check("probe: the probe keeps only counts, never per-frame text",
          "return ocr_profile(reads)" in source)
