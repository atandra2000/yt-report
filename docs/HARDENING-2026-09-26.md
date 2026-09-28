# HARDENING — 2026-09-26

Post-visual-sampling hardening pass. The question this document answers is not
"is everything green". It is five specific questions, each with a measurement
behind it or an explicit statement that the measurement could not be made.

Machine: 8 cores, 8 GB. Backend: Apple Vision through a compiled Swift helper
(`ocr_vision-d4a7551b2e0dbced`), cold macOS caches. Real artifact used for every
measurement: `/tmp/yt-replay/3b1b` — 3Blue1Brown, *But what is a neural
network?*, 18.7 min, 126 surviving frames.

---

## 1. Why did OCR take 631.6 seconds?

**It does not, and it never did on this machine. The number is not reproducible.**

Re-running the exact pass over the exact 126 frames, with a fresh cache:

| | reported in the replay | reproducible, before this pass | after this pass |
|---|---|---|---|
| cold, 126 frames | **631.6 s** | 31.6 s | **23.0 s** |
| warm, same frames | — | 0.07 s | **0.05 s** |
| helper-process spawns | — | 54 | **2** |
| Python overhead | — | 0.073 s | 0.069 s |

A 21× gap between a recorded number and the cost of the identical work, with
nothing in the artifact able to explain it, is the actual defect. The
instrumentation below exists so that the next occurrence says *which* component
grew instead of only that something did.

### What the 631.6 s was not

Checked and ruled out, each by measurement:

* **Swift compile** — `swiftc -O` on the helper takes 2.3 s, once, and the
  binary is cached. Not 600 seconds.
* **Repeated OCR of the same image** — no. The cache is content-addressed, and
  a warm pass makes zero provider calls.
* **Frame preparation** — the probe reads 16 small JPEGs; ffmpeg extraction of
  those takes 0.12 s.
* **Timing instrumentation error** — `ocr_wall_s` is a single
  `perf_counter` delta around the whole of `run_ocr`. It was correct; the run it
  measured was genuinely slow at the time, and nothing was recorded that would
  say which part.

What remains unexplained is environmental. The replay ran two full pipelines
back to back on an 8-core/8 GB machine, and the cache-file mtimes show the 126
frames were read in **two batches 22 minutes apart** (73 frames at 18:56, 53 at
19:18) — the run was interrupted or retried, and the recorded wall time covers
only the second, partial batch. I could not reproduce the condition, and I am
not going to guess at it. What changed as a result is that the number is now
attributable.

### Where the time actually goes

Cold pass, 126 frames, instrumented:

```
final_ocr_s              22.99      total
  ocr_backend_call_s     22.99      everything inside the provider
    vision_request_s     17.48      2 helper processes
      of which rereads   12.71      53 frames re-read at 2x
  python_overhead_s       0.07       cache key hashing, grouping, summary
```

The recogniser was never the problem. Reading 126 frames costs about 4.8 s
(~38 ms/frame); the pass took 23 s because **53 of the 54 helper-process spawns
were low-confidence re-reads**, each paying full process start and Vision
framework load to re-render one frame at 2x.

### The fix

`VisionProvider._reread` was the same per-frame-spawn mistake the main pass had
already stopped making. It now stages every re-read and reads them in **one**
process: 54 spawns → 2, 31.6 s → 23.0 s. No re-read was removed, no threshold
was changed, and the count of frames that trigger a re-read is identical (53
before, 53 after) — only the number of processes changed.

The text probe had the same defect and the same fix: it called the provider once
per frame, sixteen times, to do one batch of work. Now one call. Measured on 16
frames: **6.76 s → 4.02 s, 16 spawns → 1**.

### The metric block now published

`visual.json` → `summary.timings`, and `meta.json` → `visual` for the probe half.
Every field is present on a cold pass, a warm pass, and a run with no backend
at all:

```
probe_frames            probe_ocr_s          vision_helper_compile_s
probe_extract_s         probe_ocr_s_per_frame vision_helper_startup_s
final_frames            ocr_cache_hits       vision_request_s
final_ocr_s             ocr_cache_misses     vision_requests
final_ocr_s_per_frame   ocr_backend_call_s   vision_rereads
                        python_overhead_s    vision_reread_s
```

---

## 2. Does OCR run redundantly?

No, and the count is now asserted rather than assumed.

* One provider call for a whole cold batch — asserted.
* A warm pass makes **zero** provider calls and reports 126 hits / 0 misses —
  asserted.
* Two byte-identical files under different names are read once and share a cache
  entry — asserted.
* The 53 re-reads are the only second pass, they are explicitly triggered by
  `REREAD_BELOW`, and they now cost one process between them — asserted.
* The 16-frame text probe is still deliberately **not** cached, and its frames
  still go to a temporary directory that is deleted before it returns. A probe
  result that reached the cache could be read back as a reading of a frame the
  final pass never sampled. Asserted.

Python overhead is now measured separately and is 0.07 s of 23 s — 0.3%. The
cache is not the cost; the recogniser and the re-reads are.

---

## 3. Cold vs warm

One real artifact, one fresh cache directory, two consecutive runs:

| | total | OCR | cache hits | misses | helper spawns | frames processed | frames reused |
|---|---|---|---|---|---|---|---|
| **cold** | 22.99 s | 22.99 s | 0 | 126 | 2 | 126 | 0 |
| **warm** | 0.05 s | 0.00 s | 126 | 0 | 0 | 0 | 126 |

A warm run re-runs no OCR. Evidence content is identical between the two runs:
124 frames read, 106 with text, 77 verbatim-eligible, mean confidence 0.8682,
5385 characters — the same on both.

Cache invalidation boundaries, each asserted in isolation: backend identity,
backend version, languages, read threshold, re-read policy, high/medium
confidence thresholds, and frame bytes each move the key. A frame's path, a
rename, and the sampling version do **not** — folding the sampling policy into
the OCR key would re-OCR every frame of every video on every policy bump, which
is the expensive way to be wrong. A changed frame set changes the answer by
changing the content hashes, which is the mechanism that should do that job.

---

## 4. Cost of the text probe

Established, as asked, before any decision about it. The probe was **not**
removed or restructured.

| | probe_extract_s | probe_ocr_s | probe_ocr_s_per_frame | probe_frames |
|---|---|---|---|---|
| 120 s settings fixture | 0.12 | 1.43 | 0.089 | 16 |
| 120 s code fixture | 0.13 | 2.41 | 0.151 | 16 |
| 18.7 min lecture (extrapolated) | ~0.13 | ~2.4 | ~0.15 | 16 |

Roughly 2 s of fixed cost, about 10% of a full cold OCR pass on a long video.
On the *short* fixtures it is 60–70% of total extraction elapsed, because those
fixtures deduplicate to almost nothing and the probe is then nearly the whole
job. That ratio is a property of a 2-minute video, not of the probe.

What it buys: a 2× finer sampling grid where the text justifies one. `code`
samples at 6.32 s where `talking_head` would sample at 30 s — a 4.75×
difference in coverage, for 2 s. That is a good trade and the probe stays.

---

## 5. Deduplication: what was removed and what it cost

### The reported numbers

```
planned_frames (plan)          146
frames before dedupe           178
near_duplicates_dropped         52
surviving frames               126
largest gap, before dedupe     12.17 s   (the planned baseline step)
largest gap, after dedupe      48.68 s   (4 × the baseline step)
p95 gap, after dedupe          24.34 s   (2 × the baseline step)
```

The gap was not a defect in the arithmetic. 52 of 178 frames were genuinely
near-identical, and the survivors are mutually distinct — replaying
`dedupe_frames` over the 126 survivors drops **zero** of them, which is the
evidence that it removed redundancy and not coverage.

The defect was *which* frame of each run it kept.

### The fix

`dedupe_frames` kept the **first** frame of a run of identical pictures and
deleted the rest. For a video that holds one slide for a minute, that means the
frame at the **end** of the hold — where the presenter has finished speaking and
drawn the next thing — is always the one discarded. On this lecture that produced
the 48.7 s hole: four consecutive baseline steps deleted, and the change that
ended the stretch never sampled.

`dedupe_frames` now keeps the last frame of every run as well as the first
(`keep_run_tail=True`, switchable). Interior duplicates are still dropped. No
extra decode: thumbnails are taken once up front, because the run-length rule
needs to know what comes *after* a candidate and re-decoding to find out costs
more than the comparison.

A single dropped frame costs one frame; one collapsed run of ten costs ten and a
hole. `dedupe_coverage` now reports that shape directly:

```
frames_before_dedupe    frames_after_dedupe   dedupe_collapsed_intervals
dedupe_widest_collapsed_s
largest_gap_before_dedupe_s   largest_gap_after_dedupe_s
p95_gap_before_dedupe_s       p95_gap_after_dedupe_s
```

Published in both `frames.json` → `policy.coverage` and `meta.json` → `visual`.
**A removed frame is counted as gone** — not in the surviving set, not in
coverage, not available to alignment. One `largest_gap_s` could never tell "the
sampler never looked here" from "the sampler looked and nothing changed", and
those are opposite findings.

### Is the 16×16 threshold too aggressive?

Measured, on the real frames, over every pair within 16 s of each other. 218 of
221 such pairs have differing on-screen text.

| signature | text-changing pairs missed at zero false-drop |
|---|---|
| 16×16 grey (current) | 29 / 218 |
| 64×64 grey | 27 / 218 |
| 16×16 per-cell std-dev | 28 / 218 |
| 16×16 edge energy | 94 / 218 |

**A cheap pixel signature cannot separate "the same picture" from "the same
picture plus one line of new text."** Raising the resolution barely helps;
contrast and edge maps score *worse*, because they amplify compression noise.
Text occupies a few pixels in each cell and a 16-cell average absorbs it. The
threshold is left at 3.0, the measurement is recorded in the code as
`DEDUPE_UNSEEN_TEXT_RATE`, and the residual loss is bounded structurally by the
run-tail rule instead.

No duplicate was sent to a VLM. No VLM ran.

---

## 6. Real screen / code test

**No real screen recording or code-editor video exists in this repository, none
was downloaded, and the `screen_ui` and `code` profiles are therefore NOT
real-world validated.** What follows is a rendered-fixture result and is labelled
as such.

`scripts/visual_fixture.py` renders a code-editor window and a settings pane
deterministically, encodes them with ffmpeg, and runs the **whole real pipeline**
over them — ffmpeg decode, scene probe, text probe through real Apple Vision
OCR, classification, planning, extraction, deduplication. This exercises the
path end to end rather than feeding the classifier a constructed tuple.

120 s, `auto_frame_budget`:

| | code-editor | settings-pane |
|---|---|---|
| detected profile | **`code`** | **`screen_ui`** |
| reason | *180 of 212 read lines are code-shaped, across 9575 characters* | *16 of 16 probe frames carry text, 367 characters each, and the video does not cut* |
| uniform step | 6.32 s | 8.57 s |
| planned / surviving frames | 20 / 2 | 15 / 2 |
| frames per minute | 1.0 | 1.0 |
| OCR text density | 598 chars/frame | 367 chars/frame |
| code-line ratio | **0.849** | **0.242** |
| largest gap after dedupe | 118.0 s | 118.0 s |
| dedupe rate | 18/20 dropped | 13/15 dropped |
| probe cost | 2.42 s | 1.59 s |

The two discriminate correctly in both directions: the editor is 0.849
code-shaped, the settings pane 0.242 — just under the 0.25 code threshold —
and each lands on its own branch. The reason string carries the counts, so each
classification can be argued with.

Two things this does **not** establish: that a real editor window looks like
this, and that the branch survives a real screen share. Only a real video
settles that.

One incidental finding worth recording: at **40 s** both clips classified
`talking_head`, because `classify_visual_profile` short-circuits for videos
under 90 s and reads by count rather than rate — the OCR probe, which read both
clips perfectly, is never consulted. That is the documented behaviour of a
deliberate rule, not a bug, but it means the profiles do not apply to short
clips at all.

---

## 7. Figure extraction: 784 and 10

Verified against the on-disk artifact, including direct visual inspection of the
frames.

**`28 × 28 = 784` — confirmed.** Two evidence items carry the expression
verbatim (`sem-83fd96ce1f` value 784, `sem-7603e2021b` value 28), with 9 and 5
OCR provenance entries respectively, every one `read_method: "native"` and
`verified_verbatim: true`, at confidence 1.0. Raw OCR, expression structure and
provenance all preserved. `validation.json`: `valid: true`, `provenance_coverage:
1.0`, no ungrounded items, no dangling refs.

**`10` — not fabricated, and not what one might expect.** There is no
visual-derived "the output layer has 10 units" anywhere, and there could not be:
that artifact was produced with `--vlm none` (`vlm_calls: 0`), so every item
derives from Apple Vision OCR or the transcript.

* The single numeric `10` in evidence is `784×16+16×16 + 16×10` — the weight
  arithmetic printed on screen at t≈730 s, read verbatim from frames `u0061`
  and `u0062`. It carries **no subject**, which is correct: nothing on screen
  says what it counts.
* The output-layer fact — *"784 numbers as an input and spits out 10 numbers as
  an output"* — lives in a transcript-sourced definition item with a single
  `source: "transcript"` provenance entry and **no visual provenance at all**.
  Narration grounds the 10; the picture does not.
* No item anywhere describes a count derived from counting drawn nodes. The
  output column *is* drawn as ten circles labelled `0`–`9`, so a human could
  derive 10 — nothing in the pipeline did.

Frames were opened and looked at, not inferred from OCR text. `784` is written
and legible in every diagram frame. The output layer is labelled `0 1 2 3 4 5 6
7 8 9` per node; the numeral "10" and any "0–9" range label appear nowhere in
it.

**This is the desired behaviour and it is preserved.** A number stays `null`
until something grounds it; narration grounds one 10 and the screen grounds
another; the diagram stays an observation even where OCR cannot read a label.

Two pre-existing issues found in passing, not fixed here:
`detail.expression` is decorative — no validator reads it, and `validation.json`
says so; and the definition item's `verbatim` splices two snippets while citing
only the first, so its time window does not cover the clause it quotes.

---

## 8. Provider readiness

All four commands are wired in `bin/yt-report` and dispatch correctly. **No model
call was made in this phase**, per the instruction not to run them automatically,
and no credit was spent.

Verified with a real run, no credentials and no egress:

```
$ release-check                      # VLM_PROVIDER= SYNTH_PROVIDER=
release check — CONDITIONAL
  UNTESTED  connectivity  vlm: SKIPPED; synthesis: SKIPPED
  SKIPPED   vlm           no VLM provider configured; the vision columns stay UNTESTED
  SKIPPED   synthesis     no synthesis provider configured
  SKIPPED   critic        no synthesis provider configured
  SKIPPED   end-to-end    no URL given
  budget    frames 0/2  synthesis 1  critic 1  revisions 0/1  calls 0
exit 0
```

That is the honest answer: nothing was exercised, nothing is claimed, and no
false `PASS` appears anywhere. Missing credentials are `UNTESTED`, not `FAILED`.
No credential can reach an artifact or a log — both configs serialise
`api_key_present` as a boolean and the key itself never enters `to_dict()`.

---

## 9. Provider error consistency — four real defects found and fixed

An audit of the three doctors and the release harness against the six
contradictions to look for. Four were real and are fixed:

1. **`structured_vlm` passed while nested schema validation failed.**
   `release_check` read a check named `schema_valid`, which no doctor emits. The
   lookup silently fell through to `structured_enforced` — a *top-level key
   presence* test — so an observation that failed re-parse (confidence over its
   ceiling, missing content type, wrong shape) still published `PASS`. Each
   vision column now ANDs the checks that constitute it, including
   `test_observation_valid`.

2. **`vision_capable` passed while observation schema validation failed.** Same
   hole, other column. Now gated the same way.

3. **`model_completes` passed when nothing was emitted.** On an HTTP 4xx from a
   provider rejecting `response_format`, the check was recorded as a pass on the
   strength of the endpoint having replied at all — which is the transport
   answering, not the model completing. Now `False`. The capability finding is
   unchanged and still reported.

4. **`structured_enforced` passed for `prompt_only`,** the one mode that enforces
   nothing: `conforms_to_schema` short-circuited to `True`, so the doctor printed
   *"the prompt_only response carried every required key"* as a pass. Conformance
   is now actually tested for all three modes, and `schema_enforced` is `False`
   in `prompt_only` whatever the answer looked like — because asking politely in
   a prompt is the same test with nothing guaranteeing it, and conflating the
   two is how a mode that guarantees nothing got reported as guaranteeing
   everything.

5. **`DEGRADED` was produced by the row classifier and had no top-level branch.**
   A capability that is *disproven* while another worked fell through to
   `CONDITIONAL` and exited 0 — a demonstrated model-layer failure presented as a
   state a release script is told to accept. `DEGRADED` now has its own branch
   and its own non-zero exit. `CONDITIONAL` still exits 0, correctly: it means
   "not exercised", which is a state a release script should be able to record.

The shared hierarchy the brief asked for already existed in
`DoctorReport.failed(name, detail, stage)` with its downstream cascade; the
defect was that `release_check` re-derived verdicts from individual check flags
instead of consulting it. That is now fixed at the aggregator rather than by
rebuilding the doctors.

---

## 10. Tests

**2626 checks, 0 failures** — full offline suite, no network, exit 0.

New: `scripts/selftest_ocr_perf.py`, **68 checks**, covering the areas the brief
listed — OCR probe timing, final OCR timing, cache reuse, duplicate-OCR
prevention, re-read batching, dedupe coverage accounting, content-aware dedupe
safety, `screen_ui` and `code` branches, equation evidence preservation,
unitless-number subject discipline, probe/evidence separation, and OCR cache
invalidation boundaries.

**No wall-clock assertion appears anywhere in the new suite.** The 631.6 s
regression was not reproducible minutes later on the same machine; a timing
assertion would be a flaky test failing for reasons unrelated to the code. What
is asserted instead is structure — how many provider calls, with what arguments,
and which the cache removed. Those hold on any hardware.

One existing check was legitimately tightened, not weakened: the sampler
determinism test excludes declared timing keys by name and fails on any new
non-timing key. The new `probe_*` fields tripped it, which is the test working
as designed. `probe_frames` and `probe_backend` were deliberately *not* added to
the exclusion list — they are deterministic, and excusing them would let a probe
that silently stopped reading pass.

---

## 11. Remaining blockers

Real ones only.

1. **The 631.6 s is unexplained.** Not reproducible; the run it came from was a
   partial second batch after a 22-minute gap. The instrumentation will
   attribute it if it recurs; it cannot be attributed retroactively.
2. **`screen_ui` and `code` are not real-world validated.** No real screen or
   code video exists here and none was fetched. The rendered-fixture result
   shows the path works end to end and discriminates correctly; it does not show
   a real editor triggers it.
3. **`vlm-doctor` has an unguarded exception path.** `provider.analyze` is
   called with no `try` at the doctor level, and `vlm_doctor.main` has no guard
   either, while the two synthesis-side doctors do. The provider classifies the
   *expected* egress failures, so the window is narrow — but an exception outside
   that set (an unwrapped `ssl.SSLError`, an `OSError` writing the probe image)
   propagates as a traceback, and via `release_check`'s in-process call it would
   take the whole matrix down. Found by audit, not fixed this phase.
4. **`release_check --max-vlm-frames 20` does not raise the cap** — it is clamped
   to a hard-coded 2, so the flag does not do what its help text says.
5. **`connectivity` and `end-to-end` never reach the matrix.** They are run and
   printed, but their failures do not affect the top-level status. The same
   class of hole as the one fixed in item 5 above, one level up.
6. **16×16 dedupe cannot see a one-line text change** (29/218 measured). The
   run-tail rule bounds the damage; it does not eliminate it. Eliminating it
   needs a content-aware signal that does not exist cheaply, and adding one is
   an architecture change this phase was told not to make.
7. **Two evidence-integrity issues** noted in section 7: `detail.expression` is
   unvalidated by any check, and one definition item's citation does not cover
   the clause it quotes.
8. **No real provider qualification has been run.** VLM, synthesis and critic
   doctors are all `UNTESTED`. That is correct and honest, and it is the gate:
   nothing should be spent on a full model-backed video until
   `vlm-doctor` → `synthesis-doctor` → `critic-doctor` pass in that order, in that
   budget, and stop at the first failure.
