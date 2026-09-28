# yt-report

A Claude Code skill that turns a YouTube video into a **grounded, source-anchored report** — an executive summary, detailed notes organised by topic, an index of every figure, name, price and link the video contains, a timeline, and a plain statement of what could not be read.

Covers podcast, interview, lecture, screen-recording and AI-news videos. The
report is **generated from a validated evidence record**, not composed freehand,
which is what makes its grounding a property of the code rather than a promise.
Neutral third person. Never evaluates or fact-checks the content.

## What ships here

```
SKILL.md                     the skill itself — the contract an agent follows
references/genres.md         per-genre writing guidance (podcast, interview, lecture, AI news)
references/browser-fallback.md   what to do when YouTube blocks or the API breaks
scripts/yt_fetch.py          pull transcript, metadata and frames into a folder
scripts/ocr.py               read every frame's text into visual.json (also a standalone CLI)
scripts/ocr_vision.swift     the macOS Vision helper the provider drives
scripts/evidence_model.py    typed model for evidence.json — the canonical record
scripts/evidence_provenance.py  constructors for resolvable citations
scripts/align.py             pair each sentence with the frames that illustrate it
scripts/semantic.py          typed extraction: claims, figures, entities, steps, comparisons
scripts/vlm.py               vision-language pass; also one call against a single frame
scripts/build_evidence.py    run the whole second half: build, align, extract, report
scripts/evidence_validate.py  validate evidence.json; non-zero on blocking failures
scripts/synthesis.py         the evidence gate, the synthesis pipeline, the fallback ladder
scripts/release_check.py      which capabilities a provider has actually been shown to have
scripts/synth_model.py       the writer's schema, its instruction, and its client
scripts/synth_validate.py    grounding checks on a synthesized draft
scripts/chat_json.py         one structured-JSON negotiator, shared by both model passes
scripts/vlm_doctor.py        can this provider actually produce a valid observation?
scripts/synth_doctor.py      can this model write a report from validated evidence?
scripts/critic_doctor.py     can it find a planted defect, and stay quiet on a clean one?
scripts/eval_live.py         one budgeted, measured run against real providers
scripts/lang_resources.py    per-language ordinary-word lists, and how to detect the language
resources/stopwords/*.txt    ordinary vocabulary per language (en, de, es, fr, pt, it, hi)
scripts/report.py            evidence.json -> REPORT.md; a renderer, not a writer
scripts/report_validate.py   grounding checks on a finished report; blocks the write
scripts/report_fixtures.py   15 deterministic video shapes, built not recorded
scripts/label_sheet.py       tile sampled frames into labelled contact sheets
scripts/resheet.py           rebuild contact sheets from frames already on disk
scripts/selftest.py          offline self-test — no network, no ASR, no YouTube
```

`scripts/verify_report.py` was retired and has been removed. It checked that
cited *frame filenames* existed, which the report no longer cites — it renders
`[04:31 · narration]` instead — so it passed vacuously on any report citing
nothing, which is the outcome the audit flagged. `report_validate.py`
superseded it and checks the thing that can actually fail. Two validators that
can disagree is a worse state than one validator, so the old one is gone rather
than left beside the new one.

## Checking a vision provider

```bash
# can this provider actually produce a valid observation? no video, no download
VLM_API_KEY=sk-... uv run scripts/vlm_doctor.py
uv run scripts/vlm_doctor.py --vlm-provider openrouter --json
```

`vlm-doctor` walks the capability chain one step at a time and reports each
step separately, because the steps fail independently:

| check | meaning |
|---|---|
| `reachable` | the endpoint answered at all |
| `authenticated` | the credential was accepted |
| `model_exists` | the named model resolved |
| `vision_capable` | an image was accepted and described |
| `structured_requested` | a schema constraint was sent |
| `structured_enforced` | **the returned payload actually matched that schema** |
| `test_observation_valid` | the payload re-parses into a valid observation |

`qualified` is true only when all seven hold. The last three are the ones worth
reading carefully: a gateway that accepts `response_format: json_schema`,
returns HTTP 200, and answers in prose has succeeded at four of the seven and
produced nothing usable. Structured output being *requested* is not the same
claim as it being *enforced*, and only the payload can settle it.

## Checking a writing or critic provider

The same shape, the same code, a longer chain. Both take an artifact directory
or fall back to a built-in fixture, so neither needs a video or a download.

```bash
# can this model write a report from validated evidence? exactly one call
uv run scripts/synth_doctor.py --synth-provider openrouter --synth-model openai/gpt-4o-mini
uv run scripts/synth_doctor.py <artifact-dir> --json

# can the critic find a planted defect, and stay quiet about a sound draft?
# four known-answer cases, one call each
uv run scripts/critic_doctor.py --synth-provider openrouter --synth-model openai/gpt-4o-mini
```

| check | meaning |
|---|---|
| `reachable` | the endpoint answered at all |
| `authenticated` | the credential was accepted |
| `account_paid` | **the account could pay** — distinct from a rate limit, and checked first because both arrive as HTTP 429 |
| `model_exists` | the named model resolved |
| `model_completes` | something came back that was not a transport or account failure |
| `structured_requested` | a schema constraint was sent and accepted |
| `structured_enforced` | **the returned payload carried every required key** |
| `draft_valid` | the payload parsed into a schema-valid draft |
| `draft_grounded` | every unit cites resolvable evidence and survives the blocking checks |
| `report_validated` | the rendered report passes the existing validator |

`qualified` is true only when all ten hold. **HTTP 200 is never a verdict**: a
gateway that accepts `response_format`, returns 200, and answers in prose passes
six of the ten and has produced nothing usable, and a model that answers in the
wrong *shape* is rejected rather than coerced — a repaired guess is how a
fabricated table reaches a report.

The critic doctor plants one defect per case and knows the answer to each:

| case | draft | expected |
|---|---|---|
| A | clean, grounded, hedged | no findings — anything it reports here is a false positive |
| B | a figure changed in the prose | `number-error` |
| C | a hedge removed | `missing-qualifier` |
| D | a sentence no cited item supports | `unsupported-claim` |

A case that did not run within `--max-critic-calls` is printed as *not run*,
never as a pass. A critic that never answered is recorded as `no answer`, not as
`none` — the two are different findings, and conflating them turns four billing
refusals into a verdict about the model.

## A cheap measured real run

```bash
uv run scripts/eval_live.py "<url>" \
  --vlm-provider openrouter --vlm-model openai/gpt-4o-mini \
  --synth-provider openrouter --synth-model openai/gpt-4o-mini \
  --max-vlm-frames 2 --max-synthesis-calls 1 --max-critic-calls 1
```

Each budget is a hard ceiling spent *before* each call, not a counter checked
afterwards. It writes `LIVE-EVAL-<timestamp>.md` into the artifact directory:
provider, model, duration, exact call counts, cache hits and latency, then VLM
quality, extraction quality and report quality, and a blank section for the
runner's own notes. No overall score.

## Qualifying a provider for release

```bash
yt-report release-check --vlm-provider openrouter --vlm-model openai/gpt-4o-mini \
  --synth-provider openrouter --synth-model openai/gpt-4o-mini
```

`release-check` answers one question — which capabilities has this provider
actually been shown to have? — by running the cheapest sequence that can answer
it: connectivity, the VLM doctor, the synthesis doctor, the critic doctor, and
then one small real video only if you pass a URL. It prints a matrix:

```text
provider    model    vision    structured_vlm  synthesis  critic    status
openrouter  gpt-4o   UNTESTED  UNTESTED        UNTESTED   UNTESTED  CONDITIONAL
```

Each step is `PASS` (a real request held), `FAIL` (a real request came back and
did not), `UNTESTED` (no usable answer ever arrived — no credential, no credit,
no egress) or `SKIPPED` (not requested). `FAIL` requires that an answer came
back: without that rule an unreachable provider and a broken one print the same
thing, and only one of them is a bug.

`status` is `QUALIFIED`, `CONDITIONAL`, `DEGRADED` (an optional model layer
failed while the report still landed) or `UNQUALIFIED`. The release claim stays
`CONDITIONAL` until a real synthesis, VLM and critic run have all been exercised
successfully.

Budgets are ceilings and are never silently exceeded: `--max-vlm-frames 2
--max-synthesis-calls 1 --max-critic-calls 1 --max-revision-calls 1`. Use
`--json` or `--out PATH` for the machine-readable form. Credentials are never
printed.

## Writing the report with a model

The renderer in `report.py` is deterministic and remains the default and the
fallback. A model can write the prose instead, and is held to the same
validator:

```bash
# regenerate REPORT.md from the evidence already on disk — no download, no OCR
SYNTH_API_KEY=sk-... uv run scripts/build_evidence.py /tmp/video --synthesis-only \
  --synth-provider openrouter --synth-model openai/gpt-4o-mini

# or as part of a full run
SYNTH_API_KEY=sk-... uv run scripts/build_evidence.py /tmp/video \
  --synth-provider openrouter --synth-model openai/gpt-4o-mini
```

The writer sees validated evidence and nothing else — no frames, no transcript,
no filesystem paths — and returns structured JSON, not Markdown. Every
sentence carries the evidence ids it rests on, and a draft that fails validation
is discarded rather than repaired into something that passes. If no model is
configured, the model returns prose instead of JSON, or the draft fails
grounding, the deterministic renderer runs and the reason is printed. The
validator is never weakened to accommodate a model.

| `SYNTH_*` | `--synth-*` | meaning |
|---|---|---|
| `SYNTH_PROVIDER` | `--synth-provider` | `openai`, `openrouter`, `ollama`, `lmstudio`, `llamacpp`, `custom`, `none` |
| `SYNTH_MODEL` | `--synth-model` | model id |
| `SYNTH_BASE_URL` | `--synth-base-url` | OpenAI-compatible base URL |
| `SYNTH_API_KEY` | | credential; scoped to the provider, never sent to another |
| `SYNTH_OUTPUT_LANGUAGE` | `--output-language` | `auto` (follows the video) or a language code |
| | `--no-critic` | skip the second-pass critique |

## Install

The live copy an agent loads lives in `~/.claude/skills/yt-report/`. Point this repo at it so edits here are the edits there:

```bash
rm -rf ~/.claude/skills/yt-report
ln -s "$PWD" ~/.claude/skills/yt-report
```

## Requirements

- `uv` — the scripts are PEP-723 inline-script executables, so `uv run <script>` resolves their own dependencies. No project-level install step.
- `ffmpeg` 7+ — frame extraction and contact-sheet tiling.
- Optional: `mlx_whisper` (Apple Silicon) for the local ASR fallback when a video has no captions.
  It is a library the fetcher imports, not a flag; `--no-asr` turns the fallback off.
- macOS with `swiftc` — the OCR backend. It ships with the OS; the helper is compiled once and cached. Without it, runs still complete and record `ocr_status: "unavailable"`.

## Quick start

```bash
# fetch a video: transcript, meta.json, frames, contact sheets, visual.json, evidence.json
uv run scripts/yt_fetch.py "https://www.youtube.com/watch?v=..." --out /tmp/video

# no captions? local ASR is the automatic fallback; --no-asr turns it off
uv run scripts/yt_fetch.py "https://www.youtube.com/watch?v=..." --out /tmp/video --no-asr

# add or refresh the visual text layer on frames already on disk — no re-download
python3 scripts/ocr.py /tmp/video

# sheets went missing or came out empty? rebuild from frames on disk, no re-download
uv run scripts/resheet.py /tmp/video

# build, align, extract, validate, generate REPORT.md, validate the report
uv run scripts/build_evidence.py /tmp/video
uv run scripts/build_evidence.py /tmp/video --detail deep

# either gate on its own; both exit non-zero on a blocking failure
uv run scripts/evidence_validate.py /tmp/video
uv run scripts/report_validate.py  /tmp/video

# let a vision-language model interpret the frames OCR could not explain.
# Off by default: no key means no call, and every frame records why.
ANTHROPIC_API_KEY=sk-... uv run scripts/build_evidence.py /tmp/video \
    --vlm-provider anthropic --max-vlm-frames 8

# test the VLM configuration against ONE real frame — no download, no video
VLM_API_KEY=sk-... uv run scripts/vlm.py /tmp/video/frames/u0007.jpg
```

## The pipeline

```
yt_fetch.py  →  meta.json · transcript.md/.json · frames/ · frames.json · FRAMES.md
             →  ocr.py             →  visual.json         (per-frame text, confidence, boxes)
             →  build_evidence.py  →  align.py            (sentence ↔ frame pairing)
                                  →  vlm.py               (frame interpretation, opt-in)
                                  →  semantic.py          (typed items, each citing its source)
                                  →  evidence.json        (every item cites its source)
                                  →  evidence_validate    →  validation.json  (exit 1 blocks)
                                  →  synthesis.load_valid_evidence   (the only way in)
                                  →  report.py            →  REPORT.md · REPORT.json
                                  →  report_validate      →  report_validation.json
```

`frames/` is decided before the plan is made, and the plan is in `frames.json`
under `policy`. Two cheap low-resolution probes run first — one for the cuts, one
(about a dozen small frames) for the words on them — and together they pick a
sampling density: a talking head every ~30 s, slides every 12 s, a screen
recording every 8 s, code every 6 s. A uniform grid at that spacing is the
coverage guarantee and comes from the duration and the profile alone, so no
amount of cutting elsewhere in the video can move it or starve the tail; extra
frames go only to intervals the speaker pointed at or the picture changed in.
`--frames N` is a ceiling on the whole process, not per pass. `SAMPLING_VERSION`
is stamped into `frames.json` and a frame set written under a different one is
refused rather than read, because every count built on it would be a measurement
of a sampler that no longer runs.

`evidence.json` is the canonical record and the report's only input. The report
generator never sees the raw transcript or a frame, and has no free-text slot in
which to state a fact of its own — so "the report introduces no unsupported
claim" is a property of the call graph rather than a check run afterwards.
`report_validate.py` still checks it, because a property nobody tests is a
property nobody has.

## The report

`REPORT.md` is assembled from evidence fields: an item's `value`, `display`,
`qualifier`, `verbatim` and the timestamps its provenance resolves to. Provenance
is rendered inline and lightly — `[04:31 · narration]`, `[12:02 · on-screen
text]`, `[03:10 · model reading]`, `[metadata]` — so a creator's statement, a
string read off a frame, and a model's reading of a frame stay three different
kinds of claim.

Each item has exactly one **home**: a takeaway is not also a paragraph, a figure's
table row is its index and the narrative points at it. That is why the
duplicate-8-gram metric reads low without tuning. Length is derived from the
evidence — duration, segments, item count, importance, per-segment density,
visual evidence — so a short talking head is not padded and a dense lecture is
not truncated. `--detail brief|balanced|deep` sets the range; the evidence
decides where inside it a video lands.

Empty sections are not rendered. A hedge stays a hedge. An unreadable chart stays
unreadable. A conflict between two sources is shown, with both sides and both
timestamps, and is **not** resolved: that is a judgement about what to believe,
and nothing in this pipeline is allowed to make it.

`REPORT.json` holds the report's *structure* — which evidence ids each section
rendered — and no evidence text, so it cannot become a second source of truth.

## Vision-language pass

OCR reads the characters on a frame; it cannot say that a frame is a benchmark
table, that `32 tok/s` belongs to the row labelled `Qwen`, or that a bar is
missing. A VLM answers that, in a fixed structure, so the answer is checkable
rather than a caption.

**Off by default.** A model call costs money, so running one is a decision, never
a fallback that happens because a key was in the environment. With nothing
configured every frame records `vlm_status: "unavailable"` and no visual
observation is invented.

| provider | credential | endpoint |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` or `VLM_API_KEY` | Messages API, forced tool call |
| `openai` | `VLM_API_KEY` | `api.openai.com/v1` |
| `openrouter` | `VLM_API_KEY` | `openrouter.ai/api/v1` |
| `ollama` | **none** | `localhost:11434/v1` |
| `lmstudio` | **none** | `localhost:1234/v1` |
| `llamacpp` | **none** | `localhost:8080/v1` |
| `custom` | `VLM_API_KEY` | whatever `--vlm-base-url` says |

Configured by `VLM_PROVIDER`, `VLM_MODEL`, `VLM_BASE_URL`, `VLM_API_KEY`,
`VLM_TIMEOUT`, `VLM_MAX_FRAMES`, or the matching flags. The key goes in an
`Authorization` header and nowhere else: never printed, never logged, never in
the artifact, never in a cache key.

Every response is untrusted and passes the same gate whatever the provider:
parsed, then `VisionObservation.from_dict`, which is total and total means it
returns a record — a malformed one when the shape is wrong — rather than raising.
Malformed JSON, a partial object, a missing required field, a non-numeric or
out-of-range confidence, an enum outside the vocabulary, a relationship missing
an endpoint and a response naming a frame that was never asked about all become
`status: "malformed"` with the reason recorded. HTTP 401/403 latches and stops
the run; 429, 5xx and timeouts are per-frame failures; none of them is cached.

OpenAI-compatible endpoints disagree about structured output, so the provider
negotiates once — strict JSON schema, then `json_object`, then a prompt demanding
bare JSON — and records which mode answered. "The model was told to return JSON"
and "the endpoint enforced this schema" are not the same claim.

Model-derived items are marked `source: "vlm"`, capped at confidence **0.70**
(below every OCR band), and can never carry `verified_verbatim` — a description
of a frame is not a reading of its pixels.

## Visual text (OCR)

`ocr.py` reads the text off every sampled frame and writes `visual.json`. It is
an evidence-producing subsystem, not decoration: a number that is on screen and
nowhere in the transcript — a price in a screenshot, a figure in a benchmark
table — is exactly the information the transcript cannot supply, and without a
frame-backed read it has no provenance at all.

- **Provider boundary.** `OCRProvider.extract(path) -> OCRResult`. A backend sees a
  file and never a timestamp, so an OCR pass cannot invent or repair one. Vision
  is the shipped implementation on macOS; `UnavailableProvider` is the real
  fallback, so a machine with no backend records `status: "unavailable"` rather
  than behaving as though OCR ran.
- **Confidence policy.** Thresholds are set from what the backend actually returns
  (measured over 122 shipped frames: 91% of detections at 1.0, 5.7% at 0.5, 2.4%
  at 0.3). High ≥ 0.80, medium ≥ 0.50. A read at or above 0.80 may be marked
  `verified_verbatim`; below it the text is still an observation, but the
  constructor refuses to verify it and the validator rejects the claim.
- **Preprocessing.** Vision reads the extracted frame as-is. A frame whose mean
  confidence falls below 0.90 is re-read at 2× — the same pixels rendered larger,
  never assumed to hold more — with boxes mapped back to native pixel coordinates
  and `read_method: "upscale-2x"` recorded. The re-read is kept only when it
  recovers at least as much text.
- **Cache.** Keyed on frame content hash plus backend identity, version, language
  and read threshold. Changing any of them invalidates it. Re-OCRing 51 frames goes
  from ~12 s to ~0.1 s; the cache is ~11 KB per frame.
- **Deduplication.** Adjacent frames with identical text are recorded once, with
  the repeats pointing at the frame that carries the read. Normalisation folds
  whitespace and case and nothing else, so `$500` and `$5,000`, `7B` and `70B`,
  `0.20` and `2.0` and `32 GB` and `64 GB` can never collapse into one.

The validator enforces what the OCR layer claims: a read must name its backend and
status, boxes must lie on the frame, confidence must be in [0,1], a low-confidence
read may not be verified, and **the text a citation names must actually appear in
the frame's own stored read** — the check that catches a figure typed into an item
and attributed to pixels that never showed it.

## Development

Run the offline self-test before every commit. It needs no network and no API keys:

```bash
uv run scripts/selftest.py
```

It exists because two real regressions were silent — a run that produced zero contact sheets while reporting `"status": "ok"`, and a batch refactor that passed a list where a string was required. Neither surfaced downstream. It exercises the pure functions and the ffmpeg plumbing on a synthetic video so that class of break fails loudly.

Editing rules:
- Edit files here, not in `~/.claude/skills/` — that's a symlink to this repo.
- Every non-trivial logic change needs a check in `selftest.py`. That check is the runnable proof, not a comment.
- Keep `evidence_validate.py` and `report_validate.py` high-precision and low-recall. A checker that cries wolf trains people to ignore it; flag only defects identifiable without judgement, and report the ambiguous one as a metric rather than a failure.
- **Correct abstention beats unsupported detail.** A report that leaves a figure out and says so is a success; one that fills the gap is the failure this whole pipeline exists to prevent.

## Layout note

This is a product repo, not a wrapper: the source of truth for the skill is this working tree. `scripts/` is executable, `references/` is loaded on demand, and the two `SKILL.md` frontmatter fields (`name`, `description`) are the only contract the host reads before deciding to load the skill at all.
