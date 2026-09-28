---
name: yt-report
description: Turn a YouTube video into a grounded, source-anchored written report — an executive summary, detailed notes, an index of every figure, name, price and link the video contains, a timeline, and a plain statement of what could not be read. Use whenever the user pastes a YouTube link (or watch/youtu.be/shorts URL) and wants notes, a written record, a digest, or study material. Covers podcast, interview, lecture, screen-recording and AI-news videos. The report is generated from validated evidence, never written freehand: it is neutral third person, it introduces no fact the evidence does not carry, and it does not evaluate or fact-check.
---

# yt-report

Produce a **grounded written report of the video** — what it covers, what was
said, every figure it contains, and what could not be read — generated from the
validated evidence record rather than composed freehand.

## 1. Fetch

One command runs both halves and is what a user needs:

```bash
yt-report "<url>"                      # fetch → build → validate → REPORT.md
yt-report --help
```

It is `scripts/yt_fetch.py` followed by `scripts/build_evidence.py`, and it adds
no behaviour of its own. Run the two stages yourself when you want to intervene
between them — a re-run of the visual pass costs no download, which is the
common case:

```bash
uv run ~/.claude/skills/yt-report/scripts/yt_fetch.py "<url>"
```

Writes `~/Desktop/yt-reports/<slug>-<id>/`:

| file | what it is |
|---|---|
| `transcript.md` | full timestamped transcript, grouped by real YouTube chapters when they exist |
| `transcript.json` | the same fragments as data: start, duration, text — the evidence layer's unit of citation |
| `frames/` + `contact-sheet*.jpg` | sampled visual frames, and triage sheets showing all of them |
| `FRAMES.md` + `frames.json` | frame manifest with measured timestamps, human-readable and machine-readable |
| `visual.json` | every frame's OCR read: text, confidence, status, backend, per-region boxes — plus, when the visual pass ran, each frame's `vlm` block: what a model understood about the arrangement |
| `alignment.json` | which frames illustrate which sentences, with the method and the working |
| `meta.json` | title, channel, date, duration, view/like counts, description, tags, chapters |
| `evidence.json` | **the canonical record**: every item with its provenance, every segment, every frame |
| `validation.json` | the evidence layer's verdict on itself — read this before trusting the rest |
| `REPORT.md` | **the report**, generated from `evidence.json` and nothing else |
| `REPORT.json` | the report's structure: which evidence ids each section rendered. No evidence text |
| `report_validation.json` | the report's grounding verdict — the last gate before the file is written |

Captions first, local Whisper ASR as the automatic fallback — no flag, no API key, nothing uploaded. The one 720p download serves both ASR and frames, so it retries rather than losing both. Transcripts cache per video+language, so a re-run costs seconds.

Visual text is read by macOS Vision (`scripts/ocr.py`) on every frame by default. It is a separate pass from the fetch and re-runs over frames already on disk, so adding it to an old artifact directory costs no re-download: `uv run scripts/ocr.py <artifact-dir>`. Cached on frame content plus backend config, so a re-run reads nothing twice. `--no-ocr` skips it; when no backend is available the run still completes and records `ocr_status: "unavailable"` — never a silent pass that looks like a clean read.

**The visual pass answers what OCR cannot.** OCR reads characters; it cannot say that a frame is a benchmark table, that `32 tok/s` sits in the row labelled `Qwen`, that a bar is missing, or that the arrow points from the API to the retriever. A vision-language model is asked about a small, budgeted set of frames and answers in a fixed structure, so its reading is checkable rather than a caption:

```bash
uv run scripts/build_evidence.py <artifact-dir> --vlm anthropic --max-vlm-frames 8
```

It is **off by default**, because a model call costs money and that has to be a decision, not a fallback that happens because a key was in the environment. With `--vlm none`, or with a named backend and no credentials, every frame carries `vlm_status: "unavailable"` and **no visual observation is invented**. What it recovers, when it runs: a frame's content type, values with the row and column they were seen under, diagram edges with both endpoints, actions on screen, entities a second modality also names, and whether the frame supports a spoken claim.

What it is trusted with is narrower than what it returns. A value whose digits appear in neither the frame's own read nor the speech is dropped; a row label the frame does not show is dropped while the value is kept unlabelled; an entity no other modality names is dropped; what could not be read is kept as an `uncertainties` record rather than guessed. **Every refusal is written to `observations` with its reason**, so a smaller document than the model returned is a decision you can read and reverse, not a filter you have to trust. The model never overwrites the OCR read — where the two disagree, both are kept with their own provenance, and the disagreement may become an unresolved `conflict`.

Model-derived items are marked `source: "vlm"`, capped at confidence **0.70** (below every OCR band), and can never carry `verified_verbatim` — a description of a frame is not a reading of its pixels. A `vlm` citation on a frame nothing analysed is a blocking validation error, as is one whose text the stored observation does not contain.



**Alignment and semantic extraction are a third pass**, over the artifacts above, and it too re-runs on an existing directory for no download cost:

```bash
uv run ~/.claude/skills/yt-report/scripts/build_evidence.py <artifact-dir>
```

It pairs each sentence with the frames near it, then reads the OCR against what is being said, and writes `alignment.json`, `evidence.json` and `validation.json`. It is deterministic and offline: same input, same file, byte for byte.

Status: `0` ok · `2` no captions, ASR disabled or failed · `3` no captions obtained · `4` the evidence layer rejected what it built. **A non-zero status still emits the frames** — a caption-less video is not a failed fetch.

**Flags you will actually reach for:** `--lang` (non-English) · `--frames -1` (frames only, no video) · `--batch` (several URLs) · `--out DIR` · `--no-ocr` · `--vlm anthropic` (full pipeline, frames interpreted) · `--max-vlm-frames N`. The rest are in `--help`.

**Do not pass `--frames`.** The count is automatic, and the policy reads the video before it commits to one: a coarse low-resolution probe finds the cuts, a dozen small frames are read for the words on them, and those two together pick a spacing — ~1 per 30 s for a talking head, 12 s for slides, 8 s for a screen recording, 6 s for code. A uniform grid at that spacing guarantees the whole video is covered whatever else happens, and extra frames go only to the intervals the speaker pointed at or the picture changed in. **`--frames N` is a hard ceiling on the whole process**, not per pass, and `meta.json`'s `visual` block reports what each layer asked for (`uniform_candidates`, `scene_candidates`, `refinement_candidates`) beside what survived. If the numbers say refinement outnumbered the baseline, say so rather than shipping it.

**English only** — the script will not report another language's captions as English. If the audio is not the language asked for either, say so and ask.

→ `references/browser-fallback.md`, only when `download_video` itself fails. A playability oracle and URL source, **not** a caption-block bypass.

## 2. What you still read by hand

The pipeline reads the transcript, the frames and the on-screen text, and it
records what it decided about every piece of visible text. That is not the same
as having read the video, and the difference is where your judgement is still
required. Three jobs remain.

**1. Sanity-check the report against the evidence.** Open `REPORT.md` and
`evidence.json` side by side. The report is grounded by construction and checked
by `report_validate.py`; neither of those can tell you the *extraction* was
wrong. A report that says a model reads at 32 tok/s when the frame says 41 has
passed every gate in this repository, because the gates check traceability and
not eyesight. When a claim looks wrong, the fix is to open the frame — not to
edit the report.

```bash
uv run scripts/ocr.py <artifact-dir>              # re-run the visual pass, no re-download
uv run scripts/label_sheet.py <artifact-dir>      # map frame ids to contents
```

**2. Open any frame the report leans on.** `visual.json` carries every frame's
OCR read with a confidence and a backend, so you know which frames hold text
before you spend an open on them. A frame with `ocr_status: "no_text"` was read
and held nothing legible; `"unavailable"` or `"failed"` means the pass did not
happen for it and no claim about its contents may be made either way. For
**figures, prices, URLs and version strings** the read is exact and carries a
provenance the validator checks. **A quotation is still not OCR's to give** — the
engine gets prose punctuation wrong and misreads a letter in small text, and §4.4
requires a magnified re-read before a figure is written down.

**A frame read can return fewer images than you asked for, with nothing marking
the drop.** When you open frames individually rather than as a labelled sheet,
count the images before matching them to filenames by position — a dropped
result shifts every later one by one, so a run of frames each off by one means a
result was lost, not misread.

**3. Know what the extraction decided.** `observations` records what the
extractor *decided about every piece of visible text*, including the ones it did
not turn into evidence, each with the reason. Read it when a frame seems to hold
a figure the report is missing: the string is still there in
`visual[].ocr_text`, and the classification is reversible.

**Chapter titles are uploader metadata, not a verified index** — hand-typed,
generated from an early cut, or left over. On one 20-minute video 2 of 7 did not
describe what played under them. Timestamps usually survive even when titles do
not, and the Timeline labels come from them, so read them as labels and not as
findings.

## 3. Generate the report

**You do not write `REPORT.md`.** The report is generated, and generated for a
reason this skill used to get wrong: when a model composes prose from a
transcript in one pass, every fact in the output is an assertion and a fabricated
figure is indistinguishable from a verified one. The extraction layer now exists
underneath, so the report is a *renderer* over it — one that has no free-text
slot, and therefore no route by which a fact can enter that the evidence does not
already carry.

```bash
uv run ~/.claude/skills/yt-report/scripts/build_evidence.py <artifact-dir>
```

That one command runs the whole second half of the pipeline — build, align,
extract, validate, generate the report, validate the report. `--detail` sets the
length budget (`brief` / `balanced` / `deep`, default `balanced`); the evidence
decides where inside that range a particular video lands, and a video with three
extractable facts gets three facts, not padding.

**The two gates are not advisory.** `validation.json` must be clean before a
report exists at all, and the report is rendered in memory and checked *before*
it is written: an unsupported number, an invented URL, a timestamp matching no
snippet, frame or segment, a visual statement with no frame behind it, a dropped
conflict, or a heading with nothing under it all refuse the write. Re-check
either on its own at any time:

```bash
uv run scripts/evidence_validate.py <artifact-dir>     # is the evidence coherent?
uv run scripts/report_validate.py <artifact-dir>       # is the report grounded in it?
```

### Writing the report with a model (optional)

The renderer is deterministic, is the default, and is the fallback. A model can
write the prose instead. It sees validated evidence and nothing else — no
frames, no transcript, no paths — and returns structured JSON that every
sentence's evidence ids are checked against:

```bash
SYNTH_API_KEY=sk-... uv run scripts/build_evidence.py <artifact-dir> \
  --synth-provider openrouter --synth-model openai/gpt-4o-mini

# regenerate from evidence already on disk: no download, no OCR, no VLM pass
uv run scripts/build_evidence.py <artifact-dir> --synthesis-only \
  --synth-provider openrouter --synth-model openai/gpt-4o-mini
```

Off unless asked for, because a model call costs money. It writes only if the
draft survives validation; if no model is configured, if it answers in prose
instead of JSON, or if the draft fails grounding, the deterministic renderer
runs and the reason is printed. A second-pass critique may send the writer back
once; if the revision still fails, the deterministic renderer runs rather than
the check being lowered.

**`--output-language` only does anything when a writer is configured.** It is a
property of the model-written draft, and the deterministic renderer has no
translations to choose between: run `--output-language ja` with no
`--synth-provider` and the report comes out byte-identical to the
`--output-language auto` one, because the renderer's prose is a fixed English
template. What the flag *never* does is touch the evidence store — the
transcript, the extracted names, the identifiers, the URLs, the code and the
figures are the source language's whatever language the prose is in. Measured on
a German video, all four of `auto`, `en`, `ja` and `es` produced byte-identical
reports with the deterministic renderer.

### The report's shape

```markdown
# <Video title>

> **DEGRADED.** transcript only; no frame was read for on-screen text.   ← only when degraded

## Executive Summary
## Key Takeaways
## Detailed Notes          ← organised by segment; the primary narrative
## Extracted Details
### People & Organizations
### Products, Models & Tools
### Other Named Things          ← named, but the extractor established no type
### Numbers, Pricing & Benchmarks
### Steps & Workflows
### Comparisons
### Resources & Links
### Important Quotes
## Timeline
## Source & Reliability
```

**Sections with no evidence are not rendered.** A video with no procedures has no
Steps section; a talking head with nothing on screen has no Numbers section. Do
not add one to fill the shape, and do not add one by hand afterwards.

**Provenance is rendered inline and lightly** — `[04:31 · narration]`,
`[12:02 · on-screen text]`, `[03:10 · model reading]`, `[metadata]`. The tag
preserves the distinction the evidence model exists to keep: a creator's
statement, a string read off a frame, and a model's reading of a frame are three
different kinds of claim, and a report that merges them has thrown away the thing
the reader most needs.

**Each evidence item has one home.** An item that becomes a Key Takeaway is not
also a paragraph in Detailed Notes; a figure's row in the numbers table is its
index, and the narrative refers to it rather than re-deriving it. This is
structural, not stylistic — it is why the duplicate-8-gram metric in
`report_validate.py` reads low without being tuned.

### Configuring the vision-language pass

Off by default: a model call costs money, so running one is a decision, never a
fallback that happens because a key was in the environment. With no provider
configured every frame records `vlm_status: "unavailable"` and **no visual
observation is invented**.

| variable | flag | meaning |
|---|---|---|
| `VLM_PROVIDER` | `--vlm-provider` | `anthropic`, `openai`, `openrouter`, `ollama`, `lmstudio`, `llamacpp`, `custom`, `none` |
| `VLM_MODEL` | `--vlm-model` | model id for the chosen provider |
| `VLM_BASE_URL` | `--vlm-base-url` | OpenAI-compatible base URL |
| `VLM_API_KEY` | — | credential. Never printed, never logged, never written into an artifact |
| `VLM_TIMEOUT` | `--vlm-timeout` | per-call timeout, seconds |
| | `VLM_MAX_FRAMES` | `--max-vlm-frames` | hard cap on model calls for the run |

`--no-vlm` skips the pass outright (the same as `--vlm-provider none`), and
`--no-report` stops after the evidence gates if you only want the record.

```bash
export VLM_API_KEY=...                      # OpenRouter
uv run scripts/build_evidence.py <dir> --vlm-provider openrouter \
    --vlm-model anthropic/claude-sonnet-5 --max-vlm-frames 8

# a local model, no key at all
uv run scripts/build_evidence.py <dir> --vlm-provider ollama --vlm-model qwen2.5vl:7b

# any OpenAI-compatible endpoint nobody has a preset for
uv run scripts/build_evidence.py <dir> --vlm-provider custom \
    --vlm-base-url http://box.internal:8000/v1
```

Local servers are configured without a credential — requiring `VLM_API_KEY` for
a server that has no account system would make a complete configuration report as
incomplete.

**Every OpenAI-compatible endpoint behaves differently about structured output**,
so the provider negotiates once and records what it got: it asks for a strict JSON
schema, drops to `json_object` if the endpoint rejects the parameter, and falls
back to a prompt demanding bare JSON if that is refused too. Whichever mode
worked is written onto every observation, because "the model was told to return
JSON" and "the endpoint enforced this schema" are not the same claim.

**Test the configuration against one real frame before a whole run.** No
download, no video, no evidence document — and a non-zero exit on failure:

```bash
uv run scripts/vlm.py <artifact-dir>/frames/u0007.jpg
```

**When you need to know *which* part of a provider is broken**, use a doctor.
Each walks its capability chain one step at a time and reports every step
separately, because the steps fail independently and the fix differs for each.
All three are the same command from the skill's front door and all three are the
same code — one implementation of "a chain of steps where a step that never ran
is not a pass".

```bash
yt-report vlm-doctor        # can this provider produce a valid observation?
yt-report synthesis-doctor  # can this model write a report from evidence?
yt-report critic-doctor     # can it find a planted defect, and stay quiet?
yt-report eval-live <url>   # one budgeted, measured run against real providers
yt-report release-check     # which capabilities this provider has actually been shown to have
```

**`vlm-doctor`** — `reachable`, `authenticated`, `model_exists`, `vision_capable`,
`structured_requested`, `structured_enforced`, `test_observation_valid`.
`qualified` is true only when all seven hold. A gateway that accepts
`response_format`, returns 200 and answers in prose passes four of the seven
and has produced nothing usable — which is why *requested* and *enforced* are
separate verdicts, and only the returned payload settles the second.

It prints `configured=False` when nothing is set up, `provider_unavailable` when
something was asked for and could not be built, and the provider, status, content
type, structure counts and confidence cap on success. A provider that is
misnamed, mis-addressed or unauthorised looks exactly like one that had nothing
to say, and this is the only way to tell them apart before spending a run on it.

**`synthesis-doctor`** — ten steps, ending in `draft_grounded` and
`report_validated`, and it makes exactly **one** model call. With no argument it
runs against a built-in fixture carrying one of each thing the writer has to get
right (a topic, a claim, a number, a visual observation, an entity, an
uncertainty, a conflict), so it runs in seconds with no video and no download;
give it an artifact directory to qualify against that video's real evidence.

```bash
yt-report synthesis-doctor --synth-provider openrouter --synth-model openai/gpt-4o-mini
yt-report synthesis-doctor <artifact-dir> --json
```

**HTTP 200 is never a verdict.** The step that separates `account_paid` from
`model_exists` exists because an exhausted balance and a missing model arrive as
the same class of "the call did not work" and have nothing in common as fixes:
one needs a payment, the other a different model id. Billing is checked before
rate limiting, because both arrive as HTTP 429 and filing a payment refusal as a
transient rate limit tells the user to wait.

**`critic-doctor`** — four known-answer cases against a deliberately tiny
fixture: a clean draft, a figure changed in the prose, a hedge removed, and a
sentence no cited item supports. The clean case is first, because everything
after it is meaningless if the critic objects to a sound draft. One call per
case, and a case that did not run within the budget is printed as *not run*,
never as a pass.

**`eval-live <url>`** is the cheap real run: `--max-vlm-frames 2
--max-synthesis-calls 1 --max-critic-calls 1` by default. Each is a hard ceiling
spent *before* each call, so a limit that is reported rather than enforced
produces a bill and a report saying the limit was respected, and only one of
those is true. It writes `LIVE-EVAL-<timestamp>.md` recording the exact call
counts, cache hits and latency, then what came out — conformant and useful VLM
observations, entities, numbers by role, report word count, provenance coverage,
unsupported claims and numbers, and whether the conflicts survived. No overall
score: one number over six dimensions hides the one that went wrong, which is the
only reason anyone looks.

**`release-check`** is the cheap qualification sequence, and it exists because
the four capabilities are qualified in four different places and nothing
collected the answers. It runs connectivity, then the VLM doctor, then the
synthesis doctor, then the critic doctor, then — only if you pass a URL — one
small end-to-end video, and prints a matrix:

```text
provider    model    vision    structured_vlm  synthesis  critic    status
openrouter  gpt-4o   UNTESTED  UNTESTED        UNTESTED   UNTESTED  CONDITIONAL
```

Each step carries one of four states, and two of them are adjacent enough to be
confused: `PASS` (a real request held), `FAIL` (a real request came back and did
not hold — the capability is disproven), `UNTESTED` (no usable answer ever
arrived: no credential, no credit, no egress, a spent budget), and `SKIPPED`
(not requested). A step is `FAIL` only when an answer came back, which is the
whole point: converting a `FAIL` into an `UNTESTED` launders a broken provider
into an unexamined one, and converting an `UNTESTED` into a `PASS` launders no
credit into a qualification.

The matrix's `status` column is the sentence to act on: `QUALIFIED` (every
capability exercised and holding), `CONDITIONAL` (not fully exercised), `DEGRADED`
(an optional model layer failed while the report still landed), `UNQUALIFIED`
(a capability failed with nothing working beside it). The release claim is
`CONDITIONAL` until a real synthesis, VLM and critic run have all been exercised
successfully — not because the harness is strict, but because the alternative is
a claim nobody checked.

Budgets are ceilings and are never silently exceeded:
`--max-vlm-frames 2 --max-synthesis-calls 1 --max-critic-calls 1
--max-revision-calls 1`. `--json` and `--out PATH` write the machine-readable
form. Credentials are never printed.

**After changing any script here, run the self-test** — pure functions, cache,
ffmpeg on a synthetic video, the retry path against a faked downloader, every VLM
response shape and failure, and the whole report layer over fifteen video
fixtures. No network:

```bash
uv run ~/.claude/skills/yt-report/scripts/selftest.py
```

It exists because a change once shipped that produced **zero** contact sheets
while reporting `"status": "ok"`, and a refactor once passed a list where a
string was required. To rebuild the skill's own sheets from frames already on
disk, use `resheet.py`.

### Changing a script invalidates its cache — bump the version

Every cached answer is keyed on the version of the thing that produced it, so a
rule change that does not bump its version keeps serving the previous answer.
Nothing fails: the document comes back built under the old rules, well-formed,
with no indication a cache was involved. That happened three times in one
session.

All versions live in `scripts/versions.py`, so the obligation has one address:

| you changed | bump |
|---|---|
| what `semantic.py` produces from a segment — ids, item shapes, classification, subject rules | `EXTRACTION_VERSION` |
| the instructions to either model layer | `VLM_PROMPT_VERSION` / `SYNTH_PROMPT_VERSION` |
| the shape of `evidence.json`, or the synthesis JSON schema | `SCHEMA_VERSION` / `SYNTH_SCHEMA_VERSION` |

`selftest_cache.py` asserts each one moves its cache key, so the test suite fails
if a version stops being read. It is deliberately **not** derived by hashing the
source tree: that would remove the obligation, and it would do it by making every
comment edit invalidate every cached segment — a correctness mechanism whose
visible symptom is only that everything got slower for no stated reason.

## 4. Rules

Each is stated once, deliberately. Most of the reasoning is in the observation log; restating a rule does not make it hold harder.

### 4.1 Voice and stance

**Neutral third person.** "The video discusses", "the creator states" — never
"I found", "I noticed", "I think", never first-person imitation. This is not a
style preference. The report used to be written in the creator's own voice, and
the voice was load-bearing for a specific failure: §4.2's ban on meta-language
forbade the words "on screen" and "the transcript reads", so the reports that
obeyed it most carried **zero** provenance markers while the reports that broke
it carried the most. A rule that suppresses the apparatus the checker verifies
has disarmed the checker, not tightened the writing.

**Report the material; do not evaluate it.** No assessing whether a claim is
right or well-sourced. No ranking or scoring the models, products or people
covered. No fact-checking, no arithmetic on the reasoning, no repairing an
argument, no outside knowledge, no "added context". No section whose purpose is
to make up a reader's mind — not "questions worth asking", not "what this gets
wrong", not "should you trust it".

**Their opinions are theirs.** Where the creator expresses a view, the report
carries it in their words at its strength. What does not belong is a view of

*their sourcing* that they never expressed.

**Hedging is preserved, not managed.** If the creator said a deployment "could
cost $500", the report says could cost $500, and the figure is rendered as
"given as a possibility". Not flattened to *costs $500*, not inflated to *an
unsupported $500 claim*. Both are a verdict; neither is a record.

**Tension gets stated, never resolved.** If two things in the material disagree,
the report says so and stops: *"The narration states 75% while the frame reads
8%. Sources: 75% ([09:00 · narration]); 8% ([09:12 · on-screen text]). Not
resolved here."* Not "these are inconsistent", not "the real price is probably…".
Spotting the tension is the job. Settling it is not, and nothing in this
pipeline is allowed to do it — resolving a contradiction is a judgement about
which source to believe, and a generator that makes that call has stopped being a
record of what was said.

**A faithful record of a wrong claim is a success.** A claim reached by the
generator's own reasoning is a failure.

**The report must stand alone.** A reader opening it has not seen the video.

### 4.2 Accuracy, internal

These govern what goes in. None of them is ever narrated in the piece — the work surfaces as accuracy, not as method.

- **A string is from a frame only if it is visible in that frame.** The `meta.json` description is uploader metadata with no standing of its own: a title, tag, chapter name or blurb found there is not a quotation, and moving one into a quotation slot is a fabrication however true it is.
- **Never present as a quotation a line that is not on the card.** A creator who reads their slides aloud makes it look as though every spoken line is also shown. It usually is not. Check the frame, or write it as my own summary.
- **A contact sheet decides which frame to open; it never supplies contents.** Nothing goes in about a frame until that frame has been opened. A 384px tile tells you *that* a terminal is there, never what is in it.
- **Provenance does not travel along a timeline.** A source belongs to the thing it belongs to and to nothing else — not the segment, not the adjacent frame, not whatever was nearby. Both failure modes count: attributing a source I did not read, and **leaving one out when the material does show it**, which hides information.
- **The transcript garbles names; a frame does not.** When the audio mangles a proper noun, the printed name is correct and wins — replacing a legible one with the term I expected is a real error. **Never invent a correction**: a note claiming the captions "rendered X for Y" when neither string occurs anywhere is worse than no note, because it is a fabrication dressed as rigour. With no frame or transcript line to support a fix, say the name was unclear.
- **`upload_date` is not a filming date.** If the material never says when it was recorded, give the upload date and stop.
- **A capital letter is not a name.** Extraction is a capitalisation test, and capitalisation is an English-language convention. German capitalises every noun as a spelling rule, so `Aber`, `Sie`, `Dinge`, `Universums` and `Anthropic` are indistinguishable to a regex — and a German report listed all of them under *Products, Models & Tools*. Ordinary vocabulary is filtered per language, from `resources/stopwords/<lang>.txt` (en, de, es, fr, pt, it, hi), matched by German lemma so `Universums` matches `universum`. That list is **one signal**, not the classifier: corroboration (seen more than once, or in speech and on screen) and category vocabulary do the rest, and a missing noun costs one false candidate rather than a broken filter.
- **A measurement written in words is not a name.** `Millionen Lichtjahre` and `Millionen Stundenkilometern` are distances. A candidate carrying a word that names a unit of measure is a measurement, not a thing with a name.
- **An untyped name is not a product.** When no surrounding vocabulary types a candidate, it comes back as `entity` and is listed under *Other Named Things* as "type not established". `product` used to be the catch-all, and "Obsidian — product" asserted something the evidence does not. When both are wrong the failure is under-extraction, and that is the cheaper one: a false positive in a list of names is more damaging than a missing one.

### 4.3 Describing frames

- **A correct fact does not validate where I read it.** Re-open the frame and read the string off it in the same pass that writes the sentence. The mechanism is adjacency: in a fast animation neighbouring samples are near-identical, and a re-read late in a long session silently reorders the mental index. Adjacent frames are the danger zone; the tail of a sequence is worst. **An OCR read in `visual.json` is a read of the pixels, so it discharges this for the figures, prices, URLs and version strings it covers; it does not discharge it for a quotation**, because the engine varies punctuation and misreads a letter in small text.
- **Verify layout claims specifically.** A four-panel montage, a two-column comparison, a stack of diagrams — composite descriptions are where a conflated neighbour does the most damage, because every element is real somewhere in the sequence.
- **Re-read characters, not just words.** A code review quoted as `span.FontSize != 0` where the frame read `span.FontSize >= 0` — every identifier right, only the operator wrong, and it reversed the meaning. `<`/`>`, `<=`/`>=`, `!=`/`==`, `&`/`&&`, `0`/`0.0` and `1`/`-1` are the swaps that survive proofreading; so is a config key differing by two characters from the one I expected.
- **If one frame is described two different ways, one is wrong.** Chase it before anything else.
- **Describe comparative layouts as they are** — down to the detail that two generated pages carry different titles, or that one has a brand mark the other lacks. Calling that a flaw, a win, or evidence about which is better is not description.
- **Do not build the piece on a single reading of a single frame.** A distinguishing detail repeated in the overview, a section and a table multiplies one unverified reading into several confident assertions. Verify the load-bearing frame first.
- **State a count once, and make the list match it.**

### 4.4 Numbers

**Magnify before writing a figure.** A contact-sheet tile is 384px — enough to spot that a chart exists, never enough to read its values. Crop the region and upscale 3–5x, then read it:

```bash
ffmpeg -v error -y -i frames/s0007.jpg \
  -vf "crop=iw*0.5:ih*0.6:iw*0.5:ih*0.05,scale=iw*3:ih*3:flags=lanczos" /tmp/crop.png
```

If a value is still illegible, say it is illegible. Do not round it into a plausible-looking number, and never carry a number over from a different chart.

**No bar, no number.** An absent or `N/A` bar means the series was **not measured**. Check the legend and count the bars in each group first, then describe the group as showing one measured series. Filling it in invents a measurement; calling it a win inverts the chart.

**Do not call a figure unreadable without trying harder.** "Unreadable" is itself a transcription claim, and a bad one — it silently drops a correct number and closes the question. It happened when an annotation was written off as "cut off" and a better crop found it perfectly clear. Before writing it off: re-crop from a **different frame of the same page** (scroll frames often render the same figure larger), try another resampling flag, and calibrate numerically if it is a plotted point (map pixel positions to axis units). Only after those, and say what I tried.

**Transcribe every figure, including ones that look inconsistent with each other.** Reproducing them faithfully is the job. What they mean is the reader's problem.

**A figure has to be about something.** Transcription is not the same as
reporting. `21 — Browse all skills`, `173K — & system`, `127 — CA` and
`02 — PHASE TWO` are all read correctly and none of them is a fact a reader
can use, because nothing in the video says what any of them counts. They stay
in `evidence.json` and in `visual[]` — a classification withholds a figure from
one section, it never deletes an observation — and they stay out of "Numbers,
Pricing & Benchmarks".

What separates the two is `semantic_role`, computed by the extractor from six
signals and recorded with the reason: the value and its unit also appear in the
aligned speech; the label names a *measure* (`users`, `latency`, `$`) rather than
furniture (`chapter`, `views`, `subscribers`); the label is an ordinal or a
table cell; the unit makes it a price, a duration or a rate; the frame lays its
text out as a table **and** the segment is about measurements; and the frame's
own content type says what kind of picture it is. `ui` and `unknown` are
withheld; `benchmark`, `price`, `metric`, `configuration` and `count` are kept.

Two of those deserve their own line. **Table structure alone is not enough** — a
grid of numbers with labels beside them is also what a YouTube end screen looks
like, and the speaker's subject is the only thing that tells them apart. And
**`importance` is not `confidence`**: a subscriber counter read at 0.99 is `ui`,
and a benchmark bar read at 0.78 is `benchmark`. A perfectly legible interface
number is worth almost nothing, and deriving one from the other makes every
confidently-read piece of furniture look load-bearing.

### 4.5 Absence

**Never assert a negative I have not checked every frame to support.** "No frame shows this", "there is no chart" — nobody reads a negative as needing evidence, which makes it the easiest thing in a piece to get wrong. A piece once opened one contact sheet of two and concluded "exactly one" of 52 frames contained text; four opened frames refuted it, one carrying the prompt verbatim and differing from the captions in two places.

- Absence claims require checking **every** frame, or they are not made.
- If I have not looked everywhere, the honest form is scoped: "I did not see this anywhere in the material I went through."
- A universal negative is the natural place for a fabrication to hide, because it frees the writer from quoting anything. **Anything that removes the need to open a frame deserves suspicion.**

"Unreadable" and "absent" are different claims, and both are claims.

### 4.6 When something has to come out

**Removal is not annotation.** A wrong fact usually sits in four places at once — the overview, a section, a table, a closing line. "Retracted, correction follows" repairs the one spot a reviewer read and leaves the other three asserting it. So: **grep for the distinctive terms and delete every occurrence**, then re-read to confirm. Prefer deleting the words to qualifying them.

**A fix instruction can be wrong.** If told to re-pin a fact to a specific frame, open that frame and check. In two separate cases it was on neither the named frame nor the proposed one, and deleting the fact was correct. A reviewer once asserted a chart plotted a rival "further right, at ~195"; re-deriving the calibration reproduced every verified value exactly and showed no such point existed. So: if it is not there, **remove it rather than relocate it** — a plausible new home is as false as the old one. When a reviewer's premise is overturned, say so and show what was measured.

## 5. Degraded modes, stated in the report

The report states its own coverage in one line under the title, computed from the
artifact's own fields — never inferred, never left for the reader to work out
from an absence:

| state | when | what the report says |
|---|---|---|
| **FULL** | transcript + on-screen text + a model reading of selected frames | no banner |
| **DEGRADED** | transcript + OCR, VLM unavailable | "...no model reading of the frames, so diagrams, tables and charts are not interpreted" |
| **DEGRADED** | transcript only, visual unavailable | "...no frame was read for on-screen text, so nothing here comes from the visuals" |
| **FAILED** | no usable transcript and no usable ASR | no report is written |

A transcript-only report presented as though a visual analysis had run is the
specific dishonesty this state exists to prevent. If the banner is missing and
the report claims a chart, something is wrong — check `report_validation.json`.

## 6. After writing

Tell the user the path and the coverage state, and offer: a shorter version
(`--detail brief`), an Obsidian note, or follow-up Q&A on any section. If the
report was generated in DEGRADED mode, say which layer was missing before
anything else — a reader who was told a chart "shows" X needs to know whether
anything looked at a chart.

## Sibling skills

Pairs with `agent-reach` (fetches from other platforms) and `obsidian-vault` (note storage). This skill owns the YouTube path only.
