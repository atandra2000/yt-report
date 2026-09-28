# REAL-PRODUCT-EVAL.md

Real-world product qualification run, 2026-09-26. Repository frozen: no code, validator,
prompt, extraction-logic or report-generation change was made. Defects found are recorded,
not fixed.

## 1. Date / time

2026-09-26, macOS 27.0.0 (darwin), cwd `/Users/atandrabharati/Desktop/CoreProjects/yt-report`.
All model calls made ~10:15 to ~11:00 local.

## 2. Provider / model

| Role | Requested by brief | Actually exercised |
|---|---|---|
| VLM | `openrouter` / `openai/gpt-4o-mini` | `openrouter` / `stealth/space-bunny-alpha` |
| Synthesis | `openrouter` / `openai/gpt-4o-mini` | `openrouter` / `stealth/space-bunny-alpha` |
| Critic | `openrouter` / `openai/gpt-4o-mini` | `openrouter` / `stealth/space-bunny-alpha` |

`openai/gpt-4o-mini` is untestable. `https://openrouter.ai/api/v1/credits` returns
`total_credits 46 / total_usage 46.197` — the account is exhausted. Every call returned:

    HTTP 402: the provider accepted the credential but the account cannot pay for this
    call (This request requires more credits, or fewer max_tokens. You requested up to
    2000 tokens, but can only afford 948. To increase, visit
    https://openrouter.ai/settings/credits and add more credits)

release-check classified this as `http_failure_class: no-credit`, reason: "the credential
was accepted and the account cannot pay. This is not a rate limit and waiting will not
help." That classification is correct and useful.

On the user's instruction the run switched to `stealth/space-bunny-alpha` on the same
OpenRouter account. That model **is** reachable, authenticated, resolvable and billed
(`account_paid: ok` — "the call was billed"), so it produced real round-trips. The
`/credits` endpoint and the billing path disagree; recorded as an observation, not a
defect claim, since the account may hold a per-model allowance.

## 3. Release-check result

Command: `yt-report release-check --vlm-provider openrouter --vlm-model stealth/space-bunny-alpha --synth-provider openrouter --synth-model stealth/space-bunny-alpha --out /tmp/rc.json`

Overall status **UNQUALIFIED**. Budget used: 1 call, 0 frames, 0 revisions, 145.9 s.

| Step | State | Reason as reported |
|---|---|---|
| connectivity | UNTESTED | vlm: UNTESTED; synthesis: UNTESTED |
| vlm | UNTESTED | the model answered no usable observation |
| synthesis | UNTESTED | the model returned no usable content |
| critic | **FAIL** | the critic did not match the expected answer for: clean |
| end-to-end | SKIPPED | no URL given |

Matrix: `{"provider":"openrouter","model":"stealth/space-bunny-alpha","vision":"UNTESTED","structured_vlm":"UNTESTED","synthesis":"UNTESTED","critic":"FAIL","status":"UNQUALIFIED"}`

Per-check detail:

- **vlm** — `reachable ok (5022 ms)`, `authenticated ok`, `model_exists ok`,
  `account_paid` not reached. `vision_capable FAIL`: `content_type 'other' is not in the
  schema vocabulary`. `structured_requested`, `structured_enforced`, `test_observation_valid`
  all `unreachable`.
- **synthesis** — `reachable ok`, `authenticated ok`, `account_paid ok`, `model_exists ok`,
  then `model_completes FAIL`: "response hit the token limit before any content",
  140.86 s. Every downstream check `unreachable`.
- **critic** — `reachable ok`, `authenticated ok`, `account_paid ok`, `model_exists ok`,
  `model_completes ok (1/1)`, `structured_enforced ok`, `critic_reachable ok (1/1)`,
  then `cases_matched FAIL`: 0/4 matched, only 1 ran (budget).

The tool did **not** interpret HTTP 200 as success anywhere: reachable/authenticated/model_exists
are reported separately from `account_paid` and from content validity, and the 402 was
classified by cause rather than by status code.

## 4. VLM qualification

Three existing frames, chosen for content type. Frames were read at native resolution by
me first so the model's output could be checked against a known-correct reading.

| # | Frame | Content type | What the frame actually shows |
|---|---|---|---|
| 1 | `4TvH-OZhwxI/frames/u0035.jpg` | chart/diagram with figures | "AI Manager" workflow map; nodes labelled Plan 73%, Review 83%, Release 25%, Test 50/55%, Design 64%, Develop 29%; small presenter inset bottom-left |
| 2 | `aircAruvnKk/frames/u0011.jpg` | network diagram | input digit "4" -> 784 input nodes -> two boxed "Hidden layers" -> 10 output nodes labelled 0-9; a "?" in the first hidden layer |
| 3 | `4TvH-OZhwxI/frames/u0018.jpg` | screen/UI/code | VS Code diff on `Button.module.css`: `.Button[data-hovered]:not([data-disabled])`, `box-shadow: rgb(255 255 255) 1.2px ...`, `width: 100%`, agent panel "Added hover tracking so the button responds to a data-hovered flag" |

`yt-report vlm-doctor --frame <f> --vlm-provider openrouter --vlm-model stealth/space-bunny-alpha`
run once per frame. All three: **NOT QUALIFIED**.

| Frame | latency | schema failure reported |
|---|---|---|
| u0035 (chart) | 53.1 s | `visual_elements[0..3] has no 'kind'` |
| u0011 (diagram) | 80.9 s | `visual_elements is not a list`; `relationships[0..2] has no 'source'`; `a relationship is missing its 'to' endpoint` |
| u0018 (code) | 101.6 s | `visual_elements[0..5] has no 'kind'`; `numbers is not a list` |

- Schema validity: **0 of 3 valid.** All three were schema-invalid in shape, not merely
  wrong in content. `structured_output=json_schema` was requested and OpenRouter accepted
  the request; the model emitted objects missing required keys.
- Evidence created: **none.** No observation entered validated evidence. `test_observation_valid`
  never ran.
- Confidence: not reportable — no valid observation to carry one.
- Abstentions: none reported, because nothing parsed far enough to record one.
- HTTP 200 was not treated as success: `reachable ok` and `model_exists ok` are reported
  alongside the parse failure rather than collapsing to a pass.

**VLM status: FAILED.** All real responses failed schema validation. Per the brief the
schema was not loosened.

### 5. Synthesis qualification

`yt-report synthesis-doctor --synth-provider openrouter --synth-model stealth/space-bunny-alpha`
— exactly one real call, no retries.

    reachable            ok    https://openrouter.ai/api/v1 answered
    authenticated        ok    the credential was accepted
    account_paid         ok    the call was billed
    model_exists         ok    stealth/space-bunny-alpha resolved
    model_completes      FAIL  response hit the token limit before any content
    structured_requested FAIL  unreachable
    structured_enforced  FAIL  unreachable
    draft_valid          FAIL  unreachable
    draft_grounded       FAIL  unreachable
    report_validated     FAIL  unreachable
    latency              147823 ms

- schema compliance: **not reached.** The expected sequence
  provider -> structured synthesis -> grounding -> conflict -> report validation stopped
  at the first content gate.
- grounding / conflict / one-home / report validation: all `unreachable`.
- latency: 147.8 s wall for a call that produced no content.
- actual output: none. The response exhausted `max_tokens` before emitting any content.

The classification is accurate — the model was reachable, authenticated, billed and
resolvable, so this is a content failure, not a connectivity failure, and release-check
said so rather than reporting a transport error.

**Synthesis status: FAILED** (one call, gate 2 of 5, `model_completes`).

### 6. Critic qualification

`yt-report critic-doctor --synth-provider openrouter --synth-model stealth/space-bunny-alpha`
— one real run, all 4 cases, 4/4 calls spent.

    model_completes       ok    the model returned a critique for 4/4 case(s)
    structured_enforced   ok    every critique response carried every required key
    critic_reachable      ok    the model answered 4/4 case(s)
    cases_matched         FAIL  0/4 case(s) matched the expected answer

| Case | Expected | Observed | Detail |
|---|---|---|---|
| clean | none | malformed | `findings[0] is missing 'code'; findings[0] is missing 'where'` |
| changed_number | findings | malformed | same |
| removed_qualifier | findings | malformed | same |
| unsupported_sentence | findings | malformed | same |

- clean draft result: FAIL. changed-number detection: not reached. removed-hedge detection:
  not reached. unsupported-sentence detection: not reached.
- false positives: **not measurable.** All four responses were discarded as malformed
  before the clean/planted comparison ran, so the run cannot say whether the model is
  trigger-happy or trigger-shy. It only says it never produced a parseable critique.
- latency: not reported by the doctor; the four calls fit inside the 300 s command timeout.

**Critic status: FAILED.**

**Defect recorded (not fixed): `structured_enforced` and the case gate contradict each other
in the same report.** `structured_enforced ok — "every critique response carried every required
key"` is printed three lines above four FAILs saying `findings[0] is missing 'code'` and
`missing 'where'`. Both gates describe the same artifact. A reader given only the `ok` line
would conclude the critic is schema-clean.

## 7. Complete-video result

The brief gates step 6 on the provider checks succeeding. They did not (VLM FAILED,
synthesis FAILED, critic FAILED), so **no model-backed end-to-end run was performed** —
a model run could only have produced the deterministic report plus a recorded failure.

To still produce real end-to-end evidence for every non-model capability, exactly one
complete real video was processed through the deterministic (no-model) path.

**Video A (English):** "But what is a neural network? | Deep learning chapter 1",
3Blue1Brown, `aircAruvnKk`, 18:40, 37 frames.
`uv run scripts/build_evidence.py <artifact-dir> --detail balanced` -> exit 0.

**Video B (non-English):** "Das ist der furchtbarste Ort im Universum",
Dinge Erklärt - Kurzgesagt, `5_jcvQ0-vhI`, 9:17, 31 frames. Same command -> exit 0.

Both directories were copied to `/tmp/eval_D` and `/tmp/eval_G` first, so the originals
under `~/Desktop/yt-reports/` were not touched.

Both runs reported:

    VALID  evidence.json
    VALID  REPORT.md  [deterministic]
    494 words - balanced - provenance 100% - 35/55 evidence items used - repetition 5.0% - 6 sections   (A)
    104 words - balanced - provenance 100% - 17/36 evidence items used - repetition 0.0% - 3 sections   (B)

### Numbers actually measured

| Metric | Video A (en) | Video B (de) |
|---|---|---|
| transcript source | `en (human)`, asr=false | `de (human)`, asr=false |
| transcript snippets | 286 | 77 |
| transcript segments | 12 named chapters | 5 ("Part 1".."Part 5") |
| frames read by OCR | 37 / 37 | 28 / 31 |
| frames with text | 33 | 9 |
| mean OCR confidence | 0.8932 | 0.9222 |
| alignment: snippets processed | 286 | 77 |
| alignment: no-relevant-frame | 187 (65%) | 35 (45%) |
| alignment: distinct frames used | 36 / 37 | 30 / 31 |
| evidence items | 55 | 36 |
| item mix | topic 1, organization 2, link 18, claim 4, number 11, definition 14, entity 5 | topic 1, organization 1, link 27, claim 1, number 3, entity 3, **definition 0** |
| items from metadata | 22 | 31 |
| items from transcript | 23 | **5** |
| items from OCR | 10 | 0 |
| conflicts detected | 0 | 0 |
| items with two homes | 0 | 0 |
| unsupported claims / numbers / urls | 0 / 0 / 0 | 0 / 0 / 0 |
| fabricated timestamps | 0 | 0 |
| dropped hedges | 0 | 0 |
| compression ratio | 0.147 | not reported |
| timeline coverage | 0.5833 | n/a (no timeline rendered) |
| report sections | 6 | **3** |

## 8. Actual report excerpt (Video A, `REPORT.md`, read in full)

    # But what is a neural network? | Deep learning chapter 1

    *3Blue1Brown - 20171005 - 18:40*

    > **DEGRADED.** transcript + on-screen text; no model reading of the frames, so
    > diagrams, tables and charts are not interpreted.

    ## Executive Summary

    - But what is a neural network? | Deep learning chapter 1. [metadata]
    - Structure: Introduction example; Series preview; What are neurons?; Introducing
      layers; Why layers?; Edge detection example; Counting weights and biases; How
      learning relates.
    - Named: 3Blue1Brown; YouTube.

    ## Key Takeaways

    - Then the sum is largest when those middle pixels are bright but the surrounding
      pixels are darker. [10:07 - narration]
    - Now is the point I suppose I should say subscribe to stay notified about when that
      video or any new videos come out, but realistically most of you don't actually
      receive notifications from YouTube, do you? [16:27 - narration]
    - What is the best hope for what those middle layers might be doing? [05:45 - narration]
    - The goal is to have some mechanism that could conceivably combine pixels into edges,
      or edges into patterns, or patterns into digits. [08:30 - narration]
    - And that is just one neuron. [11:36 - narration]

    ## Detailed Notes

    ### Edge detection example

    The question at hand is what parameters should the network have? [08:51 - narration]
    These weights are just numbers. [09:16 - narration] Then the sum is largest when those
    middle pixels are bright but the surrounding pixels are darker. [10:07 - narration]
    That additional number is called the bias. [11:20 - narration]

    ### Recap

    Really the entire network is just a function, one that takes in 784 numbers as an input
    and spits out 10 numbers as an output. [15:39 - narration]

    ## Source & Reliability

    - Evidence by source - metadata: 22, on-screen text: 14, narration: 23.
    - **Figures from a single read.** 10 of 11 figures rest on one text read of one frame
      with nothing else naming them. They are reported as the screen was read; the labels
      are the engine's and may be garbled.

## 9. Inspection of the actual report

Judged against `transcript.md`, `evidence.json`, `visual.json` and `alignment.json` — not
against validator output.

**SUMMARY — weak.** The Executive Summary contains no prose. It echoes the title, lists
chapter names, and lists entities. It does not state the video's argument. The video's
actual argument is legible in the transcript: neurons fire on an activation value, layers
stack those neurons, and stacking is what lets pixels become edges, edges become patterns
and patterns become digits. None of that is in the summary. A reader who stops after the
Executive Summary learns that the video is by 3Blue1Brown and mentions YouTube.

The `Structure:` bullet is also the only Executive Summary bullet with no provenance
marker, while its two siblings carry `[metadata]`.

**TAKEAWAYS — mostly extracted sentences, not insights.** Four of the five are verbatim
narration lines with a timestamp. The single exception is a real claim ("pixels into edges,
or edges into patterns, or patterns into digits") and it is ranked fourth, below a
rhetorical question.

Ranking defect: bullet 2 is the Patreon/YouTube subscribe appeal — "Now is the point I
suppose I should say subscribe to stay notified...". It is promotional filler and it is
the second most prominent line in the report.

**DETAILED NOTES — coherent per section, thin in aggregate.** The Edge detection section
is the strongest passage in the report: four sentences that chain into a real explanation
(parameters -> weights -> what the weights do -> the bias). But sections are thin, and two
of them consist only of rhetorical questions: "What are we expecting here? [05:44] What is
the best hope for what those middle layers might be doing? [05:45]". Those are the narrator
asking, not the report answering.

**NUMBERS — the weakest area.** See section 12.

**ENTITIES — mixed.** `3Blue1Brown` correctly typed as an organization. But the "Other
Named Things" list is `Patreon`, `PhD`, `ReLU`, `Lisha Li`, `Amplify Partners`, all with
the hedge "type not established". `PhD` is not a named entity at all — it is the narrator
mentioning his own credential. `ReLU` is a technical term, and it is the one term in the
video a reader would most want defined; it is filed as an untyped "thing". `Lisha Li` and
`Amplify Partners` are real but are Patreon supporters credited in the outro, not content.

One transcription error survived into evidence: `Lisha Li` (the real name is Lisha Li /
Grant Sanderson's Patreon credits aside, the transcript itself reads "Lisha Li" — the
extractor copied it faithfully, which is correct behaviour).

**VISUAL — honestly absent, and the report says so.** The banner states plainly that no
model read the frames. `visual_statements: 0` and `vlm_status: unavailable` agree. The
report does not fake visual content. That is the correct failure.

But the Reliability section contradicts itself in effect: it tells the reader "10 of 11
figures rest on one text read of one frame", and then the report contains no figures at
all. There is no "Numbers, Pricing & Benchmarks" section in Video A's report, though one
exists in Video B's. The reader is told about 11 figures and shown none.

**UNCERTAINTY — preserved.** Every entity hedge ("type not established") is explicit.
`dropped_hedges: 0`, `hedged_blocks_checked: 0` (no hedged block was offered for checking,
so the metric is untested rather than passed). The "Figures from a single read" warning is
a genuine, specific uncertainty statement.

**CONFLICTS — nothing to disclose.** `conflicts_in_evidence: 0` and `conflicts: 0` in both
videos. No conflict-handling behaviour was exercised.

**REPETITION — low but non-zero.** 22 duplicate 8-grams over 438 n-grams, ratio 5.0%. The
"Key Takeaways" bullets are verbatim duplicates of the "Detailed Notes" sentences, which
is the main source: "Then the sum is largest when those middle pixels are bright..."
appears in both sections, word for word. The same claim is therefore stated twice per
report by construction.

**USEFULNESS — partial.** A person who did not watch would learn: this is chapter 1 of a
neural-network series by 3Blue1Brown; neurons have activations; layers are stacked; the
784-in / 10-out shape; edge detection is the worked example. They would not learn what an
activation is numerically, what ReLU is, how many parameters the network has, or what
"learning" means — the video spends 18 minutes on structure and defers learning entirely,
and the report's "How learning relates" segment produced no content at all.

## 10. VLM quality observations

No observation survived schema validation, so there is nothing to classify as
CORRECT / PARTIALLY CORRECT / INCORRECT / UNRESOLVED. The honest classification is
**UNRESOLVED for all three frames** — not INCORRECT, because no model claim was ever
extracted into a form a human could check.

What the failures show is shape failure, not content failure:

- Two frames returned `visual_elements` as a non-list, and elements with no `kind`. The
  model produced *something* where the array belongs. The validator named the exact
  structural defect, which is a good diagnostic.
- The diagram frame additionally returned `relationships` with no `source` on three
  entries and one relationship missing its `to` endpoint. A relationship graph with
  missing endpoints is the exact shape that would corrupt a provenance table if it were
  accepted — and it was not.
- The code frame returned `numbers` as a non-list, on a frame dense with literal numbers
  (`rgb(255 255 255)`, `1.2px`, `100%`, `2 files changed +28 -3`). The model was reaching
  for numbers and got the container wrong.

No output was silently repaired. `test_observation_valid` never ran; the run stopped at
`vision_capable` and reported the parse error verbatim.

**Nothing entered validated evidence, so VLM is not QUALIFIED and, per the brief's own
rule for the all-responses-invalid case, it is FAILED.**

## 11. Synthesis quality observations

**Not assessable.** The comparison the brief asks for — model prose against the
deterministic fallback — could not be made, because the model produced no prose at all.
There is no model output to compare, no evidence dump to diagnose, no verbosity to
measure, and no conflict handling to inspect. Recording this as UNTESTED rather than
DEGRADED would be wrong: it was tested and it failed at the content gate.

The only synthesis-side signal available is that the failure is silent from the caller's
point of view: 147.8 s, HTTP 200, billed, and zero usable content. Any caller that
checked only for a 200 or for non-empty response bytes would have recorded a success.

The deterministic fallback for comparison purposes is section 8/9 above. It is the only
report this run produced.

## 12. Numbers quality

The most concrete finding in this evaluation.

Video A's eleven `number` evidence items, with their actual values:

| value | provenance | OCR text as read | confidence |
|---|---|---|---|
| 1120 | metadata | (duration in seconds) | 1.00 |
| 2 | ocr u0021 | `2` | 0.78 (read confidence 0.63) |
| 2 | ocr u0021 | `—2` | 0.78 (0.63) |
| 3 | ocr u0021 | `—3` | 0.78 (0.63) |
| 2 | ocr u0021 | `—2` | 0.78 (0.63) |
| 4 | ocr u0021 | `4` | 0.78 (0.63) |
| 1 | ocr u0028/u0035 | `(1)` | 0.78 (0.87) |
| 0 | ocr u0028/u0035 | `(0)` | 0.78 (0.87) |
| 0 | ocr u0030 | `(0)` | 0.78 (0.77) |
| 0 | ocr u0028/u0035 | `WK, 0 Wk,1` | 0.78 (0.87) |
| 0 | ocr u0035 | `W1,0 W1,1` | 0.78 (0.85) |

These are single digits read off matrix-index subscripts and axis labels in a maths
animation. They are exactly the UI noise the brief asks about, and they are not UI noise
that was suppressed — they are what survived extraction. Meanwhile the figures the video
is actually about were not extracted at all:

- `28 x 28 = 784` is legibly written on frame u0007, which I read directly.
- `784` and `10` (input/output node counts) are labelled on frame u0011, which I read
  directly.

Neither entered evidence as a number. `784` and `10` appear in the report only because
they occur inside one narration sentence in the Recap. So: the meaningful figures are
missing, the noise digits are present, and the report shows neither as a number.

Two separate metric problems compound this:

1. `report_validation.json` reports `numbers_in_report: 44, numbers_supported: 44`.
   Forty-four "numbers" in a 494-word report. Twenty-four of them are the `[HH:MM]`
   timestamp markers. The metric is counting timestamps as figures, so "44/44 supported"
   is close to vacuous as a measure of number quality.
2. `evidence_items_used: 35 / 55` and `evidence_utilisation: 0.6364` — twenty items were
   dropped, including eleven of the eighteen links and all eleven number items. The
   numbers were extracted and then not used, which is why the reader is told about 11
   figures and shown none.

Video B, the German video, does render a "Numbers, Pricing & Benchmarks" section with
`60%` and `40%` — both real and both useful, and both correctly attributed to narration
timestamps rather than to a frame. So the figures path works when captions carry the
numbers. It fails when the numbers live on screen.

## 13. Critic quality observations

**The critic could not be assessed, and the reason matters more than the result.**

All four planted cases — including `clean`, where the correct behaviour is to return
nothing — produced the same malformed output, and each `findings[0]` was missing both
`code` and `where`. So:

- catches real grounding errors: not measurable
- notices numerical errors: not measurable
- notices unsupported statements: not measurable
- identifies missing qualifiers: not measurable
- avoids false positives: not measurable
- preserves valid prose: not measurable

The single most useful thing this run established is negative: the critic returned
findings for the `clean` draft. Whether that is a false positive or an artefact of a
malformed response is not decidable from the output, and the doctor's own summary
correctly refuses to say — it reports `observed malformed`, not `observed findings`.

Structured JSON is not evidence of a working critic, and this run does not claim one.

## 14. Multilingual observations (German, `5_jcvQ0-vhI`)

Extraction and reporting both work in German, and the run degrades structurally rather
than breaking.

Works:

- Human German captions retrieved, `de (human)`, 77 snippets, 5 segments.
- OCR ran on 31 frames, 9 with text, mean confidence 0.9222.
- Entities captured: `Incogni`, `Void`, `Super Voen`. Figures captured: `60%`, `40%`,
  correctly attributed to narration timestamps `[02:44]`, `[03:20]`.
- All 10 rendered URLs are real and all resolve to the description's own link list.
- `VALID evidence.json`, `VALID REPORT.md`, 0 unsupported claims/numbers/urls, 0
  fabricated timestamps, provenance coverage 1.0.

Degrades:

- The report has **3 sections instead of 6**. Key Takeaways, Detailed Notes and Timeline
  are absent entirely — the report is 104 words for a 9-minute video.
- The cause is visible in the item mix: `definition: 0`, and only **5** evidence items
  come from the transcript against 31 from metadata. Video A got 23 transcript items from
  286 snippets; Video B got 5 from 77. Per word that is roughly 4x worse extraction.
- The German captions arrive as chapter-level blobs (7.6 s snippets containing several
  sentences, with spacing artefacts: "Weiz", "wunderschi nen", "Durc") and the
  transcript's own segments are literally titled "Part 1".."Part 5". There is no
  sentence-level structure for the extractor to key on, and uploader chapter titles carry
  no information, so the Executive Summary's structure line is
  `Structure: Part 1; Part 2; Part 3; Part 4; Part 5`.
- Entity typing is weaker: `Super Voen` is a mangled `Supervoid`/`Super-Void` and
  `Void` is a common noun promoted to a named entity. Both filed as
  "type not established".

**Multilingual extraction: DEGRADED** — German produces a valid, honest, link-rich but
nearly contentless report where English produces a thin but readable one.

## 15. Performance and latency

| Stage | Wall time |
|---|---|
| release-check (full, incl. VLM + synthesis + 1 critic case) | 145.9 s |
| synthesis-doctor (1 call, no content) | 147.8 s |
| vlm-doctor, chart frame | 53.1 s |
| vlm-doctor, diagram frame | 80.9 s |
| vlm-doctor, code frame | 101.6 s |
| build_evidence, Video A, fully warm cache | ~0 s |
| build_evidence, Video B, 0/31 frames cached (OCR ran) | 2.8 s of OCR for 28 frames |
| build_evidence, Video A, cold OCR (prior run, from `acceptance/metrics.json`) | `ocr_wall_s` 0.012 (cached) / OCR backend `apple-vision` |

The pattern that matters: **model latency is 50-150 s per call and returns nothing usable;
local OCR is ~100 ms per frame and returns usable text.** The expensive path in this
system is the one that did not work, and the cheap one is the one that carried both
reports. At the brief's budget of 2 frames + 1 synthesis + 1 critic + 1 revision, a fully
working model run would cost roughly 3-5 minutes of wall time and a corresponding amount
of per-video latency for a reader.

## 16. Cache behaviour

- Cache is per-frame content hash. Video A: 37/37 frames served from cache
  (`cache_hit_rate 1.00`), and the whole rebuild completed in under a second.
- Video B: 0/31 cached, OCR executed live, 2.842 s for 28 frames — about 100 ms/frame.
- `visual.json` reports `ocr_wall_s` and per-frame `cached: true`, so cache state is
  observable in the artifact rather than only in logs. This is good.
- The VLM path reported `cache_hits: 0, cache_misses: 0` on every run, because no valid
  observation was ever produced to cache. A cache cannot be faulted for misses on results
  that do not exist.
- No staleness or corruption was observed: warm-cache rebuilds produced byte-identical
  validation results.

## 17. Concrete failures

Recorded, not fixed. None of these were worked around in code.

1. **VLM schema violation, 3/3 frames.** `visual_elements` returned as a non-list or with
   entries lacking `kind`; `relationships` entries missing `source` and `to`; `numbers`
   returned as a non-list. Zero valid observations, zero evidence created.
2. **Synthesis returns no content.** 147.8 s, HTTP 200, billed, `max_tokens` exhausted
   before any content. Gate 2 of 5.
3. **Critic returns malformed output, 4/4 cases.** Every `findings[0]` missing `code` and
   `where`. Zero cases matched.
4. **Validator self-contradiction.** `critic-doctor` prints
   `structured_enforced ok - "every critique response carried every required key"` in the
   same report as four FAILs naming the missing keys. Two gates, same artifact, opposite
   verdicts.
5. **Meaningful on-screen figures are never extracted.** `28 x 28 = 784` (u0007) and the
   `784` / `10` node counts (u0011) are legibly present and absent from evidence. What
   was extracted instead is ten single-digit OCR reads of matrix subscripts.
6. **A figures section is announced but not rendered.** Video A's report states
   "10 of 11 figures rest on one text read of one frame" and contains no figures section.
7. **`numbers_in_report` counts timestamps.** 44 of the 44 "numbers" in Video A are
   supported; 24 of them are `[HH:MM]` markers, so the metric overstates number quality.
8. **Subscribe boilerplate ranked as a Key Takeaway.** Video A's second bullet is the
   Patreon/YouTube notification appeal, verbatim.
9. **Executive Summary is not a summary.** Title echo + chapter list + entity list, with
   no statement of the video's argument, and the chapter-list bullet carries no
   provenance marker while its siblings do.
10. **Non-English reports lose half their structure.** German: 3 sections vs 6, 104 words,
    0 definitions, 5 transcript-derived items vs 31 metadata items. Chapter-level caption
    blobs give the extractor nothing to key on.
11. **Entity typing errors.** `PhD` and `ReLU` filed as untyped "Other Named Things";
    `Void` (common noun) filed as a named entity; `Super Voen` is a mangled transcript
    token carried through unflagged.
12. **The upload date is typed as a claim.** `20171005` appears in evidence as
    `type: claim`.
13. **Takeaways and Detailed Notes duplicate verbatim.** 22 duplicate 8-grams (5.0%),
    caused by the same narration sentence appearing in both sections by construction.
14. **Alignment discards most snippets.** 187 of 286 English snippets (65%) got
    `no-relevant-frame`; 35 of 77 German (45%). Whether that is correct restraint or
    lost content is not answerable from the artifacts.
15. **Timeline covers 58% of Video A.** Eight chapters of twelve, and the gap is not
    explained in the report.

## 18. Final capability matrix

Status vocabulary: QUALIFIED / DEGRADED / UNTESTED / FAILED.

| Capability | Status | Evidence |
|---|---|---|
| Metadata ingestion | QUALIFIED | Both videos: title, channel, upload_date, duration, description, tags, chapters, view/like counts, webpage_url all populated. `5_jcvQ0-vhI` resolved `transcript_lang de (human)`, `asr false`, `transcript_error null`. |
| Transcript | QUALIFIED | `en (human)` 286 snippets / 12 segments; `de (human)` 77 snippets / 5 segments. `transcript_loss_s: 0` both runs. No errors. |
| ASR | UNTESTED | Both videos had human captions and set `asr: false`. The ASR path was never invoked. No evidence either way. |
| OCR | QUALIFIED | `apple-vision` backend, Video A 37/37 frames read, 33 with text, mean confidence 0.8932; Video B 28/31, 9 with text, mean 0.9222. ~100 ms/frame cold. `ocr_status` reported per run. Weakness is downstream selection, not recognition (failure 5). |
| Alignment | DEGRADED | Works: 99/286 EN and 42/77 DE snippets aligned, 100 and 53 frame links, 36/37 and 30/31 distinct frames used, methods split across `overlap` and `segment-context`. But 65% (EN) and 45% (DE) of snippets get `no-relevant-frame`, and the report gives the reader no account of what was dropped. |
| Semantic extraction | DEGRADED | EN: 23 transcript-derived items, 14 `definition`, 4 `claim`, 11 `number`. DE: 5 transcript-derived items, **0** `definition`, 1 `claim`. Definition values are fragmentary ("just numbers", "we expecting here?", "just one neuron", "called the bias") rather than definitions. Ranking promotes boilerplate (failure 8). |
| VLM | FAILED | 3/3 real frames schema-invalid. `visual_elements` non-list or missing `kind`; `relationships` missing `source`/`to`; `numbers` non-list. 0 valid observations, 0 evidence created, 0 abstentions recordable. Schema not loosened. |
| Synthesis | FAILED | 1 real call, 147.8 s, HTTP 200, billed, `max_tokens` exhausted before any content. Schema/grounding/conflict/one-home/report-validation all `unreachable`. 0 model prose exists, so no deterministic-vs-model comparison was possible. |
| Critic | FAILED | 4/4 cases malformed, `findings[0]` missing `code` and `where`, 0/4 matched expected. No grounding error, missing qualifier, numerical error or false-positive behaviour could be observed. |
| Report validation | DEGRADED | Gate works and is strict where it counts: `VALID evidence.json`, `VALID REPORT.md`, 0 unsupported claims / numbers / urls / visual statements, 0 fabricated timestamps, 0 dropped hedges, 0 items with two homes, provenance coverage 1.0, 0 dangling refs, 0 ungrounded items. But it passes a report that announces 11 figures and shows none (failure 6), and `numbers_in_report` counts timestamps (failure 7). Also a self-contradiction inside `critic-doctor` (failure 4). |
| Multilingual extraction | DEGRADED | German: valid evidence, valid report, correct `60%`/`40%` figures, 10 real URLs, provenance 1.0. But 3 sections instead of 6, 104 words, 0 definitions, 5 transcript items, chapter-blob captions, `Super Voen` / `Void` entity errors. |
| End-to-end video processing | DEGRADED | Two complete real videos, exit 0, `VALID` on evidence and report, deterministic mode, honest `DEGRADED` banner, no fabricated visual content. Never exercised with a working model, so the model-backed path is unproven. Deterministic path alone is not the product's headline capability. |

## 19. Final human-quality review

**What worked.** The honesty layer is the strongest thing here. Both reports declare
`DEGRADED` and say why; neither invents visual content; `visual_statements: 0` and
`vlm_status: unavailable` match the banner; the Reliability section volunteers a
specific, checkable weakness ("10 of 11 figures rest on one text read of one frame") that
most products would suppress. The validators are genuinely strict — zero unsupported
claims, zero unsupported numbers, zero unsupported URLs, zero fabricated timestamps, zero
dropped hedges, zero double-homed items, zero dangling references across two real videos.
`release-check` did not treat HTTP 200 as success anywhere: it separated reachability,
authentication, payment, model existence and content validity into independent checks, and
classified the 402 by cause ("the account cannot pay... waiting will not help") rather than
by status code. When the model emitted a malformed observation the tool reported the exact
structural defect rather than rounding it to "error". The local OCR path is fast and
accurate, and the cache is correct and observable in the artifacts.

**What was weak.** The reports are thin. Video A is 494 words for 18:40 and omits the
video's central argument from its own summary. Video B is 104 words for 9:17 and has half
the sections. Numeric extraction is the weakest link: the figures a reader wants are not
extracted, the digits that are extracted are animation subscripts, and the coverage metric
that would have caught this counts timestamps. Entity typing puts `PhD` and `ReLU` in the
same untyped bucket. The alignment stage discards 65% of English snippets with no account
in the report.

**What was misleading.** The most misleading artifact is Video A's Reliability section,
which warns the reader about 11 figures in a report that contains no figures — a caveat
that reads as rigour while the underlying content is absent. `critic-doctor`'s
`structured_enforced ok` line is the second: it asserts full schema compliance three lines
above four failures naming the exact missing keys, and a reader scanning the summary would
believe the critic's output shape is sound. `numbers_in_report: 44 / numbers_supported: 44`
is the third — it reads as a clean bill of health for numeric fidelity. And
`total_credits 46 / total_usage 46.197` predicts total unavailability while the billing
path simultaneously reports calls as billed, which is a contradiction an operator cannot
resolve from the tool's own output.

**What was missing.** No model-backed run of any kind: VLM, synthesis and critic all
failed before producing usable output, so the model's contribution to report quality is
entirely unmeasured. No ASR evidence — both videos had human captions. No conflict
handling — zero conflicts existed in either video. No multilingual coverage beyond German,
and none in a non-Latin script. No abstention evidence from the VLM, since no observation
parsed far enough to abstain. No comparison between model synthesis and the deterministic
fallback, because there was no model prose. `hedged_blocks_checked: 0` in both runs, so
hedge preservation is untested rather than passed.

**What was genuinely useful.** A person who did not watch the 3Blue1Brown video would come
away with a correct skeleton: it is chapter 1 of a series, it is about structure not
learning, neurons carry activations, layers are stacked, the worked example is edge
detection, the network maps 784 inputs to 10 outputs, and the chapter list tells them what
the other eleven chapters cover. Every claim in the report carries a timestamp or a source
tag. The German report's 10 URLs and two percentages are correct and correctly attributed.
The `DEGRADED` banner alone would let a reader decide how much to trust the rest, which
is more than most generated reports offer.

## 20. Recommendation for the next step

**Not production-ready.** Not on this evidence, and the blocker is not the missing credits.

The deterministic path is in defensible shape: it is fast, cached, honest about its own
degradation, and its validators are strict enough that two real videos produced zero
unsupported claims and zero fabricated timestamps. But three of the four capabilities
this product is named for — VLM, synthesis, critic — have never produced a single valid
output against a real provider, and the fourth (numbers) does not work on on-screen
figures, which is the case that matters most for the visual videos the skill is aimed at.

In priority order, before any further spend:

1. **Qualify a provider that can hold a schema.** Not `stealth/space-bunny-alpha` via
   OpenRouter structured output — 3/3 VLM shape failures, 1/1 synthesis content failure,
   4/4 critic shape failures. Add credit to OpenRouter and re-run `openai/gpt-4o-mini`,
   or test the `anthropic` provider already wired in `scripts/chat_json.py`. Until one
   model returns one schema-valid observation and one non-empty draft, the entire
   model-backed half of the product is unmeasured and no amount of local testing
   substitutes.
2. **Fix the on-screen figures path** (failure 5). `28 x 28 = 784` was legible in a frame
   and never became evidence. This is a gap in what is *selected* for extraction, not in
   OCR accuracy, and it degrades every visual video regardless of which model is wired in.
3. **Reconcile the two gates that disagree** (failure 4). `structured_enforced` must not
   report `ok` on an artifact a sibling gate rejects. Until then `critic-doctor` cannot be
   trusted to tell a reader whether the critic is well-formed.
4. **Stop counting timestamps as figures** (failure 7), and render the figures section
   the report already promises (failure 6).
5. **Exercise ASR and conflicts.** Both are UNTESTED with zero evidence. A video with
   human captions disabled and a video with two conflicting on-screen values would close
   the two largest untested gaps in the matrix.

Re-run this evaluation after step 1 succeeds. Until a model produces valid output at least
once per role, the capability matrix above is the ceiling of what is known about this
product, and the honest summary is: a careful, fast, honest transcription-and-linking
report generator that cannot yet read a chart, write a sentence, or check its own work.
