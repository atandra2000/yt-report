# Acceptance report — yt-report

**Date:** 2026-09-26 · **Scope:** final acceptance, reliability and real-world
quality pass over a completed architecture · **Baseline claimed:** 1074 checks
passing, real OpenRouter VLM qualified on `openai/gpt-4o-mini`

---

## 1. Overall status: **CONDITIONAL**

Not "production-ready", and not because the tests are red — they are green, and
grew. The condition is specific and it is not a code defect:

> **The report's prose has never been written by a working model.** The
> synthesis writer and the critic are the two layers whose entire purpose is
> fluent, grounded prose and a second opinion on it, and neither could be
> exercised end to end on any provider available in this environment. Every
> report in this pass was produced by the deterministic renderer.

The deterministic path is in good shape: four materially different real videos
plus a non-English fifth, every artifact present, every gate holding, zero
unsupported claims, numbers or URLs, and a validator that caught 38 of 38
deliberate corruptions with no silent repair. What is unproven is the part a
reader would notice most — whether a model-written report is *good*, which is
the only thing the critic exists to protect.

Three further conditions, each measured rather than assumed, are listed in §15.

---

## 2. Architecture as measured

Five stages, each a gate that blocks the next.

| stage | script | produces | blocks on |
|---|---|---|---|
| fetch | `yt_fetch.py` | `meta.json`, `transcript.{json,md}`, `frames/`, `frames.json`, `FRAMES.md`, `visual.json` | captions + ASR, exit 2/3 |
| visual | `ocr.py` (macOS Vision) | OCR read per frame, in `visual.json` | records `unavailable` rather than passing silently |
| interpret | `vlm.py` | `vlm` block per selected frame | non-conforming output is refused, not repaired |
| extract | `align.py` → `semantic.py` → `build_evidence.py` | `alignment.json`, `evidence.json` | `validation.json` must be clean |
| report | `synthesis.py` → `report.py` → `report_validate.py` | `REPORT.md`, `REPORT.json`, `report_validation.json` | ungrounded report is never written |

The report is a renderer over validated evidence with no free-text slot, so it
cannot introduce a fact the evidence does not carry — the property this pass
tested hardest, by trying to make it fabricate.

---

## 3. Four real videos (plus a non-English fifth)

All fetched live from YouTube; no pre-generated evidence substituted.

| | video | duration | category | outcome |
|---|---|---|---|---|
| **A** | *This NEW Jev + Claude OS Just Changed Every AI Workflow* (Chase AI) | 22:16 | talking-head / commentary | VALID |
| **B** | *Google's Project Suncatcher is sending AI to space* | 17:47 | AI / tech news | VALID, **4 unresolved conflicts** |
| **C** | *AI agents explained — how to create and use AI agents in 2026* | 24:28 | screen-heavy technical explainer | VALID |
| **D** | *But what is a neural network? Deep learning ch. 1* (3Blue1Brown) | 18:40 | visually rich lecture, diagrams | VALID |
| **E** | *Das ist der furchtbarste Ort im Universum* (Kurzgesagt) | 9:17 | non-English, visually rich | VALID |

---

## 4. VLM qualification

| provider | model | verdict |
|---|---|---|
| `openrouter` | `openai/gpt-4o-mini` | **QUALIFIED** with a real frame — all seven doctor steps ok, `json_schema` accepted *and enforced* |
| `openrouter` | `openai/gpt-4o-mini` (no `--frame`) | **NOT QUALIFIED — doctor defect, see §13.1** |
| `openrouter` | `stealth/space-bunny-alpha` | reachable ✓, authenticated ✓, model exists ✓, vision capable ✓ — **`json_schema` not honoured on any of the three modes**; 0 conformant observations |

**A real VLM run happened and produced no evidence, honestly recorded.** On
video A: `vlm_calls: 2`, `calls_ok: 0`, `calls_failed: 2`, `cache_hits: 0`,
`vlm_wall_s: 243`. Both failures are recorded per frame in `visual.json` with
their reason, `vlm_items` in the report is **0**, and the report banner says the
frames were not interpreted. The pipeline did not invent a visual observation,
and it did not claim one.

`stealth/space-bunny-alpha` produces good visual content under `prompt_only` —
`content_type: application_ui`, seven `visual_elements`, three relationships,
three entities, a summary naming the on-screen product, and an `uncertainties`
entry disagreeing with the OCR read. It fails the contract in three consistent
ways: `content_type` outside the vocabulary, `visual_elements` items using
`type`/`text` where the schema requires `kind`/`label`/`detail`, and
`relationships` using `from` where the parser wants `source`. Those are model
behaviour, not a defect to paper over, and the pass **did not widen the schema
to accommodate them** — that is the specific failure mode this repository
exists to prevent.

Three real defects were found *by* using that model and are fixed (§13.2).

---

## 5. Real critic — **NOT EXERCISED**

| attempt | outcome |
|---|---|
| `openai/gpt-4o-mini` synthesis | HTTP 402 — OpenRouter account exhausted (`total_credits 46`, `total_usage 46.19`) |
| `stealth/space-bunny-alpha` synthesis | `response hit the token limit before any content` at 16 000 max_tokens — the model spends the budget on reasoning and emits no content |
| `stealth/space-bunny-alpha` VLM | exhausted the token budget mid-JSON (§4) |

The critic path is covered by `selftest_synthesis.py` against a scripted
transport, including the one-revision ceiling, but **a scripted critic is not a
real critic** and this item is marked UNMEASURED, not passed. §5's grounding
injection was therefore run against the validator alone — which is the property
that matters, and is stated as such in §7.

---

## 6. Non-English

Real German video (E), human German captions, no ASR needed.

| property | result |
|---|---|
| transcript preserved in source language | ✓ `transcript_lang: "de (human)"`, German throughout |
| evidence store untranslated | ✓ names, URLs, figures unchanged |
| report valid | ✓ 0 unsupported claims/numbers/URLs, provenance coverage 1.0 |
| `--output-language en` | **byte-identical to `auto`** — the flag is a no-op on the deterministic path |
| `--output-language ja`, `es` | byte-identical to `auto` |

**`--output-language` does nothing without a configured writer.** It is a
property of the model-written draft; the deterministic renderer has a fixed
English template and nothing to translate. Measured: `auto`/`en`/`ja`/`es` all
produced the same bytes. SKILL.md claimed otherwise and has been corrected.

**Real quality defect on non-English input.** The entity extractor keys on
capitalisation and filters with an English stopword list. German capitalises
every noun, so ordinary words enter the report as named products:

> `Named: Dinge Erklärt – Kurzgesagt; Erde; Incogni; Schwerkraft; Energie; **Aber**; Universums; **Sie**; **Du**; Masse.`

`Aber` (but), `Sie` (you, formal), `Du` (you, informal) are function words
presented under **Products, Models & Tools**. The English videos extract clean
entities (`Claude`, `Obsidian`, `ReLU`, `Patreon`). Severity **high for
non-English input**; detected automatically (the report still validates — it is
faithful, just not useful). Fix required; not made here, because the two
defensible fixes are a multilingual stopword corpus (a standing data
maintenance cost) or a document-level salience model, and both are architecture
this pass was told not to add.

The same failure has a small English instance: `And I` is extracted as a named
entity on video D.

---

## 7. Degradation matrix — 34/34

Every mode run through the real CLI against a real artifact directory. Provider
failures go through a local endpoint returning the real failure, so the HTTP
path, the classifier and the fallback ladder all execute.

| mode | exit | artifact state | report wording | false visual claim? |
|---|---|---|---|---|
| **A** full | 0 | complete | no banner | — |
| **B** VLM unavailable | 0 | every frame `unavailable`/`skipped` | DEGRADED, names the missing layer | none |
| **C** OCR unavailable | 0 | frames unmeasured | DEGRADED, names the missing layer | none |
| **D** visual unavailable | 0 | transcript only | DEGRADED, names the missing layer | none; no `· on-screen text]`, no `· model reading]` |
| **E** synthesis unavailable | 0 | deterministic | — | none |
| **F** no captions, ASR off | **2** | frames **still emitted**, `meta.json` records it, no `REPORT.md` | — | — |
| **G** rate limited (429) | 0 | no frame marked read | DEGRADED | none |
| **H** auth failure (401) | 0 | frames `unavailable` | DEGRADED | none |
| **I** malformed VLM response | 0 | prose rejected | DEGRADED | nothing invented |
| **J** malformed synthesis | 0 | deterministic renderer takes over, reason recorded | — | none |

Case F is the one that must not degrade, and it does not: the run exits 2, still
writes `meta.json`/`frames/`/contact sheets, and the note names the requested
language, the languages actually available, and both remedies.

**Measured defect.** With OCR unavailable, case C initially failed — because my
harness left `text` populated while clearing `result`. The corrected harness
passes. Worth recording because it is the same class as the real bug in §13.2:
the layer that must notice a changed input was keyed on part of it.

---

## 8. Grounding injection — 38/38

Seven corruptions applied to the **structured blocks** of real reports, each run
through `validate_report` with **no model in the loop**.

| corruption | applied | detected | silently repaired |
|---|---|---|---|
| 1 change one number | 9 | 9 | 0 |
| 2 add an unsupported factual sentence | 10 | 10 | 0 |
| 3 remove an important qualifier | 2 | 2 | 0 |
| 4 alter one URL | 5 | 5 | 0 |
| 5 reverse one comparison | **0** | — | **UNMEASURED** |
| 6 remove a conflict disclosure | 2 | 2 | 0 |
| 7 invent a visual relationship | 10 | 10 | 0 |
| write gate refuses and leaves the file | 1 | 1 | 0 |

**Case 5 is UNMEASURED, and would very likely fail.** No report in the set
states a comparison in a form the case can reverse. The reason is a real limit,
not a missing test: the renderer states a comparison as `**A vs B** — score: A
71%, B 79%`, and both figures are *legitimately* cited by the block, so swapping
them is not a misattribution by the evidence's own standard. Detecting a
reversal needs a claim about which side is better — a judgement the evidence
does not carry, and one §4.1 of the skill forbids making. Severity **medium**;
fix requires a comparison-direction invariant, which is new logic.

The property under test — **the validator is the final authority, the critic is
assistance** — holds on everything measured. The critic was not needed to catch
any of the 38, because it was not present.

---

## 9. Quality metrics

| metric | A | B | C | D | E (de) |
|---|---|---|---|---|---|
| duration | 22:16 | 17:47 | 24:28 | 18:40 | 9:17 |
| transcript words | 4 627 | 2 829 | 4 182 | 3 357 | 1 041 |
| report words | 791 | 835 | 743 | 464 | 114 |
| compression ratio | 0.171 | 0.295 | 0.178 | 0.138 | 0.110 |
| segments | 6 | 9 | 8 | 12 | 5 |
| evidence items | 143 | 109 | 180 | 53 | 62 |
| items used | 64 | 65 | 69 | 27 | 24 |
| provenance coverage | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **unsupported claims** | **0** | **0** | **0** | **0** | **0** |
| **unsupported numbers** | **0** | **0** | **0** | **0** | **0** |
| **unsupported URLs** | **0** | **0** | **0** | **0** | **0** |
| visual evidence items | 80 | 22 | 85 | 10 | 0 |
| VLM evidence items | 0 | 0 | 0 | 0 | 0 |
| OCR-only evidence items | 87 | 29 | 98 | 11 | 2 |
| cross-modal evidence items | 7 | 7 | 13 | 1 | 2 |
| **unresolved conflicts** | 0 | **4** | 0 | 0 | 0 |
| duplicate 8-grams | 9 | 26 | — | — | — |
| repetition ratio | 0.016 | 0.043 | 0.054 | 0.056 | 0.007 |
| frames / OCR-read | 51/51 | 71/49 | 96/57 | 37/33 | 31/9 |
| VLM calls / ok | 2/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| synthesis calls | 1 (402) | 0 | 0 | 0 | 0 |

Compression is 11–30% of transcript length, and the video with the most figures
produced (C) is not the one with the longest report — the budget follows the
evidence, as designed.

---

## 10. Performance

| stage | A | B | C | D | E |
|---|---|---|---|---|---|
| fetch (metadata + transcript + download + frames + OCR) | 89.0 s | 46.7 s | 84.6 s | 75.5 s | 51 s |
| build (align + extract + validate + report + validate) | 35.6 s | 6.8 s | 41.5 s | 9.3 s | <1 s |
| VLM pass (2 frames, `space-bunny-alpha`) | 243 s | — | — | — | — |
| synthesis (1 call, HTTP 402) | <1 s | — | — | — | — |

**The dominant cost is the download inside fetch, not any compute stage.** Frame
extraction, OCR over 37–96 frames, alignment, semantic extraction, rendering and
both gates together finish in under 42 s. A VLM call costs 60–120 s of pure
latency at this provider — two orders of magnitude more than the entire
deterministic pipeline. That is the honest argument for keeping the visual pass
off by default, and it is an argument about latency rather than money.

Model calls: 2 VLM + 1 synthesis across the whole pass. Cache hit rate on the
VLM pass 0/2; the semantic cache absorbed most repeat work.

---

## 11. Security audit — clean

Run end to end against a local endpoint with two distinctive fake credentials
(`sk-live-DO-NOT-LEAK-abc123ZZZ`, `sk-or-v1-DO-NOT-LEAK-999999xxx`) planted in
the environment.

| surface | result |
|---|---|
| stdout / stderr | no credential |
| 73 artifact paths (contents + filenames) | no credential |
| 974 cache files | no credential |
| exception traceback | no credential |
| test fixtures | 6 distinct key-shaped strings, all obviously synthetic placeholders (`sk-openai`, `sk-secret`, `sk-do-not-log-me`, `sk-or-v1-`) used as *negative* assertions |

---

## 12. Cache, reproducibility, artifacts

**Cache contract — 30 checks.** Every dimension named in the brief changes the
key, one at a time: VLM model, provider, model version, prompt version, schema
version, context, frame bytes, frame id; synthesis model, provider, prompt
version, schema version, evidence content, detail level, output language.
Failed/unavailable/malformed outcomes are uncacheable. Corrupt entries are a
miss, not a crash. Keys are content digests and hold no credential.

One real defect found here, on video B — see §13.2.

**Reproducibility.** Same video, two runs, warm cache: `evidence.json`
**byte-identical**, `REPORT.md` **byte-identical**, `REPORT.json`,
`frames.json`, `alignment.json` structurally identical, evidence ids unique,
every frame carrying a measured timestamp, provenance markers present.
Determinism is claimed for **evidence extraction and report rendering only**.
Any live model call is nondeterministic and is not covered by that claim.

**Artifacts.** All five runs carry the full set: `meta.json`, `transcript.json`,
`transcript.md`, `frames.json`, `FRAMES.md`, `frames/`, `visual.json`,
`alignment.json`, `evidence.json`, `validation.json`, `REPORT.json`,
`REPORT.md`, `report_validation.json`. Every JSON parses; provenance,
citations, frame references and report references all resolve.

---

## 13. Failures found

### 13.1 Doctor defect — a working provider reported NOT QUALIFIED

`uv run scripts/vlm_doctor.py` with no `--frame` built a **1×1 pixel PNG**,
which OpenAI rejects with `image_parse_error: unsupported image`. The failure
was then *misclassified*: `_MODE_UNSUPPORTED` in `chat_json.py` matched the words
"unsupported" and "invalid_request_error", which every OpenAI 400 carries
whatever the real problem is. So an unsupported image was reported as "endpoint
rejected the structured-output parameter", the negotiator descended all three
modes for nothing, and the doctor printed a false diagnosis naming the wrong
subsystem. `openai/gpt-4o-mini` — fully qualified on a real frame — was reported
NOT QUALIFIED on the one path a first-time user takes, because the documented
command has no `--frame`.

**Severity: high** for anyone qualifying a provider · **detected automatically**
(the doctor's own output) · **fix applied**: the classifier now checks a
media-rejection marker before a mode-rejection marker, and the built-in probe is
a valid 64×64 PNG rather than a single pixel. Four regression checks pin both.

**Verified as far as an empty account allows.** The same command now reports
`HTTP 402: This request requires more credits` — the image is accepted and the
true blocker is named, where before it named a structured-output disagreement
that was not happening. It cannot be walked through to `QUALIFIED` because the
OpenRouter account has no credit; that is the same blocker as §5, not a
property of the doctor. Qualification of this provider on a real frame was
verified earlier in this pass (§4).

### 13.2 Four defects in the pipeline, all found by real data

| # | defect | where | severity | fix |
|---|---|---|---|---|
| 1 | **The structured-output ladder reset to the top instead of descending.** After a mode was sent and ignored, the next call re-sent it. A model that cannot do strict JSON never reached the two weaker modes that would have worked. Measured: 8 frames, 8 identical non-conforming replies. | `vlm.py` | high | descend one rung per non-conforming response |
| 2 | **`relationships` accepted only one spelling of its endpoints.** `_edge` read `from` and wrote `source` over the top, so a model answering `source`/`to` had every edge discarded as "has no 'source'". | `vlm.py` | high | accept both |
| 3 | **The truncation retry was gated on `status == "failed"`.** A response cut off *inside* a JSON string arrives as `malformed`, so the remedy the code already had never fired for the case that needed it. 4 frames, 4 "Unterminated string" refusals, no retry, on a model that answered all four at a larger budget. | `vlm.py`, `chat_json.py` | high | key on `finish_reason`, shared ceiling constant |
| 4 | **The semantic cache key omitted each frame's timestamp.** The extraction *emits* a provenance timestamp, so the timestamp is part of the answer; a key without it served an answer computed against a different frame time. Real symptom: video B failed its own evidence gate with 10 `frame-timestamp-mismatch` errors (`t_start=266.0` for a frame `frames.json` placed at 266.76), all from entries cached before the frame was re-measured. | `semantic.py` | **high — this is what made a real video un-reportable** | add `t_start`/`t_end` to the key, with a regression check |

### 13.3 Four gaps in the report validator, all measured before changing anything

Each was probed first and confirmed undetected, then fixed and re-measured.

| # | gap | fix | false positives after |
|---|---|---|---|
| 1 | A fabricated sentence with no proper noun, figure, URL or marker passed every check. | content words in a factual block must appear in the evidence document | 0 / 20 documents |
| 2 | A removed hedge passed; the numbers table stated it correctly while the narrative restated it flatly. | `hedge-dropped`, reusing `synth_validate._has_hedge`; and the **renderer** now carries the hedge into the per-segment figure list | 0 |
| 3 | A whole section rendered twice passed. | `section-duplicated`, plus `item-duplicated` for an item with two narrative homes | 0 |
| 4 | A malformed timestamp (`[01:5]`, `[00:01:15]`) was silently skipped — `timestamp-unparsable` was **unreachable**, because the marker regex only matches `\d{1,2}:\d{2}`. | a citation-shaped scan for markers the parser cannot read | 0 |

The content-word rule needed a deliberate widening: 15 fixtures were not enough
and a real video added four template words (`possibility`, `referent`, `side`,
`statement`) on the first run. `acceptance/template_audit.py` exists so that
measurement is repeatable rather than a one-off, and the false-positive count is
**0 across 15 fixtures + 5 real artifacts**.

A fifth strengthening came out of the injection test: the number check was
set-membership over the whole document, so a figure changed from 20 to 27 passed
whenever 27 appeared *elsewhere in the same video* — which, on a report carrying
ninety figures, is most of the time. `number-misattributed` binds each figure to
the evidence its own block cites. My first version of that rule was dead code
(the document-wide short-circuit defeated it); the injection test caught it, and
a false positive on a narrative block whose figures live in the cited *snippet*
rather than the cited *item* was then fixed by binding to the block's whole
provenance.

### 13.4 Repository and documentation

| finding | severity | fix |
|---|---|---|
| **The installed skill was a stale pre-architecture duplicate.** `~/.claude/skills/yt-report/` was a real directory, not the symlink AGENTS.md requires. It shipped the retired `verify_report.py`, four scripts instead of twenty-five, and SKILL.md carrying **the old first-person creator voice contract this repo explicitly reversed**. Any user invoking `yt-report` got the wrong architecture and the wrong voice. | **high — user-visible** | replaced with the documented symlink; the old copy is preserved at `~/.claude/retired-skills/yt-report.retired-2026-09-25`, outside the skills-discovery directory |
| **Three vacuous tests.** `or True` in three checks, each passing unconditionally. | medium | all three replaced with assertions that would fail if the code broke; one gained a second assertion |
| `label_sheet.py` and `ocr.py` import Pillow with no dependency metadata, so the commands SKILL.md documents fail as written | medium | PEP 723 blocks added; the documented command now works |
| `yt_fetch.py --vlm` accepted two providers where `build_evidence.py` accepts eight — a second door to one subsystem with a smaller vocabulary | medium | widened to the full list |
| `--output-language` documented as changing the prose; it is a no-op without a writer | medium | SKILL.md corrected with the measurement |
| `visual.json["vlm_summary"]` used `setdefault`, so a second VLM pass left the *first* run's summary beside the new per-frame blocks | medium | now assigned, so the summary always describes the pass that just ran |
| `vlm_summary` reported `calls_failed: 8` alongside `vlm_calls: 0` for a run that asked nothing | medium | uncalled frames are counted separately, as `frames_not_analysed` |
| **Vacuous-test audit** | | 3 found, 3 fixed, 0 removed, 0 remaining. No `assert True`, no `check(..., True)`, no test that asserts only "no exception". The pre-existing `anti-vacuity` checks were inspected and are real. |

---

## 14. Human review of the reports

Performed by reading `REPORT.md` against `evidence.json`, not by scoring.

| dimension | A (commentary) | D (lecture) | verdict |
|---|---|---|---|
| **Summary** | title + section list + name list + counts | same shape | **Weak.** It indexes; it does not state the video's argument. A reader learns the shape of the video and nothing about its claim. |
| **Takeaways** | one 60-word run-on caption fragment; a Patreon plug ranked as a takeaway (D) | same | **Weak.** Transcript sentences chosen by ranking, not takeaways. No synthesis. |
| **Detailed notes** | coherent per segment | coherent | **Adequate.** Reads as a transcript digest, not a written account. |
| **Extraction** | names, figures, URLs, timeline all present | same | **Good, with noise** — see below. |
| **Visual** | nothing | nothing | **Absent, and honestly banner-marked.** The VLM produced no evidence (§4). |
| **Uncertainty** | `referent not labelled on the source`, garbled OCR kept verbatim | same | **Good.** Hedges and unreadables survive. |
| **Conflicts** | none in evidence | none | B carries 4 unresolved conflicts, all rendered with both sides. |
| **Noise** | 20 figures listed; ~6 are real, the rest UI chrome (`21 — Browse all skills`, `173K — & system`, `127 — CA`, `02 — PHASE TWO`) | `And I` as an entity | **Poor in the numbers table.** |
| **Repetition** | 1.6% | 5.6% | **Good.** |

**The one thing that must improve before this is called good enough** is the
numbers table. It lists interface chrome as "Numbers, Pricing & Benchmarks". The
system *knows*: the reliability section says "78 of 92 figures rest on one text
read of one frame with nothing else naming them" and warns the labels may be
garbled. That honesty is the architecture working — but a reader who does not
read to the bottom meets twenty numbers before meeting the caveat. Severity
**medium-high**; detected automatically but not acted on; fix not made here
because separating a benchmark from a UI counter is a judgement about context,
and this repository's rule is that ambiguous findings are metrics, not failures.

---

## 15. Remaining limitations

1. **Model-written prose and the real critic are unmeasured.** The one
   substantive reason this is CONDITIONAL. Not a code defect; a provider that
   produces a conforming observation and a draft within budget would close it,
   and the same `openai/gpt-4o-mini` that already qualifies on a real frame would
   do it given credit.
2. **VLM evidence is 0 items across all five videos.** The visual pass has never
   produced a conformant observation in this environment. Its failure handling is
   proven (degradation matrix, B/H/I); its *success* path is not.
3. **Non-English entity extraction is degraded** — English stopwords against
   German capitalisation (§6). High for non-English input.
4. **A reversed comparison is undetectable** by the current gates (§8, case 5).
   UNMEASURED and expected to fail.
5. **`--output-language` requires a writer** (§6).
6. **A failed run leaves the previous `REPORT.md` in place** rather than
   removing it. Correct per the documented contract and the right default — a
   failed run must not destroy a good report — but a reader who does not check
   `report_validation.json` can read a stale report. `validation.json` exists
   precisely to be checked, and the failure path prints that no report was
   written.

---

## 16. For a new user

```bash
export PATH="$PWD/bin:$PATH"          # once
yt-report "https://www.youtube.com/watch?v=<id>"
```

Prints the artifact directory and the report's coverage state. `--help` lists
the options. Exit 0 ok · 2 no captions, ASR off or failed · 3 no captions
obtained · 4 the evidence layer rejected what it built · 70 nothing was written.

A `REPORT.md` opens with a coverage banner when degraded. Read it before
anything else: it is the only line that tells you whether a chart in the report
was actually looked at.

---

## 17. Example artifact tree

```
~/Desktop/yt-reports/this-new-jev-claude-os-just-changed-every-ai-workflow-EOdXR6lU5ZA/
├── meta.json                 2.0K    title, channel, date, duration, description, chapters
├── transcript.json          155K     fragments: start, duration, text — the citation unit
├── transcript.md             25K     the same, timestamped and grouped by chapter
├── frames/                  51 files sampled frames
├── frames.json               5.7K    frame manifest with measured timestamps
├── FRAMES.md                 1.7K    the same, human-readable
├── contact-sheet-1.jpg      186K     triage sheet
├── contact-sheet-2.jpg      166K     triage sheet
├── visual.json              837K     per-frame OCR read + per-frame `vlm` block
├── alignment.json           266K     which frames illustrate which sentences
├── evidence.json            2.0M     the canonical record: items, segments, frames, conflicts
├── validation.json           800B    the evidence layer's verdict on itself
├── REPORT.md                6.7K     the report
├── REPORT.json               29K     which evidence ids each section rendered
├── report_validation.json   1.2K     the report's grounding verdict — the last gate
└── URL.txt                    43B
```

A representative `REPORT.md` is video A's, 109 lines, reproduced in full at
`~/Desktop/yt-reports/this-new-jev-claude-os-just-changed-every-ai-workflow-EOdXR6lU5ZA/REPORT.md`
and reviewed in §14.

---

## 18. Acceptance checklist

| criterion | status |
|---|---|
| all critical validator corruption tests pass | ✅ 38/38, 0 silently repaired |
| four materially different real videos processed | ✅ |
| at least one real VLM run produces valid evidence | ❌ 2 calls, 0 conformant — §4, §15.2 |
| real critic run completed | ❌ no provider available — §5 |
| non-English acceptance tested | ✅ tested; the output-language half is a no-op — §6 |
| degradation matrix tested | ✅ 34/34 |
| cache correctness tested | ✅ 30 checks; one real key defect found and fixed |
| reproducibility measured | ✅ byte-identical, with the deterministic boundary stated |
| security audit clean | ✅ |
| documentation synchronized | ✅ installed-skill drift, dependency metadata, provider vocabulary, `--output-language` |
| no vacuous critical tests | ✅ 3 found, 3 fixed |
| all artifacts validate | ✅ 5/5, all 14 files present |
| no unsupported claims/numbers/URLs in final reports | ✅ 0 across all five |
| visual evidence actually represented where present | ⚠️ the frames are read by OCR; **no model-derived visual evidence exists** to represent |
| unresolved conflicts remain unresolved | ✅ 4 on video B, both sides shown |
| obvious OCR garbage does not dominate | ⚠️ it does dominate the numbers table; the report says so at the bottom — §14 |
| final reports manually inspected | ✅ §14 |

---

## 19. Reproducing this pass

```bash
uv run scripts/selftest.py                                    # 1180 checks, offline
uv run --with yt-dlp --with pillow --with numpy --with youtube-transcript-api \
  python3 acceptance/degradation_matrix.py <artifact-dir>      # 34 assertions
uv run --with yt-dlp --with pillow --with numpy \
  python3 acceptance/grounding_injection.py                    # 38 corruptions
uv run --with yt-dlp --with pillow --with numpy \
  python3 acceptance/reproducibility.py <artifact-dir>         # 2 runs
uv run --with yt-dlp --with pillow --with numpy \
  python3 acceptance/secret_audit.py                           # credential audit
uv run --with yt-dlp --with pillow --with numpy \
  python3 acceptance/template_audit.py                          # validator false-positive rate
uv run --with yt-dlp --with pillow --with numpy \
  python3 acceptance/metrics.py <label>=<dir> ...              # the metrics table
```

Test count went 1074 → **1180**: three vacuous tests replaced, two new suites
(`selftest_grounding.py`, `selftest_cache.py`), four new validator rules and the
regressions that guard them. Zero failures, and the number is no longer the point
— 38 of 38 deliberate corruptions are detected with a named code and a useful
message, and the 15 synthetic fixtures are no longer the only evidence that any
rule fires.
