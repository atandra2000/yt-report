---
name: yt-report
description: Convert a YouTube video into the creator's own written notes — the overview and the stories as the creator would write them before recording, in their own voice, with every figure, name and product they mention. Use whenever the user pastes a YouTube link (or watch/youtu.be/shorts URL) and wants it turned into notes, a written record, a digest, or study material. Covers podcast, interview, lecture and AI-news videos. Produces a first-person recap, not a third-person report, and never evaluates or fact-checks.
---

# yt-report

Write the video as **the creator's own notes for the video** — as if they wrote these notes before they recorded it.

## 1. Fetch

```bash
uv run ~/.claude/skills/yt-report/scripts/yt_fetch.py "<url>"
```

Writes `~/Desktop/yt-reports/<slug>-<id>/`:

| file | what it is |
|---|---|
| `transcript.md` | full timestamped transcript, grouped by real YouTube chapters when they exist |
| `frames/` + `contact-sheet*.jpg` | sampled visual frames, and triage sheets showing all of them |
| `FRAMES.md` | frame manifest with measured timestamps |
| `meta.json` | title, channel, date, duration, view/like counts, description, tags, chapters |

Captions first, local Whisper ASR as the automatic fallback — no flag, no API key, nothing uploaded. The one 720p download serves both ASR and frames, so it retries rather than losing both. Transcripts cache per video+language, so a re-run costs seconds.

Status: `0` ok · `2` no captions, ASR disabled or failed · `3` no captions obtained. **A non-zero status still emits the frames** — a caption-less video is not a failed fetch.

**Flags you will actually reach for:** `--lang` (non-English) · `--frames -1` (frames only, no video) · `--batch` (several URLs) · `--out DIR`. The rest are in `--help`.

**Do not pass `--frames`.** The count is automatic: ~1 per 30 s, clamped to 12–48, and two samplers run in parallel, so cut-heavy footage returns more than the target. Expect 1–2x, never assume either.

**English only** — the script will not report another language's captions as English. If the audio is not the language asked for either, say so and ask.

→ `references/browser-fallback.md`, only when `download_video` itself fails. A playability oracle and URL source, **not** a caption-block bypass.

## 2. Read it all

The transcript alone misses exactly what these videos exist to show: slides, diagrams, benchmark tables, code, terminal output, UI screenshots. Both halves are required.

1. **`meta.json`** — duration, chapters, description. Descriptions carry the creator's own resource links.
2. **`transcript.md` in full.** Never sample or skip the middle; if it is too large for one read, work through it piecewise and keep a running note.
3. **Label every frame.** This maps frame IDs to contents reliably and shows you everything that exists, in one pass:

```bash
uv run --with pillow ~/.claude/skills/yt-report/scripts/label_sheet.py <artifact-dir>
```

It writes `labelled/labelled-N.jpg` tiles, each stamped with its own filename, and prints the ID list per sheet. Use it **instead of** the skill's own `contact-sheet*.jpg`, whose tiles are not in filename order.

4. **Open individually every frame you intend to describe.** A 1280px frame usually needs no cropping; crop and upscale only when text is genuinely small or the video is 4K.

**A frame read can return fewer images than you asked for, with nothing marking the drop.** The labelled sheets remove the mapping hazard; when you open frames individually instead, count the images before matching them to filenames by position — a dropped result shifts every later one by one, so a run of frames each off by one means a result was lost, not misread.

**Chapter titles are uploader metadata, not a verified index** — hand-typed, generated from an early cut, or left over. On one 20-minute video 2 of 7 did not describe what played under them. The damage is misplaced content, so skim the first lines under each title and follow the transcript for the real order. Timestamps are usually right even when titles are wrong.

## 3. Write it

Save to `<dir>/REPORT.md`, then give the user the overview in chat. Pick the shape from `references/genres.md`.

```markdown
# <title>

## Overview
<A few sentences, first person: what I cover, in order, and who it's for.>

## <Topic>
<First person. What I found, what it means, the numbers exactly, the people and
products involved, what I show. This is the substance — reproduce tables, formulas,
code and figures rather than gesturing at them.>

## <Topic>
<...>

## Tools, people and links I mention
<Table: name | what it is | where it comes up | link.>

## Terms used
<Table: term | plain-language definition | where it comes up.>
```

- **The Overview earns its place.** Two to four sentences: the stories in order, and the one thing a reader would miss if they skipped. Not a thesis, not a verdict — coverage. The common failure is restating the title with the connectives removed; if a sentence could be cut and pasted into the `<title>` line, it is not an overview.
- **First person throughout.** "I found", "I break down", "I show", "I try" — not "the video discusses", not "we see".
- **No meta-language.** Nothing about the video as a *recording*: no "on screen", no frame IDs, no "the transcript reads", no "sponsored segment", no notes on caption quality, no "in this video I". The creator does not write about their own recording; they write about their subject.
- **Headings name subjects, not tape positions.** "Google's Project Suncatcher", never "12:10 — …". A creator's notes are organised by idea.
- **Every topic gets a section.** Count the distinct topics in the transcript, then check you wrote one section each. Under-covering is the failure nobody catches, because a short piece about a five-story video reads as deliberate editing rather than a dropped segment. News roundups especially: one section per story, however thin.
- **Roughly one section per 2–5 minutes of runtime**, 150–400 words each. A 5-minute video gets a page; a 2-hour lecture gets a long piece. Do not pad.
- **No bullet-wall prologue.** Start with content.
- **Both tables are indexes, not summaries.** A row exists only if a reader would want to look that thing up: a product, paper, dataset, person with a role, or a link. A figure already given in the body does not get a table row, and a term already explained in the body does not get a Terms row. If a table has two rows, it is not earning its place.
- Define jargon the first time it appears, in the simplest words that stay accurate, at the level the creator is clearly above. Preserve numbers, names and versions **verbatim** — a version number is data, not prose.

`verify_report.py` catches stray frame citations left over from a draft. Run it only if you cited frames:

```bash
python3 ~/.claude/skills/yt-report/scripts/verify_report.py <artifact-dir>
```

**After changing any script here, run the self-test** — pure functions, cache, ffmpeg on a synthetic video, and the retry path against a faked downloader, no network:

```bash
uv run ~/.claude/skills/yt-report/scripts/selftest.py
```

It exists because a change once shipped that produced **zero** contact sheets while reporting `"status": "ok"`, and a refactor once passed a list where a string was required. To rebuild the skill's own sheets from frames already on disk, use `resheet.py`.

## 4. Rules

Each is stated once, deliberately. Most of the reasoning is in the observation log; restating a rule does not make it hold harder.

### 4.1 Voice and stance

**Write as the person making the video.** If the piece reads like a third party watching from outside, it is in the wrong voice.

**Report the material; do not evaluate it.** No assessing whether a claim is right or well-sourced. No ranking or scoring the models, products or people covered. No fact-checking, no arithmetic on the reasoning, no repairing an argument, no outside knowledge, no "added context". No section whose purpose is to make up a reader's mind — not "questions worth asking", not "what this gets wrong", not "should you trust it".

**Their opinions are theirs.** Where the creator expresses a view, the notes carry it in their words at its strength. What does not belong is a view of their *sourcing* that they never expressed.

**Hedging is preserved, not managed.** If I said the plan "could cost $500", the piece says could cost $500. Not flattened to *costs $500*, not inflated to *an unsupported $500 claim*. Both are my verdict; neither is a record.

**Tension gets stated, never resolved.** If two things in the material disagree, I say so and stop: "one capture shows 600 with a 500 override; another shows 500 outright." Not "these are inconsistent", not "the real price is probably…". Spotting the tension is the job. Settling it is not.

**A faithful record of a wrong claim is a success.** A claim I reached by my own reasoning is a failure.

**A correction I made on camera is the record, not the slip.** Creators misspeak, restart, and say "sorry, it's 500, not 600" mid-sentence. The piece carries the corrected figure, and the recovery is not news — nobody needs telling that a creator caught a stumble. What does need both numbers is a correction the creator never made on camera.

**The piece must stand alone.** A reader opening it has not seen the video. Never write "as shown earlier", "like the previous example" or "the second one" without enough in the sentence to identify what is meant. If a passage only makes sense in sequence, bring the earlier detail forward or cut the callback.

### 4.2 Accuracy, internal

These govern what goes in. None of them is ever narrated in the piece — the work surfaces as accuracy, not as method.

- **A string is from a frame only if it is visible in that frame.** The `meta.json` description is uploader metadata with no standing of its own: a title, tag, chapter name or blurb found there is not a quotation, and moving one into a quotation slot is a fabrication however true it is.
- **Never present as a quotation a line that is not on the card.** A creator who reads their slides aloud makes it look as though every spoken line is also shown. It usually is not. Check the frame, or write it as my own summary.
- **A contact sheet decides which frame to open; it never supplies contents.** Nothing goes in about a frame until that frame has been opened. A 384px tile tells you *that* a terminal is there, never what is in it.
- **Provenance does not travel along a timeline.** A source belongs to the thing it belongs to and to nothing else — not the segment, not the adjacent frame, not whatever was nearby. Both failure modes count: attributing a source I did not read, and **leaving one out when the material does show it**, which hides information.
- **The transcript garbles names; a frame does not.** When the audio mangles a proper noun, the printed name is correct and wins — replacing a legible one with the term I expected is a real error. **Never invent a correction**: a note claiming the captions "rendered X for Y" when neither string occurs anywhere is worse than no note, because it is a fabrication dressed as rigour. With no frame or transcript line to support a fix, say the name was unclear.
- **`upload_date` is not a filming date.** If the material never says when it was recorded, give the upload date and stop.

### 4.3 Describing frames

- **A correct fact does not validate where I read it.** Re-open the frame and read the string off it in the same pass that writes the sentence. The mechanism is adjacency: in a fast animation neighbouring samples are near-identical, and a re-read late in a long session silently reorders the mental index. Adjacent frames are the danger zone; the tail of a sequence is worst.
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

### 4.5 Absence

**Never assert a negative I have not checked every frame to support.** "No frame shows this", "there is no chart" — nobody reads a negative as needing evidence, which makes it the easiest thing in a piece to get wrong. A piece once opened one contact sheet of two and concluded "exactly one" of 52 frames contained text; four opened frames refuted it, one carrying the prompt verbatim and differing from the captions in two places.

- Absence claims require checking **every** frame, or they are not made.
- If I have not looked everywhere, the honest form is scoped: "I did not see this anywhere in the material I went through."
- A universal negative is the natural place for a fabrication to hide, because it frees the writer from quoting anything. **Anything that removes the need to open a frame deserves suspicion.**

"Unreadable" and "absent" are different claims, and both are claims.

### 4.6 When something has to come out

**Removal is not annotation.** A wrong fact usually sits in four places at once — the overview, a section, a table, a closing line. "Retracted, correction follows" repairs the one spot a reviewer read and leaves the other three asserting it. So: **grep for the distinctive terms and delete every occurrence**, then re-read to confirm. Prefer deleting the words to qualifying them.

**A fix instruction can be wrong.** If told to re-pin a fact to a specific frame, open that frame and check. In two separate cases it was on neither the named frame nor the proposed one, and deleting the fact was correct. A reviewer once asserted a chart plotted a rival "further right, at ~195"; re-deriving the calibration reproduced every verified value exactly and showed no such point existed. So: if it is not there, **remove it rather than relocate it** — a plausible new home is as false as the old one. When a reviewer's premise is overturned, say so and show what was measured.

## 5. After writing

Tell the user the path and offer: a shorter one-page version, an Obsidian note, or follow-up Q&A on any section.

## Sibling skills

Pairs with `agent-reach` (fetches from other platforms) and `obsidian-vault` (note storage). This skill owns the YouTube path only.
