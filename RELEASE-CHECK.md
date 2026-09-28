# RELEASE-CHECK.md

**Status: `CONDITIONAL`.** The architecture is frozen, the deterministic
pipeline is fully exercised offline, and the model-backed layers are *not*
qualified. This document records what is true today, what would change the
verdict, and the exact commands that decide it. It does not claim production
readiness, and the release claim stays `CONDITIONAL` until a real synthesis, VLM
and critic run have all been exercised successfully against a live provider.

---

## 1. Versions

| what | value | where it lives |
|---|---|---|
| architecture / pipeline | **3** | `scripts/versions.py` — `PIPELINE_VERSION` |
| evidence schema | **3** | `scripts/evidence_model.py` — `SCHEMA_VERSION` |
| extraction | **7** | `scripts/versions.py` — `EXTRACTION_VERSION` |
| VLM prompt | **2** | `scripts/versions.py` — `VLM_PROMPT_VERSION` |
| synthesis prompt | **3** | `scripts/versions.py` — `SYNTH_PROMPT_VERSION` |
| synthesis schema | **1** | `scripts/versions.py` — `SYNTH_SCHEMA_VERSION` |

The first three are the ones that decide whether an artifact on disk is still
readable. `EXTRACTION_VERSION` moved 6 → 7 in this phase; the note in
`versions.py` records exactly which extraction rules changed and why, because
the obligation to bump it is the one that gets missed.

## 2. Frozen interfaces

Unchanged in this phase, and not to be changed without a version bump and a
migration note:

```text
yt_fetch → ocr → align → vlm → semantic → evidence.json
         → evidence_validate → synthesis → report → report_validate
```

`evidence.json` is the single canonical artifact. There is one validator per
layer — `evidence_validate` for the artifact, `report_validate` for the
deterministic report, `synth_validate` for the model-written draft — and none of
them is a second source of truth.

## 3. Supported providers

| provider | base URL | vision | structured VLM | synthesis | critic |
|---|---|---|---|---|---|
| `anthropic` | from `ANTHROPIC_BASE_URL` | yes | yes | — | — |
| `openai` | `https://api.openai.com/v1` | yes | yes | yes | yes |
| `openrouter` | `https://openrouter.ai/api/v1` | yes | yes | yes | yes |
| `ollama` | `http://localhost:11434/v1` | yes | yes | yes | yes |
| `lmstudio` | `http://localhost:1234/v1` | yes | yes | yes | yes |
| `llamacpp` | `http://localhost:8080/v1` | yes | yes | yes | yes |
| `custom` | `--*-base-url` (required) | yes | yes | yes | yes |
| `fake` | — | test double only, not a fallback | | | |

The critic runs on the synthesis model; there is no separate provider to
configure. `ollama`, `lmstudio` and `llamacpp` are keyless.

## 4. Deterministic test results

```bash
uv run scripts/selftest.py
```

| | |
|---|---|
| checks passing | **1707** |
| checks skipped | **0** |
| failures | **0** |
| test functions | 173 across 9 files |
| network | none — the suite refuses to need it |
| credentials | none |

By area: `selftest.py` 46 functions, `selftest_synthesis.py` 28,
`selftest_grounding.py` 17, `selftest_doctor.py` 17, `selftest_entities.py` 17,
`selftest_numbers.py` 16, `selftest_report.py` 13, `selftest_cache.py` 11,
`selftest_vlm_provider.py` 8.

These 1707 checks are **not** of one strength. See §10, *Test Provenance and
Recovery*: one of the nine files was reconstructed after its source was lost, so
a subset of these checks is reconstruction-validated rather than
historically-regression-validated, and the release claim does not treat them as
equivalent.

Integration-shaped coverage (real ffmpeg on a synthetic video, a real cache
round-trip, the real validator over real fixtures, the full report layer over
fifteen fixtures) lives in `selftest.py`. Provider behaviour — every response
shape, every refusal, the structured-output negotiation ladder — is covered in
`selftest_vlm_provider.py` and `selftest_doctor.py` against faked transports
rather than a live endpoint, and the *live* counterpart is §6 below.

## 5. Live qualification states

**No provider was exercised in this phase.** No credential or egress was
available, so every capability below is `UNTESTED` — which is a statement that
the question was not reached, not that the answer was no.

| provider | model | vision | structured VLM | synthesis | critic | status |
|---|---|---|---|---|---|---|
| _(none exercised)_ | — | UNTESTED | UNTESTED | UNTESTED | UNTESTED | CONDITIONAL |

The last real run reached 9/10 synthesis gates before a prompt fix; the fix has
**not** been re-run against a live provider. Treat that as unverified.

`DEGRADED` is the fourth state and is distinct from all three above: an optional
model layer failed and the deterministic renderer still produced a grounded
report. The pipeline never fails because a provider is down; it degrades and
says so in `report_validation.json` under `fallback_reason`.

## 6. What this phase actually changed

Everything below is a correctness fix, a quality fix, a testability improvement
or a documentation fix. No feature, subsystem or interface was added.

**Number quality.** The table test did not recognise a table of bare decimals,
because `find_values` refuses a unitless number on its own — correct in
isolation, wrong inside a grid. The ordinary shape of a benchmark table
(`81.9` beside `Qwen`) therefore never qualified. A code-editor gutter is now
excluded structurally, by the shape of the leftmost column rather than by any
list of file extensions, and a clock time is recognised as a timestamp rather
than a measurement. §3's category matrix is now nine cases running in both
directions.

**Number subjects.** A single word directly after a personal pronoun is that
pronoun's predicate (`we saw 42 mentions` filed the figure under `saw`); a
currency written as a word is a unit (`500 dollars` filed the figure under
`dollars`, which `_clean_referent`'s own docstring already claimed it did not);
and the referent gate now consults the shipped stopword resources, so
`Die Latenz stieg auf` and `Wir sahen` are no longer German subjects. The
precision/recall tradeoff is deliberate and one-directional: `subject = null` is
preferred to a confidently wrong subject, and the price is recall on phrasings
this layer cannot parse without a parser.

**Multilingual entities.** Time adverbs, everyday nouns and category nouns were
arriving as candidates in six of the seven languages; the resources now carry
them. A category word introduced a name (`Werkzeug Docker`) and dragged itself
into it. Entity typing was narrowed to the words beside a name — a cue
forty-eight characters away was typing names it had nothing to do with, and
`According` was being filed as a `tool`. A first-person-plural Romance verb
opening a sentence is no longer a name, by inflection rather than by vocabulary.
Under-extraction is preferred to a false entity; the cost is that a name
introduced by its category word in German is dropped whole in that phrasing and
kept everywhere else.

**Synthesis contract.** The prompt said "cite it in one of the three, never in
two"; the gate exempts the summary. The instruction was stricter than the check
it was describing, so the writer was told to avoid a shape the report wants. The
prompt now names the two narrative homes, says the summary is not one of them,
and says the count is on `evidence_ids`. Five cases pin the contract: one item in
one section, the same item in a takeaway and a section, an item reached
indirectly through a summary of another, an item in a summary and a section, and
one item in a takeaway alone. The conflict invariant stays a **report-level**
rule — a paragraph may discuss one side, and the report as a whole must disclose
every side of every conflict it references.

**Output language.** `--output-language` now states, in the CLI output and in
`report_validation.json`, that
`output-language requires a configured synthesis writer for true translation`.
The deterministic renderer has no translations to choose between, and a run that
was asked for Japanese and produced English has not translated anything.

**Report contract (reviewed, unchanged).** The deterministic renderer composes
from items into fixed headings — `executive_summary`, `key_takeaways`,
`detailed_notes`, `extracted_details`, `timeline`, `reliability` — and produces
about 125 words for a whole video. That is the intended shape of a fallback: it
is valid, concise, grounded, and visibly not prose. Central argument, coherent
summary, useful takeaways, related-evidence grouping, visual interpretation,
uncertainty and conflict explanation remain the model synthesis path's work, and
nothing was done in this phase to make the renderer imitate a writer.

**Release check.** A single `yt-report release-check` command collects the four
qualification answers that previously lived in four places, with four states kept
apart, hard budget ceilings, and a provider matrix.

## 7. Known limitations

1. **Nothing model-backed is qualified.** Vision, structured VLM output,
   synthesis and the critic have all been exercised only against faked
   transports. The offline suite proves the *contract*; it does not prove any
   provider honours it.
2. **The last real synthesis run is pre-fix.** 9/10 gates, then a prompt change.
   Unverified against a live model.
3. **A strict-mode JSON schema is not the same as a valid one.** OpenAI's
   `strict` mode wants `additionalProperties: false` on every object node and
   every property listed in `required`. A hand-written schema can satisfy one
   validator and still be refused by the endpoint. Assert those invariants
   against the real endpoint before trusting a schema on it.
4. **Gateway errors bury the cause.** A fronting gateway reports "Provider
   returned error" at the top and the real complaint several levels down in
   `error.metadata.raw`. Match the cause, not the top line.
5. **A bare figure in a cell is a value now**, which widens what a table can
   promote. The gutter exclusion and the clock-time exclusion are what keep that
   safe, and both are structural — an unrecognised layout is a miss, which is the
   direction this filter is allowed to be wrong in.
6. **A German name introduced by its category word is lost.** `das Werkzeug
   Docker` is one capitalised run; the run is dropped whole. Documented in the
   resource file rather than papered over with a phrase splitter.
7. **Entity *types* are a neighbourhood reading, not a fact.** A cue two words
   away types the name. Uncertain names stay `entity`, which is the honest
   reading.
8. **Deterministic number subjects under-extract in German.** The gate refuses
   a clause it cannot parse rather than filing the figure under a preposition.
9. **OCR depends on `swiftc`.** Without it runs complete and record
   `ocr_status: "unavailable"`, which is a weaker report, not a failed one.

## 8. Commands

The normal path, unchanged:

```bash
export PATH="$PWD/bin:$PATH"
yt-report "<youtube-url>"
```

Nothing about the internal scripts is needed to produce a report.

For maintainers:

```bash
# the whole offline suite
uv run scripts/selftest.py

# the release check — cheapest useful qualification sequence
yt-report release-check \
  --vlm-provider openrouter --vlm-model openai/gpt-4o-mini \
  --synth-provider openrouter --synth-model openai/gpt-4o-mini

# ...as JSON, and written to a file
yt-report release-check --json --out release-result.json

# the individual capabilities, if you want one in isolation
yt-report vlm-doctor         --vlm-provider openrouter --vlm-model openai/gpt-4o-mini
yt-report synthesis-doctor   --synth-provider openrouter --synth-model openai/gpt-4o-mini
yt-report critic-doctor      --synth-provider openrouter --synth-model openai/gpt-4o-mini

# one small real end-to-end video, budgeted
yt-report eval-live "<url>" \
  --vlm-provider openrouter --vlm-model openai/gpt-4o-mini \
  --synth-provider openrouter --synth-model openai/gpt-4o-mini \
  --max-vlm-frames 2 --max-synthesis-calls 1 --max-critic-calls 1
```

## 9. Exact live qualification sequence

This is what turns the status from `CONDITIONAL` to `QUALIFIED`, in the minimum
budget that can. Run it in this order; each step is cheaper than the one after
it, and an early failure makes the later ones pointless.

```bash
export PATH="$PWD/bin:$PATH"
export VLM_API_KEY=...      # or SYNTH_API_KEY — the same value for one provider
export SYNTH_API_KEY=...

# STEP 1 — connectivity and credentials. Costs nothing if it works.
yt-report release-check --vlm-provider openrouter --vlm-model openai/gpt-4o-mini \
  --synth-provider openrouter --synth-model openai/gpt-4o-mini --json

# STEP 2 — VLM on a real frame. One frame, one call.
yt-report vlm-doctor --vlm-provider openrouter --vlm-model openai/gpt-4o-mini

# STEP 3 — synthesis on the built-in fixture. One draft.
yt-report synthesis-doctor --synth-provider openrouter --synth-model openai/gpt-4o-mini

# STEP 4 — the critic on four known-answer cases.
yt-report critic-doctor --synth-provider openrouter --synth-model openai/gpt-4o-mini

# STEP 5 — one small real video, if and only if 1–4 all passed.
yt-report eval-live "<a short url>" \
  --vlm-provider openrouter --vlm-model openai/gpt-4o-mini \
  --synth-provider openrouter --synth-model openai/gpt-4o-mini \
  --max-vlm-frames 2 --max-synthesis-calls 1 --max-critic-calls 1
```

The whole sequence at once, with the budget ceilings enforced:

```bash
yt-report release-check "<a short url>" \
  --vlm-provider openrouter --vlm-model openai/gpt-4o-mini \
  --synth-provider openrouter --synth-model openai/gpt-4o-mini \
  --max-vlm-frames 2 --max-synthesis-calls 1 \
  --max-critic-calls 1 --max-revision-calls 1 \
  --json --out release-result.json
```

### Reading the result honestly

- `PASS` — a real request held. Read the observation, not the checkmark. For
  vision, confirm the reading is actually correct on a chart, a diagram, a UI
  screen and a code frame. For synthesis, confirm the summary communicates the
  video's central point. Valid JSON is not good visual reasoning and not a good
  summary.
- `FAIL` — a real request came back and did not hold. The capability is
  disproven. This is the state that must never be softened.
- `UNTESTED` — no usable answer arrived. No credential, no credit, no egress, a
  spent budget. Nothing was disproven; the question was not reached. Do not
  convert it to `PASS`.
- `SKIPPED` — not requested.

`status` is `QUALIFIED` (every capability exercised and holding), `CONDITIONAL`
(not fully exercised), `DEGRADED` (an optional model layer failed while the
report still landed) or `UNQUALIFIED` (a capability failed with nothing working
beside it).

---

## 10. Test Provenance and Recovery

This section is not a footnote. It changes what the 1707 checks in §4 are
evidence *of*, and the release claim above depends on it.

### 10.1 Recovery was attempted first, and failed

`ORIGINAL TEST SOURCE NOT RECOVERED`

`scripts/selftest_entities.py` was destroyed while untracked. Every legitimate
local recovery source was searched before the reconstructed file was touched:

| source | result |
|---|---|
| `git log --all` for the path | no history — the file was never committed |
| `git reflog --all`, `git stash list` | nothing |
| `git fsck --lost-found --dangling --unreachable` | no dangling or unreachable objects |
| `__pycache__/selftest_entities.cpython-314.pyc` | **not** a recovery artifact. Its header records source size 17699 and mtime 2026-09-26 09:27:46 — byte-identical to the current reconstructed file, so it was compiled *from* the reconstruction, not before it |
| Claude Code file-history (991 entries) | no copy; the two entries naming the file are `yt_fetch.py` |
| editor local history (VS Code `User/History`) | directory does not exist |
| OS trash | unreadable (`Operation not permitted`) — not searched |
| swap / `.bak` / `~` / `.orig` / `.rej` / `.py.N` | none in the repo except `bin/yt-report.bak` (the CLI, not a test) |
| `/tmp` and `$TMPDIR` | `se.py`, `se_head.py`, `sn_head.py`, `sy_head.py` exist and are **reconstruction-session drafts** (mtimes 09:07–09:21 on 2026-09-26, i.e. written while the reconstruction was being written), not pre-loss originals |
| CodeGraph index (`CoreProjects/.codegraph`) | every row for this file carries the reconstruction's timestamp 1790395066785; it indexed the reconstruction, not a prior version |
| Time Machine / APFS local snapshots | `No machine directory found for host`; no local snapshots |
| Spotlight (`mdfind`) | only the live file and its `.pyc` |

**Not searched:** the Claude session transcripts (`.claude/projects/**/*.jsonl`),
the paste cache, and the context-mode conversation databases. The permission
classifier refused that read as PII, and it was not retried. These are the last
remaining plausible sources — a session transcript from before 09:27 on
2026-09-26 would very likely contain the file's full text. **Recovering the
original from them remains possible and was not done.** That is the single
largest remaining uncertainty in this document.

Nothing found was overwritten, and every mutation experiment restored its
source by SHA-256 comparison against a pre-run hash.

### 10.2 Which test sources are original, and which were reconstructed

- **Original:** `scripts/selftest.py` — the only test file ever committed
  (`96271c4`).
- **Unverified provenance:** the other eight files. They are untracked, so git
  holds no baseline for them. They were *not* reported lost, and nothing suggests
  they were reconstructed, but **git cannot prove it either way.** They should
  not be described as original with confidence.
- **Reconstructed:** `scripts/selftest_entities.py` only. Its module docstring
  states this explicitly, and states that its input fixtures are reconstructed
  rather than byte-for-byte original.

Because eight of nine files are untracked, the honest reading of §4 is: 173 test
functions exist and all 1707 checks pass, and the *historical regression* claim
rests on far less than that.

### 10.3 How the reconstructed tests were independently validated

Assertion labels were recovered from the baseline test log; **the fixtures were
not.** A fixture written by the same person who wrote the code under test
inherits that person's assumptions, so the reconstructed set was checked by
mutation testing instead: a deliberate defect is injected into the
implementation, and the named test must fail. A mutation that kills nothing is
reported as a gap, not dismissed.

**Entity suite — 15 mutations, 17 test functions, all 15 accounted for:**

| # | mutation | killed by |
|---|---|---|
| M01 | `is_ordinary` empty-input guard (negative control) | nothing — **correct**; the guard is redundant |
| M02 | peel German suffixes even when language was only guessed | E3 |
| M03 | stopword list never consulted | E1, E3, E9, E12, language matrix |
| M04 | drop the ordinary-vocabulary filter (broad capitalisation) | E1, E12, language matrix |
| M05 | widen the typing context window 2 → 48 words | typing-spreads |
| M06 | language detection always English, never morphology | E1, E4, E6, E9, E12, matrix |
| M07 | force every uncategorised candidate to `product` | E7, E14, typing-spreads |
| M08 | a bare capitalised word becomes name-shaped | E8, E14 |
| M09 | drop the unit-vocabulary branch of `_is_quantity_phrase` | E11 |
| M10 | keep the determiner in `The Opus` | E7 |
| M11 | drop the short all-caps category filter | **E13 (added by this audit)** |
| M12 | drop the `_AUXILIARY_LEAD` filter | nothing — see below |
| M13 | drop the Romance 1pl sentence-initial filter | language matrix |
| M14 | drop the corroboration salience bar | **E14 (added by this audit)** |
| M15 | `model` cue no longer matches | E7, typing-spreads |

**Numbers suite — 11 mutations, 16 test functions.** Killed: cross-modal
support, UI-label vocabulary, measure-label vocabulary, unit shape, table+topic
benchmark, `value_in_speech`. **N02 and N03 survived** and were characterised by
diffing the actual classifier verdict with and without the mutation: the
ordinal/cell rule changes only the `reason` string (identical role and
importance — outcome-subsumed), while the ui-chrome rule changes the role from
`ui` to `unknown`, which the existing test's `{ui, unknown}` disjunction could
not see. **N12 was added** to pin both.
**N08 (`_figure_in_claim`) survived and was proved unreachable**: brute force
over 10 speech/figure pairs found every case where it is true is already true
for the strictly broader cross-modal branch, which fires first. It is dead code,
not a coverage gap, and no test was written for it.

**Synthesis suite — 10 mutations, 28 test functions.** Killed: block grounding,
unsupported number, removed hedge, block conflict side, unsupported visual,
source kind, report-level conflict completeness, one-home. **S07 (URL) and S09
(malformed-draft wiring) survived** — both genuine: `t_synth_schema` called
`validate_draft` directly, so nothing asserted that a structural problem becomes
a blocking finding on the real entry point, and no test reached the URL rule at
all. **S11 and S12 were added**, each with a valid-fixture-passes and
invalid-fixture-fails pair.

**Four mutations were not run** (an ambiguous or absent anchor, not a coverage
result): N01, N09, N10, N11. No conclusion is claimed from them.

### 10.4 Two dead branches found, deliberately left in place

`_AUXILIARY_LEAD` and `_figure_in_claim` are both unreachable behind earlier
branches. `_AUXILIARY_LEAD` is redundant defence in depth and its words are
already caught by `_ENTITY_STOPWORDS` or a resource stopword list; removing it
changed no test. Neither was removed: this phase changes no product logic, and
neither is a proven defect. No test claims to cover either.

### 10.5 Exact remaining uncertainty

1. The original `selftest_entities.py` is probably still recoverable from a
   pre-09:27 session transcript, which was not read (see §10.1).
2. Eight of nine test files have no git history, so "original" is asserted only
   for `selftest.py`.
3. The reconstructed fixtures agree with the current implementation because
   mutation testing showed the tests *detect* deviations — not because they were
   proven to match the lost originals, which no longer exist to compare against.
4. Four number mutations (N01, N09, N10, N11) were not run.
5. No provider was executed, so §5 stands unchanged: every model-backed
   capability is `UNTESTED` and no count in this document qualifies one.
