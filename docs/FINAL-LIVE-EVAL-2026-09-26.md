# FINAL LIVE EVAL — 2026-09-26

Release qualification for `yt-report`. Offline suite re-run after the three
blockers were fixed; live provider qualification attempted and **stopped at Gate
A** on an account-balance failure.

**Top line: the deterministic pipeline is verified. No model-backed capability
is qualified. The product remains CONDITIONAL.**

---

## 1. OFFLINE VERIFIED

Full deterministic suite, no credential, no network.

| | |
|---|---|
| Total checks | **2666** |
| Failures | **0** |
| Skipped | 0 |
| Exit code | 0 |
| Test files exercised | 10 |
| `python -m compileall scripts/` | clean |

**Test files:** `selftest.py` (the runner: pure functions, cache, alignment,
semantic extraction, skill-prose invariants), `selftest_cache.py`,
`selftest_entities.py`, `selftest_numbers.py`, `selftest_ocr_perf.py`,
`selftest_grounding.py`, `selftest_report.py`, `selftest_synthesis.py`,
`selftest_vlm_provider.py`, `selftest_doctor.py`.

**Changed test counts.** Baseline before the fixes: 2626 checks, 0 failures —
identical to the figure in `HARDENING-2026-09-26.md` §10, so the baseline is
confirmed rather than assumed. After: **2666**, i.e. **+40**, from three new
test functions in `selftest_doctor.py`:

| Test | Checks | What it pins |
|---|---:|---|
| `t_vlm_doctor_survives_an_unclassified_exception` | 11 | an unclassified provider exception becomes a structured `UNTESTED` verdict, not a traceback; the provider's own classifications are unchanged; `release-check` survives it in-process |
| `t_max_vlm_frames_reaches_the_doctor` | 13 | the requested frame cap reaches the doctor up to the documented maximum, and not past it |
| `t_connectivity_and_end_to_end_reach_the_verdict` | 16 | a failing `connectivity` or `end-to-end` moves the top-level verdict; `UNTESTED` and `SKIPPED` do not |

No existing check was removed, skipped, or weakened. One existing check was
strengthened: `t_release_check_budget_is_a_ceiling` already collected
`cfg.max_frames` into a list and asserted nothing about it — the variable was
dead. It is now asserted, which is what exposed the propagation defect in §3.2.

**Confirmation that no network or provider call occurred.** Verified, not
asserted. The suite was re-run under a `sys.addaudithook` recording every
`socket.connect`, `socket.getaddrinfo`, `socket.sendto` and `urllib.Request`:

```
EGRESS EVENTS: 0
```

The first run of that instrumented suite returned **4 egress events**, all to
`localhost:11434` — a local Ollama endpoint. That was a real defect and it is
fixed; see §3.4. The second run returned zero. A blocked or refused connection
costs nothing and would not appear as egress, so this also confirms no paid
request was attempted.

---

## 2. LIVE QUALIFICATION — STOPPED AT GATE A

### Gate A — VLM doctor: **UNTESTED** (no credit)

| | |
|---|---|
| Provider | `openrouter` (the only configured provider) |
| Model | `openai/gpt-4o-mini` |
| Command | `VLM_PROVIDER=openrouter VLM_MODEL=openai/gpt-4o-mini yt-report vlm-doctor --json` |
| Latency | **965 ms** |
| Response mode | none — no observation was produced |
| Calls made | 1 |
| Cost | $0 (rejected at the account-balance check, before generation) |

Every doctor check, as the doctor recorded it:

| Check | Verdict | Detail |
|---|---|---|
| `reachable` | **ok** | `openrouter answered in 965 ms` |
| `authenticated` | **ok** | the credential was accepted |
| `model_exists` | **ok** | `openai/gpt-4o-mini resolved` |
| `vision_capable` | **FAIL** | `HTTP 402: … requires more credits … can only afford 948` |
| `structured_requested` | unreachable | — |
| `structured_enforced` | unreachable | — |
| `test_observation_valid` | unreachable | — |

Three of the seven steps held, and they are the three that cost nothing to hold.
The run then stopped at the first step that requires a bill.

**What this is not.** Transport HTTP 200 is not claimed as a pass — `reachable`
records only that the endpoint answered, and the doctor says so in its own text.
A prompt-only response is not claimed as an enforced-schema pass:
`structured_enforced` is **unreachable**, not true. No structured response was
produced, so nested schema validation was never exercised and is not reported as
passing.

**Why `UNTESTED` and not `FAILED`.** The 402 left `observation: null` and no
`result` or `cases` in the report, so the module's own rule — *did an answer come
back* — answers no. Fed the live report, its own pure functions give:

```
qualified            : False
answered()           : False
classify()           : UNTESTED
http_failure_class   : no-credit
matrix vision column : UNTESTED
```

A balance is not a capability. Nothing about this provider was disproven, and
`FAIL` is the state that must never be used for "we did not get that far".

**Root cause, confirmed independently.** The OpenRouter account is out of
credit, not merely short on this one model:

```
GET https://openrouter.ai/api/v1/credits
  total_credits : 46
  total_usage   : 46.197155895
  remaining     : -0.197155895
```

The balance is **negative**. The provider's own message ("You requested up to
2000 tokens, but can only afford 948") is the same fact seen from the other
side. No model change reaches this: the rejection is against the requested
`max_tokens` against the balance, so a cheaper model would not have helped.

**A second, independent blocker was found and not worked around.** A full
`release-check` against the configured provider was attempted to record the
aggregated verdict. It was **denied by the permission classifier** as a
paid-provider call requiring explicit authorisation. That denial is correct and
it was not circumvented. The verdict in §2.1 was instead computed by feeding the
live doctor report to `release_check`'s own `classify`, `answered`,
`_http_failure_class` and `qualification_matrix` — pure functions, no network,
no spend. It is the module's verdict, not a re-derivation of it.

### Gate B — synthesis doctor: **NOT RUN**

Not run. Gate B is gated on Gate A, and Gate A is `UNTESTED`. Running it would
have meant spending against a negative balance to learn nothing that the balance
does not already say.

### Gate C — critic doctor: **NOT RUN**

Not run, for the same reason.

### End-to-end: **NOT RUN**

No video was fetched. Part 4 is gated on all three provider gates, none of which
passed. One video was not spent; zero were.

---

## 3. Blockers closed before qualification

### 3.1 `vlm-doctor` unguarded exception path

`provider.analyze` was called with no `try` and `vlm_doctor.main` had no guard,
while the two synthesis-side doctors had both. The provider classifies the
*expected* egress failures into the VLM vocabulary and returns an observation
carrying the reason; anything outside that vocabulary — an unwrapped
`ssl.SSLError`, an `OSError` writing the probe image — escaped as a traceback,
and via `release-check`'s in-process call it took the whole five-step matrix
down.

Fixed at both layers. The distinction the report rests on is preserved:

- **Expected provider failures** are still classified by the provider, still
  distinguish auth from model from vision, and are unchanged. A 401 still breaks
  at `authenticated` with `stage: auth`.
- **Unclassified exceptions** now break at `reachable` with `stage: transport`,
  the exception named in the detail, the downstream checks marked `unreachable`
  by the existing cascade, `report.error` set, and exit code **3** — the
  documented code for a run that never reached a call.
- The chain verdict (`qualified: false`, `stage`) tells the two apart; the
  release-level verdict does not collapse them, because a refused credential is
  a configuration nobody gave the run and must never be filed as a broken
  provider. Both are `UNTESTED` at the matrix, for the same reason and by the
  same rule.

`main` additionally guards the `run_doctor` call, so a defect in the chain or in
anything it calls that it does not own still leaves a report and an exit code
rather than a stack.

### 3.2 `release-check --max-vlm-frames`

Two defects, not one.

**The clamp.** `budget_frames = max(1, min(max_frames, DEFAULT_MAX_VLM_FRAMES))`
silently reduced any request to 2. Removed. `resolve_vlm_frame_ceiling()` now
holds the rule, with the ceiling named:

- `MAX_VLM_FRAMES_CEILING = VL.DEFAULT_MAX_VLM_FRAMES` (8) — the cap the VLM
  pass already enforces on itself, and the value every document in the
  repository quotes as the maximum. Named rather than inlined, because an
  inlined literal is what made the defect invisible.
- Above the maximum, the request is reduced to it **and the reduction is visible**
  in `budget.max_vlm_frames` and in `--help`. It is not refused: the ceiling is a
  budget, and a budget the caller cannot raise is not under the caller's control.
  Nothing here raises what the VLM pass would itself have capped.
- Below 1, it resolves to 1: the VLM step here is `vlm-doctor`, which probes
  exactly one image whatever the ceiling says. A ceiling of zero would not make
  it cheaper, it would make the reported spend smaller than the run. "Send no
  frame at all" is a real mode and `eval-live` owns it.
- Negative is refused with a `ValueError`, consistent with
  `--max-revision-calls`.

**The propagation.** The flag reached the reported budget and nothing else: the
doctors read `cfg.max_frames`, and `run_release_check` never wrote the resolved
value there. So a library caller passing `max_vlm_frames=8` on a config built
with the default would run four. `run_release_check` now applies the ceiling to
the config — the object every consumer reads from — before any step runs. This
is the half the flag never had: a control that reached the report and not the
request, which is why `t_release_check_budget_is_a_ceiling` passed against it.

**The ceiling is one pool, not one per step.** Two steps send frames, and each
was handed the whole ceiling, so a run with a URL reported up to twice the number
printed beside it while every individual line stayed true. The doctor sends one
frame and is charged one frame — its actual spend, not the plan — and the live
run draws what is left. `spend.frames <= budget.max_vlm_frames` now holds for
every ceiling, which is what makes it testable.

### 3.3 `connectivity` / `end-to-end` release aggregation

Both steps were run, printed, and could fail without the verdict moving: a full
live run rejected by the report validator, with every capability column reading
`PASS`, still published `QUALIFIED` and exited 0. A row whose failure changes
nothing is a row nobody reads.

- `overall_status(matrix, steps)` now consults the matrix **and** the two steps
  it has no column for. A `FAIL` in either takes the verdict to `UNQUALIFIED`;
  `UNTESTED` and `SKIPPED` leave it exactly where it was. The cascade is
  `classify`'s own, not a second one: `FAIL` is an answer that came back and did
  not hold, `UNTESTED`/`SKIPPED` are questions never reached.
- `_connectivity` no longer derives its own state from a three-branch expression
  that merely agreed with `classify` by coincidence. `_reachability_state` is
  the same rule written once.
- **`end-to-end` no longer reads an exit code.** `eval_live` exits non-zero for a
  fetch that failed, an artifact directory that was never written, and an
  artifact with no captions — none of which is anything a provider did. Reading
  the code made all three `FAIL`. It now reads the run's own `status`: `ok` →
  `PASS`; `failed`/`raised` → `UNTESTED`, because nothing was exercised;
  `evidence-invalid`/`report-rejected` → `FAIL`, because the run completed and
  the pipeline's own gates rejected what it produced. To make that structured
  result available in-process, `eval_live.main` gained a `result_out` parameter
  that appends the result dict — a parameter rather than a module global, which
  would leak one run's answer into the next.

### 3.4 A defect found while proving the offline claim

Not in the brief. Found by instrumenting the offline suite rather than reading
its docstring, and fixed because PART 2 asks for confirmation that no provider
call occurred, which could not honestly be given while it stood.

`t_vlm_smoke_test`'s last check — "the key is not in the smoke report" — sat
*after* the `finally:` block that restored the real `provider_from_config`. It
therefore built a real `OllamaProvider` with no transport and opened a socket to
`localhost:11434`. It passed, because nothing was listening and a refused
connection is indistinguishable from the refusal the assertion expected. On a
machine with Ollama running it would have made a real paid call and asserted
whatever that model returned. Moved inside the stubbed block, where it tests what
its neighbours test.

The suite's "no network" claim is now true, and verified by the audit hook
rather than asserted in prose.

---

## 4. Remaining known limitations

Unchanged by this phase, carried forward:

1. **The 631.6 s OCR regression is unexplained.** Not reproducible. The
   instrumentation will attribute it if it recurs.
2. **`screen_ui` and `code` are not real-world validated.** Rendered fixtures
   show the paths discriminate; no real screen or code video exists here.
3. **16×16 dedupe cannot see a one-line text change** (29/218 measured). The
   run-tail rule bounds the damage; removing it needs a content-aware signal that
   is an architecture change.
4. **Two evidence-integrity issues:** `detail.expression` is unvalidated by any
   check, and one definition item's citation does not cover the clause it quotes.
5. **No real provider qualification has been achieved.** VLM, synthesis and
   critic are all `UNTESTED`. That is the gate, and it is still shut.
6. **The VLM pass has no `max_tokens` knob.** `max_tokens` is a constructor
   argument of `OpenAICompatProvider` and is not reachable from `VLMConfig` or
   from any CLI flag, so a provider with a small balance cannot be qualified at
   a lower token ceiling. This did not cause the Gate A failure — the balance is
   negative — but it is why the failure surfaced as a hard 402 rather than as a
   cheaper successful call. Plumbing it is a small, additive change and was
   **not** made: it is outside the freeze, and this phase is not the place for
   it.
7. **Multiplicative cost accounting inside a single run was not audited.** The
   per-step frame-ceiling multiplication (§3.2) was fixed at the release-check
   layer; whether `build_evidence.py` and `eval_live.py` share any other ceiling
   across passes was not re-derived this phase.

---

## 5. Exact commands used

```bash
# baseline, before the fixes
env -u VLM_PROVIDER -u SYNTH_PROVIDER -u VLM_API_KEY -u SYNTH_API_KEY \
    -u ANTHROPIC_API_KEY -u OPENAI_API_KEY PATH="$PWD/bin:$PATH" \
    yt-report selftest                       # 2626 checks, 0 failures

# the three fixes
#   scripts/vlm_doctor.py   — guard in run_doctor and in main
#   scripts/release_check.py — resolve_vlm_frame_ceiling, _reachability_state,
#                             overall_status, UNCOVERED_BY_MATRIX, end-to-end
#                             status mapping, ceiling on the config
#   scripts/eval_live.py     — main(..., result_out=None)
#   scripts/selftest_doctor.py, selftest_vlm_provider.py — new and moved tests

# PART 2 — full offline suite, twice
env -u VLM_PROVIDER -u SYNTH_PROVIDER -u VLM_API_KEY -u SYNTH_API_KEY \
    -u ANTHROPIC_API_KEY -u OPENAI_API_KEY PATH="$PWD/bin:$PATH" \
    yt-report selftest                       # 2666 checks, 0 failures, exit 0

uv run python -m compileall -q scripts/      # clean

# the same suite under an egress-recording audit hook
# (sys.addaudithook over socket.connect / socket.getaddrinfo /
#  socket.sendto / urllib.Request; prints a stack per hit)
#   first run : EGRESS EVENTS: 4  (all localhost:11434 — see §3.4)
#   after fix : EGRESS EVENTS: 0

# PART 3 — Gate A, the one live call
VLM_PROVIDER=openrouter VLM_MODEL=openai/gpt-4o-mini \
    yt-report vlm-doctor --json
#   → reachable/authenticated/model_exists ok; vision_capable FAIL (HTTP 402);
#     observation: null; latency 0.965 s; exit 1

# credit state, read-only, no generation
GET https://openrouter.ai/api/v1/credits
#   total_credits 46, total_usage 46.197155895, remaining -0.197155895

# NOT RUN — no URL was ever passed, so no video was fetched:
#   yt-report release-check <url> …
#   yt-report synthesis-doctor
#   yt-report critic-doctor
#   yt-report eval-live <url>
```

---

## 6. Token and cost usage

| Item | Value |
|---|---|
| Paid model calls attempted | 1 |
| Paid model calls completed | 0 |
| Cost incurred | **$0.00** |
| Prompt / completion tokens billed | 0 — rejected at the balance check, before generation |
| Read-only account calls | 1 (`/api/v1/credits`), free |

The one live request was rejected by OpenRouter's pre-generation balance check
because the requested `max_tokens` (2000) exceeded what the account could afford
(948 tokens). Nothing was generated and nothing was billed.

---

## 7. Verdict

| Capability | State |
|---|---|
| **Offline suite** | **VERIFIED** — 2666 checks, 0 failures, 0 egress events |
| **VLM** | **UNTESTED** — reachable, authenticated, model resolved; stopped by an exhausted account balance (HTTP 402) |
| **Synthesis** | **UNTESTED** — not run; gated behind VLM |
| **Critic** | **UNTESTED** — not run; gated behind synthesis |
| **End-to-end** | **UNTESTED** — not run; no video fetched |
| **Overall** | **CONDITIONAL** |

`UNTESTED` is not hidden behind `CONDITIONAL` here: CONDITIONAL is the release
state the module publishes when a capability is unexercised, and the three
capability states above are stated separately and explicitly.

**The product is not ready to leave the gated phase.** The three offline blockers
are closed and the deterministic pipeline is verified. What remains is not a
code problem: it is a funded provider account. Restoring credit on the
OpenRouter account is the single action that unblocks Gate A, and Gate A is the
single action that unblocks everything else. The order is unchanged and should
not be reordered: `vlm-doctor` → `synthesis-doctor` → `critic-doctor` → one
video, stopping at the first failure.
