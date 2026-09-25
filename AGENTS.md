# AGENTS.md — yt-report

Nearest repo instructions. These override anything further up.

## What this repo is

A Claude Code skill (`SKILL.md` + `references/` + `scripts/`), packaged as a product. The working tree IS the source of truth; the installed copy at `~/.claude/skills/yt-report/` is meant to be a symlink into it, never an independently edited duplicate.

## Before you edit

1. Read `SKILL.md` end to end. It is the behavioural contract, and the references and scripts all serve it.
2. Read the relevant file in `references/`. They are loaded on demand for a reason — a genre rule or a fallback path that isn't in context is one you will contradict.
3. Read the script you are about to change, top to bottom. Each carries a docstring explaining a specific past failure it exists to prevent. That failure is the reason the code is shaped the way it is.

## Invariants

- **The self-test is the proof.** Run `uv run scripts/selftest.py` before every commit. It is offline, fast, and exists because real regressions here were silent.
- **New logic gets a new check.** Non-trivial logic in `scripts/` lands with a case in `selftest.py` that fails if the logic breaks. Not a comment, not a docstring — a runnable assertion.
- **Checkers stay high-precision, low-recall.** `verify_report.py` flags only defects identifiable without judgement and reports the genuinely ambiguous one as INFO. If you add a check that can cry wolf, you have made the whole tool worse.
- **Never evaluate or fact-check.** The output is the creator's own written record. No commentary on whether they were right, no third-person framing, no verdict.
- **Every figure, name and product they mention gets carried through.** A dropped number is a defect, not an editorial choice.
- **First person, their voice.** The test is whether the speaker would recognise it as their own pre-recording notes.

## Style

Match the file you are in. These are ordinary Python scripts and prose markdown; no frameworks, no abstraction layers, no new dependencies without a reason stated in the diff.

## Committing

Run the self-test, then commit. Commit messages: imperative, one line, no body unless the "why" is not obvious from the subject.
