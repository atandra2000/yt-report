# AGENTS.md — yt-report

Nearest repo instructions. These override anything further up.

## What this repo is

A Claude Code skill (`SKILL.md` + `references/` + `scripts/`), packaged as a product. The working tree IS the source of truth; the installed copy at `~/.claude/skills/yt-report/` is meant to be a symlink into it, never an independently edited duplicate.

## Before you edit

1. Read `SKILL.md` end to end. It is the behavioural contract, and the references and scripts all serve it.
2. Read the relevant file in `references/`. They are loaded on demand for a reason — a genre rule or a fallback path that isn't in context is one you will contradict.
3. Read the script you are about to change, top to bottom. Each carries a docstring explaining a specific past failure it exists to prevent. That failure is the reason the code is shaped the way it is.
4. The pipeline is a chain of gates, and each one blocks. `evidence.json` must validate before a report exists; the report is rendered in memory and validated before `REPORT.md` is written. Do not add a path that skips one.

## Invariants

- **The self-test is the proof.** Run `uv run scripts/selftest.py` before every commit. It is offline, fast, and exists because real regressions here were silent.
- **New logic gets a new check.** Non-trivial logic in `scripts/` lands with a case in `selftest.py` that fails if the logic breaks. Not a comment, not a docstring — a runnable assertion.
- **Checkers stay high-precision, low-recall.** `evidence_validate.py` and `report_validate.py` flag only defects identifiable without judgement; the genuinely ambiguous findings are metrics, not failures. If you add a check that can cry wolf, you have made the whole tool worse.
- **Never evaluate or fact-check.** The report records the material. No commentary on whether a claim was right, no verdict, no section whose purpose is to make up a reader's mind.
- **Neutral third person.** Not first-person imitation of the speaker. The old voice contract banned the words "on screen" and "the transcript reads", which is why the reports that obeyed it most carried *zero* provenance markers — a rule that suppresses the apparatus a checker verifies has disarmed the checker, not tightened the writing.
- **The report is generated, not written.** `report.py` is a renderer over `evidence.json` and has no free-text slot, so it cannot introduce a fact the evidence does not carry. Do not add one, and do not hand-edit `REPORT.md` to add a section the generator left out — fix the generator or the evidence.
- **Every figure, name and product they mention gets carried through.** A dropped number is a defect, not an editorial choice. A *noise* figure — an unlabelled digit off a progress bar — is not, and `report.py` drops those on a rule you can read.
- **Correct abstention beats unsupported detail.** A figure that could not be read stays unreadable. A conflict between two sources stays open. Both are successes.

## Style

Match the file you are in. These are ordinary Python scripts and prose markdown; no frameworks, no abstraction layers, no new dependencies without a reason stated in the diff.

## Committing

Run the self-test, then commit. Commit messages: imperative, one line, no body unless the "why" is not obvious from the subject.
