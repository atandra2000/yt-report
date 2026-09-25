# yt-report

A Claude Code skill that turns a YouTube video into **the creator's own written notes** — the overview and the stories as the speaker would have written them before recording, in their own voice, carrying every figure, name and product they mention.

Covers podcast, interview, lecture and AI-news videos. Produces a first-person recap, not a third-person report. Never evaluates or fact-checks the content.

## What ships here

```
SKILL.md                     the skill itself — the contract an agent follows
references/genres.md         per-genre writing guidance (podcast, interview, lecture, AI news)
references/browser-fallback.md   what to do when YouTube blocks or the API breaks
scripts/yt_fetch.py          pull transcript, metadata and frames into a folder
scripts/label_sheet.py       tile sampled frames into labelled contact sheets
scripts/resheet.py           rebuild contact sheets from frames already on disk
scripts/verify_report.py     mechanical checks on a finished REPORT.md
scripts/selftest.py          offline self-test — no network, no ASR, no YouTube
```

## Install

The live copy an agent loads lives in `~/.claude/skills/yt-report/`. Point this repo at it so edits here are the edits there:

```bash
rm -rf ~/.claude/skills/yt-report
ln -s "$PWD" ~/.claude/skills/yt-report
```

## Requirements

- `uv` — the scripts are PEP-723 inline-script executables, so `uv run <script>` resolves their own dependencies. No project-level install step.
- `ffmpeg` 7+ — frame extraction and contact-sheet tiling.
- Optional: `mlx_whisper` (Apple Silicon) for the `--whisper` fallback when a video has no captions.

## Quick start

```bash
# fetch a video: transcript.md, meta.json, URL.txt, frames/, contact sheets
uv run scripts/yt_fetch.py "https://www.youtube.com/watch?v=..." --out /tmp/video

# no captions? opt into local ASR
uv run scripts/yt_fetch.py "https://www.youtube.com/watch?v=..." --out /tmp/video --whisper

# sheets went missing or came out empty? rebuild from frames on disk, no re-download
uv run scripts/resheet.py /tmp/video

# check a finished REPORT.md against its own artifact directory
python3 scripts/verify_report.py /tmp/video/REPORT.md
```

## Development

Run the offline self-test before every commit. It needs no network and no API keys:

```bash
uv run scripts/selftest.py
```

It exists because two real regressions were silent — a run that produced zero contact sheets while reporting `"status": "ok"`, and a batch refactor that passed a list where a string was required. Neither surfaced downstream. It exercises the pure functions and the ffmpeg plumbing on a synthetic video so that class of break fails loudly.

Editing rules:
- Edit files here, not in `~/.claude/skills/` — that's a symlink to this repo.
- Every non-trivial logic change needs a check in `selftest.py`. That check is the runnable proof, not a comment.
- Keep `verify_report.py` high-precision and low-recall. A checker that cries wolf trains people to ignore it; flag only defects identifiable without judgement, and report the ambiguous one as INFO.

## Layout note

This is a product repo, not a wrapper: the source of truth for the skill is this working tree. `scripts/` is executable, `references/` is loaded on demand, and the two `SKILL.md` frontmatter fields (`name`, `description`) are the only contract the host reads before deciding to load the skill at all.
