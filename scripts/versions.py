"""Every version this pipeline answers to, in one file.

The problem this solves is not tidiness. Each version used to live in the module
that used it, and that made the one obligation attached to it — *remember to bump
me when you change the thing I version* — a memory obligation held by whoever
last edited a regex. It was missed three times in one session, and every miss had
the same shape: a rule changed, the cache was warm, and a document came back
built under the old rules with nothing in the run to say a cache was involved.
The comment in `semantic.py` records two of those incidents in full; they are
silent failures, which is the only kind that survives to a finished report.

So the versions are here, together, where reading one file tells you what is
being tracked and why. The rule for using them:

    **Changing extraction logic means bumping the version that owns it.**

Which version owns which change:

* `EXTRACTION_VERSION` — anything that changes what `semantic.py` produces from a
  segment: item ids, item shapes, classification, the subject rules. This is the
  one most often forgotten, because the change usually reads as a small local fix
  rather than as a change to the extraction contract.
* `SCHEMA_VERSION` — the shape of `evidence.json` itself. Bumped when a field is
  added, removed or re-typed, so an old artifact cannot be read as a new one.
* `PROMPT_VERSION` — the instructions. There are two, one per model layer, and
  they are separate numbers on purpose: a reader comparing two runs needs to know
  which moved, and one shared counter would answer "something" to a question that
  is asking "what".
* `SYNTH_SCHEMA_VERSION` — the JSON schema the synthesis model is held to, which
  is a different thing from the prompt that describes it: a model can be asked
  the same question in a new shape without the question changing.

**Not** derived by hashing the source tree. That would remove the obligation
entirely, and it would do it by making every comment edit invalidate every cached
segment in the repository — turning a correctness mechanism into a cost that
presses people into `--no-cache` runs, which is how the guarantee actually gets
lost. An explicit bump is one edit and one commit; a tree hash is a silent,
wholesale invalidation whose only visible symptom is that everything got slower
for no stated reason.
"""
from __future__ import annotations

#: The whole pipeline, for artifacts that record which build produced them.
#: Not part of any cache key: it moves for releases, and a release does not
#: change what any component would have answered.
PIPELINE_VERSION = 3

#: What `semantic.py` produces from a segment. See the module docstring.
#:
#: 3 added the language-aware ordinary-word filter (a capitalised German noun is
#: indistinguishable from a product name to a capitalisation test), the
#: `semantic_role` classification, and the generic `entity` type.
#: 4 lowered the unit-without-a-label case from `medium` to `low`.
#: 5 moved the table-and-topic rule behind the unit rule, and completed the
#:   interface vocabulary from the observation layer's own list. The reordering
#:   matters more than it looks: the topic is a property of the *segment*, so a
#:   dashboard column of unrelated percentages sitting in a segment that happened
#:   to discuss numbers had every one of them promoted to a high-importance
#:   benchmark.
#: 6 replaced the nearest-word subject rule with a referent gate. A spoken `100%`
#:   in "this ultimately reaches 100%" was filed under `ultimately`, which is
#:   the exact silent-wrong-answer shape the version exists to prevent — and the
#:   fix changes what a segment produces, so a warm cache would have served the
#:   old subjects indefinitely.
#: 7 taught the table test that a bare figure in a cell is a value. `find_values`
#:   refuses a unitless number on its own — on a screen it is more often a list
#:   index — which left a benchmark table of bare decimals invisible to the
#:   table-plus-topic rule, so the ordinary shape of a results table never
#:   qualified. Added the code-editor gutter exclusion at the same time, because
#:   an editor is also a grid of a number and prose and a screen-recording of a
#:   benchmark discussion is a segment about benchmarks. The subject gate gained
#:   two rules in the same pass: a single word directly after a personal pronoun
#:   is that pronoun's predicate (`we saw 42 mentions` filed the figure under
#:   `saw`), and a currency written as a word is a unit (`500 dollars` under
#:   `dollars`) — which is what `_clean_referent` already claimed to do. The
#:   referent gate now consults the shipped stopword resources, so the same
#:   question ("is this just a word here?") has one answer per language instead
#:   of an English-only one; without it `Die Latenz stieg auf` and `Wir sahen`
#:   were German subjects. Entity typing was narrowed to the words beside a name
#:   (a cue forty-eight characters away typed names it had nothing to do with),
#:   a first-person-plural Romance verb at a sentence start stopped being a
#:   name, and the stopword resources gained the time words, everyday nouns and
#:   category nouns that were arriving as candidates.
#: 9 changed which frames exist, so changed what a segment produces. The
#:   sampler moved from one global budget split between a uniform grid and a
#:   scene filter — each pass capped independently, so a run could write twice
#:   what it was asked for — to a planned three-layer set whose total is bounded
#:   by construction. On an 18-minute maths animation the old grid was 30.3s
#:   apart, which left 67 of 286 snippets with no frame to align to at all. The
#:   extraction version moves because a warm segment cache would otherwise serve
#:   items derived from the old frame set, and nothing in the document would say
#:   the frames had changed.
#:
#: 8 reads the structure a frame shows, not only the unit a figure carries. A
#:   real evaluation of a maths video found `28 × 28 = 784` legible on a frame and
#:   absent from evidence, while the digits that did survive were a matrix
#:   subscript per item: `find_values` requires a unit or a currency, so a run of
#:   unitless figures joined by an operator parsed to nothing at all — and the
#:   transcript's own grammar was worse than nothing on it, reading `28 × 28` as
#:   the figure `28` carrying the unit `×` and dropping the rest of the equation.
#:   Two shapes are now read at the boundary: a *formula*, which must start with
#:   a figure and an operator (`+ Wo, 1 a1`, a misread subscript, must not), and a
#:   figure whose referent is written *after* it (`784 inputs`, `784 weights per
#:   neuron`) — `label_for_value` reads only to the left, which is why the node
#:   counts on a network diagram were unrecorded. Each figure of a formula is its
#:   own item and carries the formula verbatim, so the figures and the
#:   relationship both survive without this layer asserting an equation it cannot
#:   check. The label gate lost the glyphs: an em-dash, a bullet, a single letter
#:   with a subscript digit and anything under three characters are not referents,
#:   and each was arriving as the subject of a high-importance figure.
EXTRACTION_VERSION = 9

#: Which frames get sampled, and how densely. Separate from `EXTRACTION_VERSION`
#: because the two answer different questions and a run that changes one does not
#: necessarily change the other — but a report built on the wrong frame set is
#: wrong in exactly the same way, so the stamp goes into `frames.json` and a
#: frame set from another policy is re-extracted rather than reused.
#:
#: 2 replaced one global frame budget spent on a uniform grid and a scene
#: filter. Each ffmpeg pass was capped separately, so the run could write twice
#: the budget it was given, and the uniform grid was `duration / budget` — one
#: frame every 30s on an 18-minute maths animation, so a +/-4s alignment window
#: contained a sampled frame 3% of the time. Measured, not guessed: 187 of 286
#: snippets had no frame, 117 of them genuinely speech-only and 67 a frame that
#: was never sampled. Widening the window would have "fixed" that by pairing a
#: sentence with a frame half a minute away, which is a fabricated proximity.
#:
#: 3 added a pre-sampling *text* probe — a dozen small frames, read once and
#: discarded — and with it the two profiles a scene score cannot reach. A screen
#: recording, a code editor and a slide deck all read as "static" to a cut
#: detector, because none of them cuts; v2 gave all three the same 12-second
#: spacing and let the change layer try to make up the difference. v3 reads the
#: words first and samples `screen_ui` at 8s and `code` at 6s. The baseline grid
#: still comes from duration and profile alone, so coverage is unchanged: what
#: moved is the spacing, and only for videos that turned out to be dense.
SAMPLING_VERSION = 3


#: The shape of `evidence.json`. Imported from `evidence_model`, which is where
#: the parser lives, and re-exported here so every version is readable in one
#: place without creating an import cycle.
from evidence_model import SCHEMA_VERSION  # noqa: E402

#: The visual pass's instruction, and the synthesis writer's. Two counters, two
#: model layers; see the module docstring for why they are not one.
VLM_PROMPT_VERSION = 2
#: 3 stated the one-home rule the report gate enforces, after a real call was
#   refused for citing one item in both a takeaway and a section. The instruction
#   and the gate now name the same constraint, which is the condition/message
#:    agreement a validator needs and a prompt needs for the same reason.
SYNTH_PROMPT_VERSION = 3

#: The JSON schema the synthesis model is held to.
SYNTH_SCHEMA_VERSION = 1

__all__ = [
    "PIPELINE_VERSION", "EXTRACTION_VERSION", "SAMPLING_VERSION", "SCHEMA_VERSION",
    "VLM_PROMPT_VERSION", "SYNTH_PROMPT_VERSION", "SYNTH_SCHEMA_VERSION",
]
