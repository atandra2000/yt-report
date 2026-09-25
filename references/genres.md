# Genre shapes

The bones are the same for every video. What changes is how the creator's notes are organised.

## No chapters at all
The uploader supplied no structure, so the creator's own structure is the topic changes themselves. Find them by watching for subject shifts rather than for headings, and let each become a section. A single continuous talk still has sections — an unstructured transcript is not a reason to write one undifferentiated block.

## News roundup
**One section per story, in the order given.** Do not merge two stories into one section however thin the second is — a merged pair is how a story gets dropped without anyone noticing. Each section leads with what the story *is*, then the numbers, then the creator's own take. The creator's newsletter, community and benchmark links go at the end, where a real creator puts them.

## Interview / podcast
Organise **by topic, not chronologically** — that is the value a reader gets from notes they could not get from the recording. The host's setup questions can go unless they establish what an answer is about. Where the creator dates a claim ("in 2022 I thought…"), keep the date: it is part of the claim, and dropping it changes what was said.

## Multi-speaker panel
The pipeline runs single-channel ASR with **no diarization**, so the transcript gives words and nothing about whose mouth they came from. Attribute to a named person **only** where an on-screen lower-third names them at that moment; open the frame, the name is small text. Everything else stays unattributed. Never infer a speaker from topic, position, or voice.

## Visual explainer
The visuals carry the argument. Describe what is depicted and what it demonstrates; do not manufacture quotations from diagrams that contain no words. A diagram of a mechanism is worth a paragraph of description, and no number attached to it unless the number is there.

**Subtitles test.** Some videos burn subtitles in, so every frame carries text that is the transcript rendered — quoting it adds nothing and reading it as a separate finding is circular. `grep` a distinctive phrase against `transcript.md`: if it appears nowhere in the transcript, it is genuinely additional material worth keeping; if it does, it is just the same words drawn large.

## Screencast / terminal / notebook
Transcribe commands, paths and error strings **exactly** — these are the parts a reader cannot reconstruct. A partial command line quoted correctly beats a complete one invented. Where most of the session is code too dense or too fast to capture, say what it does rather than pretending to have it line by line.

## Lecture
The section order should follow the **argument**, not the clock, because the argument is what the creator would have written down. The Terms table earns its keep here more than anywhere else.

## Sponsored or partner section
Keep it, as its own section, at the creator's own level of enthusiasm. A creator who reads an ad warmly writes it up warmly. What is not allowed is the report's own verdict on the ad — no "this is the weakest section", no reading the sponsorship's claims for the reader.

## Explainer with no numbers
Do not manufacture a figures section. The substance is the mechanism; write about that.
