"""The model that writes prose from evidence, and nothing else.

This module holds the *shape* of a synthesis — the JSON schema the model is held
to, the instruction it is given, and the parse that refuses anything else. It
does not hold the pipeline. `synthesis.py` owns the order of operations; this
owns what a valid answer looks like.

The one idea worth stating plainly: the model is a *rearranger*, not an author.
Everything it is allowed to do — compress, group, connect, explain, drop noise —
is a function of the evidence it was handed. It is never handed a frame, a
transcript, or a path, so it cannot read one; and it is held to a schema that
makes every factual sentence carry the ids of the items it rests on, so a
sentence that cannot name its own support does not survive to be rendered.

Two failure modes are designed against specifically:

* **The confident extension.** A model that sees "model = Qwen3-8B, latency =
  100ms" will happily write "Qwen3-8B is the fastest model available." Nothing in
  a JSON schema prevents that sentence. What prevents it is the validator, which
  re-derives the sentence's content words against the cited items and refuses the
  ones that are not there — and the fact that the whole report is written in
  memory and validated before any of it reaches disk.
* **The dropped qualifier.** "could cost $500" rendered as "costs $500" passes
  every check in this repo and is still a fabrication. So hedging is compared
  item-by-item, not by keyword: an item whose status is `hedged` must be cited
  by a sentence that hedges.

Structured output is requested, then *verified* — a 200 with prose in it is a
failure, not a success. And there is no compatibility path that maps a guessed
key onto a wanted one. A model that will not answer in this shape stays
unqualified, and the deterministic renderer takes over.
"""
from __future__ import annotations

import json

import chat_json as CJ

#: Bumped whenever the instruction or the schema changes in a way that would
#: produce a different draft from the same evidence. It is part of the cache key,
#: so a stale entry can never be read back as if it were current. Held in
#: `versions` with the rest and re-exported under the names the cache key uses.
#:
#: The schema version is separate because the two move for different reasons: a
#: model can be asked the same question in a new shape, and a reader comparing
#: two runs needs to know which of the two changed.
from versions import (  # noqa: E402
    SYNTH_PROMPT_VERSION as PROMPT_VERSION,
    SYNTH_SCHEMA_VERSION,
)

#: How a sentence relates to the evidence behind it. Carried through to the
#: renderer so a claim, an observation and an inference do not read identically.
#: Declared before the schema because the schema enumerates it.
SOURCE_KINDS = frozenset({"claim", "observation", "inference", "mixed"})

#: The whole contract. `additionalProperties: false` throughout: an endpoint that
#: honours `strict` will reject anything else, and one that does not should be
#: told no rather than left free to add fields nothing validates.
SYNTH_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "takeaways", "sections"],
    "properties": {
        "summary": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/unit"},
        },
        "takeaways": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {"$ref": "#/$defs/unit"},
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "paragraphs"],
                "properties": {
                    "title": {"type": "string"},
                    "paragraphs": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"$ref": "#/$defs/paragraph"},
                    },
                },
            },
        },
    },
    "$defs": {
        "unit": {
            "type": "object",
            "additionalProperties": False,
            # `source_kind` is required, not optional. A strict-mode endpoint
            # rejects a schema whose `required` list does not cover every
            # property, and the VLM schema proved that the hard way — so it is
            # required here, which also means every sentence says whether it is
            # a claim, an observation or an inference.
            "required": ["text", "evidence_ids", "source_kind"],
            "properties": {
                "text": {"type": "string"},
                "evidence_ids": {"type": "array", "minItems": 1,
                                 "items": {"type": "string"}},
                "source_kind": {"type": "string", "enum": sorted(SOURCE_KINDS)},
            },
        },
        "paragraph": {
            "type": "object",
            "additionalProperties": False,
            "required": ["text", "evidence_ids", "source_kind"],
            "properties": {
                "text": {"type": "string"},
                "evidence_ids": {"type": "array", "minItems": 1,
                                 "items": {"type": "string"}},
                "source_kind": {"type": "string", "enum": sorted(SOURCE_KINDS)},
            },
        },
    },
}

#: The critic speaks a different schema — it reports problems, it does not write
#: prose. Kept separate so a critic cannot be mistaken for a writer, and so a
#: "critic" response can never be rendered as report text.
CRITIC_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["findings"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["code", "where", "message"],
                "properties": {
                    "code": {"type": "string"},
                    "where": {"type": "string"},
                    "message": {"type": "string"},
                },
            },
        },
        "clean": {"type": "boolean"},
    },
}

_INSTRUCTION = """\
You write a report about one video, from a set of extracted evidence items.

You are a rearranger, not an author. Everything you write must come from the
evidence you were given. You may compress, group, connect, explain and drop
low-value noise. You may not add a fact, a number, a name, a cause, a motive or
a conclusion that is not in the evidence.

Rules:

1. Every sentence you write carries the ids of the evidence it rests on, in
   `evidence_ids`. A sentence you cannot support with ids is a sentence you do
   not write. One or more ids per sentence is enough; you are not expected to
   cite per word.
2. Do not extend a claim. If the evidence says a model is presented as the
   faster option in a displayed comparison, you may write that it is presented
   as the faster option. You may not write that it is the fastest model
   available, or that it is faster, period.
3. Keep every qualifier. "could cost $500" stays "could cost $500". If an
   evidence item is hedged, the sentence citing it must be hedged too. A dropped
   hedge is a fabrication even though every word in it appears in the evidence.
4. Keep uncertainty. If the evidence records that something is unclear, say it
   is unclear. Do not fill a gap with what you happen to know.
5. Keep conflicts. When two items disagree, say they disagree and give both
   sides. Do not pick a winner, do not average them, and do not quietly drop
   one.
6. Distinguish what was said from what was seen. Use `source_kind`:
   "claim" for something the speaker asserted, "observation" for something read
   off a frame, "inference" for a relationship you are drawing between cited
   items, "mixed" for both in one sentence.
7. Numbers, names, URLs and code identifiers are copied exactly. Do not
   reformat, convert or translate them.
8. Write prose, not a list. Combine related evidence into sentences that read
   like an analyst wrote them. Do not begin sentences with "The evidence shows"
   or "Item itm-01 states". Do not repeat yourself.
9. The three parts have different jobs and must not repeat each other. The
   summary states the video's central message and nothing else. The takeaways
   are the handful of points worth carrying away, each stated once. The sections
   are where the detail lives: a section must go *beyond* what the summary said
   about its segment — the mechanism, the caveats, the numbers and what they mean
   — and must not restate a summary sentence in other words. Writing the same
   sentence in the summary and again in a section is the single most common way
   to produce a report that reads as though it says half as much as it does.
10. Prefer saying a thing once, well, to saying it twice.
11. If the visual evidence is ambiguous, say it is ambiguous. Describe what is
   legible and stop there.
12. Do not write about evidence you were not given, and do not mention item ids
    in the prose itself.

Return only the JSON object described by the schema.
"""


def _system(instruction: str, required: list[str], extra: str = "") -> str:
    return (instruction + extra
            + "\n\nAnswer with a single JSON object and nothing else: no prose "
              "before or after, no markdown fence, no commentary.\n"
              "Required top-level keys: " + ", ".join(f'"{k}"' for k in required) + ".\n")


class SynthesisResult:
    """One model answer, or the classified reason there was not one.

    `doc` is the parsed object when `ok`; `mode` is the structured-output
    constraint that was actually in force, and `enforced` says whether the
    payload carried the schema. Those are recorded rather than assumed, because
    "the model was told to return JSON" and "the endpoint enforced this JSON
    schema" are different claims and a reader comparing two runs needs to know
    which one happened.
    """

    def __init__(self, doc: dict | None, mode: str | None = None,
                 enforced: bool | None = None, error: str | None = None,
                 status: str = "ok", latency_s: float | None = None):
        self.doc = doc
        self.mode = mode
        self.enforced = enforced
        self.error = error
        self.status = status
        self.latency_s = latency_s

    @property
    def ok(self) -> bool:
        return self.error is None and isinstance(self.doc, dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "status": self.status, "mode": self.mode,
                "enforced": self.enforced, "error": self.error,
                "latency_s": round(self.latency_s, 3) if self.latency_s else None,
                "prompt_version": PROMPT_VERSION,
                "synth_schema_version": SYNTH_SCHEMA_VERSION}


class SynthesisClient:
    """A text model, held to a schema, with the credential confined to a header.

    Deliberately not a vision provider and not a general LLM wrapper: it takes
    messages, asks for JSON, and returns either a schema-valid object or a
    classified refusal. Everything the report needs from a model is in that
    sentence, and a client that could do more is a client whose failure modes
    have to be reasoned about separately.
    """

    def __init__(self, model: str, base_url: str, api_key: str = "",
                 timeout: float = 120.0, transport=None):
        self._chat = CJ.StructuredChat(
            model=model, base_url=base_url, api_key=api_key, timeout=timeout,
            max_tokens=8000, schema_name="report_draft",
            schema=SYNTH_SCHEMA, required=SYNTH_SCHEMA["required"],
            system=_system(_INSTRUCTION, SYNTH_SCHEMA["required"]), transport=transport)
        self._critic = CJ.StructuredChat(
            model=model, base_url=base_url, api_key=api_key, timeout=timeout,
            max_tokens=4000, schema_name="critique",
            schema=CRITIC_SCHEMA, required=CRITIC_SCHEMA["required"],
            system=_system(
                "You check a draft report against the evidence it claims to rest on. "
                "You do not write prose. Report only what is actually wrong.",
                CRITIC_SCHEMA["required"]), transport=transport)

    @property
    def structured_mode(self) -> str | None:
        return self._chat.structured_mode

    @property
    def schema_enforced(self) -> bool | None:
        return self._chat.schema_enforced

    def describe(self) -> dict:
        """What this client is, for the artifact. Never the credential."""
        d = self._chat.describe()
        d["prompt_version"] = PROMPT_VERSION
        d["synth_schema_version"] = SYNTH_SCHEMA_VERSION
        return d

    def draft(self, prompt: str, max_tokens: int | None = None) -> SynthesisResult:
        """One draft, or a classified failure. Never a coerced object."""
        import time

        started = time.monotonic()
        try:
            doc, mode = self._chat.complete_json([{"role": "user", "content": prompt}],
                                                 max_tokens=max_tokens)
        except CJ.ChatError as exc:
            return SynthesisResult(None, self._chat.structured_mode,
                                   self._chat.schema_enforced, exc.reason, exc.status,
                                   time.monotonic() - started)
        return SynthesisResult(doc, mode, self._chat.schema_enforced, None, "ok",
                               time.monotonic() - started)

    def critique(self, prompt: str, max_tokens: int | None = None) -> SynthesisResult:
        """Findings about a draft. Advisory: the validator decides, not this.

        `max_tokens` is the same knob the draft call takes, for the same reason:
        a probe that asks for a full report's budget gets a billing refusal on
        an account that could have afforded the probe.
        """
        return self._critic_run(prompt, max_tokens)

    def _critic_run(self, prompt: str, max_tokens: int | None = None) -> SynthesisResult:
        import time

        started = time.monotonic()
        try:
            doc, mode = self._critic.complete_json([{"role": "user", "content": prompt}],
                                                   max_tokens=max_tokens)
        except CJ.ChatError as exc:
            return SynthesisResult(None, self._critic.structured_mode,
                                   self._critic.schema_enforced, exc.reason, exc.status,
                                   time.monotonic() - started)
        return SynthesisResult(doc, mode, self._critic.schema_enforced, None, "ok",
                               time.monotonic() - started)


def validate_draft(doc: dict) -> list[str]:
    """Structural checks on a parsed draft. Returns a list of problems.

    Strict, and deliberately not forgiving. A model that returned the wrong
    *shape* has not made a small mistake — it has not read the contract, and
    every downstream stage assumes this one held. Problems are reported by
    position so a repair request can quote them back.
    """
    problems: list[str] = []

    if not isinstance(doc, dict):
        return [f"draft is {type(doc).__name__}, not an object"]
    for key in SYNTH_SCHEMA["required"]:
        if key not in doc:
            problems.append(f"missing required key {key!r}")
    for key in doc:
        if key not in SYNTH_SCHEMA["properties"]:
            problems.append(f"unexpected key {key!r} at the top level")

    for key in ("summary", "takeaways"):
        units = doc.get(key)
        if not isinstance(units, list):
            if key in doc:
                problems.append(f"{key!r} is {type(units).__name__}, not an array")
            continue
        for i, u in enumerate(units):
            problems += _check_unit(u, f"{key}[{i}]")

    sections = doc.get("sections")
    if not isinstance(sections, list):
        if "sections" in doc:
            problems.append(f"'sections' is {type(sections).__name__}, not an array")
    else:
        for i, s in enumerate(sections):
            where = f"sections[{i}]"
            if not isinstance(s, dict):
                problems.append(f"{where} is {type(s).__name__}, not an object")
                continue
            for key in ("title", "paragraphs"):
                if key not in s:
                    problems.append(f"{where} is missing {key!r}")
            if not isinstance(s.get("title"), str) or not s.get("title", "").strip():
                problems.append(f"{where} has an empty or non-string title")
            paras = s.get("paragraphs")
            if not isinstance(paras, list):
                if "paragraphs" in s:
                    problems.append(f"{where}.paragraphs is {type(paras).__name__}, not an array")
                continue
            if not paras:
                # A section with no paragraphs is a heading over nothing, which
                # renders as a claim that the video had something there. The
                # schema says minItems 1 for exactly this reason, and this is
                # where a non-enforcing endpoint's answer gets caught.
                problems.append(f"{where} has no paragraphs; it would render as an empty heading")
            for j, p in enumerate(paras):
                problems += _check_unit(p, f"{where}.paragraphs[{j}]")
    return problems


def _check_unit(unit, where: str) -> list[str]:
    problems: list[str] = []
    if not isinstance(unit, dict):
        return [f"{where} is {type(unit).__name__}, not an object"]
    for key in ("text", "evidence_ids"):
        if key not in unit:
            problems.append(f"{where} is missing {key!r}")
    text = unit.get("text")
    if not isinstance(text, str) or not text.strip():
        problems.append(f"{where} has an empty or non-string 'text'")
    ids = unit.get("evidence_ids")
    if not isinstance(ids, list) or not ids:
        problems.append(f"{where} has no 'evidence_ids'; an unsupportable sentence")
        return problems
    for eid in ids:
        if not isinstance(eid, str) or not eid.strip():
            problems.append(f"{where} has a non-string evidence id")
    if "source_kind" not in unit:
        problems.append(f"{where} is missing 'source_kind'; a sentence must say whether "
                        f"it is a claim, an observation or an inference")
    elif unit.get("source_kind") not in SOURCE_KINDS:
        problems.append(f"{where} has source_kind {unit.get('source_kind')!r}, which is not "
                        f"one of {sorted(SOURCE_KINDS)}")
    for key in unit:
        if key not in ("text", "evidence_ids", "source_kind"):
            problems.append(f"{where} has unexpected key {key!r}")
    return problems


def validate_critique(doc: dict) -> list[str]:
    """Structural checks on a parsed critique. Returns a list of problems.

    The same contract as `validate_draft`, for the same reason: a critique in
    the wrong shape cannot be read, and reading it anyway means deciding what a
    field called `type` was supposed to mean. That decision is the one this repo
    refuses to make anywhere else.

    It exists because the negotiator's `conforms_to_schema` only tests the
    schema's top-level `required` keys — `findings` was present, so the response
    counted as conformant, and a live run returned `{type, evidence_id,
    description}` where the schema asks for `{code, where, message}`. Every
    finding then scored as malformed, and a critic that had genuinely found two
    real problems was reported as one that had found nothing it could identify.

    A critique that fails here is silence, not a block: the draft has already
    passed the blocking checks, and an unreadable advisory opinion is not a
    reason to refuse a grounded report.
    """
    problems: list[str] = []
    if not isinstance(doc, dict):
        return [f"critique is {type(doc).__name__}, not an object"]
    for key in CRITIC_SCHEMA["required"]:
        if key not in doc:
            problems.append(f"missing required key {key!r}")
    for key in doc:
        if key not in CRITIC_SCHEMA["properties"]:
            problems.append(f"unexpected key {key!r} at the top level")
    findings = doc.get("findings")
    if not isinstance(findings, list):
        if "findings" in doc:
            problems.append(f"'findings' is {type(findings).__name__}, not an array")
        return problems
    item_keys = CRITIC_SCHEMA["properties"]["findings"]["items"]
    for i, f in enumerate(findings):
        where = f"findings[{i}]"
        if not isinstance(f, dict):
            problems.append(f"{where} is {type(f).__name__}, not an object")
            continue
        for key in item_keys["required"]:
            if key not in f:
                problems.append(f"{where} is missing {key!r}")
            elif not isinstance(f[key], str) or not f[key].strip():
                problems.append(f"{where}.{key} is empty or not a string")
        for key in f:
            if key not in item_keys["properties"]:
                problems.append(f"{where} has unexpected key {key!r}")
    if "clean" in doc and not isinstance(doc["clean"], bool):
        problems.append("'clean' is present but is not a boolean")
    return problems


def units(doc: dict):
    """Every `(where, text, evidence_ids, source_kind)` in a draft, in order.

    The one traversal every downstream check shares, so the validator, the
    critic's prompt and the renderer all agree on what a "unit" is and where it
    lives. Three separate traversals would eventually disagree about whether the
    summary counts, and a check that covers 90% of the units is worse than one
    that is explicit about covering all of them.
    """
    for i, u in enumerate(doc.get("summary") or []):
        if isinstance(u, dict):
            yield f"summary[{i}]", u
    for i, u in enumerate(doc.get("takeaways") or []):
        if isinstance(u, dict):
            yield f"takeaways[{i}]", u
    for i, s in enumerate(doc.get("sections") or []):
        if not isinstance(s, dict):
            continue
        for j, p in enumerate(s.get("paragraphs") or []):
            if isinstance(p, dict):
                yield f"sections[{i}].paragraphs[{j}]", p


def draft_to_json(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=2)
