"""Vision-language pass: structured understanding of a frame, marked as model-derived.

Sibling of `ocr.py` and deliberately not a replacement for it. OCR reads the
characters; a VLM says what the *arrangement* means — that a frame is a benchmark
table, that `32 tok/s` belongs to the row labelled Model A, that a bar is absent.
That is exactly the "no bar, no number" judgement the text layer cannot make, and
it is also where fabrication enters most easily, so the two are kept apart end to
end: separate provenance source, separate confidence band, separate status, and
validator rules that refuse a `vlm` citation on any frame this pass did not
actually look at.

Six decisions are load-bearing.

**Not every frame goes to the model.** `select_frames` picks a small queried set
from signals already on disk — text density, structure, numerics, proximity to
speech — under a hard budget. A talking head with no text scores zero and is
never sent, because the model would return "a person talking", which is not
evidence and costs a call. A selection policy that needed a model call to decide
which frames deserve model calls could not be re-derived when someone asks why
this frame cost money and that one did not.

**The model is given a bounded window, and told it is context.** `build_context`
carries the video title, the segment, a few seconds of nearby transcript, the
OCR read and the aligned snippet ids. Not the transcript: a frame of a table
means nothing without knowing what was being discussed, and means everything
wrong if handed the whole video. The model is given no authority over the
context — the instruction states that nothing outside the frame and the supplied
context may be asserted — and its self-reported `confidence` is capped, never
promoted to `verified_verbatim`, and never averaged with an OCR score.

**Structure, not prose.** The output schema is a tool definition the provider is
required to call, and every field is a structure a later check can read: a number
carries its row, column and label; a relationship names both ends; an
uncertainty is a field rather than a sentence's hedge. A caption cannot be
checked; a table row can. `from_dict` is total — a malformed or
schema-violating response becomes `status: "malformed"`, never an exception,
because a provider returning nonsense is a fact about the run, not a crash.

**Abstention is a first-class result.** `uncertainties` and
`status: "no_visual"` are how "I cannot read this" is recorded. A frame where
the axis is illegible is stored as partially understood with the legibility
stated, and `semantic.py` refuses to promote an uncertain value into a semantic
number. Correct abstention beats an invented figure, because an invented figure
inherits authority in the claims table, the summary and the conclusion — three
places that each repeat it as though it had been checked.

**The cache key is the question, not the answer.** It covers the frame's content
hash, the provider, the model, the model version, the prompt version, the schema
version and the context hash. Change any of them and the entry is a miss. A
cache that omits the last four does not fail loudly — it returns a plausible
answer to a question that is no longer being asked. Failures are never cached at
all: a provider outage stored as a valid empty observation is a permanent lie.

**Nothing pretends.** With no credentials and no provider named, the pass
returns `UnavailableProvider` and every frame carries `vlm_status:
"unavailable"`. The pipeline stays functional, the evidence document stays
valid, and no visual observation is invented to fill the gap.

Two provider families, one interface. `AnthropicProvider` speaks the Messages
API and forces a tool call. `OpenAICompatProvider` speaks `/chat/completions`,
which is the shape OpenRouter, Ollama, LM Studio, llama.cpp and most hosted
gateways publish, and which is therefore the cheapest way to reach a provider
nobody has written a class for. No provider is named in `semantic.py` or in the
extraction path: `get_provider` is the only door, and it takes a `VLMConfig`
assembled from the environment and the CLI. A new endpoint is a base URL, not a
code path.

**Every response is untrusted, and the same gate covers all of them.** A
response is parsed, then handed to `VisionObservation.from_dict`, which is total
and total means it returns a record — a malformed one when the shape is wrong —
rather than raising. There is no branch where a provider's answer reaches the
evidence document without passing through that function, which is why "reject
malformed output" is a property of the shape rather than a promise in prose.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from cache import JsonCache, content_key
import chat_json as CJ
from evidence_model import SCHEMA_VERSION, VisualFrame

#: Bumping this invalidates every cached analysis. It is part of the cache key
#: precisely so that changing what the model is *asked* cannot serve an answer
#: produced under the older question. Held in `versions` with the rest and
#: re-exported under this name, which is the one the cache key and its tests use.
from versions import VLM_PROMPT_VERSION as PROMPT_VERSION  # noqa: E402

#: Ceiling on a model-reported observation. Deliberately below the OCR bands: a
#: description is of pixels, and a description that is wrong is wrong in ways no
#: confidence score can bound. Held where it is; §17 of the phase brief asked for
#: justification before moving it and nothing here justifies moving it.
MAX_MODEL_CONFIDENCE = 0.70

#: What the frame *is*, as judged by the model. A model-derived classification,
#: never ground truth: it is stored beside the observation it came from, and the
#: validator checks it against this vocabulary rather than against the pixels.
CONTENT_TYPES = frozenset({
    "talking_head", "slide", "code", "terminal", "browser", "application_ui",
    "table", "chart", "diagram", "document", "product_screen", "photo",
    "mixed", "unknown",
})

#: Conservative default. Twelve frames on a 7-minute video is already a real bill;
#: the budget exists so that a longer video cannot quietly scale the spend with
#: its runtime, and so the number of calls is a decision rather than a function of
#: frame count.
DEFAULT_MAX_VLM_FRAMES = 8

#: Bounds on the context window. A frame is interpreted against its neighbourhood,
#: not against the video.
CONTEXT_WINDOW_S = 30.0
CONTEXT_MAX_TRANSCRIPT_CHARS = 900
CONTEXT_MAX_OCR_CHARS = 700
CONTEXT_MAX_EVIDENCE_ITEMS = 8

_INSTRUCTION = """You are a visual evidence extractor for a video report. You are given one frame, \
the text OCR read from that same frame, and a short window of the nearby transcript.

Describe only what is visible in this frame. Use the context to work out what the visual content is \
about, but never assert anything that is neither visible in the frame nor present in the context. \
The OCR text is a reading of the pixels, not a description: where you disagree with it, say so in \
`uncertainties` rather than silently replacing it, and do not reproduce an OCR reading you cannot \
see in the frame yourself.

Content-specific rules:
- table: put every cell in `numbers` with its `row` and `column` labels attached. Do not flatten the \
table into a list of values.
- chart: record the title, axis labels, legend and series in `visual_elements`. Extract exact numbers \
only where they are legible. Relative ordering and trends may go in `relationships`. If the axis is \
unreadable, say so in `uncertainties` and extract no numeric values from it.
- diagram: record every arrow and connection in `relationships` with both endpoint names. If the \
direction of an arrow is ambiguous, set `certain` to false rather than guessing.
- code or terminal: record the language, filenames, commands and configuration keys you can actually \
read. Do not reconstruct a file from a fragment, and never repair unreadable code silently.
- product or application: name it only when something visible supports the name — a logo you can read, \
a title, a URL. Appearance alone is not identification.

If the frame carries nothing beyond the person speaking, return content_type "talking_head" and an \
empty structure. If you cannot read part of the frame, record it in `uncertainties` instead of \
guessing. Correct abstention is a better answer than a plausible invention."""

#: The structured output contract, as a tool schema. Providers that support
#: structured output pass this verbatim; the field names are the same ones
#: `VisionObservation.from_dict` reads back, so the schema is the single place a
#: shape is declared rather than being described in prose and reimplemented.
#: shape is declared rather than being described in prose and reimplemented.
OBSERVATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "One or two sentences on what this frame shows."
        },
        "content_type": {
            "type": "string",
            "enum": ["application_ui", "browser", "chart", "code", "diagram", "document", "mixed", "photo", "product_screen", "slide", "table", "talking_head", "terminal", "unknown"]
        },
        "visual_elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string"
                    },
                    "label": {
                        "type": "string"
                    },
                    "detail": {
                        "type": "string"
                    }
                },
                "required": ["detail", "kind", "label"],
                "additionalProperties": False
            }
        },
        "text_relationships": {
            "type": "array",
            "description": "How a piece of on-screen text relates to the visual structure.",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string"
                    },
                    "role": {
                        "type": "string",
                        "description": "title | row_label | column_label | axis_label | legend | value | body"
                    },
                    "target": {
                        "type": "string"
                    }
                },
                "required": ["role", "target", "text"],
                "additionalProperties": False
            }
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string"
                    },
                    "kind": {
                        "type": "string",
                        "description": "product | model | tool | organization | person | other"
                    },
                    "confidence": {
                        "type": "number"
                    }
                },
                "required": ["confidence", "kind", "text"],
                "additionalProperties": False
            }
        },
        "numbers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string"
                    },
                    "unit": {
                        "type": "string"
                    },
                    "label": {
                        "type": "string"
                    },
                    "row": {
                        "type": "string"
                    },
                    "column": {
                        "type": "string"
                    },
                    "context": {
                        "type": "string"
                    },
                    "legible": {
                        "type": "boolean",
                        "description": "False when the value is inferred rather than read."
                    }
                },
                "required": ["column", "context", "label", "legible", "row", "unit", "value"],
                "additionalProperties": False
            }
        },
        "claims_supported": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "The spoken claim, quoted from the context."
                    },
                    "support": {
                        "type": "string",
                        "enum": ["supported", "not_supported", "unclear"]
                    },
                    "reason": {
                        "type": "string"
                    },
                    "confidence": {
                        "type": "number"
                    }
                },
                "required": ["claim", "confidence", "reason", "support"],
                "additionalProperties": False
            }
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string"
                    },
                    "target": {
                        "type": "string"
                    }
                },
                "required": ["description", "target"],
                "additionalProperties": False
            }
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {
                        "type": "string"
                    },
                    "to": {
                        "type": "string"
                    },
                    "label": {
                        "type": "string"
                    },
                    "kind": {
                        "type": "string",
                        "description": "arrow | ordering | contains | calls | none"
                    },
                    "certain": {
                        "type": "boolean"
                    }
                },
                "required": ["certain", "from", "kind", "label", "to"],
                "additionalProperties": False
            }
        },
        "uncertainties": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string"
                    },
                    "reason": {
                        "type": "string"
                    }
                },
                "required": ["reason", "text"],
                "additionalProperties": False
            }
        },
        "confidence": {
            "type": "number",
            "description": "Your confidence that this reading of the frame is correct."
        }
    },
    "required": ["actions", "claims_supported", "confidence", "content_type", "entities", "numbers", "relationships", "summary", "text_relationships", "uncertainties", "visual_elements"],
    "additionalProperties": False
}

TOOL_NAME = "record_observation"

_STR_LISTS = ("visual_elements", "text_relationships", "entities", "numbers",
              "claims_supported", "actions", "relationships", "uncertainties")

#: `from` is a Python keyword, so the relationship endpoints are renamed on the way
#: in and back on the way out. Renaming rather than `**` keeps a stray key from
#: surviving into the artifact, where a reader would take it as a recorded field.
_EDGE_IN = {"from": "source", "to": "target"}
_EDGE_OUT = {v: k for k, v in _EDGE_IN.items()}

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _as_text(v) -> str | None:
    if v is None or isinstance(v, (dict, list, bool)):
        return None
    s = str(v).strip()
    return s or None


def _as_conf(v) -> float | None:
    """A confidence inside `[0, MAX_MODEL_CONFIDENCE]`, or None.

    The cap is applied here rather than trusted: a model asked for a number will
    sometimes return 1.0, and a document carrying 1.0 for a model observation
    makes a description look like a measurement. Out-of-range and non-numeric
    values become None, which downstream means "the model declined to say" — not
    a silent 0.0, which would read as a considered judgement.
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return round(max(0.0, min(MAX_MODEL_CONFIDENCE, f)), 4)


def _as_bool(v) -> bool | None:
    return v if isinstance(v, bool) else None


def _entry_list(doc: dict, key: str, required: str) -> tuple[list[dict], list[str]]:
    """Normalise one list-of-objects field. Returns `(entries, dropped_reasons)`.

    Dropping is recorded, never silent: a response carrying four malformed
    entities is a fact about the response, and a reader deciding how far to trust
    the rest of the observation needs to know that four entries did not survive.
    """
    raw = doc.get(key)
    if raw is None:
        return [], []
    if not isinstance(raw, list):
        return [], [f"{key} is not a list"]
    out: list[dict] = []
    dropped: list[str] = []
    for i, e in enumerate(raw):
        if isinstance(e, str):
            e = {required: e}  # a bare string is a value with no structure attached
        if not isinstance(e, dict):
            dropped.append(f"{key}[{i}] is not an object")
            continue
        if _as_text(e.get(required)) is None:
            dropped.append(f"{key}[{i}] has no {required!r}")
            continue
        out.append(e)
    return out, dropped


def _edge(entry: dict) -> dict:
    """Relationship endpoints under names that are not Python keywords.

    `from` is what the schema asks for on the wire and `source` is what the
    report calls it, so the rename is not optional. But reading only `from` and
    then writing `None` over the top of a `source` the model *did* send drops a
    relationship that was answered — measured against a live gateway, one model
    answered `source`/`to` and every edge it produced was discarded with
    "has no 'source'" in the refusal. Both spellings are accepted; neither is
    preferred, because nothing in the evidence says which one is right.
    """
    src = entry.pop("from", None)
    if src is None:
        src = entry.pop("source", None)
    else:
        entry.pop("source", None)
    return {"source": src, "target": entry.pop("to", None), **entry}


@dataclass
class VisionObservation:
    """What one frame yielded, or why it yielded nothing.

    `summary` is a description, not a reading. It reaches the evidence document
    as `vlm_caption`, because that field means "the text this provenance entry
    rests on" — which a description does satisfy. What it deliberately does not
    get is `verified_verbatim`, which means "the source said exactly this": true
    of a character read off pixels, not of a sentence composed about them.

    The structured lists are the point of the whole pass. `numbers` keeps a
    value attached to the row and column it was seen under; `relationships`
    keeps both ends of an arrow; `uncertainties` keeps the part that could not be
    read. Flattening any of them into prose is how a table becomes an unordered
    bag of glyphs and a diagram becomes four strings.
    """

    frame_id: str
    summary: str | None = None
    status: str = "ok"
    content_type: str = "unknown"
    backend: str = "none"
    model: str | None = None
    model_version: str | None = None
    confidence: float | None = None
    visual_elements: list[dict] = field(default_factory=list)
    text_relationships: list[dict] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    numbers: list[dict] = field(default_factory=list)
    claims_supported: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    uncertainties: list[dict] = field(default_factory=list)
    notes: str | None = None
    error: str | None = None
    prompt_version: int = PROMPT_VERSION
    schema_version: int = SCHEMA_VERSION
    elapsed_ms: float | None = None
    cache: str | None = None
    observed_at: str | None = None

    # `caption` is the historical name for `summary`. The property keeps older
    # call sites and older cached payloads working; the artifact carries one name
    # only, because two fields holding the same sentence is a drift waiting to
    # happen between them.
    @property
    def caption(self) -> str | None:
        return self.summary

    def to_dict(self) -> dict:
        d: dict = {
            "frame_id": self.frame_id, "status": self.status,
            "content_type": self.content_type,
            "backend": self.backend, "model": self.model,
            "prompt_version": self.prompt_version, "schema_version": self.schema_version,
        }
        for k in ("model_version", "summary", "confidence", "notes", "error",
                  "elapsed_ms", "cache", "observed_at"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        for k in _STR_LISTS:
            v = getattr(self, k)
            if v:
                d[k] = [({_EDGE_OUT.get(key, key): val for key, val in e.items()}
                         if k == "relationships" else dict(e)) for e in v]
        return d

    def searchable_text(self) -> str:
        """Every string this observation rests on, for containment checks.

        The validator asks "is the cited text part of what the model actually
        recorded?" and the answer has to span more than the summary: a value the
        model read out of a table cell is recorded in `numbers`, not in the prose
        summary, and a citation of it is legitimate. A relationship is the same
        case one step along — the record stores its two endpoints, which *is* the
        edge, so the rendered text carries the edge too. Building the haystack
        here rather than in the validator keeps the question "what did this record
        say" in one place, and keeps it a rendering of the record rather than a
        second opinion about it.
        """
        parts: list[str] = [self.summary or ""]
        for key in _STR_LISTS:
            for e in getattr(self, key):
                if key == "relationships":
                    src, dst = e.get("source"), e.get("target")
                    if src and dst:
                        parts.append(f"{src} → {dst}")
                    if e.get("label"):
                        parts.append(str(e["label"]))
                for v in e.values():
                    if isinstance(v, str):
                        parts.append(v)
        return "\n".join(p for p in parts if p)

    @classmethod
    def from_dict(cls, doc: dict, frame_id: str | None = None) -> "VisionObservation":
        """Read any response, including a broken one. Never raises.

        A provider that returns nonsense has told us something true — that this
        frame was not understood — and turning that into an exception would lose
        the run over one bad frame. Anything structurally wrong lands in
        `status: "malformed"` with the reason in `notes`, so the frame is
        recorded as attempted-and-unusable rather than silently absent.
        """
        if not isinstance(doc, dict):
            return cls(frame_id=frame_id or "unknown", status="malformed",
                       notes=f"response is {type(doc).__name__}, not an object")
        problems: list[str] = []
        fid = _as_text(doc.get("frame_id")) or frame_id
        if fid is None:
            return cls(frame_id="unknown", status="malformed",
                       notes="response carries no frame_id and none was supplied")
        if frame_id is not None and fid != frame_id:
            # The frame this was asked about is not negotiable. A response
            # naming a different one is a model that answered about something
            # else — or that has started inventing ids — and either way the
            # caller-supplied id is the true one. Recorded as a problem rather
            # than silently overwritten, because a model reaching for a frame id
            # that does not exist is worth seeing in the artifact.
            problems.append(f"response names frame {fid!r}, but {frame_id!r} was asked about")
            fid = frame_id

        status = _as_text(doc.get("status")) or "ok"
        content_type = _as_text(doc.get("content_type")) or "unknown"
        if content_type not in CONTENT_TYPES:
            problems.append(f"content_type {content_type!r} is not in the schema vocabulary")
            content_type = "unknown"
        if status not in ("ok", "no_visual", "unavailable", "failed", "malformed", "skipped"):
            problems.append(f"status {status!r} is not a known outcome")
            status = "malformed"
        # A confidence the schema declares as a number, arriving as something
        # that is not one, is a type violation on a required field — not a model
        # declining to answer. The two are kept apart: `_as_conf` maps the first
        # to None and the second to a value, and only the second is a problem.
        # Values in the 0.70–1.0 band are still capped rather than refused, since
        # the cap is the policy and the model being confident is not a defect.
        raw_conf = doc.get("confidence")
        if raw_conf is not None and (isinstance(raw_conf, bool)
                                     or not isinstance(raw_conf, (int, float))
                                     or not 0.0 <= float(raw_conf) <= 1.0):
            problems.append(f"confidence {raw_conf!r} is not a number in [0, 1]")

        lists: dict[str, list[dict]] = {}
        # `from`/`to` are renamed before the required-key check rather than after,
        # so a relationship is judged on the fields the provider actually sent and
        # a missing endpoint is reported as a missing endpoint.
        source = dict(doc)
        if isinstance(source.get("relationships"), list):
            source["relationships"] = [_edge(dict(e)) if isinstance(e, dict) else e
                                       for e in source["relationships"]]
        for key, required in (("visual_elements", "kind"), ("text_relationships", "text"),
                              ("entities", "text"), ("numbers", "value"),
                              ("claims_supported", "claim"), ("actions", "description"),
                              ("relationships", "source"), ("uncertainties", "text")):
            entries, dropped = _entry_list(source, key, required)
            lists[key] = entries
            problems.extend(dropped)
        dict_edges = [e for e in (source.get("relationships") or []) if isinstance(e, dict)]
        lists["relationships"] = [r for r in lists["relationships"]
                                  if _as_text(r.get("target")) is not None]
        if len(lists["relationships"]) != len(dict_edges):
            problems.append("a relationship is missing its 'to' endpoint")

        for c in lists["claims_supported"]:
            if _as_text(c.get("support")) not in ("supported", "not_supported", "unclear"):
                problems.append(f"claim support {c.get('support')!r} is not supported/not_supported/unclear")
            c["confidence"] = _as_conf(c.get("confidence"))
        for n in lists["numbers"]:
            n["confidence"] = _as_conf(n.get("confidence"))
        for e in lists["entities"]:
            e["confidence"] = _as_conf(e.get("confidence"))
        for r in lists["relationships"]:
            if "certain" in r:
                r["certain"] = _as_bool(r["certain"])

        obs = cls(
            frame_id=fid,
            summary=_as_text(doc.get("summary") or doc.get("caption")),
            status=status,
            content_type=content_type,
            backend=_as_text(doc.get("backend")) or "none",
            model=_as_text(doc.get("model")),
            model_version=_as_text(doc.get("model_version")),
            confidence=_as_conf(doc.get("confidence")),
            notes=_as_text(doc.get("notes")),
            error=_as_text(doc.get("error")),
            elapsed_ms=doc.get("elapsed_ms") if isinstance(doc.get("elapsed_ms"), (int, float)) else None,
            cache=_as_text(doc.get("cache")),
            observed_at=_as_text(doc.get("observed_at")),
            **lists,
        )
        if problems:
            # A partial response is kept — the entries that parsed are real
            # observations — but the record is marked unusable so that nothing
            # downstream treats it as a clean read. Dropping the whole thing
            # would also be a lie, in the other direction: the model did say
            # these things, and whether to trust them is a later judgement.
            obs.status = "malformed"
            obs.notes = "; ".join(problems)[:500]
            obs.error = "schema violation"
        return obs


def _failed(frame_id: str, backend: str, model: str | None, status: str,
            reason: str) -> VisionObservation:
    """A frame the pass could not complete, recorded as itself.

    Kept as a constructor rather than an exception because a provider that
    timed out on frame 9 must not cost the other eleven frames their analysis,
    and because "the call failed" is a fact a later reader needs in the document
    rather than in a log nobody opens.
    """
    return VisionObservation(frame_id=frame_id, status=status, backend=backend,
                             model=model, notes=reason[:500], error=reason[:200])


class VisionLanguageProvider:
    """The interface a VLM backend implements.

    `analyze` receives the frame's path and a `context` dict, and returns a
    `VisionObservation`. Implementations are responsible for their own failure:
    degrade-to-`failed` is the contract, so a run that reaches the model and
    cannot complete still produces a complete, honest artifact.

    The class attributes are the provider's own description of itself, and they
    are recorded in the artifact rather than assumed by the caller. A run whose
    `model` or `structured_output` is not written down cannot be told apart from
    a run that used a different model, which is the difference between "the model
    was unsure" and "a different model was used and it was not".
    """

    name = "none"
    model = "none"
    model_version: str | None = None
    available = False
    #: Free-form capability names, recorded verbatim in the artifact. A reader
    #: asking "could this provider have produced a table reading?" reads this.
    capabilities: frozenset[str] = frozenset()
    #: Images accepted per request. One here; a provider that could take a
    #: contact sheet states so, and the selection policy may then change.
    max_images = 1
    #: Whether the provider can be made to answer in the observation shape rather
    #: than in prose. A provider that cannot is not trusted with the schema; it is
    #: wrapped in one that parses and validates whatever it does return.
    structured_output = False
    #: USD per image, when the provider publishes one. None rather than a guess:
    #: a fabricated cost figure in a run summary is worse than an absent one.
    cost_per_image_usd: float | None = None
    reason: str | None = None
    #: Whether a credential is needed before the provider can be used at all. A
    #: local server does not want one, and a provider that refused to run without
    #: a key it will never check would report `unavailable` for a configuration
    #: that is in fact complete.
    requires_key: bool = True

    def analyze(self, frame_path: str, frame_id: str, context: dict) -> VisionObservation:
        raise NotImplementedError

    def analyze_many(self, requests: list[tuple[str, str, dict]]) -> list[VisionObservation]:
        """Batch hook. The default is a sequential loop; a backend that can batch
        should override it rather than having its calls hidden in a loop."""
        return [self.analyze(p, fid, ctx) for p, fid, ctx in requests]

    def describe(self) -> dict:
        """The provider's own account of itself, for the artifact and the logs."""
        return {
            "provider": self.name, "model": self.model,
            "model_version": self.model_version, "available": self.available,
            "capabilities": sorted(self.capabilities), "max_images": self.max_images,
            "structured_output": self.structured_output,
            "cost_per_image_usd": self.cost_per_image_usd,
            "prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION,
            "reason": self.reason,
        }


class UnavailableProvider(VisionLanguageProvider):
    """The default. Says it did nothing, on every frame.

    Present because "no model ran" must be recorded rather than absent: a frame
    with no `vlm_status` is a frame whose visual meaning nobody has checked, and
    that is different from one a model checked and found uninformative.
    """

    name = "unavailable"
    available = False

    def __init__(self, reason: str = "no VLM backend configured"):
        self.reason = reason

    def analyze(self, frame_path: str, frame_id: str, context: dict) -> VisionObservation:
        return VisionObservation(frame_id=frame_id, status="unavailable",
                                 backend=self.name, notes=self.reason)


class AnthropicProvider(VisionLanguageProvider):
    """A real provider: one frame, one forced tool call, one structured record.

    Written against the Messages API with `urllib` rather than a vendor SDK
    because the repo runs under `uv run <script>` with no declared environment,
    and a provider that cannot be imported in that environment is a provider
    that silently degrades to "unavailable" on every machine but the author's.
    The one thing a dependency would buy here is a retry policy, and a retry
    policy that hides a 401 is worse than none.

    Failure is classified rather than collapsed, because the classifications mean
    different things to a later run: an authentication failure will not fix
    itself and stops the pass immediately; a rate limit is transient and is
    retried on the next frame; a timeout may be one slow frame; a response the
    schema rejects is a fact about that frame alone. None of them is cached.
    """

    name = "anthropic"
    default_model = "claude-sonnet-5"
    capabilities = frozenset({"chart", "table", "diagram", "code", "terminal",
                              "browser", "application_ui", "product_screen",
                              "claim_support", "uncertainty"})
    max_images = 1
    structured_output = True
    #: Class-level so a subclass that overrides `__init__` inherits the latch
    #: rather than raising `AttributeError` inside the failure path — which is
    #: the one place an exception turns a classified outage into an unexplained
    #: crash.
    _auth_failed = False

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 timeout: float = 90.0, base_url: str | None = None,
                 max_tokens: int = 2000):
        self.model = model or self.default_model
        self.model_version = self.model  # the alias IS the version until pinned
        self.api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        self.timeout = timeout
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL")
                         or "https://api.anthropic.com").rstrip("/")
        self.max_tokens = max_tokens
        self.available = bool(self.api_key)
        self.reason = None if self.available else (
            "anthropic requested but ANTHROPIC_API_KEY is not set; no call was made")
        #: Latched on the first authentication failure so a run with a bad key
        #: makes one attempt rather than one per frame. Reset only by a new
        #: provider instance, so a re-run is a fresh decision.
        self._auth_failed = False

    # -- request ------------------------------------------------------------- #

    def _messages(self, frame_path: str, context: dict) -> list[dict]:
        """One image block and one text block. Nothing else goes over the wire.

        The image is inlined rather than linked so the request is a pure function
        of the frame's bytes, which is what makes the cache key above it mean
        something. `media_type` is read from the file rather than assumed: a PNG
        sent as `image/jpeg` is a request that fails at the far end for a reason
        that has nothing to do with the picture.
        """
        try:
            raw = Path(frame_path).read_bytes()
        except OSError as exc:
            raise _ProviderError("frame unreadable", str(exc)) from exc
        suffix = Path(frame_path).suffix.lower()
        media = {".png": "image/png", ".gif": "image/gif",
                 ".webp": "image/webp"}.get(suffix, "image/jpeg")
        return [
            {"type": "image",
             "source": {"type": "base64", "media_type": media,
                        "data": base64.standard_b64encode(raw).decode("ascii")}},
            {"type": "text", "text": _context_text(context)},
        ]

    def analyze(self, frame_path: str, frame_id: str, context: dict) -> VisionObservation:
        if not self.available or self._auth_failed:
            return _failed(frame_id, self.name, self.model, "unavailable",
                           self.reason or "authentication previously failed for this run")
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # Tool choice is forced rather than offered. A model left free to
            # answer in prose returns prose, and prose is not checkable — the
            # whole reason this pass has a schema is that "a bar chart comparing
            # two models" cannot be tested against the pixels and
            # `numbers[0].row == "Model A"` can.
            "tools": [{"name": TOOL_NAME, "description": "Record what this frame shows.",
                       "input_schema": OBSERVATION_SCHEMA}],
            "tool_choice": {"type": "tool", "name": TOOL_NAME},
            "system": _INSTRUCTION,
            "messages": [{"role": "user", "content": self._messages(frame_path, context)}],
        }
        try:
            payload = self._post(body)
        except _ProviderError as exc:
            if exc.fatal:
                self._auth_failed = True
            return _failed(frame_id, self.name, self.model, exc.status, exc.reason)

        return _observation_from_response(payload, frame_id, self.name, self.model,
                                          self.model_version)

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                # The key goes in a header and nowhere else. It is never logged,
                # never written into the artifact, and never part of a cache key
                # — a cache directory is a file the user may share.
                "x-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("error", {}).get("message", "")
            except Exception:  # noqa: BLE001 — the body is best-effort context only
                pass
            if exc.code in (401, 403):
                raise _ProviderError("unavailable", f"HTTP {exc.code}: {detail or 'authentication failed'}",
                                     fatal=True) from exc
            if exc.code == 429:
                raise _ProviderError("failed", f"HTTP 429 rate limited: {detail}") from exc
            raise _ProviderError("failed", f"HTTP {exc.code}: {detail or exc.reason}") from exc
        except TimeoutError as exc:
            raise _ProviderError("failed", f"timeout after {self.timeout}s") from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                raise _ProviderError("failed", f"timeout after {self.timeout}s") from exc
            raise _ProviderError("failed", f"connection failed: {reason}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise _ProviderError("malformed", f"response was not JSON: {exc}") from exc


class _ProviderError(CJ.ChatError):
    """A classified provider failure. `fatal` means stop asking this run.

    A subclass rather than a copy: `run_vlm` and its tests branch on `.fatal`
    and `.downgrade`, both of which are `ChatError` attributes, and the shared
    classifier now returns the parent type. Re-declaring the constructor here
    would put the two definitions back in a position to drift.
    """


#: Base URLs for the endpoints worth naming. Presets, not branches: everything
#: below is one wire format, so an endpoint nobody has heard of is a `--vlm-
#: base-url` away rather than a class to write. Keyed by the provider alias the
#: CLI accepts.
#: Re-exported from `chat_json` rather than restated: the VLM pass and the
#: synthesis pass must resolve an alias to the same address, and a second copy of
#: this table is a second answer the day one of them is edited.
ENDPOINTS = CJ.ENDPOINTS

#: Providers reachable without a credential. A local server that runs on
#: `localhost:11434` has no account system, and demanding `VLM_API_KEY` for it
#: would make a complete configuration report as incomplete.
KEYLESS_PROVIDERS = CJ.KEYLESS_PROVIDERS

#: How the response was constrained, strongest first. Each step down is a real
#: loss of guarantee and is recorded on the observation, because "the model was
#: told to return JSON" and "the endpoint enforced this JSON schema" are not the
#: same claim and a reader comparing two runs needs to know which one happened.
STRUCTURED_MODES = CJ.STRUCTURED_MODES

#: Ceiling on the one automatic budget increase. A model that cannot answer the
#: schema in this many tokens will not answer it in more, and a retry loop that
#: scales without bound is a cost bug wearing a robustness hat.
MAX_TOKENS_CEILING = CJ.MAX_TOKENS_CEILING

_MODE_UNSUPPORTED = CJ._MODE_UNSUPPORTED


class OpenAICompatProvider(VisionLanguageProvider):
    """Any `/chat/completions` endpoint that accepts an image and returns JSON.

    Written against the same reasoning as the Anthropic provider — `urllib`,
    no SDK, because a provider that cannot be imported in the environment this
    repo runs in is a provider that silently degrades to "unavailable" on every
    machine but the author's — and against a harder constraint: this family has
    no single behaviour. Endpoints disagree about whether `response_format`
    exists, whether a schema is honoured, and whether the answer arrives in
    `content` or in a tool call.

    So the provider *negotiates*, once, and records the outcome. It asks for a
    strict JSON schema; if the endpoint refuses the parameter it drops to
    `json_object`; if that is refused too it falls back to a prompt that demands
    bare JSON, and the answer is parsed and schema-validated by the same
    `VisionObservation.from_dict` every other path uses. The mode that worked is
    latched on the instance, so eight frames cost one negotiation rather than
    eight, and it is written onto every observation — a run whose structured
    output was merely requested is not a run whose structured output was
    enforced, and only the artifact can say which happened.

    `transport` is injectable so the negotiation, the classification and the
    parse are all exercisable with no network and no credentials, which is the
    only way they get exercised on every run rather than on the days a key
    happens to be valid.
    """

    name = "openai"
    capabilities = frozenset({"chart", "table", "diagram", "code", "terminal",
                              "browser", "application_ui", "product_screen",
                              "claim_support", "uncertainty"})
    max_images = 1
    structured_output = True
    _auth_failed = False

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 timeout: float = 90.0, base_url: str | None = None,
                 max_tokens: int = 2000, provider_name: str = "openai",
                 requires_key: bool | None = None, transport=None):
        self.model = model or "gpt-4o-mini"
        self.model_version = self.model  # the alias IS the version until pinned
        self.base_url = (base_url or ENDPOINTS.get(provider_name)
                         or ENDPOINTS["openai"]).rstrip("/")
        self.name = provider_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.requires_key = (provider_name not in KEYLESS_PROVIDERS) if requires_key is None else requires_key
        if api_key is not None:
            self.api_key = api_key
        elif self.requires_key:
            # Per-provider lookup, no cross-provider fallback: see
            # `chat_json._KEY_VARS`. The previous `VLM_API_KEY or
            # OPENAI_API_KEY` would have sent an OpenAI credential to
            # OpenRouter whenever only the former was absent.
            self.api_key = CJ.key_for(provider_name) or ""
        else:
            self.api_key = ""
        self._transport = transport or _http_send
        #: Latched once a mode is accepted, so the negotiation is paid once.
        self.structured_mode: str | None = None
        #: Whether a returned response actually matched the schema that was sent.
        #: `None` until the first call answers. A gateway that accepts
        #: `response_format` and then ignores it is common enough that assuming
        #: enforcement would be a documented lie in every artifact.
        self.schema_enforced: bool | None = None
        self.available = bool(self.api_key) or not self.requires_key
        self.reason = None if self.available else (
            f"{self.name} requested but no API key is configured "
            "(set VLM_API_KEY); no call was made")
        self._auth_failed = False

    def describe(self) -> dict:
        d = super().describe()
        d["base_url"] = self.base_url
        d["requires_key"] = self.requires_key
        d["structured_mode"] = self.structured_mode
        d["schema_enforced"] = self.schema_enforced
        return d

    # -- request ------------------------------------------------------------- #

    def _messages(self, frame_path: str, context: dict) -> list[dict]:
        try:
            raw = Path(frame_path).read_bytes()
        except OSError as exc:
            raise _ProviderError("failed", f"frame unreadable: {exc}") from exc
        suffix = Path(frame_path).suffix.lower()
        media = {".png": "image/png", ".gif": "image/gif",
                 ".webp": "image/webp"}.get(suffix, "image/jpeg")
        # The OpenAI shape is the same block with a URL in place of the bytes.
        url = f"data:{media};base64," + base64.standard_b64encode(raw).decode("ascii")
        return [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": _context_text(context)},
        ]}]

    def analyze(self, frame_path: str, frame_id: str, context: dict) -> VisionObservation:
        if not self.available or self._auth_failed:
            return _failed(frame_id, self.name, self.model, "unavailable",
                           self.reason or "authentication previously failed for this run")
        try:
            messages = self._messages(frame_path, context)
        except _ProviderError as exc:
            return _failed(frame_id, self.name, self.model, exc.status, exc.reason)

        order = [self.structured_mode] if self.structured_mode else list(STRUCTURED_MODES)
        last: _ProviderError | None = None
        for mode in order:
            body: dict = {
                "model": self.model,
                "messages": [{"role": "system",
                              "content": _INSTRUCTION + (_JSON_ONLY_SUFFIX if mode == "prompt_only" else "")},
                             *messages],
                "max_tokens": self.max_tokens,
                "temperature": 0,
            }
            if mode == "json_schema":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": TOOL_NAME, "strict": True,
                                    "schema": OBSERVATION_SCHEMA}}
            elif mode == "json_object":
                body["response_format"] = {"type": "json_object"}
            try:
                payload = self._post(body)
            except _ProviderError as exc:
                if exc.fatal:
                    self._auth_failed = True
                    return _failed(frame_id, self.name, self.model, exc.status, exc.reason)
                if exc.downgrade and mode != STRUCTURED_MODES[-1]:
                    # The endpoint does not know this parameter. Keep the reason:
                    # "why is this run not schema-enforced" is answerable later
                    # only if the refusal was written down.
                    last = exc
                    continue
                return _failed(frame_id, self.name, self.model, exc.status, exc.reason)
            self.structured_mode = mode
            if _conforms_to_schema(payload, mode):
                self.schema_enforced = True
            else:
                # The endpoint accepted the parameter and the model ignored it.
                # Recording this as "the schema was enforced" is the silent
                # semantic fallback this whole class of problem is about: the
                # artifact would say structured output was requested, a reader
                # would read that as guaranteed, and the response would be
                # something else entirely. So it is verified, not assumed, and
                # the next call stops trusting the mode.
                self.schema_enforced = False
                # The next call steps *down* the ladder rather than back to the
                # top. Clearing the latch to None looks like "stop trusting the
                # mode" but is not: `order` is `[self.structured_mode] if
                # self.structured_mode else list(STRUCTURED_MODES)`, so a reset
                # to None restarts the negotiation at `json_schema` — and a model
                # that cannot produce a strict-schema payload never reaches
                # `json_object` or `prompt_only`, which are the two modes that
                # would have worked. Measured against a live gateway: eight
                # frames, eight identical non-conforming `json_schema` replies.
                if mode in STRUCTURED_MODES[:-1]:
                    self.structured_mode = STRUCTURED_MODES[
                        STRUCTURED_MODES.index(mode) + 1]
            obs = _observation_from_chat(payload, frame_id, self.name, self.model,
                                         self.model_version, mode)
            if _finish_reason(payload) == "length" and self.max_tokens < MAX_TOKENS_CEILING:
                # Keyed on the finish reason, not on `obs.status`. A model asked
                # for a large schema routinely answers past a small budget, and
                # the answer it was in the middle of writing is a good one —
                # but truncation arrives under two different statuses depending
                # on where the cut landed. Gating on `"failed"` caught only the
                # cut before any content and let the cut *inside* a JSON string
                # through as `malformed`, which reads as the model breaking the
                # contract when the budget broke it. Measured against a live
                # gateway: four frames, four "Unterminated string" refusals, no
                # retry, on a model that answered all four correctly at a larger
                # budget. The retry still happens at most once per provider.
                # Straight to the ceiling, not a single multiple of it. A
                # ×4 step from 2000 buys 8000, which a model writing 14k
                # characters of JSON for one frame still cannot finish, and
                # the remedy is the one the ceiling exists for.
                self.max_tokens = MAX_TOKENS_CEILING
                try:
                    bigger = self._post({**body, "max_tokens": self.max_tokens})
                except _ProviderError as exc:
                    if exc.fatal:
                        self._auth_failed = True
                    return _failed(frame_id, self.name, self.model, exc.status, exc.reason)
                retry = _observation_from_chat(bigger, frame_id, self.name, self.model,
                                               self.model_version, mode)
                if retry.status in ("ok", "no_visual") or \
                        _finish_reason(bigger) != "length":
                    return retry
            return obs

        reason = last.reason if last else "no usable response mode"
        return _failed(frame_id, self.name, self.model, "failed", reason)

    def _post(self, body: dict) -> dict:
        """Send, then classify. Nothing leaves this method unclassified.

        Delegates the transport and the HTTP classification to `chat_json`, the
        same path the synthesis pass uses. The injected `self._transport` is
        still what moves the bytes, so the negotiation, the classification and
        the parse stay exercisable with no network and no credentials — which is
        the only way they get exercised on every run rather than on the days a
        key happens to be valid.
        """
        headers = {"content-type": "application/json"}
        if self.api_key:
            # The key goes in a header and nowhere else: never logged, never
            # written into the artifact, never part of a cache key.
            headers["authorization"] = f"Bearer {self.api_key}"
        try:
            raw = self._transport(f"{self.base_url}/chat/completions", headers,
                                  json.dumps(body).encode("utf-8"), self.timeout)
        except urllib.error.HTTPError as exc:
            raise _classify_http(exc) from exc
        except TimeoutError as exc:
            raise _ProviderError("failed", f"timeout after {self.timeout}s") from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                raise _ProviderError("failed", f"timeout after {self.timeout}s") from exc
            raise _ProviderError("failed", f"connection failed: {reason}") from exc
        return _decode_chat_response(raw)


def _classify_http(exc: urllib.error.HTTPError) -> _ProviderError:
    """An HTTP status, translated into what it means for a later run.

    Delegates to the shared classifier so the VLM pass and the synthesis pass
    cannot disagree about what a 401 means. The `_ProviderError` shape is kept
    because `run_vlm` and its tests branch on `.fatal` and `.downgrade`.
    """
    err = CJ.classify_http(exc)
    return _ProviderError(err.status, err.reason, fatal=err.fatal, downgrade=err.downgrade)


def _http_send(url: str, headers: dict, data: bytes, timeout: float) -> str:
    """The default transport: send bytes, return the body. No judgement."""
    return CJ.http_send(url, headers, data, timeout)


def _decode_chat_response(raw) -> dict:
    """Bytes (or a str) to a payload dict, or a classified `malformed`.

    Shared with the synthesis client so "a body that is not a JSON object" is
    one judgement with one name. Note the two failure shapes stay distinct: not
    JSON at all, and JSON that is not an object.
    """
    try:
        payload = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as exc:
        raise _ProviderError("malformed", f"response was not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise _ProviderError("malformed", f"response is {type(payload).__name__}, not an object")
    return payload


#: The contract restated in the prompt, with the enum spelled out. Not belt-and-
#: braces: measured against a live gateway, an endpoint that accepted
#: `response_format: json_schema` returned a well-formed JSON object in a
#: completely different shape with a `content_type` outside the vocabulary. A
#: model that ignores the parameter still reads this, and the enum being written
#: out is what stops "screen_recording" arriving where "application_ui" belongs.
_JSON_ONLY_SUFFIX = (
    "\n\nAnswer with a single JSON object and nothing else: no prose before or "
    "after, no markdown fence, no commentary.\n"
    "Required top-level keys: "
    + ", ".join(f'"{k}"' for k in OBSERVATION_SCHEMA["required"]) + ".\n"
    '`content_type` must be exactly one of: '
    + ", ".join(sorted(CONTENT_TYPES)) + ".\n"
    '`summary` is one or two sentences on what the frame shows, and it is '
    "required.\n"
    "Every list holds objects, and each object must carry *all* of the keys "
    "named for it below — a key you rename or omit makes the whole entry "
    "discarded, and an entry that names one field but not the other is the "
    "failure this fallback exists to prevent:\n"
    + "\n".join(
        f"  {key}: " + ", ".join(sorted((OBSERVATION_SCHEMA['properties'][key]['items']
                                        .get('required') or [])))
        for key in sorted(OBSERVATION_SCHEMA["properties"])
        if isinstance(OBSERVATION_SCHEMA["properties"][key], dict)
        and OBSERVATION_SCHEMA["properties"][key].get("type") == "array"
        and isinstance(OBSERVATION_SCHEMA["properties"][key].get("items"), dict)
        and OBSERVATION_SCHEMA["properties"][key]["items"].get("type") == "object")
    + "\n"
    "Everything else in the structure is optional; send an empty list rather "
    "than a placeholder. If you cannot read the frame, still return the "
    "structure with an empty observation and an entry in `uncertainties`.")


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def _content_text(payload: dict) -> str | None:
    """The assistant's own text, wherever this endpoint family put it.

    A thin alias over the shared reader, kept under this name because the
    parsing helpers below and the provider tests both call it directly.
    """
    return CJ.content_text(payload)


def _conforms_to_schema(payload: dict, mode: str | None) -> bool:
    """Did the answer actually match the schema that was sent?

    Delegates to the shared check — the same one the synthesis pass uses to
    decide whether a draft is trustworthy. "Did it follow the schema?" is the
    load-bearing question in both places and gets exactly one implementation.
    """
    return CJ.conforms_to_schema(payload, mode, OBSERVATION_SCHEMA.get("required", ()))


def _finish_reason(payload) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("choices"), list) and payload["choices"]:
        first = payload["choices"][0]
        if isinstance(first, dict):
            return str(first.get("finish_reason") or "")
    return ""


def _no_text_reason(payload) -> tuple[str, str]:
    """Why a response carried nothing usable, as (status, reason).

    A `length` finish is not the same failure as silence. It means the model was
    answering correctly and the token budget ran out — which is a budget problem
    with an obvious remedy, and calling it "malformed" would put a fixable
    configuration error in the same bucket as a model that ignored the schema.
    """
    finish = _finish_reason(payload)
    if finish == "length":
        return "failed", ("the response was cut off by the token budget "
                          "(finish_reason 'length'); raise max_tokens and retry")
    return "malformed", f"response carried no assistant text (finish_reason {finish!r})"


def _observation_from_chat(payload: dict, frame_id: str, provider: str,
                           model: str | None, model_version: str | None,
                           mode: str | None) -> VisionObservation:
    """Parse a chat response into an observation, or say why it could not be.

    Three refusals, in order: no text at all, text that is not JSON, and JSON
    that is not the observation shape. The first two are `_failed` because
    nothing was learned about the frame; the third is a record kept and marked
    `malformed`, because the model *did* say things and dropping them silently
    would make the response look like it was never read.
    """
    text = _content_text(payload)
    if not text:
        return _failed(frame_id, provider, model, *_no_text_reason(payload))
    fence = _FENCE.match(text)
    if fence:
        text = fence.group(1)
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        return _failed(frame_id, provider, model, "malformed",
                       f"assistant text was not JSON: {exc} (structured mode {mode!r})")
    obs = VisionObservation.from_dict(doc, frame_id)
    obs.backend, obs.model, obs.model_version = provider, model, model_version
    obs.notes = "; ".join(x for x in [obs.notes, f"structured_output={mode}"] if x)[:500]
    if obs.status == "ok" and not obs.summary:
        return _failed(frame_id, provider, model, "malformed",
                       "observation carries no summary")
    return obs


def _observation_from_response(payload: dict, frame_id: str, provider: str,
                               model: str | None, model_version: str | None) -> VisionObservation:
    """Turn a Messages response into an observation, or say why it could not be.

    A response with no tool call is not an empty observation — it is a model that
    answered in prose when the schema required otherwise, which is the one case
    where guessing at the shape would be least defensible.
    """
    if not isinstance(payload, dict):
        return _failed(frame_id, provider, model, "malformed",
                       f"response is {type(payload).__name__}, not an object")
    for block in payload.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == TOOL_NAME:
            obs = VisionObservation.from_dict(dict(block.get("input") or {}), frame_id)
            obs.backend, obs.model, obs.model_version = provider, model, model_version
            if obs.status != "ok":
                # `from_dict` already downgraded and explained it.
                return obs
            if not obs.summary:
                return _failed(frame_id, provider, model, "malformed",
                               "observation carries no summary")
            return obs
    stop = payload.get("stop_reason")
    return _failed(frame_id, provider, model, "malformed",
                   f"no {TOOL_NAME} call in the response (stop_reason {stop!r})")


class FakeProvider(VisionLanguageProvider):
    """A deterministic stand-in, for the offline suite and for `--vlm fake`.

    It is a test double and is labelled `fake` on every observation it produces,
    so a document built with it can never be mistaken for one a real model wrote.
    It answers from a fixed map keyed by frame id, which makes the whole path —
    provider → observation → evidence → provenance → validator — exercisable with
    no network and no credentials, which is the only way that path can be tested
    on every run instead of on the days a key happens to be valid.

    A frame absent from the map returns `no_visual`, so an unlisted frame cannot
    quietly inherit a neighbour's reading.
    """

    name = "fake"
    capabilities = frozenset({"chart", "table", "diagram", "code", "terminal",
                              "browser", "application_ui", "product_screen",
                              "claim_support", "uncertainty", "malformed"})
    structured_output = True
    cost_per_image_usd = 0.0

    def __init__(self, observations: dict[str, dict] | None = None,
                 model: str = "fake-v1", default: dict | None = None):
        self.observations = dict(observations or {})
        self.default = default
        self.model = model
        self.model_version = model
        self.available = True
        #: Frames this provider was actually asked about, in order. A test
        #: asserting the budget was honoured reads this rather than trusting a
        #: counter the provider incremented about itself.
        self.calls: list[str] = []

    def analyze(self, frame_path: str, frame_id: str, context: dict) -> VisionObservation:
        self.calls.append(frame_id)
        doc = self.observations.get(frame_id, self.default)
        if doc is None:
            return VisionObservation(frame_id=frame_id, status="no_visual",
                                     backend=self.name, model=self.model,
                                     notes="no fixture for this frame")
        if isinstance(doc, str):
            # A raw string is a malformed response, which is a state the suite
            # has to be able to produce on purpose.
            return _failed(frame_id, self.name, self.model, "malformed", doc)
        obs = VisionObservation.from_dict({**doc, "frame_id": frame_id}, frame_id)
        obs.backend, obs.model, obs.model_version = self.name, self.model, self.model_version
        return obs


#: Names accepted by `--vlm`. The tuple exists so the CLI can reject a typo
#: instead of falling through to "unavailable" and looking like a missing key.
PROVIDERS = ("none", "anthropic", "openai", "openrouter", "ollama", "lmstudio",
             "llamacpp", "custom", "fake")

DEFAULT_ANTHROPIC_MODEL = AnthropicProvider.default_model


@dataclass
class VLMConfig:
    """Where the pass reads its settings from, in one place.

    Environment first, CLI second. A user who sets `VLM_PROVIDER=ollama` once and
    a script that passes `--vlm-provider openai` both end up here, and neither
    has to know the other's mechanism. The key is never echoed, never stored on
    disk and never written into an artifact; `describe()` reports whether one was
    present and nothing about it.

    `custom` is the escape hatch for an OpenAI-compatible endpoint nobody has a
    preset for: it is `openai` with no default base URL, so `--vlm-base-url` (or
    `VLM_BASE_URL`) is the whole configuration.
    """

    provider: str = "none"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 90.0
    max_frames: int = DEFAULT_MAX_VLM_FRAMES

    ENV = {"provider": "VLM_PROVIDER", "model": "VLM_MODEL",
           "base_url": "VLM_BASE_URL", "api_key": "VLM_API_KEY",
           "timeout": "VLM_TIMEOUT", "max_frames": "VLM_MAX_FRAMES"}

    @classmethod
    def from_env(cls, **overrides) -> "VLMConfig":
        """Build a config from the environment, then apply explicit overrides.

        A malformed `VLM_TIMEOUT` is ignored rather than fatal: a typo in a
        convenience variable should not stop a run that would otherwise be fine,
        and the effective value is visible in the artifact either way.
        """
        base: dict = {"provider": os.environ.get("VLM_PROVIDER") or "none"}
        for field_name, var in cls.ENV.items():
            raw = os.environ.get(var)
            if raw is None or field_name in base:
                continue
            if field_name in ("timeout", "max_frames"):
                try:
                    base[field_name] = float(raw) if field_name == "timeout" else int(raw)
                except ValueError:
                    continue
            else:
                base[field_name] = raw
        for k, v in overrides.items():
            if v is not None:
                base[k] = v
        return cls(**base)

    def to_public_dict(self) -> dict:
        """Everything except the key. This is what an artifact is allowed to see."""
        return {"provider": self.provider, "model": self.model,
                "base_url": self.base_url, "timeout": self.timeout,
                "max_frames": self.max_frames,
                "api_key_present": bool(self.api_key)}


def get_provider(prefer: str = "none", model: str | None = None,
                 api_key: str | None = None, timeout: float = 90.0,
                 observations: dict[str, dict] | None = None,
                 base_url: str | None = None) -> VisionLanguageProvider:
    """Pick a backend. `none` is the default and is not a failure.

    Deliberately not auto-selecting one: a VLM call costs money and latency, so
    running one is a decision the run makes explicitly, not a fallback that
    happens because a key was in the environment. A named provider with no
    credentials returns an `UnavailableProvider` carrying the reason, which is
    what keeps "I asked for a model and got none" distinguishable from "I asked
    for nothing" in the artifact.

    `prefer` is the only place a provider name is matched, so adding one is a
    line here and a class above it, and nothing downstream learns a name.
    """
    prefer = (prefer or "none").strip().lower()
    if prefer in ("none", ""):
        return UnavailableProvider("no VLM pass requested (--vlm none)")
    if prefer == "anthropic":
        p = AnthropicProvider(model=model, api_key=api_key, timeout=timeout,
                              base_url=base_url)
        return p if p.available else UnavailableProvider(p.reason or "anthropic unavailable")
    if prefer == "fake":
        if observations is None:
            return UnavailableProvider(
                "--vlm fake needs a fixture map; it is a test double, not a fallback")
        return FakeProvider(observations, model=model or "fake-v1")
    if prefer in ("custom", "openai", "openrouter", "ollama", "lmstudio", "llamacpp"):
        if prefer == "custom" and not base_url:
            return UnavailableProvider(
                "--vlm custom needs --vlm-base-url (or VLM_BASE_URL): it is an "
                "OpenAI-compatible endpoint with no default address")
        p = OpenAICompatProvider(model=model, api_key=api_key, timeout=timeout,
                                 base_url=base_url, provider_name=prefer)
        return p if p.available else UnavailableProvider(p.reason or f"{prefer} unavailable")
    return UnavailableProvider(f"no VLM backend named {prefer!r} is available")


def provider_from_config(cfg: VLMConfig,
                         observations: dict[str, dict] | None = None) -> VisionLanguageProvider:
    """The single door the pipeline uses. Nothing else constructs a provider."""
    return get_provider(cfg.provider, model=cfg.model, api_key=cfg.api_key,
                        timeout=cfg.timeout, observations=observations,
                        base_url=cfg.base_url)


# --------------------------------------------------------------------------- #
# frame selection
# --------------------------------------------------------------------------- #

#: A frame carrying this many distinct text regions is dense enough that its
#: meaning is likely structural (a table, a dashboard, a code listing) rather
#: than a sentence. Set from the observed distribution, not guessed: on the
#: shipped artifacts, talking-head frames carry a handful of regions and every
#: table/slide/UI frame carries dozens.
DENSE_REGION_COUNT = 25

#: Below this, a frame with no transcript alignment and no readable text is a
#: talking head. The model would return "a person talking", which is not evidence
#: and costs a call.
MIN_USEFUL_REGIONS = 3

_NUMERIC = re.compile(r"\d")
_WORDISH = re.compile(r"[A-Za-z]{3,}")

#: Markers for the visual kinds whose meaning is in their arrangement. Deliberately
#: coarse: these decide what is *worth a call*, not what the frame is. The content
#: type is the model's judgement, made from pixels, and is never inferred from
#: these strings.
_CODE_HINTS = ("def ", "import ", "class ", "function ", "const ", "return ",
               "=>", "->", "public ", "private ", "async ", "</", "();", "::",
               "publicKey", "#include", "SELECT ", "fn ", "let ", "var ")
_TERMINAL_HINTS = ("$ ", ">>>", "traceback", "npm ", "npx ", "pip ", "uv ", "python ",
                   "bash ", "zsh ", "error:", "warning:", "installing", "building")
_UI_HINTS = ("http://", "https://", "settings", "dashboard", "sidebar", "toolbar",
             "login", "sign in", "search", "button", "profile", "⌘", "ctrl+",
             "edit", "cancel", "save", "upload", "download", "install", "menu")
_CHART_HINTS = ("axis", "legend", "chart", "bar", "plot", "throughput", "latency",
                "accuracy", "benchmark", "score", "vs ", "%", "x-axis", "y-axis")
_DIAGRAM_HINTS = ("→", "↓", "←", "⇒", "->", "architecture", "pipeline", "workflow",
                  "flow", "stage", "layer")


def _marker_hits(text: str, hints) -> int:
    low = text.lower()
    return sum(1 for h in hints if h in low)


def score_frames(frames: list[VisualFrame], alignments: list) -> dict[str, dict]:
    """Score every frame for "could this need visual interpretation?".

    Returns a map of frame id → the signals that fired, so a selection can be
    explained after the fact rather than merely reproduced. Every signal is
    computed from what is already on disk, which is what makes the policy
    deterministic and free — and a policy that costs a model call to decide is a
    policy that cannot be re-derived when someone asks why this frame cost money
    and that one did not.

    The signals, in descending order of how likely they are to need a model:

    1. **Dense text.** Many distinct regions means a table, chart, terminal or
       UI, where the relationship between the words carries the meaning.
    2. **Structure markers.** Code, terminal, chart, diagram and UI vocabulary in
       the OCR read. Coarse on purpose: it routes a frame to a model that can see
       the arrangement, it does not decide what the frame is.
    3. **Numeric density.** Several distinct figures is the table-and-benchmark
       case, and it is the case where OCR returns digits with nothing attaching
       them to a row.
    4. **A usable read with no transcript support.** Text on screen that the
       speaker is not describing — the screen-only case, the whole product reason
       this pass exists.
    5. **Aligned to speech.** Some sentence pointed at it, so context exists.

    A frame that is none of these scores zero and is never sent.
    """
    aligned = {fid for a in alignments for fid in a.frame_ids}
    out: dict[str, dict] = {}
    for f in frames:
        if f.t_start is None:
            continue
        text = (f.ocr_text or "").strip()
        regions = len(f.ocr_regions or [])
        numeric_tokens = set(_NUM_RE.findall(text))
        signals: dict[str, float] = {}
        if regions >= DENSE_REGION_COUNT:
            signals["dense_text"] = 2.0 + min(regions, 120) / 100.0
        elif regions >= MIN_USEFUL_REGIONS:
            signals["text_regions"] = regions / 100.0
        if text:
            for name, hints, weight in (("code", _CODE_HINTS, 0.9), ("terminal", _TERMINAL_HINTS, 0.9),
                                        ("ui", _UI_HINTS, 0.6), ("chart", _CHART_HINTS, 0.7),
                                        ("diagram", _DIAGRAM_HINTS, 0.6)):
                hits = _marker_hits(text, hints)
                if hits:
                    signals[name] = round(0.3 + 0.1 * min(hits, 6) * weight, 4)
            if len(numeric_tokens) >= 3:
                signals["numeric_density"] = 0.4 + 0.1 * min(len(numeric_tokens), 10)
        if text and f.frame_id not in aligned:
            signals["screen_only"] = 1.0
        if f.frame_id in aligned:
            signals["aligned"] = 0.75
        if not signals:
            continue
        # A frame with a couple of stray regions, no readable text and nothing
        # aligned to it is a talking head with subtitles. Sending it buys "a
        # person talking" at full price.
        if not text and set(signals) <= {"text_regions"}:
            continue
        signals["_total"] = round(sum(signals.values()), 4)
        out[f.frame_id] = signals
    return out


def select_frames(frames: list[VisualFrame], alignments: list,
                  max_frames: int = DEFAULT_MAX_VLM_FRAMES) -> list[str]:
    """The queried set: which frames could need visual interpretation.

    A hard cap, and the cap is the point. Ordering by score alone would send
    every frame of a screen recording to a model, because every frame of a screen
    recording is dense; the budget is what makes the number of calls a decision
    rather than a function of the video's runtime.
    """
    scored = score_frames(frames, alignments)
    ordered = sorted(scored.items(), key=lambda kv: (-kv[1]["_total"], kv[0]))
    return [fid for fid, _ in ordered[:max(0, max_frames)]]


# --------------------------------------------------------------------------- #
# context window
# --------------------------------------------------------------------------- #

def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def _fmt_t(t: float | None) -> str:
    if t is None:
        return "unmeasured"
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def build_context(ev, frame: VisualFrame, alignments: list,
                  window_s: float = CONTEXT_WINDOW_S) -> dict:
    """The bounded neighbourhood a frame is interpreted against.

    A frame of a benchmark table means nothing without knowing what was being
    discussed, and means everything wrong if handed the whole transcript: the
    model would find the number it expected somewhere in twenty minutes of
    speech and attach it to this frame. So the window is bounded in *both* axes —
    a fixed number of seconds around the frame, and a character budget on every
    field — and it carries the aligned snippet ids so the model can be told which
    sentences this frame is supposed to illustrate.

    The known evidence for the segment goes in too, for the same reason the
    nearby speech does: "this frame is the benchmark I have already recorded a
    figure from" is checkable by a later reader, and the alternative is the model
    re-deriving a figure that is already in the document under a different name.
    """
    t = frame.t_start
    snippets = sorted(getattr(ev, "snippets", []) or [], key=lambda s: s.start_s)
    near = [s for s in snippets
            if t is not None and abs(s.start_s - t) <= window_s]
    aligned_ids = sorted({a.snippet_id for a in alignments
                          if frame.frame_id in (a.frame_ids or [])})
    aligned_text = [s for s in near if s.id in aligned_ids]
    speech = aligned_text or near

    segment = next((sg for sg in getattr(ev, "segments", [])
                    if t is not None and sg.start_s is not None
                    and sg.start_s <= t <= (sg.end_s if sg.end_s is not None else t)),
                   None)
    known: list[str] = []
    if segment is not None:
        by_id = {i.id: i for i in (getattr(ev, "items", []) or [])}
        for iid in segment.item_ids[:CONTEXT_MAX_EVIDENCE_ITEMS]:
            it = by_id.get(iid)
            if it is None:
                continue
            known.append(f"{it.type}: {it.display or it.value}"[:120])

    video = getattr(ev, "video", {}) or {}
    return {
        "video_title": _clip(video.get("title") or "", 200),
        "channel": _clip(video.get("channel") or "", 120),
        "frame_id": frame.frame_id,
        "timestamp_s": t,
        "timestamp": _fmt_t(t),
        "segment_title": _clip(segment.title, 200) if segment else None,
        "segment_id": segment.id if segment else None,
        "aligned_snippet_ids": aligned_ids[:20],
        "nearby_transcript": _clip(" ".join(s.text for s in speech), CONTEXT_MAX_TRANSCRIPT_CHARS),
        "ocr_text": _clip(frame.ocr_text or "", CONTEXT_MAX_OCR_CHARS),
        "ocr_status": frame.ocr_status,
        "known_evidence": known,
    }


def _context_text(context: dict) -> str:
    """The context as the model sees it: labelled, so a field cannot be confused
    with something the model read off the frame."""
    lines = [
        "VIDEO TITLE:", context.get("video_title") or "(none)",
        "CHANNEL:", context.get("channel") or "(none)",
        f"FRAME: {context.get('frame_id')} at {context.get('timestamp')}",
        "SEGMENT:", context.get("segment_title") or "(none)",
        "ALIGNED TRANSCRIPT IDS:", ", ".join(context.get("aligned_snippet_ids") or []) or "(none)",
        "NEARBY TRANSCRIPT:", context.get("nearby_transcript") or "(none)",
        "OCR READ OF THIS FRAME:", context.get("ocr_text") or "(no text was read)",
    ]
    if context.get("known_evidence"):
        lines.append("ALREADY-RECORDED EVIDENCE FOR THIS SEGMENT (may not be restated as new):")
        lines.extend(f"  - {k}" for k in context["known_evidence"])
    lines.append(
        "The transcript and the already-recorded evidence are CONTEXT, not content of this "
        "frame. Do not copy a figure from them into this frame's observation unless the figure "
        "is visible in the frame itself.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

def _cache_key(frame_path: Path, frame_id: str, context: dict,
               provider: VisionLanguageProvider) -> str:
    """Identity of the *question*, not of the answer.

    The frame's content hash makes a re-extracted video free; the provider,
    model and model version stop an old model's answer outliving it; the prompt
    version stops a changed instruction from serving a stale answer; the schema
    version does the same for the record shape; and the context hash means two
    frames of the same picture asked against different speech are different
    questions. A cache that omits any of the last four does not fail loudly — it
    returns a plausible answer to a question that is no longer being asked, which
    is the failure this whole layer exists to make impossible.
    """
    digest = ""
    try:
        h = hashlib.sha256()
        with frame_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()[:24]
    except OSError:
        digest = f"path:{frame_path.name}"
    return content_key(digest, frame_id, json.dumps(context, sort_keys=True, default=str),
                       provider.name, provider.model, getattr(provider, "model_version", None),
                       PROMPT_VERSION, SCHEMA_VERSION)


#: Outcomes that describe a *run* rather than a reading of the frame. None of them
#: is cached: storing "the provider was down at 14:02" as though it were an
#: observation of the frame converts a transient outage into a permanent claim
#: that the frame was uninformative.
_UNCACHEABLE = ("failed", "unavailable", "malformed")


def run_vlm(artifact_dir: str | Path, frames: list[VisualFrame], alignments: list,
            provider: VisionLanguageProvider | None = None,
            max_frames: int = DEFAULT_MAX_VLM_FRAMES,
            context_by_frame: dict[str, dict] | None = None,
            progress: bool = False,
            cache: JsonCache | None = None) -> dict:
    """Analyse the queried set. Returns the artifact payload; writes nothing.

    Frames not selected are not omitted from the answer — they come back with an
    explicit status, so the artifact distinguishes "not worth a call" from
    "called and found nothing".

    Nothing here can exceed `max_frames`. The budget is applied by the selector
    and re-checked here, because the two are the same promise stated in two
    places and a run that quietly spent eight hundred calls is not a run that
    miscalculated a score.
    """
    d = Path(artifact_dir)
    provider = provider or get_provider()
    cache = cache or JsonCache("vlm")
    max_frames = max(0, int(max_frames))
    scored = score_frames(frames, alignments)
    chosen = select_frames(frames, alignments, max_frames)
    chosen_set = set(chosen)
    context_by_frame = context_by_frame or {}

    results: dict[str, VisionObservation] = {}
    calls = hits = ok = failed = uncalled = 0
    started = time.perf_counter()

    for f in frames:
        if f.frame_id not in chosen_set:
            results[f.frame_id] = VisionObservation(
                frame_id=f.frame_id, status="skipped", backend=provider.name,
                notes="not selected for analysis")
            continue
        path = d / (f.file or f"frames/{f.frame_id}.jpg")
        if not path.exists():
            results[f.frame_id] = VisionObservation(
                frame_id=f.frame_id, status="unavailable", backend=provider.name,
                notes="frame file missing")
            continue
        ctx = context_by_frame.get(f.frame_id, {})
        key = _cache_key(path, f.frame_id, ctx, provider)
        cached = cache.get(key)
        if cached is not None:
            obs = VisionObservation.from_dict(cached, f.frame_id)
            obs.cache = "hit"
            results[f.frame_id] = obs
            if provider.available:
                calls += 1
                hits += 1
        else:
            t0 = time.perf_counter()
            try:
                obs = provider.analyze(str(path), f.frame_id, ctx)
            except Exception as exc:  # noqa: BLE001 — one bad frame must not end the run
                obs = _failed(f.frame_id, provider.name, provider.model, "failed",
                              f"{type(exc).__name__}: {exc}")
            obs.elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            obs.cache = "miss"
            obs.observed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if not obs.backend or obs.backend == "none":
                obs.backend = provider.name
            if obs.status not in _UNCACHEABLE:
                cache.put(key, obs.to_dict())
            results[f.frame_id] = obs
            if provider.available:
                calls += 1
        if results[f.frame_id].status == "ok":
            ok += 1
        elif results[f.frame_id].status == "skipped":
            pass
        elif provider.available:
            # A call was made and it did not yield a reading. Counted here rather
            # than as "anything that is not ok", because a frame the provider was
            # never asked about has no call that could have failed — and counting
            # it produced a summary reading `vlm_calls: 0, calls_failed: 8` for a
            # run that asked nothing, which is the most misleading sentence this
            # file can write.
            failed += 1
        else:
            uncalled += 1
        if progress:
            r = results[f.frame_id]
            print(f"  vlm {f.frame_id}: {r.status} ({r.content_type})", file=sys.stderr)

    wall = round(time.perf_counter() - started, 3)
    considered = len(scored)
    cost = None
    if provider.cost_per_image_usd is not None and calls:
        cost = round(provider.cost_per_image_usd * calls, 4)
    return {
        "version": 2,
        "backend": provider.name,
        "model": getattr(provider, "model", None),
        "model_version": getattr(provider, "model_version", None),
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "provider": provider.describe(),
        "summary": {
            "status": ("ok" if provider.available and ok else
                       ("skipped" if provider.name in ("unavailable",) and not chosen
                        else ("unavailable" if not provider.available else "empty"))),
            "backend": provider.name,
            "model": getattr(provider, "model", None),
            "model_version": getattr(provider, "model_version", None),
            "frames_total": len(frames),
            "candidates_considered": considered,
            "frames_selected": len(chosen),
            "frames_not_budgeted": max(0, considered - len(chosen)),
            "max_vlm_frames": max_frames,
            "frames_analysed": ok,
            "vlm_calls": calls,
            "calls_ok": ok,
            "calls_failed": failed,
            # Frames the provider was never asked about. Kept apart from
            # `calls_failed` so a run that asked nothing cannot read as a run
            # whose calls all failed.
            "frames_not_analysed": uncalled,
            "cache_hits": hits,
            "cache_misses": calls - hits,
            "cache_hit_rate": round(hits / calls, 4) if calls else None,
            "estimated_cost_usd": cost,
            "vlm_wall_s": wall,
            "notes": getattr(provider, "reason", None),
        },
        "frames": [results[f.frame_id].to_dict() for f in frames],
    }


def apply_observations(frames: list[VisualFrame], payload: dict) -> list[VisualFrame]:
    """Write each observation onto its frame, in place. Returns the same list.

    The frame record is the only place a reader looks to find out what a model
    said about a picture, so the structured record and the flat scalars are set
    together and from the same source. `vlm_caption` is the observation's summary
    and nothing else — a second, independently-editable copy of the model's words
    is how the validator's containment check stops meaning anything.
    """
    by_id = {str(f.get("frame_id")): f for f in (payload or {}).get("frames", [])}
    for f in frames:
        rec = by_id.get(f.frame_id)
        if not rec:
            continue
        obs = VisionObservation.from_dict(rec, f.frame_id)
        f.vlm_status = obs.status
        f.vlm_caption = obs.summary
        f.vlm_backend = obs.backend
        f.vlm_confidence = obs.confidence
        f.vlm_observation = obs.to_dict()
    return frames


def observation_texts(payload: dict) -> dict[str, str]:
    """frame id → every string the stored observation rests on.

    Read by the validator to answer "is the cited text part of what the model
    actually recorded for this frame?" — a question the flat caption alone cannot
    answer, because a value read out of a table cell is recorded in `numbers` and
    never appears in the summary.
    """
    out: dict[str, str] = {}
    for f in (payload or {}).get("frames", []):
        fid = str(f.get("frame_id"))
        if fid:
            out[fid] = VisionObservation.from_dict(f, fid).searchable_text()
    return out


def merge_into_visual(visual: dict, payload: dict) -> dict:
    """Fold the VLM record into `visual.json`, the one canonical visual artifact.

    Not into a `vlm.json`. A second file holding a second copy of what a model
    said about the same frames is a second source of truth, and the disagreement
    between the two would be invisible. `evidence.json` stays canonical for what
    is *claimed*; `visual.json` is the visual layer's own record, and the VLM's
    observations belong to it.

    The OCR payload is preserved exactly: the merge only adds a `vlm` key to
    frames that already have an entry, so `--no-ocr` runs and runs whose OCR
    failed are left alone.
    """
    if not visual:
        return visual
    by_id = {str(f.get("frame_id")): f for f in payload.get("frames", [])}
    for entry in visual.get("frames", []):
        rec = by_id.get(str(entry.get("frame_id")))
        if rec is not None:
            entry["vlm"] = rec
    # Assign, never `setdefault`. A second pass over the same artifact directory
    # — a different model, a different budget — rewrites every `vlm` block but
    # left the first run's summary in place, so `visual.json` carried a
    # `vlm_summary` naming one model beside per-frame blocks produced by another.
    # `setdefault` read as "do not clobber" when what it actually meant was
    # "this key does not exist yet", and the summary is the one field in the file
    # that describes the pass that just ran.
    visual["vlm_summary"] = payload.get("summary")
    return visual


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #

def smoke_test(frame_path: str | Path, cfg: VLMConfig | None = None) -> tuple[int, dict]:
    """One frame, one call, one verdict. `(exit_code, report)`.

    Exists because the failure it catches is invisible: a provider that is
    misnamed, mis-addressed or unauthorised looks exactly like a provider that
    had nothing to say, and the only way to tell them apart is to make the call
    against a file already on disk. Nothing here downloads a video, so a
    configuration can be verified in seconds instead of after a full run.

    The report says `configured=False` rather than inventing a result when
    nothing is set up, and `provider_unavailable` when something was asked for
    and could not be built. Those are different facts and a caller branching on
    them needs them kept apart.
    """
    cfg = cfg or VLMConfig.from_env()
    path = Path(frame_path).expanduser()
    report: dict = {"frame": str(path), "config": cfg.to_public_dict(),
                    "configured": False, "status": None, "provider": cfg.provider}

    if cfg.provider in ("none", ""):
        report["status"] = "configured=False"
        report["message"] = ("no VLM provider configured; set VLM_PROVIDER "
                             "(anthropic | openai | openrouter | ollama | lmstudio | "
                             "llamacpp | custom) or pass --vlm-provider")
        return 3, report
    if not path.exists():
        report["status"] = "frame_missing"
        report["message"] = f"no such frame: {path}"
        return 3, report

    provider = provider_from_config(cfg)
    report["provider"] = getattr(provider, "name", cfg.provider)
    if not provider.available:
        report["status"] = "provider_unavailable"
        report["message"] = getattr(provider, "reason", None) or "provider is not available"
        return 3, report

    report["configured"] = True
    report["provider_description"] = provider.describe()
    # The frame is analysed against no evidence document: this is a test of the
    # wire, not of the pipeline, and a context built from a video that was never
    # fetched would be a fiction.
    obs = provider.analyze(str(path), path.stem, {"frame_id": path.stem,
                                                  "video_title": None, "channel": None,
                                                  "nearby_transcript": None,
                                                  "ocr_text": None})
    report["status"] = obs.status
    report["content_type"] = obs.content_type
    report["summary"] = obs.summary
    report["confidence"] = obs.confidence
    report["notes"] = obs.notes
    report["error"] = obs.error
    report["counts"] = {k: len(getattr(obs, k)) for k in _STR_LISTS}
    if obs.status in ("ok", "no_visual"):
        print(f"  provider   {obs.backend} / {obs.model}")
        print(f"  status     {obs.status}  ({obs.content_type})")
        if obs.summary:
            print(f"  summary    {obs.summary[:300]}")
        print(f"  structure  " + "  ".join(f"{k}={n}" for k, n in report["counts"].items() if n))
        if obs.confidence is not None:
            print(f"  confidence {obs.confidence}  (model-derived; capped at {MAX_MODEL_CONFIDENCE})")
        mode = getattr(provider, "structured_mode", None)
        if mode:
            print(f"  structured {mode}")
        enforced = getattr(provider, "schema_enforced", None)
        if enforced is not None:
            # Worth its own line because the two are routinely confused: a 200
            # from a gateway that accepted `response_format` says the parameter
            # was accepted, not that the answer followed it.
            print(f"  schema     {'enforced by the endpoint' if enforced else 'NOT enforced — the answer did not follow the schema'}")
        return 0, report
    print(f"  provider   {obs.backend} / {obs.model}")
    print(f"  status     {obs.status}")
    print(f"  reason     {obs.notes or obs.error or 'unknown'}")
    return 1, report


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Check the configured VLM against one real frame. No download, "
                    "no video, no evidence document.")
    ap.add_argument("frame", help="path to one frame image (e.g. artifact/frames/u0007.jpg)")
    ap.add_argument("--vlm-provider", default=None, help="anthropic | openai | openrouter | "
                    "ollama | lmstudio | llamacpp | custom (overrides VLM_PROVIDER)")
    ap.add_argument("--vlm-model", default=None, help="model id (overrides VLM_MODEL)")
    ap.add_argument("--vlm-base-url", default=None, help="OpenAI-compatible base URL "
                    "(overrides VLM_BASE_URL)")
    ap.add_argument("--vlm-timeout", type=float, default=None,
                    help="per-call timeout in seconds (overrides VLM_TIMEOUT)")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = ap.parse_args(argv)

    cfg = VLMConfig.from_env(provider=args.vlm_provider, model=args.vlm_model,
                             base_url=args.vlm_base_url, timeout=args.vlm_timeout)
    code, report = smoke_test(args.frame, cfg)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        note = report.get("message") or report.get("notes") or ""
        print(f"\n  {'ok' if code == 0 else 'failed'}  {report['status']}  {note}\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
