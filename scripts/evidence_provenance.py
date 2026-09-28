"""Provenance constructors and the one rule that governs all of them.

A provenance entry is a pointer, so the only question that matters is whether it
resolves. These helpers are the single place a pointer is built, which is what
makes "no fabricated provenance" a property of the code rather than a habit: an
entry cannot be created without naming the source object it came from, and
`transcript_provenance` will not even accept a snippet id it was not handed.

The one rule, restated because it is the failure this whole layer exists to
prevent: **a claim is only as strong as the read that backs it.** `ocr_text` is
the evidence; `verified_verbatim` is a claim about that evidence. An entry with
no `ocr_text` therefore cannot carry `verified_verbatim=True` — the helper
refuses rather than trusting the caller, because a verified flag with nothing
behind it is exactly the laundering of an unread frame into a fact.
"""
from __future__ import annotations

from evidence_model import Provenance


def transcript_provenance(snippet, confidence: float = 0.97) -> Provenance:
    """Cite the exact transcript fragment. `t_start`/`t_end` come from the
    snippet's own measured span, never from the item's position or an estimate."""
    return Provenance(source="transcript", ref=snippet.id,
                      t_start=snippet.start_s, t_end=snippet.end_s, confidence=confidence)


def frame_provenance(frame, confidence: float = 0.9, ocr_text: str | None = None,
                     read_method: str | None = None) -> Provenance:
    """Cite a frame. With no `ocr_text` the entry cannot claim verification, and
    an unmeasured frame carries `t_start=None` rather than 0.

    Verification is granted on the same confidence policy as `ocr_provenance`: a
    text read at low confidence is a citation, not a confirmation, whichever
    source field carries it.
    """
    text = (ocr_text or "").strip() or None
    from ocr import HIGH_CONFIDENCE  # noqa: PLC0415
    return Provenance(source="frame", ref=frame.frame_id,
                      t_start=frame.t_start, t_end=frame.t_end,
                      ocr_text=text, read_method=read_method,
                      confidence=confidence,
                      verified_verbatim=bool(text) and confidence >= HIGH_CONFIDENCE)


def ocr_provenance(frame, ocr_text: str, confidence: float = 0.94,
                   read_method: str = "native") -> Provenance:
    """Cite a frame through a text read of it. The distinct `ocr` source is what
    lets a later reader tell 'I read the pixels' from 'I read the captions'.

    `verified_verbatim` is granted only above the confidence policy's high band.
    A low-confidence read is still a citation and still carries its text — it is
    recorded, inspectable and usable — but it is not allowed to assert that the
    pixels say exactly this, because that is the claim a misread character
    silently converts into a fabricated figure. The threshold lives in `ocr.py`;
    it is imported here rather than duplicated so the constructor and the
    validator cannot drift apart.
    """
    if not ocr_text.strip():
        raise ValueError(f"ocr_provenance({frame.frame_id}): empty OCR text cannot back a citation")
    from ocr import HIGH_CONFIDENCE  # noqa: PLC0415 — one threshold, two enforcement points
    return Provenance(source="ocr", ref=frame.frame_id, ocr_text=ocr_text,
                      t_start=frame.t_start, t_end=frame.t_end,
                      read_method=read_method, confidence=confidence,
                      verified_verbatim=confidence >= HIGH_CONFIDENCE)


def metadata_provenance(field: str, confidence: float = 1.0) -> Provenance:
    """Cite a metadata field by name. The field is the reference, so a reader can
    check the claim against `video` in the same document without leaving it."""
    return Provenance(source="metadata", ref=field, confidence=confidence)


def vlm_provenance(frame, caption: str, confidence: float = 0.7) -> Provenance:
    """Model-derived reading of a frame. Kept separate from `ocr` and capped
    below it on purpose: a model's description is not a reading of the pixels,
    and the two must never be averaged into one confidence."""
    return Provenance(source="vlm", ref=frame.frame_id, ocr_text=caption,
                      read_method="unknown", confidence=confidence)


def supports_verbatim(entry: Provenance) -> bool:
    """True when the entry carries text that was actually read off a source."""
    return bool((entry.ocr_text or "").strip())
