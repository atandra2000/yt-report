"""Content-addressed cache for model-derived results.

Two producers need it — the VLM pass and the semantic extractor — and they have
the same correctness requirement, which is why it is one module rather than a
helper copied into each: **a cache key must change whenever the answer would.**

The key is a hash over the inputs *and* the contract that produced the answer.
The contract half is the part that gets left out by accident, and the result is
the worst kind of stale: an old extraction silently presented as a current one,
with provenance pointing at frames whose meaning has since been re-derived. So
every caller passes, in the same tuple, the content it asked about, the model
identity, the prompt or instruction version, and the schema version. Change any
of them and the entry is a miss.

Deliberately *not* a dependency. The repo runs under `uv run <script>` with no
declared environment, and a cache that cannot be reached still has to be able to
say so rather than crash a pipeline: a write failure costs time next run, never
the run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CACHE_DIR = Path(os.environ.get("YT_REPORT_CACHE", Path.home() / ".cache" / "yt-report"))


def content_key(*parts: object) -> str:
    """A short, stable key over the caller's contract tuple.

    `parts` are stringified in order, so a caller that forgets one changes its
    key without noticing — which is why the docstring names what belongs in the
    tuple rather than leaving it to the call site. Order matters: the same
    components in a different order are a different claim about what was asked.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8", "replace"))
        h.update(b"\x1f")  # unit separator: prevents ("ab","c") colliding with ("a","bc")
    return h.hexdigest()[:32]


class JsonCache:
    """A namespaced directory of JSON payloads, addressed by `content_key`.

    A corrupt or unreadable entry is a miss, never a crash. That is the same rule
    the OCR cache follows and for the same reason: a cache that raises on a bad
    file turns a performance problem into a broken pipeline, and the pipeline is
    the thing that has to work.
    """

    def __init__(self, namespace: str, root: Path | None = None):
        self.root = (root or CACHE_DIR) / namespace

    def path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str):
        p = self.path(key)
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, payload: object) -> None:
        p = self.path(key)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False))
            tmp.replace(p)
        except (OSError, TypeError, ValueError):
            pass  # uncacheable, not unusable

    def get_or_call(self, key: str, produce):
        """Return `(payload, hit)`. `produce` runs only on a miss.

        The boolean is returned rather than logged because the caller reports a
        cache hit rate as a measurement, and a measurement that is computed from
        the cache's own opinion of itself is not one.
        """
        hit = self.get(key)
        if hit is not None:
            return hit, True
        payload = produce()
        self.put(key, payload)
        return payload, False
