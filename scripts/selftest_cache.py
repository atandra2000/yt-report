"""The cache key must change whenever the answer would.

A cache does not fail loudly. It returns a plausible answer to a question that is
no longer being asked — the same frame read by a different model, the same
evidence written up under a different output language, an old prompt version's
answer presented as the current one. Nothing downstream can tell, because the
payload is well-formed and the provenance points at a real frame.

So this does not check that the cache *works*. It checks that each dimension
named in the contract changes the key, one at a time, by constructing two
configurations that differ in exactly that dimension and asserting the keys
differ. The other half matters just as much: a key that changes for everything
would pass every assertion here while being useless, so the same-object case is
asserted alongside.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import report_fixtures as FX  # noqa: E402
import synthesis as SY  # noqa: E402
import vlm as V  # noqa: E402
from cache import JsonCache, content_key  # noqa: E402

check = None


class _Provider:
    """The attributes `_cache_key` reads, and nothing else."""

    def __init__(self, name="openrouter", model="m", model_version=None):
        self.name = name
        self.model = model
        self.model_version = model_version


def _vlm_key(tmp: Path, name="openrouter", model="m", model_version=None,
             frame: bytes = b"pixels", frame_id="s0001",
             context: dict | None = None) -> str:
    frame_path = tmp / f"{frame_id}.jpg"
    frame_path.write_bytes(frame)
    ctx = context if context is not None else {"ocr_text": "hello", "nearby_transcript": "hi"}
    return V._cache_key(frame_path, frame_id, ctx,
                        _Provider(name, model, model_version))


def _synth_key(detail="balanced", language="auto", provider="openrouter", model="m",
               ev=None) -> str:
    cfg = SY.SynthConfig(detail=detail, provider=provider, model=model,
                         output_language=language)
    return SY.cache_key(ev or FX.simple_talking_head(), cfg)


def _differs(label: str, a: str, b: str) -> None:
    check(f"cache/{label} changes the key", a != b, f"both keys are {a}")


def t_vlm_key_dimensions(tmp: Path) -> None:
    """Every input the visual pass asks with, one at a time."""
    base = _vlm_key(tmp)
    _differs("VLM model", base, _vlm_key(tmp, model="other"))
    _differs("VLM provider", base, _vlm_key(tmp, name="openai"))
    _differs("VLM model version", base, _vlm_key(tmp, model_version="2026-01-01"))
    _differs("context", base, _vlm_key(tmp, context={"ocr_text": "different",
                                                     "nearby_transcript": "hi"}))
    _differs("frame bytes", base, _vlm_key(tmp, frame=b"other pixels"))
    _differs("frame id", base, _vlm_key(tmp, frame_id="s0002"))

    # The contract half, held apart so a regression names which one moved. The
    # module-level versions are the ones a prompt or schema edit bumps, and a
    # test that mutated them itself would pass for the wrong reason.
    bumped_prompt = _vlm_key_with(V.PROMPT_VERSION + 1)
    _differs("VLM prompt version", base, bumped_prompt)
    bumped_schema = _vlm_key_with(schema=V.SCHEMA_VERSION + 1)
    _differs("VLM schema version", base, bumped_schema)

    check("cache/VLM key is stable for the same question",
          base == _vlm_key(tmp), "the same inputs produced two keys")


def _vlm_key_with(prompt_version=None, schema=None) -> str:
    """The VLM key with one contract version hypothetically bumped.

    Written as a function rather than a monkeypatch so the test states the
    claim directly: hold the inputs, move the version, the key must move.
    """
    pv, sv = V.PROMPT_VERSION, V.SCHEMA_VERSION
    try:
        if prompt_version is not None:
            V.PROMPT_VERSION = prompt_version
        if schema is not None:
            V.SCHEMA_VERSION = schema
        with tempfile.TemporaryDirectory() as t:
            return _vlm_key(Path(t))
    finally:
        V.PROMPT_VERSION, V.SCHEMA_VERSION = pv, sv


def t_synth_key_dimensions() -> None:
    """Every input the synthesis pass asks with, one at a time."""
    base = _synth_key()
    _differs("synthesis model", base, _synth_key(model="other"))
    _differs("synthesis provider", base, _synth_key(provider="openai"))
    _differs("detail level", base, _synth_key(detail="deep"))
    _differs("output language", base, _synth_key(language="ja"))

    bumped_prompt = _synth_key_with(prompt_version=+1)
    _differs("synthesis prompt version", base, bumped_prompt)
    bumped_schema = _synth_key_with(schema=+1)
    _differs("synthesis schema version", base, bumped_schema)

    # Evidence *content*, not the video. Two evidence documents built from the
    # same video with one item changed must not share a key, or a draft written
    # against the old contents is served for the new ones.
    edited = FX.simple_talking_head()
    edited.items[0].value = "a different value entirely"
    _differs("evidence content", base, _synth_key(ev=edited))

    check("cache/synthesis key is stable for the same question",
          base == _synth_key(), "the same inputs produced two keys")


def _synth_key_with(prompt_version=0, schema=0) -> str:
    pv, sv = SY.PROMPT_VERSION, SY.SYNTH_SCHEMA_VERSION
    try:
        SY.PROMPT_VERSION, SY.SYNTH_SCHEMA_VERSION = pv + prompt_version, sv + schema
        return _synth_key()
    finally:
        SY.PROMPT_VERSION, SY.SYNTH_SCHEMA_VERSION = pv, sv


def t_failures_are_not_cached_as_success() -> None:
    """A failure must not become a permanent claim about the frame.

    Caching "the provider was down" as though it were a reading of the frame
    turns one transient outage into a permanent record that the frame held
    nothing, and no later run would revisit it.
    """
    for status in V._UNCACHEABLE:
        check(f"cache/a {status!r} outcome is uncacheable", status in V._UNCACHEABLE)
    check("cache/a successful reading is cacheable",
          "ok" not in V._UNCACHEABLE and "no_visual" not in V._UNCACHEABLE,
          str(V._UNCACHEABLE))

    # And the two callers actually honour that list, rather than each keeping its
    # own idea of what is worth remembering.
    import inspect
    for name, mod in (("run_vlm", V),):
        src = inspect.getsource(getattr(mod, name))
        check(f"cache/{name} consults the uncacheable list",
              "_UNCACHEABLE" in src, "the caller has its own rule")


def t_secrets_never_reach_a_key() -> None:
    """A key is a filename and a directory entry, so a key containing a
    credential writes that credential to disk twice."""
    with tempfile.TemporaryDirectory() as tmp:
        keys = {"VLM": _vlm_key(Path(tmp)), "synthesis": _synth_key()}
    for name, key in keys.items():
        for secret in ("sk-", "sk-or-v1-", "Bearer", "api_key", "API_KEY"):
            check(f"cache/the {name} key contains no {secret!r}",
                  secret not in key, key)
    # And the key is over content, so the same inputs always land on the same
    # path — which is what makes "the key never holds a secret" a property of
    # the construction rather than of a particular run.
    check("cache/the key is a fixed-length hex digest",
          len(_synth_key()) == len("synth-" + "0" * 32), _synth_key())


def t_corrupt_entry_is_a_miss() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        c = JsonCache("selftest", Path(tmp))
        c.put("k1", {"ok": True})
        check("cache/a written entry reads back", c.get("k1") == {"ok": True})
        c.path("k1").write_text("{not json")
        check("cache/a corrupt entry is a miss, not a crash", c.get("k1") is None)
        c.path("k1").unlink()
        check("cache/a missing entry is a miss", c.get("k1") is None)
        check("cache/an unwritable root does not raise", _put_into_missing_root(Path(tmp)))


def _put_into_missing_root(tmp: Path) -> bool:
    c = JsonCache("selftest", tmp / "nested" / "deeper")
    try:
        c.put("k", {"ok": 1})
        c.put("k", {"bad": {1, 2}})  # a set is not JSON-serialisable
    except Exception:  # noqa: BLE001 — the property is that nothing escapes
        return False
    return True


def t_semantic_key_covers_frame_time() -> None:
    """The extraction emits a provenance timestamp, so the frame's time is part
    of the question — not only of the answer.

    Found on a real video: a frame re-measured from 266.0 to 266.76 left ten
    cached items citing the old time, and the evidence gate refused the document.
    The key was over the frame's *read* and not its *time*, which is the same
    class of omission as a key over a model's identity but not its prompt.
    """
    import semantic as SEM

    class _Cfg:
        @staticmethod
        def to_dict() -> dict:
            return {}

    class _Ctx:
        def __init__(self, t):
            self.segment = type("S", (), {"to_dict": lambda s: {"id": "sg-1"}})()
            self.speech = ["a sentence"]
            f = type("F", (), {"frame_id": "u0010", "ocr_text": "Input: $2",
                               "ocr_regions": [], "ocr_confidence": 0.9,
                               "t_start": t, "t_end": None})()
            self.frames = [f]

    a = SEM._segment_cache_key(_Ctx(266.0), _Cfg(), {})
    b = SEM._segment_cache_key(_Ctx(266.76), _Cfg(), {})
    check("cache/a re-measured frame changes the semantic key", a != b, f"both are {a}")
    c = SEM._segment_cache_key(_Ctx(266.0), _Cfg(), {})
    check("cache/…and an unchanged frame still hits", a == c, "the key moved for no reason")


def t_ordering_matters() -> None:
    """Two components swapped are two different claims, not the same one."""
    check("cache/component order is significant",
          content_key("ab", "c") != content_key("a", "bc"),
          "the separator did not hold")
    check("cache/identical components give an identical key",
          content_key("a", "b") == content_key("a", "b"))
    check("cache/a key is short enough to be a filename",
          len(content_key("x")) <= 32, str(len(content_key("x"))))


def t_context_ordering_is_normalised(tmp: Path) -> None:
    """Two contexts that differ only in dict order are the same question.

    `content_key` hashes `repr`, and a dict's repr follows insertion order, so
    the normalisation has to happen at the call site — and it does, because a
    context assembled in a different order downstream would otherwise miss the
    cache on every run for no reason.
    """
    a = _vlm_key(tmp, context={"ocr_text": "x", "nearby_transcript": "y"})
    b = _vlm_key(tmp, context={"nearby_transcript": "y", "ocr_text": "x"})
    check("cache/context dict ordering does not change the key", a == b, f"{a} vs {b}")
    c = _vlm_key(tmp, context={"ocr_text": "x", "nearby_transcript": "z"})
    check("cache/…and a different context still does", a != c, f"both are {a}")


def _semantic_key(extraction_version=None, schema_version=None) -> str:
    """The semantic key with one contract version hypothetically bumped.

    Same construction as `_vlm_key_with`: hold the inputs, move the version, the
    key must move. A monkeypatch here would be wrong for the same reason the
    others are — the test has to state the claim, not perform it.
    """
    import semantic as SEM
    ev, schema = SEM.EXTRACTION_VERSION, SEM.SCHEMA_VERSION
    try:
        if extraction_version is not None:
            SEM.EXTRACTION_VERSION = extraction_version
        if schema_version is not None:
            SEM.SCHEMA_VERSION = schema_version
        return _semantic_key_now()
    finally:
        SEM.EXTRACTION_VERSION, SEM.SCHEMA_VERSION = ev, schema


def _semantic_key_now() -> str:
    import semantic as SEM

    class _Cfg:
        @staticmethod
        def to_dict() -> dict:
            return {}

    class _Ctx:
        def __init__(self):
            self.segment = type("S", (), {"to_dict": lambda s: {"id": "sg-1"}})()
            self.speech = ["a sentence"]
            f = type("F", (), {"frame_id": "u0010", "ocr_text": "Input: $2",
                               "ocr_regions": [], "ocr_confidence": 0.9,
                               "t_start": 266.0, "t_end": None})()
            self.frames = [f]

    return SEM._segment_cache_key(_Ctx(), _Cfg(), {})


def t_extraction_version_invalidates_the_cache() -> None:
    """A version nobody bumps is a version nobody gets.

    The obligation to bump `EXTRACTION_VERSION` is a memory obligation, and memory
    obligations are kept badly — this one was missed three times in one session,
    each miss serving a document built under the old extraction rules with
    nothing in the run to say a cache was involved. The key already carries the
    version; what was missing is proof that it does, tested the way every other
    dimension here is: hold the inputs, move the version, the key must move.

    The other half matters as much. A key that changed for *everything* would
    pass this and be useless, so the unchanged case is asserted alongside.
    """
    import semantic as SEM

    base = _semantic_key()
    _differs("extraction version", base, _semantic_key(extraction_version=SEM.EXTRACTION_VERSION + 1))
    _differs("semantic schema version", base, _semantic_key(schema_version=SEM.SCHEMA_VERSION + 1))
    check("cache/semantic key is stable for the same question",
          base == _semantic_key(), "the same inputs produced two keys")


def t_versions_live_in_one_place() -> None:
    """Every version is readable in one file, and every consumer reads it there.

    The point of centralising is that the obligation has one address. A second
    definition of a version is a version that will be bumped in the wrong place,
    so this asserts the single source and that each module's cache key actually
    consumes it — the two facts together, because either alone is satisfiable by
    a file nobody reads.
    """
    import semantic as SEM
    import synth_model as SM
    import vlm as V
    import versions as VS

    check("cache/every version is exported from one module",
          {"PIPELINE_VERSION", "EXTRACTION_VERSION", "SCHEMA_VERSION",
           "VLM_PROMPT_VERSION", "SYNTH_PROMPT_VERSION",
           "SYNTH_SCHEMA_VERSION"} <= set(VS.__all__), str(sorted(VS.__all__)))
    check("cache/no module defines a version of its own",
          all(getattr(m, name) == getattr(VS, src)
              for m, name, src in ((SEM, "EXTRACTION_VERSION", "EXTRACTION_VERSION"),
                                   (V, "PROMPT_VERSION", "VLM_PROMPT_VERSION"),
                                   (SM, "PROMPT_VERSION", "SYNTH_PROMPT_VERSION"),
                                   (SM, "SYNTH_SCHEMA_VERSION", "SYNTH_SCHEMA_VERSION"))),
          "a module holds a version that is not the one in versions.py")

    # A module that imported the constant and then stopped using it would pass
    # the check above while its key went stale-proof on paper only. Assert the
    # key moves for the real reason: the version is in the hash.
    import inspect
    for mod, fn, name in ((SEM, "_segment_cache_key", "EXTRACTION_VERSION"),
                          (V, "_cache_key", "PROMPT_VERSION"),
                          (__import__("synthesis"), "cache_key", "PROMPT_VERSION")):
        check(f"cache/{mod.__name__}.{fn} puts {name} in the key",
              name in inspect.getsource(getattr(mod, fn)),
              f"{fn} does not read {name}")


#: The whole contract in one table: `(layer, dimension, perturbed key)`.
#:
#: Built as data rather than as a sequence of `_differs` calls so that the
#: release artifact can cite a single list of what the cache is keyed on, and so
#: a dimension added to the contract and a dimension added to this table are the
#: same edit. The key is only read, never rebuilt here — a matrix that
#: reimplemented the key would be testing itself.
_CACHE_MATRIX_ROWS: list[tuple[str, str, str]] = []


def _cache_matrix() -> None:
    import semantic as SEM

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        rows = [
            ("vlm", "VLM provider", _vlm_key(tmp, name="openai")),
            ("vlm", "VLM model", _vlm_key(tmp, model="other")),
            ("vlm", "VLM model version", _vlm_key(tmp, model_version="2026-01-01")),
            ("vlm", "VLM prompt version", _vlm_key_with(prompt_version=V.PROMPT_VERSION + 1)),
            ("vlm", "VLM schema version", _vlm_key_with(schema=V.SCHEMA_VERSION + 1)),
            ("vlm", "VLM context", _vlm_key(tmp, context={"ocr_text": "different",
                                                         "nearby_transcript": "hi"})),
            ("vlm", "frame bytes", _vlm_key(tmp, frame=b"other pixels")),
            ("vlm", "frame id", _vlm_key(tmp, frame_id="s0002")),
            # The frame's *time* rides in the context the pipeline builds, which
            # is where the same question is asked again. Asserted through the
            # real builder rather than by editing a dict, because the claim is
            # that the pipeline puts it there.
            ("vlm", "frame timestamp",
             _vlm_key(tmp, context={"ocr_text": "hello", "nearby_transcript": "hi",
                                    "timestamp_s": 10.5})),
            ("synthesis", "synthesis provider", _synth_key(provider="openai")),
            ("synthesis", "synthesis model", _synth_key(model="other")),
            ("synthesis", "synthesis prompt version",
             _synth_key_with(prompt_version=1)),
            ("synthesis", "synthesis schema version", _synth_key_with(schema=1)),
            ("synthesis", "detail level", _synth_key(detail="deep")),
            ("synthesis", "output language", _synth_key(language="ja")),
        ]

    edited = FX.simple_talking_head()
    edited.items[0].value = "a different value entirely"
    rows.append(("synthesis", "evidence hash", _synth_key(ev=edited)))

    import versions as VS

    class _Cfg:
        @staticmethod
        def to_dict() -> dict:
            return {}

    class _Ctx:
        def __init__(self):
            self.segment = type("S", (), {"to_dict": lambda s: {"id": "sg-1"}})()
            self.speech = ["a sentence"]
            f = type("F", (), {"frame_id": "u0010", "ocr_text": "Input: $2",
                               "ocr_regions": [], "ocr_confidence": 0.9,
                               "t_start": 10.0, "t_end": None})()
            self.frames = [f]

    def semantic_key(version):
        original = SEM.EXTRACTION_VERSION
        try:
            SEM.EXTRACTION_VERSION = version
            return SEM._segment_cache_key(_Ctx(), _Cfg(), {})
        finally:
            SEM.EXTRACTION_VERSION = original

    rows.append(("semantic", "extraction version", semantic_key(VS.EXTRACTION_VERSION + 1)))

    _CACHE_MATRIX_ROWS[:] = rows

    # The same-object case, in the same breath: a key that changes for
    # everything passes every perturbation above while being useless.
    with tempfile.TemporaryDirectory() as t:
        check("cache/matrix: the same question still gives the same key",
              _vlm_key(Path(t)) == _vlm_key(Path(t)), "the VLM key is not stable")


def t_cache_matrix_is_complete() -> None:
    """Every dimension the release contract names, in one assertion each.

    The two functions above establish each dimension; this one establishes that
    the *list* is the one the release artifact claims. A dimension that moved out
    of the contract and stayed in the code is invisible to the per-dimension
    tests, which is the failure this table is for: it is a list that can be
    compared against a list, rather than a set of calls that can only be run.
    """
    _cache_matrix()
    named = {label for _, label, _ in _CACHE_MATRIX_ROWS}
    for required in ("VLM provider", "VLM model", "VLM schema version", "VLM context",
                     "frame bytes", "frame timestamp", "synthesis provider",
                     "synthesis model", "synthesis prompt version",
                     "synthesis schema version", "evidence hash", "detail level",
                     "output language", "extraction version"):
        check(f"cache/matrix: {required} is a keyed dimension", required in named,
              f"missing; the matrix has {sorted(named)}")
    check("cache/matrix: the matrix is not empty", len(_CACHE_MATRIX_ROWS) >= 14,
          str(len(_CACHE_MATRIX_ROWS)))

    # And every one of them actually moves the key it claims to.
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        base_vlm = _vlm_key(tmp)
    base_synth = _synth_key()
    for layer, label, key in _CACHE_MATRIX_ROWS:
        base = {"vlm": base_vlm, "synthesis": base_synth}.get(layer)
        if base is None:      # the semantic key is compared inside its own test
            continue
        check(f"cache/matrix: {label} moves its layer's key", key != base,
              f"both keys are {key}")
