"""`vlm-doctor` — can this provider actually produce a valid observation?

Answers a question the smoke test cannot. `vlm.py <frame>` runs one call and
tells you what came back; when the answer is wrong you learn that something is
wrong, but not which of six things. This walks the capability chain one step at
a time and reports each step separately, because the steps fail independently
and the fix for each is different:

    reachable               the endpoint answered at all
    authenticated           the credential was accepted (401/403 seen as failure)
    model_exists            the named model resolved (404 seen as failure)
    vision_capable          an image was accepted and described
    structured_requested    a schema constraint was sent
    structured_enforced     the returned payload actually matched that schema
    test_observation_valid  the payload parsed into a schema-valid observation

The last three are the ones that get conflated. A gateway that accepts
`response_format: json_schema`, returns HTTP 200, and answers in prose has
succeeded at four of the seven steps and produced nothing usable — so
`structured_requested` and `structured_enforced` are separate verdicts, and
`structured_enforced` is only ever true when the returned payload carried every
required key. That check is the entire reason this command exists.

`qualified` is true only when all seven hold and a real schema-valid observation
came back. A run where the model answered but ignored the schema is recorded as
`qualified: false` with the reason, not rounded up to a pass — a provider that
cannot be trusted with a schema stays untrusted, and the point of the command is
to say so rather than to make the configuration look fine.

    python3 vlm_doctor.py                       # uses VLM_* from the environment
    python3 vlm_doctor.py --frame path.jpg      # the frame the probe describes
    python3 vlm_doctor.py --json                # machine-readable report

Exit code is 0 only for a qualified provider, 1 for a reachable-but-unqualified
one, and 3 for a configuration that never reached a call.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import chat_json as CJ
from evidence_model import SCHEMA_VERSION
from vlm import (
    MAX_MODEL_CONFIDENCE,
    OBSERVATION_SCHEMA,
    PROVIDERS,
    TOOL_NAME,
    VLMConfig,
    VisionObservation,
    provider_from_config,
)

#: The ordered verdicts. Order is the diagnostic order: a reader scanning the
#: output top to bottom learns where the chain broke, because nothing after a
#: false step can be true.
CHECKS = (
    "reachable",
    "authenticated",
    "model_exists",
    "vision_capable",
    "structured_requested",
    "structured_enforced",
    "test_observation_valid",
)

#: A 1x1 PNG. The probe needs *an image the endpoint will accept*, not a
#: meaningful one: what is under test is the wire, the credential, the model's
#: willingness to look at pixels, and the schema. A tiny built-in file means the
#: doctor needs no video, no artifact directory and no download to run — the
#: same property that makes the smoke test usable in seconds.
#:
#: Built at 64×64, not 1×1, because that is the smallest size a real vision
#: endpoint accepts. The original single pixel was rejected outright by
#: OpenAI ("you uploaded an unsupported image"), so the doctor reported
#: NOT QUALIFIED for a provider that was fully qualified, on the one path a
#: first-time user takes — the documented command with no `--frame`. A probe the
#: endpoint refuses before the model sees it tests nothing except the endpoint's
#: minimum size.
def _probe_png(size: int = 64) -> bytes:
    """A small, valid, greyscale PNG: `size`×`size`, a mid-grey field."""
    import struct
    import zlib

    raw = b"".join(b"\x00" + bytes([128, 128, 128]) * size for _ in range(size))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


_PROBE_PNG = _probe_png()



class DoctorCheck:
    """One verdict: whether it held, and what to do about it if it did not."""

    __slots__ = ("name", "ok", "detail")

    def __init__(self, name: str, ok: bool, detail: str = ""):
        self.name = name
        self.ok = ok
        self.detail = detail

    def to_dict(self) -> dict:
        return {"check": self.name, "ok": self.ok, "detail": self.detail}


class DoctorReport:
    """The whole run, as data. Every field the CLI prints comes from here.

    `checks` is the ordered step chain, and it is a constructor argument rather
    than the module constant so a second doctor — `synth_doctor.py` — reports
    through the same class. One implementation of "a chain of steps where a step
    that never ran is not a pass" is worth more than two that agree today.
    """

    def __init__(self, cfg, checks: tuple[str, ...] = CHECKS):
        self.config = cfg.to_public_dict() if hasattr(cfg, "to_public_dict") else dict(cfg or {})
        self.checks_order = tuple(checks)
        self.checks: dict[str, DoctorCheck] = {}
        self.observation: dict | None = None
        self.latency_s: float | None = None
        self.error: str | None = None
        self.stage: str = "config"
        #: Free-form facts a specific doctor wants in its JSON but that are not
        #: steps — the sample that came back, the call that was never made, the
        #: budget that stopped it. Never a verdict; `qualified` is the only one.
        self.extra: dict = {}

    def set(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks[name] = DoctorCheck(name, ok, detail)

    def failed(self, name: str, detail: str, stage: str | None = None) -> None:
        """Record a break, and mark everything downstream unreachable.

        Downstream steps are set to `False` with `unreachable` rather than left
        absent. A missing key reads as "not tested"; a false one reads as "we
        know it did not hold", which is the honest state and stops a reader
        treating a chain that broke at authentication as if the vision and schema
        checks had passed.
        """
        self.set(name, False, detail)
        if stage:
            self.stage = stage
        if name not in self.checks_order:
            return
        idx = self.checks_order.index(name)
        for later in self.checks_order[idx + 1:]:
            if later not in self.checks:
                self.checks[later] = DoctorCheck(later, False, "unreachable")

    @property
    def qualified(self) -> bool:
        """True only when every step in the chain held. A step never run is not a pass."""
        return all(self.checks.get(n) is not None and self.checks[n].ok
                   for n in self.checks_order)

    def to_dict(self) -> dict:
        out = {
            "qualified": self.qualified,
            "stage_reached": self.stage,
            "provider": self.config.get("provider"),
            "model": self.config.get("model"),
            "checks": [self.checks[n].to_dict() for n in self.checks_order if n in self.checks],
            "latency_s": round(self.latency_s, 3) if self.latency_s is not None else None,
            "observation": self.observation,
            "error": self.error,
        }
        out.update(self.extra)
        return out

    def render(self, observation_title: str = "observation") -> str:
        out = [f"\n  {'QUALIFIED' if self.qualified else 'NOT QUALIFIED'}  "
               f"{self.config.get('provider')} / {self.config.get('model')}\n"]
        for name in self.checks_order:
            c = self.checks.get(name)
            if c is None:
                out.append(f"    {name:<26} not run")
                continue
            mark = "ok  " if c.ok else "FAIL"
            out.append(f"    {name:<26} {mark}  {c.detail}".rstrip())
        if self.latency_s is not None:
            out.append(f"    {'latency':<26} {self.latency_s * 1000:.0f} ms")
        if self.observation:
            o = self.observation
            out.append(f"\n    {observation_title:<12}  {o.get('content_type') or ''}  "
                       f"confidence {o.get('confidence')}")
            if o.get("summary"):
                out.append(f"    {o['summary'][:400]}")
            counts = o.get("counts") or {}
            live = "  ".join(f"{k}={v}" for k, v in counts.items() if v)
            if live:
                out.append(f"    structure    {live}")
        if self.error:
            out.append(f"\n    {self.error}")
        return "\n".join(out) + "\n"


def _probe_image(tmp_dir: Path, frame: str | None) -> tuple[Path, bool]:
    """A file to send. The caller's frame when given, else the built-in pixel."""
    if frame:
        p = Path(frame).expanduser()
        return p, False
    p = tmp_dir / "vlm-doctor-probe.png"
    p.write_bytes(_PROBE_PNG)
    return p, True


def run_doctor(cfg: VLMConfig, frame: str | None = None) -> tuple[int, DoctorReport]:
    """Walk the capability chain. `(exit_code, report)`."""
    import tempfile

    report = DoctorReport(cfg)
    # The VLM-specific facts a reader needs to interpret the verdicts above.
    # `extra` rather than fields on the shared class, so the second doctor does
    # not inherit a schema it has no answer for.
    report.extra = {
        "schema_version": SCHEMA_VERSION,
        "schema_required": list(OBSERVATION_SCHEMA.get("required", ())),
        "max_model_confidence": MAX_MODEL_CONFIDENCE,
    }

    if cfg.provider in ("none", ""):
        report.failed("reachable",
                      "no provider configured; set VLM_PROVIDER or pass --vlm-provider",
                      stage="config")
        report.error = ("nothing was called. Set VLM_PROVIDER to one of: "
                        + ", ".join(p for p in PROVIDERS if p != "fake"))
        return 3, report

    try:
        provider = provider_from_config(cfg)
    except Exception as exc:  # noqa: BLE001 — a construction failure is a verdict
        report.failed("reachable", f"the provider could not be constructed: "
                                   f"{type(exc).__name__}: {exc}", stage="config")
        report.error = "no call was made: the provider could not be constructed"
        return 3, report
    if not provider.available:
        reason = getattr(provider, "reason", None) or "provider is not available"
        report.failed("reachable", reason, stage="config")
        report.error = "no call was made: the provider could not be constructed"
        return 3, report

    # The provider classifies the *expected* egress failures — a refused
    # credential, a 404, a rate limit — into the VLM vocabulary and returns an
    # observation carrying the reason. Everything else is unclassified by
    # definition: an unwrapped `ssl.SSLError`, an `OSError` writing the probe
    # image, a provider module that changed shape. Those escaped as a traceback
    # from here, and `release_check` calls this function in-process, so one of
    # them took the whole five-step matrix down with it and the reader saw a
    # stack instead of a verdict.
    #
    # The distinction the two `except` bodies keep is the one the report rests
    # on. An exception here is *no answer*: nothing was disproven and nothing was
    # shown, so the chain breaks at `reachable`, the downstream checks are
    # marked unreachable by the same cascade the expected failures use, and the
    # exit code is 3 — the documented code for a run that never reached a call.
    # The expected failures are still classified by the provider and still
    # distinguish auth from model from vision; only the unclassified escape
    # lands here, and it lands as `reachable` because that is where it broke.
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path, synthetic = _probe_image(Path(tmp), frame)
            if not path.exists():
                report.failed("reachable", f"no such file: {path}", stage="input")
                report.error = "supply an existing frame, or omit --frame to use the built-in probe"
                return 3, report

            # Reachable is decided by whether a call was *attempted and answered at
            # the transport level*. The probe runs the real analyze path, so every
            # earlier verdict is derived from the same exchange rather than from a
            # separate health check that could disagree with it.
            obs = provider.analyze(str(path), path.stem,
                                   {"frame_id": path.stem, "video_title": None,
                                    "channel": None, "nearby_transcript": None,
                                    "ocr_text": None})
    except Exception as exc:  # noqa: BLE001 — an unclassified escape is a verdict
        report.latency_s = time.monotonic() - started
        report.failed("reachable", f"the call raised {type(exc).__name__}: {exc}",
                      stage="transport")
        report.error = ("the endpoint could not be reached: the provider raised "
                        f"{type(exc).__name__} rather than returning a result")
        return 3, report
    report.latency_s = time.monotonic() - started

    # A call that reached the endpoint. Before this point nothing was knowable.
    report.set("reachable", True, f"{cfg.provider} answered in "
                                  f"{report.latency_s * 1000:.0f} ms")

    reason = (obs.notes or obs.error or "").lower()
    fatal_auth = obs.status == "unavailable"
    # A 401/403 is `unavailable` and fatal; a missing model is `failed` with a
    # 404. Both arrive as a non-ok status, and the two are told apart by the
    # reason text because the provider classifies into the VLM vocabulary rather
    # than into HTTP codes.
    if fatal_auth and ("401" in reason or "403" in reason or "authentication" in reason):
        report.failed("authenticated", obs.notes or obs.error or "credential rejected",
                      stage="auth")
        report.error = "the endpoint answered and refused the credential"
        return 1, report
    report.set("authenticated", True, "the credential was accepted")

    if "404" in reason or "not found" in reason:
        report.failed("model_exists", obs.notes or obs.error or "model not found",
                      stage="model")
        report.error = f"the endpoint does not resolve model {cfg.model!r}"
        return 1, report
    report.set("model_exists", True, f"{cfg.model} resolved")

    if obs.status not in ("ok", "no_visual"):
        # A non-auth, non-404 failure at this point is the vision call itself:
        # a text-only model, a rejected media type, a timeout.
        report.failed("vision_capable", obs.notes or obs.error or obs.status,
                      stage="vision")
        report.error = ("the model answered no usable observation; the frame may not have "
                        "been accepted, or the call failed for another reason")
        return 1, report
    report.set("vision_capable", True,
               "the model returned an observation for the image"
               + (" (built-in probe pixel)" if synthetic else ""))

    mode = getattr(provider, "structured_mode", None)
    report.set("structured_requested", bool(mode),
               f"constraint sent: {mode}" if mode
               else "no structured-output constraint was accepted or sent")

    # The load-bearing check. `schema_enforced` is set by the negotiator only
    # when the *returned payload* carried every required key; it is never
    # inferred from a 200, and a mode that was sent but not honoured leaves it
    # explicitly False.
    enforced = getattr(provider, "schema_enforced", None)
    if mode is None:
        report.set("structured_enforced", False,
                   "no constraint was in force, so enforcement cannot be claimed")
    elif enforced:
        report.set("structured_enforced", True,
                   f"the {mode} response carried every required key")
    else:
        report.set("structured_enforced", False,
                   f"the endpoint accepted {mode} but the answer did not follow the schema")

    # Last, and independent of the above: does the payload actually parse into a
    # valid observation? A model can satisfy every schema key and still produce
    # an observation the validator would reject.
    valid, why = _observation_valid(obs, path.stem)
    report.set("test_observation_valid", valid, why)

    report.observation = {
        "status": obs.status,
        "content_type": obs.content_type,
        "summary": obs.summary,
        "confidence": obs.confidence,
        "notes": obs.notes,
        "counts": {k: len(getattr(obs, k)) for k in
                   ("visual_elements", "text_relationships", "entities", "numbers",
                    "relationships", "actions", "uncertainties")},
    }

    if not (report.checks["structured_enforced"].ok and valid):
        report.error = ("the provider is reachable and answered, but it is not qualified: "
                        "its structured output is not enforced or its observation is invalid. "
                        "The VLM pass will reject its output rather than coerce it.")
        return 1, report
    return 0, report


def _observation_valid(obs: VisionObservation, frame_id: str) -> tuple[bool, str]:
    """Re-parse the observation the way the pipeline does, and judge the result.

    Deliberately a second, independent check on the same object: the provider
    built this one on the way past, and a provider that read its own homework is
    not evidence. `from_dict` is the same entry point `build_evidence.py` uses,
    so what passes here passes there.
    """
    if obs.status not in ("ok", "no_visual"):
        return False, f"status is {obs.status!r}: {obs.notes or obs.error or 'no reason given'}"
    if not obs.summary:
        return False, "observation carries no summary"
    try:
        again = VisionObservation.from_dict(obs.to_dict(), frame_id)
    except Exception as exc:  # noqa: BLE001 — a parse failure is the verdict
        return False, f"the observation does not re-parse: {exc}"
    if again.status != obs.status:
        return False, f"re-parsing changed the status from {obs.status!r} to {again.status!r}"
    if obs.confidence is not None and obs.confidence > MAX_MODEL_CONFIDENCE:
        return False, (f"confidence {obs.confidence} exceeds the {MAX_MODEL_CONFIDENCE} "
                       f"ceiling for model-derived readings")
    if not obs.content_type:
        return False, "observation has no content_type"
    return True, f"status {obs.status!r}, content_type {obs.content_type!r}, summary present"


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="vlm-doctor",
        description="Check whether the configured VLM can actually produce a valid "
                    "observation, step by step. No video, no download.")
    ap.add_argument("--frame", default=None,
                    help="a real frame to probe with; omit to use a built-in 1x1 pixel")
    ap.add_argument("--vlm-provider", default=None, help="provider alias (overrides VLM_PROVIDER)")
    ap.add_argument("--vlm-model", default=None, help="model id (overrides VLM_MODEL)")
    ap.add_argument("--vlm-base-url", default=None, help="base URL (overrides VLM_BASE_URL)")
    ap.add_argument("--vlm-timeout", type=float, default=None, help="per-call timeout, seconds")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)

    cfg = VLMConfig.from_env(provider=args.vlm_provider, model=args.vlm_model,
                             base_url=args.vlm_base_url, timeout=args.vlm_timeout)
    # `run_doctor` is written not to raise, and this guard exists so that stays
    # true from the outside too: a defect in the chain, or in anything it calls
    # that the chain does not own, must still leave the reader with a report
    # and an exit code rather than a stack trace. The same convention
    # `synth_doctor.main` and `critic_doctor.main` already follow.
    try:
        code, report = run_doctor(cfg, args.frame)
    except Exception as exc:  # noqa: BLE001 — the traceback is not the verdict
        print(f"vlm-doctor: the doctor itself raised {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        sys.stdout.write(report.render())
    return code


if __name__ == "__main__":
    raise SystemExit(main())
