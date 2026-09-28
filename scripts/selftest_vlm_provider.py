"""VLM provider tests: configuration, negotiation, and every refusal.

The Anthropic provider had no offline test and no way to be given a bad
response, so the paths that matter most — a 401, a 429, a timeout, a model that
answers in prose, a model that invents a frame id — were only reachable with a
live key. That is exactly backwards: the failure modes are the reason the class
exists, and they are the ones that never get exercised.

`OpenAICompatProvider` takes an injectable transport, and the classification
lives in the provider rather than in the transport, so a test double exercises
the real code rather than a bypass around it. Everything below is offline.
"""
from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import chat_json as CJ  # noqa: E402
import vlm as V  # noqa: E402

#: A one-pixel JPEG. Small enough to inline, real enough that nothing about the
#: request path cares whether it is a picture.
TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300ffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffc2000b080001000101011100"
    "ffc40014000100000000000000000000000000000009ffc4001410010000000000000000"
    "000000000000000000ffda0008010100000000013fffd9")


class _Transport:
    """Records what was sent and replays a scripted sequence of outcomes.

    Outcomes are either a payload dict (returned as the response body) or an
    exception instance (raised), because that is the full set of things a real
    endpoint does.
    """

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.sent: list[dict] = []
        self.headers: list[dict] = []
        self.urls: list[str] = []

    def __call__(self, url, headers, data, timeout):
        self.urls.append(url)
        self.headers.append(headers)
        self.sent.append(json.loads(data))
        outcome = self.outcomes.pop(0) if self.outcomes else {}
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            return outcome(self.sent[-1])
        return json.dumps(outcome)


def _http(code: int, message: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "err", {},
                                  io.BytesIO(json.dumps({"error": {"message": message}}).encode()))


def _chat(content) -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


def _frame(tmp: Path) -> str:
    p = tmp / "u0001.jpg"
    p.write_bytes(TINY_JPEG)
    return str(p)


def _provider(transport, **kw) -> V.OpenAICompatProvider:
    kw.setdefault("model", "test-model")
    kw.setdefault("api_key", "k")
    return V.OpenAICompatProvider(transport=transport, **kw)


#: A *complete* observation. `OBSERVATION_SCHEMA` is strict-mode valid, which
#: means every property is required — so a fixture carrying only the three
#: load-bearing keys is a schema violation, and a negotiation test that used one
#: would be asserting that a non-conforming answer is a conforming one. Built
#: from the schema itself so it cannot drift from it again.
GOOD = json.dumps({
    "summary": "a benchmark table", "content_type": "table", "confidence": 0.6,
    "visual_elements": [{"kind": "table"}],
    "text_relationships": [{"text": "Qwen", "role": "row_label"}],
    "entities": [{"text": "Qwen", "kind": "model"}],
    "numbers": [{"value": "32", "unit": "tok/s", "row": "Qwen", "column": "Throughput",
                 "label": "", "context": "", "legible": True}],
    "claims_supported": [],
    "actions": [],
    "relationships": [],
    "uncertainties": [],
})
FENCED = "```json\n" + GOOD + "\n```"


def t_vlm_config() -> None:
    import os
    saved = {k: os.environ.get(k) for k in
             ("VLM_PROVIDER", "VLM_MODEL", "VLM_BASE_URL", "VLM_API_KEY",
              "VLM_TIMEOUT", "VLM_MAX_FRAMES", "ANTHROPIC_API_KEY")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        cfg = V.VLMConfig.from_env()
        check("config/default is no provider", cfg.provider == "none")
        check("config/default budget is the shipped one", cfg.max_frames == V.DEFAULT_MAX_VLM_FRAMES)
        check("config/default timeout is the shipped one", cfg.timeout == 90.0)

        os.environ.update(VLM_PROVIDER="ollama", VLM_MODEL="qwen2.5vl",
                          VLM_BASE_URL="http://box:11434/v1", VLM_TIMEOUT="12",
                          VLM_MAX_FRAMES="3")
        cfg = V.VLMConfig.from_env()
        check("config/read from the environment", cfg.provider == "ollama" and cfg.model == "qwen2.5vl")
        check("config/timeout parsed as a number", cfg.timeout == 12.0)
        check("config/budget parsed as a number", cfg.max_frames == 3)

        os.environ["VLM_TIMEOUT"] = "not a number"
        check("config/a malformed variable is ignored, not fatal",
              V.VLMConfig.from_env().timeout == 90.0)

        cfg = V.VLMConfig.from_env(provider="anthropic", model=None)
        check("config/an explicit override beats the environment", cfg.provider == "anthropic")
        check("config/an omitted override does not clear the environment",
              cfg.model == "qwen2.5vl")
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    d = V.VLMConfig(provider="openai", api_key="sk-secret-value").to_public_dict()
    check("config/the key is never in the public record",
          "sk-secret-value" not in json.dumps(d) and d["api_key_present"] is True)
    check("config/the public record says only that a key exists",
          set(d) == {"provider", "model", "base_url", "timeout", "max_frames",
                     "api_key_present"})

    for name in ("openai", "openrouter", "ollama", "lmstudio", "llamacpp"):
        check(f"config/{name} is an accepted provider", name in V.PROVIDERS)
    check("config/custom is accepted", "custom" in V.PROVIDERS)
    check("config/a typo is rejected by the CLI, not by falling through",
          "gpt-4o" not in V.PROVIDERS)


def t_vlm_provider_selection() -> None:
    p = V.get_provider("none")
    check("provider/none is a decision, not a failure", p.name == "unavailable" and not p.available)
    p = V.get_provider("custom")
    check("provider/custom without a base url says so",
          isinstance(p, V.UnavailableProvider) and "base-url" in (p.reason or ""))
    p = V.get_provider("custom", base_url="http://localhost:9000/v1", api_key="k")
    check("provider/custom with a base url is a real provider",
          isinstance(p, V.OpenAICompatProvider) and p.base_url == "http://localhost:9000/v1")

    p = V.get_provider("openai", api_key="k")
    check("provider/a hosted endpoint needs a key", p.requires_key)
    p = V.get_provider("openai")
    check("provider/a hosted endpoint with no key is unavailable, not broken",
          not p.available and "VLM_API_KEY" in (p.reason or ""))
    p = V.get_provider("ollama")
    check("provider/a local endpoint needs no key", p.available and not p.requires_key)
    check("provider/a local endpoint points at its own port",
          p.base_url.endswith(":11434/v1"), p.base_url)

    p = V.get_provider("openrouter", api_key="k")
    check("provider/an alias resolves to its endpoint",
          p.base_url == "https://openrouter.ai/api/v1")
    p = V.get_provider("not-a-provider")
    check("provider/an unknown name says which name it did not know",
          "not-a-provider" in (p.reason or ""))

    p = V.get_provider("fake", observations={"u1": {"summary": "x", "content_type": "table",
                                                    "confidence": 0.5}})
    check("provider/fake still needs its fixture map", not V.get_provider("fake").available)
    check("provider/fake with a map is available", p.available and p.name == "fake")

    cfg = V.VLMConfig(provider="ollama")
    check("provider/the config door reaches the same factory",
          V.provider_from_config(cfg).name == "ollama")


def t_vlm_structured_negotiation(tmp) -> None:
    """The endpoint is asked for a schema, then asked for less, once."""
    path = _frame(tmp)

    tr = _Transport(_http(400, "response_format is not supported"), _chat(FENCED))
    pr = _provider(tr, provider_name="ollama")
    obs = pr.analyze(path, "u0001", {})
    check("negotiation/a refusal downgrades instead of failing", obs.status == "ok", obs.notes)
    check("negotiation/the mode that worked is latched", pr.structured_mode == "json_object")
    check("negotiation/the first attempt asked for a strict schema",
          tr.sent[0]["response_format"]["type"] == "json_schema")
    check("negotiation/the fallback asks only for JSON",
          tr.sent[1]["response_format"]["type"] == "json_object")

    tr = _Transport(_http(400, "response_format is not supported"),
                    _http(400, "response_format is not supported"), _chat(FENCED))
    pr = _provider(tr, provider_name="ollama")
    obs = pr.analyze(path, "u0001", {})
    check("negotiation/an endpoint speaking nothing is handled", obs.status == "ok")
    check("negotiation/the last resort is a prompt, and is recorded",
          pr.structured_mode == "prompt_only" and "prompt_only" in (obs.notes or ""))
    check("negotiation/the last resort carries no response_format",
          "response_format" not in tr.sent[2])

    tr2 = _Transport(_chat(GOOD), _chat(GOOD))
    pr = _provider(tr2, provider_name="ollama")
    pr.analyze(path, "u0001", {})
    pr.analyze(path, "u0002", {})
    check("negotiation/the negotiation is paid once, not per frame",
          pr.structured_mode == "json_schema" and len(tr2.sent) == 2, str(len(tr2.sent)))

    # The mode that was *asked for* is not the mode that was *enforced*. A
    # gateway that accepts `response_format` and answers in another shape is
    # common, and treating a 200 as proof of enforcement is the silent semantic
    # fallback this records instead of.
    pr = _provider(_Transport(_chat('{"content_type":"screen_recording","structure":{}}')),
                   provider_name="ollama")
    obs = pr.analyze(path, "u0001", {})
    check("negotiation/an accepted-but-ignored schema is not recorded as enforced",
          pr.schema_enforced is False, str(pr.schema_enforced))
    check("negotiation/the mode is not trusted again after it was ignored",
          pr.structured_mode != "json_schema", str(pr.structured_mode))
    # …and the next call steps *down* rather than back to the top. Retrying the
    # mode that was just proven ineffective spends a second paid call to get the
    # same non-conforming answer, and a model that cannot produce a strict-schema
    # payload would never reach the two weaker modes that can work for it.
    BAD = '{"content_type":"screen_recording","structure":{}}'
    tr3 = _Transport(_chat(BAD), _chat(GOOD))
    pr = _provider(tr3, provider_name="ollama")
    pr.analyze(path, "u0001", {})
    pr.analyze(path, "u0002", {})
    check("negotiation/the next call descends the ladder instead of restarting it",
          tr3.sent[1].get("response_format", {}).get("type") == "json_object",
          json.dumps({k: v for k, v in tr3.sent[1].items() if k != "messages"})[:160])

    tr4 = _Transport(_chat(BAD), _chat(BAD), _chat(BAD))
    pr = _provider(tr4, provider_name="ollama")
    for n in ("u0001", "u0002", "u0003"):
        pr.analyze(path, n, {})
    check("negotiation/a persistently non-conforming model reaches the prompt fallback",
          pr.structured_mode == "prompt_only", str(pr.structured_mode))
    check("negotiation/…and the prompt fallback is what the last call sent",
          "response_format" not in tr4.sent[2], json.dumps(list(tr4.sent[2]))[:120])
    check("negotiation/…and the response is still refused, not used",
          obs.status == "malformed", obs.status)
    pr = _provider(_Transport(_chat(GOOD)), provider_name="ollama")
    pr.analyze(path, "u0001", {})
    check("negotiation/a conforming answer is recorded as enforced",
          pr.schema_enforced is True, str(pr.schema_enforced))

    tr = _Transport(_chat(GOOD))
    pr = _provider(tr, provider_name="openai")
    pr.analyze(path, "u0001", {})
    body = tr.sent[0]
    check("negotiation/the schema sent is the observation schema",
          body["response_format"]["json_schema"]["schema"] == V.OBSERVATION_SCHEMA)
    check("negotiation/the system instruction is sent",
          any(m["role"] == "system" for m in body["messages"]))
    inline = [part for m in body["messages"] if isinstance(m["content"], list)
              for part in m["content"] if isinstance(part, dict)
              and part.get("type") == "image_url"]
    check("negotiation/the image is inlined, not linked",
          len(inline) == 1 and inline[0]["image_url"]["url"].startswith("data:image/"),
          json.dumps(inline)[:120])


def t_vlm_response_validation(tmp) -> None:
    """Every way a response can be wrong, and what each one becomes."""
    path = _frame(tmp)

    def run(payload, **kw):
        pr = _provider(_Transport(payload), provider_name="ollama", **kw)
        return pr, pr.analyze(path, "u0001", {})

    _, o = run(_chat(FENCED))
    check("response/valid: accepted", o.status == "ok" and o.content_type == "table")
    check("response/valid: confidence is kept and stays inside the model band",
          o.confidence is not None and 0 < o.confidence <= V.MAX_MODEL_CONFIDENCE,
          str(o.confidence))
    check("response/valid: the structured fields survive",
          o.numbers and o.numbers[0].get("row") == "Qwen")

    _, o = run(_chat("I looked at the frame and it seems to show a chart."))
    check("response/prose instead of JSON is malformed", o.status == "malformed")
    check("response/prose says why", "not JSON" in (o.notes or ""))

    _, o = run(_chat('{"summary":"a table","content_type":'))
    check("response/partial JSON is malformed", o.status == "malformed")

    _, o = run(_chat('{"content_type":"table","confidence":0.5}'))
    check("response/a missing required field is malformed", o.status == "malformed")
    check("response/a missing summary says which field", "summary" in (o.notes or ""))

    _, o = run(_chat('{"summary":"x","content_type":"table","confidence":"high"}'))
    check("response/a non-numeric confidence is malformed", o.status == "malformed")
    check("response/a non-numeric confidence is named", "confidence" in (o.notes or ""))

    _, o = run(_chat('{"summary":"x","content_type":"table","confidence":5}'))
    check("response/an out-of-range confidence is malformed", o.status == "malformed")

    _, o = run(_chat('{"summary":"x","content_type":"table","confidence":0.95}'))
    check("response/a confidence in the model's own band is capped, not refused",
          o.status == "ok" and o.confidence == V.MAX_MODEL_CONFIDENCE)

    _, o = run(_chat('{"summary":"x","content_type":"hologram","confidence":0.5}'))
    check("response/an enum outside the vocabulary is malformed", o.status == "malformed")
    check("response/an unknown enum degrades to 'unknown', it is not invented",
          o.content_type == "unknown")

    _, o = run(_chat('{"frame_id":"zz9999","summary":"x","content_type":"table","confidence":0.5}'))
    check("response/an invented frame id is caught", o.status == "malformed")
    check("response/the frame actually asked about is kept", o.frame_id == "u0001")
    check("response/the invented id is named in the record", "zz9999" in (o.notes or ""))

    _, o = run(_chat('{"summary":"x","content_type":"diagram","confidence":0.5,'
                     '"relationships":[{"from":"a"}]}'))
    check("response/a relationship missing an endpoint is malformed", o.status == "malformed")
    check("response/the half-relationship is named as such",
          "missing its 'to' endpoint" in (o.notes or ""), o.notes)

    _, o = run(_chat('{"summary":"x","content_type":"chart","confidence":0.5,'
                     '"numbers":[{"value":"999","legible":false}]}'))
    check("response/a fabricated figure reaches the observation as read data",
          o.status == "ok" and o.numbers[0]["value"] == "999")
    check("response/…and is marked as inferred by the model itself",
          o.numbers[0]["legible"] is False)

    _, o = run({"choices": []})
    check("response/no choices at all is malformed", o.status == "malformed")
    _, o = run({"choices": [{"message": {"content": None}}]})
    check("response/an empty message is malformed", o.status == "malformed")
    _, o = run({"choices": [{"message": {"content": None, "tool_calls": [
        {"function": {"arguments": GOOD}}]}}]})
    check("response/a tool-call answer is read, not discarded",
          o.status == "ok" and o.content_type == "table")
    _, o = run({"choices": [{"message": {"content": [
        {"type": "text", "text": GOOD}]}}]})
    check("response/content split into parts is joined", o.status == "ok")
    _, o = run({"choices": [{"message": {"content": None}, "finish_reason": "length"}]})
    check("response/a truncated answer is malformed, not empty", o.status == "malformed")


def t_vlm_transport_failures(tmp) -> None:
    path = _frame(tmp)
    cases = [
        (urllib.error.HTTPError("u", 401, "x", {}, None), "unavailable"),
        (urllib.error.HTTPError("u", 403, "x", {}, None), "unavailable"),
        (urllib.error.HTTPError("u", 429, "x", {}, None), "failed"),
        (urllib.error.HTTPError("u", 500, "x", {}, None), "failed"),
        (urllib.error.HTTPError("u", 400, "x", {}, None), "failed"),
        (TimeoutError(), "failed"),
        (urllib.error.URLError(ConnectionRefusedError()), "failed"),
    ]
    for exc, want in cases:
        pr = _provider(_Transport(exc), provider_name="ollama")
        o = pr.analyze(path, "u0001", {})
        check(f"failure/{type(exc).__name__} {getattr(exc, 'code', '')} → {want}",
              o.status == want, f"{o.status}: {o.notes}")

    pr = _provider(_Transport(_http(401, "invalid api key")), provider_name="ollama")
    o1 = pr.analyze(path, "u0001", {})
    o2 = pr.analyze(path, "u0002", {})
    check("failure/auth stops the run after one attempt",
          o1.status == "unavailable" and o2.status == "unavailable"
          and "previously failed" in (o2.notes or ""))

    pr = _provider(_Transport(lambda body: "<html>502 Bad Gateway</html>"),
                   provider_name="ollama")
    o = pr.analyze(path, "u0001", {})
    check("failure/a non-JSON body is malformed, not a crash", o.status == "malformed")

    pr = _provider(_Transport(_chat(GOOD)), provider_name="ollama")
    o = pr.analyze(str(tmp / "nope.jpg"), "u0009", {})
    check("failure/a missing frame file is a failed observation", o.status == "failed")
    check("failure/a missing frame file does not raise", o.frame_id == "u0009")


def t_vlm_media_rejection_is_not_a_mode_rejection() -> None:
    """A rejected image and a refused structured-output parameter are one 400
    with the same words in them, and they are different problems.

    The classifier once matched on "unsupported" and "invalid_request_error",
    which every OpenAI 400 carries, so `image_parse_error: "you uploaded an
    unsupported image"` was read as "the endpoint does not speak
    `response_format`". The negotiator then walked all three modes for nothing
    and the doctor reported a working provider as unconfigured, naming the wrong
    subsystem. This pins the two apart.
    """
    import vlm_doctor as D
    import struct

    image_rejected = _http(400, json.dumps({"error": {"message": (
        "You uploaded an unsupported image. Please make sure your image is valid."),
        "type": "invalid_request_error", "param": None, "code": "image_parse_error"}}))
    err = CJ.classify_http(image_rejected)
    check("media/an image rejection does not downgrade the structured mode",
          err.downgrade is False, str(err.reason))
    check("media/an image rejection says it is about the image",
          "image" in err.reason.lower(), err.reason)

    mode_refused = _http(400, json.dumps({"error": {"message": (
        "Unrecognized request argument supplied: response_format"),
        "type": "invalid_request_error", "param": "response_format", "code": None}}))
    err2 = CJ.classify_http(mode_refused)
    check("media/a response_format refusal still downgrades the mode",
          err2.downgrade is True, str(err2.reason))

    check("media/the built-in probe is large enough for a vision endpoint",
          struct.unpack(">II", D._PROBE_PNG[16:24]) == (64, 64),
          str(struct.unpack(">II", D._PROBE_PNG[16:24])))


def t_vlm_secret_hygiene(tmp) -> None:
    path = _frame(tmp)
    tr = _Transport(_chat(GOOD))
    pr = _provider(tr, provider_name="openrouter", api_key="sk-do-not-log-me")
    pr.analyze(path, "u0001", {})
    check("secrets/the key goes in an authorization header",
          tr.headers[0].get("authorization") == "Bearer sk-do-not-log-me")
    check("secrets/the key is not in the request body",
          "sk-do-not-log-me" not in json.dumps(tr.sent[0]))
    check("secrets/the key is not in the provider's own description",
          "sk-do-not-log-me" not in json.dumps(pr.describe()))
    obs = pr.analyze(path, "u0002", {})
    check("secrets/the key is not in the observation", "sk-do-not-log-me" not in json.dumps(obs.to_dict()))
    key = V._cache_key(Path(path), "u0001", {}, pr)
    check("secrets/the key is not in the cache key", "sk-do-not-log-me" not in key)


def t_vlm_smoke_test(tmp) -> None:
    path = _frame(tmp)
    code, rep = V.smoke_test(path, V.VLMConfig())
    check("smoke/no provider configured is reported as not configured",
          code == 3 and rep["status"] == "configured=False" and rep["configured"] is False)

    code, rep = V.smoke_test(str(tmp / "missing.jpg"), V.VLMConfig(provider="ollama"))
    check("smoke/a missing frame is reported, not crashed on",
          code == 3 and rep["status"] == "frame_missing")

    code, rep = V.smoke_test(path, V.VLMConfig(provider="openai", api_key=None))
    check("smoke/a provider with no key is 'provider_unavailable'",
          code == 3 and rep["status"] == "provider_unavailable", json.dumps(rep)[:200])

    tr = _Transport(_chat(GOOD))
    pr = V.OpenAICompatProvider(model="m", api_key="k", provider_name="ollama", transport=tr)
    orig = V.provider_from_config
    V.provider_from_config = lambda cfg, observations=None: pr
    try:
        code, rep = V.smoke_test(path, V.VLMConfig(provider="ollama"))
        check("smoke/a good call exits zero and says so",
              code == 0 and rep["status"] == "ok" and rep["configured"] is True, json.dumps(rep)[:200])
        check("smoke/the result names the provider and the model",
              rep["provider"] == "ollama" and rep["provider_description"]["model"] == "m")
        check("smoke/the result counts the structure it got back",
              rep["counts"]["numbers"] == 1)

        tr2 = _Transport(_chat("not json"))
        pr2 = V.OpenAICompatProvider(model="m", api_key="k", provider_name="ollama", transport=tr2)
        V.provider_from_config = lambda cfg, observations=None: pr2
        code, rep = V.smoke_test(path, V.VLMConfig(provider="ollama"))
        check("smoke/a malformed answer exits non-zero",
              code == 1 and rep["status"] == "malformed", json.dumps(rep)[:200])

        # Inside the stubbed block, where it belongs. This check sat after the
        # `finally` restored the real `provider_from_config`, so it built a real
        # `OllamaProvider` with no transport and opened a socket to
        # `localhost:11434` — the one place the offline suite left the machine.
        # It passed either way (nothing is usually listening, and the refusal
        # reads the same as a refusal), which is why it went unnoticed; on a
        # machine with Ollama running it would have spent a real model call and
        # asserted whatever that model said.
        V.provider_from_config = lambda cfg, observations=None: pr2
        code, rep = V.smoke_test(path, V.VLMConfig(provider="ollama", api_key="sk-secret"))
        check("smoke/the key is not in the smoke report",
              "sk-secret" not in json.dumps(rep), "the smoke report carried the credential")
    finally:
        V.provider_from_config = orig
