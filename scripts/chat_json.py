"""One structured-JSON negotiator, shared by every model call in this repo.

Both model-backed stages need exactly the same thing and got it separately
before this module existed: a `/chat/completions` request that asks for a strict
JSON schema, discovers whether the endpoint honours that parameter, says so when
it does not, and refuses a response that does not match the schema that was sent.
The VLM pass needed it for observations; the synthesis pass needs it for report
drafts. Two copies of that loop would be two answers to "is structured output
enforced here?", and the second one to be written is the one that gets it wrong.

Three rules the shape exists to hold:

* **Requesting is not enforcing.** `structured_mode` says which constraint was
  asked for; `schema_enforced` says the returned payload actually carried every
  required key. A gateway that accepts `response_format` and answers in prose
  returns HTTP 200 and a perfectly valid conversation, so the only evidence is
  the payload itself.
* **Refusal is a first-class outcome.** A non-conforming answer is returned as a
  classified failure, never coerced. There is no fuzzy field-name repair here,
  and there should not be: a `structure` key mapped onto `visual_elements` is a
  guess, and a guess that is silently accepted is how a fabricated table reaches
  a report. A model that will not answer in the shape stays unqualified.
* **The credential never leaves the header.** It is not logged, not written into
  an artifact, and not part of any cache key.

`transport` is injectable so negotiation, classification and parsing are
exercisable with no network and no credentials — which is the only way they get
exercised on every run rather than on the days a key happens to be valid.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

#: How the response was constrained, strongest first. Each step down is a real
#: loss of guarantee and is recorded on the answer, because "the model was told
#: to return JSON" and "the endpoint enforced this JSON schema" are not the same
#: claim and a reader comparing two runs needs to know which one happened.
STRUCTURED_MODES = ("json_schema", "json_object", "prompt_only")

#: The most a structured call will ever ask for. A model that writes a whole
#: report as JSON answers past a small budget, and the response it was in the
#: middle of writing is a good one, so a truncated answer is retried once at
#: double the budget rather than recorded as a broken contract. Defined here
#: rather than in the VLM pass because this is the lower layer: the VLM client
#: negotiates image modes itself but borrows this ceiling so the two paths have
#: one number between them.
MAX_TOKENS_CEILING = 16000

#: Base URLs for the endpoints worth naming. Presets, not branches: everything
#: below is one wire format, so an endpoint nobody has heard of is a `--base-url`
#: away rather than a class to write. Keyed by the provider alias the CLI accepts.
ENDPOINTS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "llamacpp": "http://localhost:8080/v1",
}

#: Providers reachable without a credential. A local server on `localhost:11434`
#: has no account system, and demanding an API key for it would make a complete
#: configuration report as incomplete.
KEYLESS_PROVIDERS = frozenset({"ollama", "lmstudio", "llamacpp"})

#: The aliases that mean "an OpenAI-compatible endpoint". They differ in default
#: address and in whether a credential is required, and in nothing else — which
#: is why they are one class and one wire format rather than five.
OPENAI_COMPATIBLE = ("openai", "openrouter", "ollama", "lmstudio", "llamacpp", "custom")

#: Words that mean the endpoint does not speak this structured-output shape.
#: Narrow on purpose. It once included "unsupported" and "invalid_request_error",
#: which every OpenAI 400 carries whatever the real problem is — including
#: `image_parse_error: "you uploaded an unsupported image"`. A vision endpoint
#: rejecting a 1×1 picture was therefore classified as a structured-output
#: refusal: the negotiator walked all three modes for nothing, and the doctor
#: reported a working provider as unconfigured with a reason that named the
#: wrong subsystem entirely.
_MODE_UNSUPPORTED = ("response_format", "json_schema", "json_object",
                     "not supported", "unrecognized", "unknown field")

#: A rejection of the *media* rather than of the contract. It shares the 400 and
#: often the word "unsupported" with the case above, so it is checked first and
#: reported as what it is.
_MEDIA_REJECTED = ("image", "image_url", "media_type", "audio", "video_input",
                   "unsupported image", "invalid image")

#: An authenticated account that cannot pay. A distinct class because it is the
#: one failure a user fixes by doing something other than changing the model, the
#: schema or the prompt — and because it is the one that most often arrives
#: disguised. OpenAI answers HTTP 429 with `insufficient_quota`, some gateways
#: answer HTTP 402, and several answer HTTP 400 with a billing message; all three
#: mean the same thing and none of them means what a rate limit means.
_OUT_OF_CREDITS = ("insufficient_quota", "insufficient credit", "exceeded your current quota",
                   "billing", "payment required", "out of credits", "no credits",
                   "credit balance", "spending limit", "quota exceeded",
                   "purchase more credits", "add credits")

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


class ChatError(Exception):
    """A classified model-call failure.

    `fatal` means stop asking for the rest of this run (a bad credential will not
    fix itself). `downgrade` means the endpoint rejected the structured-output
    parameter specifically, which is a negotiation step down rather than a
    failure — the distinction is the whole reason this is a type and not a
    string.
    """

    def __init__(self, status: str, reason: str, fatal: bool = False,
                 downgrade: bool = False):
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.fatal = fatal
        self.downgrade = downgrade


def _error_text(exc: urllib.error.HTTPError) -> tuple[str, str]:
    """The message an error carries, and everything searchable beneath it.

    A gateway that fronts several providers buries the real complaint: the
    top-level `error.message` reads "Provider returned error" while the
    parameter that was actually rejected sits in `error.metadata.raw`, sometimes
    several nesting levels down. Classifying on the top line alone turns a
    recoverable negotiation step into an unexplained failure, so the whole
    decoded body is returned for matching and the top message is kept for the
    message a human reads.
    """
    detail = ""
    try:
        body = exc.read().decode("utf-8", "replace")
        doc = json.loads(body)
    except Exception:  # noqa: BLE001 — the body is best-effort context only
        return "", str(exc.reason or "")
    if isinstance(doc, dict):
        detail = (doc.get("error", {}) or {}).get("message", "") or ""
    return detail, body or str(exc.reason or "")


def classify_http(exc: urllib.error.HTTPError) -> ChatError:
    """An HTTP status, translated into what it means for a later run.

    Authentication will not fix itself and stops the pass immediately; an
    exhausted balance is a different thing again and is fatal for the same
    reason but fixed by a different action; a rate limit is transient; a 4xx
    naming `response_format` is a shape the endpoint does not speak, and asking
    differently is the only thing that helps; every other status is a failure of
    that one call. None of them is cached.

    The balance check comes before the rate-limit check on purpose. Both arrive
    as HTTP 429 from the two providers most people use, and they are opposite
    problems: one clears in a minute, the other needs a payment. Filing a
    billing refusal as a transient rate limit tells the user to wait, which is
    the one action guaranteed not to help.
    """
    detail, haystack = _error_text(exc)
    text = haystack.lower()
    if any(m in text for m in _OUT_OF_CREDITS):
        return ChatError("out_of_credits",
                         f"HTTP {exc.code}: the provider accepted the credential but the "
                         f"account cannot pay for this call ({detail or exc.reason})",
                         fatal=True)
    if exc.code in (401, 403):
        return ChatError("unavailable", f"HTTP {exc.code}: {detail or 'authentication failed'}",
                         fatal=True)
    if exc.code == 429:
        return ChatError("failed", f"HTTP 429 rate limited: {detail}")
    if 400 <= exc.code < 500 and any(m in text for m in _MEDIA_REJECTED):
        return ChatError("failed",
                         f"HTTP {exc.code}: the endpoint rejected the image "
                         f"({detail or exc.reason})")
    if 400 <= exc.code < 500 and any(m in text for m in _MODE_UNSUPPORTED):
        return ChatError("failed",
                         f"HTTP {exc.code}: endpoint rejected the structured-output "
                         f"parameter ({detail or exc.reason})", downgrade=True)
    if exc.code == 404:
        return ChatError("failed",
                         f"HTTP 404: model or endpoint not found ({detail or exc.reason})",
                         fatal=True)
    return ChatError("failed", f"HTTP {exc.code}: {detail or exc.reason}")


def http_send(url: str, headers: dict, data: bytes, timeout: float) -> str:
    """The default transport: send bytes, return the body. No judgement."""
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def content_text(payload: dict) -> str | None:
    """The assistant's own text, wherever this endpoint family put it.

    `content` is a string on most servers and a list of parts on some; a few put
    the payload in a tool call's arguments instead. Reading all three is not
    leniency — an endpoint that answered in a tool call when none was offered
    has still answered, and discarding it would turn a usable response into a
    spurious "malformed".
    """
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = [p.get("text") for p in content
                 if isinstance(p, dict) and isinstance(p.get("text"), str)]
        joined = "".join(parts).strip()
        if joined:
            return joined
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict) and isinstance(call.get("function"), dict):
            args = call["function"].get("arguments")
            if isinstance(args, str) and args.strip():
                return args
    return None


def no_text_reason(payload: dict) -> tuple[str, str]:
    """Why there is no text, said as a status and a reason rather than a shrug."""
    choices = (payload or {}).get("choices") if isinstance(payload, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish = choices[0].get("finish_reason")
        if finish == "content_filter":
            return "failed", "response was withheld by a content filter"
        if finish == "length":
            return "failed", "response hit the token limit before any content"
        return "failed", f"response carried no content (finish_reason={finish!r})"
    return "failed", "response had no choices"


def finish_reason(payload: dict) -> str | None:
    choices = (payload or {}).get("choices") if isinstance(payload, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0].get("finish_reason")
    return None


def strict_json(text: str) -> dict:
    """Parse an assistant's answer as one JSON object, or raise.

    A markdown fence is stripped because that is a formatting habit rather than
    a disagreement about content. Nothing else is repaired: no substring scan for
    a brace, no trailing-comma tolerance, no key-name guessing. Every one of those
    turns a model that did not answer the question into a record that looks like
    it did, and this repo's whole failure mode is a record that looks complete
    and is not.
    """
    if not text:
        raise ValueError("empty response")
    fence = _FENCE.match(text)
    if fence:
        text = fence.group(1)
    doc = json.loads(text)
    if not isinstance(doc, dict):
        raise ValueError(f"response is {type(doc).__name__}, not an object")
    return doc


def conforms_to_schema(payload: dict, mode: str | None, required) -> bool:
    """Did the answer actually carry the schema that was sent?

    For all three modes, including `prompt_only`. The test is the schema's own
    `required` list, because that is the contract — a response missing a required
    field is one the endpoint did not deliver, whatever the HTTP status said.

    `prompt_only` used to short-circuit to `True` here, on the reasoning that
    there is nothing to enforce against. But asking for JSON in a prompt is not
    a weaker test of the answer, it is the *same* test with nothing guaranteeing
    it, and returning True unconditionally meant a prompt-only response that was
    prose was recorded as conforming. What `prompt_only` cannot claim is
    *enforcement* — that is the separate `schema_enforced` flag below, which is
    now False in this mode whatever the answer looked like. Conformance and
    enforcement are two different questions and conflating them is how a mode
    that guarantees nothing got reported as a mode that guarantees everything.
    """
    text = content_text(payload)
    if not text:
        return False
    fence = _FENCE.match(text)
    if fence:
        text = fence.group(1)
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(doc, dict):
        return False
    return all(k in doc for k in required)


class StructuredChat:
    """A text-only `/chat/completions` client that negotiates a JSON schema.

    The VLM pass keeps its own image-bearing class — it has to, because a frame
    is a different request body — but it delegates the negotiation to this one,
    so "was the schema enforced?" is answered by one implementation rather than
    two that happen to agree today.
    """

    def __init__(self, model: str, base_url: str, api_key: str = "",
                 timeout: float = 90.0, max_tokens: int = 4000,
                 schema_name: str = "response", schema: dict | None = None,
                 required=(), system: str = "", transport=None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.schema_name = schema_name
        self.schema = schema
        self.required = tuple(required)
        self.system = system
        self._transport = transport or http_send
        #: Latched once a mode is accepted, so negotiation is paid once per client
        #: rather than once per call.
        self.structured_mode: str | None = None
        #: Whether a returned response actually matched the schema that was sent.
        #: `None` until the first answer arrives. A gateway that accepts
        #: `response_format` and then ignores it is common enough that assuming
        #: enforcement would be a documented lie in every artifact.
        self.schema_enforced: bool | None = None
        self._auth_failed = False

    @property
    def available(self) -> bool:
        return not self._auth_failed

    def describe(self) -> dict:
        """What this client can do, for the artifact. Never the credential."""
        return {"model": self.model, "base_url": self.base_url,
                "structured_mode": self.structured_mode,
                "schema_enforced": self.schema_enforced,
                "api_key_present": bool(self.api_key)}

    def _body(self, messages: list[dict], mode: str) -> dict:
        body: dict = {"model": self.model, "max_tokens": self.max_tokens,
                      "temperature": 0, "messages": messages}
        if self.system:
            body["messages"] = [{"role": "system", "content": self.system}, *messages]
        if mode == "json_schema" and self.schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": self.schema_name, "strict": True,
                                "schema": self.schema}}
        elif mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        return body

    def complete(self, messages: list[dict], max_tokens: int | None = None) -> tuple[dict, str]:
        """Return `(payload, mode)`, negotiating once and latching the result.

        Raises `ChatError` on every failure, including a response that arrived
        fine and does not match the schema: a caller that wanted a validated
        object and got an unvalidated one has to be told, not handed a dict.
        """
        order = [self.structured_mode] if self.structured_mode else list(STRUCTURED_MODES)
        last: ChatError | None = None
        saved = self.max_tokens
        if max_tokens is not None:
            self.max_tokens = max_tokens
        try:
            for mode in order:
                try:
                    payload = self._post(self._body(messages, mode))
                except ChatError as exc:
                    if exc.fatal:
                        self._auth_failed = True
                        raise
                    if exc.downgrade and mode != STRUCTURED_MODES[-1]:
                        last = exc
                        continue
                    raise
                self.structured_mode = mode
                if finish_reason(payload) == "length" and self.max_tokens < MAX_TOKENS_CEILING:
                    # A model that writes a full report as JSON answers past a
                    # small budget, and the answer it was in the middle of
                    # writing is a good one. Truncation is the budget's fault,
                    # not the model's, and it arrives as `failed` when the cut
                    # lands before any content and as unparsable JSON when it
                    # lands inside a string — so the retry is keyed on the
                    # finish reason, which is the one fact both shapes agree on.
                    # The VLM pass has the same rule and the same reason for it;
                    # it lives there because that client negotiates the image
                    # modes itself rather than coming through here.
                    self.max_tokens = MAX_TOKENS_CEILING
                    payload = self._post(self._body(messages, mode))
                if not conforms_to_schema(payload, mode, self.required):
                    # The endpoint accepted the parameter and the model ignored
                    # it. Recording this as enforced is the silent semantic
                    # fallback this whole module exists to prevent, so it is
                    # verified rather than assumed, and the next call steps
                    # *down* the ladder rather than retrying the mode that was
                    # just proven ineffective.
                    self.schema_enforced = False
                    if mode in STRUCTURED_MODES[:-1]:
                        self.structured_mode = STRUCTURED_MODES[
                            STRUCTURED_MODES.index(mode) + 1]
                else:
                    # Conforming, yes. Enforced, only if the endpoint was the
                    # thing that made it so. `prompt_only` is the bottom rung and
                    # gets its JSON by being asked politely in the instruction,
                    # so a well-formed answer from it is luck the next call is
                    # equally exposed to. It is reported as unenforced so the
                    # caller can say "not enforced on this endpoint" instead of
                    # claiming a contract the provider never agreed to.
                    self.schema_enforced = mode != STRUCTURED_MODES[-1]
                return payload, mode
        finally:
            self.max_tokens = saved
        raise last or ChatError("failed", "no usable response mode")

    def complete_json(self, messages: list[dict], max_tokens: int | None = None) -> tuple[dict, str]:
        """`complete`, then the strict parse. The pair callers almost always want.

        The two failures stay distinct because they mean different things to a
        caller deciding whether to retry: no content and unparseable content are
        different diagnoses, and a repair request is only sensible for one.
        """
        payload, mode = self.complete(messages, max_tokens=max_tokens)
        text = content_text(payload)
        if not text:
            status, reason = no_text_reason(payload)
            raise ChatError(status, reason)
        try:
            return strict_json(text), mode
        except (json.JSONDecodeError, ValueError) as exc:
            raise ChatError("malformed",
                            f"assistant text was not a JSON object: {exc} "
                            f"(structured mode {mode!r})") from exc

    def _post(self, body: dict) -> dict:
        """Send, then classify. Nothing leaves this method unclassified.

        The transport is raw on purpose — it moves bytes and raises HTTP errors;
        every judgement about what an error *means* lives here, so an injected
        transport exercises the real classification rather than a bypass around
        it. That is the difference between a test double that proves the
        behaviour and one that only proves the code runs.
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
            raise classify_http(exc) from exc
        except TimeoutError as exc:
            raise ChatError("failed", f"timeout after {self.timeout}s") from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                raise ChatError("failed", f"timeout after {self.timeout}s") from exc
            raise ChatError("failed", f"connection failed: {reason}") from exc
        except OSError as exc:  # socket errors that are not URLError subclasses
            raise ChatError("failed", f"connection failed: {exc}") from exc
        try:
            payload = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as exc:
            raise ChatError("malformed", f"response was not JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ChatError("malformed", f"response is {type(payload).__name__}, not an object")
        return payload


def resolve_endpoint(provider: str, base_url: str | None) -> str | None:
    """The base URL for a provider alias, or None when it needs one supplied.

    `custom` is the escape hatch for an OpenAI-compatible endpoint nobody has a
    preset for: it has no default address on purpose, so a misconfiguration
    reports itself instead of quietly resolving to OpenAI's.
    """
    if provider == "custom" and not base_url:
        return None
    return (base_url or ENDPOINTS.get(provider) or ENDPOINTS["openai"]).rstrip("/")


#: Which environment variable holds the credential for each alias, and nothing
#: else. Strictly per-provider, with no fallback from one service's variable to
#: another's: a key is scoped to the service that issued it, and a lookup that
#: fell through to a neighbouring provider's variable would put a credential in
#: an `Authorization` header aimed at a host that never issued it. Reporting
#: "no credential configured" when the right variable is empty is a far better
#: failure than sending the wrong secret to the wrong endpoint.
_KEY_VARS = {
    "openai": ("VLM_API_KEY", "OPENAI_API_KEY"),
    "openrouter": ("VLM_API_KEY", "OPENROUTER_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    # An endpoint nobody has a preset for. Only the generic variable applies:
    # guessing which hosted service a custom URL belongs to would send a
    # credential somewhere it was never meant to go.
    "custom": ("VLM_API_KEY", "SYNTH_API_KEY"),
}


def key_for(provider: str, base_url: str | None = None) -> str | None:
    """Read the credential for a provider alias, if it wants one.

    Keyless providers get `""` rather than a lookup: a local server that runs on
    `localhost` has no account system, and demanding a key for it would make a
    complete configuration report as incomplete.
    """
    if provider in KEYLESS_PROVIDERS:
        return ""
    for var in _KEY_VARS.get(provider, ("VLM_API_KEY",)):
        val = os.environ.get(var)
        if val:
            return val
    return None
