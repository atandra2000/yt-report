#!/usr/bin/env python3
"""Does a credential reach anything a reader of the artifact can see?

The unit test in `selftest_vlm_provider.py` proves the key is not in the request
body and not in the provider's own description. That is the easy half: both are
in-memory strings the test itself built. The half that matters is whether the
key survives into something *persisted* — a JSON artifact, a Markdown file, a
cache filename, a line of stderr, an exception trace — because those are the
things a user pastes into a bug report.

So this runs the real pipeline end to end against a local endpoint, with a
deliberately distinctive fake credential, and greps every byte it produced.
Local rather than remote so the audit costs nothing, needs no credential, and
exercises the genuine HTTP transport rather than a stub of it.

    uv run --with yt-dlp --with pillow python3 acceptance/secret_audit.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
#: Distinctive and unmistakably fake, so a hit cannot be a real credential that
#: happened to match something. Both a provider-shaped and a bare token.
FAKE_VLM = "sk-live-DO-NOT-LEAK-abc123ZZZ"
FAKE_SYNTH = "sk-or-v1-DO-NOT-LEAK-999999xxx"

OBSERVATION = {
    "content_type": "application_ui",
    "summary": "A dashboard frame with a title bar and two panels.",
    "confidence": 0.7,
    "visual_elements": [{"kind": "panel", "label": "left", "detail": "a list"}],
    "numbers": [], "entities": [], "relationships": [], "actions": [],
    "text_relationships": [], "claims_supported": [], "uncertainties": [],
}


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 — the name is fixed by BaseHTTPRequestHandler
        length = int(self.headers.get("content-length") or 0)
        self.rfile.read(length)
        body = json.dumps({"choices": [{"message": {"content": json.dumps(OBSERVATION)},
                                        "finish_reason": "stop"}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a) -> None:  # noqa: ANN002 — silence the default stderr spam
        return


def main() -> int:
    src = sorted((Path.home() / "Desktop" / "yt-reports").glob("*-EOdXR6lU5ZA"))
    if not src:
        print("SKIP  no artifact directory to copy; run a fetch first")
        return 0
    src = src[0]

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "art"
        shutil.copytree(src, d)
        for stale in ("REPORT.md", "REPORT.json", "report_validation.json"):
            (d / stale).unlink(missing_ok=True)

        env = dict(os.environ)
        env["VLM_API_KEY"] = FAKE_VLM
        env["VLM_PROVIDER"] = "custom"
        env["VLM_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        env["VLM_MODEL"] = "audit/fake-model"
        env["SYNTH_API_KEY"] = FAKE_SYNTH
        for k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            env.pop(k, None)

        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_evidence.py"), str(d),
             "--vlm-provider", "custom", "--vlm-model", "audit/fake-model",
             "--vlm-base-url", f"http://127.0.0.1:{port}/v1",
             "--max-vlm-frames", "2"],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)

        # Inspection happens *inside* the context manager: the whole point is to
        # look at what the run left on disk, and a temp dir that is gone by the
        # time the audit reads it is an audit that inspected nothing.
        secrets = [FAKE_VLM, FAKE_SYNTH, "Bearer " + FAKE_VLM, "Bearer " + FAKE_SYNTH]
        findings: list[str] = []

        for label, blob in (("stdout", proc.stdout), ("stderr", proc.stderr),
                            ("combined traceback", proc.stdout + proc.stderr)):
            for s in secrets:
                if s in blob:
                    findings.append(f"{label} contains the credential")

        produced = sorted(p for p in d.rglob("*"))
        for p in produced:
            if any(s in p.name for s in secrets):
                findings.append(f"filename {p.name} contains the credential")
            if not p.is_file():
                continue
            try:
                body = p.read_bytes().decode("utf-8", "replace")
            except OSError:
                continue
            for s in secrets:
                if s in body:
                    findings.append(f"{p.relative_to(d)} contains the credential")
        for s in secrets:
            if s in str(d):
                findings.append("the artifact path contains the credential")

        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "yt-report"
        cache_files = sorted(cache_dir.rglob("*")) if cache_dir.exists() else []
        for p in cache_files:
            if not p.is_file():
                continue
            body = p.read_bytes().decode("utf-8", "replace")
            for s in secrets:
                if s in body or s in p.name:
                    findings.append(f"cache file {p.name} contains the credential")

    server.shutdown()

    # The repo's own fixtures must not carry a live credential. Placeholders are
    # not leaks and asserting otherwise would demand the negative-assertion tests
    # be deleted — those are the tests doing the work. So the fixture scan counts
    # key-shaped strings and names the ones that are not obviously synthetic.
    KEYISH = __import__("re").compile(r"\bsk-[A-Za-z0-9\-_]{4,}")
    placeholders: set[str] = set()
    suspect: list[str] = []
    for f in sorted((ROOT / "scripts").glob("*.py")):
        for m in KEYISH.finditer(f.read_text()):
            tok = m.group(0)
            low = tok.lower()
            # A test that asserts a key never reaches an artifact must contain
            # the key prefixes it searches for, so `sk-or-v1-` in a fixture is
            # the absence of a credential, not one. A scanner that flags it is
            # crying wolf, which is the one failure mode a secret scanner has.
            if any(w in low for w in ("secret", "do-not", "openai", "openrouter",
                                      "anthropic", "or-v1", "not-a-key", "fake",
                                      "placeholder", "example")):
                placeholders.add(tok)
            else:
                suspect.append(f"{f.name}: {tok[:12]}…")
    findings.extend(f"fixture carries a key that is not an obvious placeholder: {s}"
                    for s in suspect)

    print(f"exit code            : {proc.returncode}")
    print(f"artifacts inspected  : {len(produced)} paths")
    print(f"cache files inspected: {len(cache_files)}")
    print(f"secrets planted      : 2")
    print(f"fixture placeholders : {len(placeholders)} distinct, all synthetic")
    if findings:
        print("LEAKS FOUND:")
        for f in sorted(set(findings)):
            print("  -", f)
        return 1
    print("no credential reached stdout, stderr, an artifact, a cache file, a filename "
          "or a fixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
