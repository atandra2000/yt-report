"""The grounding gate, exercised against fourteen deliberate corruptions.

`selftest_report.py` proves the report layer works on clean evidence. This file
exists because that is the easier half: a rule that has never been seen to fire
is not known to fire, and a report gate that has only ever been handed documents
it generated itself has been tested against one input — its own.

So each case here takes a *valid* report from a fixture, corrupts it in one
specific way, and asserts three separate things:

* the validator **detects** it, by the specific error code, not merely "not
  valid" — `False` is equally the answer for a validator that crashed;
* the error is **useful** — it names what is wrong and where, so a reader can
  act on it without re-deriving the failure;
* the corruption is **not silently repaired** — nothing in the pipeline quietly
  restores the original text, so a defect that reached the file is a defect that
  stayed.

Where the defect is one the pipeline's own write gate must refuse, the case also
runs the real `write_and_validate_report` and asserts that no `REPORT.md` is
left behind. A gate that detects a defect but still writes the file is a gate
that has been reported, not enforced.

The fourteen are the acceptance set: unsupported claim, number, URL, visual
statement, invalid timestamp, nonexistent evidence id, dropped conflict,
removed hedge, duplicated section, excessive repetition, fabricated entity,
unsupported comparison, unsupported procedural step, malformed synthesis JSON.
Most are caught at the report gate. The last belongs to the synthesis gate,
which is where a model's *response* is judged, and the case says so rather than
forcing it through a check it does not belong to.
"""
from __future__ import annotations

import io
import json
import re
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import chat_json as CJ  # noqa: E402
import report as R  # noqa: E402
import report_fixtures as FX  # noqa: E402
import report_validate as RV  # noqa: E402
import selftest_synthesis as SS  # noqa: E402
import synth_model as SM  # noqa: E402
import synth_validate as SV  # noqa: E402
from evidence_model import save_evidence  # noqa: E402

#: Bound by `selftest.py` at import, exactly as the other suites are.
check = None


# --------------------------------------------------------------------------- #
# the harness
# --------------------------------------------------------------------------- #

def _built(name: str = "technical_lecture", detail: str = "balanced"):
    ev = FX.FIXTURES[name]()
    rep = R.generate(ev, detail)
    return ev, rep, R.render_markdown(rep)


def _codes(res) -> set[str]:
    return {e["code"] for e in res.blocking_errors}


def _message(res, code: str) -> str:
    for e in res.blocking_errors:
        if e["code"] == code:
            return e["message"]
    return ""


def _all_messages(res, code: str) -> str:
    """Every message for one code. A single fabricated sentence trips the check
    once per invented word, so asserting on the first message only proves that
    *some* word was caught, not the one the case is about."""
    return " | ".join(e["message"] for e in res.blocking_errors if e["code"] == code)


def _assert_detected(label: str, ev, rep, md, code: str,
                     authored_by: str = "deterministic") -> RV.ReportValidation:
    """The defect must be caught, by name, with a message a reader can act on.

    Asserting only `not res.valid` would pass for a validator that raised, and
    for one that fired on an unrelated rule — which is how a green suite can sit
    on top of a check that stopped working.
    """
    res = RV.validate_report(rep, ev, md, authored_by=authored_by)
    codes = _codes(res)
    check(f"{label}: detected", code in codes, f"expected {code!r}, got {sorted(codes)}")
    msg = _message(res, code)
    check(f"{label}: the error says what and where",
          bool(msg) and len(msg) > 20, f"no useful message for {code!r}")
    return res


# --------------------------------------------------------------------------- #
# 1-8: corruptions of a deterministically rendered report
# --------------------------------------------------------------------------- #

def t_unsupported_factual_claim() -> None:
    """A sentence asserting four things, none of them in the evidence.

    Deliberately free of proper nouns, figures, URLs and markers, because every
    one of those is already checked. This is the shape that slips through: an
    ordinary-looking English sentence that the evidence never says.
    """
    ev, rep, md = _built()
    forged = ("The kernel scheduler was rewritten from scratch to reduce tail "
              "latency under sustained load.")
    rep.sections[0].blocks.append(
        R.Block(forged, [rep.evidence_ids[0]], "[01:15 · narration]", kind="narrative"))
    res = _assert_detected("1 unsupported factual claim", ev, rep, R.render_markdown(rep),
                           "claim-unsupported")
    check("1 the message names the invented word",
          any(w in _all_messages(res, "claim-unsupported")
              for w in ("kernel", "scheduler", "latency")),
          _all_messages(res, "claim-unsupported")[:200])


def t_unsupported_number() -> None:
    ev, rep, md = _built()
    if "**120 ms**" not in md:
        check("1b fixture carries a figure to corrupt", False, "no **120 ms** to change")
        return
    _assert_detected("2 unsupported number", ev, rep, md.replace("**120 ms**", "**999 ms**"),
                     "number-unsupported")
    check("2 the validator does not just refuse every number",
          not any(e["code"] == "number-unsupported"
                  for e in RV.validate_report(rep, ev, md).blocking_errors),
          "a clean report was already failing the number check")


def t_unsupported_url() -> None:
    ev, rep, md = _built()
    _assert_detected("3 unsupported URL", ev, rep,
                     md + "\nSee https://invented.example.com/never for details.\n",
                     "url-unsupported")


def t_unsupported_visual_statement() -> None:
    """A chart described, with nothing in the citation that ever saw a frame."""
    ev, rep, md = _built()
    speech_only = [i for i in ev.items
                   if all(p.source == "transcript" for p in (i.provenance or []))]
    if not speech_only:
        check("4 fixture has a narration-only item to misattribute", False, "none found")
        return
    forged = R.Block("The diagram shows a retrieval pipeline end to end.",
                     [speech_only[0].id], "[01:15 · narration]", kind="visual")
    rep.sections[2].subsections[0].blocks.append(forged)
    _assert_detected("4 unsupported visual statement", ev, rep, R.render_markdown(rep),
                     "visual-unsupported")


def t_invalid_timestamp() -> None:
    ev, rep, md = _built()
    _assert_detected("5a fabricated timestamp", ev, rep,
                     md.replace("[01:15 · narration]", "[44:44 · narration]"),
                     "timestamp-fabricated")
    # A marker that is citation-shaped but unreadable is a different defect with
    # a different code. `[99:99]` is not it — that parses to 6039 s and is a
    # fabrication, because it is past the end of the video.
    _assert_detected("5b unparsable timestamp", ev, rep,
                     md.replace("[01:15 · narration]", "[01:5 · narration]"),
                     "timestamp-unparsable")
    # And the shape that made the check reachable at all: `_MARKER` only matches
    # `\d{1,2}:\d{2}`, so before the loose scan this one was invisible.
    _assert_detected("5c three-field timestamp", ev, rep,
                     md.replace("[01:15 · narration]", "[00:01:15 · narration]"),
                     "timestamp-unparsable")


def t_nonexistent_evidence_id() -> None:
    ev, rep, md = _built()
    rep.sections[0].blocks.append(
        R.Block("Something happened.", ["itm-999"], "[01:15 · narration]", kind="narrative"))
    _assert_detected("6 nonexistent evidence id", ev, rep, R.render_markdown(rep),
                     "evidence-id-unresolved")


def t_dropped_unresolved_conflict() -> None:
    """Not silently resolved, not dropped — both are refusals, and differently."""
    ev, rep, md = _built("unresolved_conflict")
    rep.sections = [s for s in rep.sections if s.key != "reliability"] + [R.Section(
        "reliability", "Source & Reliability",
        blocks=[R.Block("Evidence by source — narration: 2.", kind="meta")])]
    res = _assert_detected("7 dropped unresolved conflict", ev, rep, R.render_markdown(rep),
                           "conflict-dropped")
    check("7 the message names the conflict", "conflict" in _message(res, "conflict-dropped").lower(),
          _message(res, "conflict-dropped"))


def t_removed_hedge() -> None:
    """"could cost $500" rendered as "costs $500" — the same number, a false claim."""
    ev, rep, md = _built("hedged_claim")
    hits = [b for _, b in RV._all_blocks(rep) if "itm-003" in b.evidence_ids]
    if not hits:
        check("8 fixture carries a hedged item", False, "no block cites itm-003")
        return
    original = hits[0].text
    hits[0].text = re.sub(r"\s*\(could\)", "", original)
    hits[0].text = hits[0].text.replace("given as a possibility", "confirmed")
    res = _assert_detected("8 removed hedge", ev, rep, R.render_markdown(rep), "hedge-dropped")
    check("8 the number itself still passes — the hedge is the defect",
          not any(e["code"] == "number-unsupported" for e in res.blocking_errors),
          "the figure was flagged too, so this is not isolating the hedge check")


def t_duplicated_factual_section() -> None:
    ev, rep, md = _built()
    originals = [s for s in rep.sections if s.key == "detailed_notes"]
    if not originals:
        check("9 fixture has a narrative section to duplicate", False, "none")
        return
    s = originals[0]
    rep.sections.append(R.Section(s.key, s.title, blocks=list(s.blocks), subsections=s.subsections))
    res = _assert_detected("9 duplicated factual section", ev, rep, R.render_markdown(rep),
                           "section-duplicated")
    check("9 the message names the section twice",
          s.key in _message(res, "section-duplicated"), _message(res, "section-duplicated"))


def t_excessive_repetition() -> None:
    ev, rep, md = _built("simple_talking_head")
    body = [b for s in rep.sections for b in s.blocks if b.text]
    if not body:
        check("10 fixture has blocks to repeat", False, "none")
        return
    rep.sections[0].blocks = body * 6
    res = _assert_detected("10 excessive repetition", ev, rep, R.render_markdown(rep),
                           "repetition-excessive")
    check("10 the message carries the measured ratio",
          "%" in _message(res, "repetition-excessive"), _message(res, "repetition-excessive"))


def t_fabricated_entity() -> None:
    ev, rep, md = _built("simple_talking_head")
    rep.sections.append(R.Section("people", "People & Organizations", blocks=[
        R.Block("The Quantax Engine powers the whole pipeline.", ["itm-004"],
                "[00:12 · narration]", kind="narrative")]))
    res = _assert_detected("11 fabricated entity", ev, rep, R.render_markdown(rep),
                           "claim-unsupported")
    check("11 the message names the invented entity",
          "Quantax" in _message(res, "claim-unsupported"), _message(res, "claim-unsupported"))


def t_unsupported_comparison() -> None:
    ev, rep, md = _built()
    rep.sections.append(R.Section("cmp", "Comparisons", blocks=[
        R.Block("Redis is dramatically faster than Postgres for this workload.",
                [rep.evidence_ids[0]], "[01:15 · narration]", kind="comparison")]))
    _assert_detected("12 unsupported comparison", ev, rep, R.render_markdown(rep),
                     "claim-unsupported")
    # A comparison in words only, with no proper noun to catch it, is the version
    # that used to pass — so it gets its own case rather than riding on the above.
    ev, rep, md = _built()
    rep.sections.append(R.Section("cmp2", "Comparisons", blocks=[
        R.Block("The tuned configuration outperforms the default on every measured axis.",
                [rep.evidence_ids[0]], "[01:15 · narration]", kind="comparison")]))
    _assert_detected("12b comparison with no proper noun", ev, rep, R.render_markdown(rep),
                     "claim-unsupported")


def t_unsupported_procedural_step() -> None:
    ev, rep, md = _built()
    rep.sections.append(R.Section("steps", "Steps & Workflows", blocks=[
        R.Block("First install the toolchain.", [rep.evidence_ids[0]],
                "[01:15 · narration]", kind="step"),
        R.Block("Then rotate the signing keys.", [rep.evidence_ids[0]],
                "[01:15 · narration]", kind="step")]))
    _assert_detected("13 unsupported procedural step", ev, rep, R.render_markdown(rep),
                     "claim-unsupported")


def t_malformed_synthesis_json() -> None:
    """Case 14, and it belongs to the other gate.

    A model's *response* is judged by `synth_validate`, which is the check built
    for prose a renderer did not write. Running it through the report gate would
    be testing the wrong thing, so the case says which gate owns the defect and
    exercises that one.
    """
    ev = FX.simple_talking_head()

    # The decoder a raw model body reaches first. Prose, a bare string and a
    # JSON list are three different shapes of "not a draft", and a decoder that
    # accepts any of them is how prose gets into a report that then claims to be
    # grounded in evidence.
    for label, raw in (("prose instead of JSON", "Here is the report you asked for."),
                       ("a bare JSON string", '"a report"'),
                       ("a JSON list", '[{"text": "x"}]'),
                       ("an empty body", "")):
        try:
            doc = CJ.strict_json(raw)
        except Exception as exc:  # noqa: BLE001 — the point is that it refuses
            check(f"14 {label}: the decoder refuses it", bool(str(exc).strip()),
                  "raised with an empty message")
        else:
            res = SV.validate_draft_grounding(doc, ev)
            check(f"14 {label}: refused, as a blocking finding",
                  not res.valid and "synthesis-malformed" in {f.code for f in res.blocking},
                  f"accepted; codes={sorted({f.code for f in res.findings})}")

    # And a well-formed draft, so the case above is not a check that refuses
    # everything: a parse that never succeeds would pass all four.
    good = SS._draft([SS._unit("The video covers caching.", ["itm-001"])],
                     [SS._unit("A cache sits between the app and the database.", ["itm-002"])],
                     [{"title": "Caching", "paragraphs": [SS._unit("The second read avoids the database.", ["itm-003"])]}])
    parsed = CJ.strict_json(json.dumps(good))
    check("14 a well-formed draft does decode",
          isinstance(parsed, dict) and SM.validate_draft(parsed) == [],
          str(SM.validate_draft(parsed)[:2]))


# --------------------------------------------------------------------------- #
# the write gate: a detected defect must also mean no file
# --------------------------------------------------------------------------- #

def t_gate_blocks_the_write() -> None:
    """A gate that reports a defect and then writes the file has not enforced it.

    The corruption is injected at the renderer rather than on disk, because
    `write_and_validate_report` regenerates the report from evidence and would
    overwrite a doctored `REPORT.md` — so a file edited by hand is not the
    threat this gate has to survive. The threat is a renderer or a model that
    returns a report the gate rejects, and the question is whether the file
    still appears.
    """
    import report as report_mod
    from build_evidence import write_and_validate_report

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        save_evidence(FX.technical_lecture(), d / "evidence.json")
        (d / "REPORT.md").write_text("a report that should never survive\n")

        clean = R.generate(FX.technical_lecture(), "balanced")
        forged = R.generate(FX.technical_lecture(), "balanced")
        forged.sections[0].blocks.append(
            R.Block("The cluster sustained 999 requests per second at peak.",
                    [forged.evidence_ids[0]], "[01:15 · narration]", kind="narrative"))

        original = report_mod.generate
        report_mod.generate = lambda ev, detail: forged
        try:
            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = write_and_validate_report(d, "balanced", quiet=True)
        finally:
            report_mod.generate = original

        check("the write gate returns non-zero on an ungrounded report", rc == 1, str(rc))
        check("the write gate leaves the previous report in place rather than replacing it",
              (d / "REPORT.md").read_text() == "a report that should never survive\n",
              "REPORT.md was overwritten with the ungrounded report")
        check("the write gate says why on stderr", "BLOCK" in buf.getvalue(),
              buf.getvalue()[:200])
        verdict = json.loads((d / "report_validation.json").read_text())
        check("the write gate records the refusal with its codes",
              verdict["valid"] is False
              and "number-unsupported" in {e["code"] for e in verdict["blocking_errors"]},
              json.dumps(verdict.get("blocking_errors"))[:200])
        # And the corruption was not quietly repaired on the way to the verdict.
        check("the corruption is not silently repaired",
              "999" in json.dumps(verdict["blocking_errors"]),
              "the recorded refusal does not name the figure that caused it")


def t_evidence_gate_stops_before_the_report() -> None:
    """The other half of the chain: invalid evidence never reaches the report.

    A citation to a frame that is not in the document — valid JSON, valid schema,
    a pointer that resolves to nothing — is refused by the *evidence* gate, so no
    report is generated and no `report_validation.json` is written. The ordering
    is the property: the second gate must not be the thing that notices.
    """
    from build_evidence import write_and_validate_report

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        save_evidence(FX.technical_lecture(), d / "evidence.json")
        doc = json.loads((d / "evidence.json").read_text())
        doc["items"][0]["provenance"] = [{"source": "frame", "ref": "u9999", "confidence": 0.9}]
        (d / "evidence.json").write_text(json.dumps(doc))

        buf = io.StringIO()
        with redirect_stderr(buf):
            rc = write_and_validate_report(d, "balanced", quiet=True)
        check("the evidence gate refuses before any report exists", rc == 1, str(rc))
        check("no REPORT.md is written from invalid evidence",
              not (d / "REPORT.md").exists())
        check("no report verdict is written, because the report gate never ran",
              not (d / "report_validation.json").exists(),
              "report_validation.json exists, so the report gate ran on invalid evidence")
        check("the refusal names the evidence, on stderr", "BLOCK" in buf.getvalue(),
              buf.getvalue()[:200])


def t_clean_evidence_is_not_repaired() -> None:
    """The counterweight: nothing above is a check that refuses everything.

    A gate that blocks every write is as useless as one that blocks none, so the
    same path is run on uncorrupted evidence and must both pass and leave the
    report it generated.
    """
    from build_evidence import write_and_validate_report

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        ev = FX.technical_lecture()
        save_evidence(ev, d / "evidence.json")
        buf = io.StringIO()
        with redirect_stderr(buf):
            rc = write_and_validate_report(d, "balanced", quiet=True)
        check("the write gate passes clean evidence", rc == 0, buf.getvalue()[:200])
        check("the clean report is the one on disk",
              (d / "REPORT.md").read_text() == R.render_markdown(R.generate(FX.technical_lecture(),
                                                                          "balanced")))
