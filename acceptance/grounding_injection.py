#!/usr/bin/env python3
"""Corrupt real reports the seven ways that survive a reader's eye, and see what
catches each one.

The claim under test is one of ownership: **the validator is the final authority
and the critic is assistance, not the security boundary.** That claim is only
worth something if the validator refuses a corrupted report *on its own*, with no
model in the loop — so every case here runs through `validate_report` alone, and
the critic is reported as unexercised rather than leaned on.

Three properties are measured per case, not one:

* **detected** — the validator names the defect, by code;
* **not silently repaired** — nothing in the pipeline quietly restored the text,
  so a defect that reached the file is a defect that stayed;
* **attributable** — a case that had nothing of its shape in a given report is
  recorded as *not applicable* rather than counted as a pass, because "no
  conflict was removed" is not evidence that a removed conflict would be caught.

Corruptions are applied to the **structured blocks**, which is the surface a
model draft and a hand edit both arrive on; the report gate reads its claim,
hedge, number and visual rules from the blocks, not from the rendered text.

Every real artifact is tried, so a case one video cannot exercise is exercised
by another. Cases that no real report could exercise fall back to a fixture and
are labelled fixture-backed, so a fixture pass is never read as a real one.

    uv run --with yt-dlp --with pillow --with numpy python3 acceptance/grounding_injection.py
"""
from __future__ import annotations

import io
import json
import re
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import report as R  # noqa: E402
import report_fixtures as FX  # noqa: E402
import report_validate as RV  # noqa: E402
from build_evidence import write_and_validate_report  # noqa: E402
from evidence_model import Evidence, save_evidence  # noqa: E402


def _codes(res) -> set[str]:
    return {e["code"] for e in res.blocking_errors}


# --------------------------------------------------------------------------- #
# the seven corruptions. Each returns (report, description, expected_codes), or
# None when the report has nothing of that shape to corrupt.
# --------------------------------------------------------------------------- #

def c1_change_a_number(ev, rep):
    by_id = {i.id: i for i in ev.items}
    md = R.render_markdown(rep)
    document = {n.rstrip(".") for n in RV._NUM.findall(RV._strip_markers(md))}
    # A declared run figure (frames sampled, items used) legitimately appears in
    # a block without belonging to the item that block cites, so it cannot serve
    # as the replacement: the validator is right to wave it through.
    declared = {str(v) for v in (rep.meta.get("report_figures") or {}).values()}
    for path, b in RV._all_blocks(rep):
        if b.kind in ("link", "meta", "summary", "timeline") or not b.evidence_ids:
            continue
        own: set[str] = set()
        for i in b.evidence_ids:
            if i in by_id:
                own |= RV.evidence_numbers_for(by_id[i])
        for tok in RV._NUM.findall(RV._strip_markers(b.text)):
            if not tok.isdigit() or len(tok) < 2:
                continue
            elsewhere = sorted(n for n in document
                               if n.isdigit() and len(n) > 1 and not n.startswith("0")
                               and n not in own and n != tok
                               and n not in declared and n.replace(",", "") not in declared)
            if elsewhere:
                b.text = b.text.replace(tok, elsewhere[0], 1)
                return rep, (f"figure {tok} -> {elsewhere[0]}, a figure the document "
                             f"contains elsewhere but this block does not cite"), \
                    {"number-misattributed"}
            b.text = b.text.replace(tok, str(int(tok) + 7), 1)
            return rep, f"figure {tok} -> an invented {int(tok) + 7}", {"number-unsupported"}
    return None


def c2_unsupported_sentence(ev, rep):
    rep.sections[0].blocks.append(R.Block(
        "The vendor confirmed the figure independently before publication.",
        [rep.evidence_ids[0]], "[00:00 · narration]", kind="narrative"))
    return rep, "an invented sentence about independent confirmation", {"claim-unsupported"}


def c3_remove_a_qualifier(ev, rep):
    by_id = {i.id: i for i in ev.items}
    for path, b in RV._all_blocks(rep):
        hedged = [by_id[i] for i in b.evidence_ids if i in by_id
                  and (by_id[i].status == "hedged" or (by_id[i].qualifier or "").strip())]
        if not hedged or not b.text:
            continue
        stripped = re.sub(r"\s*\((?:may|might|could|can|would|should|possibly|perhaps|"
                          r"probably|likely|up to|about|around|approximately|roughly)"
                          r"[^)]*\)", "", b.text)
        stripped = stripped.replace("given as a possibility", "confirmed")
        if stripped == b.text:
            continue
        b.text = stripped
        return rep, f"the hedge the evidence records on {hedged[0].id} was removed", \
            {"hedge-dropped"}
    return None


def c4_alter_a_url(ev, rep):
    m = re.search(r"https?://[^\s<>\")\]]+", R.render_markdown(rep))
    if not m:
        return None
    u = m.group(0)
    for path, b in RV._all_blocks(rep):
        if u in b.text:
            b.text = b.text.replace(u, "https://example.invalid/replaced", 1)
            return rep, f"the url {u} was rewritten to a different host", {"url-unsupported"}
    return None


def c5_reverse_a_comparison(ev, rep):
    md = R.render_markdown(rep)
    opposite = {"faster": "slower", "slower": "faster", "higher": "lower", "lower": "higher",
                "larger": "smaller", "smaller": "larger", "better": "worse", "worse": "better"}
    prose = re.search(r"([A-Za-z][\w .\-]{2,30}) is (faster|slower|higher|lower|larger|"
                      r"smaller|better|worse) than ([A-Za-z][\w .\-]{2,30})", md)
    if prose:
        flipped = f"{prose.group(1)} is {opposite[prose.group(2)]} than {prose.group(3)}"
        for path, b in RV._all_blocks(rep):
            if prose.group(0) in b.text:
                b.text = b.text.replace(prose.group(0), flipped, 1)
                return rep, f"the comparison was reversed to {flipped!r}", {"claim-unsupported"}
    # The renderer's own shape: "**A vs B** — score: A 71%, B 79%." Reversing it
    # swaps the two figures, so the winner becomes the loser while every token
    # stays inside the evidence — which is exactly why set membership misses it.
    table = re.search(r"\*\*(.+?) vs (.+?)\*\* — ([^\[]+)", md)
    if not table:
        return None
    a, b, rest = table.group(1), table.group(2), table.group(3)
    na, nb = re.findall(rf"{re.escape(a)}\s+([\d.]+%)", rest), re.findall(
        rf"{re.escape(b)}\s+([\d.]+%)", rest)
    if not na or not nb:
        return None
    swapped = rest.replace(f"{a} {na[0]}", f"{a} {nb[0]}").replace(f"{b} {nb[0]}",
                                                                     f"{b} {na[0]}")
    flipped = f"**{a} vs {b}** — {swapped.strip()}"
    for path, blk in RV._all_blocks(rep):
        if table.group(0) in blk.text:
            blk.text = blk.text.replace(table.group(0), flipped, 1)
            return rep, (f"the comparison's two figures were swapped, so the ranking "
                         f"reverses: {a} now reads {nb[0]} and {b} reads {na[0]}"), \
                {"number-misattributed", "claim-unsupported"}
    return None


def c6_hide_a_conflict(ev, rep):
    if not ev.conflicts or not any(b.kind == "conflict" for _, b in RV._all_blocks(rep)):
        return None
    rep.sections = [s for s in rep.sections if s.key != "reliability"]
    rep.sections.append(R.Section(
        "reliability", "Source & Reliability",
        blocks=[R.Block("Evidence by source — narration: 2.", kind="meta")]))
    return rep, "the unresolved conflict was removed from the report", {"conflict-dropped"}


def c7_invent_a_visual_relationship(ev, rep):
    """A visual statement resting on speech: the report narrating a diagram no
    model and no frame read. Appended to whichever section exists rather than a
    fixed index, so the case runs on a report of any shape."""
    block = R.Block(
        "The arrow points from the model to the retriever, and the bar for Qwen "
        "is shorter than the bar for Llama.",
        [rep.evidence_ids[0]], "[00:00 · narration]", kind="visual")
    for s in rep.sections:
        subs = s.subsections or [s]
        subs[0].blocks.append(block)
        return rep, "an invented relationship between visual elements", \
            {"visual-unsupported"}
    return None


CASES = [
    ("1 change one number", c1_change_a_number),
    ("2 add an unsupported factual sentence", c2_unsupported_sentence),
    ("3 remove an important qualifier", c3_remove_a_qualifier),
    ("4 alter one URL", c4_alter_a_url),
    ("5 reverse one comparison", c5_reverse_a_comparison),
    ("6 remove a conflict disclosure", c6_hide_a_conflict),
    ("7 invent a visual relationship", c7_invent_a_visual_relationship),
]


def _sources():
    out = []
    root = Path.home() / "Desktop" / "yt-reports"
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if (d / "evidence.json").exists():
            out.append(("real", d.name, Evidence.from_dict(
                json.loads((d / "evidence.json").read_text()))))
    # Fixtures backstop the shapes a real video may not contain. They are
    # labelled, so a fixture pass is never read as a real-video pass.
    for name in ("unresolved_conflict", "chart_heavy", "table_heavy", "hedged_claim",
                 "technical_lecture"):
        out.append(("fixture", name, FX.FIXTURES[name]()))
    return out


def main() -> int:
    results: list[dict] = []
    totals = {label: {"applied": 0, "detected": 0, "repaired": 0} for label, _ in CASES}
    sources = _sources()

    for kind, name, ev in sources:
        clean_rep = R.generate(ev, "balanced")
        base = RV.validate_report(clean_rep, ev, R.render_markdown(clean_rep))
        if not base.valid:
            results.append({"source": name, "source_kind": kind,
                            "skipped": "the clean report does not validate",
                            "codes": sorted(_codes(base))})
            continue
        for label, fn in CASES:
            rep = R.generate(ev, "balanced")
            clean_md = R.render_markdown(rep)
            got = fn(ev, rep)
            if got is None:
                continue
            rep, what, expect = got
            md = R.render_markdown(rep)
            res = RV.validate_report(rep, ev, md)
            codes = _codes(res)
            detected = bool(expect & codes)
            repaired = md == clean_md
            totals[label]["applied"] += 1
            totals[label]["detected"] += int(detected)
            totals[label]["repaired"] += int(repaired)
            results.append({"source": name, "source_kind": kind, "case": label,
                            "corruption": what, "expected": sorted(expect),
                            "codes": sorted(codes), "detected": detected,
                            "silently_repaired": repaired})

    # The write gate: a corruption the validator refuses must leave no file.
    ev = sources[0][2]
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        save_evidence(ev, t / "evidence.json")
        (t / "REPORT.md").write_text("the report that must survive\n")
        import report as report_mod
        forged = R.generate(ev, "balanced")
        forged.sections[0].blocks.append(R.Block(
            "The vendor confirmed the figure independently before publication.",
            [forged.evidence_ids[0]], "[00:00 · narration]", kind="narrative"))
        original = report_mod.generate
        report_mod.generate = lambda e, detail: forged
        try:
            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = write_and_validate_report(t, "balanced", quiet=True)
        finally:
            report_mod.generate = original
        untouched = (t / "REPORT.md").read_text() == "the report that must survive\n"
    gate = {"case": "write gate", "detected": rc == 1 and untouched, "exit_code": rc,
            "previous_file_untouched": untouched, "reason": buf.getvalue()[-200:]}
    results.append(gate)

    print(f"  {'case':40s} applied  detected  repaired")
    for label, _ in CASES:
        t = totals[label]
        mark = "ok  " if t["applied"] and t["detected"] == t["applied"] else "FAIL"
        if not t["applied"]:
            # Zero applications is UNMEASURED, not passed. Counting it as a pass
            # is how a case nobody ran ends up looking like a case that holds.
            print(f"  N/A  {label:38s} no report in the set had this shape — "
                  f"UNMEASURED, not passed")
            continue
        print(f"  {mark} {label:38s} {t['applied']:5d}  {t['detected']:7d}  "
              f"{t['repaired']:7d}")
    print(f"  {'ok  ' if gate['detected'] else 'FAIL'} "
          f"{'write gate refuses and leaves the file':38s} "
          f"exit={gate['exit_code']} untouched={gate['previous_file_untouched']}")
    for r in results:
        if "case" in r and r.get("detected") is False and "corruption" in r:
            print(f"       missed on {r['source']} ({r['source_kind']}): expected "
                  f"{r['expected']}, got {r['codes'] or 'nothing'} — {r['corruption']}")

    out = ROOT / "acceptance" / "grounding_injection.json"
    applied = sum(t["applied"] for t in totals.values())
    detected = sum(t["detected"] for t in totals.values())
    repaired = sum(t["repaired"] for t in totals.values())
    out.write_text(json.dumps({
        "critic": "not exercised. No configured provider had credit, and the second "
                  "candidate produced neither a schema-conforming observation nor a "
                  "draft within any budget the pipeline should spend. The validator is "
                  "the boundary under test, and every case runs with no model in the "
                  "loop — which is the property that matters if the critic is "
                  "assistance rather than the boundary.",
        "totals": totals,
        "applied": applied, "detected": detected, "silently_repaired": repaired,
        "write_gate": gate,
        "results": results,
    }, indent=2) + "\n")
    print(f"\n{detected}/{applied} corruptions detected by the validator alone; "
          f"{repaired} silently repaired; write gate "
          f"{'holds' if gate['detected'] else 'FAILED'} -> {out}")
    unmeasured = [label for label, t in totals.items() if not t["applied"]]
    if unmeasured:
        print(f"\nUNMEASURED: {', '.join(unmeasured)}")
    return 0 if detected == applied and repaired == 0 and gate["detected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
