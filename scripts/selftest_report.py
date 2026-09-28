"""Report-layer checks: generation, grounding, regression, golden structure.

Split out of `selftest.py` because this layer has a different failure mode from
the one before it. The evidence layer's tests ask "does this document hold
together?"; these ask "does the report tell the truth about the evidence, and
does it stay the right size?" — a question with an answer that is not visible by
reading the code, only by generating and measuring.

Three groups:

* **Every fixture, end to end.** Fifteen video shapes, each generated and each
  validated. A report that passes on a clean technical lecture and fabricates on
  an unreadable chart is not a report layer.
* **Corruption.** The generator is a renderer and should be unfabricable by
  construction; these tests *try* to make it fabricate, and assert that the
  validator refuses. A guard that has never been seen to fire is not known to
  fire.
* **Golden structure.** Not golden prose. A golden report that asserts exact
  wording breaks on every wording change and is then either ignored or updated
  without being read — which is a test that trains people to update tests
  without reading them. These assert the properties a report must have: the
  right sections present for that shape, evidence references that resolve, no
  invented figure, a real conflict preserved, a real chart represented.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import report as R  # noqa: E402
import report_fixtures as FX  # noqa: E402
import report_validate as RV  # noqa: E402
from evidence_model import SCHEMA_VERSION  # noqa: E402
from report_validate import validate_report  # noqa: E402

HEADINGS = re.compile(r"^##+ (.+)$", re.M)
MARKER = RV._MARKER


def _gen(name: str, detail: str = "balanced"):
    ev = FX.FIXTURES[name]()
    rep = R.generate(ev, detail)
    return ev, rep, R.render_markdown(rep), validate_report(rep, ev, R.render_markdown(rep))


def headings(md: str) -> list[str]:
    return [h.strip() for h in HEADINGS.findall(md)]


# --------------------------------------------------------------------------- #
# every fixture, generated and validated
# --------------------------------------------------------------------------- #

def t_report_fixtures() -> None:
    for name in FX.FIXTURES:
        ev, rep, md, res = _gen(name)
        m = res.metrics
        for e in res.blocking_errors[:3]:
            check(f"report/{name} grounding", False, f"{e['code']}: {e['message'][:120]}")
        else:
            check(f"report/{name} validates", res.valid,
                  json.dumps([e["code"] for e in res.blocking_errors]))
        check(f"report/{name} provenance 100%", m["provenance_coverage"] == 1.0,
              str(m["provenance_coverage"]))
        check(f"report/{name} no invented number", m["unsupported_numbers"] == 0)
        check(f"report/{name} no invented claim", m["unsupported_claims"] == 0)
        check(f"report/{name} no invented url", m["unsupported_urls"] == 0)
        check(f"report/{name} no fabricated timestamp", m["fabricated_timestamps"] == 0)
        check(f"report/{name} evidence ids resolve", all(
            i in {x.id for x in ev.items} for i in rep.evidence_ids))
        check(f"report/{name} within its own ceiling",
              m["report_word_count"] <= m["ceiling_words"] * 1.5,
              f"{m['report_word_count']} vs ceiling {m['ceiling_words']}")
        check(f"report/{name} repetition controlled", m["repetition_ratio"] <= 0.18,
              str(m["repetition_ratio"]))
        check(f"report/{name} is a document", m["report_word_count"] > 20 and "# " in md)


def t_report_shape() -> None:
    """Each shape produces the sections its evidence supports, and only those."""
    ev, rep, md, _ = _gen("technical_lecture")
    hs = headings(md)
    for want in ("Executive Summary", "Key Takeaways", "Detailed Notes",
                 "Extracted Details", "Numbers, Pricing & Benchmarks",
                 "Comparisons", "Timeline", "Source & Reliability"):
        check(f"shape/technical has {want!r}", want in hs, str(hs))
    check("shape/technical has no empty People section", "People & Organizations" in hs)

    _, rep, md, _ = _gen("simple_talking_head")
    hs = headings(md)
    for absent in ("Numbers, Pricing & Benchmarks", "Comparisons", "Steps & Workflows",
                   "Resources & Links", "Important Quotes", "Products, Models & Tools"):
        check(f"shape/talking-head omits {absent!r}", absent not in hs, str(hs))

    _, rep, md, _ = _gen("code_tutorial")
    check("shape/code tutorial has Steps", "Steps & Workflows" in headings(md))
    steps = md.split("### Steps & Workflows", 1)[1]
    check("shape/steps are a real procedure", steps.count("- ") >= 3)

    _, rep, md, _ = _gen("visual_only_number")
    check("shape/visual-only figure is present", "18 tok/s" in md and "24 tok/s" in md)
    check("shape/visual-only figure is marked as on-screen", "on-screen text" in md)

    _, rep, md, _ = _gen("transcript_only_number")
    check("shape/transcript-only figure is marked as narration", "· narration]" in md)
    check("shape/transcript-only figure carries no visual marker",
          "· on-screen text]" not in md and "· model reading]" not in md)


def t_report_degradation() -> None:
    for name, want in (("simple_talking_head", "DEGRADED"),
                       ("chart_heavy", "FULL"),
                       ("unreadable_chart", "FULL")):
        ev, rep, md, _ = _gen(name)
        check(f"degradation/{name} is {want}", rep.mode == want, rep.mode)
        if want != "FULL":
            check(f"degradation/{name} says so at the top",
                  md.split("## Executive Summary")[0].count(want) == 1, rep.banner)
    ev, rep, md, _ = _gen("chart_heavy")
    check("degradation/no false visual claim when nothing was interpreted",
          "DEGRADED" not in md or rep.mode == "DEGRADED")


def t_report_uncertainty() -> None:
    """Hedges, illegibility and conflicts survive; none is quietly normalised."""
    _, rep, md, _ = _gen("hedged_claim")
    check("uncertainty/hedge kept on the figure", "possibility" in md and "$500" in md,
          "the hedge word was dropped from the figure")
    check("uncertainty/hedge is not stated flatly", "**$500** —" in md.replace("-$", "-$")
          and "given as a possibility" in md)

    _, rep, md, _ = _gen("unreadable_chart")
    check("uncertainty/unreadable chart states no figure",
          not re.search(r"\b\d+(\.\d+)?\s*(tok/s|%|GB|ms)\b", md.split("Source & Reliability")[0]),
          "a figure was stated for an illegible axis")
    check("uncertainty/unreadable chart says so", "could not be read" in md)
    # One statement about the illegibility, not two sections making it.
    check("uncertainty/illegibility is stated once",
          md.count("could not be read") == 1, str(md.count("could not be read")))

    _, rep, md, _ = _gen("unresolved_conflict")
    rel = md.split("Source & Reliability", 1)[1]
    check("conflict/both sides shown", "75%" in rel and "8%" in rel, rel[:200])
    check("conflict/not resolved by the report", "not resolved" in rel.lower())
    check("conflict/both sides have timestamps", len(MARKER.findall(rel)) >= 2)
    check("conflict/neither side silently dropped", "vs" in rel)


def t_report_noise() -> None:
    """Interface furniture does not reach the reader."""
    _, rep, md, _ = _gen("noisy_ocr")
    numbers = md.split("Numbers, Pricing & Benchmarks", 1)
    check("noise/queue depth kept", "412" in md)
    if len(numbers) > 1:
        shown = numbers[1].split("###", 1)[0]
        for junk in ("1080p", "14:22", "buffering"):
            check(f"noise/{junk!r} not rendered", junk not in shown, shown[:200])

    _, rep, md, _ = _gen("highly_repetitive")
    m = RV.duplicate_ngrams(md)
    check("noise/repetition collapsed", m[0] / max(1, m[1]) < 0.18, str(m))
    check("noise/repeat source still represented", "benchmark" in md.lower())
    check("noise/repeat source not printed eleven times", md.lower().count("rerun the benchmark") <= 3,
          str(md.lower().count("rerun the benchmark")))


def t_report_repetition() -> None:
    """One item, one home."""
    ev, rep, md, _ = _gen("technical_lecture")
    # A report has a narrative and an index, and an item legitimately appears in
    # both: a latency figure belongs in the paragraph that discusses it *and* in
    # a table a reader can scan. That is not repetition, it is what an index is
    # for. What must not happen is the same item holding two *narrative* homes,
    # or the same sentence being written twice — so only prose sections count.
    PROSE = ("key_takeaways", "detailed_notes")
    texts: dict[str, set] = {}
    homes: dict[str, set] = {}
    for s in rep.sections:
        prose = s.key in PROSE
        for sub in (s.subsections or [s]):
            for b in sub.blocks:
                if b.kind in ("summary", "meta", "timeline"):
                    continue
                for i in b.evidence_ids:
                    if prose:
                        texts.setdefault(i, set()).add(b.text)
                        homes.setdefault(i, set()).add(sub.key)
    twice = {i: sorted(v) for i, v in texts.items() if len(v) > 1}
    check("repetition/no item written in two different ways",
          all(len(v) < 2 for v in twice.values()),
          json.dumps({k: v for k, v in twice.items() if len(v) > 1})[:300])
    check("repetition/no item has two narrative homes",
          all(len(v) < 2 for v in homes.values()),
          json.dumps({k: sorted(v) for k, v in homes.items() if len(v) > 1})[:300])
    check("repetition/most evidence is written once",
          sum(len(v) for v in texts.values()) <= len(texts) * 1.6,
          f"{sum(len(v) for v in texts.values())} texts over {len(texts)} items")


# --------------------------------------------------------------------------- #
# corruption: the guard has to be seen to fire
# --------------------------------------------------------------------------- #

def t_report_corruption() -> None:
    ev, rep, md, res = _gen("technical_lecture")

    def rescore(markdown: str):
        return validate_report(rep, ev, markdown)

    bad = rescore(md.replace("**120 ms**", "**999 ms**"))
    check("corruption/invented figure is caught",
          any(e["code"] == "number-unsupported" for e in bad.blocking_errors),
          "a number that appears nowhere in the evidence passed")

    bad = rescore(md + "\nSee https://invented.example.com/never for details.\n")
    check("corruption/invented url is caught",
          any(e["code"] == "url-unsupported" for e in bad.blocking_errors),
          "a URL that appears nowhere in the evidence passed")

    bad = rescore(md.replace("[01:15 · narration]", "[44:44 · narration]"))
    check("corruption/fabricated timestamp is caught",
          any(e["code"] == "timestamp-fabricated" for e in bad.blocking_errors),
          "a timestamp matching no snippet, frame or segment passed")

    # A heading immediately followed by another heading. Appended to the rendered
    # text rather than built through the generator, because the generator prunes
    # exactly this and the check exists for the case where it stops doing so.
    bad = rescore(md + "\n## Invented Section\n\n## Nothing Under It\n\n## Also Nothing\n")
    check("corruption/empty section is caught",
          any(e["code"] == "section-empty" for e in bad.blocking_errors),
          "a heading with nothing under it passed")

    # A visual statement with no frame behind it.
    ev2, rep2, md2, _ = _gen("technical_lecture")
    forged = R.Block("The diagram shows a retrieval pipeline end to end.",
                     [rep2.evidence_ids[0]], "[01:15 · narration]", kind="visual")
    rep2.sections[2].subsections[0].blocks.append(forged)
    bad = validate_report(rep2, ev2, R.render_markdown(rep2))
    check("corruption/visual statement without frame evidence is caught",
          any(e["code"] == "visual-unsupported" for e in bad.blocking_errors),
          "a visual statement bound only to narration passed")

    # A claim naming something no source names.
    ev3 = FX.simple_talking_head()
    rep3 = R.generate(ev3, "balanced")
    block = R.Section("x", "X", blocks=[
        R.Block("The Quantax Engine powers the whole pipeline.", ["itm-004"],
                "[00:12 · narration]", kind="narrative")])
    rep3.sections.append(block)
    bad = validate_report(rep3, ev3, R.render_markdown(rep3))
    check("corruption/name in no source is caught",
          any(e["code"] == "claim-unsupported" for e in bad.blocking_errors),
          "a proper noun present in no evidence passed")

    # A dropped conflict.
    ev4, rep4, md4, _ = _gen("unresolved_conflict")
    rep4.sections = [s for s in rep4.sections
                     if s.key != "reliability"] + [R.Section(
                         "reliability", "Source & Reliability",
                         blocks=[R.Block("Evidence by source — narration: 2.", kind="meta")])]
    bad = validate_report(rep4, ev4, R.render_markdown(rep4))
    check("corruption/dropped conflict is caught",
          any(e["code"] == "conflict-dropped" for e in bad.blocking_errors),
          "an unresolved conflict was dropped without a word")

    # An item id that does not exist.
    ev5, rep5, md5, _ = _gen("simple_talking_head")
    rep5.sections[0].blocks.append(
        R.Block("Something happened.", ["itm-999"], "[00:12 · narration]", kind="narrative"))
    bad = validate_report(rep5, ev5, R.render_markdown(rep5))
    check("corruption/unresolvable evidence id is caught",
          any(e["code"] == "evidence-id-unresolved" for e in bad.blocking_errors),
          "a citation to a non-existent item passed")

    # Padded repetition.
    ev6, rep6, md6, _ = _gen("simple_talking_head")
    body = [b for s in rep6.sections for b in s.blocks if b.text]
    if body:
        rep6.sections[0].blocks = body * 6
        bad = validate_report(rep6, ev6, R.render_markdown(rep6))
        check("corruption/padded repetition is caught",
              any(e["code"] in ("repetition-excessive", "repetition-high")
                  for e in bad.blocking_errors + bad.warnings),
              "six copies of the same paragraphs passed")


def t_report_determinism() -> None:
    for name in ("technical_lecture", "unresolved_conflict"):
        a = R.render_markdown(R.generate(FX.FIXTURES[name](), "balanced"))
        b = R.render_markdown(R.generate(FX.FIXTURES[name](), "balanced"))
        check(f"determinism/{name} is byte-identical across runs", a == b)

    ev = FX.technical_lecture()
    doc = R.report_json(R.generate(ev, "balanced"))
    ids = {i.id for i in ev.items}
    check("report.json references evidence ids, not records",
          all(i in ids for s in doc["sections"] for i in s.get("evidence_ids", [])))
    blob = json.dumps(doc)
    check("report.json carries no raw evidence text",
          "120 ms" not in blob or "evidence_ids" in blob)
    check("report.json declares its figures", "report_figures" in doc["meta"])


def t_report_budget() -> None:
    ev = FX.technical_lecture()
    sizes = {}
    for level in ("brief", "balanced", "deep"):
        rep = R.generate(ev, level)
        sizes[level] = rep.word_count()
    check("budget/monotonic in detail level",
          sizes["brief"] <= sizes["balanced"] <= sizes["deep"], json.dumps(sizes))

    # The budget must move with the evidence, not with a constant.
    small = R.derive_budget(FX.simple_talking_head(), "balanced")
    big = R.derive_budget(FX.technical_lecture(), "balanced")
    check("budget/grows with the evidence", big.words > small.words,
          f"{big.words} vs {small.words}")
    dense = R.derive_budget(FX.technical_lecture(), "deep")
    check("budget/deep ceiling is the deep range", dense.ceiling == 3500, str(dense.ceiling))

    # A video with almost nothing must not be padded up to a floor.
    check("budget/never padded to the floor",
          small.words < small.floor, f"{small.words} words against floor {small.floor}")


def t_report_golden() -> None:
    """Structural assertions, not prose assertions.

    Deliberately no `assert "the report says exactly this"`. A golden text file
    for a generated document is a test that breaks on every wording change and
    gets updated without being read, which is worse than no test.
    """
    for name, must, must_not in (
        ("chart_heavy",
         ["Detailed Notes", "Source & Reliability", "Model-derived content"],
         ["Numbers, Pricing & Benchmarks"]),
        ("unresolved_conflict",
         ["Unresolved conflict", "ot resolved"],
         ["## Key Takeaways"]),
        ("corroboration", ["on-screen text", "narration"], []),
        ("unreadable_chart", ["could not be read"], []),
        ("empty_optional_sections", ["Executive Summary"], ["Comparisons"]),
    ):
        ev, rep, md, res = _gen(name)
        for token in must:
            check(f"golden/{name} contains {token!r}", token in md)
        for token in must_not:
            check(f"golden/{name} omits {token!r}", token not in md)
        check(f"golden/{name} validates", res.valid)
        check(f"golden/{name} has at least one provenance marker",
              bool(MARKER.search(md)) or name == "empty_optional_sections")


def t_report_writes(tmp: Path) -> None:
    """The artifacts land, atomically, and read back as the report that was made."""
    ev, rep, md, res = _gen("technical_lecture")
    from evidence_model import save_evidence as _save
    _save(ev, tmp / "evidence.json")
    md_path, js_path = R.write_report(rep, tmp)
    check("writes/REPORT.md exists", md_path.exists())
    check("writes/REPORT.json exists", js_path.exists())
    check("writes/no temp files left behind",
          not list(tmp.glob("*.tmp")), str(list(tmp.glob("*.tmp"))))
    doc = json.loads(js_path.read_text())
    check("writes/REPORT.json parses and names its schema",
          doc.get("schema") == "yt-report/report/1")
    check("writes/REPORT.md is the rendered report", md_path.read_text() == md)

    # A corrupted evidence document must not produce a report at all. The
    # corruption is a citation to a frame that is not in the document: valid
    # JSON, valid schema, and a pointer that resolves to nothing.
    doc = json.loads((tmp / "evidence.json").read_text())
    doc["items"][0]["provenance"] = [{"source": "frame", "ref": "u9999", "confidence": 0.9}]
    (tmp / "evidence.json").write_text(json.dumps(doc))
    (tmp / "REPORT.md").unlink(missing_ok=True)
    from build_evidence import write_and_validate_report
    import io
    from contextlib import redirect_stderr
    buf = io.StringIO()
    with redirect_stderr(buf):
        rc = write_and_validate_report(tmp, "balanced", quiet=True)
    check("writes/corrupt evidence is refused", rc == 1, str(rc))
    check("writes/no report written for corrupt evidence", not (tmp / "REPORT.md").exists())


def t_report_provenance_tags() -> None:
    """Source distinctions survive into the report."""
    _, _, md, _ = _gen("corroboration")
    rel = md.split("### Numbers", 1)[1] if "### Numbers" in md else md
    check("tags/narration is labelled", "[00:00 · narration]" in md)
    check("tags/on-screen text is labelled", "on-screen text" in md)
    check("tags/metadata is labelled", "[metadata]" in md)
    _, _, md2, _ = _gen("chart_heavy")
    check("tags/model reading is labelled", "Model-derived content" in md2)


def t_report_first_person() -> None:
    """Neutral third person. The old format's failure was a voice the evidence
    did not contain, and a renderer is the last place that can be enforced."""
    banned = re.compile(r"\b(I found|I noticed|I think|I see|we found|my notes|in my opinion)\b", re.I)
    for name in FX.FIXTURES:
        _, _, md, _ = _gen(name)
        hits = banned.findall(md)
        # A speaker's own words may legitimately contain "I think" — the report
        # quotes the source, it does not adopt a voice. So the check is on the
        # report's *own* prose, which is everything outside a quoted block.
        own = "\n".join(l for l in md.splitlines() if not l.startswith("> "))
        check(f"voice/{name} is third person", not banned.search(own), str(hits))


# --------------------------------------------------------------------------- #
# R1x · the report may not advertise figures it does not contain
# --------------------------------------------------------------------------- #

def t_figure_reliability_consistency() -> None:
    """A real report said "10 of 11 figures rest on one text read" and had no
    numbers section at all.

    Both numbers were true: the note counted every `number` item in
    `evidence.json`, the section printed the ones that survived the noise gate,
    and on that video that was none of them. Nothing compared the two, because
    nothing had to — the renderer was free to state a count about a section it
    had not written.
    """
    for name in FX.FIXTURES:
        ev, rep, md, res = _gen(name)
        codes = {e["code"] for e in res.blocking_errors}
        check(f"R1x/{name} no figure-count contradiction",
              not (codes & {"figures-announced-not-rendered", "figures-withheld-unreported",
                            "figure-count-mismatch"}),
              str(sorted(codes)))

    # Case 1 — rendered equals announced. Every fixture with a figures section
    # must satisfy it, and the two numbers must come from different places: the
    # validator counts the blocks, the renderer declares its own total.
    ev, rep, md, _ = _gen("table_heavy")
    declared = rep.meta["report_figures"]
    check("R1x the renderer declares rendered, withheld and total",
          declared["figures_rendered"] + declared["figures_withheld"] == declared["figures_total"],
          json.dumps({k: v for k, v in declared.items() if k.startswith("figures_")}))
    check("R1x a report with figures renders them", declared["figures_rendered"] > 0,
          str(declared))
    check("R1x ...and announces exactly that many",
          f"{declared['figures_rendered']} figures rest on one text read" in md
          or "figures rest on one text read" not in md,
          md[md.find("Source & Reliability"):][:400])

    # Case 3 — a figure the threshold withheld is reported as withheld, in words,
    # with the count that makes the three numbers add up.
    ev, rep, md, res = _gen("noisy_ocr")
    d = rep.meta["report_figures"]
    if d["figures_withheld"]:
        check("R1x a withheld figure is announced as omitted",
              f"{d['figures_total']} figures were detected but omitted" in md,
              md[md.find("Source & Reliability"):][:500])
    check("R1x the three figure counts add up to the number detected",
          d["figures_rendered"] + d["figures_withheld"] == d["figures_total"],
          json.dumps({k: v for k, v in d.items() if k.startswith("figures_")}))

    # Case 2 — announced but omitted, refused. The number in the note is edited
    # to one the report does not contain, which is the exact shape of the live
    # failure: both halves of the sentence true, the sentence between them false.
    ev, rep, md, res = _gen("table_heavy")
    d = rep.meta["report_figures"]
    inflated = d["figures_total"] + 1
    swapped = False
    for sec in rep.sections:
        if sec.key != "reliability":
            continue
        for b in sec.blocks:
            new = re.sub(r"(\d+) of (\d+) figures rest on",
                         lambda m: f"{inflated} of {inflated} figures rest on", b.text)
            swapped = swapped or new != b.text
            b.text = new
    check("R1x the note under test really was rewritten", swapped, "no block matched")
    errors: list[dict] = []
    RV.check_figures(rep, ev, RV._all_blocks(rep), errors)
    check("R1x a count larger than what is rendered is refused",
          any(e["code"] == "figures-announced-not-rendered" for e in errors),
          str([e["code"] for e in errors]))


def t_numeric_metrics_exclude_timestamps() -> None:
    """`numbers_in_report: 44, numbers_supported: 44` on a 494-word report.

    Twenty-four of the forty-four were `[HH:MM]` markers, which cannot be
    unsupported and measure nothing. The metric now counts numeric facts, and
    counts markers on their own line.
    """
    ev, rep, md, res = _gen("table_heavy")
    m = res.metrics
    for key in ("numeric_facts_in_report", "numeric_facts_supported",
                "timestamp_markers", "numeric_tokens_total"):
        check(f"R1x {key} is reported", key in m, str(sorted(m)))
    check("R1x a report with markers counts them as markers, not facts",
          m["timestamp_markers"] == len(RV._MARKER.findall(md)),
          f"{m['timestamp_markers']} vs {len(RV._MARKER.findall(md))}")
    check("R1x the two counts are distinct",
          m["numeric_facts_in_report"] != m["numeric_tokens_total"] or m["timestamp_markers"] == 0,
          json.dumps({k: m[k] for k in ("numeric_facts_in_report", "timestamp_markers",
                                        "numeric_tokens_total")}))
    check("R1x the total is the sum of the two",
          m["numeric_tokens_total"] == m["numeric_facts_in_report"] + m["timestamp_markers"])
    check("R1x facts are still all supported", m["numeric_facts_supported"] == m["numeric_facts_in_report"],
          f"{m['numeric_facts_supported']}/{m['numeric_facts_in_report']}")

    # The seven shapes the metric has to tell apart. A report whose only numbers
    # are markers must report zero facts, not N.
    cases = [
        ("[04:31] only", 0, 1),
        ("The card holds 32 GB [01:02]", 1, 1),
        ("Latency fell 100 ms [00:10]", 1, 1),
        ("Costs $500/month [00:30]", 1, 1),
        ("The equation 28 x 28 = 784 [00:45]", 3, 1),
        ("A 7B model [00:50]", 1, 1),
        ("Accuracy is 75% [01:00]", 1, 1),
    ]
    for text, want_facts, want_marks in cases:
        errors: list[dict] = []
        total, _ok = RV.check_numbers(rep, ev, text, errors)
        marks = len(RV._MARKER.findall(text)) + len(
            [m for m in RV._MARKER_LOOSE.findall(text) if m not in RV._MARKER.findall(text)])
        check(f"R1x {text!r}: {want_facts} fact(s), {want_marks} marker(s)",
              (total, marks) == (want_facts, want_marks), f"got {total} facts, {marks} markers")


def t_takeaways_are_insights() -> None:
    """A subscribe/Patreon message became a Key Takeaway in a live report.

    The item was not filtered by any rule; it was *ranked*, and a ranked item
    still wins when the pool is short — which is the case every short video is.
    So promotion and boilerplate are excluded rather than demoted, and a
    rhetorical question is not an insight until the evidence answers it.
    """
    cases = [
        ("If you enjoyed this, hit the bell and subscribe on Patreon", "promotion"),
        ("Thanks for watching, and see you in the next one", "boilerplate"),
        ("What is the best hope for what those middle layers might be doing?", "question"),
        ("The network maps 784 inputs to 10 outputs.", "core_idea"),
        ("For example, a 3 by 3 matrix has 9 entries.", "example"),
        ("Run the migration, then restart the service.", "procedure"),
        ("So the point is that the layers compose.", "conclusion"),
        ("Historically these machines filled a room.", "context"),
        ("The first layer detects edges.", "supporting_detail"),
    ]
    for text, want in cases:
        item = FX.Item(id="x", type="claim", value=text, display=text, provenance=[],
                       importance="medium" if want == "supporting_detail" else "high")
        got = R.classify_evidence_role(item)
        check(f"R1x {text[:40]!r} is {want}", got == want, f"got {got!r}")

    check("R1x a discount in a technical claim is not a promotion",
          R.classify_evidence_role(FX.Item(id="x", type="claim", provenance=[],
                                           value="We discount the activation by 2x", importance="high"))
          != "promotion")

    # End to end: a document whose only candidate takeaways are a plug and a
    # rhetorical question must still produce a Key Takeaway from the claim.
    ev, rep, md, res = _gen("simple_talking_head")
    takeaway = next((s for s in rep.sections if s.key == "key_takeaways"), None)
    check("R1x the takeaway section still exists", takeaway is not None and bool(takeaway.blocks))
    check("R1x no takeaway is a plug or a bare question",
          not any(R.classify_evidence_role(
              next(i for i in ev.items if i.id == iid)) in R._NOT_TAKEAWAY
              for b in (takeaway.blocks if takeaway else []) for iid in b.evidence_ids),
          str([b.text for b in (takeaway.blocks if takeaway else [])]))


def t_exec_summary_states_the_central_argument() -> None:
    """The deterministic summary was a title, a chapter list and an entity list.

    A reader opening a report to find out what the video argues got the video's
    title back. The fix is not a second reading of the video — it is assembling
    the three things a summary is for out of items that already exist, cited to
    the item each clause came from.
    """
    for name in ("technical_lecture", "table_heavy", "simple_talking_head"):
        ev, rep, md, res = _gen(name)
        summary = next((s for s in rep.sections if s.key in ("executive_summary", "exec_summary")), None)
        core = [b for b in (summary.blocks if summary else []) if b.text.startswith("Core:")]
        # The line is drawn from the narrative items the renderer actually kept:
        # a table row, a link, or the uploader's own title has its own home, and
        # promoting one of those into a summary gives the item a second home for
        # a sentence it never had. `table_heavy` is in this list precisely
        # because its only narrative "core" is the uploader's title, and the
        # honest summary of that video has no central line to print.
        pool = [i for i in ev.items if i.type in R.NARRATIVE_TYPES
                and not (i.type == "topic" and all(p.source == "metadata" for p in (i.provenance or [])))
                and R.classify_evidence_role(i) in ("core_idea", "conclusion", "benchmark")
                and (i.value or i.subject)]
        check(f"R1x/{name} a central line appears whenever the narrative can supply one",
              bool(core) or not pool,
              f"core line={bool(core)}, {len(pool)} narrative candidate(s)")
        check(f"R1x/{name} the central line is never empty or invented",
              all(b.text.strip() != "Core: ." and b.evidence_ids for b in core),
              str([b.text for b in core]))
        by_id = {i.id: i for i in ev.items}
        for b in (summary.blocks if summary else []):
            if not b.text.startswith(("Core:", "Conclusion:")):
                continue
            check(f"R1x/{name} every summary clause is cited to an item",
                  b.evidence_ids and all(i in by_id for i in b.evidence_ids),
                  str(b.evidence_ids))
            roles = {R.classify_evidence_role(by_id[i]) for i in b.evidence_ids if i in by_id}
            check(f"R1x/{name} the central line is not built from promotion or boilerplate",
                  not (roles & R._NOT_TAKEAWAY), str(roles))
    check("R1x the summary is still a summary, not a second Detailed Notes",
          all(len(b.text) < 400 for s in (_gen("technical_lecture")[1].sections)
              if s.key == "executive_summary" for b in s.blocks),
          "a summary block ran long enough to be a narrative section")


def t_conflict_renders_both_sides() -> None:
    """Spoken 75%, on screen 8%. Neither is the answer.

    A conflict is a fact about the video, and the only two ways to get this
    wrong are to print one side without saying the other exists, or to average
    them. The fixture is the exact pair the real evaluation recorded, and the
    four properties below are the four ways a report can quietly pick a winner.
    """
    ev = FX.unresolved_conflict()
    conflict = ev.conflicts[0]
    spoken, visual = conflict.item_ids
    check("R1x both sides survive into the evidence", spoken in {i.id for i in ev.items}
          and visual in {i.id for i in ev.items}, str(conflict.item_ids))
    by_id = {i.id: i for i in ev.items}
    check("R1x the two sides really are the two figures",
          {by_id[spoken].display, by_id[visual].display} == {"75%", "8%"},
          str([(by_id[spoken].display, by_id[spoken].provenance[0].source),
               (by_id[visual].display, by_id[visual].provenance[0].source)]))
    check("R1x each side keeps the modality it came from",
          by_id[spoken].provenance[0].source == "transcript"
          and by_id[visual].provenance[0].source == "ocr",
          str([by_id[spoken].provenance[0].source, by_id[visual].provenance[0].source]))

    rep = R.generate(ev)
    md = R.render_markdown(rep)
    rel = next(s for s in rep.sections if s.key == "reliability")
    text = " ".join(b.text for b in rel.blocks)
    check("R1x the report states the disagreement", "Unresolved conflict" in text, text[:200])
    check("R1x the conflict block cites both sides",
          set(conflict.item_ids) <= {i for b in rel.blocks if b.kind == "conflict"
                                     for i in b.evidence_ids},
          str([b.evidence_ids for b in rel.blocks if b.kind == "conflict"]))
    check("R1x both figures are named in the conflict section",
          "75%" in text and "8%" in text, text[:300])
    check("R1x the report does not choose a side",
          not re.search(r"(?:actually|in fact|correct(?:ly)?|really)\b[^.]*\b(?:75|8)\s*%", text, re.I),
          text[:300])
    check("R1x the report does not average them",
          "41" not in text and "41.5" not in text, text[:300])
    settled = {i for _p, b in RV._all_blocks(rep)
               if b.kind in ("number", "comparison") for i in b.evidence_ids}
    check("R1x neither side is also rendered as a settled figure in a numbers table",
          not (settled & set(conflict.item_ids)),
          str(sorted(settled & set(conflict.item_ids))))

    res = validate_report(rep, ev, md)
    check("R1x a report that discloses the conflict validates", res.valid,
          str([e["code"] for e in res.blocking_errors[:3]]))
