"""Deterministic evidence fixtures for the report layer.

Fifteen shapes of video, built in code rather than checked in as data, because a
fixture that is a JSON blob goes stale silently: change the model and the blob
still says whatever it said about the old one. These build an `Evidence` through
the same constructors the pipeline uses, so a change to the model breaks them
loudly instead of quietly.

Each fixture is a *shape*, not a video. What it has to get right is the report's
behaviour under that shape: `simple_talking_head` has nothing but speech and must
still produce a readable report; `unreadable_chart` has a chart nobody could read
and must not state a figure for it; `unresolved_conflict` has two sources that
disagree and must show both; `highly_repetitive` says one thing eleven times and
must not turn that into eleven paragraphs.
"""
from __future__ import annotations

from evidence_model import (
    SCHEMA_VERSION, Conflict, Evidence, Item, Segment, Snippet, VisualFrame,
)
from evidence_provenance import (
    metadata_provenance, ocr_provenance, transcript_provenance, vlm_provenance,
)


def _snips(pairs) -> list[Snippet]:
    out = []
    for i, (t, txt) in enumerate(pairs, start=1):
        out.append(Snippet(id=f"s{i:03d}", start_s=t, end_s=t + 8.0, text=txt))
    return out


def _video(title="Fixture video", channel="Fixture Channel", dur=600.0, **kw) -> dict:
    v = {"id": "FIXTURE", "url": "https://www.youtube.com/watch?v=FIXTURE",
         "title": title, "channel": channel, "upload_date": "20260101",
         "duration_s": dur, "description": "", "tags": [], "thumbnail": None,
         "chapters": [], "sponsors": []}
    v.update(kw)
    return v


def _frame(fid, t=None, ocr=None, status="ok", vlm=None, vlm_status=None) -> VisualFrame:
    return VisualFrame(frame_id=fid, file=f"frames/{fid}.jpg", kind="scene", t_start=t,
                       ocr_text=ocr, ocr_status=status, ocr_confidence=0.95 if ocr else None,
                       ocr_backend="vision" if ocr else None,
                       vlm_caption=(vlm or {}).get("summary") if vlm else None,
                       vlm_status=vlm_status or ("ok" if vlm else None),
                       vlm_backend="openai" if vlm else None,
                       vlm_confidence=(vlm or {}).get("confidence") if vlm else None,
                       vlm_observation=vlm)


def _ev(video, snippets, segments, items, visual=None, conflicts=None) -> Evidence:
    ev = Evidence(schema_version=SCHEMA_VERSION, video=video,
                  transcript={"language": "en", "language_source": "captions",
                              "is_generated": False, "asr_backend": None,
                              "asr_model": None,
                              "word_count": sum(len(s.text.split()) for s in snippets),
                              "snippet_count": len(snippets)},
                  snippets=snippets, segments=segments, items=items,
                  visual=visual or [], conflicts=conflicts or [])
    for seg in segments:
        for sid in seg.snippet_ids:
            sn = next((s for s in snippets if s.id == sid), None)
            if sn:
                sn.segment_id = seg.id
    return ev


def _seg(i, title, start, end, snippet_ids, item_ids, frames=None, source="chapter") -> Segment:
    return Segment(id=f"seg-{i:02d}", start_s=start, end_s=end, title=title,
                   title_source=source, snippet_ids=list(snippet_ids),
                   item_ids=list(item_ids), frame_ids=list(frames or []))


def _it(n, itype, value, prov, **kw) -> Item:
    return Item(id=f"itm-{n:03d}", type=itype, value=value, provenance=prov, **kw)


# --------------------------------------------------------------------------- #
# 1. simple talking head — speech only, no visual layer at all
# --------------------------------------------------------------------------- #

def simple_talking_head() -> Evidence:
    s = _snips([(0.0, "Welcome back to the channel. Today I want to talk about caching."),
                (12.0, "A cache sits between your application and the slow database."),
                (40.0, "The whole point is that the second read never touches the database."),
                (95.0, "That is all for today, thanks for watching.")])
    items = [
        _it(1, "topic", "Caching explained", [metadata_provenance("title")],
            notes="uploader-supplied title; not a verified summary"),
        _it(2, "organization", "Fixture Channel", [metadata_provenance("channel")]),
        _it(3, "link", "https://www.youtube.com/watch?v=FIXTURE",
            [metadata_provenance("url")], display="https://www.youtube.com/watch?v=FIXTURE"),
        _it(4, "claim", "A cache sits between the application and the database",
            [transcript_provenance(s[1])], verbatim=s[1].text, importance="high"),
        _it(5, "claim", "The second read avoids the database entirely",
            [transcript_provenance(s[2])], verbatim=s[2].text, importance="high"),
        _it(6, "definition", "a stored copy of data kept so it need not be fetched again",
            [transcript_provenance(s[1])], subject="cache", importance="medium"),
    ]
    segs = [_seg(1, "Intro", 0.0, 60.0, ["s001", "s002", "s003"], ["itm-001", "itm-004", "itm-006"]),
            _seg(2, "Wrap-up", 60.0, 110.0, ["s004"], ["itm-005"])]
    return _ev(_video("Caching, explained", dur=110.0), s, segs, items)


# --------------------------------------------------------------------------- #
# 2. technical lecture — dense, numbered, many segments
# --------------------------------------------------------------------------- #

def technical_lecture() -> Evidence:
    s = _snips([(0.0, "This lecture covers retrieval augmented generation from first principles."),
                (30.0, "The retriever returns 32 GB worth of context to the generator."),
                (75.0, "Retrieval latency was 120 ms on the small index and 480 ms on the large one."),
                (140.0, "We evaluated three checkpoints against a human labelled set."),
                (210.0, "The 8B checkpoint scored 71 percent while the 70B scored 79 percent."),
                (300.0, "That concludes the retrieval section.")])
    items = [
        _it(1, "topic", "Retrieval augmented generation", [metadata_provenance("title")],
            notes="uploader-supplied title; not a verified summary"),
        _it(2, "organization", "Fixture Channel", [metadata_provenance("channel")]),
        _it(3, "claim", "The retriever feeds context to the generator",
            [transcript_provenance(s[0])], verbatim=s[0].text, importance="high"),
        _it(4, "number", "32", [transcript_provenance(s[1])], display="32 GB", unit="GB",
            subject="retriever context", importance="high"),
        _it(5, "number", "120", [transcript_provenance(s[2])], display="120 ms", unit="ms",
            subject="retrieval latency on the small index", importance="high"),
        _it(6, "number", "480", [transcript_provenance(s[2])], display="480 ms", unit="ms",
            subject="retrieval latency on the large index", importance="high"),
        _it(7, "number", "8", [transcript_provenance(s[4])], display="8B", unit="B",
            subject="checkpoint size", importance="medium"),
        _it(8, "number", "71", [transcript_provenance(s[4])], display="71%", unit="%",
            subject="score of the 8B checkpoint", importance="high"),
        _it(9, "number", "79", [transcript_provenance(s[4])], display="79%", unit="%",
            subject="score of the 70B checkpoint", importance="high"),
        _it(10, "model", "retriever", [transcript_provenance(s[0])], importance="medium"),
        _it(11, "model", "generator", [transcript_provenance(s[0])], importance="medium"),
        _it(12, "comparison", "8B vs 70B",
            [transcript_provenance(s[4])], subject="8B", verbatim=s[4].text,
            detail={"subject_a": "8B", "subject_b": "70B",
                    "dimensions": [{"dimension": "score", "a": "71%", "b": "79%"}]},
            importance="high", notes="contrast stated by the source"),
        _it(13, "claim", "Three checkpoints were evaluated against a human labelled set",
            [transcript_provenance(s[3])], verbatim=s[3].text, importance="high"),
    ]
    segs = [_seg(1, "Background", 0.0, 70.0, ["s001", "s002"], ["itm-001", "itm-003", "itm-004", "itm-010", "itm-011"]),
            _seg(2, "Measurements", 70.0, 200.0, ["s003", "s004"], ["itm-005", "itm-006", "itm-013"]),
            _seg(3, "Results", 200.0, 290.0, ["s005"], ["itm-007", "itm-008", "itm-009", "itm-012"]),
            _seg(4, "Close", 290.0, 320.0, ["s006"], [])]
    return _ev(_video("RAG from first principles", dur=320.0), s, segs, items)


# --------------------------------------------------------------------------- #
# 3. chart-heavy — model read a chart, values carry row and column
# --------------------------------------------------------------------------- #

def chart_heavy() -> Evidence:
    s = _snips([(0.0, "Here is the throughput comparison across the three models."),
                (20.0, "Qwen leads on throughput while latency is worse.")])
    obs = {"summary": "A bar chart of throughput by model",
           "content_type": "chart", "confidence": 0.62, "backend": "openai",
           "numbers": [{"value": "32", "unit": "tok/s", "row": "Qwen", "column": "throughput"},
                       {"value": "24", "unit": "tok/s", "row": "Llama", "column": "throughput"}],
           "relationships": [{"source": "Qwen", "target": "Llama", "kind": "ordering",
                              "label": "higher throughput", "certain": True}]}
    f = _frame("s0001", 20.0, ocr="throughput tok/s  Qwen  Llama  latency ms",
               vlm=obs)
    items = [
        _it(1, "topic", "Throughput comparison", [metadata_provenance("title")],
            notes="uploader-supplied title; not a verified summary"),
        _it(2, "organization", "Fixture Channel", [metadata_provenance("channel")]),
        _it(3, "relationship", "Qwen → Llama",
            [vlm_provenance(f, obs["summary"], confidence=0.62)],
            detail={"from": "Qwen", "to": "Llama", "label": "higher throughput",
                    "kind": "ordering", "certain": True, "content_type": "chart"},
            importance="high", notes="connection seen in s0001 (chart); model-derived, not verified"),
        _it(4, "claim", "Qwen leads on throughput while latency is worse",
            [transcript_provenance(s[1])], verbatim=s[1].text, importance="high"),
    ]
    segs = [_seg(1, "Chart", 0.0, 60.0, ["s001", "s002"], ["itm-004", "itm-003"], ["s0001"])]
    return _ev(_video("Throughput comparison", dur=60.0), s, segs, items, visual=[f])


# --------------------------------------------------------------------------- #
# 4. table-heavy — a table read with cells, no narration of it
# --------------------------------------------------------------------------- #

def table_heavy() -> Evidence:
    s = _snips([(0.0, "This is the full pricing table for the three tiers.")])
    f = _frame("s0002", 8.0, ocr="Tier  Pro  Max  Price  $20  $50  $120",
               vlm={"summary": "A three-column pricing table", "content_type": "table",
                    "confidence": 0.58,
                    "numbers": [{"value": "20", "unit": "USD", "row": "Pro"},
                                {"value": "50", "unit": "USD", "row": "Max"},
                                {"value": "120", "unit": "USD", "row": "Enterprise"}]})
    items = [
        _it(1, "topic", "Pricing tiers", [metadata_provenance("title")],
            notes="uploader-supplied title; not a verified summary"),
        _it(2, "organization", "Fixture Channel", [metadata_provenance("channel")]),
        _it(3, "number", "20", [ocr_provenance(f, "$20", confidence=0.95)], display="$20",
            unit="USD", subject="Pro tier", status="asserted", importance="high"),
        _it(4, "number", "50", [ocr_provenance(f, "$50", confidence=0.95)], display="$50",
            unit="USD", subject="Max tier", status="asserted", importance="high"),
        _it(5, "claim", "Three pricing tiers are shown",
            [transcript_provenance(s[0])], verbatim=s[0].text, importance="medium"),
    ]
    segs = [_seg(1, "Pricing", 0.0, 40.0, ["s001"], ["itm-001", "itm-005"], ["s0002"])]
    return _ev(_video("Pricing tiers", dur=40.0), s, segs, items, visual=[f])


# --------------------------------------------------------------------------- #
# 5. code tutorial — a real procedure
# --------------------------------------------------------------------------- #

def code_tutorial() -> Evidence:
    s = _snips([(0.0, "First install the package with pip install litellm."),
                (15.0, "Next configure the model in the settings file."),
                (30.0, "Then run the command to verify it works."),
                (50.0, "That is the whole setup.")])
    items = [
        _it(1, "topic", "Setting up the package", [metadata_provenance("title")],
            notes="uploader-supplied title; not a verified summary"),
        _it(2, "organization", "Fixture Channel", [metadata_provenance("channel")]),
        _it(3, "step", "Install the package", [transcript_provenance(s[0])],
            detail={"index": 1, "group": "seg-01"}, importance="high", verbatim=s[0].text),
        _it(4, "step", "Configure the model", [transcript_provenance(s[1])],
            detail={"index": 2, "group": "seg-01"}, importance="high", verbatim=s[1].text),
        _it(5, "step", "Run the verification command", [transcript_provenance(s[2])],
            detail={"index": 3, "group": "seg-01"}, importance="high", verbatim=s[2].text),
        _it(6, "claim", "Three steps are needed for the setup",
            [transcript_provenance(s[0])], verbatim=s[0].text, importance="medium"),
    ]
    segs = [_seg(1, "Setup", 0.0, 70.0, ["s001", "s002", "s003", "s004"],
                 ["itm-001", "itm-003", "itm-004", "itm-005", "itm-006"])]
    return _ev(_video("Package setup", dur=70.0), s, segs, items)


# --------------------------------------------------------------------------- #
# 6. UI demonstration — an action on screen, not a narration of it
# --------------------------------------------------------------------------- #

def ui_demonstration() -> Evidence:
    s = _snips([(0.0, "Watch what happens when I click export.")])
    f = _frame("s0004", 5.0, ocr="Export  Cancel  csv",
               vlm={"summary": "An export dialog with csv selected", "content_type": "application_ui",
                    "confidence": 0.6, "actions": [{"description": "the export dialog is open with csv selected"}]})
    items = [
        _it(1, "topic", "Exporting data", [metadata_provenance("title")],
            notes="uploader-supplied title; not a verified summary"),
        _it(2, "organization", "Fixture Channel", [metadata_provenance("channel")]),
        _it(3, "visual_finding", "the export dialog is open with csv selected",
            [vlm_provenance(f, f.vlm_caption, confidence=0.6)],
            detail={"kind": "action", "content_type": "application_ui", "target": "export dialog"},
            importance="medium", notes="operation visible in s0004; a visible state, not an inferred interaction"),
        _it(4, "claim", "Clicking export opens a dialog",
            [transcript_provenance(s[0])], verbatim=s[0].text, importance="medium"),
    ]
    segs = [_seg(1, "Demo", 0.0, 30.0, ["s001"], ["itm-001", "itm-004"], ["s0004"])]
    return _ev(_video("Exporting data", dur=30.0), s, segs, items, visual=[f])


# --------------------------------------------------------------------------- #
# 7. visual-only number — never spoken, only on a frame
# --------------------------------------------------------------------------- #

def visual_only_number() -> Evidence:
    s = _snips([(0.0, "This slide has the numbers, let us look at them.")])
    f = _frame("s0007", 12.0, ocr="throughput  18 tok/s  24 tok/s")
    items = [
        _it(1, "topic", "Slide figures", [metadata_provenance("title")],
            notes="uploader-supplied title; not a verified summary"),
        _it(2, "organization", "Fixture Channel", [metadata_provenance("channel")]),
        _it(3, "number", "18", [ocr_provenance(f, "18 tok/s", confidence=0.95)],
            display="18 tok/s", unit="tok/s", subject="throughput", status="asserted",
            importance="high"),
        _it(4, "number", "24", [ocr_provenance(f, "24 tok/s", confidence=0.95)],
            display="24 tok/s", unit="tok/s", subject="throughput", status="asserted",
            importance="high"),
    ]
    segs = [_seg(1, "Slide", 0.0, 30.0, ["s001"], ["itm-001", "itm-003", "itm-004"], ["s0007"])]
    return _ev(_video("Slide figures", dur=30.0), s, segs, items, visual=[f])


# --------------------------------------------------------------------------- #
# 8. transcript-only number — never on screen
# --------------------------------------------------------------------------- #

def transcript_only_number() -> Evidence:
    s = _snips([(0.0, "The dataset has 1.2 million rows and took 3 hours to build.")])
    items = [
        _it(1, "topic", "Dataset size", [metadata_provenance("title")],
            notes="uploader-supplied title; not a verified summary"),
        _it(2, "organization", "Fixture Channel", [metadata_provenance("channel")]),
        _it(3, "number", "1.2", [transcript_provenance(s[0])], display="1.2 million",
            unit="million", subject="dataset rows", status="asserted", importance="high"),
        _it(4, "number", "3", [transcript_provenance(s[0])], display="3 hours", unit="hours",
            subject="time taken to build the dataset", status="asserted", importance="medium"),
    ]
    segs = [_seg(1, "Dataset", 0.0, 30.0, ["s001"], ["itm-001", "itm-003", "itm-004"])]
    return _ev(_video("Dataset size", dur=30.0), s, segs, items)


# --------------------------------------------------------------------------- #
# 9. corroboration — the same figure from speech, from the frame and from a model
# --------------------------------------------------------------------------- #

def corroboration() -> Evidence:
    s = _snips([(0.0, "The context window is 32K tokens.")])
    f = _frame("s0009", 6.0, ocr="context 32K tokens",
               vlm={"summary": "A slide stating a 32K context window",
                    "content_type": "slide", "confidence": 0.6,
                    "numbers": [{"value": "32", "unit": "K tokens", "row": "context"}]})
    items = [
        _it(1, "topic", "Context window", [metadata_provenance("title")],
            notes="uploader-supplied title; not a verified summary"),
        _it(2, "organization", "Fixture Channel", [metadata_provenance("channel")]),
        _it(3, "number", "32", [transcript_provenance(s[0]), ocr_provenance(f, "32K", confidence=0.95)],
            display="32K tokens", unit="tokens", subject="context window", status="asserted",
            importance="high"),
        _it(4, "claim", "The context window is 32K tokens",
            [transcript_provenance(s[0]), ocr_provenance(f, "32K", confidence=0.95)],
            verbatim=s[0].text, importance="high"),
    ]
    segs = [_seg(1, "Context", 0.0, 30.0, ["s001"], ["itm-001", "itm-004"], ["s0009"])]
    return _ev(_video("Context window", dur=30.0), s, segs, items, visual=[f])


# --------------------------------------------------------------------------- #
# 10. unresolved conflict — narration and frame disagree, and nobody wins
# --------------------------------------------------------------------------- #

def unresolved_conflict() -> Evidence:
    s = _snips([(0.0, "Accuracy is 75 percent according to the leaderboard.")])
    f = _frame("s0010", 9.0, ocr="accuracy 8%")
    a = _it(1, "number", "75", [transcript_provenance(s[0])], display="75%", unit="%",
            subject="accuracy", status="conflicting", importance="high")
    b = _it(2, "number", "8", [ocr_provenance(f, "8%", confidence=0.95)], display="8%", unit="%",
            subject="accuracy", status="conflicting", importance="high")
    items = [
        _it(3, "topic", "Accuracy figures", [metadata_provenance("title")],
            notes="uploader-supplied title; not a verified summary"),
        _it(4, "organization", "Fixture Channel", [metadata_provenance("channel")]),
        a, b,
    ]
    conflict = Conflict(item_ids=["itm-001", "itm-002"], kind="value-mismatch",
                        statement="The narration states 75% while the frame reads 8%.")
    segs = [_seg(1, "Accuracy", 0.0, 30.0, ["s001"], ["itm-003"], ["s0010"])]
    return _ev(_video("Accuracy figures", dur=30.0), s, segs, items, visual=[f],
               conflicts=[conflict])


# --------------------------------------------------------------------------- #
# 11. hedged claim — "could cost $500" must not become "$500"
# --------------------------------------------------------------------------- #

def hedged_claim() -> Evidence:
    s = _snips([(0.0, "A hosted deployment could cost $500 a month at that scale.")])
    items = [
        _it(1, "topic", "Hosting cost", [metadata_provenance("title")],
            notes="uploader-supplied title; not a verified summary"),
        _it(2, "organization", "Fixture Channel", [metadata_provenance("channel")]),
        _it(3, "number", "500", [transcript_provenance(s[0])], display="$500", unit="USD",
            subject="monthly cost of a hosted deployment", status="hedged", importance="high",
            notes="spoken as a hedge ('could'); hedge preserved"),
        _it(4, "claim", "The cost is presented as a possibility rather than a figure",
            [transcript_provenance(s[0])], verbatim=s[0].text, importance="medium"),
    ]
    segs = [_seg(1, "Cost", 0.0, 30.0, ["s001"], ["itm-001", "itm-004"])]
    return _ev(_video("Hosting cost", dur=30.0), s, segs, items)


# --------------------------------------------------------------------------- #
# 12. noisy OCR — interface furniture that must not reach the report
# --------------------------------------------------------------------------- #

def noisy_ocr() -> Evidence:
    s = _snips([(0.0, "The dashboard is showing the current queue depth.")])
    noise = ["12:04", "1080p", "0:00 / 14:22", "|||", "x", "99%", "buffering spinner"]
    f = _frame("s0012", 11.0, ocr=" | ".join(noise + ["queue depth 412", "workers 8"]))
    items = [
        _it(1, "topic", "Queue dashboard", [metadata_provenance("title")],
            notes="uploader-supplied title; not a verified summary"),
        _it(2, "organization", "Fixture Channel", [metadata_provenance("channel")]),
        _it(3, "number", "412", [ocr_provenance(f, "queue depth 412", confidence=0.93)],
            display="412", subject="queue depth", status="asserted", importance="high"),
        _it(4, "claim", "The dashboard shows the current queue depth",
            [transcript_provenance(s[0])], verbatim=s[0].text, importance="medium"),
    ]
    segs = [_seg(1, "Dashboard", 0.0, 30.0, ["s001"], ["itm-001", "itm-004"], ["s0012"])]
    return _ev(_video("Queue dashboard", dur=30.0), s, segs, items, visual=[f])


# --------------------------------------------------------------------------- #
# 13. unreadable chart — the axis could not be read, so no figure is stated
# --------------------------------------------------------------------------- #

def unreadable_chart() -> Evidence:
    s = _snips([(0.0, "This chart shows the trend over the quarter.")])
    f = _frame("s0013", 7.0, ocr="chart",
               vlm={"summary": "A line chart whose axis labels are illegible",
                    "content_type": "chart", "confidence": 0.4,
                    "uncertainties": [{"text": "the y-axis labels",
                                       "reason": "not legible at the sampled resolution"}]})
    items = [
        _it(1, "topic", "Quarterly trend", [metadata_provenance("title")],
            notes="uploader-supplied title; not a verified summary"),
        _it(2, "organization", "Fixture Channel", [metadata_provenance("channel")]),
        _it(3, "visual_finding", "the y-axis labels",
            [vlm_provenance(f, f.vlm_caption, confidence=0.4)],
            detail={"kind": "uncertainty", "content_type": "chart", "value": "the y-axis labels"},
            status="unreadable", importance="low",
            notes="the model marked 'the y-axis labels' as inferred rather than read in s0013"),
        _it(4, "claim", "A trend over the quarter is shown",
            [transcript_provenance(s[0])], verbatim=s[0].text, importance="medium"),
    ]
    segs = [_seg(1, "Trend", 0.0, 30.0, ["s001"], ["itm-001", "itm-004"], ["s0013"])]
    return _ev(_video("Quarterly trend", dur=30.0), s, segs, items, visual=[f])


# --------------------------------------------------------------------------- #
# 14. sparse — nothing optional has any content, so those sections must vanish
# --------------------------------------------------------------------------- #

def empty_optional_sections() -> Evidence:
    return simple_talking_head()


# --------------------------------------------------------------------------- #
# 15. highly repetitive — one sentence, many times
# --------------------------------------------------------------------------- #

def highly_repetitive() -> Evidence:
    texts = [(float(i * 20), f"Step {i} is to rerun the benchmark and compare the numbers.") for i in range(11)]
    s = _snips(texts)
    items = [_it(1, "topic", "Repeated benchmark advice", [metadata_provenance("title")],
                 notes="uploader-supplied title; not a verified summary"),
             _it(2, "organization", "Fixture Channel", [metadata_provenance("channel")])]
    ids = ["itm-001"]
    for i in range(11):
        n = 10 + i
        items.append(_it(n, "step", f"Rerun the benchmark and compare the numbers",
                         [transcript_provenance(s[i])], detail={"index": i + 1, "group": "seg-01"},
                         importance="low", verbatim=s[i].text))
        ids.append(f"itm-{n:03d}")
    segs = [_seg(1, "Advice", 0.0, 220.0, [x.id for x in s], ids)]
    return _ev(_video("Repeated benchmark advice", dur=220.0), s, segs, items)


FIXTURES = {
    "simple_talking_head": simple_talking_head,
    "technical_lecture": technical_lecture,
    "chart_heavy": chart_heavy,
    "table_heavy": table_heavy,
    "code_tutorial": code_tutorial,
    "ui_demonstration": ui_demonstration,
    "visual_only_number": visual_only_number,
    "transcript_only_number": transcript_only_number,
    "corroboration": corroboration,
    "unresolved_conflict": unresolved_conflict,
    "hedged_claim": hedged_claim,
    "noisy_ocr": noisy_ocr,
    "unreadable_chart": unreadable_chart,
    "empty_optional_sections": empty_optional_sections,
    "highly_repetitive": highly_repetitive,
}


# --------------------------------------------------------------------------- #
# 16. synthesis qualification — the one fixture a provider is measured against
# --------------------------------------------------------------------------- #

def synthesis_qualification() -> Evidence:
    """Seven things at once, which is the minimum a provider can be judged on.

    A smoke test that only checks "did the endpoint answer" proves nothing about
    whether the model can write this report, and one that runs a whole video
    proves it expensively. This is the middle: a document carrying, deliberately,
    one of each thing the writer has to get right —

    * a **central topic**, so there is a thesis to find rather than a list;
    * a **claim**, the kind the validator will re-derive against its evidence;
    * a **number**, so a figure has to survive being written as prose;
    * a **visual observation**, so the model must use a frame it did not read;
    * an **entity**, so a name must survive extraction;
    * an **uncertainty**, which must be kept rather than resolved; and
    * a **conflict**, which must be reported on both sides.

    Each is small enough to read in full and check by eye, which is the property
    that lets a failure be diagnosed rather than merely observed. A writer that
    drops the hedge, picks a side of the conflict, or invents a number is caught
    by `synth_validate`, and the doctor reports which of the seven it managed.
    """
    s = _snips([
        (0.0, "This video is about whether a vector database is still worth it in 2026."),
        (10.0, "The claim in the vendor's own benchmark is that it cuts p99 latency by 3x."),
        (22.0, "Their table reports 40 ms p99 against 120 ms for the incumbent."),
        (34.0, "I could not read the rest of the chart, so the comparison is incomplete."),
    ])
    f = _frame("s0010", 11.0, ocr="p99 40 ms / 120 ms",
               vlm={"summary": "A two-row comparison table of p99 latency.",
                    "confidence": 0.65, "content_type": "table"})
    items = [
        _it(1, "topic", "Vector databases in 2026", [metadata_provenance("title")],
            importance="high", notes="uploader-supplied title; not a verified summary"),
        _it(2, "claim",
            "The vendor's benchmark claims a vector database cuts p99 latency by 3x",
            [transcript_provenance(s[1])], subject="vendor benchmark",
            verbatim=s[1].text, importance="high"),
        _it(3, "number", "40", [transcript_provenance(s[2])], display="40 ms", unit="ms",
            subject="table", importance="high", semantic_role="benchmark"),
        _it(4, "number", "120", [transcript_provenance(s[2])], display="120 ms", unit="ms",
            subject="incumbent", importance="high", semantic_role="benchmark"),
        _it(5, "visual_finding",
            "A two-row comparison table of p99 latency is shown on screen",
            [vlm_provenance(f, "p99 latency", confidence=0.65)], status="asserted",
            importance="medium", semantic_role=None,
            detail={"kind": "structure", "content_type": "table"},
            notes="model-derived; not a reading of the pixels and not verified"),
        _it(6, "entity", "Vector database", [transcript_provenance(s[0])],
            subject="Vector database", importance="medium",
            notes="named in this segment; type not established"),
        _it(7, "definition", "a nearest-neighbour index over embeddings",
            [transcript_provenance(s[0])], subject="vector database",
            status="hedged", importance="low",
            notes="the speaker's gloss, hedged in the source"),
        _it(8, "topic", "Fixture Channel", [metadata_provenance("channel")],
            notes="placeholder so provenance namespaces resolve"),
    ]
    conflict = Conflict(item_ids=["itm-002", "itm-005"], kind="source-disagreement",
                        statement=("The narration presents the vendor benchmark as the "
                                   "vendor's claim; the frame shows only the two numbers "
                                   "with no methodology."))
    segs = [_seg(1, "Is it still worth it", 0.0, 45.0,
                 ["s001", "s002", "s003", "s004"],
                 ["itm-001", "itm-002", "itm-003", "itm-004", "itm-005",
                  "itm-006", "itm-007"], ["s0010"])]
    return _ev(_video("Is a vector database still worth it in 2026?", dur=45.0), s, segs,
               items, visual=[f], conflicts=[conflict])
