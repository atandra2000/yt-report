# Project record

Dated reviews of the pipeline as it stood at the time. They are a record of what
was found and changed, not living documentation — where a document here
disagrees with the code, the code is right and the document is history.

| Document | Date | What it covers |
|---|---|---|
| [AUDIT](AUDIT-2026-09-25.md) | 2026-09-25 | First full technical and product audit of the extraction path, over 8 shipped reports. |
| [ACCEPTANCE](ACCEPTANCE-2026-09-26.md) | 2026-09-26 | Final acceptance and reliability pass over the completed architecture. |
| [HARDENING](HARDENING-2026-09-26.md) | 2026-09-26 | Failure-mode work: degradation, secret leakage, injection, reproducibility. |
| [REAL-PRODUCT-EVAL](REAL-PRODUCT-EVAL.md) | 2026-09-26 | Report quality judged on real videos rather than on checks passing. |
| [FINAL-LIVE-EVAL](FINAL-LIVE-EVAL-2026-09-26.md) | 2026-09-26 | Budgeted live runs against real VLM and synthesis providers. |

The current release status is [`RELEASE-CHECK.md`](../RELEASE-CHECK.md) at the
repo root. The behavioural contract is [`SKILL.md`](../SKILL.md); the agent
contract is [`AGENTS.md`](../AGENTS.md).
