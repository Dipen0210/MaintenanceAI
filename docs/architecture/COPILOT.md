# Engineer Copilot

The Phase 4 implementation uses actual local MiniLM embeddings and Qwen3 0.6B inference. It is a constrained evidence-selection RAG system: the model chooses evidence identifiers and diagnostic roles; the backend renders verified source passages and calculated observations. It does not freely generate procedures or numerical measurements. This design limits expressiveness, but makes each displayed technical claim attributable to stored evidence. It does not establish the truth of an approved manual or prove a physical root cause.

## Data and access

Original UTF-8 Markdown/text or extractable PDFs are stored beside the configured application database. PDF extraction preserves page numbers and extracts tables. Scanned PDFs without usable text require OCR outside this importer and are rejected explicitly. Imports are limited to 20 MiB, 100 PDF pages and 1,000 chunks. Embeddings are real normalized 384-dimensional MiniLM vectors stored as SQLite BLOBs. Retrieval filters document grants, plant, equipment applicability, reviewed chunks and current approval before cosine similarity ranking. SQLite exact vector search suits the local demo; a large plant document collection needs indexed vector storage and load testing.

New revisions are pending until an authorized administrator, manager or reliability engineer reviews passages. Hypotheses require explicit supporting model classes. Simulation-only documentation cannot be retrieved for a physical plant. Approval and revocation enforce asset scope and create audit events. The supplied DEMO-2 guide is authored synthetic test material, not an OEM manual or expert-approved plant procedure.

CMMS CSV imports preserve source, asset, historical timestamp, external identifier, record type, summary and details. Source/asset permissions apply to every read. Imported historical work orders and schedule records are distinct from future approved work. Internal maintenance drafts and confirmed schedules are now provided by the Phase 6 planning workflow and exposed through the same authorized read-only tools.

## Answer construction

1. Authorize the requested asset and existing conversation owner.
2. Refuse control, credential, protection-bypass and unsafe-intervention requests before inference.
3. Run narrow, typed read-only equipment tools. Exclude poor-quality, stale, rejected, incompatible and drift-flagged current diagnostic inputs.
4. Retrieve approved equipment passages. Match diagnostic hypotheses to current compatible non-normal model classes or an anomaly flag. A manual lookup may quote a passage without asserting that the current machine has that condition.
5. Send bounded evidence and authorized recent conversation context to the provider. Previous conversation is context, never fresh evidence.
6. Constrain local generation to a JSON schema of permitted evidence identifiers and roles. Independently validate returned identifiers, current permissions, unchanged source text and unsafe content in every evidence role.
7. Separate observations, predictions, hypotheses, alternatives, references, historical records, missing information and human-review requirements. Abstain when a diagnosis lacks adequate evidence. Provider/validation failures are explicitly unavailable; there is no fabricated fallback answer.

The local provider verifies revision-pinned artifact hashes and uses no remote code. An optional HTTP adapter supports HTTPS or loopback development services. An external provider has not been deployed or evaluated. The configured provider alone sees the authorized evidence sent for inference; do not configure a remote provider for sensitive plant data without appropriate organizational authorization.

## Persistence and traces

Sessions and investigation drafts are user-owned and asset-scoped. Every citation is reauthorized when opened, exported, or reused as conversation context. Revoked evidence makes affected historical answers unavailable. Draft exports contain evidence and engineer notes; they never authorize maintenance. Feedback is stored for later review.

Trace records include request hash, prompt hash/configuration version, retrieval IDs/similarities, tool operations, validation outcome, model identity, token counts and measured latency. Local hardware cost is unknown, represented by null rather than zero. Raw provider exceptions and secrets are not logged. The standalone smoke script records raw generated IDs only for its explicitly synthetic diagnostic fixture.

## Boundaries and limitations

- Eleven read-only tools are available; there is no arbitrary SQL, filesystem, URL or industrial-write tool.
- Regex screening is deliberately conservative and may reject benign quotations or negated warnings. It is not a complete prompt-injection detector. The stronger boundaries are approved/scoped evidence, constrained selection, independent citation checks, absence of write tools and mandatory human review.
- A small local model can omit relevant evidence or fail validation. Schema-valid output is not proof of diagnostic quality; Phase 5 must measure retrieval, completeness, abstention and regressions.
- No calibrated fault certainty, real plant connectivity, expert diagnostic certification or maintenance execution is claimed.
- Evidence is timestamped. A prior response is an investigation record, not a permanent assertion of current machine condition. Obtain fresh observations and rerun analysis for a new decision.
