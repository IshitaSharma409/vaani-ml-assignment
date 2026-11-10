# Requirements Checklist — The Relay

This checklist is derived from the complete four-page assignment brief and cross-referenced to `IMPLEMENTATION_PLAN.md`. Status reflects this analysis-only stage. `Not started` means no implementation or verification has been claimed. Phase references use P1–P5 for the five plan phases.

## System and corpus requirements

| ID | Requirement | Phase / component | Verification criterion | Status |
|---|---|---|---|---|
| SYS-01 | Build a small working pipeline in Python, C++, or Go; selected plan language is Python. | P1; project setup | Clean environment can run the app and import its selected integrations. | Not started |
| SYS-02 | Pipeline has at least four steps. | P2; LangGraph | Graph inspection/test demonstrates four or more explicit processing steps. | Not started |
| SYS-03 | Pipeline makes at least one tool call. | P2; retrieval tool | Trace/test records an actual `search_research_library` invocation. | Not started |
| SYS-04 | Pipeline has at least one branch or loop. | P2; workflow | Graph tests cover conditional path and prove loop/retry bounds. | Not started |
| COR-01 | Answer only from the fixed library; unsupported information is stated as unavailable rather than guessed. | P2; prompts, evidence gate, refusal | Supported and unsupported paper-grounded questions are checked; no unsupported answer is released. | Not started |
| COR-02 | Retrieve exact Attention Is All You Need, arXiv 1706.03762 v7. | P1–P2; corpus/manifest | PDF identity and version are verified and recorded. | Downloaded and verified; ingestion pending P2 |
| COR-03 | Retrieve exact Moshi, arXiv 2410.00037 v1. | P1–P2; corpus/manifest | PDF identity and version are verified and recorded. | Downloaded and verified; ingestion pending P2 |
| COR-04 | Retrieve exact Dynamic Chunking, arXiv 2507.07955 v2. | P1–P2; corpus/manifest | PDF identity and version are verified and recorded. | Downloaded and verified; ingestion pending P2 |
| COR-05 | Every answer cites its source location. | P2; retrieval metadata/citation validator | Each released substantive claim has a citation traceable to a retrieved chunk and valid paper/page. | Not started |
| COR-06 | Never invent pages, sections, sources, or citations. | P2; deterministic validator | Tests reject unknown source IDs, wrong versions, missing/out-of-range pages. | Not started |
| COR-07 | Do not release an unverified draft; preserve answer quality under pressure. | P2–P3; validation gate | Integration test shows output is released only after validation, or a safe refusal/fallback is returned. | Not started |

## Provider and stand-in requirements

| ID | Requirement | Phase / component | Verification criterion | Status |
|---|---|---|---|---|
| LLM-01 | Use a real hosted model on the user's own Groq free tier. | P1–P2; Groq backend | Authenticated `ChatGroq` call succeeds; run is identified as real. | Three real streamed `ChatGroq` baseline calls succeeded; end-to-end pipeline pending P2 |
| LLM-02 | Use own account/key, do not use extra accounts or bypass terms/limits. | P1/P4; credentials and experiment policy | Record configuration provenance without exposing key; no alternate account or limit circumvention. | Local key used for the authorized probe and kept out of outputs; account plan/ownership not independently inspected |
| LLM-03 | Discover real service limits from documentation and provider responses/account evidence. | P1/P5; provider evidence | Table 1 gives value, method, confidence; unknowns are explicitly “not measured”/unspecified. | RPD=1,000 and TPM=8,000 from successful response headers; RPM=30 and TPD=200,000 are current official free-plan references only; concurrency/recovery not measured |
| LLM-04 | Send only a modest number of real baseline requests first. | P1; probe script | Raw records show controlled request count, prompt, timestamps, model, usage, and limitations. | Complete: three short real calls, five-second spacing, with raw timing, usage, and headers preserved |
| LLM-05 | Keep real load within safe free-tier capacity; heavy load uses stand-in. | P1/P4; scenario runner | Scenario metadata identifies backend; no uncontrolled/burst/overload real calls. | Not started |
| SIM-01 | Implement a controllable stand-in model service, separate and clearly labelled from Groq. | P3; stand-in app/backend | Stand-in can be selected while retaining same application workflow; outputs/results identify simulation. | Not started |
| SIM-02 | Stand-in can be slow and stream its reply piece by piece. | P3; simulator | Configured delays and streamed chunks are observed in local integration checks. | Not started |
| SIM-03 | Stand-in can fail and reject on configured limits. | P3; simulator/limits | Controlled failures and synthetic limit responses are reproducible from profile. | Not started |
| SIM-04 | Stand-in becomes slower under load. | P3; load model | Concurrent-load scenario produces configured load-dependent delay; formula/profile is recorded. | Not started |
| SIM-05 | Calibrate stand-in from real observations and explain where it is least like Groq. | P3/P5; profile/report | Profile links real sample IDs and labels assumptions; report describes material limitations. | Not started |

## Experiments, evaluation, and measurement

| ID | Requirement | Phase / component | Verification criterion | Status |
|---|---|---|---|---|
| EXP-01 | Run at least three load shapes, including steady, sudden burst, and sustained overload. | P4; YAML scenarios/runner | Raw request records exist for each shape; heavy traffic is simulated. | Not started |
| EXP-02 | Cover the five required table scenarios: steady, burst, sustained overload, per-minute limit halved, daily budget exhausted. | P4; scenario configuration | All five named scenarios have fixed config and raw result rows. | Not started |
| EXP-03 | Preserve raw results from both real and stand-in runs. | P1/P4; `results.csv`/metadata | Raw file identifies `source` for every request and retains failure outcomes. | Not started |
| EXP-04 | Evaluate at least 15 questions written for this corpus. | P4; evaluation dataset | At least 15 original questions have evidence-backed reference notes and expected answerability. | Not started |
| EXP-05 | Report answer quality on those questions. | P4–P5; grading | Each output is manually graded using the four allowed grades with reason/evidence. | Not started |
| EXP-06 | Compare at least three retrieval settings: piece size, pieces fetched, and reranking. | P2/P4; retrieval config/evaluation | Three+ settings record exact chunk size, overlap, top-k, reranking, and comparable results. | Not started |
| EXP-07 | Report first-token and total-time percentiles, including median and slow end, sample sizes, by scenario. | P3–P5; telemetry/report | Tables state p50/p95/p99 and sample counts/populations, computed from raw data. | Not started |
| EXP-08 | Report failures and requests given up/timeouts. | P3–P5; outcome fields/metrics | Outcome percentages are derived from all offered requests with categories retained. | Not started |
| EXP-09 | Report tokens used per answer and retries. | P3–P5; usage/retry telemetry | Aggregate all model calls per user request, with provider/estimated token provenance. | Not started |
| EXP-10 | Report one genuinely slow request split into waiting, working, retrying and locate delay relative to first token. | P3/P4; event trace | Timestamp-backed decomposition reconciles to total within documented instrumentation overhead. | Not started |
| EXP-11 | Include a measured number not inferable without execution. | P4–P5; result narrative | Cite a genuine run-derived figure and its sample/raw records. | Not started |
| EXP-12 | Describe retry policy and one case where retrying made things worse. | P3/P4/P5; retry policy/scenario | Raw trace supports a real case or a clearly labelled controlled stand-in case. | Not started |
| EXP-13 | Explain behavior at minute and daily limits: delayed, dropped/rejected, weaker answers, and rationale. | P3/P4/P5; limiter/report | Tests and raw records demonstrate terminal outcomes and no extra call after daily exhaustion. | Not started |
| EXP-14 | Describe exact user experience at 3 PM after daily budget is gone. | P4/P5; budget-exhaustion path | Implemented response is quoted accurately; no reset time is claimed unless verified. | Not started |
| EXP-15 | Identify three actual ways this library misleads a naive retriever, each with example question. | P4/P5; retrieval analysis | Each failure is evidenced by actual paper text/retrieval run, not a hypothetical candidate. | Not started |
| EXP-16 | Explain quality-versus-cost choice and near-limit retrieval behavior. | P4/P5; selected retrieval policy | Choice follows measured quality, tokens, and latency; citation checks remain enabled. | Not started |

## Required raw file and tables

| ID | Requirement | Phase / component | Verification criterion | Status |
|---|---|---|---|---|
| DATA-01 | Submit one `results.csv`, one row per request, with mandatory columns unchanged. | P3/P5; CSV writer/schema validation | Header includes all 12 exact names below; unique request IDs; rows are end-user requests. | Not started |
| DATA-02 | Preserve allowed `status` values: `ok`, `rate_limited`, `timeout`, `error`. | P3/P5; schema validator | Every status validates; richer outcomes use additional fields. | Not started |
| DATA-03 | Preserve allowed answer grades: `correct`, `wrong`, `refused_correctly`, `refused_wrongly`. | P4/P5; grader/schema | Every populated grade is an allowed value with grading basis. | Not started |
| DATA-04 | Make every table figure recomputable from raw data/evidence; use milliseconds and one decimal percentages. | P3–P5; telemetry, metric script | Deterministic metrics script reproduces tables; unspecified measures explicitly say “not measured.” | Not started |
| DATA-05 | Fill Table 1 with real limits, discovery method, confidence, in-flight requests, limit response, recovery time. | P1/P5; provider evidence | Account/documented/observed values distinguished; no invented limit or recovery time. | Not started |
| DATA-06 | Fill Tables 2a and 2b with the same required scenarios and all required metrics. | P4/P5; metrics generation | Rows match scenario set; all figures script-derived from raw data/config/evidence. | Not started |
| DATA-07 | Fill Table 3 for at least three settings with grades, tokens, first-token and total time. | P4/P5; retrieval evaluation | Exact settings and sample sizes shown; grades follow rubric. | Not started |
| DATA-08 | Preserve genuine assistant conversation history or prompts if export is unavailable. | P5; worklog/submission | Actual prompt history and contributions are preserved without fabrication or cleanup that changes events. | Not started |

Mandatory `results.csv` columns:

```text
request_id, scenario, source, start_time, first_token_ms, total_ms, status,
retries, tokens_in, tokens_out, retrieval_setting, answer_grade
```

The plan proposes additional telemetry columns (queue/work/retry timing, model and prompt IDs, token source, error class, etc.) and provider/scenario companion evidence so all requested figures can be audited. These additions preserve the assignment's mandatory columns.

## Report and submission requirements

| ID | Requirement | Phase / component | Verification criterion | Status |
|---|---|---|---|---|
| REP-01 | Main written report is no more than two pages. | P5; final report | Rendered/word-processed report is two pages or fewer. | Not started |
| REP-02 | Include prescribed headings 1–8 in order. | P5; final report | Exact heading order checked against brief. | Not started |
| REP-03 | Add “Retrieval: quality versus cost” and “How this library will mislead a naive retriever” after heading 5. | P5; final report | Heading location and substantive evidence checked. | Not started |
| REP-04 | Include “Your feedback on Vaani”: what stood out, confused, and one change. | P5; genuine review | Feedback follows an actual site visit and is dated in work notes. | Not started |
| REP-05 | Finish with “How I worked with the assistant”; three key prompts/why, caught error/how, overruled decision, distinctly user's work. | P5; worklog/report | Content is genuine and supported by preserved conversation/worklog. | Not started |
| REP-06 | Include the three result tables and all report-required findings. | P5; report/metrics | Tables match regenerated metrics and all required narrative topics are present. | Not started |
| REP-07 | Submit code, raw results, and conversation history as supporting material; make choices defensible. | P5; package/live explanation | Reproduction steps and genuine evidence present; student can explain trade-offs. | Not started |
| SEC-01 | Never include Groq key in submission. | P1/P5; secret handling | `.env` ignored; `.env.example` contains names only; secret scan/packaging check clean. | Not started |
| TIME-01 | Submit within four days of receiving brief. | P1; deadline record | Record actual receipt date and derived deadline; source PDFs' creation date is not treated as receipt. | Blocked on receipt date |
| TIME-02 | Target the stated 6–8 hours and focus on a small, deep implementation. | All phases; scope control | Track scope/time; optional work does not displace mandatory work. | Not started |
| POLICY-01 | Stay within provider terms; no extra accounts or other bypass. | P1/P4; experiment conduct | Run records and account context show compliant use. | Not started |
| POLICY-02 | No need to cover every technique; prioritize measured and defensible work. | All phases; scope | Architecture and report contain only justified components and claims. | Not started |

## Phase 1 task status

| Plan task | Current status | Evidence / remaining work |
|---|---|---|
| 1.1 Confirm deadline, scope, acceptance checklist | Partial | Checklist and scope recorded. Actual receipt date is still needed to calculate the deadline. |
| 1.2 Create Python project and virtual environment | Complete | Python 3.12.14 `.venv`, `pyproject.toml`, planned scaffold, and pinned `requirements.lock` created. |
| 1.3 Install and verify framework integrations | Complete | Dependency install succeeded. `scripts/smoke_imports.py` verified ChatGroq, prompts, LangGraph compile/invoke, and selected supporting modules. |
| 1.4 Configuration and secret handling | Complete | `.env.example`, `.gitignore`, env-backed `GroqSettings`, and missing-configuration errors implemented. The real key remains in the ignored local `.env` and is not included in output. |
| 1.5 Download exact paper PDFs | Complete | Exactly three required versioned PDFs stored under `data/papers/original/`; `scripts/download_papers.py` makes acquisition reproducible. |
| 1.6 Verify identity and create manifest | Complete | All PDFs parse; title/version URL, page count, size, and SHA-256 recorded in `data/papers/manifest.json`. |
| 1.7 Inspect structure and extraction quality | Complete for representative pages | Title/abstract, technical, figure/table, and final pages visually sampled and text-extracted; limitations recorded in `data/papers/processed/extraction_notes.md`. |
| 1.8 Verify Groq account/model options and limits | Complete with documented uncertainty | Authenticated model listing verified `openai/gpt-oss-120b` is available. Successful completion headers confirmed RPD/TPM. Official free-plan RPM/TPD values are separately labelled as documentation references; account plan/ownership and exact Limits page remain unverified. |
| 1.9 Modest real-service baseline | Complete | Three real streamed requests succeeded at five-second spacing. Provider token usage and rate-limit headers are in `experiments/raw/groq_probe.jsonl`. Three prior local client errors are retained and excluded from provider measurements in `groq_probe_runs.jsonl`. |
| 1.10 Document observed limits and uncertainty | Complete with uncertainty recorded | `experiments/raw/provider_limit_evidence.json` records documented vs header-observed values, successful calls, no observed 429, and recovery/concurrency as not measured. |

**Phase 1 technical gate: ready for review.** Corpus, environment, model access, modest baseline, and limit evidence are recorded. The assignment receipt date remains unknown, so Task 1.1's deadline calculation is incomplete. Phase 2 has not started; it requires the user's review and explicit approval.

## Requirement-to-phase summary

| Phase | Primary checklist groups |
|---|---|
| P1 | SYS-01, LLM-01–05, COR-02–04 (acquisition), SEC-01, TIME-01–02 |
| P2 | SYS-02–04, COR-01–07, retrieval and citations |
| P3 | SIM-01–05, reliability, timeouts, retries, telemetry, DATA-01–03 |
| P4 | EXP-01–16, DATA-06–07, real/stand-in scenario evidence |
| P5 | DATA-04–08, REP-01–07, SEC-01 and clean-room reproducibility |

## Source/plan alignment notes

- Brief allows Python, C++, or Go; plan selects Python and extends the requested implementation stack with LangChain/Groq/LangGraph, consistent with the user's implementation instructions.
- Brief asks for at least three load shapes and its results table names five required scenarios; the plan includes all five.
- Brief's CSV lists required columns but does not prohibit extra columns. The plan's additional fields are needed to reconstruct queue, offered rate, cost basis, and timing breakdowns. The exact required columns remain unchanged.
- No substantive conflict requiring a plan change was identified. Account-specific limits, current model availability, the receipt date, and genuine Vaani/conversation evidence remain unknown and must not be guessed.
