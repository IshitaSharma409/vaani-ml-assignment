# Requirements Checklist — The Relay

This checklist is derived from the complete four-page assignment brief and cross-referenced to `IMPLEMENTATION_PLAN.md`. Status reflects verified work through Phase 5's report and clean-room preparation; real answer sampling and final student-supplied report fields are still pending. `Not started` means no implementation or verification has been claimed. Phase references use P1–P5 for the five plan phases.

## System and corpus requirements

| ID | Requirement | Phase / component | Verification criterion | Status |
|---|---|---|---|---|
| SYS-01 | Build a small working pipeline in Python, C++, or Go; selected plan language is Python. | P1–P2; project setup and CLI | Clean environment can run the CLI and import its selected integrations. | Complete for the current CLI workflow; `smoke_imports.py`, index build, and real request path verified |
| SYS-02 | Pipeline has at least four steps. | P2; LangGraph | Graph inspection/test demonstrates four or more explicit processing steps. | Complete: explicit prepare, retrieve, evidence, draft, verify, finalize/refuse nodes; tested |
| SYS-03 | Pipeline makes at least one tool call. | P2; retrieval tool | Trace/test records an actual `search_research_library` invocation. | Complete: actual local retrieval tool invoked and traced |
| SYS-04 | Pipeline has at least one branch or loop. | P2; workflow | Graph tests cover conditional path and prove loop/retry bounds. | Complete: one retrieval refinement and one revision maximum; all major paths tested |
| COR-01 | Answer only from the fixed library; unsupported information is stated as unavailable rather than guessed. | P2; prompts, evidence gate, refusal | Supported and unsupported paper-grounded questions are checked; no unsupported answer is released. | Complete for smoke sample: three supported questions passed verification; an out-of-library question was refused |
| COR-02 | Retrieve exact Attention Is All You Need, arXiv 1706.03762 v7. | P1–P2; corpus/manifest | PDF identity and version are verified and recorded. | Complete: ingested into page-preserving chunks and locally indexed |
| COR-03 | Retrieve exact Moshi, arXiv 2410.00037 v1. | P1–P2; corpus/manifest | PDF identity and version are verified and recorded. | Complete: ingested into page-preserving chunks and locally indexed |
| COR-04 | Retrieve exact Dynamic Chunking, arXiv 2507.07955 v2. | P1–P2; corpus/manifest | PDF identity and version are verified and recorded. | Complete: ingested into page-preserving chunks and locally indexed |
| COR-05 | Every answer cites its source location. | P2; retrieval metadata/citation validator | Each released substantive claim has a citation traceable to a retrieved chunk and valid paper/page. | Complete for smoke sample: citations generated from trusted metadata and manually spot-checked |
| COR-06 | Never invent pages, sections, sources, or citations. | P2; deterministic validator | Tests reject unknown source IDs, wrong versions, missing/out-of-range pages. | Complete: validator rejects unknown chunks, empty citations, and invalid pages; tests pass |
| COR-07 | Do not release an unverified draft; preserve answer quality under pressure. | P2–P3; validation gate | Integration test shows output is released only after validation, or a safe refusal/fallback is returned. | Complete for P2: validated drafts release; provider verification errors terminate without release |

## Provider and stand-in requirements

| ID | Requirement | Phase / component | Verification criterion | Status |
|---|---|---|---|---|
| LLM-01 | Use a real hosted model on the user's own Groq free tier. | P1–P2; Groq backend | Authenticated `ChatGroq` call succeeds; run is identified as real. | Complete for P2: real `ChatGroq` end-to-end smoke passed for one question per paper |
| LLM-02 | Use own account/key, do not use extra accounts or bypass terms/limits. | P1/P4; credentials and experiment policy | Record configuration provenance without exposing key; no alternate account or limit circumvention. | Local key used for the authorized probe and kept out of outputs; account plan/ownership not independently inspected |
| LLM-03 | Discover real service limits from documentation and provider responses/account evidence. | P1/P5; provider evidence | Table 1 gives value, method, confidence; unknowns are explicitly “not measured”/unspecified. | RPD=1,000 and TPM=8,000 from successful response headers; one TPM 429 reported 7,179 used + 1,914 requested and advised 8.1975s wait; RPM/TPD are documentation references only |
| LLM-04 | Send only a modest number of real baseline requests first. | P1; probe script | Raw records show controlled request count, prompt, timestamps, model, usage, and limitations. | Complete: three short real calls, five-second spacing, with raw timing, usage, and headers preserved |
| LLM-05 | Keep real load within safe free-tier capacity; heavy load uses stand-in. | P1/P4; scenario runner | Scenario metadata identifies backend; no uncontrolled/burst/overload real calls. | Complete: no heavy load was sent to Groq; all five load scenarios used the stand-in. |
| SIM-01 | Implement a controllable stand-in model service, separate and clearly labelled from Groq. | P3; stand-in app/backend | Stand-in can be selected while retaining same application workflow; outputs/results identify simulation. | Complete: same LangGraph runs through local ChatOpenAI-compatible service; tests verify simulation labels and local source. |
| SIM-02 | Stand-in can be slow and stream its reply piece by piece. | P3; simulator | Configured delays and streamed chunks are observed in local integration checks. | Complete: chunked SSE, configurable pacing, slow-profile timeout, and cleanup are integration-tested. |
| SIM-03 | Stand-in can fail and reject on configured limits. | P3; simulator/limits | Controlled failures and synthetic limit responses are reproducible from profile. | Complete: synthetic RPM/TPM/RPD/TPD/concurrency plus forced 500, connection-style, timeout, and partial-stream paths implemented; reset tested. |
| SIM-04 | Stand-in becomes slower under load. | P3; load model | Concurrent-load scenario produces configured load-dependent delay; formula/profile is recorded. | Complete: deterministic additive delay per active request beyond the first is tested. |
| SIM-05 | Calibrate stand-in from real observations and explain where it is least like Groq. | P3/P5; profile/report | Profile links real sample IDs and labels assumptions; report describes material limitations. | Complete for P3: profile references three genuine single-user latency samples and marks pacing, failure, quota, and load behavior as assumptions; sample size remains small. |

## Experiments, evaluation, and measurement

| ID | Requirement | Phase / component | Verification criterion | Status |
|---|---|---|---|---|
| EXP-01 | Run at least three load shapes, including steady, sudden burst, and sustained overload. | P4; YAML scenarios/runner | Raw request records exist for each shape; heavy traffic is simulated. | Complete; valid runs retained; `steady` unexpectedly saturated synthetic TPM and is reported as a constant-arrival saturation run. |
| EXP-02 | Cover the five required table scenarios: steady, burst, sustained overload, per-minute limit halved, daily budget exhausted. | P4; scenario configuration | All five named scenarios have fixed config and raw result rows. | Complete; valid raw runs are in `load_runs.jsonl`, `results.csv`, and `request_traces.jsonl`. |
| EXP-03 | Preserve raw results from both real and stand-in runs. | P1/P4; `results.csv`/metadata | Raw file identifies `source` for every request and retains failure outcomes. | Complete; real nine-request sample and stand-in runs are separately labelled; invalid harness runs preserved. |
| EXP-04 | Evaluate at least 15 questions written for this corpus. | P4; evaluation dataset | At least 15 original questions have evidence-backed reference notes and expected answerability. | Complete; 18 questions (15 answerable, 3 unanswerable controls). |
| EXP-05 | Report answer quality on those questions. | P4–P5; grading | Each usable output is manually graded using the four allowed grades with reason/evidence; failures are separately counted. | Complete for the 15 answerable questions: 41 usable real outputs graded with source-page evidence across three settings; 13 operational errors/limits remain ungraded and reported separately. |
| EXP-06 | Compare at least three retrieval settings: piece size, pieces fetched, and reranking. | P2/P4; retrieval config/evaluation | Three+ settings record exact chunk size, overlap, top-k, reranking, and comparable results. | Complete; all 18 corpus questions x 3 settings have retrieval evidence; all 15 answerable questions x 3 settings have graded real outputs where successful. |
| EXP-07 | Report first-token and total-time percentiles, including median and slow end, sample sizes, by scenario. | P3–P5; telemetry/report | Tables state p50/p95/p99 and sample counts/populations, computed from raw data. | Phase 4 raw-derived stand-in metrics and real sample summary complete; small n explicitly retained. |
| EXP-08 | Report failures and requests given up/timeouts. | P3–P5; outcome fields/metrics | Outcome percentages are derived from all offered requests with categories retained. | Complete in scenario metrics and raw records; invalidated harness runs are preserved and excluded. |
| EXP-09 | Report tokens used per answer and retries. | P3–P5; usage/retry telemetry | Aggregate all model calls per user request, with provider/estimated token provenance. | Stand-in synthetic tokens and retries measured. Groq token counts are retained with provenance (provider for eight rows, mixed for the locally limited row). |
| EXP-10 | Report one genuinely slow request split into waiting, working, retrying and locate delay relative to first token. | P3/P4; event trace | Timestamp-backed decomposition reconciles to total within documented instrumentation overhead. | Complete for controlled stand-in diagnostic: retried request and queued peer are trace-backed; synthetic only. |
| EXP-11 | Include a measured number not inferable without execution. | P4–P5; result narrative | Cite a genuine run-derived figure and its sample/raw records. | Complete; medium_reranked exact page hit@3 is 12/15 and retry-peer added wait is 1,781 ms, both raw-derived. |
| EXP-12 | Describe retry policy and one case where retrying made things worse. | P3/P4/P5; retry policy/scenario | Raw trace supports a real case or a clearly labelled controlled stand-in case. | Complete; forced 500 recovered after 387 ms backoff; queued peer wait increased by 1,781 ms in controlled stand-in. |
| EXP-13 | Explain behavior at minute and daily limits: delayed, dropped/rejected, weaker answers, and rationale. | P3/P4/P5; limiter/report | Tests and raw records demonstrate terminal outcomes and no extra call after daily exhaustion. | Complete for local stand-in; minute offers are rate-limited, daily probe rejected with no retry/model call. Real limits are scoped to prior observations. |
| EXP-14 | Describe exact user experience at 3 PM after daily budget is gone. | P4/P5; budget-exhaustion path | Implemented response is quoted accurately; no reset time is claimed unless verified. | Complete; app returns an unavailable verified answer with HTTP 429 and makes no reset-time promise. |
| EXP-15 | Identify three actual ways this library misleads a naive retriever, each with example question. | P4/P5; retrieval analysis | Each failure is evidenced by actual paper text/retrieval run, not a hypothetical candidate. | Complete; three actual top-three unanswerable-control cases documented in `experiments/summaries/naive_retriever_failures.md`. |
| EXP-16 | Explain quality-versus-cost choice and near-limit retrieval behavior. | P4/P5; selected retrieval policy | Choice follows measured quality, tokens, and latency; citation checks remain enabled. | Provisional measured choice: `medium_reranked` leads exact page-hit@3; near-limit can select `small_fast` while retaining verification/citations. Semantic quality sample remains small. |

## Required raw file and tables

| ID | Requirement | Phase / component | Verification criterion | Status |
|---|---|---|---|---|
| DATA-01 | Submit one `results.csv`, one row per request, with mandatory columns unchanged. | P3/P5; CSV writer/schema validation | Header includes all 12 exact names below; unique request IDs; rows are end-user requests. | P3 implementation complete: exact columns lead the schema, additive audit fields follow, duplicate IDs are rejected, accepted/rejected request rows are tested. Actual experiment rows await P4. |
| DATA-02 | Preserve allowed `status` values: `ok`, `rate_limited`, `timeout`, `error`. | P3/P5; schema validator | Every status validates; richer outcomes use additional fields. | Complete in P3 schema/writer tests; detailed user outcomes retain timeout, abandonment, refusal, and rejection separately. |
| DATA-03 | Preserve allowed answer grades: `correct`, `wrong`, `refused_correctly`, `refused_wrongly`. | P4/P5; grader/schema | Every populated grade is an allowed value with grading basis. | Four usable outputs graded with reason/evidence; failures remain null; allowed values and grade coverage validated. |
| DATA-04 | Make every table figure recomputable from raw data/evidence; use milliseconds and one decimal percentages. | P3–P5; telemetry, metric script | Deterministic metrics script reproduces tables; unspecified measures explicitly say “not measured.” | Phase 4 scripts regenerate raw-derived load and answer summary JSON; report formatting remains P5. |
| DATA-05 | Fill Table 1 with real limits, discovery method, confidence, in-flight requests, limit response, recovery time. | P1/P5; provider evidence | Account/documented/observed values distinguished; no invented limit or recovery time. | Not started |
| DATA-06 | Fill Tables 2a and 2b with the same required scenarios and all required metrics. | P4/P5; metrics generation | Rows match scenario set; all figures script-derived from raw data/config/evidence. | Phase 4 raw-derived metrics complete; final assignment tables remain P5. |
| DATA-07 | Fill Table 3 for at least three settings with grades, tokens, first-token and total time. | P4/P5; retrieval evaluation | Exact settings and sample sizes shown; grades follow rubric. | Phase 4 comparison and summary complete; failed calls and token provenance disclosed; final table remains P5. |
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
| REP-01 | Main written report is no more than two pages. | P5; final report | Rendered/word-processed report is two pages or fewer. | Draft PDF renders to two pages and has been visually reviewed; final PDF awaits student-specific sections |
| REP-02 | Include prescribed headings 1–8 in order. | P5; final report | Exact heading order checked against brief. | Draft includes prescribed sequence; generator verifies headings |
| REP-03 | Add “Retrieval: quality versus cost” and “How this library will mislead a naive retriever” after heading 5. | P5; final report | Heading location and substantive evidence checked. | Draft includes measured retrieval evidence and three observed failures |
| REP-04 | Include “Your feedback on Vaani”: what stood out, confused, and one change. | P5; genuine review | Feedback follows an actual site visit and is dated in work notes. | Assistant recorded public-page observations; student's personal response pending |
| REP-05 | Finish with “How I worked with the assistant”; three key prompts/why, caught error/how, overruled decision, distinctly user's work. | P5; worklog/report | Content is genuine and supported by preserved conversation/worklog. | Prompt excerpts/worklog recorded; user-specific caught error, overrule, and contribution pending |
| REP-06 | Include the three result tables and all report-required findings. | P5; report/metrics | Tables match regenerated metrics and all required narrative topics are present. | Four required tables are regenerated from raw evidence, including 15 questions/settings, all grades, and separately reported failures |
| REP-07 | Submit code, raw results, and conversation history as supporting material; make choices defensible. | P5; package/live explanation | Reproduction steps and genuine evidence present; student can explain trade-offs. | Code, raw records, plan, report, and reproducibility steps are present; student still needs to review and own the explanation |
| SEC-01 | Never include Groq key in submission. | P1/P5; secret handling | `.env` ignored; `.env.example` contains names only; secret scan/packaging check clean. | `.env` is ignored; a scan of 105 project text files found no Groq key-pattern hits. The private `.env` was excluded from the scan and clean-room copy. |
| TIME-01 | Submit within four days of receiving brief. | P1; deadline record | Record actual receipt date and derived deadline; source PDFs' creation date is not treated as receipt. | Blocked on receipt date |
| TIME-02 | Target the stated 6–8 hours and focus on a small, deep implementation. | All phases; scope control | Track scope/time; optional work does not displace mandatory work. | Elapsed effort was not reliably tracked; no claim of meeting the target duration is made. |
| POLICY-01 | Stay within provider terms; no extra accounts or other bypass. | P1/P4; experiment conduct | Run records and account context show compliant use. | One user's configured Groq key was used for short calls spaced at least 45 seconds in the original sample and one minute in the supplement; all heavy load was stand-in only. No account bypass was used. |
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
| 1.8 Verify Groq account/model options and limits | Complete with documented uncertainty | Authenticated model listing verified `openai/gpt-oss-120b` is available. Successful completion headers confirmed RPD/TPM; a later real TPM 429 confirmed the 8,000 TPM limit. RPM/TPD remain documentation references; account plan/ownership and exact Limits page remain unverified. |
| 1.9 Modest real-service baseline | Complete | Three real streamed requests succeeded at five-second spacing. Provider token usage and rate-limit headers are in `experiments/raw/groq_probe.jsonl`. Three prior local client errors are retained and excluded from provider measurements in `groq_probe_runs.jsonl`. |
| 1.10 Document observed limits and uncertainty | Complete with uncertainty recorded | `experiments/raw/provider_limit_evidence.json` records documentation vs account/header evidence and the TPM 429; exact recovery boundary and concurrency remain unmeasured. |

**Phase 1 gate: complete for implementation.** Corpus, environment, model access, modest baseline, and limit evidence are recorded. The assignment receipt date remains unknown, so Task 1.1's deadline calculation is incomplete. The user authorized Phase 2 with “next.”

## Phase 2 task status

| Plan task | Current status | Evidence |
|---|---|---|
| 2.1 Page-aware PDF ingestion | Complete | One page document per source PDF page, carrying paper ID/title/version/page/source/timestamp. |
| 2.2 Configurable page-bounded chunking | Complete | LangChain recursive splitter, stable IDs, three configs; chunk counts and lengths in `data/vectorstore/relay_chroma/index_manifest.json`. |
| 2.3 Local embedding and Chroma | Complete | Pinned local MiniLM revision; three collections built deterministically and rebuildable with `scripts/build_index.py`. |
| 2.4 Three retrieval settings | Complete | `small_fast`, `medium`, `medium_reranked` are selectable; smoke records expose setting and score metadata. |
| 2.5 Optional reranking | Complete | Local BM25 reranks five candidates to three; score and compute time preserved; unit test verifies ranking. |
| 2.6 Explicit LangChain retrieval tool | Complete | `search_research_library` validates inputs and returns typed source metadata; invocation count/time in workflow state. |
| 2.7 Citation validation | Complete | Formatter uses only retrieved metadata; tests reject unknown IDs and invalid pages. |
| 2.8 Answer/refusal prompts | Complete | Versioned source-controlled prompts require claim citations and abstention. |
| 2.9 Bounded LangGraph workflow | Complete | Retrieval refinement and answer revision each max once; tests cover normal, refusal, revision, and provider failure paths. |
| 2.10 Real-Groq correctness smoke | Complete | Three supported questions passed generation, citation validation, and verifier; out-of-library question refused; records in `phase2_groq_smoke.jsonl`. |

**Phase 2 gate: passed for the small manual sample.** This confirms workflow functionality, not broad answer quality or retrieval-setting performance.

## Phase 3 task status

| Plan task | Current status | Evidence / limitation |
|---|---|---|
| 3.1 Bounded FIFO admission | Complete | Queue order, queue-full rejection, queue metrics, and one-row-per-accepted/rejected request tested in `test_admission_api.py`. |
| 3.2 Concurrency and request/token budgets | Complete | Model semaphore, persistent daily/minute local budget accounting and estimated-token reservation implemented; concurrency and each budget dimension have unit coverage. Local counters cannot see use outside this app. |
| 3.3 Deadline and abandonment | Complete | End-to-end deadline, cancellation signal, API disconnect polling, timeout and abandonment trace distinctions implemented; timeout and abandonment integration paths pass. An in-flight synchronous SDK call remains bounded by its configured 30-second call timeout. |
| 3.4 Bounded retries and failure classification | Complete | Network/5xx and minute-window rate limits are retryable under bounded backoff; daily exhaustion, auth, invalid requests, and terminal timeouts are not. Retry-After is preserved. Unit tests cover retry success and terminal daily errors. |
| 3.5 Structured telemetry | Complete | JSON operational logs omit prompts; monotonic/UTC timings, queue/retry/retrieval/verification, model and user response first-token times, token provenance and per-call JSONL are implemented. |
| 3.6 Required result schema | Complete for implementation | Pydantic limits statuses/grades; exact mandatory CSV header precedes additive audit columns; duplicate request rows are rejected. Actual experiment records are not populated until P4. |
| 3.7 Local stand-in API | Complete | FastAPI chat-completions JSON and SSE interface with usage metadata; same LangGraph passes through LangChain's local backend. |
| 3.8 Stand-in capacity and failure behavior | Complete | Synthetic RPM/TPM/RPD/TPD/concurrency, reset, forced errors, partial streams, pacing, and load slowdown are implemented and locally tested. |
| 3.9 Calibration | Complete with stated limits | Profile traces three real Phase 1 latency samples and IDs; pacing, concurrency curve, quotas, and failures remain explicitly synthetic assumptions. |
| 3.10 Reliability integration tests | Complete | Queue, timeout/abandonment, API, retry, budget caps, local SSE, forced failures, reset, client timeout, concurrency, and same-graph stand-in tests pass. No Groq load was sent. |

**Phase 3 gate: passed for implementation and local verification.** The test suite has 35 passing tests (including 11 local API/stand-in integration tests). Phase 4 subsequently ran all five fixed scenarios and recorded their raw evidence. Real-provider behavior is represented only by the already-recorded modest Phase 1/2 observations.

## Phase 4 task status

| Plan task | Current status | Evidence / limitation |
|---|---|---|
| 4.1–4.2 Questions and grading rubric | Complete | 18 questions, four-grade rubric, and 41 usable real outputs manually checked against PDF evidence; 13 operational failures are not answer grades. |
| 4.3 Retrieval comparison | Complete | All 18 questions searched under all three settings; raw chunks/scores and derived exact page/version recall are preserved. Real answer sample spans all 15 answerable questions x 3 settings. |
| 4.4 Naive retrieval failures | Complete | Three corpus-scope false-positive examples from actual top-three results documented in `experiments/summaries/naive_retriever_failures.md`. |
| 4.5 Operating point and near-limit policy | Complete provisionally | `medium_reranked` selected for highest measured exact page-hit@3 (12/15); `small_fast` remains selectable near limit; verifier/citation validation remain mandatory. Answer quality sample is too small for a broad semantic-quality claim. |
| 4.6–4.8 Scenarios and runs | Complete | Five fixed YAML stand-in scenarios and valid runs; invalid harness failures preserved in `load_runs.jsonl`. A separate retry diagnostic is also preserved. One scenario's “below capacity” assumption was disproven by its measured TPM saturation. |
| 4.7 Low-rate real pipeline sample | Complete with limitation | 54 Groq requests across 15 unique questions x three settings, with nine repeated comparisons from a resume-filter source-label mismatch. Forty-one usable outputs and 13 operational failures/limits; calls were spaced one minute apart and no real load test was run. |
| 4.9 Percentiles and queue analysis | Complete for measured samples | `phase4_load_metrics.json` states populations and NumPy linear percentiles. Controlled diagnostic: one request completed in 1,797 ms including 387 ms retry wait; its queued peer waited 1,781 ms. |
| 4.10 Limits and retries | Complete for tested local policy | Synthetic minute/daily limits and no-reset-promise path recorded; a forced 500 improved after retry while delaying its peer. At daily exhaustion the exact answer is “The assistant could not complete a verified answer. Please try again later.” API returns HTTP 429; no reset time is promised. Provider daily reset timing remains unverified. |

**Phase 4 gate: passed with scoped limitations.** Five stand-in scenarios and all retrieval comparisons have raw-derived summaries. Phase 5 extended the real answer sample to all 15 answerable questions at a controlled one-request-per-minute pace. Nine repeated comparisons are retained and disclosed. Stand-in figures describe the simulator only.

## Phase 5 task status

| Plan task | Current status | Evidence / limitation |
|---|---|---|
| 5.1–5.2 Final tables and report | In progress | Four raw-derived tables and a visually reviewed two-page draft PDF are generated. The final PDF requires authentic student reflection and Vaani feedback. |
| 5.3 Submission validation | Complete for current evidence | Validator checked 231 unique end-user rows and reconciled token/retry totals to traces. The refreshed clean-room copy validated 362 unique rows after replaying all five stand-in scenarios. |
| 5.4 Clean-room reproduction | Complete | Fresh environment passed framework imports and `pip check`; copied project rebuilt the index, reproduced retrieval metrics and all five stand-in load scenarios, regenerated reports, and passed 42 tests. No Groq key was copied or used. Sustained overload completed 3/60 on replay versus 2/60 in the retained run, reflecting timing-sensitive synthetic capacity. |
| 5.5 Final documentation and handoff | In progress | README/checklist/worklog reflect actual evidence and remaining user-supplied fields. The actual receipt date and personal responses are not inferable. |

**Phase 5 gate: technical work verified; final handoff pending student input.** The answer sample is graded, clean-room reproduction and validation passed, and the two-page draft was visually checked. The report cannot be final until the student supplies genuine Vaani feedback and personal reflection; the brief receipt date is also needed to calculate its four-day deadline.

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
