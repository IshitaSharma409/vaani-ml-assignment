# Assistant Worklog (in progress)

This is a contemporaneous technical worklog for the implementation and Phase 2 smoke checks. It is not a substitute for the user's complete conversation history or their own reflection, which must be preserved and completed during Phase 5.

## 2026-11-10 — Phase 1 and Phase 2

- The user authorized the next project phase by replying “next.” Phase 1 had set up the environment, acquired the exact PDFs, and recorded the real Groq baseline and provider evidence.
- Built page-aware PDF ingestion, page-bounded LangChain chunking, three local Chroma collections, explicit `search_research_library` tool, metadata-based citation rendering/validation, and bounded LangGraph retrieval/refinement/draft/verification/revision/refusal paths.
- The local embedding model is `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The optional reranker is a local BM25 ranker over five bounded vector candidates, selected under the plan's allowed “another explicit and reproducible local re-ranking method” option.
- All three retrieval settings were run against one question per paper. The results, exact setting parameters, source pages, vector scores, BM25 scores, and retrieval timings are saved in `experiments/raw/phase2_retrieval_smoke.jsonl`; this is a smoke test, not a comparative quality result.

### Smoke questions actually sent through the harness

The first three are supported-paper questions; the fourth tests refusal. Outputs and request traces are preserved verbatim in `experiments/raw/phase2_groq_smoke.jsonl`.

1. “Why are dot products scaled by the square root of key dimension d_k in attention?” — real Groq answer passed citation validation and verification, cited `1706.03762v7`, PDF page 4. Request `3f35b57a-545d-4945-be98-c5c9d6ee7a0c`.
2. “What is Mimi and how does it tokenize audio using semantic and acoustic information?” — real Groq answer passed citation validation and verification, cited `2410.00037v1`, PDF pages 9, 10, and 24. Request `d31db086-4a33-4f0a-9714-9cf47c5e86dd`.
3. “How does H-Net learn content-aware chunk boundaries through end-to-end training?” — real Groq answer passed citation validation and verification, cited `2507.07955v2`, PDF pages 6 and 14. Request `d4c45859-1d7b-4264-90df-02163ee752f3`.
4. “What is the boiling point of water at sea level in Celsius?” — the fixed library did not support an answer; the workflow refined retrieval once and refused without invoking the answer model. The final successful refusal is request `5a31a167-869c-4e06-b104-9d8742f462bc`.

### Errors found and corrections made

- Initial real-model attempts in the restricted process environment failed with local connection errors; they are recorded as errors with unavailable token counts. The authorized outbound test path then succeeded. These local failures are not presented as provider latency or provider outages.
- One genuine TPM 429 occurred when the workflow attempted its single answer revision after the model rejected an initial draft. Groq reported `8000 TPM`, `7179 used`, `1914 requested`, and advised waiting `8.1975` seconds. A later run passed after a longer cooldown. This is a real provider limit event, not a simulated result. The error row is classified as an error rather than a correct refusal; its originally returned text is retained in the raw record.
- That event exposed a workflow issue: verifier/model-service failures could enter the content-revision branch. The graph now terminates on a model-service error, returns an operational error, releases no draft, and makes no extra revision call. A dedicated test covers this behavior. Token metadata is left unavailable when the provider returns no usage values.
- The first two successful answers used `relay-rag-v1`; the TPM event motivated a shorter `relay-rag-v2` prompt that caps answers at two concise claims and shortens verifier feedback. The version and SHA-256 of each prompt template are attached to each raw request record.
- Citation spot checks against the original PDFs confirmed the cited page locations for the three successful supported samples. The deterministic validator now also rejects a wrong version for the fixed corpus, a missing page, an out-of-range page, and an unknown chunk.
- The first index attempt encountered a OneDrive directory reparse point at `data/vectorstore/`; the index now writes to `data/vectorstore/relay_chroma/` beneath the planned directory and rebuilds without removing the parent.

### Work that belongs to the user / remains for later

- The user has not supplied the date they received the brief; the deadline cannot be calculated from PDF metadata.
- The user should review and understand the architecture and evidence before proceeding. Do not invent personal feedback on Vaani, learning reflections, an overruled decision, or which work was distinctly theirs. Add the user's genuine contribution and conversation context during the final report phase.
- The above three Groq answers are a correctness smoke sample, not a quality benchmark or retrieval-setting comparison. Heavy traffic, reliability/load scenarios, the stand-in, and the final evaluation/report have not been run.

## Phase 3 — local reliability and stand-in implementation (2026-11-10)

The user authorized proceeding to the next implementation phase. I implemented the bounded request queue, configured model-call semaphore, persistent local budgets, deadline/abandonment signals, explicit retry wrapper, request telemetry and results writer, FastAPI endpoints, and the OpenAI-compatible local stand-in. The same existing LangGraph now runs through a local `ChatOpenAI` adapter. The simulator's deterministic response text is explicitly labelled and does not assert paper facts.

The stand-in latency profile uses the three recorded real Groq first-token/total-time samples. The sample is too small to estimate a reliable distribution. Inter-chunk pacing, load slowdown, synthetic quota values and failure modes are assumptions/controls and are documented as such. The five Phase 4 scenario configurations are recorded with `configured_not_run`; no experimental result is claimed.

Local verification found and corrected a budget-test expectation that exceeded its configured RPM, a forced-failure reset expectation, a partial-stream control bug, and a test query that did not pass the graph's lexical evidence gate. Verification then passed: 35 pytest tests; the service/framework import smoke; and `pip check` (no broken requirements). The integration tests use asyncio and loopback only; they did not call Groq. The user still needs to provide the assignment receipt date for the deadline calculation; final evaluation and reporting remain for later phases.

## Phase 4 — evaluation and load evidence (2026-11-10)

The user authorized the next phase with the actual prompt “next.” I ran 18 paper-grounded questions (15 answerable and three unanswerable controls) under three retrieval settings and preserved retrieved chunk IDs, versions, pages, scores, and timings in `experiments/raw/retrieval_evaluation.jsonl`. Exact expected page/version hit@3 was 9/15 for `small_fast`, 11/15 for `medium`, and 12/15 for `medium_reranked`; on that local retrieval metric I selected `medium_reranked` as the default while retaining explicit setting selection for near-limit requests.

The low-rate real Groq answer sample used three paper-balanced questions across all settings, sequentially spaced by 45 seconds. Four of nine outputs completed and were compared directly with PDF evidence; the remaining five were operational failures or local safety limits and are not answer grades. Provider token counts are retained for eight request rows and one locally limited row has mixed provenance. This is a small exploratory sample, not a broad quality claim.

All five fixed stand-in load scenarios used the same API and graph. Initial invalid harness attempts remain in `load_runs.jsonl` with reasons; corrected runs are marked valid. The scenario description “below configured capacity” for `steady` was an assistant-generated assumption: the preserved run showed only 3 of 30 successes because synthetic TPM saturated. I left the fixed pre-run configuration intact and documented the result as a constant-arrival saturation case. In a separate controlled retry test, a forced 500 recovered after 387 ms backoff; the queued peer waited 1,781 ms. The peer's 3,187 ms total reconciles to 1,781 ms queue wait + 1,406 ms work, with no retry on the peer; its first model token arrived at 2,531 ms, so its long wait preceded that token. All are stand-in results.

During Phase 5 preparation, I first described Groq answer token usage as unavailable. Checking `results.csv` and `tokens_source` showed provider counts on eight rows and mixed provenance on one; I corrected the summary and README. I also found blank CSV cells broke Pydantic validation in the grade propagation script; I normalized nullable cells before validation. These corrections were made by the assistant. The available conversation does not record the student personally catching an assistant error or overruling an implementation decision; the report must not claim either without the student's confirmation.

## Phase 5 — report and reproducibility work (2026-11-10)

The actual prompt “next” authorized this phase. The assignment brief requires four tables, the exact section sequence, a two-page maximum, a visit to Vaani, and personal reflection. I visited the public Vaani landing page on 2026-11-10 and recorded only what the page displayed in `vaani_visit_notes.md`; the student's personal reactions remain pending. The user has also been asked for the brief receipt date, one learning, and one implementation decision they changed or overruled.

Phase 5 tooling in progress: `scripts/build_final_tables.py` generates the four tables from raw rows/config/provider evidence; `scripts/validate_submission.py` verifies mandatory CSV header order, unique IDs, allowed values, trace coverage, and token/retry aggregates. The first validation passed for 186 unique request rows. The complete test suite passed 38 tests after Phase 4. Final clean-room environment reproduction and PDF visual inspection are still required.

### Actual prompts preserved from the conversation

1. **“MASTER PROMPT: IMPLEMENT THE INTERNSHIP PROJECT”** — supplied the assignment hierarchy, exact five-phase process, required stack, no-fabrication rules, and first-turn analysis-only gate. It shaped both the technical scope and the user approval gates.
2. **“should i write api key under "" pr not”** — asked about quoting the secret in `.env`; the assistant clarified the exact syntax and reiterated not to paste a key into chat.
3. **“next”** — repeated at each phase boundary. Each reply explicitly authorized proceeding to the next phase; the assistant stopped at phase gates until that approval.

The actual Codex thread remains available for the full chronology; this worklog selects relevant prompt and verification excerpts rather than reproducing raw tool output or secrets. Student-specific claims about an assistant error they personally caught, an implementation choice they overruled, and what work is distinctly theirs remain unconfirmed and must be completed in the student's own words.

### Phase 5 evidence update — 2026-11-10

The supplemental Groq run covered all 15 answerable question IDs in all three retrieval settings. The original three questions were repeated because the resume filter compared the command value `groq` with stored source label `real`; this added nine calls. All 45 supplemental requests remained one minute apart. The complete sample has 54 requests, 41 successful outputs manually graded against the versioned source PDF pages, and 13 operational failures/limits left ungraded. The resume filter now maps CLI backends to persisted source labels, with a regression test.

The summary and final tables were regenerated from raw evidence. The workspace CSV validator checked 231 unique end-user requests and reconciled trace/token/retry aggregates. A copied project in a fresh virtual environment passed selected framework imports, index rebuild, retrieval evaluation (9/15, 11/15, 12/15 exact page hits at three), all five stand-in load scenarios, final metrics/table regeneration, CSV validation (362 source-plus-replay unique requests), 42 tests, and `pip check`. Most replayed scenario outcomes match the retained runs; sustained overload produced 3/60 successes on replay versus 2/60 originally, which shows the timing sensitivity of the synthetic capacity model. No key was copied and no Groq request was made in the clean-room copy.

The complete workspace test suite passed 42 tests and `pip check` reported no broken requirements. The PDF draft generator rendered two pages; page images were visually reviewed after correcting a table-header contrast issue and refreshing the final metrics. The final PDF must wait for the student's own Vaani reaction and reflection. Receipt date, one personal learning, one decision the student overruled, and any assistant error the student personally caught are still not established in this conversation; no personal account has been invented. The resume-filter mismatch and its correction are assistant-owned and should not be attributed to the student.
