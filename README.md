# The Relay

The Relay is a citation-grounded research assistant that answers engineering questions from three fixed arXiv papers while measuring behavior under constrained model capacity. The authoritative requirements are in `The_Relay_Assignment_Brief.pdf`; the implementation roadmap is `IMPLEMENTATION_PLAN.md`.

## Current progress

Phases 1–4 are implemented and their technical gates passed. Phase 5 has regenerated the report tables, completed clean-room reproduction, and validated the submission; the PDF remains a draft until student-specific reflections are added. The project ingests the fixed three-paper corpus, compares three local Chroma retrieval settings, and runs one bounded LangGraph through Groq or a clearly labelled local stand-in. Phase 4 added 18 corpus questions and five stand-in load scenarios. The low-rate Groq sample covers all 15 answerable questions under three settings, with nine extra repeated comparisons disclosed in the raw evidence. No load was sent to Groq. The four-day deadline remains uncalculated because the assignment receipt date is unknown.

## Requirements and architecture

See [PROJECT_UNDERSTANDING.md](PROJECT_UNDERSTANDING.md), [REQUIREMENTS_CHECKLIST.md](REQUIREMENTS_CHECKLIST.md), and [ARCHITECTURE.md](ARCHITECTURE.md) for the assignment analysis and planned structure. The implementation uses LangChain components with a LangGraph workflow; Groq is the real model backend, and a separate local stand-in is reserved for load experiments.

## Paper corpus

The original PDFs are under `data/papers/original/`. `data/papers/manifest.json` records each required version, source, download date, page count, and SHA-256 digest. The vector index and processed extraction will be rebuildable from these sources.

## Phase 1 Groq baseline and Phase 2 smoke evidence

On 2026-11-10, three real `ChatGroq` streaming calls used `openai/gpt-oss-120b`, a fixed short prompt, and five-second spacing. All three succeeded. First model-token times were 697.1, 620.5, and 455.1 ms (median 620.5 ms; n=3); total times were 702.9, 624.4, and 460.7 ms (median 624.4 ms; n=3). Provider usage was 78 input and 35 output tokens per call. This is a small model probe, not an end-to-end assistant measurement. A preceding set of three attempts failed locally before completion requests were sent; they remain in the raw file and are marked invalid for provider latency analysis in `groq_probe_runs.jsonl`.

Baseline responses reported 1,000 requests/day and 8,000 tokens/minute in their rate-limit headers. A later real Phase 2 answer-revision call produced one TPM 429: the provider reported a limit of 8,000 TPM, 7,179 used, 1,914 requested, and an advised retry delay of 8.1975 seconds. A subsequent call succeeded after a longer cooldown; the exact recovery boundary was not measured. The current published free-plan table lists 30 requests/minute and 200,000 tokens/day for this model, but exact account values for those dimensions were not independently read from the account Limits page. Concurrency remains unmeasured. See `experiments/raw/provider_limit_evidence.json` for the dated evidence and caveats.

## Environment setup

Python 3.11+ is required. Phase 1 targets Python 3.12 and uses `pyproject.toml` for dependency management.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
python scripts/smoke_imports.py
```

`requirements.lock` pins the packages resolved and checked on Windows with CPython 3.12.14. Do not add credentials to the repository. Copy `.env.example` to `.env` for a local real run, then supply the model and key from your own Groq account; keep `.env` private and untracked. Real calls should be spaced to stay within observed TPM capacity; heavy-load scenarios are reserved for the local stand-in.

## Build and query the paper index

```powershell
python scripts/build_index.py
python scripts/smoke_retrieval.py
python scripts/ask.py "Why are dot products scaled in attention?" --setting medium
python -m pytest -q
```

The index uses local `sentence-transformers/all-MiniLM-L6-v2` at pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The first run downloads/loads the model if it is not cached. Chroma data is rebuildable and ignored by version control. Retrieval settings are `small_fast`, `medium`, and `medium_reranked`; the last reranks five vector candidates down to three with local BM25. The measured default is `medium_reranked`; explicit settings are available for evaluation and near-limit use.

The retrieval smoke records for three paper-specific questions across all three settings are saved in `experiments/raw/phase2_retrieval_smoke.jsonl`; the generator writes retrieval parameters, source pages, vector/reranker scores, and timing fields. Phase 4 evaluated all 18 questions across all three settings. Exact expected paper/version/page hit@3 was 9/15 (0.600) for `small_fast`, 11/15 (0.733) for `medium`, and 12/15 (0.800) for `medium_reranked`; the remaining three questions per setting are unanswerable controls. Reranking added a measured mean 0.65 ms local rerank time. Given the small single-corpus sample, `medium_reranked` is the measured retrieval default, not a general quality guarantee. Details are in `experiments/summaries/retrieval_evaluation.json` and `naive_retriever_failures.md`.

`ask.py` buffers the answer until the graph's deterministic citation checks and model verifier pass. For the controlled Phase 2 real smoke sample, use `python scripts/run_phase2_smoke.py`; this makes real Groq calls for supported questions and uses local retrieval/refusal for the out-of-library question. It consumes account quota. The exact prompts and request records from the completed sample are kept in `experiments/raw/phase2_groq_smoke.jsonl`.

## HTTP API and local stand-in

The application API is `POST /ask` with `{"question":"...", "retrieval_setting":"medium"}`; `GET /health` reports service readiness and queue capacity. Use the local stand-in for reliability development and experiments. In separate terminals:

```powershell
python -m uvicorn relay.standin.app:app --app-dir src --host 127.0.0.1 --port 8001
$env:RELAY_BACKEND = "standin"
python -m uvicorn relay.api:app --app-dir src --host 127.0.0.1 --port 8000
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType 'application/json' -Body '{"question":"What does the paper say about attention?","retrieval_setting":"medium"}'
```

Set `RELAY_BACKEND=groq` (the `.env.example` default) for a real run; it requires `GROQ_API_KEY` and `GROQ_MODEL` in the private local `.env`. A real request consumes Groq quota. The stand-in binds to loopback in this command, returns explicitly simulated JSON/SSE responses and synthetic usage/limit metadata, and uses a fixed local dummy credential in its LangChain client. Its reset endpoint is `POST http://127.0.0.1:8001/admin/reset` and is intentionally loopback-only. Simulator settings and measured-sample calibration are in `config/standin_profile.yaml`; the five planned load shapes and initial synthetic parameters are in `config/scenarios.yaml`.

Queue capacity, workers, end-to-end deadline, concurrent model calls, local RPM/TPM/RPD/TPD safety caps, and retry behavior are configured in `config/app.yaml`. Local daily budget counters persist at `experiments/raw/local_budget_state.json`; they do not include use outside this app. The model wrapper buffers its stream until the answer is validated, so first-model-token latency can precede first-user-visible-answer latency. Request outcomes append one row per request to `experiments/raw/results.csv` and a detailed record to `experiments/raw/request_traces.jsonl`. The mandatory CSV columns remain first and unchanged; additive columns record queue, work, retries, token provenance, and errors. `first_token_ms` means first user-visible response time; `first_model_token_ms` separately records model stream timing. Stand-in usage is labelled `simulated`, never provider usage.

## Application, experiments, tests, and results

Phase 4 used the same API and graph for five local stand-in scenarios. Reproduce the runs with `python scripts/run_load_test.py`; the existing raw runs include an invalidated harness attempt and corrected valid runs. Regenerate stand-in summaries with `python scripts/build_phase4_metrics.py`. The scenario named `steady` offered about 0.516 requests/sec but saturated the synthetic token budget (3/30 completed); the label's “below capacity” assumption was wrong, so interpret it as a constant-arrival saturation run. A separate retry diagnostic forced one recoverable 500: retry yielded success after 387 ms backoff and delayed a queued peer by an additional 1,781 ms. These are synthetic stand-in behaviors.

The low-rate Groq comparison covered all 15 answerable questions across three settings. The original nine calls sampled three questions; a supplemental 45-call run completed the full comparison at one-minute spacing. A resume-filter source-label mismatch repeated those three questions once per setting, so the raw set has 54 calls and nine duplicate comparisons. Forty-one calls returned manually gradable answers; 13 operational failures/limits remain visible and ungraded. Provider token usage is recorded by provenance, including one locally limited mixed-provenance row. This is an exploratory fixed-corpus evaluation, not a general model-quality estimate. Rebuild its summary with `python scripts/build_phase4_evaluation_summary.py`.

Regenerate the final result tables with `python scripts/build_final_tables.py` and validate `results.csv` plus its token/retry aggregates against request traces with `python scripts/validate_submission.py`. Run `python -m pytest -q` for tests. Real Groq evidence and simulated evidence remain labelled separately; never interpret simulated measurements as provider measurements.

For a fresh environment, install `requirements.lock`, install the package with `python -m pip install -e .`, run `python scripts/smoke_imports.py`, rebuild the index with `python scripts/build_index.py`, and repeat retrieval-only evaluation with `python scripts/run_eval.py --mode retrieval`. The five local-only load scenarios run with `python scripts/run_load_test.py`; they use the stand-in and do not spend Groq quota. Regenerate the answer summary, final tables, and validator with `python scripts/build_phase4_evaluation_summary.py`, `python scripts/build_final_tables.py`, and `python scripts/validate_submission.py`. The clean-room project copy completed these steps and passed all 42 tests. Build a two-page PDF draft with `python scripts/build_final_report.py --draft`; the final PDF is withheld until the student-authored reflection and Vaani feedback sections are supplied. `reportlab` is an optional reporting dependency included in `requirements.lock`.

## Current limitations and pending inputs

- Assignment receipt date is unknown, so the four-day deadline cannot yet be calculated.
- Account-specific model availability was confirmed; RPD and TPM were confirmed in response headers, and one TPM 429 was observed. RPM/TPD and concurrency remain unverified. The provider-suggested wait was observed, but exact recovery time was not.
- Official Groq rate-limit documentation was checked on 2026-11-10; see `experiments/raw/provider_limit_evidence.json`. It distinguishes documented limit behavior from account-specific values, which still require your Limits page.
- The Groq key is held only in the ignored local `.env`; it is not included in project outputs.
- PDF extraction notes are in `data/papers/processed/extraction_notes.md`; complex figure glyphs need page-level review during ingestion.
- `medium_reranked` is selected based on the Phase 4 top-three expected-page comparison. Answer grades cover 15 questions from this fixed corpus; the sample does not establish broad semantic answer superiority.
- The model verifier is a second model judgment, not a formal entailment proof. Citation identities and page bounds are deterministic; semantic support was manually graded on 41 outputs over the assignment's 15 answerable questions, with 13 operational failures excluded from grade denominators.
- Stand-in calibration has only three single-user latency samples. Its inter-chunk pacing, load slowdown, limits, failures, and deterministic answer behavior are assumptions or controls, not measurements of Groq internals.
- Groq token counts are recorded by provenance in `phase4_answer_metrics.json`; the one locally limited request is marked mixed rather than provider-only. No missing values are imputed. The local stand-in uses synthetic tokens and should only be used for synthetic load analysis.
