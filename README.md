# The Relay

The Relay is a planned citation-grounded research assistant that answers engineering questions from three fixed arXiv papers while measuring behavior under constrained model capacity. The authoritative requirements are in `The_Relay_Assignment_Brief.pdf`; the implementation roadmap is `IMPLEMENTATION_PLAN.md`.

## Current progress

Phase 1 technical setup is complete and ready for review. The exact versioned PDF corpus is downloaded, parsed, visually sampled, and recorded in the manifest. The Python 3.12.14 environment and pinned dependency lock installed successfully; LangChain, LangGraph, Groq integration, and supporting imports passed the smoke check. Groq model access is verified and three short real streamed baseline calls succeeded. Their request-level records and provider headers are in `experiments/raw/groq_probe.jsonl`; `provider_limit_evidence.json` distinguishes account headers from published free-plan references. No full-pipeline or load-test performance results are claimed. The assignment receipt date is still needed to calculate the four-day deadline.

## Requirements and architecture

See [PROJECT_UNDERSTANDING.md](PROJECT_UNDERSTANDING.md), [REQUIREMENTS_CHECKLIST.md](REQUIREMENTS_CHECKLIST.md), and [ARCHITECTURE.md](ARCHITECTURE.md) for the assignment analysis and planned structure. The implementation uses LangChain components with a LangGraph workflow; Groq is the real model backend, and a separate local stand-in is reserved for load experiments.

## Paper corpus

The original PDFs are under `data/papers/original/`. `data/papers/manifest.json` records each required version, source, download date, page count, and SHA-256 digest. The vector index and processed extraction will be rebuildable from these sources.

## Phase 1 Groq baseline

On 2026-11-10, three real `ChatGroq` streaming calls used `openai/gpt-oss-120b`, a fixed short prompt, and five-second spacing. All three succeeded. First model-token times were 697.1, 620.5, and 455.1 ms (median 620.5 ms; n=3); total times were 702.9, 624.4, and 460.7 ms (median 624.4 ms; n=3). Provider usage was 78 input and 35 output tokens per call. This is a small model probe, not an end-to-end assistant measurement. A preceding set of three attempts failed locally before completion requests were sent; they remain in the raw file and are marked invalid for provider latency analysis in `groq_probe_runs.jsonl`.

Successful responses reported 1,000 requests/day and 8,000 tokens/minute in their rate-limit headers. The current published free-plan table lists 30 requests/minute and 200,000 tokens/day for this model, but exact account values for those dimensions were not independently read from the account Limits page. No 429 response, recovery time, or in-flight concurrency limit was measured. Do not treat the documentation reference as an account-specific measurement.

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

`requirements.lock` pins the packages resolved and smoke-checked on Windows with CPython 3.12.14. Do not add credentials to the repository. Copy `.env.example` to `.env` for a local real run, then supply the model and key from your own Groq account; keep `.env` private and untracked. The initial provider probe was limited to the three recorded calls; further real-provider measurements will stay within verified account limits, and heavy-load scenarios are reserved for the local stand-in.

## Application, experiments, tests, and results

These commands will be documented as their features are implemented in Phases 2–5. No app, experiment runner, test suite, results CSV, or measured performance report exists yet. Real Groq results and simulated stand-in results will be labelled separately. Never interpret simulated measurements as Groq measurements.

## Current limitations and pending inputs

- Assignment receipt date is unknown, so the four-day deadline cannot yet be calculated.
- Account-specific model availability was confirmed; RPD and TPM were confirmed in response headers. RPM/TPD, concurrency, and reset/recovery remain unverified for this account.
- Official Groq rate-limit documentation was checked on 2026-11-10; see `experiments/raw/provider_limit_evidence.json`. It distinguishes documented limit behavior from account-specific values, which still require your Limits page.
- The Groq key is held only in the ignored local `.env`; it is not included in project outputs.
- PDF extraction notes are in `data/papers/processed/extraction_notes.md`; complex figure glyphs need page-level review during ingestion.
- Retrieval settings in `config/retrieval.yaml` are plan-proposed starting points, not evaluated choices.
