# Project Understanding — The Relay

## Source authority and scope

The authoritative assignment is `The_Relay_Assignment_Brief.pdf`. `IMPLEMENTATION_PLAN.md` translates that brief into a five-phase roadmap and proposes the implementation architecture. This document records the analysis baseline. Phase 1 setup has since been authorized and is underway: the planned scaffold/configuration, environment, exact paper corpus, and corpus checks have been added. A modest three-call Groq baseline is now preserved; no end-to-end assistant pipeline or full experiment set exists yet.

The assignment asks for a small, defensible research-answering system that retrieves from a fixed three-paper library, produces location-cited answers, and checks them before release. The engineering question is how to balance response speed, answer quality, cost/quota, and availability when multi-step model calls, tools, and user arrivals are unpredictable. The system must never trade correctness for speed by releasing unsupported claims.

## Objectives and expected outcome

The completed project is expected to provide:

1. A working Python pipeline with at least four steps, a tool call, and a conditional branch or bounded loop.
2. Retrieval only from the exact required versions of three arXiv papers, with page-grounded citations and a safe response when evidence is absent.
3. Real inference through the user's own Groq free-tier account, with a modest measured baseline and no quota circumvention.
4. A separately identifiable, controllable local stand-in that can be slow, fail, rate-limit, slow under load, and stream incrementally; calibration and its differences from Groq must be documented.
5. Reproducible experiments for real and simulated traffic, retrieval settings, answer quality, latency, failures, retries, tokens, and queueing.
6. Raw `results.csv` evidence, required result tables, and a maximum two-page report with the prescribed headings, Vaani feedback, and genuine assistant-work history.
7. A reproducible, explainable submission another developer can run and the student can defend in a live discussion.

The paper corpus is fixed:

| Paper | Required version |
|---|---|
| Attention Is All You Need (arXiv 1706.03762) | v7 |
| Moshi: a speech-text foundation model for real-time dialogue (arXiv 2410.00037) | v1 |
| Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (arXiv 2507.07955) | v2 |

## System concept

The proposed request path uses admission control and a bounded queue, initializes trace state, normalizes the question, calls a retrieval tool against the local fixed corpus, checks evidence, and conditionally refines retrieval once. Supported evidence proceeds to cited-answer generation and verification; a single bounded revision is permitted after verification failure. Unsupported or still-invalid answers are refused safely. The result and per-stage measurements are recorded. Draft model output is buffered until the citation/grounding gate passes.

LangChain supplies model, prompt, retriever, and tool components. LangGraph is justified by the explicit shared state, evidence routing, bounded retrieval recovery and answer revision, and predictable refusal/error termination. A single graph is proposed; no multi-agent system is needed. A backend boundary lets the same application/workflow use `ChatGroq` for genuine provider calls or the local stand-in for simulation. Simulated results must always be labelled as such.

## Existing directory and proposed layout

At the original analysis inspection, the project directory contained only:

```text
assignment/
├── The_Relay_Assignment_Brief.pdf
└── IMPLEMENTATION_PLAN.md
```

The plan's target layout places `README.md`, `IMPLEMENTATION_PLAN.md`, and project configuration at the repository root, with `config/`, `data/`, `src/relay/`, `tests/`, `scripts/`, `experiments/`, and `submissions/` beneath it. It does not define a `docs/` directory. To avoid inventing a new structural convention, these three requested analysis documents are placed at the root beside the plan. This documents the current state and does not create the implementation skeleton or begin Phase 1.

## Five implementation phases

| Phase | Objective | Gate / expected evidence |
|---|---|---|
| 1. Project setup, corpus acquisition, real Groq baseline | Set up the Python/framework environment; obtain and verify the exact PDFs; configure secrets safely; make a modest real Groq probe; record limits and uncertainty. | Reproducible environment, exact corpus/manifest, real baseline, dated evidence, and a safe request budget before larger experiments. |
| 2. Local corpus, RAG, LangGraph assistant | Build page-aware ingestion, chunking/index and three retrieval configurations; expose an explicit retrieval tool; implement citations/refusal; compile and smoke-test the graph. | Normal, refinement, verification, and refusal paths terminate; citation locations resolve to the source PDFs; small real-Groq end-to-end check succeeds. |
| 3. Request control, reliability, observability, stand-in | Add bounded admission, concurrency/usage budgets, deadlines, explicit retries, per-request/stage telemetry, and a configurable streaming simulator calibrated to Phase 1. | Same graph works with both backends; local controlled reliability checks and telemetry reconcile; no load is sent to Groq. |
| 4. Evaluation, retrieval experiments, load testing | Create 15+ paper-grounded questions; grade outputs; compare 3+ retrieval settings; run five required scenarios, with heavy loads on the stand-in; analyze raw data. | Raw results support the required tables, retrieval comparison, slow-request trace, retry-worsened case, and budget-exhaustion behavior. |
| 5. Reporting and final submission | Generate metrics from raw data; validate schema and costs; complete report, Vaani feedback, worklog/history, README, clean-room reproduction. | Required report has all headings and is <=2 pages; artifacts are reproducible and contain no secret or fabricated evidence. |

## Constraints, dependencies, and risks

- Use Python 3.11+ per the requested implementation brief, while selecting versions verified to work together for LangChain, LangGraph, and `langchain-groq`/`ChatGroq`. The plan further proposes FastAPI/Uvicorn, `httpx`, local page-aware PDF processing, Chroma, local sentence-transformer embeddings, optional local cross-encoder reranking, YAML/CSV/JSONL, and pytest. Optional libraries should be retained only if they serve the defined experiments.
- The user's local Groq key is configured in the ignored `.env` and was not exposed in output. The selected model is accessible. Successful response headers verify RPD and TPM; RPM/TPD are retained only as current official free-plan references, while concurrency and recovery remain unmeasured.
- Groq limits and model availability change. Capture dated console/documentation observations and direct response metadata; never infer account-specific RPM/TPM/RPD/TPD/concurrency.
- The assignment receipt date is not stated. PDF creation metadata is not proof of receipt, so the four-day deadline cannot yet be calculated.
- Exact paper versioning, extraction quality, page-number conventions, citation validity, token usage in streaming, hidden framework retries, and simulator realism need explicit verification.
- The required CSV has a fixed minimum schema, while report metrics also need queue, duration, cost basis, scenario metadata, and other data. The plan's additive telemetry fields and companion evidence preserve the required columns while making additional metrics auditable.
- Heavy load, daily exhaustion, and artificial failures must be simulated locally; never present simulated behavior as measured Groq behavior.
- Required genuine Vaani feedback and conversation history cannot be invented. Preserve actual observations and interaction history when those phases are reached.

## Analysis boundary

No plan-versus-assignment conflict was found in the core objective, technologies, fixed corpus, experiments, or deliverables. The plan expands the brief's minimum three load shapes to five named scenarios for the required tables; this is consistent with the brief's examples and table requirements. Its additional CSV telemetry is an additive interpretation of the brief's required columns and supports the brief's requirement that results be recomputable. The plan appropriately labels its architecture/configuration values as proposals rather than measured facts.

Phase 1 is in progress following the user's explicit “next” authorization. Local setup, corpus work, model-access verification, and the modest real baseline are complete. The deadline still needs the actual assignment receipt date; Phase 2 has not begun and requires the user's review and explicit approval.
