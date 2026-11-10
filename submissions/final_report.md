# The Relay: bounded research assistant

**Engineering experiment report | 10 November 2026**

## 1. What I built and why

The Relay answers engineering questions from exactly three versioned research papers. A LangGraph prepares the request, retrieves page-bounded passages, drafts and verifies cited claims, and refuses when evidence is missing. I used a local stand-in for load experiments so heavy traffic would not be sent to the Groq free tier. The key design choice is to buffer the answer until citation and support checks finish; this costs time but avoids releasing an unchecked draft.

## 2. The real service and my stand-in

The model used was `openai/gpt-oss-120b`. Real response headers showed 8,000 TPM and 1,000 RPD; a real TPM 429 reported 7,179 tokens used plus 1,914 requested and advised an 8.1975-second wait. Account RPM, TPD, concurrency, and exact recovery were not confirmed. The local stand-in reuses the same application graph and was calibrated from three real single-user latency/usage samples. Its quotas, slowdown, retries, and load results are synthetic, not Groq measurements.

{{TABLE1}}

## 3. Results

The stand-in ran all five fixed load shapes. Its constant-arrival `steady` run saturated synthetic TPM: 3 of 30 requests completed. That measured outcome disproved the configuration's “below capacity” assumption. Across the three settings, the low-rate Groq sample covers all 15 answerable questions; 41 of 54 requests returned answers that were manually graded against their cited source pages. Thirteen operational failures/limits remain ungraded and separately counted. A resume-filter label mismatch caused three prior questions to be repeated once per setting (nine extra requests); the 45 supplemental calls remained spaced one minute apart.

{{TABLE2A}}

{{TABLE2B}}

<!-- PAGEBREAK -->

## 4. Where the time goes

In the controlled stand-in retry diagnostic, the queued peer took 3,187 ms: 1,781 ms waiting in queue, 1,406 ms doing active work, and 0 ms retrying itself. Its first model token arrived at 2,531 ms, so most of its delay happened before that token. The preceding request's 387 ms retry backoff kept the only worker occupied and caused the peer's queue wait. The numbers reconcile to the recorded trace.

## 5. When a limit is hit

At a minute limit, the application returns `rate_limited` (HTTP 429) rather than silently retrying every request; overload that fills the bounded queue is rejected. At daily exhaustion, the stand-in's one measured probe returned HTTP 429 with no model retry. The system does not weaken a response or bypass citation checks; it returns no synthesized answer.

## Retrieval: quality versus cost

All 18 questions were searched under each setting. Exact expected paper/version/page hit@3 was 9/15 for `small_fast`, 11/15 for `medium`, and 12/15 for `medium_reranked`; the latter added 0.7 ms mean local BM25 rerank time. The balanced real answer sample covers 15 answerable questions per setting; answer-grade denominators exclude operational failures and are shown explicitly. `medium_reranked` is the measured default. Near the local safety limit, callers can select `small_fast`, while citation and verification checks stay enabled.

{{TABLE3}}

## How this library will mislead a naive retriever

- A query for Moshi's commercial per-minute price retrieves dialogue/training passages; the papers publish no such price.
- A question about the first production Transformer deployment retrieves translation benchmark tables; the corpus gives no deployment system or launch date.
- A request for a direct H-Net versus Moshi LibriSpeech score mixes unrelated H-Net and Moshi benchmark tables; there is no head-to-head experiment.

## 6. Retries

The client retries bounded network/5xx and recoverable minute-limit failures, but does not retry authentication, invalid requests, daily exhaustion, or terminal timeouts. In the stand-in diagnostic, retry recovered one forced 500 after 387 ms. It delayed a queued peer by 1,781 ms, showing the queueing cost of holding a worker during backoff.

## 7. What the person sees

At 3 PM after daily exhaustion, the response body says: **“The assistant could not complete a verified answer. Please try again later.”** The API returns HTTP 429. It does not promise a reset time that has not been verified.

## 8. What I'm unsure about, what I left out, what I learned

The least certain assumption is how well a stand-in calibrated from three single-user observations predicts real-provider tail latency and concurrency; if it is wrong, the local overload percentiles do not predict Groq behavior. I deliberately left out a user interface and cloud deployment. **Student reflection required:** add one thing learned in your own words; it is not in the conversation evidence yet.

## Your feedback on Vaani

**Student feedback required.** I reviewed the public landing page on 10 November 2026; it highlights Teacher AI's brief-to-agent flow, clarifying questions, and prompt-diff approval. Add what stood out to you, what confused you, and one change you would make after your own review. Do not present this assistant's page observation as your personal reaction.

## How I worked with the assistant

The three prompts that shaped the work were: the master project prompt (scope, stack, evidence rules, and approval gates); “should i write api key under \"\" pr not” (safe credential setup); and repeated “next” approvals (phase-by-phase control). The assistant implemented and tested the pipeline, experiments, and reports; the user supplied the assignment, configured the private key locally, and approved each phase. The assistant found that its “below capacity” scenario assumption was wrong by comparing the run with the 3/30 completion result. **Student confirmation required:** identify an error you personally caught, one decision you overruled, and which choices or conclusions are distinctly yours. The available thread does not establish those personal claims.
