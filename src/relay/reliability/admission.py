"""Bounded FIFO admission and asynchronous execution in front of LangGraph."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from relay.observability.csv_writer import ResultWriter
from relay.observability.logging import log_event
from relay.observability.timing import RequestTrace, reset_current_trace, set_current_trace
from relay.schemas import QuestionResponse, ResultRecord


logger = logging.getLogger(__name__)


@dataclass
class _Job:
    trace: RequestTrace
    future: asyncio.Future


class AdmissionController:
    """FIFO queue with immediate overload rejection and bounded worker count."""

    def __init__(self, workflow: Any, *, capacity: int = 8, workers: int = 4,
                 deadline_seconds: float = 60, source: str = "real", model_name: str = "unknown",
                 result_writer: ResultWriter | None = None):
        if capacity < 1 or workers < 1 or deadline_seconds <= 0:
            raise ValueError("Queue capacity, workers, and deadline must be positive")
        self.workflow = workflow
        self.queue: asyncio.Queue[_Job] = asyncio.Queue(maxsize=capacity)
        self.worker_count, self.deadline_seconds = workers, deadline_seconds
        self.source, self.model_name = source, model_name
        root = Path(__file__).resolve().parents[3]
        self.writer = result_writer or ResultWriter(root / "experiments/raw/results.csv", root / "experiments/raw/request_traces.jsonl")
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._peak_queue = 0
        self._peak_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._workers = [asyncio.create_task(self._worker(), name=f"relay-worker-{i}")
                         for i in range(self.worker_count)]

    async def close(self) -> None:
        if not self._running:
            return
        await self.queue.join()
        self._running = False
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def submit(self, question: str, scenario: str = "interactive", retrieval_setting: str = "medium_reranked") -> QuestionResponse:
        if not self._running:
            await self.start()
        loop = asyncio.get_running_loop()
        trace = RequestTrace(question=question, scenario=scenario, source=self.source,
                             retrieval_setting=retrieval_setting, model_name=self.model_name,
                             deadline_seconds=self.deadline_seconds)
        trace.mark_enqueued(self.queue.qsize())
        future = loop.create_future()
        job = _Job(trace=trace, future=future)
        try:
            self.queue.put_nowait(job)
            self._peak_queue = max(self._peak_queue, self.queue.qsize())
        except asyncio.QueueFull:
            trace.final_status = "error"
            trace.final_response = "The service is busy. Please retry shortly."
            trace.error_class = "QueueFull"
            trace.http_status = 503
            trace.finish()
            self._write_trace(trace, "rejected")
            log_event(logger, logging.WARNING, "request_rejected", request_id=trace.request_id,
                      outcome="rejected", queue_depth=trace.queue_depth_at_arrival)
            return self._response(trace, "rejected")
        remaining = max(0.0, trace.deadline_monotonic - time.monotonic())
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=remaining)
        except asyncio.TimeoutError:
            trace.cancel("deadline")
            trace.final_status = "timeout"
            trace.final_response = "The request exceeded its time limit. Please try again."
            trace.error_class = trace.error_class or "RequestDeadlineExceeded"
            trace.first_user_token_ms = (time.monotonic() - trace.started_monotonic) * 1000
            trace.finish()
            return self._response(trace, "timed_out")
        except asyncio.CancelledError:
            trace.cancel("abandoned")
            raise

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get()
            trace = job.trace
            try:
                trace.mark_dequeued()
                if trace.cancel_event.is_set():
                    trace.final_status = "timeout" if trace.cancel_reason == "deadline" else "error"
                    trace.final_response = "Request was cancelled before processing began."
                    trace.error_class = "RequestCancelled"
                    response = self._response(trace, "abandoned" if trace.cancel_reason == "abandoned" else "timed_out")
                else:
                    token = set_current_trace(trace)
                    try:
                        state = await asyncio.to_thread(self.workflow.invoke, {
                            "request_id": trace.request_id, "user_query": trace.question,
                            "scenario": trace.scenario, "source": trace.source,
                            "retrieval_setting": trace.retrieval_setting,
                        })
                        trace.state = state
                        trace.tool_calls = state.get("tool_calls", [])
                        trace.stage_timings = state.get("stage_timings", {})
                        if trace.cancel_event.is_set():
                            trace.final_status = "timeout" if trace.cancel_reason in {"deadline", "model_timeout"} else "error"
                            trace.final_response = "The request was cancelled before a verified answer could be released."
                        else:
                            trace.final_status = state.get("final_status", "error")
                            trace.final_response = state.get("final_response", "")
                        if trace.final_status == "error" and trace.error_class is None:
                            trace.error_class = state.get("failure_stage", "WorkflowError")
                    finally:
                        reset_current_trace(token)
                    response = self._response(trace, self._user_outcome(trace))
                if not job.future.done():
                    job.future.set_result(response)
            except Exception as exc:
                trace.final_status = "error"
                trace.final_response = "The request could not be completed. Please try again later."
                trace.error_class = type(exc).__name__
                trace.error_detail = re.sub(r"gsk_[A-Za-z0-9_-]+", "[REDACTED]", str(exc).replace("\n", " "))[:400]
                trace.finish()
                response = self._response(trace, "error")
                if not job.future.done():
                    job.future.set_result(response)
            finally:
                trace.finish()
                outcome = self._user_outcome(trace)
                self._write_trace(trace, outcome)
                log_event(logger, logging.INFO, "request_finished", request_id=trace.request_id,
                          source=trace.source, status=self._response(trace, outcome).status, outcome=outcome,
                          total_ms=round(trace.total_ms, 1), queue_wait_ms=round(trace.queue_wait_ms, 1),
                          model_calls=len(trace.model_calls))
                self.queue.task_done()

    def _user_outcome(self, trace: RequestTrace) -> str:
        if trace.cancel_reason == "abandoned":
            return "abandoned"
        if trace.cancel_reason or trace.final_status == "timeout" or any(
            call.get("outcome") == "timeout" for call in trace.model_calls
        ):
            return "timed_out"
        if trace.final_status in {"ok", "refused_correctly"}:
            return "refused" if trace.state.get("final_status") == "refused_correctly" else "success"
        if trace.http_status == 429 or any(call.get("http_status") == 429 or call.get("outcome") in {
            "minute_rate_limit", "rate_limit_unknown", "daily_budget", "concurrency_limit", "requests_per_minute", "tokens_per_minute", "requests_per_day", "tokens_per_day"
        } for call in trace.model_calls):
            return "error"
        return "error"

    def _response(self, trace: RequestTrace, outcome: str) -> QuestionResponse:
        status = trace.final_status or "error"
        if status == "refused_correctly":
            status = "ok"
        if status == "error" and any(call.get("outcome") == "timeout" for call in trace.model_calls):
            status = "timeout"
        if status == "error" and any(call.get("outcome") in {
            "minute_rate_limit", "rate_limit_unknown", "daily_budget", "concurrency_limit", "requests_per_minute", "tokens_per_minute", "requests_per_day", "tokens_per_day"
        } for call in trace.model_calls):
            status = "rate_limited"
        if status not in {"ok", "rate_limited", "timeout", "error"}:
            status = "error"
        return QuestionResponse(request_id=trace.request_id, status=status, user_outcome=outcome,
                                answer=trace.final_response, model_name=trace.model_name)

    def _write_trace(self, trace: RequestTrace, outcome: str) -> None:
        trace.finish()
        if trace.first_user_token_ms is None and trace.final_response and outcome != "abandoned":
            trace.first_user_token_ms = trace.total_ms
        trace.model_calls = list(trace.model_calls)
        input_values = [call.get("tokens_in") for call in trace.model_calls if call.get("tokens_in") is not None]
        output_values = [call.get("tokens_out") for call in trace.model_calls if call.get("tokens_out") is not None]
        sources = {call.get("token_source", "unavailable") for call in trace.model_calls}
        token_source = next(iter(sources)) if len(sources) == 1 else (
            "mixed" if sources else "unavailable"
        )
        retries = sum(int(call.get("retry_count", 0)) for call in trace.model_calls)
        status = trace.final_status or "error"
        if status == "refused_correctly":
            status = "ok"
        if status == "error" and any(call.get("outcome") == "timeout" for call in trace.model_calls):
            status = "timeout"
        elif status == "error" and any(call.get("outcome") in {
            "minute_rate_limit", "rate_limit_unknown", "daily_budget", "concurrency_limit", "requests_per_minute", "tokens_per_minute", "requests_per_day", "tokens_per_day"
        } for call in trace.model_calls):
            status = "rate_limited"
        elif status not in {"ok", "rate_limited", "timeout", "error"}:
            status = "error"
        calls_total = sum(int(call.get("attempts", 1)) for call in trace.model_calls)
        stage_calls = trace.state.get("call_usage", [])
        verification_ms = sum(float(call.get("elapsed_seconds", 0)) * 1000 for call in stage_calls if call.get("stage") == "verification")
        retrieval_ms = float(trace.stage_timings.get("retrieval_seconds", 0)) * 1000
        record = ResultRecord(
            request_id=trace.request_id, scenario=trace.scenario, source=trace.source,
            start_time=trace.start_time, first_token_ms=trace.first_user_token_ms,
            total_ms=trace.total_ms, status=status, retries=retries,
            tokens_in=sum(input_values) if input_values else None,
            tokens_out=sum(output_values) if output_values else None,
            retrieval_setting=trace.retrieval_setting, answer_grade=trace.answer_grade,
            user_outcome=outcome, queue_wait_ms=trace.queue_wait_ms, active_work_ms=trace.active_work_ms,
            retry_wait_ms=trace.retry_wait_ms, retrieval_ms=retrieval_ms, verification_ms=verification_ms,
            first_model_token_ms=trace.first_model_token_ms, first_user_token_ms=trace.first_user_token_ms,
            llm_calls=calls_total, queue_depth_at_arrival=trace.queue_depth_at_arrival,
            peak_queue_length=self._peak_queue, model_name=trace.model_name,
            tokens_source=token_source, error_class=trace.error_class, http_status=trace.http_status,
            retry_after_s=trace.retry_after_s,
        )
        detailed = {
            "request_id": trace.request_id, "scenario": trace.scenario, "source": trace.source,
            "question": trace.question, "answer": trace.final_response, "status": status,
            "user_outcome": outcome, "start_time": trace.start_time,
            "finish_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_ms": trace.total_ms, "queue_wait_ms": trace.queue_wait_ms,
            "active_work_ms": trace.active_work_ms, "retry_wait_ms": trace.retry_wait_ms,
            "first_model_token_ms": trace.first_model_token_ms, "first_user_token_ms": trace.first_user_token_ms,
            "model_calls": trace.model_calls, "tool_calls": trace.tool_calls,
            "stage_timings": trace.stage_timings, "retrieval_setting": trace.retrieval_setting,
            "model_name": trace.model_name, "error_class": trace.error_class, "error_detail": trace.error_detail,
            "http_status": trace.http_status, "retry_after_s": trace.retry_after_s,
            "cancel_reason": trace.cancel_reason,
        }
        self.writer.write(record, detailed)
