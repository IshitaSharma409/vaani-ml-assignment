from __future__ import annotations

import asyncio
import csv
import threading
import time

from fastapi.testclient import TestClient

from relay.api import create_app
from relay.observability.csv_writer import ResultWriter
from relay.reliability.admission import AdmissionController


class FakeWorkflow:
    def __init__(self, block_first: bool = False, delay: float = 0):
        self.order = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block_first = block_first
        self.delay = delay

    def invoke(self, state):
        question = state["user_query"]
        self.order.append(question)
        self.entered.set()
        if self.block_first and len(self.order) == 1:
            self.release.wait(2)
        if self.delay:
            time.sleep(self.delay)
        return {"final_status": "ok", "final_response": f"Processed: {question}",
                "call_usage": [], "tool_calls": [], "stage_timings": {}}


def test_fifo_queue_rejects_full_admission_and_writes_one_row_each(tmp_path):
    async def scenario():
        workflow = FakeWorkflow(block_first=True)
        writer = ResultWriter(tmp_path / "results.csv", tmp_path / "trace.jsonl")
        admission = AdmissionController(workflow, capacity=1, workers=1, deadline_seconds=3,
                                        result_writer=writer)
        await admission.start()
        first = asyncio.create_task(admission.submit("first"))
        assert await asyncio.to_thread(workflow.entered.wait, 1)
        second = asyncio.create_task(admission.submit("second"))
        await asyncio.sleep(0.02)
        rejected = await admission.submit("third")
        assert rejected.user_outcome == "rejected"
        workflow.release.set()
        results = await asyncio.gather(first, second)
        await admission.close()
        assert [item.answer for item in results] == ["Processed: first", "Processed: second"]
        assert workflow.order == ["first", "second"]
        assert all(item.status == "ok" for item in results)
        with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 3
        assert len({row["request_id"] for row in rows}) == 3
        assert next(row for row in rows if row["scenario"] == "interactive" and row["status"] == "ok")["first_token_ms"]

    asyncio.run(scenario())


def test_deadline_returns_promptly_then_records_timeout(tmp_path):
    async def scenario():
        workflow = FakeWorkflow(delay=0.12)
        writer = ResultWriter(tmp_path / "results.csv", tmp_path / "trace.jsonl")
        admission = AdmissionController(workflow, capacity=1, workers=1, deadline_seconds=0.025,
                                        result_writer=writer)
        response = await admission.submit("slow")
        assert response.status == "timeout"
        await admission.close()
        with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1 and rows[0]["status"] == "timeout"
        assert rows[0]["user_outcome"] == "timed_out"

    asyncio.run(scenario())


def test_client_abandonment_stops_followup_request_and_is_logged(tmp_path):
    async def scenario():
        workflow = FakeWorkflow(block_first=True)
        writer = ResultWriter(tmp_path / "results.csv", tmp_path / "trace.jsonl")
        admission = AdmissionController(workflow, capacity=1, workers=1, deadline_seconds=3,
                                        result_writer=writer)
        await admission.start()
        request = asyncio.create_task(admission.submit("abandoned"))
        assert await asyncio.to_thread(workflow.entered.wait, 1)
        request.cancel()
        try:
            await request
        except asyncio.CancelledError:
            pass
        workflow.release.set()
        await admission.close()
        with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        assert row["user_outcome"] == "abandoned"
        assert row["first_token_ms"] == ""

    asyncio.run(scenario())


def test_fastapi_endpoint_uses_injected_admission_queue(tmp_path):
    workflow = FakeWorkflow()
    writer = ResultWriter(tmp_path / "results.csv", tmp_path / "trace.jsonl")
    admission = AdmissionController(workflow, capacity=2, workers=1, result_writer=writer)
    with TestClient(create_app(admission)) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.post("/ask", json={"question": "What is this?", "retrieval_setting": "medium"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Processed: What is this?"
