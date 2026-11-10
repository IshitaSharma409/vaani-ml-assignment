from __future__ import annotations

import asyncio
import time

import httpx
from fastapi.testclient import TestClient

from relay.graph.workflow import compile_workflow
from relay.llm.standin_backend import StandInBackend
from relay.observability.timing import RequestTrace, reset_current_trace, set_current_trace
from relay.reliability.retry import RetryingModel, RetryPolicy
from relay.standin.app import create_standin_app
from relay.standin.profiles import load_profile
from relay.standin.simulator import ChatCompletionRequest, ChatMessage, SimulatorRejection, StandInSimulator


def fast_profile(**limit_updates):
    profile = load_profile()
    latency = profile.latency.model_copy(update={"first_token_ms_samples": [0.0], "total_ms_samples": [0.0], "inter_chunk_ms": 0.0})
    limits = profile.limits.model_copy(update=limit_updates)
    return profile.model_copy(update={"latency": latency, "limits": limits})


def request(stream=False):
    return {"model": "relay-standin-v1", "stream": stream, "messages": [
        {"role": "system", "content": "Answer from provided context."},
        {"role": "user", "content": 'Question. Retrieved excerpts: [{"chunk_id":"abcdef012345"}]'},
    ]}


def test_standin_normal_json_has_openai_shape_usage_and_simulation_labels():
    app = create_standin_app(fast_profile())
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=request())
    body = response.json()
    assert response.status_code == 200
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["total_tokens"] > 0
    assert body["simulation"]["label"] == "stand-in"
    assert response.headers["x-simulation"] == "true"


def test_standin_sse_streams_chunks_and_terminates_cleanly():
    with TestClient(create_standin_app(fast_profile())) as client:
        response = client.post("/v1/chat/completions", json=request(stream=True))
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    content_lines = [line for line in lines if '"content"' in line]
    assert response.headers["content-type"].startswith("text/event-stream")
    assert len(content_lines) > 1
    assert lines[-1] == "data: [DONE]"


def test_standin_rate_limits_reset_and_forced_server_failure():
    profile = fast_profile().model_copy(update={
        "limits": load_profile().limits.model_copy(update={"requests_per_day": 2}),
        "failures": load_profile().failures.model_copy(update={"forced_failure_statuses": [500]}),
    })
    with TestClient(create_standin_app(profile)) as client:
        failed = client.post("/v1/chat/completions", json=request())
        # The forced 500 does not count tokens; the next succeeds and consumes the single daily request.
        first_success = client.post("/v1/chat/completions", json=request())
        exhausted = client.post("/v1/chat/completions", json=request())
        reset = client.post("/admin/reset")
        after_reset_forced = client.post("/v1/chat/completions", json=request())
        after_reset_success = client.post("/v1/chat/completions", json=request())
    assert failed.status_code == 500
    assert first_success.status_code == 200
    assert exhausted.status_code == 429
    assert exhausted.headers["retry-after"] == "86400"
    assert reset.status_code == 200
    assert after_reset_forced.status_code == 500
    assert after_reset_success.status_code == 200


def test_standin_can_force_partial_sse_and_connection_style_failure():
    profile = fast_profile().model_copy(update={
        "failures": load_profile().failures.model_copy(update={"forced_failure_statuses": ["partial_stream", "connection_error"], "partial_stream_after_chunks": 1})
    })
    with TestClient(create_standin_app(profile)) as client:
        partial = client.post("/v1/chat/completions", json=request(stream=True))
        connection = client.post("/v1/chat/completions", json=request())
    assert partial.status_code == 200
    assert "[DONE]" not in partial.text
    assert connection.status_code == 503
    assert connection.headers["x-simulated-failure"] == "connection_error"


def test_standin_capacity_and_load_slowdown_are_deterministic():
    profile = load_profile().model_copy(update={
        "limits": load_profile().limits.model_copy(update={"concurrent_requests": 2}),
    })
    simulator = StandInSimulator(profile)
    payload = request()
    from relay.standin.simulator import ChatCompletionRequest
    first = simulator.prepare(ChatCompletionRequest.model_validate(payload))
    second = simulator.prepare(ChatCompletionRequest.model_validate(payload))
    assert second.active_load_delay_ms == profile.latency.load_slowdown_ms_per_extra_active_request
    try:
        try:
            simulator.prepare(ChatCompletionRequest.model_validate(payload))
        except SimulatorRejection as exc:
            assert exc.status_code == 429
        else:
            raise AssertionError("third concurrent request should be rate limited")
    finally:
        simulator.release()
        simulator.release()


def test_slow_profile_can_hit_a_client_timeout_and_releases_active_slot():
    profile = load_profile().model_copy(update={
        "latency": load_profile().latency.model_copy(update={"first_token_ms_samples": [100], "total_ms_samples": [100]})
    })
    app = create_standin_app(profile)

    async def call():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://standin") as client:
            await client.post("/v1/chat/completions", json=request())

    async def scenario():
        try:
            await asyncio.wait_for(call(), timeout=0.01)
        except asyncio.TimeoutError:
            pass
        await asyncio.sleep(0.02)

    asyncio.run(scenario())
    assert app.state.simulator.active_requests == 0


def test_same_graph_runs_with_langchain_local_backend():
    """Use a loopback Uvicorn server to exercise ChatOpenAI through the unchanged graph."""
    import socket
    import threading
    import uvicorn

    from langchain_core.documents import Document

    profile = fast_profile()
    server_app = create_standin_app(profile)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(server_app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started

    class FakeRetrieval:
        def search(self, query, setting, top_k=None):
            return [{"chunk_id": "abcdef012345", "chunk_text": "Evidence excerpt.", "paper_id": "1706.03762",
                     "title": "Attention Is All You Need", "arxiv_version": "7", "pdf_page_number": 4, "score": 0.5}], {
                         "vector_search_seconds": 0.0, "rerank_seconds": 0.0}

    try:
        model = StandInBackend(f"http://127.0.0.1:{port}/v1", max_tokens=256).create_model()
        wrapped = RetryingModel(model, RetryPolicy(max_attempts=1), output_token_reservation=256)
        graph = compile_workflow(FakeRetrieval(), wrapped, {"1706.03762": 15})
        trace = RequestTrace("What is supported?")
        token = set_current_trace(trace)
        try:
            state = graph.invoke({"request_id": "standin-test", "user_query": "What evidence excerpt is supported?", "source": "stand-in"})
        finally:
            reset_current_trace(token)
        assert state["final_status"] == "ok"
        assert "SIMULATED stand-in output" in state["final_response"]
        assert trace.model_calls and trace.first_model_token_ms is not None
    finally:
        server.should_exit = True
        thread.join(timeout=5)
