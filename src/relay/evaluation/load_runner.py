"""Local-only HTTP load runner for the five predeclared stand-in scenarios."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import httpx
import uvicorn
import yaml

from relay.api import create_app
from relay.graph.workflow import compile_workflow, corpus_page_counts
from relay.llm.backend import create_model_backend
from relay.llm.standin_backend import StandInBackend
from relay.reliability.admission import AdmissionController
from relay.reliability.retry import RetryPolicy, RetryingModel
from relay.retrieval.index import RetrievalService
from relay.standin.app import create_standin_app
from relay.standin.profiles import StandInProfile, load_profile


ROOT = Path(__file__).resolve().parents[3]
SCENARIO_PATH = ROOT / "config/scenarios.yaml"
SCENARIOS = ("steady", "burst", "sustained_overload", "per_minute_limit_halved", "daily_budget_exhausted")
LOAD_QUESTIONS = [
    "Why does scaled dot-product attention divide by the square root of d_k?",
    "What is Mimi and how does its codec represent semantic and acoustic information?",
    "How does H-Net choose content-aware chunk boundaries?",
]


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@contextmanager
def _serve(app: Any) -> Iterator[str]:
    port = _free_loopback_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
        log_level="error", access_log=False, lifespan="on"))
    thread = threading.Thread(target=server.run, name=f"relay-local-{port}", daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        if not thread.is_alive():
            raise RuntimeError("Local Uvicorn server exited before starting")
        time.sleep(0.02)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2)
        raise TimeoutError("Timed out starting a loopback-only local service")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("Local Uvicorn service failed to stop cleanly")


def _scenario_config() -> dict[str, Any]:
    value = yaml.safe_load(SCENARIO_PATH.read_text(encoding="utf-8"))
    if value.get("backend") != "stand-in" or not set(SCENARIOS).issubset(value.get("scenarios", {})):
        raise ValueError("Scenario configuration must define all five local stand-in scenarios")
    return value


def _request_trace_map(request_ids: set[str]) -> dict[str, dict[str, Any]]:
    path = ROOT / "experiments/raw/request_traces.jsonl"
    if not path.exists():
        return {}
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("request_id") in request_ids:
            found[record["request_id"]] = record
    return found


def _profile_for_scenario(scenario: dict[str, Any]) -> StandInProfile:
    profile = load_profile()
    overrides = scenario.get("profile_overrides", {})
    allowed = {"requests_per_minute", "tokens_per_minute", "requests_per_day", "tokens_per_day", "concurrent_requests"}
    if not set(overrides).issubset(allowed):
        raise ValueError("Unsupported stand-in limit override")
    return profile.model_copy(update={"limits": profile.limits.model_copy(update=overrides)})


async def _send_load(api_url: str, standin_url: str, scenario_name: str,
                     scenario: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    count = int(scenario["request_count"])
    bootstrap_count = int(scenario.get("setup_requests_before_measurement", 0))
    measured_count = max(0, count - bootstrap_count)
    async with httpx.AsyncClient(base_url=standin_url, timeout=90) as bootstrap_client:
        bootstrap = []
        for _ in range(bootstrap_count):
            response = await bootstrap_client.post("/v1/chat/completions", json={
                "model": "relay-standin-v1", "messages": [{"role": "user", "content": "synthetic daily budget setup"}],
            })
            bootstrap.append({"http_status": response.status_code,
                              "simulation": response.headers.get("x-simulation", "false")})
    async with httpx.AsyncClient(base_url=api_url, timeout=90) as client:
        started = time.monotonic()
        responses: list[httpx.Response | None] = [None] * measured_count

        async def submit(index: int) -> None:
            question = LOAD_QUESTIONS[index % len(LOAD_QUESTIONS)]
            responses[index] = await client.post("/ask", json={
                "question": question, "scenario": scenario_name, "retrieval_setting": "medium",
            })

        tasks = []
        if scenario.get("arrival_pattern") == "simultaneous":
            tasks = [asyncio.create_task(submit(index)) for index in range(measured_count)]
        else:
            rate = float(scenario.get("offered_requests_per_second", 1.0))
            if rate <= 0:
                raise ValueError("Offered request rate must be positive")
            for index in range(measured_count):
                scheduled = started + index / rate
                await asyncio.sleep(max(0.0, scheduled - time.monotonic()))
                tasks.append(asyncio.create_task(submit(index)))
        if tasks:
            await asyncio.gather(*tasks)
        finish = time.monotonic()
    request_results = []
    for response in responses:
        assert response is not None
        try:
            body = response.json()
        except ValueError:
            body = {"detail": response.text[:300]}
        detail = body.get("detail", body)
        request_results.append({"http_status": response.status_code,
            "status": detail.get("status") if isinstance(detail, dict) else None,
            "user_outcome": detail.get("user_outcome") if isinstance(detail, dict) else None,
            "request_id": detail.get("request_id") if isinstance(detail, dict) else None})
    offered_elapsed = max(0.0, finish - started)
    return request_results, {"bootstrap_calls": bootstrap, "offered_requests": measured_count,
        "offered_elapsed_seconds": offered_elapsed,
        "offered_rate_per_second": measured_count / offered_elapsed if offered_elapsed else None}


async def run_scenario(name: str, scenario: dict[str, Any], retrieval: RetrievalService) -> dict[str, Any]:
    run_started = datetime.now(timezone.utc)
    profile = _profile_for_scenario(scenario)
    standin_app = create_standin_app(profile)
    with _serve(standin_app) as standin_url:
        model, source = create_model_backend("standin", base_url=f"{standin_url}/v1")
        graph = compile_workflow(retrieval, model, corpus_page_counts())
        app_settings = yaml.safe_load((ROOT / "config/app.yaml").read_text(encoding="utf-8"))
        request_settings = app_settings["request"]
        admission = AdmissionController(graph, capacity=request_settings["queue_capacity"],
            workers=request_settings["workers"], deadline_seconds=request_settings["deadline_seconds"],
            source=source, model_name=profile.model_id)
        application = create_app(admission)
        with _serve(application) as api_url:
            results, offered = await _send_load(api_url, standin_url, name, scenario)
    trace_map = _request_trace_map({item["request_id"] for item in results if item.get("request_id")})
    internal_errors = [
        {"request_id": request_id, "error_class": trace.get("error_class"),
         "error_detail": trace.get("error_detail")}
        for request_id, trace in trace_map.items() if trace.get("error_detail")
    ]
    bootstrap_errors = [call for call in offered.get("bootstrap_calls", [])
                        if call.get("http_status") != 200 or call.get("simulation") != "true"]
    scenario_errors = []
    if name in {"daily_budget_exhausted", "per_minute_limit_halved"} and not any(
        item["http_status"] == 429 for item in results
    ):
        scenario_errors.append({"requirement": "configured limit scenario must produce an application-level HTTP 429"})
    status_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    for result in results:
        status = result["status"] or f"http_{result['http_status']}"
        outcome = result["user_outcome"] or "unparsed_error"
        status_counts[status] = status_counts.get(status, 0) + 1
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
    return {
        "run_id": f"{name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
        "started_at": run_started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(), "scenario": name,
        "source": "stand-in", "profile_id": profile.profile_id, "seed": profile.seed,
        "profile_limits": profile.limits.model_dump(), "scenario_config": scenario,
        **offered, "completed_http_responses": len(results), "status_counts": status_counts,
        "user_outcome_counts": outcome_counts, "requests": results,
        "valid": not bool(internal_errors or bootstrap_errors or scenario_errors),
        "invalid_reason": ({"internal_errors": internal_errors, "bootstrap_errors": bootstrap_errors,
                            "scenario_errors": scenario_errors}
                           if internal_errors or bootstrap_errors or scenario_errors else None),
    }


async def run_all() -> list[dict[str, Any]]:
    config = _scenario_config()
    retrieval = RetrievalService()
    results = []
    for name in SCENARIOS:
        record = await run_scenario(name, config["scenarios"][name], retrieval)
        results.append(record)
        output = ROOT / "experiments/raw/load_runs.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({key: record[key] for key in (
            "scenario", "offered_requests", "offered_rate_per_second", "status_counts", "user_outcome_counts"
        )}, ensure_ascii=False))
    return results


async def run_retry_diagnostic(retrieval: RetrievalService) -> dict[str, Any]:
    """Compare two queued users with/without one forced 500 then successful retry."""
    app_config = yaml.safe_load((ROOT / "config/app.yaml").read_text(encoding="utf-8"))
    observations = {}
    for label, forced in (("control", []), ("forced_retry", [500])):
        base = load_profile()
        profile = base.model_copy(update={
            "limits": base.limits.model_copy(update={"tokens_per_minute": 100_000,
                "requests_per_minute": 100, "requests_per_day": 100}),
            "failures": base.failures.model_copy(update={"forced_failure_statuses": forced,
                "fail_probability": 0.0, "fail_every_n_requests": 0}),
        })
        with _serve(create_standin_app(profile)) as standin_url:
            model, source = create_model_backend("standin", base_url=f"{standin_url}/v1")
            graph = compile_workflow(retrieval, model, corpus_page_counts())
            settings = app_config["request"]
            admission = AdmissionController(graph, capacity=2, workers=1, deadline_seconds=60,
                source=source, model_name=profile.model_id)
            with _serve(create_app(admission)) as api_url:
                async with httpx.AsyncClient(base_url=api_url, timeout=90) as client:
                    payload = {"question": LOAD_QUESTIONS[0], "scenario": f"retry_diagnostic_{label}",
                               "retrieval_setting": "medium"}
                    response_pair = await asyncio.gather(
                        client.post("/ask", json=payload), client.post("/ask", json=payload)
                    )
        observations[label] = [
            {"http_status": response.status_code,
             **(response.json().get("detail", response.json()) if isinstance(response.json(), dict) else {})}
            for response in response_pair
        ]
    all_ids = {item.get("request_id") for group in observations.values() for item in group if item.get("request_id")}
    trace_map = _request_trace_map(all_ids)
    for label, group in observations.items():
        for item in group:
            trace = trace_map.get(item.get("request_id"), {})
            item["queue_wait_ms"] = trace.get("queue_wait_ms")
            item["retry_wait_ms"] = trace.get("retry_wait_ms")
            item["model_calls"] = trace.get("model_calls", [])
            item["total_ms"] = trace.get("total_ms")
    control_peer = observations["control"][1]
    retry_peer = observations["forced_retry"][1]
    record = {
        "run_id": f"retry-diagnostic-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
        "timestamp": datetime.now(timezone.utc).isoformat(), "source": "stand-in",
        "purpose": "measure one retry improving a forced 500 outcome while delaying a queued peer",
        "control": observations["control"], "forced_retry": observations["forced_retry"],
        "peer_queue_wait_delta_ms": (float(retry_peer.get("queue_wait_ms") or 0)
            - float(control_peer.get("queue_wait_ms") or 0)),
        "valid": all(item.get("http_status") == 200 for group in observations.values() for item in group),
        "assumptions": ["same seed/profile and two synchronous arrivals", "one worker intentionally isolates queueing", "all responses are synthetic"],
    }
    out = ROOT / "experiments/raw/retry_diagnostic.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, help="Run one scenario; default runs all five.")
    parser.add_argument("--retry-diagnostic", action="store_true",
                        help="Run a separate local pair comparing retry success and peer queue delay.")
    args = parser.parse_args()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    config = _scenario_config()
    if args.retry_diagnostic:
        record = asyncio.run(run_retry_diagnostic(RetrievalService()))
        print(json.dumps({"valid": record["valid"], "peer_queue_wait_delta_ms": record["peer_queue_wait_delta_ms"]}, indent=2))
    elif args.scenario:
        record = asyncio.run(run_scenario(args.scenario, config["scenarios"][args.scenario], RetrievalService()))
        output = ROOT / "experiments/raw/load_runs.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({key: record[key] for key in ("scenario", "offered_requests", "status_counts", "user_outcome_counts")}, indent=2))
    else:
        asyncio.run(run_all())


if __name__ == "__main__":
    main()
