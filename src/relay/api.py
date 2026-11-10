"""FastAPI question endpoint backed by admission control and the shared graph."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request

from relay.graph.workflow import compile_workflow
from relay.llm.backend import create_model_backend
from relay.observability.csv_writer import ResultWriter
from relay.observability.logging import configure_logging
from relay.reliability.admission import AdmissionController
from relay.retrieval.index import RetrievalService
from relay.schemas import QuestionRequest, QuestionResponse


ROOT = Path(__file__).resolve().parents[2]


def _default_admission() -> AdmissionController:
    app_config = yaml.safe_load((ROOT / "config/app.yaml").read_text(encoding="utf-8"))
    model, source = create_model_backend()
    model_name = getattr(model.model, "model_name", getattr(model.model, "model", "unknown"))
    workflow = compile_workflow(RetrievalService(), model)
    request = app_config["request"]
    return AdmissionController(workflow, capacity=request["queue_capacity"], workers=request["workers"],
        deadline_seconds=request["deadline_seconds"], source=source, model_name=str(model_name))


def create_app(admission: AdmissionController | None = None) -> FastAPI:
    """Create an app; default backend and vector index initialize only at startup."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        if app.state.admission is None:
            app.state.admission = _default_admission()
        await app.state.admission.start()
        yield
        await app.state.admission.close()

    app = FastAPI(title="The Relay", version="0.3.0", lifespan=lifespan)
    app.state.admission = admission

    @app.get("/health")
    async def health():
        service = app.state.admission
        return {"status": "ok", "queue_depth": service.queue.qsize() if service else None,
                "queue_capacity": service.queue.maxsize if service else None,
                "source": service.source if service else None}

    @app.post("/ask", response_model=QuestionResponse)
    async def ask(payload: QuestionRequest, request: Request):
        task = asyncio.create_task(service_submit(app.state.admission, payload))
        try:
            while not task.done():
                if await request.is_disconnected():
                    task.cancel()
                    raise HTTPException(status_code=499, detail="Client disconnected")
                await asyncio.sleep(0.05)
            result = await task
        except asyncio.CancelledError:
            task.cancel()
            raise
        if result.status == "rate_limited":
            raise HTTPException(status_code=429, detail=result.model_dump())
        if result.status == "timeout":
            raise HTTPException(status_code=504, detail=result.model_dump())
        if result.user_outcome == "rejected":
            raise HTTPException(status_code=503, detail=result.model_dump())
        if result.status == "error":
            raise HTTPException(status_code=502, detail=result.model_dump())
        return result

    return app


async def service_submit(admission: AdmissionController, payload: QuestionRequest) -> QuestionResponse:
    return await admission.submit(payload.question, payload.scenario, payload.retrieval_setting)


app = create_app()
