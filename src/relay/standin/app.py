"""FastAPI OpenAI-compatible local stand-in with SSE and synthetic limits."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from relay.standin.profiles import StandInProfile, load_profile
from relay.standin.simulator import ChatCompletionRequest, SimulationCall, SimulatorRejection, StandInSimulator


def _chunk(completion_id: str, model: str, delta: dict[str, Any], finish_reason: str | None = None,
           usage: dict[str, int] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"id": completion_id, "object": "chat.completion.chunk", "created": int(time.time()),
        "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
    if usage is not None:
        result["choices"] = []
        result["usage"] = usage
    return result


def create_standin_app(profile: StandInProfile | None = None, simulator: StandInSimulator | None = None) -> FastAPI:
    active_profile = profile or (simulator.profile if simulator else load_profile())
    engine = simulator or StandInSimulator(active_profile)
    app = FastAPI(title="The Relay Local Stand-In", version="0.3.0")
    app.state.simulator = engine
    app.state.profile = active_profile

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend": "stand-in", "profile_id": active_profile.profile_id,
                "active_requests": engine.active_requests}

    @app.post("/admin/reset")
    async def reset(request: Request):
        host = request.client.host if request.client else ""
        if host not in {"127.0.0.1", "::1", "testclient"}:
            raise HTTPException(status_code=403, detail="Reset is allowed only from loopback")
        return engine.reset()

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, response: Response):
        try:
            call = engine.prepare(payload)
        except SimulatorRejection as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.body, headers=exc.headers)
        created = int(time.time())
        usage = {"prompt_tokens": call.input_tokens, "completion_tokens": call.output_tokens,
                 "total_tokens": call.input_tokens + call.output_tokens}
        response.headers.update({"X-Standin-Profile": call.profile_id, "X-Simulation": "true",
            "X-RateLimit-Limit-Requests": str(active_profile.limits.requests_per_day),
            "X-RateLimit-Limit-Tokens": str(active_profile.limits.tokens_per_minute)})
        if not payload.stream:
            try:
                await engine.complete(call)
                return {"id": call.completion_id, "object": "chat.completion", "created": created,
                    "model": payload.model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": call.output}, "finish_reason": "stop"}],
                    "usage": usage, "simulation": {"profile_id": call.profile_id, "label": "stand-in"}}
            finally:
                engine.release()

        async def events():
            try:
                yield "data: " + json.dumps(_chunk(call.completion_id, payload.model, {"role": "assistant"})) + "\n\n"
                async for piece in engine.stream_chunks(call):
                    yield "data: " + json.dumps(_chunk(call.completion_id, payload.model, {"content": piece}), ensure_ascii=False) + "\n\n"
                    if call.partial_stream_after_chunks is not None and len(piece) > 0:
                        # The partial-stream control intentionally closes before finish/DONE.
                        call.partial_stream_after_chunks -= 1
                        if call.partial_stream_after_chunks <= 0:
                            return
                yield "data: " + json.dumps(_chunk(call.completion_id, payload.model, {}, "stop")) + "\n\n"
                yield "data: " + json.dumps(_chunk(call.completion_id, payload.model, {}, usage=usage)) + "\n\n"
                yield "data: [DONE]\n\n"
            finally:
                engine.release()

        return StreamingResponse(events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "X-Standin-Profile": call.profile_id, "X-Simulation": "true",
                     "X-RateLimit-Limit-Requests": str(active_profile.limits.requests_per_day),
                     "X-RateLimit-Limit-Tokens": str(active_profile.limits.tokens_per_minute)})

    return app


app = create_standin_app()
