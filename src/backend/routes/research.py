import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config import OLLAMA_BASE_URL, SEARX_URL
from lib.research_config import build_research_config, write_research_config
from lib.research_trace import ResearchTrace

TRACES_DIR = Path(os.environ.get("RESEARCH_TRACES_DIR", str(Path(__file__).resolve().parent.parent.parent.parent / "chat_histories" / "research_traces")))

router = APIRouter(prefix="/api", tags=["research"])

_env_lock = asyncio.Lock()


class ResearchRequest(BaseModel):
    query: str
    fast_model: str
    smart_model: str
    strategic_model: str
    depth: int = 2
    breadth: int = 4
    reasoning_effort: str | None = None
    report_type: str = "deep"


class SSEWriter:
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def send(self, event: str, data: Any) -> None:
        payload = json.dumps(data, default=str)
        await self.queue.put((event, payload))

    async def send_step(self, step: str, event_type: str, details: dict | None = None) -> None:
        await self.send("step", {
            "step": step,
            "event_type": event_type,
            "details": details or {},
            "timestamp": time.time(),
        })

    async def send_progress(self, progress) -> None:
        await self.send("progress", {
            "current_depth": getattr(progress, "current_depth", 0),
            "total_depth": getattr(progress, "total_depth", 0),
            "current_breadth": getattr(progress, "current_breadth", 0),
            "total_breadth": getattr(progress, "total_breadth", 0),
            "current_query": getattr(progress, "current_query", None),
            "completed_queries": getattr(progress, "completed_queries", 0),
            "total_queries": getattr(progress, "total_queries", 0),
        })

    async def send_result(self, report: str, sources: list[str], costs: float) -> None:
        await self.send("result", {
            "report": report,
            "sources": sources,
            "costs": costs,
        })

    async def send_error(self, message: str) -> None:
        await self.send("error", {"message": message})


class TraceLogHandler:
    def __init__(self, writer: SSEWriter, trace: ResearchTrace):
        self.writer = writer
        self.trace = trace

    async def on_tool_start(self, tool_name: str, **kwargs) -> None:
        self.trace.add_step(tool_name, "tool", kwargs)
        await self.writer.send_step(tool_name, "tool", kwargs)

    async def on_agent_action(self, action: str, **kwargs) -> None:
        details = {"action": action, **kwargs}
        self.trace.add_step(action, "action", details)
        await self.writer.send_step(action, "action", details)

    async def on_research_step(self, step: str, details: dict = None, **kwargs) -> None:
        merged = {**(details or {}), **kwargs}
        self.trace.add_step(step, "research", merged)
        await self.writer.send_step(step, "research", merged)


def on_progress_factory(writer: SSEWriter, trace: ResearchTrace):
    def on_progress(progress):
        trace.update_progress(progress)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(writer.send_progress(progress))
        except RuntimeError:
            pass
    return on_progress


@router.post("/research")
async def research(req: ResearchRequest):
    from gpt_researcher import GPTResearcher

    config = build_research_config(
        fast_model=req.fast_model,
        smart_model=req.smart_model,
        strategic_model=req.strategic_model,
        depth=req.depth,
        breadth=req.breadth,
        reasoning_effort=req.reasoning_effort,
    )
    config_path = write_research_config(config)
    run_id = uuid.uuid4().hex[:12]
    trace = ResearchTrace(query=req.query, run_id=run_id)
    queue: asyncio.Queue = asyncio.Queue()
    writer = SSEWriter(queue)

    log_handler = TraceLogHandler(writer, trace)

    async def event_generator() -> AsyncGenerator[dict, None]:
        report: str | None = None
        sources: list[str] = []
        costs: float = 0.0

        await writer.send_step("connected", "research", {"query": req.query})

        async def run_research():
            nonlocal report, sources, costs
            try:
                import logging
                _log = logging.getLogger("research_route")
                _log.info("[research] Waiting for _env_lock...")
                async with _env_lock:
                    _log.info("[research] _env_lock acquired")
                    saved_env: dict[str, str | None] = {}
                    env_keys = ["OLLAMA_API_BASE", "OLLAMA_BASE_URL", "REASONING_EFFORT", "SEARX_URL"]
                    for key in env_keys:
                        saved_env[key] = os.environ.get(key)

                    try:
                        os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL
                        os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL
                        os.environ["SEARX_URL"] = SEARX_URL
                        if req.reasoning_effort and req.reasoning_effort != "none":
                            os.environ["REASONING_EFFORT"] = req.reasoning_effort

                        _log.info("[research] Creating GPTResearcher...")
                        researcher = GPTResearcher(
                            query=req.query,
                            report_type=req.report_type,
                            config_path=config_path,
                            log_handler=log_handler,
                        )
                        _log.info("[research] Starting conduct_research...")
                        on_progress = on_progress_factory(writer, trace)
                        await writer.send_step("start", "research", {"query": req.query, "report_type": req.report_type})
                        await researcher.conduct_research(on_progress=on_progress)
                        _log.info("[research] conduct_research done, writing report...")
                        report = await researcher.write_report()
                        sources = researcher.get_source_urls()
                        costs = researcher.get_costs()
                        _log.info(f"[research] Report written, {len(sources)} sources, cost=${costs:.4f}")
                    finally:
                        for key, val in saved_env.items():
                            if val is None:
                                os.environ.pop(key, None)
                            else:
                                os.environ[key] = val
                        import contextlib
                        with contextlib.suppress(OSError):
                            os.unlink(config_path)

                _log.info("[research] Saving trace...")
                trace.finish()
                trace.save(TRACES_DIR)
                _log.info("[research] Sending result event...")
                await writer.send_result(report or "", sources, costs)
                _log.info("[research] Done!")
            except Exception as e:
                import traceback
                _log = logging.getLogger("research_route")
                _log.error(f"[research] FAILED: {e}")
                traceback.print_exc()
                await writer.send_error(str(e))
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_research())

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                event, data = item
                yield {"event": event, "data": data}
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    return EventSourceResponse(event_generator())


@router.get("/research-traces")
async def list_traces() -> dict[str, Any]:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    traces = []
    for f in sorted(TRACES_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            traces.append({
                "run_id": data["run_id"],
                "query": data["query"],
                "started_at": data["started_at"],
                "finished_at": data["finished_at"],
                "duration_s": data.get("duration_s"),
                "step_count": len(data.get("steps", [])),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return {"traces": traces}


@router.get("/research-traces/{run_id}")
async def get_trace(run_id: str) -> dict[str, Any]:
    path = TRACES_DIR / f"{run_id}.json"
    if not path.exists():
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": "Trace not found"})
    data = json.loads(path.read_text())
    return data