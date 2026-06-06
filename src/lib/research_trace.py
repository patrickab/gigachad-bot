import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class TraceStep:
    step: str
    event_type: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceProgress:
    current_depth: int
    total_depth: int
    current_breadth: int
    total_breadth: int
    current_query: str | None = None
    completed_queries: int = 0
    total_queries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResearchTrace:
    def __init__(self, query: str, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.query = query
        self.steps: list[TraceStep] = []
        self.progress: TraceProgress | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None

    def add_step(self, step: str, event_type: str, details: dict[str, Any] | None = None) -> TraceStep:
        trace_step = TraceStep(step=step, event_type=event_type, details=details or {})
        self.steps.append(trace_step)
        return trace_step

    def update_progress(self, progress) -> None:
        self.progress = TraceProgress(
            current_depth=getattr(progress, "current_depth", 0),
            total_depth=getattr(progress, "total_depth", 0),
            current_breadth=getattr(progress, "current_breadth", 0),
            total_breadth=getattr(progress, "total_breadth", 0),
            current_query=getattr(progress, "current_query", None),
            completed_queries=getattr(progress, "completed_queries", 0),
            total_queries=getattr(progress, "total_queries", 0),
        )

    def finish(self) -> None:
        self.finished_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.finished_at - self.started_at, 2) if self.finished_at else None,
            "steps": [s.to_dict() for s in self.steps],
            "progress": self.progress.to_dict() if self.progress else None,
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: Path) -> "ResearchTrace":
        data = json.loads(path.read_text())
        trace = cls(query=data["query"], run_id=data["run_id"])
        trace.started_at = data["started_at"]
        trace.finished_at = data["finished_at"]
        for s in data["steps"]:
            trace.steps.append(TraceStep(**s))
        if data.get("progress"):
            trace.progress = TraceProgress(**data["progress"])
        return trace
