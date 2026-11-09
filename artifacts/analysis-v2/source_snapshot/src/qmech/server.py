from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .frontends import parse_text
from .runtime import environment_report
from .schema import DesignSpec, PlannerSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GenerateRequest(BaseModel):
    spec: DesignSpec
    settings: PlannerSettings = Field(default_factory=PlannerSettings)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def create_app(output: Path | str = "artifacts/runs", web_root: Path | None = None) -> FastAPI:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qmech-generator")

    @asynccontextmanager
    async def lifespan(app):
        yield
        executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="QMech", description="Measured quantum structure planning research prototype",
                  lifespan=lifespan)

    @app.middleware("http")
    async def cache_policy(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.endswith(".html") or request.url.path.startswith("/api/"):
            # Keep updated controls and polling state consistent across local rebuilds.
            response.headers["Cache-Control"] = "no-store"
        return response
    lock = Lock()
    jobs: dict[str, dict] = {}
    runtime = environment_report()

    def run_job(id: str, request: GenerateRequest):
        from .pipeline import generate
        with lock:
            jobs[id]["status"] = "running"
        try:
            summary = generate(request.spec, request.settings, output/id)
            with lock:
                jobs[id] = {"id": id, "status": "completed", "summary": summary}
        except Exception as error:  # noqa: BLE001 -- persist errors for the UI; never turn errors into success.
            with lock:
                jobs[id] = {"id": id, "status": "failed", "error": f"{type(error).__name__}: {error}"}

    def resolve(id: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", id):
            raise HTTPException(404, "Unknown run")
        directory = PROJECT_ROOT/"artifacts"/"examples"/id[8:] if id.startswith("example-") else output/id
        if not directory.is_dir():
            raise HTTPException(404, "Unknown run")
        return directory

    @app.get("/api/status")
    def status():
        return {"runtime": runtime, "prototype": "bounded mechanical grammar",
                "hardware_verified": False, "quantum_speedup_verified": False,
                "image_frontend_available": False,
                "noise_model": "generic; use QMECH_DEVICE_PROFILE for explicit device parameters"}

    @app.post("/api/parse")
    def parse(request: TextRequest):
        try:
            spec, warnings = parse_text(request.text)
            return {"spec": spec.model_dump(), "warnings": warnings}
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/generate", status_code=202)
    def submit(request: GenerateRequest):
        with lock:
            active = sum(job["status"] in {"queued", "running"} for job in jobs.values())
            if active >= 4:
                raise HTTPException(429, "Generation queue is full")
            id = uuid4().hex[:12]
            jobs[id] = {"id": id, "status": "queued"}
        executor.submit(run_job, id, request)
        return {"id": id, "status": "queued"}

    @app.get("/api/jobs/{id}")
    def job(id: str):
        with lock:
            if id not in jobs:
                raise HTTPException(404, "Unknown job")
            return jobs[id].copy()

    @app.get("/api/runs")
    def runs():
        result = []
        for family in ["drive", "clamp"]:
            path = PROJECT_ROOT/"artifacts"/"examples"/family/"run.json"
            if path.exists():
                result.append({"id": f"example-{family}", "summary": json.loads(path.read_text())})
        for path in sorted(output.glob("*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
            summary = json.loads(path.read_text())
            if summary["status"] == "generated":
                result.append({"id": path.parent.name, "summary": summary})
        return result

    @app.get("/api/runs/{id}/{filename}")
    def file(id: str, filename: str):
        allowed = {"run.json", "assembly.step", "assembly.glb", "structure.json",
                   "trace.json", "design.json", "problem.json", "disassembly.json"}
        if filename not in allowed:
            raise HTTPException(404, "Unknown artifact")
        path = resolve(id)/filename
        if not path.is_file():
            raise HTTPException(404, "Artifact not yet available")
        return FileResponse(path)

    @app.get("/api/experiments")
    def experiments():
        return [{"name": path.parent.name, "summary": json.loads(path.read_text())}
                for path in sorted((PROJECT_ROOT/"artifacts"/"benchmarks").glob("*/summary.json"))]

    web_root = web_root or PROJECT_ROOT/"web"/"dist"
    if web_root.exists():
        app.mount("/", StaticFiles(directory=web_root, html=True), name="viewer")
    return app

