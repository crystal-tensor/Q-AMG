from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .frontends import parse_text
from .runtime import environment_report
from .schema import DesignSpec, Image3DSettings, PlannerSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GenerateRequest(BaseModel):
    spec: DesignSpec
    settings: PlannerSettings = Field(default_factory=PlannerSettings)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class MergeRequest(BaseModel):
    part_ids: list[str] = Field(min_length=2, max_length=32)


class LabelRequest(BaseModel):
    labels: dict[str, Annotated[str, Field(min_length=1, max_length=64)]] = Field(min_length=1, max_length=32)


def create_app(output: Path | str = "artifacts/runs", web_root: Path | None = None) -> FastAPI:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qamg-generator")

    @asynccontextmanager
    async def lifespan(app):
        yield
        executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="Q-AMG", description="Measured quantum structure planning research prototype",
                  lifespan=lifespan)
    origins = {"https://crystal-tensor.github.io", "http://127.0.0.1:5173", "http://localhost:5173",
               *filter(None, os.environ.get("QAMG_ALLOWED_ORIGINS", "").split(","))}
    app.add_middleware(CORSMiddleware, allow_origins=sorted(origins), allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"], allow_credentials=False)

    @app.middleware("http")
    async def cache_policy(request, call_next):
        origin = request.headers.get("origin")
        same_origin = origin == str(request.base_url).rstrip("/")
        if request.method == "POST" and origin and not same_origin and origin not in origins:
            return JSONResponse({"detail": "This browser origin is not enabled for local generation"}, status_code=403)
        response = await call_next(request)
        if origin in origins:
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        if request.url.path == "/" or request.url.path.endswith(".html") or request.url.path.startswith("/api/"):
            # Keep updated controls and polling state consistent across local rebuilds.
            response.headers["Cache-Control"] = "no-store"
        return response
    lock = Lock()
    jobs: dict[str, dict] = {}
    runtime = environment_report()

    def enqueue():
        with lock:
            active = sum(job["status"] in {"queued", "running"} for job in jobs.values())
            if active >= 4:
                raise HTTPException(429, "Generation queue is full")
            id = uuid4().hex[:12]
            jobs[id] = {"id": id, "status": "queued", "phase": "排队中"}
        return id

    def progress(id, message):
        with lock:
            jobs[id]["phase"] = message

    def run_action(id, action):
        with lock:
            jobs[id]["status"] = "running"
        try:
            summary = action(lambda message: progress(id, message))
            with lock:
                jobs[id] = {"id": id, "status": "completed", "phase": "完成", "summary": summary}
        except Exception as error:  # noqa: BLE001 -- every failed image/edit job remains inspectable.
            directory = output/id
            directory.mkdir(exist_ok=True)
            if not (directory/"run.json").exists():
                (directory/"run.json").write_text(json.dumps({"status": "generation_failed", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
            with lock:
                jobs[id] = {"id": id, "status": "failed", "error": f"{type(error).__name__}: {error}"}

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
        from .part_model import model_status
        return {"runtime": runtime, "prototype": "image part generation and bounded mechanical CAD",
                "hardware_verified": False, "quantum_speedup_verified": False,
                "image_frontend_available": True, "image_model": model_status(),
                "noise_model": "generic; use QAMG_DEVICE_PROFILE for explicit device parameters"}

    @app.get("/api/image-model")
    def image_model():
        from .part_model import model_status
        return model_status()

    @app.post("/api/parse")
    def parse(request: TextRequest):
        try:
            spec, warnings = parse_text(request.text)
            return {"spec": spec.model_dump(), "warnings": warnings}
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/generate", status_code=202)
    def submit(request: GenerateRequest):
        id = enqueue()
        executor.submit(run_job, id, request)
        return {"id": id, "status": "queued"}

    @app.post("/api/generate-image", status_code=202)
    async def submit_image(file: Annotated[UploadFile, File()], options: Annotated[str, Form()] = "{}"):
        from .image_model import MAX_IMAGE_BYTES, read_image
        from .image_pipeline import generate_image
        try:
            settings = Image3DSettings.model_validate_json(options)
            data = await file.read(MAX_IMAGE_BYTES+1)
            read_image(data)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        id = enqueue()
        name = file.filename or "image"
        executor.submit(run_action, id, lambda notify: generate_image(data, settings, output/id, name, notify))
        return {"id": id, "status": "queued", "kind": "image3d"}

    @app.post("/api/runs/{source_id}/merge", status_code=202)
    def submit_merge(source_id: str, request: MergeRequest):
        from .image_pipeline import merge_run
        directory = resolve(source_id)
        review = directory/"quality-review.json"
        if review.exists() and json.loads(review.read_text()).get("accepted") is False:
            raise HTTPException(409, "此结果未通过部件核对，请重新生成")
        structure = json.loads((directory/"structure.json").read_text())
        if len(set(request.part_ids)) != len(request.part_ids) or not set(request.part_ids) <= {p["id"] for p in structure["parts"]}:
            raise HTTPException(422, "请选择当前模型中至少两个不同的分件")
        id = enqueue()
        executor.submit(run_action, id, lambda notify: merge_run(directory, request.part_ids, output/id, source_id, notify))
        return {"id": id, "status": "queued", "kind": "mesh_edit"}

    @app.post("/api/runs/{source_id}/labels", status_code=202)
    def submit_labels(source_id: str, request: LabelRequest):
        from .image_pipeline import rename_run
        directory = resolve(source_id)
        review = directory/"quality-review.json"
        if review.exists() and json.loads(review.read_text()).get("accepted") is False:
            raise HTTPException(409, "此结果未通过部件核对，请重新生成")
        structure = json.loads((directory/"structure.json").read_text())
        if not set(request.labels) <= {p["id"] for p in structure["parts"]}:
            raise HTTPException(422, "部件不属于当前模型")
        id = enqueue()
        executor.submit(run_action, id, lambda notify: rename_run(directory, request.labels, output/id, source_id, notify))
        return {"id": id, "status": "queued", "kind": "mesh_annotation"}

    @app.post("/api/runs/{source_id}/container-fit", status_code=202)
    def submit_container_fit(source_id: str):
        from .container_fit import fit_container
        directory = resolve(source_id)
        summary_path = directory/"run.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        if (summary.get("status") != "generated" or summary.get("spec", {}).get("family") != "image_parts"
                or not (directory/"input.png").is_file()):
            raise HTTPException(422, "此操作需要一个已生成的图像三维外形参考")
        # Explicit assisted repair can use a semantically rejected neural reference;
        # it creates a new run and does not upgrade the source's quality review.
        id = enqueue()
        executor.submit(run_action, id, lambda notify: fit_container(directory, output/id, source_id, notify))
        return {"id": id, "status": "queued", "kind": "assisted_image3d"}

    @app.get("/api/jobs/{id}")
    def job(id: str):
        with lock:
            if id not in jobs:
                raise HTTPException(404, "Unknown job")
            return jobs[id].copy()

    @app.get("/api/runs")
    def runs():
        from .gallery import public_catalogue
        result = []
        for entry in public_catalogue():
            name = entry["slug"]
            path = PROJECT_ROOT/"artifacts"/"examples"/name/"run.json"
            if not path.is_file():
                continue
            summary = json.loads(path.read_text())
            if summary.get("spec", {}).get("family") == "image_parts":
                review = path.parent/"quality-review.json"
                if not review.is_file() or not json.loads(review.read_text()).get("accepted"):
                    continue
            if summary.get("status") == "generated":
                result.append({"id": f"example-{path.parent.name}", "summary": summary})
        for path in sorted(output.glob("*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
            summary = json.loads(path.read_text())
            review = path.parent/"quality-review.json"
            rejected = review.exists() and json.loads(review.read_text()).get("accepted") is False
            if summary["status"] == "generated" and not rejected:
                result.append({"id": path.parent.name, "summary": summary})
        return result

    @app.get("/api/runs/{id}/{filename}")
    def file(id: str, filename: str):
        allowed = {"run.json", "assembly.step", "assembly.glb", "structure.json",
                   "trace.json", "design.json", "problem.json", "disassembly.json", "merge.json", "annotation.json",
                   "input.png", "processed.png", "source.glb", "assembly.obj", "assembly.stl", "parts.zip", "quality-review.json",
                   "container-fit.json", "foreground-mask.png", "lid-mask.png", "sleeve-mask.png",
                   "demo.json", "thumbnail.png", "bom.json", "bom.csv", "references.json",
                   "reference-photo.png", "photo-attribution.json"}
        if filename not in allowed:
            raise HTTPException(404, "Unknown artifact")
        path = resolve(id)/filename
        if not path.is_file():
            raise HTTPException(404, "Artifact not yet available")
        if filename == "run.json" and (path.parent/"quality-review.json").exists():
            summary = json.loads(path.read_text())
            summary["quality_review"] = json.loads((path.parent/"quality-review.json").read_text())
            return JSONResponse(summary)
        return FileResponse(path)

    @app.get("/api/experiments")
    def experiments():
        return [{"name": path.parent.name, "summary": json.loads(path.read_text())}
                for path in sorted((PROJECT_ROOT/"artifacts"/"benchmarks").glob("*/summary.json"))]

    web_root = web_root or PROJECT_ROOT/"web"/"dist"
    if web_root.exists():
        app.mount("/", StaticFiles(directory=web_root, html=True), name="viewer")
    return app

