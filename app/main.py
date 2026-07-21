"""FastAPI app: upload a street video, process it, serve results.

Endpoints
    GET  /                     -> upload page
    GET  /results?job=<id>     -> results dashboard page
    POST /api/upload           -> accept video, start processing, return job id
    GET  /api/status/{id}      -> job status + progress
    GET  /api/results/{id}     -> analytics JSON (when done)
    GET  /api/video/{id}       -> annotated MP4 (range requests supported)
    GET  /api/original/{id}    -> original uploaded MP4
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline.config import PipelineConfig
from .jobs import store

# Load .env before any os.getenv() below. Real environment variables already set
# in the shell win over the file, so the README's PowerShell overrides still work.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

# Cache-busting token, refreshed each server start, injected into asset URLs so a
# browser never serves a stale theme.css / app.js after a redeploy.
VERSION = str(int(time.time()))

BASE = Path(__file__).resolve().parent.parent
WEB = BASE / "web"
DATA = BASE / "data" / "jobs"

MAX_UPLOAD_MB = int(os.getenv("ITS_MAX_UPLOAD_MB", "500"))
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
VIDEO_MIME = {
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska", ".webm": "video/webm",
}

app = FastAPI(title="ITS Traffic Analytics — Elsewedy")


@app.middleware("http")
async def no_cache(request: Request, call_next):
    resp = await call_next(request)
    # Pages and JSON must never be stale. Videos must NOT be no-store: the
    # player issues range requests while seeking, and forbidding cache makes the
    # browser re-fetch a multi-megabyte file on every scrub. Static assets are
    # already cache-busted by the ?v= token.
    if request.url.path.startswith(("/api/video/", "/api/original/", "/static/")):
        resp.headers.setdefault("Cache-Control", "private, max-age=3600")
    else:
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


def _flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


def _cfg() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.model = os.getenv("ITS_MODEL", cfg.model)
    cfg.imgsz = int(os.getenv("ITS_IMGSZ", cfg.imgsz))
    cfg.frame_stride = max(int(os.getenv("ITS_FRAME_STRIDE", cfg.frame_stride)), 1)
    cfg.conf = float(os.getenv("ITS_CONF", cfg.conf))
    cfg.stable_ids = _flag("ITS_STABLE_IDS", cfg.stable_ids)
    cfg.roi_gated_tracking = _flag("ITS_ROI_GATED", cfg.roi_gated_tracking)
    cfg.plates = _flag("ITS_PLATES", cfg.plates)
    cfg.plate_model = os.getenv("ITS_PLATE_MODEL", cfg.plate_model)
    cfg.plate_ocr_model = os.getenv("ITS_PLATE_OCR_MODEL", cfg.plate_ocr_model)
    return cfg


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text(encoding="utf-8").replace("__V__", VERSION)


@app.get("/results", response_class=HTMLResponse)
def results_page():
    return (WEB / "results.html").read_text(encoding="utf-8").replace("__V__", VERSION)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")

    job_id = uuid.uuid4().hex[:12]
    job = store.create(job_id, file.filename or f"{job_id}{ext}")
    input_path = job.dir / f"input{ext}"

    size = 0
    limit = MAX_UPLOAD_MB * 1024 * 1024
    try:
        with open(input_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB limit")
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "Uploaded file is empty")
    except BaseException:
        # Rejected or aborted upload: drop the half-written file and the job dir
        # so data/jobs/ doesn't fill with orphaned 'queued' jobs.
        store.discard(job)
        raise

    store.submit(job, input_path, _cfg())
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {
        "id": job.id, "status": job.status, "progress": job.progress,
        "message": job.message, "filename": job.filename, "error": job.error,
    }


@app.get("/api/results/{job_id}")
def results(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    af = job.dir / "analytics.json"
    if not af.exists():
        raise HTTPException(409, f"not ready (status: {job.status})")
    return JSONResponse(content=json.loads(af.read_text()))


@app.get("/api/video/{job_id}")
def video(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    vf = job.dir / "annotated.mp4"
    if not vf.exists():
        raise HTTPException(404, "annotated video not ready")
    # No `filename=` — that sets Content-Disposition: attachment, which is wrong
    # for a <video> element playing the file inline.
    return FileResponse(str(vf), media_type="video/mp4")


@app.get("/api/original/{job_id}")
def original(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    for f in job.dir.glob("input.*"):
        return FileResponse(str(f), media_type=VIDEO_MIME.get(f.suffix.lower(), "video/mp4"))
    # Jobs produced by the CLI have no uploaded copy — fall back to the source the
    # pipeline recorded, but only if it is still inside the project directory, so
    # a hand-edited analytics.json cannot turn this into an arbitrary file read.
    src = _recorded_source(job.dir)
    if src is not None:
        return FileResponse(str(src),
                            media_type=VIDEO_MIME.get(src.suffix.lower(), "video/mp4"))
    raise HTTPException(404, "original not available for this job")


def _recorded_source(job_dir: Path) -> Path | None:
    af = job_dir / "analytics.json"
    if not af.exists():
        return None
    try:
        raw = json.loads(af.read_text()).get("video", {}).get("source_path")
    except (ValueError, OSError):
        return None
    if not raw:
        return None
    try:
        p = Path(raw).resolve()
        p.relative_to(BASE)          # must live under the project root
    except (ValueError, OSError):
        return None
    return p if p.is_file() and p.suffix.lower() in ALLOWED_EXT else None


# Static assets (theme.css, app.js, logo, ...) under /static
app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
