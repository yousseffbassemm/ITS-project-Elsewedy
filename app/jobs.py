"""Job registry + background processing runner.

Video processing is CPU-bound and slow (no GPU), so each job runs in a worker
thread from a small pool rather than blocking the event loop. Job state is kept
in memory and mirrored to data/jobs/{id}/job.json so a page refresh (or a
restart) can still find finished results.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pipeline.config import PipelineConfig
from pipeline.process_video import process_video

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "jobs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Job ids are used as directory names; keep them to characters that cannot
# escape DATA_DIR or mean anything special to the filesystem.
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")

# One worker: on a CPU-only box, running two heavy YOLO jobs at once just makes
# both slower. Bump if you later add a GPU.
_executor = ThreadPoolExecutor(max_workers=1)


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"          # queued | processing | done | error
    progress: float = 0.0
    message: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None

    @property
    def dir(self) -> Path:
        return DATA_DIR / self.id

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        # Written from the worker thread while the API may be reading it, and
        # rewritten on every progress tick. A plain write_text truncates first,
        # so a poll landing mid-write sees invalid JSON. Write then rename —
        # os.replace is atomic, so a reader sees either the old or new file.
        target = self.dir / "job.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        os.replace(tmp, target)


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, filename: str) -> Job:
        job = Job(id=job_id, filename=filename)
        with self._lock:
            self._jobs[job_id] = job
        job.save()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        # job_id lands in a filesystem path. Starlette already refuses encoded
        # traversal in a path parameter, but this class of bug is not worth
        # delegating to framework behaviour — allow only safe id characters.
        if not _SAFE_ID.fullmatch(job_id or ""):
            return None
        with self._lock:
            job = self._jobs.get(job_id)
        if job:
            return job
        # Fall back to disk (e.g. after a restart).
        jf = DATA_DIR / job_id / "job.json"
        if jf.exists():
            try:
                data = json.loads(jf.read_text())
            except (ValueError, OSError):
                data = None
            if isinstance(data, dict):
                # Only pass keys that are actually present, so a file written by
                # an older schema falls back to the dataclass defaults instead of
                # setting status=None — which would leave the UI polling forever.
                fields = {k: v for k, v in data.items()
                          if k in Job.__dataclass_fields__ and v is not None}
                fields["id"] = job_id
                fields.setdefault("filename", job_id)
                job = Job(**fields)
                with self._lock:
                    self._jobs[job_id] = job
                return job
        # Also surface jobs produced directly by the pipeline CLI (no job.json,
        # but a finished analytics.json): treat them as completed.
        if (DATA_DIR / job_id / "analytics.json").exists():
            job = Job(id=job_id, filename=job_id, status="done", progress=100.0,
                      message="done")
            with self._lock:
                self._jobs[job_id] = job
            return job
        return None

    def discard(self, job: Job) -> None:
        """Forget a job and delete its directory (used for rejected uploads)."""
        with self._lock:
            self._jobs.pop(job.id, None)
        shutil.rmtree(job.dir, ignore_errors=True)

    def _update(self, job: Job, **kw) -> None:
        for k, v in kw.items():
            setattr(job, k, v)
        job.save()

    def submit(self, job: Job, input_path: Path, cfg: PipelineConfig) -> None:
        _executor.submit(self._run, job, input_path, cfg)

    def reconcile_startup(self) -> int:
        """Mark jobs left mid-flight by a previous process as failed.

        A job.json on disk that says 'queued' or 'processing' but has no live
        worker — because the server was killed or restarted — would otherwise be
        polled by the UI forever, showing 0% queued with nothing behind it. This
        is exactly what a hard restart produces. On startup, turn those into a
        clear error the page can display and stop polling, rather than a
        permanent spinner.

        A job is only stale if it did NOT finish: presence of analytics.json
        means it completed before the restart and must be left alone.
        """
        fixed = 0
        for jf in DATA_DIR.glob("*/job.json"):
            try:
                data = json.loads(jf.read_text())
            except (ValueError, OSError):
                continue
            if data.get("status") in ("queued", "processing") and \
                    not (jf.parent / "analytics.json").exists():
                data["status"] = "error"
                data["message"] = "interrupted"
                data["error"] = ("This job was interrupted when the server "
                                 "restarted. Please upload the video again.")
                tmp = jf.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(data, indent=2))
                os.replace(tmp, jf)
                fixed += 1
        return fixed

    def _run(self, job: Job, input_path: Path, cfg: PipelineConfig) -> None:
        try:
            self._update(job, status="processing", progress=1.0, message="loading model")

            def cb(pct: float, msg: str):
                self._update(job, progress=round(pct, 1), message=msg)

            process_video(str(input_path), str(job.dir), cfg, cb)
            self._update(job, status="done", progress=100.0, message="done")
        except BaseException as exc:  # noqa: BLE001
            # BaseException, not Exception: a worker killed by KeyboardInterrupt
            # or a MemoryError would otherwise leave the job on disk saying
            # "processing" with nothing behind it, and the dashboard would poll
            # it forever — the same permanent spinner reconcile_startup exists
            # to clean up, but for a server that never restarts.
            self._update(
                job, status="error", message="failed",
                error=f"{exc}\n{traceback.format_exc()[-1500:]}",
            )
            raise


store = JobStore()
