from __future__ import annotations

import logging
import threading
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from media_tools.web import storage

logger = logging.getLogger(__name__)

JobRunner = Callable[["Job"], dict[str, Any]]


@dataclass
class Job:
    id: str
    kind: str
    state: str = "pending"
    progress: float | None = None
    message: str = "En cola"
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "result": self.result,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class JobManager:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="media-tools",
        )
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, kind: str) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind)
        storage.create_job_dir(job.id)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def submit(self, job: Job, runner: JobRunner) -> None:
        self._executor.submit(self._run, job, runner)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        job = self.get(job_id)
        if job is None:
            return None
        with self._lock:
            return job.snapshot()

    def update(
        self,
        job: Job,
        *,
        state: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            if state is not None:
                job.state = state
            if progress is not None:
                job.progress = max(0.0, min(1.0, progress))
            if message is not None:
                job.message = message
            if error is not None:
                job.error = error
            if result is not None:
                job.result = result
            job.updated_at = datetime.now(UTC)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            existed = self._jobs.pop(job_id, None) is not None
        storage.remove_job_dir(job_id)
        return existed

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job: Job, runner: JobRunner) -> None:
        self.update(job, state="running", progress=None, message="Procesando...")
        try:
            result = runner(job)
        except Exception as exc:
            logger.warning(
                "El trabajo %s (%s) falló: %s\n%s",
                job.id,
                job.kind,
                exc,
                traceback.format_exc(),
            )
            self.update(
                job,
                state="error",
                message="El trabajo falló",
                error=str(exc) or exc.__class__.__name__,
            )
        else:
            self.update(
                job,
                state="done",
                progress=1.0,
                message="Completado",
                result=result,
            )
