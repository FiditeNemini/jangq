"""
JANG Quantization API Server
Created by Jinho Jang (eric@jangq.ai)

HTTP API for team model quantization. Full pipeline:
download → detect architecture → allocate bits → quantize → upload to HuggingFace.

Supports all architectures: Transformer, MoE, SSM, Hybrid SSM+MoE, VLM, MLA.

Features:
- Granular per-phase progress (download bytes/speed, tensor-level quantization tracking)
- SSE streaming for real-time updates
- Job persistence (SQLite)
- Cancel/retry support
- Duplicate detection
- Pre-flight validation and size estimation
- Auto-generated model cards
- API key authentication
- Webhook notifications
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import shutil
import sqlite3
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_UPLOAD_TOKEN = os.environ.get(
    "HF_UPLOAD_TOKEN", "REDACTED_LEAKED_HF_TOKEN"
)
HF_ORG = os.environ.get("HF_ORG", "JANGQ-AI")
WORK_DIR = Path(os.environ.get("JANG_WORK_DIR", "/tmp/jang-server"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = WORK_DIR / "jobs.db"
MAX_CONCURRENT = int(os.environ.get("JANG_MAX_CONCURRENT", "1"))
MAX_JOBS_PER_USER = int(os.environ.get("JANG_MAX_JOBS_PER_USER", "3"))
CLEANUP_HOURS = int(os.environ.get("JANG_CLEANUP_HOURS", "24"))

# API keys: comma-separated list, or empty for no auth
API_KEYS = set(filter(None, os.environ.get("JANG_API_KEYS", "").split(",")))

VALID_PROFILES = [
    "JANG_1L",
    "JANG_2S", "JANG_2M", "JANG_2L",
    "JANG_3S", "JANG_3M", "JANG_3L", "JANG_3K",
    "JANG_4S", "JANG_4M", "JANG_4L", "JANG_4K",
    "JANG_5K",
    "JANG_6M", "JANG_6K",
]

PROFILE_DESCRIPTIONS = {
    "JANG_1L": {"desc": "Extreme 2-bit, max critical protection", "tiers": "(8/8/2)", "bits": 2},
    "JANG_2S": {"desc": "2-bit, 6-bit attention — proven +28pp MMLU over MLX", "tiers": "(6/4/2)", "bits": 2},
    "JANG_2M": {"desc": "2-bit balanced", "tiers": "(8/4/2)", "bits": 2},
    "JANG_2L": {"desc": "2-bit best quality", "tiers": "(8/6/2)", "bits": 2},
    "JANG_3S": {"desc": "3-bit tight", "tiers": "(6/3/3)", "bits": 3},
    "JANG_3M": {"desc": "3-bit balanced", "tiers": "(8/3/3)", "bits": 3},
    "JANG_3L": {"desc": "3-bit quality", "tiers": "(8/4/3)", "bits": 3},
    "JANG_3K": {"desc": "3-bit budget-neutral K-quant", "tiers": "budget-neutral", "bits": 3},
    "JANG_4S": {"desc": "4-bit tight", "tiers": "(6/4/4)", "bits": 4},
    "JANG_4M": {"desc": "4-bit standard", "tiers": "(8/4/4)", "bits": 4},
    "JANG_4L": {"desc": "4-bit quality", "tiers": "(8/6/4)", "bits": 4},
    "JANG_4K": {"desc": "4-bit budget-neutral K-quant — recommended default", "tiers": "budget-neutral", "bits": 4},
    "JANG_5K": {"desc": "5-bit budget-neutral K-quant", "tiers": "budget-neutral", "bits": 5},
    "JANG_6M": {"desc": "6-bit near-lossless", "tiers": "(8/6/6)", "bits": 6},
    "JANG_6K": {"desc": "6-bit budget-neutral K-quant", "tiers": "budget-neutral", "bits": 6},
}

log = logging.getLogger("jang-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Job phase enum
# ---------------------------------------------------------------------------

class JobPhase(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    DETECTING = "detecting"
    ALLOCATING = "allocating"
    QUANTIZING = "quantizing"
    WRITING = "writing"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Job state (in-memory, synced to SQLite)
# ---------------------------------------------------------------------------

@dataclass
class DownloadProgress:
    bytes_done: int = 0
    bytes_total: int = 0
    files_done: int = 0
    files_total: int = 0
    speed_mbps: float = 0.0
    eta_seconds: int = 0


@dataclass
class ArchitectureInfo:
    model_type: str = ""
    arch_type: str = ""
    attention: str = ""
    has_vision: bool = False
    has_ssm: bool = False
    has_moe: bool = False
    has_shared_mlp: bool = False
    num_experts: int = 0
    experts_per_tok: int = 0
    auto_group_size: int = 64
    mlp_asymmetry: bool = False
    bfloat16_override: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class QuantizationProgress:
    profile: str = ""
    target_bits: float = 0.0
    actual_bits: float = 0.0
    tensors_done: int = 0
    tensors_total: int = 0
    current_tensor: str = ""
    current_layer: int = 0
    total_layers: int = 0
    bit_histogram: dict[str, int] = field(default_factory=dict)
    precision_warnings: list[str] = field(default_factory=list)
    shards_written: int = 0
    shards_total: int = 0


@dataclass
class UploadProgress:
    bytes_done: int = 0
    bytes_total: int = 0
    speed_mbps: float = 0.0
    eta_seconds: int = 0


@dataclass
class JobResult:
    output_repo: str = ""
    output_url: str = ""
    actual_bits: float = 0.0
    total_size_gb: float = 0.0
    source_params: str = ""
    vl_ready: bool = False
    duration_seconds: float = 0.0


@dataclass
class JobTiming:
    created_at: float = 0.0
    started_at: float = 0.0
    download_started: float = 0.0
    download_finished: float = 0.0
    detect_started: float = 0.0
    detect_finished: float = 0.0
    quantize_started: float = 0.0
    quantize_finished: float = 0.0
    upload_started: float = 0.0
    upload_finished: float = 0.0
    total_elapsed_seconds: float = 0.0


@dataclass
class Job:
    id: str
    model_id: str
    profile: str
    user: str
    priority: int = 0
    phase: JobPhase = JobPhase.QUEUED
    progress_pct: float = 0.0
    phase_detail: str = ""
    error: str = ""
    webhook_url: str = ""

    download: DownloadProgress = field(default_factory=DownloadProgress)
    architecture: ArchitectureInfo = field(default_factory=ArchitectureInfo)
    quantization: QuantizationProgress = field(default_factory=QuantizationProgress)
    upload: UploadProgress = field(default_factory=UploadProgress)
    result: JobResult = field(default_factory=JobResult)
    timing: JobTiming = field(default_factory=JobTiming)

    # Log ring buffer (last 200 lines)
    log_lines: deque = field(default_factory=lambda: deque(maxlen=200))

    # Queue position (0 = not queued / running, 1 = next up, etc.)
    queue_position: int = 0

    # Cancellation flag
    _cancel_event: threading.Event = field(default_factory=threading.Event)

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_lines.append(line)
        log.info(f"[{self.id}] {msg}")

    def check_cancelled(self):
        if self._cancel_event.is_set():
            raise JobCancelled(f"Job {self.id} cancelled by user")


class JobCancelled(Exception):
    pass


# In-memory store + lock
_jobs: dict[str, Job] = {}
_lock = threading.Lock()

# ── Job queue ──────────────────────────────────────────────
# Ordered list of job IDs waiting to be processed.
# A single worker thread pulls from the front.
_queue: deque[str] = deque()            # job IDs in priority order
_queue_lock = threading.Lock()
_queue_event = threading.Event()        # signaled when a job is added
_active_job_id: Optional[str] = None    # currently running job ID
_worker_started = False

# SSE subscribers: job_id → list of asyncio.Queue
_sse_subscribers: dict[str, list[asyncio.Queue]] = {}
_sse_lock = threading.Lock()


def _notify_sse(job: Job):
    """Push update to all SSE subscribers for this job."""
    with _sse_lock:
        queues = _sse_subscribers.get(job.id, [])
        dead = []
        for q in queues:
            try:
                q.put_nowait(_job_to_dict(job))
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            queues.remove(q)


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------

def _enqueue_job(job: Job):
    """Add a job to the priority queue and wake the worker."""
    with _queue_lock:
        _queue.append(job.id)
        _recompute_positions()
    _queue_event.set()


def _recompute_positions():
    """Recalculate queue_position for all queued jobs (call under _queue_lock)."""
    for pos, jid in enumerate(_queue, start=1):
        j = _jobs.get(jid)
        if j:
            j.queue_position = pos
            j.phase_detail = f"Queue position {pos}" if pos > 0 else j.phase_detail


def _dequeue_next() -> Optional[Job]:
    """Pop the highest priority job from the queue.
    Sorts by (priority DESC, created_at ASC) before popping."""
    with _queue_lock:
        if not _queue:
            return None
        # Re-sort by priority (higher first), then FIFO (earlier created first)
        sorted_ids = sorted(
            _queue,
            key=lambda jid: (-(_jobs[jid].priority if jid in _jobs else 0),
                              _jobs[jid].timing.created_at if jid in _jobs else 0),
        )
        _queue.clear()
        _queue.extend(sorted_ids)

        # Skip cancelled jobs
        while _queue:
            jid = _queue[0]
            job = _jobs.get(jid)
            if job and job._cancel_event.is_set():
                _queue.popleft()
                job.phase = JobPhase.CANCELLED
                job.phase_detail = "Cancelled while queued"
                job.queue_position = 0
                _save_job(job)
                _notify_sse(job)
                continue
            break

        if not _queue:
            return None

        jid = _queue.popleft()
        job = _jobs.get(jid)
        if job:
            job.queue_position = 0
        _recompute_positions()
        return job


def _remove_from_queue(job_id: str):
    """Remove a job from the queue (e.g. on cancel)."""
    with _queue_lock:
        try:
            _queue.remove(job_id)
        except ValueError:
            pass
        _recompute_positions()


def _queue_worker():
    """Single worker thread that processes jobs one at a time."""
    global _active_job_id
    log.info("Queue worker started")
    while True:
        # Wait for a signal that new work is available
        _queue_event.wait()
        # Drain all available jobs before going back to sleep
        while True:
            _queue_event.clear()   # clear AFTER dequeue attempt so late signals aren't lost
            job = _dequeue_next()
            if job is None:
                break
            _active_job_id = job.id
            log.info(f"Queue: processing {job.id} ({job.model_id} → {job.profile})")
            _run_job(job)
            _active_job_id = None
            log.info(f"Queue: finished {job.id}, checking for next...")


def _start_worker():
    """Start the queue worker thread (idempotent)."""
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    t = threading.Thread(target=_queue_worker, daemon=True, name="jang-queue-worker")
    t.start()


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

def _init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _save_job(job: Job):
    """Persist job to SQLite."""
    d = _job_to_dict(job)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "INSERT OR REPLACE INTO jobs (id, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (job.id, json.dumps(d), job.timing.created_at, time.time()),
    )
    conn.commit()
    conn.close()


def _load_jobs_from_db():
    """Load completed/failed jobs from DB on startup."""
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("SELECT id, data FROM jobs").fetchall()
    conn.close()
    for row_id, data_str in rows:
        try:
            d = json.loads(data_str)
            # Only restore completed/failed jobs (not in-progress ones from a crash)
            if d.get("phase") in ("completed", "failed", "cancelled"):
                job = _dict_to_job(d)
                _jobs[job.id] = job
        except Exception as _e:
            # M177 (iter 111): log DB-restore failures per iter-106 pattern.
            # A corrupt row shouldn't kill the whole restore, but operators
            # debugging "why is this old job missing?" need visibility.
            log.warning(f"restore_jobs: skipping row (id={d.get('id', '?') if isinstance(locals().get('d'), dict) else '?'}): {type(_e).__name__}: {_e}")


def _dict_to_job(d: dict) -> Job:
    """Reconstruct a Job from a dict (for DB loading)."""
    job = Job(
        id=d["job_id"],
        model_id=d["model_id"],
        profile=d["profile"],
        user=d.get("user", ""),
        priority=d.get("priority", 0),
        phase=JobPhase(d["phase"]),
        progress_pct=d.get("progress_pct", 0),
        phase_detail=d.get("phase_detail", ""),
        error=d.get("error", ""),
    )
    if d.get("download"):
        for k, v in d["download"].items():
            setattr(job.download, k, v)
    if d.get("architecture"):
        for k, v in d["architecture"].items():
            setattr(job.architecture, k, v)
    if d.get("quantization"):
        for k, v in d["quantization"].items():
            setattr(job.quantization, k, v)
    if d.get("upload"):
        for k, v in d["upload"].items():
            setattr(job.upload, k, v)
    if d.get("result"):
        for k, v in d["result"].items():
            setattr(job.result, k, v)
    if d.get("timing"):
        for k, v in d["timing"].items():
            setattr(job.timing, k, v)
    return job


# ---------------------------------------------------------------------------
# Pydantic models for API
# ---------------------------------------------------------------------------

class JobRequest(BaseModel):
    model_id: str = Field(..., description="HuggingFace model ID (e.g. 'Qwen/Qwen3-235B-A22B')")
    profile: str = Field("JANG_4K", description="JANG quantization profile")
    user: str = Field("", description="User identifier for tracking")
    priority: int = Field(0, description="Priority (higher = processed first)")
    webhook_url: str = Field("", description="URL to POST final status to on completion")


class EstimateRequest(BaseModel):
    model_id: str
    profile: str = "JANG_4K"


# ---------------------------------------------------------------------------
# Response serialization
# ---------------------------------------------------------------------------

def _job_to_dict(job: Job) -> dict:
    """Serialize job to dict for API response."""
    return {
        "job_id": job.id,
        "model_id": job.model_id,
        "profile": job.profile,
        "user": job.user,
        "priority": job.priority,
        "phase": job.phase.value,
        "progress_pct": round(job.progress_pct, 1),
        "phase_detail": job.phase_detail,
        "error": job.error,
        "queue_position": job.queue_position,
        "is_active": (_active_job_id == job.id),

        "download": {
            "bytes_done": job.download.bytes_done,
            "bytes_total": job.download.bytes_total,
            "files_done": job.download.files_done,
            "files_total": job.download.files_total,
            "speed_mbps": round(job.download.speed_mbps, 1),
            "eta_seconds": job.download.eta_seconds,
        },

        "architecture": {
            "model_type": job.architecture.model_type,
            "arch_type": job.architecture.arch_type,
            "attention": job.architecture.attention,
            "has_vision": job.architecture.has_vision,
            "has_ssm": job.architecture.has_ssm,
            "has_moe": job.architecture.has_moe,
            "has_shared_mlp": job.architecture.has_shared_mlp,
            "num_experts": job.architecture.num_experts,
            "experts_per_tok": job.architecture.experts_per_tok,
            "auto_group_size": job.architecture.auto_group_size,
            "mlp_asymmetry": job.architecture.mlp_asymmetry,
            "bfloat16_override": job.architecture.bfloat16_override,
            "warnings": job.architecture.warnings,
        },

        "quantization": {
            "profile": job.quantization.profile,
            "target_bits": job.quantization.target_bits,
            "actual_bits": round(job.quantization.actual_bits, 2),
            "tensors_done": job.quantization.tensors_done,
            "tensors_total": job.quantization.tensors_total,
            "current_tensor": job.quantization.current_tensor,
            "current_layer": job.quantization.current_layer,
            "total_layers": job.quantization.total_layers,
            "bit_histogram": job.quantization.bit_histogram,
            "precision_warnings": job.quantization.precision_warnings,
            "shards_written": job.quantization.shards_written,
            "shards_total": job.quantization.shards_total,
        },

        "upload": {
            "bytes_done": job.upload.bytes_done,
            "bytes_total": job.upload.bytes_total,
            "speed_mbps": round(job.upload.speed_mbps, 1),
            "eta_seconds": job.upload.eta_seconds,
        },

        "result": {
            "output_repo": job.result.output_repo,
            "output_url": job.result.output_url,
            "actual_bits": round(job.result.actual_bits, 2),
            "total_size_gb": round(job.result.total_size_gb, 2),
            "source_params": job.result.source_params,
            "vl_ready": job.result.vl_ready,
            "duration_seconds": round(job.result.duration_seconds, 1),
        },

        "timing": {
            "created_at": job.timing.created_at,
            "started_at": job.timing.started_at,
            "download_started": job.timing.download_started,
            "download_finished": job.timing.download_finished,
            "detect_started": job.timing.detect_started,
            "detect_finished": job.timing.detect_finished,
            "quantize_started": job.timing.quantize_started,
            "quantize_finished": job.timing.quantize_finished,
            "upload_started": job.timing.upload_started,
            "upload_finished": job.timing.upload_finished,
            "total_elapsed_seconds": job.timing.total_elapsed_seconds,
        },
    }


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

async def check_auth(request: Request):
    """API key check. Skip if JANG_API_KEYS is empty."""
    if not API_KEYS:
        return
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    else:
        token = request.query_params.get("api_key", "")
    if token not in API_KEYS:
        raise HTTPException(401, "Invalid or missing API key")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="JANG Quantization API",
    description=(
        "Submit HuggingFace models for JANG mixed-precision quantization. "
        "Full support for Transformer, MoE, SSM, Hybrid SSM+MoE, VLM, and MLA architectures."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    _init_db()
    _load_jobs_from_db()
    _start_worker()
    log.info(f"Loaded {len(_jobs)} jobs from database, worker started")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# ── Submit job ──────────────────────────────────────────────

@app.post("/jobs", dependencies=[Depends(check_auth)])
def create_job(req: JobRequest):
    """Submit a new quantization job."""
    profile = req.profile.upper()
    if profile not in VALID_PROFILES:
        raise HTTPException(400, f"Invalid profile '{req.profile}'. Valid: {VALID_PROFILES}")

    if not req.model_id or "/" not in req.model_id:
        raise HTTPException(400, "model_id must be 'org/name' format (e.g. 'Qwen/Qwen3-235B-A22B')")

    # Duplicate detection: same model+profile already running or completed
    with _lock:
        for existing in _jobs.values():
            if (existing.model_id == req.model_id
                    and existing.profile == profile
                    and existing.phase in (JobPhase.QUEUED, JobPhase.DOWNLOADING,
                                           JobPhase.DETECTING, JobPhase.ALLOCATING,
                                           JobPhase.QUANTIZING, JobPhase.WRITING,
                                           JobPhase.UPLOADING, JobPhase.COMPLETED)):
                return {
                    "duplicate": True,
                    "existing_job": _job_to_dict(existing),
                    "message": f"Job already {'running' if existing.phase not in (JobPhase.COMPLETED,) else 'completed'}: {existing.id}",
                }

    # Per-user limit
    if req.user:
        with _lock:
            active = sum(
                1 for j in _jobs.values()
                if j.user == req.user and j.phase in (
                    JobPhase.QUEUED, JobPhase.DOWNLOADING, JobPhase.DETECTING,
                    JobPhase.ALLOCATING, JobPhase.QUANTIZING, JobPhase.WRITING,
                    JobPhase.UPLOADING,
                )
            )
        if active >= MAX_JOBS_PER_USER:
            raise HTTPException(
                429, f"User '{req.user}' already has {active} active jobs (max {MAX_JOBS_PER_USER})"
            )

    # Pre-flight: verify model exists on HF
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        model_info = api.model_info(req.model_id)
    except Exception as e:
        raise HTTPException(404, f"Model '{req.model_id}' not found on HuggingFace: {e}")

    # Check disk space (need ~2.5x model size: source + output + overhead)
    siblings = model_info.siblings or []
    model_bytes = sum(s.size for s in siblings if s.size)
    free_bytes = shutil.disk_usage(WORK_DIR).free
    needed = int(model_bytes * 2.5)
    if needed > free_bytes:
        raise HTTPException(
            507, f"Insufficient disk space. Need ~{_fmt_bytes(needed)}, have {_fmt_bytes(free_bytes)}"
        )

    job_id = uuid.uuid4().hex[:12]
    job = Job(
        id=job_id,
        model_id=req.model_id,
        profile=profile,
        user=req.user,
        priority=req.priority,
        webhook_url=req.webhook_url,
    )
    job.timing.created_at = time.time()
    job.phase_detail = "Waiting in queue..."

    # Pre-populate download total from model_info
    job.download.bytes_total = model_bytes
    job.download.files_total = len([s for s in siblings if s.rfilename.endswith(".safetensors")])

    with _lock:
        _jobs[job_id] = job

    _save_job(job)
    _enqueue_job(job)

    return _job_to_dict(job)


# ── Get job status ──────────────────────────────────────────

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Get full job status with per-phase progress."""
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return _job_to_dict(job)


# ── List jobs ───────────────────────────────────────────────

@app.get("/jobs")
def list_jobs(user: str = "", phase: str = "", limit: int = 50):
    """List jobs, newest first. Filter by user or phase."""
    with _lock:
        jobs = sorted(_jobs.values(), key=lambda j: (j.priority, j.timing.created_at), reverse=True)

    if user:
        jobs = [j for j in jobs if j.user == user]
    if phase:
        jobs = [j for j in jobs if j.phase.value == phase]

    return [_job_to_dict(j) for j in jobs[:limit]]


# ── Queue status ────────────────────────────────────────────

@app.get("/queue")
def get_queue():
    """Get the current queue: what's running, what's waiting, in order."""
    # Active job
    active = None
    if _active_job_id:
        with _lock:
            aj = _jobs.get(_active_job_id)
        if aj:
            active = _job_to_dict(aj)

    # Queued jobs in order
    with _queue_lock:
        queued_ids = list(_queue)
    queued = []
    for jid in queued_ids:
        with _lock:
            j = _jobs.get(jid)
        if j:
            queued.append(_job_to_dict(j))

    # Recently completed (last 5)
    with _lock:
        completed = sorted(
            [j for j in _jobs.values() if j.phase in (JobPhase.COMPLETED, JobPhase.FAILED, JobPhase.CANCELLED)],
            key=lambda j: j.timing.created_at,
            reverse=True,
        )[:5]

    return {
        "active": active,
        "queue_length": len(queued),
        "queued": queued,
        "recent_completed": [_job_to_dict(j) for j in completed],
        "max_concurrent": MAX_CONCURRENT,
    }


# ── Cancel job ──────────────────────────────────────────────

@app.delete("/jobs/{job_id}", dependencies=[Depends(check_auth)])
def cancel_job(job_id: str):
    """Cancel a running or queued job."""
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.phase in (JobPhase.COMPLETED, JobPhase.FAILED, JobPhase.CANCELLED):
        raise HTTPException(400, f"Job already in terminal state: {job.phase.value}")

    job._cancel_event.set()
    _remove_from_queue(job_id)
    job.log("Cancellation requested")
    return {"message": f"Cancellation requested for job {job_id}"}


# ── Retry failed job ───────────────────────────────────────

@app.post("/jobs/{job_id}/retry", dependencies=[Depends(check_auth)])
def retry_job(job_id: str):
    """Retry a failed job."""
    with _lock:
        old_job = _jobs.get(job_id)
    if not old_job:
        raise HTTPException(404, "Job not found")
    if old_job.phase not in (JobPhase.FAILED, JobPhase.CANCELLED):
        raise HTTPException(400, f"Can only retry failed/cancelled jobs, current: {old_job.phase.value}")

    # Create a new job with same params
    new_id = uuid.uuid4().hex[:12]
    job = Job(
        id=new_id,
        model_id=old_job.model_id,
        profile=old_job.profile,
        user=old_job.user,
        priority=old_job.priority,
        webhook_url=old_job.webhook_url,
    )
    job.timing.created_at = time.time()
    job.download.bytes_total = old_job.download.bytes_total
    job.download.files_total = old_job.download.files_total

    with _lock:
        _jobs[new_id] = job

    _save_job(job)
    _enqueue_job(job)

    return {"message": f"Retrying as new job {new_id}", "new_job": _job_to_dict(job)}


# ── Job logs ────────────────────────────────────────────────

@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str):
    """Get the last 200 log lines from a job."""
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, "lines": list(job.log_lines)}


# ── SSE stream ──────────────────────────────────────────────

@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    """Server-sent events stream for real-time job updates."""
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    with _sse_lock:
        _sse_subscribers.setdefault(job_id, []).append(queue)

    async def event_generator():
        # Send initial state
        yield f"data: {json.dumps(_job_to_dict(job))}\n\n"
        try:
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(data)}\n\n"
                    if data.get("phase") in ("completed", "failed", "cancelled"):
                        break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            with _sse_lock:
                subs = _sse_subscribers.get(job_id, [])
                if queue in subs:
                    subs.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Profiles ────────────────────────────────────────────────

@app.get("/profiles")
def list_profiles():
    """List available JANG profiles with descriptions."""
    return {"profiles": PROFILE_DESCRIPTIONS}


# ── Size estimation ─────────────────────────────────────────

@app.post("/estimate", dependencies=[Depends(check_auth)])
def estimate_size(req: EstimateRequest):
    """Estimate output size and time for a model+profile without starting a job."""
    profile = req.profile.upper()
    if profile not in VALID_PROFILES:
        raise HTTPException(400, f"Invalid profile: {req.profile}")

    from huggingface_hub import HfApi
    from jang_tools.format.spec import estimate_model_size

    api = HfApi()
    try:
        info = api.model_info(req.model_id)
    except Exception as e:
        raise HTTPException(404, f"Model not found: {e}")

    # Get param count from config if available
    config = {}
    try:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(req.model_id, "config.json")
        config = json.loads(Path(config_path).read_text())
    except Exception as _e:
        # M177 (iter 111): log HF config fetch failures per iter-106 pattern.
        # Common causes: HF unreachable, model gated/private without token,
        # corrupt cached config. config={} fallback is safe — downstream
        # _extract_param_count tolerates empty — but operators need
        # visibility into why enrichment fell through for a specific repo.
        log.warning(f"estimate: HF config fetch failed for {req.model_id}: {type(_e).__name__}: {_e}")

    # Try to extract param count
    num_params = _extract_param_count(config, info)
    source_bytes = sum(s.size for s in (info.siblings or []) if s.size)

    # Get target bits from profile
    target_bits = _profile_to_bits(profile)
    est = estimate_model_size(num_params, target_bits) if num_params else None

    # Detect architecture from config
    arch_info = _detect_arch_from_config(config)

    # Profile recommendation
    recommendations = _recommend_profiles(config, arch_info)

    return {
        "model_id": req.model_id,
        "profile": profile,
        "source_size_gb": round(source_bytes / (1024**3), 2),
        "num_params": num_params,
        "params_str": _params_to_str(num_params) if num_params else "unknown",
        "estimated_output": est,
        "architecture": arch_info,
        "recommendations": recommendations,
    }


# ── Architecture recommendation ─────────────────────────────

@app.get("/recommend/{model_id:path}", dependencies=[Depends(check_auth)])
def recommend_profile(model_id: str):
    """Detect architecture and recommend best profiles for a model."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    try:
        info = api.model_info(model_id)
    except Exception as e:
        raise HTTPException(404, f"Model not found: {e}")

    config = {}
    try:
        config_path = hf_hub_download(model_id, "config.json")
        config = json.loads(Path(config_path).read_text())
    except Exception as _e:
        # M177 (iter 111): second HF config-fetch site — same rationale as
        # the estimate-endpoint fetch above. Fall through with config={}
        # but log so operators can see which model's config fetch failed.
        log.warning(f"recommend: HF config fetch failed for {model_id}: {type(_e).__name__}: {_e}")

    arch_info = _detect_arch_from_config(config)
    recommendations = _recommend_profiles(config, arch_info)
    num_params = _extract_param_count(config, info)

    return {
        "model_id": model_id,
        "architecture": arch_info,
        "params": _params_to_str(num_params) if num_params else "unknown",
        "recommendations": recommendations,
    }


# ── Admin: purge old jobs ───────────────────────────────────

@app.post("/admin/purge", dependencies=[Depends(check_auth)])
def purge_old_jobs(hours: int = CLEANUP_HOURS):
    """Delete completed/failed jobs older than N hours."""
    cutoff = time.time() - hours * 3600
    removed = 0
    with _lock:
        to_remove = [
            jid for jid, j in _jobs.items()
            if j.phase in (JobPhase.COMPLETED, JobPhase.FAILED, JobPhase.CANCELLED)
            and j.timing.created_at < cutoff
        ]
        for jid in to_remove:
            del _jobs[jid]
            removed += 1

    # Also purge from DB
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,))
    conn.commit()
    conn.close()

    return {"purged": removed, "remaining": len(_jobs)}


# ── Health ──────────────────────────────────────────────────

@app.get("/health")
def health():
    with _queue_lock:
        queue_len = len(_queue)
    disk = shutil.disk_usage(WORK_DIR)
    active_model = None
    if _active_job_id:
        with _lock:
            aj = _jobs.get(_active_job_id)
        if aj:
            active_model = aj.model_id
    return {
        "status": "ok",
        "processing": _active_job_id,
        "processing_model": active_model,
        "queue_length": queue_len,
        "total_jobs": len(_jobs),
        "max_concurrent": MAX_CONCURRENT,
        "disk_free_gb": round(disk.free / (1024**3), 1),
        "disk_total_gb": round(disk.total / (1024**3), 1),
    }


# ---------------------------------------------------------------------------
# Background pipeline
# ---------------------------------------------------------------------------

def _run_job(job: Job):
    """Full pipeline: download → detect → allocate → quantize → write → upload.
    Called by the queue worker thread — runs one at a time."""
    try:
        job.timing.started_at = time.time()
        job.queue_position = 0
        job.log(f"Starting pipeline: {job.model_id} → {job.profile}")
        _notify_sse(job)

        _phase_download(job)
        _phase_detect(job)
        _phase_quantize(job)   # allocate + quantize + write happen inside convert_model
        _phase_upload(job)

        job.phase = JobPhase.COMPLETED
        job.progress_pct = 100.0
        elapsed = time.time() - job.timing.started_at
        job.timing.total_elapsed_seconds = elapsed
        job.result.duration_seconds = elapsed
        job.phase_detail = f"Done in {_fmt_duration(elapsed)}"
        job.log(f"Completed: {job.result.output_url}")

        _save_job(job)
        _notify_sse(job)
        _fire_webhook(job)

    except JobCancelled:
        job.phase = JobPhase.CANCELLED
        job.phase_detail = "Cancelled by user"
        job.timing.total_elapsed_seconds = time.time() - job.timing.started_at
        job.log("Job cancelled")
        _save_job(job)
        _notify_sse(job)
        _fire_webhook(job)

    except Exception as e:
        job.phase = JobPhase.FAILED
        job.error = f"{e}\n{traceback.format_exc()}"
        job.phase_detail = f"Failed during {job.phase.value}: {e}"
        job.timing.total_elapsed_seconds = time.time() - job.timing.started_at
        job.log(f"FAILED: {e}")
        _save_job(job)
        _notify_sse(job)
        _fire_webhook(job)

    finally:
        # Cleanup work dirs (keep logs in DB)
        for d in [WORK_DIR / job.id / "source", WORK_DIR / job.id / "output"]:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)


# ── Phase 1: Download ──────────────────────────────────────

def _phase_download(job: Job):
    from huggingface_hub import snapshot_download

    job.phase = JobPhase.DOWNLOADING
    job.progress_pct = 0.0
    job.timing.download_started = time.time()
    job.log(f"Downloading {job.model_id} ({_fmt_bytes(job.download.bytes_total)})...")
    _notify_sse(job)

    dest = WORK_DIR / job.id / "source"
    dest.mkdir(parents=True, exist_ok=True)

    # Speed tracking state
    speed_tracker = {"last_bytes": 0, "last_time": time.time()}

    def _update_download_progress():
        """Scan dest dir for actual downloaded bytes."""
        try:
            total = 0
            file_count = 0
            for f in dest.rglob("*"):
                if f.is_file() and not f.name.startswith("."):
                    total += f.stat().st_size
                    if f.suffix == ".safetensors":
                        file_count += 1
            job.download.bytes_done = total
            job.download.files_done = file_count

            now = time.time()
            dt = now - speed_tracker["last_time"]
            if dt >= 2.0:
                db = total - speed_tracker["last_bytes"]
                job.download.speed_mbps = round((db / dt) / (1024 * 1024), 1)
                if job.download.speed_mbps > 0 and job.download.bytes_total > 0:
                    remaining = job.download.bytes_total - total
                    job.download.eta_seconds = int(remaining / (db / dt)) if db > 0 else 0
                speed_tracker["last_bytes"] = total
                speed_tracker["last_time"] = now

            if job.download.bytes_total > 0:
                job.progress_pct = round(min(total / job.download.bytes_total * 100, 99.9), 1)
            job.phase_detail = (
                f"Downloading: {_fmt_bytes(total)} / {_fmt_bytes(job.download.bytes_total)} "
                f"({job.download.speed_mbps} MB/s)"
            )
        except Exception:
            pass

    # Progress monitor thread
    stop_monitor = threading.Event()

    def _monitor():
        while not stop_monitor.is_set():
            _update_download_progress()
            _notify_sse(job)
            job.check_cancelled()
            stop_monitor.wait(3.0)

    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()

    try:
        snapshot_download(
            job.model_id,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
    finally:
        stop_monitor.set()
        monitor.join(timeout=5)

    # Final size
    actual = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    sf_count = sum(1 for f in dest.rglob("*.safetensors"))
    job.download.bytes_done = actual
    job.download.bytes_total = actual
    job.download.files_done = sf_count
    job.progress_pct = 100.0
    job.timing.download_finished = time.time()
    dl_time = job.timing.download_finished - job.timing.download_started
    job.phase_detail = f"Downloaded {_fmt_bytes(actual)} in {_fmt_duration(dl_time)}"
    job.log(job.phase_detail)
    _notify_sse(job)


# ── Phase 2: Detect architecture ───────────────────────────

def _phase_detect(job: Job):
    from jang_tools.architectures import detect_architecture, summarize_architecture

    job.phase = JobPhase.DETECTING
    job.progress_pct = 0.0
    job.timing.detect_started = time.time()
    job.phase_detail = "Detecting architecture..."
    _notify_sse(job)

    source_path = WORK_DIR / job.id / "source"
    arch = detect_architecture(source_path)

    job.architecture.model_type = arch.model_type
    job.architecture.arch_type = arch.arch_type.value
    job.architecture.attention = arch.attention_type.value
    job.architecture.has_vision = arch.has_vision_encoder
    job.architecture.has_ssm = arch.has_ssm_layers
    job.architecture.has_moe = arch.has_moe_layers
    job.architecture.has_shared_mlp = getattr(arch, "has_shared_mlp", False)
    job.architecture.num_experts = arch.num_experts
    job.architecture.experts_per_tok = arch.num_experts_per_tok

    # Auto-adjustments
    warnings = []
    if arch.has_moe_layers and arch.num_experts >= 150:
        job.architecture.auto_group_size = 128
        warnings.append(f"{arch.num_experts} experts: auto group_size=128 (speed fix)")
    if arch.num_experts >= 512:
        job.architecture.mlp_asymmetry = True
        warnings.append("512+ experts: MLP asymmetry floor applied (gate_proj=4-bit, down_proj=3-bit)")
    hidden = 0
    try:
        cfg = json.loads((source_path / "config.json").read_text())
        hidden = cfg.get("hidden_size", cfg.get("text_config", {}).get("hidden_size", 0))
    except Exception as _e:
        # M177 (iter 111): log local config.json read failures. hidden=0
        # fallback disables the 512+experts+large-hidden bfloat16 override
        # heuristic — operator debugging "why isn't bfloat16 being forced
        # on this MoE?" needs to know the config read fell through.
        log.warning(f"architecture check: config.json read failed at {source_path}: {type(_e).__name__}: {_e}")
    if arch.num_experts >= 512 and hidden >= 4096:
        job.architecture.bfloat16_override = True
        warnings.append("512+ experts + large hidden: bfloat16 activations forced")
    if arch.has_vision_encoder:
        warnings.append("Vision encoder detected: VL preprocessor files will be preserved")
    if arch.has_ssm_layers:
        warnings.append("SSM layers detected: state matrices classified as CRITICAL")

    job.architecture.warnings = warnings
    job.progress_pct = 100.0
    job.timing.detect_finished = time.time()

    summary = summarize_architecture(arch)
    job.phase_detail = summary
    job.log(f"Architecture: {summary}")
    for w in warnings:
        job.log(f"  Warning: {w}")
    _notify_sse(job)


# ── Phase 3-5: Quantize (allocate + quantize + write) ──────

def _phase_quantize(job: Job):
    """Run the full JANG convert pipeline with progress hooks."""
    from jang_tools.convert import convert_model
    from jang_tools.allocate import JANG_PROFILES, is_k_quant, k_quant_target

    job.phase = JobPhase.QUANTIZING
    job.progress_pct = 0.0
    job.timing.quantize_started = time.time()

    source_path = WORK_DIR / job.id / "source"
    profile = job.profile
    job.quantization.profile = profile

    # Derive target_bits
    if is_k_quant(profile):
        target_bits = k_quant_target(profile)
    else:
        for ch in profile.replace("JANG_", ""):
            if ch.isdigit():
                target_bits = float(ch)
                break
        else:
            target_bits = 4.0
    job.quantization.target_bits = target_bits

    # Build output name
    model_short = job.model_id.split("/")[-1]
    for suffix in ["-BF16", "-bf16", "-FP16", "-fp16"]:
        if model_short.endswith(suffix):
            model_short = model_short[:-len(suffix)]
    output_name = f"{model_short}-{profile}"
    output_path = WORK_DIR / job.id / "output" / output_name
    output_path.mkdir(parents=True, exist_ok=True)

    job.phase_detail = f"Quantizing → {output_name}"
    job.log(f"Quantizing: {profile} (target {target_bits} bits)")
    _notify_sse(job)

    # Capture stdout to extract progress from convert_model's print statements
    log_capture = _LogCapture(job)

    job.check_cancelled()

    with redirect_stdout(log_capture), redirect_stderr(log_capture):
        result = convert_model(
            model_path=str(source_path),
            output_path=str(output_path),
            target_bits=target_bits,
            profile=profile,
            quantization_method="mse",
        )

    actual_bits = result.get("actual_bits", target_bits)
    total_gb = result.get("total_weight_gb", 0)
    job.quantization.actual_bits = actual_bits
    job.progress_pct = 100.0
    job.timing.quantize_finished = time.time()
    qt = job.timing.quantize_finished - job.timing.quantize_started
    job.phase_detail = f"Quantized: {actual_bits:.2f} bits, {total_gb} GB in {_fmt_duration(qt)}"
    job.log(job.phase_detail)

    # Check VL readiness
    vl_ready = (output_path / "preprocessor_config.json").exists()
    job.result.vl_ready = vl_ready
    job.result.actual_bits = actual_bits
    job.result.total_size_gb = total_gb

    # Extract param count from result or jang_config
    jang_cfg_path = output_path / "jang_config.json"
    if jang_cfg_path.exists():
        jc = json.loads(jang_cfg_path.read_text())
        job.result.source_params = jc.get("source_model", {}).get("parameters", "")

    _notify_sse(job)


# ── Phase 6: Upload ────────────────────────────────────────

def _phase_upload(job: Job):
    from huggingface_hub import HfApi

    job.phase = JobPhase.UPLOADING
    job.progress_pct = 0.0
    job.timing.upload_started = time.time()

    output_base = WORK_DIR / job.id / "output"
    output_dirs = [d for d in output_base.iterdir() if d.is_dir()]
    if not output_dirs:
        raise RuntimeError("No output directory after quantization")
    output_path = output_dirs[0]
    repo_name = output_path.name
    repo_id = f"{HF_ORG}/{repo_name}"

    # Calculate total upload size
    upload_bytes = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    job.upload.bytes_total = upload_bytes

    job.result.output_repo = repo_id
    job.result.output_url = f"https://huggingface.co/{repo_id}"
    job.phase_detail = f"Uploading {_fmt_bytes(upload_bytes)} to {repo_id}..."
    job.log(job.phase_detail)
    _notify_sse(job)

    job.check_cancelled()

    # Generate model card
    readme_content = _generate_model_card(job, output_path)
    readme_path = output_path / "README.md"
    readme_path.write_text(readme_content)

    api = HfApi(token=HF_UPLOAD_TOKEN)

    # Create repo
    api.create_repo(repo_id, exist_ok=True, repo_type="model")

    # Upload with progress monitor
    speed_tracker = {"last_bytes": 0, "last_time": time.time()}

    # We can't get per-byte progress from upload_folder easily,
    # so we track by counting uploaded files
    total_files = sum(1 for f in output_path.rglob("*") if f.is_file())
    job.quantization.shards_total = total_files

    api.upload_folder(
        folder_path=str(output_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"JANG {job.profile}: {job.model_id}",
    )

    job.upload.bytes_done = upload_bytes
    job.progress_pct = 100.0
    job.timing.upload_finished = time.time()
    ut = job.timing.upload_finished - job.timing.upload_started
    job.phase_detail = f"Uploaded {_fmt_bytes(upload_bytes)} in {_fmt_duration(ut)}"
    job.log(job.phase_detail)
    _notify_sse(job)


# ---------------------------------------------------------------------------
# Model card generation
# ---------------------------------------------------------------------------

def _generate_model_card(job: Job, output_path: Path) -> str:
    """Auto-generate a README.md for the HuggingFace repo."""
    arch = job.architecture
    quant = job.quantization
    result = job.result

    arch_badges = []
    if arch.has_moe:
        arch_badges.append(f"MoE ({arch.num_experts} experts, top-{arch.experts_per_tok})")
    if arch.has_ssm:
        arch_badges.append("SSM")
    if arch.has_vision:
        arch_badges.append("VLM")
    if arch.attention == "mla":
        arch_badges.append("MLA")
    elif arch.attention == "gqa":
        arch_badges.append("GQA")
    if arch.has_shared_mlp:
        arch_badges.append("Shared MLP")
    arch_str = " | ".join(arch_badges) if arch_badges else arch.arch_type

    # Read jang_config for bit histogram
    bit_hist = ""
    jang_cfg_path = output_path / "jang_config.json"
    if jang_cfg_path.exists():
        jc = json.loads(jang_cfg_path.read_text())
        bit_widths = jc.get("quantization", {}).get("bit_widths_used", [])
        if bit_widths:
            bit_hist = f"Bit widths used: {', '.join(str(b) for b in bit_widths)}"

    warnings_section = ""
    if arch.warnings:
        warnings_section = "\n### Architecture Notes\n" + "\n".join(f"- {w}" for w in arch.warnings)

    card = f"""---
tags:
- jang
- mlx
- quantized
- apple-silicon
- {arch.arch_type}
license: apache-2.0
base_model: {job.model_id}
---

# {output_path.name}

JANG mixed-precision quantization of [{job.model_id}](https://huggingface.co/{job.model_id}).

| Property | Value |
|----------|-------|
| Profile | **{job.profile}** |
| Actual bits | **{result.actual_bits:.2f}** |
| Size | **{result.total_size_gb:.1f} GB** |
| Source params | {result.source_params} |
| Architecture | {arch_str} |
| Model type | `{arch.model_type}` |
| Attention | `{arch.attention}` |
| VL ready | {"Yes" if result.vl_ready else "No"} |
| Format | JANG v2 (MLX-native, instant load) |

{bit_hist}
{warnings_section}

## Usage

```bash
pip install jang[mlx]
```

```python
from jang_tools import load_for_inference

model, tokenizer = load_for_inference("{HF_ORG}/{output_path.name}")
```

{"### VLM Usage" if result.vl_ready else ""}
{"```python" if result.vl_ready else ""}
{"from jang_tools import load_jang_vlm_model" if result.vl_ready else ""}
{"" if result.vl_ready else ""}
{"model, processor = load_jang_vlm_model('" + HF_ORG + "/" + output_path.name + "')" if result.vl_ready else ""}
{"```" if result.vl_ready else ""}

## What is JANG?

JANG (Jang Adaptive N-bit Grading) is mixed-precision quantization for Apple Silicon.
Unlike uniform quantization, JANG classifies each weight tensor by sensitivity
(CRITICAL/IMPORTANT/COMPRESS) and assigns optimal bit widths per tier.

- CRITICAL (attention Q/K/V/O, MoE routers, SSM state): highest precision
- IMPORTANT (embeddings, VL connectors): medium precision
- COMPRESS (MLP/FFN, routed experts): aggressive compression

Result: same file size as uniform quantization, significantly better quality.

---
Quantized by [JANG Quantization API](https://jangq.ai) | Created by Jinho Jang (eric@jangq.ai)
"""
    return card


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------

def _fire_webhook(job: Job):
    """POST final job status to webhook URL if configured."""
    if not job.webhook_url:
        return
    try:
        import urllib.request
        data = json.dumps(_job_to_dict(job)).encode()
        req = urllib.request.Request(
            job.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        job.log(f"Webhook delivered to {job.webhook_url}")
    except Exception as e:
        job.log(f"Webhook failed: {e}")


# ---------------------------------------------------------------------------
# Log capture (intercept convert_model prints for progress extraction)
# ---------------------------------------------------------------------------

class _LogCapture(io.TextIOBase):
    """Captures stdout/stderr from convert_model and extracts progress."""

    def __init__(self, job: Job):
        self.job = job
        self._buffer = ""

    def write(self, s: str) -> int:
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            self.job.log(line)
            self._parse_line(line)
            _notify_sse(self.job)
        return len(s)

    def flush(self):
        pass

    def _parse_line(self, line: str):
        """Extract progress info from convert_model's print output."""
        # [1/5] Detecting architecture...
        if "[1/5]" in line:
            self.job.phase = JobPhase.DETECTING
            self.job.phase_detail = "Detecting architecture..."
        # [2/5] Calibrating or Skipping
        elif "[2/5]" in line:
            self.job.phase_detail = line.strip()
        # [3/5] Allocating bits...
        elif "[3/5]" in line:
            self.job.phase = JobPhase.ALLOCATING
            self.job.phase_detail = "Allocating bits..."
        # Actual bits: 4.02
        elif "Actual bits:" in line:
            try:
                bits = float(line.split("Actual bits:")[1].strip())
                self.job.quantization.actual_bits = bits
            except (ValueError, IndexError):
                pass
        # Total blocks: 12,345
        elif "Total blocks:" in line:
            try:
                blocks = int(line.split("Total blocks:")[1].strip().replace(",", ""))
                self.job.phase_detail = f"Allocated {blocks:,} blocks"
            except (ValueError, IndexError):
                pass
        # [4/5] Quantizing...
        elif "[4/5]" in line:
            self.job.phase = JobPhase.QUANTIZING
            self.job.phase_detail = "Quantizing tensors..."
        # tqdm progress: "  Quantizing:  47%|..."
        elif "Quantizing:" in line and "%" in line:
            try:
                pct_str = line.split("%")[0].split()[-1]
                pct = float(pct_str)
                self.job.progress_pct = pct
                # Extract tensor count if available: "156/330"
                if "/" in line:
                    parts = line.split("|")
                    for p in parts:
                        if "/" in p:
                            nums = p.strip().split("/")
                            if len(nums) == 2 and nums[0].strip().isdigit():
                                self.job.quantization.tensors_done = int(nums[0].strip())
                                self.job.quantization.tensors_total = int(nums[1].strip().split()[0])
                                break
            except (ValueError, IndexError):
                pass
        # Gate passthrough
        elif "Gate passthrough" in line:
            self.job.log(line)
        # [5/5] Writing...
        elif "[5/5]" in line:
            self.job.phase = JobPhase.WRITING
            self.job.phase_detail = "Writing JANG v2 model..."
        # Precision warnings
        elif "PRECISION WARNING" in line:
            self.job.quantization.precision_warnings.append(line.strip())
        # MLP asymmetry
        elif "MLP asymmetry" in line:
            self.job.architecture.mlp_asymmetry = True
        # Architecture info
        elif "Architecture:" in line:
            try:
                self.job.architecture.arch_type = line.split("Architecture:")[1].strip()
            except (IndexError):
                pass
        elif "MoE:" in line:
            try:
                parts = line.split("MoE:")[1].strip()
                if "experts" in parts:
                    self.job.architecture.num_experts = int(parts.split()[0])
            except (ValueError, IndexError):
                pass
        # Bit allocation histogram: "  4: 12,345 blocks (85.2%)"
        elif "blocks (" in line and "%" in line:
            try:
                parts = line.strip().split(":")
                if len(parts) >= 2 and parts[0].strip().isdigit():
                    bit_width = parts[0].strip()
                    count = int(parts[1].strip().split()[0].replace(",", ""))
                    self.job.quantization.bit_histogram[bit_width] = count
            except (ValueError, IndexError):
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_bytes(n: int) -> str:
    if n == 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def _profile_to_bits(profile: str) -> float:
    """Extract target bits from profile name."""
    from jang_tools.allocate import is_k_quant, k_quant_target
    if is_k_quant(profile):
        return k_quant_target(profile)
    for ch in profile.replace("JANG_", ""):
        if ch.isdigit():
            return float(ch)
    return 4.0


def _extract_param_count(config: dict, model_info=None) -> int:
    """Try to extract parameter count from model config or HF metadata."""
    # Direct field
    for key in ("num_parameters", "n_params"):
        if key in config:
            return int(config[key])

    # Calculate from dimensions
    hidden = config.get("hidden_size", config.get("text_config", {}).get("hidden_size", 0))
    n_layers = config.get("num_hidden_layers", config.get("text_config", {}).get("num_hidden_layers", 0))
    vocab = config.get("vocab_size", config.get("text_config", {}).get("vocab_size", 0))
    intermediate = config.get("intermediate_size", config.get("text_config", {}).get("intermediate_size", 0))

    if hidden and n_layers and vocab:
        # Rough estimate: embedding + n_layers * (4*h^2 + 2*h*inter) + lm_head
        attn_params = 4 * hidden * hidden  # Q, K, V, O
        ffn_params = 2 * hidden * (intermediate or 4 * hidden)  # gate+up, down
        n_experts = config.get("num_local_experts", config.get("text_config", {}).get("num_local_experts", 1))
        layer_params = attn_params + ffn_params * max(n_experts, 1)
        total = vocab * hidden + n_layers * layer_params + vocab * hidden
        return total

    # From model_info safetensors metadata
    if model_info and hasattr(model_info, "safetensors"):
        st = model_info.safetensors
        if st and hasattr(st, "total"):
            return st.total

    return 0


def _params_to_str(n: int) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.1f}T"
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return str(n)


def _detect_arch_from_config(config: dict) -> dict:
    """Lightweight architecture detection from config.json (no download needed)."""
    model_type = config.get("model_type", config.get("text_config", {}).get("model_type", "unknown"))
    arch_list = config.get("architectures", [])

    has_vision = bool(config.get("vision_config")) or any("Conditional" in a for a in arch_list)
    has_moe = config.get("num_local_experts", config.get("text_config", {}).get("num_local_experts", 0)) > 1
    num_experts = config.get("num_local_experts", config.get("text_config", {}).get("num_local_experts", 0))
    experts_per_tok = config.get("num_experts_per_tok", config.get("text_config", {}).get("num_experts_per_tok", 0))
    has_mla = bool(config.get("kv_lora_rank", config.get("text_config", {}).get("kv_lora_rank", 0)))
    has_ssm = model_type in ("jamba", "zamba", "zamba2", "mamba", "mamba2") or bool(config.get("attn_type_list"))

    if has_moe and has_ssm:
        arch_type = "hybrid_moe_ssm"
    elif has_ssm:
        arch_type = "hybrid_ssm"
    elif has_moe:
        arch_type = "moe"
    elif has_vision:
        arch_type = "vision_language"
    else:
        arch_type = "transformer"

    attention = "mla" if has_mla else "gqa"

    return {
        "model_type": model_type,
        "arch_type": arch_type,
        "attention": attention,
        "has_vision": has_vision,
        "has_ssm": has_ssm,
        "has_moe": has_moe,
        "num_experts": num_experts,
        "experts_per_tok": experts_per_tok,
    }


def _recommend_profiles(config: dict, arch_info: dict) -> list[dict]:
    """Recommend profiles based on architecture."""
    recs = []
    num_experts = arch_info.get("num_experts", 0)
    has_moe = arch_info.get("has_moe", False)
    has_ssm = arch_info.get("has_ssm", False)

    if has_moe:
        # MoE models benefit most from JANG — routed experts compress well
        recs.append({
            "profile": "JANG_2S",
            "reason": "MoE models have massive compressible expert blocks. 2-bit COMPRESS with 6-bit attention is proven optimal for MoE.",
        })
        recs.append({
            "profile": "JANG_4K",
            "reason": "Budget-neutral 4-bit: same size as uniform 4-bit, smarter bit allocation. Safe default.",
        })
        if num_experts >= 512:
            recs.append({
                "profile": "JANG_2L",
                "reason": f"512+ experts ({num_experts}): use L profile for maximum attention protection. MLP asymmetry floor auto-applied.",
            })
    elif has_ssm:
        recs.append({
            "profile": "JANG_4K",
            "reason": "Hybrid SSM models: SSM state matrices are CRITICAL tier. K-quant gives best balance.",
        })
        recs.append({
            "profile": "JANG_3L",
            "reason": "3-bit with 4-bit IMPORTANT tier protects SSM projections.",
        })
    else:
        # Dense transformer — JANG helps less on dense models
        recs.append({
            "profile": "JANG_4K",
            "reason": "Budget-neutral 4-bit: recommended default for dense transformers.",
        })
        recs.append({
            "profile": "JANG_3K",
            "reason": "Budget-neutral 3-bit for smaller size with smart allocation.",
        })

    return recs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8420"))
    auth_mode = "API keys" if API_KEYS else "none (open)"

    print(f"""
  JANG Quantization API v2.0.0
  Created by Jinho Jang (eric@jangq.ai)

  Server:       http://0.0.0.0:{port}
  Upload org:   {HF_ORG}
  Work dir:     {WORK_DIR}
  DB:           {DB_PATH}
  Queue mode:   sequential (1 job at a time)
  Auth:         {auth_mode}
  Per-user max: {MAX_JOBS_PER_USER}
""")
    uvicorn.run(app, host="0.0.0.0", port=port)
