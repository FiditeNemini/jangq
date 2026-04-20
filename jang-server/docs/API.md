# JANG Quantization API

**Version 2.0.0** | Created by Jinho Jang (eric@jangq.ai)

Team-facing HTTP API for quantizing HuggingFace models using JANG mixed-precision quantization. Handles the full pipeline: download from HF, detect architecture, allocate bits per tier, quantize with MLX, upload to HuggingFace — all with real-time progress tracking.

Supports all architectures: Transformer, MoE, SSM, Hybrid SSM+MoE, VLM, MLA, GatedDeltaNet.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [Environment Variables](#environment-variables)
- [Endpoints](#endpoints)
  - [POST /jobs — Submit Job](#post-jobs)
  - [GET /jobs/{job_id} — Job Status](#get-jobsjob_id)
  - [GET /jobs — List Jobs](#get-jobs)
  - [GET /queue — Queue Status](#get-queue)
  - [DELETE /jobs/{job_id} — Cancel Job](#delete-jobsjob_id)
  - [POST /jobs/{job_id}/retry — Retry Job](#post-jobsjob_idretry)
  - [GET /jobs/{job_id}/logs — Job Logs](#get-jobsjob_idlogs)
  - [GET /jobs/{job_id}/stream — SSE Stream](#get-jobsjob_idstream)
  - [GET /profiles — List Profiles](#get-profiles)
  - [POST /estimate — Size Estimation](#post-estimate)
  - [GET /recommend/{model_id} — Profile Recommendation](#get-recommendmodel_id)
  - [POST /admin/purge — Purge Old Jobs](#post-adminpurge)
  - [GET /health — Health Check](#get-health)
- [Job Lifecycle](#job-lifecycle)
- [Queue System](#queue-system)
- [Response Schema](#response-schema)
- [Profiles Reference](#profiles-reference)
- [Webhooks](#webhooks)
- [SSE Streaming](#sse-streaming)
- [Error Handling](#error-handling)

---

## Quick Start

```bash
cd jang-server
pip install -r requirements.txt
python server.py
```

Server starts at `http://0.0.0.0:8420`.

```bash
# Submit a job
curl -X POST http://localhost:8420/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "Qwen/Qwen3-8B",
    "profile": "JANG_4K",
    "user": "alice"
  }'

# Check status
curl http://localhost:8420/jobs/<job_id>

# Watch the queue
curl http://localhost:8420/queue

# Stream real-time updates
curl -N http://localhost:8420/jobs/<job_id>/stream
```

---

## Authentication

Authentication is optional. Set `JANG_API_KEYS` to enable it.

**Methods:**

1. **Bearer token (preferred):** `Authorization: Bearer <key>` — accepted on ALL HTTP methods.
2. **Query parameter (SSE only):** `?api_key=<key>` — accepted on **GET requests only**. Intended solely for browsers' `EventSource` API (`GET /jobs/{id}/stream`), which cannot set custom headers. Do not use on POST/DELETE: tokens in non-GET URLs leak to server logs, browser history, proxy logs, and terminal history.

**Protected endpoints (require auth when `JANG_API_KEYS` is set):**
- `POST /jobs`, `POST /jobs/{id}/retry`, `POST /estimate`, `POST /admin/purge`
- `DELETE /jobs/{id}`
- `GET /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/logs`, `GET /jobs/{id}/stream`, `GET /queue`, `GET /recommend/{model_id}`

**Public endpoints (never require auth):**
- `GET /health`, `GET /profiles`

If `JANG_API_KEYS` is empty (default), all endpoints are open.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_UPLOAD_TOKEN` | _(empty)_ | HuggingFace token for uploading quantized models (set via env, never hardcode) |
| `HF_ORG` | `JANGQ-AI` | HuggingFace org/user to upload models under |
| `JANG_WORK_DIR` | `/tmp/jang-server` | Working directory for downloads, output, and SQLite DB |
| `JANG_API_KEYS` | _(empty)_ | Comma-separated API keys. Empty = no auth |
| `JANG_MAX_JOBS_PER_USER` | `3` | Max active (queued + running) jobs per user |
| `JANG_CLEANUP_HOURS` | `24` | Auto-purge completed jobs older than N hours |
| `PORT` | `8420` | Server port |

---

## Endpoints

### POST /jobs

Submit a new quantization job. The model is validated on HuggingFace, disk space is checked, and the job enters the queue.

**Request body:**

```json
{
  "model_id": "Qwen/Qwen3-235B-A22B",
  "profile": "JANG_4K",
  "user": "alice",
  "priority": 0,
  "webhook_url": "https://hooks.slack.com/..."
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_id` | string | yes | — | HuggingFace model ID in `org/name` format |
| `profile` | string | no | `JANG_4K` | JANG quantization profile (see [Profiles](#profiles-reference)) |
| `user` | string | no | `""` | User identifier for tracking and per-user limits |
| `priority` | int | no | `0` | Queue priority. Higher values are processed first |
| `webhook_url` | string | no | `""` | URL to POST final job status when complete/failed |

**Pre-flight checks (all happen before queueing):**

1. Profile validation
2. Model ID format check (`org/name`)
3. Duplicate detection — if same model+profile is already queued/running/completed, returns the existing job
4. Per-user active job limit check
5. Model existence verification on HuggingFace
6. Disk space check (needs ~2.5x source model size)

**Response:** Full [Job object](#response-schema)

**Duplicate response:**

```json
{
  "duplicate": true,
  "existing_job": { ... },
  "message": "Job already running: abc123def456"
}
```

**Error codes:** `400` (bad input), `404` (model not found on HF), `429` (user limit), `507` (disk full)

---

### GET /jobs/{job_id}

Get full status of a job including all per-phase progress details.

**Response:** Full [Job object](#response-schema)

**Error codes:** `404` (job not found)

---

### GET /jobs

List all jobs, sorted by priority (desc) then created time (desc).

**Query parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `user` | string | Filter to jobs from this user |
| `phase` | string | Filter to jobs in this phase (e.g. `queued`, `quantizing`, `completed`) |
| `limit` | int | Max results (default 50) |

**Response:** Array of [Job objects](#response-schema)

---

### GET /queue

Get the current queue state in one call: what's running, what's waiting, recent history.

**Response:**

```json
{
  "active": { /* full job object of currently processing job, or null */ },
  "queue_length": 3,
  "queued": [
    { /* job object, queue_position: 1 */ },
    { /* job object, queue_position: 2 */ },
    { /* job object, queue_position: 3 */ }
  ],
  "recent_completed": [
    { /* last 5 completed/failed/cancelled jobs */ }
  ],
  "max_concurrent": 1
}
```

---

### DELETE /jobs/{job_id}

Cancel a queued or running job.

- **Queued jobs:** Removed from queue immediately. Other jobs' positions are recomputed.
- **Running jobs:** Cancellation flag is set. The pipeline checks this flag between phases and stops at the next phase boundary.

**Response:**

```json
{
  "message": "Cancellation requested for job abc123def456"
}
```

**Error codes:** `400` (already in terminal state), `404` (not found)

---

### POST /jobs/{job_id}/retry

Retry a failed or cancelled job. Creates a new job with the same parameters.

**Response:**

```json
{
  "message": "Retrying as new job def789abc012",
  "new_job": { /* full job object */ }
}
```

**Error codes:** `400` (not in failed/cancelled state), `404` (not found)

---

### GET /jobs/{job_id}/logs

Get the last 200 log lines from the job's pipeline execution.

**Response:**

```json
{
  "job_id": "abc123def456",
  "lines": [
    "[14:32:01] Starting pipeline: Qwen/Qwen3-8B → JANG_4K",
    "[14:32:02] Downloading 16.2 GB...",
    "[14:35:44] Downloaded 16.2 GB in 3.7m",
    "[14:35:44] Architecture: moe, Attention: gqa, MoE: 128 experts top-8",
    "[14:35:45] [3/5] Allocating bits...",
    "[14:35:45] Using K-quant: JANG_4K (target: 4.0 avg bits)",
    "[14:35:45] Actual bits: 4.02",
    "..."
  ]
}
```

---

### GET /jobs/{job_id}/stream

Server-sent events (SSE) stream for real-time job updates. Pushes the full job object on every state change.

**Content-Type:** `text/event-stream`

**Events:**

```
data: {"job_id": "abc123", "phase": "downloading", "progress_pct": 45.2, ...}

data: {"job_id": "abc123", "phase": "quantizing", "progress_pct": 12.0, ...}

: keepalive

data: {"job_id": "abc123", "phase": "completed", "progress_pct": 100.0, ...}
```

- Each `data:` line contains the full job JSON
- Keepalive comments (`: keepalive`) are sent every 30 seconds
- Stream ends automatically when job reaches `completed`, `failed`, or `cancelled`

**Usage with JavaScript:**

```javascript
const es = new EventSource('/jobs/abc123def456/stream');
es.onmessage = (e) => {
  const job = JSON.parse(e.data);
  console.log(`${job.phase}: ${job.progress_pct}% — ${job.phase_detail}`);
  if (['completed', 'failed', 'cancelled'].includes(job.phase)) {
    es.close();
  }
};
```

**Usage with curl:**

```bash
curl -N http://localhost:8420/jobs/abc123def456/stream
```

---

### GET /profiles

List all available JANG quantization profiles.

**Response:**

```json
{
  "profiles": {
    "JANG_4K": {
      "desc": "4-bit budget-neutral K-quant — recommended default",
      "tiers": "budget-neutral",
      "bits": 4
    },
    ...
  }
}
```

---

### POST /estimate

Estimate output size for a model+profile without starting a job. Downloads only `config.json` from HuggingFace.

**Request body:**

```json
{
  "model_id": "Qwen/Qwen3-235B-A22B",
  "profile": "JANG_4K"
}
```

**Response:**

```json
{
  "model_id": "Qwen/Qwen3-235B-A22B",
  "profile": "JANG_4K",
  "source_size_gb": 440.5,
  "num_params": 235000000000,
  "params_str": "235.0B",
  "estimated_output": {
    "nominal_bits": 4.0,
    "effective_bits": 4.5,
    "weight_bytes": 132150000000,
    "weight_gb": 123.07
  },
  "architecture": {
    "model_type": "qwen3_5_moe",
    "arch_type": "hybrid_moe_ssm",
    "attention": "gqa",
    "has_vision": false,
    "has_ssm": true,
    "has_moe": true,
    "num_experts": 128,
    "experts_per_tok": 8
  },
  "recommendations": [
    {
      "profile": "JANG_2S",
      "reason": "MoE models have massive compressible expert blocks..."
    },
    {
      "profile": "JANG_4K",
      "reason": "Budget-neutral 4-bit: same size as uniform 4-bit, smarter bit allocation..."
    }
  ]
}
```

---

### GET /recommend/{model_id}

Detect a model's architecture and get profile recommendations without downloading the full model.

**Example:** `GET /recommend/Qwen/Qwen3-235B-A22B`

**Response:**

```json
{
  "model_id": "Qwen/Qwen3-235B-A22B",
  "architecture": {
    "model_type": "qwen3_5_moe",
    "arch_type": "hybrid_moe_ssm",
    "attention": "gqa",
    "has_vision": false,
    "has_ssm": true,
    "has_moe": true,
    "num_experts": 128,
    "experts_per_tok": 8
  },
  "params": "235.0B",
  "recommendations": [
    {"profile": "JANG_2S", "reason": "..."},
    {"profile": "JANG_4K", "reason": "..."}
  ]
}
```

---

### POST /admin/purge

Delete completed/failed/cancelled jobs older than N hours from memory and database.

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `hours` | int | 24 | Delete jobs older than this many hours |

**Response:**

```json
{
  "purged": 12,
  "remaining": 5
}
```

---

### GET /health

Server health check with queue and disk status.

**Response:**

```json
{
  "status": "ok",
  "processing": "abc123def456",
  "processing_model": "Qwen/Qwen3-235B-A22B",
  "queue_length": 2,
  "total_jobs": 47,
  "max_concurrent": 1,
  "disk_free_gb": 812.3,
  "disk_total_gb": 1862.0
}
```

- `processing`: job ID of the currently running job, or `null`
- `processing_model`: model ID of the running job, or `null`
- `queue_length`: number of jobs waiting

---

## Job Lifecycle

```
                         ┌──────────────────────────────────────────────┐
                         │                                              │
  POST /jobs ──▶ QUEUED ──▶ DOWNLOADING ──▶ DETECTING ──▶ ALLOCATING   │
                   │                                          │        │
                   │ (cancel)                                 ▼        │
                   ▼                                    QUANTIZING     │
               CANCELLED                                     │        │
                                                             ▼        │
                                                         WRITING      │
                                                             │        │
                                                             ▼        │
                                                        UPLOADING     │
                                                             │        │
                                                     ┌───────┴───┐    │
                                                     ▼           ▼    │
                                                 COMPLETED    FAILED ─┘
                                                              (retry)
```

**Phases:**

| Phase | Description | Progress tracked |
|-------|-------------|-----------------|
| `queued` | Waiting in queue | `queue_position` (1, 2, 3...) |
| `downloading` | Downloading from HuggingFace | bytes, files, speed (MB/s), ETA |
| `detecting` | Analyzing model architecture | Architecture type, MoE/SSM/VL flags, warnings |
| `allocating` | Assigning bit widths to tensors | Profile, target bits, actual bits, bit histogram |
| `quantizing` | Running mx.quantize() on each tensor | Tensors done/total, current tensor, current layer |
| `writing` | Writing shards to disk | Shards written/total |
| `uploading` | Uploading to HuggingFace | bytes, speed, ETA |
| `completed` | Done | Output repo URL, actual bits, size, duration, VL readiness |
| `failed` | Error occurred | Error message + traceback, which phase failed |
| `cancelled` | Cancelled by user | Whether cancelled while queued or mid-pipeline |

---

## Queue System

Jobs are processed **one at a time** in a single worker thread. Quantization is GPU/RAM intensive, so concurrent runs would OOM.

**Ordering:** Priority (higher first) then FIFO (earlier submitted first).

**Queue position:** Every queued job has a `queue_position` field (1 = next up, 2 = after that, etc.). Positions are recomputed whenever a job finishes or is cancelled.

**Active job:** The `is_active` field is `true` only for the currently processing job. `GET /queue` returns the full active job object under `active`.

**Cancel while queued:** The job is removed from the queue immediately and all positions shift down.

**Cancel while running:** A flag is set. The pipeline checks this flag between phases. The running phase completes, then the job transitions to `cancelled`.

---

## Response Schema

Every job endpoint returns this structure:

```json
{
  "job_id": "abc123def456",
  "model_id": "Qwen/Qwen3-235B-A22B",
  "profile": "JANG_4K",
  "user": "alice",
  "priority": 0,
  "phase": "quantizing",
  "progress_pct": 47.3,
  "phase_detail": "Quantizing tensors...",
  "error": "",
  "queue_position": 0,
  "is_active": true,

  "download": {
    "bytes_done": 90000000000,
    "bytes_total": 90000000000,
    "files_done": 24,
    "files_total": 24,
    "speed_mbps": 0.0,
    "eta_seconds": 0
  },

  "architecture": {
    "model_type": "qwen3_5_moe",
    "arch_type": "hybrid_moe_ssm",
    "attention": "gqa",
    "has_vision": false,
    "has_ssm": true,
    "has_moe": true,
    "has_shared_mlp": false,
    "num_experts": 128,
    "experts_per_tok": 8,
    "auto_group_size": 128,
    "mlp_asymmetry": false,
    "bfloat16_override": false,
    "warnings": ["128 experts: auto group_size=128 (speed fix)"]
  },

  "quantization": {
    "profile": "JANG_4K",
    "target_bits": 4.0,
    "actual_bits": 4.02,
    "tensors_done": 156,
    "tensors_total": 330,
    "current_tensor": "model.layers.24.self_attn.q_proj.weight",
    "current_layer": 24,
    "total_layers": 48,
    "bit_histogram": {"2": 1200, "4": 2800, "8": 100},
    "precision_warnings": [],
    "shards_written": 0,
    "shards_total": 0
  },

  "upload": {
    "bytes_done": 0,
    "bytes_total": 0,
    "speed_mbps": 0.0,
    "eta_seconds": 0
  },

  "result": {
    "output_repo": "",
    "output_url": "",
    "actual_bits": 0.0,
    "total_size_gb": 0.0,
    "source_params": "",
    "vl_ready": false,
    "duration_seconds": 0.0
  },

  "timing": {
    "created_at": 1743700000.0,
    "started_at": 1743700005.0,
    "download_started": 1743700005.0,
    "download_finished": 1743700300.0,
    "detect_started": 1743700300.0,
    "detect_finished": 1743700301.0,
    "quantize_started": 1743700301.0,
    "quantize_finished": 0.0,
    "upload_started": 0.0,
    "upload_finished": 0.0,
    "total_elapsed_seconds": 0.0
  }
}
```

**Top-level fields:**

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | 12-char hex identifier |
| `model_id` | string | Source HuggingFace model ID |
| `profile` | string | JANG profile used |
| `user` | string | Who submitted the job |
| `priority` | int | Queue priority (higher = first) |
| `phase` | string | Current phase (see [Job Lifecycle](#job-lifecycle)) |
| `progress_pct` | float | 0-100, progress within current phase |
| `phase_detail` | string | Human-readable status message |
| `error` | string | Error message + traceback (only when `phase=failed`) |
| `queue_position` | int | Position in queue (0 = not queued) |
| `is_active` | bool | `true` if this is the currently processing job |

**`download` object:**

| Field | Type | Description |
|-------|------|-------------|
| `bytes_done` | int | Bytes downloaded so far |
| `bytes_total` | int | Total model size in bytes |
| `files_done` | int | Safetensor files downloaded |
| `files_total` | int | Total safetensor files |
| `speed_mbps` | float | Rolling download speed in MB/s |
| `eta_seconds` | int | Estimated seconds remaining |

**`architecture` object:**

| Field | Type | Description |
|-------|------|-------------|
| `model_type` | string | HuggingFace model_type (e.g. `qwen3_5_moe`, `llama`, `jamba`) |
| `arch_type` | string | JANG classification: `transformer`, `moe`, `hybrid_ssm`, `hybrid_moe_ssm`, `vision_language`, `mamba` |
| `attention` | string | Attention mechanism: `mha`, `gqa`, `mqa`, `mla`, `none` |
| `has_vision` | bool | Vision encoder detected |
| `has_ssm` | bool | SSM (Mamba/DeltaNet) layers detected |
| `has_moe` | bool | Mixture-of-Experts layers detected |
| `has_shared_mlp` | bool | Shared dense MLP alongside MoE (e.g. Gemma 4) |
| `num_experts` | int | Number of routed experts |
| `experts_per_tok` | int | Experts activated per token |
| `auto_group_size` | int | Quantization group size (auto-bumped to 128 for 150+ experts) |
| `mlp_asymmetry` | bool | MLP asymmetry floor applied (512+ experts) |
| `bfloat16_override` | bool | bfloat16 activations forced (512+ experts + large hidden) |
| `warnings` | string[] | Architecture-specific warnings and auto-adjustments |

**`quantization` object:**

| Field | Type | Description |
|-------|------|-------------|
| `profile` | string | Profile name |
| `target_bits` | float | Requested average bits per weight |
| `actual_bits` | float | Actual average bits after allocation |
| `tensors_done` | int | Tensors quantized so far |
| `tensors_total` | int | Total tensors to quantize |
| `current_tensor` | string | Name of tensor currently being quantized |
| `current_layer` | int | Current layer number |
| `total_layers` | int | Total layers in model |
| `bit_histogram` | object | Counts of blocks at each bit width, e.g. `{"2": 1200, "4": 2800, "8": 100}` |
| `precision_warnings` | string[] | Precision floor violations detected during allocation |
| `shards_written` | int | Output shards written to disk |
| `shards_total` | int | Total shards to write |

**`upload` object:**

| Field | Type | Description |
|-------|------|-------------|
| `bytes_done` | int | Bytes uploaded so far |
| `bytes_total` | int | Total output size in bytes |
| `speed_mbps` | float | Upload speed in MB/s |
| `eta_seconds` | int | Estimated seconds remaining |

**`result` object (populated on completion):**

| Field | Type | Description |
|-------|------|-------------|
| `output_repo` | string | HuggingFace repo ID (e.g. `JANGQ-AI/Qwen3-235B-A22B-JANG_4K`) |
| `output_url` | string | Full URL: `https://huggingface.co/JANGQ-AI/Qwen3-235B-A22B-JANG_4K` |
| `actual_bits` | float | Final average bits per weight |
| `total_size_gb` | float | Quantized model size in GB |
| `source_params` | string | Source model parameter count (e.g. `"235B"`) |
| `vl_ready` | bool | `true` if VL preprocessor files are included |
| `duration_seconds` | float | Total pipeline duration |

**`timing` object (all values are Unix timestamps, 0.0 if not yet reached):**

| Field | Type | Description |
|-------|------|-------------|
| `created_at` | float | When the job was submitted |
| `started_at` | float | When the worker picked up the job |
| `download_started` | float | Download phase start |
| `download_finished` | float | Download phase end |
| `detect_started` | float | Architecture detection start |
| `detect_finished` | float | Architecture detection end |
| `quantize_started` | float | Quantization start (includes allocate + quantize + write) |
| `quantize_finished` | float | Quantization end |
| `upload_started` | float | Upload start |
| `upload_finished` | float | Upload end |
| `total_elapsed_seconds` | float | Total wall time from start to finish |

---

## Profiles Reference

Profiles are a 3-tuple of `(CRITICAL_bits, IMPORTANT_bits, COMPRESS_bits)`.

**Tier classification:**
- **CRITICAL** — Attention Q/K/V/O, MLA projections, MoE router gates, shared experts, SSM state matrices, output head
- **IMPORTANT** — Embeddings, VL connectors, SSM timestep/in/out projections
- **COMPRESS** — MLP/FFN layers, routed MoE experts

| Profile | Tiers (C/I/Comp) | Nominal Bits | Best For |
|---------|-----------------|-------------|----------|
| `JANG_1L` | 8/8/2 | 2-bit | Maximum protection at 2-bit. Largest 2-bit output. |
| `JANG_2S` | 6/4/2 | 2-bit | Proven: +28pp MMLU over MLX uniform 2-bit on 122B MoE. |
| `JANG_2M` | 8/4/2 | 2-bit | Balanced 2-bit. Good default for 2-bit MoE. |
| `JANG_2L` | 8/6/2 | 2-bit | Best quality 2-bit. Required for 512+ expert models. |
| `JANG_3S` | 6/3/3 | 3-bit | Tightest 3-bit. |
| `JANG_3M` | 8/3/3 | 3-bit | Balanced 3-bit. |
| `JANG_3L` | 8/4/3 | 3-bit | Quality 3-bit. Good for SSM hybrids. |
| `JANG_3K` | budget-neutral | 3-bit | K-quant: same size as uniform 3-bit, smarter allocation. |
| `JANG_4S` | 6/4/4 | 4-bit | Tight 4-bit. |
| `JANG_4M` | 8/4/4 | 4-bit | Standard 4-bit. |
| `JANG_4L` | 8/6/4 | 4-bit | Quality 4-bit. |
| `JANG_4K` | budget-neutral | 4-bit | **Recommended default.** K-quant: same size as uniform 4-bit. |
| `JANG_5K` | budget-neutral | 5-bit | K-quant 5-bit. |
| `JANG_6M` | 8/6/6 | 6-bit | Near-lossless. |
| `JANG_6K` | budget-neutral | 6-bit | K-quant 6-bit. |

**K-quant profiles** (`JANG_3K`, `JANG_4K`, `JANG_5K`, `JANG_6K`) use budget-neutral allocation: total bits equals uniform quantization at that level, but bits are redistributed from COMPRESS to CRITICAL tensors. Same file size, better quality.

**Architecture-specific auto-adjustments:**
- 150+ experts: `group_size` auto-bumped to 128 (prevents Metal `gather_qmm` cache pressure)
- 512+ experts: MLP asymmetry floors applied (`gate_proj` min 4-bit, `down_proj` min 3-bit)
- 512+ experts + hidden_size >= 4096: bfloat16 activations forced (prevents float16 overflow)

---

## Webhooks

If `webhook_url` is provided when submitting a job, the server will POST the full job JSON to that URL when the job reaches a terminal state (`completed`, `failed`, or `cancelled`).

**Webhook request:**
- Method: `POST`
- Content-Type: `application/json`
- Body: Full [Job object](#response-schema)
- Timeout: 10 seconds

**Example with Slack incoming webhook:**

```bash
curl -X POST http://localhost:8420/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "Qwen/Qwen3-8B",
    "profile": "JANG_4K",
    "user": "alice",
    "webhook_url": "https://hooks.slack.com/services/T.../B.../..."
  }'
```

---

## SSE Streaming

For real-time UI updates without polling, use Server-Sent Events:

```
GET /jobs/{job_id}/stream
```

The stream pushes the full job JSON on every state change (phase transitions, progress updates, architecture detection results, etc.). A keepalive comment is sent every 30 seconds to prevent connection drops.

The stream automatically closes when the job reaches a terminal state.

---

## Error Handling

**HTTP error codes:**

| Code | Meaning |
|------|---------|
| `400` | Bad request (invalid profile, bad model_id format, job in wrong state for operation) |
| `401` | Missing or invalid API key |
| `404` | Job not found, or model not found on HuggingFace |
| `429` | Per-user job limit exceeded |
| `507` | Insufficient disk space |

**Pipeline failures:**

When a job fails mid-pipeline, the `error` field contains the exception message and full Python traceback. The `phase` field shows which phase failed. The `timing` object shows which phases completed and their durations.

Work directories are cleaned up automatically after both success and failure.

**Architecture-specific failures the pipeline handles:**
- FP8 source models (MiniMax, DeepSeek-V3): auto-dequantized to float32 before re-quantizing
- bfloat16 tensors: loaded via raw byte conversion (numpy doesn't support bf16)
- Fused `gate_up_proj` (Qwen3.5 MoE): auto-split into separate tensors
- Per-expert 2D tensors (MiniMax/Mixtral naming): auto-stacked into 3D for QuantizedSwitchLinear
- Large tensors (>100M elements): chunked to avoid Metal OOM
- Wrong `eos_token_id` (Qwen3.5): auto-corrected
