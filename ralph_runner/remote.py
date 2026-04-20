"""SSH + rsync orchestration for Mac Studio.

Mac Studio host is `macstudio` (Tailscale short name resolving to 100.76.98.16).
Ralph never touches /Volumes/EricsLLMDrive (read-only) or anything outside
~/jang-ralph-workspace/.
"""
from __future__ import annotations
import subprocess
from dataclasses import dataclass
from typing import Iterable, Sequence

REMOTE_HOST = "macstudio"
REMOTE_WORKSPACE = "/Users/eric/jang-ralph-workspace"
DEFAULT_EXCLUDES = (
    ".git", ".venv", "__pycache__", "*.egg-info", "*.pyc",
    "JANGStudio/build", "JANGStudio/DerivedData",
    "ralph_runner/results", "ralph_runner/baselines",
    "*.safetensors",  # don't rsync large weight files around
)


@dataclass
class RemoteResult:
    returncode: int
    stdout: str
    stderr: str


def build_rsync_args(src: str, dst: str, excludes: Iterable[str] | None = None) -> list[str]:
    exc = tuple(excludes) if excludes is not None else DEFAULT_EXCLUDES
    args = ["rsync", "-az", "--delete"]
    for e in exc:
        args += ["--exclude", e]
    args += [src, dst]
    return args


def build_ssh_args(host: str, command: str, timeout: int = 10) -> list[str]:
    return ["ssh", "-o", f"ConnectTimeout={timeout}", "-o", "BatchMode=yes", host, command]


def run_remote(command: str, host: str = REMOTE_HOST, timeout: float = 3600) -> RemoteResult:
    r = subprocess.run(
        build_ssh_args(host, command),
        capture_output=True, text=True, timeout=timeout,
    )
    return RemoteResult(r.returncode, r.stdout, r.stderr)


def sync_tree(local_src: str, remote_subpath: str, timeout: float = 1800) -> RemoteResult:
    """rsync a local directory to macstudio's workspace. local_src should end with /.

    M118 (iter 41): previously had NO timeout. A hung rsync (network glitch,
    macstudio becomes unreachable mid-transfer, disk-full on the remote side)
    stalled the whole Ralph iteration forever — iter 12's M55 lock prevents
    concurrent instances but doesn't address a single hung one. 30-minute
    default matches the expected wall-clock for transferring a jang-tools tree
    (~100 MB) with headroom for Tailscale latency spikes; callers can override
    for tighter budgets.
    """
    if not local_src.endswith("/"):
        local_src += "/"
    dst = f"{REMOTE_HOST}:{REMOTE_WORKSPACE}/{remote_subpath}"
    try:
        r = subprocess.run(build_rsync_args(local_src, dst),
                           capture_output=True, text=True, timeout=timeout)
        return RemoteResult(r.returncode, r.stdout, r.stderr)
    except subprocess.TimeoutExpired as e:
        return RemoteResult(
            returncode=124,   # conventional timeout exit code
            stdout=(e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or "")),
            stderr=f"[timeout] rsync to {dst} exceeded {timeout}s",
        )


def pull_tree(remote_subpath: str, local_dst: str, timeout: float = 1800) -> RemoteResult:
    """rsync a remote subpath down to local_dst. M118: same hang protection
    as sync_tree. Pulls can be big (full audit result dir) so 30 min default
    matches; callers override for tighter budgets.
    """
    src = f"{REMOTE_HOST}:{REMOTE_WORKSPACE}/{remote_subpath}/"
    try:
        r = subprocess.run(build_rsync_args(src, local_dst, excludes=()),
                           capture_output=True, text=True, timeout=timeout)
        return RemoteResult(r.returncode, r.stdout, r.stderr)
    except subprocess.TimeoutExpired as e:
        return RemoteResult(
            returncode=124,
            stdout=(e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or "")),
            stderr=f"[timeout] rsync from {src} exceeded {timeout}s",
        )


def remote_free_gb(host: str = REMOTE_HOST) -> float:
    """Return free disk space on remote home volume (GB)."""
    r = run_remote("df -g ~ | tail -1 | awk '{print $4}'", host=host, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"remote df failed: {r.stderr}")
    return float(r.stdout.strip())


def remote_ok(host: str = REMOTE_HOST) -> bool:
    """Quick health check — is macstudio reachable at all?"""
    r = run_remote("echo ok", host=host, timeout=10)
    return r.returncode == 0 and r.stdout.strip() == "ok"
