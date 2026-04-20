"""Ralph Runner entry point — one iteration at a time.

Commands:
  --status     Print status.json summary + exit
  --next       Pick the next pending combo, run it, update state, exit.
  --tier N     Activate tier N; all active combos enter state as 'pending'
  --reset      Clear state.json (keeps results/logs)

Status strings printed to stdout (Ralph reads these):
  "ALL GREEN"         — all combos passed, nothing pending
  "NONE PENDING"      — all combos attempted but some failed
  "COMBO <slug> STARTED" — one combo picked up
  "COMBO <slug> GREEN"   — one combo finished green
  "COMBO <slug> FAILED: <reason>" — one combo failed
  "BLOCKED: <reason>"  — cannot proceed (e.g., macstudio unreachable)
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from .remote import REMOTE_HOST, REMOTE_WORKSPACE, remote_ok, remote_free_gb, run_remote, sync_tree

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "results" / "state.json"
RESULTS_DIR = ROOT / "results"
JANG_REPO_ROOT = ROOT.parent   # /Users/eric/jang


def _yaml() -> YAML:
    return YAML(typ="safe")


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"combos": {}, "active_tier": None, "created": dt.datetime.now().isoformat()}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def slug(model_repo_or_path: str, profile: str) -> str:
    return f"{model_repo_or_path.replace('/', '__').replace(' ', '_')}__{profile}"


def activate_tier(tier: int) -> dict[str, Any]:
    """Merge tier N's combos into state.json as 'pending' (if not already tracked)."""
    models = _yaml().load((ROOT / "models.yaml").read_text())
    profiles = _yaml().load((ROOT / "profiles.yaml").read_text())
    tier_cfg = next((t for t in models["tiers"] if t["tier"] == tier), None)
    if tier_cfg is None:
        raise SystemExit(f"tier {tier} not defined in models.yaml")
    tier_profiles = profiles.get(f"tier_{tier}_profiles", {})
    state = load_state()
    state["active_tier"] = tier
    for m in tier_cfg["models"]:
        key = m.get("hf_repo") or m.get("local_path")
        if m.get("skip"):
            for prof_list in tier_profiles.values():
                for p in prof_list:
                    s = slug(key, p)
                    if s not in state["combos"]:
                        state["combos"][s] = {
                            "model": key, "profile": p,
                            "status": "skipped", "reason": m["skip"],
                            "tier": tier,
                        }
            continue
        for p in tier_profiles.get("jang", []):
            s = slug(key, p)
            if s not in state["combos"]:
                state["combos"][s] = {"model": key, "profile": p, "family": "jang",
                                       "status": "pending", "tier": tier}
        if m.get("supports_jangtq"):
            for p in tier_profiles.get("jangtq", []):
                s = slug(key, p)
                if s not in state["combos"]:
                    state["combos"][s] = {"model": key, "profile": p, "family": "jangtq",
                                           "status": "pending", "tier": tier}
    save_state(state)
    return state


def pick_next(state: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    for s, info in state["combos"].items():
        if info.get("status") == "pending":
            return s, info
    return None


def ensure_source_model(hf_repo: str) -> str:
    """Ensure model is cached on macstudio. Returns the snapshot path."""
    cmd = (
        "python3 -c 'from huggingface_hub import snapshot_download; "
        f"print(snapshot_download(repo_id=\"{hf_repo}\"))'"
    )
    r = run_remote(cmd, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"hf snapshot_download failed: {r.stderr[-500:]}")
    # Take the last line — snapshot_download prints the path
    lines = [l for l in r.stdout.strip().splitlines() if l.strip().startswith("/")]
    if not lines:
        raise RuntimeError(f"no path in snapshot_download output: {r.stdout[-500:]}")
    return lines[-1]


def run_convert_remote(src: str, profile: str, out_slug: str, family: str) -> dict[str, Any]:
    """Run conversion on macstudio. Returns dict with returncode/wall/stdout/stderr."""
    out = f"{REMOTE_WORKSPACE}/out/{out_slug}"
    run_remote(f"rm -rf {out}")
    if family == "jang":
        cmd = (
            f"cd {REMOTE_WORKSPACE}/jang/jang-tools && "
            f"python3 -m jang_tools --progress=json --quiet-text "
            f"convert {src} -o {out} -p {profile} 2>&1"
        )
    else:
        cmd = (
            f"cd {REMOTE_WORKSPACE}/jang/jang-tools && "
            f"python3 -m jang_tools.convert_qwen35_jangtq --progress=json --quiet-text "
            f"{src} {out} {profile} 2>&1"
        )
    t0 = dt.datetime.now()
    r = run_remote(cmd, timeout=7200)
    wall = (dt.datetime.now() - t0).total_seconds()
    return {
        "returncode": r.returncode,
        "wall_time_s": wall,
        "stdout_tail": r.stdout[-4000:],
        "stderr_tail": r.stderr[-4000:],
        "output_path": out,
    }


def cmd_status() -> int:
    state = load_state()
    combos = state.get("combos", {})
    if not combos:
        print("NO COMBOS (run --tier N first)")
        return 0
    counts: dict[str, int] = {}
    for info in combos.values():
        counts[info["status"]] = counts.get(info["status"], 0) + 1
    for k in ("pending", "running", "green", "failed", "skipped"):
        print(f"  {k}: {counts.get(k, 0)}")
    for s, info in combos.items():
        print(f"  - {info['status']:<8} {s}  {info.get('reason', '')}")
    if counts.get("pending", 0) == 0 and counts.get("failed", 0) == 0 and counts.get("running", 0) == 0:
        print("ALL GREEN")
    elif counts.get("pending", 0) == 0 and counts.get("running", 0) == 0:
        print("NONE PENDING")
    return 0


def run_audits_remote(output_path: str, convert_wall_s: float) -> dict[str, Any]:
    """SSH to macstudio + run ralph_runner/audit.py on the converted model dir. Capture JSON."""
    # Make sure ralph_runner/ is on macstudio (same rsync scope as jang-tools)
    print(f"[ralph] sync ralph_runner -> macstudio (for audit)")
    sync_tree(str(JANG_REPO_ROOT / "ralph_runner"), "jang/ralph_runner")
    cmd = (
        f"cd {REMOTE_WORKSPACE}/jang && "
        f"python3 -m ralph_runner.audit "
        f"--model {output_path} "
        f"--convert-wall-s {convert_wall_s:.3f} "
        f"--json"
    )
    print(f"[ralph] audit: {cmd}")
    r = run_remote(cmd, timeout=1800)
    if r.returncode != 0:
        return {
            "overall": "fail",
            "required_fail_count": 1,
            "error": f"audit_invocation_rc={r.returncode}",
            "stderr_tail": r.stderr[-500:],
            "rows": {},
        }
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {
            "overall": "fail",
            "required_fail_count": 1,
            "error": f"audit_output_parse: {e}",
            "stdout_tail": r.stdout[-500:],
            "rows": {},
        }


def cmd_next() -> int:
    if not remote_ok():
        print("BLOCKED: macstudio not reachable via SSH")
        return 0
    state = load_state()
    picked = pick_next(state)
    if picked is None:
        return cmd_status()
    s, info = picked
    # bootstrap workspace on first run
    run_remote(f"mkdir -p {REMOTE_WORKSPACE}/jang {REMOTE_WORKSPACE}/out {REMOTE_WORKSPACE}/logs")
    print(f"[ralph] sync jang-tools -> macstudio")
    sync_tree(str(JANG_REPO_ROOT / "jang-tools"), "jang/jang-tools")
    # mark running
    info["status"] = "running"
    info["started"] = dt.datetime.now().isoformat()
    save_state(state)
    print(f"COMBO {s} STARTED")
    # pick src
    try:
        model = info["model"]
        if model.startswith("/") or model.startswith("~/"):
            src = model.replace("~/", "/Users/eric/")
        else:
            src = ensure_source_model(model)
    except Exception as e:
        info["status"] = "failed"
        info["error"] = f"source_fetch: {e}"
        save_state(state)
        print(f"COMBO {s} FAILED: {info['error']}")
        return 0
    # run convert
    result = run_convert_remote(src, info["profile"], s, info.get("family", "jang"))
    # record
    day = dt.datetime.now().strftime("%Y-%m-%d")
    run_dir = RESULTS_DIR / day / s
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "convert.json").write_text(json.dumps(result, indent=2))
    if result["returncode"] == 0:
        audit_result = run_audits_remote(result["output_path"], result["wall_time_s"])
        (run_dir / "audit.json").write_text(json.dumps(audit_result, indent=2))
        overall = audit_result.get("overall", "fail")
        required_fails = audit_result.get("required_fail_count", 0)
        if overall == "pass" and required_fails == 0:
            info["status"] = "green"
            info["wall_s"] = result["wall_time_s"]
            save_state(state)
            print(f"COMBO {s} GREEN wall={result['wall_time_s']:.1f}s audit=pass")
        else:
            info["status"] = "failed"
            info["error"] = f"audit_failed: {required_fails} required rows failed"
            info["audit_fails"] = [
                {"row": k, "hint": r.get("hint", "")}
                for k, r in audit_result.get("rows", {}).items()
                if r.get("required") and r.get("status") == "fail"
            ]
            save_state(state)
            print(f"COMBO {s} FAILED: audit ({required_fails} required fails)")
    else:
        info["status"] = "failed"
        info["error"] = f"convert_rc={result['returncode']}"
        info["stderr_tail"] = result["stderr_tail"][-500:]
        save_state(state)
        print(f"COMBO {s} FAILED: convert exit {result['returncode']}")
    # cleanup output to free disk
    run_remote(f"rm -rf {result['output_path']}")
    return 0


def cmd_reset() -> int:
    if STATE_PATH.exists():
        STATE_PATH.unlink()
    print("state reset")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="ralph_runner")
    p.add_argument("--tier", type=int, help="Activate tier N")
    p.add_argument("--next", action="store_true", help="Run next pending combo")
    p.add_argument("--status", action="store_true", help="Print status and exit")
    p.add_argument("--reset", action="store_true", help="Clear state.json")
    args = p.parse_args()
    if args.reset:
        sys.exit(cmd_reset())
    if args.tier is not None:
        activate_tier(args.tier)
    if args.status:
        sys.exit(cmd_status())
    if args.next:
        sys.exit(cmd_next())
    # default: status
    sys.exit(cmd_status())


if __name__ == "__main__":
    main()
