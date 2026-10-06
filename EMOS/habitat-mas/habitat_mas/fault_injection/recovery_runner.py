"""Sequential isolated processes with immutable attempts and restartable status."""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from habitat_mas.fault_injection.recovery_batch import (
    GROUPS, ROOT, build_command, digest, jobs, validate_manifest, write_json,
)
from habitat_mas.fault_injection.recovery_analysis import analyze, summarize_attempt, artifact_hashes, artifacts_match


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_one(manifest, job, python, timeout, retry_errors=False):
    directory = Path(manifest["output"]) / "runs" / job["job_id"]
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "status.json"
    threshold = manifest["protocol"]["wait_threshold"]
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status["state"] == "running" and status.get("pid"):
            try:
                os.kill(status["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError(f"Previous worker PID {status['pid']} is still alive; inspect it before resuming")
        if status["state"] == "complete":
            current = summarize_attempt(job, status["attempt"], status["returncode"], threshold)
            if current["valid"] and artifacts_match(status):
                return status
            if not artifacts_match(status):
                current.update(valid=False, success=None, invalid_reason="completed_artifacts_changed")
            status.update(state="error", result=current)
            write_json(status_path, status)
        # A held experiment lock prevents concurrent jobs. A stale running status
        # is deliberately not silently retried; --retry-errors creates a new attempt.
        if not retry_errors:
            return {**status, "state": "error"}
    number = 1
    while (directory / f"attempt_{number:03d}").exists():
        number += 1
    attempt = directory / f"attempt_{number:03d}"
    attempt.mkdir()
    command = build_command(manifest, job["task"], job["group"], job["agent"], job["repeat"], attempt, python)
    write_json(attempt / "command.json", command)
    status = {"state": "running", "attempt": str(attempt), "started_at": time.time(),
              "job_id": job["job_id"], "timed_out": False}
    write_json(status_path, status)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / p) for p in ("habitat-mas", "habitat-lab", "habitat-baselines")]
                                      + [env.get("PYTHONPATH", "")])
    env["PYTHONUNBUFFERED"] = "1"
    process = None
    interrupted = False
    with (attempt / "console.log").open("w") as log:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            status["pid"] = process.pid
            write_json(status_path, status)
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            status["timed_out"] = True
            stop_process(process)
            returncode = 124
        except KeyboardInterrupt:
            interrupted = True
            if process is not None:
                stop_process(process)
            returncode = 130
        except OSError as error:
            log.write(str(error) + "\n")
            returncode = 127
    result = summarize_attempt(job, attempt, returncode, threshold)
    status.update(state="complete" if result["valid"] else "error", returncode=returncode,
                  ended_at=time.time(), result=result, artifact_sha256=artifact_hashes(attempt))
    write_json(attempt / "result.json", status)
    write_json(status_path, status)
    if interrupted:
        raise KeyboardInterrupt
    return status


def run_batch(manifest_path, python, *, limit_tasks=None, timeout=1800,
              retry_errors=False, dry_run=False, groups=None):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    selected_groups = set(GROUPS if groups is None else groups)
    if not selected_groups or not selected_groups.issubset(GROUPS):
        raise ValueError("groups must be a nonempty selection of: " + ", ".join(GROUPS))
    validate_manifest(manifest)
    output = Path(manifest["output"])
    # Advisory locks release automatically after interruption or process death.
    with (output / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another batch runner is active in this experiment")
        lock_manifest = output / "manifest.sha256"
        expected = digest(manifest_path)
        if lock_manifest.exists() and lock_manifest.read_text().strip() != expected:
            raise ValueError("Manifest changed after first run; prepare a new experiment")
        lock_manifest.write_text(expected + "\n")
        scheduled = [job for job in jobs(manifest, limit_tasks) if job["group"] in selected_groups]
        if dry_run:
            commands = [{"job_id": job["job_id"], "command": build_command(
                manifest, job["task"], job["group"], job["agent"], job["repeat"],
                output / "runs" / job["job_id"] / "attempt_001", python)} for job in scheduled]
            write_json(output / "commands.json", commands)
            print(f"Prepared {len(commands)} commands; no simulator or model started.", flush=True)
            return 0
        failed = False
        def interrupt_on_term(signum, frame):
            raise KeyboardInterrupt
        previous_handler = signal.signal(signal.SIGTERM, interrupt_on_term)
        try:
            for index, job in enumerate(scheduled, 1):
                print(f"[{index}/{len(scheduled)}] {job['job_id']}", flush=True)
                status = run_one(manifest, job, python, timeout, retry_errors)
                print(f"  {status['state']}: {status.get('result', {}).get('invalid_reason', '')}", flush=True)
                if status["state"] != "complete":
                    failed = True
                    print(f"Stopped on infrastructure/validation error. Inspect {status['attempt']}/console.log", flush=True)
                    break
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            analyze(manifest)
        return 1 if failed else 0
