from __future__ import annotations

import json
from pathlib import Path
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_gate0_gate2_loso_resumable_runner_closure_v1 import (
    HeartbeatWriter,
    JobKey,
    JobLogger,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
)


def main() -> int:
    tmp_root = ROOT / "experiments" / f"loso_io_test_{uuid.uuid4().hex}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        heartbeat_path = tmp_root / "heartbeat.json"
        logger_path = tmp_root / "heartbeat_test.log"
        logger = JobLogger(logger_path)
        job = JobKey(dataset="weissbart_tf64", subject_id="P00", model="eegnet", seed=0)

        writer = HeartbeatWriter(heartbeat_path, batch_interval=1, time_interval_seconds=0)
        for batch in range(1000):
            writer.update(job=job, phase="train", epoch=batch // 100, batch=batch, logger=logger)

        heartbeat_payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))

        import run_gate0_gate2_loso_resumable_runner_closure_v1 as runner_mod

        original_atomic_write_json = runner_mod.atomic_write_json

        heartbeat_call_state = {"count": 0}

        def flaky_heartbeat_atomic_write_json(path: Path, payload: object, *, attempts: int = 5, sleep_seconds: float = 0.05) -> None:
            if Path(path) == heartbeat_path and heartbeat_call_state["count"] == 0:
                heartbeat_call_state["count"] += 1
                raise PermissionError(13, "simulated heartbeat permission denied")
            return original_atomic_write_json(path, payload, attempts=attempts, sleep_seconds=sleep_seconds)

        runner_mod.atomic_write_json = flaky_heartbeat_atomic_write_json
        try:
            writer.update(job=job, phase="val", epoch=99, batch=1001, logger=logger)
        finally:
            runner_mod.atomic_write_json = original_atomic_write_json

        logger_text = logger_path.read_text(encoding="utf-8")
        heartbeat_warning_logged = "heartbeat_warning" in logger_text

        for idx in range(1000):
            atomic_write_json(tmp_root / "state.json", {"i": idx})
            atomic_write_csv(
                tmp_root / "rows.csv",
                [{"i": idx, "label": f"row_{idx}"}],
                ["i", "label"],
            )
            atomic_write_text(tmp_root / "note.txt", f"note_{idx}")

        original_replace = Path.replace
        replace_state = {"remaining_failures": 2}

        def flaky_replace(self: Path, target: Path):
            if self.name.startswith("retry_success.json.") and replace_state["remaining_failures"] > 0:
                replace_state["remaining_failures"] -= 1
                raise PermissionError(13, "simulated replace permission denied")
            return original_replace(self, target)

        Path.replace = flaky_replace
        try:
            atomic_write_json(tmp_root / "retry_success.json", {"status": "ok"})
        finally:
            Path.replace = original_replace

        original_replace = Path.replace
        failure_state = {"remaining_failures": 99}

        def always_fail_replace(self: Path, target: Path):
            if self.name.startswith("retry_fail.json."):
                failure_state["remaining_failures"] -= 1
                raise PermissionError(13, "simulated persistent permission denied")
            return original_replace(self, target)

        persistent_failure_raised = False
        Path.replace = always_fail_replace
        try:
            try:
                atomic_write_json(tmp_root / "retry_fail.json", {"status": "should_fail"})
            except PermissionError:
                persistent_failure_raised = True
        finally:
            Path.replace = original_replace

        result = {
            "heartbeat_write_count": 1000,
            "heartbeat_file_exists": heartbeat_path.exists(),
            "heartbeat_last_payload": heartbeat_payload,
            "heartbeat_permissionerror_downgraded_to_warning": heartbeat_warning_logged,
            "critical_atomic_write_repeat_count": 1000,
            "retry_success_after_transient_permissionerror": json.loads((tmp_root / "retry_success.json").read_text(encoding="utf-8"))["status"] == "ok",
            "persistent_failure_raised_after_retries": persistent_failure_raised,
            "warning_count": writer.warning_count,
        }
        print(json.dumps(result, indent=2))
        return 0
    finally:
        for child in tmp_root.glob("*"):
            if child.is_file():
                child.unlink(missing_ok=True)
        tmp_root.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
