from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_CSV = ROOT / "registry" / "model_registry.csv"


@dataclass(frozen=True)
class ModelEntry:
    model_name: str
    implementation_type: str
    source_paper: str
    source_repo_or_local_path: str
    runner_path: str
    config_path: str
    supported_tasks: str
    current_validation_status: str
    known_deviations: str


def load_model_registry(path: str | Path = REGISTRY_CSV) -> list[ModelEntry]:
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return [ModelEntry(**row) for row in rows]


def validate_registry_entries(entries: list[ModelEntry], repo_root: Path = ROOT) -> dict[str, object]:
    errors: list[str] = []
    rows: list[dict[str, object]] = []
    required_fields = [
        "model_name",
        "implementation_type",
        "source_paper",
        "source_repo_or_local_path",
        "runner_path",
        "config_path",
        "supported_tasks",
        "current_validation_status",
        "known_deviations",
    ]

    for entry in entries:
        missing = [field for field in required_fields if not getattr(entry, field)]
        runner_exists = (repo_root / entry.runner_path).exists() if entry.runner_path else False
        config_exists = (repo_root / entry.config_path).exists() if entry.config_path else False
        if missing:
            errors.append(f"{entry.model_name}: missing required registry fields: {', '.join(missing)}")
        if not runner_exists:
            errors.append(f"{entry.model_name}: missing runner path {entry.runner_path}")
        if not config_exists:
            errors.append(f"{entry.model_name}: missing config path {entry.config_path}")
        rows.append(
            {
                "model_name": entry.model_name,
                "runner_exists": runner_exists,
                "config_exists": config_exists,
                "implementation_type": entry.implementation_type,
                "current_validation_status": entry.current_validation_status,
            }
        )

    return {
        "num_entries": len(entries),
        "rows": rows,
        "errors": errors,
    }
