from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_CSV = ROOT / "registry" / "model_registry.csv"


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    display_name: str
    implementation_type: str
    backend: str
    status: str
    source_paper: str
    source_repo: str
    source_ref: str
    input_definition: str
    output_definition: str
    current_task_support: str
    known_deviations: str
    comparable_scope: str
    current_validation_evidence: str
    runner_script: str
    config_path: str
    notes: str


def load_model_registry(path: str | Path = REGISTRY_CSV) -> list[ModelEntry]:
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        return [ModelEntry(**row) for row in rows]


def validate_registry_entries(entries: list[ModelEntry], repo_root: Path = ROOT) -> dict[str, object]:
    errors: list[str] = []
    rows: list[dict[str, object]] = []
    required_text_fields = [
        "model_id",
        "display_name",
        "implementation_type",
        "backend",
        "status",
        "source_repo",
        "source_ref",
        "runner_script",
    ]

    for entry in entries:
        missing = [field for field in required_text_fields if not getattr(entry, field)]
        runner_exists = (repo_root / entry.runner_script).exists() if entry.runner_script else False
        config_exists = (repo_root / entry.config_path).exists() if entry.config_path else False
        if missing:
            errors.append(f"{entry.model_id}: missing required registry fields: {', '.join(missing)}")
        if not runner_exists:
            errors.append(f"{entry.model_id}: missing runner script {entry.runner_script}")
        if entry.config_path and not config_exists:
            errors.append(f"{entry.model_id}: missing config path {entry.config_path}")
        rows.append(
            {
                "model_id": entry.model_id,
                "runner_exists": runner_exists,
                "config_exists": config_exists,
                "implementation_type": entry.implementation_type,
                "status": entry.status,
            }
        )

    return {
        "num_entries": len(entries),
        "rows": rows,
        "errors": errors,
    }
