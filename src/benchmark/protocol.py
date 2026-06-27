from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_PATH = ROOT / "configs" / "baseline_protocol.json"


@dataclass(frozen=True)
class BaselineProtocol:
    raw: dict

    @property
    def name(self) -> str:
        return str(self.raw["protocol_name"])

    @property
    def dataset_id(self) -> str:
        return str(self.raw["primary_dataset"]["dataset_id"])

    @property
    def primary_metric(self) -> str:
        return str(self.raw["metrics"]["primary"])

    @property
    def stable_main_baselines(self) -> list[str]:
        return list(self.raw["reporting_tiers"]["stable_main_baselines"])

    @property
    def stable_secondary_baselines(self) -> list[str]:
        return list(self.raw["reporting_tiers"]["stable_secondary_baselines"])

    @property
    def advanced_candidate_models(self) -> list[str]:
        return list(self.raw["reporting_tiers"]["advanced_candidate_models"])

    @property
    def diagnostic_only_methods(self) -> list[str]:
        return list(self.raw["reporting_tiers"]["diagnostic_only_methods"])


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> BaselineProtocol:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return BaselineProtocol(raw=raw)
