from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "draft_zh"


REQUIRED_FILES = [
    PAPER / "main.tex",
    PAPER / "sections" / "introduction_zh.tex",
    PAPER / "sections" / "methods_zh.tex",
    PAPER / "sections" / "benchmark_design_zh.tex",
    PAPER / "sections" / "results_placeholder_zh.tex",
    PAPER / "sections" / "discussion_zh.tex",
    PAPER / "tables" / "model_inventory_table.tex",
    PAPER / "tables" / "dataset_task_matrix.tex",
    PAPER / "tables" / "metric_schema_table.tex",
    PAPER / "figures" / "README.md",
]


def main() -> int:
    missing = [path.relative_to(ROOT).as_posix() for path in REQUIRED_FILES if not path.exists()]
    report = {
        "paper_root": PAPER.relative_to(ROOT).as_posix(),
        "required_files": [path.relative_to(ROOT).as_posix() for path in REQUIRED_FILES],
        "missing_files": missing,
        "ok": len(missing) == 0,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
