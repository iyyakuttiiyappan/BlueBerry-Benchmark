from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blueberry_multitask.config import load_config, output_dirs


TASK_METRICS = {
    "detection": ("map50", True, "mAP50"),
    "segmentation": ("miou_foreground", True, "foreground mIoU"),
    "counting": ("mae", False, "MAE"),
    "classification": ("macro_f1", True, "macro F1"),
}


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_unified(method: Any) -> bool:
    return str(method).startswith("berrymtl_")


def _best_row(frame: pd.DataFrame, task: str, unified: bool | None) -> pd.Series:
    metric, higher_is_better, _ = TASK_METRICS[task]
    subset = frame[frame["task"].astype(str).eq(task)].copy()
    if unified is not None:
        subset = subset[subset["method"].map(_is_unified).eq(unified)].copy()
    subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
    subset = subset.dropna(subset=[metric])
    if subset.empty:
        raise ValueError(f"No rows found for task={task!r}, unified={unified!r}, metric={metric!r}")
    return subset.sort_values(metric, ascending=not higher_is_better).iloc[0]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_if_exists(source: Path, dest: Path) -> None:
    if source.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def _task_metrics_from_row(row: pd.Series, task: str) -> dict[str, Any]:
    run_dir = Path(str(row["run_dir"]))
    metrics = _read_json(run_dir / "test_metrics.json")
    metric, _, _ = TASK_METRICS[task]
    if metric in row and pd.notna(row[metric]):
        metrics[metric] = float(row[metric])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate the best trained specialist for each task into a modular four-task pipeline.")
    parser.add_argument("--config", default="configs/fresh_benchmark_514.yaml")
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    dirs = output_dirs(config)
    paper_tables = dirs["paper_ready"] / "tables"
    all_runs_path = paper_tables / "all_task_runs.csv"
    if not all_runs_path.exists():
        raise FileNotFoundError(all_runs_path)
    runs = pd.read_csv(all_runs_path)

    run_name = args.output_name or f"{_stamp()}_best_specialists_integrated_pipeline"
    out_dir = dirs["analysis"] / "integrated_specialists" / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    specialists = {task: _best_row(runs, task, unified=False) for task in TASK_METRICS}
    best_unified_by_task = {task: _best_row(runs, task, unified=True) for task in TASK_METRICS}

    delta_summary = pd.read_csv(paper_tables / "unified_delta_m_summary.csv")
    best_single_unified_method = str(delta_summary.sort_values("delta_m_percent", ascending=False).iloc[0]["method"])
    single_unified_rows = {
        task: runs[(runs["task"].astype(str).eq(task)) & (runs["method"].astype(str).eq(best_single_unified_method))].iloc[0]
        for task in TASK_METRICS
    }

    integrated_metrics: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for task, (metric, higher_is_better, metric_label) in TASK_METRICS.items():
        specialist = specialists[task]
        best_unified = best_unified_by_task[task]
        single_unified = single_unified_rows[task]
        specialist_score = float(pd.to_numeric(pd.Series([specialist[metric]]), errors="coerce").iloc[0])
        best_unified_score = float(pd.to_numeric(pd.Series([best_unified[metric]]), errors="coerce").iloc[0])
        single_unified_score = float(pd.to_numeric(pd.Series([single_unified[metric]]), errors="coerce").iloc[0])
        best_unified_gap = best_unified_score - specialist_score
        single_unified_gap = single_unified_score - specialist_score
        if not higher_is_better:
            best_unified_gap = specialist_score - best_unified_score
            single_unified_gap = specialist_score - single_unified_score
        comparison_rows.append(
            {
                "task": task,
                "metric": metric_label,
                "higher_is_better": higher_is_better,
                "integrated_specialist_method": str(specialist["method"]),
                "integrated_specialist_display_name": str(specialist["display_name"]),
                "integrated_specialist_score": specialist_score,
                "best_single_unified_method": best_single_unified_method,
                "best_single_unified_display_name": str(single_unified["display_name"]),
                "best_single_unified_score": single_unified_score,
                "single_unified_gap_vs_integrated": single_unified_gap,
                "best_unified_for_task_method": str(best_unified["method"]),
                "best_unified_for_task_display_name": str(best_unified["display_name"]),
                "best_unified_for_task_score": best_unified_score,
                "best_unified_for_task_gap_vs_integrated": best_unified_gap,
            }
        )
        integrated_metrics[task] = {
            **_task_metrics_from_row(specialist, task),
            "method": "best_specialists_integrated_pipeline",
            "display_name": "Best-Specialists Integrated Pipeline",
            "task": task,
            "source_method": str(specialist["method"]),
            "source_display_name": str(specialist["display_name"]),
            "source_run_dir": str(specialist["run_dir"]),
        }

    metadata = {
        "method": "best_specialists_integrated_pipeline",
        "display_name": "Best-Specialists Integrated Pipeline",
        "family": "Modular specialist fusion",
        "is_single_shared_model": False,
        "is_end_to_end_trainable": False,
        "purpose": "Upper-bound modular pipeline using the best already-trained specialist for each task.",
        "config": str(Path(args.config).resolve()),
        "best_single_unified_method": best_single_unified_method,
        "tasks": {
            task: {
                "metric": TASK_METRICS[task][2],
                "source_method": str(row["method"]),
                "source_display_name": str(row["display_name"]),
                "source_run_dir": str(row["run_dir"]),
            }
            for task, row in specialists.items()
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "test_metrics_by_task.json").write_text(json.dumps(integrated_metrics, indent=2), encoding="utf-8")

    for task, row in specialists.items():
        source = Path(str(row["run_dir"]))
        dest = out_dir / task
        _copy_if_exists(source / "test_metrics.json", dest / "test_metrics.json")
        _copy_if_exists(source / "metadata.json", dest / "metadata.json")
        _copy_if_exists(source / "predictions_test.csv", dest / "predictions_test.csv")
        _copy_if_exists(source / "per_class_test.csv", dest / "per_class_test.csv")
        _copy_if_exists(source / "confusion_matrix_test.csv", dest / "confusion_matrix_test.csv")
        _copy_if_exists(source / "classification_report_test.csv", dest / "classification_report_test.csv")
        _copy_if_exists(source / "sample_prediction_overlay.jpg", dest / "sample_prediction_overlay.jpg")

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out_dir / "integrated_specialist_vs_unified.csv", index=False)
    comparison.to_csv(paper_tables / "integrated_specialist_vs_unified.csv", index=False)
    comparison.to_markdown(paper_tables / "integrated_specialist_vs_unified.md", index=False)

    summary_lines = [
        "# Best-Specialists Integrated Pipeline",
        "",
        "This is a modular upper-bound pipeline, not a single shared-backbone model.",
        "It reuses the best trained specialist for each task and compares it with the best single unified model.",
        "",
        comparison.to_markdown(index=False),
        "",
        f"Run directory: `{out_dir.resolve()}`",
    ]
    (paper_tables / "integrated_specialist_pipeline_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"run_dir={out_dir}")
    print(f"comparison={paper_tables / 'integrated_specialist_vs_unified.csv'}")


if __name__ == "__main__":
    main()
