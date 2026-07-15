#!/usr/bin/env python
"""Summarize specialist-guided distillation against specialist and unified baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TASK_SPECS = {
    "detection": {
        "metric": "mAP50",
        "column": "map50",
        "higher_is_better": True,
        "specialist_method": "fasterrcnn_resnet50_fpn_thr005",
    },
    "segmentation": {
        "metric": "foreground mIoU",
        "column": "miou_foreground",
        "higher_is_better": True,
        "specialist_method": "fpn_convnextv2_tiny_tta",
    },
    "counting": {
        "metric": "MAE",
        "column": "mae",
        "higher_is_better": False,
        "specialist_method": "count_efficientnetv2_s",
    },
    "classification": {
        "metric": "macro F1",
        "column": "macro_f1",
        "higher_is_better": True,
        "specialist_method": "resnet50",
    },
}

METHODS = [
    {
        "method": "integrated_specialists",
        "display_name": "Integrated best specialists",
        "type": "four-model upper bound",
    },
    {
        "method": "berrymtl_centerdet_hitile_quality",
        "display_name": "BerryMTL-HiTile-QualityDet",
        "type": "single unified model",
    },
    {
        "method": "berrymtl_teacher_aligned_det",
        "display_name": "BerryMTL-TeacherAlignedDet",
        "type": "single unified model",
    },
    {
        "method": "berrymtl_specialist_guided_distill",
        "display_name": "BerryMTL-SpecialistGuidedDistill",
        "type": "single unified model",
    },
    {
        "method": "berrymtl_specialist_adapter_fusion",
        "display_name": "BerryMTL-SpecialistAdapterFusion",
        "type": "single unified model",
    },
    {
        "method": "berrymtl_specialist_adapter_fusion_uncertainty",
        "display_name": "BerryMTL-SpecialistAdapterFusion-UW",
        "type": "single unified model",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="outputs/fresh_benchmark_514",
        help="Benchmark output root.",
    )
    return parser.parse_args()


def _score(
    runs: pd.DataFrame,
    *,
    task: str,
    method: str,
    column: str,
) -> float:
    row = runs[(runs["task"] == task) & (runs["method"] == method)]
    if row.empty:
        raise ValueError(f"Missing {task}/{method} in all_task_runs.csv")
    value = row.iloc[-1][column]
    if pd.isna(value):
        raise ValueError(f"Missing {column} value for {task}/{method}")
    return float(value)


def _relative_delta(score: float, reference: float, higher_is_better: bool) -> float:
    if reference == 0:
        return 0.0
    if higher_is_better:
        return 100.0 * (score - reference) / reference
    return 100.0 * (reference - score) / reference


def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    rounded = df.copy()
    for col in rounded.columns:
        if pd.api.types.is_float_dtype(rounded[col]):
            rounded[col] = rounded[col].map(lambda x: f"{x:.4f}")
    path.write_text(rounded.to_markdown(index=False), encoding="utf-8")


def _plot_delta(comparison: pd.DataFrame, fig_path: Path) -> None:
    plot_df = comparison[comparison["method"] != "integrated_specialists"].copy()
    label_map = {
        "BerryMTL-HiTile-QualityDet": "HiTile\nQualityDet",
        "BerryMTL-TeacherAlignedDet": "Teacher\nAlignedDet",
        "BerryMTL-SpecialistGuidedDistill": "Specialist-Guided\nDistill",
        "BerryMTL-SpecialistAdapterFusion": "Specialist\nAdapterFusion",
        "BerryMTL-SpecialistAdapterFusion-UW": "AdapterFusion\n+ UW",
    }
    labels = [label_map.get(name, name) for name in plot_df["display_name"]]
    values = plot_df["delta_m_percent"]

    colors = ["#4C78A8" if v < 0 else "#2F855A" for v in values]
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 16,
        }
    )
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.8)
    ax.axhline(0, color="#333333", linewidth=1.2)
    ax.set_ylabel("Mean task delta vs specialists (%)")
    ax.set_title("Specialist-Guided Unified Ablations", pad=18)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    max_abs = max(float(values.abs().max()), 1.0)
    y_min = min(float(values.min()) - 0.16 * max_abs, -0.08 * max_abs)
    y_max = max(float(values.max()) + 0.20 * max_abs, 0.08 * max_abs)
    ax.set_ylim(y_min, y_max)
    label_offset = 0.04 * max_abs
    for bar, value in zip(bars, values):
        y = value + (label_offset if value >= 0 else -label_offset)
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:+.2f}%",
            ha="center",
            va=va,
            fontsize=15,
            fontweight="bold",
        )
    ax.tick_params(axis="x", rotation=0)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    table_dir = root / "paper_ready" / "tables"
    figure_dir = root / "paper_ready" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(table_dir / "all_task_runs.csv")

    specialist_scores = {
        task: _score(
            runs,
            task=task,
            method=spec["specialist_method"],
            column=spec["column"],
        )
        for task, spec in TASK_SPECS.items()
    }

    rows: list[dict[str, object]] = []
    task_delta_rows: list[dict[str, object]] = []
    for method in METHODS:
        task_scores: dict[str, float] = {}
        task_deltas: dict[str, float] = {}
        for task, spec in TASK_SPECS.items():
            if method["method"] == "integrated_specialists":
                score = specialist_scores[task]
            else:
                score = _score(
                    runs,
                    task=task,
                    method=method["method"],
                    column=spec["column"],
                )
            delta = _relative_delta(
                score,
                specialist_scores[task],
                spec["higher_is_better"],
            )
            task_scores[task] = score
            task_deltas[task] = delta
            task_delta_rows.append(
                {
                    "method": method["method"],
                    "display_name": method["display_name"],
                    "task": task,
                    "metric": spec["metric"],
                    "score": score,
                    "specialist_score": specialist_scores[task],
                    "delta_vs_specialist_percent": delta,
                }
            )

        rows.append(
            {
                "method": method["method"],
                "display_name": method["display_name"],
                "model_type": method["type"],
                "detection_mAP50": task_scores["detection"],
                "segmentation_fg_mIoU": task_scores["segmentation"],
                "counting_MAE": task_scores["counting"],
                "classification_macro_F1": task_scores["classification"],
                "delta_detection_percent": task_deltas["detection"],
                "delta_segmentation_percent": task_deltas["segmentation"],
                "delta_counting_percent": task_deltas["counting"],
                "delta_classification_percent": task_deltas["classification"],
                "delta_m_percent": sum(task_deltas.values()) / len(task_deltas),
            }
        )

    comparison = pd.DataFrame(rows)
    task_deltas = pd.DataFrame(task_delta_rows)

    comparison_path = table_dir / "specialist_guided_distillation_comparison.csv"
    task_delta_path = table_dir / "specialist_guided_distillation_task_deltas.csv"
    comparison.to_csv(comparison_path, index=False)
    task_deltas.to_csv(task_delta_path, index=False)
    _write_markdown(
        comparison,
        table_dir / "specialist_guided_distillation_comparison.md",
    )
    _write_markdown(
        task_deltas,
        table_dir / "specialist_guided_distillation_task_deltas.md",
    )

    fig_path = figure_dir / "specialist_guided_distillation_delta.png"
    _plot_delta(comparison, fig_path)

    summary = {
        "comparison_csv": str(comparison_path),
        "task_deltas_csv": str(task_delta_path),
        "delta_plot": str(fig_path),
        "best_single_unified_by_delta_m": comparison[
            comparison["model_type"] == "single unified model"
        ]
        .sort_values("delta_m_percent", ascending=False)
        .iloc[0]
        .to_dict(),
    }
    (table_dir / "specialist_guided_distillation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
