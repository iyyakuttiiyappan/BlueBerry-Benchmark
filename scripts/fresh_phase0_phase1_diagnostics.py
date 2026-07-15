from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from blueberry_multitask.annotations import prepare_annotations
from blueberry_multitask.config import load_config, output_dirs
from blueberry_multitask.metrics import detection_metrics, box_iou
from blueberry_multitask.ours import _split


TASK_METRICS = {
    "detection": ("map50", True),
    "segmentation": ("miou_foreground", True),
    "counting": ("mae", False),
    "classification": ("macro_f1", True),
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_tables(frame: pd.DataFrame, csv_path: Path, md_path: Path | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    if md_path is not None:
        md_path.write_text(frame.to_markdown(index=False), encoding="utf-8")


def _method_cfg(config: dict[str, Any], task: str, method: str) -> dict[str, Any]:
    cfg = dict(config.get("task_defaults", {}).get(task, {}))
    cfg.update(config.get("tasks", {}).get(task, {}).get(method, {}))
    for key in [
        "ours",
        "ours_roi_attention",
        "ours_roi_attention_finetune",
        "ours_centerdet",
        "ours_centerdet_plus",
        "ours_centerdet_agnostic",
        "ours_centerdet_tiletrain",
        "ours_centerdet_decoupled",
        "ours_centerdet_decoupled_residual",
        "ours_centerdet_shared_matched",
        "ours_centerdet_aligned_highres",
        "ours_centerdet_highres_residual",
        "ours_centerdet_hitile_quality",
        "ours_centerdet_teacher_aligned",
    ]:
        section = config.get(key, {})
        if section.get("method") == method:
            merged = dict(section)
            merged.update(cfg)
            return {**cfg, **section}
    return cfg


def _is_unified(method: str) -> bool:
    return str(method).startswith("berrymtl")


def _best_rows(summary: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    output: dict[str, dict[str, pd.Series]] = {}
    for task, (metric, higher) in TASK_METRICS.items():
        frame = summary[(summary["task"] == task) & summary[metric].notna()].copy()
        output[task] = {}
        for role, mask in [("specialist", ~frame["method"].map(_is_unified)), ("unified", frame["method"].map(_is_unified))]:
            candidates = frame[mask].copy()
            if candidates.empty:
                continue
            output[task][role] = candidates.sort_values(metric, ascending=not higher).iloc[0]
    return output


def compute_delta_m(summary: pd.DataFrame, tables: Path) -> pd.DataFrame:
    baselines = _best_rows(summary)
    baseline_rows = []
    for task, roles in baselines.items():
        if "specialist" not in roles:
            continue
        metric, higher = TASK_METRICS[task]
        row = roles["specialist"]
        baseline_rows.append(
            {
                "task": task,
                "metric": metric,
                "higher_is_better": higher,
                "specialist_method": row["method"],
                "specialist_display_name": row.get("display_name", row["method"]),
                "specialist_score": float(row[metric]),
            }
        )
    baseline_df = pd.DataFrame(baseline_rows)
    _write_tables(baseline_df, tables / "delta_m_specialist_references.csv", tables / "delta_m_specialist_references.md")

    unified_methods = sorted(summary[summary["method"].map(_is_unified)]["method"].dropna().unique())
    rows = []
    for method in unified_methods:
        method_rows = summary[summary["method"] == method]
        task_deltas: dict[str, float] = {}
        record: dict[str, Any] = {
            "method": method,
            "display_name": method_rows["display_name"].dropna().iloc[0] if "display_name" in method_rows and method_rows["display_name"].notna().any() else method,
        }
        complete = True
        for task, (metric, higher) in TASK_METRICS.items():
            if task not in baselines or "specialist" not in baselines[task]:
                complete = False
                continue
            task_row = method_rows[method_rows["task"] == task]
            if task_row.empty or pd.isna(task_row.iloc[0].get(metric)):
                complete = False
                continue
            specialist = float(baselines[task]["specialist"][metric])
            unified = float(task_row.iloc[0][metric])
            if abs(specialist) < 1e-12:
                delta = np.nan
            elif higher:
                delta = (unified - specialist) / specialist * 100.0
            else:
                delta = (specialist - unified) / specialist * 100.0
            task_deltas[task] = float(delta)
            record[f"{task}_{metric}"] = unified
            record[f"{task}_delta_percent"] = float(delta)
        record["complete_four_task_variant"] = bool(complete)
        record["delta_m_percent"] = float(np.nanmean(list(task_deltas.values()))) if task_deltas else np.nan
        rows.append(record)
    frame = pd.DataFrame(rows).sort_values("delta_m_percent", ascending=False, na_position="last")
    _write_tables(frame, tables / "unified_delta_m_summary.csv", tables / "unified_delta_m_summary.md")
    return frame


def _image_size_for_run(config: dict[str, Any], run: pd.Series, metadata: dict[str, Any]) -> int:
    if metadata.get("image_size") not in (None, ""):
        return int(float(metadata["image_size"]))
    cfg = _method_cfg(config, "detection", str(run["method"]))
    return int(cfg.get("image_size", config.get("task_defaults", {}).get("detection", {}).get("image_size", 768)))


def _targets_for_detection(images: pd.DataFrame, instances: pd.DataFrame, image_size: int) -> list[dict[str, np.ndarray]]:
    test_images = images.reset_index(drop=True)
    output: list[dict[str, np.ndarray]] = []
    for _, row in test_images.iterrows():
        group = instances[instances["stem"] == row["stem"]]
        if group.empty:
            boxes = np.zeros((0, 4), dtype=float)
            labels = np.zeros((0,), dtype=int)
        else:
            boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float).copy()
            boxes[:, [0, 2]] *= float(image_size) / float(row.aligned_width)
            boxes[:, [1, 3]] *= float(image_size) / float(row.aligned_height)
            labels = group["det_label"].to_numpy(dtype=int)
        output.append({"boxes": boxes, "labels": labels})
    return output


def _predictions_from_csv(path: Path, image_count: int, threshold: float | None = None) -> list[dict[str, np.ndarray]]:
    frame = pd.read_csv(path)
    if threshold is not None and "score" in frame.columns:
        frame = frame[frame["score"] >= float(threshold)].copy()
    output: list[dict[str, np.ndarray]] = []
    for image_idx in range(image_count):
        group = frame[frame["batch_image_index"] == image_idx]
        if group.empty:
            boxes = np.zeros((0, 4), dtype=float)
            labels = np.zeros((0,), dtype=int)
            scores = np.zeros((0,), dtype=float)
        else:
            boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
            labels = group["label"].to_numpy(dtype=int)
            scores = group["score"].to_numpy(dtype=float)
        output.append({"boxes": boxes, "labels": labels, "scores": scores})
    return output


def _agnostic(items: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    output = []
    for item in items:
        output.append(
            {
                "boxes": item["boxes"],
                "labels": np.ones((len(item["boxes"]),), dtype=int),
                "scores": item.get("scores", np.ones((len(item["boxes"]),), dtype=float)),
            }
        )
    return output


def _matched_box_classes(predictions: list[dict[str, np.ndarray]], targets: list[dict[str, np.ndarray]], iou_threshold: float) -> tuple[list[int], list[int], pd.DataFrame]:
    true_labels: list[int] = []
    pred_labels: list[int] = []
    rows = []
    for image_idx, (prediction, target) in enumerate(zip(predictions, targets)):
        pred_boxes = prediction["boxes"]
        pred_cls = prediction["labels"]
        pred_scores = prediction["scores"]
        gt_boxes = target["boxes"]
        gt_cls = target["labels"]
        matched_gt = np.zeros((len(gt_boxes),), dtype=bool)
        order = np.argsort(-pred_scores) if len(pred_scores) else np.asarray([], dtype=int)
        for pred_idx in order:
            if len(gt_boxes) == 0:
                continue
            overlaps = box_iou(pred_boxes[pred_idx : pred_idx + 1], gt_boxes)[0]
            best = int(np.argmax(overlaps))
            if overlaps[best] < iou_threshold or matched_gt[best]:
                continue
            matched_gt[best] = True
            y_true = int(gt_cls[best])
            y_pred = int(pred_cls[pred_idx])
            true_labels.append(y_true)
            pred_labels.append(y_pred)
            rows.append(
                {
                    "batch_image_index": image_idx,
                    "gt_label": y_true,
                    "pred_label": y_pred,
                    "score": float(pred_scores[pred_idx]),
                    "iou": float(overlaps[best]),
                }
            )
    return true_labels, pred_labels, pd.DataFrame(rows)


def detection_split_diagnostics(config: dict[str, Any], detection_summary: pd.DataFrame, tables: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dirs = output_dirs(config)
    classes = list(config["classes"])
    images = _split(pd.read_csv(dirs["annotations"] / "image_manifest.csv"), "test")
    instances_all = pd.read_csv(dirs["annotations"] / "instances.csv")
    instances = instances_all[instances_all["stem"].isin(set(images["stem"]))]

    rows = []
    per_class_rows = []
    best_unified: dict[str, Any] | None = None
    best_unified_map = -np.inf
    for _, run in detection_summary.iterrows():
        run_dir = Path(str(run["run_dir"]))
        pred_path = run_dir / "predictions_test.csv"
        metadata = _read_json(run_dir / "metadata.json")
        if not pred_path.exists():
            continue
        image_size = _image_size_for_run(config, run, metadata)
        targets = _targets_for_detection(images, instances, image_size)
        predictions = _predictions_from_csv(pred_path, len(images))
        loc_metrics, _ = detection_metrics(_agnostic(predictions), _agnostic(targets), 1)
        y_true, y_pred, matched = _matched_box_classes(predictions, targets, iou_threshold=0.50)
        if y_true:
            class_acc = float(accuracy_score(y_true, y_pred))
            class_macro = float(f1_score(y_true, y_pred, labels=list(range(1, len(classes) + 1)), average="macro", zero_division=0))
        else:
            class_acc = 0.0
            class_macro = 0.0
        row = {
            "method": run["method"],
            "display_name": run.get("display_name", run["method"]),
            "is_unified": _is_unified(str(run["method"])),
            "image_size": image_size,
            "class_sensitive_map50": float(run["map50"]),
            "localization_map50": loc_metrics["map50"],
            "localization_map50_95": loc_metrics["map50_95"],
            "localization_precision50": loc_metrics["precision50"],
            "localization_recall50": loc_metrics["recall50"],
            "matched_boxes_iou50": int(len(y_true)),
            "matched_box_class_accuracy": class_acc,
            "matched_box_class_macro_f1": class_macro,
            "run_dir": str(run_dir.resolve()),
        }
        rows.append(row)
        if row["is_unified"] and row["class_sensitive_map50"] > best_unified_map:
            best_unified_map = row["class_sensitive_map50"]
            best_unified = {
                "row": row,
                "y_true": y_true,
                "y_pred": y_pred,
                "matched": matched,
            }

        per_class_path = run_dir / "per_class_test.csv"
        if not per_class_path.exists():
            per_class_path = run_dir / "detection_per_class_test_named.csv"
        if _is_unified(str(run["method"])) and per_class_path.exists():
            per_class = pd.read_csv(per_class_path)
            per_class["method"] = run["method"]
            per_class["display_name"] = run.get("display_name", run["method"])
            if "class_name" not in per_class.columns:
                per_class["class_name"] = per_class["class_id"].map({idx + 1: name for idx, name in enumerate(classes)})
            per_class_rows.append(per_class)

    split = pd.DataFrame(rows).sort_values("class_sensitive_map50", ascending=False)
    rounded = split.copy()
    for col in [
        "class_sensitive_map50",
        "localization_map50",
        "localization_map50_95",
        "localization_precision50",
        "localization_recall50",
        "matched_box_class_accuracy",
        "matched_box_class_macro_f1",
    ]:
        rounded[col] = rounded[col].map(lambda value: round(float(value), 4))
    _write_tables(rounded, tables / "detection_localization_classification_split.csv", tables / "detection_localization_classification_split.md")

    per_class_frame = pd.concat(per_class_rows, ignore_index=True) if per_class_rows else pd.DataFrame()
    if not per_class_frame.empty:
        per_class_frame = per_class_frame.sort_values(["method", "class_id"])
        _write_tables(per_class_frame, tables / "unified_detector_per_class_ap.csv", tables / "unified_detector_per_class_ap.md")

    if best_unified is not None:
        labels = list(range(1, len(classes) + 1))
        matrix = confusion_matrix(best_unified["y_true"], best_unified["y_pred"], labels=labels)
        confusion = pd.DataFrame(matrix, index=classes, columns=classes)
        confusion.to_csv(tables / "unified_detector_matched_box_confusion.csv")
        (tables / "unified_detector_matched_box_confusion.md").write_text(confusion.to_markdown(), encoding="utf-8")
        best_unified["matched"].to_csv(tables / "unified_detector_matched_box_matches.csv", index=False)
    return split, per_class_frame


def threshold_sweep(config: dict[str, Any], detection_summary: pd.DataFrame, tables: Path) -> pd.DataFrame:
    dirs = output_dirs(config)
    images = _split(pd.read_csv(dirs["annotations"] / "image_manifest.csv"), "test")
    instances_all = pd.read_csv(dirs["annotations"] / "instances.csv")
    instances = instances_all[instances_all["stem"].isin(set(images["stem"]))]
    thresholds = [0.0, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]
    rows = []
    for _, run in detection_summary[detection_summary["method"].map(_is_unified)].iterrows():
        run_dir = Path(str(run["run_dir"]))
        pred_path = run_dir / "predictions_test.csv"
        if not pred_path.exists():
            continue
        metadata = _read_json(run_dir / "metadata.json")
        image_size = _image_size_for_run(config, run, metadata)
        targets = _targets_for_detection(images, instances, image_size)
        target_agnostic = _agnostic(targets)
        pred_frame = pd.read_csv(pred_path)
        stored_min_score = float(pred_frame["score"].min()) if not pred_frame.empty and "score" in pred_frame else np.nan
        for threshold in thresholds:
            predictions = _predictions_from_csv(pred_path, len(images), threshold=threshold)
            metrics, _ = detection_metrics(predictions, targets, len(config["classes"]))
            loc_metrics, _ = detection_metrics(_agnostic(predictions), target_agnostic, 1)
            rows.append(
                {
                    "method": run["method"],
                    "display_name": run.get("display_name", run["method"]),
                    "threshold": threshold,
                    "stored_min_score": stored_min_score,
                    "class_sensitive_map50": metrics["map50"],
                    "class_sensitive_recall50": metrics["recall50"],
                    "class_sensitive_precision50": metrics["precision50"],
                    "localization_map50": loc_metrics["map50"],
                    "localization_recall50": loc_metrics["recall50"],
                    "localization_precision50": loc_metrics["precision50"],
                    "run_dir": str(run_dir.resolve()),
                }
            )
    sweep = pd.DataFrame(rows)
    _write_tables(sweep, tables / "unified_detector_threshold_sweep.csv", tables / "unified_detector_threshold_sweep.md")
    if not sweep.empty:
        best = sweep.sort_values(["method", "class_sensitive_map50"], ascending=[True, False]).groupby("method", as_index=False).head(1)
        best = best.sort_values("class_sensitive_map50", ascending=False)
        _write_tables(best, tables / "unified_detector_threshold_sweep_best.csv", tables / "unified_detector_threshold_sweep_best.md")
    return sweep


def protocol_audit(config: dict[str, Any], summary: pd.DataFrame, tables: Path) -> pd.DataFrame:
    dirs = output_dirs(config)
    split_counts = pd.read_csv(dirs["annotations"] / "image_manifest.csv")["split"].value_counts().rename_axis("split").reset_index(name="images")
    split_counts["split_seed"] = int(config.get("split", {}).get("seed", config.get("training", {}).get("seed", 42)))
    _write_tables(split_counts, tables / "locked_split_counts.csv", tables / "locked_split_counts.md")

    best = _best_rows(summary)
    rows = []
    for task, roles in best.items():
        metric, higher = TASK_METRICS[task]
        for role, run in roles.items():
            cfg = _method_cfg(config, task, str(run["method"]))
            rows.append(
                {
                    "task": task,
                    "role": role,
                    "method": run["method"],
                    "display_name": run.get("display_name", run["method"]),
                    "metric": metric,
                    "score": float(run[metric]),
                    "higher_is_better": higher,
                    "image_size": run.get("image_size", cfg.get("image_size")),
                    "model_or_backbone": run.get("resolved_model_name", cfg.get("model_name", "")),
                    "configured_epochs": cfg.get("epochs", config.get("training", {}).get("epochs")),
                    "best_epoch": run.get("best_epoch", ""),
                    "train_seconds": run.get("train_seconds", ""),
                    "tta": bool(run.get("tta", False) in [True, "True", "true", 1, "1"] or "tta" in str(run["method"]).lower()),
                }
            )
    audit = pd.DataFrame(rows)
    _write_tables(audit, tables / "protocol_audit_best_models.csv", tables / "protocol_audit_best_models.md")
    notes = [
        "# Protocol Audit",
        "",
        f"- Split seed: `{config.get('split', {}).get('seed', config.get('training', {}).get('seed', 42))}`.",
        f"- Training seed: `{config.get('training', {}).get('seed', 42)}`.",
        "- All rows use the locked train/validation/test split stored in `outputs/fresh_benchmark/01_annotations/image_manifest.csv`.",
        "- Important remaining differences: specialist detectors use torchvision two-stage/RetinaNet heads at 768 px, while unified CenterDet variants use a ConvNeXtV2 shared backbone at 512 px; the best specialist segmentation result uses flip TTA.",
    ]
    (tables / "protocol_audit_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return audit


def unified_vs_individual_best(summary: pd.DataFrame, detection_split: pd.DataFrame, tables: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_row(task: str, metric_label: str, individual: pd.Series, unified: pd.Series, metric: str, higher: bool) -> None:
        individual_score = float(individual[metric])
        unified_score = float(unified[metric])
        gap = unified_score - individual_score
        if higher:
            percent = abs(gap) / max(abs(individual_score), 1e-12) * 100.0
            outcome = f"{percent:.1f}% {'higher' if gap >= 0 else 'lower'} than individual"
        else:
            percent = abs(gap) / max(abs(individual_score), 1e-12) * 100.0
            outcome = f"{percent:.1f}% {'lower MAE' if gap <= 0 else 'higher MAE'} than individual"
        rows.append(
            {
                "task": task,
                "metric": metric_label,
                "best_individual_model": individual.get("display_name", individual.get("method", "")),
                "individual_score": round(individual_score, 4),
                "best_unified_model": unified.get("display_name", unified.get("method", "")),
                "unified_score": round(unified_score, 4),
                "absolute_gap": round(gap, 4),
                "relative_outcome": outcome,
            }
        )

    if not detection_split.empty:
        for task, metric, label in [
            ("detection_class_sensitive", "class_sensitive_map50", "mAP50 higher is better"),
            ("detection_binary_berry", "localization_map50", "mAP50 higher is better"),
        ]:
            frame = detection_split[detection_split[metric].notna()].copy()
            specialists = frame[~frame["is_unified"].astype(bool)]
            unified = frame[frame["is_unified"].astype(bool)]
            if not specialists.empty and not unified.empty:
                add_row(
                    task,
                    label,
                    specialists.sort_values(metric, ascending=False).iloc[0],
                    unified.sort_values(metric, ascending=False).iloc[0],
                    metric,
                    higher=True,
                )

    best = _best_rows(summary)
    for task, metric_label in [
        ("segmentation", "foreground mIoU higher is better"),
        ("counting", "MAE lower is better"),
        ("classification", "macro F1 higher is better"),
    ]:
        if task not in best or "specialist" not in best[task] or "unified" not in best[task]:
            continue
        metric, higher = TASK_METRICS[task]
        add_row(task, metric_label, best[task]["specialist"], best[task]["unified"], metric, higher=higher)

    frame = pd.DataFrame(rows)
    _write_tables(frame, tables / "unified_vs_individual_best.csv", tables / "unified_vs_individual_best.md")
    return frame


def phase3_decoupled_summary(summary: pd.DataFrame, delta: pd.DataFrame, detection_split: pd.DataFrame, tables: Path) -> pd.DataFrame:
    phase3_methods = [
        "berrymtl_centerdet_agnostic",
        "berrymtl_centerdet_plus",
        "berrymtl_centerdet_tiletrain",
        "berrymtl_centerdet_shared_matched",
        "berrymtl_centerdet_decoupled",
        "berrymtl_centerdet_decoupled_calibrated",
        "berrymtl_centerdet_decoupled_residual",
        "berrymtl_centerdet_decoupled_residual_calibrated",
        "berrymtl_centerdet_aligned_highres",
        "berrymtl_centerdet_aligned_highres_calibrated",
        "berrymtl_centerdet_highres_residual",
        "berrymtl_centerdet_highres_residual_calibrated",
        "berrymtl_centerdet_hitile_quality",
        "berrymtl_centerdet_hitile_quality_calibrated",
        "berrymtl_teacher_aligned_det",
    ]
    rows: list[dict[str, Any]] = []
    baselines = _best_rows(summary)
    specialist_row: dict[str, Any] = {
        "role": "specialist_reference",
        "method": "best_specialists_by_task",
        "display_name": "Best specialist per task",
        "delta_m_percent": 0.0,
    }
    for task, (metric, _) in TASK_METRICS.items():
        if task in baselines and "specialist" in baselines[task]:
            row = baselines[task]["specialist"]
            specialist_row[f"{task}_{metric}"] = float(row[metric])
            specialist_row[f"{task}_reference_method"] = row["method"]
    rows.append(specialist_row)

    for method in phase3_methods:
        method_rows = summary[summary["method"] == method].copy()
        if method_rows.empty:
            continue
        record: dict[str, Any] = {
            "role": "unified_variant",
            "method": method,
            "display_name": method_rows["display_name"].dropna().iloc[0] if method_rows["display_name"].notna().any() else method,
        }
        for task, (metric, higher) in TASK_METRICS.items():
            task_rows = method_rows[(method_rows["task"] == task) & method_rows[metric].notna()].copy()
            if task_rows.empty:
                continue
            task_row = task_rows.sort_values(metric, ascending=not higher).iloc[0]
            record[f"{task}_{metric}"] = float(task_row[metric])
        delta_rows = delta[delta["method"] == method]
        if not delta_rows.empty:
            delta_row = delta_rows.iloc[0]
            for col in [
                "detection_delta_percent",
                "segmentation_delta_percent",
                "counting_delta_percent",
                "classification_delta_percent",
                "delta_m_percent",
            ]:
                if col in delta_row and pd.notna(delta_row[col]):
                    record[col] = float(delta_row[col])
        split_rows = detection_split[detection_split["method"] == method]
        if not split_rows.empty:
            split_row = split_rows.iloc[0]
            for source, target in [
                ("localization_map50", "detection_localization_map50"),
                ("localization_recall50", "detection_localization_recall50"),
                ("matched_box_class_accuracy", "detection_matched_box_class_accuracy"),
                ("matched_box_class_macro_f1", "detection_matched_box_class_macro_f1"),
            ]:
                if source in split_row and pd.notna(split_row[source]):
                    record[target] = float(split_row[source])
        rows.append(record)

    frame = pd.DataFrame(rows)
    metric_cols = [col for col in frame.columns if col not in {"role", "method", "display_name"}]
    for col in metric_cols:
        if pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = frame[col].map(lambda value: round(float(value), 4) if pd.notna(value) else value)
    frame = frame.replace({np.nan: ""})
    _write_tables(frame, tables / "phase3_decoupled_decoder_ablation.csv", tables / "phase3_decoupled_decoder_ablation.md")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 0/1 diagnostics for the fresh unified benchmark.")
    parser.add_argument("--config", default="configs/fresh_benchmark.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    prepare_annotations(config)
    dirs = output_dirs(config)
    tables = dirs["paper_ready"] / "tables"
    summary_path = tables / "all_task_runs.csv"
    detection_path = tables / "detection_summary.csv"
    if not summary_path.exists() or not detection_path.exists():
        raise FileNotFoundError("Run scripts/fresh_summarize.py before diagnostics.")
    summary = pd.read_csv(summary_path)
    detection_summary = pd.read_csv(detection_path)
    delta = compute_delta_m(summary, tables)
    split, _ = detection_split_diagnostics(config, detection_summary, tables)
    sweep = threshold_sweep(config, detection_summary, tables)
    audit = protocol_audit(config, summary, tables)
    comparison = unified_vs_individual_best(summary, split, tables)
    phase3 = phase3_decoupled_summary(summary, delta, split, tables)
    print(f"delta_m_rows={len(delta)}")
    print(f"detection_split_rows={len(split)}")
    print(f"threshold_sweep_rows={len(sweep)}")
    print(f"protocol_audit_rows={len(audit)}")
    print(f"unified_vs_individual_rows={len(comparison)}")
    print(f"phase3_rows={len(phase3)}")
    print(f"tables={tables.resolve()}")


if __name__ == "__main__":
    main()
