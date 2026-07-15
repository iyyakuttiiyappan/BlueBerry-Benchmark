from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torchvision.transforms import functional as VF

from blueberry_multitask.annotations import prepare_annotations
from blueberry_multitask.config import load_config, output_dirs
from blueberry_multitask.datasets import IMAGENET_MEAN, IMAGENET_STD
from blueberry_multitask.ours_centernet import BerryMTLCenterDetNet
from blueberry_multitask.utils import resolve_device, set_seed


CLASS_COLORS = {
    0: "#35a853",  # green_immature
    1: "#f2a7c6",  # pale_pink
    2: "#8e44ad",  # pink_turns_purple
    3: "#2f6fd6",  # fully_ripe
    4: "#b63f45",  # over_ripe
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _model_from_run(config: dict[str, Any], cfg: dict[str, Any], metadata: dict[str, Any], device: torch.device) -> BerryMTLCenterDetNet:
    class_names = list(config["classes"])
    num_classes = len(class_names) + 1
    return BerryMTLCenterDetNet(
        str(metadata.get("model_name", cfg.get("model_name", "convnextv2_tiny.fcmae_ft_in22k_in1k"))),
        num_classes=num_classes,
        pretrained=False,
        decoder_channels=int(cfg.get("decoder_channels", 128)),
        roi_channels=int(cfg.get("roi_channels", 192)),
        roi_size=int(cfg.get("roi_size", 7)),
        detection_classes=int(metadata.get("detection_classes", cfg.get("detection_classes", num_classes - 1))),
        decoupled_decoder=bool(metadata.get("decoupled_decoder", cfg.get("decoupled_decoder", False))),
        dense_count_residual=bool(metadata.get("dense_count_residual", cfg.get("dense_count_residual", False))),
        task_aligned_detection=bool(metadata.get("task_aligned_detection", cfg.get("task_aligned_detection", False))),
        highres_detection=bool(metadata.get("highres_detection", cfg.get("highres_detection", False))),
        roi_global_context=bool(metadata.get("roi_global_context", cfg.get("roi_global_context", False))),
        adapter_fusion=bool(metadata.get("adapter_fusion", cfg.get("adapter_fusion", False))),
        adapter_bottleneck=int(metadata.get("adapter_bottleneck", cfg.get("adapter_bottleneck", 32))),
    ).to(device)


def _latest_final_run(dirs: dict[str, Path]) -> Path:
    root = dirs["analysis"] / "ours" / "runs"
    patterns = [
        "*_berrymtl_specialist_adapter_fusion_uncertainty_seed*",
        "*_berrymtl_specialist_adapter_fusion_seed*",
        "*_berrymtl_specialist_guided_distill_seed*",
        "*_berrymtl_teacher_aligned_det_seed*",
        "*_berrymtl_centerdet_hitile_quality_seed*",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    candidates = [path for path in candidates if (path / "best.pt").exists()]
    if not candidates:
        raise FileNotFoundError(f"No unified checkpoint found under {root}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _teacher_dir_from_metadata(metadata: dict[str, Any], dirs: dict[str, Path]) -> Path:
    raw = metadata.get("teacher_dir")
    if raw:
        path = Path(str(raw))
        if path.exists():
            return path
    return dirs["analysis"] / "specialist_teachers" / "best_individual_514"


def _absolute(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _normalise(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr - float(np.nanmin(arr))
    denom = float(np.nanmax(arr))
    if denom <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / denom, 0.0, 1.0)


def _image_tensor(path: str | Path, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(image).copy()
    tensor = VF.normalize(VF.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0)
    return tensor, arr


def _scaled_boxes(image_row: pd.Series, stem_instances: pd.DataFrame, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    if stem_instances.empty:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    boxes = stem_instances[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32, copy=True)
    boxes[:, [0, 2]] *= float(image_size) / max(float(image_row["aligned_width"]), 1.0)
    boxes[:, [1, 3]] *= float(image_size) / max(float(image_row["aligned_height"]), 1.0)
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, image_size)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, image_size)
    labels = stem_instances["class_index"].to_numpy(dtype=np.int64, copy=True)
    return boxes, labels


def _draw_boxes(
    ax: plt.Axes,
    boxes: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    confidences: np.ndarray | None = None,
    max_boxes: int = 110,
    linewidth: float = 1.2,
) -> None:
    if len(boxes) == 0:
        return
    order = np.arange(len(boxes))
    if confidences is not None and len(confidences) == len(boxes):
        order = np.argsort(confidences)[::-1]
    for idx in order[:max_boxes]:
        box = boxes[int(idx)]
        label = int(labels[int(idx)])
        color = CLASS_COLORS.get(label, "#ffffff")
        rect = plt.Rectangle(
            (box[0], box[1]),
            max(1.0, box[2] - box[0]),
            max(1.0, box[3] - box[1]),
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            alpha=0.88,
        )
        ax.add_patch(rect)


def _draw_corner_text(ax: plt.Axes, text: str, fontsize: int = 13) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color="#111111",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )


def _overlay_heatmap(ax: plt.Axes, image: np.ndarray, heatmap: np.ndarray, cmap: str, text: str = "") -> None:
    norm = _normalise(heatmap)
    ax.imshow(image)
    ax.imshow(norm, cmap=cmap, alpha=np.clip(0.18 + 0.58 * norm, 0.0, 0.72), vmin=0.0, vmax=1.0)
    if text:
        _draw_corner_text(ax, text)


def _teacher_detection_heatmap(frame: pd.DataFrame, image_size: int, max_boxes: int = 220) -> tuple[np.ndarray, int]:
    heat = np.zeros((image_size, image_size), dtype=np.float32)
    if frame.empty:
        return heat, 0
    frame = frame.sort_values("score", ascending=False).head(max_boxes)
    yy_all, xx_all = np.mgrid[0:image_size, 0:image_size]
    for row in frame.itertuples(index=False):
        x1, y1, x2, y2 = float(row.x1), float(row.y1), float(row.x2), float(row.y2)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        sigma = max(3.0, 0.18 * min(max(1.0, x2 - x1), max(1.0, y2 - y1)))
        x_low = max(0, int(cx - 4 * sigma))
        x_high = min(image_size, int(cx + 4 * sigma + 1))
        y_low = max(0, int(cy - 4 * sigma))
        y_high = min(image_size, int(cy + 4 * sigma + 1))
        if x_high <= x_low or y_high <= y_low:
            continue
        patch = np.exp(-(((xx_all[y_low:y_high, x_low:x_high] - cx) ** 2 + (yy_all[y_low:y_high, x_low:x_high] - cy) ** 2) / (2.0 * sigma * sigma)))
        heat[y_low:y_high, x_low:x_high] = np.maximum(heat[y_low:y_high, x_low:x_high], float(row.score) * patch.astype(np.float32))
    return heat, int(len(frame))


def _teacher_mask(mask_path: str | Path, image_size: int) -> np.ndarray:
    path = _absolute(mask_path)
    if not path.exists():
        return np.zeros((image_size, image_size), dtype=np.float32)
    with Image.open(path) as raw:
        mask = raw.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.float32)


def _tensor_map(tensor: torch.Tensor, image_size: int, mode: str = "bilinear") -> np.ndarray:
    value = tensor.detach().float()
    if value.ndim == 3:
        value = value.unsqueeze(0)
    value = F.interpolate(value, size=(image_size, image_size), mode=mode, align_corners=False if mode == "bilinear" else None)
    return value.squeeze().cpu().numpy().astype(np.float32)


def _classification_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    if len(true) == 0:
        return float("nan")
    return float((pred[: len(true)] == true[: len(pred)]).mean())


def _select_examples(images: pd.DataFrame, run_dir: Path, teacher_dir: Path, num_samples: int) -> pd.DataFrame:
    count_path = run_dir / "counting_predictions_test.csv"
    det_path = run_dir / "detection_predictions_test.csv"
    teacher_count_path = teacher_dir / "counting_teacher_test.csv"
    rows = images.copy().reset_index(drop=True)
    rows["image_index"] = np.arange(len(rows))
    if count_path.exists():
        count_df = pd.read_csv(count_path)[["stem", "y_pred", "error"]].rename(columns={"y_pred": "unified_count", "error": "unified_count_error"})
        rows = rows.merge(count_df, on="stem", how="left")
    if teacher_count_path.exists():
        teacher_count = pd.read_csv(teacher_count_path)[["stem", "teacher_count"]]
        rows = rows.merge(teacher_count, on="stem", how="left")
    if det_path.exists():
        det_counts = pd.read_csv(det_path).groupby("batch_image_index").size().rename("unified_det_boxes").reset_index()
        rows = rows.merge(det_counts, left_on="image_index", right_on="batch_image_index", how="left").drop(columns=["batch_image_index"], errors="ignore")
    rows["unified_count"] = rows["unified_count"].fillna(rows["Total"].astype(float))
    rows["teacher_count"] = rows["teacher_count"].fillna(np.nan)
    rows["unified_det_boxes"] = rows["unified_det_boxes"].fillna(0).astype(int)
    rows["count_abs_error"] = (rows["unified_count"].astype(float) - rows["Total"].astype(float)).abs()
    rows["det_count_gap"] = (rows["unified_det_boxes"].astype(float) - rows["Total"].astype(float)).abs()
    rows["rel_count_abs_error"] = rows["count_abs_error"] / (rows["Total"].astype(float) + 1.0)
    rows["rel_det_count_gap"] = rows["det_count_gap"] / (rows["Total"].astype(float) + 1.0)
    rows["evidence_score"] = rows["rel_det_count_gap"] - rows["rel_count_abs_error"] + 0.35 * (rows["Total"].astype(float) / max(float(rows["Total"].max()), 1.0))
    count_cutoff = float(rows["count_abs_error"].quantile(0.50))
    total_cutoff = float(rows["Total"].quantile(0.40))
    candidates = rows[(rows["count_abs_error"] <= count_cutoff) & (rows["Total"] >= total_cutoff)].copy()
    if len(candidates) < max(1, num_samples):
        candidates = rows[rows["count_abs_error"] <= float(rows["count_abs_error"].quantile(0.65))].copy()
    if len(candidates) < max(1, num_samples):
        candidates = rows.copy()
    selected = candidates.sort_values(["evidence_score", "Total"], ascending=[False, False]).head(max(1, num_samples))
    return selected.reset_index(drop=True)


def _copy_outputs(output_dir: Path, dirs: dict[str, Path]) -> None:
    figures_dir = dirs["paper_ready"] / "figures"
    qualitative_dir = dirs["paper_ready"] / "qualitative_montages_large"
    figures_dir.mkdir(parents=True, exist_ok=True)
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "unified_vs_specialist_heatmap_montage_large.jpg",
        "detection_counting_paradox_examples_large.jpg",
    ]:
        source = output_dir / name
        if source.exists():
            shutil.copy2(source, figures_dir / name)
            shutil.copy2(source, qualitative_dir / name)


def _write_explanation(output_dir: Path, dirs: dict[str, Path], run_dir: Path, cka_path: Path | None) -> None:
    cka_text = ""
    if cka_path is not None and cka_path.exists():
        cka = pd.read_csv(cka_path, index_col=0)
        def value(a: str, b: str) -> str:
            if a in cka.index and b in cka.columns:
                return f"{float(cka.loc[a, b]):.3f}"
            return "n/a"
        cka_text = (
            f"\n\nKey CKA values from the unified checkpoint: segmentation-counting={value('segmentation_head', 'counting_head')}, "
            f"detection-classification={value('detection_head', 'classification_roi_head')}, "
            f"detection-counting={value('detection_head', 'counting_head')}."
        )
    text = f"""# Feature-Space and Heatmap Interpretation

Source unified run: `{run_dir}`

The unified feature-space heatmap is a representation-similarity plot, not an accuracy table. Higher off-diagonal CKA means two task heads preserve similar geometry from the shared encoder. This is the visual argument for the unified model: the four outputs are not four unrelated pipelines; they reuse berry-aware evidence before specializing.

The specialist reference heatmap is intentionally diagonal because four individual models have no train-time shared task representation. That is useful as a contrast, but it should be described as a structural reference, not as empirical CKA between specialist backbones.{cka_text}

## Why counting can work when detection is weaker

Detection is an instance-level decision: every berry needs a sufficiently accurate box, class score, and non-suppressed peak. Dense clusters, occlusion, adjacent berries, and class-score/localization mismatch can hurt mAP even when the model sees the berry mass.

Counting is a global/dense task: the model can integrate distributed foreground and density evidence over a cluster. It does not need every berry to survive NMS as a separate high-confidence box. Therefore, a model can underperform on class-sensitive detection mAP while still producing a strong count, especially in crowded blueberry clusters.

Use the montage as follows:

1. Unified detection heatmaps show where the model forms discrete berry peaks.
2. Unified density maps show broader count evidence over berry-heavy regions.
3. Unified segmentation and specialist masks show that dense foreground support remains strong.
4. When density/foreground covers the cluster but detections are sparse or misclassified, the count can remain close to the true count while mAP drops.
"""
    out_md = output_dir / "feature_space_heatmap_interpretation.md"
    out_md.write_text(text, encoding="utf-8")
    tables_dir = dirs["paper_ready"] / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_md, tables_dir / out_md.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create unified-vs-specialist heatmap evidence montages.")
    parser.add_argument("--config", default="configs/fresh_benchmark_514.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--profile-key", default="ours_centerdet_teacher_aligned")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    config = load_config(args.config)
    prepare_annotations(config)
    dirs = output_dirs(config)
    set_seed(int(config.get("training", {}).get("seed", 42)))
    device = resolve_device(args.device)
    cfg = {**config.get("ours", {}), **config.get(args.profile_key, {})}
    run_dir = Path(args.run_dir) if args.run_dir else _latest_final_run(dirs)
    metadata = _load_json(run_dir / "metadata.json")
    image_size = int(metadata.get("image_size", cfg.get("image_size", 768)))
    teacher_dir = _teacher_dir_from_metadata(metadata, dirs)

    output_dir = dirs["paper_ready"] / "feature_space" / run_dir.name / "heatmap_evidence"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _model_from_run(config, cfg, metadata, device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    images = pd.read_csv(dirs["annotations"] / "image_manifest.csv")
    instances = pd.read_csv(dirs["annotations"] / "instances.csv")
    split_images = images[images["split"] == args.split].reset_index(drop=True)
    split_instances = instances[instances["split"] == args.split].reset_index(drop=True)
    instances_by_stem = {str(stem): group.reset_index(drop=True) for stem, group in split_instances.groupby("stem", sort=False)}
    selected = _select_examples(split_images, run_dir, teacher_dir, int(args.num_samples))

    det_teacher = pd.read_csv(teacher_dir / f"detection_teacher_{args.split}.csv") if (teacher_dir / f"detection_teacher_{args.split}.csv").exists() else pd.DataFrame()
    cls_teacher = pd.read_csv(teacher_dir / f"classification_teacher_{args.split}.csv") if (teacher_dir / f"classification_teacher_{args.split}.csv").exists() else pd.DataFrame()
    seg_manifest = pd.read_csv(teacher_dir / "segmentation_teacher_manifest.csv") if (teacher_dir / "segmentation_teacher_manifest.csv").exists() else pd.DataFrame()
    seg_paths = dict(zip(seg_manifest["stem"].astype(str), seg_manifest["teacher_mask_path"].astype(str))) if not seg_manifest.empty else {}

    class_names = list(config["classes"])
    rows_out: list[dict[str, Any]] = []
    fig, axes = plt.subplots(len(selected), 8, figsize=(31.0, 4.5 * len(selected)), squeeze=False, constrained_layout=True)
    titles = [
        "Image + GT boxes",
        "Unified detection\npeak heatmap",
        "Specialist detection\npseudo heatmap",
        "Unified segmentation\nforeground",
        "Specialist segmentation\nmask",
        "Unified counting\ndensity",
        "Unified ROI\nclassification",
        "Specialist ROI\nclassification",
    ]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=18, weight="bold", pad=12)

    for row_idx, image_row in selected.iterrows():
        stem = str(image_row["stem"])
        image_tensor, image_np = _image_tensor(str(image_row["image_path"]), image_size)
        stem_instances = instances_by_stem.get(stem, pd.DataFrame())
        gt_boxes, gt_labels = _scaled_boxes(image_row, stem_instances, image_size)
        box_tensor = torch.as_tensor(gt_boxes, dtype=torch.float32, device=device)
        image_tensor = image_tensor.to(device)

        with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda" and bool(config.get("training", {}).get("amp", True))):
            output = model(image_tensor, boxes=[box_tensor])

        det_prob = torch.sigmoid(output["det_heatmap"].float())
        if "det_quality" in output:
            det_prob = det_prob * torch.sigmoid(output["det_quality"].float())
        det_map = _tensor_map(det_prob.max(dim=1, keepdim=True).values, image_size)
        seg_fg = torch.softmax(output["seg"].float(), dim=1)[:, 1:].sum(dim=1, keepdim=True)
        seg_map = _tensor_map(seg_fg, image_size)
        density = F.softplus(output["density"].float())
        density_map = _tensor_map(density, image_size)
        unified_count = float(output["count"].detach().float().cpu().item())

        cls_pred = np.zeros((0,), dtype=np.int64)
        cls_conf = np.zeros((0,), dtype=np.float32)
        if "cls" in output and output["cls"].numel() > 0:
            probs = torch.softmax(output["cls"].float(), dim=1).detach().cpu().numpy()
            cls_pred = probs.argmax(axis=1).astype(np.int64)
            cls_conf = probs.max(axis=1).astype(np.float32)
        unified_cls_acc = _classification_accuracy(cls_pred, gt_labels)

        teacher_det_frame = det_teacher[det_teacher["stem"].astype(str) == stem] if not det_teacher.empty else pd.DataFrame()
        teacher_det_map, teacher_det_count = _teacher_detection_heatmap(teacher_det_frame, image_size)
        teacher_seg = _teacher_mask(seg_paths.get(stem, ""), image_size)
        teacher_cls_frame = cls_teacher[cls_teacher["stem"].astype(str) == stem] if not cls_teacher.empty else pd.DataFrame()
        teacher_cls_pred = teacher_cls_frame["teacher_y_pred"].to_numpy(dtype=np.int64, copy=True) if not teacher_cls_frame.empty else np.zeros((0,), dtype=np.int64)
        teacher_cls_conf = teacher_cls_frame["teacher_confidence"].to_numpy(dtype=np.float32, copy=True) if not teacher_cls_frame.empty else np.zeros((0,), dtype=np.float32)
        teacher_cls_acc = _classification_accuracy(teacher_cls_pred, gt_labels)
        true_count = float(image_row["Total"])
        teacher_count = float(image_row["teacher_count"]) if pd.notna(image_row.get("teacher_count", np.nan)) else float("nan")

        ax = axes[row_idx, 0]
        ax.imshow(image_np)
        _draw_boxes(ax, gt_boxes, gt_labels, class_names, max_boxes=140, linewidth=1.15)
        _draw_corner_text(ax, f"Example {row_idx + 1}\nGT count {true_count:.0f}", fontsize=12)

        _overlay_heatmap(axes[row_idx, 1], image_np, det_map, "magma", f"peaks from unified\nboxes in CSV {int(image_row['unified_det_boxes'])}")
        _overlay_heatmap(axes[row_idx, 2], image_np, teacher_det_map, "magma", f"specialist boxes\n{teacher_det_count}")
        _overlay_heatmap(axes[row_idx, 3], image_np, seg_map, "viridis", "dense foreground\nshared with count")
        _overlay_heatmap(axes[row_idx, 4], image_np, teacher_seg, "viridis", "specialist mask")
        _overlay_heatmap(axes[row_idx, 5], image_np, density_map, "inferno", f"true {true_count:.0f}\nunified {unified_count:.1f}\nspecialist {teacher_count:.1f}" if np.isfinite(teacher_count) else f"true {true_count:.0f}\nunified {unified_count:.1f}")

        axes[row_idx, 6].imshow(image_np)
        _draw_boxes(axes[row_idx, 6], gt_boxes, cls_pred, class_names, confidences=cls_conf, max_boxes=140, linewidth=1.15)
        _draw_corner_text(axes[row_idx, 6], f"GT-box ROI cls\nacc {unified_cls_acc:.2f}" if np.isfinite(unified_cls_acc) else "GT-box ROI cls")

        axes[row_idx, 7].imshow(image_np)
        _draw_boxes(axes[row_idx, 7], gt_boxes, teacher_cls_pred, class_names, confidences=teacher_cls_conf, max_boxes=140, linewidth=1.15)
        _draw_corner_text(axes[row_idx, 7], f"specialist ROI cls\nacc {teacher_cls_acc:.2f}" if np.isfinite(teacher_cls_acc) else "specialist ROI cls")

        rows_out.append(
            {
                "stem": stem,
                "true_count": true_count,
                "unified_count": unified_count,
                "specialist_count": teacher_count,
                "unified_count_abs_error": abs(unified_count - true_count),
                "unified_detection_csv_boxes": int(image_row["unified_det_boxes"]),
                "specialist_detection_boxes": int(teacher_det_count),
                "unified_gt_roi_class_accuracy": unified_cls_acc,
                "specialist_gt_roi_class_accuracy": teacher_cls_acc,
                "selection_evidence_score": float(image_row["evidence_score"]),
            }
        )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, image_size)
        ax.set_ylim(image_size, 0)

    legend_labels = [f"{idx + 1}: {name}" for idx, name in enumerate(class_names)]
    fig.text(0.5, 0.005, "ROI box colors: " + "   ".join(legend_labels), ha="center", va="bottom", fontsize=15)
    montage_path = output_dir / "unified_vs_specialist_heatmap_montage_large.jpg"
    fig.savefig(montage_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(montage_path, output_dir / "detection_counting_paradox_examples_large.jpg")

    pd.DataFrame(rows_out).to_csv(output_dir / "heatmap_evidence_examples.csv", index=False)
    tables_dir = dirs["paper_ready"] / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_dir / "heatmap_evidence_examples.csv", tables_dir / "heatmap_evidence_examples.csv")
    cka_path = dirs["paper_ready"] / "feature_space" / run_dir.name / "task_space" / "unified_task_cka_similarity.csv"
    _write_explanation(output_dir, dirs, run_dir, cka_path if cka_path.exists() else None)
    _copy_outputs(output_dir, dirs)
    print(f"heatmap_evidence_dir={output_dir}")
    print(f"montage={montage_path}")


if __name__ == "__main__":
    main()
