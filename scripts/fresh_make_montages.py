from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.transforms import functional as TF

from blueberry_multitask.annotations import PALETTE, prepare_annotations
from blueberry_multitask.config import load_config, output_dirs
from blueberry_multitask.datasets import IMAGENET_MEAN, IMAGENET_STD
from blueberry_multitask.metrics import box_iou
from blueberry_multitask.ours_centernet import BerryMTLCenterDetNet
from blueberry_multitask.utils import resolve_device


FONT_SCALE = 1.55

CLASS_COLORS = {
    "green_immature": (64, 160, 43),
    "pale_pink": (241, 146, 178),
    "pink_turns_purple": (138, 79, 184),
    "fully_ripe": (46, 96, 200),
    "over_ripe": (185, 57, 63),
}

ABLATION_METHODS = [
    ("berrymtl_centerdet_agnostic", "Agnostic+ROI"),
    ("berrymtl_centerdet_plus", "Contrastive"),
    ("berrymtl_centerdet_tiletrain", "TileTrain"),
    ("berrymtl_centerdet_shared_matched", "Shared"),
    ("berrymtl_centerdet_decoupled_residual", "Decoupled+Residual"),
    ("berrymtl_centerdet_aligned_highres_calibrated", "Aligned+HighRes"),
    ("berrymtl_centerdet_highres_residual", "HighRes"),
    ("berrymtl_centerdet_hitile_quality", "HiTile+Quality"),
    ("berrymtl_centerdet_hitile_quality_calibrated", "Quality+Refine"),
    ("berrymtl_teacher_aligned_det", "TeacherAligned"),
    ("berrymtl_specialist_adapter_fusion", "AdapterFusion"),
    ("berrymtl_specialist_adapter_fusion_uncertainty", "AdapterFusion-UW"),
]

MAIN_METHOD = "berrymtl_specialist_adapter_fusion_uncertainty"


def _case_label(prefix: str, index: int, *lines: str) -> str:
    visible = [line for line in lines if line]
    return "\n".join([f"{prefix} {index + 1}", *visible])


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    size = max(8, int(round(size * FONT_SCALE)))
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    return default


def _number(value: Any, default: float) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def _add_dominant_class(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "dominant_class" in out.columns:
        return out
    class_columns = [name for name in CLASS_COLORS if name in out.columns]
    if not class_columns:
        out["dominant_class"] = "unknown"
        return out
    values = out[class_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    out["dominant_class"] = values.idxmax(axis=1)
    return out


def _read_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _read_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def _fit(image: Image.Image, width: int, height: int, bg: str = "white") -> Image.Image:
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), bg)
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int, max_lines: int = 2) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
            if len(lines) >= max_lines:
                break
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines


def _cell(
    image: Image.Image,
    title: str,
    lines: list[str] | None = None,
    width: int = 300,
    height: int = 330,
    title_fill: tuple[int, int, int] = (25, 25, 25),
    border: tuple[int, int, int] = (216, 220, 225),
) -> Image.Image:
    lines = lines or []
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width - 1, height - 1), outline=border, width=1)
    title_font = _font(17, bold=True)
    small = _font(13)
    y = 11
    for line in _wrap(draw, title, title_font, width - 18, max_lines=2):
        draw.text((9, y), line, fill=title_fill, font=title_font)
        y += 30
    footer_h = 25 * min(3, len(lines)) + 12 if lines else 0
    image_y = 76
    image_h = max(80, height - image_y - footer_h - 8)
    canvas.paste(_fit(image, width - 16, image_h), (8, image_y))
    y = height - footer_h + 2
    for line in lines[:3]:
        draw.text((9, y), line, fill=(42, 48, 56), font=small)
        y += 24
    return canvas


def _make_matrix(
    headers: list[str],
    row_labels: list[str],
    cells: list[list[Image.Image]],
    output_path: Path,
    title: str,
    cell_w: int = 235,
    cell_h: int = 285,
    row_label_w: int = 180,
    header_h: int = 86,
    title_h: int = 86,
    legend: bool = True,
) -> None:
    cols = len(headers)
    rows = len(row_labels)
    legend_h = 64 if legend else 0
    width = row_label_w + cols * cell_w
    height = title_h + header_h + rows * cell_h + legend_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(24, bold=True)
    header_font = _font(15, bold=True)
    row_font = _font(14, bold=True)
    draw.text((18, 18), title, fill=(22, 26, 32), font=title_font)
    y0 = title_h
    draw.rectangle((0, y0, width, y0 + header_h), fill=(244, 246, 248))
    for col, header in enumerate(headers):
        x = row_label_w + col * cell_w
        for idx, line in enumerate(_wrap(draw, header, header_font, cell_w - 16, max_lines=2)):
            draw.text((x + 8, y0 + 14 + idx * 30), line, fill=(25, 25, 25), font=header_font)
    for row_idx, label in enumerate(row_labels):
        y = title_h + header_h + row_idx * cell_h
        draw.rectangle((0, y, row_label_w, y + cell_h), fill=(250, 250, 250))
        for idx, line in enumerate(_wrap(draw, label, row_font, row_label_w - 18, max_lines=4)):
            draw.text((10, y + 14 + idx * 28), line, fill=(28, 34, 42), font=row_font)
        for col_idx in range(cols):
            canvas.paste(cells[row_idx][col_idx], (row_label_w + col_idx * cell_w, y))
    if legend:
        _draw_legend(canvas, (16, height - legend_h + 9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=94)


def _draw_legend(canvas: Image.Image, xy: tuple[int, int]) -> None:
    draw = ImageDraw.Draw(canvas)
    font = _font(13)
    x, y = xy
    draw.text((x, y + 1), "Class colors:", fill=(55, 55, 55), font=font)
    x += 135
    for class_name, color in CLASS_COLORS.items():
        draw.rectangle((x, y + 4, x + 21, y + 25), fill=color, outline=(80, 80, 80))
        draw.text((x + 28, y + 1), class_name, fill=(45, 45, 45), font=font)
        x += 32 + max(145, len(class_name) * 11)


def _semantic_overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.42) -> Image.Image:
    resized = image.resize((mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized.convert("RGB")).astype(np.float32)
    color = np.zeros_like(rgb)
    for label, palette_color in PALETTE.items():
        if int(label) == 0:
            continue
        color[mask == int(label)] = palette_color
    active = mask > 0
    blended = rgb.copy()
    blended[active] = (1.0 - alpha) * rgb[active] + alpha * color[active]
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def _draw_boxes(
    image: Image.Image,
    boxes: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    scores: np.ndarray | None = None,
    show_scores: bool = False,
    width: int = 3,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(12, bold=True)
    for idx, box in enumerate(boxes):
        label_id = int(labels[idx]) if len(labels) else 1
        class_name = classes[label_id - 1] if 1 <= label_id <= len(classes) else "berry"
        color = CLASS_COLORS.get(class_name, (240, 120, 40))
        x1, y1, x2, y2 = [float(v) for v in box]
        for offset in range(width):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
        if show_scores and scores is not None:
            text = f"{float(scores[idx]):.2f}"
            bbox = draw.textbbox((0, 0), text, font=font)
            tx, ty = int(x1), max(0, int(y1) - (bbox[3] - bbox[1]) - 3)
            draw.rectangle((tx, ty, tx + bbox[2] - bbox[0] + 5, ty + bbox[3] - bbox[1] + 3), fill=color)
            draw.text((tx + 2, ty + 1), text, fill="white", font=font)
    return canvas


def _gt_boxes(instances: pd.DataFrame, stem: str, image_size: int, image_row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    frame = instances[instances["stem"].astype(str) == str(stem)]
    if frame.empty:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=int)
    boxes = frame[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float).copy()
    boxes[:, [0, 2]] *= image_size / float(image_row["aligned_width"])
    boxes[:, [1, 3]] *= image_size / float(image_row["aligned_height"])
    labels = frame["det_label"].to_numpy(dtype=int)
    return boxes, labels


def _profile_for_method(config: dict[str, Any], method: str) -> dict[str, Any]:
    base = method
    for suffix in ["_calibrated", "_sahi_original", "_sahi_resized", "_sahi"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    cfg = dict(config.get("ours", {}))
    for value in config.values():
        if isinstance(value, dict) and value.get("method") == base:
            cfg.update(value)
            break
    return cfg


def _task_row(summary: pd.DataFrame, task: str, method: str) -> pd.Series:
    frame = summary[(summary["task"] == task) & (summary["method"] == method)].copy()
    if frame.empty:
        raise KeyError(f"No {task} row found for method {method}.")
    return frame.iloc[0]


def _available_methods(summary: pd.DataFrame) -> list[tuple[str, str]]:
    available = set(summary["method"].astype(str))
    return [(method, label) for method, label in ABLATION_METHODS if method in available]


def _find_analysis_run(dirs: dict[str, Path], method: str) -> Path:
    runs_dir = dirs["analysis"] / "ours" / "runs"
    candidates = sorted(
        [path for path in runs_dir.glob(f"*_{method}_seed*") if (path / "best.pt").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No analysis run with best.pt found for {method}.")


def _load_model(
    config: dict[str, Any],
    summary: pd.DataFrame,
    dirs: dict[str, Path],
    method: str,
    device: torch.device,
) -> tuple[BerryMTLCenterDetNet, int]:
    row = _task_row(summary, "detection", method)
    cfg = _profile_for_method(config, method)
    image_size = int(_number(row.get("image_size"), float(cfg.get("image_size", 512))))
    resolved = str(row.get("resolved_model_name", ""))
    model_name = resolved.split(":", 1)[1] if ":" in resolved else str(cfg.get("model_name", "convnextv2_tiny.fcmae_ft_in22k_in1k"))
    classes = list(config["classes"])
    detection_classes = int(_number(row.get("detection_classes"), 1 if _bool(row.get("class_agnostic_detection"), False) else len(classes)))
    model = BerryMTLCenterDetNet(
        model_name,
        num_classes=len(classes) + 1,
        pretrained=False,
        decoder_channels=int(cfg.get("decoder_channels", 128)),
        roi_channels=int(cfg.get("roi_channels", 192)),
        roi_size=int(cfg.get("roi_size", 7)),
        detection_classes=detection_classes,
        decoupled_decoder=_bool(row.get("decoupled_decoder"), _bool(cfg.get("decoupled_decoder"), False)),
        dense_count_residual=_bool(row.get("dense_count_residual"), _bool(cfg.get("dense_count_residual"), False)),
        task_aligned_detection=_bool(row.get("task_aligned_detection"), _bool(cfg.get("task_aligned_detection"), False)),
        highres_detection=_bool(row.get("highres_detection"), _bool(cfg.get("highres_detection"), False)),
        roi_global_context=_bool(row.get("roi_global_context"), _bool(cfg.get("roi_global_context"), False)),
        adapter_fusion=_bool(row.get("adapter_fusion"), _bool(cfg.get("adapter_fusion"), False)),
        adapter_bottleneck=int(_number(row.get("adapter_bottleneck"), float(cfg.get("adapter_bottleneck", 32)))),
    ).to(device)
    run_dir = _find_analysis_run(dirs, method)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model, image_size


@torch.inference_mode()
def _predict_segmentations(
    config: dict[str, Any],
    summary: pd.DataFrame,
    dirs: dict[str, Path],
    methods: list[str],
    selected_images: pd.DataFrame,
    device: torch.device,
) -> dict[tuple[str, str], np.ndarray]:
    output: dict[tuple[str, str], np.ndarray] = {}
    for method in methods:
        model, image_size = _load_model(config, summary, dirs, method, device)
        for row in selected_images.itertuples(index=False):
            image = _read_rgb(row.image_path).resize((image_size, image_size), Image.Resampling.BILINEAR)
            tensor = TF.normalize(TF.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0).to(device)
            pred = model(tensor)["seg"].argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
            output[(method, str(row.stem))] = pred
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output


def _prediction_run(dirs: dict[str, Path], method: str) -> Path:
    run = _find_analysis_run(dirs, method)
    return run


def _load_detection_predictions(dirs: dict[str, Path], method: str) -> pd.DataFrame:
    run = _prediction_run(dirs, method)
    path = run / "detection_predictions_test.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _load_count_predictions(dirs: dict[str, Path], method: str) -> pd.DataFrame:
    run = _prediction_run(dirs, method)
    return pd.read_csv(run / "counting_predictions_test.csv")


def _load_class_predictions(dirs: dict[str, Path], method: str) -> pd.DataFrame:
    run = _prediction_run(dirs, method)
    return pd.read_csv(run / "classification_predictions_test.csv")


def _select_images(image_df: pd.DataFrame, count_predictions: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    test = image_df[image_df["split"] == "test"].copy().reset_index(drop=True)
    counts = count_predictions[["stem", "error"]].copy() if "error" in count_predictions.columns else pd.DataFrame()
    if not counts.empty:
        test = test.merge(counts, on="stem", how="left")
    else:
        test["error"] = 0.0
    selected: list[int] = []
    quantiles = np.linspace(0.18, 0.92, n)
    for q in quantiles:
        target = float(test["Total"].quantile(q))
        candidates = test[~test.index.isin(selected)].copy()
        if candidates.empty:
            break
        score = (candidates["Total"] - target).abs() + 0.20 * candidates["error"].fillna(0).abs()
        selected.append(int(score.idxmin()))
    return _add_dominant_class(test.loc[selected].reset_index(drop=True))


def _detection_frame_for_stem(
    pred_df: pd.DataFrame,
    stem_to_index: dict[str, int],
    stem: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = stem_to_index[str(stem)]
    frame = pred_df[pred_df["batch_image_index"] == idx]
    if frame.empty:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=int), np.zeros((0,), dtype=float)
    return (
        frame[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float),
        frame["label"].to_numpy(dtype=int),
        frame["score"].to_numpy(dtype=float),
    )


def _count_for_stem(pred_df: pd.DataFrame, stem: str) -> tuple[float, float, float]:
    row = pred_df[pred_df["stem"].astype(str) == str(stem)].iloc[0]
    return float(row["y_true"]), float(row["y_pred"]), float(row["error"])


def _error_overlay(image: Image.Image, truth: np.ndarray, pred: np.ndarray) -> Image.Image:
    resized = image.resize((truth.shape[1], truth.shape[0]), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized.convert("RGB")).astype(np.float32)
    error = rgb.copy()
    truth_fg = truth > 0
    pred_fg = pred > 0
    tp = truth_fg & pred_fg & (truth == pred)
    class_confusion = truth_fg & pred_fg & (truth != pred)
    fn = truth_fg & ~pred_fg
    fp = ~truth_fg & pred_fg
    error[tp] = (0.55 * error[tp] + 0.45 * np.array([73, 170, 90])).clip(0, 255)
    error[class_confusion] = (0.45 * error[class_confusion] + 0.55 * np.array([245, 180, 70])).clip(0, 255)
    error[fn] = (0.35 * error[fn] + 0.65 * np.array([40, 100, 220])).clip(0, 255)
    error[fp] = (0.35 * error[fp] + 0.65 * np.array([220, 55, 60])).clip(0, 255)
    return Image.fromarray(error.clip(0, 255).astype(np.uint8))


def _match_counts_agnostic(pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray) -> dict[str, int]:
    order = np.argsort(-pred_scores) if len(pred_scores) else np.asarray([], dtype=int)
    matched = np.zeros((len(gt_boxes),), dtype=bool)
    tp = fp = 0
    for pred_idx in order:
        if len(gt_boxes) == 0:
            fp += 1
            continue
        overlaps = box_iou(pred_boxes[pred_idx : pred_idx + 1], gt_boxes)[0]
        best = int(np.argmax(overlaps))
        if overlaps[best] >= 0.50 and not matched[best]:
            matched[best] = True
            tp += 1
        else:
            fp += 1
    return {"tp": int(tp), "fp": int(fp), "fn": int((~matched).sum())}


def _with_instance_index(instances: pd.DataFrame) -> pd.DataFrame:
    frame = instances.copy()
    frame["within_image_index"] = frame.groupby("stem", sort=False).cumcount()
    return frame


def _crop_prediction_rows(instances: pd.DataFrame, class_preds: pd.DataFrame) -> pd.DataFrame:
    inst = _with_instance_index(instances)
    preds = class_preds.copy()
    preds["within_image_index"] = preds["instance_index"].astype(int)
    merged = inst.merge(
        preds[["stem", "within_image_index", "true_class", "pred_class", "confidence"]],
        on=["stem", "within_image_index"],
        how="inner",
        suffixes=("", "_pred"),
    )
    return merged


def _classification_crop_strip(
    rows: pd.DataFrame,
    max_items: int = 6,
    cols: int | None = None,
    cell_w: int = 142,
    cell_h: int = 190,
) -> Image.Image:
    cells: list[Image.Image] = []
    for row in rows.head(max_items).itertuples(index=False):
        crop = _read_rgb(row.crop_path)
        correct = str(row.true_class) == str(row.pred_class)
        border = CLASS_COLORS.get(str(row.true_class), (180, 180, 180))
        title = f"{row.true_class}"
        lines = [f"pred {row.pred_class}", f"conf {float(row.confidence):.2f}", "correct" if correct else "wrong"]
        cell = _cell(crop, title, lines, width=cell_w, height=cell_h, border=border)
        cells.append(cell)
    if not cells:
        return Image.new("RGB", (cell_w, cell_h), "white")
    cols = int(cols or len(cells))
    rows_n = int(math.ceil(len(cells) / cols))
    canvas = Image.new("RGB", (cols * cell_w, rows_n * cell_h), "white")
    for idx, cell in enumerate(cells):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        canvas.paste(cell, (x, y))
    return canvas


def make_task_montages(
    config: dict[str, Any],
    summary: pd.DataFrame,
    dirs: dict[str, Path],
    image_df: pd.DataFrame,
    instances: pd.DataFrame,
    selected: pd.DataFrame,
    seg_predictions: dict[tuple[str, str], np.ndarray],
    out_dir: Path,
) -> list[Path]:
    classes = list(config["classes"])
    out: list[Path] = []
    test_images = image_df[image_df["split"] == "test"].reset_index(drop=True)
    stem_to_index = {str(row.stem): idx for idx, row in enumerate(test_images.itertuples(index=False))}
    det_preds = _load_detection_predictions(dirs, MAIN_METHOD)
    count_preds = _load_count_predictions(dirs, MAIN_METHOD)
    cls_preds = _load_class_predictions(dirs, MAIN_METHOD)
    crop_rows = _crop_prediction_rows(instances[instances["split"] == "test"], cls_preds)
    method_row = _task_row(summary, "detection", MAIN_METHOD)
    image_size = int(_number(method_row.get("image_size"), 512))

    headers = ["RGB", "Detection", "Segmentation", "Counting", "Classification crops"]
    rows: list[str] = []
    matrix: list[list[Image.Image]] = []
    for sample_idx, row in enumerate(selected.itertuples(index=False)):
        stem = str(row.stem)
        image_row = image_df[image_df["stem"].astype(str) == stem].iloc[0]
        image = _read_rgb(row.image_path).resize((image_size, image_size), Image.Resampling.BILINEAR)
        gt_boxes, gt_labels = _gt_boxes(instances, stem, image_size, image_row)
        pred_boxes, pred_labels, pred_scores = _detection_frame_for_stem(det_preds, stem_to_index, stem)
        truth_mask = _read_mask(row.semantic_mask_path)
        truth_mask = np.asarray(Image.fromarray(truth_mask).resize((image_size, image_size), Image.Resampling.NEAREST))
        pred_mask = seg_predictions[(MAIN_METHOD, stem)]
        y_true, y_pred, error = _count_for_stem(count_preds, stem)
        crops = crop_rows[crop_rows["stem"].astype(str) == stem].sort_values("confidence", ascending=False)
        rows.append(_case_label("Example", sample_idx, f"true count {int(y_true)}"))
        matrix.append(
            [
                _cell(image, "Original", [f"dominant {row.dominant_class}", f"berries {int(row.Total)}"], width=355, height=455),
                _cell(
                    _draw_boxes(image, pred_boxes, pred_labels, classes, pred_scores, show_scores=False),
                    "Predicted boxes",
                    [f"pred boxes {len(pred_boxes)}", f"GT boxes {len(gt_boxes)}"],
                    width=355,
                    height=455,
                ),
                _cell(
                    _semantic_overlay(image, pred_mask),
                    "Predicted mask",
                    [f"GT fg pixels {int((truth_mask > 0).sum())}", f"pred fg pixels {int((pred_mask > 0).sum())}"],
                    width=355,
                    height=455,
                ),
                _cell(image, "Count regression", [f"true {y_true:.0f}", f"pred {y_pred:.1f}", f"error {error:+.1f}"], width=355, height=455),
                _cell(
                    _classification_crop_strip(crops, max_items=1, cols=1, cell_w=300, cell_h=360),
                    "ROI classes",
                    [f"shown {min(1, len(crops))} crop"],
                    width=355,
                    height=455,
                ),
            ]
        )
    path = out_dir / "unified_four_task_examples_large.jpg"
    _make_matrix(headers, rows, matrix, path, title="Unified BerryMTL Qualitative Examples", cell_w=355, cell_h=455, row_label_w=255)
    out.append(path)

    task_specs = [
        ("detection_examples_unified_large.jpg", "Detection examples", ["Ground truth", "Prediction"]),
        ("segmentation_examples_unified_large.jpg", "Segmentation examples", ["Ground truth", "Prediction"]),
        ("counting_examples_unified_large.jpg", "Counting examples", ["Image", "Count"]),
        ("classification_examples_unified_large.jpg", "Classification examples", ["Image crops"]),
    ]
    for filename, title, headers_task in task_specs:
        rows_task: list[str] = []
        matrix_task: list[list[Image.Image]] = []
        for sample_idx, row in enumerate(selected.itertuples(index=False)):
            stem = str(row.stem)
            image_row = image_df[image_df["stem"].astype(str) == stem].iloc[0]
            image = _read_rgb(row.image_path).resize((image_size, image_size), Image.Resampling.BILINEAR)
            rows_task.append(_case_label("Example", sample_idx, f"Total {int(row.Total)}"))
            if filename.startswith("detection"):
                gt_boxes, gt_labels = _gt_boxes(instances, stem, image_size, image_row)
                pred_boxes, pred_labels, pred_scores = _detection_frame_for_stem(det_preds, stem_to_index, stem)
                matrix_task.append(
                    [
                        _cell(_draw_boxes(image, gt_boxes, gt_labels, classes), "Ground truth", [f"GT boxes {len(gt_boxes)}"], width=520, height=470),
                        _cell(_draw_boxes(image, pred_boxes, pred_labels, classes, pred_scores), "Prediction", [f"pred boxes {len(pred_boxes)}"], width=520, height=470),
                    ]
                )
            elif filename.startswith("segmentation"):
                truth_mask = _read_mask(row.semantic_mask_path)
                truth_mask = np.asarray(Image.fromarray(truth_mask).resize((image_size, image_size), Image.Resampling.NEAREST))
                pred_mask = seg_predictions[(MAIN_METHOD, stem)]
                matrix_task.append(
                    [
                        _cell(_semantic_overlay(image, truth_mask), "Ground truth", [f"foreground {int((truth_mask > 0).sum())} px"], width=520, height=470),
                        _cell(_semantic_overlay(image, pred_mask), "Prediction", [f"foreground {int((pred_mask > 0).sum())} px"], width=520, height=470),
                    ]
                )
            elif filename.startswith("counting"):
                y_true, y_pred, error = _count_for_stem(count_preds, stem)
                matrix_task.append(
                    [
                        _cell(image, "Image", [f"dominant {row.dominant_class}"], width=520, height=470),
                        _cell(image, "Predicted count", [f"true {y_true:.0f}", f"pred {y_pred:.1f}", f"error {error:+.1f}"], width=520, height=470),
                    ]
                )
            else:
                crops = crop_rows[crop_rows["stem"].astype(str) == stem].sort_values("confidence", ascending=False)
                matrix_task.append(
                    [
                        _cell(
                            _classification_crop_strip(crops, max_items=6, cols=3, cell_w=250, cell_h=360),
                            "ROI crop predictions",
                            [f"shown {min(6, len(crops))} crops"],
                            width=900,
                            height=820,
                        )
                    ]
                )
        path = out_dir / filename
        _make_matrix(
            headers_task,
            rows_task,
            matrix_task,
            path,
            title=title,
            cell_w=520 if len(headers_task) == 2 else 900,
            cell_h=470 if len(headers_task) == 2 else 820,
            row_label_w=255,
        )
        out.append(path)
    return out


def make_ablation_montages(
    config: dict[str, Any],
    summary: pd.DataFrame,
    dirs: dict[str, Path],
    image_df: pd.DataFrame,
    instances: pd.DataFrame,
    selected: pd.DataFrame,
    seg_predictions: dict[tuple[str, str], np.ndarray],
    out_dir: Path,
) -> list[Path]:
    classes = list(config["classes"])
    methods = _available_methods(summary)
    method_ids = [method for method, _ in methods]
    test_images = image_df[image_df["split"] == "test"].reset_index(drop=True)
    stem_to_index = {str(row.stem): idx for idx, row in enumerate(test_images.itertuples(index=False))}
    headers = ["GT"] + [label for _, label in methods]
    row_labels = [_case_label("Example", idx, f"Total {int(row.Total)}") for idx, row in enumerate(selected.itertuples(index=False))]
    out: list[Path] = []
    det_w, det_h = 330, 430
    count_w, count_h = 330, 355

    det_preds_by_method = {method: _load_detection_predictions(dirs, method) for method in method_ids}
    det_cells: list[list[Image.Image]] = []
    for row in selected.itertuples(index=False):
        stem = str(row.stem)
        method_row = _task_row(summary, "detection", method_ids[0])
        image_size = int(_number(method_row.get("image_size"), 512))
        image_row = image_df[image_df["stem"].astype(str) == stem].iloc[0]
        image = _read_rgb(row.image_path).resize((image_size, image_size), Image.Resampling.BILINEAR)
        gt_boxes, gt_labels = _gt_boxes(instances, stem, image_size, image_row)
        row_cells = [_cell(_draw_boxes(image, gt_boxes, gt_labels, classes), "GT boxes", [f"{len(gt_boxes)} boxes"], width=det_w, height=det_h)]
        for method, label in methods:
            pred_boxes, pred_labels, pred_scores = _detection_frame_for_stem(det_preds_by_method[method], stem_to_index, stem)
            row_cells.append(
                _cell(
                    _draw_boxes(image, pred_boxes, pred_labels, classes, pred_scores),
                    label,
                    [f"pred {len(pred_boxes)}", f"mAP50 {_number(_task_row(summary, 'detection', method).get('map50'), 0):.3f}"],
                    width=det_w,
                    height=det_h,
                )
            )
        det_cells.append(row_cells)
    path = out_dir / "ablation_detection_comparison_montage_large.jpg"
    _make_matrix(headers, row_labels, det_cells, path, title="Detection Ablation Visual Comparison", cell_w=det_w, cell_h=det_h, row_label_w=255)
    out.append(path)

    seg_cells: list[list[Image.Image]] = []
    for row in selected.itertuples(index=False):
        stem = str(row.stem)
        image_size = int(_number(_task_row(summary, "detection", method_ids[0]).get("image_size"), 512))
        image = _read_rgb(row.image_path).resize((image_size, image_size), Image.Resampling.BILINEAR)
        truth_mask = _read_mask(row.semantic_mask_path)
        truth_mask = np.asarray(Image.fromarray(truth_mask).resize((image_size, image_size), Image.Resampling.NEAREST))
        row_cells = [_cell(_semantic_overlay(image, truth_mask), "GT mask", [f"fg {int((truth_mask > 0).sum())} px"], width=det_w, height=det_h)]
        for method, label in methods:
            pred = seg_predictions[(method, stem)]
            row_cells.append(
                _cell(
                    _semantic_overlay(image, pred),
                    label,
                    [f"mIoUfg {_number(_task_row(summary, 'segmentation', method).get('miou_foreground'), 0):.3f}"],
                    width=det_w,
                    height=det_h,
                )
            )
        seg_cells.append(row_cells)
    path = out_dir / "ablation_segmentation_comparison_montage_large.jpg"
    _make_matrix(headers, row_labels, seg_cells, path, title="Segmentation Ablation Visual Comparison", cell_w=det_w, cell_h=det_h, row_label_w=255)
    out.append(path)

    count_preds_by_method = {method: _load_count_predictions(dirs, method) for method in method_ids}
    count_cells: list[list[Image.Image]] = []
    for row in selected.itertuples(index=False):
        stem = str(row.stem)
        image = _read_rgb(row.image_path)
        y_true = float(row.Total)
        row_cells = [_cell(image, "Ground truth", [f"count {y_true:.0f}"], width=count_w, height=count_h)]
        for method, label in methods:
            truth, pred, error = _count_for_stem(count_preds_by_method[method], stem)
            mae = _number(_task_row(summary, "counting", method).get("mae"), 0)
            row_cells.append(_cell(image, label, [f"pred {pred:.1f}", f"error {error:+.1f}", f"MAE {mae:.2f}"], width=count_w, height=count_h))
        count_cells.append(row_cells)
    path = out_dir / "ablation_counting_comparison_montage_large.jpg"
    _make_matrix(headers, row_labels, count_cells, path, title="Counting Ablation Visual Comparison", cell_w=count_w, cell_h=count_h, row_label_w=255, legend=False)
    out.append(path)

    class_cells = _classification_ablation_cells(config, summary, dirs, instances, methods, out_dir)
    out.extend(class_cells)
    out.append(_metric_ablation_summary(summary, methods, out_dir))
    return out


def make_failure_montages(
    config: dict[str, Any],
    summary: pd.DataFrame,
    dirs: dict[str, Path],
    image_df: pd.DataFrame,
    instances: pd.DataFrame,
    seg_predictions: dict[tuple[str, str], np.ndarray],
    failure_sets: dict[str, pd.DataFrame],
    out_dir: Path,
) -> list[Path]:
    classes = list(config["classes"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    test_images = image_df[image_df["split"] == "test"].reset_index(drop=True)
    stem_to_index = {str(row.stem): idx for idx, row in enumerate(test_images.itertuples(index=False))}
    det_preds = _load_detection_predictions(dirs, MAIN_METHOD)
    count_preds = _load_count_predictions(dirs, MAIN_METHOD)
    cls_preds = _load_class_predictions(dirs, MAIN_METHOD)
    crop_rows = _crop_prediction_rows(instances[instances["split"] == "test"], cls_preds)
    image_size = int(_number(_task_row(summary, "detection", MAIN_METHOD).get("image_size"), 512))

    det_cells: list[list[Image.Image]] = []
    det_rows: list[str] = []
    for case_idx, row in enumerate(failure_sets["detection"].itertuples(index=False)):
        stem = str(row.stem)
        image_row = image_df[image_df["stem"].astype(str) == stem].iloc[0]
        image = _read_rgb(row.image_path).resize((image_size, image_size), Image.Resampling.BILINEAR)
        gt_boxes, gt_labels = _gt_boxes(instances, stem, image_size, image_row)
        pred_boxes, pred_labels, pred_scores = _detection_frame_for_stem(det_preds, stem_to_index, stem)
        stats = _match_counts_agnostic(pred_boxes, pred_scores, gt_boxes)
        det_rows.append(_case_label("Failure", case_idx, f"GT {len(gt_boxes)} / pred {len(pred_boxes)}"))
        det_cells.append(
            [
                _cell(_draw_boxes(image, gt_boxes, gt_labels, classes), "Ground truth", [f"GT boxes {len(gt_boxes)}"], width=520, height=470),
                _cell(_draw_boxes(image, pred_boxes, pred_labels, classes, pred_scores), "Prediction", [f"pred boxes {len(pred_boxes)}"], width=520, height=470),
                _failure_text_cell(
                    "Detection failure",
                    [f"TP {stats['tp']}", f"FP {stats['fp']}", f"FN {stats['fn']}", "Dense scenes create many low-quality proposals."],
                    width=520,
                    height=470,
                ),
            ]
        )
    path = out_dir / "detection_failure_cases_large.jpg"
    _make_matrix(["GT boxes", "Prediction", "Failure signal"], det_rows, det_cells, path, title="Detection Failure Cases", cell_w=520, cell_h=470, row_label_w=260)
    out.append(path)

    seg_cells: list[list[Image.Image]] = []
    seg_rows: list[str] = []
    for case_idx, row in enumerate(failure_sets["segmentation"].itertuples(index=False)):
        stem = str(row.stem)
        image = _read_rgb(row.image_path).resize((image_size, image_size), Image.Resampling.BILINEAR)
        truth = _read_mask(row.semantic_mask_path)
        truth = np.asarray(Image.fromarray(truth).resize((image_size, image_size), Image.Resampling.NEAREST))
        pred = seg_predictions[(MAIN_METHOD, stem)]
        fg_iou = float(((truth > 0) & (pred > 0)).sum() / max(1, ((truth > 0) | (pred > 0)).sum()))
        seg_rows.append(_case_label("Failure", case_idx, f"fg IoU {fg_iou:.3f}"))
        seg_cells.append(
            [
                _cell(_semantic_overlay(image, truth), "Ground truth", [f"fg {int((truth > 0).sum())} px"], width=470, height=450),
                _cell(_semantic_overlay(image, pred), "Prediction", [f"fg {int((pred > 0).sum())} px"], width=470, height=450),
                _cell(_error_overlay(image, truth, pred), "Error map", ["green TP", "blue FN, red FP", "yellow class swap"], width=470, height=450),
            ]
        )
    path = out_dir / "segmentation_failure_cases_large.jpg"
    _make_matrix(["Ground truth", "Prediction", "Error map"], seg_rows, seg_cells, path, title="Segmentation Failure Cases", cell_w=470, cell_h=450, row_label_w=260)
    out.append(path)

    count_cells: list[list[Image.Image]] = []
    count_rows: list[str] = []
    for case_idx, row in enumerate(failure_sets["counting"].itertuples(index=False)):
        stem = str(row.stem)
        image = _read_rgb(row.image_path)
        y_true, y_pred, error = _count_for_stem(count_preds, stem)
        count_rows.append(_case_label("Failure", case_idx, f"error {error:+.1f}"))
        count_cells.append(
            [
                _cell(image, "Image", [f"true count {y_true:.0f}"], width=560, height=470),
                _failure_text_cell(
                    "Counting failure",
                    [f"true {y_true:.0f}", f"pred {y_pred:.1f}", f"error {error:+.1f}", "Crowding/occlusion changes visible density."],
                    width=560,
                    height=470,
                ),
            ]
        )
    path = out_dir / "counting_failure_cases_large.jpg"
    _make_matrix(["Image", "Error"], count_rows, count_cells, path, title="Counting Failure Cases", cell_w=560, cell_h=470, row_label_w=260, legend=False)
    out.append(path)

    class_fail = failure_sets["classification"]
    cls_cells: list[list[Image.Image]] = []
    cls_rows: list[str] = []
    for case_idx, row in enumerate(class_fail.itertuples(index=False)):
        hit = crop_rows[crop_rows["instance_id"] == row.instance_id]
        if hit.empty:
            continue
        crop_row = hit.iloc[0]
        crop = _read_rgb(crop_row["crop_path"])
        cls_rows.append(_case_label("Failure", case_idx, f"true {crop_row['true_class']}"))
        cls_cells.append(
            [
                _cell(crop, "Crop", [f"true {crop_row['true_class']}"], width=420, height=430, border=CLASS_COLORS.get(str(crop_row["true_class"]), (180, 180, 180))),
                _failure_text_cell(
                    "Classification failure",
                    [
                        f"pred {crop_row['pred_class']}",
                        f"confidence {float(crop_row['confidence']):.2f}",
                        "Adjacent ripeness stages are visually close.",
                    ],
                    width=540,
                    height=430,
                    accent=(190, 60, 60),
                ),
            ]
        )
    path = out_dir / "classification_failure_cases_large.jpg"
    _make_matrix(["Crop", "Prediction"], cls_rows, cls_cells, path, title="Classification Failure Cases", cell_w=540, cell_h=430, row_label_w=360, legend=False)
    out.append(path)
    return out


def _failure_text_cell(
    title: str,
    lines: list[str],
    width: int,
    height: int,
    accent: tuple[int, int, int] = (70, 90, 130),
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(216, 220, 225))
    draw.rectangle((0, 0, 10, height - 1), fill=accent)
    draw.text((28, 28), title, fill=(22, 26, 32), font=_font(21, bold=True))
    y = 98
    for line in lines:
        for wrapped in _wrap(draw, line, _font(17), width - 56, max_lines=2):
            draw.text((28, y), wrapped, fill=(45, 50, 58), font=_font(17))
            y += 38
        y += 8
    return canvas


def select_failure_sets(
    dirs: dict[str, Path],
    image_df: pd.DataFrame,
    instances: pd.DataFrame,
    n: int,
) -> dict[str, pd.DataFrame]:
    test = image_df[image_df["split"] == "test"].copy().reset_index(drop=True)
    det_preds = _load_detection_predictions(dirs, MAIN_METHOD)
    stem_to_index = {str(row.stem): idx for idx, row in enumerate(test.itertuples(index=False))}
    image_size = 512
    det_rows = []
    for row in test.itertuples(index=False):
        image_row = image_df[image_df["stem"].astype(str) == str(row.stem)].iloc[0]
        gt_boxes, _ = _gt_boxes(instances, str(row.stem), image_size, image_row)
        pred_boxes, _, pred_scores = _detection_frame_for_stem(det_preds, stem_to_index, str(row.stem))
        stats = _match_counts_agnostic(pred_boxes, pred_scores, gt_boxes)
        det_rows.append({"stem": str(row.stem), "failure_score": stats["fp"] + 2 * stats["fn"], **stats})
    det_fail = pd.DataFrame(det_rows)
    det_selected = (
        test.merge(det_fail, on="stem", how="left")
        .sort_values(["failure_score", "fn", "fp"], ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    seg_path = _prediction_run(dirs, MAIN_METHOD) / "segmentation_per_image_test.csv"
    seg_metrics = pd.read_csv(seg_path)
    seg_selected = (
        test.merge(seg_metrics[["stem", "foreground_iou"]], on="stem", how="left")
        .sort_values("foreground_iou", ascending=True)
        .head(n)
        .reset_index(drop=True)
    )

    count_preds = _load_count_predictions(dirs, MAIN_METHOD)
    count_preds["abs_error"] = count_preds["error"].abs()
    count_selected = (
        test.merge(count_preds[["stem", "y_true", "y_pred", "error", "abs_error"]], on="stem", how="left")
        .sort_values("abs_error", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    cls_preds = _crop_prediction_rows(instances[instances["split"] == "test"], _load_class_predictions(dirs, MAIN_METHOD))
    wrong = cls_preds[cls_preds["true_class"].astype(str) != cls_preds["pred_class"].astype(str)].copy()
    if wrong.empty:
        wrong = cls_preds.copy()
    class_selected = wrong.sort_values("confidence", ascending=False).head(max(4, n * 2)).reset_index(drop=True)
    return {
        "detection": det_selected,
        "segmentation": seg_selected,
        "counting": count_selected,
        "classification": class_selected,
    }


def _classification_ablation_cells(
    config: dict[str, Any],
    summary: pd.DataFrame,
    dirs: dict[str, Path],
    instances: pd.DataFrame,
    methods: list[tuple[str, str]],
    out_dir: Path,
) -> list[Path]:
    main_preds = _load_class_predictions(dirs, MAIN_METHOD)
    crop_rows = _crop_prediction_rows(instances[instances["split"] == "test"], main_preds)
    crop_rows["correct"] = crop_rows["true_class"].astype(str) == crop_rows["pred_class"].astype(str)
    selected_parts = []
    wrong = crop_rows[~crop_rows["correct"]].sort_values("confidence", ascending=False).head(6)
    selected_parts.append(wrong)
    for class_name in config["classes"]:
        selected_parts.append(
            crop_rows[(crop_rows["true_class"] == class_name) & (crop_rows["correct"])].sort_values("confidence", ascending=False).head(2)
        )
    selected = pd.concat(selected_parts, ignore_index=True).drop_duplicates("instance_id").head(6)
    pred_by_method = {}
    for method, _ in methods:
        pred_by_method[method] = _crop_prediction_rows(instances[instances["split"] == "test"], _load_class_predictions(dirs, method))
    headers = ["Crop"] + [label for _, label in methods]
    row_labels = [_case_label("Crop", idx, f"true {row.true_class}") for idx, row in enumerate(selected.itertuples(index=False))]
    cells: list[list[Image.Image]] = []
    for row in selected.itertuples(index=False):
        crop = _read_rgb(row.crop_path)
        cell_w, cell_h = 300, 320
        row_cells = [_cell(crop, "GT crop", [str(row.true_class)], width=cell_w, height=cell_h, border=CLASS_COLORS.get(str(row.true_class), (180, 180, 180)))]
        for method, label in methods:
            frame = pred_by_method[method]
            hit = frame[frame["instance_id"] == row.instance_id]
            if hit.empty:
                pred_class = "missing"
                conf = 0.0
            else:
                pred_class = str(hit.iloc[0]["pred_class"])
                conf = float(hit.iloc[0]["confidence"])
            correct = pred_class == str(row.true_class)
            fill = (36, 130, 70) if correct else (190, 60, 60)
            text_img = Image.new("RGB", (cell_w, 230), "white")
            draw = ImageDraw.Draw(text_img)
            draw.rectangle((0, 0, cell_w - 1, 229), outline=(215, 220, 225))
            y = 18
            for line in _wrap(draw, pred_class, _font(16, bold=True), cell_w - 20, max_lines=2):
                draw.text((10, y), line, fill=fill, font=_font(16, bold=True))
                y += 30
            draw.text((10, y + 4), f"conf {conf:.2f}", fill=(45, 50, 58), font=_font(14))
            draw.text((10, y + 36), "correct" if correct else "wrong", fill=fill, font=_font(14, bold=True))
            f1 = _number(_task_row(summary, "classification", method).get("macro_f1"), 0)
            draw.text((10, y + 68), f"F1 {f1:.3f}", fill=(45, 50, 58), font=_font(13))
            row_cells.append(_cell(text_img, label, width=cell_w, height=cell_h, border=(215, 220, 225)))
        cells.append(row_cells)
    path = out_dir / "ablation_classification_comparison_montage_large.jpg"
    _make_matrix(headers, row_labels, cells, path, title="Classification Ablation Visual Comparison", cell_w=300, cell_h=320, row_label_w=335, legend=False)
    return [path]


def _metric_ablation_summary(summary: pd.DataFrame, methods: list[tuple[str, str]], out_dir: Path) -> Path:
    cell_w, cell_h = 285, 190
    headers = ["Variant", "Detection", "Segmentation", "Counting", "Classification", "Delta-m"]
    rows: list[str] = []
    cells: list[list[Image.Image]] = []
    delta_path = out_dir.parent / "tables" / "unified_delta_m_summary.csv"
    if not delta_path.exists():
        delta_path = out_dir.parent / "tables" / "phase3_decoupled_decoder_ablation.csv"
    delta = pd.read_csv(delta_path) if delta_path.exists() else pd.DataFrame()
    for method, label in methods:
        rows.append(label)
        det = _number(_task_row(summary, "detection", method).get("map50"), 0)
        seg = _number(_task_row(summary, "segmentation", method).get("miou_foreground"), 0)
        mae = _number(_task_row(summary, "counting", method).get("mae"), 0)
        f1 = _number(_task_row(summary, "classification", method).get("macro_f1"), 0)
        if not delta.empty and method in set(delta["method"].astype(str)):
            dm = _number(delta[delta["method"].astype(str) == method].iloc[0].get("delta_m_percent"), 0)
        else:
            dm = 0
        values = [
            ("mAP50", det, True),
            ("mIoUfg", seg, True),
            ("MAE", mae, False),
            ("Macro F1", f1, True),
            ("Delta-m", dm, True),
        ]
        row_cells = [_metric_cell(label, "", 0, True, cell_w, cell_h)]
        for name, value, higher in values:
            row_cells.append(_metric_cell(name, f"{value:.3f}" if name != "MAE" else f"{value:.2f}", value, higher, cell_w, cell_h))
        cells.append(row_cells)
    path = out_dir / "ablation_metric_summary_montage_large.jpg"
    _make_matrix(headers, rows, cells, path, title="Ablation Metric Summary", cell_w=cell_w, cell_h=cell_h, row_label_w=245, legend=False)
    return path


def _metric_cell(title: str, value_text: str, value: float, higher: bool, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(216, 220, 225))
    draw.text((12, 18), title, fill=(35, 40, 48), font=_font(15, bold=True))
    if value_text:
        color = (35, 110, 70) if (higher and value >= 0) or (not higher and value <= 10) else (70, 70, 70)
        draw.text((12, 58), value_text, fill=color, font=_font(25, bold=True))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-ready qualitative and ablation montage figures.")
    parser.add_argument("--config", default="configs/fresh_benchmark.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--failure-examples", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    prepare_annotations(config)
    dirs = output_dirs(config)
    paper = dirs["paper_ready"]
    qual_dir = paper / "qualitative_montages_large"
    ablation_dir = paper / "ablation_montages_large"
    failure_dir = paper / "failure_montages_large"
    qual_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(paper / "tables" / "all_task_runs.csv")
    image_df = pd.read_csv(dirs["annotations"] / "image_manifest.csv")
    instances = pd.read_csv(dirs["annotations"] / "instances.csv")
    count_preds = _load_count_predictions(dirs, MAIN_METHOD)
    selected = _select_images(image_df, count_preds, n=args.examples)
    failure_sets = select_failure_sets(dirs, image_df, instances, n=args.failure_examples)
    seg_stems = set(selected["stem"].astype(str)) | set(failure_sets["segmentation"]["stem"].astype(str))
    seg_selection = image_df[image_df["stem"].astype(str).isin(seg_stems)].copy().reset_index(drop=True)
    methods = [method for method, _ in _available_methods(summary)]
    methods_for_seg = sorted(set(methods + [MAIN_METHOD]))
    device = resolve_device(args.device)
    seg_predictions = _predict_segmentations(config, summary, dirs, methods_for_seg, seg_selection, device)

    written = []
    written.extend(make_task_montages(config, summary, dirs, image_df, instances, selected, seg_predictions, qual_dir))
    written.extend(make_ablation_montages(config, summary, dirs, image_df, instances, selected, seg_predictions, ablation_dir))
    written.extend(make_failure_montages(config, summary, dirs, image_df, instances, seg_predictions, failure_sets, failure_dir))

    manifest = {
        "selected_examples": selected[["stem", "filename", "Total", "dominant_class"]].to_dict(orient="records"),
        "failure_examples": {
            key: value[[col for col in ["stem", "filename", "Total", "dominant_class"] if col in value.columns]].to_dict(orient="records")
            for key, value in failure_sets.items()
            if key != "classification"
        },
        "classification_failure_examples": failure_sets["classification"][
            [col for col in ["instance_index", "stem", "filename", "true_class", "pred_class", "confidence"] if col in failure_sets["classification"].columns]
        ].to_dict(orient="records"),
        "methods": _available_methods(summary),
        "files": [str(path.resolve()) for path in written],
    }
    manifest_path = paper / "tables" / "qualitative_montage_manifest_large.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"montage_files={len(written)}")
    print(f"manifest={manifest_path}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
