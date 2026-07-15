from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from blueberry_multitask.config import load_config, output_dirs


CLASS_COLORS = {
    "green_immature": (64, 160, 43),
    "pale_pink": (241, 146, 178),
    "pink_turns_purple": (138, 79, 184),
    "fully_ripe": (46, 96, 200),
    "over_ripe": (185, 57, 63),
}

CLASS_LABELS = {
    "green_immature": "green\nimmature",
    "pale_pink": "pale\npink",
    "pink_turns_purple": "pink ->\npurple",
    "fully_ripe": "fully\nripe",
    "over_ripe": "over\nripe",
}

MAIN_METHOD = "berrymtl_specialist_adapter_fusion_uncertainty"


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
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


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if not text:
        return 0, 0
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=3)
    return int(box[2] - box[0]), int(box[3] - box[1])


def _read_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    fitted = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return canvas


def _crop_prediction_rows(instances: pd.DataFrame, class_preds: pd.DataFrame) -> pd.DataFrame:
    inst = instances.copy()
    inst["within_image_index"] = inst.groupby("stem", sort=False).cumcount()
    preds = class_preds.copy()
    preds["within_image_index"] = preds["instance_index"].astype(int)
    return inst.merge(
        preds[["stem", "within_image_index", "true_class", "pred_class", "confidence"]],
        on=["stem", "within_image_index"],
        how="inner",
        suffixes=("", "_pred"),
    )


def _find_run(dirs: dict[str, Path], method: str) -> Path:
    candidates = sorted(
        dirs["analysis"].glob(f"**/*{method}*/classification_predictions_test.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            dirs["root"].glob(f"**/*{method}*/classification_predictions_test.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(f"No classification_predictions_test.csv found for {method}")
    return candidates[0].parent


def _select_balanced(rows: pd.DataFrame, classes: list[str], per_class: int) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    used_stems: set[str] = set()
    for class_name in classes:
        group = rows[rows["true_class"].astype(str) == class_name].copy()
        group["correct"] = group["true_class"].astype(str) == group["pred_class"].astype(str)
        group["area_rank"] = pd.to_numeric(group.get("area", 0), errors="coerce").fillna(0)
        preferred = group[group["correct"]].sort_values(["confidence", "area_rank"], ascending=False)
        fallback = group.sort_values(["correct", "confidence", "area_rank"], ascending=False)
        picks: list[dict[str, Any]] = []
        for frame in [preferred, fallback]:
            for _, row in frame.iterrows():
                if len(picks) >= per_class:
                    break
                stem = str(row["stem"])
                if stem in used_stems and len(group["stem"].astype(str).unique()) >= per_class:
                    continue
                if any(str(item["instance_id"]) == str(row["instance_id"]) for item in picks):
                    continue
                picks.append(row.to_dict())
                used_stems.add(stem)
            if len(picks) >= per_class:
                break
        if picks:
            selected.append(pd.DataFrame(picks))
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True)


def _tile(row: pd.Series, width: int, height: int) -> Image.Image:
    pad = 12
    title_h = 70
    footer_h = 88
    border = CLASS_COLORS.get(str(row.true_class), (180, 180, 180))
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width - 1, height - 1), outline=border, width=8)

    title = CLASS_LABELS.get(str(row.true_class), str(row.true_class).replace("_", " "))
    draw.multiline_text((pad, 10), title, fill=(25, 32, 42), font=_font(25, bold=True), spacing=2)

    crop_h = height - title_h - footer_h - 2 * pad
    crop = _fit_image(_read_rgb(row.crop_path), width - 2 * pad, crop_h)
    canvas.paste(crop, (pad, title_h))

    correct = str(row.true_class) == str(row.pred_class)
    status = "correct" if correct else "wrong"
    status_color = (34, 126, 75) if correct else (178, 55, 58)
    pred_label = str(row.pred_class).replace("_", " ")
    conf = float(row.confidence)
    y = title_h + crop_h + 10
    draw.text((pad, y), f"pred: {pred_label}", fill=(25, 32, 42), font=_font(17, bold=True))
    draw.text((pad, y + 28), f"conf: {conf:.2f}", fill=(70, 76, 86), font=_font(16))
    draw.text((pad, y + 54), status, fill=status_color, font=_font(18, bold=True))
    return canvas


def _legend(classes: list[str], width: int) -> Image.Image:
    height = 82
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 14), "Ripeness-classification examples: class-balanced test crops", fill=(25, 32, 42), font=_font(26, bold=True))
    x = 18
    y = 52
    for class_name in classes:
        color = CLASS_COLORS.get(class_name, (180, 180, 180))
        label = class_name.replace("_", " ")
        draw.rectangle((x, y, x + 24, y + 24), fill=color)
        draw.text((x + 32, y - 1), label, fill=(40, 45, 55), font=_font(14))
        x += 32 + _text_size(draw, label, _font(14))[0] + 22
    return canvas


def make_mixed_tile(config_path: str, per_class: int, output_name: str) -> tuple[Path, Path]:
    config = load_config(config_path)
    dirs = output_dirs(config)
    classes = list(config["classes"])
    run_dir = _find_run(dirs, MAIN_METHOD)
    preds = pd.read_csv(run_dir / "classification_predictions_test.csv")
    instances = pd.read_csv(dirs["annotations"] / "instances.csv")
    rows = _crop_prediction_rows(instances[instances["split"] == "test"], preds)
    rows = _select_balanced(rows, classes, per_class=per_class)
    if rows.empty:
        raise RuntimeError("No class-balanced classification rows could be selected.")

    tile_w, tile_h = 260, 390
    gap = 14
    label_w = 150
    cols = len(classes)
    total_w = label_w + cols * tile_w + (cols + 1) * gap
    total_h = 82 + per_class * tile_h + (per_class + 1) * gap
    canvas = Image.new("RGB", (total_w, total_h), (250, 251, 253))
    canvas.paste(_legend(classes, total_w), (0, 0))
    draw = ImageDraw.Draw(canvas)

    for col, class_name in enumerate(classes):
        header_x = label_w + gap + col * (tile_w + gap)
        draw.text((header_x + 6, 88), class_name.replace("_", "\n"), fill=(35, 40, 48), font=_font(15, bold=True), spacing=1)

    for row_idx in range(per_class):
        y = 82 + gap + row_idx * (tile_h + gap)
        draw.text((18, y + 18), f"Example {row_idx + 1}", fill=(35, 40, 48), font=_font(22, bold=True))
        for col, class_name in enumerate(classes):
            group = rows[rows["true_class"].astype(str) == class_name].reset_index(drop=True)
            if row_idx >= len(group):
                continue
            x = label_w + gap + col * (tile_w + gap)
            canvas.paste(_tile(group.iloc[row_idx], tile_w, tile_h), (x, y))

    out_dir = dirs["paper_ready"] / "qualitative_montages_large"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name
    canvas.save(out_path, quality=95)

    table_path = dirs["paper_ready"] / "tables" / "classification_mixed_tile_examples.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = ["instance_id", "stem", "filename", "true_class", "pred_class", "confidence", "crop_path"]
    rows[[col for col in keep_cols if col in rows.columns]].to_csv(table_path, index=False)

    figures_dir = dirs["paper_ready"] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(figures_dir / output_name, quality=95)
    return out_path, table_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a class-balanced mixed ripeness-classification tile image.")
    parser.add_argument("--config", default="configs/fresh_benchmark_514.yaml")
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--output-name", default="classification_examples_unified_large.jpg")
    args = parser.parse_args()
    out_path, table_path = make_mixed_tile(args.config, args.per_class, args.output_name)
    print(f"figure={out_path.resolve()}")
    print(f"examples={table_path.resolve()}")


if __name__ == "__main__":
    main()
