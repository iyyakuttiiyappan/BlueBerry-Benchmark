from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blueberry_multitask.annotations import prepare_annotations
from blueberry_multitask.config import load_config, output_dirs
from blueberry_multitask.datasets import (
    CropClassificationDataset,
    CountingDataset,
    DetectionDataset,
    SegmentationDataset,
    detection_collate,
)
from blueberry_multitask.engine import _loader_kwargs, _split, _task_cfg
from blueberry_multitask.models import create_detection_model, create_segmentation_model, create_timm_head_model
from blueberry_multitask.utils import resolve_device, set_seed


BEST_SPECIALISTS = {
    "classification": "resnet50",
    "counting": "count_efficientnetv2_s",
    "segmentation": "fpn_convnextv2_tiny_tta",
    "detection": "fasterrcnn_resnet50_fpn_thr005",
}


def _latest_completed_run(dirs: dict[str, Path], task: str, method: str) -> Path:
    root = dirs[task] / "runs"
    candidates = sorted(root.glob(f"*_{method}_seed*"), key=lambda path: path.stat().st_mtime, reverse=True)
    candidates = [path for path in candidates if (path / "best.pt").exists()]
    if not candidates:
        raise FileNotFoundError(f"No completed run found for {task}/{method} under {root}")
    return candidates[0]


def _load_state(model: torch.nn.Module, run_dir: Path, device: torch.device) -> None:
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@torch.inference_mode()
def export_classification(config: dict[str, Any], dirs: dict[str, Path], out_dir: Path, device: torch.device) -> dict[str, Any]:
    method = BEST_SPECIALISTS["classification"]
    run_dir = _latest_completed_run(dirs, "classification", method)
    method_cfg = _task_cfg(config, "classification", method, {})
    image_size = int(method_cfg.get("image_size", 224))
    crops = pd.read_csv(dirs["annotations"] / "classification_crops.csv")
    classes = list(config["classes"])
    created = create_timm_head_model(method_cfg, num_outputs=len(classes), pretrained=False)
    model = created.model.to(device)
    _load_state(model, run_dir, device)
    model.eval()
    loader_kwargs = _loader_kwargs({**config, "training": {**config.get("training", {}), "num_workers": 0}}, device)
    rows: list[pd.DataFrame] = []
    for split in ["train", "val", "test"]:
        frame = _split(crops, split)
        loader = DataLoader(CropClassificationDataset(frame, image_size, augment=False), batch_size=64, shuffle=False, **loader_kwargs)
        split_rows: list[dict[str, Any]] = []
        for images, labels, paths in tqdm(loader, desc=f"class-teacher-{split}", leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()
            pred = probs.argmax(axis=1)
            conf = probs.max(axis=1)
            for idx, path in enumerate(paths):
                row = {
                    "split": split,
                    "crop_path": str(path),
                    "teacher_y_pred": int(pred[idx]),
                    "teacher_confidence": float(conf[idx]),
                    "teacher_pred_class": classes[int(pred[idx])],
                    "y_true": int(labels[idx]),
                }
                for class_idx, class_name in enumerate(classes):
                    row[f"prob_{class_name}"] = float(probs[idx, class_idx])
                split_rows.append(row)
        split_df = pd.DataFrame(split_rows)
        split_df = split_df.merge(
            crops[["instance_id", "stem", "crop_path", "class_name", "label"]],
            on="crop_path",
            how="left",
        )
        split_df.to_csv(out_dir / f"classification_teacher_{split}.csv", index=False)
        rows.append(split_df)
    all_rows = pd.concat(rows, ignore_index=True)
    all_rows.to_csv(out_dir / "classification_teacher_all.csv", index=False)
    return {"method": method, "run_dir": str(run_dir), "rows": int(len(all_rows)), "image_size": image_size}


@torch.inference_mode()
def export_counting(config: dict[str, Any], dirs: dict[str, Path], out_dir: Path, device: torch.device) -> dict[str, Any]:
    method = BEST_SPECIALISTS["counting"]
    run_dir = _latest_completed_run(dirs, "counting", method)
    method_cfg = _task_cfg(config, "counting", method, {})
    image_size = int(method_cfg.get("image_size", 384))
    images = pd.read_csv(dirs["annotations"] / "image_manifest.csv")
    created = create_timm_head_model(method_cfg, num_outputs=1, pretrained=False)
    model = created.model.to(device)
    _load_state(model, run_dir, device)
    model.eval()
    loader_kwargs = _loader_kwargs({**config, "training": {**config.get("training", {}), "num_workers": 0}}, device)
    all_rows: list[pd.DataFrame] = []
    path_to_stem = dict(zip(images["image_path"].astype(str), images["stem"].astype(str)))
    for split in ["train", "val", "test"]:
        frame = _split(images, split)
        loader = DataLoader(
            CountingDataset(frame, image_size, str(method_cfg.get("target", "total")), augment=False),
            batch_size=32,
            shuffle=False,
            **loader_kwargs,
        )
        rows = []
        for batch_images, targets, paths in tqdm(loader, desc=f"count-teacher-{split}", leave=False):
            batch_images = batch_images.to(device, non_blocking=True)
            pred = model(batch_images).float().detach().cpu().numpy().reshape(-1)
            true = targets.numpy().reshape(-1)
            for idx, path in enumerate(paths):
                rows.append(
                    {
                        "split": split,
                        "stem": path_to_stem.get(str(path), Path(str(path)).stem),
                        "image_path": str(path),
                        "teacher_count": float(pred[idx]),
                        "true_count": float(true[idx]),
                    }
                )
        split_df = pd.DataFrame(rows)
        split_df.to_csv(out_dir / f"counting_teacher_{split}.csv", index=False)
        all_rows.append(split_df)
    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(out_dir / "counting_teacher_all.csv", index=False)
    return {"method": method, "run_dir": str(run_dir), "rows": int(len(all_df)), "image_size": image_size}


@torch.inference_mode()
def export_detection(config: dict[str, Any], dirs: dict[str, Path], out_dir: Path, device: torch.device) -> dict[str, Any]:
    method = BEST_SPECIALISTS["detection"]
    run_dir = _latest_completed_run(dirs, "detection", method)
    method_cfg = _task_cfg(config, "detection", method, {})
    image_size = int(method_cfg.get("image_size", 768))
    score_threshold = float(method_cfg.get("score_threshold", 0.05))
    classes = list(config["classes"])
    images = pd.read_csv(dirs["annotations"] / "image_manifest.csv")
    instances = pd.read_csv(dirs["annotations"] / "instances.csv")
    created = create_detection_model(method_cfg, num_classes_with_background=len(classes) + 1, pretrained=False)
    model = created.model.to(device)
    _load_state(model, run_dir, device)
    model.eval()
    loader_kwargs = _loader_kwargs({**config, "training": {**config.get("training", {}), "num_workers": 0}}, device)
    all_rows: list[pd.DataFrame] = []
    for split in ["train", "val", "test"]:
        frame = _split(images, split)
        split_instances = instances[instances["stem"].isin(set(frame["stem"].astype(str)))]
        loader = DataLoader(
            DetectionDataset(frame, split_instances, image_size, include_masks=False, augment=False),
            batch_size=2,
            shuffle=False,
            collate_fn=detection_collate,
            **loader_kwargs,
        )
        rows = []
        offset = 0
        stems = frame["stem"].astype(str).tolist()
        for batch_images, _targets in tqdm(loader, desc=f"det-teacher-{split}", leave=False):
            batch_images = [image.to(device, non_blocking=True) for image in batch_images]
            outputs = model(batch_images)
            for batch_idx, output in enumerate(outputs):
                stem = stems[offset + batch_idx]
                scores = output["scores"].detach().cpu().numpy()
                keep = scores >= score_threshold
                boxes = output["boxes"].detach().cpu().numpy()[keep]
                labels = output["labels"].detach().cpu().numpy()[keep]
                kept_scores = scores[keep]
                for box, label, score in zip(boxes, labels, kept_scores):
                    rows.append(
                        {
                            "split": split,
                            "stem": stem,
                            "label": int(label),
                            "score": float(score),
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                        }
                    )
            offset += len(batch_images)
        split_df = pd.DataFrame(rows)
        split_df.to_csv(out_dir / f"detection_teacher_{split}.csv", index=False)
        all_rows.append(split_df)
    all_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    all_df.to_csv(out_dir / "detection_teacher_all.csv", index=False)
    return {"method": method, "run_dir": str(run_dir), "rows": int(len(all_df)), "image_size": image_size, "score_threshold": score_threshold}


@torch.inference_mode()
def export_segmentation(config: dict[str, Any], dirs: dict[str, Path], out_dir: Path, device: torch.device) -> dict[str, Any]:
    method = BEST_SPECIALISTS["segmentation"]
    run_dir = _latest_completed_run(dirs, "segmentation", method)
    method_cfg = _task_cfg(config, "segmentation", method, {})
    image_size = int(method_cfg.get("image_size", 512))
    classes = list(config["classes"])
    images = pd.read_csv(dirs["annotations"] / "image_manifest.csv")
    created = create_segmentation_model(method_cfg, num_classes=len(classes) + 1, pretrained=False)
    model = created.model.to(device)
    _load_state(model, run_dir, device)
    model.eval()
    mask_root = out_dir / "segmentation_teacher_masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    loader_kwargs = _loader_kwargs({**config, "training": {**config.get("training", {}), "num_workers": 0}}, device)
    path_to_stem = dict(zip(images["image_path"].astype(str), images["stem"].astype(str)))
    manifest_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        frame = _split(images, split)
        loader = DataLoader(SegmentationDataset(frame, image_size, augment=False), batch_size=4, shuffle=False, **loader_kwargs)
        for batch_images, _masks, paths in tqdm(loader, desc=f"seg-teacher-{split}", leave=False):
            batch_images = batch_images.to(device, non_blocking=True)
            output = model(batch_images)
            logits = output["out"] if isinstance(output, dict) else output
            flipped = torch.flip(batch_images, dims=[3])
            flipped_output = model(flipped)
            flipped_logits = flipped_output["out"] if isinstance(flipped_output, dict) else flipped_output
            logits = 0.5 * (logits + torch.flip(flipped_logits, dims=[3]))
            pred = logits.detach().argmax(dim=1).cpu().numpy().astype(np.uint8)
            for idx, path in enumerate(paths):
                stem = path_to_stem.get(str(path), Path(str(path)).stem)
                mask_path = mask_root / split / f"{stem}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pred[idx], mode="L").save(mask_path)
                manifest_rows.append({"split": split, "stem": stem, "image_path": str(path), "teacher_mask_path": str(mask_path)})
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "segmentation_teacher_manifest.csv", index=False)
    return {"method": method, "run_dir": str(run_dir), "rows": int(len(manifest)), "image_size": image_size, "flip_tta": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export train/val/test predictions from best specialist teachers.")
    parser.add_argument("--config", default="configs/fresh_benchmark_514.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("training", {}).get("seed", 42)))
    prepare_annotations(config)
    dirs = output_dirs(config)
    out_dir = dirs["analysis"] / "specialist_teachers" / "best_individual_514"
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.json"
    if metadata_path.exists() and not args.force:
        print(f"teacher_dir={out_dir}")
        print(f"metadata={metadata_path}")
        return
    device = resolve_device(args.device)
    metadata = {
        "classification": export_classification(config, dirs, out_dir, device),
        "counting": export_counting(config, dirs, out_dir, device),
        "detection": export_detection(config, dirs, out_dir, device),
        "segmentation": export_segmentation(config, dirs, out_dir, device),
    }
    _json(metadata_path, metadata)
    print(f"teacher_dir={out_dir}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
