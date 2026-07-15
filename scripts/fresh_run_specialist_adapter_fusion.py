from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fresh_run_specialist_guided_distill import (  # noqa: E402
    SOURCE_METHOD,
    SOURCE_PROFILE,
    SpecialistGuidedDataset,
    _distill_epoch,
    _flat_values,
    _latest_checkpoint,
    _load_teacher_dir,
    specialist_collate,
)

from blueberry_multitask.annotations import prepare_annotations  # noqa: E402
from blueberry_multitask.config import load_config, output_dirs  # noqa: E402
from blueberry_multitask.ours import _class_weights, _dice_loss, _loader_kwargs, _mirror_task_outputs, _split  # noqa: E402
from blueberry_multitask.ours_attention import (  # noqa: E402
    BerryMTLInstanceDataset,
    _flatten_labels,
    _instance_class_weights,
    instance_collate,
)
from blueberry_multitask.ours_centernet import (  # noqa: E402
    BerryMTLCenterDetNet,
    FocalCrossEntropyLoss,
    _center_detection_loss,
    _center_epoch,
    _center_targets,
    _density_losses,
    _evaluate_center,
    _roi_quality_targets,
    _supervised_contrastive_loss,
)
from blueberry_multitask.plots import save_confusion_matrix, save_history_plot, save_prediction_overlay  # noqa: E402
from blueberry_multitask.utils import json_dump, now_stamp, resolve_device, set_seed  # noqa: E402


METHOD = "berrymtl_specialist_adapter_fusion"
DISPLAY = "BerryMTL-SpecialistAdapterFusion (ours)"
BALANCED_METHOD = "berrymtl_specialist_adapter_fusion_uncertainty"
BALANCED_DISPLAY = "BerryMTL-SpecialistAdapterFusion-UW (ours)"


class UncertaintyTaskBalancer(nn.Module):
    def __init__(self, tasks: tuple[str, ...] = ("segmentation", "detection", "counting", "classification")):
        super().__init__()
        self.tasks = tuple(tasks)
        self.log_vars = nn.ParameterDict({task: nn.Parameter(torch.zeros(())) for task in self.tasks})

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.zeros((), dtype=next(iter(losses.values())).dtype, device=next(iter(losses.values())).device)
        for task in self.tasks:
            loss = losses[task]
            log_var = self.log_vars[task].clamp(-5.0, 5.0)
            total = total + 0.5 * torch.exp(-log_var) * loss + 0.5 * log_var
        return total

    def metrics(self) -> dict[str, float]:
        output: dict[str, float] = {}
        for task in self.tasks:
            log_var = float(self.log_vars[task].detach().cpu())
            output[f"uw_log_var_{task}"] = log_var
            output[f"uw_weight_{task}"] = float(torch.exp(-self.log_vars[task].detach().cpu()).item())
        return output


def _set_trainable(model: nn.Module, freeze_encoder: bool) -> None:
    for name, parameter in model.named_parameters():
        if freeze_encoder and name.startswith("encoder."):
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True


def _optimizer_groups(model: nn.Module, cfg: dict[str, Any], balancer: nn.Module | None = None) -> list[dict[str, Any]]:
    adapter_params = []
    encoder_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "adapter" in name:
            adapter_params.append(parameter)
        elif name.startswith("encoder."):
            encoder_params.append(parameter)
        else:
            head_params.append(parameter)
    groups: list[dict[str, Any]] = []
    if adapter_params:
        groups.append({"params": adapter_params, "lr": float(cfg.get("adapter_lr", 0.00016)), "name": "adapters"})
    if head_params:
        groups.append({"params": head_params, "lr": float(cfg.get("head_lr", 0.000030)), "name": "heads"})
    if encoder_params:
        groups.append({"params": encoder_params, "lr": float(cfg.get("encoder_lr", 0.000004)), "name": "encoder"})
    if balancer is not None:
        groups.append({"params": list(balancer.parameters()), "lr": float(cfg.get("balancer_lr", 0.00008)), "name": "loss_balancer"})
    return groups


def _make_optimizer(model: nn.Module, cfg: dict[str, Any], balancer: nn.Module | None = None) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        _optimizer_groups(model, cfg, balancer=balancer),
        weight_decay=float(cfg.get("adapter_weight_decay", cfg.get("weight_decay", 0.04))),
    )


def _safe_load_base(model: BerryMTLCenterDetNet, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    result = model.load_state_dict(checkpoint["model"], strict=False)
    bad_missing = [key for key in result.missing_keys if "adapter" not in key]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(
            "Unexpected checkpoint mismatch while initializing adapter fusion: "
            f"missing={bad_missing}, unexpected={list(result.unexpected_keys)}"
        )


def _balanced_distill_epoch(
    model: BerryMTLCenterDetNet,
    loader: DataLoader,
    seg_criterion: nn.Module,
    cls_criterion: nn.Module,
    count_loss: nn.Module,
    balancer: UncertaintyTaskBalancer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    num_classes: int,
    image_size: int,
    cfg: dict[str, Any],
) -> dict[str, float]:
    model.train(True)
    balancer.train(True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    totals: dict[str, float] = {}
    image_count = 0
    roi_count = 0
    detection_classes = int(cfg.get("detection_classes", 1 if bool(cfg.get("class_agnostic_detection", True)) else num_classes - 1))
    class_agnostic_detection = bool(cfg.get("class_agnostic_detection", True))
    for batch in loader:
        (
            images,
            masks,
            targets,
            _stems,
            boxes,
            labels,
            teacher_masks,
            teacher_counts,
            teacher_boxes,
            teacher_labels,
            teacher_cls,
            teacher_cls_conf,
        ) = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        teacher_masks = teacher_masks.to(device, non_blocking=True)
        teacher_counts = teacher_counts.to(device, non_blocking=True)
        boxes = [value.to(device, non_blocking=True) for value in boxes]
        labels = [value.to(device, non_blocking=True) for value in labels]
        teacher_boxes = [value.to(device, non_blocking=True) for value in teacher_boxes]
        teacher_labels = [value.to(device, non_blocking=True) for value in teacher_labels]
        cls_labels = _flatten_labels(labels, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(images, boxes=boxes)
            logits = output["seg"]
            counts = output["count"].float()
            center_targets = _center_targets(
                boxes,
                labels,
                image_size=image_size,
                heatmap_shape=output["det_heatmap"].shape[-2:],
                num_classes=num_classes,
                detection_classes=detection_classes,
                class_agnostic=class_agnostic_detection,
                device=device,
            )
            seg_loss = seg_criterion(logits, masks) + float(cfg.get("dice_loss_weight", 0.52)) * _dice_loss(logits, masks, num_classes)
            c_loss = count_loss(counts, targets.float())
            d_loss, _ = _center_detection_loss(output, center_targets)
            density_map_loss, density_count_loss, _ = _density_losses(output, center_targets["density"], targets.float(), count_loss)
            density_loss = density_map_loss + float(cfg.get("density_count_loss_weight", 0.09)) * density_count_loss
            if cls_labels.numel() > 0:
                cls_logits = output["cls"]
                r_loss = cls_criterion(cls_logits, cls_labels)
                contrastive_loss = _supervised_contrastive_loss(output["cls_features"], cls_labels, float(cfg.get("contrastive_temperature", 0.12)))
            else:
                cls_logits = torch.zeros((0, num_classes - 1), device=device)
                r_loss = torch.zeros((), device=device)
                contrastive_loss = torch.zeros((), device=device)

            roi_quality_loss = torch.zeros((), device=device)
            if float(cfg.get("roi_quality_loss_weight", 0.0)) > 0:
                quality_boxes, quality_targets = _roi_quality_targets(
                    boxes,
                    image_size=image_size,
                    jitter_count=int(cfg.get("roi_quality_jitter_count", 2)),
                    background_count=int(cfg.get("roi_quality_background_count", 8)),
                )
                if quality_targets.numel() > 0:
                    _, _, quality_logits = model.classify_rois(
                        output["roi_feature"],
                        logits,
                        quality_boxes,
                        image_size=image_size,
                        return_features=True,
                        return_quality=True,
                    )
                    roi_quality_loss = F.binary_cross_entropy_with_logits(
                        quality_logits.float().view(-1),
                        quality_targets.to(device=quality_logits.device, dtype=torch.float32).view(-1),
                    )

            teacher_seg_loss = F.cross_entropy(logits.float(), teacher_masks)
            teacher_det_loss = torch.zeros((), device=device)
            if sum(int(value.numel() > 0) for value in teacher_boxes) > 0:
                teacher_targets = _center_targets(
                    teacher_boxes,
                    teacher_labels,
                    image_size=image_size,
                    heatmap_shape=output["det_heatmap"].shape[-2:],
                    num_classes=num_classes,
                    detection_classes=detection_classes,
                    class_agnostic=class_agnostic_detection,
                    device=device,
                )
                teacher_det_loss, _ = _center_detection_loss(output, teacher_targets)
            teacher_cls_loss = torch.zeros((), device=device)
            teacher_cls_labels = _flat_values(teacher_cls, device, torch.long)
            teacher_cls_weight = _flat_values(teacher_cls_conf, device, torch.float32)
            if cls_logits.shape[0] == teacher_cls_labels.numel() and teacher_cls_labels.numel() > 0:
                keep = teacher_cls_weight >= float(cfg.get("teacher_class_min_confidence", 0.40))
                if bool(keep.any()):
                    cls_each = F.cross_entropy(cls_logits.float()[keep], teacher_cls_labels[keep], reduction="none")
                    weights = teacher_cls_weight[keep].clamp(0.20, 1.0)
                    teacher_cls_loss = (cls_each * weights).sum() / weights.sum().clamp_min(1e-6)
            teacher_count_loss = count_loss(counts, teacher_counts.float())

            task_losses = {
                "segmentation": seg_loss + float(cfg.get("teacher_segmentation_loss_weight", 0.06)) * teacher_seg_loss,
                "detection": (
                    float(cfg.get("detection_loss_weight", 1.25)) * d_loss
                    + float(cfg.get("roi_quality_loss_weight", 0.28)) * roi_quality_loss
                    + float(cfg.get("teacher_detection_loss_weight", 0.10)) * teacher_det_loss
                ),
                "counting": (
                    float(cfg.get("count_loss_weight", 0.016)) * c_loss
                    + float(cfg.get("density_loss_weight", 0.025)) * density_loss
                    + float(cfg.get("teacher_count_loss_weight", 0.0)) * teacher_count_loss
                ),
                "classification": (
                    float(cfg.get("classification_loss_weight", 0.72)) * r_loss
                    + float(cfg.get("contrastive_loss_weight", 0.05)) * contrastive_loss
                    + float(cfg.get("teacher_classification_loss_weight", 0.20)) * teacher_cls_loss
                ),
            }
            loss = balancer(task_losses)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(balancer.parameters()), float(cfg.get("grad_clip_norm", 1.0)))
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(images.shape[0])
        image_count += batch_size
        roi_count += int(max(1, cls_labels.numel()))
        metrics = {
            "loss": loss,
            "seg_loss": seg_loss,
            "count_loss": c_loss,
            "det_loss": d_loss,
            "roi_cls_loss": r_loss,
            "teacher_seg_loss": teacher_seg_loss,
            "teacher_det_loss": teacher_det_loss,
            "teacher_cls_loss": teacher_cls_loss,
            "teacher_count_loss": teacher_count_loss,
            **{f"task_{key}_loss": value for key, value in task_losses.items()},
        }
        for key, value in metrics.items():
            denom = roi_count if key in {"roi_cls_loss", "teacher_cls_loss"} else image_count
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
        for key, value in balancer.metrics().items():
            totals[key] = value * image_count
    return {key: value / max(1, image_count) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BerryMTL with specialist-initialized task adapters.")
    parser.add_argument("--config", default="configs/fresh_benchmark_514.yaml")
    parser.add_argument("--teacher-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--loss-balancer", choices=["fixed", "uncertainty"], default="fixed")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("training", {}).get("seed", 42))
    set_seed(seed)
    prepare_annotations(config)
    dirs = output_dirs(config)
    teacher_dir = _load_teacher_dir(dirs, args.teacher_dir)
    teacher_metadata = json.loads((teacher_dir / "metadata.json").read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    classes = list(config["classes"])
    num_classes = len(classes) + 1
    method = BALANCED_METHOD if args.loss_balancer == "uncertainty" else METHOD
    display = BALANCED_DISPLAY if args.loss_balancer == "uncertainty" else DISPLAY

    cfg = dict(config.get(SOURCE_PROFILE, {}))
    cfg.update(
        {
            "method": method,
            "display_name": display,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "adapter_fusion": True,
            "adapter_bottleneck": int(cfg.get("adapter_bottleneck", 48)),
            "adapter_lr": 0.00016,
            "head_lr": 0.000030,
            "encoder_lr": 0.000004,
            "balancer_lr": 0.00008,
            "loss_balancer": args.loss_balancer,
            "teacher_segmentation_loss_weight": 0.06,
            "teacher_detection_loss_weight": 0.10,
            "teacher_classification_loss_weight": 0.20,
            "teacher_count_loss_weight": 0.0,
            "teacher_class_min_confidence": 0.40,
            "grad_clip_norm": 1.0,
        }
    )
    image_size = int(cfg.get("image_size", 768))
    detection_classes = int(cfg.get("detection_classes", 1 if bool(cfg.get("class_agnostic_detection", True)) else num_classes - 1))
    class_agnostic_detection = bool(cfg.get("class_agnostic_detection", True))

    images = pd.read_csv(dirs["annotations"] / "image_manifest.csv")
    instances = pd.read_csv(dirs["annotations"] / "instances.csv")
    train_images = _split(images, "train", args.limit)
    val_images = _split(images, "val", args.limit)
    test_images = _split(images, "test", args.limit)
    train_instances = instances[instances["stem"].isin(set(train_images["stem"].astype(str)))]
    val_instances = instances[instances["stem"].isin(set(val_images["stem"].astype(str)))]
    test_instances = instances[instances["stem"].isin(set(test_images["stem"].astype(str)))]
    loader_config = {**config, "training": {**config.get("training", {}), "num_workers": 0}}
    loader_kwargs = _loader_kwargs(loader_config, device)

    det_teacher_size = int(teacher_metadata.get("detection", {}).get("image_size", image_size))
    train_loader = DataLoader(
        SpecialistGuidedDataset(
            train_images,
            train_instances,
            image_size=image_size,
            teacher_dir=teacher_dir,
            detection_score_min=float(cfg.get("teacher_detection_score_min", 0.05)),
            detection_max_boxes=int(cfg.get("teacher_detection_max_boxes", 220)),
            detection_teacher_image_size=det_teacher_size,
        ),
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=True,
        collate_fn=specialist_collate,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        BerryMTLInstanceDataset(val_images, val_instances, image_size, False),
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=False,
        collate_fn=instance_collate,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        BerryMTLInstanceDataset(test_images, test_instances, image_size, False),
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=False,
        collate_fn=instance_collate,
        **loader_kwargs,
    )

    model = BerryMTLCenterDetNet(
        str(cfg.get("model_name", "convnextv2_tiny.fcmae_ft_in22k_in1k")),
        num_classes=num_classes,
        pretrained=False,
        decoder_channels=int(cfg.get("decoder_channels", 128)),
        roi_channels=int(cfg.get("roi_channels", 192)),
        roi_size=int(cfg.get("roi_size", 7)),
        detection_classes=detection_classes,
        decoupled_decoder=bool(cfg.get("decoupled_decoder", True)),
        dense_count_residual=bool(cfg.get("dense_count_residual", True)),
        task_aligned_detection=bool(cfg.get("task_aligned_detection", True)),
        highres_detection=bool(cfg.get("highres_detection", True)),
        roi_global_context=bool(cfg.get("roi_global_context", True)),
        adapter_fusion=True,
        adapter_bottleneck=int(cfg.get("adapter_bottleneck", 48)),
    ).to(device)
    source_checkpoint = _latest_checkpoint(dirs, SOURCE_METHOD)
    _safe_load_base(model, source_checkpoint, device)
    balancer = UncertaintyTaskBalancer().to(device) if args.loss_balancer == "uncertainty" else None

    freeze_encoder_epochs = int(args.freeze_encoder_epochs)
    _set_trainable(model, freeze_encoder=freeze_encoder_epochs > 0)
    optimizer = _make_optimizer(model, cfg, balancer=balancer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(cfg.get("epochs", args.epochs))))

    seg_weights = _class_weights(train_images, num_classes).to(device)
    cls_weights = _instance_class_weights(train_instances, len(classes)).to(device)
    roi_class_weight_power = float(cfg.get("roi_class_weight_power", 1.18))
    if abs(roi_class_weight_power - 1.0) > 1e-6:
        cls_weights = torch.pow(cls_weights, roi_class_weight_power)
        cls_weights = cls_weights / torch.clamp(cls_weights.mean(), min=1e-9)
        cls_weights = torch.clamp(cls_weights, min=0.20, max=12.0)
    seg_loss = nn.CrossEntropyLoss(weight=seg_weights)
    cls_loss = FocalCrossEntropyLoss(
        cls_weights,
        gamma=float(cfg.get("roi_focal_gamma", 1.35)),
        label_smoothing=float(cfg.get("label_smoothing", 0.025)),
    )
    count_loss = nn.SmoothL1Loss()
    amp = bool(config.get("training", {}).get("amp", True))

    run_dir = dirs["analysis"] / "ours" / "runs" / f"{now_stamp()}_{method}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.get("_config_path")
    if config_path and Path(config_path).exists():
        shutil.copy2(config_path, run_dir / "config.yaml")

    history: list[dict[str, float]] = []
    best_value: float | None = None
    best_epoch = 0
    start = time.perf_counter()
    was_encoder_frozen = freeze_encoder_epochs > 0
    for epoch in range(1, int(cfg.get("epochs", args.epochs)) + 1):
        should_freeze_encoder = epoch <= freeze_encoder_epochs
        if was_encoder_frozen != should_freeze_encoder:
            _set_trainable(model, freeze_encoder=should_freeze_encoder)
            optimizer = _make_optimizer(model, cfg, balancer=balancer)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(cfg.get("epochs", args.epochs)) - epoch + 1),
            )
            was_encoder_frozen = should_freeze_encoder

        if balancer is None:
            train_metrics = _distill_epoch(
                model,
                train_loader,
                seg_loss,
                cls_loss,
                count_loss,
                optimizer,
                device,
                amp,
                num_classes,
                image_size,
                cfg,
            )
        else:
            train_metrics = _balanced_distill_epoch(
                model,
                train_loader,
                seg_loss,
                cls_loss,
                count_loss,
                balancer,
                optimizer,
                device,
                amp,
                num_classes,
                image_size,
                cfg,
            )
        val_metrics = _center_epoch(
            model,
            val_loader,
            seg_loss,
            cls_loss,
            count_loss,
            optimizer,
            device,
            amp,
            False,
            num_classes,
            classes,
            image_size,
            float(cfg.get("count_loss_weight", 0.016)),
            float(cfg.get("dice_loss_weight", 0.52)),
            float(cfg.get("classification_loss_weight", 0.72)),
            float(cfg.get("detection_loss_weight", 1.25)),
            float(cfg.get("density_loss_weight", 0.025)),
            float(cfg.get("density_count_loss_weight", 0.09)),
            float(cfg.get("contrastive_loss_weight", 0.05)),
            float(cfg.get("contrastive_temperature", 0.12)),
            float(cfg.get("roi_quality_loss_weight", 0.28)),
            int(cfg.get("roi_quality_jitter_count", 2)),
            int(cfg.get("roi_quality_background_count", 8)),
            float(cfg.get("classification_score_weight", 0.30)),
            float(cfg.get("count_score_weight", 0.002)),
            float(cfg.get("detection_score_weight", 0.012)),
            detection_classes,
            class_agnostic_detection,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "encoder_frozen": should_freeze_encoder,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        history_df = pd.DataFrame(history)
        history_df.to_csv(run_dir / "history.csv", index=False)
        save_history_plot(history_df.rename(columns={"val_joint_score": "val_joint"}), run_dir / "training_curves.png", "joint")
        score = float(val_metrics["joint_score"])
        if best_value is None or score > best_value:
            best_value = score
            best_epoch = epoch
            state = {"model": model.state_dict(), "epoch": epoch}
            if balancer is not None:
                state["loss_balancer"] = balancer.state_dict()
            torch.save(state, run_dir / "best.pt")

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = _evaluate_center(
        model,
        test_loader,
        test_images,
        test_instances,
        device,
        amp,
        image_size,
        classes,
        score_threshold=float(cfg.get("score_threshold", 0.008)),
        top_k=int(cfg.get("top_k", 450)),
        nms_iou=float(cfg.get("nms_iou", 0.32)),
        classify_detection_boxes=bool(cfg.get("classify_detection_boxes", True)),
        class_agnostic_detection=class_agnostic_detection,
        segmentation_support_power=float(cfg.get("segmentation_support_power", 0.35)),
        segmentation_support_threshold=float(cfg.get("segmentation_support_threshold", 0.035)),
        segmentation_box_refine=bool(cfg.get("segmentation_box_refine", False)),
        segmentation_box_refine_threshold=float(cfg.get("segmentation_box_refine_threshold", 0.30)),
        segmentation_box_refine_expansion=float(cfg.get("segmentation_box_refine_expansion", 0.10)),
        segmentation_box_refine_blend=float(cfg.get("segmentation_box_refine_blend", 0.25)),
        segmentation_box_refine_min_pixels=int(cfg.get("segmentation_box_refine_min_pixels", 6)),
        count_aware_topk=bool(cfg.get("count_aware_topk", True)),
        count_aware_multiplier=float(cfg.get("count_aware_multiplier", 1.60)),
        count_aware_bias=float(cfg.get("count_aware_bias", 12.0)),
        count_aware_min=int(cfg.get("count_aware_min", 18)),
        count_aware_max=int(cfg.get("count_aware_max", 110)),
        roi_quality_inference_power=float(cfg.get("roi_quality_inference_power", 0.45)),
    )
    train_seconds = time.perf_counter() - start
    common = {"best_epoch": best_epoch, "best_val_metric": best_value, "train_seconds": train_seconds}

    test["segmentation_per_class"].to_csv(run_dir / "segmentation_per_class_test.csv", index=False)
    test["segmentation_per_image"].to_csv(run_dir / "segmentation_per_image_test.csv", index=False)
    test["counting_predictions"].to_csv(run_dir / "counting_predictions_test.csv", index=False)
    test["detection_per_class"].to_csv(run_dir / "detection_per_class_test.csv", index=False)
    test["detection_predictions"].to_csv(run_dir / "detection_predictions_test.csv", index=False)
    test["classification_predictions"].to_csv(run_dir / "classification_predictions_test.csv", index=False)
    test["classification_report"].to_csv(run_dir / "classification_report_test.csv", index=False)
    test["classification_confusion"].to_csv(run_dir / "confusion_matrix_test.csv")
    save_confusion_matrix(test["classification_confusion"], run_dir / "confusion_matrix_test.png", title=f"{display}: ROI Classification")
    if test["sample"] is not None:
        image_path, target, pred = test["sample"]
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        save_prediction_overlay(image, target, pred, run_dir / "sample_prediction_overlay.jpg")

    combined = {
        "classification": {**test["classification"], **common},
        "counting": {**test["counting"], **common},
        "segmentation": {**test["segmentation"], **common},
        "detection": {**test["detection"], **common},
    }
    json_dump(combined, run_dir / "test_metrics_by_task.json")
    metadata = {
        "method": method,
        "display_name": display,
        "family": "Specialist-initialized adapter fusion" if balancer is None else "Specialist adapter fusion + uncertainty weighting",
        "seed": seed,
        "device": str(device),
        "image_size": image_size,
        "model_name": str(cfg.get("model_name")),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "teacher_dir": str(teacher_dir.resolve()),
        "adapter_fusion": True,
        "adapter_bottleneck": int(cfg.get("adapter_bottleneck", 48)),
        "freeze_encoder_epochs": freeze_encoder_epochs,
        "adapter_lr": float(cfg.get("adapter_lr", 0.00016)),
        "loss_balancer": args.loss_balancer,
        "balancer_lr": float(cfg.get("balancer_lr", 0.00008)),
        "head_lr": float(cfg.get("head_lr", 0.000030)),
        "encoder_lr": float(cfg.get("encoder_lr", 0.000004)),
        "teacher_segmentation_loss_weight": float(cfg.get("teacher_segmentation_loss_weight", 0.06)),
        "teacher_detection_loss_weight": float(cfg.get("teacher_detection_loss_weight", 0.10)),
        "teacher_classification_loss_weight": float(cfg.get("teacher_classification_loss_weight", 0.20)),
        "teacher_count_loss_weight": float(cfg.get("teacher_count_loss_weight", 0.0)),
        "class_agnostic_detection": bool(class_agnostic_detection),
        "detection_classes": int(detection_classes),
        "decoupled_decoder": bool(cfg.get("decoupled_decoder", True)),
        "dense_count_residual": bool(cfg.get("dense_count_residual", True)),
        "task_aligned_detection": bool(cfg.get("task_aligned_detection", True)),
        "highres_detection": bool(cfg.get("highres_detection", True)),
        "roi_global_context": bool(cfg.get("roi_global_context", True)),
    }
    if balancer is not None:
        metadata.update(balancer.metrics())
    json_dump(metadata, run_dir / "metadata.json")

    base_meta = {
        **metadata,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    }
    _mirror_task_outputs(
        config,
        run_dir,
        "classification",
        {**test["classification"], **common},
        {**base_meta, "task": "classification"},
        {
            "predictions_test.csv": run_dir / "classification_predictions_test.csv",
            "classification_report_test.csv": run_dir / "classification_report_test.csv",
            "confusion_matrix_test.csv": run_dir / "confusion_matrix_test.csv",
            "confusion_matrix_test.png": run_dir / "confusion_matrix_test.png",
        },
    )
    _mirror_task_outputs(
        config,
        run_dir,
        "counting",
        {**test["counting"], **common},
        {**base_meta, "task": "counting"},
        {"predictions_test.csv": run_dir / "counting_predictions_test.csv"},
    )
    _mirror_task_outputs(
        config,
        run_dir,
        "segmentation",
        {**test["segmentation"], **common},
        {**base_meta, "task": "segmentation"},
        {
            "per_class_test.csv": run_dir / "segmentation_per_class_test.csv",
            "per_image_test.csv": run_dir / "segmentation_per_image_test.csv",
            "sample_prediction_overlay.jpg": run_dir / "sample_prediction_overlay.jpg",
        },
    )
    det_per_class = test["detection_per_class"].copy()
    det_per_class["class_name"] = det_per_class["class_id"].map({idx + 1: name for idx, name in enumerate(classes)})
    det_per_class.to_csv(run_dir / "detection_per_class_test_named.csv", index=False)
    _mirror_task_outputs(
        config,
        run_dir,
        "detection",
        {**test["detection"], **common},
        {**base_meta, "task": "detection"},
        {
            "predictions_test.csv": run_dir / "detection_predictions_test.csv",
            "per_class_test.csv": run_dir / "detection_per_class_test_named.csv",
        },
    )
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
