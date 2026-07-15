from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from blueberry_multitask.annotations import prepare_annotations
from blueberry_multitask.config import load_config, output_dirs
from blueberry_multitask.ours import _loader_kwargs
from blueberry_multitask.ours_attention import BerryMTLInstanceDataset, instance_collate
from blueberry_multitask.ours_centernet import CENTER_METHOD, BerryMTLCenterDetNet
from blueberry_multitask.utils import json_dump, resolve_device, set_seed


TASK_COLORS = {
    "shared_encoder": "#4c4c4c",
    "detection_head": "#e76f51",
    "segmentation_head": "#2a9d8f",
    "classification_roi_head": "#b05ac9",
    "counting_head": "#457b9d",
}
TASK_DISPLAY = {
    "shared_encoder": "Shared encoder",
    "detection_head": "Detection",
    "segmentation_head": "Segmentation",
    "classification_roi_head": "Classification",
    "counting_head": "Counting",
}


def _latest_centerdet_run(dirs: dict[str, Path]) -> Path:
    root = dirs["analysis"] / "ours" / "runs"
    candidates = sorted(
        list(root.glob("*_berrymtl_specialist_adapter_fusion_uncertainty_seed*"))
        + list(root.glob("*_berrymtl_specialist_adapter_fusion_seed*"))
        + list(root.glob("*_berrymtl_specialist_guided_distill_seed*"))
        + list(root.glob("*_berrymtl_teacher_aligned_det_seed*"))
        + list(root.glob("*_berrymtl_centerdet_hitile_quality_seed*"))
        + list(root.glob("*_berrymtl_centerdet_highres_residual_seed*"))
        + list(root.glob("*_berrymtl_centerdet_aligned_highres_seed*"))
        + list(root.glob("*_berrymtl_centerdet_decoupled_residual_seed*"))
        + list(root.glob(f"*_{CENTER_METHOD}_calibrated_seed*"))
        + list(root.glob(f"*_{CENTER_METHOD}_seed*")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No CenterDet run found under {root}")
    return candidates[0]


def _pool_feature(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim == 4:
        return F.adaptive_avg_pool2d(feature.float(), 1).flatten(1)
    return feature.float().flatten(1)


def _load_metadata(run_dir: Path) -> dict[str, object]:
    path = run_dir / "metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _model_from_run(
    config: dict[str, object],
    cfg: dict[str, object],
    metadata: dict[str, object],
    device: torch.device,
) -> BerryMTLCenterDetNet:
    class_names = list(config["classes"])  # type: ignore[index]
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


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = StandardScaler().fit_transform(np.asarray(x, dtype=np.float64))
    y = StandardScaler().fit_transform(np.asarray(y, dtype=np.float64))
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    numerator = float(np.sum(xy * xy))
    denominator = float(np.sqrt(np.sum(xx * xx) * np.sum(yy * yy)))
    return numerator / max(denominator, 1e-12)


def _extract_task_representations(
    model: BerryMTLCenterDetNet,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    captured: dict[str, torch.Tensor] = {}
    hooks = [
        model.lateral[-1].register_forward_hook(lambda module, inputs, output: captured.__setitem__("shared_encoder", output.detach())),
        model.seg_head[2].register_forward_hook(lambda module, inputs, output: captured.__setitem__("segmentation_head", output.detach())),
        model.center_stem.register_forward_hook(lambda module, inputs, output: captured.__setitem__("detection_head", output.detach())),
        model.class_head[5].register_forward_hook(lambda module, inputs, output: captured.__setitem__("classification_roi_head", output.detach())),
        model.count_head[5].register_forward_hook(lambda module, inputs, output: captured.__setitem__("counting_head", output.detach())),
    ]
    if getattr(model, "decoupled_decoder", False):
        hooks.append(model.dense_count_head[4].register_forward_hook(lambda module, inputs, output: captured.__setitem__("counting_dense_head", output.detach())))
    representations: dict[str, list[np.ndarray]] = {name: [] for name in TASK_COLORS}
    rows: list[dict[str, object]] = []
    model.eval()
    try:
        with torch.inference_mode():
            for images, masks, targets, stems, boxes, labels in tqdm(loader, desc="task features", leave=False):
                images = images.to(device, non_blocking=True)
                boxes = [value.to(device, non_blocking=True) for value in boxes]
                captured.clear()
                with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                    _ = model(images, boxes=boxes)
                batch_size = images.shape[0]
                for task_name in ["shared_encoder", "segmentation_head", "detection_head"]:
                    pooled = _pool_feature(captured[task_name]).detach().cpu().numpy()
                    representations[task_name].append(pooled)

                count_parts = []
                if "counting_head" in captured:
                    count_parts.append(_pool_feature(captured["counting_head"]).detach().cpu().numpy())
                if "counting_dense_head" in captured:
                    count_parts.append(_pool_feature(captured["counting_dense_head"]).detach().cpu().numpy())
                if count_parts:
                    representations["counting_head"].append(np.concatenate(count_parts, axis=1))
                else:
                    representations["counting_head"].append(np.zeros((batch_size, 256), dtype=np.float32))

                roi_hidden = captured.get("classification_roi_head")
                cls_rows = []
                if roi_hidden is not None and roi_hidden.numel() > 0:
                    roi_np = roi_hidden.float().detach().cpu().numpy()
                    offset = 0
                    for box_tensor in boxes:
                        count = int(box_tensor.shape[0])
                        if count > 0:
                            cls_rows.append(roi_np[offset : offset + count].mean(axis=0))
                        else:
                            cls_rows.append(np.zeros((roi_np.shape[1],), dtype=np.float32))
                        offset += count
                else:
                    cls_rows = [np.zeros((192,), dtype=np.float32) for _ in range(batch_size)]
                representations["classification_roi_head"].append(np.stack(cls_rows, axis=0))

                for stem, target, box_tensor in zip(stems, targets, boxes):
                    rows.append(
                        {
                            "stem": str(stem),
                            "split_count": float(target.item()),
                            "num_gt_boxes": int(box_tensor.shape[0]),
                        }
                    )
    finally:
        for handle in hooks:
            handle.remove()
    arrays = {name: np.concatenate(chunks, axis=0) for name, chunks in representations.items()}
    return arrays, pd.DataFrame(rows)


def _plot_cka(cka: pd.DataFrame, output_dir: Path) -> None:
    labels = [TASK_DISPLAY[name] for name in cka.index]
    plot_df = cka.copy()
    plot_df.index = labels
    plot_df.columns = labels
    plt.figure(figsize=(9.8, 8.0))
    sns.heatmap(
        plot_df,
        annot=True,
        fmt=".2f",
        cmap="mako",
        vmin=0.0,
        vmax=1.0,
        square=True,
        annot_kws={"fontsize": 14, "weight": "bold"},
        cbar_kws={"label": "Linear CKA"},
    )
    plt.title("Unified Model: Task Representation Similarity", fontsize=18, weight="bold", pad=14)
    plt.xticks(fontsize=13, rotation=28, ha="right")
    plt.yticks(fontsize=13, rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "unified_task_cka_similarity_heatmap.png", dpi=220)
    plt.close()


def _plot_specialist_reference(cka: pd.DataFrame, output_dir: Path) -> None:
    tasks = [task for task in cka.index if task != "shared_encoder"]
    unified = cka.loc[tasks, tasks].copy()
    unified.index = [TASK_DISPLAY[name] for name in tasks]
    unified.columns = [TASK_DISPLAY[name] for name in tasks]
    specialist = pd.DataFrame(np.eye(len(tasks), dtype=float), index=unified.index, columns=unified.columns)
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.0), constrained_layout=True)
    sns.heatmap(unified, annot=True, fmt=".2f", cmap="mako", vmin=0.0, vmax=1.0, square=True, annot_kws={"fontsize": 13, "weight": "bold"}, cbar=False, ax=axes[0])
    sns.heatmap(specialist, annot=True, fmt=".2f", cmap="mako", vmin=0.0, vmax=1.0, square=True, annot_kws={"fontsize": 13, "weight": "bold"}, cbar=False, ax=axes[1])
    axes[0].set_title("Unified BerryMTL task sharing", fontsize=17, weight="bold", pad=12)
    axes[1].set_title("Four specialist pipelines\n(no shared task representation)", fontsize=17, weight="bold", pad=12)
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=28, labelsize=12)
        ax.tick_params(axis="y", labelrotation=0, labelsize=12)
    fig.savefig(output_dir / "unified_vs_specialist_task_similarity_reference.png", dpi=220)
    plt.close(fig)


def _plot_mds(cka: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    tasks = list(cka.index)
    distance = np.clip(1.0 - cka.to_numpy(dtype=float), 0.0, 1.0)
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto").fit_transform(distance)
    frame = pd.DataFrame({"task": tasks, "task_label": [TASK_DISPLAY[name] for name in tasks], "mds_1": coords[:, 0], "mds_2": coords[:, 1]})
    frame.to_csv(output_dir / "unified_task_mds_coordinates.csv", index=False)

    plt.figure(figsize=(9.6, 7.4))
    ax = plt.gca()
    for _, row in frame.iterrows():
        color = TASK_COLORS[row["task"]]
        ax.scatter(row["mds_1"], row["mds_2"], s=520, color=color, edgecolor="black", linewidth=1.2, zorder=3)
        ax.annotate(
            row["task_label"],
            (row["mds_1"], row["mds_2"]),
            xytext=(9, 8),
            textcoords="offset points",
            fontsize=13,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.82},
        )
    for idx, task_a in enumerate(tasks):
        for jdx, task_b in enumerate(tasks):
            if jdx <= idx:
                continue
            similarity = float(cka.loc[task_a, task_b])
            linewidth = 0.5 + 3.0 * similarity
            ax.plot(
                [coords[idx, 0], coords[jdx, 0]],
                [coords[idx, 1], coords[jdx, 1]],
                color="#999999",
                alpha=0.18 + 0.45 * similarity,
                linewidth=linewidth,
                zorder=1,
            )
    ax.set_title("Task Heads in the Unified Representation Space", fontsize=18, weight="bold", pad=14)
    ax.set_xlabel("MDS 1", fontsize=14)
    ax.set_ylabel("MDS 2", fontsize=14)
    ax.tick_params(labelsize=12)
    x_span = max(float(coords[:, 0].max() - coords[:, 0].min()), 1e-3)
    y_span = max(float(coords[:, 1].max() - coords[:, 1].min()), 1e-3)
    ax.set_xlim(float(coords[:, 0].min() - 0.18 * x_span), float(coords[:, 0].max() + 0.28 * x_span))
    ax.set_ylim(float(coords[:, 1].min() - 0.18 * y_span), float(coords[:, 1].max() + 0.18 * y_span))
    ax.grid(True, color="#dddddd", linewidth=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "unified_task_space_mds.png", dpi=220)
    plt.close()
    return frame


def _plot_shared_bar(cka: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for task_name in cka.index:
        if task_name == "shared_encoder":
            continue
        rows.append({"task": task_name, "task_label": TASK_DISPLAY[task_name], "cka_to_shared": float(cka.loc["shared_encoder", task_name])})
    frame = pd.DataFrame(rows).sort_values("cka_to_shared", ascending=False)
    frame.to_csv(output_dir / "unified_task_similarity_to_shared.csv", index=False)
    plt.figure(figsize=(9.8, 6.2))
    ax = sns.barplot(
        data=frame,
        x="task_label",
        y="cka_to_shared",
        hue="task",
        palette=TASK_COLORS,
        legend=False,
    )
    ax.set_ylim(0, 1)
    ax.set_title("How Strongly Each Head Retains Shared Encoder Structure", fontsize=18, weight="bold", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Linear CKA to shared encoder", fontsize=14)
    ax.tick_params(axis="x", labelsize=13, rotation=18)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "unified_task_similarity_to_shared_bar.png", dpi=220)
    plt.close()


def _plot_schema(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 5.7))
    ax.axis("off")
    boxes = {
        "input": (0.04, 0.42, 0.15, 0.16, "Input Image", "#f7f7f7"),
        "shared": (0.30, 0.40, 0.20, 0.20, "Shared Encoder + FPN", TASK_COLORS["shared_encoder"]),
        "det": (0.66, 0.74, 0.23, 0.13, "Detection Head", TASK_COLORS["detection_head"]),
        "seg": (0.66, 0.53, 0.23, 0.13, "Segmentation Head", TASK_COLORS["segmentation_head"]),
        "cls": (0.66, 0.32, 0.23, 0.13, "ROI Classification Head", TASK_COLORS["classification_roi_head"]),
        "cnt": (0.66, 0.11, 0.23, 0.13, "Counting Head", TASK_COLORS["counting_head"]),
    }
    for key, (x, y, w, h, label, color) in boxes.items():
        text_color = "white" if key in {"shared", "det", "seg", "cls", "cnt"} else "#222222"
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#222222", linewidth=1.2, alpha=0.94)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=13, weight="bold", color=text_color)
    def arrow(a: str, b: str) -> None:
        ax.annotate(
            "",
            xy=(boxes[b][0], boxes[b][1] + boxes[b][3] / 2),
            xytext=(boxes[a][0] + boxes[a][2], boxes[a][1] + boxes[a][3] / 2),
            arrowprops={"arrowstyle": "->", "linewidth": 1.8, "color": "#333333"},
        )
    arrow("input", "shared")
    for key in ["det", "seg", "cls", "cnt"]:
        arrow("shared", key)
    ax.text(0.30, 0.67, "Common feature space", ha="center", fontsize=14, color="#333333")
    ax.text(0.66, 0.91, "Task-specific lightweight heads", ha="left", fontsize=14, color="#333333")
    plt.tight_layout()
    plt.savefig(output_dir / "unified_task_shared_space_schema.png", dpi=220)
    plt.close()


def _copy_to_paper_figures(output_dir: Path, dirs: dict[str, Path]) -> None:
    figures_dir = dirs["paper_ready"] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "unified_task_cka_similarity_heatmap.png",
        "unified_task_space_mds.png",
        "unified_task_similarity_to_shared_bar.png",
        "unified_task_shared_space_schema.png",
        "unified_vs_specialist_task_similarity_reference.png",
    ]:
        source = output_dir / name
        if source.exists():
            shutil.copy2(source, figures_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize how unified task heads share representation space.")
    parser.add_argument("--config", default="configs/fresh_benchmark.yaml")
    parser.add_argument("--run-dir", default=None, help="CenterDet run directory. Defaults to latest calibrated run.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--splits", default="test", help="Comma-separated splits to analyze.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--profile-key", default="ours_centerdet")
    args = parser.parse_args()

    config = load_config(args.config)
    prepare_annotations(config)
    dirs = output_dirs(config)
    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)
    device = resolve_device(args.device)
    cfg = {**config.get("ours", {}), **config.get(args.profile_key, {})}

    run_dir = Path(args.run_dir) if args.run_dir else _latest_centerdet_run(dirs)
    metadata = _load_metadata(run_dir)
    image_size = int(metadata.get("image_size", cfg.get("image_size", 512)))
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 3))
    output_dir = dirs["paper_ready"] / "feature_space" / run_dir.name / "task_space"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = pd.read_csv(dirs["annotations"] / "image_manifest.csv")
    instances = pd.read_csv(dirs["annotations"] / "instances.csv")
    selected_splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    images = images[images["split"].isin(selected_splits)].reset_index(drop=True)
    instances = instances[instances["split"].isin(selected_splits)].reset_index(drop=True)

    model = _model_from_run(config, cfg, metadata, device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    loader_kwargs = _loader_kwargs(config, device)
    loader = DataLoader(
        BerryMTLInstanceDataset(images, instances, image_size, False),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=instance_collate,
        **loader_kwargs,
    )
    representations, image_rows = _extract_task_representations(
        model,
        loader,
        device,
        amp=bool(config.get("training", {}).get("amp", True)),
    )
    image_rows.to_csv(output_dir / "task_space_image_manifest.csv", index=False)
    task_order = ["shared_encoder", "detection_head", "segmentation_head", "classification_roi_head", "counting_head"]
    cka_values = np.eye(len(task_order), dtype=float)
    for idx, task_a in enumerate(task_order):
        for jdx, task_b in enumerate(task_order):
            if jdx <= idx:
                continue
            value = _linear_cka(representations[task_a], representations[task_b])
            cka_values[idx, jdx] = value
            cka_values[jdx, idx] = value
    cka = pd.DataFrame(cka_values, index=task_order, columns=task_order)
    cka.to_csv(output_dir / "unified_task_cka_similarity.csv", index_label="task")
    _plot_cka(cka, output_dir)
    _plot_specialist_reference(cka, output_dir)
    _plot_mds(cka, output_dir)
    _plot_shared_bar(cka, output_dir)
    _plot_schema(output_dir)
    _copy_to_paper_figures(output_dir, dirs)
    json_dump(
        {
            "source_run": str(run_dir.resolve()),
            "source_method": str(metadata.get("method", "")),
            "source_display_name": str(metadata.get("display_name", "")),
            "splits": sorted(selected_splits),
            "num_images": int(len(image_rows)),
            "task_feature_shapes": {name: list(value.shape) for name, value in representations.items()},
            "mean_task_similarity_off_diagonal": float((cka_values.sum() - len(task_order)) / (len(task_order) * (len(task_order) - 1))),
        },
        output_dir / "unified_task_space_summary.json",
    )
    print(f"task_space_dir={output_dir}")


if __name__ == "__main__":
    main()
