from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align


@dataclass(frozen=True)
class BerryClusterOutputSpec:
    semantic_classes: int
    ripeness_classes: int
    density_channels: int
    cluster_token_dim: int


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.0):
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class MultiScaleFusion(nn.Module):
    def __init__(self, channels: list[int], hidden_dim: int):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(channel, hidden_dim, kernel_size=1) for channel in channels])
        self.refine = nn.Sequential(
            ConvNormAct(hidden_dim * len(channels), hidden_dim, kernel_size=3, dropout=0.06),
            ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, dropout=0.04),
        )

    @staticmethod
    def _nchw(feature: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if feature.ndim == 4 and feature.shape[1] != expected_channels and feature.shape[-1] == expected_channels:
            return feature.permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, features: list[torch.Tensor], channels: list[int]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        projected = []
        for feature, channel, projection in zip(features, channels, self.projections):
            feature = self._nchw(feature, channel)
            projected.append(F.interpolate(projection(feature), size=target_size, mode="bilinear", align_corners=False))
        return self.refine(torch.cat(projected, dim=1))


class ClusterTokenHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        token_dim: int,
        ripeness_classes: int,
        roi_size: int,
        attention_layers: int,
        attention_heads: int,
    ):
        super().__init__()
        self.roi_size = int(roi_size)
        self.project = nn.Sequential(
            nn.Conv2d(feature_dim + 1, token_dim, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=attention_heads,
            dim_feedforward=token_dim * 3,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_mixer = nn.TransformerEncoder(encoder_layer, num_layers=attention_layers)
        self.ripeness_head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, ripeness_classes))
        self.cluster_count_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, 1),
        )
        self.cluster_quality_head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))
        self.yield_proxy_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, 1),
        )

    @staticmethod
    def _rois_from_boxes(boxes: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        rows = []
        for batch_idx, batch_boxes in enumerate(boxes):
            if batch_boxes.numel() == 0:
                continue
            batch_boxes = batch_boxes.to(device=device, dtype=torch.float32)
            batch_index = torch.full((batch_boxes.shape[0], 1), float(batch_idx), device=device)
            rows.append(torch.cat([batch_index, batch_boxes], dim=1))
        if not rows:
            return torch.zeros((0, 5), dtype=torch.float32, device=device)
        return torch.cat(rows, dim=0)

    def forward(
        self,
        feature: torch.Tensor,
        foreground: torch.Tensor,
        cluster_boxes: list[torch.Tensor],
        image_size: int,
    ) -> dict[str, torch.Tensor]:
        rois = self._rois_from_boxes(cluster_boxes, feature.device)
        if rois.numel() == 0:
            empty_tokens = torch.zeros((0, self.ripeness_head[1].in_features), dtype=feature.dtype, device=feature.device)
            return {
                "cluster_tokens": empty_tokens,
                "cluster_ripeness_logits": torch.zeros((0, self.ripeness_head[-1].out_features), dtype=feature.dtype, device=feature.device),
                "cluster_count": torch.zeros((0, 1), dtype=feature.dtype, device=feature.device),
                "cluster_quality": torch.zeros((0, 1), dtype=feature.dtype, device=feature.device),
                "yield_proxy": torch.zeros((0, 1), dtype=feature.dtype, device=feature.device),
            }

        spatial_scale = feature.shape[-1] / float(image_size)
        pooled = roi_align(feature, rois, output_size=(self.roi_size, self.roi_size), spatial_scale=spatial_scale, aligned=True)
        pooled_fg = roi_align(foreground, rois, output_size=(self.roi_size, self.roi_size), spatial_scale=1.0, aligned=True)
        tokens = self.project(torch.cat([pooled, pooled_fg], dim=1))

        counts_per_image = [int(batch_boxes.shape[0]) for batch_boxes in cluster_boxes]
        mixed_parts = []
        offset = 0
        for count in counts_per_image:
            if count <= 0:
                continue
            local = tokens[offset : offset + count].unsqueeze(0)
            mixed_parts.append(self.token_mixer(local).squeeze(0))
            offset += count
        mixed = torch.cat(mixed_parts, dim=0) if mixed_parts else tokens
        return {
            "cluster_tokens": mixed,
            "cluster_ripeness_logits": self.ripeness_head(mixed),
            "cluster_count": F.softplus(self.cluster_count_head(mixed)),
            "cluster_quality": self.cluster_quality_head(mixed),
            "yield_proxy": F.softplus(self.yield_proxy_head(mixed)),
        }


class BerryClusterNet(nn.Module):
    """
    Cluster-aware multi-task model for dense blueberry scenes.

    The intended novelty is not a new generic detector block. It is the task coupling:
    class-agnostic centers separate individual berries, classwise density counts berries
    inside merged clusters, semantic masks localize ripeness regions, and cluster tokens
    estimate per-cluster count/ripeness/yield proxies.
    """

    def __init__(
        self,
        model_name: str,
        semantic_classes: int,
        ripeness_classes: int,
        pretrained: bool = True,
        hidden_dim: int = 160,
        token_dim: int = 192,
        roi_size: int = 7,
        attention_layers: int = 2,
        attention_heads: int = 4,
    ):
        super().__init__()
        import timm

        self.semantic_classes = int(semantic_classes)
        self.ripeness_classes = int(ripeness_classes)
        self.density_channels = int(ripeness_classes + 1)
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.encoder_channels = list(self.encoder.feature_info.channels())
        self.fusion = MultiScaleFusion(self.encoder_channels, hidden_dim)
        self.semantic_head = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, dropout=0.08),
            nn.Conv2d(hidden_dim, semantic_classes, kernel_size=1),
        )
        self.boundary_head = nn.Sequential(ConvNormAct(hidden_dim, hidden_dim // 2), nn.Conv2d(hidden_dim // 2, 1, 1))
        self.center_head = nn.Sequential(ConvNormAct(hidden_dim, hidden_dim // 2), nn.Conv2d(hidden_dim // 2, 1, 1))
        self.size_head = nn.Sequential(ConvNormAct(hidden_dim, hidden_dim // 2), nn.Conv2d(hidden_dim // 2, 2, 1))
        self.offset_head = nn.Sequential(ConvNormAct(hidden_dim, hidden_dim // 2), nn.Conv2d(hidden_dim // 2, 2, 1))
        self.density_head = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, dropout=0.04),
            nn.Conv2d(hidden_dim, self.density_channels, kernel_size=1),
        )
        self.global_count_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(hidden_dim, self.density_channels),
        )
        self.cluster_token_head = ClusterTokenHead(
            feature_dim=hidden_dim,
            token_dim=token_dim,
            ripeness_classes=ripeness_classes,
            roi_size=roi_size,
            attention_layers=attention_layers,
            attention_heads=attention_heads,
        )
        self._init_sparse_prediction_heads()

    def _init_sparse_prediction_heads(self) -> None:
        # Berry centers and density maps are sparse. A negative initial bias prevents
        # the density integral from starting as thousands of false berries.
        for module, bias in [
            (self.boundary_head[-1], -4.0),
            (self.center_head[-1], -4.0),
            (self.density_head[-1], -6.0),
        ]:
            if isinstance(module, nn.Conv2d) and module.bias is not None:
                nn.init.constant_(module.bias, bias)

    @property
    def output_spec(self) -> BerryClusterOutputSpec:
        return BerryClusterOutputSpec(
            semantic_classes=self.semantic_classes,
            ripeness_classes=self.ripeness_classes,
            density_channels=self.density_channels,
            cluster_token_dim=self.cluster_token_head.ripeness_head[1].in_features,
        )

    def forward(self, x: torch.Tensor, cluster_boxes: list[torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        fused = self.fusion(features, self.encoder_channels)
        semantic_logits = F.interpolate(self.semantic_head(fused), size=input_size, mode="bilinear", align_corners=False)
        boundary_logits = F.interpolate(self.boundary_head(fused), size=input_size, mode="bilinear", align_corners=False)
        density_logits = self.density_head(fused)
        center_logits = self.center_head(fused)
        output: dict[str, torch.Tensor] = {
            "semantic": semantic_logits,
            "boundary": boundary_logits,
            "center_heatmap": center_logits,
            "size": F.softplus(self.size_head(fused)),
            "offset": torch.sigmoid(self.offset_head(fused)),
            "density": F.softplus(density_logits),
            "global_count": F.softplus(self.global_count_head(fused)),
            "feature": fused,
        }
        if cluster_boxes is not None:
            foreground = torch.softmax(semantic_logits, dim=1)[:, 1:].sum(dim=1, keepdim=True)
            output.update(
                self.cluster_token_head(
                    fused,
                    foreground,
                    cluster_boxes=cluster_boxes,
                    image_size=int(input_size[-1]),
                )
            )
        return output


def create_berrycluster_model(config: dict[str, Any], pretrained: bool | None = None) -> BerryClusterNet:
    classes = list(config["classes"])
    cfg = dict(config.get("berrycluster", {}))
    if pretrained is None:
        pretrained = bool(config.get("training", {}).get("pretrained", True))
    return BerryClusterNet(
        model_name=str(cfg.get("model_name", "convnextv2_tiny.fcmae_ft_in22k_in1k")),
        semantic_classes=len(classes) + 1,
        ripeness_classes=len(classes),
        pretrained=bool(pretrained),
        hidden_dim=int(cfg.get("hidden_dim", 160)),
        token_dim=int(cfg.get("token_dim", 192)),
        roi_size=int(cfg.get("roi_size", 7)),
        attention_layers=int(cfg.get("attention_layers", 2)),
        attention_heads=int(cfg.get("attention_heads", 4)),
    )
