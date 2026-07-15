from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from blueberry_multitask.berrycluster import ConvNormAct, MultiScaleFusion


def _make_query_grid(num_queries: int) -> torch.Tensor:
    side = int(num_queries**0.5)
    if side * side < num_queries:
        side += 1
    coords = torch.linspace(0.5 / side, 1.0 - 0.5 / side, side)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    return grid[:num_queries].contiguous()


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


@dataclass(frozen=True)
class OcclusionQueryOutputSpec:
    queries: int
    ripeness_classes: int
    mask_stride: int
    token_dim: int


class OcclusionAwareQueryNet(nn.Module):
    """
    Query-based berry disentangling model for small occluded blueberries.

    The architecture is deliberately not "tile + NMS". It combines four signals:
    anchored berry queries for individual instances, semantic support for green
    berry/leaf ambiguity, classwise density for count conservation, and query masks
    for visible instance segmentation.
    """

    def __init__(
        self,
        model_name: str,
        ripeness_classes: int,
        pretrained: bool = True,
        hidden_dim: int = 192,
        token_dim: int = 192,
        num_queries: int = 384,
        memory_grid: int = 16,
        decoder_layers: int = 3,
        decoder_heads: int = 6,
        mask_dim: int = 192,
        identity_dim: int = 32,
    ) -> None:
        super().__init__()
        import timm

        self.ripeness_classes = int(ripeness_classes)
        self.num_queries = int(num_queries)
        self.memory_grid = int(memory_grid)
        self.identity_dim = int(identity_dim)
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.encoder_channels = list(self.encoder.feature_info.channels())
        self.fusion = MultiScaleFusion(self.encoder_channels, hidden_dim)
        self.memory_projection = nn.Sequential(
            ConvNormAct(hidden_dim, token_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d((self.memory_grid, self.memory_grid)),
        )
        self.memory_pos = nn.Parameter(torch.zeros(1, self.memory_grid * self.memory_grid, token_dim))
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, token_dim) * 0.02)
        self.query_reference = nn.Parameter(_inverse_sigmoid(_make_query_grid(self.num_queries)))
        self.denoise_query_embed = nn.Parameter(torch.randn(1, token_dim) * 0.02)
        self.denoise_label_embed = nn.Embedding(self.ripeness_classes, token_dim)
        self.reference_embed = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=decoder_heads,
            dim_feedforward=token_dim * 4,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.query_norm = nn.LayerNorm(token_dim)

        self.objectness_head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))
        self.center_head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 2))
        self.size_head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 2))
        self.ripeness_head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, self.ripeness_classes))
        self.visibility_head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))
        self.occlusion_head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))
        self.count_quality_head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))
        self.identity_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, self.identity_dim),
        )

        self.mask_features = nn.Sequential(
            ConvNormAct(hidden_dim, mask_dim, kernel_size=3, dropout=0.04),
            ConvNormAct(mask_dim, mask_dim, kernel_size=3, dropout=0.02),
        )
        self.mask_embed_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, mask_dim),
        )
        self.semantic_support_head = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, dropout=0.05),
            nn.Conv2d(hidden_dim, self.ripeness_classes + 1, kernel_size=1),
        )
        self.cluster_cardinality_head = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim // 2, kernel_size=3, dropout=0.04),
            nn.Conv2d(hidden_dim // 2, self.ripeness_classes + 1, kernel_size=1),
        )
        self.density_head = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim // 2, kernel_size=3, dropout=0.04),
            nn.Conv2d(hidden_dim // 2, self.ripeness_classes + 1, kernel_size=1),
        )
        self._init_heads()

    def _init_heads(self) -> None:
        for module, bias in [
            (self.objectness_head[-1], -3.0),
            (self.visibility_head[-1], 1.0),
            (self.occlusion_head[-1], -1.0),
            (self.cluster_cardinality_head[-1], -8.0),
            (self.density_head[-1], -6.0),
        ]:
            if isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
                nn.init.constant_(module.bias, bias)

    def _decode_queries(
        self,
        queries: torch.Tensor,
        reference: torch.Tensor,
        memory: torch.Tensor,
        mask_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query_tokens = self.query_norm(self.decoder(tgt=queries, memory=memory))
        centers = torch.sigmoid(self.center_head(query_tokens) + _inverse_sigmoid(reference))
        mask_embed = self.mask_embed_head(query_tokens)
        count_quality_logits = self.count_quality_head(query_tokens).squeeze(-1)
        identity_embeddings = F.normalize(self.identity_head(query_tokens), dim=-1)
        return {
            "objectness_logits": self.objectness_head(query_tokens).squeeze(-1),
            "centers": centers,
            "sizes": torch.sigmoid(self.size_head(query_tokens)),
            "ripeness_logits": self.ripeness_head(query_tokens),
            "visibility": torch.sigmoid(self.visibility_head(query_tokens)).squeeze(-1),
            "occlusion_logits": self.occlusion_head(query_tokens).squeeze(-1),
            "count_quality_logits": count_quality_logits,
            "count_quality": torch.sigmoid(count_quality_logits),
            "query_masks": torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features),
            "query_tokens": query_tokens,
            "identity_embeddings": identity_embeddings,
        }

    @property
    def output_spec(self) -> OcclusionQueryOutputSpec:
        return OcclusionQueryOutputSpec(
            queries=self.num_queries,
            ripeness_classes=self.ripeness_classes,
            mask_stride=4,
            token_dim=self.query_embed.shape[-1],
        )

    def forward(
        self,
        x: torch.Tensor,
        denoise_centers: torch.Tensor | None = None,
        denoise_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        fused = self.fusion(features, self.encoder_channels)
        memory = self.memory_projection(fused).flatten(2).transpose(1, 2)
        memory = memory + self.memory_pos[:, : memory.shape[1]]
        reference = self.query_reference.sigmoid()
        queries = self.query_embed + self.reference_embed(reference)
        queries = queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        mask_features = self.mask_features(fused)
        decoded = self._decode_queries(queries, reference.unsqueeze(0), memory, mask_features)

        output = {
            "objectness_logits": decoded["objectness_logits"],
            "centers": decoded["centers"],
            "sizes": decoded["sizes"],
            "ripeness_logits": decoded["ripeness_logits"],
            "visibility": decoded["visibility"],
            "occlusion_logits": decoded["occlusion_logits"],
            "count_quality_logits": decoded["count_quality_logits"],
            "count_quality": decoded["count_quality"],
            "query_masks": decoded["query_masks"],
            "semantic_support": F.interpolate(
                self.semantic_support_head(fused),
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ),
            "cluster_cardinality": F.softplus(self.cluster_cardinality_head(fused)),
            "density": F.softplus(self.density_head(fused)),
            "feature": fused,
            "query_tokens": decoded["query_tokens"],
            "identity_embeddings": decoded["identity_embeddings"],
        }
        if denoise_centers is not None and denoise_labels is not None and denoise_centers.numel() > 0:
            denoise_reference = denoise_centers.clamp(1e-4, 1.0 - 1e-4)
            denoise_queries = (
                self.denoise_query_embed.view(1, 1, -1)
                + self.reference_embed(denoise_reference)
                + self.denoise_label_embed(denoise_labels.clamp(0, self.ripeness_classes - 1))
            )
            denoise_decoded = self._decode_queries(denoise_queries, denoise_reference, memory, mask_features)
            output.update(
                {
                    "denoise_objectness_logits": denoise_decoded["objectness_logits"],
                    "denoise_centers": denoise_decoded["centers"],
                    "denoise_sizes": denoise_decoded["sizes"],
                    "denoise_ripeness_logits": denoise_decoded["ripeness_logits"],
                    "denoise_visibility": denoise_decoded["visibility"],
                    "denoise_occlusion_logits": denoise_decoded["occlusion_logits"],
                    "denoise_count_quality_logits": denoise_decoded["count_quality_logits"],
                    "denoise_count_quality": denoise_decoded["count_quality"],
                    "denoise_query_masks": denoise_decoded["query_masks"],
                    "denoise_query_tokens": denoise_decoded["query_tokens"],
                    "denoise_identity_embeddings": denoise_decoded["identity_embeddings"],
                }
            )
        return output


def create_occlusion_query_model(config: dict[str, Any], pretrained: bool | None = None) -> OcclusionAwareQueryNet:
    classes = list(config["classes"])
    cfg = dict(config.get("occlusion_query", {}))
    if pretrained is None:
        pretrained = bool(config.get("training", {}).get("pretrained", True))
    return OcclusionAwareQueryNet(
        model_name=str(cfg.get("model_name", "pvt_v2_b1")),
        ripeness_classes=len(classes),
        pretrained=bool(pretrained),
        hidden_dim=int(cfg.get("hidden_dim", 192)),
        token_dim=int(cfg.get("token_dim", 192)),
        num_queries=int(cfg.get("num_queries", 384)),
        memory_grid=int(cfg.get("memory_grid", 16)),
        decoder_layers=int(cfg.get("decoder_layers", 3)),
        decoder_heads=int(cfg.get("decoder_heads", 6)),
        mask_dim=int(cfg.get("mask_dim", 192)),
        identity_dim=int(cfg.get("identity_dim", 32)),
    )
