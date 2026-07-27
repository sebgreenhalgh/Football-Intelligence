"""Deterministic, lightweight model components for Football Observation Reasoner v0."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from football_intelligence.football_observation_reasoner.contracts import (
    CandidateState,
    EntityRole,
    KitState,
    PairRelation,
    ParticipationState,
    PitchState,
    SceneCandidateAssessment,
    SceneEnergyResult,
    TeamAffiliation,
)

NODE_HEAD_CLASSES: dict[str, tuple[str, ...]] = {
    "candidate_state": tuple(value.value for value in CandidateState),
    "role": tuple(value.value for value in EntityRole),
    "team": tuple(value.value for value in TeamAffiliation),
    "kit": tuple(value.value for value in KitState),
    "pitch": tuple(value.value for value in PitchState),
    "participation": tuple(value.value for value in ParticipationState),
}
PAIR_RELATION_CLASSES = tuple(value.value for value in PairRelation)


def freeze_visual_encoder(encoder: nn.Module) -> nn.Module:
    """Freeze an encoder in evaluation mode and clear any stale parameter gradients."""

    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return encoder


def assert_visual_encoder_frozen(encoder: nn.Module) -> None:
    """Raise when a supposedly frozen encoder could update weights or running state."""

    trainable = [name for name, parameter in encoder.named_parameters() if parameter.requires_grad]
    gradients = [name for name, parameter in encoder.named_parameters() if parameter.grad is not None]
    if trainable:
        raise RuntimeError(f"visual encoder has trainable parameters: {trainable}")
    if gradients:
        raise RuntimeError(f"visual encoder has parameter gradients: {gradients}")
    if encoder.training:
        raise RuntimeError("visual encoder must remain in evaluation mode")


class FrozenVisualEncoder(nn.Module):
    """Keep an official pretrained visual encoder frozen even when parent heads train."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = freeze_visual_encoder(encoder)
        self.training = False

    def train(self, mode: bool = True) -> FrozenVisualEncoder:
        del mode
        self.training = False
        self.encoder.eval()
        return self

    def forward(self, crops: Tensor) -> Tensor:
        assert_visual_encoder_frozen(self.encoder)
        with torch.no_grad():
            output = self.encoder(crops)
        if not isinstance(output, Tensor):
            raise TypeError("frozen visual encoder must return one tensor")
        return output.detach()


@dataclass(frozen=True)
class MaskedMultitaskLossResult:
    total: Tensor
    by_head: dict[str, Tensor]
    labelled_counts: dict[str, int]


@dataclass(frozen=True)
class MaskedFootpointLossResult:
    """Masked robust regression evidence for a two-coordinate footpoint head."""

    total: Tensor
    labelled_count: int
    robust_error: Tensor
    mean_log_variance: Tensor


def masked_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    availability: Tensor,
    *,
    class_weights: Tensor | None = None,
) -> tuple[Tensor, int]:
    """Compute cross entropy only for rows whose target is available."""

    if logits.ndim != 2:
        raise ValueError("masked cross entropy logits must have shape [rows, classes]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("masked cross entropy targets must have one value per row")
    if availability.ndim != 1 or availability.shape[0] != logits.shape[0]:
        raise ValueError("masked cross entropy availability must have one value per row")
    mask = availability.to(device=logits.device, dtype=torch.bool)
    count = int(mask.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    selected_targets = targets.to(device=logits.device, dtype=torch.long)[mask]
    weights = None
    if class_weights is not None:
        if class_weights.ndim != 1 or class_weights.shape[0] != logits.shape[1]:
            raise ValueError("class_weights must contain one value per logit class")
        weights = class_weights.to(device=logits.device, dtype=logits.dtype)
        if not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError("class_weights must be finite and strictly positive")
    return F.cross_entropy(logits[mask], selected_targets, weight=weights), count


def masked_multitask_cross_entropy(
    logits_by_head: Mapping[str, Tensor],
    targets_by_head: Mapping[str, Tensor],
    availability_by_head: Mapping[str, Tensor],
    *,
    loss_weights: Mapping[str, float] | None = None,
    class_weights_by_head: Mapping[str, Tensor] | None = None,
) -> MaskedMultitaskLossResult:
    """Combine masked classification heads without treating unknown labels as negatives."""

    if not logits_by_head:
        raise ValueError("at least one multitask head is required")
    missing_targets = set(logits_by_head) - set(targets_by_head)
    missing_masks = set(logits_by_head) - set(availability_by_head)
    if missing_targets or missing_masks:
        raise ValueError(
            f"masked multitask inputs are incomplete: targets={sorted(missing_targets)}, masks={sorted(missing_masks)}"
        )
    weights = dict(loss_weights or {})
    class_weights = dict(class_weights_by_head or {})
    unknown_weights = set(weights) - set(logits_by_head)
    if unknown_weights:
        raise ValueError(f"loss weights reference unknown heads: {sorted(unknown_weights)}")
    unknown_class_weights = set(class_weights) - set(logits_by_head)
    if unknown_class_weights:
        raise ValueError(f"class weights reference unknown heads: {sorted(unknown_class_weights)}")
    by_head: dict[str, Tensor] = {}
    counts: dict[str, int] = {}
    active: list[tuple[Tensor, float]] = []
    for name in sorted(logits_by_head):
        weight = float(weights.get(name, 1.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"loss weight for {name} must be finite and non-negative")
        loss, count = masked_cross_entropy(
            logits_by_head[name],
            targets_by_head[name],
            availability_by_head[name],
            class_weights=class_weights.get(name),
        )
        by_head[name] = loss
        counts[name] = count
        if count and weight > 0.0:
            active.append((loss, weight))
    if active:
        denominator = sum(weight for _, weight in active)
        total = sum(loss * weight for loss, weight in active) / denominator
    else:
        total = sum(logits.sum() * 0.0 for logits in logits_by_head.values())
    return MaskedMultitaskLossResult(total=total, by_head=by_head, labelled_counts=counts)


def masked_heteroscedastic_footpoint_loss(
    mean: Tensor,
    log_variance: Tensor,
    targets: Tensor,
    availability: Tensor,
    *,
    huber_delta: float = 0.25,
    minimum_log_variance: float = -8.0,
    maximum_log_variance: float = 6.0,
) -> MaskedFootpointLossResult:
    """Apply a masked robust heteroscedastic loss to normalized x/y residuals.

    The caller supplies target coordinates normalized by visible-box height.  A
    smooth-L1 data term limits the influence of uncertain or truncated human
    footpoints, while the learned log variance preserves coordinate-specific
    aleatoric uncertainty.  Unavailable rows contribute exactly zero.
    """

    expected_shape = (mean.shape[0], 2) if mean.ndim == 2 else None
    if mean.ndim != 2 or mean.shape[1] != 2:
        raise ValueError("footpoint mean must have shape [rows, 2]")
    if log_variance.shape != expected_shape or targets.shape != expected_shape:
        raise ValueError("footpoint log variance and targets must match mean shape [rows, 2]")
    if availability.ndim != 1 or availability.shape[0] != mean.shape[0]:
        raise ValueError("footpoint availability must contain one value per row")
    if not math.isfinite(huber_delta) or huber_delta <= 0.0:
        raise ValueError("huber_delta must be finite and positive")
    if (
        not math.isfinite(minimum_log_variance)
        or not math.isfinite(maximum_log_variance)
        or minimum_log_variance >= maximum_log_variance
    ):
        raise ValueError("footpoint log-variance bounds must be finite and increasing")
    if not torch.isfinite(mean).all() or not torch.isfinite(log_variance).all():
        raise ValueError("footpoint predictions must be finite")
    mask = availability.to(device=mean.device, dtype=torch.bool)
    count = int(mask.sum().item())
    if count == 0:
        zero = mean.sum() * 0.0 + log_variance.sum() * 0.0
        return MaskedFootpointLossResult(
            total=zero,
            labelled_count=0,
            robust_error=zero,
            mean_log_variance=zero,
        )
    selected_targets = targets.to(device=mean.device, dtype=mean.dtype)[mask]
    if not torch.isfinite(selected_targets).all():
        raise ValueError("available footpoint targets must be finite")
    selected_mean = mean[mask]
    selected_log_variance = log_variance[mask].clamp(
        min=minimum_log_variance,
        max=maximum_log_variance,
    )
    robust_error = F.smooth_l1_loss(
        selected_mean,
        selected_targets,
        beta=huber_delta,
        reduction="none",
    )
    # Gaussian-style attenuation applied to a robust data term.  The factor of
    # one half keeps this on the conventional heteroscedastic NLL scale.
    nll = 0.5 * (torch.exp(-selected_log_variance) * robust_error + selected_log_variance)
    return MaskedFootpointLossResult(
        total=nll.mean(),
        labelled_count=count,
        robust_error=robust_error.mean(),
        mean_log_variance=selected_log_variance.mean(),
    )


class HeteroscedasticFootpointHead(nn.Module):
    """Predict a normalized footpoint mean and per-coordinate log variance."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("footpoint head hidden dimension must be positive")
        self.mean = nn.Linear(hidden_dim, 2)
        self.log_variance = nn.Linear(hidden_dim, 2)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        if hidden.ndim != 2:
            raise ValueError("footpoint head input must have shape [rows, hidden]")
        return self.mean(hidden), self.log_variance(hidden)


class SoftSceneEnergyRanker(nn.Module):
    """Small trainable scene-ranking head that never makes hard decisions."""

    def __init__(self, feature_dim: int, *, hidden_dim: int = 16, seed: int = 7) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("scene ranker dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.initialization_seed = int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            self.energy = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1),
            )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"scene ranker features must have shape [rows, {self.feature_dim}]")
        if not torch.isfinite(features).all():
            raise ValueError("scene ranker features must be finite")
        return self.energy(features).squeeze(1)


def masked_scene_ranking_loss(
    energies: Tensor,
    clean_targets: Tensor,
    availability: Tensor,
    scene_ids: Tensor,
    *,
    margin: float = 0.2,
) -> tuple[Tensor, int]:
    """Rank clean candidates below non-clean candidates within each scene.

    This is a soft structured objective only: it returns a scalar loss and does
    not accept, suppress, invent, or delete any candidate.
    """

    row_count = energies.shape[0]
    if energies.ndim != 1:
        raise ValueError("scene energies must have shape [rows]")
    if any(value.ndim != 1 or value.shape[0] != row_count for value in (clean_targets, availability, scene_ids)):
        raise ValueError("scene ranking targets, availability, and IDs must contain one value per row")
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("scene ranking margin must be finite and non-negative")
    if not torch.isfinite(energies).all():
        raise ValueError("scene energies must be finite")
    available = availability.to(device=energies.device, dtype=torch.bool)
    clean = clean_targets.to(device=energies.device, dtype=torch.bool)
    groups = scene_ids.to(device=energies.device, dtype=torch.long)
    losses: list[Tensor] = []
    pair_count = 0
    for scene_id in torch.unique(groups[available], sorted=True):
        scene = available & (groups == scene_id)
        clean_energies = energies[scene & clean]
        non_clean_energies = energies[scene & ~clean]
        if not clean_energies.numel() or not non_clean_energies.numel():
            continue
        differences = margin + clean_energies[:, None] - non_clean_energies[None, :]
        losses.append(F.softplus(differences).mean())
        pair_count += int(differences.numel())
    if not losses:
        return energies.sum() * 0.0, 0
    return torch.stack(losses).mean(), pair_count


class LightweightGraphReasoner(nn.Module):
    """A small source-frame graph-attention model with joint node and pair heads."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        *,
        hidden_dim: int = 64,
        seed: int = 7,
    ) -> None:
        super().__init__()
        if node_feature_dim <= 0 or edge_feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("graph feature and hidden dimensions must be positive")
        self.node_feature_dim = int(node_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.initialization_seed = int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            self.node_encoder = nn.Sequential(
                nn.Linear(self.node_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.edge_encoder = nn.Sequential(
                nn.Linear(self.edge_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            pair_width = hidden_dim * 3
            self.edge_attention = nn.Linear(pair_width, 1)
            self.edge_message = nn.Sequential(
                nn.Linear(pair_width, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.node_update = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.node_norm = nn.LayerNorm(hidden_dim)
            self.node_heads = nn.ModuleDict(
                {name: nn.Linear(hidden_dim, len(classes)) for name, classes in sorted(NODE_HEAD_CLASSES.items())}
            )
            self.footpoint_head = HeteroscedasticFootpointHead(hidden_dim)
            self.pair_head = nn.Sequential(
                nn.Linear(pair_width, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, len(PAIR_RELATION_CLASSES)),
            )

    def _validate_inputs(self, node_features: Tensor, edge_index: Tensor, edge_features: Tensor) -> None:
        if node_features.ndim != 2 or node_features.shape[1] != self.node_feature_dim:
            raise ValueError(f"node_features must have shape [nodes, {self.node_feature_dim}]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        if edge_index.dtype != torch.long:
            raise ValueError("edge_index must use torch.long indices")
        if edge_features.ndim != 2 or edge_features.shape != (edge_index.shape[1], self.edge_feature_dim):
            raise ValueError(f"edge_features must have shape [edges, {self.edge_feature_dim}]")
        if not torch.isfinite(node_features).all() or not torch.isfinite(edge_features).all():
            raise ValueError("graph features must be finite")
        node_count = node_features.shape[0]
        if edge_index.numel():
            if int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= node_count:
                raise ValueError("edge_index references a missing node")
            left, right = edge_index[0], edge_index[1]
            if torch.any(left == right):
                raise ValueError("graph reasoner does not accept self edges")
            low, high = torch.minimum(left, right), torch.maximum(left, right)
            keys = low * max(1, node_count) + high
            if torch.unique(keys).numel() != keys.numel():
                raise ValueError("graph reasoner requires one undirected edge per candidate pair")

    def forward(self, node_features: Tensor, edge_index: Tensor, edge_features: Tensor) -> dict[str, Tensor]:
        self._validate_inputs(node_features, edge_index, edge_features)
        node_hidden = self.node_encoder(node_features)
        edge_count = edge_index.shape[1]
        if edge_count:
            low = torch.minimum(edge_index[0], edge_index[1])
            high = torch.maximum(edge_index[0], edge_index[1])
            canonical_keys = low * max(1, node_features.shape[0]) + high
            order = torch.argsort(canonical_keys, stable=True)
            low_sorted, high_sorted = low[order], high[order]
            edge_hidden_sorted = self.edge_encoder(edge_features[order])
            pair_hidden_sorted = torch.cat(
                (node_hidden[low_sorted], node_hidden[high_sorted], edge_hidden_sorted), dim=1
            )
            attention = torch.sigmoid(self.edge_attention(pair_hidden_sorted))
            messages = self.edge_message(pair_hidden_sorted) * attention
            aggregate = messages.new_zeros((node_features.shape[0], self.hidden_dim))
            # Each undirected message contributes symmetrically to both ends.
            # CPU index_add is deterministic and avoids one full edge mask per
            # node when evaluating the complete held-out graph.
            aggregate.index_add_(0, low_sorted, messages)
            aggregate.index_add_(0, high_sorted, messages)
            updated = self.node_norm(node_hidden + self.node_update(torch.cat((node_hidden, aggregate), dim=1)))
            pair_logits_sorted = self.pair_head(
                torch.cat((updated[low_sorted], updated[high_sorted], edge_hidden_sorted), dim=1)
            )
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(edge_count, device=order.device)
            pair_logits = pair_logits_sorted[inverse]
        else:
            aggregate = node_hidden.new_zeros((node_features.shape[0], self.hidden_dim))
            updated = self.node_norm(node_hidden + self.node_update(torch.cat((node_hidden, aggregate), dim=1)))
            pair_logits = node_hidden.new_empty((0, len(PAIR_RELATION_CLASSES)))
        output = {f"{name}_logits": head(updated) for name, head in self.node_heads.items()}
        footpoint_mean, footpoint_log_variance = self.footpoint_head(updated)
        output["footpoint_mean"] = footpoint_mean
        output["footpoint_log_variance"] = footpoint_log_variance
        output["pair_relation_logits"] = pair_logits
        output["node_embeddings"] = updated
        return output


def warning_only_scene_energy(
    candidates: Sequence[SceneCandidateAssessment],
    *,
    expected_visible_person_count: int | None = None,
    unresolved_threshold: float = 0.5,
) -> SceneEnergyResult:
    """Score scene coherence and emit warnings without changing candidate acceptance."""

    if expected_visible_person_count is not None and expected_visible_person_count < 0:
        raise ValueError("expected visible-person count cannot be negative")
    if not 0.0 <= unresolved_threshold <= 1.0:
        raise ValueError("unresolved threshold must be in [0, 1]")
    identifiers = [candidate.candidate_uuid for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("scene candidates must have unique UUIDs")
    accepted = tuple(
        sorted(candidate.candidate_uuid for candidate in candidates if candidate.accepted_as_independent_person)
    )
    count_under = expected_visible_person_count is not None and len(accepted) < expected_visible_person_count
    count_over = expected_visible_person_count is not None and len(accepted) > expected_visible_person_count
    active_goalkeepers: dict[TeamAffiliation, int] = {
        TeamAffiliation.TEAM_1: 0,
        TeamAffiliation.TEAM_2: 0,
    }
    unresolved_count = 0
    risk_values: list[float] = []
    for candidate in candidates:
        axes = candidate.axes
        if (
            candidate.accepted_as_independent_person
            and axes.role is EntityRole.GOALKEEPER
            and axes.team in active_goalkeepers
            and axes.pitch is PitchState.ON_PITCH
            and axes.participation is not ParticipationState.OFF_PITCH_SUBSTITUTE_OR_WARMING
        ):
            active_goalkeepers[axes.team] += 1
        unresolved = (
            axes.candidate_state is CandidateState.AMBIGUOUS_UNRESOLVED
            or candidate.unresolved_probability >= unresolved_threshold
        )
        unresolved_count += int(unresolved)
        risk_values.append(
            max(candidate.unresolved_probability, candidate.duplicate_probability, candidate.merged_probability)
        )
    goalkeeper_conflict = any(count > 1 for count in active_goalkeepers.values())
    unresolved_warning = unresolved_count > 0
    reasons: list[str] = []
    if count_under:
        reasons.append("COUNT_UNDER_RESOLUTION_WARNING")
    if count_over:
        reasons.append("COUNT_OVER_RESOLUTION_WARNING")
    if goalkeeper_conflict:
        reasons.append("GOALKEEPER_TEAM_CONFLICT_WARNING")
    if unresolved_warning:
        reasons.append("UNRESOLVED_SCENE_WARNING")
    mean_risk = sum(risk_values) / len(risk_values) if risk_values else 0.0
    warning_penalty = 0.08 * sum((count_under, count_over, goalkeeper_conflict, unresolved_warning))
    coherent_score = round(max(0.0, min(1.0, 1.0 - mean_risk - warning_penalty)), 8)
    return SceneEnergyResult(
        coherent_scene_score=coherent_score,
        accepted_candidate_uuids_before=accepted,
        accepted_candidate_uuids_after=accepted,
        count_under_resolution_warning=count_under,
        count_over_resolution_warning=count_over,
        goalkeeper_team_conflict_warning=goalkeeper_conflict,
        unresolved_scene_warning=unresolved_warning,
        warning_reasons=tuple(reasons),
    )


def graph_model_specification(model: LightweightGraphReasoner) -> dict[str, Any]:
    """Return a stable architecture specification without parameter values or runtime state."""

    return {
        "schema_version": "football_intelligence.football_observation_reasoner.graph_model.v1",
        "model_type": "LIGHTWEIGHT_SOURCE_FRAME_GRAPH_ATTENTION",
        "node_feature_dim": model.node_feature_dim,
        "edge_feature_dim": model.edge_feature_dim,
        "hidden_dim": model.hidden_dim,
        "initialization_seed": model.initialization_seed,
        "node_heads": {name: list(classes) for name, classes in sorted(NODE_HEAD_CLASSES.items())},
        "footpoint_head": {
            "mean_coordinates": 2,
            "heteroscedastic_log_variance_coordinates": 2,
            "coordinate_parameterization": "VISIBLE_HEIGHT_NORMALIZED_BOX_BOTTOM_RESIDUAL_XY",
        },
        "pair_relation_classes": list(PAIR_RELATION_CLASSES),
        "identity_head_present": False,
        "temporal_acceptance_head_present": False,
        "hard_count_head_present": False,
        "deterministic_no_dropout": True,
    }
