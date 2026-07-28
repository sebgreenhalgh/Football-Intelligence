"""Deterministic grouped training and calibration helpers for M5.5G.7B.

The helpers in this module operate on already-materialized node and pair
features.  They deliberately do not own a visual encoder, a pitch polygon
model, identity state, temporal state, or scene-count prior.  In particular,
``pitch`` remains an auxiliary learned head: authoritative pitch-state routing
is performed elsewhere from the human-confirmed polygon and the estimated
footpoint with uncertainty.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from football_intelligence.football_observation_reasoner.hierarchical_selection import (
    HierarchicalSoftConditioningNodeModel,
    MultitaskNodeMLP,
    SymmetricPairMLP,
)
from football_intelligence.football_observation_reasoner.models import (
    NODE_HEAD_CLASSES,
    PAIR_RELATION_CLASSES,
    masked_heteroscedastic_footpoint_loss,
)
from football_intelligence.review_chassis.hashing import stable_hash

DEVELOPMENT_SCOPE = "SINGLE_MATCH_GROUPED_DEVELOPMENT_ONLY"
DEFAULT_OUTER_FOLD_COUNT = 5
DEFAULT_INNER_FOLD_COUNT = 4
DEFAULT_SPLIT_SEED = "M5_5G7B_NESTED_GROUPED_CALIBRATION_V1"
DEFAULT_TRAINING_SEED = 5720
CLASSIFICATION_HEADS = tuple(sorted(NODE_HEAD_CLASSES))
FORBIDDEN_HEAD_TOKENS = ("certainty", "identity", "temporal", "count")


def _as_group_ids(group_ids: Sequence[Any]) -> tuple[str, ...]:
    values = tuple(str(value).strip() for value in group_ids)
    if not values or any(not value for value in values):
        raise ValueError("group_ids must contain at least one non-empty group identifier")
    return values


def _as_outer_folds(outer_fold_ids: Sequence[int], row_count: int, fold_count: int) -> tuple[int, ...]:
    if fold_count < 2:
        raise ValueError("grouped outer evaluation requires at least two folds")
    if len(outer_fold_ids) != row_count:
        raise ValueError("outer_fold_ids must contain one assignment per row")
    folds = tuple(int(value) for value in outer_fold_ids)
    if any(value < 0 or value >= fold_count for value in folds):
        raise ValueError(f"outer fold assignments must be in [0, {fold_count})")
    if set(folds) != set(range(fold_count)):
        raise ValueError("every declared outer fold must contain at least one row")
    return folds


def validate_grouped_outer_folds(
    group_ids: Sequence[Any],
    outer_fold_ids: Sequence[int],
    *,
    fold_count: int = DEFAULT_OUTER_FOLD_COUNT,
) -> dict[str, Any]:
    """Validate exact inherited outer folds without changing an assignment.

    This function accepts assignments but never generates replacements.  It is
    therefore suitable for validating the immutable G7A five-fold manifest.
    """

    groups = _as_group_ids(group_ids)
    folds = _as_outer_folds(outer_fold_ids, len(groups), fold_count)
    folds_by_group: dict[str, set[int]] = defaultdict(set)
    for group, fold in zip(groups, folds, strict=True):
        folds_by_group[group].add(fold)
    split_groups = sorted(group for group, values in folds_by_group.items() if len(values) != 1)
    if split_groups:
        raise ValueError(f"source groups span outer folds: {split_groups}")
    fold_rows = {fold: sum(value == fold for value in folds) for fold in range(fold_count)}
    fold_groups = {
        fold: sum(next(iter(values)) == fold for values in folds_by_group.values()) for fold in range(fold_count)
    }
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.grouped_outer_fold_validation.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "row_count": len(groups),
        "group_count": len(folds_by_group),
        "fold_count": fold_count,
        "fold_row_counts": fold_rows,
        "fold_group_counts": fold_groups,
        "source_group_cross_fold_count": 0,
        "assignments_changed": False,
        "passed": True,
    }
    payload["validation_hash"] = stable_hash(payload)
    return payload


def grouped_inner_fold_assignments(
    group_ids: Sequence[Any],
    outer_fold_ids: Sequence[int],
    held_out_outer_fold: int,
    *,
    outer_fold_count: int = DEFAULT_OUTER_FOLD_COUNT,
    inner_fold_count: int = DEFAULT_INNER_FOLD_COUNT,
    seed: str = DEFAULT_SPLIT_SEED,
) -> tuple[int, ...]:
    """Assign outer-training groups deterministically to inner folds.

    Rows in the held-out outer fold receive ``-1``.  The remaining source
    groups are ordered only by a fixed hash of the group identifier and then
    distributed round-robin.  Labels and model performance never enter the
    assignment.
    """

    groups = _as_group_ids(group_ids)
    folds = _as_outer_folds(outer_fold_ids, len(groups), outer_fold_count)
    validate_grouped_outer_folds(groups, folds, fold_count=outer_fold_count)
    if held_out_outer_fold < 0 or held_out_outer_fold >= outer_fold_count:
        raise ValueError("held_out_outer_fold is outside the declared outer fold range")
    if inner_fold_count < 2:
        raise ValueError("nested grouped calibration requires at least two inner folds")

    training_groups = sorted(
        {group for group, fold in zip(groups, folds, strict=True) if fold != held_out_outer_fold},
        key=lambda group: (
            stable_hash({"seed": seed, "outer_fold": held_out_outer_fold, "source_group_id": group}),
            group,
        ),
    )
    if len(training_groups) < inner_fold_count:
        raise ValueError("outer-training data has fewer source groups than requested inner folds")
    inner_by_group = {group: index % inner_fold_count for index, group in enumerate(training_groups)}
    assignments = tuple(
        -1 if fold == held_out_outer_fold else inner_by_group[group] for group, fold in zip(groups, folds, strict=True)
    )
    if set(value for value in assignments if value >= 0) != set(range(inner_fold_count)):
        raise RuntimeError("deterministic inner-fold allocation produced an empty fold")
    return assignments


@dataclass(frozen=True)
class NestedGroupedSplit:
    """One outer-test and inner-calibration split with group-disjoint rows."""

    outer_fold: int
    inner_fold: int
    training_indices: tuple[int, ...]
    calibration_indices: tuple[int, ...]
    outer_test_indices: tuple[int, ...]
    training_group_ids: tuple[str, ...]
    calibration_group_ids: tuple[str, ...]
    outer_test_group_ids: tuple[str, ...]
    split_hash: str


def nested_grouped_oof_splits(
    group_ids: Sequence[Any],
    outer_fold_ids: Sequence[int],
    *,
    outer_fold_count: int = DEFAULT_OUTER_FOLD_COUNT,
    inner_fold_count: int = DEFAULT_INNER_FOLD_COUNT,
    seed: str = DEFAULT_SPLIT_SEED,
) -> tuple[NestedGroupedSplit, ...]:
    """Build fixed nested grouped splits for leakage-safe calibration.

    For a given outer fold, every outer-training row appears in exactly one
    inner calibration split.  Outer-test rows never appear in either training
    or calibration for that outer fold.
    """

    groups = _as_group_ids(group_ids)
    folds = _as_outer_folds(outer_fold_ids, len(groups), outer_fold_count)
    validate_grouped_outer_folds(groups, folds, fold_count=outer_fold_count)
    result: list[NestedGroupedSplit] = []
    for outer_fold in range(outer_fold_count):
        inner_assignments = grouped_inner_fold_assignments(
            groups,
            folds,
            outer_fold,
            outer_fold_count=outer_fold_count,
            inner_fold_count=inner_fold_count,
            seed=seed,
        )
        outer_test = tuple(index for index, fold in enumerate(folds) if fold == outer_fold)
        for inner_fold in range(inner_fold_count):
            calibration = tuple(index for index, value in enumerate(inner_assignments) if value == inner_fold)
            training = tuple(
                index for index, value in enumerate(inner_assignments) if value >= 0 and value != inner_fold
            )
            training_groups = tuple(sorted({groups[index] for index in training}))
            calibration_groups = tuple(sorted({groups[index] for index in calibration}))
            outer_test_groups = tuple(sorted({groups[index] for index in outer_test}))
            if set(training_groups) & set(calibration_groups):
                raise RuntimeError("a source group crossed inner training and calibration")
            if (set(training_groups) | set(calibration_groups)) & set(outer_test_groups):
                raise RuntimeError("an outer-test source group leaked into inner fitting")
            split_payload = {
                "seed": seed,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "training_indices": training,
                "calibration_indices": calibration,
                "outer_test_indices": outer_test,
                "training_group_ids": training_groups,
                "calibration_group_ids": calibration_groups,
                "outer_test_group_ids": outer_test_groups,
            }
            result.append(
                NestedGroupedSplit(
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    training_indices=training,
                    calibration_indices=calibration,
                    outer_test_indices=outer_test,
                    training_group_ids=training_groups,
                    calibration_group_ids=calibration_groups,
                    outer_test_group_ids=outer_test_groups,
                    split_hash=stable_hash(split_payload),
                )
            )
    return tuple(result)


def frozen_feature_tensor(features: Tensor) -> Tensor:
    """Return finite, detached float features that cannot backpropagate to an encoder."""

    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("precomputed features must have shape [rows, dimensions]")
    if not torch.isfinite(features).all():
        raise ValueError("precomputed features must be finite")
    return features.detach().to(dtype=torch.float32)


def _check_head_names(names: Sequence[str]) -> None:
    invalid = sorted(
        name
        for name in names
        if name not in CLASSIFICATION_HEADS or any(token in name.lower() for token in FORBIDDEN_HEAD_TOKENS)
    )
    if invalid:
        raise ValueError(f"unsupported or forbidden trainable heads: {invalid}")


def class_balanced_weights(
    targets: Tensor,
    availability: Tensor,
    class_count: int,
) -> tuple[Tensor, tuple[int, ...]]:
    """Return inverse-frequency weights computed from available training rows only."""

    if class_count < 2:
        raise ValueError("class_count must be at least two")
    if targets.ndim != 1 or availability.ndim != 1 or targets.shape != availability.shape:
        raise ValueError("targets and availability must be aligned one-dimensional tensors")
    mask = availability.to(device=targets.device, dtype=torch.bool)
    selected = targets.to(dtype=torch.long)[mask]
    if selected.numel() and (int(selected.min()) < 0 or int(selected.max()) >= class_count):
        raise ValueError("available targets contain an out-of-range class index")
    counts = torch.bincount(selected, minlength=class_count)
    supported = counts > 0
    weights = torch.ones(class_count, dtype=torch.float32, device=targets.device)
    if supported.any():
        inverse = selected.numel() / counts[supported].to(dtype=torch.float32)
        inverse = inverse / inverse.mean()
        weights[supported] = inverse
    return weights, tuple(int(value) for value in counts.tolist())


@dataclass(frozen=True)
class MaskedClassificationLoss:
    total: Tensor
    by_head: dict[str, Tensor]
    labelled_counts: dict[str, int]
    class_counts: dict[str, tuple[int, ...]]
    class_weights: dict[str, Tensor]


def masked_class_balanced_multitask_loss(
    logits_by_head: Mapping[str, Tensor],
    targets_by_head: Mapping[str, Tensor],
    availability_by_head: Mapping[str, Tensor],
    *,
    loss_weights: Mapping[str, float] | None = None,
) -> MaskedClassificationLoss:
    """Apply independent masked, class-balanced CE losses to explicit axes."""

    if not logits_by_head:
        raise ValueError("at least one classification head is required")
    names = tuple(logits_by_head)
    _check_head_names(names)
    if set(names) != set(targets_by_head) or set(names) != set(availability_by_head):
        raise ValueError("logits, targets, and availability must name the same heads")
    supplied_loss_weights = dict(loss_weights or {})
    if set(supplied_loss_weights) - set(names):
        raise ValueError("loss_weights contains a head that is not being trained")

    by_head: dict[str, Tensor] = {}
    labelled_counts: dict[str, int] = {}
    counts_by_head: dict[str, tuple[int, ...]] = {}
    weights_by_head: dict[str, Tensor] = {}
    active: list[tuple[Tensor, float]] = []
    for name in sorted(names):
        logits = logits_by_head[name]
        targets = targets_by_head[name]
        availability = availability_by_head[name]
        expected_classes = len(NODE_HEAD_CLASSES[name])
        if logits.ndim != 2 or logits.shape[1] != expected_classes:
            raise ValueError(f"{name} logits must have shape [rows, {expected_classes}]")
        if targets.ndim != 1 or availability.ndim != 1 or targets.shape != availability.shape:
            raise ValueError(f"{name} targets and availability must be aligned vectors")
        if targets.shape[0] != logits.shape[0]:
            raise ValueError(f"{name} targets do not align with logits")
        mask = availability.to(device=logits.device, dtype=torch.bool)
        class_weights, class_counts = class_balanced_weights(
            targets.to(device=logits.device),
            mask,
            expected_classes,
        )
        labelled_count = int(mask.sum())
        if labelled_count:
            loss = F.cross_entropy(
                logits[mask],
                targets.to(device=logits.device, dtype=torch.long)[mask],
                weight=class_weights.to(device=logits.device, dtype=logits.dtype),
            )
        else:
            loss = logits.sum() * 0.0
        head_weight = float(supplied_loss_weights.get(name, 1.0))
        if not math.isfinite(head_weight) or head_weight < 0.0:
            raise ValueError(f"loss weight for {name} must be finite and non-negative")
        by_head[name] = loss
        labelled_counts[name] = labelled_count
        counts_by_head[name] = class_counts
        weights_by_head[name] = class_weights.detach().cpu()
        if labelled_count and head_weight > 0.0:
            active.append((loss, head_weight))
    if active:
        total = sum(loss * weight for loss, weight in active) / sum(weight for _, weight in active)
    else:
        total = sum(logits.sum() * 0.0 for logits in logits_by_head.values())
    return MaskedClassificationLoss(
        total=total,
        by_head=by_head,
        labelled_counts=labelled_counts,
        class_counts=counts_by_head,
        class_weights=weights_by_head,
    )


def _state_dict_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class NodeTrainingReceipt:
    variant: str
    seed: int
    epochs: int
    training_row_count: int
    labelled_counts: dict[str, int]
    class_counts: dict[str, tuple[int, ...]]
    final_loss: float
    state_dict_sha256: str
    input_features_detached: bool
    human_certainty_head_present: bool
    visual_encoder_trained: bool


def train_masked_multitask_node_model(
    model: MultitaskNodeMLP | HierarchicalSoftConditioningNodeModel,
    features: Tensor,
    targets_by_head: Mapping[str, Tensor],
    availability_by_head: Mapping[str, Tensor],
    *,
    training_indices: Sequence[int] | None = None,
    loss_weights: Mapping[str, float] | None = None,
    footpoint_targets: Tensor | None = None,
    footpoint_availability: Tensor | None = None,
    footpoint_loss_weight: float = 1.0,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = DEFAULT_TRAINING_SEED,
) -> NodeTrainingReceipt:
    """Train N2/N3 on an explicit training subset of detached cached features."""

    if not isinstance(model, (MultitaskNodeMLP, HierarchicalSoftConditioningNodeModel)):
        raise TypeError("model must be the bounded G7B N2 or N3 primitive")
    if epochs <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("epochs and learning_rate must be positive")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not math.isfinite(footpoint_loss_weight) or footpoint_loss_weight < 0.0:
        raise ValueError("footpoint_loss_weight must be finite and non-negative")
    _check_head_names(tuple(targets_by_head))
    if set(targets_by_head) != set(availability_by_head):
        raise ValueError("targets and availability must name the same heads")

    detached_features = frozen_feature_tensor(features)
    row_count = detached_features.shape[0]
    if any(target.ndim != 1 or target.shape[0] != row_count for target in targets_by_head.values()):
        raise ValueError("every classification target must contain one value per input row")
    if any(mask.ndim != 1 or mask.shape[0] != row_count for mask in availability_by_head.values()):
        raise ValueError("every classification mask must contain one value per input row")
    indices = tuple(range(row_count)) if training_indices is None else tuple(int(value) for value in training_indices)
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= row_count:
        raise ValueError("training_indices must be a non-empty unique in-range subset")
    index = torch.tensor(indices, device=detached_features.device, dtype=torch.long)
    training_features = detached_features.index_select(0, index)
    training_targets = {
        name: target.to(device=training_features.device).index_select(0, index)
        for name, target in targets_by_head.items()
    }
    training_availability = {
        name: mask.to(device=training_features.device).index_select(0, index)
        for name, mask in availability_by_head.items()
    }
    use_footpoint = footpoint_targets is not None or footpoint_availability is not None
    if use_footpoint:
        if footpoint_targets is None or footpoint_availability is None:
            raise ValueError("footpoint targets and availability must be supplied together")
        if footpoint_targets.shape != (row_count, 2) or footpoint_availability.shape != (row_count,):
            raise ValueError("footpoint targets/mask must have shapes [rows, 2] and [rows]")
        training_footpoints = footpoint_targets.to(device=training_features.device).index_select(0, index)
        training_footpoint_mask = footpoint_availability.to(device=training_features.device).index_select(0, index)
    else:
        training_footpoints = None
        training_footpoint_mask = None

    model_device = next(model.parameters()).device
    if training_features.device != model_device:
        training_features = training_features.to(model_device)
        training_targets = {name: value.to(model_device) for name, value in training_targets.items()}
        training_availability = {name: value.to(model_device) for name, value in training_availability.items()}
        if use_footpoint:
            assert training_footpoints is not None and training_footpoint_mask is not None
            training_footpoints = training_footpoints.to(model_device)
            training_footpoint_mask = training_footpoint_mask.to(model_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    final_classification: MaskedClassificationLoss | None = None
    final_loss: Tensor | None = None
    devices = [model_device.index] if model_device.type == "cuda" and model_device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if model_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            output = model(training_features)
            if any(any(token in name.lower() for token in FORBIDDEN_HEAD_TOKENS) for name in output):
                raise RuntimeError("model emitted a forbidden certainty/identity/temporal/count head")
            logits = {name: output[f"{name}_logits"] for name in training_targets}
            classification = masked_class_balanced_multitask_loss(
                logits,
                training_targets,
                training_availability,
                loss_weights=loss_weights,
            )
            loss = classification.total
            if use_footpoint and footpoint_loss_weight > 0.0:
                assert training_footpoints is not None and training_footpoint_mask is not None
                footpoint = masked_heteroscedastic_footpoint_loss(
                    output["footpoint_mean"],
                    output["footpoint_log_variance"],
                    training_footpoints,
                    training_footpoint_mask,
                )
                if footpoint.labelled_count:
                    loss = loss + footpoint_loss_weight * footpoint.total
            if not torch.isfinite(loss):
                raise RuntimeError("node training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            final_classification = classification
            final_loss = loss.detach()
    assert final_classification is not None and final_loss is not None
    model.eval()
    if features.grad is not None:
        raise RuntimeError("a gradient reached the supplied frozen feature tensor")
    variant = "N2" if isinstance(model, MultitaskNodeMLP) else "N3"
    return NodeTrainingReceipt(
        variant=variant,
        seed=seed,
        epochs=epochs,
        training_row_count=len(indices),
        labelled_counts=final_classification.labelled_counts,
        class_counts=final_classification.class_counts,
        final_loss=float(final_loss.cpu()),
        state_dict_sha256=_state_dict_sha256(model),
        input_features_detached=True,
        human_certainty_head_present=False,
        visual_encoder_trained=False,
    )


def symmetric_endpoint_features(left: Tensor, right: Tensor, pair_features: Tensor | None = None) -> Tensor:
    """Build commutative endpoint features for P1/P3 and audit P2 inputs."""

    if left.ndim != 2 or right.shape != left.shape:
        raise ValueError("left and right endpoint tensors must have the same two-dimensional shape")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("endpoint features must be finite")
    if pair_features is None:
        pair = left.new_empty((left.shape[0], 0))
    else:
        if pair_features.ndim != 2 or pair_features.shape[0] != left.shape[0]:
            raise ValueError("pair features must have one row per endpoint pair")
        if not torch.isfinite(pair_features).all():
            raise ValueError("pair features must be finite")
        pair = pair_features
    return torch.cat((left + right, torch.abs(left - right), left * right, pair), dim=1)


class InterpretablePairLinear(nn.Module):
    """P1 linear classifier over an explicitly named symmetric tabular matrix."""

    def __init__(self, feature_names: Sequence[str], *, seed: int = 5711) -> None:
        super().__init__()
        names = tuple(str(value).strip() for value in feature_names)
        if not names or any(not value for value in names) or len(set(names)) != len(names):
            raise ValueError("P1 feature names must be non-empty and unique")
        if any("identity" in value.lower() or "temporal" in value.lower() for value in names):
            raise ValueError("P1 cannot consume identity or temporal features")
        self.feature_names = names
        self.initialization_seed = int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            self.classifier = nn.Linear(len(names), len(PAIR_RELATION_CLASSES))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError(f"P1 features must have shape [rows, {len(self.feature_names)}]")
        if not torch.isfinite(features).all():
            raise ValueError("P1 features must be finite")
        return self.classifier(features)

    def coefficient_ledger(self) -> dict[str, dict[str, float]]:
        weights = self.classifier.weight.detach().cpu()
        return {
            relation: {name: float(weights[class_index, index]) for index, name in enumerate(self.feature_names)}
            for class_index, relation in enumerate(PAIR_RELATION_CLASSES)
        }

    def specification(self) -> dict[str, Any]:
        return {
            "schema_version": "football_intelligence.m5_5g7b.p1_interpretable_pair_linear.v1",
            "variant": "P1",
            "feature_names": list(self.feature_names),
            "pair_order_invariant_input_required": True,
            "strictly_out_of_fold_node_probabilities_required_when_present": True,
            "identity_features_present": False,
            "temporal_features_present": False,
        }


class TwoStagePairClassifier(nn.Module):
    """P3 duplicate gate followed by distinct/merged/insufficient routing."""

    def __init__(self, feature_dim: int, *, hidden_dim: int = 64, seed: int = 5713) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("P3 dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.initialization_seed = int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            self.trunk = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
            )
            self.duplicate_gate = nn.Linear(self.hidden_dim, 1)
            self.non_duplicate_router = nn.Linear(self.hidden_dim, 3)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"P3 features must have shape [rows, {self.feature_dim}]")
        if not torch.isfinite(features).all():
            raise ValueError("P3 features must be finite")
        hidden = self.trunk(features)
        duplicate_logit = self.duplicate_gate(hidden).squeeze(1)
        duplicate_log_probability = F.logsigmoid(duplicate_logit)
        non_duplicate_log_probability = F.logsigmoid(-duplicate_logit)
        routed = F.log_softmax(self.non_duplicate_router(hidden), dim=1)
        # PAIR_RELATION_CLASSES is duplicate, distinct, merged, insufficient.
        return torch.cat(
            (duplicate_log_probability[:, None], non_duplicate_log_probability[:, None] + routed),
            dim=1,
        )

    def specification(self) -> dict[str, Any]:
        return {
            "schema_version": "football_intelligence.m5_5g7b.p3_two_stage_pair_classifier.v1",
            "variant": "P3",
            "stage_1": "SAME_PERSON_DUPLICATE_VERSUS_NOT_DUPLICATE",
            "stage_2": ["DISTINCT_PEOPLE", "MERGED_CONTAINS_BOTH", "INSUFFICIENT_EVIDENCE"],
            "pair_order_invariant_input_required": True,
            "identity_features_present": False,
            "temporal_features_present": False,
        }


def pair_model_logits(
    model: InterpretablePairLinear | SymmetricPairMLP | TwoStagePairClassifier,
    pair_features: Tensor,
    *,
    left_features: Tensor | None = None,
    right_features: Tensor | None = None,
) -> Tensor:
    """Apply P1/P2/P3 through one shape-checked interface."""

    if isinstance(model, SymmetricPairMLP):
        if left_features is None or right_features is None:
            raise ValueError("P2 requires both endpoint feature tensors")
        return model(left_features, right_features, pair_features)
    if left_features is not None or right_features is not None:
        raise ValueError("P1/P3 consume a precomputed symmetric tabular matrix")
    return model(pair_features)


def train_masked_pair_model(
    model: InterpretablePairLinear | SymmetricPairMLP | TwoStagePairClassifier,
    pair_features: Tensor,
    targets: Tensor,
    availability: Tensor,
    *,
    left_features: Tensor | None = None,
    right_features: Tensor | None = None,
    training_indices: Sequence[int] | None = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 5721,
) -> dict[str, Any]:
    """Train a pair model with masked class-balanced CE on one grouped pool."""

    if pair_features.ndim != 2 or pair_features.shape[0] == 0 or not torch.isfinite(pair_features).all():
        raise ValueError("pair_features must be a finite non-empty matrix")
    row_count = pair_features.shape[0]
    if targets.shape != (row_count,) or availability.shape != (row_count,):
        raise ValueError("pair targets and availability must contain one value per pair")
    if epochs <= 0 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("pair training hyperparameters are invalid")
    indices = tuple(range(row_count)) if training_indices is None else tuple(int(value) for value in training_indices)
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= row_count:
        raise ValueError("training_indices must be a non-empty unique in-range subset")
    index = torch.tensor(indices, dtype=torch.long, device=pair_features.device)
    pair = frozen_feature_tensor(pair_features).index_select(0, index)
    left = frozen_feature_tensor(left_features).index_select(0, index) if left_features is not None else None
    right = frozen_feature_tensor(right_features).index_select(0, index) if right_features is not None else None
    target = targets.to(device=pair.device).index_select(0, index)
    mask = availability.to(device=pair.device).index_select(0, index).to(dtype=torch.bool)
    class_weights, class_counts = class_balanced_weights(target, mask, len(PAIR_RELATION_CLASSES))
    if not int(mask.sum()):
        raise ValueError("pair training requires at least one available target")
    model_device = next(model.parameters()).device
    pair, target, mask = pair.to(model_device), target.to(model_device), mask.to(model_device)
    class_weights = class_weights.to(model_device)
    left = left.to(model_device) if left is not None else None
    right = right.to(model_device) if right is not None else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    final_loss: Tensor | None = None
    devices = [model_device.index] if model_device.type == "cuda" and model_device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if model_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            logits = pair_model_logits(model, pair, left_features=left, right_features=right)
            loss = F.cross_entropy(logits[mask], target.to(dtype=torch.long)[mask], weight=class_weights)
            if not torch.isfinite(loss):
                raise RuntimeError("pair training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            final_loss = loss.detach()
    assert final_loss is not None
    model.eval()
    return {
        "schema_version": "football_intelligence.m5_5g7b.pair_training_receipt.v1",
        "development_scope": DEVELOPMENT_SCOPE,
        "variant": model.specification()["variant"],
        "seed": seed,
        "epochs": epochs,
        "training_row_count": len(indices),
        "labelled_count": int(mask.sum()),
        "class_counts": list(class_counts),
        "final_loss": float(final_loss.cpu()),
        "state_dict_sha256": _state_dict_sha256(model),
        "input_features_detached": True,
        "pair_order_invariant_input_required": True,
    }


def _available_rows(
    values: Tensor,
    targets: Tensor,
    availability: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if values.ndim != 2 or targets.ndim != 1 or values.shape[0] != targets.shape[0]:
        raise ValueError("values and targets must have shapes [rows, classes] and [rows]")
    if availability is None:
        mask = torch.ones(values.shape[0], device=values.device, dtype=torch.bool)
    else:
        if availability.shape != targets.shape:
            raise ValueError("availability must contain one value per target")
        mask = availability.to(device=values.device, dtype=torch.bool)
    selected_targets = targets.to(device=values.device, dtype=torch.long)[mask]
    selected_values = values[mask]
    if selected_targets.numel() and (int(selected_targets.min()) < 0 or int(selected_targets.max()) >= values.shape[1]):
        raise ValueError("available targets contain an out-of-range class index")
    return selected_values, selected_targets


def multiclass_brier_score(
    probabilities: Tensor,
    targets: Tensor,
    availability: Tensor | None = None,
) -> float | None:
    """Return the mean unscaled multiclass Brier score (sum over classes)."""

    selected, selected_targets = _available_rows(probabilities, targets, availability)
    if not selected.shape[0]:
        return None
    if not torch.isfinite(selected).all() or torch.any(selected < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not torch.allclose(selected.sum(dim=1), torch.ones(selected.shape[0], device=selected.device), atol=1e-5):
        raise ValueError("probability rows must sum to one")
    truth = F.one_hot(selected_targets, num_classes=selected.shape[1]).to(dtype=selected.dtype)
    return float(torch.sum((selected - truth) ** 2, dim=1).mean().cpu())


def macro_f1_score(
    predictions: Tensor,
    targets: Tensor,
    class_count: int,
    availability: Tensor | None = None,
    *,
    include_unsupported: bool = False,
) -> float | None:
    """Return macro F1 over supported classes unless explicitly requested otherwise."""

    if predictions.ndim != 1 or targets.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError("predictions and targets must be aligned vectors")
    if class_count < 2:
        raise ValueError("class_count must be at least two")
    mask = (
        torch.ones(predictions.shape[0], device=predictions.device, dtype=torch.bool)
        if availability is None
        else availability.to(device=predictions.device, dtype=torch.bool)
    )
    if mask.shape != predictions.shape:
        raise ValueError("availability must contain one value per prediction")
    predicted = predictions.to(dtype=torch.long)[mask]
    actual = targets.to(device=predictions.device, dtype=torch.long)[mask]
    if not actual.numel():
        return None
    if min(int(actual.min()), int(predicted.min())) < 0 or max(int(actual.max()), int(predicted.max())) >= class_count:
        raise ValueError("prediction or target contains an out-of-range class index")
    scores: list[float] = []
    for class_index in range(class_count):
        support = int((actual == class_index).sum())
        if not include_unsupported and support == 0:
            continue
        true_positive = int(((predicted == class_index) & (actual == class_index)).sum())
        false_positive = int(((predicted == class_index) & (actual != class_index)).sum())
        false_negative = int(((predicted != class_index) & (actual == class_index)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores) if scores else None


def expected_calibration_error(
    probabilities: Tensor,
    targets: Tensor,
    availability: Tensor | None = None,
    *,
    bin_count: int = 10,
) -> float | None:
    """Top-class ECE with fixed confidence bins."""

    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    selected, selected_targets = _available_rows(probabilities, targets, availability)
    if not selected.shape[0]:
        return None
    confidence, prediction = selected.max(dim=1)
    if not torch.isfinite(confidence).all() or torch.any(confidence < 0.0) or torch.any(confidence > 1.0):
        raise ValueError("probabilities must produce confidence in [0, 1]")
    correctness = prediction.eq(selected_targets).to(dtype=torch.float32)
    bin_indices = torch.clamp((confidence * bin_count).to(dtype=torch.long), max=bin_count - 1)
    total = confidence.shape[0]
    error = 0.0
    for bin_index in range(bin_count):
        mask = bin_indices == bin_index
        if mask.any():
            gap = abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
            error += int(mask.sum()) / total * gap
    return error


def selective_risk_curve(
    probabilities: Tensor,
    targets: Tensor,
    availability: Tensor | None = None,
    *,
    coverage_levels: Sequence[float] = (0.25, 0.5, 0.75, 0.9, 1.0),
) -> list[dict[str, Any]]:
    """Return deterministic risk/coverage points ranked by top-class confidence."""

    selected, selected_targets = _available_rows(probabilities, targets, availability)
    row_count = selected.shape[0]
    if not row_count:
        return []
    confidence, prediction = selected.max(dim=1)
    order = sorted(range(row_count), key=lambda index: (-float(confidence[index]), index))
    rows: list[dict[str, Any]] = []
    for requested in coverage_levels:
        coverage = float(requested)
        if not math.isfinite(coverage) or coverage <= 0.0 or coverage > 1.0:
            raise ValueError("coverage levels must be finite values in (0, 1]")
        retained = max(1, math.ceil(row_count * coverage))
        retained_indices = torch.tensor(order[:retained], device=selected.device, dtype=torch.long)
        errors = prediction.index_select(0, retained_indices).ne(selected_targets.index_select(0, retained_indices))
        rows.append(
            {
                "requested_coverage": coverage,
                "actual_coverage": retained / row_count,
                "retained": retained,
                "denominator": row_count,
                "minimum_retained_confidence": float(confidence.index_select(0, retained_indices).min().cpu()),
                "risk": float(errors.to(dtype=torch.float32).mean().cpu()),
            }
        )
    return rows


@dataclass(frozen=True)
class TemperatureScalingResult:
    temperature: float
    labelled_count: int
    nll_before: float | None
    nll_after: float | None
    fitted_on_grouped_inner_holdout: bool


def fit_temperature_scaling(
    logits: Tensor,
    targets: Tensor,
    availability: Tensor | None = None,
    *,
    minimum_temperature: float = 0.05,
    maximum_temperature: float = 20.0,
    iterations: int = 64,
) -> TemperatureScalingResult:
    """Fit one deterministic scalar temperature on supplied calibration rows."""

    selected, selected_targets = _available_rows(logits, targets, availability)
    if not selected.shape[0]:
        return TemperatureScalingResult(1.0, 0, None, None, True)
    if (
        minimum_temperature <= 0.0
        or maximum_temperature <= minimum_temperature
        or iterations <= 0
        or not math.isfinite(minimum_temperature)
        or not math.isfinite(maximum_temperature)
    ):
        raise ValueError("temperature bounds and iterations are invalid")
    selected = selected.detach().to(dtype=torch.float64)
    selected_targets = selected_targets.detach()

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        return float(F.cross_entropy(selected / temperature, selected_targets).cpu())

    low, high = math.log(minimum_temperature), math.log(maximum_temperature)
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    left = high - golden * (high - low)
    right = low + golden * (high - low)
    left_value, right_value = objective(left), objective(right)
    for _ in range(iterations):
        if left_value <= right_value:
            high, right, right_value = right, left, left_value
            left = high - golden * (high - low)
            left_value = objective(left)
        else:
            low, left, left_value = left, right, right_value
            right = low + golden * (high - low)
            right_value = objective(right)
    candidates = [(0.0, objective(0.0)), (left, left_value), (right, right_value)]
    best_log_temperature, best_nll = min(candidates, key=lambda row: (row[1], abs(row[0])))
    return TemperatureScalingResult(
        temperature=math.exp(best_log_temperature),
        labelled_count=selected.shape[0],
        nll_before=objective(0.0),
        nll_after=best_nll,
        fitted_on_grouped_inner_holdout=True,
    )


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError("logits must be a finite [rows, classes] tensor")
    return torch.softmax(logits / temperature, dim=1)


@dataclass(frozen=True)
class AbstentionThresholdResult:
    threshold: float
    calibration_count: int
    retained_count: int
    coverage: float
    empirical_risk: float | None


def fit_abstention_threshold(
    probabilities: Tensor,
    targets: Tensor,
    availability: Tensor | None = None,
    *,
    maximum_risk: float,
    minimum_coverage: float = 0.0,
) -> AbstentionThresholdResult:
    """Choose maximum calibration coverage satisfying a frozen risk limit."""

    if not math.isfinite(maximum_risk) or maximum_risk < 0.0 or maximum_risk > 1.0:
        raise ValueError("maximum_risk must be in [0, 1]")
    if not math.isfinite(minimum_coverage) or minimum_coverage < 0.0 or minimum_coverage > 1.0:
        raise ValueError("minimum_coverage must be in [0, 1]")
    selected, selected_targets = _available_rows(probabilities, targets, availability)
    row_count = selected.shape[0]
    if not row_count:
        return AbstentionThresholdResult(math.nextafter(1.0, math.inf), 0, 0, 0.0, None)
    confidence, prediction = selected.max(dim=1)
    candidates = sorted({float(value) for value in confidence.tolist()}, reverse=True)
    accepted: list[tuple[int, float, float]] = []
    for threshold in candidates:
        retain = confidence >= threshold
        retained_count = int(retain.sum())
        coverage = retained_count / row_count
        risk = float(prediction[retain].ne(selected_targets[retain]).to(dtype=torch.float32).mean())
        if coverage >= minimum_coverage and risk <= maximum_risk:
            accepted.append((retained_count, threshold, risk))
    if not accepted:
        return AbstentionThresholdResult(math.nextafter(1.0, math.inf), row_count, 0, 0.0, None)
    retained_count, threshold, risk = max(accepted, key=lambda row: (row[0], -row[1]))
    return AbstentionThresholdResult(
        threshold=threshold,
        calibration_count=row_count,
        retained_count=retained_count,
        coverage=retained_count / row_count,
        empirical_risk=risk,
    )


@dataclass(frozen=True)
class HeadCalibration:
    """Independent temperature and abstention evidence for one semantic head."""

    head_name: str
    temperature: float
    abstention_threshold: float
    calibration_count: int
    nll_before: float | None
    nll_after: float | None
    retained_count: int
    calibration_coverage: float
    calibration_risk: float | None
    fitted_on_grouped_inner_holdout: bool


def fit_head_calibration(
    head_name: str,
    logits: Tensor,
    targets: Tensor,
    availability: Tensor,
    *,
    calibration_indices: Sequence[int],
    maximum_risk: float,
    minimum_coverage: float = 0.0,
) -> HeadCalibration:
    """Fit one head only on explicit inner-held-out calibration indices."""

    _check_head_names((head_name,))
    if logits.ndim != 2 or targets.shape != (logits.shape[0],) or availability.shape != targets.shape:
        raise ValueError("head logits, targets, and availability are not row-aligned")
    indices = tuple(int(value) for value in calibration_indices)
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= logits.shape[0]:
        raise ValueError("calibration_indices must be a non-empty unique in-range inner-held-out subset")
    index = torch.tensor(indices, device=logits.device, dtype=torch.long)
    calibration_logits = logits.index_select(0, index)
    calibration_targets = targets.to(device=logits.device).index_select(0, index)
    calibration_availability = availability.to(device=logits.device).index_select(0, index)
    temperature = fit_temperature_scaling(
        calibration_logits,
        calibration_targets,
        calibration_availability,
    )
    probabilities = apply_temperature(calibration_logits, temperature.temperature)
    threshold = fit_abstention_threshold(
        probabilities,
        calibration_targets,
        calibration_availability,
        maximum_risk=maximum_risk,
        minimum_coverage=minimum_coverage,
    )
    return HeadCalibration(
        head_name=head_name,
        temperature=temperature.temperature,
        abstention_threshold=threshold.threshold,
        calibration_count=temperature.labelled_count,
        nll_before=temperature.nll_before,
        nll_after=temperature.nll_after,
        retained_count=threshold.retained_count,
        calibration_coverage=threshold.coverage,
        calibration_risk=threshold.empirical_risk,
        fitted_on_grouped_inner_holdout=True,
    )


def fit_multitask_head_calibrations(
    logits_by_head: Mapping[str, Tensor],
    targets_by_head: Mapping[str, Tensor],
    availability_by_head: Mapping[str, Tensor],
    *,
    calibration_indices: Sequence[int],
    maximum_risk_by_head: Mapping[str, float],
    minimum_coverage_by_head: Mapping[str, float] | None = None,
) -> dict[str, HeadCalibration]:
    """Fit separate calibration and abstention parameters for every supplied head."""

    names = tuple(logits_by_head)
    _check_head_names(names)
    if set(names) != set(targets_by_head) or set(names) != set(availability_by_head):
        raise ValueError("multitask calibration inputs must name the same heads")
    if set(maximum_risk_by_head) != set(names):
        raise ValueError("maximum_risk_by_head must explicitly cover every calibrated head")
    minimum = dict(minimum_coverage_by_head or {})
    if set(minimum) - set(names):
        raise ValueError("minimum_coverage_by_head references an unknown head")
    return {
        name: fit_head_calibration(
            name,
            logits_by_head[name],
            targets_by_head[name],
            availability_by_head[name],
            calibration_indices=calibration_indices,
            maximum_risk=float(maximum_risk_by_head[name]),
            minimum_coverage=float(minimum.get(name, 0.0)),
        )
        for name in sorted(names)
    }


def abstaining_predictions(probabilities: Tensor, threshold: float, *, unresolved_index: int = -1) -> Tensor:
    """Return independent head predictions, using unresolved_index below threshold."""

    if probabilities.ndim != 2 or not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be a finite [rows, classes] tensor")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    confidence, prediction = probabilities.max(dim=1)
    unresolved = torch.full_like(prediction, int(unresolved_index))
    return torch.where(confidence >= threshold, prediction, unresolved)


def classification_metrics(
    probabilities: Tensor,
    targets: Tensor,
    availability: Tensor | None = None,
    *,
    bin_count: int = 10,
) -> dict[str, Any]:
    """Return the common G7B accuracy, macro-F1, Brier, ECE and risk curve."""

    selected, selected_targets = _available_rows(probabilities, targets, availability)
    denominator = selected.shape[0]
    if not denominator:
        return {
            "denominator": 0,
            "accuracy": None,
            "macro_f1": None,
            "brier_score": None,
            "expected_calibration_error": None,
            "selective_risk": [],
        }
    prediction = selected.argmax(dim=1)
    return {
        "denominator": denominator,
        "accuracy": float(prediction.eq(selected_targets).to(dtype=torch.float32).mean().cpu()),
        "macro_f1": macro_f1_score(prediction, selected_targets, selected.shape[1]),
        "brier_score": multiclass_brier_score(selected, selected_targets),
        "expected_calibration_error": expected_calibration_error(
            selected,
            selected_targets,
            bin_count=bin_count,
        ),
        "selective_risk": selective_risk_curve(selected, selected_targets),
    }


def class_distribution(targets: Tensor, availability: Tensor, class_names: Sequence[str]) -> dict[str, int]:
    """Expose exact masked denominators without fabricating unsupported classes."""

    _, counts = class_balanced_weights(targets, availability, len(class_names))
    return dict(zip(class_names, counts, strict=True))


def assert_no_human_certainty_head(model: nn.Module) -> None:
    """Fail closed if a constant-label human-certainty head was introduced."""

    offending = sorted(name for name, _ in model.named_modules() if "certainty" in name.lower())
    if offending:
        raise RuntimeError(f"human-certainty head is forbidden: {offending}")


__all__ = [
    "AbstentionThresholdResult",
    "CLASSIFICATION_HEADS",
    "DEFAULT_INNER_FOLD_COUNT",
    "DEFAULT_OUTER_FOLD_COUNT",
    "DEFAULT_SPLIT_SEED",
    "DEVELOPMENT_SCOPE",
    "HeadCalibration",
    "HierarchicalSoftConditioningNodeModel",
    "InterpretablePairLinear",
    "MaskedClassificationLoss",
    "MultitaskNodeMLP",
    "NestedGroupedSplit",
    "NodeTrainingReceipt",
    "SymmetricPairMLP",
    "TemperatureScalingResult",
    "TwoStagePairClassifier",
    "abstaining_predictions",
    "apply_temperature",
    "assert_no_human_certainty_head",
    "class_balanced_weights",
    "class_distribution",
    "classification_metrics",
    "expected_calibration_error",
    "fit_abstention_threshold",
    "fit_head_calibration",
    "fit_multitask_head_calibrations",
    "fit_temperature_scaling",
    "frozen_feature_tensor",
    "grouped_inner_fold_assignments",
    "macro_f1_score",
    "masked_class_balanced_multitask_loss",
    "multiclass_brier_score",
    "nested_grouped_oof_splits",
    "pair_model_logits",
    "selective_risk_curve",
    "symmetric_endpoint_features",
    "train_masked_multitask_node_model",
    "train_masked_pair_model",
    "validate_grouped_outer_folds",
]
