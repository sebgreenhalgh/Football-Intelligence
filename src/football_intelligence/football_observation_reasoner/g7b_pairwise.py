"""Leakage-safe pair-relation development models for M5 5G7B.

This module is deliberately local to pair evidence.  It neither creates person
identities nor consumes temporal state.  Every prediction is for one frozen
candidate pair in one source group.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from football_intelligence.football_observation_reasoner.dataset import sample_group_balanced_edges
from football_intelligence.football_observation_reasoner.hierarchical_selection import SymmetricPairMLP
from football_intelligence.football_observation_reasoner.models import PAIR_RELATION_CLASSES

PAIR_CLASSES = tuple(PAIR_RELATION_CLASSES)
POSITIVE_PAIR_CLASSES = frozenset({"SAME_PERSON_DUPLICATE", "MERGED_CONTAINS_BOTH"})
_IGNORED_PAIR_FEATURES = frozenset({"feature_hash", "human_identity_feature_used"})
_PROHIBITED_FEATURE_FRAGMENTS = (
    "future_frame",
    "identity_id",
    "next_frame",
    "previous_frame",
    "temporal",
    "track_id",
    "trajectory",
)


@dataclass(frozen=True)
class PairwiseOOFConfig:
    """Frozen, bounded training settings for P1--P3."""

    negative_ratio: float = 3.0
    minimum_negatives_per_group: int = 1
    logistic_c: float = 1.0
    p2_hidden_dim: int = 64
    p2_epochs: int = 80
    p2_learning_rate: float = 0.01
    p2_weight_decay: float = 0.0001
    seed: int = 5713

    def validate(self) -> None:
        if not math.isfinite(self.negative_ratio) or self.negative_ratio < 0.0:
            raise ValueError("negative_ratio must be finite and non-negative")
        if self.minimum_negatives_per_group < 0:
            raise ValueError("minimum_negatives_per_group must be non-negative")
        if not math.isfinite(self.logistic_c) or self.logistic_c <= 0.0:
            raise ValueError("logistic_c must be finite and positive")
        if self.p2_hidden_dim <= 0 or self.p2_epochs <= 0:
            raise ValueError("P2 hidden dimension and epoch count must be positive")
        if not math.isfinite(self.p2_learning_rate) or self.p2_learning_rate <= 0.0:
            raise ValueError("P2 learning rate must be finite and positive")
        if not math.isfinite(self.p2_weight_decay) or self.p2_weight_decay < 0.0:
            raise ValueError("P2 weight decay must be finite and non-negative")

    def specification(self) -> dict[str, Any]:
        payload = {
            "schema_version": "football_intelligence.m5_5g7b.pairwise_oof_config.v1",
            "negative_ratio": self.negative_ratio,
            "minimum_negatives_per_group": self.minimum_negatives_per_group,
            "logistic_c": self.logistic_c,
            "p2_hidden_dim": self.p2_hidden_dim,
            "p2_epochs": self.p2_epochs,
            "p2_learning_rate": self.p2_learning_rate,
            "p2_weight_decay": self.p2_weight_decay,
            "seed": self.seed,
            "visual_encoder_trainable": False,
            "identity_tracking_present": False,
            "temporal_prediction_present": False,
        }
        payload["specification_hash"] = _stable_hash(payload)
        return payload


def _stable_hash(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _finite_scalar(name: str, value: Any) -> tuple[float, float]:
    if value is None:
        return 0.0, 1.0
    if isinstance(value, (bool, np.bool_)):
        return float(value), 0.0
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"pair feature {name!r} must be numeric, boolean, or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"pair feature {name!r} must be finite")
    return result, 0.0


def _directional_pair_key(name: str) -> tuple[str, str] | None:
    if name.startswith("left_"):
        return name[5:], f"right_{name[5:]}"
    if name.startswith("right_"):
        return name[6:], f"left_{name[6:]}"
    if name.endswith("_left"):
        return name[:-5], f"{name[:-5]}_right"
    if name.endswith("_right"):
        return name[:-6], f"{name[:-6]}_left"
    return None


def _assert_feature_name_is_safe(name: str) -> None:
    lowered = name.lower()
    if any(fragment in lowered for fragment in _PROHIBITED_FEATURE_FRAGMENTS):
        raise ValueError(f"pair feature {name!r} is prohibited identity/temporal evidence")
    if any(fragment in lowered for fragment in ("target", "label", "ground_truth", "human_decision")):
        raise ValueError(f"pair feature {name!r} leaks a training target")


def canonicalize_pair_feature_mapping(pair_features: Mapping[str, Any]) -> dict[str, float]:
    """Convert G7A pair evidence to an endpoint-order-invariant numeric mapping.

    Directional endpoint pairs such as ``left_containment_fraction`` and
    ``right_containment_fraction`` are represented by commutative statistics.
    Every retained scalar has an explicit missingness indicator.
    """

    if not isinstance(pair_features, Mapping) or not pair_features:
        raise ValueError("pair_features must be a non-empty mapping")
    if bool(pair_features.get("human_identity_feature_used", False)):
        raise ValueError("human identity evidence is forbidden in G7B pair features")
    output: dict[str, float] = {}
    consumed: set[str] = set()
    for name in sorted(str(key) for key in pair_features):
        if name in consumed or name in _IGNORED_PAIR_FEATURES:
            continue
        _assert_feature_name_is_safe(name)
        directional = _directional_pair_key(name)
        if directional is not None:
            base, counterpart = directional
            if counterpart not in pair_features:
                raise ValueError(f"directional pair feature {name!r} has no {counterpart!r} counterpart")
            left_name, right_name = (
                (name, counterpart)
                if name.startswith("left_") or name.endswith("_left")
                else (
                    counterpart,
                    name,
                )
            )
            left, left_missing = _finite_scalar(left_name, pair_features[left_name])
            right, right_missing = _finite_scalar(right_name, pair_features[right_name])
            prefix = f"endpoint_{base}"
            output[f"{prefix}__min"] = min(left, right)
            output[f"{prefix}__max"] = max(left, right)
            output[f"{prefix}__mean"] = 0.5 * (left + right)
            output[f"{prefix}__abs_difference"] = abs(left - right)
            output[f"{prefix}__missing_count"] = left_missing + right_missing
            consumed.update((name, counterpart))
            continue
        value, missing = _finite_scalar(name, pair_features[name])
        output[name] = value
        output[f"{name}__missing"] = missing
        consumed.add(name)
    if not output:
        raise ValueError("pair feature mapping has no permitted numeric evidence")
    return dict(sorted(output.items()))


def canonicalize_g7a_edge_features(edge_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Canonicalize a frozen edge population and return a stable feature ledger."""

    feature_rows: dict[str, dict[str, float]] = {}
    endpoint_rows: dict[str, tuple[str, str]] = {}
    for index, edge in enumerate(edge_rows):
        edge_id = _edge_id(edge, index)
        if edge_id in feature_rows:
            raise ValueError(f"duplicate pair edge_uuid: {edge_id}")
        left, right = _endpoints(edge)
        feature_rows[edge_id] = canonicalize_pair_feature_mapping(edge.get("pair_features") or {})
        endpoint_rows[edge_id] = tuple(sorted((left, right)))
    feature_names = tuple(sorted({name for row in feature_rows.values() for name in row}))
    if not feature_names:
        raise ValueError("no canonical pair features were created")
    vectors = {
        edge_id: tuple(float(row.get(name, 0.0)) for name in feature_names)
        for edge_id, row in sorted(feature_rows.items())
    }
    audit = {
        "schema_version": "football_intelligence.m5_5g7b.canonical_pair_features.v1",
        "edge_count": len(feature_rows),
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "endpoint_order_invariant": True,
        "directional_endpoint_values_retained": False,
        "identity_features_present": False,
        "temporal_features_present": False,
        "endpoint_hash": _stable_hash(endpoint_rows),
        "feature_matrix_hash": _stable_hash(vectors),
    }
    audit["audit_hash"] = _stable_hash(audit)
    return {
        "feature_names": feature_names,
        "features_by_edge_uuid": feature_rows,
        "vectors_by_edge_uuid": vectors,
        "audit": audit,
    }


def _probability_groups(probabilities: Mapping[str, Any]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for name, raw_value in sorted(probabilities.items()):
        _assert_feature_name_is_safe(str(name))
        value, missing = _finite_scalar(str(name), raw_value)
        if missing:
            raise ValueError(f"OOF node probability {name!r} cannot be null")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"OOF node probability {name!r} must be in [0, 1]")
        head = str(name).split("::", 1)[0] if "::" in str(name) else "node"
        grouped[head].append(value)
    if not grouped:
        raise ValueError("OOF node probability mapping cannot be empty")
    for head, values in grouped.items():
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError(f"OOF node probabilities for head {head!r} must sum to one")
    return grouped


def append_supplied_oof_node_probability_summaries(
    canonical_payload: Mapping[str, Any],
    edge_rows: Sequence[Mapping[str, Any]],
    supplied_by_candidate_uuid: Mapping[str, Mapping[str, Any]],
    fold_by_source_group: Mapping[str, int],
    *,
    additional_excluded_source_group_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Append symmetric node-probability summaries after strict OOF validation.

    The function never fits or reconstructs node predictions.  Callers must
    supply probabilities and their exact fit-group provenance.  For every edge,
    its source group must be absent from both endpoint models' fit groups.
    """

    base_rows = canonical_payload.get("features_by_edge_uuid") or {}
    additionally_excluded = frozenset(str(value) for value in additional_excluded_source_group_ids)
    if "" in additionally_excluded:
        raise ValueError("additional excluded source-group IDs cannot be empty")
    combined_rows: dict[str, dict[str, float]] = {}
    ledger: list[dict[str, Any]] = []
    probability_names: tuple[str, ...] | None = None
    for index, edge in enumerate(edge_rows):
        edge_id = _edge_id(edge, index)
        if edge_id not in base_rows:
            raise ValueError(f"canonical features are missing edge {edge_id}")
        left, right = _endpoints(edge)
        source_group = str(edge.get("source_group_id") or "").strip()
        if not source_group or source_group not in fold_by_source_group:
            raise ValueError(f"edge {edge_id} lacks a frozen source-group fold")
        held_out_fold = int(fold_by_source_group[source_group])
        endpoint_records: list[Mapping[str, Any]] = []
        for candidate_uuid in (left, right):
            record = supplied_by_candidate_uuid.get(candidate_uuid)
            if record is None:
                raise ValueError(f"caller did not supply OOF node probabilities for {candidate_uuid}")
            if str(record.get("source_group_id") or "") != source_group:
                raise ValueError(f"OOF node source group mismatch for {candidate_uuid}")
            if int(record.get("held_out_fold", -1)) != held_out_fold:
                raise ValueError(f"OOF node held-out fold mismatch for {candidate_uuid}")
            provenance_kind = str(record.get("provenance_kind") or "")
            if provenance_kind not in {"INNER_GROUP_OOF", "OUTER_GROUP_OOF"}:
                raise ValueError(f"OOF node provenance kind is invalid for {candidate_uuid}")
            fit_groups = tuple(sorted(str(value) for value in (record.get("model_fit_source_group_ids") or ())))
            if not fit_groups:
                raise ValueError(f"OOF node fit-group provenance is missing for {candidate_uuid}")
            forbidden_fit_groups = {source_group, *additionally_excluded}
            leaked_groups = sorted(forbidden_fit_groups & set(fit_groups))
            if leaked_groups:
                raise ValueError(f"OOF node probability for {candidate_uuid} leaks source group(s) {leaked_groups}")
            probabilities = record.get("probabilities")
            if not isinstance(probabilities, Mapping):
                raise ValueError(f"OOF node probabilities are missing for {candidate_uuid}")
            _probability_groups(probabilities)
            current_names = tuple(sorted(str(name) for name in probabilities))
            if probability_names is None:
                probability_names = current_names
            if current_names != probability_names:
                raise ValueError("OOF node probability schemas differ between candidates")
            endpoint_records.append(record)
        assert probability_names is not None
        augmented = {str(name): float(value) for name, value in base_rows[edge_id].items()}
        left_probabilities = endpoint_records[0]["probabilities"]
        right_probabilities = endpoint_records[1]["probabilities"]
        for name in probability_names:
            left_value = float(left_probabilities[name])
            right_value = float(right_probabilities[name])
            prefix = f"oof_node_probability::{name}"
            augmented[f"{prefix}::min"] = min(left_value, right_value)
            augmented[f"{prefix}::max"] = max(left_value, right_value)
            augmented[f"{prefix}::mean"] = 0.5 * (left_value + right_value)
            augmented[f"{prefix}::abs_difference"] = abs(left_value - right_value)
            augmented[f"{prefix}::product"] = left_value * right_value
        combined_rows[edge_id] = dict(sorted(augmented.items()))
        ledger.append(
            {
                "edge_uuid": edge_id,
                "source_group_id": source_group,
                "held_out_fold": held_out_fold,
                "endpoint_candidate_uuids": sorted((left, right)),
                "endpoint_provenance_kinds": sorted(str(row["provenance_kind"]) for row in endpoint_records),
                "source_group_excluded_from_both_model_fits": True,
                "additional_excluded_source_group_ids": sorted(additionally_excluded),
                "all_additional_groups_excluded_from_both_model_fits": True,
                "probability_schema_hash": _stable_hash(probability_names),
            }
        )
    feature_names = tuple(sorted({name for row in combined_rows.values() for name in row}))
    vectors = {
        edge_id: tuple(float(row.get(name, 0.0)) for name in feature_names)
        for edge_id, row in sorted(combined_rows.items())
    }
    ledger.sort(key=lambda row: row["edge_uuid"])
    audit = {
        "schema_version": "football_intelligence.m5_5g7b.supplied_oof_node_probability_join.v1",
        "edge_count": len(ledger),
        "probability_names": list(probability_names or ()),
        "caller_supplied_only": True,
        "node_models_fitted_by_pair_module": False,
        "all_edge_groups_excluded_from_node_model_fits": all(
            row["source_group_excluded_from_both_model_fits"] for row in ledger
        ),
        "additional_excluded_source_group_ids": sorted(additionally_excluded),
        "all_additional_groups_excluded_from_node_model_fits": all(
            row["all_additional_groups_excluded_from_both_model_fits"] for row in ledger
        ),
        "endpoint_order_invariant_summaries": True,
        "ledger_hash": _stable_hash(ledger),
        "feature_matrix_hash": _stable_hash(vectors),
    }
    audit["audit_hash"] = _stable_hash(audit)
    return {
        "feature_names": feature_names,
        "features_by_edge_uuid": combined_rows,
        "vectors_by_edge_uuid": vectors,
        "oof_ledger": ledger,
        "audit": audit,
    }


def _edge_id(edge: Mapping[str, Any], index: int) -> str:
    identifier = str(edge.get("edge_uuid") or f"edge_{index:08d}").strip()
    if not identifier:
        raise ValueError("pair edge identifier cannot be empty")
    return identifier


def _endpoints(edge: Mapping[str, Any]) -> tuple[str, str]:
    left = str(edge.get("left_candidate_uuid") or "").strip()
    right = str(edge.get("right_candidate_uuid") or "").strip()
    if not left or not right or left == right:
        raise ValueError("pair edges require two distinct candidate UUIDs")
    return left, right


def _target(edge: Mapping[str, Any]) -> str:
    target = str(edge.get("target_relation") or "").strip()
    if target not in PAIR_CLASSES:
        raise ValueError(f"pair edge has invalid target relation: {target!r}")
    return target


def _dense_matrix(
    edge_rows: Sequence[Mapping[str, Any]],
    vectors_by_edge_uuid: Mapping[str, Sequence[float]],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    width: int | None = None
    for index, edge in enumerate(edge_rows):
        edge_id = _edge_id(edge, index)
        if edge_id not in vectors_by_edge_uuid:
            raise ValueError(f"pair feature vector is missing for edge {edge_id}")
        vector = np.asarray(vectors_by_edge_uuid[edge_id], dtype=np.float64).reshape(-1)
        if not vector.size or not np.isfinite(vector).all():
            raise ValueError(f"pair feature vector for {edge_id} must be non-empty and finite")
        if width is None:
            width = int(vector.size)
        if vector.size != width:
            raise ValueError("pair feature vectors must have one frozen width")
        rows.append(vector)
    return np.stack(rows) if rows else np.empty((0, int(width or 0)), dtype=np.float64)


def _class_balanced_weights(targets: Sequence[int]) -> torch.Tensor:
    counts = Counter(int(value) for value in targets)
    total = len(targets)
    represented = len(counts)
    return torch.tensor(
        [total / (represented * counts[index]) if counts[index] else 0.0 for index in range(len(PAIR_CLASSES))],
        dtype=torch.float32,
    )


def _fit_logistic_probabilities(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    classes: Sequence[int],
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    output = np.zeros((test_features.shape[0], len(classes)), dtype=np.float64)
    observed = np.unique(train_targets)
    if observed.size == 1:
        output[:, int(observed[0])] = 1.0
        return output, {
            "model_kind": "DETERMINISTIC_SINGLE_CLASS",
            "observed_classes": [int(observed[0])],
            "coefficient_rows": [],
        }
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(train_features, train_targets)
    predicted = model.predict_proba(test_features)
    for source_index, class_index in enumerate(model.classes_):
        output[:, int(class_index)] = predicted[:, source_index]
    if model.coef_.shape[0] == 1 and len(model.classes_) == 2:
        binary_coefficient = model.coef_[0]
        binary_intercept = float(model.intercept_[0])
        coefficients = [
            {
                "class_index": int(model.classes_[0]),
                "intercept": -binary_intercept,
                "coefficients": [float(-value) for value in binary_coefficient],
                "binary_reference_encoding": "NEGATED_POSITIVE_CLASS_LOG_ODDS",
            },
            {
                "class_index": int(model.classes_[1]),
                "intercept": binary_intercept,
                "coefficients": [float(value) for value in binary_coefficient],
                "binary_reference_encoding": "POSITIVE_CLASS_LOG_ODDS",
            },
        ]
    else:
        coefficients = [
            {
                "class_index": int(class_index),
                "intercept": float(model.intercept_[row_index]),
                "coefficients": [float(value) for value in model.coef_[row_index]],
            }
            for row_index, class_index in enumerate(model.classes_)
        ]
    return output, {
        "model_kind": "CLASS_BALANCED_MULTINOMIAL_LOGISTIC",
        "observed_classes": [int(value) for value in model.classes_],
        "coefficient_rows": coefficients,
    }


def _fit_p1(
    train_pair: np.ndarray,
    train_targets: np.ndarray,
    test_pair: np.ndarray,
    *,
    config: PairwiseOOFConfig,
    fold: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = StandardScaler().fit(train_pair)
    probabilities, model_ledger = _fit_logistic_probabilities(
        scaler.transform(train_pair),
        train_targets,
        scaler.transform(test_pair),
        classes=range(len(PAIR_CLASSES)),
        c_value=config.logistic_c,
        seed=config.seed + fold,
    )
    return probabilities, {
        **model_ledger,
        "variant": "P1",
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
    }


def _endpoint_feature_arrays(
    edges: Sequence[Mapping[str, Any]],
    node_features_by_candidate_uuid: Mapping[str, Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    left_rows: list[np.ndarray] = []
    right_rows: list[np.ndarray] = []
    width: int | None = None
    for edge in edges:
        left, right = _endpoints(edge)
        if left not in node_features_by_candidate_uuid or right not in node_features_by_candidate_uuid:
            raise ValueError(f"P2 caller-supplied node features are missing for edge {_edge_id(edge, 0)}")
        left_vector = np.asarray(node_features_by_candidate_uuid[left], dtype=np.float64).reshape(-1)
        right_vector = np.asarray(node_features_by_candidate_uuid[right], dtype=np.float64).reshape(-1)
        if not left_vector.size or left_vector.shape != right_vector.shape:
            raise ValueError("P2 endpoint node vectors must have equal non-empty shapes")
        if not np.isfinite(left_vector).all() or not np.isfinite(right_vector).all():
            raise ValueError("P2 endpoint node vectors must be finite")
        if width is None:
            width = int(left_vector.size)
        if left_vector.size != width:
            raise ValueError("P2 endpoint node vectors must have one frozen width")
        left_rows.append(left_vector)
        right_rows.append(right_vector)
    empty = np.empty((0, int(width or 0)), dtype=np.float64)
    return (np.stack(left_rows), np.stack(right_rows)) if left_rows else (empty, empty.copy())


def _fit_p2(
    train_edges: Sequence[Mapping[str, Any]],
    test_edges: Sequence[Mapping[str, Any]],
    train_pair: np.ndarray,
    train_targets: np.ndarray,
    test_pair: np.ndarray,
    node_features_by_candidate_uuid: Mapping[str, Sequence[float]],
    *,
    config: PairwiseOOFConfig,
    fold: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_left, train_right = _endpoint_feature_arrays(train_edges, node_features_by_candidate_uuid)
    test_left, test_right = _endpoint_feature_arrays(test_edges, node_features_by_candidate_uuid)
    node_scaler = StandardScaler().fit(np.concatenate((train_left, train_right), axis=0))
    pair_scaler = StandardScaler().fit(train_pair)
    train_left_tensor = torch.as_tensor(node_scaler.transform(train_left), dtype=torch.float32)
    train_right_tensor = torch.as_tensor(node_scaler.transform(train_right), dtype=torch.float32)
    train_pair_tensor = torch.as_tensor(pair_scaler.transform(train_pair), dtype=torch.float32)
    target_tensor = torch.as_tensor(train_targets, dtype=torch.long)
    seed = config.seed + 1000 + fold
    model = SymmetricPairMLP(
        train_left.shape[1],
        train_pair.shape[1],
        hidden_dim=config.p2_hidden_dim,
        seed=seed,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.p2_learning_rate,
        weight_decay=config.p2_weight_decay,
    )
    class_weights = _class_balanced_weights(train_targets.tolist())
    model.train()
    loss_value = 0.0
    for _ in range(config.p2_epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_left_tensor, train_right_tensor, train_pair_tensor)
        loss = F.cross_entropy(logits, target_tensor, weight=class_weights)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().item())
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.as_tensor(node_scaler.transform(test_left), dtype=torch.float32),
            torch.as_tensor(node_scaler.transform(test_right), dtype=torch.float32),
            torch.as_tensor(pair_scaler.transform(test_pair), dtype=torch.float32),
        )
        swapped_logits = model(
            torch.as_tensor(node_scaler.transform(test_right), dtype=torch.float32),
            torch.as_tensor(node_scaler.transform(test_left), dtype=torch.float32),
            torch.as_tensor(pair_scaler.transform(test_pair), dtype=torch.float32),
        )
        if not torch.equal(logits, swapped_logits):
            raise RuntimeError("P2 endpoint-order invariance failed")
        probabilities = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)
    state_digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        state_digest.update(name.encode("utf-8"))
        state_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return probabilities, {
        "variant": "P2",
        "model_kind": "COMPACT_SYMMETRIC_MLP",
        "final_training_loss": loss_value,
        "model_state_hash": state_digest.hexdigest(),
        "model_specification": model.specification(),
        "class_weights": [float(value) for value in class_weights],
        "endpoint_order_invariance_verified": True,
    }


def _binary_probability(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    observed = np.unique(train_targets)
    if observed.size == 1:
        probability = np.full(test_features.shape[0], float(observed[0]), dtype=np.float64)
        return probability, {"model_kind": "DETERMINISTIC_SINGLE_CLASS", "observed_classes": observed.tolist()}
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    ).fit(train_features, train_targets)
    positive_index = list(model.classes_).index(1)
    return model.predict_proba(test_features)[:, positive_index], {
        "model_kind": "CLASS_BALANCED_BINARY_LOGISTIC",
        "observed_classes": [int(value) for value in model.classes_],
        "coefficient": [float(value) for value in model.coef_.reshape(-1)],
        "intercept": float(model.intercept_[0]),
    }


def _fit_p3(
    train_pair: np.ndarray,
    train_targets: np.ndarray,
    test_pair: np.ndarray,
    *,
    config: PairwiseOOFConfig,
    fold: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = StandardScaler().fit(train_pair)
    train_scaled = scaler.transform(train_pair)
    test_scaled = scaler.transform(test_pair)
    duplicate_index = PAIR_CLASSES.index("SAME_PERSON_DUPLICATE")
    stage_one_targets = (train_targets == duplicate_index).astype(np.int64)
    duplicate_probability, stage_one = _binary_probability(
        train_scaled,
        stage_one_targets,
        test_scaled,
        c_value=config.logistic_c,
        seed=config.seed + 2000 + fold,
    )
    rest_indices = [index for index in range(len(PAIR_CLASSES)) if index != duplicate_index]
    rest_mask = train_targets != duplicate_index
    if not np.any(rest_mask):
        rest_probabilities = np.zeros((test_pair.shape[0], len(PAIR_CLASSES)), dtype=np.float64)
        rest_probabilities[:, PAIR_CLASSES.index("INSUFFICIENT_EVIDENCE")] = 1.0
        stage_two: dict[str, Any] = {
            "model_kind": "NO_REST_TRAINING_ROWS_SAFE_FALLBACK",
            "observed_classes": [],
        }
    else:
        rest_probabilities, stage_two = _fit_logistic_probabilities(
            train_scaled[rest_mask],
            train_targets[rest_mask],
            test_scaled,
            classes=range(len(PAIR_CLASSES)),
            c_value=config.logistic_c,
            seed=config.seed + 3000 + fold,
        )
        rest_probabilities[:, duplicate_index] = 0.0
        rest_sum = rest_probabilities[:, rest_indices].sum(axis=1, keepdims=True)
        if np.any(rest_sum <= 0.0):
            raise RuntimeError("P3 rest-stage probabilities have zero mass")
        rest_probabilities[:, rest_indices] /= rest_sum
    probabilities = rest_probabilities * (1.0 - duplicate_probability[:, None])
    probabilities[:, duplicate_index] = duplicate_probability
    return probabilities, {
        "variant": "P3",
        "model_kind": "TWO_STAGE_DUPLICATE_VS_REST_THEN_REST_RELATION",
        "stage_one": stage_one,
        "stage_two": stage_two,
        "duplicate_stage_is_binary": True,
        "rest_classes": [PAIR_CLASSES[index] for index in rest_indices],
    }


def pair_metrics_and_screen(ledger: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return complete classwise metrics, confusion counts, and frozen screens."""

    confusion = {target: {prediction: 0 for prediction in PAIR_CLASSES} for target in PAIR_CLASSES}
    for row in ledger:
        target = str(row.get("target_relation") or "")
        prediction = str(row.get("predicted_relation") or "")
        if target not in PAIR_CLASSES or prediction not in PAIR_CLASSES:
            raise ValueError("pair metric ledger contains an invalid class")
        confusion[target][prediction] += 1
    per_class: dict[str, dict[str, Any]] = {}
    for relation in PAIR_CLASSES:
        true_positive = confusion[relation][relation]
        target_support = sum(confusion[relation].values())
        predicted_support = sum(confusion[target][relation] for target in PAIR_CLASSES)
        per_class[relation] = {
            "true_positive": true_positive,
            "false_positive": predicted_support - true_positive,
            "false_negative": target_support - true_positive,
            "target_support": target_support,
            "predicted_support": predicted_support,
            "recall_numerator": true_positive,
            "recall_denominator": target_support,
            "recall": true_positive / target_support if target_support else None,
            "precision_numerator": true_positive,
            "precision_denominator": predicted_support,
            "precision": true_positive / predicted_support if predicted_support else None,
        }

    def screen_row(relation: str, metric: str, threshold: float) -> dict[str, Any]:
        row = per_class[relation]
        value = row[metric]
        return {
            "relation": relation,
            "metric": metric,
            "threshold": threshold,
            "numerator": row[f"{metric}_numerator"],
            "denominator": row[f"{metric}_denominator"],
            "value": value,
            "passed": value is not None and value >= threshold,
        }

    screens = [
        screen_row("SAME_PERSON_DUPLICATE", "recall", 0.90),
        screen_row("SAME_PERSON_DUPLICATE", "precision", 0.90),
        screen_row("MERGED_CONTAINS_BOTH", "recall", 0.50),
        screen_row("MERGED_CONTAINS_BOTH", "precision", 0.80),
    ]
    population = len(ledger)
    correct = sum(confusion[relation][relation] for relation in PAIR_CLASSES)
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.pairwise_metrics_and_screen.v1",
        "pair_classes": list(PAIR_CLASSES),
        "labelled_edge_denominator": population,
        "correct_count": correct,
        "accuracy": correct / population if population else None,
        "confusion_matrix": confusion,
        "per_class": per_class,
        "screens": screens,
        "pair_screen_passed": all(row["passed"] for row in screens),
        "identity_tracking_metric": False,
        "temporal_metric": False,
        "ledger": [dict(row) for row in ledger],
        "ledger_hash": _stable_hash(list(ledger)),
    }
    payload["metrics_hash"] = _stable_hash(payload)
    return payload


def grouped_oof_pairwise_evaluation(
    edge_rows: Sequence[Mapping[str, Any]],
    canonical_feature_payload: Mapping[str, Any],
    fold_by_source_group: Mapping[str, int],
    node_features_by_candidate_uuid: Mapping[str, Sequence[float]],
    *,
    config: PairwiseOOFConfig | None = None,
    variants: Sequence[str] = ("P1", "P2", "P3"),
    feature_payload_by_outer_fold: Mapping[int | str, Mapping[str, Any]] | None = None,
    runtime_edge_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train and evaluate P1--P3 with source-group-exclusive outer folds.

    All held-out labelled edges are evaluated exactly once.  Negative sampling
    is performed independently inside each training fold; every duplicate and
    merged training edge is retained.  When pair features include stacked node
    probabilities, callers pass a distinct payload for each outer fold.  Its
    outer-test rows must use outer-OOF node predictions, while its outer-train
    rows use inner-OOF predictions fitted without any outer-test source group.
    """

    settings = config or PairwiseOOFConfig()
    settings.validate()
    requested = tuple(dict.fromkeys(str(value) for value in variants))
    if not requested or not set(requested) <= {"P1", "P2", "P3"}:
        raise ValueError("pair variants must be a non-empty subset of P1, P2, and P3")
    labelled: list[Mapping[str, Any]] = []
    seen_edges: set[str] = set()
    for index, edge in enumerate(edge_rows):
        if not bool(edge.get("target_available", edge.get("target_relation") is not None)):
            continue
        edge_id = _edge_id(edge, index)
        if edge_id in seen_edges:
            raise ValueError(f"duplicate labelled edge_uuid: {edge_id}")
        seen_edges.add(edge_id)
        _endpoints(edge)
        _target(edge)
        group = str(edge.get("source_group_id") or "").strip()
        if not group or group not in fold_by_source_group:
            raise ValueError(f"labelled edge {edge_id} lacks a frozen source-group fold")
        if bool(edge.get("identity_relation_created", False)):
            raise ValueError("pair evaluator does not accept created identity relations")
        labelled.append(edge)
    labelled.sort(key=lambda edge: _edge_id(edge, 0))
    if not labelled:
        raise ValueError("pair evaluation requires labelled edges")
    runtime_edges = list(labelled if runtime_edge_rows is None else runtime_edge_rows)
    runtime_edges.sort(key=lambda edge: _edge_id(edge, 0))
    runtime_ids: list[str] = []
    for index, edge in enumerate(runtime_edges):
        edge_id = _edge_id(edge, index)
        runtime_ids.append(edge_id)
        _endpoints(edge)
        group = str(edge.get("source_group_id") or "").strip()
        if not group or group not in fold_by_source_group:
            raise ValueError(f"runtime edge {edge_id} lacks a frozen source-group fold")
        if bool(edge.get("identity_relation_created", False)):
            raise ValueError("pair evaluator does not accept created identity relations")
    if len(runtime_ids) != len(set(runtime_ids)):
        raise ValueError("runtime edge rows contain duplicate edge_uuid values")
    fold_values = sorted({int(fold_by_source_group[str(edge["source_group_id"])]) for edge in labelled})
    if len(fold_values) < 2:
        raise ValueError("grouped OOF pair evaluation requires at least two non-empty folds")
    default_vectors_by_edge = canonical_feature_payload.get("vectors_by_edge_uuid") or {}
    default_feature_names = tuple(str(value) for value in (canonical_feature_payload.get("feature_names") or ()))
    if not default_feature_names:
        raise ValueError("canonical pair feature specification is missing")
    per_fold_payloads: dict[int, Mapping[str, Any]] = {}
    if feature_payload_by_outer_fold is not None:
        for fold in fold_values:
            payload = feature_payload_by_outer_fold.get(fold)
            if payload is None:
                payload = feature_payload_by_outer_fold.get(str(fold))
            if payload is None:
                raise ValueError(f"stacked pair features are missing outer fold {fold}")
            per_fold_payloads[fold] = payload
        extra_fold_keys = {int(key) for key in feature_payload_by_outer_fold if str(key).lstrip("-").isdigit()} - set(
            fold_values
        )
        if extra_fold_keys:
            raise ValueError(f"stacked pair features contain unknown outer folds: {sorted(extra_fold_keys)}")
        feature_name_rows = {
            tuple(str(value) for value in (payload.get("feature_names") or ()))
            for payload in per_fold_payloads.values()
        }
        if len(feature_name_rows) != 1 or not next(iter(feature_name_rows), ()):
            raise ValueError("outer-fold pair feature specifications must be identical and non-empty")
        feature_names = next(iter(feature_name_rows))
    else:
        feature_names = default_feature_names
    results: dict[str, dict[str, Any]] = {
        variant: {"prediction_rows": [], "runtime_prediction_rows": [], "fold_ledger": []} for variant in requested
    }
    class_index = {name: index for index, name in enumerate(PAIR_CLASSES)}
    for fold in fold_values:
        held_out = [edge for edge in labelled if int(fold_by_source_group[str(edge["source_group_id"])]) == fold]
        held_out_runtime = [
            edge for edge in runtime_edges if int(fold_by_source_group[str(edge["source_group_id"])]) == fold
        ]
        training_pool = [edge for edge in labelled if int(fold_by_source_group[str(edge["source_group_id"])]) != fold]
        held_out_groups = {str(edge["source_group_id"]) for edge in held_out}
        training_groups = {str(edge["source_group_id"]) for edge in training_pool}
        if not held_out or not training_pool or held_out_groups & training_groups:
            raise ValueError(f"fold {fold} is not source-group exclusive")
        if feature_payload_by_outer_fold is None:
            vectors_by_edge = default_vectors_by_edge
            pair_feature_audit: Mapping[str, Any] = canonical_feature_payload.get("audit") or {}
        else:
            fold_feature_payload = per_fold_payloads[fold]
            vectors_by_edge = fold_feature_payload.get("vectors_by_edge_uuid") or {}
            pair_feature_audit = fold_feature_payload.get("audit") or {}
            if str(pair_feature_audit.get("schema_version") or "").endswith("supplied_oof_node_probability_join.v1"):
                excluded_groups = set(
                    str(value) for value in (pair_feature_audit.get("additional_excluded_source_group_ids") or ())
                )
                if not held_out_groups <= excluded_groups:
                    raise ValueError(
                        f"outer-fold {fold} node-probability features do not exclude all outer-test groups"
                    )
                if not bool(pair_feature_audit.get("all_edge_groups_excluded_from_node_model_fits")) or not bool(
                    pair_feature_audit.get("all_additional_groups_excluded_from_node_model_fits")
                ):
                    raise ValueError(f"outer-fold {fold} node-probability feature audit failed")
        sample_seed = f"M5_5G7B_PAIR::{settings.seed}::fold={fold}"
        selected = list(
            sample_group_balanced_edges(
                training_pool,
                negative_ratio=settings.negative_ratio,
                minimum_negatives_per_group=settings.minimum_negatives_per_group,
                seed=sample_seed,
            )
        )
        training_positive_ids = {
            _edge_id(edge, index) for index, edge in enumerate(training_pool) if _target(edge) in POSITIVE_PAIR_CLASSES
        }
        selected_ids = {_edge_id(edge, index) for index, edge in enumerate(selected)}
        selected_negative_counts = Counter(
            str(edge["source_group_id"]) for edge in selected if _target(edge) not in POSITIVE_PAIR_CLASSES
        )
        missing_positive_ids = sorted(training_positive_ids - selected_ids)
        if missing_positive_ids:
            raise RuntimeError(f"fold {fold} dropped positive pair rows: {missing_positive_ids}")
        train_pair = _dense_matrix(selected, vectors_by_edge)
        test_pair = _dense_matrix(held_out_runtime, vectors_by_edge)
        train_targets = np.asarray([class_index[_target(edge)] for edge in selected], dtype=np.int64)
        if np.unique(train_targets).size < 2:
            raise ValueError(f"fold {fold} sampled training set has fewer than two pair classes")
        training_hash = _stable_hash(sorted(selected_ids))
        common_fold_ledger = {
            "held_out_fold": fold,
            "held_out_source_group_ids": sorted(held_out_groups),
            "training_source_group_ids": sorted(training_groups),
            "source_group_overlap_count": len(held_out_groups & training_groups),
            "training_pool_edge_count": len(training_pool),
            "selected_training_edge_count": len(selected),
            "held_out_evaluation_edge_count": len(held_out),
            "held_out_runtime_edge_count": len(held_out_runtime),
            "positive_training_edge_count": len(training_positive_ids),
            "positive_training_edge_selected_count": len(training_positive_ids & selected_ids),
            "missing_positive_edge_ids": missing_positive_ids,
            "all_duplicate_and_merged_training_pairs_preserved": not missing_positive_ids,
            "negative_training_counts_by_source_group": dict(sorted(selected_negative_counts.items())),
            "group_balanced_deterministic_negative_sampling": True,
            "negative_sampling_seed": sample_seed,
            "selected_training_edge_uuid_hash": training_hash,
            "pair_feature_matrix_hash": str(pair_feature_audit.get("feature_matrix_hash") or ""),
            "outer_fold_specific_pair_features_used": feature_payload_by_outer_fold is not None,
        }
        for variant in requested:
            if variant == "P1":
                probabilities, model_ledger = _fit_p1(
                    train_pair,
                    train_targets,
                    test_pair,
                    config=settings,
                    fold=fold,
                )
            elif variant == "P2":
                probabilities, model_ledger = _fit_p2(
                    selected,
                    held_out_runtime,
                    train_pair,
                    train_targets,
                    test_pair,
                    node_features_by_candidate_uuid,
                    config=settings,
                    fold=fold,
                )
            else:
                probabilities, model_ledger = _fit_p3(
                    train_pair,
                    train_targets,
                    test_pair,
                    config=settings,
                    fold=fold,
                )
            if probabilities.shape != (len(held_out_runtime), len(PAIR_CLASSES)):
                raise RuntimeError(f"{variant} returned an invalid probability matrix")
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
                raise RuntimeError(f"{variant} returned invalid probabilities")
            if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
                raise RuntimeError(f"{variant} probabilities do not sum to one")
            fold_ledger = {**common_fold_ledger, "model": model_ledger}
            fold_ledger["fold_ledger_hash"] = _stable_hash(fold_ledger)
            results[variant]["fold_ledger"].append(fold_ledger)
            probability_by_edge: dict[str, np.ndarray] = {}
            for row_index, edge in enumerate(held_out_runtime):
                edge_id = _edge_id(edge, row_index)
                probability_by_edge[edge_id] = probabilities[row_index]
                prediction_index = int(np.argmax(probabilities[row_index]))
                runtime_prediction_row = {
                    "edge_uuid": edge_id,
                    "source_group_id": str(edge["source_group_id"]),
                    "source_frame_sha256": str(edge.get("source_frame_sha256") or ""),
                    "candidate_uuids": sorted(_endpoints(edge)),
                    "universe": str(edge.get("universe") or "UNKNOWN"),
                    "case_family": str(edge.get("case_family") or "UNKNOWN"),
                    "held_out_fold": fold,
                    "target_available": bool(edge.get("target_available", edge.get("target_relation") is not None)),
                    "predicted_relation": PAIR_CLASSES[prediction_index],
                    "probabilities": {
                        relation: float(probabilities[row_index, index]) for index, relation in enumerate(PAIR_CLASSES)
                    },
                    "selected_training_edge_uuid_hash": training_hash,
                    "held_out_source_group_excluded_from_training": True,
                }
                results[variant]["runtime_prediction_rows"].append(runtime_prediction_row)
            for row_index, edge in enumerate(held_out):
                edge_id = _edge_id(edge, row_index)
                edge_probabilities = probability_by_edge[edge_id]
                prediction_index = int(np.argmax(edge_probabilities))
                prediction_row = {
                    "edge_uuid": edge_id,
                    "source_group_id": str(edge["source_group_id"]),
                    "source_frame_sha256": str(edge.get("source_frame_sha256") or ""),
                    "candidate_uuids": sorted(_endpoints(edge)),
                    "universe": str(edge.get("universe") or "UNKNOWN"),
                    "case_family": str(edge.get("case_family") or "UNKNOWN"),
                    "held_out_fold": fold,
                    "target_relation": _target(edge),
                    "predicted_relation": PAIR_CLASSES[prediction_index],
                    "probabilities": {
                        relation: float(edge_probabilities[index]) for index, relation in enumerate(PAIR_CLASSES)
                    },
                    "selected_training_edge_uuid_hash": training_hash,
                    "held_out_source_group_excluded_from_training": True,
                }
                results[variant]["prediction_rows"].append(prediction_row)
    expected_ids = sorted(_edge_id(edge, index) for index, edge in enumerate(labelled))
    for variant, result in results.items():
        result["prediction_rows"].sort(key=lambda row: row["edge_uuid"])
        result["runtime_prediction_rows"].sort(key=lambda row: row["edge_uuid"])
        predicted_ids = [str(row["edge_uuid"]) for row in result["prediction_rows"]]
        if predicted_ids != expected_ids or len(predicted_ids) != len(set(predicted_ids)):
            raise RuntimeError(f"{variant} OOF ledger does not partition labelled edges exactly once")
        predicted_runtime_ids = [str(row["edge_uuid"]) for row in result["runtime_prediction_rows"]]
        if predicted_runtime_ids != runtime_ids or len(predicted_runtime_ids) != len(set(predicted_runtime_ids)):
            raise RuntimeError(f"{variant} OOF runtime ledger does not partition the edge graph exactly once")
        result["metrics"] = pair_metrics_and_screen(result["prediction_rows"])
        result["all_labelled_edges_predicted_exactly_once"] = True
        result["all_runtime_edges_predicted_exactly_once"] = True
        result["runtime_edge_count"] = len(runtime_edges)
        result["all_positive_training_pairs_preserved"] = all(
            row["all_duplicate_and_merged_training_pairs_preserved"] for row in result["fold_ledger"]
        )
        result["source_group_leakage_count"] = sum(row["source_group_overlap_count"] for row in result["fold_ledger"])
        result["prediction_ledger_hash"] = _stable_hash(result["prediction_rows"])
        result["runtime_prediction_ledger_hash"] = _stable_hash(result["runtime_prediction_rows"])
        result["fold_ledger_hash"] = _stable_hash(result["fold_ledger"])
        result["variant_hash"] = _stable_hash(result)
    payload = {
        "schema_version": "football_intelligence.m5_5g7b.grouped_oof_pairwise_evaluation.v1",
        "development_scope": "SINGLE_MATCH_GROUPED_DEVELOPMENT_ONLY",
        "pair_classes": list(PAIR_CLASSES),
        "feature_names": list(feature_names),
        "feature_specification_hash": _stable_hash(feature_names),
        "outer_fold_specific_pair_features_used": feature_payload_by_outer_fold is not None,
        "stacked_node_probabilities_must_be_nested": True,
        "labelled_edge_count": len(labelled),
        "runtime_edge_count": len(runtime_edges),
        "folds": fold_values,
        "variants": results,
        "config": settings.specification(),
        "identity_tracking_present": False,
        "temporal_prediction_present": False,
        "all_variants_source_group_leakage_free": all(
            result["source_group_leakage_count"] == 0 for result in results.values()
        ),
        "all_variants_preserve_positive_training_pairs": all(
            result["all_positive_training_pairs_preserved"] for result in results.values()
        ),
    }
    payload["evaluation_hash"] = _stable_hash(payload)
    return payload


__all__ = [
    "PAIR_CLASSES",
    "POSITIVE_PAIR_CLASSES",
    "PairwiseOOFConfig",
    "append_supplied_oof_node_probability_summaries",
    "canonicalize_g7a_edge_features",
    "canonicalize_pair_feature_mapping",
    "grouped_oof_pairwise_evaluation",
    "pair_metrics_and_screen",
]
