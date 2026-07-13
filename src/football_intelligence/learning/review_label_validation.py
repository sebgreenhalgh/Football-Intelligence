from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _spatial_region(example: dict[str, Any]) -> str:
    metadata = example.get("selection_metadata") if isinstance(example.get("selection_metadata"), dict) else {}
    region = metadata.get("spatial_region_key") or example.get("spatial_region_key")
    if region:
        return str(region)
    return f"round_{example.get('review_round', 'unknown')}"


def class_sufficiency_readiness(
    examples: list[dict[str, Any]],
    *,
    task_type: str,
    required_labels: set[str] | None = None,
    min_examples: int = 5,
    min_clusters_per_class_for_application: int = 5,
) -> dict[str, Any]:
    usable = [
        row for row in examples if row.get("task_type") == task_type and row.get("label_usable_for_training") is True
    ]
    label_counts = Counter(str(row.get("normalized_training_label")) for row in usable)
    clusters_by_label: dict[str, set[str]] = defaultdict(set)
    regions_by_label: dict[str, set[str]] = defaultdict(set)
    for row in usable:
        label = str(row.get("normalized_training_label"))
        clusters_by_label[label].add(str(row.get("equivalence_cluster_id") or row.get("review_case_id")))
        regions_by_label[label].add(_spatial_region(row))

    distinct_classes = set(label_counts)
    required_ok = True if not required_labels else required_labels.issubset(distinct_classes)
    two_classes = len(distinct_classes) >= 2
    enough_examples = len(usable) >= min_examples
    grouped_validation_possible = two_classes and len({row.get("equivalence_cluster_id") for row in usable}) >= 3
    leakage_risk = any(
        not cluster for row in usable for cluster in [row.get("equivalence_cluster_id") or row.get("review_case_id")]
    )
    min_clusters_observed = min((len(value) for value in clusters_by_label.values()), default=0)
    application_ready = (
        enough_examples
        and two_classes
        and required_ok
        and grouped_validation_possible
        and min_clusters_observed >= min_clusters_per_class_for_application
        and not leakage_risk
    )
    status = "READY_FOR_TRAINING" if enough_examples and two_classes and required_ok else "BLOCKED_LABEL_DIVERSITY"
    if task_type == "visual_continuity_edge_review" and not required_ok:
        status = "BLOCKED_SINGLE_CLASS_REVIEW_LABELS"
    return {
        "artifact": f"m5_4e_{task_type}_training_readiness",
        "task_type": task_type,
        "usable_example_count": len(usable),
        "distinct_class_count": len(distinct_classes),
        "examples_per_class": dict(sorted(label_counts.items())),
        "equivalence_clusters_per_class": {
            label: len(clusters) for label, clusters in sorted(clusters_by_label.items())
        },
        "spatial_regions_per_class": {label: len(regions) for label, regions in sorted(regions_by_label.items())},
        "leakage_risk": leakage_risk,
        "train_validation_feasible": grouped_validation_possible,
        "required_labels": sorted(required_labels or []),
        "required_labels_present": required_ok,
        "min_clusters_per_class_for_application": min_clusters_per_class_for_application,
        "application_ready": application_ready,
        "status": status,
        "blocked_reason": None if status == "READY_FOR_TRAINING" else status,
    }
