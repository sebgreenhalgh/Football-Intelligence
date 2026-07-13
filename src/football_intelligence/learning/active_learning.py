from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from football_intelligence.review.schemas import safety_payload, stable_hash


def equivalence_key(row: dict[str, Any]) -> str:
    explicit = row.get("static_persistence_signature") or row.get("equivalence_key")
    category = str(row.get("category") or row.get("review_category") or "unknown")
    if explicit and "static" in category:
        return f"{category}:{explicit}"
    frame_bucket = int(row.get("source_frame_sequence", row.get("frame_sequence", 0))) // 20
    region = row.get("spatial_region_key") or row.get("spatial_context") or row.get("source_region") or "unknown_region"
    reason_family = ",".join(sorted(str(reason) for reason in row.get("uncertainty_reasons", [])[:2]))
    candidate_key = explicit or row.get("candidate_id", row.get("edge_id", "x"))
    return f"{category}:{region}:f{frame_bucket}:{reason_family}:{candidate_key}"


def build_review_equivalence_clusters(pool_rows: list[dict[str, Any]]) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        clusters[equivalence_key(row)].append(row)
    cluster_rows: list[dict[str, Any]] = []
    for index, (key, members) in enumerate(sorted(clusters.items()), start=1):
        cluster_id = f"m5_4d_review_cluster_{index:04d}"
        representative = max(
            members,
            key=lambda row: (
                float(row.get("information_gain_score", 0.0)),
                float(row.get("model_uncertainty", 0.0)),
                -int(row.get("priority_hint", 999999)),
            ),
        )
        for member in members:
            member["equivalence_cluster_id"] = cluster_id
            member["representative_of_count"] = len(members)
            member["is_cluster_representative"] = member is representative
        cluster_rows.append(
            {
                "equivalence_cluster_id": cluster_id,
                "equivalence_key": key,
                "member_count": len(members),
                "representative_candidate_id": representative.get("candidate_id") or representative.get("edge_id"),
                "category": representative.get("category"),
                "review_task_type": representative.get("task_type"),
                "selection_reason": representative.get("selection_reason"),
            }
        )
    return {
        "artifact": "m5_4d_review_equivalence_clusters",
        "cluster_count": len(cluster_rows),
        "rows": cluster_rows,
        **safety_payload(),
    }


def select_diverse_review_rounds(
    pool_rows: list[dict[str, Any]],
    *,
    round_size: int = 20,
    round_count: int = 3,
) -> dict[str, Any]:
    selected_ids: set[str] = set()
    selected_clusters: set[str] = set()
    rounds: list[list[dict[str, Any]]] = []
    exclusions: list[dict[str, Any]] = []
    ordered = sorted(
        pool_rows,
        key=lambda row: (
            float(row.get("information_gain_score", 0.0)),
            float(row.get("model_uncertainty", 0.0)),
            row.get("category", ""),
        ),
        reverse=True,
    )
    categories_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        categories_by_name[str(row.get("category"))].append(row)

    def can_take(
        row: dict[str, Any],
        *,
        category_counts: Counter[str],
        frame_buckets: Counter[int],
        category_cap: int,
        frame_cap: int,
    ) -> tuple[bool, str | None]:
        row_id = str(row.get("candidate_id") or row.get("edge_id"))
        cluster_id = str(row.get("equivalence_cluster_id"))
        category = str(row.get("category"))
        frame_bucket = int(row.get("source_frame_sequence", row.get("frame_sequence", 0))) // 30
        if row_id in selected_ids:
            return False, "candidate_already_selected"
        if cluster_id in selected_clusters:
            return False, "equivalence_cluster_already_selected"
        if category_counts[category] >= category_cap:
            return False, "category_round_cap"
        if frame_buckets[frame_bucket] >= frame_cap:
            return False, "frame_range_round_cap"
        return True, None

    def take(
        row: dict[str, Any],
        *,
        round_index: int,
        round_rows: list[dict[str, Any]],
        category_counts: Counter[str],
        frame_buckets: Counter[int],
    ) -> None:
        row_id = str(row.get("candidate_id") or row.get("edge_id"))
        cluster_id = str(row.get("equivalence_cluster_id"))
        category = str(row.get("category"))
        frame_bucket = int(row.get("source_frame_sequence", row.get("frame_sequence", 0))) // 30
        selected = dict(row)
        selected["review_round"] = round_index
        selected["round_position"] = len(round_rows) + 1
        selected["why_selected"] = selected.get("selection_reason") or "diverse_high_information_case"
        round_rows.append(selected)
        selected_ids.add(row_id)
        selected_clusters.add(cluster_id)
        category_counts[category] += 1
        frame_buckets[frame_bucket] += 1

    for round_index in range(1, round_count + 1):
        category_counts: Counter[str] = Counter()
        frame_buckets: Counter[int] = Counter()
        round_rows: list[dict[str, Any]] = []
        category_order = sorted(
            categories_by_name,
            key=lambda category: max(
                float(row.get("information_gain_score", 0.0)) for row in categories_by_name[category]
            ),
            reverse=True,
        )
        while len(round_rows) < min(round_size, len(category_order)):
            progressed = False
            for category in category_order:
                if len(round_rows) >= round_size:
                    break
                if category_counts[category] > 0:
                    continue
                for row in categories_by_name[category]:
                    ok, _reason = can_take(
                        row, category_counts=category_counts, frame_buckets=frame_buckets, category_cap=5, frame_cap=20
                    )
                    if ok:
                        take(
                            row,
                            round_index=round_index,
                            round_rows=round_rows,
                            category_counts=category_counts,
                            frame_buckets=frame_buckets,
                        )
                        progressed = True
                        break
            if not progressed:
                break
        for row in ordered:
            if len(round_rows) >= round_size:
                break
            ok, reason = can_take(
                row, category_counts=category_counts, frame_buckets=frame_buckets, category_cap=5, frame_cap=8
            )
            if ok:
                take(
                    row,
                    round_index=round_index,
                    round_rows=round_rows,
                    category_counts=category_counts,
                    frame_buckets=frame_buckets,
                )
            elif reason not in {"candidate_already_selected"}:
                exclusions.append(
                    {"candidate_id": str(row.get("candidate_id") or row.get("edge_id")), "reason": reason}
                )
        rounds.append(round_rows)
    all_selected = [row for round_rows in rounds for row in round_rows]
    return {
        "artifact": "m5_4d_active_review_rounds",
        "rounds": rounds,
        "total_selected": len(all_selected),
        "category_distribution": dict(sorted(Counter(row["category"] for row in all_selected).items())),
        "selection_hash": stable_hash(
            [
                {
                    "candidate_id": row.get("candidate_id") or row.get("edge_id"),
                    "round": row["review_round"],
                    "category": row["category"],
                    "cluster": row.get("equivalence_cluster_id"),
                }
                for row in all_selected
            ]
        ),
        "exclusions": exclusions,
        **safety_payload(),
    }


def diversity_audit(rounds_payload: dict[str, Any]) -> dict[str, Any]:
    selected = [row for round_rows in rounds_payload.get("rounds", []) for row in round_rows]
    clusters = [row.get("equivalence_cluster_id") for row in selected]
    categories = Counter(row.get("category") for row in selected)
    duplicates = [cluster for cluster, count in Counter(clusters).items() if count > 1]
    return {
        "artifact": "m5_4d_review_diversity_audit",
        "passed": not duplicates and len(categories) >= 10 and rounds_payload.get("total_selected", 0) <= 60,
        "total_selected": rounds_payload.get("total_selected", 0),
        "category_distribution": dict(sorted(categories.items())),
        "duplicate_selected_clusters": duplicates,
        "supports_up_to_60_cases": rounds_payload.get("total_selected", 0) <= 60,
        "facade_cluster_principal_slot_cap": "one_per_equivalence_cluster",
        **safety_payload(),
    }
