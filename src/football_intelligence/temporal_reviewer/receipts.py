"""Receipt cardinality helpers shared by audits and release gates."""

from __future__ import annotations

from pathlib import Path

from football_intelligence.temporal_reviewer.contracts import contained_path


def acknowledgement_path(root: Path, event_id: str) -> Path:
    return contained_path(root, "receipts", "acknowledgements", f"ack-{event_id}.json")


def action_receipt_path(root: Path, action_id: str) -> Path:
    return contained_path(root, "receipts", "actions", f"action-ack-{action_id}.json")


__all__ = ["acknowledgement_path", "action_receipt_path"]
