from __future__ import annotations

from football_intelligence.replay.runner import source_mutation_report


def test_true_m4_source_mutation_report_detects_no_changes_for_equal_inventory() -> None:
    inventory = {"schema_version": "x", "roots": [{"relative_uri": "a", "inventory_hash": "h"}]}
    report = source_mutation_report(inventory, inventory)
    assert report["passed"] is True
