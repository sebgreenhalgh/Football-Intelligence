"""Stdlib fallback for the focused G7F-F integrity tests."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from scripts import check_g7f_f_provisional as frozen
from scripts.check_project_metadata import check as check_metadata


@unittest.skipUnless(frozen.MANIFEST.is_file(), "external immutable manifest unavailable")
class ProvisionalFreezeTests(unittest.TestCase):
    def test_frozen_evidence_gold_and_registry(self) -> None:
        result = frozen.check()
        self.assertTrue(result["valid"])
        self.assertEqual(result["candidate_c"], "g7f_b_r1_recall_conf_012")
        self.assertFalse(result["blind_repeat_identities_accessed"])

    def test_changed_candidate_identity_fails_closed(self) -> None:
        original = frozen.read

        def altered(path):
            result = original(path)
            if path == frozen.STAGE / "candidate_identity_mapping.json":
                result = copy.deepcopy(result)
                result["candidates"]["Candidate C"]["role"] = "LOCAL_DEFAULT_RERUN"
            return result

        with patch.object(frozen, "read", side_effect=altered):
            with self.assertRaisesRegex(ValueError, "role mismatch"):
                frozen.check()

    def test_false_promotion_fails_closed(self) -> None:
        original = frozen.read

        def altered(path):
            result = original(path)
            if path == frozen.MANIFEST:
                result = copy.deepcopy(result)
                result["production_ready"] = True
            return result

        with patch.object(frozen, "read", side_effect=altered):
            with self.assertRaisesRegex(ValueError, "production promotion"):
                frozen.check()

    def test_canonical_metadata_and_links(self) -> None:
        self.assertTrue(check_metadata(frozen.ROOT)["valid"])


if __name__ == "__main__":
    unittest.main()
