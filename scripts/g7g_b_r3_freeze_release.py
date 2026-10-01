"""Publish the display-only R3 binding without changing any prior release."""
import json
try:
    from scripts import g7g_b_r3_run_temporal_gold_pilot as gate
except ModuleNotFoundError:
    import g7g_b_r3_run_temporal_gold_pilot as gate
from football_intelligence.gold.corpus import canonical, digest, file_hash, immutable_write


def freeze():
    repo = gate.REPO
    predecessor = repo / "configs/reviewers/temporal_r2_launch_v2.json"
    old = json.loads(predecessor.read_bytes())
    paths = ["src/football_intelligence/gold/temporal_reviewer_r3.py",
             *("src/football_intelligence/gold/temporal_reviewer_r3_static/" + name
               for name in ("index.html", "app.js", "styles.css", "view.js"))]
    bundle = {str((repo / p).relative_to(repo)): file_hash(repo / p) for p in paths}
    sources = {**old["source_sha256"], **{p: file_hash(repo / p) for p in paths + [gate.RUNNER_PATH]}}
    binding = {
        "schema_version": "football_intelligence.g7g_b.reviewer_release.v3",
        "reviewer_release": gate.RELEASE, "reviewer_sha256": digest(canonical(bundle)),
        "bundle_source_sha256": bundle, "source_sha256": sources,
        "supersedes_config_sha256": file_hash(predecessor),
        "pilot_selection_sha256": gate.r1.PILOT_SHA,
        "starting_code_commit": "98f9450d6c2a22ecfdbcaf16c9c828285388a08e",
        "maximum_zoom_multiplier": 24, "image_smoothing_disabled_above_fit_multiplier": 8,
        "candidate_blind": True, "context_video_ui": False, "display_only": True,
        "truth_schemas_changed": False, "truth_builders_changed": False,
        "canonical_gold_mutation": False, "production_ready": False,
    }
    immutable_write(gate.RELEASE_CONFIG, canonical(binding))
    gate.validate_launch_binding()
    return binding


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2, sort_keys=True))
