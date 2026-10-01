"""Read-only, empty-real-root instrumentation; no reviewer server is started."""
import argparse
import contextlib
import io
import json
import time
from unittest.mock import patch
import cv2
try:
    from scripts import g7g_b_r3_run_temporal_gold_pilot as gate
except ModuleNotFoundError:
    import g7g_b_r3_run_temporal_gold_pilot as gate
from football_intelligence.gold import temporal_reviewer_r3
from football_intelligence.gold.corpus import canonical, file_hash, immutable_write, require

STAGE = gate.r1.STAGE.parent / "G7G_B_R4_FAST_FRAME_ASSET_LAUNCH_VALIDATION_REPAIR_v1"


def measure(mode):
    before = {p.relative_to(gate.r1.REAL).as_posix(): file_hash(p)
              for p in gate.r1.REAL.rglob("*") if p.is_file()}
    require(not before, "Benchmark requires empty real root; otherwise use TEMP")
    hashes, decodes, rgb_calls = [], [], []
    original_hash, original_decode, original_rgb = file_hash, cv2.imdecode, cv2.cvtColor

    def hashed(path):
        if path.suffix == ".png":
            hashes.append(str(path.resolve()))
        return original_hash(path)

    def decoded(*args, **kwargs):
        require(mode != "fast", "Fast path unexpectedly decoded a PNG")
        decodes.append(1)
        return original_decode(*args, **kwargs)

    def rgb(*args, **kwargs):
        require(mode != "fast", "Fast path unexpectedly converted RGB")
        rgb_calls.append(1)
        return original_rgb(*args, **kwargs)

    output = io.StringIO()
    start = time.perf_counter()
    with patch.object(temporal_reviewer_r3, "file_hash", hashed), patch.object(cv2, "imdecode", decoded), patch.object(cv2, "cvtColor", rgb), contextlib.redirect_stdout(output):
        result = gate.check() if mode == "fast" else gate.audit_assets()
    elapsed = time.perf_counter() - start
    after = {p.relative_to(gate.r1.REAL).as_posix(): file_hash(p)
             for p in gate.r1.REAL.rglob("*") if p.is_file()}
    require(before == after, "Real root changed")
    _, integrity = gate.validate_frozen_gold_fast()
    require(len(hashes) == len(set(hashes)) == 9, "Expected exactly nine constructor PNG hashes")
    require(len(decodes) == len(rgb_calls) == (0 if mode == "fast" else 9), "Decode/RGB count mismatch")
    report = {"mode": mode, "elapsed_seconds": elapsed, "frame_asset_file_hashes": len(hashes),
              "frame_assets_decoded": len(decodes), "RGB_conversions": len(rgb_calls),
              "real_root_unchanged": True, "gold_unchanged": True,
              "gold_integrity": integrity, "result": result, "messages": output.getvalue(),
              "production_ready": False}
    immutable_write(STAGE / (mode.upper() + "_MEASUREMENT.json"), canonical(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fast", "audit-assets"))
    args = parser.parse_args()
    print(json.dumps(measure(args.mode), indent=2, sort_keys=True))
