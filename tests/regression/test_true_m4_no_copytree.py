from __future__ import annotations

from pathlib import Path


def test_true_m4_build_code_does_not_use_package_mirroring() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "src/football_intelligence/replay/source_access.py",
        root / "src/football_intelligence/replay/m1_node_recovery.py",
        root / "src/football_intelligence/replay/frame_lookup.py",
        root / "src/football_intelligence/replay/true_m4_engine.py",
        root / "src/football_intelligence/replay/true_m4_renderer.py",
        root / "src/football_intelligence/replay/true_m4_documents.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    banned = [
        "shutil.copytree",
        "shutil.copy",
        "copyfile",
        "mirror_preserved_m4_package",
        "step2m4_sparse_handoff_package",
    ]
    assert not any(pattern in text for pattern in banned)
