from pathlib import Path

from football_intelligence.dataset_inventory import discover_matches, sha256_file


def test_streaming_hash(tmp_path: Path) -> None:
    path = tmp_path / "source.bin"
    path.write_bytes(b"g7c" * 1000)
    assert sha256_file(path) == "4bf4328cbb72f93a13718a8c1cbceb012446f2c0d3321a1d61b4c572f11f94cf"


def test_discover_matches_requires_exact_count(tmp_path: Path) -> None:
    for name in ("128057", "128058"):
        (tmp_path / name).mkdir()
    assert [p.name for p in discover_matches(tmp_path, 2)] == ["128057", "128058"]
