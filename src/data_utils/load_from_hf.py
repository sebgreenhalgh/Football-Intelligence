"""Fetch SoccerTrack v2 from Hugging Face and load a Match.

    from src.data_utils.load_from_hf import load_match_from_hf
    m = load_match_from_hf("117093")
    for ev in m.bas_events():
        print(ev.half, ev.t_ms, ev.label)

`snapshot_download` caches locally under HF_HOME so repeat calls are free.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from src.data_utils.soccertrack_v2 import Match, load_match

HF_REPO_ID = "atomscott/soccertrack-v2"


def snapshot(
    match_ids: Optional[Iterable[str]] = None,
    *,
    revision: str = "main",
    cache_dir: Optional[str | Path] = None,
    repo_id: str = HF_REPO_ID,
) -> Path:
    """Download (or reuse cached) SoccerTrack v2 files from Hugging Face.

    When `match_ids` is provided, only gsr/bas/mot/raw/videos entries for those
    matches are fetched. Returns the local directory that mirrors the dataset
    root, suitable for passing to `src.data_utils.soccertrack_v2.load_match`.
    """
    from huggingface_hub import snapshot_download  # deferred import

    allow_patterns: Optional[list[str]] = None
    if match_ids is not None:
        allow_patterns = []
        for mid in match_ids:
            allow_patterns += [
                f"gsr/{mid}/*",
                f"bas/{mid}/*",
                f"mot/{mid}/*",
                f"raw/{mid}/*",
                f"videos/{mid}*",
            ]

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        allow_patterns=allow_patterns,
    )
    return Path(path)


def load_match_from_hf(
    match_id: str,
    *,
    revision: str = "main",
    cache_dir: Optional[str | Path] = None,
    repo_id: str = HF_REPO_ID,
) -> Match:
    """Download just the files needed for `match_id` and return a lazy Match."""
    root = snapshot(
        match_ids=[str(match_id)],
        revision=revision,
        cache_dir=cache_dir,
        repo_id=repo_id,
    )
    return load_match(root, match_id=str(match_id))
