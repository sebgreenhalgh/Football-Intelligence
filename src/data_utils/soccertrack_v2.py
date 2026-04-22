"""SoccerTrack v2 dataset loader.

Thin, dependency-free reader for GSR / BAS JSON annotations. See docs/format-gsr.md
and docs/format-bas.md for the on-disk schema.

    from src.data_utils.soccertrack_v2 import load_match

    m = load_match("path/to/dataset", match_id="117093")
    for frame in m.gsr_frames(half=1):
        for p in frame.entities:
            print(frame.image_id, p.track_id, p.role, p.x, p.y)

    for ev in m.bas_events():
        print(ev.half, ev.t_ms, ev.label, ev.team)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional

Role = Literal["player", "goalkeeper", "referee", "other"]
TeamSide = Literal["left", "right"]

FPS: int = 25
BAS_LABELS: tuple[str, ...] = (
    "Pass",
    "Drive",
    "Header",
    "High Pass",
    "Out",
    "Cross",
    "Throw In",
    "Shot",
    "Ball Player Block",
    "Player Successful Tackle",
    "Free Kick",
    "Goal",
)


@dataclass(frozen=True)
class Player:
    """Per-frame GSR record for one entity."""

    track_id: int
    role: Role
    jersey_number: Optional[int]
    team_side: Optional[TeamSide]
    x: float
    y: float
    player_id: Optional[str | int] = None
    bbox_image: Optional[tuple[int, int, int, int]] = None
    bbox_pitch: Optional[tuple[float, float, float, float]] = None


@dataclass(frozen=True)
class Frame:
    """All GSR entities observed in a single panoramic frame of a half."""

    half: int
    image_id: int
    entities: tuple[Player, ...]

    @property
    def t_ms(self) -> int:
        """Milliseconds since kickoff of this half (assuming FPS = 25)."""
        return int(round(self.image_id * 1000 / FPS))


@dataclass(frozen=True)
class Event:
    """One BAS annotation."""

    half: int
    clock: str  # "mm:ss" within the half
    t_ms: int  # ms since kickoff of `half`
    label: str
    team: Optional[TeamSide] = None
    player_id: Optional[str | int] = None
    visibility: Optional[str] = None

    @property
    def image_id(self) -> int:
        """Frame index in the half video (assuming FPS = 25)."""
        return int(round(self.t_ms * FPS / 1000))


@dataclass
class Match:
    """One SoccerTrack v2 match. Lazy: JSONs are parsed on first access per half."""

    root: Path
    match_id: str
    _gsr_cache: dict[int, tuple[Frame, ...]] = field(default_factory=dict, repr=False)
    _bas_cache: Optional[tuple[Event, ...]] = field(default=None, repr=False)

    # ---- GSR -----------------------------------------------------------------

    def gsr_path(self, half: int) -> Path:
        return self.root / "gsr" / self.match_id / f"{self.match_id}_{_half_suffix(half)}.json"

    def gsr_frames(self, half: int) -> tuple[Frame, ...]:
        if half not in self._gsr_cache:
            self._gsr_cache[half] = tuple(_parse_gsr(self.gsr_path(half), half))
        return self._gsr_cache[half]

    def gsr_frame(self, half: int, image_id: int) -> Optional[Frame]:
        """O(log n) lookup of the Frame at `image_id` (None if absent)."""
        frames = self.gsr_frames(half)
        lo, hi = 0, len(frames)
        while lo < hi:
            mid = (lo + hi) // 2
            if frames[mid].image_id < image_id:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(frames) and frames[lo].image_id == image_id:
            return frames[lo]
        return None

    # ---- BAS -----------------------------------------------------------------

    def bas_path(self) -> Path:
        return self.root / "bas" / self.match_id / f"{self.match_id}_12_class_events.json"

    def bas_events(self) -> tuple[Event, ...]:
        if self._bas_cache is None:
            self._bas_cache = tuple(_parse_bas(self.bas_path()))
        return self._bas_cache

    def events_of(self, label: str) -> tuple[Event, ...]:
        if label not in BAS_LABELS:
            raise ValueError(f"Unknown BAS label: {label!r}. Expected one of {BAS_LABELS}.")
        return tuple(e for e in self.bas_events() if e.label == label)


# ---- Parsers -----------------------------------------------------------------


def _half_suffix(half: int) -> str:
    if half == 1:
        return "1st"
    if half == 2:
        return "2nd"
    raise ValueError(f"half must be 1 or 2, got {half!r}")


def _parse_gsr(path: Path, half: int) -> Iterator[Frame]:
    if not path.exists():
        raise FileNotFoundError(f"GSR annotation not found: {path}")
    records = json.loads(path.read_text())
    grouped: dict[int, list[Player]] = {}
    for r in records:
        image_id = int(r["image_id"])
        grouped.setdefault(image_id, []).append(_player_from(r))
    for image_id in sorted(grouped):
        yield Frame(half=half, image_id=image_id, entities=tuple(grouped[image_id]))


def _player_from(r: dict) -> Player:
    bbox_image = tuple(r["bbox_image"]) if "bbox_image" in r and r["bbox_image"] is not None else None
    bbox_pitch = tuple(r["bbox_pitch"]) if "bbox_pitch" in r and r["bbox_pitch"] is not None else None
    return Player(
        track_id=int(r["track_id"]),
        role=r["role"],
        jersey_number=_maybe_int(r.get("jersey_number")),
        team_side=r.get("team_side"),
        x=float(r["x"]),
        y=float(r["y"]),
        player_id=r.get("player_id"),
        bbox_image=bbox_image,  # type: ignore[arg-type]
        bbox_pitch=bbox_pitch,  # type: ignore[arg-type]
    )


def _parse_bas(path: Path) -> Iterator[Event]:
    if not path.exists():
        raise FileNotFoundError(f"BAS annotation not found: {path}")
    data = json.loads(path.read_text())
    for a in data["annotations"]:
        half_str, clock = a["gameTime"].split(" - ")
        label = a["label"]
        if label not in BAS_LABELS:
            raise ValueError(f"Unknown BAS label in {path.name}: {label!r}")
        yield Event(
            half=int(half_str),
            clock=clock,
            t_ms=int(a["position"]),
            label=label,
            team=a.get("team"),
            player_id=a.get("player_id"),
            visibility=a.get("visibility"),
        )


def _maybe_int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    return int(v)


# ---- Public entry points -----------------------------------------------------


def load_match(root: str | Path, match_id: str) -> Match:
    """Return a lazy Match handle rooted at `root` for `match_id`.

    `root` is the top-level folder with `gsr/`, `bas/`, `mot/`, `videos/` subdirectories.
    """
    root = Path(root)
    if not (root / "gsr").is_dir():
        raise FileNotFoundError(f"Expected `gsr/` under {root}. Is this a SoccerTrack v2 root?")
    return Match(root=root, match_id=str(match_id))


def list_matches(root: str | Path) -> list[str]:
    """Return available match IDs (sorted) under a SoccerTrack v2 root."""
    root = Path(root)
    gsr_root = root / "gsr"
    if not gsr_root.is_dir():
        return []
    return sorted(p.name for p in gsr_root.iterdir() if p.is_dir())


def iter_matches(root: str | Path) -> Iterable[Match]:
    for mid in list_matches(root):
        yield load_match(root, mid)
