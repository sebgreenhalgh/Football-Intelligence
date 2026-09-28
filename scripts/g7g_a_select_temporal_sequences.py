"""Freeze eight candidate-blind, source-video-bound temporal pilot sequences.

No detector artifacts, scores, models or candidate boxes are opened. This is
selection and source decoding only; it does not create human Gold events.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from football_intelligence.gold.corpus import GoldCorpus, canonical, file_hash, immutable_write


REPO = Path(__file__).resolve().parents[1]
EXTERNAL = REPO.parent
STAGE = EXTERNAL / "experiments/football_observation_reasoner/part 9/G7G_A_TEMPORAL_GOLD_CORPUS_AND_SEQUENCE_FOUNDATION_v1"
GOLD = EXTERNAL / "datasets/gold_corpus"
OFFSETS_SECONDS = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
CONTEXT_SECONDS = 4.0
SELECTION_SCHEMA = "football_intelligence.gold.temporal_sequence_selection.v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_images(corpus: GoldCorpus) -> list[dict]:
    manifest = corpus.manifest()
    sources = corpus._rows(manifest, "source_registry")
    roots = [source for source in sources if source.get("images")]
    require(len(roots) == 1, "Exactly one canonical dense source-image registry is required")
    return roots[0]["images"]


def video_info(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate,nb_frames,width,height", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    fps = Fraction(stream["avg_frame_rate"])
    require(fps > 0 and int(stream["nb_frames"]) > 0, "Source video lacks frame/fps metadata")
    return {"fps_numerator": fps.numerator, "fps_denominator": fps.denominator, "frame_count": int(stream["nb_frames"]), "width": int(stream["width"]), "height": int(stream["height"])}


def video_sha(path: Path) -> str:
    print(f"hashing source video: {path.name}", flush=True)
    return file_hash(path)


def frame_rgb_sha(frame: np.ndarray) -> str:
    return hashlib.sha256(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()


def decode(cap: cv2.VideoCapture, index: int) -> np.ndarray:
    require(cap.set(cv2.CAP_PROP_POS_FRAMES, index), f"Cannot seek to source frame {index}")
    ok, frame = cap.read()
    require(ok and frame is not None, f"Cannot decode source frame {index}")
    return frame


def clip_bytes(path: Path, start_seconds: float, duration_seconds: float, target: Path) -> bytes:
    # Encode only a view derivative; the immutable source video is never touched.
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".building.mp4")
    require(not temp.exists(), f"Stale context-build file; inspect before retry: {temp}")
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{start_seconds:.6f}", "-i", str(path), "-t", f"{duration_seconds:.6f}", "-vf", "scale=1280:-2", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-movflags", "+faststart", "-y", str(temp)],
            check=True,
        )
        data = temp.read_bytes()
        require(len(data) > 0, "Context derivative is empty")
        return data
    finally:
        temp.unlink(missing_ok=True)


def choose_anchors(images: list[dict], active: dict[str, dict]) -> list[tuple[str, dict]]:
    eligible = []
    for image in images:
        frame_id = "gf-" + image["source_frame_sha256"]
        if frame_id in active:
            eligible.append({**image, "gold_frame_id": frame_id, "active": active[frame_id]})
    scored = [x for x in eligible if x["active"]["split"] == "DENSE_GOLD_INTERNAL_VALIDATION"]
    by_match = defaultdict(list)
    for item in eligible:
        by_match[item["active"]["match_id"]].append(item)
    require(len(by_match) == 6, "Six source matches with canonical Gold are required")

    def usable(item: dict) -> bool:
        """Deterministic source-only rejection screen before freezing an anchor."""
        video = EXTERNAL / item["source_video_relative_path"]
        if not video.is_file():
            return False
        try:
            metadata = video_info(video)
            fps = metadata["fps_numerator"] / metadata["fps_denominator"]
            anchor_index = int(item["frame_index_zero_based"])
            indices = [anchor_index + round(fps * offset) for offset in OFFSETS_SECONDS]
            if min(indices) < 0 or max(indices) >= metadata["frame_count"]:
                return False
            if anchor_index - round(fps * CONTEXT_SECONDS) < 0 or anchor_index + round(fps * CONTEXT_SECONDS) >= metadata["frame_count"]:
                return False
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                return False
            try:
                decoded = [decode(cap, index) for index in indices]
            finally:
                cap.release()
            if frame_rgb_sha(decoded[4]) != item["source_frame_sha256"]:
                return False
            thumbs = [cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32) for frame in decoded]
            return all(float(np.mean(np.abs(left - right)) / 255) < 0.20 for left, right in zip(thumbs, thumbs[1:]))
        except (RuntimeError, ValueError, OSError):
            return False

    primary = []
    for match_id, rows in sorted(by_match.items()):
        ordered = sorted(rows, key=lambda x: (x["active"]["split"] != "DENSE_GOLD_INTERNAL_VALIDATION", Path(x["path"]).stem))
        chosen = next((item for item in ordered if usable(item)), None)
        require(chosen is not None, f"No source-valid canonical anchor for match {match_id}")
        primary.append(chosen)
    primary_videos = {x["active"]["match_id"]: x["source_video_relative_path"] for x in primary}
    extras = sorted(
        (x for x in scored if x not in primary and x["source_video_relative_path"] == primary_videos[x["active"]["match_id"]]),
        key=lambda x: (x["active"]["match_id"], Path(x["path"]).stem),
    )
    reserves = []
    for item in extras:
        if item["active"]["match_id"] not in {x["active"]["match_id"] for x in reserves} and usable(item):
            reserves.append(item)
        if len(reserves) == 2:
            break
    require(len(primary) == 6 and len(reserves) == 2, "Cannot select six primary plus two reserves")
    return [("PRIMARY", x) for x in primary] + [("RESERVE", x) for x in reserves]


def main() -> None:
    corpus = GoldCorpus(GOLD)
    status = corpus.status()
    require(status["detection_gold_frames"] == 16 and status["detection_gold_people"] == 862, "Canonical Gold baseline mismatch")
    require(corpus.manifest()["layers_available"] == ["DETECTION"], "Unexpected active Gold layer")
    active = {row["gold_frame_id"]: row for row in corpus.annotations(layer="DETECTION", active=True)}
    selected = choose_anchors(source_images(corpus), active)
    video_cache = {}
    sequences = []
    assets = STAGE / "sequence_assets"
    for role, anchor in selected:
        match_id = anchor["active"]["match_id"]
        relative_video = anchor["source_video_relative_path"]
        video = EXTERNAL / relative_video
        require(video.is_file(), f"Missing source video: {relative_video}")
        if relative_video not in video_cache:
            video_cache[relative_video] = {**video_info(video), "source_video_sha256": video_sha(video)}
        metadata = video_cache[relative_video]
        fps = Fraction(metadata["fps_numerator"], metadata["fps_denominator"])
        anchor_index = int(anchor["frame_index_zero_based"])
        indices = [anchor_index + round(float(fps) * offset) for offset in OFFSETS_SECONDS]
        require(all(0 <= i < metadata["frame_count"] for i in indices), "Required frame falls outside video")
        require(len(set(indices)) == 9 and indices[4] == anchor_index, "Invalid nine-frame sampling")
        context_first = anchor_index - round(float(fps) * CONTEXT_SECONDS)
        context_last = anchor_index + round(float(fps) * CONTEXT_SECONDS)
        require(context_first >= 0 and context_last < metadata["frame_count"], "Context window falls outside video")
        capture = cv2.VideoCapture(str(video))
        require(capture.isOpened(), "Cannot open source video")
        decoded = []
        try:
            for index in indices:
                frame = decode(capture, index)
                require((frame.shape[1], frame.shape[0]) == (metadata["width"], metadata["height"]), "Decoded frame dimensions changed")
                decoded.append(frame)
        finally:
            capture.release()
        hashes = [frame_rgb_sha(frame) for frame in decoded]
        require(hashes[4] == anchor["source_frame_sha256"], f"Anchor cannot be bound to canonical Gold: {Path(anchor['path']).name}")
        thumbs = [cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32) for frame in decoded]
        cut_scores = [round(float(np.mean(np.abs(left - right)) / 255), 6) for left, right in zip(thumbs, thumbs[1:])]
        require(max(cut_scores) < 0.20, f"Hard camera-cut screening failed at {Path(anchor['path']).name}: {cut_scores}")
        frame_ids = ["gf-" + value for value in hashes]
        identity = {
            "source_video_sha256": metadata["source_video_sha256"],
            "source_frame_numbers_zero_based": indices,
            "source_rgb_sha256": hashes,
            "sampling_rule": "nearest source frame to fixed offsets -1.00 through +1.00 seconds; Python round ties-to-even",
        }
        sequence_id = "gs-" + hashlib.sha256(canonical(identity)).hexdigest()
        frame_rows = []
        for order, (index, frame, frame_id, source_sha) in enumerate(zip(indices, decoded, frame_ids, hashes, strict=True), start=1):
            relative_asset = f"sequence_assets/{sequence_id}/frame-{order:02d}.png"
            ok, encoded = cv2.imencode(".png", frame)
            require(ok, "Cannot encode source frame PNG")
            immutable_write(STAGE / relative_asset, encoded.tobytes())
            active_row = active.get(frame_id)
            frame_rows.append({
                "order": order,
                "gold_frame_id": frame_id,
                "source_frame_number_zero_based": index,
                "actual_timestamp_seconds": round(float(Fraction(index, 1) / fps), 6),
                "source_rgb_sha256": source_sha,
                "asset_path": relative_asset,
                "asset_sha256": file_hash(STAGE / relative_asset),
                "existing_canonical_detection_event_sha256": active_row["annotation_event_sha256"] if active_row else None,
                "detection_read_only": active_row is not None,
            })
        context_relative = f"sequence_assets/{sequence_id}/context.mp4"
        context_path = STAGE / context_relative
        context_data = clip_bytes(video, float(Fraction(context_first, 1) / fps), float(Fraction(context_last - context_first + 1, 1) / fps), context_path)
        immutable_write(context_path, context_data)
        sequences.append({
            "role": role,
            "gold_sequence_id": sequence_id,
            "anonymized_source_match_id": f"Match-{sorted({x['active']['match_id'] for _, x in selected}).index(match_id)+1:02d}",
            "source_match_id_for_provenance": match_id,
            "source_video_relative_path": relative_video,
            "source_video_sha256": metadata["source_video_sha256"],
            "source_video_bytes": video.stat().st_size,
            "source_fps": {"numerator": fps.numerator, "denominator": fps.denominator},
            "source_video_frame_count": metadata["frame_count"],
            "source_width": metadata["width"],
            "source_height": metadata["height"],
            "anchor_gold_frame_id": anchor["gold_frame_id"],
            "anchor_detection_event_sha256": anchor["active"]["annotation_event_sha256"],
            "anchor_selection_rationale": "earliest scored canonical DETECTION anchor by immutable DG image ID per match" if role == "PRIMARY" else "next scored canonical anchor on a selected primary source video, distinct match first",
            "annotation_frame_sampling_rule": "nine nearest valid source frames at offsets -1.00,-0.75,-0.50,-0.25,0,+0.25,+0.50,+0.75,+1.00 seconds",
            "annotation_frames": frame_rows,
            "context_window": {"first_source_frame_zero_based": context_first, "last_source_frame_zero_based": context_last, "path": context_relative, "sha256": file_hash(context_path), "purpose": "human visual context only; not Gold truth"},
            "camera_cut_selection_statistic": {"metric": "adjacent annotation-frame 64x36 grayscale mean absolute difference / 255", "scores": cut_scores, "rejection_threshold": 0.20},
            "schema_versions": {"selection": SELECTION_SCHEMA, "detection": "existing canonical dense person Gold", "tracklet": "football_intelligence.gold.tracklet_sequence.v1", "ball": "football_intelligence.gold.ball_sequence.v1", "match_state": "football_intelligence.gold.match_state_sequence.v1"},
        })
        print(f"selected {role} {sequence_id} anchor={Path(anchor['path']).name} match={match_id}", flush=True)
    require(len(sequences) == 8 and len({x["gold_sequence_id"] for x in sequences}) == 8, "Sequence identity collision")
    manifest = {
        "schema_version": SELECTION_SCHEMA,
        "gold_release": "gold-v0.1.0",
        "gold_corpus_manifest_sha256": file_hash(GOLD / "corpus_manifest.json"),
        "gold_release_manifest_sha256": file_hash(GOLD / "releases/gold-v0.1.0/manifest.json"),
        "candidate_data_used": False,
        "source_selection_only": True,
        "primary_sequence_count": 6,
        "reserve_sequence_count": 2,
        "source_match_count": 6,
        "selection_rule": "one earliest scored canonical DETECTION anchor per match; two next same-video scored anchors from distinct matches in match-ID order; reject only failed bounds/hash/decode/camera-cut, then use next eligible same-match canonical anchor",
        "sequences": sequences,
        "production_ready": False,
    }
    immutable_write(STAGE / "TEMPORAL_SEQUENCE_SELECTION_v1.json", canonical(manifest))
    print(json.dumps({"selection_sha256": file_hash(STAGE / "TEMPORAL_SEQUENCE_SELECTION_v1.json"), "primary": 6, "reserve": 2, "annotation_frames": 72, "candidate_data_used": False}), flush=True)


if __name__ == "__main__":
    main()
