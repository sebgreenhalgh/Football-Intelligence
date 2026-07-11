import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_intelligence.paths import DATABASE_PATH, ensure_dir


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def stable_run_id(match_id: str, run_type: str, clip_id: str | None, parameters: dict[str, Any]) -> str:
    scope = clip_id or "match"
    digest = stable_hash(json_dumps(parameters))
    return f"{match_id}_{scope}_{run_type}_{digest}"


def connect(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS matches (
            match_id TEXT PRIMARY KEY,
            source_name TEXT NOT NULL,
            competition TEXT NULL,
            home_team TEXT NULL,
            away_team TEXT NULL,
            match_date TEXT NULL,
            notes TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS source_videos (
            video_id TEXT PRIMARY KEY,
            match_id TEXT NOT NULL,
            half INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            fps REAL NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            frame_count INTEGER NOT NULL,
            duration_seconds REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS clips (
            clip_id TEXT PRIMARY KEY,
            match_id TEXT NOT NULL,
            source_video_id TEXT NOT NULL,
            half INTEGER NOT NULL,
            start_seconds REAL NOT NULL,
            end_seconds REAL NOT NULL,
            duration_seconds REAL NOT NULL,
            file_path TEXT NOT NULL,
            fps REAL NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            frame_count INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
            FOREIGN KEY (source_video_id) REFERENCES source_videos(video_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS processing_runs (
            run_id TEXT PRIMARY KEY,
            run_type TEXT NOT NULL,
            match_id TEXT NOT NULL,
            clip_id TEXT NULL,
            script_name TEXT NOT NULL,
            model_name TEXT NULL,
            model_version TEXT NULL,
            parameters_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            notes TEXT NOT NULL,
            FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
            FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS frames (
            frame_id TEXT PRIMARY KEY,
            match_id TEXT NOT NULL,
            clip_id TEXT NOT NULL,
            frame_index INTEGER NOT NULL,
            source_frame_index INTEGER NULL,
            timestamp_seconds REAL NOT NULL,
            file_path TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            extraction_run_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
            FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE,
            FOREIGN KEY (extraction_run_id) REFERENCES processing_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS detections (
            detection_id TEXT PRIMARY KEY,
            match_id TEXT NOT NULL,
            clip_id TEXT NOT NULL,
            frame_id TEXT NOT NULL,
            detection_run_id TEXT NOT NULL,
            object_type TEXT NOT NULL,
            class_id INTEGER NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            x1 REAL NOT NULL,
            y1 REAL NOT NULL,
            x2 REAL NOT NULL,
            y2 REAL NOT NULL,
            center_x REAL NOT NULL,
            center_y REAL NOT NULL,
            width REAL NOT NULL,
            height REAL NOT NULL,
            area REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
            FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE,
            FOREIGN KEY (frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE,
            FOREIGN KEY (detection_run_id) REFERENCES processing_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS tracks (
            track_id TEXT PRIMARY KEY,
            match_id TEXT NOT NULL,
            clip_id TEXT NOT NULL,
            object_type TEXT NOT NULL,
            team_id TEXT NULL,
            player_id TEXT NULL,
            tracker_name TEXT NOT NULL,
            tracking_run_id TEXT NOT NULL,
            start_time_seconds REAL NOT NULL,
            end_time_seconds REAL NOT NULL,
            num_observations INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
            FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE,
            FOREIGN KEY (tracking_run_id) REFERENCES processing_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS track_observations (
            track_observation_id TEXT PRIMARY KEY,
            track_id TEXT NOT NULL,
            detection_id TEXT NOT NULL,
            frame_id TEXT NOT NULL,
            timestamp_seconds REAL NOT NULL,
            center_x REAL NOT NULL,
            center_y REAL NOT NULL,
            x1 REAL NOT NULL,
            y1 REAL NOT NULL,
            x2 REAL NOT NULL,
            y2 REAL NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (track_id) REFERENCES tracks(track_id) ON DELETE CASCADE,
            FOREIGN KEY (detection_id) REFERENCES detections(detection_id) ON DELETE CASCADE,
            FOREIGN KEY (frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS pitch_calibrations (
            calibration_id TEXT PRIMARY KEY,
            match_id TEXT NOT NULL,
            clip_id TEXT NULL,
            frame_id TEXT NULL,
            method TEXT NOT NULL,
            status TEXT NOT NULL,
            homography_json TEXT NULL,
            source TEXT NOT NULL,
            notes TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
            FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE,
            FOREIGN KEY (frame_id) REFERENCES frames(frame_id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS physical_metrics_windows (
            metric_id TEXT PRIMARY KEY,
            match_id TEXT NOT NULL,
            clip_id TEXT NOT NULL,
            track_id TEXT NOT NULL,
            window_start_seconds REAL NOT NULL,
            window_end_seconds REAL NOT NULL,
            distance_pixels REAL NULL,
            top_speed_pixels_per_second REAL NULL,
            avg_speed_pixels_per_second REAL NULL,
            distance_meters REAL NULL,
            top_speed_mps REAL NULL,
            top_speed_kmh REAL NULL,
            avg_speed_mps REAL NULL,
            calibration_id TEXT NULL,
            metric_run_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
            FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE,
            FOREIGN KEY (track_id) REFERENCES tracks(track_id) ON DELETE CASCADE,
            FOREIGN KEY (metric_run_id) REFERENCES processing_runs(run_id) ON DELETE CASCADE,
            FOREIGN KEY (calibration_id) REFERENCES pitch_calibrations(calibration_id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS assets (
            asset_id TEXT PRIMARY KEY,
            match_id TEXT NOT NULL,
            clip_id TEXT NULL,
            asset_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            created_by_run_id TEXT NULL,
            notes TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
            FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE,
            FOREIGN KEY (created_by_run_id) REFERENCES processing_runs(run_id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_frames_clip_time ON frames(clip_id, timestamp_seconds);
        CREATE INDEX IF NOT EXISTS idx_detections_frame ON detections(frame_id);
        CREATE INDEX IF NOT EXISTS idx_detections_clip_type ON detections(clip_id, object_type);
        CREATE INDEX IF NOT EXISTS idx_track_observations_track_time
            ON track_observations(track_id, timestamp_seconds);
        CREATE INDEX IF NOT EXISTS idx_assets_match_clip ON assets(match_id, clip_id, asset_type);

        CREATE TABLE IF NOT EXISTS detection_quality_reviews (
            review_id TEXT PRIMARY KEY,
            match_id TEXT NOT NULL,
            clip_id TEXT NOT NULL,
            frame_id TEXT NOT NULL,
            visible_players_estimate INTEGER NULL,
            detected_players_count INTEGER NULL,
            false_positives_count INTEGER NULL,
            missed_players_estimate INTEGER NULL,
            acceptable_for_tracking INTEGER NULL,
            reviewer_notes TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
            FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE,
            FOREIGN KEY (frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS detection_run_metrics (
            metric_id TEXT PRIMARY KEY,
            processing_run_id TEXT NOT NULL,
            match_id TEXT NOT NULL,
            clip_id TEXT NOT NULL,
            frames_processed INTEGER NOT NULL,
            total_detections INTEGER NOT NULL,
            avg_detections_per_frame REAL NOT NULL,
            median_detections_per_frame REAL NOT NULL,
            frames_with_zero_detections INTEGER NOT NULL,
            frames_with_16plus_detections INTEGER NOT NULL,
            notes TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (processing_run_id) REFERENCES processing_runs(run_id) ON DELETE CASCADE,
            FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
            FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()


def upsert_processing_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    run_type: str,
    match_id: str,
    clip_id: str | None,
    script_name: str,
    model_name: str | None,
    model_version: str | None,
    parameters: dict[str, Any],
    notes: str,
) -> None:
    conn.execute(
        """
        INSERT INTO processing_runs (
            run_id, run_type, match_id, clip_id, script_name, model_name, model_version,
            parameters_json, created_at, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            run_type = excluded.run_type,
            match_id = excluded.match_id,
            clip_id = excluded.clip_id,
            script_name = excluded.script_name,
            model_name = excluded.model_name,
            model_version = excluded.model_version,
            parameters_json = excluded.parameters_json,
            created_at = excluded.created_at,
            notes = excluded.notes
        """,
        (
            run_id,
            run_type,
            match_id,
            clip_id,
            script_name,
            model_name,
            model_version,
            json.dumps(parameters, sort_keys=True),
            utc_now(),
            notes,
        ),
    )


def upsert_asset(
    conn: sqlite3.Connection,
    *,
    asset_id: str,
    match_id: str,
    clip_id: str | None,
    asset_type: str,
    file_path: Path,
    created_by_run_id: str | None,
    notes: str,
) -> None:
    conn.execute(
        """
        INSERT INTO assets (
            asset_id, match_id, clip_id, asset_type, file_path, created_by_run_id, notes, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(asset_id) DO UPDATE SET
            match_id = excluded.match_id,
            clip_id = excluded.clip_id,
            asset_type = excluded.asset_type,
            file_path = excluded.file_path,
            created_by_run_id = excluded.created_by_run_id,
            notes = excluded.notes,
            created_at = excluded.created_at
        """,
        (
            asset_id,
            match_id,
            clip_id,
            asset_type,
            str(file_path.resolve()),
            created_by_run_id,
            notes,
            utc_now(),
        ),
    )


def fetch_one(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    return conn.execute(sql, params).fetchone()


def fetch_all(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    return list(conn.execute(sql, params).fetchall())
