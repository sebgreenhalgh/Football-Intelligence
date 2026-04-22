"""Evaluation utilities for SoccerTrack v2.

Submodules:
    gs_hota   — Game State Reconstruction (GSR) scorer (GS-HOTA, via SoccerNet impl).
    bas_map   — Ball Action Spotting (BAS) scorer (tolerant mAP).
    mot_hota  — Multi-Object Tracking (MOT) scorer (HOTA / IDF1 / MOTA via TrackEval).

Each submodule exposes `score_many(pred_root, gt_root, match_ids, ...)` returning a dict
and is also runnable as a CLI: `python -m src.evaluation.<name> --help`.
"""
