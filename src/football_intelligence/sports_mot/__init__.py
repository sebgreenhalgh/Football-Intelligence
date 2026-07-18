"""Match-local, visual-only sports-MOT research interfaces."""

from football_intelligence.sports_mot.architecture import (
    ADAPTER_SPECS,
    ANNOTATION_STATES,
    PitchParticipantGate,
    build_common_observation_graph,
    build_mhsag_artifacts,
    evaluate_gold_paths,
    run_tracking_adapter,
)

__all__ = [
    "ADAPTER_SPECS",
    "ANNOTATION_STATES",
    "PitchParticipantGate",
    "build_common_observation_graph",
    "build_mhsag_artifacts",
    "evaluate_gold_paths",
    "run_tracking_adapter",
]
