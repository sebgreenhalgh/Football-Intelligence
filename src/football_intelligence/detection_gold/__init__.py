"""Detection-gold annotation schemas, evaluation contracts, and persistence."""

from football_intelligence.detection_gold.matching import evaluate_detection_gold
from football_intelligence.detection_gold.models import validate_case_annotation
from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence

__all__ = [
    "DetectionGoldPilotPersistence",
    "evaluate_detection_gold",
    "validate_case_annotation",
]
