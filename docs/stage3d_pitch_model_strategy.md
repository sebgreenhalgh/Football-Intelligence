# Stage 3D Pitch Model Strategy

Stage 3D is calibration-label infrastructure for future automatic pitch calibration. It is visual QA only and must not be used for speed, distance, fatigue, player load, team shape, pass, dribble, or tactical metrics.

## Why Manual Clicking Does Not Scale

Manual clicking is useful for bootstrap labels, debugging, and low-confidence exception handling. It is not a scalable production workflow because every new camera angle, venue, lens distortion, and weather/lighting condition can require fresh calibration review.

The long-term workflow should use manual labels as training data, not repeated production labour.

## Why Classical CV Is Useful But Insufficient

The Stage 3D.0 classical CV fallback found useful pitch-line structure, but the auto/manual comparison showed large alignment errors. Classical thresholding and Hough lines are good proposal generators, especially for touchlines and high-contrast markings, but they struggle with panoramic distortion, partial occlusion, shadows, bench areas, and curved apparent lines.

Classical CV should remain a candidate proposal and debugging layer, not the final calibration authority.

## Recommended Learned Model Path

1. Use a SoccerNet/Roboflow-style pitch keypoint or line model if one is available locally.
2. Fine-tune on this project’s calibration labels, including manual keypoints, boundary polylines, accepted auto proposals, and rejected negatives.
3. Run model predictions through the same visual-only confidence gate used by Stage 3D.
4. Require manual review only when confidence is low, touchline/official-lane constraints disagree, or comparison checks fail.
5. Keep calibration visual-only until it has been separately validated for metric reconstruction.

## Candidate Model Types

- Keypoint detector: predicts named pitch landmarks such as centre spot, touchline intersections, penalty box corners, and goal-line/touchline corners.
- Line segmentation model: predicts pitch-line masks or polylines for touchlines, halfway line, box lines, goal lines, and centre circle.
- Hybrid keypoint + polyline model: uses line segmentation for robust boundaries and keypoints for semantic anchoring.

## Recommended First Target

Train a pitch line/keypoint detector for panoramic football footage. The first useful model does not need metric-grade calibration; it should reliably propose touchlines, halfway line, box/goal-line structures, and named anchor points with confidence scores good enough to reduce manual calibration review.

## Current Stage 3D.1 Role

Stage 3D.1 creates review packs, a proposal review CSV, reviewed label JSONs, negative examples, a pitch-calibration training dataset structure, and a benchmark report. Its job is to turn manual calibration work and automatic proposals into reusable model-training assets.
