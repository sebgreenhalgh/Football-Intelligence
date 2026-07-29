"use strict";

/* Shared source-coordinate math for the R2 reviewer, Node audit, and browser. */
(function expose(factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  globalThis.TargetBoxCalibration = api;
})(function buildCalibration() {
  const EPSILON = 1e-9;
  const DEFAULT_LAYOUTS = {
    full: { css_width: 1280, css_height: 720 },
    context: { css_width: 1280, css_height: 720 },
    closeup: { css_width: 800, css_height: 720 },
  };
  const CONTEXT_PROFILE = { padding_multiplier: 14, min_width: 800, min_height: 450, aspect: 16 / 9 };
  const CLOSEUP_PROFILE = { padding_multiplier: 4.5, min_width: 420, min_height: 236.25, aspect: 16 / 9 };

  function finite(value) { return typeof value === "number" && Number.isFinite(value); }
  function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
  function rounded(value) { return Math.round(value * 1e6) / 1e6; }
  function rectangle(left, top, right, bottom) {
    return { left, top, right, bottom, width: right - left, height: bottom - top };
  }
  function boxFromArray(values) {
    if (!Array.isArray(values) || values.length !== 4) throw new Error("A source box must have four coordinates.");
    return rectangle(...values);
  }
  function validateBox(box, sourceWidth, sourceHeight) {
    const valid = [box.left, box.top, box.right, box.bottom, sourceWidth, sourceHeight].every(finite)
      && sourceWidth > 0 && sourceHeight > 0 && box.left >= 0 && box.top >= 0
      && box.right <= sourceWidth && box.bottom <= sourceHeight
      && box.right > box.left && box.bottom > box.top;
    return { valid, reason: valid ? null : "box is not finite, ordered, or source-bounded" };
  }
  function cropForBox(box, sourceWidth, sourceHeight, profile, centerOffset = { x: 0, y: 0 }) {
    const minimumWidth = Math.max(box.width, profile.min_width);
    const minimumHeight = Math.max(box.height, profile.min_height);
    let width = Math.max(box.width * profile.padding_multiplier, minimumWidth);
    let height = Math.max(box.height * profile.padding_multiplier, minimumHeight);
    if (width / height < profile.aspect) width = height * profile.aspect;
    else height = width / profile.aspect;
    width = Math.min(sourceWidth, Math.max(width, box.width));
    height = Math.min(sourceHeight, Math.max(height, box.height));
    let left = (box.left + box.right) / 2 + (centerOffset.x || 0) - width / 2;
    let top = (box.top + box.bottom) / 2 + (centerOffset.y || 0) - height / 2;
    left = clamp(left, 0, sourceWidth - width);
    top = clamp(top, 0, sourceHeight - height);
    left = clamp(left, box.right - width, box.left);
    top = clamp(top, box.bottom - height, box.top);
    return rectangle(rounded(left), rounded(top), rounded(left + width), rounded(top + height));
  }
  function containTransform(sourceRect, cssWidth, cssHeight) {
    if (![sourceRect.width, sourceRect.height, cssWidth, cssHeight].every(finite)
      || sourceRect.width <= 0 || sourceRect.height <= 0 || cssWidth <= 0 || cssHeight <= 0) {
      throw new Error("A transform needs positive finite source and CSS dimensions.");
    }
    const scale = Math.min(cssWidth / sourceRect.width, cssHeight / sourceRect.height);
    const contentWidth = sourceRect.width * scale;
    const contentHeight = sourceRect.height * scale;
    return {
      source: sourceRect,
      css_width: cssWidth,
      css_height: cssHeight,
      scale,
      content: rectangle((cssWidth - contentWidth) / 2, (cssHeight - contentHeight) / 2,
        (cssWidth + contentWidth) / 2, (cssHeight + contentHeight) / 2),
    };
  }
  function sourcePointToDisplay(transform, point) {
    return {
      x: transform.content.left + (point.x - transform.source.left) * transform.scale,
      y: transform.content.top + (point.y - transform.source.top) * transform.scale,
    };
  }
  function displayPointToSource(transform, point) {
    if (point.x < transform.content.left - EPSILON || point.x > transform.content.right + EPSILON
      || point.y < transform.content.top - EPSILON || point.y > transform.content.bottom + EPSILON) return null;
    return {
      x: transform.source.left + (point.x - transform.content.left) / transform.scale,
      y: transform.source.top + (point.y - transform.content.top) / transform.scale,
    };
  }
  function sourceBoxToDisplay(transform, box) {
    const topLeft = sourcePointToDisplay(transform, { x: box.left, y: box.top });
    const bottomRight = sourcePointToDisplay(transform, { x: box.right, y: box.bottom });
    return rectangle(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y);
  }
  function displayBoxToSource(transform, box) {
    const topLeft = displayPointToSource(transform, { x: box.left, y: box.top });
    const bottomRight = displayPointToSource(transform, { x: box.right, y: box.bottom });
    if (!topLeft || !bottomRight) return null;
    return rectangle(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y);
  }
  function contains(outer, inner) {
    return inner.left >= outer.left - EPSILON && inner.top >= outer.top - EPSILON
      && inner.right <= outer.right + EPSILON && inner.bottom <= outer.bottom + EPSILON;
  }
  function maxDifference(first, second) {
    return Math.max(...["left", "top", "right", "bottom"].map((key) => Math.abs(first[key] - second[key])));
  }
  function viewAudit(box, crop, layout) {
    const transform = containTransform(crop, layout.css_width, layout.css_height);
    const displayed = sourceBoxToDisplay(transform, box);
    const sourceRoundTrip = displayBoxToSource(transform, displayed);
    const displayRoundTrip = sourceBoxToDisplay(transform, sourceRoundTrip);
    return {
      crop_source_xyxy: [rounded(crop.left), rounded(crop.top), rounded(crop.right), rounded(crop.bottom)],
      displayed_rectangle_css_xyxy: [rounded(displayed.left), rounded(displayed.top), rounded(displayed.right), rounded(displayed.bottom)],
      image_content_css_xyxy: [rounded(transform.content.left), rounded(transform.content.top), rounded(transform.content.right), rounded(transform.content.bottom)],
      source_round_trip_error_px: rounded(maxDifference(box, sourceRoundTrip)),
      display_round_trip_error_css_px: rounded(maxDifference(displayed, displayRoundTrip)),
    };
  }
  function auditTarget(scene, target, layouts = DEFAULT_LAYOUTS) {
    const box = boxFromArray(target.source_box_xyxy);
    const sourceWidth = scene.source_width;
    const sourceHeight = scene.source_height;
    const boxValidation = validateBox(box, sourceWidth, sourceHeight);
    const fullCrop = rectangle(0, 0, sourceWidth, sourceHeight);
    const contextCrop = cropForBox(box, sourceWidth, sourceHeight, CONTEXT_PROFILE);
    const closeCrop = cropForBox(box, sourceWidth, sourceHeight, CLOSEUP_PROFILE);
    const full = viewAudit(box, fullCrop, layouts.full);
    const context = viewAudit(box, contextCrop, layouts.context);
    const closeup = viewAudit(box, closeCrop, layouts.closeup);
    const sourceError = Math.max(full.source_round_trip_error_px, context.source_round_trip_error_px, closeup.source_round_trip_error_px);
    const cssError = Math.max(full.display_round_trip_error_css_px, context.display_round_trip_error_css_px, closeup.display_round_trip_error_css_px);
    const cropContained = contains(contextCrop, box) && contains(closeCrop, box);
    return {
      scene_id: scene.scene_id,
      target_id: target.target_id,
      source_frame_dimensions: [sourceWidth, sourceHeight],
      source_box_xyxy: target.source_box_xyxy,
      crop_source_xyxy: closeup.crop_source_xyxy,
      full_frame_displayed_rectangle: full.displayed_rectangle_css_xyxy,
      context_displayed_rectangle: context.displayed_rectangle_css_xyxy,
      closeup_displayed_rectangle: closeup.displayed_rectangle_css_xyxy,
      source_to_display_to_source_error_px: sourceError,
      display_to_source_to_display_error_css_px: cssError,
      in_bounds: boxValidation.valid,
      box_area_source_px2: rounded(box.width * box.height),
      crop_containment: cropContained,
      views: { full, context, closeup },
      passed: boxValidation.valid && cropContained && sourceError <= 0.5 && cssError <= 1.0,
    };
  }
  function cssCanvasMetrics(cssWidth, cssHeight, dpr) {
    return { css_width: cssWidth, css_height: cssHeight, dpr, backing_width: Math.round(cssWidth * dpr), backing_height: Math.round(cssHeight * dpr) };
  }
  return {
    CLOSEUP_PROFILE, CONTEXT_PROFILE, DEFAULT_LAYOUTS, auditTarget, boxFromArray, containTransform,
    contains, cropForBox, cssCanvasMetrics, displayBoxToSource, displayPointToSource, sourceBoxToDisplay,
    sourcePointToDisplay, validateBox,
  };
});
