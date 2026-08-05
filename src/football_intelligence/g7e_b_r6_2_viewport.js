"use strict";

// Canonical display-only viewport geometry for the R6.2 panorama and detail views.
// Values are CSS-pixel or normalized source-space values. DPR is applied only
// when the canvas backing store is prepared by the renderer.
window.R62Viewport = (() => {
  const MIN_ZOOM = 1;
  const MAX_ZOOM = 8;
  const finite = (value, fallback) => Number.isFinite(value) ? Number(value) : fallback;
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

  function createState(value = {}) {
    return {
      zoom: clamp(finite(value.zoom, 1), MIN_ZOOM, MAX_ZOOM),
      focalX: clamp(finite(value.focalX, .5), 0, 1),
      focalY: clamp(finite(value.focalY, .5), 0, 1),
      panMode: value.panMode === true,
    };
  }

  function dimensions(sourceWidth, sourceHeight, viewportWidth, viewportHeight) {
    const sw = Math.max(1, finite(sourceWidth, 1));
    const sh = Math.max(1, finite(sourceHeight, 1));
    const vw = Math.max(1, finite(viewportWidth, 1));
    const vh = Math.max(1, finite(viewportHeight, 1));
    return { sw, sh, vw, vh, fitScale: Math.min(vw / sw, vh / sh) };
  }

  function transform(value, sourceWidth, sourceHeight, viewportWidth, viewportHeight) {
    const state = createState(value);
    const { sw, sh, vw, vh, fitScale } = dimensions(sourceWidth, sourceHeight, viewportWidth, viewportHeight);
    const scale = fitScale * state.zoom;
    const width = sw * scale;
    const height = sh * scale;
    let left = vw / 2 - state.focalX * width;
    let top = vh / 2 - state.focalY * height;
    if (width <= vw) left = (vw - width) / 2;
    else left = clamp(left, vw - width, 0);
    if (height <= vh) top = (vh - height) / 2;
    else top = clamp(top, vh - height, 0);
    const focalX = width <= vw ? .5 : clamp((vw / 2 - left) / width, 0, 1);
    const focalY = height <= vh ? .5 : clamp((vh / 2 - top) / height, 0, 1);
    return {
      sourceWidth: sw, sourceHeight: sh, viewportWidth: vw, viewportHeight: vh,
      fitScale, scale, width, height, left, top,
      state: { zoom: state.zoom, focalX, focalY, panMode: state.panMode },
    };
  }

  function sourceToLocal(point, geometry) {
    return [geometry.left + point[0] * geometry.scale, geometry.top + point[1] * geometry.scale];
  }

  function localToSource(point, geometry) {
    return [(point[0] - geometry.left) / geometry.scale, (point[1] - geometry.top) / geometry.scale];
  }

  function pointInsideSource(point, geometry, tolerance = 1e-7) {
    return point[0] >= -tolerance && point[1] >= -tolerance &&
      point[0] <= geometry.sourceWidth + tolerance && point[1] <= geometry.sourceHeight + tolerance;
  }

  function zoomAtLocal(value, localX, localY, factor, sourceWidth, sourceHeight, viewportWidth, viewportHeight) {
    const before = transform(value, sourceWidth, sourceHeight, viewportWidth, viewportHeight);
    const source = localToSource([localX, localY], before);
    const zoom = clamp(before.state.zoom * finite(factor, 1), MIN_ZOOM, MAX_ZOOM);
    const { sw, sh, vw, vh, fitScale } = dimensions(sourceWidth, sourceHeight, viewportWidth, viewportHeight);
    const scale = fitScale * zoom;
    const width = sw * scale;
    const height = sh * scale;
    const proposedLeft = localX - source[0] * scale;
    const proposedTop = localY - source[1] * scale;
    return transform(
      {
        zoom,
        focalX: width <= vw ? .5 : (vw / 2 - proposedLeft) / width,
        focalY: height <= vh ? .5 : (vh / 2 - proposedTop) / height,
        panMode: before.state.panMode,
      },
      sw, sh, vw, vh,
    ).state;
  }

  function panFromStart(value, deltaX, deltaY, sourceWidth, sourceHeight, viewportWidth, viewportHeight) {
    const before = transform(value, sourceWidth, sourceHeight, viewportWidth, viewportHeight);
    return transform(
      {
        zoom: before.state.zoom,
        focalX: before.state.focalX - finite(deltaX, 0) / before.width,
        focalY: before.state.focalY - finite(deltaY, 0) / before.height,
        panMode: before.state.panMode,
      },
      sourceWidth, sourceHeight, viewportWidth, viewportHeight,
    ).state;
  }

  function fit(value = {}) {
    return { zoom: 1, focalX: .5, focalY: .5, panMode: value.panMode === true };
  }

  return Object.freeze({
    MIN_ZOOM, MAX_ZOOM, createState, transform, sourceToLocal, localToSource,
    pointInsideSource, zoomAtLocal, panFromStart, fit,
  });
})();
