/* Pure display transform. Never serialize this into an annotation document. */
"use strict";
const View = Object.freeze({
  fit(w, h, vw, vh) {
    const scale = Math.min(vw / w, vh / h);
    return {scale, x: (vw - w * scale) / 2, y: (vh - h * scale) / 2};
  },
  source(v, p) { return {x: (p.x - v.x) / v.scale, y: (p.y - v.y) / v.scale}; },
  screen(v, p) { return {x: p.x * v.scale + v.x, y: p.y * v.scale + v.y}; },
  zoom(v, factor, p, minimum) {
    const source = this.source(v, p);
    const scale = Math.max(minimum, Math.min(minimum * 8, v.scale * factor));
    return {scale, x: p.x - source.x * scale, y: p.y - source.y * scale};
  },
  pan(v, dx, dy) { return {...v, x: v.x - dx, y: v.y - dy}; }
});
if (typeof module !== "undefined") module.exports = View;
