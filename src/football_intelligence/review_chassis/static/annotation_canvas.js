(function () {
  const SCHEMA_VERSION = "football_intelligence.review_chassis.spatial_annotation.v2";
  const MIN_BBOX_SIZE = 2;

  function clamp(value, min, max) {
    return Math.min(Math.max(Number(value) || 0, min), max);
  }

  function round(value) {
    return Math.round(Number(value) * 1000) / 1000;
  }

  function normalizeBbox(bbox, width, height) {
    const x1 = clamp(Math.min(bbox.x1, bbox.x2), 0, width);
    const x2 = clamp(Math.max(bbox.x1, bbox.x2), 0, width);
    const y1 = clamp(Math.min(bbox.y1, bbox.y2), 0, height);
    const y2 = clamp(Math.max(bbox.y1, bbox.y2), 0, height);
    return {x1: round(x1), y1: round(y1), x2: round(x2), y2: round(y2)};
  }

  function bboxValid(bbox) {
    return bbox && bbox.x2 - bbox.x1 >= MIN_BBOX_SIZE && bbox.y2 - bbox.y1 >= MIN_BBOX_SIZE;
  }

  function parseNote(noteText, fallbackCaseId) {
    try {
      const parsed = JSON.parse(noteText || "{}");
      return parsed?.spatial_annotation || parsed || {case_id: fallbackCaseId};
    } catch {
      return {case_id: fallbackCaseId};
    }
  }

  function candidateArea(candidate) {
    const box = candidate.bbox || {};
    return Math.max(0, Number(box.x2) - Number(box.x1)) * Math.max(0, Number(box.y2) - Number(box.y1));
  }

  function hitCandidates(candidates, point, scale) {
    const tolerance = 10 / Math.max(scale || 1, 0.01);
    return (candidates || [])
      .filter((candidate) => {
        const box = candidate.bbox || {};
        return point.x >= box.x1 - tolerance && point.x <= box.x2 + tolerance
          && point.y >= box.y1 - tolerance && point.y <= box.y2 + tolerance;
      })
      .sort((a, b) => candidateArea(a) - candidateArea(b)
        || Number(a.anonymous_candidate_number) - Number(b.anonymous_candidate_number));
  }

  function pointerPoint(event, viewport, transform) {
    const rect = viewport.getBoundingClientRect();
    return {
      x: round((event.clientX - rect.left - transform.panX) / transform.scale),
      y: round((event.clientY - rect.top - transform.panY) / transform.scale),
    };
  }

  function annotationToNote(annotation) {
    return JSON.stringify({spatial_annotation: annotation}, null, 2);
  }

  function validationForDecision(decision, annotation) {
    const errors = [];
    if (decision === "TARGET_VISIBLE_DRAW_BBOX" && !bboxValid(annotation.reviewer_bbox)) {
      errors.push("Draw or type a valid target bbox first.");
    }
    if (
      decision === "TARGET_VISIBLE_SELECT_EXISTING_DETECTION"
      && !annotation.existing_candidate_number
    ) {
      errors.push("Select an anonymous detection box first.");
    }
    if (annotation.partial_or_occluded === true) {
      const hasPoint = Array.isArray(annotation.occlusion_points) && annotation.occlusion_points.length > 0;
      const hasReason = annotation.occlusion_location_status === "not_localizable"
        && String(annotation.occlusion_not_localizable_reason || "").trim();
      if (!hasPoint && !hasReason) {
        errors.push("Partial/occluded annotations need an occlusion point or a not-localizable reason.");
      }
    }
    return errors;
  }

  class SpatialAnnotationCanvas {
    constructor(root, options) {
      this.root = root;
      this.options = options;
      this.caseData = options.caseData;
      this.asset = options.asset;
      this.candidates = options.candidates || [];
      this.noteElement = options.noteElement;
      this.onChange = options.onChange;
      this.transform = {scale: 1, panX: 0, panY: 0};
      this.mode = "draw";
      this.drag = null;
      this.history = [];
      this.imageSize = {
        width: Number(this.asset?.metadata?.width || this.asset?.metadata?.original_width || 0),
        height: Number(this.asset?.metadata?.height || this.asset?.metadata?.original_height || 0),
      };
      this.annotation = this.fromNote();
      this.render();
    }

    fromNote() {
      const parsed = parseNote(this.noteElement?.value || "", this.caseData.case_id);
      const annotation = {
        schema_version: SCHEMA_VERSION,
        case_id: this.caseData.case_id,
        coordinate_space: "original_image_pixels",
        target_frame_sequence: this.caseData.target_frame_sequence,
        annotation_source: parsed.annotation_source || "none",
        confidence: parsed.confidence || "uncertain",
        bbox_size_category: parsed.bbox_size_category || "uncertain",
        partial_or_occluded: parsed.partial_or_occluded === true,
        occlusion_location_status: parsed.occlusion_location_status || "not_applicable",
      };
      const legacyBox = parsed.reviewer_bbox || (
        parsed.bbox_x1 !== undefined
          ? {x1: parsed.bbox_x1, y1: parsed.bbox_y1, x2: parsed.bbox_x2, y2: parsed.bbox_y2}
          : null
      );
      if (legacyBox && this.imageSize.width && this.imageSize.height) {
        annotation.reviewer_bbox = normalizeBbox(legacyBox, this.imageSize.width, this.imageSize.height);
      }
      const number = parsed.existing_candidate_number || parsed.selected_anonymous_candidate_number;
      if (number) {
        annotation.existing_candidate_number = Number(number);
        annotation.selected_anonymous_candidate_number = Number(number);
      }
      const footpoint = parsed.footpoint || (
        parsed.footpoint_x !== undefined ? {x: parsed.footpoint_x, y: parsed.footpoint_y} : null
      );
      if (footpoint) annotation.footpoint = this.clampPoint(footpoint);
      const points = parsed.occlusion_points || (parsed.occlusion_point ? [parsed.occlusion_point] : []);
      if (Array.isArray(points) && points.length) {
        annotation.occlusion_points = points.map((point) => ({kind: "occlusion_location", ...this.clampPoint(point)}));
        annotation.occlusion_location_status = "marked";
      }
      return annotation;
    }

    render() {
      this.root.innerHTML = `
        <div class="spatialToolbar" role="toolbar" aria-label="Spatial annotation tools">
          <button type="button" data-tool="draw">Draw bbox</button>
          <button type="button" data-tool="edit">Move/resize bbox</button>
          <button type="button" data-tool="select">Select detection</button>
          <button type="button" data-tool="occlusion">Occlusion point</button>
          <button type="button" data-tool="footpoint">Footpoint</button>
          <button type="button" data-tool="pan">Pan</button>
          <button type="button" data-zoom="fit">Fit</button>
          <button type="button" data-zoom="in">Zoom in</button>
          <button type="button" data-zoom="out">Zoom out</button>
          <button type="button" data-fullscreen="true">Fullscreen</button>
          <button type="button" data-undo="true">Undo annotation</button>
          <button type="button" data-clear="true">Clear</button>
        </div>
        <div class="largeImageViewport" tabindex="0">
          <div class="imageLayer">
            <img alt="Full-resolution target frame" draggable="false">
            <svg class="annotationSvg" xmlns="http://www.w3.org/2000/svg"></svg>
          </div>
        </div>
        <div class="overlapPanel hidden"></div>
        <div class="annotationFallbackGrid"></div>
        <p class="annotationHint"></p>
      `;
      this.viewport = this.root.querySelector(".largeImageViewport");
      this.layer = this.root.querySelector(".imageLayer");
      this.image = this.root.querySelector("img");
      this.svg = this.root.querySelector("svg");
      this.overlapPanel = this.root.querySelector(".overlapPanel");
      this.fallbackGrid = this.root.querySelector(".annotationFallbackGrid");
      this.hint = this.root.querySelector(".annotationHint");
      this.image.src = this.options.imageUrl;
      this.image.addEventListener("load", () => {
        this.imageSize.width = this.image.naturalWidth || this.imageSize.width;
        this.imageSize.height = this.image.naturalHeight || this.imageSize.height;
        this.layer.style.width = `${this.imageSize.width}px`;
        this.layer.style.height = `${this.imageSize.height}px`;
        this.svg.setAttribute("viewBox", `0 0 ${this.imageSize.width} ${this.imageSize.height}`);
        this.svg.setAttribute("width", this.imageSize.width);
        this.svg.setAttribute("height", this.imageSize.height);
        this.fit();
        this.drawOverlay();
      });
      this.bindEvents();
      this.renderFallback();
      this.writeNote(false);
    }

    bindEvents() {
      this.root.querySelectorAll("[data-tool]").forEach((button) => {
        button.addEventListener("click", () => this.setMode(button.dataset.tool));
      });
      this.root.querySelector("[data-zoom='fit']").addEventListener("click", () => this.fit());
      this.root.querySelector("[data-zoom='in']").addEventListener("click", () => this.zoom(1.25));
      this.root.querySelector("[data-zoom='out']").addEventListener("click", () => this.zoom(0.8));
      this.root.querySelector("[data-fullscreen]").addEventListener("click", () => this.viewport.requestFullscreen?.());
      this.root.querySelector("[data-undo]").addEventListener("click", () => this.undo());
      this.root.querySelector("[data-clear]").addEventListener("click", () => this.clear());
      this.viewport.addEventListener("pointerdown", (event) => this.pointerDown(event));
      this.viewport.addEventListener("pointermove", (event) => this.pointerMove(event));
      this.viewport.addEventListener("pointerup", (event) => this.pointerUp(event));
      this.viewport.addEventListener("pointercancel", (event) => this.pointerUp(event));
      this.viewport.addEventListener("wheel", (event) => {
        if (!event.ctrlKey && !event.metaKey && !event.altKey) return;
        event.preventDefault();
        this.zoom(event.deltaY < 0 ? 1.1 : 0.9);
      }, {passive: false});
      this.fallbackGrid.addEventListener("input", () => this.readFallback());
    }

    setMode(mode) {
      this.mode = mode;
      this.root.querySelectorAll("[data-tool]").forEach((button) => {
        button.classList.toggle("activeTool", button.dataset.tool === mode);
      });
      this.hint.textContent = `Mode: ${mode}`;
    }

    fit() {
      if (!this.imageSize.width || !this.viewport.clientWidth) return;
      const scale = Math.min(1, (this.viewport.clientWidth - 24) / this.imageSize.width);
      this.transform.scale = Math.max(0.2, scale);
      this.transform.panX = 8;
      this.transform.panY = 8;
      this.applyTransform();
    }

    zoom(multiplier) {
      this.transform.scale = clamp(this.transform.scale * multiplier, 0.15, 8);
      this.applyTransform();
    }

    applyTransform() {
      this.layer.style.transform = `translate(${this.transform.panX}px, ${this.transform.panY}px) scale(${this.transform.scale})`;
    }

    saveHistory() {
      this.history.push(JSON.stringify(this.annotation));
      if (this.history.length > 25) this.history.shift();
    }

    undo() {
      const previous = this.history.pop();
      if (!previous) return;
      this.annotation = JSON.parse(previous);
      this.drawOverlay();
      this.renderFallback();
      this.writeNote();
    }

    clear() {
      this.saveHistory();
      this.annotation = {
        schema_version: SCHEMA_VERSION,
        case_id: this.caseData.case_id,
        coordinate_space: "original_image_pixels",
        target_frame_sequence: this.caseData.target_frame_sequence,
        annotation_source: "none",
        confidence: "uncertain",
        bbox_size_category: "uncertain",
        partial_or_occluded: false,
        occlusion_location_status: "not_applicable",
      };
      this.drawOverlay();
      this.renderFallback();
      this.writeNote();
    }

    pointerDown(event) {
      if (!this.imageSize.width) return;
      this.viewport.setPointerCapture?.(event.pointerId);
      const point = pointerPoint(event, this.viewport, this.transform);
      if (this.mode === "pan") {
        this.drag = {kind: "pan", startX: event.clientX, startY: event.clientY, panX: this.transform.panX, panY: this.transform.panY};
        return;
      }
      if (this.mode === "select") {
        this.selectCandidate(point);
        return;
      }
      if (this.mode === "occlusion") {
        this.saveHistory();
        this.annotation.occlusion_points = [{kind: "occlusion_location", ...this.clampPoint(point)}];
        this.annotation.occlusion_location_status = "marked";
        this.annotation.partial_or_occluded = true;
        this.commit();
        return;
      }
      if (this.mode === "footpoint") {
        this.saveHistory();
        this.annotation.footpoint = this.clampPoint(point);
        this.commit();
        return;
      }
      if (this.mode === "draw") {
        this.saveHistory();
        this.drag = {kind: "draw", start: point, current: point};
        this.annotation.annotation_source = "drawn_bbox";
      }
      if (this.mode === "edit") {
        this.startEdit(point);
      }
    }

    pointerMove(event) {
      if (!this.drag) return;
      const point = pointerPoint(event, this.viewport, this.transform);
      if (this.drag.kind === "pan") {
        this.transform.panX = this.drag.panX + event.clientX - this.drag.startX;
        this.transform.panY = this.drag.panY + event.clientY - this.drag.startY;
        this.applyTransform();
        return;
      }
      if (this.drag.kind === "draw") {
        this.drag.current = point;
        this.annotation.reviewer_bbox = normalizeBbox({
          x1: this.drag.start.x,
          y1: this.drag.start.y,
          x2: point.x,
          y2: point.y,
        }, this.imageSize.width, this.imageSize.height);
        this.drawOverlay();
      }
      if (this.drag.kind === "move") {
        const dx = point.x - this.drag.start.x;
        const dy = point.y - this.drag.start.y;
        const box = this.drag.original;
        this.annotation.reviewer_bbox = this.moveBox(box, dx, dy);
        this.drawOverlay();
      }
      if (this.drag.kind === "resize") {
        this.annotation.reviewer_bbox = this.resizeBox(this.drag.original, this.drag.handle, point);
        this.drawOverlay();
      }
    }

    pointerUp() {
      if (!this.drag) return;
      this.drag = null;
      this.commit();
    }

    startEdit(point) {
      const box = this.annotation.reviewer_bbox;
      if (!box) return;
      const handle = this.nearestResizeHandle(box, point);
      if (handle) {
        this.saveHistory();
        this.drag = {kind: "resize", handle, original: {...box}};
        return;
      }
      if (point.x >= box.x1 && point.x <= box.x2 && point.y >= box.y1 && point.y <= box.y2) {
        this.saveHistory();
        this.drag = {kind: "move", start: point, original: {...box}};
      }
    }

    nearestResizeHandle(box, point) {
      const tolerance = 14 / Math.max(this.transform.scale || 1, 0.01);
      const handles = {
        nw: {x: box.x1, y: box.y1},
        ne: {x: box.x2, y: box.y1},
        sw: {x: box.x1, y: box.y2},
        se: {x: box.x2, y: box.y2},
      };
      for (const [name, handle] of Object.entries(handles)) {
        if (Math.abs(point.x - handle.x) <= tolerance && Math.abs(point.y - handle.y) <= tolerance) {
          return name;
        }
      }
      return null;
    }

    moveBox(box, dx, dy) {
      const width = box.x2 - box.x1;
      const height = box.y2 - box.y1;
      const x1 = clamp(box.x1 + dx, 0, Math.max(0, this.imageSize.width - width));
      const y1 = clamp(box.y1 + dy, 0, Math.max(0, this.imageSize.height - height));
      return {x1: round(x1), y1: round(y1), x2: round(x1 + width), y2: round(y1 + height)};
    }

    resizeBox(box, handle, point) {
      const next = {...box};
      if (handle.includes("w")) next.x1 = point.x;
      if (handle.includes("e")) next.x2 = point.x;
      if (handle.includes("n")) next.y1 = point.y;
      if (handle.includes("s")) next.y2 = point.y;
      return normalizeBbox(next, this.imageSize.width, this.imageSize.height);
    }

    clampPoint(point) {
      return {
        x: round(clamp(point.x, 0, this.imageSize.width || Number.MAX_SAFE_INTEGER)),
        y: round(clamp(point.y, 0, this.imageSize.height || Number.MAX_SAFE_INTEGER)),
      };
    }

    selectCandidate(point) {
      const hits = hitCandidates(this.candidates, point, this.transform.scale);
      this.overlapPanel.classList.toggle("hidden", hits.length <= 1);
      if (!hits.length) return;
      if (hits.length > 1) {
        this.overlapPanel.innerHTML = `<strong>Overlapping detections</strong>${hits.map((candidate) => `
          <button type="button" data-overlap-candidate="${candidate.anonymous_candidate_number}">
            #${candidate.anonymous_candidate_number}
          </button>`).join("")}`;
        this.overlapPanel.querySelectorAll("[data-overlap-candidate]").forEach((button) => {
          button.addEventListener("click", () => {
            const selected = hits.find((item) => Number(item.anonymous_candidate_number) === Number(button.dataset.overlapCandidate));
            if (selected) this.applyCandidate(selected, hits);
          });
        });
      }
      this.applyCandidate(hits[0], hits);
    }

    applyCandidate(candidate, hits) {
      this.saveHistory();
      this.annotation.annotation_source = "existing_detection";
      this.annotation.existing_candidate_number = Number(candidate.anonymous_candidate_number);
      this.annotation.selected_anonymous_candidate_number = Number(candidate.anonymous_candidate_number);
      this.annotation.reviewer_bbox = normalizeBbox(candidate.bbox, this.imageSize.width, this.imageSize.height);
      this.annotation.overlapping_candidate_numbers = hits.map((item) => Number(item.anonymous_candidate_number));
      this.commit();
    }

    commit() {
      this.drawOverlay();
      this.renderFallback();
      this.writeNote();
    }

    writeNote(trigger = true) {
      if (this.annotation.reviewer_bbox) {
        this.annotation.bbox_x1 = this.annotation.reviewer_bbox.x1;
        this.annotation.bbox_y1 = this.annotation.reviewer_bbox.y1;
        this.annotation.bbox_x2 = this.annotation.reviewer_bbox.x2;
        this.annotation.bbox_y2 = this.annotation.reviewer_bbox.y2;
      }
      if (this.annotation.footpoint) {
        this.annotation.footpoint_x = this.annotation.footpoint.x;
        this.annotation.footpoint_y = this.annotation.footpoint.y;
      }
      if (this.noteElement) {
        this.noteElement.value = annotationToNote(this.annotation);
        if (trigger) this.noteElement.dispatchEvent(new Event("input", {bubbles: true}));
      }
      if (this.onChange) this.onChange(this.annotation);
    }

    renderFallback() {
      const box = this.annotation.reviewer_bbox || {};
      const foot = this.annotation.footpoint || {};
      const occ = (this.annotation.occlusion_points || [])[0] || {};
      this.fallbackGrid.innerHTML = `
        <label>bbox x1<input data-field="bbox_x1" type="number" step="0.1" value="${box.x1 ?? ""}"></label>
        <label>bbox y1<input data-field="bbox_y1" type="number" step="0.1" value="${box.y1 ?? ""}"></label>
        <label>bbox x2<input data-field="bbox_x2" type="number" step="0.1" value="${box.x2 ?? ""}"></label>
        <label>bbox y2<input data-field="bbox_y2" type="number" step="0.1" value="${box.y2 ?? ""}"></label>
        <label>candidate #<input data-field="existing_candidate_number" type="number" step="1" min="1" value="${this.annotation.existing_candidate_number ?? ""}"></label>
        <label>foot x<input data-field="footpoint_x" type="number" step="0.1" value="${foot.x ?? ""}"></label>
        <label>foot y<input data-field="footpoint_y" type="number" step="0.1" value="${foot.y ?? ""}"></label>
        <label>occlusion x<input data-field="occlusion_x" type="number" step="0.1" value="${occ.x ?? ""}"></label>
        <label>occlusion y<input data-field="occlusion_y" type="number" step="0.1" value="${occ.y ?? ""}"></label>
        <label>confidence<select data-field="confidence">${["high", "medium", "low", "uncertain"].map((value) => `<option value="${value}"${this.annotation.confidence === value ? " selected" : ""}>${value}</option>`).join("")}</select></label>
        <label>bbox size<select data-field="bbox_size_category">${["small", "medium", "large", "partial", "uncertain"].map((value) => `<option value="${value}"${this.annotation.bbox_size_category === value ? " selected" : ""}>${value}</option>`).join("")}</select></label>
        <label>partial/occluded<select data-field="partial_or_occluded">${["false", "true"].map((value) => `<option value="${value}"${String(this.annotation.partial_or_occluded) === value ? " selected" : ""}>${value}</option>`).join("")}</select></label>
      `;
    }

    readFallback() {
      const values = {};
      this.fallbackGrid.querySelectorAll("[data-field]").forEach((input) => {
        values[input.dataset.field] = input.value;
      });
      this.saveHistory();
      if (values.bbox_x1 !== "" && values.bbox_y1 !== "" && values.bbox_x2 !== "" && values.bbox_y2 !== "") {
        this.annotation.reviewer_bbox = normalizeBbox({
          x1: values.bbox_x1,
          y1: values.bbox_y1,
          x2: values.bbox_x2,
          y2: values.bbox_y2,
        }, this.imageSize.width, this.imageSize.height);
        this.annotation.annotation_source = "drawn_bbox";
      }
      if (values.existing_candidate_number) {
        this.annotation.existing_candidate_number = Number(values.existing_candidate_number);
        this.annotation.selected_anonymous_candidate_number = Number(values.existing_candidate_number);
      }
      if (values.footpoint_x !== "" && values.footpoint_y !== "") {
        this.annotation.footpoint = this.clampPoint({x: values.footpoint_x, y: values.footpoint_y});
      }
      if (values.occlusion_x !== "" && values.occlusion_y !== "") {
        this.annotation.occlusion_points = [{kind: "occlusion_location", ...this.clampPoint({x: values.occlusion_x, y: values.occlusion_y})}];
        this.annotation.occlusion_location_status = "marked";
      }
      this.annotation.confidence = values.confidence || "uncertain";
      this.annotation.bbox_size_category = values.bbox_size_category || "uncertain";
      this.annotation.partial_or_occluded = values.partial_or_occluded === "true";
      this.drawOverlay();
      this.writeNote();
    }

    drawOverlay() {
      if (!this.svg) return;
      const candidateMarkup = this.candidates.map((candidate) => {
        const box = candidate.bbox || {};
        const selected = Number(candidate.anonymous_candidate_number) === Number(this.annotation.existing_candidate_number);
        return `<g class="candidateBox${selected ? " selectedCandidateBox" : ""}">
          <rect x="${box.x1}" y="${box.y1}" width="${box.x2 - box.x1}" height="${box.y2 - box.y1}"></rect>
          <text x="${box.x1}" y="${Math.max(12, box.y1 - 3)}">#${candidate.anonymous_candidate_number}</text>
        </g>`;
      }).join("");
      const box = this.annotation.reviewer_bbox;
      const reviewerBox = box ? `<g class="reviewerBox">
        <rect x="${box.x1}" y="${box.y1}" width="${box.x2 - box.x1}" height="${box.y2 - box.y1}"></rect>
        <circle cx="${box.x1}" cy="${box.y1}" r="5"></circle>
        <circle cx="${box.x2}" cy="${box.y1}" r="5"></circle>
        <circle cx="${box.x1}" cy="${box.y2}" r="5"></circle>
        <circle cx="${box.x2}" cy="${box.y2}" r="5"></circle>
      </g>` : "";
      const foot = this.annotation.footpoint
        ? `<g class="footpointMarker"><circle cx="${this.annotation.footpoint.x}" cy="${this.annotation.footpoint.y}" r="8"></circle><text x="${this.annotation.footpoint.x + 10}" y="${this.annotation.footpoint.y}">foot</text></g>`
        : "";
      const occlusion = (this.annotation.occlusion_points || []).map((point) => `
        <g class="occlusionMarker"><path d="M ${point.x - 10} ${point.y} L ${point.x + 10} ${point.y} M ${point.x} ${point.y - 10} L ${point.x} ${point.y + 10}"></path><circle cx="${point.x}" cy="${point.y}" r="6"></circle></g>
      `).join("");
      this.svg.innerHTML = `${candidateMarkup}${reviewerBox}${foot}${occlusion}`;
    }
  }

  function validateDecision(decision, noteText) {
    const annotation = parseNote(noteText, "");
    return validationForDecision(decision, annotation);
  }

  window.ReviewAnnotationCanvas = {
    SpatialAnnotationCanvas,
    normalizeBbox,
    hitCandidates,
    validateDecision,
    annotationToNote,
  };
}());
