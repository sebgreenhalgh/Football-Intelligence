(function () {
  "use strict";

  const MODULES = {
    detection_gold_player_static: {label: "Player static", eyebrow: "PLAYER INSTANCE"},
    detection_gold_dense_region: {label: "Dense region", eyebrow: "VISIBLE MASK"},
    detection_gold_temporal_player: {label: "Temporal player", eyebrow: "TEMPORAL VISIBILITY"},
    detection_gold_pitch_boundary: {label: "Pitch / boundary", eyebrow: "PITCH AND ROLE"},
    detection_gold_football_burst: {label: "Football burst", eyebrow: "FOOTBALL VISIBILITY"},
  };
  const LAYERS = ["RAW", "CONFIDENCE", "PRE_NMS", "POST_NMS", "FUSED"];
  const SVG_NS = "http://www.w3.org/2000/svg";
  const runtime = {
    manifest: null,
    uiConfig: null,
    state: null,
    api: null,
    activeIndex: 0,
    frameIndex: 0,
    view: "focal",
    tool: "select",
    selectedCandidate: null,
    selectedObjectByCase: {},
    redrawVisibleObjectUuid: null,
    drafts: {},
    geometryDrafts: {},
    history: [],
    maskPoints: [],
    drawingStart: null,
    panStart: null,
    viewTransform: {scale: 1, x: 0, y: 0},
    playing: false,
    playTimer: null,
    evidenceBlocked: false,
    evidenceToken: 0,
    verifiedAssets: new Set(),
    db: null,
    outbox: [],
    flushing: false,
    serverStateHash: "",
    serverSequence: 0,
    activeSeconds: 0,
    lastActiveTick: Date.now(),
  };

  const byId = (id) => document.getElementById(id);
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const currentCase = () => runtime.manifest.cases[runtime.activeIndex];
  const records = () => currentCase()?.visible_metadata?.frame_records || [];
  const currentRecord = () => records()[runtime.frameIndex] || records()[0];
  const moduleInfo = () => MODULES[currentCase()?.task_type] || {label: "Unknown", eyebrow: "REVIEW"};
  const evidenceUrl = (relativePath) =>
    `/evidence/${encodeURIComponent(currentCase().case_id)}/${relativePath.split("/").map(encodeURIComponent).join("/")}`;

  function uid(prefix) {
    return `${prefix}-${Date.now().toString(36)}-${crypto.randomUUID()}`;
  }

  function openDatabase() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(`fi_detection_gold_${runtime.manifest.review_id}`, 1);
      request.onupgradeneeded = () => {
        const database = request.result;
        if (!database.objectStoreNames.contains("outbox")) database.createObjectStore("outbox", {keyPath: "client_event_id"});
        if (!database.objectStoreNames.contains("drafts")) database.createObjectStore("drafts", {keyPath: "case_id"});
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  function dbRequest(storeName, mode, operation) {
    if (!runtime.db) return Promise.resolve(undefined);
    return new Promise((resolve, reject) => {
      const transaction = runtime.db.transaction(storeName, mode);
      const store = transaction.objectStore(storeName);
      const request = operation(store);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  const dbPut = (store, value) => dbRequest(store, "readwrite", (target) => target.put(value));
  const dbDelete = (store, key) => dbRequest(store, "readwrite", (target) => target.delete(key));
  const dbAll = (store) => dbRequest(store, "readonly", (target) => target.getAll());

  function sourceBinding(caseData) {
    return clone(caseData.visible_metadata.source_binding);
  }

  function annotationObjects(annotation = draft()) {
    return annotation.player_instances || annotation.visible_masks || [];
  }

  function objectKind(annotation = draft()) {
    return annotation.player_instances ? "Person" : "Mask";
  }

  function selectedObject(annotation = draft()) {
    const objects = annotationObjects(annotation);
    const caseId = currentCase().case_id;
    const selectedUuid = runtime.selectedObjectByCase[caseId];
    const selected = objects.find((item) => item.annotation_uuid === selectedUuid);
    if (selected) return selected;
    if (!objects.length) {
      delete runtime.selectedObjectByCase[caseId];
      return null;
    }
    runtime.selectedObjectByCase[caseId] = objects[0].annotation_uuid;
    return objects[0];
  }

  function selectObject(annotationUuid) {
    const exists = annotationObjects().some((item) => item.annotation_uuid === annotationUuid);
    if (!exists) return;
    runtime.selectedObjectByCase[currentCase().case_id] = annotationUuid;
    runtime.redrawVisibleObjectUuid = null;
    persistDraft();
    renderAnnotationForm();
    renderOverlay();
  }

  function objectLabel(annotationUuid, annotation = draft()) {
    const index = annotationObjects(annotation).findIndex((item) => item.annotation_uuid === annotationUuid);
    return index >= 0 ? `${objectKind(annotation)} ${index + 1}` : "Removed target";
  }

  function focalScopeApplies() {
    return ["detection_gold_player_static", "detection_gold_dense_region"].includes(currentCase()?.task_type);
  }

  function requireFocalAnnotationScope() {
    if (!focalScopeApplies() || runtime.view === "focal") return true;
    showError("Static and dense annotations apply to the focal ROI only. Switch to Focal before drawing or editing.");
    return false;
  }

  function defaultAnnotation(caseData) {
    const binding = sourceBinding(caseData);
    const frameRows = caseData.visible_metadata.frame_records || [];
    if (caseData.task_type === "detection_gold_player_static") {
      return {
        schema_version: "m5_5g1a_detection_gold_v1",
        source_binding: binding,
        visible_person_count: 0,
        player_instances: [],
        candidate_relations: [],
        earliest_failure_stage: "UNRESOLVED",
        note: "",
      };
    }
    if (caseData.task_type === "detection_gold_dense_region") {
      return {
        schema_version: "m5_5g1a_detection_gold_v1",
        source_binding: binding,
        dense_region_uuid: uid("dense"),
        trigger_reason: caseData.visible_metadata.pilot_stratum,
        human_visible_person_count: 0,
        visible_masks: [],
        candidate_relations: [],
        uncertain_or_ignore: false,
        note: "",
      };
    }
    if (caseData.task_type === "detection_gold_temporal_player") {
      return {
        schema_version: "m5_5g1a_detection_gold_v1",
        source_binding: binding,
        frames: frameRows.map((row) => ({
          frame_sequence: row.frame_sequence,
          source_frame_sha256: row.source_frame_sha256,
          state: "UNRESOLVED",
          current_frame_pixel_support: false,
          candidate_uuids: [],
        })),
        contact_strip_reviewed: false,
        stable_run_accepted: false,
        note: "",
      };
    }
    if (caseData.task_type === "detection_gold_pitch_boundary") {
      return {
        schema_version: "m5_5g1a_detection_gold_v1",
        source_binding: binding,
        footpoint: clone(caseData.visible_metadata.machine_footpoint || {x: 0, y: 0}),
        footpoint_uncertainty_pixels: 5,
        pitch_state: "BOUNDARY_UNCERTAIN",
        coarse_role: "UNKNOWN",
        primary_on_pitch_supply_eligible: false,
        note: "",
      };
    }
    return {
      schema_version: "m5_5g1a_detection_gold_v1",
      source_binding: binding,
      frames: frameRows.map((row) => ({
        frame_sequence: row.frame_sequence,
        source_frame_sha256: row.source_frame_sha256,
        state: "UNRESOLVED",
      })),
      full_contact_strip_reviewed: false,
      note: "",
    };
  }

  function draft(caseData = currentCase()) {
    if (!runtime.drafts[caseData.case_id]) {
      runtime.drafts[caseData.case_id] = clone(
        runtime.state.annotations?.[caseData.case_id] || defaultAnnotation(caseData)
      );
    }
    return runtime.drafts[caseData.case_id];
  }

  function caseGeometryDrafts(caseData = currentCase()) {
    if (!runtime.geometryDrafts[caseData.case_id]) runtime.geometryDrafts[caseData.case_id] = {};
    return runtime.geometryDrafts[caseData.case_id];
  }

  function currentGeometryDraft() {
    return caseGeometryDrafts()[String(currentRecord()?.frame_sequence)] || null;
  }

  async function persistDraft() {
    const caseData = currentCase();
    await dbPut("drafts", {
      case_id: caseData.case_id,
      annotation: draft(),
      geometry_drafts: clone(runtime.geometryDrafts[caseData.case_id] || {}),
      selected_annotation_uuid: runtime.selectedObjectByCase[caseData.case_id] || null,
      updated_at: new Date().toISOString(),
    });
    setSaveState("Draft stored locally", false);
  }

  function pushHistory() {
    runtime.history.push({
      case_id: currentCase().case_id,
      annotation: clone(draft()),
      selected_annotation_uuid: runtime.selectedObjectByCase[currentCase().case_id] || null,
    });
    if (runtime.history.length > 60) runtime.history.shift();
  }

  function undo() {
    const previous = runtime.history.pop();
    if (!previous || previous.case_id !== currentCase().case_id) return;
    runtime.drafts[previous.case_id] = previous.annotation;
    if (previous.selected_annotation_uuid) {
      runtime.selectedObjectByCase[previous.case_id] = previous.selected_annotation_uuid;
    } else {
      delete runtime.selectedObjectByCase[previous.case_id];
    }
    runtime.redrawVisibleObjectUuid = null;
    persistDraft();
    renderAnnotationForm();
    renderOverlay();
  }

  function setSaveState(message, error = false) {
    byId("dgSaveState").textContent = message;
    byId("dgSaveState").classList.toggle("isError", error);
  }

  function updateServerState() {
    byId("dgServerState").textContent = `server ${runtime.serverSequence} | pending ${runtime.outbox.length}`;
  }

  function activeSeconds() {
    const now = Date.now();
    if (document.visibilityState === "visible") {
      runtime.activeSeconds += Math.max(0, Math.round((now - runtime.lastActiveTick) / 1000));
    }
    runtime.lastActiveTick = now;
    return runtime.activeSeconds;
  }

  function getBounds() {
    const row = currentRecord();
    const width = Number(row.image_width || 2730);
    const height = Number(row.image_height || 720);
    if (runtime.view === "focal" && row.focal_bounds) return clone(row.focal_bounds);
    return {x1: 0, y1: 0, x2: width, y2: height};
  }

  function applyViewTransform() {
    const transform = runtime.viewTransform;
    byId("dgStage").style.transform = `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`;
  }

  function clampViewTransform() {
    const viewport = byId("dgViewport").getBoundingClientRect();
    const scale = runtime.viewTransform.scale;
    runtime.viewTransform.x = Math.min(0, Math.max(viewport.width * (1 - scale), runtime.viewTransform.x));
    runtime.viewTransform.y = Math.min(0, Math.max(viewport.height * (1 - scale), runtime.viewTransform.y));
    applyViewTransform();
  }

  function fitEvidence() {
    runtime.viewTransform = {scale: 1, x: 0, y: 0};
    applyViewTransform();
  }

  function zoomAt(clientX, clientY, requestedScale) {
    const viewport = byId("dgViewport").getBoundingClientRect();
    const before = runtime.viewTransform;
    const scale = Math.max(1, Math.min(8, requestedScale));
    const x = clientX - viewport.left;
    const y = clientY - viewport.top;
    runtime.viewTransform = {
      scale,
      x: x - ((x - before.x) * scale) / before.scale,
      y: y - ((y - before.y) * scale) / before.scale,
    };
    clampViewTransform();
  }

  function toViewBox(box) {
    const bounds = getBounds();
    return {
      x1: box.x1 - bounds.x1,
      y1: box.y1 - bounds.y1,
      x2: box.x2 - bounds.x1,
      y2: box.y2 - bounds.y1,
    };
  }

  function pointToView(point) {
    const bounds = getBounds();
    return {x: point.x - bounds.x1, y: point.y - bounds.y1};
  }

  function pointerOriginal(event) {
    const svg = byId("dgOverlay");
    const rectangle = svg.getBoundingClientRect();
    const bounds = getBounds();
    const sourceWidth = bounds.x2 - bounds.x1;
    const sourceHeight = bounds.y2 - bounds.y1;
    const sourceRatio = sourceWidth / sourceHeight;
    const renderedRatio = rectangle.width / rectangle.height;
    const contentWidth = sourceRatio > renderedRatio ? rectangle.width : rectangle.height * sourceRatio;
    const contentHeight = sourceRatio > renderedRatio ? rectangle.width / sourceRatio : rectangle.height;
    const contentLeft = rectangle.left + (rectangle.width - contentWidth) / 2;
    const contentTop = rectangle.top + (rectangle.height - contentHeight) / 2;
    return {
      x: bounds.x1 + Math.max(0, Math.min(1, (event.clientX - contentLeft) / contentWidth)) * sourceWidth,
      y: bounds.y1 + Math.max(0, Math.min(1, (event.clientY - contentTop) / contentHeight)) * sourceHeight,
    };
  }

  function makeSvg(tag, attributes = {}) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
    return node;
  }

  function renderOverlay() {
    const svg = byId("dgOverlay");
    svg.replaceChildren();
    const bounds = getBounds();
    svg.setAttribute("viewBox", `0 0 ${bounds.x2 - bounds.x1} ${bounds.y2 - bounds.y1}`);
    const enabled = new Set(
      [...document.querySelectorAll("[data-dg-layer]:checked")].map((element) => element.dataset.dgLayer)
    );
    const row = currentRecord();
    for (const candidate of row?.candidates || []) {
      if (!enabled.has(candidate.stage)) continue;
      const box = toViewBox(candidate.bbox_original_pixels);
      const rectangle = makeSvg("rect", {
        x: box.x1,
        y: box.y1,
        width: box.x2 - box.x1,
        height: box.y2 - box.y1,
        class: `dgProposal dgLayer${candidate.stage} ${runtime.selectedCandidate?.diagnostic_uuid === candidate.diagnostic_uuid ? "selected" : ""}`,
        tabindex: 0,
      });
      rectangle.addEventListener("click", (event) => {
        if (runtime.tool !== "select") return;
        event.stopPropagation();
        runtime.selectedCandidate = candidate;
        renderOverlay();
        renderProvenance();
        renderAnnotationForm();
      });
      svg.appendChild(rectangle);
    }
    const annotation = draft();
    const selectedUuid = selectedObject(annotation)?.annotation_uuid;
    const appendSelectableBox = (rawBox, item, geometryClass = "") => {
      if (!rawBox) return;
      const box = toViewBox(rawBox);
      const rectangle = makeSvg("rect", {
        x: box.x1,
        y: box.y1,
        width: box.x2 - box.x1,
        height: box.y2 - box.y1,
        class: `dgHumanBox ${geometryClass} ${selectedUuid === item.annotation_uuid ? "selectedObject" : ""}`,
        "data-dg-object-uuid": item.annotation_uuid,
        tabindex: 0,
      });
      rectangle.addEventListener("click", (event) => {
        if (runtime.tool !== "select") return;
        event.stopPropagation();
        selectObject(item.annotation_uuid);
      });
      svg.appendChild(rectangle);
    };
    for (const item of annotation.player_instances || []) {
      appendSelectableBox(item.visible_body_box, item, "dgVisibleBodyBox");
      appendSelectableBox(item.full_body_box, item, "dgSupplementaryBox");
      appendSelectableBox(item.optional_head_box, item, "dgHeadBox");
      const point = pointToView(item.footpoint);
      svg.appendChild(makeSvg("circle", {
        cx: point.x,
        cy: point.y,
        r: 4,
        class: `dgFootpoint ${selectedUuid === item.annotation_uuid ? "selectedObject" : ""}`,
      }));
    }
    for (const item of annotation.visible_masks || []) {
      appendSelectableBox(item.visible_body_box, item, "dgVisibleBodyBox");
      appendSelectableBox(item.full_body_box, item, "dgSupplementaryBox");
      appendSelectableBox(item.optional_head_box, item, "dgHeadBox");
      const polygon = item.polygon_original_pixels.map(pointToView).map((point) => `${point.x},${point.y}`).join(" ");
      const node = makeSvg("polygon", {
        points: polygon,
        class: `dgHumanMask ${selectedUuid === item.annotation_uuid ? "selectedObject" : ""}`,
        "data-dg-object-uuid": item.annotation_uuid,
        tabindex: 0,
      });
      node.addEventListener("click", (event) => {
        if (runtime.tool !== "select") return;
        event.stopPropagation();
        selectObject(item.annotation_uuid);
      });
      svg.appendChild(node);
    }
    if (annotation.frames) {
      const frame = annotation.frames[runtime.frameIndex];
      if (frame?.visible_body_box) {
        const box = toViewBox(frame.visible_body_box);
        svg.appendChild(makeSvg("rect", {
          x: box.x1,
          y: box.y1,
          width: box.x2 - box.x1,
          height: box.y2 - box.y1,
          class: "dgHumanBox dgTemporalObservation",
        }));
      }
      if (frame?.footpoint) {
        const point = pointToView(frame.footpoint);
        svg.appendChild(makeSvg("circle", {cx: point.x, cy: point.y, r: 4, class: "dgFootpoint"}));
      }
      if (frame?.centre_point) {
        const point = pointToView(frame.centre_point);
        svg.appendChild(makeSvg("circle", {cx: point.x, cy: point.y, r: 5, class: "dgBallPoint"}));
      }
      if (frame?.apparent_ellipse) {
        const centre = pointToView({x: frame.apparent_ellipse.centre_x, y: frame.apparent_ellipse.centre_y});
        svg.appendChild(makeSvg("ellipse", {
          cx: centre.x,
          cy: centre.y,
          rx: frame.apparent_ellipse.radius_x,
          ry: frame.apparent_ellipse.radius_y,
          class: "dgBallEllipse",
        }));
      }
      if (frame?.blur_trail_endpoints?.length === 2) {
        const start = pointToView(frame.blur_trail_endpoints[0]);
        const end = pointToView(frame.blur_trail_endpoints[1]);
        svg.appendChild(makeSvg("line", {
          x1: start.x,
          y1: start.y,
          x2: end.x,
          y2: end.y,
          class: "dgBallTrail",
        }));
      }
    }
    if (annotation.footpoint) {
      const point = pointToView(annotation.footpoint);
      svg.appendChild(makeSvg("circle", {cx: point.x, cy: point.y, r: 4, class: "dgFootpoint"}));
    }
    if (runtime.maskPoints.length) {
      const polygon = runtime.maskPoints.map(pointToView).map((point) => `${point.x},${point.y}`).join(" ");
      svg.appendChild(makeSvg("polyline", {points: polygon, class: "dgMaskDraft"}));
    }
    const geometryDraft = currentGeometryDraft();
    if (geometryDraft?.visible_body_box) {
      const box = toViewBox(geometryDraft.visible_body_box);
      svg.appendChild(makeSvg("rect", {
        x: box.x1,
        y: box.y1,
        width: box.x2 - box.x1,
        height: box.y2 - box.y1,
        class: "dgGeometryDraft",
      }));
    }
    const pitchPolygon = currentCase().visible_metadata.pitch_polygon_vertices || [];
    if (currentCase().task_type === "detection_gold_pitch_boundary" && pitchPolygon.length) {
      const polygon = pitchPolygon.map(pointToView).map((point) => `${point.x},${point.y}`).join(" ");
      svg.insertBefore(makeSvg("polygon", {points: polygon, class: "dgPitchPolygon"}), svg.firstChild);
    }
    if (focalScopeApplies() && runtime.view === "panorama" && row.focal_bounds) {
      const focal = toViewBox(row.focal_bounds);
      svg.appendChild(makeSvg("rect", {
        x: focal.x1,
        y: focal.y1,
        width: focal.x2 - focal.x1,
        height: focal.y2 - focal.y1,
        class: "dgFocalScopeRoi",
      }));
      const label = makeSvg("text", {x: focal.x1 + 8, y: focal.y1 + 18, class: "dgFocalScopeLabel"});
      label.textContent = "ANNOTATION ROI";
      svg.appendChild(label);
    }
  }

  function candidateInstance(candidate, boxOverride = null) {
    const box = clone(boxOverride || candidate.bbox_original_pixels);
    return {
      annotation_uuid: uid("person"),
      visible_body_box: box,
      footpoint: {x: (box.x1 + box.x2) / 2, y: box.y2},
      footpoint_uncertainty_pixels: 8,
      visibility_state: "UNRESOLVED",
      occlusion_fraction: 0,
      occlusion_type: "UNKNOWN",
      truncation_flags: [],
      minimum_visible_dimensions: {width_pixels: box.x2 - box.x1, height_pixels: box.y2 - box.y1},
      ambiguity_ignore: false,
      pitch_state: "BOUNDARY_UNCERTAIN",
      coarse_role: "UNKNOWN",
    };
  }

  function acceptSelectedPerson(boxOverride = null) {
    if (!requireFocalAnnotationScope()) return;
    const annotation = draft();
    const selectedMachineCandidate = boxOverride ? null : runtime.selectedCandidate;
    const candidate = selectedMachineCandidate || {bbox_original_pixels: boxOverride};
    if (!candidate.bbox_original_pixels && !boxOverride) throw new Error("Select or draw a visible person first");
    if (selectedMachineCandidate && selectedMachineCandidate.class_name !== "person") {
      throw new Error("Player annotations may use only person proposals");
    }
    pushHistory();
    const instance = candidateInstance(candidate, boxOverride);
    if (currentCase().task_type === "detection_gold_player_static") {
      annotation.player_instances.push(instance);
      annotation.visible_person_count = annotation.player_instances.length;
      runtime.selectedObjectByCase[currentCase().case_id] = instance.annotation_uuid;
    } else if (currentCase().task_type === "detection_gold_temporal_player") {
      annotation.stable_run_accepted = false;
      annotation.frames[runtime.frameIndex] = {
        frame_sequence: currentRecord().frame_sequence,
        source_frame_sha256: currentRecord().source_frame_sha256,
        state: "OBSERVED",
        visible_body_box: instance.visible_body_box,
        footpoint: instance.footpoint,
        current_frame_pixel_support: true,
        candidate_uuids: selectedMachineCandidate ? [selectedMachineCandidate.diagnostic_uuid] : [],
      };
    }
    persistDraft();
    renderAnnotationForm();
    renderOverlay();
  }

  function finishMask() {
    if (!requireFocalAnnotationScope()) return;
    if (runtime.maskPoints.length < 3) throw new Error("A visible mask needs at least three points");
    const xs = runtime.maskPoints.map((point) => point.x);
    const ys = runtime.maskPoints.map((point) => point.y);
    const box = {x1: Math.min(...xs), y1: Math.min(...ys), x2: Math.max(...xs), y2: Math.max(...ys)};
    const annotation = draft();
    pushHistory();
    const maskId = uid("mask");
    annotation.visible_masks.push({
      annotation_uuid: maskId,
      polygon_original_pixels: clone(runtime.maskPoints),
      mask_quality: "UNCERTAIN",
      visible_body_box: box,
      occlusion_order: annotation.visible_masks.length,
      pairwise_overlap_annotation_uuids: [],
      truncation_flags: [],
      current_frame_pixel_support: true,
    });
    annotation.human_visible_person_count = annotation.visible_masks.length;
    runtime.selectedObjectByCase[currentCase().case_id] = maskId;
    runtime.maskPoints = [];
    persistDraft();
    renderAnnotationForm();
    renderOverlay();
  }

  function handleOverlayClick(event) {
    if (runtime.tool === "select") return;
    if (!requireFocalAnnotationScope()) return;
    const point = pointerOriginal(event);
    const annotation = draft();
    if (runtime.tool === "footpoint") {
      pushHistory();
      if (currentCase().task_type === "detection_gold_pitch_boundary") annotation.footpoint = point;
      else {
        const person = selectedObject(annotation);
        if (!person || !annotation.player_instances) return showError("Select a person before placing its footpoint.");
        person.footpoint = point;
      }
      persistDraft();
    } else if (runtime.tool === "mask") {
      runtime.maskPoints.push(point);
      renderAnnotationForm();
    } else if (runtime.tool === "ball" && currentCase().task_type === "detection_gold_football_burst") {
      pushHistory();
      annotation.frames[runtime.frameIndex] = {
        frame_sequence: currentRecord().frame_sequence,
        source_frame_sha256: currentRecord().source_frame_sha256,
        state: "VISIBLE_CLEAR",
        centre_point: point,
        geometry_uncertainty_pixels: 3,
      };
      persistDraft();
      renderAnnotationForm();
    }
    renderOverlay();
  }

  function handlePointerDown(event) {
    if (runtime.tool === "pan" && event.button === 0) {
      runtime.panStart = {
        clientX: event.clientX,
        clientY: event.clientY,
        x: runtime.viewTransform.x,
        y: runtime.viewTransform.y,
      };
      event.currentTarget.setPointerCapture(event.pointerId);
      event.preventDefault();
      return;
    }
    if (!["box", "fullbox", "headbox", "ellipse", "trail"].includes(runtime.tool) || event.button !== 0) return;
    if (!requireFocalAnnotationScope()) return;
    runtime.drawingStart = pointerOriginal(event);
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  function handlePointerMove(event) {
    if (!runtime.panStart) return;
    runtime.viewTransform.x = runtime.panStart.x + event.clientX - runtime.panStart.clientX;
    runtime.viewTransform.y = runtime.panStart.y + event.clientY - runtime.panStart.clientY;
    clampViewTransform();
  }

  function handlePointerUp(event) {
    if (runtime.panStart) {
      runtime.panStart = null;
      event.currentTarget.releasePointerCapture(event.pointerId);
      return;
    }
    if (!runtime.drawingStart || !["box", "fullbox", "headbox", "ellipse", "trail"].includes(runtime.tool)) return;
    const end = pointerOriginal(event);
    const start = runtime.drawingStart;
    const box = {
      x1: Math.min(start.x, end.x),
      y1: Math.min(start.y, end.y),
      x2: Math.max(start.x, end.x),
      y2: Math.max(start.y, end.y),
    };
    runtime.drawingStart = null;
    event.currentTarget.releasePointerCapture(event.pointerId);
    if (box.x2 - box.x1 < 2 || box.y2 - box.y1 < 2) return;
    const annotation = draft();
    if (runtime.tool === "box") {
      if (runtime.redrawVisibleObjectUuid) {
        const target = annotationObjects(annotation).find(
          (item) => item.annotation_uuid === runtime.redrawVisibleObjectUuid
        );
        if (!target) return showError("The selected object no longer exists. Select it again before redrawing.");
        pushHistory();
        target.visible_body_box = box;
        if (target.minimum_visible_dimensions) {
          target.minimum_visible_dimensions = {
            width_pixels: box.x2 - box.x1,
            height_pixels: box.y2 - box.y1,
          };
        }
        runtime.redrawVisibleObjectUuid = null;
        persistDraft();
        renderAnnotationForm();
        renderOverlay();
        return;
      }
      acceptSelectedPerson(box);
      return;
    }
    pushHistory();
    if (["fullbox", "headbox"].includes(runtime.tool)) {
      const target = selectedObject(annotation);
      if (!target) {
        showError("Select a person or visible mask before adding full-body or head geometry.");
        return;
      }
      target[runtime.tool === "fullbox" ? "full_body_box" : "optional_head_box"] = box;
    } else if (currentCase().task_type === "detection_gold_football_burst") {
      const frame = annotation.frames[runtime.frameIndex];
      const centre = {x: (box.x1 + box.x2) / 2, y: (box.y1 + box.y2) / 2};
      if (runtime.tool === "ellipse") {
        frame.state = frame.state === "VISIBLE_BLURRED" ? frame.state : "VISIBLE_CLEAR";
        frame.centre_point = centre;
        frame.apparent_ellipse = {
          centre_x: centre.x,
          centre_y: centre.y,
          radius_x: (box.x2 - box.x1) / 2,
          radius_y: (box.y2 - box.y1) / 2,
          rotation_degrees: 0,
        };
        frame.geometry_uncertainty_pixels = 3;
      } else {
        frame.state = "VISIBLE_BLURRED";
        frame.centre_point = centre;
        frame.blur_trail_endpoints = [start, end];
        frame.blur_trail_width = Math.max(2, Math.min(box.x2 - box.x1, box.y2 - box.y1));
        frame.geometry_uncertainty_pixels = 4;
      }
    }
    persistDraft();
    renderAnnotationForm();
    renderOverlay();
  }

  function selectMarkup(id, values, selected) {
    return `<select id="${id}">${values.map((value) => `<option value="${value}" ${value === selected ? "selected" : ""}>${value.replaceAll("_", " ")}</option>`).join("")}</select>`;
  }

  function baseNote(annotation) {
    return `<label class="dgField">Optional note<textarea id="dgNote" rows="3" maxlength="1000">${annotation.note || ""}</textarea></label>`;
  }

  function candidateCoverage(annotation) {
    const required = new Set(currentCase().visible_metadata.candidate_uuids || []);
    const covered = new Set((annotation.candidate_relations || []).map((row) => row.candidate_uuid));
    return {covered: [...covered].filter((value) => required.has(value)).length, total: required.size};
  }

  function candidateBindingControls(annotation, {dense = false} = {}) {
    const coverage = candidateCoverage(annotation);
    const selectedRelation = (annotation.candidate_relations || []).find(
      (row) => row.candidate_uuid === runtime.selectedCandidate?.diagnostic_uuid
    );
    const targets = annotationObjects(annotation);
    const selectedTargets = new Set(selectedRelation?.annotation_uuids || []);
    const relationOptions = [
      "BACKGROUND",
      "CLEAN_SINGLE_INSTANCE",
      "DUPLICATE_OF_INSTANCE",
      "MERGED_MULTIPLE_INSTANCES",
      "PARTIAL_INSTANCE",
      "AMBIGUOUS",
    ];
    const relationMarkup = `<select id="dgCandidateRelation" ${runtime.selectedCandidate ? "" : "disabled"}>
      <option value="" ${selectedRelation ? "" : "selected"}>Select relation...</option>
      ${relationOptions.map((value) => `<option value="${value}" ${value === selectedRelation?.relation ? "selected" : ""}>${value.replaceAll("_", " ")}</option>`).join("")}
    </select>`;
    const targetMarkup = targets.length
      ? targets.map((item) => {
        const label = objectLabel(item.annotation_uuid, annotation);
        return `<label class="dgTargetOption ${selectedTargets.has(item.annotation_uuid) ? "selected" : ""}">
          <input data-dg-target-uuid="${item.annotation_uuid}" type="checkbox"
            ${selectedTargets.has(item.annotation_uuid) ? "checked" : ""}
            ${runtime.selectedCandidate && selectedRelation?.relation !== "BACKGROUND" ? "" : "disabled"}>
          <span>${label}</span><small>${item.annotation_uuid.slice(-8)}</small>
        </label>`;
      }).join("")
      : "<p class=\"dgEmptyHint\">Create the human objects before binding a proposal.</p>";
    const selectedLabel = selectedRelation
      ? selectedRelation.annotation_uuids.map((uuid) => objectLabel(uuid, annotation)).join(", ") || "No human targets"
      : "Unbound";
    const remaining = Math.max(0, coverage.total - coverage.covered);
    return `
      <section class="dgSummaryBand"><strong>${coverage.covered}/${coverage.total}</strong><span>machine candidates bound</span></section>
      <p class="dgBindingStatus">${runtime.selectedCandidate ? "Selected proposal" : "Select a proposal in the focal image"} | targets: ${selectedLabel}</p>
      <label class="dgField">Selected candidate relation ${relationMarkup}</label>
      <fieldset class="dgTargetSelector" ${runtime.selectedCandidate ? "" : "disabled"}>
        <legend>Explicit human target${dense ? " mask" : ""} selection</legend>
        ${targetMarkup}
      </fieldset>
      ${dense ? `<label class="dgField">Candidate-to-mask coverage <input id="dgCandidateMaskCoverage" type="number" min="0" max="1" step="0.01" value="${selectedRelation?.candidate_visible_mask_coverage ?? ""}" ${runtime.selectedCandidate && selectedRelation && selectedRelation.relation !== "BACKGROUND" && selectedRelation.annotation_uuids.length ? "" : "disabled"}></label>` : ""}
      <button id="dgClearCandidateBinding" type="button" ${selectedRelation ? "" : "disabled"}>Clear selected candidate binding</button>
      <button id="dgMarkRemainingBackground" type="button" ${remaining ? "" : "disabled"}>Review ${remaining} remaining as background...</button>`;
  }

  function renderPlayerForm(annotation) {
    const selected = selectedObject(annotation);
    return `
      <section class="dgSummaryBand"><strong>${annotation.visible_person_count}</strong><span>visible people annotated</span></section>
      <div class="dgScopeReminder">ANNOTATE FOCAL ROI ONLY</div>
      <button id="dgAcceptCandidate" type="button" ${runtime.selectedCandidate ? "" : "disabled"}>Create unresolved draft geometry from proposal</button>
      <button id="dgRedrawSelectedVisible" type="button" ${selected ? "" : "disabled"}>Redraw selected visible box</button>
      <button id="dgRemoveSelected" type="button" ${selected ? "" : "disabled"}>Remove selected person</button>
      ${candidateBindingControls(annotation)}
      <label class="dgField">Earliest failure ${selectMarkup("dgFailureStage", ["NO_VALID_RAW_PROPOSAL", "BAD_RAW_LOCALIZATION", "VALID_PROPOSAL_LOW_CONFIDENCE", "VALID_PROPOSALS_NMS_COLLAPSED", "DUPLICATED_AFTER_VIEW_FUSION", "PITCH_GATE_ERROR", "RENDERER_OR_PROVENANCE_ERROR", "UNRESOLVED"], annotation.earliest_failure_stage)}</label>
      ${selected ? `
        <section class="dgSelectedObject"><strong>${objectLabel(selected.annotation_uuid, annotation)} selected</strong><small>${selected.annotation_uuid.slice(-8)}</small></section>
        <div class="dgUnresolvedNotice ${[selected.visibility_state, selected.occlusion_type, selected.pitch_state, selected.coarse_role].some((value) => ["UNRESOLVED", "UNKNOWN", "BOUNDARY_UNCERTAIN"].includes(value)) ? "" : "isResolved"}">Review every unresolved semantic field; proposal geometry is not truth.</div>
        <label class="dgField">Selected person visibility ${selectMarkup("dgPersonVisibility", ["VISIBLE", "PARTIALLY_VISIBLE", "HEAVILY_OCCLUDED", "UNRESOLVED"], selected.visibility_state)}</label>
        <label class="dgField">Occlusion ${selectMarkup("dgOcclusionType", ["NONE", "PERSON", "EQUIPMENT", "FRAME_EDGE", "SCENE_STRUCTURE", "UNKNOWN"], selected.occlusion_type)}</label>
        <label class="dgField">Occlusion fraction <input id="dgOcclusionFraction" type="number" min="0" max="1" step="0.05" value="${selected.occlusion_fraction}"></label>
        <label class="dgField">Footpoint uncertainty <input id="dgPersonFootpointUncertainty" type="number" min="0" max="200" step="1" value="${selected.footpoint_uncertainty_pixels}"></label>
        <label class="dgField">Pitch state ${selectMarkup("dgPersonPitchState", ["ON_PITCH", "OFF_PITCH", "BOUNDARY_UNCERTAIN"], selected.pitch_state)}</label>
        <label class="dgField">Coarse role ${selectMarkup("dgPersonRole", ["PLAYER", "GOALKEEPER", "REFEREE", "OFFICIAL", "STAFF_OR_SPECTATOR", "UNKNOWN"], selected.coarse_role)}</label>
        <label class="toggleChip dgWideToggle"><input id="dgPersonAmbiguityIgnore" type="checkbox" ${selected.ambiguity_ignore ? "checked" : ""}><span>Ambiguous / ignore instance</span></label>
        <div class="dgField">Truncation ${["LEFT", "TOP", "RIGHT", "BOTTOM"].map((flag) => `<label class="toggleChip"><input data-dg-truncation="${flag}" type="checkbox" ${selected.truncation_flags.includes(flag) ? "checked" : ""}><span>${flag}</span></label>`).join("")}</div>
      ` : ""}
      <div class="dgInstanceList">${annotation.player_instances.map((item, index) => `<button data-dg-object-select="${item.annotation_uuid}" class="${selected?.annotation_uuid === item.annotation_uuid ? "selected" : ""}" type="button"><strong>Person ${index + 1}</strong><span>${item.visibility_state} | ${item.pitch_state} | ${item.coarse_role}</span><small>${item.annotation_uuid.slice(-8)}</small></button>`).join("") || "<p>No human instance yet.</p>"}</div>
      ${baseNote(annotation)}`;
  }

  function renderDenseForm(annotation) {
    const selected = selectedObject(annotation);
    return `
      <section class="dgSummaryBand"><strong>${annotation.human_visible_person_count}</strong><span>visible masks</span></section>
      <div class="dgScopeReminder">ANNOTATE FOCAL ROI ONLY</div>
      <button id="dgFinishMask" type="button" ${runtime.maskPoints.length >= 3 ? "" : "disabled"}>Close visible mask</button>
      <button id="dgRedrawSelectedVisible" type="button" ${selected ? "" : "disabled"}>Redraw selected visible box</button>
      <button id="dgRemoveSelected" type="button" ${selected ? "" : "disabled"}>Remove selected mask</button>
      ${candidateBindingControls(annotation, {dense: true})}
      <label class="toggleChip dgWideToggle"><input id="dgDenseIgnore" type="checkbox" ${annotation.uncertain_or_ignore ? "checked" : ""}><span>Uncertain / ignore</span></label>
      ${selected ? `
        <section class="dgSelectedObject"><strong>${objectLabel(selected.annotation_uuid, annotation)} selected</strong><small>${selected.annotation_uuid.slice(-8)}</small></section>
        <div class="dgUnresolvedNotice ${selected.mask_quality === "UNCERTAIN" ? "" : "isResolved"}">Visible masks begin uncertain. Confirm quality explicitly.</div>
        <label class="dgField">Selected mask quality ${selectMarkup("dgMaskQuality", ["PRECISE", "COARSE", "UNCERTAIN", "IGNORE"], selected.mask_quality)}</label>
        <label class="dgField">Occlusion order <input id="dgOcclusionOrder" type="number" min="0" step="1" value="${selected.occlusion_order}"></label>
        <label class="dgField">Occluder ${selectMarkup("dgOccluderUuid", ["", ...annotation.visible_masks.filter((item) => item.annotation_uuid !== selected.annotation_uuid).map((item) => item.annotation_uuid)], selected.occluder_uuid || "")}</label>
        <div class="dgField">Pairwise overlap ${annotation.visible_masks.filter((item) => item.annotation_uuid !== selected.annotation_uuid).map((item) => `<label class="toggleChip"><input data-dg-overlap-uuid="${item.annotation_uuid}" type="checkbox" ${selected.pairwise_overlap_annotation_uuids.includes(item.annotation_uuid) ? "checked" : ""}><span>${objectLabel(item.annotation_uuid, annotation)}</span></label>`).join("") || "No other masks yet"}</div>
      ` : ""}
      <div class="dgInstanceList">${annotation.visible_masks.map((item, index) => `<button data-dg-object-select="${item.annotation_uuid}" class="${selected?.annotation_uuid === item.annotation_uuid ? "selected" : ""}" type="button"><strong>Mask ${index + 1}</strong><span>${item.mask_quality}; order ${item.occlusion_order}</span><small>${item.annotation_uuid.slice(-8)}</small></button>`).join("") || "<p>Select Mask and mark each visible contour.</p>"}</div>
      ${baseNote(annotation)}`;
  }

  function renderTemporalForm(annotation) {
    const frame = annotation.frames[runtime.frameIndex];
    return `
      <section class="dgSummaryBand"><strong>${runtime.frameIndex + 1}/11</strong><span>frame states</span></section>
      <label class="dgField">Current state ${selectMarkup("dgTemporalState", ["OBSERVED", "OBSERVED_WITH_TEMPORAL_REFINEMENT", "OCCLUDED_PREDICTED", "NOT_VISIBLE", "UNRESOLVED"], frame.state)}</label>
      <button id="dgUseTemporalCandidate" type="button" ${runtime.selectedCandidate ? "" : "disabled"}>Use selected observed proposal</button>
      <label class="toggleChip dgWideToggle"><input id="dgContactReviewed" type="checkbox" ${annotation.contact_strip_reviewed ? "checked" : ""}><span>Full 11-frame contact strip reviewed</span></label>
      <button id="dgCopyGeometryNext" type="button" ${frame.visible_body_box ? "" : "disabled"}>Copy geometry to next frame as draft</button>
      <button id="dgConfirmGeometryDraft" type="button" ${currentGeometryDraft() ? "" : "disabled"}>Confirm copied geometry on this frame</button>
      <button id="dgJumpUnresolved" type="button">Jump to next unresolved frame</button>
      <button id="dgJumpMachineUncertain" type="button">Jump to machine-uncertain frame</button>
      <button id="dgAcceptStableRun" type="button" ${annotation.contact_strip_reviewed && annotation.frames.every((item) => item.state !== "UNRESOLVED") ? "" : "disabled"}>Confirm reviewed frame-by-frame run</button>
      <div class="dgStateGrid">${annotation.frames.map((item, index) => `<button data-dg-frame-jump="${index}" class="${index === runtime.frameIndex ? "active" : ""}" type="button"><span>${index + 1}</span><small>${item.state}</small></button>`).join("")}</div>
      ${baseNote(annotation)}`;
  }

  function renderPitchForm(annotation) {
    return `
      <label class="dgField">Pitch state ${selectMarkup("dgPitchState", ["ON_PITCH", "OFF_PITCH", "BOUNDARY_UNCERTAIN"], annotation.pitch_state)}</label>
      <label class="dgField">Coarse role ${selectMarkup("dgCoarseRole", ["PLAYER", "GOALKEEPER", "REFEREE", "OFFICIAL", "STAFF_OR_SPECTATOR", "UNKNOWN"], annotation.coarse_role)}</label>
      <label class="dgField">Footpoint uncertainty <input id="dgFootpointUncertainty" type="number" min="0" max="200" step="1" value="${annotation.footpoint_uncertainty_pixels}"></label>
      <label class="toggleChip dgWideToggle"><input id="dgSupplyEligible" type="checkbox" ${annotation.primary_on_pitch_supply_eligible ? "checked" : ""} ${annotation.pitch_state === "ON_PITCH" ? "" : "disabled"}><span>Eligible for primary on-pitch supply</span></label>
      <p class="dgCoordinateReadout">Footpoint ${annotation.footpoint.x.toFixed(1)}, ${annotation.footpoint.y.toFixed(1)}</p>
      ${baseNote(annotation)}`;
  }

  function renderFootballForm(annotation) {
    const frame = annotation.frames[runtime.frameIndex];
    return `
      <section class="dgSummaryBand"><strong>${runtime.frameIndex + 1}/9</strong><span>football frames</span></section>
      <label class="dgField">Current state ${selectMarkup("dgFootballState", ["VISIBLE_CLEAR", "VISIBLE_BLURRED", "PARTIALLY_OCCLUDED_VISIBLE", "FULLY_OCCLUDED_PREDICTED", "NOT_VISIBLE", "OUT_OF_FRAME", "UNRESOLVED"], frame.state)}</label>
      <label class="dgField">Hard negative ${selectMarkup("dgHardNegative", ["", "PITCH_MARKING", "LINE_INTERSECTION", "LOGO_OR_TEXT", "EQUIPMENT_OR_CONE", "SHOE_OR_SOCK", "HEAD", "HIGHLIGHT_OR_REFLECTION", "COMPRESSION_ARTEFACT", "UNKNOWN"], frame.hard_negative_category || "")}</label>
      <label class="toggleChip dgWideToggle"><input id="dgBallContactReviewed" type="checkbox" ${annotation.full_contact_strip_reviewed ? "checked" : ""}><span>Full 9-frame contact strip reviewed</span></label>
      <button id="dgApplyBallState" type="button" ${annotation.full_contact_strip_reviewed && ["NOT_VISIBLE", "OUT_OF_FRAME"].includes(frame.state) ? "" : "disabled"}>Apply reviewed no-ball state to burst</button>
      <button id="dgJumpUnresolved" type="button">Jump to next unresolved frame</button>
      <button id="dgJumpMachineUncertain" type="button">Jump to machine-uncertain frame</button>
      <div class="dgStateGrid">${annotation.frames.map((item, index) => `<button data-dg-frame-jump="${index}" class="${index === runtime.frameIndex ? "active" : ""}" type="button"><span>${index + 1}</span><small>${item.state}</small></button>`).join("")}</div>
      ${baseNote(annotation)}`;
  }

  function bindCommonForm(annotation) {
    byId("dgNote")?.addEventListener("input", (event) => {
      annotation.note = event.target.value;
      persistDraft();
    });
    for (const button of document.querySelectorAll("[data-dg-frame-jump]")) {
      button.addEventListener("click", () => setFrame(Number(button.dataset.dgFrameJump)));
    }
  }

  function upsertSelectedCandidateRelation(annotation, relation, annotationUuids, coverageValue = undefined) {
    if (!runtime.selectedCandidate) return null;
    const candidateUuid = runtime.selectedCandidate.diagnostic_uuid;
    const prior = (annotation.candidate_relations || []).find((row) => row.candidate_uuid === candidateUuid);
    const row = {
      candidate_uuid: candidateUuid,
      relation,
      annotation_uuids: [...new Set(annotationUuids)],
    };
    const coverage = coverageValue === undefined ? prior?.candidate_visible_mask_coverage : coverageValue;
    if (relation !== "BACKGROUND" && coverage !== undefined && coverage !== null) {
      row.candidate_visible_mask_coverage = coverage;
    }
    annotation.candidate_relations = (annotation.candidate_relations || []).filter(
      (item) => item.candidate_uuid !== candidateUuid
    );
    annotation.candidate_relations.push(row);
    return row;
  }

  function removeSelectedAnnotation(annotation) {
    const selected = selectedObject(annotation);
    if (!selected) return;
    const label = objectLabel(selected.annotation_uuid, annotation);
    const affected = (annotation.candidate_relations || []).filter(
      (row) => row.annotation_uuids.includes(selected.annotation_uuid)
    );
    if (!window.confirm(`Remove ${label}? ${affected.length} affected candidate binding(s) will be cleared and must be reviewed again.`)) return;
    pushHistory();
    if (annotation.player_instances) {
      annotation.player_instances = annotation.player_instances.filter(
        (item) => item.annotation_uuid !== selected.annotation_uuid
      );
      annotation.visible_person_count = annotation.player_instances.length;
    } else {
      annotation.visible_masks = annotation.visible_masks.filter(
        (item) => item.annotation_uuid !== selected.annotation_uuid
      );
      annotation.human_visible_person_count = annotation.visible_masks.length;
      for (const mask of annotation.visible_masks) {
        mask.pairwise_overlap_annotation_uuids = (mask.pairwise_overlap_annotation_uuids || []).filter(
          (uuid) => uuid !== selected.annotation_uuid
        );
        if (mask.occluder_uuid === selected.annotation_uuid) delete mask.occluder_uuid;
      }
    }
    annotation.candidate_relations = (annotation.candidate_relations || []).filter(
      (row) => !row.annotation_uuids.includes(selected.annotation_uuid)
    );
    const replacement = annotationObjects(annotation)[0];
    if (replacement) runtime.selectedObjectByCase[currentCase().case_id] = replacement.annotation_uuid;
    else delete runtime.selectedObjectByCase[currentCase().case_id];
    runtime.redrawVisibleObjectUuid = null;
    persistDraft();
    renderAnnotationForm();
    renderOverlay();
    if (affected.length) showError(`${affected.length} candidate binding(s) were cleared. Rebind them before saving.`);
  }

  function renderAnnotationForm() {
    const annotation = draft();
    const task = currentCase().task_type;
    const form = byId("dgAnnotationForm");
    if (task === "detection_gold_player_static") form.innerHTML = renderPlayerForm(annotation);
    else if (task === "detection_gold_dense_region") form.innerHTML = renderDenseForm(annotation);
    else if (task === "detection_gold_temporal_player") form.innerHTML = renderTemporalForm(annotation);
    else if (task === "detection_gold_pitch_boundary") form.innerHTML = renderPitchForm(annotation);
    else form.innerHTML = renderFootballForm(annotation);
    bindCommonForm(annotation);

    byId("dgAcceptCandidate")?.addEventListener("click", () => {
      try { acceptSelectedPerson(); } catch (error) { showError(error.message); }
    });
    for (const button of document.querySelectorAll("[data-dg-object-select]")) {
      button.addEventListener("click", () => selectObject(button.dataset.dgObjectSelect));
    }
    byId("dgRedrawSelectedVisible")?.addEventListener("click", () => {
      const selected = selectedObject(annotation);
      if (!selected || !requireFocalAnnotationScope()) return;
      runtime.redrawVisibleObjectUuid = selected.annotation_uuid;
      runtime.tool = "box";
      document.querySelectorAll("[data-dg-tool]").forEach((item) => {
        item.classList.toggle("active", item.dataset.dgTool === "box");
      });
      setSaveState(`Draw the replacement visible box for ${objectLabel(selected.annotation_uuid, annotation)}`, false);
    });
    byId("dgRemoveSelected")?.addEventListener("click", () => removeSelectedAnnotation(annotation));
    byId("dgCandidateRelation")?.addEventListener("change", (event) => {
      if (!runtime.selectedCandidate) return;
      if (runtime.selectedCandidate.class_name !== "person") return showError("Player relations require a person proposal.");
      const relation = event.target.value;
      if (!relation) return;
      const prior = (annotation.candidate_relations || []).find(
        (row) => row.candidate_uuid === runtime.selectedCandidate.diagnostic_uuid
      );
      const ids = relation === "BACKGROUND" ? [] : (prior?.annotation_uuids || []);
      pushHistory();
      upsertSelectedCandidateRelation(annotation, relation, ids, relation === "BACKGROUND" ? null : undefined);
      persistDraft();
      renderAnnotationForm();
    });
    for (const toggle of document.querySelectorAll("[data-dg-target-uuid]")) {
      toggle.addEventListener("change", () => {
        if (!runtime.selectedCandidate) return;
        const relation = byId("dgCandidateRelation")?.value;
        if (!relation || relation === "BACKGROUND") return showError("Select a non-background relation before choosing human targets.");
        const ids = [...document.querySelectorAll("[data-dg-target-uuid]:checked")].map(
          (item) => item.dataset.dgTargetUuid
        );
        pushHistory();
        upsertSelectedCandidateRelation(annotation, relation, ids);
        persistDraft();
        renderAnnotationForm();
      });
    }
    byId("dgCandidateMaskCoverage")?.addEventListener("input", (event) => {
      if (!runtime.selectedCandidate) return;
      const relation = (annotation.candidate_relations || []).find(
        (row) => row.candidate_uuid === runtime.selectedCandidate.diagnostic_uuid
      );
      if (!relation || relation.relation === "BACKGROUND" || !relation.annotation_uuids.length) {
        event.target.value = "";
        return showError("Choose a non-background relation and explicit target masks before entering coverage.");
      }
      if (event.target.value === "") {
        delete relation.candidate_visible_mask_coverage;
        persistDraft();
        return;
      }
      const value = Number(event.target.value);
      if (!Number.isFinite(value) || value < 0 || value > 1) return showError("Candidate-to-mask coverage must be between 0 and 1.");
      relation.candidate_visible_mask_coverage = value;
      persistDraft();
    });
    byId("dgClearCandidateBinding")?.addEventListener("click", () => {
      if (!runtime.selectedCandidate) return;
      pushHistory();
      annotation.candidate_relations = (annotation.candidate_relations || []).filter(
        (row) => row.candidate_uuid !== runtime.selectedCandidate.diagnostic_uuid
      );
      persistDraft();
      renderAnnotationForm();
      renderOverlay();
    });
    byId("dgMarkRemainingBackground")?.addEventListener("click", () => {
      const covered = new Set(annotation.candidate_relations.map((row) => row.candidate_uuid));
      const remaining = (currentCase().visible_metadata.candidate_uuids || []).filter(
        (candidateUuid) => !covered.has(candidateUuid)
      );
      if (!remaining.length) return;
      const confirmed = window.confirm(
        `Mark ${remaining.length} unbound candidate row(s) as BACKGROUND? Confirm that all enabled proposal layers and every visible person in the focal ROI were reviewed. Existing bindings will not change.`
      );
      if (!confirmed) return;
      pushHistory();
      for (const candidateUuid of remaining) {
        annotation.candidate_relations.push({candidate_uuid: candidateUuid, relation: "BACKGROUND", annotation_uuids: []});
      }
      persistDraft(); renderAnnotationForm(); renderOverlay();
    });
    byId("dgFailureStage")?.addEventListener("change", (event) => { annotation.earliest_failure_stage = event.target.value; persistDraft(); });
    byId("dgFinishMask")?.addEventListener("click", finishMask);
    byId("dgDenseIgnore")?.addEventListener("change", (event) => { annotation.uncertain_or_ignore = event.target.checked; persistDraft(); });
    byId("dgMaskQuality")?.addEventListener("change", (event) => { const mask = selectedObject(annotation); if (mask) mask.mask_quality = event.target.value; persistDraft(); renderAnnotationForm(); renderOverlay(); });
    byId("dgOcclusionOrder")?.addEventListener("input", (event) => { const mask = selectedObject(annotation); if (mask) mask.occlusion_order = Number(event.target.value); persistDraft(); });
    byId("dgOccluderUuid")?.addEventListener("input", (event) => { const mask = selectedObject(annotation); if (!mask) return; if (event.target.value) mask.occluder_uuid = event.target.value; else delete mask.occluder_uuid; persistDraft(); });
    for (const toggle of document.querySelectorAll("[data-dg-overlap-uuid]")) {
      toggle.addEventListener("change", () => {
        const mask = selectedObject(annotation);
        if (!mask) return;
        const selected = new Set(mask.pairwise_overlap_annotation_uuids || []);
        if (toggle.checked) selected.add(toggle.dataset.dgOverlapUuid); else selected.delete(toggle.dataset.dgOverlapUuid);
        mask.pairwise_overlap_annotation_uuids = [...selected].sort();
        persistDraft();
      });
    }
    byId("dgPersonVisibility")?.addEventListener("change", (event) => { const person = selectedObject(annotation); if (person) person.visibility_state = event.target.value; persistDraft(); renderAnnotationForm(); renderOverlay(); });
    byId("dgOcclusionType")?.addEventListener("change", (event) => { const person = selectedObject(annotation); if (person) person.occlusion_type = event.target.value; persistDraft(); renderAnnotationForm(); });
    byId("dgOcclusionFraction")?.addEventListener("input", (event) => { const person = selectedObject(annotation); if (person) person.occlusion_fraction = Number(event.target.value); persistDraft(); });
    byId("dgPersonFootpointUncertainty")?.addEventListener("input", (event) => { const person = selectedObject(annotation); if (person) person.footpoint_uncertainty_pixels = Number(event.target.value); persistDraft(); });
    byId("dgPersonPitchState")?.addEventListener("change", (event) => { const person = selectedObject(annotation); if (person) person.pitch_state = event.target.value; persistDraft(); renderAnnotationForm(); });
    byId("dgPersonRole")?.addEventListener("change", (event) => { const person = selectedObject(annotation); if (person) person.coarse_role = event.target.value; persistDraft(); renderAnnotationForm(); });
    byId("dgPersonAmbiguityIgnore")?.addEventListener("change", (event) => { const person = selectedObject(annotation); if (person) person.ambiguity_ignore = event.target.checked; persistDraft(); });
    for (const toggle of document.querySelectorAll("[data-dg-truncation]")) {
      toggle.addEventListener("change", () => {
        const person = selectedObject(annotation);
        if (!person) return;
        const selected = new Set(person.truncation_flags || []);
        if (toggle.checked) selected.add(toggle.dataset.dgTruncation); else selected.delete(toggle.dataset.dgTruncation);
        person.truncation_flags = [...selected].sort();
        persistDraft();
      });
    }
    byId("dgTemporalState")?.addEventListener("change", (event) => setTemporalState(event.target.value));
    byId("dgUseTemporalCandidate")?.addEventListener("click", () => acceptSelectedPerson());
    byId("dgContactReviewed")?.addEventListener("change", (event) => { annotation.contact_strip_reviewed = event.target.checked; persistDraft(); renderAnnotationForm(); });
    byId("dgCopyGeometryNext")?.addEventListener("click", copyGeometryToNextFrame);
    byId("dgConfirmGeometryDraft")?.addEventListener("click", confirmGeometryDraft);
    byId("dgJumpUnresolved")?.addEventListener("click", jumpToUnresolvedFrame);
    byId("dgJumpMachineUncertain")?.addEventListener("click", jumpToMachineUncertainFrame);
    byId("dgAcceptStableRun")?.addEventListener("click", acceptStableRun);
    byId("dgPitchState")?.addEventListener("change", (event) => {
      annotation.pitch_state = event.target.value;
      if (annotation.pitch_state !== "ON_PITCH") annotation.primary_on_pitch_supply_eligible = false;
      persistDraft(); renderAnnotationForm();
    });
    byId("dgCoarseRole")?.addEventListener("change", (event) => { annotation.coarse_role = event.target.value; persistDraft(); });
    byId("dgFootpointUncertainty")?.addEventListener("input", (event) => { annotation.footpoint_uncertainty_pixels = Number(event.target.value); persistDraft(); });
    byId("dgSupplyEligible")?.addEventListener("change", (event) => { annotation.primary_on_pitch_supply_eligible = event.target.checked; persistDraft(); });
    byId("dgFootballState")?.addEventListener("change", (event) => setFootballState(event.target.value));
    byId("dgHardNegative")?.addEventListener("change", (event) => {
      const frame = annotation.frames[runtime.frameIndex];
      if (event.target.value) {
        frame.state = "NOT_VISIBLE";
        delete frame.centre_point;
        frame.hard_negative_category = event.target.value;
      } else delete frame.hard_negative_category;
      persistDraft(); renderAnnotationForm(); renderOverlay();
    });
    byId("dgBallContactReviewed")?.addEventListener("change", (event) => { annotation.full_contact_strip_reviewed = event.target.checked; persistDraft(); renderAnnotationForm(); });
    byId("dgApplyBallState")?.addEventListener("click", applyFootballStateToBurst);
  }

  function setTemporalState(value) {
    const annotation = draft();
    pushHistory();
    annotation.stable_run_accepted = false;
    const row = currentRecord();
    if (["OBSERVED", "OBSERVED_WITH_TEMPORAL_REFINEMENT"].includes(value)) {
      const existing = annotation.frames[runtime.frameIndex];
      if (!runtime.selectedCandidate && existing?.visible_body_box && existing.current_frame_pixel_support) {
        annotation.frames[runtime.frameIndex] = {...clone(existing), state: value};
        persistDraft(); renderAnnotationForm(); renderOverlay();
        return;
      }
      if (!runtime.selectedCandidate) return showError("Select a current-frame person proposal or draw a current-frame visible box first.");
      if (runtime.selectedCandidate.class_name !== "person" || !runtime.selectedCandidate.diagnostic_uuid) {
        return showError("Temporal observed states require a current-frame person proposal.");
      }
      const box = clone(runtime.selectedCandidate.bbox_original_pixels);
      annotation.frames[runtime.frameIndex] = {
        frame_sequence: row.frame_sequence,
        source_frame_sha256: row.source_frame_sha256,
        state: value,
        visible_body_box: box,
        footpoint: {x: (box.x1 + box.x2) / 2, y: box.y2},
        current_frame_pixel_support: true,
        candidate_uuids: [runtime.selectedCandidate.diagnostic_uuid],
      };
    } else {
      annotation.frames[runtime.frameIndex] = {
        frame_sequence: row.frame_sequence,
        source_frame_sha256: row.source_frame_sha256,
        state: value,
        current_frame_pixel_support: false,
        candidate_uuids: [],
      };
    }
    persistDraft(); renderAnnotationForm(); renderOverlay();
  }

  function copyGeometryToNextFrame() {
    const annotation = draft();
    const current = annotation.frames[runtime.frameIndex];
    if (!current.visible_body_box || runtime.frameIndex >= records().length - 1) return;
    const nextIndex = runtime.frameIndex + 1;
    const next = records()[nextIndex];
    caseGeometryDrafts()[String(next.frame_sequence)] = {
      visible_body_box: clone(current.visible_body_box),
      footpoint: clone(current.footpoint || {
        x: (current.visible_body_box.x1 + current.visible_body_box.x2) / 2,
        y: current.visible_body_box.y2,
      }),
      copied_from_frame_sequence: current.frame_sequence,
      human_truth: false,
    };
    persistDraft();
    setFrame(nextIndex);
  }

  function confirmGeometryDraft() {
    const geometry = currentGeometryDraft();
    if (!geometry) return;
    const annotation = draft();
    pushHistory();
    annotation.stable_run_accepted = false;
    annotation.frames[runtime.frameIndex] = {
      frame_sequence: currentRecord().frame_sequence,
      source_frame_sha256: currentRecord().source_frame_sha256,
      state: "OBSERVED_WITH_TEMPORAL_REFINEMENT",
      visible_body_box: clone(geometry.visible_body_box),
      footpoint: clone(geometry.footpoint),
      current_frame_pixel_support: true,
      candidate_uuids: [],
    };
    delete caseGeometryDrafts()[String(currentRecord().frame_sequence)];
    persistDraft(); renderAnnotationForm(); renderOverlay();
  }

  function jumpToUnresolvedFrame() {
    const frames = draft().frames || [];
    const index = frames.findIndex((frame, candidateIndex) => candidateIndex > runtime.frameIndex && frame.state === "UNRESOLVED");
    const wrapped = index >= 0 ? index : frames.findIndex((frame) => frame.state === "UNRESOLVED");
    if (wrapped >= 0) setFrame(wrapped);
  }

  function jumpToMachineUncertainFrame() {
    const ranked = records().map((row, index) => {
      const scores = (row.candidates || []).map((candidate) => Number(candidate.score || 0)).sort((a, b) => b - a);
      const margin = scores.length > 1 ? scores[0] - scores[1] : scores[0] || 0;
      return {index, margin, candidateCount: scores.length};
    }).sort((left, right) => left.candidateCount - right.candidateCount || left.margin - right.margin || left.index - right.index);
    if (ranked.length) setFrame(ranked[0].index);
  }

  function acceptStableRun() {
    const annotation = draft();
    if (!annotation.contact_strip_reviewed || annotation.frames.some((frame) => frame.state === "UNRESOLVED")) return;
    pushHistory();
    annotation.stable_run_accepted = true;
    persistDraft(); renderAnnotationForm(); renderOverlay();
  }

  function setFootballState(value) {
    const annotation = draft();
    const row = currentRecord();
    pushHistory();
    const frame = {frame_sequence: row.frame_sequence, source_frame_sha256: row.source_frame_sha256, state: value};
    if (["VISIBLE_CLEAR", "VISIBLE_BLURRED", "PARTIALLY_OCCLUDED_VISIBLE"].includes(value)) {
      const candidate = runtime.selectedCandidate;
      if (!candidate || candidate.class_name !== "sports_ball") { showError("Select a sports-ball proposal or use the Ball tool to mark its centre."); return; }
      const box = candidate.bbox_original_pixels;
      frame.centre_point = {x: (box.x1 + box.x2) / 2, y: (box.y1 + box.y2) / 2};
      frame.apparent_ellipse = {centre_x: frame.centre_point.x, centre_y: frame.centre_point.y, radius_x: (box.x2 - box.x1) / 2, radius_y: (box.y2 - box.y1) / 2, rotation_degrees: 0};
      frame.geometry_uncertainty_pixels = 3;
    }
    annotation.frames[runtime.frameIndex] = frame;
    persistDraft(); renderAnnotationForm(); renderOverlay();
  }

  function applyFootballStateToBurst() {
    const annotation = draft();
    if (!annotation.full_contact_strip_reviewed || !["NOT_VISIBLE", "OUT_OF_FRAME"].includes(annotation.frames[runtime.frameIndex].state)) return;
    const source = clone(annotation.frames[runtime.frameIndex]);
    pushHistory();
    annotation.frames = records().map((row) => ({...clone(source), frame_sequence: row.frame_sequence, source_frame_sha256: row.source_frame_sha256}));
    persistDraft(); renderAnnotationForm(); renderOverlay();
  }

  function renderProvenance() {
    const container = byId("dgProvenance");
    const candidate = runtime.selectedCandidate;
    if (!candidate) {
      container.innerHTML = "<p>Select a proposal to inspect its raw-to-fused binding.</p>";
      return;
    }
    container.innerHTML = `<dl>
      <div><dt>Diagnostic UUID</dt><dd>${candidate.diagnostic_uuid}</dd></div>
      <div><dt>Stage</dt><dd>${candidate.stage}</dd></div>
      <div><dt>Class</dt><dd>${candidate.class_name}</dd></div>
      <div><dt>Score</dt><dd>${candidate.score.toFixed(4)}</dd></div>
      <div><dt>Inference view</dt><dd>${candidate.inference_view}</dd></div>
      <div><dt>Source row</dt><dd>${candidate.source_row_sha256}</dd></div>
      <div><dt>Coordinate space</dt><dd>${candidate.coordinate_space}</dd></div>
    </dl>`;
  }

  async function sha256Buffer(buffer) {
    const digest = await crypto.subtle.digest("SHA-256", buffer);
    return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
  }

  async function verifyImage(path, expectedHash) {
    const cacheKey = `${currentCase().case_id}:${path}:${expectedHash}`;
    if (runtime.verifiedAssets.has(cacheKey)) return true;
    const response = await fetch(evidenceUrl(path), {cache: "no-store"});
    if (!response.ok || !response.headers.get("content-type")?.startsWith("image/")) throw new Error("image route failed");
    const buffer = await response.arrayBuffer();
    if (!buffer.byteLength) throw new Error("empty image");
    const actual = await sha256Buffer(buffer);
    if (actual !== expectedHash) throw new Error("image hash mismatch");
    const image = new Image();
    image.src = URL.createObjectURL(new Blob([buffer], {type: response.headers.get("content-type")}));
    await image.decode();
    if (!image.naturalWidth || !image.naturalHeight) throw new Error("image decode failed");
    URL.revokeObjectURL(image.src);
    runtime.verifiedAssets.add(cacheKey);
    return true;
  }

  async function renderEvidence() {
    const token = ++runtime.evidenceToken;
    const row = currentRecord();
    if (!row) return;
    runtime.evidenceBlocked = true;
    byId("dgSaveCase").disabled = true;
    byId("dgEvidenceBlocker").classList.add("isHidden");
    byId("dgEvidenceStatus").textContent = "Verifying exact source bytes";
    const basePath = runtime.view === "focal" ? row.focal_asset_path : row.panorama_asset_path;
    const baseHash = runtime.view === "focal" ? row.focal_asset_sha256 : row.source_frame_sha256;
    const image = byId("dgBaseImage");
    image.src = evidenceUrl(basePath);
    const context = records();
    const previous = context[Math.max(0, runtime.frameIndex - 1)];
    const next = context[Math.min(context.length - 1, runtime.frameIndex + 1)];
    byId("dgPreviousImage").src = evidenceUrl(previous.panorama_asset_path);
    byId("dgCurrentContext").src = evidenceUrl(row.panorama_asset_path);
    byId("dgNextImage").src = evidenceUrl(next.panorama_asset_path);
    try {
      const contactChecks = records().map((record) => verifyImage(
        record.contact_asset_path || record.focal_asset_path,
        record.contact_asset_sha256 || record.focal_asset_sha256,
      ));
      await Promise.all([
        image.decode(),
        verifyImage(basePath, baseHash),
        verifyImage(row.panorama_asset_path, row.source_frame_sha256),
        verifyImage(previous.panorama_asset_path, previous.source_frame_sha256),
        verifyImage(next.panorama_asset_path, next.source_frame_sha256),
        ...contactChecks,
      ]);
      if (token !== runtime.evidenceToken) return;
      runtime.evidenceBlocked = false;
      byId("dgEvidenceStatus").textContent = `Evidence verified | ${row.image_width} x ${row.image_height} | ${row.source_frame_sha256.slice(0, 12)}`;
      byId("dgSaveCase").disabled = false;
    } catch (error) {
      if (token !== runtime.evidenceToken) return;
      byId("dgEvidenceBlocker").classList.remove("isHidden");
      byId("dgEvidenceBlocker").textContent = `Evidence unavailable: ${error.message}`;
      byId("dgEvidenceStatus").textContent = "Evidence blocked";
      runtime.evidenceBlocked = true;
    }
    renderOverlay();
  }

  function renderContactStrip() {
    const container = byId("dgContactStrip");
    container.replaceChildren();
    records().forEach((row, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = index === runtime.frameIndex ? "active" : "";
      const image = document.createElement("img");
      image.alt = `Frame ${index + 1}`;
      image.src = evidenceUrl(row.contact_asset_path || row.focal_asset_path);
      const label = document.createElement("span");
      label.textContent = String(index + 1);
      button.append(image, label);
      button.addEventListener("click", () => setFrame(index));
      container.appendChild(button);
    });
    container.classList.toggle("isCompact", records().length <= 3);
  }

  function renderProgress() {
    const counts = runtime.state.counts || {};
    const reviewed = counts.reviewed || Object.keys(runtime.state.annotations || {}).length;
    byId("dgCaseProgress").textContent = `Case ${runtime.activeIndex + 1} of ${runtime.manifest.cases.length}`;
    byId("dgModuleProgress").textContent = `${moduleInfo().label} | ${reviewed} saved`;
    byId("dgProgressBar").style.width = `${(reviewed / runtime.manifest.cases.length) * 100}%`;
    byId("dgComplete").disabled = Boolean(runtime.state.completed) || reviewed !== runtime.manifest.cases.length || runtime.outbox.length > 0;
  }

  function renderCase() {
    runtime.selectedCandidate = null;
    runtime.maskPoints = [];
    runtime.redrawVisibleObjectUuid = null;
    const caseData = currentCase();
    const info = moduleInfo();
    byId("dgTitle").textContent = runtime.uiConfig.review_title;
    byId("dgModuleEyebrow").textContent = info.eyebrow;
    byId("dgCaseTitle").textContent = `${info.label} ${caseData.visible_metadata.module_case_number}`;
    byId("dgQuestion").textContent = caseData.concise_question;
    byId("dgScopeBadge").classList.toggle("isHidden", !focalScopeApplies());
    runtime.frameIndex = Math.min(runtime.frameIndex, Math.max(0, records().length - 1));
    byId("dgTimeline").max = String(Math.max(0, records().length - 1));
    byId("dgTimeline").value = String(runtime.frameIndex);
    byId("dgFrameReadout").textContent = `Frame ${currentRecord().frame_sequence} | ${currentRecord().timestamp_seconds.toFixed(3)}s`;
    document.querySelectorAll("[data-dg-module]").forEach((button) => button.classList.toggle("active", button.dataset.dgModule === caseData.task_type));
    byId("dgReopenCase").disabled = !(runtime.state.annotations || {})[caseData.case_id];
    renderProgress();
    renderContactStrip();
    renderAnnotationForm();
    renderProvenance();
    renderEvidence();
  }

  function setFrame(index) {
    runtime.frameIndex = Math.max(0, Math.min(index, records().length - 1));
    runtime.selectedCandidate = null;
    runtime.redrawVisibleObjectUuid = null;
    byId("dgTimeline").value = String(runtime.frameIndex);
    renderCase();
  }

  function showError(message) {
    const element = byId("dgFormError");
    element.textContent = message;
    element.classList.remove("isHidden");
  }

  function clearError() {
    byId("dgFormError").classList.add("isHidden");
  }

  function canonicalJson(value) {
    if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
    if (value && typeof value === "object") {
      return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
    }
    return JSON.stringify(value);
  }

  function validateDraftForSave(caseData, annotation) {
    if (canonicalJson(annotation.source_binding) !== canonicalJson(sourceBinding(caseData))) {
      throw new Error("Source binding diverged from the immutable case evidence. Reload this case before saving.");
    }
    const width = Number(annotation.source_binding.image_width);
    const height = Number(annotation.source_binding.image_height);
    const validatePoint = (point, label) => {
      if (!point) return;
      if (![point.x, point.y].every(Number.isFinite) || point.x < 0 || point.x > width || point.y < 0 || point.y > height) {
        throw new Error(`${label} lies outside the source image.`);
      }
    };
    const validateBox = (box, label) => {
      if (!box) return;
      if (![box.x1, box.y1, box.x2, box.y2].every(Number.isFinite)
        || box.x1 < 0 || box.y1 < 0 || box.x2 > width || box.y2 > height
        || box.x2 <= box.x1 || box.y2 <= box.y1) {
        throw new Error(`${label} lies outside source pixels or has invalid dimensions.`);
      }
    };
    for (const item of [...(annotation.player_instances || []), ...(annotation.visible_masks || [])]) {
      validateBox(item.visible_body_box, "Visible geometry");
      validateBox(item.full_body_box, "Full-body geometry");
      validateBox(item.optional_head_box, "Head geometry");
      validatePoint(item.footpoint, "Footpoint");
      for (const point of item.polygon_original_pixels || []) validatePoint(point, "Mask point");
    }
    validatePoint(annotation.footpoint, "Pitch footpoint");

    if (["detection_gold_player_static", "detection_gold_dense_region"].includes(caseData.task_type)) {
      const objects = annotationObjects(annotation);
      const objectIds = new Set(objects.map((item) => item.annotation_uuid));
      const required = new Set(caseData.visible_metadata.candidate_uuids || []);
      const relations = annotation.candidate_relations || [];
      const actual = relations.map((row) => row.candidate_uuid);
      if (actual.length !== new Set(actual).size || actual.length !== required.size || actual.some((uuid) => !required.has(uuid))) {
        throw new Error("Candidate coverage is incomplete. Bind every candidate or use the confirmed background action.");
      }
      for (const relation of relations) {
        const ids = [...new Set(relation.annotation_uuids || [])];
        if (ids.some((uuid) => !objectIds.has(uuid))) throw new Error("A candidate relation references a removed human target. Rebind it.");
        if (relation.relation === "BACKGROUND" && ids.length !== 0) throw new Error("BACKGROUND requires zero human targets.");
        if (["CLEAN_SINGLE_INSTANCE", "DUPLICATE_OF_INSTANCE", "PARTIAL_INSTANCE"].includes(relation.relation) && ids.length !== 1) {
          throw new Error(`${relation.relation} requires exactly one explicitly selected human target.`);
        }
        if (relation.relation === "MERGED_MULTIPLE_INSTANCES" && ids.length < 2) {
          throw new Error("MERGED_MULTIPLE_INSTANCES requires an explicit subset of at least two targets.");
        }
        const maskCoverage = relation.candidate_visible_mask_coverage;
        if (relation.relation === "BACKGROUND" && maskCoverage !== undefined) {
          throw new Error("BACKGROUND candidates cannot carry visible-mask coverage.");
        }
        if (maskCoverage !== undefined && (!Number.isFinite(maskCoverage) || maskCoverage < 0 || maskCoverage > 1 || !ids.length)) {
          throw new Error("Dense candidate-to-mask coverage must be between 0 and 1 and bound to selected target masks.");
        }
      }
    }

    if (caseData.task_type === "detection_gold_temporal_player") {
      if (!annotation.contact_strip_reviewed || annotation.frames.some((frame) => frame.state === "UNRESOLVED")) {
        throw new Error("Review the full temporal strip and resolve every frame before saving.");
      }
      const expected = caseData.visible_metadata.frame_records || [];
      if (annotation.frames.length !== expected.length) throw new Error("Temporal frame coverage diverged from the immutable strip.");
      annotation.frames.forEach((frame, index) => {
        const row = expected[index];
        if (frame.frame_sequence !== row.frame_sequence || frame.source_frame_sha256 !== row.source_frame_sha256) {
          throw new Error("Temporal frame or source hash binding diverged.");
        }
        if (!(frame.candidate_uuids || []).every((uuid) => typeof uuid === "string" && uuid.length > 0)) {
          throw new Error("Temporal candidate UUIDs cannot contain null or undefined values.");
        }
        validateBox(frame.visible_body_box, "Temporal visible geometry");
        validatePoint(frame.footpoint, "Temporal footpoint");
      });
    }
    if (caseData.task_type === "detection_gold_football_burst") {
      if (!annotation.full_contact_strip_reviewed || annotation.frames.some((frame) => frame.state === "UNRESOLVED")) {
        throw new Error("Review the full football strip and resolve every frame before saving.");
      }
    }
  }

  async function enqueueSave() {
    clearError();
    if (runtime.evidenceBlocked) return showError("Case saving is blocked until exact evidence verifies.");
    const caseData = currentCase();
    try {
      validateDraftForSave(caseData, draft());
    } catch (error) {
      showError(error.message);
      return;
    }
    const clientEventId = uid("detection-case");
    const item = {
      event_type: "DETECTION_CASE_SAVED",
      review_id: runtime.manifest.review_id,
      reviewer_session_id: runtime.uiConfig.question_contract.reviewer_session_id,
      case_id: caseData.case_id,
      annotation: clone(draft()),
      client_event_id: clientEventId,
      idempotency_key: `${runtime.manifest.review_id}:${caseData.case_id}:${clientEventId}`,
      expected_server_state_hash: runtime.serverStateHash,
      last_viewed_case_id: caseData.case_id,
      elapsed_active_seconds: activeSeconds(),
      input_source: "detection_gold_ui",
    };
    await dbPut("outbox", item);
    runtime.outbox.push(item);
    updateServerState();
    setSaveState("Pending server", false);
    await flushOutbox();
    if (!(runtime.state.annotations || {})[caseData.case_id]) return;
    await dbDelete("drafts", caseData.case_id);
    delete runtime.drafts[caseData.case_id];
    delete runtime.geometryDrafts[caseData.case_id];
    goNextUnresolved();
  }

  async function flushOutbox() {
    if (runtime.flushing) return;
    runtime.flushing = true;
    try {
      while (runtime.outbox.length) {
        const item = runtime.outbox[0];
        item.expected_server_state_hash = runtime.serverStateHash;
        const response = await runtime.api("/api/review/detection-gold-event", {method: "POST", body: JSON.stringify(item)});
        runtime.state = response;
        runtime.serverStateHash = response.ack.server_state_hash;
        runtime.serverSequence = response.ack.server_event_sequence;
        await dbDelete("outbox", item.client_event_id);
        runtime.outbox.shift();
        setSaveState("Saved to server", false);
        updateServerState();
      }
    } catch (error) {
      setSaveState("Offline - safely queued", true);
      showError(error.message);
    } finally {
      runtime.flushing = false;
      renderProgress();
    }
  }

  function goNextUnresolved() {
    const annotations = runtime.state.annotations || {};
    const start = runtime.activeIndex;
    for (let offset = 1; offset <= runtime.manifest.cases.length; offset += 1) {
      const index = (start + offset) % runtime.manifest.cases.length;
      if (!annotations[runtime.manifest.cases[index].case_id]) {
        runtime.activeIndex = index;
        runtime.frameIndex = 0;
        runtime.view = "focal";
        byId("dgFocalView").classList.add("active");
        byId("dgPanoramaView").classList.remove("active");
        renderCase();
        return;
      }
    }
    renderCase();
  }

  async function reopenCase() {
    const caseData = currentCase();
    try {
      const clientEventId = uid("reopen");
      const response = await runtime.api("/api/review/detection-gold-reopen", {
        method: "POST",
        body: JSON.stringify({
          event_type: "DETECTION_CASE_REOPENED",
          review_id: runtime.manifest.review_id,
          reviewer_session_id: runtime.uiConfig.question_contract.reviewer_session_id,
          client_event_id: clientEventId,
          idempotency_key: `${runtime.manifest.review_id}:${caseData.case_id}:${clientEventId}`,
          expected_server_state_hash: runtime.serverStateHash,
          case_id: caseData.case_id,
          input_source: "detection_gold_ui",
        }),
      });
      runtime.state = response;
      runtime.serverStateHash = response.ack.server_state_hash;
      runtime.serverSequence = response.ack.server_event_sequence;
      runtime.drafts[caseData.case_id] = clone(response.annotations?.[caseData.case_id] || defaultAnnotation(caseData));
      runtime.geometryDrafts[caseData.case_id] = {};
      setSaveState("Case reopened", false);
      renderCase();
    } catch (error) { showError(error.message); }
  }

  async function completePilot() {
    clearError();
    try {
      await flushOutbox();
      const response = await runtime.api("/api/review/detection-gold-complete", {
        method: "POST",
        body: JSON.stringify({
          client_event_id: uid("complete"),
          idempotency_key: `${runtime.manifest.review_id}:complete`,
          expected_server_state_hash: runtime.serverStateHash,
          pending_outbox_events: runtime.outbox.length,
          evidence_blocker_count: runtime.evidenceBlocked ? 1 : 0,
          unresolved_draft_count: (await dbAll("drafts")).length,
          unresolved_divergence: false,
          elapsed_active_seconds: activeSeconds(),
        }),
      });
      runtime.state = response;
      runtime.serverStateHash = response.ack.server_state_hash;
      runtime.serverSequence = response.ack.server_event_sequence;
      setSaveState("Pilot completed", false);
      renderProgress();
    } catch (error) { showError(error.message); }
  }

  function bind() {
    byId("dgPrevCase").addEventListener("click", () => { runtime.activeIndex = Math.max(0, runtime.activeIndex - 1); runtime.frameIndex = 0; runtime.view = "focal"; byId("dgFocalView").classList.add("active"); byId("dgPanoramaView").classList.remove("active"); renderCase(); });
    byId("dgNextCase").addEventListener("click", goNextUnresolved);
    byId("dgPreviousFrame").addEventListener("click", () => setFrame(runtime.frameIndex - 1));
    byId("dgNextFrame").addEventListener("click", () => setFrame(runtime.frameIndex + 1));
    byId("dgTimeline").addEventListener("input", (event) => setFrame(Number(event.target.value)));
    byId("dgFocalView").addEventListener("click", () => { runtime.view = "focal"; runtime.redrawVisibleObjectUuid = null; fitEvidence(); byId("dgFocalView").classList.add("active"); byId("dgPanoramaView").classList.remove("active"); renderEvidence(); renderAnnotationForm(); });
    byId("dgPanoramaView").addEventListener("click", () => { runtime.view = "panorama"; runtime.redrawVisibleObjectUuid = null; fitEvidence(); byId("dgPanoramaView").classList.add("active"); byId("dgFocalView").classList.remove("active"); renderEvidence(); renderAnnotationForm(); });
    document.querySelectorAll("[data-dg-layer]").forEach((toggle) => toggle.addEventListener("change", renderOverlay));
    document.querySelectorAll("[data-dg-tool]").forEach((button) => button.addEventListener("click", () => {
      runtime.tool = button.dataset.dgTool;
      runtime.redrawVisibleObjectUuid = null;
      document.querySelectorAll("[data-dg-tool]").forEach((item) => item.classList.toggle("active", item === button));
    }));
    document.querySelectorAll("[data-dg-module]").forEach((button) => button.addEventListener("click", () => {
      const index = runtime.manifest.cases.findIndex((caseData) => caseData.task_type === button.dataset.dgModule && !(runtime.state.annotations || {})[caseData.case_id]);
      runtime.activeIndex = index >= 0 ? index : runtime.manifest.cases.findIndex((caseData) => caseData.task_type === button.dataset.dgModule);
      runtime.frameIndex = 0; runtime.view = "focal"; byId("dgFocalView").classList.add("active"); byId("dgPanoramaView").classList.remove("active"); renderCase();
    }));
    byId("dgOverlay").addEventListener("click", handleOverlayClick);
    byId("dgOverlay").addEventListener("pointerdown", handlePointerDown);
    byId("dgOverlay").addEventListener("pointermove", handlePointerMove);
    byId("dgOverlay").addEventListener("pointerup", handlePointerUp);
    byId("dgOverlay").addEventListener("pointercancel", () => { runtime.panStart = null; runtime.drawingStart = null; });
    byId("dgViewport").addEventListener("wheel", (event) => {
      event.preventDefault();
      zoomAt(event.clientX, event.clientY, runtime.viewTransform.scale * (event.deltaY < 0 ? 1.18 : 0.85));
    }, {passive: false});
    byId("dgUndo").addEventListener("click", undo);
    byId("dgFit").addEventListener("click", fitEvidence);
    byId("dgZoomOut").addEventListener("click", () => {
      const rectangle = byId("dgViewport").getBoundingClientRect();
      zoomAt(rectangle.left + rectangle.width / 2, rectangle.top + rectangle.height / 2, runtime.viewTransform.scale / 1.25);
    });
    byId("dgZoomIn").addEventListener("click", () => {
      const rectangle = byId("dgViewport").getBoundingClientRect();
      zoomAt(rectangle.left + rectangle.width / 2, rectangle.top + rectangle.height / 2, runtime.viewTransform.scale * 1.25);
    });
    byId("dgSaveCase").addEventListener("click", enqueueSave);
    byId("dgReopenCase").addEventListener("click", reopenCase);
    byId("dgComplete").addEventListener("click", completePilot);
    byId("dgPlay").addEventListener("click", () => {
      runtime.playing = !runtime.playing;
      byId("dgPlay").textContent = runtime.playing ? "Pause" : "Play";
      clearInterval(runtime.playTimer);
      if (runtime.playing) runtime.playTimer = setInterval(() => setFrame((runtime.frameIndex + 1) % records().length), 350);
    });
    document.addEventListener("keydown", (event) => {
      if (document.body.dataset.presentation !== "detection_gold_pilot" || ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
      if (event.ctrlKey && event.key.toLowerCase() === "z") { event.preventDefault(); undo(); }
      else if (event.key === "ArrowLeft") { event.preventDefault(); setFrame(runtime.frameIndex - 1); }
      else if (event.key === "ArrowRight") { event.preventDefault(); setFrame(runtime.frameIndex + 1); }
      else if (event.key.toLowerCase() === "f") byId("dgFocalView").click();
      else if (event.key.toLowerCase() === "p") byId("dgPanoramaView").click();
      else if (event.key.toLowerCase() === "s") byId("dgSaveCase").click();
    });
    document.addEventListener("visibilitychange", activeSeconds);
    window.addEventListener("resize", clampViewTransform);
  }

  async function mount({manifest, uiConfig, state, api}) {
    runtime.manifest = manifest;
    runtime.uiConfig = uiConfig;
    runtime.state = state;
    runtime.api = api;
    runtime.serverStateHash = state.server_state_hash || "";
    runtime.serverSequence = Number(state.event_sequence || 0);
    runtime.db = await openDatabase();
    const storedDrafts = await dbAll("drafts");
    for (const row of storedDrafts) {
      runtime.drafts[row.case_id] = row.annotation;
      runtime.geometryDrafts[row.case_id] = row.geometry_drafts || {};
      if (row.selected_annotation_uuid) runtime.selectedObjectByCase[row.case_id] = row.selected_annotation_uuid;
    }
    runtime.outbox = await dbAll("outbox");
    const recovery = await api("/api/review/detection-gold-recover", {
      method: "POST",
      body: JSON.stringify({
        pending_outbox_events: runtime.outbox.length,
        evidence_blocker_count: 0,
        unresolved_draft_count: storedDrafts.length,
        unresolved_divergence: false,
      }),
    });
    runtime.state = recovery.materialized_state;
    runtime.state.counts = recovery.completion_eligibility;
    runtime.serverStateHash = recovery.server_state_hash;
    runtime.serverSequence = recovery.server_event_sequence;
    runtime.activeIndex = Math.max(0, manifest.cases.findIndex((caseData) => caseData.case_id === (state.resume_case_id || recovery.materialized_state.last_viewed_case_id)));
    bind();
    updateServerState();
    await flushOutbox();
    renderCase();
    setInterval(activeSeconds, 10000);
  }

  window.DetectionGoldPilot = {mount};
})();
