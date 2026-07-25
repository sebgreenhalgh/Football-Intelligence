(function () {
  "use strict";

  const MODULES = {
    detection_gold_player_static: {label: "Player static", eyebrow: "PLAYER INSTANCE"},
    detection_gold_dense_region: {label: "Dense region", eyebrow: "VISIBLE MASK"},
    detection_gold_temporal_player: {label: "Temporal player", eyebrow: "TEMPORAL VISIBILITY"},
    detection_gold_pitch_boundary: {label: "Pitch / boundary", eyebrow: "PITCH AND ROLE"},
    detection_gold_football_burst: {label: "Football burst", eyebrow: "FOOTBALL VISIBILITY"},
  };
  const NOVICE_MODULES = {
    detection_gold_player_static: {label: "Visible people", eyebrow: "MARK PEOPLE", question: "Find and mark every visible person in the highlighted area."},
    detection_gold_dense_region: {label: "Crowded people", eyebrow: "TRACE PEOPLE", question: "Trace the visible shape of each person in this crowded area."},
    detection_gold_temporal_player: {label: "Person over time", eyebrow: "CHECK EACH FRAME", question: "Follow the same person through this short set of frames."},
    detection_gold_pitch_boundary: {label: "Playing-field position", eyebrow: "MARK THE FEET", question: "Decide where this person's feet are relative to the playing field."},
    detection_gold_football_burst: {label: "Football over time", eyebrow: "FIND THE FOOTBALL", question: "Check the full image for the football in each frame."},
  };
  const LAYERS = ["RAW", "CONFIDENCE", "PRE_NMS", "POST_NMS", "FUSED"];
  const C2_CLIENT_BUILD_ID = "m5_5g1a_r3_r4_c2_pitch_boundary_v1";
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
    novice: false,
    incrementalR3: false,
    tranches: {},
    trancheOrder: [],
    currentTrancheId: null,
    wizard: null,
    lastCaseId: null,
    positionsByCase: {},
    revisionAwareR3R1: false,
    clientBuildId: null,
    indexedDbNamespace: null,
    firstLoadReconciliation: null,
    completionReplayActive: false,
    completionReplayTimer: null,
    c2PitchBoundary: false,
  };

  const byId = (id) => document.getElementById(id);
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const activeCases = () => {
    if (!runtime.incrementalR3) return runtime.manifest.cases;
    const caseIds = new Set(runtime.tranches[runtime.currentTrancheId]?.case_ids || []);
    return runtime.manifest.cases.filter((caseData) => caseIds.has(caseData.case_id));
  };
  function refreshTrancheOptions() {
    if (!runtime.incrementalR3) return;
    const select = byId("dgTrancheSelect");
    select.replaceChildren();
    for (const trancheId of runtime.trancheOrder) {
      const tranche = runtime.tranches[trancheId];
      const reviewed = tranche.case_ids.filter((caseId) => runtime.state.annotations?.[caseId]).length;
      const completed = Boolean(runtime.state.tranche_completions?.[trancheId]);
      const option = document.createElement("option");
      option.value = trancheId;
      option.textContent = `${tranche.label} - ${completed ? "completed" : `${reviewed}/${tranche.case_ids.length} saved`}`;
      option.selected = trancheId === runtime.currentTrancheId;
      select.appendChild(option);
    }
  }
  const currentCase = () => activeCases()[runtime.activeIndex];
  const records = () => currentCase()?.visible_metadata?.frame_records || [];
  const currentRecord = () => records()[runtime.frameIndex] || records()[0];
  const moduleInfo = () => MODULES[currentCase()?.task_type] || {label: "Unknown", eyebrow: "REVIEW"};
  const evidenceUrl = (relativePath) =>
    `/evidence/${encodeURIComponent(currentCase().case_id)}/${relativePath.split("/").map(encodeURIComponent).join("/")}`;

  function c2PitchBoundary(caseData = currentCase()) {
    return runtime.c2PitchBoundary && caseData?.task_type === "detection_gold_pitch_boundary";
  }

  function staticFrameLocked(caseData = currentCase()) {
    return runtime.incrementalR3 && (["detection_gold_player_static", "detection_gold_dense_region"].includes(caseData?.task_type)
      || c2PitchBoundary(caseData));
  }

  function authoritativeBinding(caseData = currentCase()) {
    return runtime.uiConfig?.question_contract?.static_authoritative_bindings?.[caseData?.case_id] || null;
  }

  function authoritativeFrameIndex(caseData = currentCase()) {
    if (!staticFrameLocked(caseData)) return runtime.wizard?.initialFrameIndex(caseData) || 0;
    const binding = authoritativeBinding(caseData);
    const index = (caseData.visible_metadata.frame_records || []).findIndex((row) =>
      Number(row.frame_sequence) === Number(binding?.frame_sequence)
      && row.source_frame_sha256 === binding?.source_frame_sha256
    );
    if (index < 0) throw new Error("Authoritative static frame is not present in the immutable case evidence.");
    return index;
  }

  function authoritativeCandidateUuids(caseData = currentCase()) {
    if (!staticFrameLocked(caseData)) return [...(caseData.visible_metadata.candidate_uuids || [])];
    return [...(authoritativeBinding(caseData)?.candidate_uuids || [])];
  }

  function uid(prefix) {
    return `${prefix}-${Date.now().toString(36)}-${crypto.randomUUID()}`;
  }

  function openDatabase() {
    return new Promise((resolve, reject) => {
      const databaseName = runtime.indexedDbNamespace || `fi_detection_gold_${runtime.manifest.review_id}`;
      const request = indexedDB.open(databaseName, runtime.incrementalR3 ? 2 : 1);
      request.onupgradeneeded = () => {
        const database = request.result;
        if (!database.objectStoreNames.contains("outbox")) database.createObjectStore("outbox", {keyPath: "client_event_id"});
        if (!database.objectStoreNames.contains("drafts")) database.createObjectStore("drafts", {keyPath: "case_id"});
        if (runtime.incrementalR3 && !database.objectStoreNames.contains("session")) database.createObjectStore("session", {keyPath: "key"});
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
  const dbGet = (store, key) => dbRequest(store, "readonly", (target) => target.get(key));
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
    return ["detection_gold_player_static", "detection_gold_dense_region"].includes(currentCase()?.task_type)
      || c2PitchBoundary();
  }

  function requireFocalAnnotationScope() {
    if (!focalScopeApplies() || runtime.view === "focal") return true;
    showError("Static and dense annotations apply to the focal ROI only. Switch to Focal before drawing or editing.");
    return false;
  }

  function setTool(tool) {
    runtime.tool = tool;
    runtime.redrawVisibleObjectUuid = null;
    document.querySelectorAll("[data-dg-tool]").forEach((item) => {
      item.classList.toggle("active", item.dataset.dgTool === tool);
    });
    const labels = {
      box: "Draw the visible person on the image.",
      footpoint: "Click where the feet touch the ground.",
      mask: "Click around the visible outline, then finish the outline.",
      ball: "Click the centre of the football.",
      trail: "Drag from one end of the blur trail to the other.",
    };
    if (labels[tool]) setSaveState(labels[tool], false);
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
      if (c2PitchBoundary(caseData)) {
        return {
          schema_version: "m5_5g1a_c2_pitch_boundary_v1",
          source_binding: binding,
          visible_person_count: 0,
          player_instances: [],
          candidate_relations: [],
          note: "",
        };
      }
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
    runtime.positionsByCase[caseData.case_id] = {
      frame_index: runtime.frameIndex,
      view: runtime.view,
      selected_candidate_uuid: runtime.selectedCandidate?.diagnostic_uuid || null,
    };
    await dbPut("drafts", {
      case_id: caseData.case_id,
      annotation: draft(),
      geometry_drafts: clone(runtime.geometryDrafts[caseData.case_id] || {}),
      selected_annotation_uuid: runtime.selectedObjectByCase[caseData.case_id] || null,
      wizard_state: runtime.wizard?.snapshot(caseData.case_id) || null,
      position: clone(runtime.positionsByCase[caseData.case_id]),
      undo_history: clone(runtime.history.filter((row) => row.case_id === caseData.case_id).slice(-20)),
      updated_at: new Date().toISOString(),
    });
    if (runtime.incrementalR3) {
      await dbPut("session", {
        key: "navigation",
        current_tranche_id: runtime.currentTrancheId,
        case_id: caseData.case_id,
        frame_index: runtime.frameIndex,
        wizard_step: runtime.wizard?.state()?.step || 1,
        updated_at: new Date().toISOString(),
      });
    }
    setSaveState("Draft stored locally", false);
  }

  function pushHistory() {
    runtime.history.push({
      case_id: currentCase().case_id,
      annotation: clone(draft()),
      selected_annotation_uuid: runtime.selectedObjectByCase[currentCase().case_id] || null,
      wizard_state: runtime.wizard?.snapshot(currentCase().case_id) || null,
    });
    if (runtime.history.length > 60) runtime.history.shift();
  }

  function undo() {
    const previous = runtime.history.pop();
    if (!previous || previous.case_id !== currentCase().case_id) return;
    runtime.drafts[previous.case_id] = previous.annotation;
    if (runtime.wizard && previous.wizard_state) runtime.wizard.replace(previous.case_id, previous.wizard_state);
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
    refreshTrancheOptions();
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

  function focusSelectedPerson() {
    const person = selectedObject();
    if (!person?.visible_body_box) return;
    const bounds = getBounds();
    const viewport = byId("dgViewport").getBoundingClientRect();
    const centreX = (person.visible_body_box.x1 + person.visible_body_box.x2) / 2 - bounds.x1;
    const centreY = (person.visible_body_box.y1 + person.visible_body_box.y2) / 2 - bounds.y1;
    const scale = 4;
    runtime.viewTransform = {
      scale,
      x: viewport.width / 2 - (centreX / (bounds.x2 - bounds.x1)) * viewport.width * scale,
      y: viewport.height / 2 - (centreY / (bounds.y2 - bounds.y1)) * viewport.height * scale,
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

  function clipPolygonToBounds(points, bounds) {
    const clip = (input, inside, intersection) => {
      if (!input.length) return [];
      const output = [];
      let previous = input[input.length - 1];
      let previousInside = inside(previous);
      input.forEach((current) => {
        const currentInside = inside(current);
        if (currentInside) {
          if (!previousInside) output.push(intersection(previous, current));
          output.push(current);
        } else if (previousInside) {
          output.push(intersection(previous, current));
        }
        previous = current;
        previousInside = currentInside;
      });
      return output;
    };
    const verticalIntersection = (boundary) => (start, end) => {
      const delta = end.x - start.x;
      const ratio = Math.abs(delta) < 1e-12 ? 0 : (boundary - start.x) / delta;
      return {x: boundary, y: start.y + ratio * (end.y - start.y)};
    };
    const horizontalIntersection = (boundary) => (start, end) => {
      const delta = end.y - start.y;
      const ratio = Math.abs(delta) < 1e-12 ? 0 : (boundary - start.y) / delta;
      return {x: start.x + ratio * (end.x - start.x), y: boundary};
    };
    let output = points.map((point) => ({x: Number(point.x), y: Number(point.y)}));
    output = clip(output, (point) => point.x >= bounds.x1, verticalIntersection(bounds.x1));
    output = clip(output, (point) => point.x <= bounds.x2, verticalIntersection(bounds.x2));
    output = clip(output, (point) => point.y >= bounds.y1, horizontalIntersection(bounds.y1));
    return clip(output, (point) => point.y <= bounds.y2, horizontalIntersection(bounds.y2));
  }

  function clipSegmentToBounds(start, end, bounds) {
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    let lower = 0;
    let upper = 1;
    const tests = [
      [-dx, start.x - bounds.x1],
      [dx, bounds.x2 - start.x],
      [-dy, start.y - bounds.y1],
      [dy, bounds.y2 - start.y],
    ];
    for (const [direction, distance] of tests) {
      if (Math.abs(direction) < 1e-12) {
        if (distance < 0) return null;
        continue;
      }
      const ratio = distance / direction;
      if (direction < 0) lower = Math.max(lower, ratio);
      else upper = Math.min(upper, ratio);
      if (lower > upper) return null;
    }
    return [
      {x: start.x + lower * dx, y: start.y + lower * dy},
      {x: start.x + upper * dx, y: start.y + upper * dy},
    ];
  }

  function projectPitchPolygon(points) {
    const bounds = getBounds();
    const sourcePoints = points.map((point) => ({x: Number(point.x), y: Number(point.y)}));
    const fillSource = runtime.view === "focal" ? clipPolygonToBounds(sourcePoints, bounds) : sourcePoints;
    const segments = [];
    sourcePoints.forEach((start, index) => {
      const end = sourcePoints[(index + 1) % sourcePoints.length];
      const clipped = runtime.view === "focal" ? clipSegmentToBounds(start, end, bounds) : [start, end];
      if (clipped) segments.push(clipped.map(pointToView));
    });
    return {
      fill: fillSource.map(pointToView),
      segments,
    };
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
    const novicePolicy = runtime.wizard?.overlayPolicy() || null;
    const row = currentRecord();
    const noviceCandidateNodes = [];
    let noviceCandidateRendered = false;
    for (const candidate of row?.candidates || []) {
      if (runtime.novice) {
        if (!novicePolicy?.showMachine || candidate.diagnostic_uuid !== novicePolicy.candidateUuid) continue;
        if (noviceCandidateRendered) continue;
        noviceCandidateRendered = true;
      } else if (!enabled.has(candidate.stage)) continue;
      const box = toViewBox(candidate.bbox_original_pixels);
      const rectangle = makeSvg("rect", {
        x: box.x1,
        y: box.y1,
        width: box.x2 - box.x1,
        height: box.y2 - box.y1,
        class: `dgProposal dgLayer${candidate.stage} ${runtime.selectedCandidate?.diagnostic_uuid === candidate.diagnostic_uuid ? "selected" : ""} ${runtime.novice ? "dgNoviceCandidate" : ""}`,
        tabindex: 0,
      });
      rectangle.addEventListener("click", (event) => {
        if (runtime.tool !== "select" || (runtime.novice && novicePolicy?.humanInteractive === false)) return;
        event.stopPropagation();
        runtime.selectedCandidate = candidate;
        renderOverlay();
        renderProvenance();
        renderAnnotationForm();
      });
      if (runtime.novice) noviceCandidateNodes.push(rectangle);
      else svg.appendChild(rectangle);
      if (runtime.novice) {
        const viewWidth = bounds.x2 - bounds.x1;
        const anchorAtRight = box.x1 > viewWidth * 0.65;
        const label = makeSvg("text", {
          x: anchorAtRight ? box.x2 : box.x1 + 4,
          y: box.y2 + 18 <= bounds.y2 - bounds.y1 ? box.y2 + 18 : Math.max(16, box.y1 - 6),
          class: "dgNoviceCandidateLabel",
          "text-anchor": anchorAtRight ? "end" : "start",
        });
        label.textContent = novicePolicy?.candidateLabel || "Machine suggestion";
        noviceCandidateNodes.push(label);
      }
    }
    const annotation = draft();
    const selectedUuid = selectedObject(annotation)?.annotation_uuid;
    const appendSelectableBox = (rawBox, item, geometryClass = "", objectIndex = null) => {
      if (!rawBox) return;
      const box = toViewBox(rawBox);
      const rectangle = makeSvg("rect", {
        x: box.x1,
        y: box.y1,
        width: box.x2 - box.x1,
        height: box.y2 - box.y1,
        class: `dgHumanBox ${geometryClass} ${selectedUuid === item.annotation_uuid ? "selectedObject" : ""} ${runtime.novice && novicePolicy?.humanInteractive === false ? "dgCandidateReference" : ""}`,
        "data-dg-object-uuid": item.annotation_uuid,
        tabindex: 0,
      });
      rectangle.addEventListener("click", (event) => {
        if (runtime.tool !== "select" || (runtime.novice && novicePolicy?.humanInteractive === false)) return;
        event.stopPropagation();
        selectObject(item.annotation_uuid);
      });
      svg.appendChild(rectangle);
      if (runtime.novice && Number.isInteger(objectIndex) && geometryClass === "dgVisibleBodyBox") {
        const label = makeSvg("text", {
          x: box.x1 + 4,
          y: Math.max(16, box.y1 - 6),
          class: "dgNovicePersonLabel",
        });
        label.textContent = `Person ${objectIndex + 1}`;
        svg.appendChild(label);
      }
    };
    for (const [index, item] of (annotation.player_instances || []).entries()) {
      appendSelectableBox(item.visible_body_box, item, "dgVisibleBodyBox", index);
      appendSelectableBox(item.full_body_box, item, "dgSupplementaryBox");
      appendSelectableBox(item.optional_head_box, item, "dgHeadBox");
      if (item.footpoint && (!runtime.novice || novicePolicy?.footpointUuids?.includes(item.annotation_uuid))) {
        const point = pointToView(item.footpoint);
        if (Number(item.footpoint_uncertainty_pixels) > 0) {
          svg.appendChild(makeSvg("circle", {
            cx: point.x,
            cy: point.y,
            r: Number(item.footpoint_uncertainty_pixels),
            class: "dgFootpointUncertainty",
          }));
        }
        svg.appendChild(makeSvg("circle", {
          cx: point.x,
          cy: point.y,
          r: 4,
          class: `dgFootpoint ${selectedUuid === item.annotation_uuid ? "selectedObject" : ""}`,
        }));
        if (novicePolicy?.estimatedFootpointUuids?.includes(item.annotation_uuid)) {
          const label = makeSvg("text", {
            x: point.x + 8,
            y: Math.max(14, point.y - 8),
            class: "dgEstimatedFootpointLabel",
          });
          label.textContent = "Estimated because the feet are not visible";
          svg.appendChild(label);
        }
      }
    }
    for (const [index, item] of (annotation.visible_masks || []).entries()) {
      appendSelectableBox(item.visible_body_box, item, "dgVisibleBodyBox", index);
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
      if (frame?.footpoint && (!runtime.novice || novicePolicy?.showTemporalFootpoint)) {
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
    if (annotation.footpoint && (!runtime.novice || novicePolicy?.pitchFootpointSet)) {
      const point = pointToView(annotation.footpoint);
      if (Number(annotation.footpoint_uncertainty_pixels) > 0) {
        svg.appendChild(makeSvg("circle", {
          cx: point.x,
          cy: point.y,
          r: Number(annotation.footpoint_uncertainty_pixels),
          class: "dgFootpointUncertainty",
        }));
      }
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
      const projection = projectPitchPolygon(pitchPolygon);
      const group = makeSvg("g", {class: "dgPitchProjection"});
      group.style.pointerEvents = "none";
      if (projection.fill.length >= 3) {
        const fillPoints = projection.fill.map((point) => `${point.x},${point.y}`).join(" ");
        group.appendChild(makeSvg("polygon", {points: fillPoints, class: "dgPitchPolygonFill"}));
      }
      const boundaryPath = projection.segments
        .map(([start, end]) => `M ${start.x} ${start.y} L ${end.x} ${end.y}`)
        .join(" ");
      if (c2PitchBoundary() && boundaryPath) {
        const tolerancePixels = Number(currentCase().visible_metadata.pitch_polygon_tolerance_pixels || 10);
        const toleranceBand = makeSvg("path", {
          d: boundaryPath,
          class: "dgPitchToleranceBand",
          "stroke-width": tolerancePixels * 2,
        });
        toleranceBand.style.pointerEvents = "none";
        group.appendChild(toleranceBand);
      }
      if (boundaryPath) {
        const boundary = makeSvg("path", {d: boundaryPath, class: "dgPitchPolygon", "stroke-width": 2});
        boundary.style.pointerEvents = "none";
        group.appendChild(boundary);
      }
      svg.insertBefore(group, svg.firstChild);
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
    noviceCandidateNodes.forEach((node) => svg.appendChild(node));
  }

  function candidateInstance(candidate, boxOverride = null) {
    const box = clone(boxOverride || candidate.bbox_original_pixels);
    if (c2PitchBoundary()) {
      return {
        annotation_uuid: uid("person"),
        visible_body_box: box,
        footpoint: null,
        footpoint_status: "CANNOT_TELL",
        footpoint_uncertainty_pixels: 20,
        pitch_state: "BOUNDARY_UNCERTAIN",
        pitch_state_certainty: "UNCERTAIN",
        coarse_role: "UNKNOWN",
        minimum_visible_dimensions: {width_pixels: box.x2 - box.x1, height_pixels: box.y2 - box.y1},
      };
    }
    return {
      annotation_uuid: uid("person"),
      visible_body_box: box,
      footpoint: {x: (box.x1 + box.x2) / 2, y: box.y2},
      footpoint_uncertainty_pixels: runtime.incrementalR3 ? 3 : 8,
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

  function estimateHiddenFootpoint(person) {
    const binding = draft().source_binding;
    const box = person.visible_body_box;
    const visibleHeight = Math.max(2, Number(box.y2) - Number(box.y1));
    person.footpoint = {
      x: (Number(box.x1) + Number(box.x2)) / 2,
      y: Math.min(Number(binding.image_height), Number(box.y2) + Math.max(4, visibleHeight * 0.35)),
    };
    if (Math.abs(person.footpoint.y - Number(box.y2)) < 0.5) {
      person.footpoint.y = Math.max(0, Number(box.y2) - Math.max(4, visibleHeight * 0.2));
    }
    person.footpoint_uncertainty_pixels = Math.max(20, Math.round(visibleHeight * 0.5));
    return clone(person.footpoint);
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
    if (currentCase().task_type === "detection_gold_player_static" || c2PitchBoundary()) {
      annotation.player_instances.push(instance);
      annotation.visible_person_count = annotation.player_instances.length;
      runtime.selectedObjectByCase[currentCase().case_id] = instance.annotation_uuid;
    } else if (currentCase().task_type === "detection_gold_temporal_player") {
      annotation.stable_run_accepted = false;
      annotation.frames[runtime.frameIndex] = {
        frame_sequence: currentRecord().frame_sequence,
        source_frame_sha256: currentRecord().source_frame_sha256,
        state: runtime.novice ? (runtime.wizard.state().desired_frame_state || "OBSERVED") : "OBSERVED",
        visible_body_box: instance.visible_body_box,
        footpoint: instance.footpoint,
        current_frame_pixel_support: true,
        candidate_uuids: selectedMachineCandidate ? [selectedMachineCandidate.diagnostic_uuid] : [],
      };
    }
    persistDraft();
    if (runtime.novice) {
      runtime.wizard.objectCreated(instance.annotation_uuid);
      return;
    }
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
    if (runtime.novice) {
      runtime.wizard.objectCreated(maskId);
      return;
    }
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
      if (currentCase().task_type === "detection_gold_pitch_boundary" && !c2PitchBoundary()) {
        annotation.footpoint = point;
      } else if (currentCase().task_type === "detection_gold_temporal_player") {
        const frame = annotation.frames[runtime.frameIndex];
        if (!frame?.visible_body_box) return showError("Mark the visible person before placing the foot point.");
        frame.footpoint = point;
      } else {
        const person = selectedObject(annotation);
        if (!person || !annotation.player_instances) return showError("Select a person before placing its footpoint.");
        person.footpoint = point;
        if (c2PitchBoundary()) {
          person.footpoint_status = "OBSERVED_APPROXIMATE";
          person.footpoint_uncertainty_pixels = Math.max(8, Number(person.footpoint_uncertainty_pixels || 8));
        }
      }
      persistDraft();
      if (runtime.novice) {
        runtime.wizard.footpointPlaced();
        return;
      }
    } else if (runtime.tool === "mask") {
      runtime.maskPoints.push(point);
      renderAnnotationForm();
    } else if (runtime.tool === "ball" && currentCase().task_type === "detection_gold_football_burst") {
      pushHistory();
      annotation.frames[runtime.frameIndex] = {
        frame_sequence: currentRecord().frame_sequence,
        source_frame_sha256: currentRecord().source_frame_sha256,
        state: runtime.novice ? (runtime.wizard.state().desired_frame_state || "VISIBLE_CLEAR") : "VISIBLE_CLEAR",
        centre_point: point,
        geometry_uncertainty_pixels: 3,
      };
      persistDraft();
      if (runtime.novice) {
        runtime.wizard.footballPointPlaced();
        return;
      }
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
        runtime.wizard?.objectGeometryChanged(target.annotation_uuid);
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
    if (runtime.novice && runtime.tool === "trail") {
      runtime.wizard.footballTrailDrawn();
      return;
    }
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

  function renderC2PitchForm(annotation) {
    const selected = selectedObject(annotation);
    return `<section class="dgC2Advanced">
      <strong>${annotation.visible_person_count} visible people</strong>
      <p>Pitch state and role are human labels. They never determine which machine box belongs to a person.</p>
      <div class="dgInstanceList">${annotation.player_instances.map((person, index) => `<button data-dg-object-select="${person.annotation_uuid}" class="${selected?.annotation_uuid === person.annotation_uuid ? "selected" : ""}" type="button"><strong>Person ${index + 1}</strong><span>${person.coarse_role} | ${person.pitch_state} | ${person.footpoint_status}</span></button>`).join("") || "<p>No visible people marked.</p>"}</div>
      <button id="dgC2FocusPerson" type="button" ${selected ? "" : "disabled"}>Focus selected person</button>
      <button id="dgRemoveSelected" type="button" ${selected ? "" : "disabled"}>Remove selected person</button>
      ${candidateBindingControls(annotation)}
      ${baseNote(annotation)}
    </section>`;
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

  function invalidateAdvancedPersonEdit(person, {candidateRelevant = true} = {}) {
    if (!person || !runtime.revisionAwareR3R1) return;
    runtime.wizard?.objectSemanticChanged(person.annotation_uuid, {
      candidateRelevant,
      reopenQuestions: true,
    });
  }

  function invalidateAdvancedCandidateEdit(candidateUuid, reason) {
    if (!candidateUuid || !runtime.revisionAwareR3R1) return;
    runtime.wizard?.candidateAnswerEdited(candidateUuid, reason);
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
    annotation.candidate_relations = (annotation.candidate_relations || []).map((row) => {
      if (!row.annotation_uuids.includes(selected.annotation_uuid)) return row;
      const updated = {...row, annotation_uuids: row.annotation_uuids.filter(
        (uuid) => uuid !== selected.annotation_uuid
      )};
      if (!updated.annotation_uuids.length) delete updated.candidate_visible_mask_coverage;
      return updated;
    });
    const replacement = annotationObjects(annotation)[0];
    if (replacement) runtime.selectedObjectByCase[currentCase().case_id] = replacement.annotation_uuid;
    else delete runtime.selectedObjectByCase[currentCase().case_id];
    runtime.redrawVisibleObjectUuid = null;
    if (runtime.novice) {
      runtime.wizard.objectDeleted(selected.annotation_uuid);
    }
    persistDraft();
    if (runtime.novice) renderCase();
    else {
      renderAnnotationForm();
      renderOverlay();
    }
    if (affected.length) showError("Some machine-box answers need checking again because a person was removed.");
  }

  function deleteAllAnnotations() {
    const annotation = draft();
    const objects = annotationObjects(annotation);
    if (!objects.length) return;
    if (!window.confirm(`Delete all ${objects.length} marked people from this unsaved case? Saved cases and other tranches will not change.`)) return;
    pushHistory();
    const deletedUuids = objects.map((row) => row.annotation_uuid);
    if (annotation.player_instances) {
      annotation.player_instances = [];
      annotation.visible_person_count = 0;
    } else {
      annotation.visible_masks = [];
      annotation.human_visible_person_count = 0;
    }
    annotation.candidate_relations = (annotation.candidate_relations || []).map((row) => {
      if (!row.annotation_uuids.some((uuid) => deletedUuids.includes(uuid))) return row;
      const updated = {...row, annotation_uuids: []};
      delete updated.candidate_visible_mask_coverage;
      return updated;
    });
    delete runtime.selectedObjectByCase[currentCase().case_id];
    runtime.redrawVisibleObjectUuid = null;
    runtime.wizard?.allObjectsDeleted(deletedUuids);
    persistDraft();
    renderCase();
    showError("Start again by drawing the visible people. Earlier machine-box answers need review.");
  }

  async function restartCurrentCase() {
    const caseData = currentCase();
    if ((runtime.state.annotations || {})[caseData.case_id]) {
      showError("This case is already saved on the server and cannot be restarted as an unsaved draft.");
      return;
    }
    if (!window.confirm("Restart this unsaved case? Only this browser draft will be cleared. Saved cases and tranche completion will remain unchanged.")) return;
    runtime.drafts[caseData.case_id] = defaultAnnotation(caseData);
    runtime.geometryDrafts[caseData.case_id] = {};
    delete runtime.positionsByCase[caseData.case_id];
    delete runtime.selectedObjectByCase[caseData.case_id];
    runtime.history = runtime.history.filter((row) => row.case_id !== caseData.case_id);
    runtime.selectedCandidate = null;
    runtime.maskPoints = [];
    runtime.redrawVisibleObjectUuid = null;
    runtime.frameIndex = authoritativeFrameIndex(caseData);
    runtime.view = "focal";
    runtime.wizard?.reset(caseData.case_id);
    await dbDelete("drafts", caseData.case_id);
    await dbPut("session", {
      key: "navigation",
      current_tranche_id: runtime.currentTrancheId,
      case_id: caseData.case_id,
      frame_index: runtime.frameIndex,
      wizard_step: 1,
      updated_at: new Date().toISOString(),
    });
    runtime.lastCaseId = null;
    renderCase();
    setSaveState("This case restarted from a clean Step 1 draft", false);
  }

  function renderAnnotationForm() {
    const annotation = draft();
    const task = currentCase().task_type;
    const form = byId("dgAnnotationForm");
    let legacyMarkup;
    if (task === "detection_gold_player_static") legacyMarkup = renderPlayerForm(annotation);
    else if (task === "detection_gold_dense_region") legacyMarkup = renderDenseForm(annotation);
    else if (task === "detection_gold_temporal_player") legacyMarkup = renderTemporalForm(annotation);
    else if (task === "detection_gold_pitch_boundary") legacyMarkup = c2PitchBoundary()
      ? renderC2PitchForm(annotation)
      : renderPitchForm(annotation);
    else legacyMarkup = renderFootballForm(annotation);
    form.innerHTML = runtime.novice
      ? `${runtime.wizard.render(annotation)}<details class="nwAdvancedDetails"><summary>Advanced details</summary><div class="nwAdvancedBody">${legacyMarkup}</div></details>`
      : legacyMarkup;
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
    byId("dgC2FocusPerson")?.addEventListener("click", focusSelectedPerson);
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
      invalidateAdvancedCandidateEdit(
        runtime.selectedCandidate.diagnostic_uuid,
        "A machine-box relation changed in advanced details and needs guided confirmation.",
      );
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
        invalidateAdvancedCandidateEdit(
          runtime.selectedCandidate.diagnostic_uuid,
          "Machine-box targets changed in advanced details and need guided confirmation.",
        );
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
      invalidateAdvancedCandidateEdit(
        runtime.selectedCandidate.diagnostic_uuid,
        "Machine-box mask coverage changed in advanced details and needs guided confirmation.",
      );
      persistDraft();
    });
    byId("dgClearCandidateBinding")?.addEventListener("click", () => {
      if (!runtime.selectedCandidate) return;
      pushHistory();
      annotation.candidate_relations = (annotation.candidate_relations || []).filter(
        (row) => row.candidate_uuid !== runtime.selectedCandidate.diagnostic_uuid
      );
      invalidateAdvancedCandidateEdit(
        runtime.selectedCandidate.diagnostic_uuid,
        "A machine-box binding was cleared and must be reviewed again.",
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
        invalidateAdvancedCandidateEdit(
          candidateUuid,
          "A machine box was marked as background in advanced details and needs guided confirmation.",
        );
      }
      persistDraft(); renderAnnotationForm(); renderOverlay();
    });
    byId("dgFailureStage")?.addEventListener("change", (event) => { annotation.earliest_failure_stage = event.target.value; persistDraft(); });
    byId("dgFinishMask")?.addEventListener("click", finishMask);
    byId("dgDenseIgnore")?.addEventListener("change", (event) => { annotation.uncertain_or_ignore = event.target.checked; persistDraft(); });
    byId("dgMaskQuality")?.addEventListener("change", (event) => { const mask = selectedObject(annotation); if (mask) { invalidateAdvancedPersonEdit(mask); mask.mask_quality = event.target.value; } persistDraft(); renderAnnotationForm(); renderOverlay(); });
    byId("dgOcclusionOrder")?.addEventListener("input", (event) => { const mask = selectedObject(annotation); if (mask) { invalidateAdvancedPersonEdit(mask); mask.occlusion_order = Number(event.target.value); } persistDraft(); });
    byId("dgOccluderUuid")?.addEventListener("input", (event) => { const mask = selectedObject(annotation); if (!mask) return; invalidateAdvancedPersonEdit(mask); if (event.target.value) mask.occluder_uuid = event.target.value; else delete mask.occluder_uuid; persistDraft(); });
    for (const toggle of document.querySelectorAll("[data-dg-overlap-uuid]")) {
      toggle.addEventListener("change", () => {
        const mask = selectedObject(annotation);
        if (!mask) return;
        invalidateAdvancedPersonEdit(mask);
        const selected = new Set(mask.pairwise_overlap_annotation_uuids || []);
        if (toggle.checked) selected.add(toggle.dataset.dgOverlapUuid); else selected.delete(toggle.dataset.dgOverlapUuid);
        mask.pairwise_overlap_annotation_uuids = [...selected].sort();
        persistDraft();
      });
    }
    byId("dgPersonVisibility")?.addEventListener("change", (event) => { const person = selectedObject(annotation); if (person) { invalidateAdvancedPersonEdit(person); person.visibility_state = event.target.value; } persistDraft(); renderAnnotationForm(); renderOverlay(); });
    byId("dgOcclusionType")?.addEventListener("change", (event) => { const person = selectedObject(annotation); if (person) { invalidateAdvancedPersonEdit(person); person.occlusion_type = event.target.value; } persistDraft(); renderAnnotationForm(); });
    byId("dgOcclusionFraction")?.addEventListener("input", (event) => { const person = selectedObject(annotation); if (person) { invalidateAdvancedPersonEdit(person); person.occlusion_fraction = Number(event.target.value); } persistDraft(); });
    byId("dgPersonFootpointUncertainty")?.addEventListener("input", (event) => { const person = selectedObject(annotation); if (person) person.footpoint_uncertainty_pixels = Number(event.target.value); persistDraft(); });
    byId("dgPersonPitchState")?.addEventListener("change", (event) => { const person = selectedObject(annotation); if (person) { invalidateAdvancedPersonEdit(person, {candidateRelevant: false}); person.pitch_state = event.target.value; } persistDraft(); renderAnnotationForm(); });
    byId("dgPersonRole")?.addEventListener("change", (event) => { const person = selectedObject(annotation); if (person) { invalidateAdvancedPersonEdit(person, {candidateRelevant: false}); person.coarse_role = event.target.value; } persistDraft(); renderAnnotationForm(); });
    byId("dgPersonAmbiguityIgnore")?.addEventListener("change", (event) => { const person = selectedObject(annotation); if (person) { invalidateAdvancedPersonEdit(person); person.ambiguity_ignore = event.target.checked; } persistDraft(); });
    for (const toggle of document.querySelectorAll("[data-dg-truncation]")) {
      toggle.addEventListener("change", () => {
        const person = selectedObject(annotation);
        if (!person) return;
        invalidateAdvancedPersonEdit(person);
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
    runtime.wizard?.bind();
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
    if (runtime.novice) {
      runtime.wizard.geometryCopiedToFrame(nextIndex);
      return;
    }
    persistDraft();
    setFrame(nextIndex);
  }

  function confirmGeometryDraft() {
    const geometry = currentGeometryDraft();
    if (!geometry) return false;
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
    if (!runtime.novice) {
      persistDraft();
      renderAnnotationForm();
      renderOverlay();
    }
    return true;
  }

  function rejectGeometryDraft() {
    if (!currentGeometryDraft()) return false;
    delete caseGeometryDrafts()[String(currentRecord().frame_sequence)];
    if (!runtime.novice) {
      persistDraft();
      renderAnnotationForm();
      renderOverlay();
    }
    return true;
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
    if (byId("nwSaveCase")) byId("nwSaveCase").disabled = true;
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
      if (byId("nwSaveCase")) byId("nwSaveCase").disabled = false;
    } catch (error) {
      if (token !== runtime.evidenceToken) return;
      byId("dgEvidenceBlocker").classList.remove("isHidden");
      byId("dgEvidenceBlocker").textContent = `Evidence unavailable: ${error.message}`;
      byId("dgEvidenceStatus").textContent = "Evidence blocked";
      runtime.evidenceBlocked = true;
      if (byId("nwSaveCase")) byId("nwSaveCase").disabled = true;
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
      if (staticFrameLocked()) {
        button.disabled = true;
        button.title = index === authoritativeFrameIndex() ? "Authoritative middle frame" : "Reference image only";
      } else {
        button.addEventListener("click", () => setFrame(index));
      }
      container.appendChild(button);
    });
    container.classList.toggle("isCompact", records().length <= 3);
  }

  function renderProgress() {
    const counts = runtime.state.counts || {};
    const allReviewed = counts.reviewed ?? Object.keys(runtime.state.annotations || {}).length;
    const cases = activeCases();
    const tranche = counts.tranches?.[runtime.currentTrancheId];
    const reviewed = runtime.incrementalR3
      ? (tranche?.reviewed ?? cases.filter((row) => runtime.state.annotations?.[row.case_id]).length)
      : allReviewed;
    byId("dgCaseProgress").textContent = runtime.incrementalR3
      ? `Case ${runtime.activeIndex + 1} of ${cases.length} in ${runtime.tranches[runtime.currentTrancheId]?.label || runtime.currentTrancheId}`
      : `Case ${runtime.activeIndex + 1} of ${cases.length}`;
    byId("dgModuleProgress").textContent = `${moduleInfo().label} | ${reviewed} saved`;
    byId("dgProgressBar").style.width = `${cases.length ? (reviewed / cases.length) * 100 : 0}%`;
    if (runtime.incrementalR3) {
      const completed = Boolean(runtime.state.tranche_completions?.[runtime.currentTrancheId]);
      byId("dgCompleteTranche").disabled = completed || reviewed !== cases.length || runtime.outbox.length > 0;
      byId("dgTrancheStatus").textContent = completed ? "Tranche completed" : `${reviewed}/${cases.length} saved`;
    }
    const allTranchesComplete = !runtime.incrementalR3
      || runtime.trancheOrder.every((trancheId) => runtime.state.tranche_completions?.[trancheId]);
    byId("dgComplete").disabled = Boolean(runtime.state.completed)
      || allReviewed !== runtime.manifest.cases.length
      || runtime.outbox.length > 0
      || !allTranchesComplete;
  }

  function setFrameSilently(index) {
    runtime.frameIndex = staticFrameLocked()
      ? authoritativeFrameIndex()
      : Math.max(0, Math.min(index, records().length - 1));
    if (byId("dgTimeline")) byId("dgTimeline").value = String(runtime.frameIndex);
  }

  function currentFrameCandidates(className = null) {
    const required = new Set(authoritativeCandidateUuids());
    return (currentRecord()?.candidates || [])
      .filter((candidate) => (!className || candidate.class_name === className)
        && (!required.size || required.has(candidate.diagnostic_uuid)))
      .sort((left, right) => Number(right.score || 0) - Number(left.score || 0)
        || left.diagnostic_uuid.localeCompare(right.diagnostic_uuid));
  }

  function setFirstFrameCandidate(className = null) {
    runtime.selectedCandidate = currentFrameCandidates(className)[0] || null;
  }

  function nextFrameCandidate() {
    const candidates = currentFrameCandidates("person");
    if (!candidates.length) return;
    const current = candidates.findIndex(
      (candidate) => candidate.diagnostic_uuid === runtime.selectedCandidate?.diagnostic_uuid
    );
    runtime.selectedCandidate = candidates[(current + 1) % candidates.length];
    renderOverlay();
    renderAnnotationForm();
  }

  function useTemporalCandidate(desiredState) {
    if (!runtime.selectedCandidate) setFirstFrameCandidate("person");
    if (!runtime.selectedCandidate) return showError("No machine suggestion is available. Draw the visible person instead.");
    runtime.wizard.state().desired_frame_state = desiredState || "OBSERVED";
    acceptSelectedPerson();
  }

  function upsertCandidateRelation(annotation, candidate, relation, annotationUuids, coverageValue) {
    runtime.selectedCandidate = candidate;
    return upsertSelectedCandidateRelation(annotation, relation, annotationUuids, coverageValue);
  }

  function renderCase() {
    const caseData = currentCase();
    const caseChanged = runtime.lastCaseId !== caseData.case_id;
    if (caseChanged) {
      runtime.selectedCandidate = null;
      runtime.maskPoints = [];
      runtime.redrawVisibleObjectUuid = null;
      const savedPosition = runtime.positionsByCase[caseData.case_id];
      runtime.frameIndex = staticFrameLocked(caseData)
        ? authoritativeFrameIndex(caseData)
        : (savedPosition?.frame_index ?? (runtime.wizard?.initialFrameIndex(caseData) || 0));
      runtime.view = savedPosition?.view || (runtime.novice && caseData.task_type === "detection_gold_football_burst" ? "panorama" : "focal");
      byId("dgFocalView").classList.toggle("active", runtime.view === "focal");
      byId("dgPanoramaView").classList.toggle("active", runtime.view === "panorama");
    }
    runtime.frameIndex = staticFrameLocked(caseData)
      ? authoritativeFrameIndex(caseData)
      : Math.min(runtime.frameIndex, Math.max(0, records().length - 1));
    if (runtime.novice) runtime.wizard.syncCandidate();
    const info = runtime.novice ? NOVICE_MODULES[caseData.task_type] : moduleInfo();
    runtime.lastCaseId = caseData.case_id;
    byId("dgTitle").textContent = runtime.uiConfig.review_title;
    byId("dgModuleEyebrow").textContent = info.eyebrow;
    byId("dgCaseTitle").textContent = `${info.label} ${caseData.visible_metadata.module_case_number}`;
    byId("dgQuestion").textContent = runtime.novice ? info.question : caseData.concise_question;
    byId("dgScopeBadge").textContent = runtime.novice ? "MARK THE HIGHLIGHTED AREA ONLY" : "ANNOTATE FOCAL ROI ONLY";
    byId("dgScopeBadge").classList.toggle("isHidden", !focalScopeApplies());
    byId("dgStaticFrameInstruction").classList.toggle("isHidden", !staticFrameLocked(caseData));
    byId("dgTimelineRow").classList.toggle("isStaticLocked", staticFrameLocked(caseData));
    for (const id of ["dgPreviousFrame", "dgNextFrame", "dgTimeline", "dgPlay"]) byId(id).disabled = staticFrameLocked(caseData);
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
    if (staticFrameLocked()) {
      setFrameSilently(authoritativeFrameIndex());
      setSaveState("Reference frames are view-only; the middle frame remains locked.", false);
      return;
    }
    setFrameSilently(index);
    runtime.selectedCandidate = null;
    runtime.redrawVisibleObjectUuid = null;
    persistDraft();
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

    if (["detection_gold_player_static", "detection_gold_dense_region"].includes(caseData.task_type)
      || c2PitchBoundary(caseData)) {
      if (staticFrameLocked(caseData)) {
        const binding = authoritativeBinding(caseData);
        const row = currentRecord();
        if (runtime.frameIndex !== authoritativeFrameIndex(caseData)
          || Number(row.frame_sequence) !== Number(binding.frame_sequence)
          || row.source_frame_sha256 !== binding.source_frame_sha256
          || annotation.source_binding.source_frame_sha256 !== binding.source_frame_sha256) {
          throw new Error("Static annotation is not bound to the authoritative middle frame.");
        }
      }
      const objects = annotationObjects(annotation);
      const objectIds = new Set(objects.map((item) => item.annotation_uuid));
      const required = new Set(authoritativeCandidateUuids(caseData));
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
      if (c2PitchBoundary(caseData)) {
        if (annotation.visible_person_count !== annotation.player_instances.length) {
          throw new Error("Visible-person count must match every marked C2 person.");
        }
        for (const person of annotation.player_instances) {
          const observed = ["OBSERVED_CLEAR", "OBSERVED_APPROXIMATE"].includes(person.footpoint_status);
          if (observed !== Boolean(person.footpoint)) {
            throw new Error("Observed feet require a point; hidden or unresolved feet must not carry one.");
          }
          if (!person.coarse_role || !person.pitch_state || !person.pitch_state_certainty) {
            throw new Error("Review role, pitch state, and certainty for every visible person.");
          }
        }
      }
    }

    if (caseData.task_type === "detection_gold_temporal_player") {
      const unresolvedAllowed = runtime.uiConfig.question_contract.reviewed_unresolved_states_allowed === true;
      if (!annotation.contact_strip_reviewed || (!unresolvedAllowed && annotation.frames.some((frame) => frame.state === "UNRESOLVED"))) {
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
      const unresolvedAllowed = runtime.uiConfig.question_contract.reviewed_unresolved_states_allowed === true;
      if (!annotation.full_contact_strip_reviewed || (!unresolvedAllowed && annotation.frames.some((frame) => frame.state === "UNRESOLVED"))) {
        throw new Error("Review the full football strip and resolve every frame before saving.");
      }
    }
  }

  async function enqueueSave() {
    clearError();
    if (runtime.evidenceBlocked) return showError("Case saving is blocked until exact evidence verifies.");
    const caseData = currentCase();
    try {
      runtime.wizard?.validateForSave();
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
      wizard_state: runtime.wizard?.snapshot(caseData.case_id) || null,
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
    const cases = activeCases();
    const start = runtime.activeIndex;
    for (let offset = 1; offset <= cases.length; offset += 1) {
      const index = (start + offset) % cases.length;
      if (!annotations[cases[index].case_id]) {
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

  async function switchTranche(trancheId) {
    if (!runtime.incrementalR3 || !runtime.tranches[trancheId]) return;
    runtime.currentTrancheId = trancheId;
    const annotations = runtime.state.annotations || {};
    runtime.activeIndex = Math.max(0, activeCases().findIndex((caseData) => !annotations[caseData.case_id]));
    runtime.frameIndex = 0;
    runtime.view = "focal";
    runtime.lastCaseId = null;
    await dbPut("session", {
      key: "navigation",
      current_tranche_id: trancheId,
      case_id: currentCase()?.case_id || null,
      frame_index: 0,
      wizard_step: runtime.wizard?.state()?.step || 1,
      updated_at: new Date().toISOString(),
    });
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
      if (runtime.wizard && response.wizard_states?.[caseData.case_id]) {
        runtime.wizard.restore(caseData.case_id, response.wizard_states[caseData.case_id]);
      }
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

  function completionFailureMessage(error) {
    const status = error.httpStatus ? `HTTP ${error.httpStatus}` : "local preflight";
    const code = error.errorCode || "COMPLETION_BLOCKED";
    const preservation = error.savedAnnotationsUnchanged ? " Saved cases remain unchanged." : "";
    const guidance = error.retryGuidance ? ` ${error.retryGuidance}` : "";
    return `Completion failed (${status}, ${code}): ${error.message}.${preservation}${guidance}`;
  }

  async function completionRequestPayload() {
    runtime.outbox = await dbAll("outbox");
    if (runtime.outbox.length) {
      throw new Error("Completion is blocked while case-save events are pending; save or recover them first.");
    }
    const caseIds = new Set(runtime.tranches[runtime.currentTrancheId].case_ids);
    const savedAnnotations = runtime.state.annotations || {};
    const unresolvedDrafts = (await dbAll("drafts")).filter(
      (row) => caseIds.has(row.case_id) && !savedAnnotations[row.case_id]
    );
    return {
      review_id: runtime.manifest.review_id,
      reviewer_session_id: runtime.uiConfig.question_contract.reviewer_session_id,
      tranche_id: runtime.currentTrancheId,
      client_event_id: `${runtime.manifest.review_id}:complete-tranche:${runtime.currentTrancheId}`,
      idempotency_key: `${runtime.manifest.review_id}:complete-tranche:${runtime.currentTrancheId}`,
      expected_server_state_hash: runtime.serverStateHash,
      pending_outbox_events: runtime.outbox.length,
      evidence_blocker_count: runtime.evidenceBlocked ? 1 : 0,
      unresolved_draft_count: unresolvedDrafts.length,
      unresolved_divergence: false,
      elapsed_active_seconds: activeSeconds(),
    };
  }

  async function applyCompletionRequest(payload) {
    const response = await runtime.api("/api/review/detection-gold-tranche-complete", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    runtime.state = response;
    runtime.serverStateHash = response.ack.server_state_hash;
    runtime.serverSequence = response.ack.server_event_sequence;
    if (!response.completion_ack?.bundle_valid) throw new Error("Server completion bundle acknowledgement is invalid.");
    await dbDelete("session", "pending_tranche_completion");
    clearInterval(runtime.completionReplayTimer);
    runtime.completionReplayTimer = null;
    setSaveState(
      `${runtime.tranches[payload.tranche_id].label} completed | Saved to server | pending 0`,
      false
    );
    renderProgress();
    return response;
  }

  async function replayQueuedTrancheCompletion() {
    if (runtime.completionReplayActive || runtime.state.tranche_completions?.[runtime.currentTrancheId]) return;
    const queued = await dbGet("session", "pending_tranche_completion");
    if (!queued?.payload || queued.payload.tranche_id !== runtime.currentTrancheId) return;
    runtime.completionReplayActive = true;
    try {
      setSaveState("Replaying queued completion from saved server state...", false);
      await applyCompletionRequest(queued.payload);
    } catch (error) {
      if (error.httpStatus) {
        const message = completionFailureMessage(error);
        setSaveState(message, true);
        showError(message);
      } else {
        setSaveState("Completion queued offline | saved cases unchanged | pending completion 1", false);
      }
    } finally {
      runtime.completionReplayActive = false;
    }
  }

  async function completeCurrentTranche() {
    clearError();
    const button = byId("dgCompleteTranche");
    button.disabled = true;
    setSaveState("Completing tranche from saved server state...", false);
    let payload = null;
    try {
      const queued = await dbGet("session", "pending_tranche_completion");
      payload = queued?.payload || await completionRequestPayload();
      await applyCompletionRequest(payload);
    } catch (error) {
      if (!error.httpStatus && payload) {
        await dbPut("session", {
          key: "pending_tranche_completion",
          payload,
          queued_at: new Date().toISOString(),
          contains_case_save_payload: false,
        });
        setSaveState("Completion queued offline | saved cases unchanged | pending completion 1", false);
      } else {
        const message = completionFailureMessage(error);
        setSaveState(message, true);
        showError(message);
      }
    } finally {
      button.disabled = Boolean(runtime.state.tranche_completions?.[runtime.currentTrancheId]);
    }
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
      setTool(button.dataset.dgTool);
    }));
    document.querySelectorAll("[data-dg-module]").forEach((button) => button.addEventListener("click", () => {
      const cases = activeCases();
      const index = cases.findIndex((caseData) => caseData.task_type === button.dataset.dgModule && !(runtime.state.annotations || {})[caseData.case_id]);
      const fallback = cases.findIndex((caseData) => caseData.task_type === button.dataset.dgModule);
      if (index < 0 && fallback < 0) return;
      runtime.activeIndex = index >= 0 ? index : fallback;
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
    byId("nwFit")?.addEventListener("click", fitEvidence);
    byId("nwZoomOut")?.addEventListener("click", () => byId("dgZoomOut").click());
    byId("nwZoomIn")?.addEventListener("click", () => byId("dgZoomIn").click());
    byId("nwAdvancedToggle")?.addEventListener("click", () => {
      const details = document.querySelector(".nwAdvancedDetails");
      if (details) details.open = !details.open;
    });
    byId("dgSaveCase").addEventListener("click", enqueueSave);
    byId("dgReopenCase").addEventListener("click", reopenCase);
    byId("dgComplete").addEventListener("click", completePilot);
    byId("dgCompleteTranche").addEventListener("click", completeCurrentTranche);
    byId("dgTrancheSelect").addEventListener("change", (event) => switchTranche(event.target.value));
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
    window.addEventListener("online", replayQueuedTrancheCompletion);
  }

  async function mount({manifest, uiConfig, state, api}) {
    runtime.manifest = manifest;
    runtime.uiConfig = uiConfig;
    runtime.state = state;
    runtime.api = api;
    runtime.clientBuildId = uiConfig.question_contract.client_build_id || null;
    runtime.c2PitchBoundary = runtime.clientBuildId === C2_CLIENT_BUILD_ID;
    runtime.revisionAwareR3R1 = uiConfig.question_contract.revision_aware_wizard_state === true;
    runtime.indexedDbNamespace = uiConfig.question_contract.indexeddb_namespace || null;
    runtime.novice = uiConfig.question_contract.novice_guided_wizard === true;
    runtime.incrementalR3 = uiConfig.question_contract.incremental_gold_tranches === true;
    runtime.tranches = runtime.incrementalR3 ? (uiConfig.question_contract.gold_tranches || {}) : {};
    runtime.trancheOrder = runtime.incrementalR3 ? (uiConfig.question_contract.tranche_order || []) : [];
    runtime.currentTrancheId = runtime.incrementalR3
      ? (state.active_tranche_id || uiConfig.question_contract.default_tranche_id || runtime.trancheOrder[0])
      : null;
    document.body.classList.toggle("detectionGoldNovice", runtime.novice);
    document.body.classList.toggle("detectionGoldIncremental", runtime.incrementalR3);
    byId("dgTrancheControls").classList.toggle("isHidden", !runtime.incrementalR3);
    if (runtime.incrementalR3) refreshTrancheOptions();
    runtime.serverStateHash = state.server_state_hash || "";
    runtime.serverSequence = Number(state.event_sequence || 0);
    runtime.db = await openDatabase();
    let firstRepairLoad = false;
    let navigation = null;
    if (runtime.incrementalR3) {
      const sessionRows = await dbAll("session");
      const reconciliationKey = `${runtime.clientBuildId || "r3"}_first_load_reconciled`;
      const repairMarker = sessionRows.find((row) => row.key === reconciliationKey);
      firstRepairLoad = runtime.revisionAwareR3R1
        && uiConfig.question_contract.first_load_server_reconciliation === true
        && !repairMarker;
      if (firstRepairLoad) {
        const initialDrafts = await dbAll("drafts");
        const initialOutbox = await dbAll("outbox");
        for (const row of initialDrafts) await dbDelete("drafts", row.case_id);
        for (const row of initialOutbox) await dbDelete("outbox", row.client_event_id);
        await dbPut("session", {
          key: reconciliationKey,
          client_build_id: runtime.clientBuildId,
          reconciled_at: new Date().toISOString(),
          stale_prior_namespace_imported: false,
        });
        runtime.firstLoadReconciliation = {
          stale_prior_namespace_imported: false,
          cleared_new_namespace_draft_count: initialDrafts.length,
          cleared_new_namespace_outbox_count: initialOutbox.length,
        };
        const forcedTranche = uiConfig.question_contract.first_load_forced_tranche_id
          || uiConfig.question_contract.default_tranche_id;
        if (runtime.tranches[forcedTranche]) {
          runtime.currentTrancheId = forcedTranche;
          byId("dgTrancheSelect").value = forcedTranche;
        }
      } else {
        navigation = sessionRows.find((row) => row.key === "navigation");
      }
      if (navigation?.current_tranche_id && runtime.tranches[navigation.current_tranche_id]) {
        runtime.currentTrancheId = navigation.current_tranche_id;
        byId("dgTrancheSelect").value = runtime.currentTrancheId;
      }
    }
    if (runtime.novice) {
      if (!window.DetectionGoldNoviceWizard) throw new Error("Novice wizard module failed to load");
      runtime.wizard = window.DetectionGoldNoviceWizard.create({
        caseData: currentCase,
        annotation: draft,
        objects: annotationObjects,
        objectIndex: (annotationUuid) => annotationObjects().findIndex((item) => item.annotation_uuid === annotationUuid),
        selectedCandidate: () => runtime.selectedCandidate,
        selectObject,
        setSelectedCandidate: (candidate) => { runtime.selectedCandidate = candidate; },
        frameIndex: () => runtime.frameIndex,
        records,
        record: currentRecord,
        setFrame,
        setFrameSilently,
        currentFrameCandidates,
        setFirstFrameCandidate,
        nextFrameCandidate,
        setTool,
        useTemporalCandidate,
        copyGeometry: copyGeometryToNextFrame,
        hasGeometryDraft: () => Boolean(currentGeometryDraft()),
        confirmGeometryDraft,
        rejectGeometryDraft,
        upsertCandidateRelation,
        pushHistory,
        persist: persistDraft,
        render: renderCase,
        undo,
        removeSelected: () => removeSelectedAnnotation(draft()),
        deleteAllObjects: deleteAllAnnotations,
        restartCase: restartCurrentCase,
        isSaved: () => Boolean((runtime.state.annotations || {})[currentCase().case_id]),
        finishMask,
        maskPointCount: () => runtime.maskPoints.length,
        save: enqueueSave,
        reviewId: () => runtime.manifest.review_id,
        incrementalR3: () => runtime.incrementalR3,
        authoritativeFrameIndex,
        authoritativeBinding,
        authoritativeCandidateUuids,
        currentTrancheId: () => runtime.currentTrancheId,
        revisionAware: () => runtime.revisionAwareR3R1,
        c2PitchBoundary: () => runtime.c2PitchBoundary,
        estimateHiddenFootpoint,
        focusSelectedPerson,
      });
    }
    const storedDrafts = await dbAll("drafts");
    for (const row of storedDrafts) {
      runtime.drafts[row.case_id] = row.annotation;
      runtime.geometryDrafts[row.case_id] = row.geometry_drafts || {};
      runtime.positionsByCase[row.case_id] = row.position || {};
      if (row.selected_annotation_uuid) runtime.selectedObjectByCase[row.case_id] = row.selected_annotation_uuid;
      if (runtime.wizard && row.wizard_state) runtime.wizard.restore(row.case_id, row.wizard_state);
      for (const historyRow of row.undo_history || []) runtime.history.push(historyRow);
    }
    runtime.outbox = await dbAll("outbox");
    const recovery = await api("/api/review/detection-gold-recover", {
      method: "POST",
      body: JSON.stringify({
        write_sidecar: !runtime.revisionAwareR3R1,
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
    if (!firstRepairLoad && runtime.revisionAwareR3R1 && runtime.state.active_tranche_id && runtime.tranches[runtime.state.active_tranche_id]) {
      runtime.currentTrancheId = runtime.state.active_tranche_id;
      byId("dgTrancheSelect").value = runtime.currentTrancheId;
    }
    const resumeId = state.resume_case_id || recovery.materialized_state.last_viewed_case_id;
    const firstServerUnsaved = activeCases().findIndex(
      (caseData) => !(runtime.state.annotations || {})[caseData.case_id]
    );
    runtime.activeIndex = firstRepairLoad
      ? Math.max(0, firstServerUnsaved)
      : Math.max(0, activeCases().findIndex((caseData) => caseData.case_id === resumeId));
    bind();
    updateServerState();
    await flushOutbox();
    renderCase();
    if (firstRepairLoad) {
      setSaveState(
        uiConfig.question_contract.first_load_notice
          || "Six saved Tranche B cases were restored from the server. The unsaved Case 7 draft was cleared because the annotation workflow was repaired.",
        false,
      );
    }
    runtime.wizard?.showTour(false);
    await replayQueuedTrancheCompletion();
    if (!runtime.state.tranche_completions?.[runtime.currentTrancheId]) {
      runtime.completionReplayTimer = setInterval(replayQueuedTrancheCompletion, 2000);
    }
    setInterval(activeSeconds, 10000);
  }

  window.DetectionGoldPilot = {mount};
})();
