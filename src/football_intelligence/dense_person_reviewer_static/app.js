"use strict";

const ASSERTION = "I have reviewed the full image and annotated every individually evaluable visible human.";
const ADJUDICATION_ASSERTION = "I re-reviewed the full image, addressed the calibration audit findings, and annotated every individually evaluable visible human.";

const InteractionMode = Object.freeze({
  PAN_EDIT: "PAN_EDIT",
  DRAW_PERSON: "DRAW_PERSON",
  ADD_VISIBLE_COMPONENT: "ADD_VISIBLE_COMPONENT",
  DRAW_IGNORE_REGION: "DRAW_IGNORE_REGION",
});

const DRAWING_MODES = new Set([
  InteractionMode.DRAW_PERSON,
  InteractionMode.ADD_VISIBLE_COMPONENT,
  InteractionMode.DRAW_IGNORE_REGION,
]);

const MODE_LABELS = Object.freeze({
  [InteractionMode.PAN_EDIT]: "PAN / EDIT",
  [InteractionMode.DRAW_PERSON]: "DRAW PERSON",
  [InteractionMode.ADD_VISIBLE_COMPONENT]: "ADD VISIBLE PART",
  [InteractionMode.DRAW_IGNORE_REGION]: "DRAW IGNORE REGION",
});

const OperationState = Object.freeze({
  IDLE: "IDLE",
  SAVING_FOR_NAVIGATION: "SAVING_FOR_NAVIGATION",
  LOADING_IMAGE: "LOADING_IMAGE",
  FINALIZING: "FINALIZING",
});

const BUSY_REJECTION_MESSAGE = "Please wait for the current save/load to finish.";

const $ = id => document.getElementById(id);

const state = {
  bootstrap: null,
  queue: [],
  index: 0,
  serverRevision: 0,
  finalized: false,
  passKind: "FIRST_PASS",
  document: null,
  adjudicationMetadata: null,
  repairChecklist: [],
  image: new Image(),
  scale: 1,
  panX: 0,
  panY: 0,
  mode: InteractionMode.PAN_EDIT,
  working: [],
  drawingContext: null,
  contextTool: null,
  spaceHeld: false,
  selected: null,
  dragVertex: null,
  panning: null,
  history: [],
  future: [],
  autosaveTimer: null,
  actionCounter: 0,
  dirty: false,
  mutationVersion: 0,
  saveInFlight: null,
  operation: OperationState.IDLE,
  operationToken: 0,
  currentImageId: null,
  revisionImageId: null,
  statusImageId: null,
  coherenceFailure: false,
  lastInvariant: null,
};

Object.defineProperty(state, "loading", {
  get: () => state.operation !== OperationState.IDLE,
});

function blankDocument() {
  return {
    people: [],
    ignore_regions: [],
    reviewed_exhaustiveness_strips: [],
    unfinished_polygon: null,
    completion_assertion: null,
  };
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function actionId(prefix) {
  state.actionCounter += 1;
  return `${prefix}-${Date.now()}-${state.actionCounter}`;
}

function currentItem() {
  return state.queue[state.index];
}

function isBusy() {
  return state.operation !== OperationState.IDLE;
}

function syncSelector() {
  if ($("imageSelect")) $("imageSelect").value = String(state.index);
}

function isDrawingMode(mode = state.mode) {
  return DRAWING_MODES.has(mode);
}

function isAdjudication() {
  return state.passKind === "CALIBRATION_ADJUDICATION";
}

function effectiveMode() {
  return state.spaceHeld && isDrawingMode() ? "TEMPORARY_PAN" : state.mode;
}

function hasWorkingPolygon() {
  return state.working.length > 0;
}

function pushHistory() {
  state.history.push(clone(state.document));
  if (state.history.length > 100) state.history.shift();
  state.future = [];
}

function mutateDocument(callback) {
  if (state.finalized || isBusy() || state.coherenceFailure) return false;
  pushHistory();
  callback();
  state.mutationVersion += 1;
  state.dirty = true;
  renderAll();
  scheduleAutosave();
  return true;
}

function imagePoint(event) {
  const rect = $("canvas").getBoundingClientRect();
  return {
    x: (event.clientX - rect.left - state.panX) / state.scale,
    y: (event.clientY - rect.top - state.panY) / state.scale,
  };
}

function boundedImagePoint(event) {
  const point = imagePoint(event);
  return {
    x: Math.max(0, Math.min(state.image.width - 1, point.x)),
    y: Math.max(0, Math.min(state.image.height - 1, point.y)),
  };
}

function canvasPoint(point) {
  return {x: point.x * state.scale + state.panX, y: point.y * state.scale + state.panY};
}

function setStatus(message, error = false) {
  state.statusImageId = null;
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function setImageStatus(imageId, message, error = false) {
  state.statusImageId = imageId;
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function beginOperation(operation, message) {
  state.operation = operation;
  state.operationToken += 1;
  state.panning = null;
  state.dragVertex = null;
  state.spaceHeld = false;
  syncSelector();
  if (message) {
    const imageId = state.currentImageId || currentItem()?.anonymous_dense_image_id || null;
    if (imageId) setImageStatus(imageId, message);
    else setStatus(message);
  }
  renderAll();
  renderControls();
  return state.operationToken;
}

function transitionOperation(token, operation, message) {
  if (token !== state.operationToken || !isBusy()) throw new Error("STALE_OPERATION_TOKEN");
  state.operation = operation;
  syncSelector();
  if (message) {
    const imageId = state.currentImageId || currentItem()?.anonymous_dense_image_id || null;
    if (imageId) setImageStatus(imageId, message);
    else setStatus(message);
  }
  renderAll();
  renderControls();
}

function leaveOperation(token) {
  if (token !== state.operationToken) return false;
  state.operation = OperationState.IDLE;
  syncSelector();
  renderAll();
  renderControls();
  return true;
}

function rejectBusyOperation() {
  if (!isBusy()) return false;
  syncSelector();
  setStatus(BUSY_REJECTION_MESSAGE, true);
  showNotice(BUSY_REJECTION_MESSAGE);
  renderControls();
  return true;
}

function clientStateInvariant() {
  if (isBusy() || !state.document) return {ok: true, checked: false, operation: state.operation};
  const item = currentItem();
  const option = $("imageSelect").selectedOptions[0];
  const checks = {
    current_item_exists: Boolean(item),
    selector_index_matches: $("imageSelect").value === String(state.index),
    selector_text_matches: Boolean(item && option?.textContent.includes(item.anonymous_dense_image_id)),
    current_image_matches: Boolean(item && state.currentImageId === item.anonymous_dense_image_id),
    revision_image_matches: Boolean(item && state.revisionImageId === item.anonymous_dense_image_id),
    status_image_matches: Boolean(!state.statusImageId || (item && state.statusImageId === item.anonymous_dense_image_id)),
    workflow_badge_matches: Boolean(item && $("workflowBadge").textContent === item.workflow_group),
  };
  return {ok: Object.values(checks).every(Boolean), checked: true, image_id: item?.anonymous_dense_image_id, checks};
}

function enforceClientStateInvariant() {
  const result = clientStateInvariant();
  state.lastInvariant = result;
  if (result.ok) return true;
  state.coherenceFailure = true;
  syncSelector();
  setStatus("Client image state is inconsistent. Reload the reviewer before editing.", true);
  renderControls();
  return false;
}

let noticeTimer = null;
function showNotice(message) {
  clearTimeout(noticeTimer);
  $("interactionNotice").textContent = message;
  $("interactionNotice").hidden = false;
  noticeTimer = setTimeout(() => { $("interactionNotice").hidden = true; }, 2600);
}

async function request(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: {"Content-Type": "application/json", ...(options.headers || {})},
  });
  const payload = await response.json();
  if (!response.ok) {
    const error = new Error(payload.message || response.statusText);
    error.payload = payload;
    throw error;
  }
  return payload;
}

async function loadBootstrap() {
  state.bootstrap = await request("/api/bootstrap");
  state.queue = state.bootstrap.queue;
  state.passKind = state.bootstrap.pass_kind;
  const reminder = state.bootstrap.scope_reminder;
  const showReminder = Array.isArray(reminder) && reminder.length === 2;
  $("scopeReminder").hidden = !showReminder;
  document.body.classList.toggle("has-scope-reminder", showReminder);
  if (showReminder) {
    $("scopeReminderPrimary").textContent = reminder[0];
    $("scopeReminderSecondary").textContent = reminder[1];
  }
  $("blindBadge").textContent = isAdjudication() ? "Candidate-free" : "Candidate-blind";
  $("imageSelect").innerHTML = state.queue
    .map((row, index) => `<option value="${index}">${row.anonymous_dense_image_id} - ${row.workflow_group}</option>`)
    .join("");
  await loadImage(0, true);
}

function blockForWorkingPolygon() {
  if (!hasWorkingPolygon()) return false;
  const message = "Finish or cancel the current polygon first.";
  setStatus(message, true);
  showNotice(message);
  return true;
}

async function loadImage(index, initial = false) {
  if (!state.queue.length) {
    syncSelector();
    return false;
  }
  if (rejectBusyOperation()) return false;
  if (!initial && blockForWorkingPolygon()) {
    syncSelector();
    return false;
  }
  const requestedIndex = Number.isInteger(index) ? index : state.index;
  const targetIndex = Math.max(0, Math.min(requestedIndex, state.queue.length - 1));
  if (!initial && state.document && targetIndex === state.index) {
    syncSelector();
    return true;
  }
  const sourceId = state.currentImageId;
  const targetItem = state.queue[targetIndex];
  const token = beginOperation(
    OperationState.SAVING_FOR_NAVIGATION,
    sourceId ? `${sourceId} - saving before navigation...` : `Loading ${targetItem.anonymous_dense_image_id}...`,
  );
  try {
    await saveDraft();
    transitionOperation(
      token,
      OperationState.LOADING_IMAGE,
      sourceId
        ? `${sourceId} - loading ${targetItem.anonymous_dense_image_id}...`
        : `Loading ${targetItem.anonymous_dense_image_id}...`,
    );
    const saved = await request(
      `/api/state?image_id=${encodeURIComponent(targetItem.anonymous_dense_image_id)}`,
    );
    if (
      saved.anonymous_dense_image_id !== targetItem.anonymous_dense_image_id
      || saved.pass_kind !== state.passKind
    ) {
      throw new Error("Target state response named a different image.");
    }
    if (!Number.isInteger(saved.revision) || !saved.document) {
      throw new Error("Target state response is incomplete.");
    }
    const nextDocument = clone(saved.document || blankDocument());
    const nextMetadata = clone(saved.adjudication_metadata || {
      adjudication_assertion: null,
      repair_checklist_addressed: false,
    });
    const nextChecklist = clone(saved.repair_checklist || targetItem.repair_checklist || []);
    const nextImage = await new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error(`Image asset failed to load for ${targetItem.anonymous_dense_image_id}.`));
      image.src = targetItem.image_url;
    });
    if (token !== state.operationToken || state.operation !== OperationState.LOADING_IMAGE) {
      throw new Error("Navigation transaction was superseded.");
    }

    state.index = targetIndex;
    state.currentImageId = targetItem.anonymous_dense_image_id;
    state.revisionImageId = targetItem.anonymous_dense_image_id;
    state.serverRevision = saved.revision;
    state.finalized = saved.finalized;
    state.document = nextDocument;
    state.adjudicationMetadata = nextMetadata;
    state.repairChecklist = nextChecklist;
    state.image = nextImage;
    state.history = [];
    state.future = [];
    state.working = [];
    state.drawingContext = null;
    state.contextTool = null;
    state.spaceHeld = false;
    state.selected = null;
    state.dragVertex = null;
    state.panning = null;
    state.dirty = false;
    state.mutationVersion = 0;
    state.mode = InteractionMode.PAN_EDIT;
    syncSelector();
    fit("fitWidth");
    $("reveal").hidden = state.bootstrap?.tools?.post_finalize_comparison !== true || isAdjudication() || !state.finalized;
    $("revealPanel").hidden = true;
    $("assertion").checked = state.document.completion_assertion === ASSERTION;
    $("adjudicationAssertion").checked = state.adjudicationMetadata.adjudication_assertion === ADJUDICATION_ASSERTION;
    $("repairChecklistAddressed").checked = state.adjudicationMetadata.repair_checklist_addressed === true;
    leaveOperation(token);
    setImageStatus(
      targetItem.anonymous_dense_image_id,
      `${targetItem.anonymous_dense_image_id} - revision ${state.serverRevision}${state.finalized ? " - finalized" : ""}`,
    );
    renderAll();
    return enforceClientStateInvariant();
  } catch (error) {
    leaveOperation(token);
    syncSelector();
    const imageId = state.currentImageId;
    const errorCode = error.payload?.error_code;
    const errorDetail = errorCode ? `${errorCode}: ${error.message}` : error.message;
    if (imageId) setImageStatus(imageId, `${imageId} - navigation failed: ${errorDetail}`, true);
    else setStatus(`Image load failed: ${error.message}`, true);
    renderAll();
    renderControls();
    enforceClientStateInvariant();
    return false;
  }
}

function fit(kind) {
  if (!state.image.width || !state.image.height) return;
  const viewport = $("viewport");
  const scaleX = viewport.clientWidth / state.image.width;
  const scaleY = viewport.clientHeight / state.image.height;
  state.scale = kind === "fitHeight" ? scaleY : scaleX;
  state.panX = (viewport.clientWidth - state.image.width * state.scale) / 2;
  state.panY = (viewport.clientHeight - state.image.height * state.scale) / 2;
  render();
}

function zoomAt(factor, clientX, clientY) {
  if (!state.image.width) return;
  const rect = $("canvas").getBoundingClientRect();
  const anchorX = clientX ?? rect.left + $("viewport").clientWidth / 2;
  const anchorY = clientY ?? rect.top + $("viewport").clientHeight / 2;
  const imageX = (anchorX - rect.left - state.panX) / state.scale;
  const imageY = (anchorY - rect.top - state.panY) / state.scale;
  const nextScale = Math.max(0.03, Math.min(40, state.scale * factor));
  state.panX = anchorX - rect.left - imageX * nextScale;
  state.panY = anchorY - rect.top - imageY * nextScale;
  state.scale = nextScale;
  render();
}

function drawPolygon(ctx, polygon, stroke, fill, selected = false) {
  if (!polygon || !polygon.length) return;
  ctx.beginPath();
  const first = canvasPoint(polygon[0]);
  ctx.moveTo(first.x, first.y);
  polygon.slice(1).forEach(point => {
    const current = canvasPoint(point);
    ctx.lineTo(current.x, current.y);
  });
  if (polygon.length > 2) ctx.closePath();
  ctx.fillStyle = fill;
  if (polygon.length > 2) ctx.fill();
  ctx.strokeStyle = selected ? "#f9ed78" : stroke;
  ctx.lineWidth = selected ? 4 : 2;
  ctx.stroke();
  polygon.forEach(point => {
    const current = canvasPoint(point);
    ctx.beginPath();
    ctx.arc(current.x, current.y, selected ? 5.5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = selected ? "#f9ed78" : stroke;
    ctx.fill();
  });
}

function personComponents(person) {
  return person.visible_mask_components || person.canonical_components || [];
}

function regionPolygon(region) {
  return region.polygon || (region.canonical_components || [])[0] || [];
}

function render() {
  if (!state.document) return;
  const canvas = $("canvas");
  const viewport = $("viewport");
  const dpr = window.devicePixelRatio || 1;
  canvas.width = viewport.clientWidth * dpr;
  canvas.height = viewport.clientHeight * dpr;
  canvas.style.width = `${viewport.clientWidth}px`;
  canvas.style.height = `${viewport.clientHeight}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, viewport.clientWidth, viewport.clientHeight);
  ctx.drawImage(state.image, state.panX, state.panY, state.image.width * state.scale, state.image.height * state.scale);

  const stripWidth = state.image.width / 8;
  for (let index = 0; index < 8; index += 1) {
    if (state.document.reviewed_exhaustiveness_strips.includes(index)) {
      const x = state.panX + index * stripWidth * state.scale;
      ctx.fillStyle = "rgba(20, 210, 145, .06)";
      ctx.fillRect(x, state.panY, stripWidth * state.scale, state.image.height * state.scale);
    }
  }

  state.document.people.forEach((person, personIndex) => {
    personComponents(person).forEach(polygon => {
      drawPolygon(
        ctx,
        polygon,
        "#66dcff",
        "rgba(40, 160, 220, .18)",
        state.selected?.kind === "person" && state.selected.i === personIndex,
      );
    });
  });
  state.document.ignore_regions.forEach((region, regionIndex) => {
    drawPolygon(
      ctx,
      regionPolygon(region),
      "#ff9b53",
      "rgba(255, 110, 40, .18)",
      state.selected?.kind === "ignore" && state.selected.i === regionIndex,
    );
  });
  if (state.working.length) drawPolygon(ctx, state.working, "#f8ef74", "rgba(248, 239, 116, .08)", true);
}

function renderMode() {
  const effective = effectiveMode();
  $("modeBadge").dataset.mode = state.mode;
  $("modeBadge").textContent = `MODE: ${MODE_LABELS[state.mode]}`;
  $("viewport").dataset.effectiveMode = effective;
  $("viewport").classList.toggle("is-panning", Boolean(state.panning));
  document.querySelectorAll("[data-mode-button]").forEach(button => {
    button.classList.toggle("active", button.dataset.modeButton === state.mode);
  });
  const drawing = isDrawingMode();
  $("finishPolygon").disabled = state.finalized || isBusy() || state.coherenceFailure || !drawing;
  $("cancelPolygon").disabled = state.finalized || isBusy() || state.coherenceFailure || !drawing;
  $("workingCount").textContent = drawing
    ? `${state.working.length} point${state.working.length === 1 ? "" : "s"} - ${state.spaceHeld ? "temporary pan" : MODE_LABELS[state.mode].toLowerCase()}`
    : "No active polygon";
  $("help").textContent = state.spaceHeld
    ? "Temporary pan: drag to move the image. Release Space to resume drawing; no point can be added now."
    : drawing
      ? "Click boundary points. Hold Space and drag to pan. Enter finishes; Escape cancels."
      : "Pan: drag the background. Edit: drag a nearby vertex. Click a shape to select it.";
}

function selectedRecord() {
  if (state.selected?.kind === "person") return state.document.people[state.selected.i] || null;
  if (state.selected?.kind === "ignore") return state.document.ignore_regions[state.selected.i] || null;
  return null;
}

function renderInspector() {
  const personContextVisible = state.contextTool === "person" || state.mode === InteractionMode.DRAW_PERSON;
  const ignoreContextVisible = state.contextTool === "ignore" || state.mode === InteractionMode.DRAW_IGNORE_REGION;
  $("personContext").hidden = !personContextVisible;
  $("ignoreContext").hidden = !ignoreContextVisible;

  const selected = selectedRecord();
  $("deletePerson").hidden = state.finalized || state.selected?.kind !== "person";
  $("deleteIgnore").hidden = state.finalized || state.selected?.kind !== "ignore";
  $("editRelevanceLabel").hidden = state.selected?.kind !== "person" || !selected;
  if (state.selected?.kind === "person" && selected) {
    $("selectedKind").textContent = "Person";
    $("selectionDetails").className = "";
    $("selectionDetails").innerHTML = `<strong>${selected.instance_id}</strong><br>Relevance: ${selected.relevance}<br>Visible components: ${personComponents(selected).length}`;
    $("editRelevance").value = selected.relevance;
  } else if (state.selected?.kind === "ignore" && selected) {
    $("selectedKind").textContent = "Ignore region";
    $("selectionDetails").className = "";
    $("selectionDetails").innerHTML = `<strong>${selected.ignore_region_id}</strong><br>Reason: ${selected.reason}`;
  } else {
    $("selectedKind").textContent = "None";
    $("selectionDetails").className = "empty-state";
    $("selectionDetails").textContent = "Click a person or ignore region to inspect it.";
  }

  const peopleRows = state.document.people.map((person, index) => (
    `<li data-kind="person" data-i="${index}" class="${state.selected?.kind === "person" && state.selected.i === index ? "selected" : ""}">`
      + `<strong>${person.instance_id}</strong><br>${person.relevance || "Relevance required"} - ${personComponents(person).length} visible part(s)</li>`
  ));
  const ignoreRows = state.document.ignore_regions.map((region, index) => (
    `<li data-kind="ignore" data-i="${index}" class="${state.selected?.kind === "ignore" && state.selected.i === index ? "selected" : ""}">`
      + `<strong>${region.ignore_region_id}</strong><br>${region.reason}</li>`
  ));
  $("instances").innerHTML = [...peopleRows, ...ignoreRows].join("") || '<li class="empty-state">No instances yet.</li>';
  $("instanceCount").textContent = `${state.document.people.length} people / ${state.document.ignore_regions.length} ignore`;

  $("strips").innerHTML = Array.from({length: 8}, (_, index) => (
    `<button type="button" data-strip="${index}" class="${state.document.reviewed_exhaustiveness_strips.includes(index) ? "reviewed" : ""}"`
      + ` aria-pressed="${state.document.reviewed_exhaustiveness_strips.includes(index)}">${index + 1}</button>`
  )).join("");
  $("stripProgress").textContent = `${state.document.reviewed_exhaustiveness_strips.length}/8 reviewed`;

  const adjudication = isAdjudication();
  $("auditChecklistPanel").hidden = !adjudication;
  $("adjudicationAssertionLabel").hidden = !adjudication;
  if (adjudication) {
    const rows = state.repairChecklist.map(finding => {
      const item = document.createElement("li");
      const region = document.createElement("span");
      region.className = "region";
      region.textContent = finding.source_region;
      item.append(region, document.createTextNode(` ${finding.required_correction_type}`));
      return item;
    });
    $("auditChecklist").replaceChildren(...rows);
  }
}

function renderControls() {
  const busy = isBusy();
  const coherent = !state.coherenceFailure;
  const hasDocument = Boolean(state.document);
  const mutable = hasDocument && !state.finalized && !busy && coherent;
  $("previous").disabled = busy || !coherent || !hasDocument || state.index <= 0;
  $("next").disabled = busy || !coherent || !hasDocument || state.index >= state.queue.length - 1;
  $("imageSelect").disabled = busy || !coherent || !hasDocument;
  $("panEdit").disabled = !mutable;
  $("newPerson").disabled = !mutable;
  $("newIgnore").disabled = !mutable;
  $("addComponent").disabled = !mutable || state.selected?.kind !== "person";
  $("undo").disabled = !mutable || !state.history.length;
  $("redo").disabled = !mutable || !state.future.length;
  $("assertion").disabled = !mutable;
  $("editRelevance").disabled = !mutable;
  $("repairChecklistAddressed").disabled = !mutable;
  $("adjudicationAssertion").disabled = !mutable;
  $("deletePerson").disabled = !mutable;
  $("deleteIgnore").disabled = !mutable;
  $("relevance").disabled = !mutable;
  $("ignoreReason").disabled = !mutable;
  $("strips").querySelectorAll("button").forEach(button => { button.disabled = !mutable; });
  document.querySelectorAll("[data-view]").forEach(button => { button.disabled = busy || !coherent; });
  const adjudicationReady = hasDocument && (!isAdjudication() || (
    state.document.reviewed_exhaustiveness_strips.length === 8
    && state.document.completion_assertion === ASSERTION
    && state.adjudicationMetadata?.repair_checklist_addressed === true
    && state.adjudicationMetadata?.adjudication_assertion === ADJUDICATION_ASSERTION
    && !hasWorkingPolygon()
  ));
  $("finalize").disabled = !mutable || !adjudicationReady;
  $("finalize").textContent = isAdjudication() ? "Finalize immutable adjudication" : "Finalize immutable frame";
  $("reveal").hidden = state.bootstrap?.tools?.post_finalize_comparison !== true || isAdjudication() || !state.finalized;
  $("reveal").disabled = busy || !coherent;
  $("workflowBadge").textContent = currentItem()?.workflow_group || "Loading";
}

function renderAll() {
  if (!state.document) {
    renderControls();
    return;
  }
  render();
  renderMode();
  renderInspector();
  renderControls();
}

function clearDrawingState() {
  state.working = [];
  state.drawingContext = null;
  state.contextTool = null;
  state.spaceHeld = false;
}

function enterPanEdit({discardWorking = false} = {}) {
  if (isBusy() || state.coherenceFailure) return false;
  if (hasWorkingPolygon() && !discardWorking) {
    blockForWorkingPolygon();
    return false;
  }
  clearDrawingState();
  state.mode = InteractionMode.PAN_EDIT;
  state.contextTool = null;
  renderAll();
  return true;
}

function enterDrawPerson() {
  if (state.finalized || isBusy() || state.coherenceFailure || blockForWorkingPolygon()) return false;
  const relevance = $("relevance").value;
  if (!relevance) {
    state.contextTool = "person";
    renderAll();
    $("relevance").focus();
    setStatus("Choose relevance to start a person.", true);
    return false;
  }
  state.mode = InteractionMode.DRAW_PERSON;
  state.contextTool = "person";
  state.drawingContext = {relevance};
  state.working = [];
  renderAll();
  setStatus("Draw person: click boundary points; Enter finishes; hold Space to pan.");
  return true;
}

function enterAddVisibleComponent() {
  if (state.finalized || isBusy() || state.coherenceFailure || blockForWorkingPolygon()) return false;
  const person = state.selected?.kind === "person" ? state.document.people[state.selected.i] : null;
  if (!person) {
    setStatus("Select an existing person before adding a visible part.", true);
    showNotice("Select an existing person first.");
    return false;
  }
  state.mode = InteractionMode.ADD_VISIBLE_COMPONENT;
  state.contextTool = null;
  state.drawingContext = {personInstanceId: person.instance_id};
  state.working = [];
  renderAll();
  setStatus(`Add visible part to ${person.instance_id}: Enter finishes; hold Space to pan.`);
  return true;
}

function enterDrawIgnoreRegion() {
  if (state.finalized || isBusy() || state.coherenceFailure || blockForWorkingPolygon()) return false;
  const reason = $("ignoreReason").value;
  if (!reason) {
    state.contextTool = "ignore";
    renderAll();
    $("ignoreReason").focus();
    setStatus("Choose a reason to start an ignore region.", true);
    return false;
  }
  state.mode = InteractionMode.DRAW_IGNORE_REGION;
  state.contextTool = "ignore";
  state.drawingContext = {reason};
  state.working = [];
  renderAll();
  setStatus("Draw ignore region: click boundary points; Enter finishes; hold Space to pan.");
  return true;
}

function addWorkingVertex(point) {
  if (state.finalized || isBusy() || state.coherenceFailure || state.spaceHeld || !isDrawingMode()) return false;
  state.working.push(point);
  render();
  renderMode();
  return true;
}

function nextInstanceId(prefix, records, key) {
  const pattern = new RegExp(`^${prefix}-(\\d+)$`);
  const maximum = records.reduce((current, record) => {
    const match = pattern.exec(record[key] || "");
    return match ? Math.max(current, Number(match[1])) : current;
  }, 0);
  return `${prefix}-${String(maximum + 1).padStart(3, "0")}`;
}

function finishPolygon() {
  if (isBusy() || state.coherenceFailure || !isDrawingMode()) return false;
  if (state.working.length < 3) {
    setStatus("Polygon needs at least three vertices.", true);
    showNotice("Add at least three points, or Cancel.");
    return false;
  }
  const polygon = clone(state.working);
  const mode = state.mode;
  const context = clone(state.drawingContext || {});
  let committedSelection = null;
  const changed = mutateDocument(() => {
    if (mode === InteractionMode.DRAW_PERSON) {
      const instanceId = nextInstanceId("person", state.document.people, "instance_id");
      state.document.people.push({
        instance_id: instanceId,
        relevance: context.relevance,
        visible_mask_components: [polygon],
      });
      committedSelection = {kind: "person", i: state.document.people.length - 1, c: 0};
    } else if (mode === InteractionMode.ADD_VISIBLE_COMPONENT) {
      const index = state.document.people.findIndex(person => person.instance_id === context.personInstanceId);
      if (index < 0) throw new Error("Selected person no longer exists.");
      state.document.people[index].visible_mask_components.push(polygon);
      committedSelection = {
        kind: "person",
        i: index,
        c: state.document.people[index].visible_mask_components.length - 1,
      };
    } else if (mode === InteractionMode.DRAW_IGNORE_REGION) {
      const ignoreId = nextInstanceId("ignore", state.document.ignore_regions, "ignore_region_id");
      state.document.ignore_regions.push({ignore_region_id: ignoreId, reason: context.reason, polygon});
      committedSelection = {kind: "ignore", i: state.document.ignore_regions.length - 1, c: 0};
    }
  });
  if (!changed) return false;
  clearDrawingState();
  state.mode = InteractionMode.PAN_EDIT;
  state.selected = committedSelection;
  renderAll();
  setStatus("Polygon committed. Mode returned to Pan / Edit.");
  return true;
}

function cancelPolygon() {
  if (isBusy() || state.coherenceFailure) return false;
  const hadWork = hasWorkingPolygon();
  clearDrawingState();
  state.mode = InteractionMode.PAN_EDIT;
  renderAll();
  setStatus(hadWork ? "Polygon canceled. Document unchanged; mode is Pan / Edit." : "Mode: Pan / Edit.");
  return true;
}

function nearestVertex(point) {
  let best = null;
  const consider = (kind, index, component, polygon) => {
    polygon.forEach((vertex, vertexIndex) => {
      const distance = Math.hypot(vertex.x - point.x, vertex.y - point.y);
      if (distance < 10 / state.scale && (!best || distance < best.distance)) {
        best = {kind, i: index, c: component, v: vertexIndex, distance, changed: false};
      }
    });
  };
  state.document.people.forEach((person, index) => {
    personComponents(person).forEach((polygon, component) => consider("person", index, component, polygon));
  });
  state.document.ignore_regions.forEach((region, index) => consider("ignore", index, 0, regionPolygon(region)));
  return best;
}

function pointInsidePolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const a = polygon[i];
    const b = polygon[j];
    const intersects = ((a.y > point.y) !== (b.y > point.y))
      && point.x < ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y) + a.x;
    if (intersects) inside = !inside;
  }
  return inside;
}

function instanceAt(point) {
  for (let index = state.document.ignore_regions.length - 1; index >= 0; index -= 1) {
    if (pointInsidePolygon(point, regionPolygon(state.document.ignore_regions[index]))) {
      return {kind: "ignore", i: index, c: 0};
    }
  }
  for (let index = state.document.people.length - 1; index >= 0; index -= 1) {
    const components = personComponents(state.document.people[index]);
    for (let component = components.length - 1; component >= 0; component -= 1) {
      if (pointInsidePolygon(point, components[component])) return {kind: "person", i: index, c: component};
    }
  }
  return null;
}

function beginPan(event) {
  state.panning = {
    x: event.clientX,
    y: event.clientY,
    panX: state.panX,
    panY: state.panY,
  };
  renderMode();
}

function undo() {
  if (state.finalized || isBusy() || state.coherenceFailure || !state.history.length || blockForWorkingPolygon()) return;
  state.future.push(clone(state.document));
  state.document = state.history.pop();
  state.selected = null;
  state.mutationVersion += 1;
  state.dirty = true;
  renderAll();
  scheduleAutosave();
}

function redo() {
  if (state.finalized || isBusy() || state.coherenceFailure || !state.future.length || blockForWorkingPolygon()) return;
  state.history.push(clone(state.document));
  state.document = state.future.pop();
  state.selected = null;
  state.mutationVersion += 1;
  state.dirty = true;
  renderAll();
  scheduleAutosave();
}

function scheduleAutosave() {
  clearTimeout(state.autosaveTimer);
  state.autosaveTimer = setTimeout(() => {
    saveDraft().catch(error => setStatus(`${error.payload?.error_code || "SAVE_FAILED"}: ${error.message}`, true));
  }, 500);
}

async function saveDraft() {
  clearTimeout(state.autosaveTimer);
  if (state.coherenceFailure) throw new Error("Client image state is inconsistent; reload before saving.");
  if (state.saveInFlight) {
    await state.saveInFlight.promise;
    return state.dirty ? saveDraft() : null;
  }
  if (state.finalized || !state.dirty || !state.document) return null;
  const imageId = state.currentImageId;
  if (!imageId || state.revisionImageId !== imageId || currentItem()?.anonymous_dense_image_id !== imageId) {
    state.coherenceFailure = true;
    renderControls();
    throw new Error("Client image identity is inconsistent; draft was not sent.");
  }
  const expectedRevision = state.serverRevision;
  const savingVersion = state.mutationVersion;
  const documentSnapshot = clone(state.document);
  const metadataSnapshot = clone(state.adjudicationMetadata);
  const passKind = state.passKind;
  const saveRecord = {imageId, expectedRevision, document: documentSnapshot, mutationVersion: savingVersion, promise: null};
  state.saveInFlight = saveRecord;
  setImageStatus(imageId, `Saving ${imageId} draft...`);
  saveRecord.promise = (async () => {
    try {
      const response = await request("/api/action", {
        method: "POST",
        body: JSON.stringify({
          action_id: actionId("autosave"),
          action_type: "SAVE_DRAFT",
          anonymous_dense_image_id: imageId,
          pass_kind: passKind,
          expected_revision: expectedRevision,
          document: documentSnapshot,
          ...(passKind === "CALIBRATION_ADJUDICATION"
            ? {adjudication_metadata: metadataSnapshot}
            : {}),
        }),
      });
      if (
        response.anonymous_dense_image_id !== imageId
        || response.pass_kind !== passKind
        || response.revision !== expectedRevision + 1
      ) {
        state.coherenceFailure = true;
        throw new Error(`Stale save response rejected for ${imageId}.`);
      }
      if (
        state.currentImageId !== imageId
        || state.revisionImageId !== imageId
        || state.serverRevision !== expectedRevision
      ) {
        state.coherenceFailure = true;
        throw new Error(`Save response for ${imageId} no longer matches the authoritative image.`);
      }
      state.serverRevision = response.revision;
      state.revisionImageId = imageId;
      state.dirty = state.mutationVersion !== savingVersion;
      if (state.dirty && !isBusy()) scheduleAutosave();
      setImageStatus(imageId, `Saved ${imageId} revision ${state.serverRevision}`);
      if (!isBusy()) {
        renderControls();
        enforceClientStateInvariant();
      }
      return response;
    } catch (error) {
      setImageStatus(imageId, `${imageId} - ${error.payload?.error_code || "SAVE_FAILED"}: ${error.message}`, true);
      throw error;
    } finally {
      if (state.saveInFlight === saveRecord) state.saveInFlight = null;
    }
  })();
  return saveRecord.promise;
}

async function recoverAfterFinalizeFailure(imageId) {
  if (state.currentImageId !== imageId) return false;
  try {
    const saved = await request(`/api/state?image_id=${encodeURIComponent(imageId)}`);
    if (saved.anonymous_dense_image_id !== imageId || !Number.isInteger(saved.revision) || !saved.document) {
      return false;
    }
    if (state.currentImageId !== imageId) return false;
    state.serverRevision = saved.revision;
    state.revisionImageId = imageId;
    state.finalized = saved.finalized;
    state.document = clone(saved.document);
    state.adjudicationMetadata = clone(saved.adjudication_metadata || state.adjudicationMetadata);
    state.repairChecklist = clone(saved.repair_checklist || state.repairChecklist);
    state.dirty = false;
    state.history = [];
    state.future = [];
    if (state.finalized) {
      clearDrawingState();
      state.mode = InteractionMode.PAN_EDIT;
      state.selected = null;
    }
    $("assertion").checked = state.document.completion_assertion === ASSERTION;
    $("adjudicationAssertion").checked = state.adjudicationMetadata?.adjudication_assertion === ADJUDICATION_ASSERTION;
    $("repairChecklistAddressed").checked = state.adjudicationMetadata?.repair_checklist_addressed === true;
    return true;
  } catch (_) {
    return false;
  }
}

async function finalize() {
  if (rejectBusyOperation()) return false;
  if (state.finalized || state.coherenceFailure || blockForWorkingPolygon()) return false;
  const imageId = state.currentImageId;
  const token = beginOperation(OperationState.FINALIZING, `${imageId} - finalizing...`);
  try {
    await saveDraft();
    if (
      token !== state.operationToken
      || state.operation !== OperationState.FINALIZING
      || state.currentImageId !== imageId
      || state.revisionImageId !== imageId
    ) {
      throw new Error("Finalization transaction lost its image binding.");
    }
    const expectedRevision = state.serverRevision;
    const documentSnapshot = clone(state.document);
    const metadataSnapshot = clone(state.adjudicationMetadata);
    const passKind = state.passKind;
    const response = await request("/api/action", {
      method: "POST",
      body: JSON.stringify({
        action_id: actionId("finalize"),
        action_type: "FINALIZE",
        anonymous_dense_image_id: imageId,
        pass_kind: passKind,
        expected_revision: expectedRevision,
        document: documentSnapshot,
        ...(passKind === "CALIBRATION_ADJUDICATION" ? {adjudication_metadata: metadataSnapshot} : {}),
      }),
    });
    if (
      response.anonymous_dense_image_id !== imageId
      || response.pass_kind !== passKind
      || response.revision !== expectedRevision + 1
      || state.currentImageId !== imageId
      || state.serverRevision !== expectedRevision
      || token !== state.operationToken
    ) {
      state.coherenceFailure = true;
      throw new Error(`Stale finalization response rejected for ${imageId}.`);
    }
    state.serverRevision = response.revision;
    state.revisionImageId = imageId;
    state.finalized = true;
    state.dirty = false;
    clearDrawingState();
    state.mode = InteractionMode.PAN_EDIT;
    leaveOperation(token);
    renderAll();
    setImageStatus(imageId, `Finalized ${imageId} - immutable event ${response.event_id}`);
    renderControls();
    enforceClientStateInvariant();
    return true;
  } catch (error) {
    await recoverAfterFinalizeFailure(imageId);
    leaveOperation(token);
    syncSelector();
    setImageStatus(imageId, `${imageId} - ${error.payload?.error_code || "FINALIZE_FAILED"}: ${error.message}`, true);
    renderAll();
    enforceClientStateInvariant();
    return false;
  }
}

async function reveal() {
  if (isBusy() || state.coherenceFailure) return false;
  try {
    const payload = await request("/api/action", {
      method: "POST",
      body: JSON.stringify({
        action_id: actionId("reveal"),
        action_type: "REVEAL_CANDIDATES",
        anonymous_dense_image_id: currentItem().anonymous_dense_image_id,
        pass_kind: state.passKind,
        expected_revision: state.serverRevision,
      }),
    });
    $("revealPayload").textContent = JSON.stringify(payload.candidate_comparison, null, 2);
    $("revealPanel").hidden = false;
    return true;
  } catch (error) {
    setStatus(`${error.payload?.error_code || "REVEAL_FAILED"}: ${error.message}`, true);
    return false;
  }
}

function deleteSelected() {
  if (state.finalized || isBusy() || state.coherenceFailure || blockForWorkingPolygon()) return;
  const selected = selectedRecord();
  if (!selected) return;
  const label = state.selected.kind === "person" ? selected.instance_id : selected.ignore_region_id;
  if (!window.confirm(`Delete ${label} from this mutable draft?`)) return;
  const kind = state.selected.kind;
  const index = state.selected.i;
  mutateDocument(() => {
    if (kind === "person") state.document.people.splice(index, 1);
    else state.document.ignore_regions.splice(index, 1);
  });
  state.selected = null;
  enterPanEdit({discardWorking: true});
  setStatus(`${label} deleted; draft revision queued for save.`);
}

$("canvas").addEventListener("pointerdown", event => {
  if (state.finalized || isBusy() || state.coherenceFailure || (event.button !== 0 && event.button !== 1)) return;
  event.preventDefault();
  const point = boundedImagePoint(event);
  if (event.button === 1 || state.spaceHeld) {
    beginPan(event);
    return;
  }
  if (state.mode === InteractionMode.PAN_EDIT) {
    const vertex = nearestVertex(point);
    if (vertex) {
      pushHistory();
      state.dragVertex = vertex;
      state.selected = {kind: vertex.kind, i: vertex.i, c: vertex.c};
      renderAll();
      return;
    }
    const instance = instanceAt(point);
    if (instance) {
      state.selected = instance;
      renderAll();
      return;
    }
    beginPan(event);
    return;
  }
  addWorkingVertex(point);
});

window.addEventListener("pointermove", event => {
  if (!state.document) return;
  const point = boundedImagePoint(event);
  $("cursor").textContent = `x ${point.x.toFixed(1)}, y ${point.y.toFixed(1)}`;
  if (state.dragVertex) {
    const polygon = state.dragVertex.kind === "person"
      ? personComponents(state.document.people[state.dragVertex.i])[state.dragVertex.c]
      : regionPolygon(state.document.ignore_regions[state.dragVertex.i]);
    polygon[state.dragVertex.v] = point;
    state.dragVertex.changed = true;
    state.dirty = true;
    render();
  } else if (state.panning) {
    state.panX = state.panning.panX + event.clientX - state.panning.x;
    state.panY = state.panning.panY + event.clientY - state.panning.y;
    render();
  }
});

window.addEventListener("pointerup", () => {
  if (state.dragVertex) {
    if (state.dragVertex.changed) {
      state.mutationVersion += 1;
      scheduleAutosave();
    }
    else state.history.pop();
    state.dragVertex = null;
    renderAll();
  }
  state.panning = null;
  renderMode();
});

$("canvas").addEventListener("wheel", event => {
  event.preventDefault();
  if (isBusy() || state.coherenceFailure) return;
  zoomAt(event.deltaY < 0 ? 1.15 : 1 / 1.15, event.clientX, event.clientY);
}, {passive: false});

$("instances").addEventListener("click", event => {
  if (isBusy() || state.coherenceFailure) return;
  const item = event.target.closest("li[data-kind]");
  if (!item) return;
  state.selected = {kind: item.dataset.kind, i: Number(item.dataset.i), c: 0};
  renderAll();
});

$("strips").addEventListener("click", event => {
  const button = event.target.closest("button[data-strip]");
  if (!button || state.finalized || isBusy() || state.coherenceFailure || blockForWorkingPolygon()) return;
  const strip = Number(button.dataset.strip);
  mutateDocument(() => {
    const values = new Set(state.document.reviewed_exhaustiveness_strips);
    values.has(strip) ? values.delete(strip) : values.add(strip);
    state.document.reviewed_exhaustiveness_strips = [...values].sort((a, b) => a - b);
  });
});

$("assertion").addEventListener("change", () => {
  if (isBusy() || state.coherenceFailure) {
    $("assertion").checked = state.document.completion_assertion === ASSERTION;
    return;
  }
  if (blockForWorkingPolygon()) {
    $("assertion").checked = state.document.completion_assertion === ASSERTION;
    return;
  }
  mutateDocument(() => {
    state.document.completion_assertion = $("assertion").checked ? ASSERTION : null;
  });
});

function mutateAdjudicationMetadata(callback) {
  if (!isAdjudication() || state.finalized || isBusy() || state.coherenceFailure || blockForWorkingPolygon()) return false;
  callback();
  state.mutationVersion += 1;
  state.dirty = true;
  renderAll();
  scheduleAutosave();
  return true;
}

$("repairChecklistAddressed").addEventListener("change", () => {
  const checked = $("repairChecklistAddressed").checked;
  if (!mutateAdjudicationMetadata(() => { state.adjudicationMetadata.repair_checklist_addressed = checked; })) {
    $("repairChecklistAddressed").checked = state.adjudicationMetadata?.repair_checklist_addressed === true;
  }
});

$("adjudicationAssertion").addEventListener("change", () => {
  const checked = $("adjudicationAssertion").checked;
  if (!mutateAdjudicationMetadata(() => {
    state.adjudicationMetadata.adjudication_assertion = checked ? ADJUDICATION_ASSERTION : null;
  })) {
    $("adjudicationAssertion").checked = state.adjudicationMetadata?.adjudication_assertion === ADJUDICATION_ASSERTION;
  }
});

$("editRelevance").addEventListener("change", () => {
  if (isBusy() || state.coherenceFailure || state.selected?.kind !== "person") return;
  const index = state.selected.i;
  mutateDocument(() => { state.document.people[index].relevance = $("editRelevance").value; });
});

$("relevance").addEventListener("change", () => {
  if (state.contextTool === "person" && $("relevance").value) enterDrawPerson();
});
$("ignoreReason").addEventListener("change", () => {
  if (state.contextTool === "ignore" && $("ignoreReason").value) enterDrawIgnoreRegion();
});
$("imageSelect").addEventListener("change", () => loadImage(Number($("imageSelect").value)));
$("previous").addEventListener("click", () => loadImage(state.index - 1));
$("next").addEventListener("click", () => loadImage(state.index + 1));
$("panEdit").addEventListener("click", () => enterPanEdit());
$("newPerson").addEventListener("click", enterDrawPerson);
$("addComponent").addEventListener("click", enterAddVisibleComponent);
$("newIgnore").addEventListener("click", enterDrawIgnoreRegion);
$("finishPolygon").addEventListener("click", finishPolygon);
$("cancelPolygon").addEventListener("click", cancelPolygon);
$("deletePerson").addEventListener("click", deleteSelected);
$("deleteIgnore").addEventListener("click", deleteSelected);
$("undo").addEventListener("click", undo);
$("redo").addEventListener("click", redo);
$("finalize").addEventListener("click", finalize);
$("reveal").addEventListener("click", reveal);

document.querySelectorAll("[data-view]").forEach(button => {
  button.addEventListener("click", () => {
    if (isBusy() || state.coherenceFailure) return;
    const action = button.dataset.view;
    if (action === "fitWidth" || action === "fitHeight") fit(action);
    else if (action === "zoomIn") zoomAt(1.25);
    else if (action === "zoomOut") zoomAt(1 / 1.25);
    else if (action === "resetView") fit("fitWidth");
  });
});

function shortcutTarget(event) {
  return event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement || event.target instanceof HTMLTextAreaElement;
}

window.addEventListener("keydown", event => {
  if (event.code === "Space" && isDrawingMode() && !state.finalized && !isBusy() && !state.coherenceFailure) {
    event.preventDefault();
    if (!state.spaceHeld) {
      state.spaceHeld = true;
      renderMode();
    }
    return;
  }
  if (shortcutTarget(event)) {
    if (event.key === "Escape") {
      event.target.blur();
      cancelPolygon();
    }
    return;
  }
  if (event.key === "Enter") {
    event.preventDefault();
    finishPolygon();
  } else if (event.key === "Escape") {
    event.preventDefault();
    cancelPolygon();
  } else if (event.key.toLowerCase() === "p") {
    enterDrawPerson();
  } else if (event.key.toLowerCase() === "c") {
    enterAddVisibleComponent();
  } else if (event.key.toLowerCase() === "i") {
    enterDrawIgnoreRegion();
  } else if (["v", "a"].includes(event.key.toLowerCase())) {
    enterPanEdit();
  } else if (event.key.toLowerCase() === "z") {
    undo();
  } else if (event.key.toLowerCase() === "y") {
    redo();
  } else if (/^[1-8]$/.test(event.key)) {
    document.querySelector(`[data-strip="${Number(event.key) - 1}"]`)?.click();
  } else if (event.key === "Delete") {
    deleteSelected();
  }
});

window.addEventListener("keyup", event => {
  if (event.code === "Space" && state.spaceHeld) {
    state.spaceHeld = false;
    state.panning = null;
    renderMode();
  }
});

window.addEventListener("blur", () => {
  if (state.spaceHeld) {
    state.spaceHeld = false;
    state.panning = null;
    renderMode();
  }
});
window.addEventListener("resize", render);

loadBootstrap().catch(error => setStatus(error.message, true));
