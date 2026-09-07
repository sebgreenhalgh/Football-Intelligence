"use strict";

const IMAGE_ID = "DG-004";
const TARGET_STRIP = 2;
const Mode = Object.freeze({PAN_EDIT: "PAN_EDIT", DRAW_PERSON: "DRAW_PERSON", ADD_COMPONENT: "ADD_COMPONENT"});
const MODE_LABEL = Object.freeze({
  [Mode.PAN_EDIT]: "PAN / REVIEW",
  [Mode.DRAW_PERSON]: "DRAW OMITTED PERSON",
  [Mode.ADD_COMPONENT]: "ADD VISIBLE COMPONENT",
});
const COMPLETION_ASSERTION = "I have reviewed the full image and annotated every individually evaluable visible human.";
const ADJUDICATION_ASSERTION = "I re-reviewed the full image, addressed the calibration audit findings, and annotated every individually evaluable visible human.";

const state = {
  bootstrap: null,
  document: null,
  metadata: null,
  revision: 0,
  finalized: false,
  busy: false,
  image: null,
  newPersonId: null,
  mode: Mode.PAN_EDIT,
  polygon: [],
  zoom: 1,
  spaceDown: false,
  pan: null,
};
window.state = state;

const byId = id => document.getElementById(id);
const clone = value => JSON.parse(JSON.stringify(value));

function actionId(prefix) {
  return `${prefix}-${Date.now()}-${crypto.randomUUID()}`;
}

function targetPerson() {
  if (!state.document || !state.newPersonId) return null;
  const matches = state.document.people.filter(person => person.instance_id === state.newPersonId);
  if (matches.length > 1) throw new Error("Frozen new person ID is duplicated");
  return matches[0] || null;
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.message || `${response.status} ${response.statusText}`);
  return payload;
}

function setStatus(message, kind = "") {
  const node = byId("status");
  node.textContent = message;
  node.className = `status ${kind}`.trim();
}

function setBusy(value) {
  state.busy = value;
  renderControls();
}

function setMode(mode) {
  state.mode = mode;
  if (mode === Mode.PAN_EDIT) state.polygon = [];
  byId("modeBadge").textContent = MODE_LABEL[mode];
  renderControls();
  draw();
}

function renderControls() {
  if (!state.document) return;
  const target = targetPerson();
  const drawing = state.mode !== Mode.PAN_EDIT;
  const disabled = state.busy || state.finalized;
  byId("panEdit").classList.toggle("active", state.mode === Mode.PAN_EDIT);
  byId("drawNewPerson").classList.toggle("active", state.mode === Mode.DRAW_PERSON);
  byId("addComponent").classList.toggle("active", state.mode === Mode.ADD_COMPONENT);
  byId("drawNewPerson").disabled = disabled || Boolean(target);
  byId("addComponent").disabled = disabled || !target;
  byId("finishPolygon").disabled = disabled || !drawing || state.polygon.length < 3;
  byId("cancelPolygon").disabled = disabled || !drawing;
  byId("targetNonMatch").checked = target?.relevance === "NON_MATCH_RELEVANT";
  byId("targetMatch").checked = target?.relevance === "MATCH_RELEVANT";
  byId("targetNonMatch").disabled = disabled || !target || drawing;
  byId("targetMatch").disabled = disabled || !target || drawing;
  byId("personState").textContent = target
    ? `${target.visible_mask_components.length} component${target.visible_mask_components.length === 1 ? "" : "s"}`
    : "Not drawn";
  const strips = [...state.document.reviewed_exhaustiveness_strips].sort((a, b) => a - b);
  byId("stripProgress").textContent = `${strips.length} / 8`;
  document.querySelectorAll("#stripButtons button").forEach(button => {
    button.classList.toggle("reviewed", strips.includes(Number(button.dataset.strip)));
    button.disabled = disabled || drawing;
  });
  byId("omissionConfirmed").checked = state.metadata.repair_checklist_addressed === true;
  byId("denseAssertion").checked = state.document.completion_assertion === COMPLETION_ASSERTION;
  byId("adjudicationAssertion").checked = state.metadata.adjudication_assertion === ADJUDICATION_ASSERTION;
  for (const id of ["omissionConfirmed", "denseAssertion", "adjudicationAssertion"])
    byId(id).disabled = disabled || drawing;
  const ready = Boolean(target)
    && ["MATCH_RELEVANT", "NON_MATCH_RELEVANT"].includes(target.relevance)
    && strips.length === 8
    && state.metadata.repair_checklist_addressed === true
    && state.document.completion_assertion === COMPLETION_ASSERTION
    && state.metadata.adjudication_assertion === ADJUDICATION_ASSERTION;
  byId("finalizeSequence2").disabled = disabled || drawing || !ready;
  byId("revision").textContent = `${IMAGE_ID} - sequence 2 - revision ${state.revision}${state.finalized ? " - immutable" : ""}`;
  byId("zoomValue").textContent = `${Math.round(state.zoom * 100)}%`;
  const shell = byId("canvasShell");
  shell.classList.toggle("drawing", drawing && !state.spaceDown);
  shell.classList.toggle("panning", Boolean(state.pan) || state.spaceDown);
}

function drawComponent(context, component, fill, stroke, width) {
  if (!component?.length) return;
  context.beginPath();
  context.moveTo(component[0].x, component[0].y);
  for (const point of component.slice(1)) context.lineTo(point.x, point.y);
  context.closePath();
  context.fillStyle = fill;
  context.strokeStyle = stroke;
  context.lineWidth = width;
  context.fill();
  context.stroke();
}

function draw() {
  if (!state.image || !state.document || !state.image.complete || !state.image.naturalWidth) return;
  const canvas = byId("annotationCanvas");
  if (canvas.width !== state.image.naturalWidth || canvas.height !== state.image.naturalHeight) {
    canvas.width = state.image.naturalWidth;
    canvas.height = state.image.naturalHeight;
  }
  canvas.style.width = `${Math.round(canvas.width * state.zoom)}px`;
  canvas.style.height = `${Math.round(canvas.height * state.zoom)}px`;
  const context = canvas.getContext("2d");
  context.drawImage(state.image, 0, 0);
  for (const person of state.document.people) {
    const target = person.instance_id === state.newPersonId;
    for (const component of person.visible_mask_components || person.canonical_components || []) {
      drawComponent(
        context,
        component,
        target ? "rgba(255, 209, 91, .34)" : "rgba(73, 216, 232, .12)",
        target ? "#ffd15b" : "#49d8e8",
        target ? 7 : 2,
      );
    }
    const components = person.visible_mask_components || person.canonical_components || [];
    if (target && components[0]?.length) {
      const points = components.flat();
      const left = Math.min(...points.map(point => point.x));
      const top = Math.min(...points.map(point => point.y));
      context.font = "700 28px system-ui";
      context.fillStyle = "#10151c";
      context.fillRect(left - 5, Math.max(0, top - 38), 175, 34);
      context.fillStyle = "#ffd15b";
      context.fillText(state.newPersonId, left, Math.max(27, top - 11));
    }
  }
  if (state.polygon.length) {
    context.beginPath();
    context.moveTo(state.polygon[0].x, state.polygon[0].y);
    for (const point of state.polygon.slice(1)) context.lineTo(point.x, point.y);
    context.strokeStyle = "#ff7f73";
    context.lineWidth = 6;
    context.stroke();
    for (const point of state.polygon) {
      context.beginPath();
      context.arc(point.x, point.y, 7, 0, Math.PI * 2);
      context.fillStyle = "#ff7f73";
      context.fill();
    }
  }
}

function setZoom(value, anchorX = null, anchorY = null) {
  if (!state.image) return;
  const shell = byId("canvasShell");
  const previous = state.zoom;
  const next = Math.min(2.5, Math.max(.12, value));
  const localX = anchorX ?? shell.clientWidth / 2;
  const localY = anchorY ?? shell.clientHeight / 2;
  const sourceX = (shell.scrollLeft + localX) / previous;
  const sourceY = (shell.scrollTop + localY) / previous;
  state.zoom = next;
  draw();
  shell.scrollLeft = sourceX * next - localX;
  shell.scrollTop = sourceY * next - localY;
  renderControls();
}

function fitView() {
  if (!state.image) return;
  const shell = byId("canvasShell");
  const fit = Math.min(
    (shell.clientWidth - 18) / state.image.naturalWidth,
    (shell.clientHeight - 18) / state.image.naturalHeight,
  );
  state.zoom = Math.max(.12, fit);
  draw();
  shell.scrollLeft = 0;
  shell.scrollTop = 0;
  renderControls();
}

function goToStrip(strip) {
  if (!state.image) return;
  const shell = byId("canvasShell");
  if (state.zoom < .55) setZoom(.55, 0, 0);
  const stripWidth = state.image.naturalWidth * state.zoom / 8;
  shell.scrollLeft = Math.max(0, strip * stripWidth - (shell.clientWidth - stripWidth) / 2);
  shell.scrollTop = 0;
}

function sourcePoint(event) {
  const rect = byId("annotationCanvas").getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(state.image.naturalWidth, (event.clientX - rect.left) * state.image.naturalWidth / rect.width)),
    y: Math.max(0, Math.min(state.image.naturalHeight, (event.clientY - rect.top) * state.image.naturalHeight / rect.height)),
  };
}

async function postAction(actionType, fixedActionId = null) {
  const payload = {
    action_id: fixedActionId || actionId(actionType.toLowerCase()),
    action_type: actionType,
    anonymous_dense_image_id: IMAGE_ID,
    pass_kind: "CALIBRATION_ADJUDICATION",
    expected_revision: state.revision,
    document: clone(state.document),
    adjudication_metadata: clone(state.metadata),
  };
  return requestJson("/api/action", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
}

async function saveMutation(message) {
  setBusy(true);
  setStatus("Saving the locked DG-004 sequence-2 draft...");
  try {
    const response = await postAction("SAVE_DRAFT");
    if (response.anonymous_dense_image_id !== IMAGE_ID) throw new Error("Save response image identity mismatch");
    state.revision = response.revision;
    state.document = response.document;
    state.metadata = response.adjudication_metadata;
    setStatus(message);
    draw();
  } catch (error) {
    setStatus(error.message, "error");
    await loadState();
  } finally {
    setBusy(false);
  }
}

async function mutate(mutator, message) {
  if (state.busy || state.finalized || state.mode !== Mode.PAN_EDIT) return;
  mutator();
  renderControls();
  await saveMutation(message);
}

async function finishPolygon() {
  if (state.busy || state.finalized || state.polygon.length < 3 || state.mode === Mode.PAN_EDIT) return;
  const component = clone(state.polygon);
  if (state.mode === Mode.DRAW_PERSON) {
    if (targetPerson()) throw new Error("The one permitted new person already exists");
    state.document.people.push({
      instance_id: state.newPersonId,
      relevance: null,
      visible_mask_components: [component],
    });
  } else {
    const target = targetPerson();
    if (!target) throw new Error("Draw the omitted person before adding a visible component");
    target.visible_mask_components.push(component);
  }
  state.polygon = [];
  state.mode = Mode.PAN_EDIT;
  byId("modeBadge").textContent = MODE_LABEL[state.mode];
  renderControls();
  draw();
  await saveMutation("Visible geometry saved. Select the new person's relevance explicitly.");
}

async function loadState() {
  const serverState = await requestJson(`/api/state?image_id=${IMAGE_ID}`);
  state.document = serverState.document;
  state.metadata = serverState.adjudication_metadata;
  state.revision = serverState.revision;
  state.finalized = serverState.finalized;
  state.newPersonId = serverState.dg004_sequence2_new_person_id;
  state.mode = Mode.PAN_EDIT;
  state.polygon = [];
  byId("newPersonHeading").textContent = `${state.newPersonId} visible mask + relevance`;
  byId("modeBadge").textContent = MODE_LABEL[state.mode];
  renderControls();
  draw();
  if (state.finalized) setStatus("DG-004 sequence 2 is immutably finalized.", "success");
}

async function finalize() {
  if (state.busy || state.finalized || byId("finalizeSequence2").disabled) return;
  setBusy(true);
  setStatus("Finalizing one immutable DG-004 sequence-2 event and acknowledgement...");
  try {
    const fixedId = actionId("finalize");
    const response = await postAction("FINALIZE", fixedId);
    if (response.anonymous_dense_image_id !== IMAGE_ID || response.finalized !== true)
      throw new Error("Finalize response identity mismatch");
    state.revision = response.revision;
    state.finalized = true;
    state.mode = Mode.PAN_EDIT;
    setStatus(`Sequence 2 finalized: ${response.event_id}`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
}

async function boot() {
  try {
    state.bootstrap = await requestJson("/api/bootstrap");
    state.newPersonId = state.bootstrap.repair.new_person_id;
    byId("guidancePrimary").textContent = state.bootstrap.guidance[0];
    byId("guidanceAction").textContent = state.bootstrap.guidance[1];
    for (let strip = 0; strip < 8; strip += 1) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.strip = String(strip);
      button.textContent = `Strip ${strip}${strip === TARGET_STRIP ? " - target" : ""}`;
      button.classList.toggle("target-strip", strip === TARGET_STRIP);
      button.addEventListener("click", () => {
        goToStrip(strip);
        mutate(() => {
          const values = new Set(state.document.reviewed_exhaustiveness_strips);
          if (values.has(strip)) values.delete(strip); else values.add(strip);
          state.document.reviewed_exhaustiveness_strips = [...values].sort((a, b) => a - b);
        }, `Strip ${strip} confirmation saved.`);
      });
      byId("stripButtons").append(button);
    }
    byId("panEdit").addEventListener("click", () => setMode(Mode.PAN_EDIT));
    byId("drawNewPerson").addEventListener("click", () => {
      if (!targetPerson()) { state.polygon = []; setMode(Mode.DRAW_PERSON); goToStrip(TARGET_STRIP); }
    });
    byId("addComponent").addEventListener("click", () => {
      if (targetPerson()) { state.polygon = []; setMode(Mode.ADD_COMPONENT); goToStrip(TARGET_STRIP); }
    });
    byId("finishPolygon").addEventListener("click", finishPolygon);
    byId("cancelPolygon").addEventListener("click", () => setMode(Mode.PAN_EDIT));
    byId("targetNonMatch").addEventListener("change", () => mutate(
      () => { targetPerson().relevance = "NON_MATCH_RELEVANT"; },
      `${state.newPersonId} relevance saved as NON_MATCH_RELEVANT.`,
    ));
    byId("targetMatch").addEventListener("change", () => mutate(
      () => { targetPerson().relevance = "MATCH_RELEVANT"; },
      `${state.newPersonId} relevance saved as MATCH_RELEVANT.`,
    ));
    byId("omissionConfirmed").addEventListener("change", event => mutate(
      () => { state.metadata.repair_checklist_addressed = event.target.checked; },
      "Omission-repair confirmation saved.",
    ));
    byId("denseAssertion").addEventListener("change", event => mutate(
      () => { state.document.completion_assertion = event.target.checked ? COMPLETION_ASSERTION : null; },
      "Dense-Gold assertion state saved.",
    ));
    byId("adjudicationAssertion").addEventListener("change", event => mutate(
      () => { state.metadata.adjudication_assertion = event.target.checked ? ADJUDICATION_ASSERTION : null; },
      "Adjudication assertion state saved.",
    ));
    byId("finalizeSequence2").addEventListener("click", finalize);
    byId("zoomIn").addEventListener("click", () => setZoom(state.zoom * 1.25));
    byId("zoomOut").addEventListener("click", () => setZoom(state.zoom / 1.25));
    byId("fitView").addEventListener("click", fitView);
    byId("targetStrip").addEventListener("click", () => goToStrip(TARGET_STRIP));
    const shell = byId("canvasShell");
    const canvas = byId("annotationCanvas");
    canvas.addEventListener("pointerdown", event => {
      if (state.busy || state.finalized) return;
      if (state.mode === Mode.PAN_EDIT || state.spaceDown || event.button === 1) {
        state.pan = {x: event.clientX, y: event.clientY, left: shell.scrollLeft, top: shell.scrollTop};
        canvas.setPointerCapture(event.pointerId);
      } else if (event.button === 0) {
        state.polygon.push(sourcePoint(event));
        renderControls();
        draw();
      }
    });
    canvas.addEventListener("pointermove", event => {
      if (!state.pan) return;
      shell.scrollLeft = state.pan.left - (event.clientX - state.pan.x);
      shell.scrollTop = state.pan.top - (event.clientY - state.pan.y);
    });
    const stopPan = () => { state.pan = null; renderControls(); };
    canvas.addEventListener("pointerup", stopPan);
    canvas.addEventListener("pointercancel", stopPan);
    canvas.addEventListener("contextmenu", event => event.preventDefault());
    shell.addEventListener("wheel", event => {
      if (!event.ctrlKey) return;
      event.preventDefault();
      const rect = shell.getBoundingClientRect();
      setZoom(state.zoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15), event.clientX - rect.left, event.clientY - rect.top);
    }, {passive: false});
    window.addEventListener("keydown", event => {
      if (event.code === "Space" && !event.repeat) { state.spaceDown = true; renderControls(); event.preventDefault(); }
      if (event.key === "Escape") setMode(Mode.PAN_EDIT);
      if (event.key === "Enter" && state.mode !== Mode.PAN_EDIT) finishPolygon();
    });
    window.addEventListener("keyup", event => {
      if (event.code === "Space") { state.spaceDown = false; state.pan = null; renderControls(); }
    });
    state.image = new Image();
    state.image.onload = () => { fitView(); draw(); };
    state.image.src = state.bootstrap.queue[0].image_url;
    await loadState();
    if (!state.finalized) setStatus("Go to strip 2, draw exactly the omitted person, then classify and confirm 8/8.");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

window.sequence2Api = {postAction, loadState, targetPerson, setMode, finishPolygon, goToStrip};
boot();
