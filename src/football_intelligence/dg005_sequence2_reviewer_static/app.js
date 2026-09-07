"use strict";

const IMAGE_ID = "DG-005";
const TARGET_ID = "person-030";
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
};
window.state = state;

const byId = id => document.getElementById(id);
const clone = value => JSON.parse(JSON.stringify(value));

function actionId(prefix) {
  return `${prefix}-${Date.now()}-${crypto.randomUUID()}`;
}

function targetPerson() {
  const matches = state.document.people.filter(person => person.instance_id === TARGET_ID);
  if (matches.length !== 1) throw new Error("Locked target person is not unique");
  return matches[0];
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

function renderControls() {
  if (!state.document) return;
  const target = targetPerson();
  byId("targetNonMatch").checked = target.relevance === "NON_MATCH_RELEVANT";
  byId("targetMatch").checked = target.relevance === "MATCH_RELEVANT";
  byId("currentRelevance").textContent = target.relevance;
  const strips = [...state.document.reviewed_exhaustiveness_strips].sort((a, b) => a - b);
  byId("stripProgress").textContent = `${strips.length} / 8`;
  document.querySelectorAll("#stripButtons button").forEach(button => {
    button.classList.toggle("reviewed", strips.includes(Number(button.dataset.strip)));
    button.disabled = state.busy || state.finalized;
  });
  byId("officialConfirmed").checked = state.metadata.repair_checklist_addressed === true;
  byId("denseAssertion").checked = state.document.completion_assertion === COMPLETION_ASSERTION;
  byId("adjudicationAssertion").checked = state.metadata.adjudication_assertion === ADJUDICATION_ASSERTION;
  for (const id of ["targetNonMatch", "targetMatch", "officialConfirmed", "denseAssertion", "adjudicationAssertion"])
    byId(id).disabled = state.busy || state.finalized;
  const ready = target.relevance === "MATCH_RELEVANT"
    && strips.length === 8
    && state.metadata.repair_checklist_addressed === true
    && state.document.completion_assertion === COMPLETION_ASSERTION
    && state.metadata.adjudication_assertion === ADJUDICATION_ASSERTION;
  byId("finalizeSequence2").disabled = state.busy || state.finalized || !ready;
  byId("revision").textContent = `${IMAGE_ID} · sequence 2 · revision ${state.revision}${state.finalized ? " · immutable" : ""}`;
}

function draw() {
  if (!state.image || !state.document) return;
  const canvas = byId("annotationCanvas");
  canvas.width = state.image.naturalWidth;
  canvas.height = state.image.naturalHeight;
  const context = canvas.getContext("2d");
  context.drawImage(state.image, 0, 0);
  for (const person of state.document.people) {
    const target = person.instance_id === TARGET_ID;
    const components = person.visible_mask_components || person.canonical_components || [];
    for (const component of components) {
      if (!component.length) continue;
      context.beginPath();
      context.moveTo(component[0].x, component[0].y);
      for (const point of component.slice(1)) context.lineTo(point.x, point.y);
      context.closePath();
      context.fillStyle = target ? "rgba(255, 209, 91, .35)" : "rgba(73, 216, 232, .14)";
      context.strokeStyle = target ? "#ffd15b" : "#49d8e8";
      context.lineWidth = target ? 8 : 2;
      context.fill();
      context.stroke();
    }
    if (target && components[0]?.length) {
      const left = Math.min(...components[0].map(point => point.x));
      const top = Math.min(...components[0].map(point => point.y));
      context.font = "700 30px system-ui";
      context.fillStyle = "#10151c";
      context.fillRect(left - 6, top - 42, 185, 38);
      context.fillStyle = "#ffd15b";
      context.fillText("person-030", left, top - 13);
    }
  }
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
  setStatus("Saving locked sequence-2 confirmation…");
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
  if (state.busy || state.finalized) return;
  mutator();
  renderControls();
  await saveMutation(message);
}

async function loadState() {
  const serverState = await requestJson(`/api/state?image_id=${IMAGE_ID}`);
  state.document = serverState.document;
  state.metadata = serverState.adjudication_metadata;
  state.revision = serverState.revision;
  state.finalized = serverState.finalized;
  renderControls();
  draw();
  if (state.finalized) setStatus("Sequence 2 is immutably finalized.", "success");
}

async function finalize() {
  if (state.busy || state.finalized || byId("finalizeSequence2").disabled) return;
  setBusy(true);
  setStatus("Finalizing one immutable sequence-2 event and acknowledgement…");
  try {
    const response = await postAction("FINALIZE");
    if (response.anonymous_dense_image_id !== IMAGE_ID || response.finalized !== true)
      throw new Error("Finalize response identity mismatch");
    state.revision = response.revision;
    state.finalized = true;
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
    byId("guidancePrimary").textContent = state.bootstrap.guidance[0];
    byId("guidanceAction").textContent = state.bootstrap.guidance[1];
    for (let strip = 0; strip < 8; strip += 1) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.strip = String(strip);
      button.textContent = `Strip ${strip + 1}`;
      button.addEventListener("click", () => mutate(() => {
        const values = new Set(state.document.reviewed_exhaustiveness_strips);
        if (values.has(strip)) values.delete(strip); else values.add(strip);
        state.document.reviewed_exhaustiveness_strips = [...values].sort((a, b) => a - b);
      }, `Strip ${strip + 1} confirmation saved.`));
      byId("stripButtons").append(button);
    }
    byId("targetNonMatch").addEventListener("change", () => mutate(
      () => { targetPerson().relevance = "NON_MATCH_RELEVANT"; },
      "person-030 remains NON_MATCH_RELEVANT; finalization stays blocked.",
    ));
    byId("targetMatch").addEventListener("change", () => mutate(
      () => { targetPerson().relevance = "MATCH_RELEVANT"; },
      "person-030 correction saved as MATCH_RELEVANT.",
    ));
    byId("officialConfirmed").addEventListener("change", event => mutate(
      () => { state.metadata.repair_checklist_addressed = event.target.checked; },
      "Assistant-referee confirmation saved.",
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
    state.image = new Image();
    state.image.onload = draw;
    state.image.src = state.bootstrap.queue[0].image_url;
    await loadState();
    if (!state.finalized) setStatus("Confirm the correction, all 8 strips, and both exact assertions.");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

window.sequence2Api = {postAction, loadState, targetPerson};
boot();
