"use strict";

// R6 is intentionally a server-state renderer.  Human controls dispatch intent;
// only a complete canonical draft returned by /api/action replaces app.draft.
const $ = (id) => document.getElementById(id);
const ui = {};
[
  "modePill", "progressText", "progressFill", "saveState", "helpButton", "practiceBanner",
  "resetPractice", "legacyDraftNotice", "resetLegacyPractice", "previewBanner", "blockingError",
  "reviewShell", "welcomeScreen", "completionScreen", "caseEyebrow", "caseTitle", "overlayToggle",
  "subjectToggle", "idToggle", "panoramaWrap", "panoramaCanvas", "assetState", "mappingState",
  "fitButton", "zoomOutButton", "zoomInButton", "resetViewButton", "zoomSubjectButton",
  "fullScreenButton", "zoomPercent", "lockViewToggle", "firstFrameButton", "previousFrameButton",
  "playButton", "nextFrameButton", "centreFrameButton", "lastFrameButton", "timeline", "focusWrap",
  "focusCanvas", "focusAssetState", "focusFitButton", "focusZoomOutButton", "focusZoomInButton",
  "focusResetButton", "focusZoomSubjectButton", "focusZoomPercent", "subjectReference", "questionStep",
  "visualModeAuto", "visualModeOriginal", "visualModeEnhanced", "visualModeStatus",
  "subjectPill", "questionLegend", "questionKicker", "questionTitle", "questionHelp", "answerArea",
  "backButton", "continueButton", "startRealButton", "startPracticeButton", "completionTitle",
  "completionCount", "trancheReceipt", "lastEvent", "globalReceiptRow", "globalReceipt", "pauseMessage",
  "reviewCompletedButton", "nextTrancheButton", "confirmDialog", "helpDrawer", "closeHelp",
].forEach((id) => { ui[id] = $(id); });

const REVISION = "G7E_B_R6_SERVER_AUTHORITATIVE_ACTION_REDUCER_V1";
const app = {
  mode: null, cases: [], current: null, draft: null, contract: null, contractHash: null,
  actionContract: null, actionContractHash: null, frame: 4, image: null, focusImage: null,
  assetReady: false, mappingVerified: false, pending: false, readOnly: false,
  inputMode: "pan", view: { zoom: 1, centerX: .5, centerY: .5 },
  focusView: { zoom: 1 }, drag: null, productionBundleSha256: null,
  visualPreference: localStorage.getItem("fi.temporal_review.visual_mode") || "AUTO", resolvedVisualMode: "ORIGINAL",
};
if (!["AUTO", "ORIGINAL", "ENHANCED"].includes(app.visualPreference)) app.visualPreference = "AUTO";

function block(message, kind = "runtime") {
  ui.blockingError.classList.remove("hidden");
  ui.blockingError.dataset.errorKind = kind;
  ui.blockingError.textContent = message;
  ui.saveState.textContent = "Stopped safely";
}
function clearBlock() { ui.blockingError.classList.add("hidden"); ui.blockingError.textContent = ""; }
async function api(path, payload) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const body = await response.json();
  if (!response.ok || body.ok === false) {
    const error = new Error(body.error || body.errors?.[0]?.message || `HTTP ${response.status}`);
    error.payload = body; error.httpStatus = response.status; throw error;
  }
  return body;
}
async function getJson(path) { const response = await fetch(path); if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); }
async function sha256Hex(buffer) { return [...new Uint8Array(await crypto.subtle.digest("SHA-256", buffer))].map((v) => v.toString(16).padStart(2, "0")).join(""); }

function familyOf(key = app.draft?.current_question_instance_key || "") { return key.split("|").at(-1); }
function parseKey(key = app.draft?.current_question_instance_key || "") {
  const parts = key.split("|");
  const token = parts.find((v) => /^SUBJECT_[ABC]$/.test(v)) || null;
  const framePart = parts.find((v) => /^frame_\d+$/.test(v));
  return { family: parts.at(-1), token, subject: token ? token.charCodeAt(8) - 65 : null, frame: framePart ? Number(framePart.slice(6)) : null };
}
function currentAnswer() { return app.draft?.answered_domain_values?.[app.draft.current_question_instance_key]; }
function currentLifecycle() { return app.draft?.question_lifecycle?.[app.draft.current_question_instance_key]; }
function frameRow(subject, frame) { return app.draft.subjects[subject].frame_observations[frame]; }
function candidates(frame = app.frame) { return app.current?.frame_candidates?.[frame] || []; }
function label(domain, value) { return app.contract?.domain_labels?.[domain]?.[value] || String(value).replaceAll("_", " "); }

async function dispatch(actionType, payload = {}, questionKey = app.draft.current_question_instance_key) {
  if (app.pending || app.readOnly) return null;
  app.pending = true; setControls(false); ui.saveState.textContent = "Saving this answer…";
  const actionId = crypto.randomUUID();
  const envelope = {
    schema_version: "football_intelligence.g7e_b_r6.browser_action.v1",
    action_id: actionId,
    idempotency_key: actionId,
    review_revision: REVISION,
    contract_hash: app.actionContractHash,
    mode: app.mode,
    tranche_id: app.current.tranche_id,
    burst_id: app.current.burst_id,
    expected_draft_revision: app.draft.draft_version,
    expected_draft_sha256: app.draft.draft_content_sha256,
    question_instance_key: questionKey,
    action_type: actionType,
    payload,
    client_timestamp: new Date().toISOString(),
  };
  try {
    const response = await api("/api/action", envelope);
    app.draft = response.draft;
    ui.saveState.textContent = "Progress saved to server";
    clearBlock();
    await alignFrameToQuestion();
    renderQuestion(); draw();
    return response;
  } catch (error) {
    block(`ACTION_REJECTED · ${error.payload?.error_code || error.message}`, "server-action");
    throw error;
  } finally { app.pending = false; setControls(true); }
}

function setControls(enabled) {
  document.querySelectorAll("button").forEach((button) => {
    if (!button.closest(".topbar") && !button.closest(".help-drawer") && !button.closest(".visual-mode-switch")) {
      button.disabled = !enabled || app.readOnly;
    }
  });
}

function domainOptions(domain) { return (app.contract?.domain_enums?.[domain] || []).map((value) => [value, label(domain, value)]); }
function answerCards(domain, options = domainOptions(domain), actionType = "ANSWER_QUESTION") {
  ui.answerArea.innerHTML = "";
  const selected = currentAnswer();
  options.forEach(([value, text], index) => {
    const button = document.createElement("button");
    button.type = "button"; button.className = `answer-card${selected === value ? " selected" : ""}`;
    button.dataset.value = value; button.dataset.shortcut = String(index + 1);
    button.innerHTML = `<span>${text}</span>`;
    button.onclick = () => dispatch(actionType, { value });
    ui.answerArea.appendChild(button);
  });
}

const TITLES = {
  original_focus: ["What does the yellow original focus box contain?", "Judge the yellow box only. Blue is nearby context."],
  context_subject: ["Is there a relevant match person in the blue context area?", "The yellow box contained no relevant person."],
  uncertain_focus_path: ["What would you like to do with the uncertain yellow box?", "You do not need to force a guess."],
  multi_subject_b: ["Should another person in the yellow box become Subject B?", "Add only a genuinely separate person."],
  anchor: ["Click the subject in the clearest frame", "Choose a frame, then click the person."],
  location: ["Where is this subject in this frame?", "Answer first, then click visible people in the large frame."],
  marker_review: ["Do the nine markers follow the same subject?", "Check only this burst-local subject."],
  supply: ["Which boxes provide useful evidence for this subject?", "Answer, then click the exact model box or boxes."],
  relationship: ["How do the selected boxes relate?", "This follow-up is specific to the current supply answer."],
  occlusion: ["Describe the occlusion sequence", "Confirm the entry, maintained occlusion, and exit pattern."],
  continuity: ["Is this the same burst-local subject?", "No identity beyond this short burst is requested."],
  role: ["What is this person's role?", "Judge this burst only."],
  participation: ["How are they participating?", "Judge this burst only."],
  certainty: ["How sure are you?", "Not sure is always available."],
  additional_subject: ["Is there another subject to review?", "Add only when the evidence requires it."],
  missed_check: ["Can you see any relevant person with no useful model box?", "Review the whole nine-frame burst."],
  missed_mark: ["Click the centre of each missed relevant person", "Use any frame; click Done marking when the set is complete."],
  summary: ["Review this burst before saving", "This summary appears only after server authorization."],
};

function setQuestion(title, help, kicker = "SERVER-AUTHORITATIVE · ONE STEP") {
  ui.questionTitle.textContent = title; ui.questionHelp.textContent = help; ui.questionKicker.textContent = kicker;
}
function relationshipOptions(part) {
  const row = frameRow(part.subject, part.frame);
  const supply = app.contract.relationship_compatibility.supply_states[row.observation_supply];
  const branch = app.contract.relationship_compatibility.question_families[supply.question_family];
  return branch.options;
}
function questionReady(part) {
  const state = currentLifecycle();
  if (state !== "ANSWERED" && state !== "SKIPPED_NOT_APPLICABLE") return false;
  if (part.family === "anchor") return Array.isArray(app.draft.subjects[part.subject]?.anchor_source_xy);
  if (part.family === "location") {
    const row = frameRow(part.subject, part.frame);
    if (["VISIBLE_COMPLETE", "VISIBLE_PARTIAL"].includes(row.visibility)) return row.human_confirmed === true;
  }
  if (part.family === "supply") {
    const row = frameRow(part.subject, part.frame);
    const branch = app.contract.relationship_compatibility.supply_states[row.observation_supply];
    if (!branch) return false;
    const count = row.selected_candidate_ids.length;
    return count >= Number(branch.minimum_selected_count) && (branch.maximum_selected_count === null || count <= Number(branch.maximum_selected_count));
  }
  if (part.family === "missed_mark") return app.draft.missed_marking_complete && app.draft.missed_person_marks.length > 0;
  return true;
}

function renderQuestion() {
  if (!app.draft) return;
  const part = parseKey(); const [title, help] = TITLES[part.family] || [part.family, "Server-backed question"];
  setQuestion(title, help);
  ui.questionStep.textContent = `Question · ${part.family.replaceAll("_", " ")}`;
  ui.subjectPill.textContent = part.token ? part.token.replace("SUBJECT_", "Subject ") : "";
  ui.answerArea.innerHTML = ""; app.inputMode = "pan";
  const domain = app.contract.question_families[part.family]?.domain;
  if (domain) {
    const options = part.family === "relationship" ? relationshipOptions(part) : domainOptions(domain);
    answerCards(domain, options, ["marker_review", "occlusion", "continuity"].includes(part.family) ? "CONFIRM_SUBJECT_CONTINUITY" : "ANSWER_QUESTION");
  }
  if (part.family === "anchor") {
    app.inputMode = "subject-location";
    ui.answerArea.innerHTML = `<div class="click-required"><b>${Array.isArray(app.draft.subjects[part.subject]?.anchor_source_xy) ? "Human anchor recorded" : "Waiting for a source-coordinate click"}</b><br>Use the large football frame.</div>`;
  }
  if (part.family === "location") {
    const row = frameRow(part.subject, part.frame);
    if (["VISIBLE_COMPLETE", "VISIBLE_PARTIAL", "FULLY_OCCLUDED_EXPECTED_PRESENT"].includes(row.visibility)) app.inputMode = "subject-location";
    const note = document.createElement("div"); note.className = "click-required";
    note.innerHTML = `<b>${row.human_confirmed ? "Location acknowledged" : "Click the subject when visible"}</b><br>Frame ${part.frame + 1} · canonical source coordinates.`;
    ui.answerArea.appendChild(note);
  }
  if (part.family === "supply") {
    app.inputMode = "candidate-selection";
    const row = frameRow(part.subject, part.frame); const note = document.createElement("div"); note.className = "selection-status";
    note.innerHTML = `<b>${row.selected_candidate_ids.length} selected box${row.selected_candidate_ids.length === 1 ? "" : "es"}</b><br>Click exact white boxes in the large frame to select or deselect.`;
    ui.answerArea.appendChild(note);
  }
  if (part.family === "missed_mark") {
    app.inputMode = "missed-mark";
    const list = document.createElement("div"); list.className = "click-required";
    list.innerHTML = `<b>${app.draft.missed_person_marks.length} missed people marked</b><br>Click the large frame to add. ${app.draft.missed_marking_complete ? "Done marking acknowledged." : ""}`;
    app.draft.missed_person_marks.forEach((mark, index) => {
      const remove = document.createElement("button"); remove.type = "button"; remove.textContent = `Remove mark ${index + 1}`;
      remove.onclick = () => dispatch("REMOVE_MISSED_PERSON_MARK", { mark_id: mark.mark_id }); list.appendChild(remove);
    });
    const done = document.createElement("button"); done.id = "doneMarkingButton"; done.type = "button"; done.className = "primary"; done.textContent = "Done marking";
    done.disabled = app.draft.missed_person_marks.length === 0; done.onclick = () => dispatch("COMPLETE_MISSED_PERSON_MARKING"); list.appendChild(done);
    ui.answerArea.appendChild(list);
  }
  if (part.family === "summary") renderSummary();
  ui.backButton.disabled = app.readOnly || app.pending || !(app.draft.navigation_history || []).length;
  ui.continueButton.textContent = part.family === "summary" ? "Check and save" : "Continue";
  ui.continueButton.disabled = app.readOnly || app.pending || !questionReady(part) || (part.family === "summary" && !app.draft.summary_ready);
  ui.questionLegend.classList.toggle("hidden", !["original_focus", "context_subject"].includes(part.family));
}

function renderSummary() {
  if (!app.draft.summary_ready) { block("SUMMARY_BARRIER · the server has not authorized this summary", "summary"); return; }
  const subjectRows = app.draft.subjects.map((subject) => `<li><b>${subject.subject_token.replace("_", " ")}</b> · ${subject.role?.replaceAll("_", " ") || "role pending"}</li>`).join("");
  ui.answerArea.innerHTML = `<div class="release-gate"><b>SERVER-VERIFIED SUMMARY</b><br>Draft ${app.draft.draft_version} · ${app.draft.draft_content_sha256}</div><ul class="summary-list"><li><b>Yellow box:</b> ${app.draft.answers.original_focus_box_answer?.replaceAll("_", " ")}</li>${subjectRows || "<li>No burst-local subject was followed.</li>"}<li><b>Whole-burst missed-person check:</b> ${app.draft.answers.missed_check}</li><li><b>Missed-person marks:</b> ${app.draft.missed_person_marks.length}</li></ul>`;
  if (app.draft.real_draft_recovery || app.draft.migration_record) {
    const recovered = document.createElement("div"); recovered.id = "realDraftRecovered"; recovered.className = "click-required";
    recovered.innerHTML = "<b>Real draft recovered — no event created</b><br>All 27 human marks and every answer/coordinate were preserved. Lifecycle metadata only was repaired."; ui.answerArea.prepend(recovered);
  }
}

async function saveFinal() {
  if (!app.draft.summary_ready || app.readOnly) return;
  app.pending = true; ui.continueButton.disabled = true; ui.saveState.textContent = "Server preflight…";
  const request = { mode: app.mode, burst_id: app.draft.burst_id, draft_version: app.draft.draft_version, draft_content_sha256: app.draft.draft_content_sha256, optimistic_lock_token: app.draft.optimistic_lock_token };
  try {
    const preflight = await api("/api/final-save-preflight", request);
    const result = await api("/api/save", { ...request, proposed_event_id: preflight.proposed_event_id, idempotency_key: preflight.idempotency_key });
    ui.saveState.textContent = `SAVED — SERVER ACKNOWLEDGED · ${result.event_id}`; app.readOnly = true;
    if (result.tranche_complete) renderCompletion({ ...result, tranche_id: app.current.tranche_id, last_event_id: result.event_id });
    else await loadMode(app.mode);
  } catch (error) { block(`FINAL SAVE ERROR · ${error.payload?.error_code || error.message}`, "final-save"); }
  finally { app.pending = false; renderQuestion(); }
}

async function advance() { if (familyOf() === "summary") return saveFinal(); await dispatch("NAVIGATE_FORWARD"); }
async function back() { await dispatch("NAVIGATE_BACK"); }

function resolvedVisualMode(frame) {
  const preference = ["AUTO", "ORIGINAL", "ENHANCED"].includes(app.visualPreference) ? app.visualPreference : "AUTO";
  return preference === "AUTO" ? (frame.auto_visual_mode || "ORIGINAL") : preference;
}
function visualRecord(frame) {
  const mode = resolvedVisualMode(frame);
  const record = frame.visual_modes?.[mode] || {
    panorama_url: frame.panorama_url, panorama_sha256: frame.panorama_sha256,
    focus_url: frame.focus_url, focus_sha256: frame.focus_sha256,
  };
  return { mode, record };
}
function setAssetMessage(element, message) {
  element.textContent = message;
  element.classList.toggle("visible", Boolean(message));
}
function updateVisualModeUi(frame = app.current?.frames?.[app.frame]) {
  const resolved = frame ? resolvedVisualMode(frame) : "ORIGINAL";
  app.resolvedVisualMode = resolved;
  document.querySelectorAll("[data-visual-mode]").forEach((button) => button.classList.toggle("active", button.dataset.visualMode === app.visualPreference));
  ui.visualModeStatus.textContent = app.visualPreference === "AUTO" ? `Auto · ${resolved === "ENHANCED" ? "Enhanced" : "Original"}` : resolved === "ENHANCED" ? "Enhanced · review only" : "Original · source truth";
}
async function setVisualPreference(mode) {
  if (!["AUTO", "ORIGINAL", "ENHANCED"].includes(mode) || app.pending) return;
  app.visualPreference = mode;
  localStorage.setItem("fi.temporal_review.visual_mode", mode);
  updateVisualModeUi();
  await loadFrame(app.frame);
}
async function verifiedImage(frame, kind) {
  const { record } = visualRecord(frame);
  const url = record[`${kind}_url`];
  const expected = record[`${kind}_sha256`];
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok || !response.headers.get("content-type")?.startsWith("image/")) throw new Error(`${kind} asset HTTP/MIME failure`);
  const bytes = await response.arrayBuffer(); const digest = await sha256Hex(bytes);
  if (digest !== expected) throw new Error(`${kind} asset hash mismatch`);
  const bitmap = await createImageBitmap(new Blob([bytes], { type: response.headers.get("content-type") }));
  return bitmap;
}
async function loadFrame(sequence) {
  app.frame = Math.max(0, Math.min(8, sequence)); app.assetReady = false; app.mappingVerified = false;
  setAssetMessage(ui.assetState, "Loading verified football frame…"); setAssetMessage(ui.focusAssetState, "Loading verified detail…");
  const frame = app.current.frames[app.frame];
  try {
    [app.image, app.focusImage] = await Promise.all([verifiedImage(frame, "panorama"), verifiedImage(frame, "focus")]);
    app.assetReady = true; app.mappingVerified = frame.source_width === app.current.source_width && frame.source_height === app.current.source_height;
    if (!app.mappingVerified || !app.image.width || !app.image.height) throw new Error("decoded frame mapping failed");
    setAssetMessage(ui.assetState, ""); setAssetMessage(ui.focusAssetState, ""); ui.mappingState.textContent = "VERIFIED"; updateVisualModeUi(frame); draw(); renderTimeline();
  } catch (error) { setAssetMessage(ui.assetState, "Image unavailable"); setAssetMessage(ui.focusAssetState, "Image unavailable"); setControls(false); block(`IMAGE_ASSET_ERROR · ${error.message}`, "image-asset"); }
}
async function alignFrameToQuestion() { const part = parseKey(); if (Number.isInteger(part.frame) && part.frame !== app.frame) await loadFrame(part.frame); }

function transform(canvas) {
  const rect = canvas.getBoundingClientRect(); const dpr = devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr)); canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const fit = Math.min(canvas.width / app.current.source_width, canvas.height / app.current.source_height);
  const scale = fit * app.view.zoom; const width = app.current.source_width * scale; const height = app.current.source_height * scale;
  const x = (canvas.width - width) / 2 + (.5 - app.view.centerX) * Math.max(0, width - canvas.width);
  const y = (canvas.height - height) / 2 + (.5 - app.view.centerY) * Math.max(0, height - canvas.height);
  return { dpr, scale, x, y, width, height };
}
function sourceToCanvas(point, t) { return [t.x + point[0] * t.scale, t.y + point[1] * t.scale]; }
function canvasToSource(event) {
  const rect = ui.panoramaCanvas.getBoundingClientRect(); const t = transform(ui.panoramaCanvas);
  return [((event.clientX - rect.left) * t.dpr - t.x) / t.scale, ((event.clientY - rect.top) * t.dpr - t.y) / t.scale];
}
function drawBox(ctx, box, t, colour, width = 3) {
  const [x1, y1] = sourceToCanvas([box[0], box[1]], t); const [x2, y2] = sourceToCanvas([box[2], box[3]], t);
  ctx.strokeStyle = colour; ctx.lineWidth = width * t.dpr; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
}
function draw() {
  if (!app.assetReady || !app.image) return;
  const canvas = ui.panoramaCanvas; const ctx = canvas.getContext("2d"); const t = transform(canvas);
  ctx.fillStyle = "#090f20"; ctx.fillRect(0, 0, canvas.width, canvas.height); ctx.drawImage(app.image, t.x, t.y, t.width, t.height);
  if (ui.overlayToggle.checked) candidates().forEach((candidate) => drawBox(ctx, candidate.source_box_xyxy, t, "rgba(255,255,255,.88)", 1.5));
  if (app.frame === 4) {
    (app.current.candidates || []).forEach((originalFocus) =>
      drawBox(ctx, originalFocus.source_box_xyxy, t, "#ffd84f", 4));
    ctx.setLineDash([12, 8]);
    drawBox(ctx, app.current.focus_crop_source_xyxy, t, "#59a7ff", 3);
    ctx.setLineDash([]);
  }
  const part = parseKey();
  if (part.family === "supply") frameRow(part.subject, part.frame).selected_candidate_ids.forEach((id) => { const c = candidates().find((row) => row.candidate_id === id); if (c) drawBox(ctx, c.source_box_xyxy, t, "#2cc9a0", 4); });
  if (ui.subjectToggle.checked) app.draft?.subjects?.forEach((subject, index) => { const row = subject.frame_observations[app.frame]; const point = Number.isFinite(row?.subject_location_source_x) ? [row.subject_location_source_x, row.subject_location_source_y] : (subject.anchor_frame_sequence === app.frame ? subject.anchor_source_xy : null); if (point) { const [x, y] = sourceToCanvas(point, t); ctx.fillStyle = ["#2cc9a0", "#9a72e8", "#e7a51a"][index]; ctx.beginPath(); ctx.arc(x, y, 8 * t.dpr, 0, Math.PI * 2); ctx.fill(); } });
  app.draft?.missed_person_marks?.filter((mark) => mark.frame_sequence === app.frame).forEach((mark, index) => { const [x, y] = sourceToCanvas(mark.source_xy, t); ctx.fillStyle = "#ff5e6d"; ctx.beginPath(); ctx.arc(x, y, 9 * t.dpr, 0, Math.PI * 2); ctx.fill(); ctx.fillStyle = "white"; ctx.font = `${12 * t.dpr}px sans-serif`; ctx.fillText(String(index + 1), x + 11 * t.dpr, y); });
  const focus = ui.focusCanvas; const fctx = focus.getContext("2d"); const frect = focus.getBoundingClientRect(); const dpr = devicePixelRatio || 1; focus.width = Math.max(1, Math.round(frect.width * dpr)); focus.height = Math.max(1, Math.round(frect.height * dpr)); fctx.fillStyle = "#090f20"; fctx.fillRect(0, 0, focus.width, focus.height); const fs = Math.min(focus.width / app.focusImage.width, focus.height / app.focusImage.height) * app.focusView.zoom; const fw = app.focusImage.width * fs, fh = app.focusImage.height * fs; fctx.drawImage(app.focusImage, (focus.width - fw) / 2, (focus.height - fh) / 2, fw, fh);
  ui.zoomPercent.textContent = `${Math.round(app.view.zoom * 100)}%`; ui.focusZoomPercent.textContent = `${Math.round(app.focusView.zoom * 100)}%`;
}

async function handleCanvasClick(event) {
  if (!app.assetReady || !app.mappingVerified || app.pending || app.readOnly) return;
  const point = canvasToSource(event); if (point.some((v, i) => v < 0 || v > (i ? app.current.source_height : app.current.source_width))) return;
  const part = parseKey();
  if (app.inputMode === "subject-location") return dispatch("SET_SUBJECT_LOCATION", { source_xy: point, frame_sequence: app.frame, approximate_hidden_location: currentAnswer() === "FULLY_OCCLUDED_EXPECTED_PRESENT" });
  if (app.inputMode === "missed-mark") return dispatch("ADD_MISSED_PERSON_MARK", { source_xy: point, frame_sequence: app.frame, mark_id: crypto.randomUUID() });
  if (app.inputMode === "candidate-selection") {
    const hits = candidates().filter((c) => point[0] >= c.source_box_xyxy[0] && point[0] <= c.source_box_xyxy[2] && point[1] >= c.source_box_xyxy[1] && point[1] <= c.source_box_xyxy[3]).sort((a, b) => (a.source_box_xyxy[2]-a.source_box_xyxy[0])*(a.source_box_xyxy[3]-a.source_box_xyxy[1]) - (b.source_box_xyxy[2]-b.source_box_xyxy[0])*(b.source_box_xyxy[3]-b.source_box_xyxy[1]));
    if (!hits.length) return; const selected = frameRow(part.subject, part.frame).selected_candidate_ids.includes(hits[0].candidate_id);
    return dispatch(selected ? "DESELECT_CANDIDATE" : "SELECT_CANDIDATE", { candidate_id: hits[0].candidate_id });
  }
}
function renderTimeline() { ui.timeline.innerHTML = app.current.frames.map((frame, index) => { const { mode, record } = visualRecord(frame); return `<button type="button" data-frame="${index}" data-frame-visual-mode="${mode}" class="${index === app.frame ? "active" : ""}"><img src="${record.panorama_url}" alt="Frame ${index + 1} ${mode.toLowerCase()} view"><b>${index + 1}</b><span>${frame.relative_offset_seconds > 0 ? "+" : ""}${frame.relative_offset_seconds.toFixed(1)}s</span></button>`; }).join(""); ui.timeline.querySelectorAll("[data-frame]").forEach((button) => { button.onclick = () => loadFrame(Number(button.dataset.frame)); }); }

async function loadMode(mode) {
  clearBlock(); app.mode = mode; const bootstrap = await getJson(`/api/bootstrap?mode=${encodeURIComponent(mode)}`);
  ui.previewBanner.textContent = "R6.1 FINAL-BYTE REVIEW PREVIEW — NO NEW REAL HUMAN TRUTH";
  ui.previewBanner.classList.toggle("hidden", bootstrap.acceptance_temporary !== true);
  if (bootstrap.release_gate?.required && !bootstrap.release_gate.valid && mode === "real" && bootstrap.state?.editable !== true) throw new Error(`REAL_REVIEW_TEMPORARILY_LOCKED · ${bootstrap.release_gate.failures.join(", ")}`);
  app.cases = bootstrap.cases; app.contract = bootstrap.canonical_contract; app.contractHash = bootstrap.canonical_contract_sha256; app.actionContract = bootstrap.server_action_contract; app.actionContractHash = bootstrap.server_action_contract_sha256;
  const state = bootstrap.state; if ((mode === "practice" && state.all_practice_complete) || state.tranche_complete) return renderCompletion(state);
  const burstId = state.first_incomplete_burst_id; app.current = app.cases.find((row) => row.burst_id === burstId); if (!app.current) throw new Error("first incomplete burst is unavailable");
  app.draft = state.draft || (await api("/api/initialize-draft", { mode, burst_id: burstId })).draft; app.readOnly = false; app.frame = Number(app.draft.current_frame_sequence ?? 4);
  ui.welcomeScreen.classList.add("hidden"); ui.completionScreen.classList.add("hidden"); ui.reviewShell.classList.remove("hidden"); ui.practiceBanner.classList.toggle("hidden", mode !== "practice");
  ui.modePill.textContent = mode === "practice" ? "Practice · not human truth" : app.current.tranche_id.replace("_", " "); ui.progressText.textContent = `${state.completed_count} of ${state.total_count}`; ui.progressFill.style.width = `${100 * state.completed_count / state.total_count}%`; ui.caseEyebrow.textContent = `MATCH ${app.current.match_id} · ${app.current.half.replaceAll("_", " ")}`; ui.caseTitle.textContent = app.current.burst_id;
  await loadFrame(app.frame); renderQuestion();
}
function renderCompletion(state) { ui.reviewShell.classList.add("hidden"); ui.welcomeScreen.classList.add("hidden"); ui.completionScreen.classList.remove("hidden"); ui.completionTitle.textContent = state.all_cases_complete ? "ALL CASES COMPLETE" : `${state.tranche_id || "PRACTICE"} COMPLETE`; ui.completionCount.textContent = `${state.completed_count || 20} complete`; ui.trancheReceipt.textContent = state.tranche_completion_receipt_id || "practice"; ui.lastEvent.textContent = state.last_event_id || "—"; ui.globalReceiptRow.classList.toggle("hidden", !state.global_completion_receipt_id); ui.globalReceipt.textContent = state.global_completion_receipt_id || "—"; }

ui.startRealButton.onclick = () => loadMode("real").catch((error) => block(error.message, "boot"));
ui.startPracticeButton.onclick = () => loadMode("practice").catch((error) => block(error.message, "boot"));
ui.visualModeAuto.onclick = () => setVisualPreference("AUTO");
ui.visualModeOriginal.onclick = () => setVisualPreference("ORIGINAL");
ui.visualModeEnhanced.onclick = () => setVisualPreference("ENHANCED");
ui.nextTrancheButton.onclick = async () => { try { await api("/api/tranche/start-next", { mode: "real", tranche_id: app.current.tranche_id }); await loadMode("real"); } catch (error) { block(`TRANCHE_START_ERROR · ${error.message}`, "server-action"); } };
ui.continueButton.onclick = () => advance(); ui.backButton.onclick = () => back(); ui.panoramaCanvas.addEventListener("click", handleCanvasClick);
ui.overlayToggle.onchange = draw; ui.subjectToggle.onchange = draw; ui.idToggle.onchange = draw;
ui.firstFrameButton.onclick = () => loadFrame(0); ui.previousFrameButton.onclick = () => loadFrame(app.frame - 1); ui.nextFrameButton.onclick = () => loadFrame(app.frame + 1); ui.centreFrameButton.onclick = () => loadFrame(4); ui.lastFrameButton.onclick = () => loadFrame(8);
ui.fitButton.onclick = () => { app.view = { zoom: 1, centerX: .5, centerY: .5 }; draw(); }; ui.resetViewButton.onclick = ui.fitButton.onclick;
ui.zoomInButton.onclick = () => { app.view.zoom = Math.min(8, app.view.zoom * 1.25); draw(); }; ui.zoomOutButton.onclick = () => { app.view.zoom = Math.max(1, app.view.zoom / 1.25); draw(); };
ui.focusZoomInButton.onclick = () => { app.focusView.zoom = Math.min(8, app.focusView.zoom * 1.25); draw(); }; ui.focusZoomOutButton.onclick = () => { app.focusView.zoom = Math.max(1, app.focusView.zoom / 1.25); draw(); }; ui.focusFitButton.onclick = ui.focusResetButton.onclick = () => { app.focusView.zoom = 1; draw(); };
ui.fullScreenButton.onclick = () => ui.panoramaWrap.requestFullscreen();
ui.helpButton.onclick = () => { ui.helpDrawer.setAttribute("aria-hidden", "false"); ui.helpDrawer.classList.add("open"); }; ui.closeHelp.onclick = () => { ui.helpDrawer.setAttribute("aria-hidden", "true"); ui.helpDrawer.classList.remove("open"); };
window.addEventListener("resize", draw);
window.addEventListener("keydown", (event) => { if (/^[1-9]$/.test(event.key)) document.querySelector(`[data-shortcut="${event.key}"]`)?.click(); if (event.key === "Enter") ui.continueButton.click(); });

window.__G7E_B_R6__ = { app, dispatch, loadMode, loadFrame, renderQuestion, sourceToCanvas, canvasToSource, saveFinal, productionActionOrigin: "REAL_DOM_ACTIONS" };

fetch("/review.js", { cache: "no-store" }).then((response) => response.arrayBuffer()).then(sha256Hex).then((digest) => { app.productionBundleSha256 = digest; ui.saveState.textContent = "Server-backed action reviewer ready"; }).catch((error) => block(`BUNDLE_HASH_ERROR · ${error.message}`, "boot"));
