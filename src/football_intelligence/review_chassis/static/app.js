let manifest = null;
let uiConfig = null;
let state = null;
let activeIndex = 0;
let elapsedSeconds = 0;
let timerStarted = Date.now();
let activeTimeAccumulated = 0;
let activeTimeLastTick = Date.now();
let activeTimeVisible = document.visibilityState === "visible";
let activeAnnotationEditor = null;
const frameStepper = {};
let premiumMode = false;
let premiumInitialized = false;
const premiumFrames = {};
const premiumDrafts = {};
let premiumView = "focal";
let premiumPlaying = false;
let premiumSpeed = 1;
let premiumPlayTimer = null;
let premiumRenderToken = 0;
let localPremiumConfigured = false;
let stablePremiumConfigured = false;
let errorAtlasConfigured = false;
let goldMode = false;
let goldFrameIndex = 0;
let goldActiveStrand = "A";
let goldPitchVertices = [];
let goldPitchOriginalVertices = [];
let goldPitchDragIndex = null;
let goldDrawingStart = null;
const goldDrafts = {};
const goldHistory = [];
const goldViewport = {
  pitch: {zoom: 1, panX: 0, panY: 0},
  frame: {zoom: 1, panX: 0, panY: 0},
};
let goldEvidenceGeneration = 0;
let goldEvidenceBlocked = false;
let goldPointerPan = null;
let goldPolygonSidecar = null;
let goldPolygonSaveTimer = null;
let goldPolygonHistory = [];
let goldPolygonRedo = [];
let goldPolygonGateBlocked = false;
let goldSeedMode = null;
let goldSeedDrawingStart = null;
const goldPersistence = {
  db: null, ready: false, flushing: false, pending: [], serverSequence: 0,
  serverStateHash: "", clientSequence: 0, lastAck: null, blocked: false,
};

const $ = (id) => document.getElementById(id);
const isTyping = () => ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {"Content-Type": "application/json"},
    ...options,
  });
  if (!response.ok) {
    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : {message: await response.text()};
    const error = new Error(payload.message || `Request failed with HTTP ${response.status}`);
    error.httpStatus = response.status;
    error.errorCode = payload.error_code || "HTTP_REQUEST_FAILED";
    error.failedChecks = payload.failed_checks || [];
    error.savedAnnotationsUnchanged = payload.saved_annotations_unchanged === true;
    error.retryGuidance = payload.retry_guidance || "Reload server state before retrying.";
    throw error;
  }
  return response.json();
}

function goldPersistenceStatus(label) {
  const state = $("goldSaveState");
  if (state) { state.textContent = label; state.className = `saveState ${label.includes("Saved") ? "saved" : label.includes("Error") ? "error" : "unsaved"}`; }
  const status = $("goldPersistenceStatus");
  if (status) status.textContent = label;
  const pending = $("goldPendingCount");
  if (pending) pending.textContent = String(goldPersistence.pending.length);
  const sequence = $("goldServerSequence");
  if (sequence) sequence.textContent = String(goldPersistence.serverSequence);
  const hash = $("goldServerHash");
  if (hash) hash.textContent = goldPersistence.serverStateHash ? goldPersistence.serverStateHash.slice(0, 12) : "-";
  if (manifest && state) goldUpdateCompletionGate();
}

function goldPersistenceKey() { return `gold_durable_outbox_${manifest?.review_id || "review"}`; }

function goldIdbRequest(request) {
  return new Promise((resolve, reject) => { request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error); });
}

async function goldOpenOutbox() {
  if (!window.indexedDB) return null;
  const request = indexedDB.open("m5_5f1a4_gold_outbox", 1);
  request.onupgradeneeded = () => request.result.createObjectStore("events", {keyPath: "client_event_id"});
  try { return await goldIdbRequest(request); } catch { return null; }
}

async function goldOutboxRead() {
  if (goldPersistence.db) return await goldIdbRequest(goldPersistence.db.transaction("events", "readonly").objectStore("events").getAll());
  try { return JSON.parse(localStorage.getItem(goldPersistenceKey()) || "[]"); } catch { return []; }
}

async function goldOutboxPut(event) {
  if (goldPersistence.db) { await goldIdbRequest(goldPersistence.db.transaction("events", "readwrite").objectStore("events").put(event)); return; }
  localStorage.setItem(goldPersistenceKey(), JSON.stringify(goldPersistence.pending));
}

async function goldOutboxDelete(id) {
  if (goldPersistence.db) { await goldIdbRequest(goldPersistence.db.transaction("events", "readwrite").objectStore("events").delete(id)); return; }
  localStorage.setItem(goldPersistenceKey(), JSON.stringify(goldPersistence.pending));
}

async function goldInitPersistence() {
  if (goldPersistence.ready) return;
  goldPersistence.db = await goldOpenOutbox();
  goldPersistence.pending = await goldOutboxRead();
  goldPersistence.ready = true;
  goldPersistenceStatus(goldPersistence.pending.length ? "Pending locally" : "Saved to server");
  window.__goldPersistenceDiagnostics = goldPersistence;
  window.addEventListener("online", () => void goldFlushOutbox());
  void goldFlushOutbox();
}

function goldEventBase(caseData, eventType, payload, frame = null, strand = null) {
  goldPersistence.clientSequence += 1;
  const id = crypto.randomUUID();
  return {
    review_id: manifest.review_id, reviewer_session_id: uiConfig.question_contract?.reviewer_session_id || "m5_5f1a4_crash_safe_gold_annotation_reviewer",
    client_event_id: id, idempotency_key: id, client_event_sequence: goldPersistence.clientSequence,
    event_type: eventType, sequence_id: caseData?.case_id || null, frame: frame == null ? null : Number(frame), strand,
    payload, approved_polygon_hash: goldPolygonSidecar?.approved_polygon_hash || null,
    client_timestamp: new Date().toISOString(), prior_server_state_hash: goldPersistence.serverStateHash || null,
  };
}

async function goldFlushOutbox() {
  if (!goldPersistence.ready || goldPersistence.flushing || goldPersistence.blocked) return;
  goldPersistence.flushing = true;
  try {
    while (goldPersistence.pending.length) {
      const event = goldPersistence.pending[0];
      goldPersistenceStatus("Uploading");
      let ack;
      event.prior_server_state_hash = goldPersistence.serverStateHash || null;
      try { ack = await api("/api/review/gold-event", {method: "POST", body: JSON.stringify(event)}); }
      catch (error) { goldPersistenceStatus(navigator.onLine ? "Retrying" : "Offline — queued locally"); break; }
      if (ack?.state) state = ack.state;
      goldPersistence.serverSequence = Number(ack.server_event_sequence || goldPersistence.serverSequence);
      goldPersistence.serverStateHash = ack.server_state_hash || goldPersistence.serverStateHash;
      goldPersistence.lastAck = ack;
      goldPersistence.pending.shift();
      await goldOutboxDelete(event.client_event_id);
      goldPersistenceStatus("Saved to server");
    }
    if (!goldPersistence.pending.length) goldPersistenceStatus("Saved to server");
  } finally { goldPersistence.flushing = false; }
  goldRenderMetrics(goldCase());
}

function goldQueueEvent(eventType, caseData, payload, frame = null, strand = null) {
  if (!goldPersistence.ready || !caseData) return;
  const event = goldEventBase(caseData, eventType, payload, frame, strand);
  goldPersistence.pending.push(event);
  goldPersistenceStatus("Pending locally");
  void goldOutboxPut(event).then(goldFlushOutbox);
}

async function goldQueueEventAndFlush(eventType, caseData, payload, frame = null, strand = null) {
  const event = goldEventBase(caseData, eventType, payload, frame, strand);
  goldPersistence.pending.push(event);
  goldPersistenceStatus("Pending locally");
  await goldOutboxPut(event);
  await goldFlushOutbox();
  if (goldPersistence.pending.length) throw new Error("server acknowledgement is pending");
}

function goldHydrateFromServer() {
  const materialized = state?.gold_materialized?.sequences || {};
  for (const [caseId, sequence] of Object.entries(materialized)) {
    const pendingForCase = goldPersistence.pending.some((event) => event.sequence_id === caseId);
    if (pendingForCase && goldDrafts[caseId]?.dirty) continue;
    const draft = goldDefaultDraft();
    draft.seed_confirmation = sequence.seed_confirmation ? JSON.parse(JSON.stringify(sequence.seed_confirmation)) : null;
    draft.annotations = JSON.parse(JSON.stringify(sequence.frames || {}));
    draft.note = sequence.note == null ? "" : String(sequence.note);
    draft.hydrated = true;
    draft.dirty = false;
    draft.server_finalized = Boolean(sequence.finalized);
    goldDrafts[caseId] = draft;
    if (sequence.finalized) localStorage.removeItem(goldDraftKey({case_id: caseId}));
  }
  goldPersistence.serverSequence = Number(state?.server_sequence || state?.event_sequence || 0);
  goldPersistence.serverStateHash = state?.server_state_hash || "";
  goldPersistenceStatus(goldPersistence.pending.length ? "Pending locally" : "Saved to server");
}

function activeCase() {
  return manifest.cases[activeIndex];
}

function decisions() {
  return state?.decisions || {};
}

function noteFor(caseId) {
  return state?.notes?.[caseId] || "";
}

function evidenceUrl(caseId, relativePath) {
  return `/evidence/${encodeURIComponent(caseId)}/${relativePath.split("/").map(encodeURIComponent).join("/")}`;
}

async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

async function goldAuditAndLoadImage(caseData, asset, target) {
  const generation = goldEvidenceGeneration;
  if (!asset) throw new Error("The configured evidence asset is missing.");
  const url = evidenceUrl(caseData.case_id, asset.relative_path);
  const response = await fetch(url, {cache: "no-store"});
  if (!response.ok) throw new Error(`Evidence request returned HTTP ${response.status}.`);
  const contentType = response.headers.get("Content-Type") || "";
  const contentLength = Number(response.headers.get("Content-Length") || 0);
  const bytes = await response.arrayBuffer();
  if (!contentType.toLowerCase().startsWith("image/")) throw new Error("Evidence route did not return an image.");
  if (!contentLength || !bytes.byteLength) throw new Error("Evidence image is empty.");
  const actualHash = await sha256Hex(bytes);
  if (asset.sha256 && actualHash !== asset.sha256) throw new Error("Evidence hash does not match the manifest.");
  const blobUrl = URL.createObjectURL(new Blob([bytes], {type: contentType}));
  try {
    const image = new Image();
    image.decoding = "async";
    image.src = blobUrl;
    await image.decode();
    if (!image.naturalWidth || !image.naturalHeight) throw new Error("Decoded evidence has zero natural dimensions.");
    if (generation !== goldEvidenceGeneration) return null;
    target.src = blobUrl;
    target.dataset.naturalWidth = String(image.naturalWidth);
    target.dataset.naturalHeight = String(image.naturalHeight);
    target.dataset.evidenceHash = actualHash;
    target.dataset.evidenceReady = "true";
    target.onload = () => URL.revokeObjectURL(blobUrl);
    return image;
  } catch (error) {
    URL.revokeObjectURL(blobUrl);
    throw error;
  }
}

function goldSetEvidenceBlocker(message = "Evidence unavailable. Annotation and completion are disabled.") {
  goldEvidenceBlocked = true;
  [$("goldEvidenceBlocker"), $("goldEvidenceBlockerFrame")].forEach((blocker) => {
    if (blocker) { blocker.textContent = message; blocker.classList.remove("isHidden"); }
  });
  const status = $("goldEvidenceStatus");
  if (status) status.textContent = `Evidence unavailable: ${message}`;
  document.querySelectorAll("#goldShell button, #goldShell input, #goldShell textarea").forEach((control) => {
    if (!control.matches("#goldComplete, #goldUndo")) control.disabled = true;
  });
  goldUpdateCompletionGate();
}

function goldClearEvidenceBlocker() {
  goldEvidenceBlocked = false;
  [$("goldEvidenceBlocker"), $("goldEvidenceBlockerFrame")].forEach((blocker) => blocker?.classList.add("isHidden"));
  const status = $("goldEvidenceStatus");
  if (status) status.textContent = "Evidence verified: image route, decode, dimensions and hash passed.";
  document.querySelectorAll("#goldShell button, #goldShell input, #goldShell textarea").forEach((control) => {
    if (!control.matches("#goldComplete")) control.disabled = false;
  });
}

function goldSetPolygonGate(message) {
  goldPolygonGateBlocked = true;
  const blocker = $("goldEvidenceBlockerFrame");
  if (blocker) { blocker.textContent = message; blocker.classList.remove("isHidden"); }
  document.querySelectorAll("#goldAnnotationPanel button, #goldAnnotationTools button, #goldAnnotationTools textarea").forEach((control) => { control.disabled = true; });
  const status = $("goldEvidenceStatus");
  if (status) status.textContent = message;
  goldUpdateCompletionGate();
}

function goldClearPolygonGate() {
  goldPolygonGateBlocked = false;
  const blocker = $("goldEvidenceBlockerFrame");
  if (blocker && !goldEvidenceBlocked) blocker.classList.add("isHidden");
  document.querySelectorAll("#goldAnnotationPanel button, #goldAnnotationTools button, #goldAnnotationTools textarea").forEach((control) => { control.disabled = false; });
  goldUpdateCompletionGate();
}

function assetSort(a, b) {
  const order = uiConfig.asset_panel_order || [];
  const index = (asset) => {
    const found = order.findIndex((item) => item.asset_type === asset.asset_type && (!item.group_id || item.group_id === asset.group_id));
    return found >= 0 ? found : 999;
  };
  return index(a) - index(b) || a.label.localeCompare(b.label);
}

function groupedSequenceAssets(caseData) {
  const groups = {};
  for (const asset of caseData.evidence_assets.filter((item) => item.asset_type === "image_sequence" && !item.metadata?.annotation_base && assetVisible(caseData, item))) {
    const key = asset.group_id || "default";
    groups[key] = groups[key] || [];
    groups[key].push(asset);
  }
  for (const assets of Object.values(groups)) {
    assets.sort((a, b) => (a.frame_sequences[0] || 0) - (b.frame_sequences[0] || 0));
  }
  return groups;
}

function annotationFrameAssets(caseData) {
  return caseData.evidence_assets
    .filter((item) => item.asset_type === "image_sequence" && item.metadata?.annotation_base === true && assetVisible(caseData, item))
    .sort((a, b) => (a.frame_sequences[0] || 0) - (b.frame_sequences[0] || 0));
}

function currentAnnotationAsset(caseData) {
  const assets = annotationFrameAssets(caseData);
  if (!assets.length) return null;
  const key = `${caseData.case_id}:annotation_frames`;
  const defaultIndex = Number(caseData.visible_metadata?.target_frame_index ?? 0);
  frameStepper[key] = Math.max(0, Math.min(frameStepper[key] ?? defaultIndex, assets.length - 1));
  return assets[frameStepper[key]];
}

function currentAnnotationCandidates(caseData, asset) {
  const byFrame = caseData.visible_metadata?.safe_anonymous_candidates_by_frame || {};
  return byFrame[String(asset?.frame_sequences?.[0])] || caseData.visible_metadata?.safe_anonymous_candidates || [];
}

function currentDecision(caseData) {
  return decisions()[caseData.case_id];
}

function activeTimeNow() {
  const now = Date.now();
  if (activeTimeVisible) activeTimeAccumulated += Math.max(0, now - activeTimeLastTick) / 1000;
  activeTimeLastTick = now;
  return Math.max(0, Math.ceil(activeTimeAccumulated));
}

function revealMap(caseData) {
  return state?.reveal_state?.[caseData.case_id] || {};
}

function revealKey(asset) {
  return asset.reveal_group_id || asset.asset_id;
}

function assetVisible(caseData, asset) {
  const policy = asset.visibility_policy || "always_visible";
  const decision = currentDecision(caseData);
  const revealed = revealMap(caseData)[revealKey(asset)] === true;
  if (state?.completed === true && asset.visible_after_completion === true) return true;
  if (policy === "always_visible") return true;
  if (policy === "hidden_always_reviewer") return false;
  if (policy === "completion_only") return state?.completed === true;
  if (policy === "hidden_until_decision") {
    if (!decision) return false;
    const values = asset.visible_after_decision_values || [];
    return values.length === 0 || values.includes(decision);
  }
  if (policy === "hidden_until_explicit_reveal") return revealed;
  return false;
}

function revealControl(caseData, asset) {
  const policy = asset.visibility_policy || "always_visible";
  if (!["hidden_until_decision", "hidden_until_explicit_reveal"].includes(policy)) return "";
  if (assetVisible(caseData, asset)) return "";
  const decision = currentDecision(caseData);
  const requiresDecision = asset.reveal_requires_existing_decision || policy === "hidden_until_decision";
  const disabled = requiresDecision && !decision ? " disabled" : "";
  const label = asset.reveal_button_label || (requiresDecision ? "Available after decision" : "Reveal hidden evidence");
  return `<article class="assetCard hiddenAssetNotice">
    <h3>Hidden evidence</h3>
    <button type="button" data-reveal-asset="${asset.asset_id}" data-reveal-group="${asset.reveal_group_id || ""}"${disabled}>
      ${label}
    </button>
  </article>`;
}

function renderImageStepper(caseData, assets, groupId) {
  if (!assets.length) return "";
  const caseKey = `${caseData.case_id}:${groupId}`;
  frameStepper[caseKey] = Math.max(0, Math.min(frameStepper[caseKey] || 0, assets.length - 1));
  const current = assets[frameStepper[caseKey]];
  const frameLabel = current.frame_sequences.length ? `Frame ${current.frame_sequences.join(", ")}` : current.label;
  return `
    <div class="stepper" data-stepper="${caseKey}">
      <div class="stepperControls">
        <button type="button" data-stepper-prev="${caseKey}">Previous frame</button>
        <span>${frameLabel}</span>
        <button type="button" data-stepper-next="${caseKey}">Next frame</button>
      </div>
      <img class="evidenceImage" src="${evidenceUrl(caseData.case_id, current.relative_path)}" alt="${current.label}">
    </div>`;
}

function renderAsset(caseData, asset) {
  if (!assetVisible(caseData, asset)) return revealControl(caseData, asset);
  const url = evidenceUrl(caseData.case_id, asset.relative_path);
  if (asset.asset_type === "animated_gif") {
    return `
      <article class="assetCard">
        <h3>${asset.label}</h3>
        <div class="gifControls">
          <button type="button" data-gif-restart="${asset.asset_id}">Restart GIF</button>
        </div>
        <img class="evidenceImage temporalGif" id="gif_${asset.asset_id}" src="${url}" alt="${asset.label}">
      </article>`;
  }
  if (asset.asset_type === "metadata_json") {
    return `
      <article class="assetCard">
        <h3>${asset.label}</h3>
        <a href="${url}" target="_blank" rel="noreferrer">Open metadata</a>
      </article>`;
  }
  if (asset.asset_type === "image_sequence") return "";
  return `
    <article class="assetCard">
      <h3>${asset.label}</h3>
      <img class="evidenceImage" src="${url}" alt="${asset.label}">
    </article>`;
}

function renderAssets(caseData) {
  const sequenceGroups = groupedSequenceAssets(caseData);
  const sequenceCards = Object.entries(sequenceGroups).map(([groupId, assets]) => `
    <article class="assetCard">
      <h3>${assets[0]?.label || "Frame sequence"}</h3>
      ${renderImageStepper(caseData, assets, groupId)}
    </article>`);
  const renderedGroupIds = new Set(Object.keys(sequenceGroups));
  const panels = uiConfig.layout === "multi_candidate_comparison" ? (uiConfig.comparison_panels || []) : [];
  const panelGroupIds = new Set(panels.map((panel) => panel.asset_group_id));
  const comparisonPanels = panels.map((panel) => {
    const assets = caseData.evidence_assets
      .filter((asset) => asset.group_id === panel.asset_group_id && asset.asset_type !== "image_sequence")
      .sort(assetSort)
      .map((asset) => renderAsset(caseData, asset))
      .join("");
    return `<article class="comparisonPanel">
      <h3>${panel.label || panel.asset_group_id}</h3>
      ${assets}
    </article>`;
  });
  const normalAssets = caseData.evidence_assets
    .filter((asset) => asset.asset_type !== "image_sequence")
    .filter((asset) => !panelGroupIds.has(asset.group_id))
    .filter((asset) => !(uiConfig.spatial_annotation_enabled && (asset.metadata?.primary_annotation_image === true || asset.metadata?.annotation_base === true)))
    .sort(assetSort)
    .map((asset) => renderAsset(caseData, asset));
  const hiddenSequenceControls = caseData.evidence_assets
    .filter((asset) => asset.asset_type === "image_sequence" && !asset.metadata?.annotation_base && !assetVisible(caseData, asset))
    .filter((asset) => !renderedGroupIds.has(asset.group_id || "default"))
    .map((asset) => revealControl(caseData, asset));
  const comparisonMarkup = comparisonPanels.length
    ? [`<section class="comparisonGrid">${comparisonPanels.join("")}</section>`]
    : [];
  $("assetPanels").innerHTML = [...comparisonMarkup, ...normalAssets, ...sequenceCards, ...hiddenSequenceControls].join("");
}

function renderMetadata(caseData) {
  const fields = uiConfig.visible_metadata_fields || [];
  const rows = fields.map((field) => {
    const value = caseData.visible_metadata?.[field] ?? caseData[field] ?? "";
    return `<tr><th>${field}</th><td>${String(value)}</td></tr>`;
  });
  $("metadataPanel").innerHTML = rows.length ? `<table>${rows.join("")}</table>` : "";
  const canShowHidden = currentDecision(caseData) && revealMap(caseData).__case_metadata__ === true;
  const serverRevealPayloads = state?.server_reveal_payloads?.[caseData.case_id] || {};
  const serverReveal = serverRevealPayloads.__case_metadata__;
  const legacyReveal = {
    hidden_metadata: caseData.hidden_metadata || {},
    reveal_metadata: caseData.reveal_metadata || {},
  };
  const revealPayload = serverReveal || legacyReveal;
  $("hiddenMetadata").textContent = canShowHidden ? JSON.stringify({
    server_reveal_payload: revealPayload,
  }, null, 2) : "Hidden until a decision is saved or reveal is recorded.";
}

function parseAnnotationNote() {
  try {
    const parsed = JSON.parse($("note").value || "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function annotationValue(key) {
  const parsed = parseAnnotationNote();
  return parsed?.spatial_annotation?.[key] ?? "";
}

function intervalAnnotationValues() {
  const parsed = parseAnnotationNote();
  return parsed?.spatial_annotation || {};
}

function renderIntervalControls(caseData) {
  const root = $("intervalAnnotationControls");
  if (!root) return;
  const values = intervalAnnotationValues();
  const frames = caseData.visible_metadata?.frame_sequences || [];
  const asset = currentAnnotationAsset(caseData);
  const candidates = currentAnnotationCandidates(caseData, asset);
  const optionList = (items, selected, emptyLabel) => [
    `<option value="">${emptyLabel}</option>`,
    ...items.map((item) => `<option value="${item}"${String(selected) === String(item) ? " selected" : ""}>${item}</option>`),
  ].join("");
  root.innerHTML = `
    <h4>Interval controls</h4>
    <div class="annotationGrid intervalControlGrid">
      <label>Deficit start frame<select data-interval-field="deficit_start_frame">${optionList(frames, values.deficit_start_frame, "Unselected")}</select></label>
      <label>Deficit end frame<select data-interval-field="deficit_end_frame">${optionList(frames, values.deficit_end_frame, "Unselected")}</select></label>
      <label>Merged detection<select data-interval-field="merged_detection_number">${optionList(candidates.map((item) => item.anonymous_candidate_number), values.merged_detection_number, "None / unresolved")}</select></label>
      <label>Re-entry path<select data-interval-field="reentry_path_selection">${optionList(["PATH_A", "PATH_B", "PATH_C", "NO_REENTRY", "UNRESOLVED"], values.reentry_path_selection || "UNRESOLVED", "Choose a path")}</select></label>
      <label>Partial / occluded<select data-interval-field="partial_or_occluded">${optionList(["false", "true"], String(values.partial_or_occluded), "Unspecified")}</select></label>
    </div>
    <button type="button" data-interval-apply="true">Save interval annotation</button>
    <p class="annotationHint">All coordinates remain in original-image pixels. Decisions remain visual review labels only.</p>`;
}

function applyIntervalAnnotation() {
  const caseData = activeCase();
  const current = intervalAnnotationValues();
  const annotation = {
    ...current,
    schema_version: "football_intelligence.review_chassis.occlusion_interval_annotation.v1",
    case_id: caseData.case_id,
    coordinate_space: "original_image_pixels",
  };
  document.querySelectorAll("[data-interval-field]").forEach((input) => {
    const key = input.dataset.intervalField;
    if (!key || input.value === "") return;
    annotation[key] = key === "merged_detection_number" || key.endsWith("_frame")
      ? Number(input.value)
      : key === "partial_or_occluded" ? input.value === "true" : input.value;
  });
  $("note").value = JSON.stringify({spatial_annotation: annotation}, null, 2);
  saveNote();
}

function renderSpatialAnnotation(caseData) {
  const panel = $("annotationPanel");
  activeAnnotationEditor = null;
  if (!uiConfig.spatial_annotation_enabled) {
    panel.classList.add("hidden");
    panel.innerHTML = "";
    return;
  }
  const schema = uiConfig.spatial_annotation_schema || {};
  const interactive = schema.interactive_canvas_enabled === true
    || String(uiConfig.spatial_annotation_mode || "").includes("interactive");
  if (interactive && window.ReviewAnnotationCanvas) {
    const asset = currentAnnotationAsset(caseData)
      || caseData.evidence_assets.find((item) => item.metadata?.primary_annotation_image === true)
      || caseData.evidence_assets.find((item) => item.asset_id === "target_full_resolution")
      || caseData.evidence_assets.find((item) => item.metadata?.full_resolution === true);
    if (!asset) {
      panel.classList.remove("hidden");
      panel.innerHTML = "<h3>Spatial annotation</h3><p>Full-resolution annotation image is unavailable.</p>";
      return;
    }
    panel.classList.remove("hidden");
    panel.innerHTML = `
      <h3>${schema.title || "Interactive spatial annotation"}</h3>
      <div id="interactiveAnnotationRoot"></div>
      <div id="intervalAnnotationControls"></div>`;
    const frameAssets = annotationFrameAssets(caseData);
    activeAnnotationEditor = new window.ReviewAnnotationCanvas.SpatialAnnotationCanvas(
      $("interactiveAnnotationRoot"),
      {
        caseData,
        asset,
        imageUrl: evidenceUrl(caseData.case_id, asset.relative_path),
        candidates: currentAnnotationCandidates(caseData, asset),
        layerRows: caseData.visible_metadata?.geometry_layers || [],
        layerVisibility: caseData.visible_metadata?.layer_visibility || {},
        frameAssets,
        selectedFrameIndex: frameStepper[`${caseData.case_id}:annotation_frames`] || 0,
        noteElement: $("note"),
        onChange: () => {},
      },
    );
    if (uiConfig.spatial_annotation_mode === "occlusion_interval") renderIntervalControls(caseData);
    return;
  }
  const sizeCategories = schema.bbox_size_categories || ["small", "medium", "large", "uncertain"];
  const confidenceValues = schema.confidence_values || ["high", "medium", "low", "uncertain"];
  panel.classList.remove("hidden");
  panel.innerHTML = `
    <h3>${schema.title || "Spatial annotation"}</h3>
    <div class="annotationGrid">
      <label>bbox x1<input data-annotation-field="bbox_x1" type="number" step="0.1" value="${annotationValue("bbox_x1")}"></label>
      <label>bbox y1<input data-annotation-field="bbox_y1" type="number" step="0.1" value="${annotationValue("bbox_y1")}"></label>
      <label>bbox x2<input data-annotation-field="bbox_x2" type="number" step="0.1" value="${annotationValue("bbox_x2")}"></label>
      <label>bbox y2<input data-annotation-field="bbox_y2" type="number" step="0.1" value="${annotationValue("bbox_y2")}"></label>
      <label>footpoint x<input data-annotation-field="footpoint_x" type="number" step="0.1" value="${annotationValue("footpoint_x")}"></label>
      <label>footpoint y<input data-annotation-field="footpoint_y" type="number" step="0.1" value="${annotationValue("footpoint_y")}"></label>
      <label>existing candidate #<input data-annotation-field="existing_candidate_number" type="number" step="1" min="1" value="${annotationValue("existing_candidate_number")}"></label>
      <label>confidence<select data-annotation-field="confidence">
        ${confidenceValues.map((value) => `<option value="${value}"${annotationValue("confidence") === value ? " selected" : ""}>${value}</option>`).join("")}
      </select></label>
      <label>bbox size<select data-annotation-field="bbox_size_category">
        ${sizeCategories.map((value) => `<option value="${value}"${annotationValue("bbox_size_category") === value ? " selected" : ""}>${value}</option>`).join("")}
      </select></label>
      <label>partial or occluded<select data-annotation-field="partial_or_occluded">
        ${["", "false", "true"].map((value) => `<option value="${value}"${String(annotationValue("partial_or_occluded")) === value ? " selected" : ""}>${value || "unspecified"}</option>`).join("")}
      </select></label>
    </div>
    <div class="annotationActions">
      <button type="button" data-annotation-apply="true">Write annotation JSON to notes</button>
    </div>`;
}

function applySpatialAnnotation() {
  const caseData = activeCase();
  const annotation = {
    schema_version: "football_intelligence.review_chassis.spatial_annotation.v1",
    case_id: caseData.case_id,
    mode: uiConfig.spatial_annotation_mode || "spatial_annotation",
  };
  document.querySelectorAll("[data-annotation-field]").forEach((input) => {
    const key = input.dataset.annotationField;
    if (!key) return;
    const value = input.value;
    if (value !== "") annotation[key] = value;
  });
  $("note").value = JSON.stringify({spatial_annotation: annotation}, null, 2);
  saveNote();
}

function renderDecisions(caseData) {
  const current = decisions()[caseData.case_id];
  const allowed = new Set(caseData.allowed_decisions || []);
  if (uiConfig.question_contract?.seed_confirmation_required === true && caseData.task_type === "gold_strand_frame_annotation") {
    allowed.add("SEQUENCE_REJECTED");
  }
  $("decisionButtons").innerHTML = uiConfig.decisions.filter((option) => allowed.has(option.value)).map((option) => {
    const selected = current === option.value ? " selected" : "";
    return `<button type="button" class="decision ${option.style}${selected}" data-decision="${option.value}">
      <strong>${option.key}</strong> ${option.label}
    </button>`;
  }).join("");
}

function renderCaseList() {
  $("caseList").innerHTML = manifest.cases.map((caseData, index) => {
    const done = decisions()[caseData.case_id] ? "done" : "";
    const active = index === activeIndex ? "active" : "";
    return `<button type="button" class="caseButton ${done} ${active}" data-case-index="${index}">
      ${index + 1}. ${caseData.case_id}
    </button>`;
  }).join("");
}

function render() {
  if (!manifest || !uiConfig || !state) return;
  const caseData = activeCase();
  document.title = uiConfig.page_title;
  $("reviewTitle").textContent = uiConfig.review_title;
  $("warning").textContent = uiConfig.visual_warning;
  $("instructions").textContent = uiConfig.task_instructions;
  $("caseTitle").textContent = `${activeIndex + 1} / ${manifest.cases.length}: ${caseData.case_id}`;
  $("question").textContent = caseData.concise_question;
  $("counts").textContent = `${state.counts.reviewed} reviewed, ${state.counts.remaining} remaining`;
  $("notesPanel").classList.toggle("hidden", !uiConfig.notes_enabled);
  $("undoBtn").classList.toggle("hidden", !uiConfig.undo_enabled);
  $("note").value = noteFor(caseData.case_id);
  renderCaseList();
  renderDecisions(caseData);
  renderAssets(caseData);
  renderSpatialAnnotation(caseData);
  renderMetadata(caseData);
}

function setStatus(text, failed = false) {
  $("status").textContent = text;
  $("status").classList.toggle("failed", failed);
}

async function saveDecision(decision, inputSource = "click") {
  const caseData = activeCase();
  if (uiConfig.spatial_annotation_enabled && window.ReviewAnnotationCanvas) {
    const errors = window.ReviewAnnotationCanvas.validateDecision(decision, $("note").value, caseData);
    if (errors.length) {
      setStatus(`Annotation required: ${errors.join(" ")}`, true);
      return;
    }
  }
  const body = {
    case_id: caseData.case_id,
    decision,
    note: $("note").value,
    input_source: inputSource,
    reveal_state: {[caseData.case_id]: $("revealPanel").open},
    last_viewed_case_id: caseData.case_id,
    elapsed_active_seconds: Math.max(elapsedSeconds + Math.floor((Date.now() - timerStarted) / 1000), activeTimeNow()),
  };
  state = await api("/api/review/decision", {method: "POST", body: JSON.stringify(body)});
  setStatus("Saved");
  if (uiConfig.decisions_advance_automatically) {
    activeIndex = Math.min(manifest.cases.length - 1, activeIndex + 1);
  }
  render();
}

async function saveNote() {
  if (!uiConfig.notes_enabled) return;
  const caseData = activeCase();
  state = await api("/api/review/note", {
    method: "POST",
    body: JSON.stringify({case_id: caseData.case_id, note: $("note").value}),
  });
  setStatus("Note saved");
}

async function reveal(assetId, revealGroupId) {
  const caseData = activeCase();
  state = await api("/api/review/reveal", {
    method: "POST",
    body: JSON.stringify({
      case_id: caseData.case_id,
      asset_id: assetId || null,
      reveal_group_id: revealGroupId || assetId || "__case_metadata__",
      input_source: "click",
    }),
  });
  setStatus("Reveal recorded");
  render();
}

async function undo() {
  state = await api("/api/review/undo", {method: "POST", body: "{}"});
  setStatus("Undo saved");
  render();
}

async function completeReview() {
  try {
    state = await api("/api/review/complete", {method: "POST", body: JSON.stringify({elapsed_active_seconds: Math.max(elapsedSeconds, activeTimeNow())})});
    setStatus("Completed");
    render();
  } catch (error) {
    setStatus(`Completion blocked: ${error.message}`, true);
  }
}

async function exportReview() {
  $("exportBox").textContent = JSON.stringify(await api("/api/review/export"), null, 2);
  $("exportBox").classList.remove("hidden");
}

function go(delta) {
  activeIndex = Math.max(0, Math.min(manifest.cases.length - 1, activeIndex + delta));
  render();
}

function decisionForKey(key) {
  const lowered = key.toLowerCase();
  return uiConfig.decisions.find((option) => option.key.toLowerCase() === lowered)?.value;
}

document.addEventListener("keydown", (event) => {
  if (premiumMode || goldMode) return;
  if (isTyping()) {
    if (event.key === "Escape") document.activeElement.blur();
    return;
  }
  if (event.ctrlKey && event.key.toLowerCase() === "z" && uiConfig.undo_enabled) {
    event.preventDefault();
    undo();
    return;
  }
  const decision = decisionForKey(event.key);
  if (decision && activeCase().allowed_decisions.includes(decision)) {
    event.preventDefault();
    saveDecision(decision, "keyboard");
    return;
  }
  if (event.key === "ArrowLeft") go(-1);
  if (event.key === "ArrowRight") go(1);
  if (event.key.toLowerCase() === "n" && uiConfig.notes_enabled) $("note").focus();
});

document.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const decision = target.closest("[data-decision]")?.dataset.decision;
  if (decision) saveDecision(decision, "click");
  const caseIndex = target.closest("[data-case-index]")?.dataset.caseIndex;
  if (caseIndex !== undefined) {
    activeIndex = Number(caseIndex);
    render();
  }
  const prev = target.closest("[data-stepper-prev]")?.dataset.stepperPrev;
  if (prev) {
    frameStepper[prev] = Math.max(0, (frameStepper[prev] || 0) - 1);
    render();
  }
  const next = target.closest("[data-stepper-next]")?.dataset.stepperNext;
  if (next) {
    const [caseId, groupId] = next.split(":");
    const caseData = manifest.cases.find((item) => item.case_id === caseId);
    const count = groupedSequenceAssets(caseData)[groupId]?.length || 1;
    frameStepper[next] = Math.min(count - 1, (frameStepper[next] || 0) + 1);
    render();
  }
  const gif = target.closest("[data-gif-restart]")?.dataset.gifRestart;
  if (gif) {
    const image = $(`gif_${gif}`);
    const base = image.src.split("?")[0];
    image.src = `${base}?restart=${Date.now()}`;
  }
  const revealAsset = target.closest("[data-reveal-asset]")?.dataset.revealAsset;
  if (revealAsset) {
    const group = target.closest("[data-reveal-asset]")?.dataset.revealGroup;
    reveal(revealAsset, group);
  }
  if (target.closest("[data-annotation-apply]")) applySpatialAnnotation();
  if (target.closest("[data-interval-apply]")) applyIntervalAnnotation();
  const canvasPrev = target.closest("[data-canvas-prev]")?.dataset.canvasPrev;
  if (canvasPrev) {
    frameStepper[canvasPrev] = Math.max(0, (frameStepper[canvasPrev] || 0) - 1);
    render();
  }
  const canvasNext = target.closest("[data-canvas-next]")?.dataset.canvasNext;
  if (canvasNext) {
    const caseData = manifest.cases.find((item) => item.case_id === activeCase().case_id);
    const count = annotationFrameAssets(caseData).length || 1;
    frameStepper[canvasNext] = Math.min(count - 1, (frameStepper[canvasNext] || 0) + 1);
    render();
  }
});

let noteTimer = null;
$("note").addEventListener("input", () => {
  clearTimeout(noteTimer);
  noteTimer = setTimeout(saveNote, 400);
});
$("prevBtn").onclick = () => go(-1);
$("nextBtn").onclick = () => go(1);
$("undoBtn").onclick = undo;
$("completeBtn").onclick = completeReview;
$("exportBtn").onclick = exportReview;
$("revealPanel").addEventListener("toggle", () => {
  const caseData = activeCase();
  if ($("revealPanel").open && caseData && currentDecision(caseData)) reveal(null, "__case_metadata__");
});

setInterval(() => {
  elapsedSeconds += Math.floor((Date.now() - timerStarted) / 1000);
  timerStarted = Date.now();
}, 1000);

document.addEventListener("visibilitychange", () => {
  activeTimeNow();
  activeTimeVisible = document.visibilityState === "visible";
  activeTimeLastTick = Date.now();
});

function premiumCase() {
  return manifest.cases[activeIndex];
}

function premiumDraftKey(caseData) {
  const prefix = ({stable_local_strand_continuity: "m5_5f0_draft", development_error_atlas: "m5_5f1c_atlas_draft"})[uiConfig?.presentation_mode] || "m5_5e2_draft";
  return `${prefix}_${manifest.review_id}_${caseData.case_id}`;
}

function premiumDefaultDraft() {
  return {
    answers: {},
    conclusion: "",
    subtype: "",
    overrideReason: "",
    confirmed: false,
    note: "",
    annotation: {},
    seed_action: "",
    first_failure_frame: "",
    seed_correction: "",
    seed_rejection_reason: "",
  };
}

function premiumGetDraft(caseData) {
  if (premiumDrafts[caseData.case_id]) return premiumDrafts[caseData.case_id];
  const saved = localStorage.getItem(premiumDraftKey(caseData));
  try {
    premiumDrafts[caseData.case_id] = {...premiumDefaultDraft(), ...(saved ? JSON.parse(saved) : {})};
  } catch {
    premiumDrafts[caseData.case_id] = premiumDefaultDraft();
  }
  return premiumDrafts[caseData.case_id];
}

function premiumSaveDraft(caseData) {
  localStorage.setItem(premiumDraftKey(caseData), JSON.stringify(premiumGetDraft(caseData)));
  $("premiumSaveState").textContent = "Unsaved";
  $("premiumSaveState").className = "saveState unsaved";
}

function premiumEvidenceAssets(caseData) {
  return Object.fromEntries(caseData.evidence_assets.map((item) => [item.asset_id, item]));
}

function premiumRecord(caseData) {
  const records = caseData.visible_metadata?.frame_records || [];
  return records[Math.max(0, Math.min(premiumFrames[caseData.case_id] || 0, records.length - 1))];
}

function premiumFrameIndex(caseData) {
  const records = caseData.visible_metadata?.frame_records || [];
  premiumFrames[caseData.case_id] = Math.max(0, Math.min(premiumFrames[caseData.case_id] || 0, Math.max(0, records.length - 1)));
  return premiumFrames[caseData.case_id];
}

function premiumAssetUrl(caseData, assetId) {
  const asset = premiumEvidenceAssets(caseData)[assetId];
  return asset ? evidenceUrl(caseData.case_id, asset.relative_path) : "";
}

function premiumSetStatus(text, kind = "") {
  const target = $("premiumSaveState");
  target.textContent = text;
  target.className = `saveState ${kind}`.trim();
}

function premiumAnswerValues(caseData) {
  const draft = premiumGetDraft(caseData);
  return draft.answers || {};
}

function premiumCollectAnswers(caseData) {
  const answers = {};
  document.querySelectorAll("#premiumReviewForm input[type=radio]:checked").forEach((input) => {
    answers[input.name] = input.value;
  });
  premiumGetDraft(caseData).answers = answers;
  return answers;
}

function premiumSuggestion(answers) {
  if (uiConfig?.presentation_mode === "development_error_atlas") return null;
  if (uiConfig?.presentation_mode === "stable_local_strand_continuity") {
    return answers.continuity_outcome ? {code: answers.continuity_outcome, reason: "The selected structured continuity outcome is ready to save."} : null;
  }
  if (["local_encounter_strands", "stable_local_strand_continuity"].includes(uiConfig?.presentation_mode)) {
    if (answers.incoming_people_supported === "no") return {code: "I", reason: "The local evidence does not support two independently observed incoming strands."};
    if (answers.incoming_people_supported === "unclear") return {code: "U", reason: "The incoming local evidence is unresolved."};
    if (answers.during_state === "both_remain_independently_visible") return {code: "O", reason: "Both local strands remain independently observed through the interval."};
    if (answers.during_state === "detector_duplicate_or_false_positive_artifact") return {code: "X", reason: "The local evidence is more consistent with a detector duplicate or false-positive artifact."};
    if (answers.during_state === "strand_evidence_inconsistent") return {code: "S", reason: "The local strand evidence is inconsistent or switches, so no strand conclusion is supported."};
    if (answers.outgoing_people_supported === "no") return {code: "P", reason: "The local evidence does not support two independently observed outgoing strands."};
    if (answers.outgoing_people_supported === "unclear" || answers.path_continuity_plausible === "unclear") return {code: "U", reason: "The outgoing or continuity evidence remains unresolved."};
    if (answers.path_continuity_plausible === "no") return {code: "S", reason: "The proposed local A/B continuation is not visually plausible."};
    const localSubtype = {
      one_person_becomes_missing: "observed_missing_observed",
      one_shared_or_merged_observation: "shared_or_merged_observation",
      partial_body_or_fragment_only: "partial_or_fragment_observation",
      other_two_to_one_collapse: "two_to_one_collapse",
    }[answers.during_state];
    if (answers.incoming_people_supported === "yes" && answers.outgoing_people_supported === "yes" && localSubtype) return {code: "G", subtype: localSubtype, reason: "The before, interval, after and local A/B continuity evidence support a bounded observation-deficit review case."};
    return {code: "U", reason: "The local strand answers do not support a more specific conclusion."};
  }
  if (answers.incoming_people_supported === "no") return {code: "I", reason: "The before evidence does not support two independently visible people."};
  if (answers.incoming_people_supported === "unclear") return {code: "U", reason: "The before evidence is not decisive enough for a supported precondition."};
  if (answers.during_state === "both_remain_independently_visible") return {code: "O", reason: "Both people remain independently visible during the interval."};
  if (answers.during_state === "detector_duplicate_or_false_positive_artifact") return {code: "X", reason: "The during evidence is more consistent with a detector or duplicate artifact."};
  if (answers.outgoing_people_supported === "no") return {code: "P", reason: "The after evidence does not support two independently visible people again."};
  if (answers.outgoing_people_supported === "unclear" || answers.path_continuity_plausible === "unclear") return {code: "U", reason: "One or more required after or continuity answers remains unclear."};
  if (answers.path_continuity_plausible === "no") return {code: "U", reason: "The incoming and outgoing paths are not visually plausible as a continuation."};
  const subtype = {
    one_person_becomes_missing: "observed_missing_observed",
    one_shared_or_merged_observation: "shared_or_merged_observation",
    partial_body_or_fragment_only: "partial_or_fragment_observation",
    other_two_to_one_collapse: "two_to_one_collapse",
  }[answers.during_state];
  if (answers.incoming_people_supported === "yes" && answers.outgoing_people_supported === "yes" && subtype) return {code: "G", subtype, reason: "The before, during, after and continuity answers support a genuine observation-deficit interval."};
  return {code: "U", reason: "The four answers do not support a more specific conclusion."};
}

function premiumConclusionLabel(code) {
  return ({G: "G - Genuine observation-deficit interval", O: "O - Ordinary crossing; observations remain independent", X: "X - Detector or duplicate artifact", I: "I - Insufficient incoming evidence", P: "P - Insufficient outgoing evidence", S: "S - Strand evidence inconsistent", U: "U - Unresolved"})[code] || "Select after answering";
}

function premiumCanonicalLabel(code, subtype) {
  if (code === "G") return ({two_to_one_collapse: "GENUINE_TWO_TO_ONE_COLLAPSE", observed_missing_observed: "GENUINE_OBSERVED_MISSING_OBSERVED", shared_or_merged_observation: "GENUINE_MERGED_OBSERVATION_INTERVAL", partial_or_fragment_observation: "PARTIAL_FRAGMENT_OBSERVATION_DEFICIT"})[subtype] || "";
  return ({O: "ORDINARY_CROSSING_INDEPENDENT_OBSERVATIONS_REMAIN", X: "DETECTOR_DUPLICATE_OR_FALSE_POSITIVE_ARTIFACT", I: "INSUFFICIENT_INCOMING_PRECONDITION", P: "INSUFFICIENT_OUTGOING_POSTCONDITION", S: "STRAND_EVIDENCE_INCONSISTENT", U: "EVIDENCE_UNRESOLVED"})[code] || "";
}

function premiumConfigureLocalContract() {
  if (localPremiumConfigured || uiConfig?.presentation_mode !== "local_encounter_strands") return;
  localPremiumConfigured = true;
  $("premiumReviewTitle").textContent = uiConfig.review_title || "Local encounter strand review";
  const task = document.querySelector(".taskCard p");
  if (task) task.textContent = "Review only the local encounter. Strand A is cyan, Strand B is magenta, and a shared gold box means one observation supports both. Missing means no observed box. Predictions are off by default.";
  const labels = {
    incoming_people_supported: "Before the interval, are two local people independently visible as Strand A and Strand B?",
    during_state: "What happens to the local A/B encounter during the interval?",
    outgoing_people_supported: "After the interval, are two local people independently visible again?",
    path_continuity_plausible: "Is the local A/B continuation visually plausible without a strand switch?",
  };
  document.querySelectorAll(".evidenceQuestion").forEach((fieldset) => {
    const key = fieldset.dataset.question;
    const legend = fieldset.querySelector("legend");
    if (legend && labels[key]) legend.innerHTML = `<span class="questionNumber">${legend.querySelector(".questionNumber")?.textContent || ""}</span> ${labels[key]}`;
  });
  const during = document.querySelector('[data-question="during_state"] .radioStack');
  if (during) during.innerHTML = [
    ["both_remain_independently_visible", "Both remain independently observed"],
    ["one_person_becomes_missing", "One strand becomes missing (no observed box)"],
    ["one_shared_or_merged_observation", "One shared gold observation supports both strands"],
    ["partial_body_or_fragment_only", "One strand is partial or fragmentary"],
    ["other_two_to_one_collapse", "Two-to-one collapse into one local observation"],
    ["detector_duplicate_or_false_positive_artifact", "Detector duplicate or false-positive artifact"],
    ["strand_evidence_inconsistent", "Strand evidence is inconsistent or switches"],
    ["unclear", "Unclear"],
  ].map(([value, label]) => `<label><input type="radio" name="during_state" value="${value}">${label}</label>`).join("");
  const conclusion = $("premiumConclusion");
  if (conclusion && !conclusion.querySelector('option[value="S"]')) conclusion.insertAdjacentHTML("beforeend", '<option value="S">S - Strand evidence inconsistent</option>');
}

function stableOutcomeLabel(value) {
  return ({PASS: "PASS - Stable local continuation", A_SWITCH: "A_SWITCH - Strand A switches", B_SWITCH: "B_SWITCH - Strand B switches", BOTH_SWITCH: "BOTH_SWITCH - Both strands switch", A_LOST: "A_LOST - Strand A is lost", B_LOST: "B_LOST - Strand B is lost", BOTH_LOST: "BOTH_LOST - Both strands are lost", DETECTION_SUPPLY_FAILURE: "DETECTION_SUPPLY_FAILURE - Local supply is insufficient", AMBIGUOUS_BUT_SAFE_ABSTENTION: "AMBIGUOUS_BUT_SAFE_ABSTENTION - Tracker abstains safely", BAD_CASE: "BAD_CASE - Case is not suitable", UNRESOLVED: "UNRESOLVED - Evidence is unresolved"})[value] || value || "Select a structured outcome";
}

function stableSeedLabel(value) {
  return ({CONFIRM: "CONFIRM - Proposed A/B seeds are usable", SWAP_A_B: "SWAP_A_B - Swap the proposed A/B seeds", CORRECT_A: "CORRECT_A - Correct Strand A seed", CORRECT_B: "CORRECT_B - Correct Strand B seed", REJECT_BAD_SEED_CASE: "REJECT_BAD_SEED_CASE - Reject this seed case"})[value] || value || "No seed action selected";
}

function stableAutoSummary(draft) {
  const frame = draft.first_failure_frame ? ` First failure frame: ${draft.first_failure_frame}.` : "";
  if (draft.answers?.seed_action === "REJECT_BAD_SEED_CASE") {
    return `Seed action: ${stableSeedLabel(draft.answers.seed_action)}. Rejection reason: ${draft.seed_rejection_reason || "not selected"}.`;
  }
  return `Seed action: ${stableSeedLabel(draft.answers?.seed_action)}. Outcome: ${stableOutcomeLabel(draft.answers?.continuity_outcome)}.${frame}`;
}

function premiumConfigureStableContract() {
  if (stablePremiumConfigured || uiConfig?.presentation_mode !== "stable_local_strand_continuity") return;
  stablePremiumConfigured = true;
  $("premiumReviewTitle").textContent = uiConfig.review_title || "Stable local strand benchmark";
  const task = document.querySelector(".taskCard p");
  if (task) task.textContent = "Confirm or correct the temporary anonymous seeds first. Then judge continuity only. Cyan is Strand A, magenta is Strand B, predictions are off by default, and notes are optional for structured outcomes.";
  const fields = document.querySelectorAll(".evidenceQuestion");
  const seedOptions = [["CONFIRM", "Confirm proposed seeds"], ["SWAP_A_B", "Swap A/B"], ["CORRECT_A", "Correct A"], ["CORRECT_B", "Correct B"], ["REJECT_BAD_SEED_CASE", "Reject bad seed case"]];
  const outcomeOptions = [["PASS", "PASS"], ["A_SWITCH", "A_SWITCH"], ["B_SWITCH", "B_SWITCH"], ["BOTH_SWITCH", "BOTH_SWITCH"], ["A_LOST", "A_LOST"], ["B_LOST", "B_LOST"], ["BOTH_LOST", "BOTH_LOST"], ["DETECTION_SUPPLY_FAILURE", "Detection supply failure"], ["AMBIGUOUS_BUT_SAFE_ABSTENTION", "Ambiguous but safe abstention"], ["BAD_CASE", "Bad case"], ["UNRESOLVED", "Unresolved"]];
  const radioMarkup = (name, options) => options.map(([value, label]) => `<label><input type="radio" name="${name}" value="${value}">${label}</label>`).join("");
  if (fields[0]) { fields[0].dataset.question = "seed_action"; fields[0].innerHTML = `<legend><span class="questionNumber">1</span> Confirm or correct the proposed anonymous A/B seeds.</legend><div class="radioStack">${radioMarkup("seed_action", seedOptions)}</div><p class="helper">This action is local to this short sequence and does not create identity.</p>`; }
  if (fields[1]) { fields[1].dataset.question = "continuity_outcome"; fields[1].innerHTML = `<legend><span class="questionNumber">2</span> What is the continuity outcome after reviewing the sequence?</legend><div class="radioStack">${radioMarkup("continuity_outcome", outcomeOptions)}</div>`; }
  if (fields[2]) { fields[2].dataset.question = "first_failure_frame"; fields[2].innerHTML = `<legend><span class="questionNumber">3</span> First failure frame, only for a switch or loss.</legend><label>Frame number<input id="premiumFirstFailureFrame" type="number" min="0" step="1" placeholder="Optional unless switch/loss"></label><p class="helper">Leave blank for PASS, safe abstention, bad case or unresolved unless useful.</p>`; }
  const rejection = uiConfig.question_contract?.seed_rejection_contract;
  if (fields[3]) {
    fields[3].dataset.question = "seed_correction";
    const reasonOptions = (rejection?.rejection_reasons || []).map((value) => `<option value="${value}">${value.replaceAll("_", " ")}</option>`).join("");
    const rejectionMarkup = rejection ? `<label id="premiumSeedRejectionWrap" class="isHidden">Structured rejection reason<select id="premiumSeedRejectionReason"><option value="">Select rejection reason</option>${reasonOptions}</select></label>` : "";
    fields[3].innerHTML = `<legend><span class="questionNumber">4</span> Seed correction or rejection detail.</legend><label>Correction note<input id="premiumSeedCorrection" type="text" maxlength="240" autocomplete="off" placeholder="Required only for Correct A or Correct B"></label>${rejectionMarkup}`;
  }
  const conclusionCard = document.querySelector(".conclusionCard");
  if (conclusionCard) conclusionCard.classList.add("stableConclusionCard");
  const noteLabel = document.querySelector('label[for="premiumNote"]');
  if (noteLabel) noteLabel.textContent = "Optional note (required only for BAD_CASE, UNRESOLVED or manual override).";
  $("premiumNote").placeholder = "Optional for structured outcomes.";
  $("premiumConclusion").classList.add("isHidden");
  $("premiumSubtypeWrap").classList.add("isHidden");
  $("premiumOverrideWrap").classList.add("isHidden");
  $("premiumConfirm").closest("label")?.classList.add("isHidden");
}

function premiumConfigureErrorAtlasContract() {
  if (errorAtlasConfigured || uiConfig?.presentation_mode !== "development_error_atlas") return;
  errorAtlasConfigured = true;
  $("premiumReviewTitle").textContent = uiConfig.review_title || "Development error atlas";
  const task = document.querySelector(".taskCard p");
  if (task) task.textContent = "Audit the synchronized full-panorama failure evidence. Gold is read-only. Compare the legacy path, repaired path, candidates, motion region, tracklets and top-K alternative.";
  const configuredQuestions = uiConfig.question_contract?.evidence_questions || [];
  const form = $("premiumReviewForm");
  let fields = [...form.querySelectorAll(".evidenceQuestion")];
  while (fields.length < configuredQuestions.length) {
    const field = document.createElement("fieldset");
    field.className = "evidenceQuestion";
    fields[fields.length - 1].insertAdjacentElement("afterend", field);
    fields = [...form.querySelectorAll(".evidenceQuestion")];
  }
  const options = uiConfig.question_contract?.answer_values || ["YES", "NO", "UNRESOLVED"];
  configuredQuestions.forEach((question, index) => {
    const field = fields[index];
    field.dataset.question = question.key;
    field.innerHTML = `<legend><span class="questionNumber">${index + 1}</span> ${question.label}</legend><div class="radioStack">${options.map((value) => `<label><input type="radio" name="${question.key}" value="${value}">${value.replaceAll("_", " ")}</label>`).join("")}</div>`;
  });
  fields.slice(configuredQuestions.length).forEach((field) => field.classList.add("isHidden"));
  $("premiumConclusion").innerHTML = `<option value="">Choose an audit outcome</option>${uiConfig.decisions.map((option) => `<option value="${option.value}">${option.label}</option>`).join("")}`;
  $("premiumSubtypeWrap").classList.add("isHidden");
  $("premiumOverrideWrap").classList.add("isHidden");
  $("premiumConfirm").closest("label")?.classList.add("isHidden");
  const noteLabel = document.querySelector('label[for="premiumNote"]');
  if (noteLabel) noteLabel.textContent = "Optional note";
  $("premiumNote").placeholder = "Optional";
}

function premiumApplySeedRejectionState() {
  if (uiConfig?.presentation_mode !== "stable_local_strand_continuity") return;
  const contract = uiConfig.question_contract?.seed_rejection_contract;
  if (!contract) return;
  const rejected = document.querySelector('input[name="seed_action"]:checked')?.value === (contract.rejection_action || "REJECT_BAD_SEED_CASE");
  document.querySelectorAll('input[name="continuity_outcome"]').forEach((input) => {
    input.disabled = rejected;
    if (rejected) input.checked = false;
  });
  const failure = $("premiumFirstFailureFrame");
  if (failure) { failure.disabled = rejected; if (rejected) failure.value = ""; }
  const outcomeField = document.querySelector('[data-question="continuity_outcome"]');
  if (outcomeField) outcomeField.classList.toggle("isHidden", rejected);
  const reasonWrap = $("premiumSeedRejectionWrap");
  if (reasonWrap) reasonWrap.classList.toggle("isHidden", !rejected);
  if (!rejected && $("premiumSeedRejectionReason")) $("premiumSeedRejectionReason").value = "";
}

function premiumRenderQuestions(caseData) {
  const values = premiumAnswerValues(caseData);
  document.querySelectorAll("#premiumReviewForm input[type=radio]").forEach((input) => {
    input.checked = values[input.name] === input.value;
  });
  const draft = premiumGetDraft(caseData);
  $("premiumNote").value = draft.note || "";
  $("premiumConclusion").value = draft.conclusion || "";
  $("premiumSubtype").value = draft.subtype || "";
  $("premiumOverrideReason").value = draft.overrideReason || "";
  $("premiumConfirm").checked = draft.confirmed === true;
  $("premiumAnnotationStart").value = draft.annotation?.start_frame ?? "";
  $("premiumAnnotationEnd").value = draft.annotation?.end_frame ?? "";
  $("premiumMergeRegion").value = draft.annotation?.merge_region ?? "";
  if (uiConfig?.presentation_mode === "stable_local_strand_continuity") {
    if ($("premiumFirstFailureFrame")) $("premiumFirstFailureFrame").value = draft.first_failure_frame || "";
    if ($("premiumSeedCorrection")) $("premiumSeedCorrection").value = draft.seed_correction || "";
    if ($("premiumSeedRejectionReason")) $("premiumSeedRejectionReason").value = draft.seed_rejection_reason || "";
    premiumApplySeedRejectionState();
  }
  premiumRenderSuggestion(caseData);
}

function premiumRenderSuggestion(caseData) {
  const draft = premiumGetDraft(caseData);
  if (uiConfig?.presentation_mode === "development_error_atlas") {
    const option = uiConfig.decisions.find((item) => item.value === draft.conclusion);
    $("premiumSuggestionTitle").textContent = option?.label || "Choose an audit outcome";
    $("premiumSuggestionReason").textContent = "The five evidence answers and one structured outcome will be persisted together. Notes are optional.";
    $("premiumSuggestionState").textContent = draft.conclusion ? "Structured outcome" : "Awaiting outcome";
    $("premiumSuggestionState").className = `suggestionState ${draft.conclusion ? "confirmed" : ""}`;
    return;
  }
  if (uiConfig?.presentation_mode === "stable_local_strand_continuity") {
    const outcome = draft.answers?.continuity_outcome;
    $("premiumSuggestionTitle").textContent = outcome ? stableOutcomeLabel(outcome) : "Choose a continuity outcome";
    $("premiumSuggestionReason").textContent = stableAutoSummary(draft);
    $("premiumSuggestionState").textContent = outcome ? "Structured outcome" : "Awaiting outcome";
    $("premiumSuggestionState").className = `suggestionState ${outcome ? "confirmed" : ""}`;
    return;
  }
  const answers = premiumAnswerValues(caseData);
  const required = ["incoming_people_supported", "during_state", "outgoing_people_supported", "path_continuity_plausible"];
  const complete = required.every((key) => answers[key]);
  const suggestion = complete ? premiumSuggestion(answers) : null;
  const selected = draft.conclusion || suggestion?.code || "";
  $("premiumSuggestionTitle").textContent = suggestion ? `Suggested: ${premiumConclusionLabel(suggestion.code)}` : "Complete the four evidence questions";
  $("premiumSuggestionReason").textContent = suggestion?.reason || "The viewer will suggest a human-facing conclusion after the evidence answers are complete.";
  $("premiumSuggestionState").textContent = suggestion ? (draft.confirmed ? "Confirmed" : "Confirmation required") : "Answer the questions";
  $("premiumSuggestionState").className = `suggestionState ${draft.confirmed ? "confirmed" : ""}`;
  if (suggestion && !draft.conclusion) {
    draft.conclusion = suggestion.code;
    draft.subtype = suggestion.subtype || "";
  }
  $("premiumConclusion").value = selected;
  $("premiumSubtypeWrap").classList.toggle("isHidden", selected !== "G");
  $("premiumOverrideWrap").classList.toggle("isHidden", !suggestion || selected === suggestion.code);
  if (selected === "G" && !draft.subtype && suggestion?.subtype) {
    draft.subtype = suggestion.subtype;
    $("premiumSubtype").value = draft.subtype;
  }
}

function premiumApplyView() {
  const stage = $("premiumStage");
  stage.dataset.view = premiumView;
  const caseData = premiumCase();
  if (["local_encounter_strands", "stable_local_strand_continuity", "development_error_atlas"].includes(uiConfig?.presentation_mode)) {
    stage.style.setProperty("--focal-scale", "1");
    stage.style.setProperty("--focal-shift-x", "0%");
    stage.style.setProperty("--focal-shift-y", "0%");
    stage.dataset.view = premiumView;
    document.querySelectorAll("[data-premium-view]").forEach((button) => button.classList.toggle("active", button.dataset.premiumView === premiumView));
    $("premiumLocatorToggle").disabled = premiumView !== "panorama";
    if (premiumView !== "panorama") $("premiumLocatorToggle").checked = false;
    return;
  }
  const region = caseData.visible_metadata?.focal_region || {};
  const width = Number(caseData.visible_metadata?.source_width || 2730);
  const height = Number(caseData.visible_metadata?.source_height || 720);
  const rw = Math.max(1, Number(region.x2 || width * 0.3) - Number(region.x1 || width * 0.2));
  const rh = Math.max(1, Number(region.y2 || height * 0.8) - Number(region.y1 || height * 0.2));
  const scale = Math.min(3.2, Math.max(1.35, Math.min(width / rw, height / rh) * 0.82));
  const cx = ((Number(region.x1 || width / 2) + Number(region.x2 || width / 2)) / 2) / width;
  const cy = ((Number(region.y1 || height / 2) + Number(region.y2 || height / 2)) / 2) / height;
  stage.style.setProperty("--focal-scale", String(premiumView === "focal" ? scale : 1));
  stage.style.setProperty("--focal-shift-x", `${(0.5 - cx) * 100}%`);
  stage.style.setProperty("--focal-shift-y", `${(0.5 - cy) * 100}%`);
  document.querySelectorAll("[data-premium-view]").forEach((button) => button.classList.toggle("active", button.dataset.premiumView === premiumView));
  $("premiumLocatorToggle").disabled = premiumView !== "panorama";
  if (premiumView !== "panorama") $("premiumLocatorToggle").checked = false;
}

function premiumSetLayerVisibility() {
  const predicted = $("premiumPredictedToggle").checked;
  const alternative = $("premiumAlternativeToggle").checked;
  const labels = $("premiumLabelsToggle").checked;
  const locator = $("premiumLocatorToggle").checked && premiumView === "panorama";
  $("premiumObservedLayer").classList.toggle("isHidden", !$("premiumObservedToggle").checked);
  $("premiumAllDetectionsLayer").classList.toggle("isHidden", !$("premiumAllDetectionsToggle").checked);
  $("premiumPredictedLayer").classList.toggle("isHidden", !predicted);
  $("premiumAlternativeLayer").classList.toggle("isHidden", !alternative);
  $("premiumLabelsLayer").classList.toggle("isHidden", !labels);
  $("premiumLocatorLayer").classList.toggle("isHidden", !locator);
}

function premiumPrimeFrame(caseData) {
  const record = premiumRecord(caseData);
  if (!record) return;
  const assets = premiumEvidenceAssets(caseData);
  const prefix = premiumView === "panorama" ? "panorama_" : "";
  const base = assets[record.assets[`${prefix}base`] || record.assets.base];
  const observed = assets[record.assets[`${prefix}observed`] || record.assets.observed];
  if (base) $("premiumBaseLayer").src = premiumAssetUrl(caseData, base.asset_id);
  if (observed) $("premiumObservedLayer").src = premiumAssetUrl(caseData, observed.asset_id);
}

async function premiumLoadFrame(caseData) {
  const record = premiumRecord(caseData);
  if (!record) return;
  const assets = premiumEvidenceAssets(caseData);
  const enabled = ["base", "observed", "all_detections", "predicted", "alternative_hypothesis", "labels", "locator"].filter((layer) => {
    if (layer === "predicted") return $("premiumPredictedToggle").checked;
    if (layer === "alternative_hypothesis") return $("premiumAlternativeToggle").checked;
    if (layer === "all_detections") return $("premiumAllDetectionsToggle").checked;
    if (layer === "labels") return $("premiumLabelsToggle").checked;
    if (layer === "locator") return $("premiumLocatorToggle").checked && premiumView === "panorama";
    return true;
  });
  const token = ++premiumRenderToken;
  $("premiumSyncStatus").textContent = "Checking frame...";
  const loaded = await Promise.all(enabled.map((layer) => new Promise((resolve, reject) => {
    const prefix = premiumView === "panorama" ? "panorama_" : "";
    const item = assets[record.assets[`${prefix}${layer}`] || record.assets[layer]];
    if (!item) return reject(new Error(`${layer} layer is unavailable`));
    const image = new Image();
    image.onload = () => resolve({layer, image, item});
    image.onerror = () => reject(new Error(`${layer} image failed to load`));
    image.src = premiumAssetUrl(caseData, item.asset_id);
  })));
  if (token !== premiumRenderToken) return;
  const width = loaded[0].image.naturalWidth;
  const height = loaded[0].image.naturalHeight;
  if (loaded.some((item) => item.image.naturalWidth !== width || item.image.naturalHeight !== height)) throw new Error("enabled evidence layers have mismatched dimensions");
  const targets = {base: $("premiumBaseLayer"), observed: $("premiumObservedLayer"), all_detections: $("premiumAllDetectionsLayer"), predicted: $("premiumPredictedLayer"), alternative_hypothesis: $("premiumAlternativeLayer"), labels: $("premiumLabelsLayer"), locator: $("premiumLocatorLayer")};
  loaded.forEach(({layer, item}) => { targets[layer].src = premiumAssetUrl(caseData, item.asset_id); targets[layer].dataset.frame = String(record.frame_sequence); targets[layer].dataset.timestamp = String(record.timestamp_seconds); });
  $("premiumEvidenceBlocker").classList.add("isHidden");
  $("premiumSyncStatus").textContent = "Synchronized";
  $("premiumFrameReadout").textContent = `Frame ${record.frame_sequence}`;
  $("premiumTimeReadout").textContent = `Time ${Number(record.timestamp_seconds).toFixed(1)}s`;
  $("premiumPhaseReadout").textContent = record.phase;
  $("premiumTimeline").value = String(premiumFrameIndex(caseData));
  premiumApplyView();
  premiumSetLayerVisibility();
}

function premiumRenderMetadata(caseData) {
  const metadata = caseData.visible_metadata || {};
  const interval = metadata.candidate_interval || {};
  const frames = metadata.frame_window || {};
  const records = metadata.frame_records || [];
  const timestamps = records.length ? [records[0].timestamp_seconds, records[records.length - 1].timestamp_seconds] : [];
  $("premiumMetadata").innerHTML = `<div><span>Frames</span><strong>${frames.start ?? "-"}-${frames.end ?? "-"}</strong></div><div><span>Candidate interval</span><strong>${interval.start ?? "-"}-${interval.end ?? "-"}</strong></div><div><span>Time</span><strong>${timestamps.length ? `${Number(timestamps[0]).toFixed(1)}-${Number(timestamps[1]).toFixed(1)} seconds` : "-"}</strong></div><div><span>Source</span><strong>${metadata.source_rate || "canonical 10 FPS"}</strong></div>`;
}

function premiumRenderTimeline(caseData) {
  const records = caseData.visible_metadata?.frame_records || [];
  const phases = {BEFORE: [], INTERVAL: [], AFTER: []};
  records.forEach((record, index) => phases[record.phase]?.push(index));
  const marker = (id, phase, fallback) => { const values = phases[phase] || []; $(id).style.left = `${100 * (values[0] ?? fallback) / Math.max(1, records.length - 1)}%`; };
  marker("premiumBeforeMarker", "BEFORE", 0);
  marker("premiumIntervalStartMarker", "INTERVAL", Math.floor(records.length / 2));
  marker("premiumIntervalMiddleMarker", "INTERVAL", Math.floor(records.length / 2));
  const interval = phases.INTERVAL || [];
  $("premiumIntervalEndMarker").style.left = `${100 * (interval[interval.length - 1] ?? Math.floor(records.length / 2)) / Math.max(1, records.length - 1)}%`;
  marker("premiumAfterMarker", "AFTER", records.length - 1);
}

function premiumRender() {
  if (!premiumMode || !manifest) return;
  const caseData = premiumCase();
  premiumConfigureLocalContract();
  premiumConfigureStableContract();
  premiumConfigureErrorAtlasContract();
  premiumFrameIndex(caseData);
  const reviewed = Object.keys(state?.decisions || {}).length;
  $("premiumReviewTitle").textContent = uiConfig.review_title || (["local_encounter_strands", "stable_local_strand_continuity", "development_error_atlas"].includes(uiConfig.presentation_mode) ? "Visual continuity audit" : "Simplified temporal review");
  $("premiumCaseProgress").textContent = `Case ${activeIndex + 1} of ${manifest.cases.length}`;
  $("premiumProgressBar").style.width = `${100 * reviewed / Math.max(1, manifest.cases.length)}%`;
  $("premiumCaseTitle").textContent = `Case ${activeIndex + 1}`;
  $("premiumQuestion").textContent = uiConfig.question_contract?.primary_question || "Review the synchronized before, during and after evidence.";
  $("premiumComplete").disabled = reviewed < manifest.cases.length;
  premiumRenderQuestions(caseData);
  premiumRenderMetadata(caseData);
  premiumRenderTimeline(caseData);
  premiumApplyView();
  premiumSetLayerVisibility();
  premiumPrimeFrame(caseData);
  premiumLoadFrame(caseData).catch((error) => {
    $("premiumEvidenceBlocker").textContent = `Evidence unavailable: ${error.message}`;
    $("premiumEvidenceBlocker").classList.remove("isHidden");
    $("premiumSyncStatus").textContent = "Evidence blocked";
  });
  window.setTimeout(() => {
    if ($( "premiumBaseLayer").naturalWidth === 0 && premiumCase()?.case_id === caseData.case_id) {
      premiumLoadFrame(caseData).catch(() => {});
    }
  }, 350);
}

function premiumGo(delta) {
  activeIndex = Math.max(0, Math.min(manifest.cases.length - 1, activeIndex + delta));
  premiumPlaying = false;
  clearInterval(premiumPlayTimer);
  $("premiumPlay").textContent = "Play";
  premiumRender();
}

function premiumStep(delta) {
  const caseData = premiumCase();
  const count = (caseData.visible_metadata?.frame_records || []).length;
  premiumFrames[caseData.case_id] = Math.max(0, Math.min(count - 1, premiumFrameIndex(caseData) + delta));
  premiumLoadFrame(caseData).catch(() => {});
}

function premiumJump(phase) {
  const records = premiumCase().visible_metadata?.frame_records || [];
  const index = phase === "start" ? 0 : records.length - 1;
  const target = phase === "start" ? records.findIndex((item) => item.phase === "BEFORE") : [...records].reverse().findIndex((item) => item.phase === "AFTER");
  premiumFrames[premiumCase().case_id] = target >= 0 ? (phase === "start" ? target : records.length - 1 - target) : index;
  premiumLoadFrame(premiumCase()).catch(() => {});
}

function premiumJumpPhase(delta) {
  const records = premiumCase().visible_metadata?.frame_records || [];
  const current = records[premiumFrameIndex(premiumCase())]?.phase;
  const order = ["BEFORE", "INTERVAL", "AFTER"];
  const next = order[Math.max(0, Math.min(order.length - 1, order.indexOf(current) + delta))];
  const index = records.findIndex((item) => item.phase === next);
  if (index >= 0) { premiumFrames[premiumCase().case_id] = index; premiumLoadFrame(premiumCase()).catch(() => {}); }
}

function premiumTogglePlay() {
  premiumPlaying = !premiumPlaying;
  $("premiumPlay").textContent = premiumPlaying ? "Pause" : "Play";
  $("premiumPlay").setAttribute("aria-pressed", String(premiumPlaying));
  clearInterval(premiumPlayTimer);
  if (premiumPlaying) premiumPlayTimer = setInterval(() => {
    const records = premiumCase().visible_metadata?.frame_records || [];
    if (premiumFrameIndex(premiumCase()) >= records.length - 1) { premiumPlaying = false; clearInterval(premiumPlayTimer); $("premiumPlay").textContent = "Play"; return; }
    premiumStep(1);
  }, 300 / premiumSpeed);
}

function premiumValidateDraft(caseData) {
  const draft = premiumGetDraft(caseData);
  if (uiConfig?.presentation_mode === "development_error_atlas") {
    const questions = uiConfig.question_contract?.evidence_questions || [];
    const errors = [];
    if (!questions.every((question) => draft.answers?.[question.key])) errors.push("Answer all five evidence questions.");
    if (!draft.conclusion || !uiConfig.decisions.some((option) => option.value === draft.conclusion)) errors.push("Choose one structured audit outcome.");
    return errors;
  }
  if (uiConfig?.presentation_mode === "stable_local_strand_continuity") {
    const seed = draft.answers?.seed_action;
    const outcome = draft.answers?.continuity_outcome;
    const rejection = uiConfig.question_contract?.seed_rejection_contract;
    const errors = [];
    if (!seed) errors.push("Confirm or correct the proposed seeds.");
    if (rejection && seed === (rejection.rejection_action || "REJECT_BAD_SEED_CASE")) {
      if (!draft.seed_rejection_reason || !(rejection.rejection_reasons || []).includes(draft.seed_rejection_reason)) errors.push("Choose a structured seed rejection reason.");
      return errors;
    }
    if (!outcome) errors.push("Choose one continuity outcome.");
    if (["CORRECT_A", "CORRECT_B"].includes(seed) && !String(draft.seed_correction || "").trim()) errors.push("Describe the corrected seed briefly.");
    if (["A_SWITCH", "B_SWITCH", "BOTH_SWITCH", "A_LOST", "B_LOST", "BOTH_LOST"].includes(outcome) && !String(draft.first_failure_frame || "").trim()) errors.push("Choose the first failure frame for this outcome.");
    if (["BAD_CASE", "UNRESOLVED"].includes(outcome) && !String(draft.note || "").trim()) errors.push("Add a note for this outcome.");
    return errors;
  }
  const required = ["incoming_people_supported", "during_state", "outgoing_people_supported", "path_continuity_plausible"];
  const errors = [];
  if (!required.every((key) => draft.answers?.[key])) errors.push("Answer all four evidence questions.");
  if (!draft.conclusion) errors.push("Choose the suggested conclusion or an override.");
  if (draft.conclusion === "G" && !draft.subtype) errors.push("Choose a genuine subtype.");
  const suggestion = premiumSuggestion(draft.answers || {});
  if (suggestion && draft.conclusion !== suggestion.code && !String(draft.overrideReason || "").trim()) errors.push("Add a brief reason for overriding the suggestion.");
  if (!draft.confirmed) errors.push("Confirm the conclusion before saving.");
  if (!String(draft.note || "").trim()) errors.push("Add one concise note describing before, during and after.");
  return errors;
}

async function premiumSaveAndNext(event) {
  event.preventDefault();
  const caseData = premiumCase();
  const draft = premiumGetDraft(caseData);
  premiumCollectAnswers(caseData);
  if (uiConfig?.presentation_mode === "development_error_atlas") {
    draft.note = $("premiumNote").value;
    draft.conclusion = $("premiumConclusion").value;
    const errors = premiumValidateDraft(caseData);
    if (errors.length) { $("premiumError").textContent = errors.join(" "); $("premiumError").classList.remove("isHidden"); return; }
    $("premiumError").classList.add("isHidden");
    const structuredReview = {evidence_answers: draft.answers, audit_outcome: draft.conclusion, note: String(draft.note || "").trim() || null, gold_labels_mutated: false};
    premiumSetStatus("Saving", "saving");
    $("premiumSaveNext").disabled = true;
    try {
      state = await api("/api/review/decision", {method: "POST", body: JSON.stringify({case_id: caseData.case_id, decision: draft.conclusion, note: String(draft.note || "").trim(), structured_review: structuredReview, input_source: "save_and_next", last_viewed_case_id: caseData.case_id, elapsed_active_seconds: activeTimeNow()})});
      localStorage.removeItem(premiumDraftKey(caseData));
      delete premiumDrafts[caseData.case_id];
      premiumSetStatus("Saved", "saved");
      if (activeIndex < manifest.cases.length - 1) activeIndex += 1;
      premiumRender();
    } catch (error) {
      premiumSetStatus("Error", "error");
      $("premiumError").textContent = `Save failed: ${error.message}`;
      $("premiumError").classList.remove("isHidden");
    } finally { $("premiumSaveNext").disabled = false; }
    return;
  }
  if (uiConfig?.presentation_mode === "stable_local_strand_continuity") {
    draft.note = $("premiumNote").value;
    draft.first_failure_frame = $("premiumFirstFailureFrame")?.value || "";
    draft.seed_correction = $("premiumSeedCorrection")?.value || "";
    draft.seed_rejection_reason = $("premiumSeedRejectionReason")?.value || "";
    const errors = premiumValidateDraft(caseData);
    if (errors.length) { $("premiumError").textContent = errors.join(" "); $("premiumError").classList.remove("isHidden"); return; }
    $("premiumError").classList.add("isHidden");
    const rejection = uiConfig.question_contract?.seed_rejection_contract;
    const isRejected = rejection && draft.answers.seed_action === (rejection.rejection_action || "REJECT_BAD_SEED_CASE");
    const outcome = isRejected ? (rejection.rejection_decision || "BAD_SEED_CASE") : draft.answers.continuity_outcome;
    const structuredReview = {seed_action: draft.answers.seed_action, continuity_outcome: isRejected ? null : outcome, first_failure_frame: isRejected ? null : (draft.first_failure_frame || null), seed_correction: draft.seed_correction || null, seed_rejection_reason: isRejected ? draft.seed_rejection_reason : null, auto_generated_summary: stableAutoSummary(draft), note: String(draft.note || "").trim() || null};
    premiumSetStatus("Saving", "saving");
    $("premiumSaveNext").disabled = true;
    try {
      state = await api("/api/review/decision", {method: "POST", body: JSON.stringify({case_id: caseData.case_id, decision: outcome, note: String(draft.note || "").trim(), structured_review: structuredReview, input_source: "save_and_next", last_viewed_case_id: caseData.case_id, elapsed_active_seconds: activeTimeNow()})});
      localStorage.removeItem(premiumDraftKey(caseData));
      delete premiumDrafts[caseData.case_id];
      premiumSetStatus("Saved", "saved");
      if (activeIndex < manifest.cases.length - 1) activeIndex += 1;
      premiumRender();
    } catch (error) {
      premiumSetStatus("Error", "error");
      $("premiumError").textContent = `Save failed: ${error.message}`;
      $("premiumError").classList.remove("isHidden");
    } finally { $("premiumSaveNext").disabled = false; }
    return;
  }
  draft.note = $("premiumNote").value;
  draft.conclusion = $("premiumConclusion").value;
  draft.subtype = $("premiumSubtype").value;
  draft.overrideReason = $("premiumOverrideReason").value;
  draft.confirmed = $("premiumConfirm").checked;
  draft.annotation = {start_frame: $("premiumAnnotationStart").value || null, end_frame: $("premiumAnnotationEnd").value || null, merge_region: $("premiumMergeRegion").value || null};
  const errors = premiumValidateDraft(caseData);
  if (errors.length) { $("premiumError").textContent = errors.join(" "); $("premiumError").classList.remove("isHidden"); return; }
  $("premiumError").classList.add("isHidden");
  const canonical = premiumCanonicalLabel(draft.conclusion, draft.subtype);
  if (!canonical) { $("premiumError").textContent = "The conclusion could not be mapped safely."; $("premiumError").classList.remove("isHidden"); return; }
  const structuredReview = {answers: draft.answers, suggested_conclusion: premiumSuggestion(draft.answers).code, confirmed_conclusion: draft.conclusion, canonical_label: canonical, genuine_subtype: draft.conclusion === "G" ? draft.subtype : null, override_reason: draft.conclusion === premiumSuggestion(draft.answers).code ? null : draft.overrideReason.trim(), note: draft.note.trim(), optional_annotation: draft.annotation};
  premiumSetStatus("Saving", "saving");
  $("premiumSaveNext").disabled = true;
  try {
    state = await api("/api/review/decision", {method: "POST", body: JSON.stringify({case_id: caseData.case_id, decision: canonical, note: draft.note.trim(), structured_review: structuredReview, input_source: "save_and_next", last_viewed_case_id: caseData.case_id, elapsed_active_seconds: activeTimeNow()})});
    localStorage.removeItem(premiumDraftKey(caseData));
    delete premiumDrafts[caseData.case_id];
    premiumSetStatus("Saved", "saved");
    if (activeIndex < manifest.cases.length - 1) activeIndex += 1;
    premiumRender();
  } catch (error) {
    premiumSetStatus("Error", "error");
    $("premiumError").textContent = `Save failed: ${error.message}`;
    $("premiumError").classList.remove("isHidden");
  } finally {
    $("premiumSaveNext").disabled = false;
  }
}

function premiumBind() {
  if (premiumInitialized) return;
  premiumInitialized = true;
  document.querySelectorAll("[data-premium-view]").forEach((button) => button.addEventListener("click", () => { premiumView = button.dataset.premiumView; premiumApplyView(); premiumSetLayerVisibility(); premiumLoadFrame(premiumCase()).catch(() => {}); }));
  document.querySelectorAll("[data-premium-step]").forEach((button) => button.addEventListener("click", () => premiumStep(Number(button.dataset.premiumStep))));
  document.querySelectorAll("[data-premium-jump]").forEach((button) => button.addEventListener("click", () => premiumJump(button.dataset.premiumJump)));
  document.querySelectorAll("[data-premium-speed]").forEach((button) => button.addEventListener("click", () => { premiumSpeed = Number(button.dataset.premiumSpeed); document.querySelectorAll("[data-premium-speed]").forEach((item) => item.classList.toggle("activeSpeed", item === button)); if (premiumPlaying) { premiumTogglePlay(); premiumTogglePlay(); } }));
  $("premiumPlay").addEventListener("click", premiumTogglePlay);
  $("premiumPrev").addEventListener("click", () => premiumGo(-1));
  $("premiumNext").addEventListener("click", () => premiumGo(1));
  $("premiumTimeline").addEventListener("input", (event) => { premiumFrames[premiumCase().case_id] = Number(event.target.value); premiumLoadFrame(premiumCase()).catch(() => {}); });
  ["premiumObservedToggle", "premiumAllDetectionsToggle", "premiumPredictedToggle", "premiumLabelsToggle", "premiumLocatorToggle", "premiumAlternativeToggle"].forEach((id) => $(id).addEventListener("change", () => { premiumSetLayerVisibility(); premiumLoadFrame(premiumCase()).catch(() => {}); }));
  $("premiumReviewForm").addEventListener("change", () => { const draft = premiumGetDraft(premiumCase()); premiumApplySeedRejectionState(); premiumCollectAnswers(premiumCase()); premiumRenderSuggestion(premiumCase()); premiumSaveDraft(premiumCase()); });
  $("premiumReviewForm").addEventListener("input", () => { const draft = premiumGetDraft(premiumCase()); draft.note = $("premiumNote").value; draft.overrideReason = $("premiumOverrideReason").value; draft.annotation = {start_frame: $("premiumAnnotationStart").value || null, end_frame: $("premiumAnnotationEnd").value || null, merge_region: $("premiumMergeRegion").value || null}; if (uiConfig?.presentation_mode === "stable_local_strand_continuity") { draft.first_failure_frame = $("premiumFirstFailureFrame")?.value || ""; draft.seed_correction = $("premiumSeedCorrection")?.value || ""; draft.seed_rejection_reason = $("premiumSeedRejectionReason")?.value || ""; } premiumSaveDraft(premiumCase()); });
  $("premiumConclusion").addEventListener("change", () => { premiumGetDraft(premiumCase()).conclusion = $("premiumConclusion").value; premiumGetDraft(premiumCase()).confirmed = false; premiumRenderSuggestion(premiumCase()); premiumSaveDraft(premiumCase()); });
  $("premiumSubtype").addEventListener("change", () => { premiumGetDraft(premiumCase()).subtype = $("premiumSubtype").value; premiumGetDraft(premiumCase()).confirmed = false; premiumSaveDraft(premiumCase()); });
  $("premiumConfirm").addEventListener("change", () => { premiumGetDraft(premiumCase()).confirmed = $("premiumConfirm").checked; premiumSaveDraft(premiumCase()); });
  $("premiumReviewForm").addEventListener("submit", premiumSaveAndNext);
  $("premiumComplete").addEventListener("click", async () => { try { state = await api("/api/review/complete", {method: "POST", body: JSON.stringify({elapsed_active_seconds: activeTimeNow()})}); premiumSetStatus("Completed", "saved"); } catch (error) { $("premiumError").textContent = `Completion blocked: ${error.message}`; $("premiumError").classList.remove("isHidden"); } });
  $("premiumHelp").addEventListener("click", () => $("premiumHelpDialog").classList.remove("isHidden"));
  $("premiumHelpClose").addEventListener("click", () => $("premiumHelpDialog").classList.add("isHidden"));
  $("premiumHelpDialog").addEventListener("click", (event) => { if (event.target === $("premiumHelpDialog")) $("premiumHelpDialog").classList.add("isHidden"); });
  document.addEventListener("keydown", (event) => {
    if (!premiumMode) return;
    if (event.key === "Escape") { $("premiumHelpDialog").classList.add("isHidden"); return; }
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    if (event.key === "ArrowLeft" && event.shiftKey) { event.preventDefault(); premiumJumpPhase(-1); }
    else if (event.key === "ArrowRight" && event.shiftKey) { event.preventDefault(); premiumJumpPhase(1); }
    else if (event.key === "ArrowLeft") { event.preventDefault(); premiumStep(-1); }
    else if (event.key === "ArrowRight") { event.preventDefault(); premiumStep(1); }
    else if (event.key === " ") { event.preventDefault(); premiumTogglePlay(); }
  });
}

function goldCase() {
  return manifest.cases[activeIndex];
}

function goldDraftKey(caseData) {
  return `gold_strand_${manifest.review_id}_${caseData.case_id}`;
}

function goldPolygonStorageKey() {
  return `gold_polygon_${manifest.review_id}_pitch`;
}

function goldPolygonPayload() {
  const metadata = manifest?.cases?.find((item) => item.task_type === "pitch_polygon_approval")?.visible_metadata || {};
  return {
    vertices_original_pixels: goldPitchVertices.map((point) => ({x: Number(point.x), y: Number(point.y)})),
    tolerance_pixels: Number($("goldPitchTolerance").value),
    source_image_hash: metadata.source_frame_sha256,
    image_width: Number(metadata.image_width),
    image_height: Number(metadata.image_height),
  };
}

function goldPolygonApproved() {
  return Boolean(goldPolygonSidecar?.is_approved);
}

function goldPolygonBackup() {
  try { localStorage.setItem(goldPolygonStorageKey(), JSON.stringify({...goldPolygonPayload(), saved_at: new Date().toISOString()})); } catch {}
}

function goldLegacyPolygonCandidate(key, raw) {
  if (!key || /sequence|annotation/i.test(key) || /frame_annotations|annotations/i.test(raw)) return null;
  let value;
  try { value = JSON.parse(raw); } catch { return null; }
  const candidate = value?.polygon || value?.pitch_polygon || value;
  const vertices = candidate?.vertices_original_pixels || candidate?.polygon_vertices || candidate?.vertices;
  if (!Array.isArray(vertices) || vertices.length < 4) return null;
  const metadata = manifest.cases.find((item) => item.task_type === "pitch_polygon_approval")?.visible_metadata || {};
  if (candidate.source_image_hash && candidate.source_image_hash !== metadata.source_frame_sha256) return null;
  if (candidate.source_frame_sha256 && candidate.source_frame_sha256 !== metadata.source_frame_sha256) return null;
  const dimensions = candidate.source_dimensions || {};
  if (dimensions.width && Number(dimensions.width) !== Number(metadata.image_width)) return null;
  if (dimensions.height && Number(dimensions.height) !== Number(metadata.image_height)) return null;
  return {
    key,
    vertices_original_pixels: vertices,
    tolerance_pixels: Number(candidate.tolerance_pixels ?? metadata.tolerance_pixels),
    source_image_hash: metadata.source_frame_sha256,
    image_width: Number(metadata.image_width),
    image_height: Number(metadata.image_height),
  };
}

async function goldLoadPolygonSidecar() {
  if (!uiConfig.question_contract?.polygon_sidecar?.enabled) return;
  goldPolygonSidecar = await api("/api/review/polygon");
  const draft = goldPolygonSidecar.draft || {};
  if (draft.vertices_original_pixels?.length && draft.status !== "PROPOSAL") {
    goldPitchVertices = draft.vertices_original_pixels.map((point) => ({x: Number(point.x), y: Number(point.y)}));
  }
  if (draft.tolerance_pixels != null) $("goldPitchTolerance").value = String(draft.tolerance_pixels);
  if (draft.status === "DRAFT") {
    $("goldSaveState").textContent = "Saved draft";
    $("goldSaveState").className = "saveState unsaved";
  }
  if (goldPolygonSidecar.is_approved) {
    goldPitchVertices = goldPolygonSidecar.approved.vertices_original_pixels.map((point) => ({x: Number(point.x), y: Number(point.y)}));
    $("goldSaveState").textContent = "Pitch approved";
    $("goldSaveState").className = "saveState saved";
    return;
  }
  const candidates = [];
  for (const storage of [localStorage, sessionStorage]) {
    for (const key of Object.keys(storage)) {
      const candidate = goldLegacyPolygonCandidate(key, storage.getItem(key) || "");
      if (candidate) candidates.push(candidate);
    }
  }
  if (!candidates.length) return;
  const candidate = candidates.sort((a, b) => a.key.localeCompare(b.key))[0];
  try {
    const migrated = await api("/api/review/polygon/migrate", {method: "POST", body: JSON.stringify(candidate)});
    goldPolygonSidecar = migrated;
    goldPitchVertices = migrated.draft.vertices_original_pixels.map((point) => ({x: Number(point.x), y: Number(point.y)}));
    $("goldPitchTolerance").value = String(migrated.draft.tolerance_pixels);
    $("goldSaveState").textContent = "Recovered";
    $("goldSaveState").className = "saveState recovered";
    $("goldPitchMessage").textContent = "Recovered your previous polygon edit";
    try { localStorage.removeItem(candidate.key); sessionStorage.removeItem(candidate.key); } catch {}
  } catch (error) {
    $("goldPitchMessage").textContent = `Draft recovery needs review: ${error.message}`;
  }
}

async function goldSavePolygonDraft(source = "explicit_save") {
  if (!goldPolygonIsValid()) throw new Error("The polygon is not valid in the source image.");
  goldPolygonBackup();
  const saved = await api("/api/review/polygon/draft", {method: "POST", body: JSON.stringify({...goldPolygonPayload(), migration_source: source})});
  goldPolygonSidecar = saved;
  $("goldSaveState").textContent = "Saved";
  $("goldSaveState").className = "saveState saved";
  goldUpdateCompletionGate();
  return saved;
}

function goldSchedulePolygonDraftSave() {
  goldPolygonBackup();
  clearTimeout(goldPolygonSaveTimer);
  $("goldSaveState").textContent = "Saving";
  $("goldSaveState").className = "saveState unsaved";
  goldPolygonSaveTimer = setTimeout(() => goldSavePolygonDraft("vertex_drag").catch((error) => {
    $("goldSaveState").textContent = "Error";
    $("goldPitchMessage").textContent = `Draft save failed: ${error.message}`;
  }), 180);
}

function goldDefaultDraft() {
  return {
    annotations: {}, note: "", clicks: 0, accepted_in_runs: 0, manual_bbox_count: 0,
    seed_confirmation: null, dirty: false, hydrated: false, server_finalized: false,
  };
}

function goldDraft(caseData) {
  if (goldDrafts[caseData.case_id]) return goldDrafts[caseData.case_id];
  try {
    const stored = JSON.parse(localStorage.getItem(goldDraftKey(caseData)) || "{}");
    goldDrafts[caseData.case_id] = {
      ...goldDefaultDraft(),
      ...stored,
      dirty: Object.keys(stored).length > 0,
      hydrated: false,
    };
  } catch {
    goldDrafts[caseData.case_id] = goldDefaultDraft();
  }
  return goldDrafts[caseData.case_id];
}

function goldPersistDraft(caseData) {
  const draft = goldDraft(caseData);
  draft.dirty = true;
  draft.server_finalized = false;
  localStorage.setItem(goldDraftKey(caseData), JSON.stringify(draft));
  if (goldPersistence.ready) goldPersistenceStatus("Pending locally");
}

function goldRecords(caseData = goldCase()) {
  return caseData.visible_metadata?.frame_records || [];
}

function goldRecord(caseData = goldCase(), index = goldFrameIndex) {
  const records = goldRecords(caseData);
  return records[Math.max(0, Math.min(index, records.length - 1))];
}

function goldAsset(caseData, assetId) {
  const asset = caseData.evidence_assets.find((item) => item.asset_id === assetId);
  return asset ? evidenceUrl(caseData.case_id, asset.relative_path) : "";
}

function goldAnnotation(caseData, frameSequence) {
  const draft = goldDraft(caseData);
  const key = String(frameSequence);
  draft.annotations[key] = draft.annotations[key] || {A: null, B: null};
  return draft.annotations[key];
}

function goldSeedContractEnabled() {
  return uiConfig?.question_contract?.seed_confirmation_required === true;
}

function goldSeedConfirmed(caseData = goldCase()) {
  const seed = goldDraft(caseData).seed_confirmation;
  return !goldSeedContractEnabled() || Boolean(seed && seed.status === "CONFIRMED" && seed.A && seed.B);
}

function goldSeedRejected(caseData = goldCase()) {
  return Boolean(goldDraft(caseData).seed_confirmation?.status === "REJECTED");
}

function goldProposalUsable(record) {
  const proposal = record?.proposed_annotations || {};
  return Boolean(
    proposal.A?.state === "OBSERVED_EXISTING_DETECTION"
      && proposal.B?.state === "OBSERVED_EXISTING_DETECTION"
      && proposal.A.anonymous_detection_id
      && proposal.B.anonymous_detection_id
      && proposal.A.anonymous_detection_id !== proposal.B.anonymous_detection_id
  );
}

function goldSeedRecord(caseData = goldCase()) {
  const records = goldRecords(caseData);
  const requested = Number(caseData?.visible_metadata?.seed_frame_index || 0);
  return records[Math.max(0, Math.min(records.length - 1, requested))];
}

function goldSeedValues(caseData = goldCase()) {
  const record = goldSeedRecord(caseData);
  const draftSeed = goldDraft(caseData).seed_confirmation;
  const proposal = record?.proposed_annotations || {};
  return {
    A: draftSeed?.A || proposal.A || null,
    B: draftSeed?.B || proposal.B || null,
    source_frame_sequence: Number(record?.frame_sequence),
  };
}

function goldSeedValueForDetection(detection) {
  return detection ? {
    state: "OBSERVED_EXISTING_DETECTION",
    anonymous_detection_id: detection.anonymous_detection_id,
    observation_quality: detection.observation_quality || "UNRESOLVED_MACHINE_OBSERVATION",
  } : null;
}

function goldSeedUsable(caseData = goldCase()) {
  const values = goldSeedValues(caseData);
  const a = values.A;
  const b = values.B;
  if (!a || !b) return false;
  if (a.state !== "OBSERVED_EXISTING_DETECTION" && a.state !== "OBSERVED_MANUAL_BBOX") return false;
  if (b.state !== "OBSERVED_EXISTING_DETECTION" && b.state !== "OBSERVED_MANUAL_BBOX") return false;
  if (a.state === "OBSERVED_EXISTING_DETECTION" && b.state === "OBSERVED_EXISTING_DETECTION") {
    return a.anonymous_detection_id !== b.anonymous_detection_id;
  }
  return true;
}

function goldPushHistory(caseData) {
  goldHistory.push({case_id: caseData.case_id, draft: JSON.parse(JSON.stringify(goldDraft(caseData)))});
  if (goldHistory.length > 100) goldHistory.shift();
}

function goldUndo() {
  const prior = goldHistory.pop();
  if (!prior) return;
  goldDrafts[prior.case_id] = prior.draft;
  const caseData = goldCase();
  if (caseData.case_id === prior.case_id) {
    goldPersistDraft(caseData);
    goldRenderFrame();
  }
}

function goldSetActiveStrand(strand) {
  goldActiveStrand = strand;
  $("goldStrandA").className = strand === "A" ? "activeA" : "";
  $("goldStrandB").className = strand === "B" ? "activeB" : "";
}

function goldSetState(strand, value, caseData = goldCase(), record = goldRecord(caseData)) {
  if (!record) return;
  if (!goldSeedConfirmed(caseData)) return;
  if (value?.state === "OBSERVED_EXISTING_DETECTION") {
    const other = strand === "A" ? "B" : "A";
    const existing = goldAnnotation(caseData, record.frame_sequence)[other];
    if (existing?.state === "OBSERVED_EXISTING_DETECTION"
      && existing.anonymous_detection_id === value.anonymous_detection_id) {
      $("goldError").textContent = "A and B must use distinct observations on this frame.";
      $("goldError").classList.remove("isHidden");
      return;
    }
  }
  goldPushHistory(caseData);
  goldAnnotation(caseData, record.frame_sequence)[strand] = value;
  goldDraft(caseData).clicks += 1;
  goldPersistDraft(caseData);
  goldQueueEvent(value?.state === "OBSERVED_MANUAL_BBOX" ? "MANUAL_BBOX_SET" : "FRAME_STATE_SET", caseData, {value}, record.frame_sequence, strand);
  goldRenderFrame();
}

function goldApplyProposal(run = false) {
  const caseData = goldCase();
  if (caseData.task_type !== "gold_strand_frame_annotation" || !goldSeedConfirmed(caseData)) return;
  goldPushHistory(caseData);
  const records = goldRecords(caseData);
  const indices = run ? records.map((_, index) => index).filter((index) => index >= goldFrameIndex) : [goldFrameIndex];
  let accepted = 0;
  for (const index of indices) {
    const record = records[index];
    const proposal = record.proposed_annotations || {};
    if (!goldProposalUsable(record)) continue;
    goldAnnotation(caseData, record.frame_sequence).A = {...proposal.A};
    goldAnnotation(caseData, record.frame_sequence).B = {...proposal.B};
    goldQueueEvent("FRAME_STATE_SET", caseData, {value: {...proposal.A}}, record.frame_sequence, "A");
    goldQueueEvent("FRAME_STATE_SET", caseData, {value: {...proposal.B}}, record.frame_sequence, "B");
    accepted += 2;
  }
  const draft = goldDraft(caseData);
  draft.clicks += 1;
  if (run) draft.accepted_in_runs += accepted;
  goldPersistDraft(caseData);
  if (!run) {
    const record = records[goldFrameIndex];
    const values = goldAnnotation(caseData, record.frame_sequence);
    goldQueueEvent("PAIR_ACCEPTED", caseData, {values}, record.frame_sequence, null);
  } else {
    goldQueueEvent("STABLE_RUN_ACCEPTED", caseData, {start_frame: records[goldFrameIndex]?.frame_sequence, frame_count: indices.length, frames: indices.map((index) => records[index].frame_sequence)}, null, null);
  }
  if (!run && goldFrameIndex < records.length - 1) goldFrameIndex += 1;
  goldRenderFrame();
}

function goldAssetUrl(caseData, record) {
  const assetId = record.contact_strip_asset_id || record.base_asset_id;
  const asset = caseData.evidence_assets.find((item) => item.asset_id === assetId);
  return asset ? evidenceUrl(caseData.case_id, asset.relative_path) : "";
}

function goldPreviewStableRun() {
  const caseData = goldCase();
  if (!caseData || !goldSeedConfirmed(caseData)) return;
  const records = goldRecords(caseData).slice(goldFrameIndex);
  const strip = $("goldRunContactStrip");
  strip.replaceChildren();
  let gaps = 0;
  let uncertain = 0;
  for (const record of records) {
    const usable = goldProposalUsable(record);
    gaps += Number(!usable);
    uncertain += Number(Boolean(record.machine_uncertain));
    const figure = document.createElement("figure");
    if (!usable) figure.classList.add("gap");
    else if (record.machine_uncertain) figure.classList.add("uncertain");
    const image = document.createElement("img");
    image.src = goldAssetUrl(caseData, record);
    image.alt = `Frame ${record.frame_sequence} with cyan A and magenta B proposal`;
    const caption = document.createElement("figcaption");
    caption.textContent = `Frame ${record.frame_sequence}${usable ? "" : " - proposal gap"}${record.machine_uncertain ? " - uncertain" : ""}`;
    figure.append(image, caption);
    strip.appendChild(figure);
  }
  $("goldRunSummary").textContent = `Frames ${records[0]?.frame_sequence ?? "-"} to ${records.at(-1)?.frame_sequence ?? "-"}; ${records.length * 2} strand-frame events will be persisted.`;
  $("goldRunWarnings").textContent = gaps || uncertain
    ? `${gaps} proposal gaps and ${uncertain} machine-uncertain frames. Only complete, visible proposals may be run-accepted.`
    : "No proposal gaps or machine-uncertain frames are present in this run.";
  $("goldRunConfirm").disabled = records.length === 0 || gaps > 0;
  $("goldRunDialog").showModal();
}

function goldAcceptProposal(run = false) {
  if (run) {
    goldPreviewStableRun();
    return;
  }
  goldApplyProposal(false);
}

function goldFindNextFrame(predicate) {
  const records = goldRecords();
  for (let offset = 1; offset <= records.length; offset += 1) {
    const index = (goldFrameIndex + offset) % records.length;
    if (predicate(records[index])) {
      goldFrameIndex = index;
      goldRenderFrame();
      return;
    }
  }
}

function goldShiftSequence(direction) {
  const draft = goldDraft(goldCase());
  if (goldPersistence.pending.length || draft?.dirty) {
    $("goldError").textContent = "Save and reconcile this sequence before changing sequences.";
    $("goldError").classList.remove("isHidden");
    return;
  }
  const indices = manifest.cases
    .map((caseData, index) => ({caseData, index}))
    .filter((row) => row.caseData.task_type === "gold_strand_frame_annotation")
    .map((row) => row.index);
  const current = indices.indexOf(activeIndex);
  if (current < 0) return;
  activeIndex = indices[Math.max(0, Math.min(indices.length - 1, current + direction))];
  goldFrameIndex = 0;
  goldRender();
}

function goldStateLabel(value) {
  if (!value) return "Not annotated";
  if (value.state === "OBSERVED_EXISTING_DETECTION") return "Accepted existing detection";
  if (value.state === "OBSERVED_MANUAL_BBOX") return "Manual observed bbox";
  return String(value.state).replaceAll("_", " ").toLowerCase();
}

function goldCropBox(record, bbox) {
  return {
    x1: Number(bbox.x1) - Number(record.roi.x1),
    y1: Number(bbox.y1) - Number(record.roi.y1),
    x2: Number(bbox.x2) - Number(record.roi.x1),
    y2: Number(bbox.y2) - Number(record.roi.y1),
  };
}

function goldSvgRect(svg, box, className, data = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  element.setAttribute("x", box.x1);
  element.setAttribute("y", box.y1);
  element.setAttribute("width", Math.max(1, box.x2 - box.x1));
  element.setAttribute("height", Math.max(1, box.y2 - box.y1));
  element.setAttribute("class", className);
  Object.entries(data).forEach(([key, value]) => { element.dataset[key] = value; });
  svg.appendChild(element);
  return element;
}

function goldSvgLabel(svg, box, label, className) {
  const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
  text.setAttribute("x", Math.max(2, box.x1));
  text.setAttribute("y", Math.max(20, box.y1 - 5));
  text.setAttribute("class", `goldDetectionLabel ${className}`);
  text.textContent = label;
  svg.appendChild(text);
}

function goldDetectionBox(record, detection) {
  return goldCropBox(record, detection.bbox_original_pixels);
}

function goldRenderDetectionPair(svg, record, values, accepted = false, prefix = "") {
  for (const strand of ["A", "B"]) {
    const value = values[strand];
    if (!value || value.state !== "OBSERVED_EXISTING_DETECTION") continue;
    const detection = (record.anonymous_detections || []).find(
      (item) => item.anonymous_detection_id === value.anonymous_detection_id
    );
    if (!detection) continue;
    const style = `goldDetection ${accepted ? "" : "proposal "}${strand === "A" ? "strandA" : "strandB"}`;
    const box = goldDetectionBox(record, detection);
    goldSvgRect(svg, box, style, {detectionId: detection.anonymous_detection_id, strand, proposal: String(!accepted)});
    goldSvgLabel(svg, box, strand, strand === "A" ? "strandA" : "strandB");
  }
}

function goldRenderDetections(caseData, record) {
  const svg = $("goldDetectionSvg");
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${record.crop_width} ${record.crop_height}`);
  const annotation = goldAnnotation(caseData, record.frame_sequence);
  const proposal = record.proposed_annotations || {};
  for (const detection of record.anonymous_detections || []) {
    const selectedA = annotation.A?.anonymous_detection_id === detection.anonymous_detection_id;
    const selectedB = annotation.B?.anonymous_detection_id === detection.anonymous_detection_id;
    const proposalA = !annotation.A && proposal.A?.state === "OBSERVED_EXISTING_DETECTION"
      && proposal.A.anonymous_detection_id === detection.anonymous_detection_id;
    const proposalB = !annotation.B && proposal.B?.state === "OBSERVED_EXISTING_DETECTION"
      && proposal.B.anonymous_detection_id === detection.anonymous_detection_id;
    const className = `goldDetection${selectedA ? " strandA" : ""}${selectedB ? " strandB" : ""}${proposalA ? " proposal strandA" : ""}${proposalB ? " proposal strandB" : ""}`;
    const box = goldCropBox(record, detection.bbox_original_pixels);
    const rectangle = goldSvgRect(svg, box, className, {
      detectionId: detection.anonymous_detection_id,
    });
    if (selectedA || proposalA) goldSvgLabel(svg, box, "A", "strandA");
    if (selectedB || proposalB) goldSvgLabel(svg, box, "B", "strandB");
    rectangle.addEventListener("click", (event) => {
      event.stopPropagation();
      goldSetState(goldActiveStrand, {
        state: "OBSERVED_EXISTING_DETECTION",
        anonymous_detection_id: detection.anonymous_detection_id,
        observation_quality: detection.observation_quality || "UNRESOLVED",
      });
    });
  }
  for (const strand of ["A", "B"]) {
    const value = annotation[strand];
    if (value?.state === "OBSERVED_MANUAL_BBOX") {
      goldSvgRect(svg, goldCropBox(record, value.bbox_original_pixels), `goldManualBox ${strand === "A" ? "strandA" : "strandB"}`);
    }
  }
}

function goldDrawSeedCrop(canvas, image, record, value) {
  if (!canvas || !image || !value?.anonymous_detection_id || !image.naturalWidth) return;
  const detection = (record.anonymous_detections || []).find(
    (item) => item.anonymous_detection_id === value.anonymous_detection_id
  );
  if (!detection) return;
  const box = goldCropBox(record, detection.bbox_original_pixels);
  const pad = 18;
  const x = Math.max(0, Math.floor(box.x1 - pad));
  const y = Math.max(0, Math.floor(box.y1 - pad));
  const right = Math.min(image.naturalWidth, Math.ceil(box.x2 + pad));
  const bottom = Math.min(image.naturalHeight, Math.ceil(box.y2 + pad));
  const width = Math.max(1, right - x);
  const height = Math.max(1, bottom - y);
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  context.drawImage(image, x, y, width, height, 0, 0, width, height);
}

function goldRenderSeedDetections(caseData, record) {
  const svg = $("goldSeedSvg");
  if (!svg || !record) return;
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${record.crop_width} ${record.crop_height}`);
  const values = goldSeedValues(caseData);
  for (const detection of record.anonymous_detections || []) {
    const isA = values.A?.anonymous_detection_id === detection.anonymous_detection_id;
    const isB = values.B?.anonymous_detection_id === detection.anonymous_detection_id;
    const box = goldDetectionBox(record, detection);
    const className = `goldDetection${isA ? " strandA" : ""}${isB ? " strandB" : ""}`;
    const rectangle = goldSvgRect(svg, box, className, {detectionId: detection.anonymous_detection_id});
    if (isA) goldSvgLabel(svg, box, "A", "strandA");
    if (isB) goldSvgLabel(svg, box, "B", "strandB");
    rectangle.addEventListener("click", (event) => {
      event.stopPropagation();
      if (!goldSeedMode) return;
      const target = goldSeedMode === "BOTH" ? (goldDraft(caseData).seed_confirmation?.A ? "B" : "A") : goldSeedMode;
      const next = goldDraft(caseData).seed_confirmation || {status: "EDITING", A: null, B: null};
      const other = target === "A" ? "B" : "A";
      if (next[other]?.anonymous_detection_id === detection.anonymous_detection_id) {
        $("goldSeedMessage").textContent = "A and B must use distinct available detections.";
        return;
      }
      next[target] = goldSeedValueForDetection(detection);
      next.status = "EDITING";
      goldDraft(caseData).seed_confirmation = next;
      if (goldSeedMode === "BOTH" && target === "A") goldSeedMode = "BOTH";
      else goldSeedMode = null;
      goldPersistDraft(caseData);
      goldRenderSeed(caseData);
    });
  }
  for (const strand of ["A", "B"]) {
    const value = values[strand];
    if (value?.state !== "OBSERVED_MANUAL_BBOX") continue;
    const box = goldCropBox(record, value.bbox_original_pixels);
    goldSvgRect(svg, box, `goldManualBox ${strand === "A" ? "strandA" : "strandB"}`);
    goldSvgLabel(svg, box, strand, strand === "A" ? "strandA" : "strandB");
  }
  const message = $("goldSeedMessage");
  if (message) message.textContent = goldSeedMode ? `Select the replacement for Strand ${goldSeedMode === "BOTH" ? "A" : goldSeedMode}.` : "Choose Confirm, Swap, Correct, or Reject. Do not infer A or B from white detections.";
}

async function goldRenderSeed(caseData = goldCase()) {
  const record = goldSeedRecord(caseData);
  if (!record) return;
  const records = goldRecords(caseData);
  const seedIndex = Math.max(0, records.indexOf(record));
  const generation = ++goldEvidenceGeneration;
  goldClearEvidenceBlocker();
  try {
    const images = await Promise.all([
      goldLoadContextImage($("goldSeedPreviousImage"), caseData, Math.max(0, seedIndex - 1)),
      goldLoadContextImage($("goldSeedCurrentImage"), caseData, seedIndex),
      goldLoadContextImage($("goldSeedNextImage"), caseData, Math.min(records.length - 1, seedIndex + 1)),
    ]);
    if (generation !== goldEvidenceGeneration) return;
    $("goldSeedCurrentCanvasWrap").style.aspectRatio = `${Number(record.crop_width)} / ${Number(record.crop_height)}`;
    goldRenderSeedDetections(caseData, record);
    goldDrawSeedCrop($("goldSeedCropA"), images[1], record, goldSeedValues(caseData).A);
    goldDrawSeedCrop($("goldSeedCropB"), images[1], record, goldSeedValues(caseData).B);
    goldApplyViewport("frame");
  } catch (error) {
    if (generation === goldEvidenceGeneration) goldSetEvidenceBlocker(error.message);
  }
  const seed = goldDraft(caseData).seed_confirmation;
  $("goldSeedStatus").textContent = seed?.status === "EDITING" ? "Corrected pair pending confirmation" : "Seed confirmation pending";
  $("goldSeedConfirm").disabled = seed?.status === "REJECTED" || !goldSeedUsable(caseData);
  $("goldSeedSaveRejected").disabled = seed?.status !== "REJECTED";
}

function goldStartSeedCorrection(strand) {
  goldSeedMode = strand;
  const caseData = goldCase();
  const seed = goldDraft(caseData).seed_confirmation || {status: "EDITING", A: null, B: null};
  seed.seed_action = strand === "BOTH" ? "CORRECT_BOTH" : (strand === "A" ? "CORRECT_A" : (strand === "B" ? "CORRECT_B" : seed.seed_action || "CONFIRM"));
  seed.status = "EDITING";
  goldDraft(caseData).seed_confirmation = seed;
  goldPersistDraft(caseData);
  $("goldSeedMessage").textContent = `Select the replacement for Strand ${strand === "BOTH" ? "A" : strand}.`;
  goldRenderSeed(caseData);
}

function goldConfirmSeed() {
  const caseData = goldCase();
  if (!goldSeedUsable(caseData)) return;
  const values = goldSeedValues(caseData);
  const confirmed = {
    status: "CONFIRMED",
    seed_action: goldDraft(caseData).seed_confirmation?.seed_action || "CONFIRM",
    source_frame_sequence: values.source_frame_sequence,
    A: {...values.A},
    B: {...values.B},
  };
  const authoritative = state?.gold_materialized?.sequences?.[caseData.case_id];
  if (authoritative?.seed_confirmation
    && JSON.stringify(authoritative.seed_confirmation) === JSON.stringify(confirmed)) {
    const draft = goldDraft(caseData);
    draft.seed_confirmation = JSON.parse(JSON.stringify(authoritative.seed_confirmation));
    draft.dirty = false;
    draft.hydrated = true;
    draft.server_finalized = Boolean(authoritative.finalized);
    localStorage.removeItem(goldDraftKey(caseData));
    goldSeedMode = null;
    goldRenderAnnotationCase(caseData);
    return;
  }
  goldPushHistory(caseData);
  goldDraft(caseData).seed_confirmation = confirmed;
  goldPersistDraft(caseData);
  goldQueueEvent("SEED_CONFIRMED", caseData, {seed_confirmation: goldDraft(caseData).seed_confirmation}, null, null);
  goldSeedMode = null;
  goldRenderAnnotationCase(caseData);
}

function goldSwapSeed() {
  const caseData = goldCase();
  const values = goldSeedValues(caseData);
  if (!values.A || !values.B) return;
  goldPushHistory(caseData);
  goldDraft(caseData).seed_confirmation = {
    status: "EDITING",
    seed_action: "SWAP_A_B",
    source_frame_sequence: values.source_frame_sequence,
    A: {...values.B},
    B: {...values.A},
  };
  goldPersistDraft(caseData);
  goldQueueEvent("SEED_SWAPPED", caseData, {seed_confirmation: goldDraft(caseData).seed_confirmation}, null, null);
  goldRenderSeed(caseData);
}

function goldRejectSeed() {
  const caseData = goldCase();
  const reason = $("goldSeedRejectionReason").value;
  const note = String($("goldSeedRejectionNote").value || "").trim();
  if (!reason || (reason === "OTHER" && !note)) {
    $("goldSeedMessage").textContent = "Choose a structured rejection reason; Other requires a short note.";
    return;
  }
  goldPushHistory(caseData);
  goldDraft(caseData).seed_confirmation = {
    status: "REJECTED",
    seed_action: "REJECT_SEQUENCE",
    seed_rejection_reason: reason,
    note: note || null,
    source_frame_sequence: Number(goldSeedRecord(caseData)?.frame_sequence),
    A: null,
    B: null,
  };
  goldPersistDraft(caseData);
  goldQueueEvent("SEED_REJECTED", caseData, {seed_confirmation: goldDraft(caseData).seed_confirmation}, null, null);
  $("goldSeedStatus").textContent = "Sequence rejected; no frame labels will be collected.";
  $("goldSeedMessage").textContent = "Save this sequence to record the structured rejection.";
  $("goldSeedConfirm").disabled = true;
  $("goldSeedSaveRejected").disabled = false;
}

function goldSeedBeginManualDraw(event) {
  if (!goldSeedMode?.startsWith("MANUAL_")) return;
  goldSeedDrawingStart = goldPoint(event, $("goldSeedSvg"));
}

function goldSeedFinishManualDraw(event) {
  if (!goldSeedDrawingStart) return;
  const caseData = goldCase();
  const record = goldSeedRecord(caseData);
  const end = goldPoint(event, $("goldSeedSvg"));
  const crop = {
    x1: Math.min(goldSeedDrawingStart.x, end.x), y1: Math.min(goldSeedDrawingStart.y, end.y),
    x2: Math.max(goldSeedDrawingStart.x, end.x), y2: Math.max(goldSeedDrawingStart.y, end.y),
  };
  goldSeedDrawingStart = null;
  if (crop.x2 - crop.x1 < 3 || crop.y2 - crop.y1 < 3) return;
  const target = goldSeedMode === "MANUAL_B" ? "B" : "A";
  const next = goldDraft(caseData).seed_confirmation || {status: "EDITING", A: null, B: null};
  next[target] = {
    state: "OBSERVED_MANUAL_BBOX",
    bbox_original_pixels: {
      x1: crop.x1 + Number(record.roi.x1), y1: crop.y1 + Number(record.roi.y1),
      x2: crop.x2 + Number(record.roi.x1), y2: crop.y2 + Number(record.roi.y1),
    },
    manual_correction_reason: "VISIBLE_PERSON_WITHOUT_USABLE_OBSERVATION_BANK_ROW",
  };
  next.status = "EDITING";
  goldDraft(caseData).seed_confirmation = next;
  goldSeedMode = null;
  goldPersistDraft(caseData);
  goldRenderSeed(caseData);
}

function goldRenderMetrics(caseData) {
  const draft = goldDraft(caseData);
  const total = goldRecords(caseData).length * 2;
  const annotated = Object.values(draft.annotations).reduce((count, value) => count + Number(Boolean(value.A)) + Number(Boolean(value.B)), 0);
  $("goldRunAccepted").textContent = `${Math.round(100 * draft.accepted_in_runs / Math.max(1, total))}%`;
  $("goldManualCount").textContent = String(draft.manual_bbox_count);
  $("goldClickCount").textContent = String(draft.clicks);
  $("goldSaveSequence").disabled = annotated !== total || goldEvidenceBlocked || goldPolygonGateBlocked || !goldPolygonApproved() || !goldSeedConfirmed(caseData);
  $("goldFrameProgress").textContent = `${annotated} of ${total} strand-frames annotated`;
  $("goldProgressBar").style.width = `${100 * annotated / Math.max(1, total)}%`;
  goldUpdateCompletionGate();
}

function goldApplyViewport(kind) {
  const viewport = goldViewport[kind];
  const wrap = $(kind === "pitch" ? "goldPitchCanvasWrap" : "goldCurrentCanvasWrap");
  if (!wrap) return;
  wrap.style.transform = `translate(${viewport.panX}px, ${viewport.panY}px) scale(${viewport.zoom})`;
  wrap.style.transformOrigin = "0 0";
  wrap.dataset.zoom = String(viewport.zoom);
}

function goldFitViewport(kind) {
  goldViewport[kind] = {zoom: 1, panX: 0, panY: 0};
  goldApplyViewport(kind);
}

function goldZoomViewport(kind, factor, clientX = null, clientY = null) {
  const wrap = $(kind === "pitch" ? "goldPitchCanvasWrap" : "goldCurrentCanvasWrap");
  if (!wrap) return;
  const viewport = goldViewport[kind];
  const rect = wrap.getBoundingClientRect();
  const focalX = clientX == null ? rect.width / 2 : clientX - rect.left;
  const focalY = clientY == null ? rect.height / 2 : clientY - rect.top;
  const oldZoom = viewport.zoom;
  const nextZoom = Math.max(1, Math.min(5, oldZoom * factor));
  const imageX = (focalX - viewport.panX) / oldZoom;
  const imageY = (focalY - viewport.panY) / oldZoom;
  viewport.zoom = nextZoom;
  viewport.panX = focalX - imageX * nextZoom;
  viewport.panY = focalY - imageY * nextZoom;
  goldApplyViewport(kind);
}

function goldBindViewport(kind, wrapId) {
  const wrap = $(wrapId);
  if (!wrap || wrap.dataset.viewportBound === "true") return;
  wrap.dataset.viewportBound = "true";
  wrap.addEventListener("wheel", (event) => {
    event.preventDefault();
    goldZoomViewport(kind, event.deltaY < 0 ? 1.12 : 0.89, event.clientX, event.clientY);
  }, {passive: false});
  wrap.addEventListener("pointerdown", (event) => {
    if (event.target.closest("svg circle, svg rect, button")) return;
    goldPointerPan = {kind, pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, panX: goldViewport[kind].panX, panY: goldViewport[kind].panY};
    wrap.setPointerCapture(event.pointerId);
  });
  wrap.addEventListener("pointermove", (event) => {
    if (!goldPointerPan || goldPointerPan.pointerId !== event.pointerId || goldPointerPan.kind !== kind) return;
    goldViewport[kind].panX = goldPointerPan.panX + event.clientX - goldPointerPan.startX;
    goldViewport[kind].panY = goldPointerPan.panY + event.clientY - goldPointerPan.startY;
    goldApplyViewport(kind);
  });
  wrap.addEventListener("pointerup", () => { goldPointerPan = null; });
  wrap.addEventListener("dblclick", (event) => goldZoomViewport(kind, 1.5, event.clientX, event.clientY));
}

async function goldLoadContextImage(target, caseData, index) {
  const record = goldRecord(caseData, index);
  if (!record) { target.removeAttribute("src"); target.dataset.evidenceReady = "true"; return target; }
  const asset = caseData.evidence_assets.find((item) => item.asset_id === record.base_asset_id);
  await goldAuditAndLoadImage(caseData, asset, target);
  return target;
}

async function goldRenderFrame() {
  const caseData = goldCase();
  if (!caseData || caseData.task_type !== "gold_strand_frame_annotation") return;
  const records = goldRecords(caseData);
  goldFrameIndex = Math.max(0, Math.min(goldFrameIndex, records.length - 1));
  const record = records[goldFrameIndex];
  $("goldCurrentCanvasWrap").style.aspectRatio = `${Number(record.crop_width)} / ${Number(record.crop_height)}`;
  const generation = ++goldEvidenceGeneration;
  goldClearEvidenceBlocker();
  try {
    await Promise.all([
      goldLoadContextImage($("goldPreviousImage"), caseData, goldFrameIndex - 1),
      goldLoadContextImage($("goldCurrentImage"), caseData, goldFrameIndex),
      goldLoadContextImage($("goldNextImage"), caseData, goldFrameIndex + 1),
    ]);
    if (generation !== goldEvidenceGeneration) return;
    goldRenderDetections(caseData, record);
    goldApplyViewport("frame");
  } catch (error) {
    if (generation === goldEvidenceGeneration) goldSetEvidenceBlocker(error.message);
  }
  $("goldTimeline").max = String(records.length - 1);
  $("goldTimeline").value = String(goldFrameIndex);
  $("goldFrameNumber").textContent = `Frame ${record.frame_sequence}`;
  $("goldTimestamp").textContent = `${Number(record.timestamp_seconds).toFixed(1)} seconds`;
  const annotation = goldAnnotation(caseData, record.frame_sequence);
  $("goldAState").textContent = goldStateLabel(annotation.A);
  $("goldBState").textContent = goldStateLabel(annotation.B);
  goldRenderMetrics(caseData);
  if (goldPolygonApproved()) goldClearPolygonGate();
  else goldSetPolygonGate("Approve the revised pitch polygon before annotating frames.");
}

function goldPoint(event, svg) {
  const rectangle = svg.getBoundingClientRect();
  const view = svg.viewBox.baseVal;
  return {
    x: view.x + (event.clientX - rectangle.left) * view.width / rectangle.width,
    y: view.y + (event.clientY - rectangle.top) * view.height / rectangle.height,
  };
}

function goldPolygonIsValid() {
  const metadata = manifest?.cases?.find((item) => item.task_type === "pitch_polygon_approval")?.visible_metadata || {};
  if (goldPitchVertices.length < 4) return false;
  if (!goldPitchVertices.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y)
    && point.x >= 0 && point.y >= 0 && point.x <= Number(metadata.image_width) && point.y <= Number(metadata.image_height))) return false;
  for (let index = 0; index < goldPitchVertices.length; index += 1) {
    const next = goldPitchVertices[(index + 1) % goldPitchVertices.length];
    if (Math.hypot(goldPitchVertices[index].x - next.x, goldPitchVertices[index].y - next.y) < 0.001) return false;
  }
  const area = Math.abs(goldPitchVertices.reduce((sum, point, index) => {
    const next = goldPitchVertices[(index + 1) % goldPitchVertices.length];
    return sum + point.x * next.y - next.x * point.y;
  }, 0) / 2);
  return area >= Number(metadata.image_width) * Number(metadata.image_height) * 0.001;
}

function goldHasUnresolvedDrafts() {
  const frameDrafts = Object.values(goldDrafts).some((draft) => draft.dirty === true);
  const polygonDraft = Boolean(goldPolygonSidecar?.draft?.status === "DRAFT" && !goldPolygonApproved());
  return frameDrafts || polygonDraft;
}

function goldUpdateCompletionGate() {
  const eligibility = state?.completion_eligibility || {};
  const serverEligible = eligibility.eligible === true;
  const finalized = Number(eligibility.sequences_finalized || 0);
  const expected = Number(eligibility.expected_sequence_count || 0);
  const confirmed = Number(eligibility.confirmed_sequences || 0);
  const confirmedComplete = Number(eligibility.confirmed_sequences_complete || 0);
  const rejected = Number(eligibility.rejected_sequences || 0);
  const rejectedComplete = Number(eligibility.rejected_sequences_complete || 0);
  const persistedFrameStates = Number(eligibility.persisted_strand_frame_states || 0);
  const requiredFrameStates = Number(eligibility.required_strand_frame_states || 0);
  const unresolvedDrafts = goldHasUnresolvedDrafts();
  const complete = $("goldComplete");
  const clientReady = goldPersistence.pending.length === 0 && !goldPersistence.blocked
    && !goldEvidenceBlocked && !unresolvedDrafts && !document.querySelector(".goldInvalidBBox");
  if (complete) {
    complete.textContent = state?.completed === true ? "Review finalized" : "Finalize review";
    complete.disabled = state?.completed === true || !(serverEligible && clientReady);
  }
  const checklist = $("goldCompletionChecklist");
  if (checklist) checklist.textContent = `Confirmed sequences: ${confirmedComplete}/${confirmed} | Rejected sequences: ${rejectedComplete}/${rejected} | Finalized sequences: ${finalized}/${expected} | Required frame states: ${requiredFrameStates} | Persisted frame states: ${persistedFrameStates} | Pending events: ${goldPersistence.pending.length} | Evidence ${goldEvidenceBlocked ? "blocked" : "clear"} | Draft ${unresolvedDrafts ? "unsaved" : "clear"}`;
}

function goldRenderPitch() {
  const caseData = goldCase();
  const metadata = caseData.visible_metadata || {};
  const svg = $("goldPitchSvg");
  $("goldPitchCanvasWrap").style.aspectRatio = `${Number(metadata.image_width)} / ${Number(metadata.image_height)}`;
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${metadata.image_width} ${metadata.image_height}`);
  const points = goldPitchVertices.map((point) => `${point.x},${point.y}`).join(" ");
  const tolerance = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
  tolerance.setAttribute("points", points);
  tolerance.setAttribute("class", "goldTolerance");
  svg.appendChild(tolerance);
  const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
  polygon.setAttribute("points", points);
  polygon.setAttribute("class", "goldPolygon");
  svg.appendChild(polygon);
  for (const sample of metadata.sample_footpoints || []) {
    const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    marker.setAttribute("cx", sample.x);
    marker.setAttribute("cy", sample.y);
    marker.setAttribute("r", 7);
    marker.setAttribute("class", `goldPitchSample ${String(sample.zone || "").toLowerCase()}`);
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = String(sample.zone || "sample footpoint").replaceAll("_", " ");
    marker.appendChild(title);
    svg.appendChild(marker);
  }
  goldPitchVertices.forEach((point, index) => {
    const vertex = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    vertex.setAttribute("cx", point.x);
    vertex.setAttribute("cy", point.y);
    vertex.setAttribute("r", 9);
    vertex.setAttribute("class", "goldVertex");
    vertex.addEventListener("pointerdown", (event) => {
      goldPolygonHistory.push(JSON.parse(JSON.stringify(goldPitchVertices)));
      goldPolygonRedo = [];
      goldPitchDragIndex = index;
      vertex.setPointerCapture(event.pointerId);
    });
    vertex.addEventListener("pointermove", (event) => {
      if (goldPitchDragIndex !== index) return;
      goldPitchVertices[index] = goldPoint(event, svg);
      vertex.setAttribute("cx", goldPitchVertices[index].x);
      vertex.setAttribute("cy", goldPitchVertices[index].y);
      const updated = goldPitchVertices.map((value) => `${value.x},${value.y}`).join(" ");
      polygon.setAttribute("points", updated);
      tolerance.setAttribute("points", updated);
      $("goldPitchValidation").textContent = goldPolygonIsValid() ? "Polygon valid in original-image pixels." : "Polygon is outside the image bounds.";
      goldUpdateCompletionGate();
    });
    vertex.addEventListener("pointerup", () => {
      goldPitchDragIndex = null;
      goldSchedulePolygonDraftSave();
    });
    svg.appendChild(vertex);
  });
  $("goldPitchValidation").textContent = goldPolygonIsValid() ? "Polygon valid in original-image pixels." : "Polygon is outside the image bounds.";
  goldApplyViewport("pitch");
}

async function goldSavePitch(decision) {
  if (goldEvidenceBlocked || !goldPolygonIsValid()) {
    $("goldPitchMessage").textContent = "Approval is blocked until the evidence and polygon are valid.";
    return;
  }
  try {
    await goldSavePolygonDraft(decision === "PITCH_POLYGON_REVISION_REQUIRED" ? "needs_revision" : "before_approval");
    if (decision === "PITCH_POLYGON_APPROVED") {
      goldPolygonSidecar = await api("/api/review/polygon/approve", {method: "POST", body: JSON.stringify(goldPolygonPayload())});
      $("goldPitchMessage").textContent = `Pitch approved. Approved polygon hash: ${goldPolygonSidecar.approved_polygon_hash.slice(0, 12)}`;
      activeIndex = Math.min(manifest.cases.length - 1, activeIndex + 1);
      goldFrameIndex = 0;
    } else {
      goldPolygonSidecar = await api("/api/review/polygon/revision", {method: "POST", body: JSON.stringify(goldPolygonPayload())});
      $("goldPitchMessage").textContent = "Needs revision recorded. The revised draft remains editable and unapproved.";
    }
    goldRender();
  } catch (error) {
    $("goldPitchMessage").textContent = `Polygon save blocked: ${error.message}`;
  }
}

async function goldSaveRevisedPolygon() {
  try {
    await goldSavePolygonDraft("explicit_save");
    $("goldPitchMessage").textContent = "Revised polygon saved as a match-local draft.";
    goldRenderPitch();
  } catch (error) { $("goldPitchMessage").textContent = `Draft save failed: ${error.message}`; }
}

async function goldEditApprovedPolygon() {
  if (!goldPolygonSidecar?.approved) return;
  goldPolygonHistory.push(JSON.parse(JSON.stringify(goldPitchVertices)));
  goldPolygonRedo = [];
  goldPitchVertices = goldPolygonSidecar.approved.vertices_original_pixels.map((point) => ({x: Number(point.x), y: Number(point.y)}));
  await goldSavePolygonDraft("edit_approved_polygon");
  activeIndex = manifest.cases.findIndex((caseData) => caseData.task_type === "pitch_polygon_approval");
  $("goldPitchMessage").textContent = "Approved polygon is being edited. Re-approve it before annotation.";
  await goldRender();
}

async function goldRevokeApproval() {
  if (!window.confirm("Revoke polygon approval? Frame annotation and completion will be blocked until reapproval.")) return;
  try {
    goldPolygonSidecar = await api("/api/review/polygon/revoke", {method: "POST", body: JSON.stringify({reason: "reviewer_requested"})});
    activeIndex = 0;
    $("goldPitchMessage").textContent = "Polygon approval revoked. Reapproval is required before annotation.";
    goldRender();
  } catch (error) { $("goldPitchMessage").textContent = `Revoke failed: ${error.message}`; }
}

function goldUndoPolygon() {
  const prior = goldPolygonHistory.pop();
  if (!prior) return;
  goldPolygonRedo.push(JSON.parse(JSON.stringify(goldPitchVertices)));
  goldPitchVertices = prior;
  goldSchedulePolygonDraftSave();
  goldRenderPitch();
}

function goldRedoPolygon() {
  const next = goldPolygonRedo.pop();
  if (!next) return;
  goldPolygonHistory.push(JSON.parse(JSON.stringify(goldPitchVertices)));
  goldPitchVertices = next;
  goldSchedulePolygonDraftSave();
  goldRenderPitch();
}

async function goldRenderPitchCase(caseData) {
  $("goldPitchPanel").classList.remove("isHidden");
  $("goldAnnotationPanel").classList.add("isHidden");
  $("goldSeedPanel").classList.add("isHidden");
  $("goldAnnotationTools").classList.add("isHidden");
  $("goldTaskEyebrow").textContent = "PITCH GATE";
  $("goldCaseTitle").textContent = "Approve the playable-pitch boundary";
  $("goldQuestion").textContent = caseData.concise_question;
  $("goldSequenceProgress").textContent = "Pitch gate approval";
  $("goldFrameProgress").textContent = "Required before gold annotation";
  $("goldProgressBar").style.width = goldPolygonApproved() ? "100%" : "0%";
  const metadata = caseData.visible_metadata;
  goldPitchOriginalVertices = metadata.polygon_vertices.map((point) => ({...point}));
  const sourceVertices = goldPolygonSidecar?.is_approved
    ? goldPolygonSidecar.approved.vertices_original_pixels
    : goldPolygonSidecar?.draft?.vertices_original_pixels || goldPitchOriginalVertices;
  goldPitchVertices = sourceVertices.map((point) => ({x: Number(point.x), y: Number(point.y)}));
  $("goldPitchTolerance").value = String(goldPolygonSidecar?.draft?.tolerance_pixels ?? metadata.tolerance_pixels);
  $("goldSaveRevisedPolygon").disabled = false;
  $("goldApproveRevisedPolygon").disabled = goldPolygonApproved();
  $("goldEditApprovedPolygon").disabled = !goldPolygonApproved();
  $("goldRevokeApproval").disabled = !goldPolygonApproved();
  const generation = ++goldEvidenceGeneration;
  goldClearEvidenceBlocker();
  goldClearPolygonGate();
  try {
    const asset = caseData.evidence_assets.find((item) => item.asset_id === metadata.base_asset_id);
    await goldAuditAndLoadImage(caseData, asset, $("goldPitchImage"));
    if (generation !== goldEvidenceGeneration) return;
    goldRenderPitch();
  } catch (error) {
    if (generation === goldEvidenceGeneration) goldSetEvidenceBlocker(error.message);
  }
}

function goldRenderAnnotationCase(caseData) {
  $("goldPitchPanel").classList.add("isHidden");
  const rejected = goldSeedRejected(caseData);
  const confirmed = goldSeedConfirmed(caseData);
  $("goldSeedPanel").classList.toggle("isHidden", confirmed);
  $("goldAnnotationPanel").classList.toggle("isHidden", !confirmed || rejected);
  $("goldAnnotationTools").classList.toggle("isHidden", !confirmed || rejected);
  $("goldTaskEyebrow").textContent = rejected ? "SEQUENCE REJECTED" : (confirmed ? "FRAME-LEVEL A/B GOLD" : "SEED CONFIRMATION REQUIRED");
  $("goldCaseTitle").textContent = `Sequence ${activeIndex} of ${manifest.cases.length - 1}`;
  $("goldQuestion").textContent = confirmed
    ? "Confirm each temporary A/B strand frame state. Cyan A and magenta B remain visible through the sequence."
    : "Confirm the visibly labelled cyan A and magenta B pair before frame annotation begins.";
  $("goldSequenceProgress").textContent = `Sequence ${activeIndex} of ${manifest.cases.length - 1}`;
  const characteristics = caseData.visible_metadata?.challenge_characteristics || [];
  $("goldChallengeCharacteristics").textContent = characteristics.length
    ? `Challenge characteristics: ${characteristics.map((value) => String(value).replaceAll("_", " ")).join("; ")}.`
    : "";
  const allowedStates = new Set(uiConfig?.question_contract?.annotation_states || []);
  document.querySelectorAll("[data-gold-state]").forEach((button) => {
    button.hidden = allowedStates.size > 0 && !allowedStates.has(button.dataset.goldState);
  });
  $("goldNote").value = goldDraft(caseData).note || "";
  if (confirmed) goldRenderFrame();
  else goldRenderSeed(caseData);
  if (!goldPolygonApproved()) goldSetPolygonGate("Approve the revised pitch polygon before annotating frames.");
  else if (confirmed) goldClearPolygonGate();
}

async function goldRender() {
  const caseData = goldCase();
  if (!caseData) return;
  $("goldTitle").textContent = uiConfig.review_title || "Gold strand annotation";
  goldUpdateCompletionGate();
  if (caseData.task_type === "pitch_polygon_approval") await goldRenderPitchCase(caseData);
  else goldRenderAnnotationCase(caseData);
}

async function goldSaveSequence() {
  if (goldEvidenceBlocked || goldPolygonGateBlocked || !goldPolygonApproved()) return;
  const caseData = goldCase();
  const draft = goldDraft(caseData);
  if (!goldSeedConfirmed(caseData) && !goldSeedRejected(caseData)) {
    $("goldError").textContent = "Confirm or reject the visible A/B seed pair before saving this sequence.";
    $("goldError").classList.remove("isHidden");
    return;
  }
  const seedConfirmation = draft.seed_confirmation;
  const isRejected = goldSeedRejected(caseData);
  const records = goldRecords(caseData);
  const frameAnnotations = records.map((record) => ({
    frame_sequence: Number(record.frame_sequence),
    A: draft.annotations[String(record.frame_sequence)]?.A,
    B: draft.annotations[String(record.frame_sequence)]?.B,
  }));
  if (!isRejected && frameAnnotations.some((item) => !item.A || !item.B)) {
    $("goldError").textContent = "Annotate both strands on every frame before saving.";
    $("goldError").classList.remove("isHidden");
    return;
  }
  try {
    await goldFlushOutbox();
    if (goldPersistence.pending.length) throw new Error("save is blocked until pending events are acknowledged");
    await goldQueueEventAndFlush("SEQUENCE_SAVED", caseData, {
      decision: isRejected ? "SEQUENCE_REJECTED" : "SEQUENCE_ANNOTATED",
      frame_annotations: isRejected ? [] : frameAnnotations,
      seed_confirmation: seedConfirmation,
      seed_state_hash: JSON.stringify(seedConfirmation),
      approved_polygon_hash: goldPolygonSidecar?.approved_polygon_hash,
      approved_polygon_manifest_hash: goldPolygonSidecar?.approved_polygon_manifest_hash,
      note: String($("goldNote").value || "").trim() || null,
      interaction_metrics: {clicks: draft.clicks, accepted_in_runs: draft.accepted_in_runs, manual_bbox_count: draft.manual_bbox_count, active_seconds: activeTimeNow()},
    });
    state = await api("/api/review/state");
    goldHydrateFromServer();
    localStorage.removeItem(goldDraftKey(caseData));
    if (activeIndex < manifest.cases.length - 1) {
      activeIndex += 1;
      goldFrameIndex = 0;
    }
    goldPersistenceStatus("Saved to server");
    goldRender();
  } catch (error) {
    $("goldError").textContent = `Save failed: ${error.message}`;
    $("goldError").classList.remove("isHidden");
  }
}

function goldBeginManualDraw(event) {
  if (!$("goldDrawManual").classList.contains("drawing")) return;
  goldDrawingStart = goldPoint(event, $("goldDetectionSvg"));
}

function goldFinishManualDraw(event) {
  if (!goldDrawingStart) return;
  const caseData = goldCase();
  const record = goldRecord(caseData);
  const end = goldPoint(event, $("goldDetectionSvg"));
  const crop = {
    x1: Math.min(goldDrawingStart.x, end.x), y1: Math.min(goldDrawingStart.y, end.y),
    x2: Math.max(goldDrawingStart.x, end.x), y2: Math.max(goldDrawingStart.y, end.y),
  };
  goldDrawingStart = null;
  $("goldDrawManual").classList.remove("drawing");
  if (crop.x2 - crop.x1 < 3 || crop.y2 - crop.y1 < 3) return;
  goldSetState(goldActiveStrand, {
    state: "OBSERVED_MANUAL_BBOX",
    bbox_original_pixels: {
      x1: crop.x1 + Number(record.roi.x1), y1: crop.y1 + Number(record.roi.y1),
      x2: crop.x2 + Number(record.roi.x1), y2: crop.y2 + Number(record.roi.y1),
    },
    manual_correction_reason: "VISIBLE_PERSON_WITHOUT_USABLE_OBSERVATION_BANK_ROW",
  });
  goldDraft(caseData).manual_bbox_count += 1;
  goldPersistDraft(caseData);
}

function goldBind() {
  goldBindViewport("pitch", "goldPitchCanvasWrap");
  goldBindViewport("frame", "goldCurrentCanvasWrap");
  document.querySelectorAll("[data-gold-zoom]").forEach((button) => button.addEventListener("click", () => {
    const kind = button.dataset.goldZoom === "pitch" ? "pitch" : "frame";
    const action = button.dataset.goldAction;
    if (action === "fit" || action === "fit-polygon") goldFitViewport(kind);
    else goldZoomViewport(kind, action === "out" ? 0.8 : 1.25);
  }));
  $("goldUndo").addEventListener("click", () => goldCase()?.task_type === "pitch_polygon_approval" ? goldUndoPolygon() : goldUndo());
  $("goldRedo").addEventListener("click", goldRedoPolygon);
  $("goldResetPolygon").addEventListener("click", () => {
    if (!window.confirm("Reset the polygon to the immutable proposal?")) return;
    goldPolygonHistory.push(JSON.parse(JSON.stringify(goldPitchVertices)));
    goldPolygonRedo = [];
    goldPitchVertices = goldPitchOriginalVertices.map((point) => ({...point}));
    goldSchedulePolygonDraftSave();
    goldRenderPitch();
    goldFitViewport("pitch");
  });
  $("goldSaveRevisedPolygon").addEventListener("click", goldSaveRevisedPolygon);
  $("goldApproveRevisedPolygon").addEventListener("click", () => goldSavePitch("PITCH_POLYGON_APPROVED"));
  $("goldEditApprovedPolygon").addEventListener("click", goldEditApprovedPolygon);
  $("goldRevokeApproval").addEventListener("click", goldRevokeApproval);
  $("goldApprovePolygon").addEventListener("click", () => goldSavePitch("PITCH_POLYGON_APPROVED"));
  $("goldRequestRevision").addEventListener("click", () => goldSavePitch("PITCH_POLYGON_REVISION_REQUIRED"));
  $("goldStrandA").addEventListener("click", () => goldSetActiveStrand("A"));
  $("goldStrandB").addEventListener("click", () => goldSetActiveStrand("B"));
  $("goldPreviousFrame").addEventListener("click", () => { goldFrameIndex = Math.max(0, goldFrameIndex - 1); goldRenderFrame(); });
  $("goldNextFrame").addEventListener("click", () => { goldFrameIndex = Math.min(goldRecords().length - 1, goldFrameIndex + 1); goldRenderFrame(); });
  $("goldTimeline").addEventListener("input", (event) => { goldFrameIndex = Number(event.target.value); goldRenderFrame(); });
  $("goldAcceptFrame").addEventListener("click", () => goldAcceptProposal(false));
  $("goldAcceptRun").addEventListener("click", () => goldAcceptProposal(true));
  $("goldRunConfirm").addEventListener("click", () => {
    $("goldRunDialog").close();
    goldApplyProposal(true);
  });
  $("goldNextUnannotated").addEventListener("click", () => goldFindNextFrame((record) => {
    const value = goldAnnotation(goldCase(), record.frame_sequence);
    return !value.A || !value.B;
  }));
  $("goldNextUncertain").addEventListener("click", () => goldFindNextFrame((record) => Boolean(record.machine_uncertain)));
  $("goldNextDistractor").addEventListener("click", () => goldFindNextFrame((record) => Boolean(record.high_distractor)));
  $("goldSaveSequence").addEventListener("click", goldSaveSequence);
  $("goldSeedConfirm").addEventListener("click", goldConfirmSeed);
  $("goldSeedSwap").addEventListener("click", goldSwapSeed);
  $("goldSeedCorrectA").addEventListener("click", () => goldStartSeedCorrection("A"));
  $("goldSeedCorrectB").addEventListener("click", () => goldStartSeedCorrection("B"));
  $("goldSeedCorrectBoth").addEventListener("click", () => goldStartSeedCorrection("BOTH"));
  $("goldSeedDrawManual").addEventListener("click", () => goldStartSeedCorrection(`MANUAL_${goldSeedMode === "B" ? "B" : "A"}`));
  $("goldSeedReject").addEventListener("click", goldRejectSeed);
  $("goldSeedSaveRejected").addEventListener("click", goldSaveSequence);
  $("goldSeedSvg").addEventListener("pointerdown", goldSeedBeginManualDraw);
  $("goldSeedSvg").addEventListener("pointerup", goldSeedFinishManualDraw);
  $("goldNote").addEventListener("input", () => {
    const caseData = goldCase();
    goldDraft(caseData).note = $("goldNote").value;
    goldPersistDraft(caseData);
    goldQueueEvent("NOTE_UPDATED", caseData, {note: goldDraft(caseData).note});
  });
  $("goldDrawManual").addEventListener("click", () => $("goldDrawManual").classList.toggle("drawing"));
  $("goldDetectionSvg").addEventListener("pointerdown", goldBeginManualDraw);
  $("goldDetectionSvg").addEventListener("pointerup", goldFinishManualDraw);
  document.querySelectorAll("[data-gold-state]").forEach((button) => button.addEventListener("click", () => goldSetState(goldActiveStrand, {state: button.dataset.goldState})));
  $("goldComplete").addEventListener("click", async () => {
    if ($("goldComplete").disabled) return;
    try {
      await goldFlushOutbox();
      if (goldPersistence.pending.length) throw new Error("completion is blocked while events are pending");
      const completionAck = await api("/api/review/gold-complete", {method: "POST", body: JSON.stringify(goldEventBase(null, "REVIEW_COMPLETED", {
        elapsed_active_seconds: activeTimeNow(),
        pending_outbox_events: goldPersistence.pending.length,
        evidence_blocker_count: goldEvidenceBlocked ? 1 : 0,
        unresolved_draft_count: goldHasUnresolvedDrafts() ? 1 : 0,
        unresolved_divergence: goldPersistence.blocked,
      }))});
      state = completionAck.state || state;
      goldPersistence.serverSequence = Number(completionAck.server_event_sequence || state?.server_sequence || goldPersistence.serverSequence);
      goldPersistence.serverStateHash = completionAck.server_state_hash || state?.server_state_hash || goldPersistence.serverStateHash;
      goldPersistenceStatus("Saved to server");
      goldUpdateCompletionGate();
    } catch (error) {
      $("goldError").textContent = `Completion blocked: ${error.message}`;
      $("goldError").classList.remove("isHidden");
    }
  });
  document.addEventListener("keydown", (event) => {
    if (!goldMode || isTyping()) return;
    if (event.ctrlKey && event.key.toLowerCase() === "z") { event.preventDefault(); goldUndo(); return; }
    if (goldCase()?.task_type !== "gold_strand_frame_annotation") return;
    if (!goldSeedConfirmed(goldCase())) {
      if (event.key === " ") { event.preventDefault(); goldConfirmSeed(); }
      else if (event.key.toLowerCase() === "s") { event.preventDefault(); goldSwapSeed(); }
      else if (event.key.toLowerCase() === "a") { event.preventDefault(); goldStartSeedCorrection("A"); }
      else if (event.key.toLowerCase() === "b") { event.preventDefault(); goldStartSeedCorrection("B"); }
      else if (event.key.toLowerCase() === "x") { event.preventDefault(); goldRejectSeed(); }
      return;
    }
    if (event.key === " ") { event.preventDefault(); goldAcceptProposal(false); }
    else if (event.key === "Enter") { event.preventDefault(); goldAcceptProposal(true); }
    else if (event.key.toLowerCase() === "a") goldSetActiveStrand("A");
    else if (event.key.toLowerCase() === "b") goldSetActiveStrand("B");
    else if (event.key === "1") goldSetState("A", {state: "VISIBLE_NO_VALID_DETECTION"});
    else if (event.key === "2") goldSetState("B", {state: "VISIBLE_NO_VALID_DETECTION"});
    else if (event.key.toLowerCase() === "u") goldSetState(goldActiveStrand, {state: "AMBIGUOUS"});
    else if (event.key.toLowerCase() === "n") goldSetState(goldActiveStrand, {state: "NOT_VISIBLE_IN_PANORAMA"});
    else if (event.key.toLowerCase() === "o") goldSetState(goldActiveStrand, {state: "OUTSIDE_DYNAMIC_VIEW_BUT_VISIBLE_IN_PANORAMA"});
    else if (event.key === "ArrowLeft" && event.shiftKey) { event.preventDefault(); goldShiftSequence(-1); }
    else if (event.key === "ArrowRight" && event.shiftKey) { event.preventDefault(); goldShiftSequence(1); }
    else if (event.key === "ArrowLeft") { event.preventDefault(); goldFrameIndex = Math.max(0, goldFrameIndex - 1); goldRenderFrame(); }
    else if (event.key === "ArrowRight") { event.preventDefault(); goldFrameIndex = Math.min(goldRecords().length - 1, goldFrameIndex + 1); goldRenderFrame(); }
  });
}

async function load() {
  manifest = await api("/api/review/manifest");
  uiConfig = await api("/api/review/ui-config");
  state = await api("/api/review/state");
  if (uiConfig.presentation_mode === "detection_gold_pilot") {
    document.body.dataset.presentation = uiConfig.presentation_mode;
    document.body.classList.add("detectionGoldPresentation");
    $("legacyShell").classList.add("isHidden");
    $("premiumShell").classList.add("isHidden");
    $("goldShell").classList.add("isHidden");
    $("detectionGoldShell").classList.remove("isHidden");
    $("detectionGoldShell").setAttribute("aria-hidden", "false");
    await window.DetectionGoldPilot.mount({manifest, uiConfig, state, api});
    return;
  }
  goldMode = uiConfig.presentation_mode === "gold_strand_annotation";
  if (goldMode) {
    await goldInitPersistence();
    goldHydrateFromServer();
    await goldLoadPolygonSidecar();
    let resume = state.resume_case_id;
    const pitchIndex = manifest.cases.findIndex((caseData) => caseData.task_type === "pitch_polygon_approval");
    if (goldPolygonApproved() && resume === manifest.cases[pitchIndex]?.case_id) {
      resume = manifest.cases.find((caseData) => caseData.task_type === "gold_strand_frame_annotation")?.case_id;
    }
    activeIndex = Math.max(0, manifest.cases.findIndex((caseData) => caseData.case_id === resume));
    document.body.dataset.presentation = uiConfig.presentation_mode;
    document.body.classList.add("goldPresentation");
    $("legacyShell").classList.add("isHidden");
    $("goldShell").classList.remove("isHidden");
    $("goldShell").setAttribute("aria-hidden", "false");
    goldBind();
    goldRender();
    return;
  }
  premiumMode = ["simplified_temporal", "local_encounter_strands", "stable_local_strand_continuity", "development_error_atlas"].includes(uiConfig.presentation_mode);
  if (premiumMode) {
    const alternativeEnabled = uiConfig.question_contract?.alternative_hypothesis_toggle_enabled === true;
    $("premiumAlternativeToggleWrap").classList.toggle("isHidden", !alternativeEnabled);
    $("premiumAlternativeToggle").checked = false;
    document.body.dataset.presentation = uiConfig.presentation_mode;
    $("legacyShell").classList.add("isHidden");
    $("premiumShell").classList.remove("isHidden");
    $("premiumShell").setAttribute("aria-hidden", "false");
    premiumBind();
    premiumRender();
    return;
  }
  const resume = state.resume_case_id;
  activeIndex = Math.max(0, manifest.cases.findIndex((caseData) => caseData.case_id === resume));
  render();
}

load().catch((error) => setStatus(`Load failed: ${error.message}`, true));
