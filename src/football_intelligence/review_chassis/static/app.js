let manifest = null;
let uiConfig = null;
let state = null;
let activeIndex = 0;
let elapsedSeconds = 0;
let timerStarted = Date.now();
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

const $ = (id) => document.getElementById(id);
const isTyping = () => ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {"Content-Type": "application/json"},
    ...options,
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
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
    elapsed_active_seconds: elapsedSeconds + Math.floor((Date.now() - timerStarted) / 1000),
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
    state = await api("/api/review/complete", {method: "POST", body: JSON.stringify({elapsed_active_seconds: elapsedSeconds})});
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
  if (premiumMode) return;
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

function premiumCase() {
  return manifest.cases[activeIndex];
}

function premiumDraftKey(caseData) {
  const prefix = uiConfig?.presentation_mode === "stable_local_strand_continuity" ? "m5_5f0_draft" : "m5_5e2_draft";
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
  if (fields[3]) { fields[3].dataset.question = "seed_correction"; fields[3].innerHTML = `<legend><span class="questionNumber">4</span> Optional seed correction detail.</legend><label>Correction note<input id="premiumSeedCorrection" type="text" maxlength="240" autocomplete="off" placeholder="Required only for Correct A or Correct B"></label>`; }
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
  }
  premiumRenderSuggestion(caseData);
}

function premiumRenderSuggestion(caseData) {
  const draft = premiumGetDraft(caseData);
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
  if (["local_encounter_strands", "stable_local_strand_continuity"].includes(uiConfig?.presentation_mode)) {
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
  const labels = $("premiumLabelsToggle").checked;
  const locator = $("premiumLocatorToggle").checked && premiumView === "panorama";
  $("premiumObservedLayer").classList.toggle("isHidden", !$("premiumObservedToggle").checked);
  $("premiumAllDetectionsLayer").classList.toggle("isHidden", !$("premiumAllDetectionsToggle").checked);
  $("premiumPredictedLayer").classList.toggle("isHidden", !predicted);
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
  const enabled = ["base", "observed", "all_detections", "predicted", "labels", "locator"].filter((layer) => {
    if (layer === "predicted") return $("premiumPredictedToggle").checked;
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
  const targets = {base: $("premiumBaseLayer"), observed: $("premiumObservedLayer"), all_detections: $("premiumAllDetectionsLayer"), predicted: $("premiumPredictedLayer"), labels: $("premiumLabelsLayer"), locator: $("premiumLocatorLayer")};
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
  premiumFrameIndex(caseData);
  const reviewed = Object.keys(state?.decisions || {}).length;
  $("premiumReviewTitle").textContent = uiConfig.review_title || (["local_encounter_strands", "stable_local_strand_continuity"].includes(uiConfig.presentation_mode) ? "Local strand continuity review" : "Simplified temporal review");
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
  if (uiConfig?.presentation_mode === "stable_local_strand_continuity") {
    const seed = draft.answers?.seed_action;
    const outcome = draft.answers?.continuity_outcome;
    const errors = [];
    if (!seed) errors.push("Confirm or correct the proposed seeds.");
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
  if (uiConfig?.presentation_mode === "stable_local_strand_continuity") {
    draft.note = $("premiumNote").value;
    draft.first_failure_frame = $("premiumFirstFailureFrame")?.value || "";
    draft.seed_correction = $("premiumSeedCorrection")?.value || "";
    const errors = premiumValidateDraft(caseData);
    if (errors.length) { $("premiumError").textContent = errors.join(" "); $("premiumError").classList.remove("isHidden"); return; }
    $("premiumError").classList.add("isHidden");
    const outcome = draft.answers.continuity_outcome;
    const structuredReview = {seed_action: draft.answers.seed_action, continuity_outcome: outcome, first_failure_frame: draft.first_failure_frame || null, seed_correction: draft.seed_correction || null, auto_generated_summary: stableAutoSummary(draft), note: String(draft.note || "").trim() || null};
    premiumSetStatus("Saving", "saving");
    $("premiumSaveNext").disabled = true;
    try {
      state = await api("/api/review/decision", {method: "POST", body: JSON.stringify({case_id: caseData.case_id, decision: outcome, note: String(draft.note || "").trim(), structured_review: structuredReview, input_source: "save_and_next", last_viewed_case_id: caseData.case_id})});
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
    state = await api("/api/review/decision", {method: "POST", body: JSON.stringify({case_id: caseData.case_id, decision: canonical, note: draft.note.trim(), structured_review: structuredReview, input_source: "save_and_next", last_viewed_case_id: caseData.case_id})});
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
  ["premiumObservedToggle", "premiumAllDetectionsToggle", "premiumPredictedToggle", "premiumLabelsToggle", "premiumLocatorToggle"].forEach((id) => $(id).addEventListener("change", () => { premiumSetLayerVisibility(); premiumLoadFrame(premiumCase()).catch(() => {}); }));
  $("premiumReviewForm").addEventListener("change", () => { const draft = premiumGetDraft(premiumCase()); premiumCollectAnswers(premiumCase()); premiumRenderSuggestion(premiumCase()); premiumSaveDraft(premiumCase()); });
  $("premiumReviewForm").addEventListener("input", () => { const draft = premiumGetDraft(premiumCase()); draft.note = $("premiumNote").value; draft.overrideReason = $("premiumOverrideReason").value; draft.annotation = {start_frame: $("premiumAnnotationStart").value || null, end_frame: $("premiumAnnotationEnd").value || null, merge_region: $("premiumMergeRegion").value || null}; if (uiConfig?.presentation_mode === "stable_local_strand_continuity") { draft.first_failure_frame = $("premiumFirstFailureFrame")?.value || ""; draft.seed_correction = $("premiumSeedCorrection")?.value || ""; } premiumSaveDraft(premiumCase()); });
  $("premiumConclusion").addEventListener("change", () => { premiumGetDraft(premiumCase()).conclusion = $("premiumConclusion").value; premiumGetDraft(premiumCase()).confirmed = false; premiumRenderSuggestion(premiumCase()); premiumSaveDraft(premiumCase()); });
  $("premiumSubtype").addEventListener("change", () => { premiumGetDraft(premiumCase()).subtype = $("premiumSubtype").value; premiumGetDraft(premiumCase()).confirmed = false; premiumSaveDraft(premiumCase()); });
  $("premiumConfirm").addEventListener("change", () => { premiumGetDraft(premiumCase()).confirmed = $("premiumConfirm").checked; premiumSaveDraft(premiumCase()); });
  $("premiumReviewForm").addEventListener("submit", premiumSaveAndNext);
  $("premiumComplete").addEventListener("click", async () => { try { state = await api("/api/review/complete", {method: "POST", body: JSON.stringify({})}); premiumSetStatus("Completed", "saved"); } catch (error) { $("premiumError").textContent = `Completion blocked: ${error.message}`; $("premiumError").classList.remove("isHidden"); } });
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

async function load() {
  manifest = await api("/api/review/manifest");
  uiConfig = await api("/api/review/ui-config");
  state = await api("/api/review/state");
  premiumMode = ["simplified_temporal", "local_encounter_strands", "stable_local_strand_continuity"].includes(uiConfig.presentation_mode);
  if (premiumMode) {
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
