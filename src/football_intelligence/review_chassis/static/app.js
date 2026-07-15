let manifest = null;
let uiConfig = null;
let state = null;
let activeIndex = 0;
let elapsedSeconds = 0;
let timerStarted = Date.now();
let activeAnnotationEditor = null;
const frameStepper = {};

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
  for (const asset of caseData.evidence_assets.filter((item) => item.asset_type === "image_sequence" && assetVisible(caseData, item))) {
    const key = asset.group_id || "default";
    groups[key] = groups[key] || [];
    groups[key].push(asset);
  }
  for (const assets of Object.values(groups)) {
    assets.sort((a, b) => (a.frame_sequences[0] || 0) - (b.frame_sequences[0] || 0));
  }
  return groups;
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
    .filter((asset) => !(uiConfig.spatial_annotation_enabled && asset.metadata?.primary_annotation_image === true))
    .sort(assetSort)
    .map((asset) => renderAsset(caseData, asset));
  const hiddenSequenceControls = caseData.evidence_assets
    .filter((asset) => asset.asset_type === "image_sequence" && !assetVisible(caseData, asset))
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
    const asset = caseData.evidence_assets.find((item) => item.metadata?.primary_annotation_image === true)
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
      <div id="interactiveAnnotationRoot"></div>`;
    activeAnnotationEditor = new window.ReviewAnnotationCanvas.SpatialAnnotationCanvas(
      $("interactiveAnnotationRoot"),
      {
        caseData,
        asset,
        imageUrl: evidenceUrl(caseData.case_id, asset.relative_path),
        candidates: caseData.visible_metadata?.safe_anonymous_candidates || caseData.competing_candidates || [],
        noteElement: $("note"),
        onChange: () => {},
      },
    );
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
    const errors = window.ReviewAnnotationCanvas.validateDecision(decision, $("note").value);
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

async function load() {
  manifest = await api("/api/review/manifest");
  uiConfig = await api("/api/review/ui-config");
  state = await api("/api/review/state");
  const resume = state.resume_case_id;
  activeIndex = Math.max(0, manifest.cases.findIndex((caseData) => caseData.case_id === resume));
  render();
}

load().catch((error) => setStatus(`Load failed: ${error.message}`, true));
