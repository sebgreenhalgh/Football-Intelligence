# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_intelligence.review.schemas import WORKBENCH_VERSION

FALLBACK_WARNING = "Browser-only recovery mode - decisions are not yet durably saved to the project."

KEYBOARD_SHORTCUTS = {
    "A": "accept_continuity",
    "R": "reject_continuity",
    "U": "unresolved",
    "ArrowLeft": "previous",
    "ArrowRight": "next",
    "Space": "play_pause_temporal_evidence",
    "Z": "toggle_zoom",
    "C": "toggle_context",
    "D": "toggle_engineering_details",
    "N": "focus_note_field",
    "Ctrl+Z": "undo_most_recent_decision_change",
}


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def build_workbench(workbench_root: Path) -> dict[str, Any]:
    workbench_root.mkdir(parents=True, exist_ok=True)
    write_text(workbench_root / "index.html", INDEX_HTML)
    write_text(workbench_root / "styles.css", STYLES_CSS)
    write_text(workbench_root / "app.js", APP_JS)
    write_text(workbench_root / "fallback.html", FALLBACK_HTML)
    manifest = {
        "schema_version": "m5_4b.workbench_static_assets.v1",
        "workbench_version": WORKBENCH_VERSION,
        "index_path": str(workbench_root / "index.html"),
        "fallback_path": str(workbench_root / "fallback.html"),
        "keyboard_shortcuts": KEYBOARD_SHORTCUTS,
        "static_fallback_warning": FALLBACK_WARNING,
        "raw_json_primary_interface": False,
        "uses_durable_project_autosave_server": True,
        "browser_storage_is_fallback_only": True,
    }
    (workbench_root / "workbench_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>M5.4B Visual Review Workbench</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="topbar">
    <div>
      <h1>M5.4B Visual Review Workbench</h1>
      <p class="safety">VISUAL_ONLY_NOT_METRIC. Continuity review only; no identity, slot, metric, event, tactical, or physical conclusions.</p>
    </div>
    <div class="status-block" aria-live="polite">
      <strong id="saveStatus">Loading...</strong>
      <span id="lastSaved"></span>
      <span id="timer">00:00</span>
      <span id="stopIndicator"></span>
    </div>
  </header>

  <main class="layout">
    <aside class="sidebar">
      <div class="counts">
        <span id="caseCounter">Case 0 of 0</span>
        <span id="reviewedCounter">Reviewed 0</span>
        <span id="remainingCounter">Remaining 0</span>
        <span id="unresolvedCounter">Unresolved 0</span>
      </div>
      <label class="check"><input type="checkbox" id="unresolvedOnly"> Unresolved only</label>
      <nav id="caseList" aria-label="Review cases"></nav>
      <section class="shortcuts">
        <h2>Shortcuts</h2>
        <p><kbd>A</kbd> Accept <kbd>R</kbd> Reject <kbd>U</kbd> Unresolved</p>
        <p><kbd>Left</kbd>/<kbd>Right</kbd> Previous/next <kbd>Space</kbd> Play/pause</p>
        <p><kbd>Z</kbd> Zoom <kbd>C</kbd> Context <kbd>D</kbd> Details <kbd>N</kbd> Note <kbd>Ctrl+Z</kbd> Undo</p>
      </section>
    </aside>

    <section class="review">
      <div id="fallbackBanner" class="fallback hidden">Browser-only recovery mode - decisions are not yet durably saved to the project.</div>
      <div class="case-head">
        <div>
          <p id="category" class="eyebrow"></p>
          <h2 id="question">Loading review case...</h2>
          <p id="uncertainty" class="muted"></p>
        </div>
        <div class="frame-gap" id="frameGap"></div>
      </div>

      <div class="tabs" role="tablist">
        <button class="tab active" data-tab="primary">Comparison</button>
        <button class="tab" data-tab="context">Context</button>
        <button class="tab" data-tab="temporal">Temporal</button>
      </div>

      <section id="primaryPanel" class="evidence-panel"></section>
      <section id="contextPanel" class="evidence-panel hidden"></section>
      <section id="temporalPanel" class="evidence-panel hidden"></section>

      <section class="decision-panel" aria-label="Decision controls">
        <button id="acceptBtn" class="decision accept" data-decision="accept_continuity"><span>A</span> Accept continuity</button>
        <button id="rejectBtn" class="decision reject" data-decision="reject_continuity"><span>R</span> Reject continuity</button>
        <button id="unresolvedBtn" class="decision unresolved" data-decision="unresolved"><span>U</span> Unresolved</button>
      </section>

      <label class="note-label" for="note">Optional note</label>
      <textarea id="note" rows="4" placeholder="Add review note. Notes autosave after a short pause."></textarea>
      <div class="retry-row"><button id="retryBtn" class="secondary hidden">Retry save</button></div>

      <details id="engineeringDetails">
        <summary>Engineering details</summary>
        <pre id="engineeringJson"></pre>
      </details>

      <footer class="nav-row">
        <button id="prevBtn" class="secondary">Previous</button>
        <button id="nextBtn" class="secondary">Next</button>
        <button id="completeBtn" class="secondary">Complete review</button>
        <button id="exportBtn" class="secondary">Export JSON</button>
      </footer>
      <pre id="exportBox" class="export hidden"></pre>
    </section>
  </main>
  <script src="app.js"></script>
</body>
</html>
"""


STYLES_CSS = """
:root {
  color-scheme: light;
  --ink: #15171a;
  --muted: #5e6673;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --line: #cbd2dc;
  --accent: #005fcc;
  --accept: #0b6b31;
  --reject: #9f1d20;
  --unresolved: #6b4b00;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: Segoe UI, system-ui, sans-serif; color: var(--ink); background: var(--bg); }
.topbar { display: flex; justify-content: space-between; gap: 16px; padding: 14px 18px; background: #20242b; color: white; }
h1 { margin: 0 0 4px; font-size: 20px; }
h2 { margin: 0; font-size: 24px; line-height: 1.25; }
.safety { margin: 0; color: #e7ebf2; font-size: 13px; }
.status-block { display: grid; gap: 4px; text-align: right; min-width: 190px; }
.layout { display: grid; grid-template-columns: minmax(260px, 330px) 1fr; min-height: calc(100vh - 82px); }
.sidebar { border-right: 1px solid var(--line); padding: 14px; background: #eef1f5; overflow-y: auto; }
.counts { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 12px; font-size: 13px; }
.check { display: block; margin-bottom: 10px; }
#caseList { display: grid; gap: 6px; }
.case-button { text-align: left; border: 1px solid var(--line); background: white; border-radius: 6px; padding: 8px; cursor: pointer; }
.case-button.active { border-color: var(--accent); outline: 2px solid var(--accent); }
.case-button.done::after { content: " saved"; font-weight: 700; color: var(--accept); }
.shortcuts { margin-top: 18px; padding-top: 12px; border-top: 1px solid var(--line); font-size: 13px; }
kbd { display: inline-block; min-width: 22px; padding: 2px 5px; border: 1px solid #8d98a7; border-radius: 4px; background: white; text-align: center; font-weight: 700; }
.review { padding: 18px; overflow-y: auto; }
.fallback { border: 2px solid var(--reject); background: #fff4f4; padding: 10px; margin-bottom: 12px; font-weight: 700; }
.hidden { display: none !important; }
.case-head { display: flex; justify-content: space-between; gap: 12px; align-items: start; margin-bottom: 12px; }
.eyebrow { margin: 0 0 4px; text-transform: uppercase; font-size: 12px; font-weight: 800; color: var(--accent); letter-spacing: 0; }
.muted { color: var(--muted); margin: 6px 0 0; }
.frame-gap { border: 1px solid var(--line); background: var(--panel); border-radius: 6px; padding: 8px 10px; min-width: 110px; text-align: center; font-weight: 700; }
.tabs { display: flex; gap: 8px; margin: 12px 0; }
.tab { border: 1px solid var(--line); background: white; padding: 8px 12px; border-radius: 6px; cursor: pointer; }
.tab.active { border-color: var(--accent); background: #eaf2ff; font-weight: 700; }
.evidence-panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
.evidence-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.evidence-grid.primary { grid-template-columns: 1fr; }
figure { margin: 0; }
figcaption { margin-top: 5px; color: var(--muted); font-size: 13px; }
img, video { max-width: 100%; height: auto; border: 1px solid var(--line); background: #111; }
.zoom img, .zoom video { width: 100%; max-width: none; }
.decision-panel { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 14px 0; }
.decision { min-height: 58px; border: 3px solid transparent; border-radius: 8px; color: white; font-size: 17px; font-weight: 800; cursor: pointer; }
.decision span { display: inline-grid; place-items: center; min-width: 28px; height: 28px; border: 2px solid white; border-radius: 999px; margin-right: 8px; }
.accept { background: var(--accept); }
.reject { background: var(--reject); }
.unresolved { background: var(--unresolved); }
.decision.selected { outline: 4px solid #111; }
.secondary { border: 1px solid #7a8492; background: white; color: var(--ink); border-radius: 6px; padding: 9px 12px; cursor: pointer; }
.note-label { display: block; font-weight: 700; margin: 12px 0 5px; }
textarea { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 10px; font: inherit; }
.retry-row { min-height: 34px; margin-top: 6px; }
details { margin-top: 12px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; }
.nav-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.export { max-height: 280px; overflow: auto; background: #111; color: #e9edf4; padding: 12px; border-radius: 6px; }
#stopIndicator { color: #ffd166; font-weight: 900; }
@media (max-width: 900px) {
  .topbar, .case-head { flex-direction: column; }
  .layout { grid-template-columns: 1fr; }
  .sidebar { border-right: 0; border-bottom: 1px solid var(--line); max-height: 44vh; }
  .decision-panel, .evidence-grid { grid-template-columns: 1fr; }
}
"""


APP_JS = r"""
const SAVE_OK = "Saved OK";
let manifest = null;
let state = null;
let activeIndex = 0;
let activeTab = "primary";
let zoomed = false;
let contextVisible = true;
let noteTimer = null;
let pendingSave = null;
let lastFailedSave = null;
let elapsedSeconds = 0;
let timerStarted = Date.now();

const $ = (id) => document.getElementById(id);
const isTyping = () => {
  const el = document.activeElement;
  return el && (el.tagName === "TEXTAREA" || el.tagName === "INPUT" || el.isContentEditable);
};

function setStatus(text, failed=false) {
  $("saveStatus").textContent = text;
  $("saveStatus").style.color = failed ? "#ffd166" : "";
  $("retryBtn").classList.toggle("hidden", !failed);
}

function asset(caseItem, type) {
  return caseItem.evidence_manifest.evidence_assets.find(a => a.asset_type === type || a.asset_id === type);
}

function evidenceUrl(caseItem, assetItem) {
  return `/evidence/${caseItem.review_case_id}/${assetItem.relative_path}`;
}

async function api(path, options={}) {
  const response = await fetch(path, {
    headers: {"Content-Type": "application/json"},
    ...options
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

async function load() {
  try {
    manifest = await api("/api/review/manifest");
    state = await api("/api/review/state");
    elapsedSeconds = Number(state.elapsed_active_seconds || 0);
    const resumeId = state.resume_case_id || (manifest.review_cases[0] || {}).review_case_id;
    activeIndex = Math.max(0, manifest.review_cases.findIndex(c => c.review_case_id === resumeId));
    setStatus(SAVE_OK);
    render();
  } catch (err) {
    $("fallbackBanner").classList.remove("hidden");
    setStatus("Server unavailable", true);
    const cached = localStorage.getItem("m54b_browser_recovery_export");
    $("exportBox").textContent = cached || "";
    $("exportBox").classList.toggle("hidden", !cached);
  }
}

function activeCase() {
  return manifest.review_cases[activeIndex];
}

function decisions() {
  return (state && state.decisions) || {};
}

function notes() {
  return (state && state.notes) || {};
}

function counts() {
  const all = manifest.review_cases.length;
  const values = Object.values(decisions());
  const reviewed = values.length;
  const unresolved = values.filter(v => v === "unresolved").length;
  return {all, reviewed, unresolved, remaining: Math.max(0, all - reviewed)};
}

function renderCaseList() {
  const unresolvedOnly = $("unresolvedOnly").checked;
  $("caseList").innerHTML = "";
  manifest.review_cases.forEach((caseItem, index) => {
    if (unresolvedOnly && decisions()[caseItem.review_case_id]) return;
    const btn = document.createElement("button");
    btn.className = "case-button";
    if (index === activeIndex) btn.classList.add("active");
    if (decisions()[caseItem.review_case_id]) btn.classList.add("done");
    btn.textContent = `${index + 1}. ${caseItem.category} f${caseItem.source_frame_sequence}->${caseItem.target_frame_sequence}`;
    btn.onclick = () => { activeIndex = index; render(); };
    $("caseList").appendChild(btn);
  });
}

function renderEvidence(caseItem) {
  const side = asset(caseItem, "side_by_side");
  const src = asset(caseItem, "source_crop");
  const tgt = asset(caseItem, "target_crop");
  const srcContext = asset(caseItem, "source_context");
  const tgtContext = asset(caseItem, "target_context");
  const strip = asset(caseItem, "temporal_strip");
  const clip = asset(caseItem, "temporal_clip") || caseItem.evidence_manifest.evidence_assets.find(a => a.media_type === "video/mp4");
  const gif = caseItem.evidence_manifest.evidence_assets.find(a => a.media_type === "image/gif");
  $("primaryPanel").innerHTML = `
    <div class="evidence-grid primary">
      <figure><img alt="Side by side source and target detections" src="${evidenceUrl(caseItem, side)}"><figcaption>Side-by-side comparison</figcaption></figure>
    </div>
    <div class="evidence-grid">
      <figure><img alt="Source detection crop" src="${evidenceUrl(caseItem, src)}"><figcaption>Source crop</figcaption></figure>
      <figure><img alt="Target detection crop" src="${evidenceUrl(caseItem, tgt)}"><figcaption>Target crop</figcaption></figure>
    </div>`;
  $("contextPanel").innerHTML = `
    <div class="evidence-grid">
      <figure><img alt="Source wider context" src="${evidenceUrl(caseItem, srcContext)}"><figcaption>Source wider context</figcaption></figure>
      <figure><img alt="Target wider context" src="${evidenceUrl(caseItem, tgtContext)}"><figcaption>Target wider context</figcaption></figure>
    </div>`;
  const media = clip
    ? `<video id="temporalMedia" controls src="${evidenceUrl(caseItem, clip)}"></video>`
    : `<img id="temporalMedia" alt="Animated temporal evidence" src="${evidenceUrl(caseItem, gif)}">`;
  $("temporalPanel").innerHTML = `
    <div class="evidence-grid primary">
      <figure><img alt="Temporal strip" src="${evidenceUrl(caseItem, strip)}"><figcaption>Short temporal strip</figcaption></figure>
      <figure>${media}<figcaption>Playable temporal evidence</figcaption></figure>
    </div>`;
}

function render() {
  if (!manifest || !state) return;
  const caseItem = activeCase();
  const c = counts();
  $("caseCounter").textContent = `Case ${activeIndex + 1} of ${c.all}`;
  $("reviewedCounter").textContent = `Reviewed ${c.reviewed}`;
  $("remainingCounter").textContent = `Remaining ${c.remaining}`;
  $("unresolvedCounter").textContent = `Unresolved ${c.unresolved}`;
  $("category").textContent = caseItem.category;
  $("question").textContent = caseItem.concise_question;
  $("uncertainty").textContent = (caseItem.uncertainty_reasons || []).join(", ");
  $("frameGap").textContent = `Gap ${caseItem.evidence_manifest.frame_gap}`;
  $("note").value = notes()[caseItem.review_case_id] || "";
  $("engineeringJson").textContent = JSON.stringify({
    review_case_id: caseItem.review_case_id,
    candidate_artifact_id: caseItem.candidate_artifact_id,
    candidate_hash: caseItem.candidate_hash,
    evidence_hash: caseItem.evidence_hash,
    source_frame_sequence: caseItem.source_frame_sequence,
    target_frame_sequence: caseItem.target_frame_sequence,
    safety_payload: caseItem.safety_payload
  }, null, 2);
  renderEvidence(caseItem);
  renderCaseList();
  document.querySelectorAll(".decision").forEach(btn => {
    btn.classList.toggle("selected", decisions()[caseItem.review_case_id] === btn.dataset.decision);
    btn.onclick = () => saveDecision(btn.dataset.decision);
  });
  showTab(activeTab);
  applyZoom();
}

function showTab(tab) {
  activeTab = tab;
  document.querySelectorAll(".tab").forEach(btn => btn.classList.toggle("active", btn.dataset.tab === tab));
  $("primaryPanel").classList.toggle("hidden", tab !== "primary");
  $("contextPanel").classList.toggle("hidden", tab !== "context" || !contextVisible);
  $("temporalPanel").classList.toggle("hidden", tab !== "temporal");
}

function applyZoom() {
  document.querySelectorAll(".evidence-panel").forEach(panel => panel.classList.toggle("zoom", zoomed));
}

async function saveDecision(decision) {
  const caseItem = activeCase();
  const body = {
    review_case_id: caseItem.review_case_id,
    decision,
    note: $("note").value,
    last_viewed_case_id: caseItem.review_case_id,
    elapsed_active_seconds: elapsedSeconds + Math.floor((Date.now() - timerStarted) / 1000)
  };
  pendingSave = body;
  setStatus("Saving...");
  try {
    state = await api("/api/review/decision", {method: "POST", body: JSON.stringify(body)});
    $("lastSaved").textContent = `Last saved ${new Date().toLocaleTimeString()}`;
    lastFailedSave = null;
    setStatus(SAVE_OK);
    render();
  } catch (err) {
    lastFailedSave = () => saveDecision(decision);
    localStorage.setItem("m54b_browser_recovery_export", JSON.stringify({pending: body, error: String(err)}, null, 2));
    $("fallbackBanner").classList.remove("hidden");
    setStatus(`Save failed: ${err.message}`, true);
  }
}

async function saveNote() {
  const caseItem = activeCase();
  const body = {
    review_case_id: caseItem.review_case_id,
    note: $("note").value,
    last_viewed_case_id: caseItem.review_case_id,
    elapsed_active_seconds: elapsedSeconds + Math.floor((Date.now() - timerStarted) / 1000)
  };
  pendingSave = body;
  setStatus("Saving...");
  try {
    state = await api("/api/review/note", {method: "POST", body: JSON.stringify(body)});
    $("lastSaved").textContent = `Last saved ${new Date().toLocaleTimeString()}`;
    lastFailedSave = null;
    setStatus(SAVE_OK);
  } catch (err) {
    lastFailedSave = saveNote;
    localStorage.setItem("m54b_browser_recovery_export", JSON.stringify({pending: body, error: String(err)}, null, 2));
    $("fallbackBanner").classList.remove("hidden");
    setStatus(`Save failed: ${err.message}`, true);
  }
}

function go(delta) {
  activeIndex = Math.max(0, Math.min(manifest.review_cases.length - 1, activeIndex + delta));
  render();
}

function nextUnresolved() {
  const index = manifest.review_cases.findIndex(c => !decisions()[c.review_case_id]);
  if (index >= 0) activeIndex = index;
  render();
}

async function undo() {
  setStatus("Saving...");
  try {
    state = await api("/api/review/undo", {method: "POST", body: "{}"});
    setStatus(SAVE_OK);
    render();
  } catch (err) {
    setStatus(`Undo failed: ${err.message}`, true);
  }
}

async function completeReview() {
  setStatus("Saving...");
  try {
    state = await api("/api/review/complete", {method: "POST", body: JSON.stringify({elapsed_active_seconds: elapsedSeconds})});
    setStatus(SAVE_OK);
    render();
  } catch (err) {
    setStatus(`Complete failed: ${err.message}`, true);
  }
}

async function exportJson() {
  const payload = await api("/api/review/export");
  $("exportBox").textContent = JSON.stringify(payload, null, 2);
  $("exportBox").classList.remove("hidden");
}

function togglePlay() {
  const media = $("temporalMedia");
  if (media && media.tagName === "VIDEO") {
    if (media.paused) media.play(); else media.pause();
  }
}

document.addEventListener("keydown", (event) => {
  if (isTyping()) {
    if (event.key === "Escape") document.activeElement.blur();
    return;
  }
  if (event.ctrlKey && event.key.toLowerCase() === "z") { event.preventDefault(); undo(); return; }
  const key = event.key.toLowerCase();
  if (key === "a") saveDecision("accept_continuity");
  if (key === "r") saveDecision("reject_continuity");
  if (key === "u") saveDecision("unresolved");
  if (event.key === "ArrowLeft") go(-1);
  if (event.key === "ArrowRight") go(1);
  if (event.code === "Space") { event.preventDefault(); togglePlay(); }
  if (key === "z") { zoomed = !zoomed; applyZoom(); }
  if (key === "c") { contextVisible = !contextVisible; showTab("context"); }
  if (key === "d") $("engineeringDetails").open = !$("engineeringDetails").open;
  if (key === "n") $("note").focus();
});

document.addEventListener("click", (event) => {
  if (event.target.classList.contains("tab")) showTab(event.target.dataset.tab);
});

$("note").addEventListener("input", () => {
  clearTimeout(noteTimer);
  noteTimer = setTimeout(saveNote, 450);
});
$("prevBtn").onclick = () => go(-1);
$("nextBtn").onclick = () => go(1);
$("completeBtn").onclick = completeReview;
$("exportBtn").onclick = exportJson;
$("retryBtn").onclick = () => { if (lastFailedSave) lastFailedSave(); };
$("unresolvedOnly").onchange = () => { if ($("unresolvedOnly").checked) nextUnresolved(); else render(); };

setInterval(() => {
  const total = elapsedSeconds + Math.floor((Date.now() - timerStarted) / 1000);
  const minutes = String(Math.floor(total / 60)).padStart(2, "0");
  const seconds = String(total % 60).padStart(2, "0");
  $("timer").textContent = `${minutes}:${seconds}`;
  $("stopIndicator").textContent = total >= 600 ? "Ten-minute stop point reached" : "";
}, 1000);

load();
"""


FALLBACK_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>M5.4B Browser Recovery</title>
  <style>
    body { font-family: Segoe UI, system-ui, sans-serif; margin: 24px; background: #fff8f4; color: #15171a; }
    .warning { border: 3px solid #9f1d20; padding: 14px; font-weight: 800; background: white; }
    textarea { width: 100%; min-height: 260px; margin-top: 12px; }
    button { padding: 10px 14px; margin-top: 12px; }
  </style>
</head>
<body>
  <h1>M5.4B Browser Recovery</h1>
  <p class="warning">Browser-only recovery mode - decisions are not yet durably saved to the project.</p>
  <p>Use this only if the localhost review server is unavailable. Export JSON and re-enter decisions through the server-backed workbench.</p>
  <textarea id="box"></textarea>
  <br>
  <button id="export">Export browser recovery JSON</button>
  <script>
    const key = "m54b_browser_recovery_export";
    const box = document.getElementById("box");
    box.value = localStorage.getItem(key) || "{}";
    box.addEventListener("input", () => localStorage.setItem(key, box.value));
    document.getElementById("export").onclick = () => {
      const blob = new Blob([box.value], {type: "application/json"});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "m54b_browser_recovery_export.json";
      a.click();
      URL.revokeObjectURL(url);
    };
  </script>
</body>
</html>
"""
