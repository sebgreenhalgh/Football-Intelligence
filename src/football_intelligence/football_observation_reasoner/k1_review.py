# ruff: noqa: E501
"""Isolated target-only K1 team/role/kit person review package.

This module deliberately does not register a legacy review-chassis mode.  K1 has
its own browser namespace, package-local decision ledger, HTTP routes, and
completion transaction so that it cannot import or complete an earlier review.
"""

from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import os
import re
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

from football_intelligence.review_chassis.completion import (
    COMPLETION_FILENAMES,
    validate_completion_bundle,
    write_completion_transaction,
)
from football_intelligence.review_chassis.hashing import sha256_file, stable_hash

REVIEW_ID = "m5_5g7a_k1_team_role_kit_person_gold_v1"
TRANCHE_ID = "K1_TEAM_ROLE_KIT_PERSON_GOLD"
INDEXEDDB_NAMESPACE = "fi_m5_5g7a_k1_team_role_kit_person_gold_v1"
DEFAULT_REVIEWER_SESSION_ID = "m5_5g7a_k1_team_role_kit_person_gold_reviewer"
HOST = "127.0.0.1"
PORT = 8811
MAXIMUM_TARGET_COUNT = 144

ROLE_VALUES = (
    "OUTFIELD_PLAYER",
    "GOALKEEPER",
    "REFEREE",
    "OTHER_MATCH_OFFICIAL",
    "STAFF_OR_SPECTATOR",
    "UNKNOWN_ROLE",
)
TEAM_VALUES = ("TEAM_1", "TEAM_2", "NO_TEAM", "UNKNOWN_TEAM")
KIT_VALUES = (
    "MATCH_OUTFIELD_KIT",
    "MATCH_GOALKEEPER_KIT",
    "WARMUP_OR_BIB",
    "OFFICIAL_KIT",
    "STAFF_OR_SPECTATOR_CLOTHING",
    "UNKNOWN_KIT",
)
PITCH_VALUES = ("ON_PITCH", "OFF_PITCH", "BOUNDARY_UNCERTAIN", "UNKNOWN_PITCH_STATE")
PARTICIPATION_VALUES = (
    "ACTIVE_ON_PITCH",
    "OFF_PITCH_SUBSTITUTE_OR_WARMING",
    "OFF_PITCH_NON_PLAYER",
    "UNKNOWN_PARTICIPATION",
)
CERTAINTY_VALUES = ("CERTAIN", "PROBABLE", "UNCERTAIN")

TARGET_STRATA = {
    "team_1_on_pitch_outfield": 24,
    "team_2_on_pitch_outfield": 24,
    "team_1_goalkeeper": 12,
    "team_2_goalkeeper": 12,
    "team_1_off_pitch_warmup_player": 16,
    "team_2_off_pitch_warmup_player": 16,
    "referee_or_official": 12,
    "staff_or_spectator": 12,
    "ambiguous_or_occluded_control": 16,
}

GUIDANCE = (
    "A substitute remains a Player even when wearing a warmup top or bib whose colours differ from the team's "
    "match jersey.",
    "Each team has its own goalkeeper. Goalkeeper role and team affiliation are separate answers.",
    "A reserve goalkeeper may be off pitch and may wear warmup clothing.",
)

__all__ = [
    "CERTAINTY_VALUES",
    "DEFAULT_REVIEWER_SESSION_ID",
    "GUIDANCE",
    "HOST",
    "INDEXEDDB_NAMESPACE",
    "KIT_VALUES",
    "K1CaseSpec",
    "K1ContextFrame",
    "K1ReviewPersistence",
    "K1ServerConfig",
    "K1StateDivergenceError",
    "MAXIMUM_TARGET_COUNT",
    "PARTICIPATION_VALUES",
    "PITCH_VALUES",
    "PORT",
    "REVIEW_ID",
    "ROLE_VALUES",
    "TARGET_STRATA",
    "TEAM_VALUES",
    "TRANCHE_ID",
    "build_k1_package",
    "build_ui_config",
    "create_server",
    "load_k1_package",
    "render_k1_html",
    "render_k1_javascript",
    "render_launcher",
    "serve",
    "validate_k1_package",
]

_SAFE_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

_FORBIDDEN_KEYS = {
    "answer",
    "answers",
    "expected_answer",
    "expected_label",
    "ground_truth",
    "hidden_answer",
    "hidden_metadata",
    "identity_id",
    "identity_label",
    "person_identity",
    "player_id",
    "player_identity_id",
    "prefill",
    "prefilled_answer",
    "stable_identity_id",
    "track_id",
}
_ALLOWED_NON_LABEL_EXPECTED_KEYS = {"expected_server_state_hash"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, data)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _assert_no_forbidden_keys(value: Any, *, location: str = "payload") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            identity_key = "identity" in key and not key.endswith("_forbidden")
            answer_hint_key = key.startswith(("expected_", "hidden_", "prefill")) and key not in (
                _ALLOWED_NON_LABEL_EXPECTED_KEYS
            )
            if key in _FORBIDDEN_KEYS or identity_key or answer_hint_key:
                raise ValueError(f"forbidden answer or identity key at {location}.{raw_key}")
            _assert_no_forbidden_keys(child, location=f"{location}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_forbidden_keys(child, location=f"{location}[{index}]")


def _normalise_box(value: Mapping[str, Any]) -> dict[str, float]:
    required = {"x1", "y1", "x2", "y2"}
    if set(value) != required:
        raise ValueError("target_box must contain exactly x1, y1, x2, and y2")
    try:
        box = {axis: float(value[axis]) for axis in ("x1", "y1", "x2", "y2")}
    except (TypeError, ValueError) as exc:
        raise ValueError("target_box coordinates must be numeric") from exc
    if min(box.values()) < 0 or box["x2"] <= box["x1"] or box["y2"] <= box["y1"]:
        raise ValueError("target_box must have non-negative coordinates and positive area")
    return box


@dataclass(frozen=True)
class K1ContextFrame:
    """A nearby, non-authoritative image shown only as temporal context."""

    position: str
    image_path: Path
    claimed_sha256: str | None = None


@dataclass(frozen=True)
class K1CaseSpec:
    """Answer-free source material for one highlighted K1 target."""

    case_id: str
    source_group_id: str
    source_image_path: Path
    crop_image_path: Path
    target_box: Mapping[str, float]
    context_frames: tuple[K1ContextFrame, ...] = ()
    claimed_source_sha256: str | None = None
    claimed_crop_sha256: str | None = None


def _coerce_case_spec(value: K1CaseSpec | Mapping[str, Any]) -> K1CaseSpec:
    if isinstance(value, K1CaseSpec):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("each case must be K1CaseSpec or a mapping")
    contexts: list[K1ContextFrame] = []
    for context in value.get("context_frames", ()):
        if isinstance(context, K1ContextFrame):
            contexts.append(context)
        elif isinstance(context, Mapping):
            contexts.append(
                K1ContextFrame(
                    position=str(context["position"]),
                    image_path=Path(context["image_path"]),
                    claimed_sha256=str(context["claimed_sha256"]) if context.get("claimed_sha256") else None,
                )
            )
        else:
            raise TypeError("context frames must be K1ContextFrame or mappings")
    return K1CaseSpec(
        case_id=str(value["case_id"]),
        source_group_id=str(value["source_group_id"]),
        source_image_path=Path(value["source_image_path"]),
        crop_image_path=Path(value["crop_image_path"]),
        target_box=dict(value["target_box"]),
        context_frames=tuple(contexts),
        claimed_source_sha256=str(value["claimed_source_sha256"]) if value.get("claimed_source_sha256") else None,
        claimed_crop_sha256=str(value["claimed_crop_sha256"]) if value.get("claimed_crop_sha256") else None,
    )


def _image_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        raise ValueError(f"K1 evidence must use a browser-readable raster image: {path}")
    return suffix


def _copy_asset(source: Path, target: Path, claimed_sha256: str | None = None) -> str:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    actual = sha256_file(source)
    if claimed_sha256 is not None and claimed_sha256 != actual:
        raise ValueError(f"claimed evidence hash mismatch for {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    copied = sha256_file(target)
    if copied != actual:
        raise RuntimeError(f"evidence copy hash mismatch for {target}")
    return copied


def build_ui_config(*, manifest_hash: str) -> dict[str, Any]:
    """Return the answer-free, five-question novice UI contract."""

    return {
        "schema_version": "football_intelligence.m5_5g7a.k1_ui_config.v1",
        "review_id": REVIEW_ID,
        "tranche_id": TRANCHE_ID,
        "manifest_hash": manifest_hash,
        "presentation_mode": "team_role_kit_person_gold",
        "persistence_mode": "k1_target_person_gold_v1",
        "indexeddb_namespace": INDEXEDDB_NAMESPACE,
        "fresh_indexeddb_namespace": True,
        "prior_indexeddb_namespace_import_forbidden": True,
        "target_only": True,
        "current_frame_authoritative": True,
        "nearby_frames_context_only": True,
        "completion_requires_all_cases": True,
        "first_load_notice": "This is a fresh K1 review. No prior decisions or browser drafts are imported.",
        "guidance": list(GUIDANCE),
        "questions": [
            {
                "number": 1,
                "prompt": "What is this person's football role?",
                "fields": [{"name": "role", "options": list(ROLE_VALUES)}],
            },
            {
                "number": 2,
                "prompt": "Which team are they affiliated with?",
                "fields": [{"name": "team_affiliation", "options": list(TEAM_VALUES)}],
            },
            {
                "number": 3,
                "prompt": "What type of clothing/kit are they wearing?",
                "fields": [{"name": "kit_state", "options": list(KIT_VALUES)}],
            },
            {
                "number": 4,
                "prompt": "Are they active on the pitch, off-pitch warming/substitute, or something else?",
                "fields": [
                    {"name": "pitch_state", "label": "Pitch location", "options": list(PITCH_VALUES)},
                    {
                        "name": "participation_state",
                        "label": "Participation",
                        "options": list(PARTICIPATION_VALUES),
                    },
                ],
            },
            {
                "number": 5,
                "prompt": "How certain are you?",
                "fields": [{"name": "certainty", "options": list(CERTAINTY_VALUES)}],
            },
        ],
    }


def render_k1_html() -> str:
    """Render the standalone browser shell; labels are supplied only by the reviewer."""

    guidance = "\n".join(f"<li>{line}</li>" for line in GUIDANCE)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>K1 team, role and kit review</title>
  <style>
    :root {{ color-scheme: dark; font: 16px/1.45 system-ui, sans-serif; background: #10141b; color: #f4f7fb; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    header, main {{ width: min(1180px, 96vw); margin: 0 auto; }}
    header {{ padding: 1rem 0; }}
    .notice, .guidance, .question, .context {{ background: #19212c; border: 1px solid #344257; border-radius: .7rem; padding: 1rem; }}
    .guidance {{ margin: .8rem 0; }}
    .layout {{ display: grid; grid-template-columns: minmax(0, 3fr) minmax(280px, 2fr); gap: 1rem; }}
    .current-wrap {{ position: relative; background: #07090d; min-height: 320px; display: grid; place-items: center; }}
    #current-image {{ display: block; max-width: 100%; max-height: 68vh; }}
    #target-box {{ position: absolute; border: 4px solid #ffcf33; box-shadow: 0 0 0 2px #111; pointer-events: none; }}
    #target-box::before {{ content: "TARGET"; position: absolute; left: -2px; top: -2rem; background: #ffcf33; color: #111; font-weight: 800; padding: .2rem .45rem; }}
    #target-crop {{ max-width: 100%; max-height: 220px; border: 3px solid #ffcf33; }}
    .context-images {{ display: flex; gap: .6rem; overflow-x: auto; }}
    .context-images figure {{ min-width: 180px; margin: 0; }}
    .context-images img {{ width: 100%; max-height: 150px; object-fit: contain; background: #07090d; }}
    fieldset {{ border: 0; padding: 0; margin: .8rem 0; }}
    label.option {{ display: block; margin: .35rem 0; padding: .55rem; background: #222d3b; border-radius: .35rem; cursor: pointer; }}
    button {{ padding: .7rem 1rem; margin: .35rem; border: 0; border-radius: .35rem; font-weight: 700; cursor: pointer; }}
    button:disabled {{ opacity: .45; cursor: not-allowed; }}
    .status {{ min-height: 1.5rem; color: #9fd4ff; }}
    .danger {{ color: #ff9c9c; }}
    @media (max-width: 780px) {{ .layout {{ grid-template-columns: 1fr; }} .current-wrap {{ min-height: 220px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>K1 target-person review</h1>
    <p class="notice">Label the highlighted target person only. Other people are context. The Current frame is authoritative; nearby frames are context only.</p>
    <ul class="guidance">{guidance}</ul>
    <div id="progress"></div><div id="status" class="status" role="status" aria-live="polite"></div>
  </header>
  <main>
    <div class="layout">
      <section>
        <h2 id="case-title">Loading…</h2>
        <div class="current-wrap"><img id="current-image" alt="Authoritative Current frame"><div id="target-box" hidden></div></div>
        <h3>Highlighted target crop</h3><img id="target-crop" alt="Highlighted target crop">
        <div class="context"><h3>Nearby context only — do not annotate these frames</h3><div id="context-images" class="context-images"></div></div>
      </section>
      <section class="question">
        <div id="question"></div>
        <button id="question-back" type="button">Previous question</button>
        <button id="question-next" type="button">Next question</button>
      </section>
    </div>
    <nav>
      <button id="case-previous" type="button">Previous case</button>
      <button id="case-next" type="button">Next case</button>
      <button id="complete-review" type="button">Complete K1 review</button>
    </nav>
  </main>
  <script src="/app.js" defer></script>
</body>
</html>
"""


def render_k1_javascript() -> str:
    """Render a dependency-free IndexedDB draft/outbox client."""

    return r""""use strict";

const api = {
  manifest: "/api/k1/manifest", config: "/api/k1/ui-config", state: "/api/k1/state",
  event: "/api/k1/event", complete: "/api/k1/complete"
};
let manifest = null, config = null, serverState = null, database = null;
let caseIndex = 0, questionIndex = 0, answers = {};
const byId = (id) => document.getElementById(id);
const eventId = () => globalThis.crypto?.randomUUID?.() || `k1-${Date.now()}-${Math.random().toString(16).slice(2)}`;
const evidenceUrl = (caseId, relativePath) => `/evidence/${encodeURIComponent(caseId)}/${relativePath.split("/").map(encodeURIComponent).join("/")}`;

function requestJson(url, options = {}) {
  return fetch(url, {cache: "no-store", headers: {"Content-Type": "application/json"}, ...options}).then(async (response) => {
    const text = await response.text();
    let payload = {};
    try { payload = text ? JSON.parse(text) : {}; } catch (_) { payload = {error: text}; }
    if (!response.ok) { const error = new Error(payload.error || text || `HTTP ${response.status}`); error.status = response.status; throw error; }
    return payload;
  });
}

function openDatabase(name) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(name, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains("drafts")) db.createObjectStore("drafts", {keyPath: "case_id"});
      if (!db.objectStoreNames.contains("outbox")) db.createObjectStore("outbox", {keyPath: "client_event_id"});
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function dbRequest(storeName, mode, operation) {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(storeName, mode);
    const request = operation(transaction.objectStore(storeName));
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
const dbGet = (store, key) => dbRequest(store, "readonly", (value) => value.get(key));
const dbAll = (store) => dbRequest(store, "readonly", (value) => value.getAll());
const dbPut = (store, value) => dbRequest(store, "readwrite", (target) => target.put(value));
const dbDelete = (store, key) => dbRequest(store, "readwrite", (target) => target.delete(key));

function setStatus(message, danger = false) {
  byId("status").textContent = message;
  byId("status").classList.toggle("danger", danger);
}

function currentCase() { return manifest.cases[caseIndex]; }
function fieldsForQuestion() { return config.questions[questionIndex].fields; }
function questionAnswered() { return fieldsForQuestion().every((field) => Boolean(answers[field.name])); }
function annotationComplete() { return config.questions.flatMap((question) => question.fields).every((field) => Boolean(answers[field.name])); }

async function persistDraft() {
  await dbPut("drafts", {case_id: currentCase().case_id, answers: {...answers}, question_index: questionIndex, updated_at: new Date().toISOString()});
}

async function loadCase(index) {
  caseIndex = Math.max(0, Math.min(index, manifest.cases.length - 1));
  questionIndex = 0;
  const reviewCase = currentCase();
  const serverAnnotation = serverState.annotations?.[reviewCase.case_id]?.annotation;
  const draft = await dbGet("drafts", reviewCase.case_id);
  answers = {...(serverAnnotation || {}), ...(draft?.answers || {})};
  renderEvidence();
  renderQuestion();
  renderProgress();
}

function renderEvidence() {
  const reviewCase = currentCase();
  byId("case-title").textContent = `Current target ${caseIndex + 1} of ${manifest.cases.length}`;
  const image = byId("current-image");
  image.src = evidenceUrl(reviewCase.case_id, reviewCase.current_frame.relative_path);
  byId("target-crop").src = evidenceUrl(reviewCase.case_id, reviewCase.target_crop.relative_path);
  image.onload = () => {
    const box = reviewCase.target.bbox_original_pixels;
    const scaleX = image.clientWidth / image.naturalWidth, scaleY = image.clientHeight / image.naturalHeight;
    const overlay = byId("target-box");
    overlay.style.left = `${image.offsetLeft + box.x1 * scaleX}px`;
    overlay.style.top = `${image.offsetTop + box.y1 * scaleY}px`;
    overlay.style.width = `${(box.x2 - box.x1) * scaleX}px`;
    overlay.style.height = `${(box.y2 - box.y1) * scaleY}px`;
    overlay.hidden = false;
  };
  const context = byId("context-images"); context.replaceChildren();
  for (const frame of reviewCase.context_frames) {
    const figure = document.createElement("figure"), img = document.createElement("img"), caption = document.createElement("figcaption");
    img.src = evidenceUrl(reviewCase.case_id, frame.relative_path); img.alt = `${frame.position} context frame`;
    caption.textContent = `${frame.position} — context only`; figure.append(img, caption); context.append(figure);
  }
}

function renderQuestion() {
  const question = config.questions[questionIndex], root = byId("question"); root.replaceChildren();
  const heading = document.createElement("h2"); heading.textContent = `Question ${question.number} of 5`;
  const prompt = document.createElement("p"); prompt.textContent = question.prompt; root.append(heading, prompt);
  for (const field of question.fields) {
    const fieldset = document.createElement("fieldset"), legend = document.createElement("legend");
    legend.textContent = field.label || question.prompt; fieldset.append(legend);
    for (const option of field.options) {
      const label = document.createElement("label"), input = document.createElement("input");
      label.className = "option"; input.type = "radio"; input.name = field.name; input.value = option;
      input.checked = answers[field.name] === option;
      input.addEventListener("change", async () => { answers[field.name] = option; await persistDraft(); renderQuestion(); });
      label.append(input, document.createTextNode(` ${option.replaceAll("_", " ")}`)); fieldset.append(label);
    }
    root.append(fieldset);
  }
  byId("question-back").disabled = questionIndex === 0;
  byId("question-next").disabled = !questionAnswered();
  byId("question-next").textContent = questionIndex === config.questions.length - 1 ? "Save target" : "Next question";
}

function renderProgress() {
  const reviewed = Object.keys(serverState.annotations || {}).length;
  byId("progress").textContent = `${reviewed} of ${manifest.cases.length} targets acknowledged by the server.`;
  byId("case-previous").disabled = caseIndex === 0;
  byId("case-next").disabled = caseIndex === manifest.cases.length - 1;
  byId("complete-review").disabled = reviewed !== manifest.cases.length || serverState.completed;
}

async function queueCurrentCase() {
  if (!annotationComplete()) { setStatus("Answer every question before saving.", true); return; }
  const reviewCase = currentCase(), id = eventId();
  const annotation = {
    schema_version: "football_intelligence.m5_5g7a.k1_annotation.v1",
    role: answers.role, team_affiliation: answers.team_affiliation, kit_state: answers.kit_state,
    pitch_state: answers.pitch_state, participation_state: answers.participation_state, certainty: answers.certainty,
    source_frame_sha256: reviewCase.source_frame_sha256, target_crop_sha256: reviewCase.target_crop_sha256,
    target_binding_sha256: reviewCase.target_binding_sha256
  };
  const event = {
    event_type: "K1_CASE_SAVED", review_id: config.review_id, reviewer_session_id: serverState.reviewer_session_id,
    case_id: reviewCase.case_id, annotation, client_event_id: id, idempotency_key: id,
    expected_server_state_hash: serverState.server_state_hash, created_at: new Date().toISOString()
  };
  await dbPut("outbox", event); setStatus("Queued locally; waiting for server acknowledgement.");
  await flushOutbox();
}

async function flushOutbox() {
  const events = (await dbAll("outbox")).sort((a, b) => String(a.created_at).localeCompare(String(b.created_at)));
  let mayRebaseAfterOwnAck = false;
  for (const queued of events) {
    if (mayRebaseAfterOwnAck) { queued.expected_server_state_hash = serverState.server_state_hash; await dbPut("outbox", queued); }
    try {
      const response = await requestJson(api.event, {method: "POST", body: JSON.stringify(queued)});
      serverState = response.state; await dbDelete("outbox", queued.client_event_id); await dbDelete("drafts", queued.case_id);
      mayRebaseAfterOwnAck = true; setStatus("Saved after server acknowledgement.");
    } catch (error) {
      setStatus(error.status === 409 ? `State divergence: ${error.message}` : "Offline: the durable outbox will retry after reconnect.", true);
      break;
    }
  }
  renderProgress();
}

async function completeReview() {
  const outbox = await dbAll("outbox"), drafts = await dbAll("drafts");
  if (outbox.length || drafts.length) { setStatus("Completion is blocked while drafts or outbox events remain.", true); return; }
  const id = eventId();
  const payload = {
    event_type: "K1_REVIEW_COMPLETED", review_id: config.review_id, reviewer_session_id: serverState.reviewer_session_id,
    client_event_id: id, idempotency_key: id, expected_server_state_hash: serverState.server_state_hash,
    pending_outbox_events: 0, unresolved_draft_count: 0, evidence_blocker_count: 0, unresolved_divergence: false
  };
  try {
    const response = await requestJson(api.complete, {method: "POST", body: JSON.stringify(payload)});
    serverState = response.state; setStatus("K1 review completed atomically after server acknowledgement."); renderProgress();
  } catch (error) { setStatus(`Completion blocked: ${error.message}`, true); }
}

byId("question-back").addEventListener("click", () => { if (questionIndex > 0) { questionIndex -= 1; renderQuestion(); } });
byId("question-next").addEventListener("click", async () => {
  if (!questionAnswered()) return;
  if (questionIndex < config.questions.length - 1) { questionIndex += 1; await persistDraft(); renderQuestion(); }
  else await queueCurrentCase();
});
byId("case-previous").addEventListener("click", () => loadCase(caseIndex - 1));
byId("case-next").addEventListener("click", () => loadCase(caseIndex + 1));
byId("complete-review").addEventListener("click", completeReview);
window.addEventListener("resize", renderEvidence);

(async () => {
  try {
    [manifest, config, serverState] = await Promise.all([requestJson(api.manifest), requestJson(api.config), requestJson(api.state)]);
    database = await openDatabase(config.indexeddb_namespace);
    setStatus(config.first_load_notice);
    await flushOutbox();
    const resume = Math.max(0, manifest.cases.findIndex((item) => item.case_id === serverState.resume_case_id));
    await loadCase(resume);
  } catch (error) { setStatus(`K1 review failed to start: ${error.message}`, true); }
})();
"""


def render_launcher(*, package_root: Path, repo_root: Path | None = None) -> str:
    """Render a fixed-port PowerShell launcher with no fallback-port behavior."""

    repository = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    package = package_root.resolve()

    def quote(path: Path) -> str:
        return str(path).replace("'", "''")

    return f"""$ErrorActionPreference = 'Stop'
$PortInUse = Get-NetTCPConnection -LocalAddress '{HOST}' -LocalPort {PORT} -State Listen -ErrorAction SilentlyContinue
if ($PortInUse) {{
    throw 'K1 review requires {HOST}:{PORT}; the launcher will not move ports.'
}}
$RepoRoot = '{quote(repository)}'
$PackageRoot = '{quote(package)}'
Push-Location -LiteralPath $RepoRoot
try {{
    uv run python -m football_intelligence.football_observation_reasoner.k1_review serve --package-root $PackageRoot --host {HOST} --port {PORT} --reviewer-session-id {DEFAULT_REVIEWER_SESSION_ID}
}} finally {{
    Pop-Location
}}
"""


def _decisions_have_human_work(decisions_root: Path) -> bool:
    if not decisions_root.exists():
        return False
    if any((decisions_root / name).exists() for name in COMPLETION_FILENAMES):
        return True
    events = decisions_root / "k1_review_events.jsonl"
    if events.is_file() and events.read_text(encoding="utf-8").strip():
        return True
    state_path = decisions_root / "k1_review_state.json"
    if state_path.is_file():
        state = _read_json(state_path)
        if int(state.get("event_sequence", 0)) > 0 or state.get("annotations") or state.get("completed"):
            return True
    known = {"k1_review_events.jsonl", "k1_review_state.json"}
    for path in decisions_root.rglob("*"):
        if path.is_file() and path.name not in known and "snapshots" not in path.parts:
            return True
    snapshots = decisions_root / "snapshots"
    return snapshots.exists() and any(path.is_file() for path in snapshots.rglob("*"))


def _clear_empty_decision_binding(decisions_root: Path) -> None:
    for name in ("k1_review_events.jsonl", "k1_review_state.json"):
        path = decisions_root / name
        if path.exists():
            path.unlink()
    snapshots = decisions_root / "snapshots"
    if snapshots.exists() and not any(snapshots.iterdir()):
        snapshots.rmdir()


def build_k1_package(
    *,
    package_root: Path,
    cases: Sequence[K1CaseSpec | Mapping[str, Any]],
    stage_id: str,
    selection_spec_sha256: str,
    quota_shortfalls: Mapping[str, int],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Materialize a fresh, answer-free K1 package.

    Rebuilding an empty package is allowed, but a decision root containing any
    human event, annotation, snapshot, or completion artifact is immutable.
    """

    root = package_root.resolve()
    decisions_root = root / "decisions"
    if _decisions_have_human_work(decisions_root):
        raise ValueError("refusing to rebuild K1 package because its real decisions root contains human work")
    specs = [_coerce_case_spec(value) for value in cases]
    if not specs or len(specs) > MAXIMUM_TARGET_COUNT:
        raise ValueError(f"K1 requires between 1 and {MAXIMUM_TARGET_COUNT} highlighted targets")
    case_ids = [spec.case_id for spec in specs]
    if len(case_ids) != len(set(case_ids)) or any(_SAFE_CASE_ID.fullmatch(value) is None for value in case_ids):
        raise ValueError("K1 case IDs must be unique, path-safe identifiers")
    if not selection_spec_sha256 or len(selection_spec_sha256) != 64:
        raise ValueError("selection_spec_sha256 must be an exact SHA-256 digest")
    if set(quota_shortfalls) != set(TARGET_STRATA):
        raise ValueError("quota_shortfalls must report every frozen K1 target stratum")
    shortfalls = {key: int(quota_shortfalls[key]) for key in TARGET_STRATA}
    if any(value < 0 or value > TARGET_STRATA[key] for key, value in shortfalls.items()):
        raise ValueError("quota shortfalls must be between zero and the frozen target quota")
    selected_from_quotas = sum(TARGET_STRATA[key] - value for key, value in shortfalls.items())
    if selected_from_quotas != len(specs):
        raise ValueError("reported K1 quota shortfalls do not reconcile to the selected target count")

    root.mkdir(parents=True, exist_ok=True)
    evidence_root = root / "evidence"
    manifest_cases: list[dict[str, Any]] = []
    for spec in specs:
        _assert_no_forbidden_keys(spec.target_box, location=f"case[{spec.case_id}].target_box")
        if not spec.source_group_id.strip():
            raise ValueError(f"source_group_id is required for {spec.case_id}")
        box = _normalise_box(spec.target_box)
        case_root = evidence_root / spec.case_id
        source_name = f"current_source{_image_suffix(spec.source_image_path)}"
        crop_name = f"target_crop{_image_suffix(spec.crop_image_path)}"
        source_hash = _copy_asset(spec.source_image_path, case_root / source_name, spec.claimed_source_sha256)
        crop_hash = _copy_asset(spec.crop_image_path, case_root / crop_name, spec.claimed_crop_sha256)
        target_binding_hash = stable_hash(
            {
                "case_id": spec.case_id,
                "source_frame_sha256": source_hash,
                "target_crop_sha256": crop_hash,
                "bbox_original_pixels": box,
            }
        )
        contexts: list[dict[str, Any]] = []
        for index, context in enumerate(spec.context_frames):
            if not context.position.strip() or context.position.lower() == "current":
                raise ValueError("context frame positions must be non-empty and cannot be Current")
            position_slug = re.sub(r"[^a-z0-9_-]+", "_", context.position.lower()).strip("_")
            if not position_slug:
                raise ValueError("context frame positions must contain a path-safe name")
            context_name = f"context_{index:02d}_{position_slug}{_image_suffix(context.image_path)}"
            context_hash = _copy_asset(context.image_path, case_root / context_name, context.claimed_sha256)
            contexts.append(
                {
                    "position": context.position,
                    "relative_path": context_name,
                    "sha256": context_hash,
                    "authoritative": False,
                    "editable": False,
                    "context_only": True,
                }
            )
        manifest_cases.append(
            {
                "case_id": spec.case_id,
                "source_group_id": spec.source_group_id,
                "target_only": True,
                "other_people_context_only": True,
                "source_frame_sha256": source_hash,
                "target_crop_sha256": crop_hash,
                "target_binding_sha256": target_binding_hash,
                "target": {
                    "highlight_label": "TARGET",
                    "bbox_original_pixels": box,
                    "binding_sha256": target_binding_hash,
                },
                "current_frame": {
                    "position": "Current",
                    "relative_path": source_name,
                    "sha256": source_hash,
                    "authoritative": True,
                    "editable": False,
                },
                "target_crop": {
                    "relative_path": crop_name,
                    "sha256": crop_hash,
                    "authoritative": True,
                    "editable": False,
                },
                "context_frames": contexts,
            }
        )

    manifest = {
        "schema_version": "football_intelligence.m5_5g7a.k1_review_manifest.v1",
        "created_at": utc_now(),
        "review_id": REVIEW_ID,
        "stage_id": str(stage_id),
        "tranche_id": TRANCHE_ID,
        "selection_spec_sha256": selection_spec_sha256,
        "selection_frozen_before_human_answers": True,
        "maximum_target_count": MAXIMUM_TARGET_COUNT,
        "target_count": len(manifest_cases),
        "source_group_count": len({case["source_group_id"] for case in manifest_cases}),
        "target_only": True,
        "one_highlighted_person_per_case": True,
        "exhaustive_frame_annotation_forbidden": True,
        "identity_labels_forbidden": True,
        "quota_targets": copy.deepcopy(TARGET_STRATA),
        "quota_shortfalls": shortfalls,
        "cases": manifest_cases,
    }
    _assert_no_forbidden_keys(manifest, location="manifest")
    manifest_hash = stable_hash(manifest)
    ui_config = build_ui_config(manifest_hash=manifest_hash)
    _assert_no_forbidden_keys(ui_config, location="ui_config")
    _atomic_write_json(root / "k1_manifest.json", manifest)
    _atomic_write_json(root / "k1_ui_config.json", ui_config)
    _atomic_write_bytes(root / "index.html", render_k1_html().encode("utf-8"))
    _atomic_write_bytes(root / "app.js", render_k1_javascript().encode("utf-8"))
    _atomic_write_bytes(
        root / "launch_team_role_kit_review.ps1",
        render_launcher(package_root=root, repo_root=repo_root).encode("utf-8"),
    )
    decisions_root.mkdir(parents=True, exist_ok=True)
    _clear_empty_decision_binding(decisions_root)
    validation = validate_k1_package(root)
    _atomic_write_json(root / "k1_package_validation.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"generated K1 package is invalid: {validation['errors']}")
    return {
        "package_root": str(root),
        "manifest_path": str(root / "k1_manifest.json"),
        "ui_config_path": str(root / "k1_ui_config.json"),
        "decisions_root": str(decisions_root),
        "manifest_hash": manifest_hash,
        "ui_config_hash": stable_hash(ui_config),
        "validation": validation,
    }


def validate_k1_package(package_root: Path) -> dict[str, Any]:
    root = package_root.resolve()
    errors: list[str] = []
    try:
        manifest = _read_json(root / "k1_manifest.json")
        config = _read_json(root / "k1_ui_config.json")
        _assert_no_forbidden_keys(manifest, location="manifest")
        _assert_no_forbidden_keys(config, location="ui_config")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [str(exc)]}
    if manifest.get("review_id") != REVIEW_ID or config.get("review_id") != REVIEW_ID:
        errors.append("K1 review ID mismatch")
    if config.get("indexeddb_namespace") != INDEXEDDB_NAMESPACE:
        errors.append("K1 IndexedDB namespace mismatch")
    expected_manifest_hash = stable_hash(manifest)
    if config.get("manifest_hash") != expected_manifest_hash:
        errors.append("K1 manifest hash mismatch")
    if config != build_ui_config(manifest_hash=expected_manifest_hash):
        errors.append("K1 UI config deviates from the answer-free novice contract")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAXIMUM_TARGET_COUNT:
        errors.append("K1 target count is invalid")
        cases = []
    if manifest.get("target_count") != len(cases):
        errors.append("K1 target count binding mismatch")
    case_ids = [str(case.get("case_id", "")) for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(set(case_ids)):
        errors.append("K1 case IDs must be unique")
    selection_hash = str(manifest.get("selection_spec_sha256", ""))
    if len(selection_hash) != 64 or any(character not in "0123456789abcdef" for character in selection_hash):
        errors.append("K1 selection-specification SHA-256 binding is invalid")
    if manifest.get("source_group_count") != len(
        {str(case.get("source_group_id")) for case in cases if isinstance(case, dict)}
    ):
        errors.append("K1 source-group count binding mismatch")
    shortfalls = manifest.get("quota_shortfalls", {})
    if not isinstance(shortfalls, dict) or set(shortfalls) != set(TARGET_STRATA):
        errors.append("K1 quota shortfalls do not cover every frozen target stratum")
    else:
        try:
            selected_from_quotas = sum(TARGET_STRATA[key] - int(shortfalls[key]) for key in TARGET_STRATA)
        except (TypeError, ValueError):
            selected_from_quotas = -1
        if selected_from_quotas != len(cases):
            errors.append("K1 quota shortfalls do not reconcile to the target count")
    for case in cases:
        if not isinstance(case, dict) or set(case.get("target", {})) != {
            "highlight_label",
            "bbox_original_pixels",
            "binding_sha256",
        }:
            errors.append("each K1 case must have exactly one answer-free highlighted target")
            continue
        case_id = str(case.get("case_id", ""))
        if _SAFE_CASE_ID.fullmatch(case_id) is None:
            errors.append(f"unsafe K1 case ID: {case_id}")
            continue
        if case.get("target", {}).get("highlight_label") != "TARGET":
            errors.append(f"target highlight missing for {case.get('case_id')}")
        current = case.get("current_frame", {})
        crop = case.get("target_crop", {})
        if (
            current.get("position") != "Current"
            or current.get("authoritative") is not True
            or current.get("editable") is not False
        ):
            errors.append(f"Current frame is not authoritative for {case.get('case_id')}")
        if current.get("sha256") != case.get("source_frame_sha256"):
            errors.append(f"Current source hash binding mismatch for {case.get('case_id')}")
        if crop.get("sha256") != case.get("target_crop_sha256") or crop.get("editable") is not False:
            errors.append(f"target crop hash binding mismatch for {case.get('case_id')}")
        for frame in case.get("context_frames", []):
            if (
                frame.get("authoritative") is not False
                or frame.get("editable") is not False
                or frame.get("context_only") is not True
            ):
                errors.append(f"context frame authority mismatch for {case.get('case_id')}")
        expected_binding = stable_hash(
            {
                "case_id": case.get("case_id"),
                "source_frame_sha256": case.get("source_frame_sha256"),
                "target_crop_sha256": case.get("target_crop_sha256"),
                "bbox_original_pixels": case.get("target", {}).get("bbox_original_pixels"),
            }
        )
        if expected_binding != case.get("target_binding_sha256") or expected_binding != case["target"].get(
            "binding_sha256"
        ):
            errors.append(f"target binding hash mismatch for {case.get('case_id')}")
        for asset in [current, crop, *case.get("context_frames", [])]:
            relative = Path(str(asset.get("relative_path", "")))
            case_root = (root / "evidence" / case_id).resolve()
            path = (case_root / relative).resolve()
            if not path.is_relative_to(case_root) or not path.is_file():
                errors.append(f"missing or unsafe K1 evidence for {case.get('case_id')}")
            elif sha256_file(path) != asset.get("sha256"):
                errors.append(f"evidence hash mismatch for {case.get('case_id')}:{relative.as_posix()}")
        current_path = (root / "evidence" / case_id / str(current.get("relative_path", ""))).resolve()
        try:
            from PIL import Image

            with Image.open(current_path) as image:
                image_width, image_height = image.size
            box = _normalise_box(case["target"]["bbox_original_pixels"])
            if box["x2"] > image_width or box["y2"] > image_height:
                errors.append(f"target box exceeds authoritative Current frame for {case.get('case_id')}")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"target box/source bounds cannot be verified for {case.get('case_id')}: {exc}")
    questions = config.get("questions", [])
    field_names = [field.get("name") for question in questions for field in question.get("fields", [])]
    if len(questions) != 5 or field_names != [
        "role",
        "team_affiliation",
        "kit_state",
        "pitch_state",
        "participation_state",
        "certainty",
    ]:
        errors.append("K1 questions do not preserve the six independent stored axes across five questions")
    if list(config.get("guidance", [])) != list(GUIDANCE):
        errors.append("K1 goalkeeper/warmup guidance mismatch")
    return {
        "passed": not errors,
        "errors": errors,
        "review_id": REVIEW_ID,
        "indexeddb_namespace": INDEXEDDB_NAMESPACE,
        "target_count": len(cases),
        "manifest_hash": stable_hash(manifest),
        "ui_config_hash": stable_hash(config),
    }


class K1StateDivergenceError(ValueError):
    """The client tried to apply an event to a different server state."""


def _canonical_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": state.get("schema_version"),
        "review_id": state.get("review_id"),
        "stage_id": state.get("stage_id"),
        "manifest_hash": state.get("manifest_hash"),
        "ui_config_hash": state.get("ui_config_hash"),
        "event_sequence": int(state.get("event_sequence", 0)),
        "annotations": state.get("annotations", {}),
        "event_receipts": state.get("event_receipts", {}),
        "client_event_ids": state.get("client_event_ids", {}),
        "completed": bool(state.get("completed")),
        "completed_at": state.get("completed_at"),
        "completion_transaction_id": state.get("completion_transaction_id"),
    }


@dataclass
class K1ReviewPersistence:
    manifest: dict[str, Any]
    ui_config: dict[str, Any]
    decisions_root: Path
    reviewer_session_id: str = DEFAULT_REVIEWER_SESSION_ID
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.decisions_root = self.decisions_root.resolve()
        if self.manifest.get("review_id") != REVIEW_ID or self.ui_config.get("review_id") != REVIEW_ID:
            raise ValueError("K1 persistence refuses a non-K1 review ID")
        if self.ui_config.get("indexeddb_namespace") != INDEXEDDB_NAMESPACE:
            raise ValueError("K1 persistence refuses a predecessor browser namespace")
        if self.ui_config.get("manifest_hash") != self.manifest_hash:
            raise ValueError("K1 manifest/UI binding mismatch")

    @property
    def state_path(self) -> Path:
        return self.decisions_root / "k1_review_state.json"

    @property
    def events_path(self) -> Path:
        return self.decisions_root / "k1_review_events.jsonl"

    @property
    def snapshots_root(self) -> Path:
        return self.decisions_root / "snapshots"

    @property
    def manifest_hash(self) -> str:
        return stable_hash(self.manifest)

    @property
    def ui_config_hash(self) -> str:
        return stable_hash(self.ui_config)

    @property
    def case_map(self) -> dict[str, dict[str, Any]]:
        return {str(case["case_id"]): case for case in self.manifest["cases"]}

    def _empty_state(self, *, created_at: str | None = None) -> dict[str, Any]:
        timestamp = created_at or utc_now()
        return {
            "schema_version": "football_intelligence.m5_5g7a.k1_review_state.v1",
            "created_at": timestamp,
            "updated_at": timestamp,
            "review_id": REVIEW_ID,
            "stage_id": self.manifest["stage_id"],
            "tranche_id": TRANCHE_ID,
            "reviewer_session_id": self.reviewer_session_id,
            "manifest_hash": self.manifest_hash,
            "ui_config_hash": self.ui_config_hash,
            "event_sequence": 0,
            "annotations": {},
            "event_receipts": {},
            "client_event_ids": {},
            "elapsed_active_seconds": 0,
            "completed": False,
            "completed_at": None,
            "completion_transaction_id": None,
            "identity_tracking_performed": False,
            "temporal_predictions_used": False,
            "promotion_authorized": False,
        }

    def _server_state_hash(self, state: Mapping[str, Any]) -> str:
        return stable_hash(_canonical_state(state))

    def _load_events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line_number, line in enumerate(self.events_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"K1 event ledger is corrupt at line {line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"K1 event ledger line {line_number} is not an object")
            events.append(payload)
        return events

    def _assert_binding(self, state: Mapping[str, Any]) -> None:
        if state.get("review_id") != REVIEW_ID:
            raise ValueError("K1 decision state review ID mismatch")
        if state.get("manifest_hash") != self.manifest_hash:
            raise ValueError("K1 decision state manifest hash mismatch")
        if state.get("ui_config_hash") != self.ui_config_hash:
            raise ValueError("K1 decision state UI-config hash mismatch")
        if state.get("reviewer_session_id") != self.reviewer_session_id:
            raise ValueError("K1 decision state reviewer-session mismatch")

    def _apply_event(self, state: dict[str, Any], event: Mapping[str, Any]) -> dict[str, Any]:
        sequence = int(state["event_sequence"]) + 1
        if event.get("event_sequence") != sequence:
            raise ValueError("K1 event sequence is not contiguous")
        if event.get("review_id") != REVIEW_ID or event.get("manifest_hash") != self.manifest_hash:
            raise ValueError("K1 event binding mismatch")
        event_without_hash = {key: value for key, value in event.items() if key != "event_hash"}
        if event.get("event_hash") != stable_hash(event_without_hash):
            raise ValueError("K1 event hash mismatch")
        if event.get("prior_server_state_hash") != self._server_state_hash(state):
            raise ValueError("K1 event prior-state hash mismatch")
        event_type = event.get("event_type")
        if event_type == "K1_CASE_SAVED":
            case_id = str(event.get("case_id"))
            annotation = event.get("annotation")
            self._validate_annotation(case_id, annotation)
            state["annotations"][case_id] = {
                "annotation": copy.deepcopy(annotation),
                "saved_at": event["timestamp"],
                "event_sequence": sequence,
            }
        elif event_type == "REVIEW_COMPLETED":
            if set(state["annotations"]) != set(self.case_map):
                raise ValueError("K1 completion event precedes complete target coverage")
            state["completed"] = True
            state["completed_at"] = event["timestamp"]
            state["completion_transaction_id"] = event["completion_transaction_id"]
        else:
            raise ValueError(f"unsupported K1 event type: {event_type}")
        receipt = {
            "client_event_id": event["client_event_id"],
            "client_payload_hash": event["client_payload_hash"],
            "event_sequence": sequence,
            "event_type": event_type,
        }
        state["event_receipts"][event["idempotency_key"]] = receipt
        state["client_event_ids"][event["client_event_id"]] = event["idempotency_key"]
        state["event_sequence"] = sequence
        state["updated_at"] = event["timestamp"]
        state["elapsed_active_seconds"] = max(
            int(state.get("elapsed_active_seconds", 0)), int(event.get("elapsed_active_seconds", 0))
        )
        if event.get("server_state_hash") != self._server_state_hash(state):
            raise ValueError("K1 event resulting-state hash mismatch")
        return state

    def _replay(self, events: Sequence[Mapping[str, Any]], *, created_at: str | None = None) -> dict[str, Any]:
        state = self._empty_state(created_at=created_at)
        for event in events:
            state = self._apply_event(state, event)
        return state

    def ensure_state(self) -> dict[str, Any]:
        with self._lock:
            self.decisions_root.mkdir(parents=True, exist_ok=True)
            self.snapshots_root.mkdir(parents=True, exist_ok=True)
            self.events_path.touch(exist_ok=True)
            persisted = _read_json(self.state_path) if self.state_path.exists() else None
            if persisted is not None:
                self._assert_binding(persisted)
            replayed = self._replay(
                self._load_events(),
                created_at=str(persisted.get("created_at")) if persisted is not None else None,
            )
            if persisted is None or _canonical_state(persisted) != _canonical_state(replayed):
                _atomic_write_json(self.state_path, replayed)
            if replayed.get("completed"):
                self._ensure_completion_bundle(replayed)
            return replayed

    def _counts(self, state: Mapping[str, Any]) -> dict[str, Any]:
        annotations = state.get("annotations", {})
        return {
            "total_cases": len(self.case_map),
            "reviewed": len(annotations),
            "remaining": max(0, len(self.case_map) - len(annotations)),
            "completed": bool(state.get("completed")),
        }

    def _response(self, state: Mapping[str, Any], *, ack: Mapping[str, Any] | None = None) -> dict[str, Any]:
        response = copy.deepcopy(dict(state))
        response["server_state_hash"] = self._server_state_hash(state)
        response["counts"] = self._counts(state)
        response["resume_case_id"] = next(
            (case_id for case_id in self.case_map if case_id not in state.get("annotations", {})),
            next(iter(self.case_map), None),
        )
        if ack is not None:
            response["ack"] = dict(ack)
        return response

    def state(self) -> dict[str, Any]:
        return self._response(self.ensure_state())

    def _validate_annotation(self, case_id: str, annotation: Any) -> None:
        if case_id not in self.case_map:
            raise ValueError(f"unknown K1 case: {case_id}")
        if not isinstance(annotation, Mapping):
            raise ValueError("K1 annotation must be an object")
        _assert_no_forbidden_keys(annotation, location="annotation")
        required = {
            "schema_version",
            "role",
            "team_affiliation",
            "kit_state",
            "pitch_state",
            "participation_state",
            "certainty",
            "source_frame_sha256",
            "target_crop_sha256",
            "target_binding_sha256",
        }
        if set(annotation) != required:
            raise ValueError("K1 annotation must contain exactly the independent answer axes and evidence bindings")
        if annotation.get("schema_version") != "football_intelligence.m5_5g7a.k1_annotation.v1":
            raise ValueError("K1 annotation schema mismatch")
        enum_contract = {
            "role": ROLE_VALUES,
            "team_affiliation": TEAM_VALUES,
            "kit_state": KIT_VALUES,
            "pitch_state": PITCH_VALUES,
            "participation_state": PARTICIPATION_VALUES,
            "certainty": CERTAINTY_VALUES,
        }
        for field_name, allowed in enum_contract.items():
            if annotation.get(field_name) not in allowed:
                raise ValueError(f"invalid K1 {field_name}")
        case = self.case_map[case_id]
        for field_name in ("source_frame_sha256", "target_crop_sha256", "target_binding_sha256"):
            if annotation.get(field_name) != case[field_name]:
                raise ValueError(f"K1 annotation {field_name} mismatch")

    def _client_payload_hash(self, payload: Mapping[str, Any], *, completion: bool) -> str:
        fields = (
            (
                "event_type",
                "review_id",
                "reviewer_session_id",
                "client_event_id",
                "idempotency_key",
                "pending_outbox_events",
                "unresolved_draft_count",
                "evidence_blocker_count",
                "unresolved_divergence",
            )
            if completion
            else (
                "event_type",
                "review_id",
                "reviewer_session_id",
                "case_id",
                "annotation",
                "client_event_id",
                "idempotency_key",
            )
        )
        return stable_hash({field_name: payload.get(field_name) for field_name in fields})

    def _validate_envelope(self, payload: Mapping[str, Any], *, completion: bool) -> tuple[str, str, str]:
        _assert_no_forbidden_keys(payload, location="event")
        expected_type = "K1_REVIEW_COMPLETED" if completion else "K1_CASE_SAVED"
        if payload.get("event_type") != expected_type:
            raise ValueError(f"K1 event_type must be {expected_type}")
        if payload.get("review_id") != REVIEW_ID:
            raise ValueError("K1 event review ID mismatch")
        if payload.get("reviewer_session_id") != self.reviewer_session_id:
            raise ValueError("K1 event reviewer-session mismatch")
        client_event_id = str(payload.get("client_event_id", "")).strip()
        idempotency_key = str(payload.get("idempotency_key", "")).strip()
        if not client_event_id or not idempotency_key:
            raise ValueError("K1 events require client_event_id and idempotency_key")
        return client_event_id, idempotency_key, self._client_payload_hash(payload, completion=completion)

    def _idempotent_receipt(
        self,
        state: Mapping[str, Any],
        *,
        client_event_id: str,
        idempotency_key: str,
        client_payload_hash: str,
    ) -> dict[str, Any] | None:
        receipt = state.get("event_receipts", {}).get(idempotency_key)
        if receipt is not None:
            if (
                receipt.get("client_event_id") != client_event_id
                or receipt.get("client_payload_hash") != client_payload_hash
            ):
                raise ValueError("K1 idempotency key was reused for a different payload")
            return dict(receipt)
        prior_key = state.get("client_event_ids", {}).get(client_event_id)
        if prior_key is not None and prior_key != idempotency_key:
            raise ValueError("K1 client_event_id was reused with a different idempotency key")
        return None

    def _check_expected_hash(self, state: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
        expected = payload.get("expected_server_state_hash")
        if not isinstance(expected, str) or not expected:
            raise ValueError("expected_server_state_hash is required")
        actual = self._server_state_hash(state)
        if expected != actual:
            raise K1StateDivergenceError(f"expected server state {expected}, found {actual}")

    def _new_event(
        self,
        *,
        state: Mapping[str, Any],
        payload: Mapping[str, Any],
        event_type: str,
        client_payload_hash: str,
        extra: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        timestamp = utc_now()
        event: dict[str, Any] = {
            "schema_version": "football_intelligence.m5_5g7a.k1_event.v1",
            "event_type": event_type,
            "event_sequence": int(state["event_sequence"]) + 1,
            "timestamp": timestamp,
            "review_id": REVIEW_ID,
            "reviewer_session_id": self.reviewer_session_id,
            "manifest_hash": self.manifest_hash,
            "ui_config_hash": self.ui_config_hash,
            "client_event_id": str(payload["client_event_id"]),
            "idempotency_key": str(payload["idempotency_key"]),
            "client_payload_hash": client_payload_hash,
            "prior_server_state_hash": self._server_state_hash(state),
            "elapsed_active_seconds": int(payload.get("elapsed_active_seconds", 0)),
            **dict(extra),
        }
        candidate = self._apply_event_without_integrity(copy.deepcopy(dict(state)), event)
        event["server_state_hash"] = self._server_state_hash(candidate)
        event["event_hash"] = stable_hash(event)
        replay_candidate = self._apply_event(copy.deepcopy(dict(state)), event)
        return event, replay_candidate

    def _apply_event_without_integrity(self, state: dict[str, Any], event: Mapping[str, Any]) -> dict[str, Any]:
        event_type = event["event_type"]
        sequence = int(event["event_sequence"])
        if event_type == "K1_CASE_SAVED":
            state["annotations"][str(event["case_id"])] = {
                "annotation": copy.deepcopy(event["annotation"]),
                "saved_at": event["timestamp"],
                "event_sequence": sequence,
            }
        elif event_type == "REVIEW_COMPLETED":
            state["completed"] = True
            state["completed_at"] = event["timestamp"]
            state["completion_transaction_id"] = event["completion_transaction_id"]
        receipt = {
            "client_event_id": event["client_event_id"],
            "client_payload_hash": event["client_payload_hash"],
            "event_sequence": sequence,
            "event_type": event_type,
        }
        state["event_receipts"][event["idempotency_key"]] = receipt
        state["client_event_ids"][event["client_event_id"]] = event["idempotency_key"]
        state["event_sequence"] = sequence
        state["updated_at"] = event["timestamp"]
        state["elapsed_active_seconds"] = max(
            int(state.get("elapsed_active_seconds", 0)), int(event.get("elapsed_active_seconds", 0))
        )
        return state

    def _persist_event(self, event: Mapping[str, Any], state: Mapping[str, Any]) -> None:
        _append_jsonl(self.events_path, event)
        _atomic_write_json(self.state_path, state)
        _atomic_write_json(
            self.snapshots_root / f"k1_state_{int(event['event_sequence']):06d}.json",
            {
                "schema_version": "football_intelligence.m5_5g7a.k1_snapshot.v1",
                "created_at": event["timestamp"],
                "event_sequence": event["event_sequence"],
                "server_state_hash": self._server_state_hash(state),
                "state": state,
            },
        )

    def save_event(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            client_id, key, payload_hash = self._validate_envelope(payload, completion=False)
            state = self.ensure_state()
            receipt = self._idempotent_receipt(
                state, client_event_id=client_id, idempotency_key=key, client_payload_hash=payload_hash
            )
            if receipt is not None:
                return {"state": self._response(state), "ack": {**receipt, "idempotent_retry": True}}
            if state.get("completed"):
                raise ValueError("completed K1 reviews are immutable")
            self._check_expected_hash(state, payload)
            case_id = str(payload.get("case_id", ""))
            self._validate_annotation(case_id, payload.get("annotation"))
            event, candidate = self._new_event(
                state=state,
                payload=payload,
                event_type="K1_CASE_SAVED",
                client_payload_hash=payload_hash,
                extra={"case_id": case_id, "annotation": copy.deepcopy(payload["annotation"])},
            )
            self._persist_event(event, candidate)
            ack = {
                "client_event_id": client_id,
                "idempotency_key": key,
                "event_sequence": event["event_sequence"],
                "server_state_hash": event["server_state_hash"],
                "idempotent_retry": False,
            }
            return {"state": self._response(candidate), "ack": ack}

    def recover_authoritative_state(self) -> dict[str, Any]:
        state = self.ensure_state()
        return {
            "state": self._response(state),
            "event_count": len(self._load_events()),
            "replayed_from_event_ledger": True,
        }

    def _axis_counts(self, state: Mapping[str, Any]) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for field_name in ("role", "team_affiliation", "kit_state", "pitch_state", "participation_state", "certainty"):
            values = [row["annotation"][field_name] for row in state["annotations"].values()]
            counts[field_name] = {value: values.count(value) for value in sorted(set(values))}
        return counts

    def _completion_payloads(self, state: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        state_hash = self._server_state_hash(state)
        transaction_id = str(state["completion_transaction_id"])
        counts = self._counts(state)
        common = {
            "review_id": REVIEW_ID,
            "stage_id": self.manifest["stage_id"],
            "manifest_hash": self.manifest_hash,
            "ui_config_hash": self.ui_config_hash,
            "decision_state_hash": state_hash,
            "completion_transaction_id": transaction_id,
        }
        k1_summary = {
            "tranche_id": TRANCHE_ID,
            "selection_spec_sha256": self.manifest["selection_spec_sha256"],
            "exact_target_count": len(self.case_map),
            "source_group_count": self.manifest["source_group_count"],
            "quota_targets": self.manifest["quota_targets"],
            "quota_shortfalls": self.manifest["quota_shortfalls"],
            "axis_label_counts": self._axis_counts(state),
            "target_only": True,
            "current_frame_authoritative": True,
            "nearby_frames_context_only": True,
            "identity_labels_present": False,
        }
        completed_review = {
            "schema_version": "football_intelligence.m5_5g7a.k1_completed_review.v1",
            "created_at": state["completed_at"],
            **common,
            "state": copy.deepcopy(dict(state)),
            "summary": {**counts, "completed": True, "k1": k1_summary},
            "human_approved": False,
            "promotion_authorized": False,
        }
        completed_manifest = {
            "schema_version": "football_intelligence.m5_5g7a.k1_completed_manifest.v1",
            "created_at": state["completed_at"],
            **common,
            "human_approved": False,
            "promotion_authorized": False,
        }
        completed_summary = {
            "schema_version": "football_intelligence.m5_5g7a.k1_completed_summary.v1",
            "created_at": state["completed_at"],
            **common,
            **counts,
            "completed": True,
            "reviewer_session_id": self.reviewer_session_id,
            "k1": k1_summary,
            "human_approved": False,
            "promotion_authorized": False,
        }
        return completed_review, completed_manifest, completed_summary

    def _write_completion_bundle(
        self, state: Mapping[str, Any], *, fail_after_replace: int | None = None
    ) -> dict[str, Any]:
        completed_review, completed_manifest, completed_summary = self._completion_payloads(state)
        return write_completion_transaction(
            decisions_root=self.decisions_root,
            completed_review=completed_review,
            completed_events=self.events_path.read_bytes(),
            completed_manifest=completed_manifest,
            completed_summary=completed_summary,
            fail_after_replace=fail_after_replace,
        )

    def _ensure_completion_bundle(self, state: Mapping[str, Any]) -> None:
        validation = validate_completion_bundle(self.decisions_root)
        if not validation.get("passed"):
            self._write_completion_bundle(state)

    def complete(self, payload: Mapping[str, Any], *, fail_after_replace: int | None = None) -> dict[str, Any]:
        with self._lock:
            client_id, key, payload_hash = self._validate_envelope(payload, completion=True)
            state = self.ensure_state()
            receipt = self._idempotent_receipt(
                state, client_event_id=client_id, idempotency_key=key, client_payload_hash=payload_hash
            )
            if receipt is not None:
                if not state.get("completed"):
                    raise ValueError("K1 completion receipt exists without completed state")
                self._ensure_completion_bundle(state)
                return {
                    "state": self._response(state),
                    "ack": {**receipt, "idempotent_retry": True},
                    "completion": validate_completion_bundle(self.decisions_root),
                }
            if state.get("completed"):
                raise ValueError("completed K1 reviews are immutable")
            self._check_expected_hash(state, payload)
            if set(state["annotations"]) != set(self.case_map):
                raise ValueError("K1 completion is blocked until every target is server-acknowledged")
            blockers = {
                "pending_outbox_events": int(payload.get("pending_outbox_events", -1)),
                "unresolved_draft_count": int(payload.get("unresolved_draft_count", -1)),
                "evidence_blocker_count": int(payload.get("evidence_blocker_count", -1)),
            }
            if any(value != 0 for value in blockers.values()) or payload.get("unresolved_divergence") is not False:
                raise ValueError("K1 completion is blocked by drafts, outbox events, evidence, or state divergence")
            transaction_id = f"k1_completion_{uuid.uuid4().hex}"
            event, candidate = self._new_event(
                state=state,
                payload=payload,
                event_type="REVIEW_COMPLETED",
                client_payload_hash=payload_hash,
                extra={
                    "completion_transaction_id": transaction_id,
                    "pending_outbox_events": 0,
                    "unresolved_draft_count": 0,
                    "evidence_blocker_count": 0,
                    "unresolved_divergence": False,
                },
            )
            prior_events = self.events_path.read_bytes()
            prior_state = copy.deepcopy(state)
            snapshot_path = self.snapshots_root / f"k1_state_{int(event['event_sequence']):06d}.json"
            self._persist_event(event, candidate)
            try:
                completion = self._write_completion_bundle(candidate, fail_after_replace=fail_after_replace)
            except Exception:
                _atomic_write_bytes(self.events_path, prior_events)
                _atomic_write_json(self.state_path, prior_state)
                if snapshot_path.exists():
                    snapshot_path.unlink()
                raise
            ack = {
                "client_event_id": client_id,
                "idempotency_key": key,
                "event_sequence": event["event_sequence"],
                "server_state_hash": event["server_state_hash"],
                "idempotent_retry": False,
            }
            return {"state": self._response(candidate), "ack": ack, "completion": completion}


@dataclass(frozen=True)
class K1ServerConfig:
    package_root: Path
    reviewer_session_id: str = DEFAULT_REVIEWER_SESSION_ID
    host: str = HOST
    port: int = PORT

    def __post_init__(self) -> None:
        if self.host != HOST or self.port != PORT:
            raise ValueError(f"K1 server is fixed to {HOST}:{PORT} and will not move ports")


def load_k1_package(package_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = validate_k1_package(package_root)
    if not validation["passed"]:
        raise ValueError(f"invalid K1 package: {validation['errors']}")
    root = package_root.resolve()
    return _read_json(root / "k1_manifest.json"), _read_json(root / "k1_ui_config.json")


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, *, status: int = 200) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _request_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    size = int(handler.headers.get("Content-Length", "0"))
    payload = json.loads(handler.rfile.read(size).decode("utf-8")) if size else {}
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


class K1HTTPServer(ThreadingHTTPServer):
    def __init__(self, config: K1ServerConfig):
        self.config = config
        self.package_root = config.package_root.resolve()
        self.manifest, self.ui_config = load_k1_package(self.package_root)
        self.persistence = K1ReviewPersistence(
            manifest=self.manifest,
            ui_config=self.ui_config,
            decisions_root=self.package_root / "decisions",
            reviewer_session_id=config.reviewer_session_id,
        )
        self.persistence.ensure_state()
        super().__init__((config.host, config.port), K1RequestHandler)


class K1RequestHandler(BaseHTTPRequestHandler):
    server: K1HTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path in {"/", "/index.html"}:
                self._serve_file(self.server.package_root / "index.html")
            elif path == "/app.js":
                self._serve_file(self.server.package_root / "app.js")
            elif path == "/api/k1/manifest":
                _json_response(self, self.server.manifest)
            elif path == "/api/k1/ui-config":
                _json_response(self, self.server.ui_config)
            elif path == "/api/k1/state":
                _json_response(self, self.server.persistence.state())
            elif path == "/api/k1/recover":
                _json_response(self, self.server.persistence.recover_authoritative_state())
            elif path.startswith("/evidence/"):
                self._serve_evidence(path)
            else:
                _json_response(self, {"error": "not found"}, status=404)
        except Exception as exc:  # pragma: no cover - HTTP boundary
            _json_response(self, {"error": str(exc)}, status=500)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = _request_body(self)
            if path == "/api/k1/event":
                _json_response(self, self.server.persistence.save_event(payload))
            elif path == "/api/k1/complete":
                _json_response(self, self.server.persistence.complete(payload))
            else:
                _json_response(self, {"error": "not found"}, status=404)
        except K1StateDivergenceError as exc:
            _json_response(self, {"error": str(exc), "error_code": "STATE_DIVERGENCE"}, status=409)
        except Exception as exc:
            _json_response(self, {"error": str(exc)}, status=400)

    def _serve_evidence(self, request_path: str) -> None:
        parts = request_path.removeprefix("/evidence/").split("/", 1)
        if len(parts) != 2:
            _json_response(self, {"error": "evidence not found"}, status=404)
            return
        case_id, relative_text = unquote(parts[0]), unquote(parts[1])
        if case_id not in self.server.persistence.case_map:
            _json_response(self, {"error": "evidence not found"}, status=404)
            return
        root = (self.server.package_root / "evidence" / case_id).resolve()
        target = (root / Path(relative_text)).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            _json_response(self, {"error": "evidence not found"}, status=404)
            return
        self._serve_file(target)

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            _json_response(self, {"error": "not found"}, status=404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def create_server(config: K1ServerConfig) -> K1HTTPServer:
    """Create the fixed-address K1 server without touching legacy server routing."""

    return K1HTTPServer(config)


def serve(
    *,
    package_root: Path,
    reviewer_session_id: str = DEFAULT_REVIEWER_SESSION_ID,
    host: str = HOST,
    port: int = PORT,
) -> None:
    """Serve K1 only at its contracted local address."""

    server = create_server(
        K1ServerConfig(
            package_root=package_root,
            reviewer_session_id=reviewer_session_id,
            host=host,
            port=port,
        )
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="K1 target-only team/role/kit person review")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--package-root", type=Path, required=True)
    serve_parser.add_argument("--reviewer-session-id", default=DEFAULT_REVIEWER_SESSION_ID)
    serve_parser.add_argument("--host", default=HOST)
    serve_parser.add_argument("--port", type=int, default=PORT)
    arguments = parser.parse_args(argv)
    if arguments.command == "serve":
        serve(
            package_root=arguments.package_root,
            reviewer_session_id=arguments.reviewer_session_id,
            host=arguments.host,
            port=arguments.port,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
