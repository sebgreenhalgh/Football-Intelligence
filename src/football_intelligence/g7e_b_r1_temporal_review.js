"use strict";

const $ = (id) => document.getElementById(id);
const ui = {};
for (const id of [
  "modePill", "progressText", "progressFill", "saveState", "helpButton", "practiceBanner", "resetPractice",
  "legacyDraftNotice", "resetLegacyPractice", "previewBanner", "blockingError", "reviewShell", "welcomeScreen",
  "completionScreen", "caseEyebrow", "caseTitle", "overlayToggle", "subjectToggle", "idToggle", "evidenceCard",
  "panoramaWrap", "panoramaCanvas", "assetState", "mappingState", "fitButton", "zoomOutButton", "zoomInButton",
  "resetViewButton", "zoomSubjectButton", "fullScreenButton", "zoomPercent", "lockViewToggle", "firstFrameButton",
  "previousFrameButton", "playButton", "nextFrameButton", "centreFrameButton", "lastFrameButton", "timeline",
  "focusWrap", "focusCanvas", "focusAssetState", "focusFitButton", "focusZoomOutButton", "focusZoomInButton",
  "focusResetButton", "focusZoomSubjectButton", "focusZoomPercent", "subjectReference", "subjectReferenceToken",
  "subjectReferenceCanvas", "subjectReferenceTitle", "subjectReferenceMeta", "questionStep", "subjectPill",
  "questionLegend", "questionKicker", "questionTitle", "questionHelp", "answerArea", "backButton", "continueButton",
  "startRealButton", "startPracticeButton", "completionTitle", "completionCount", "trancheReceipt", "lastEvent",
  "globalReceiptRow", "globalReceipt", "pauseMessage", "reviewCompletedButton", "nextTrancheButton", "confirmDialog",
  "confirmTitle", "helpDrawer", "closeHelp",
]) ui[id] = $(id);

const REVISION = "G7E_B_R1_SUBJECT_GUIDANCE_AND_ZOOM_REPAIR_V1";
const SUBJECT_COLOURS = ["#2cc9a0", "#9a72e8", "#e7a51a"];
const LOCATION_VALUES = [
  ["VISIBLE_COMPLETE", "Visible — click the person"],
  ["VISIBLE_PARTIAL", "Partly visible — click the visible part"],
  ["FULLY_OCCLUDED_EXPECTED_PRESENT", "Hidden behind someone, but probably still there"],
  ["OUT_OF_FRAME_OR_LEFT_SCENE", "Left the picture"],
  ["NOT_PRESENT", "Not present"],
  ["UNCERTAIN", "Not sure"],
];
const SUPPLY_VALUES = [
  ["ONE_USEFUL_CANDIDATE", "One useful box"],
  ["MULTIPLE_CANDIDATES", "More than one box for this same person"],
  ["MERGED_WITH_OTHER_PEOPLE", "One box covers several people"],
  ["FRAGMENT_ONLY", "Only a body fragment has a box"],
  ["NO_CANDIDATE", "No useful box"],
  ["UNCERTAIN", "Not sure"],
];
const LABELS = Object.fromEntries([...LOCATION_VALUES, ...SUPPLY_VALUES]);

const app = {
  mode: null,
  cases: [],
  allCases: [],
  current: null,
  data: null,
  frame: 4,
  questionKey: "original_focus",
  history: [],
  speed: 1,
  timer: null,
  playing: false,
  image: null,
  focusImage: null,
  assetReady: false,
  mappingVerified: false,
  readOnly: false,
  inputMode: "pan",
  inputSubject: null,
  pointer: null,
  drawQueued: false,
  view: { zoom: 1, centerX: 0.5, centerY: 0.5 },
  focusView: { zoom: 1, centerX: 0.5, centerY: 0.5 },
  performance: [],
};

function observation() {
  return {
    visibility: null,
    subject_location_source_x: null,
    subject_location_source_y: null,
    human_confirmed: false,
    approximate_hidden_location: false,
    observation_supply: null,
    selected_candidate_ids: [],
    occlusion_phase: "NONE",
  };
}

function newSubject(index, source) {
  return {
    subject_token: `SUBJECT_${"ABC"[index]}`,
    subject_definition_source: source,
    anchor_frame_sequence: null,
    anchor_source_xy: null,
    frame_observations: Array.from({ length: 9 }, observation),
    marker_continuity_confirmation: null,
    candidate_relationship: null,
    occlusion_confirmed: false,
    continuity: null,
    role: null,
    participation: null,
    certainty: null,
  };
}

function blankData(caseRow) {
  return {
    review_revision: REVISION,
    burst_id: caseRow.burst_id,
    current_question: "original_focus",
    current_frame_sequence: 4,
    playback_speed: 1,
    answers: {},
    subjects: [],
    candidate_mappings: [],
    missed_person_marks: [],
  };
}

function currentSubject(index = activeSubjectIndex()) { return app.data?.subjects[index] || null; }
function activeSubjectIndex() {
  const match = app.questionKey.match(/^subject_(\d+)_/);
  return match ? Number(match[1]) : 0;
}
function subjectLetter(index) { return "ABC"[index] || "A"; }
function ensureSubjects(count, source) {
  while (app.data.subjects.length < count) app.data.subjects.push(newSubject(app.data.subjects.length, source));
  app.data.subjects = app.data.subjects.slice(0, count);
}
function relevantVisibility(value) {
  return ["VISIBLE_COMPLETE", "VISIBLE_PARTIAL", "FULLY_OCCLUDED_EXPECTED_PRESENT", "UNCERTAIN"].includes(value);
}
function selectionRequired(value) {
  return ["ONE_USEFUL_CANDIDATE", "MULTIPLE_CANDIDATES", "MERGED_WITH_OTHER_PEOPLE", "FRAGMENT_ONLY"].includes(value);
}
function frameCandidates(index = app.frame) { return app.current?.frame_candidates?.[index] || []; }
function needsRelationship(subject) {
  return subject.frame_observations.some((row) => ["MULTIPLE_CANDIDATES", "MERGED_WITH_OTHER_PEOPLE", "FRAGMENT_ONLY"].includes(row.observation_supply));
}
function needsOcclusion(subject) {
  const values = subject.frame_observations.map((row) => row.visibility);
  const reappears = values.some((value, index) => index > 0 && ["VISIBLE_COMPLETE", "VISIBLE_PARTIAL"].includes(value) && values.slice(0, index).some((prior) => ["FULLY_OCCLUDED_EXPECTED_PRESENT", "NOT_PRESENT"].includes(prior)));
  return reappears || values.some((value) => ["VISIBLE_PARTIAL", "FULLY_OCCLUDED_EXPECTED_PRESENT"].includes(value));
}
function needsContinuity(subject) {
  return needsOcclusion(subject) || needsRelationship(subject) || subject.marker_continuity_confirmation === "CANNOT_TELL";
}

function questionSequence() {
  const q = ["original_focus"];
  const focus = app.data.answers.original_focus_box_answer;
  if (focus === "NO_RELEVANT_PERSON") q.push("context_subject");
  if (focus === "NOT_SURE") q.push("uncertain_focus_path");
  for (let index = 0; index < app.data.subjects.length; index += 1) {
    q.push(`subject_${index}_anchor`);
    if (index === 0 && focus === "MORE_THAN_ONE_RELEVANT_PERSON") q.push("multi_subject_b");
    for (let frame = 0; frame < 9; frame += 1) q.push(`subject_${index}_location_${frame}`);
    q.push(`subject_${index}_marker_review`);
    for (let frame = 0; frame < 9; frame += 1) {
      if (relevantVisibility(app.data.subjects[index].frame_observations[frame].visibility)) q.push(`subject_${index}_supply_${frame}`);
    }
    if (needsRelationship(app.data.subjects[index])) q.push(`subject_${index}_relationship`);
    if (needsOcclusion(app.data.subjects[index])) q.push(`subject_${index}_occlusion`);
    if (needsContinuity(app.data.subjects[index])) q.push(`subject_${index}_continuity`);
    q.push(`subject_${index}_role`, `subject_${index}_participation`, `subject_${index}_certainty`);
  }
  if (app.data.subjects.length > 0 && app.data.subjects.length < 3) q.push("additional_subject");
  q.push("missed_check");
  if (app.data.answers.missed_check === "YES") q.push("missed_mark");
  q.push("summary");
  return q;
}

function parseQuestion(key) {
  const match = key.match(/^subject_(\d+)_(location|supply)_(\d+)$/);
  if (match) return { subject: Number(match[1]), kind: match[2], frame: Number(match[3]) };
  const simple = key.match(/^subject_(\d+)_(.+)$/);
  return simple ? { subject: Number(simple[1]), kind: simple[2], frame: null } : { subject: null, kind: key, frame: null };
}

function reviewReady() { return app.assetReady && app.mappingVerified && !app.readOnly; }
function setQuestion(title, help, kicker = "ONE PLAIN-ENGLISH STEP", showLegend = false) {
  ui.questionTitle.textContent = title;
  ui.questionHelp.textContent = help;
  ui.questionKicker.textContent = kicker;
  ui.questionLegend.classList.toggle("hidden", !showLegend);
}
function answerCards(options, selected, onSelect) {
  ui.answerArea.innerHTML = "";
  options.forEach(([value, label, help], index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `answer-card${selected === value ? " selected" : ""}`;
    button.dataset.value = value;
    button.dataset.shortcut = String(index + 1);
    button.innerHTML = `<span>${label}</span>${help ? `<small>${help}</small>` : ""}`;
    button.disabled = !reviewReady();
    button.setAttribute("aria-pressed", selected === value ? "true" : "false");
    button.onclick = () => onSelect(value);
    ui.answerArea.appendChild(button);
  });
}

async function chooseOriginalFocus(value) {
  app.data.answers.original_focus_box_answer = value;
  delete app.data.answers.context_subject_answer;
  delete app.data.answers.uncertain_focus_path;
  if (["ONE_RELEVANT_MATCH_PERSON", "PART_OF_ONE_RELEVANT_MATCH_PERSON"].includes(value)) {
    ensureSubjects(1, "YELLOW_ORIGINAL_FOCUS_CANDIDATE");
  } else if (value === "MORE_THAN_ONE_RELEVANT_PERSON") {
    ensureSubjects(1, "YELLOW_MULTI_PERSON_HUMAN_SELECTION");
  } else {
    app.data.subjects = [];
  }
  await saveDraft();
  renderQuestion();
}

async function chooseContext(value) {
  app.data.answers.context_subject_answer = value;
  if (value === "YES_ONE_PERSON") ensureSubjects(1, "BLUE_CONTEXT_HUMAN_SELECTION");
  else if (value === "YES_MORE_THAN_ONE_PERSON") ensureSubjects(2, "BLUE_CONTEXT_HUMAN_SELECTION");
  else app.data.subjects = [];
  await saveDraft(); renderQuestion();
}

async function chooseUncertainPath(value) {
  app.data.answers.uncertain_focus_path = value;
  if (value === "UNCERTAIN_SUBJECT_A") ensureSubjects(1, "UNCERTAIN_HUMAN_SELECTION");
  else if (value === "NO_SUBJECT") app.data.subjects = [];
  await saveDraft(); renderQuestion();
}

function renderQuestion() {
  if (!app.current || !app.data) return;
  const sequence = questionSequence();
  if (!sequence.includes(app.questionKey)) app.questionKey = sequence[0];
  const position = sequence.indexOf(app.questionKey);
  const part = parseQuestion(app.questionKey);
  ui.questionStep.textContent = `Question ${position + 1} of ${sequence.length}`;
  ui.subjectPill.textContent = part.subject === null ? "" : `Subject ${subjectLetter(part.subject)}`;
  ui.backButton.disabled = app.history.length === 0 || app.readOnly;
  ui.continueButton.textContent = app.questionKey === "summary" ? "Save burst" : "Continue";
  ui.continueButton.disabled = true;
  app.inputMode = "pan";
  app.inputSubject = part.subject;
  updateSubjectReference(part.subject);

  if (part.kind === "original_focus") {
    setQuestion("What does the yellow original focus box contain?", "Judge the yellow box only. The blue dashed area is nearby context.", "START WITH THE YELLOW ORIGINAL BOX", true);
    answerCards([
      ["ONE_RELEVANT_MATCH_PERSON", "One relevant match person"],
      ["PART_OF_ONE_RELEVANT_MATCH_PERSON", "Part of one relevant match person"],
      ["MORE_THAN_ONE_RELEVANT_PERSON", "More than one relevant person"],
      ["NO_RELEVANT_PERSON", "No relevant person"],
      ["NOT_SURE", "Not sure"],
    ], app.data.answers.original_focus_box_answer, chooseOriginalFocus);
    ui.continueButton.disabled = !app.data.answers.original_focus_box_answer;
    return;
  }
  if (part.kind === "context_subject") {
    setQuestion("Is there a relevant player, goalkeeper, or match official nearby in the blue context area who is worth following through this clip?", "The yellow original box contains no relevant person. Judge the blue context area now.", "OPTIONAL NEARBY SUBJECT", true);
    answerCards([["YES_ONE_PERSON", "Yes — one person"], ["YES_MORE_THAN_ONE_PERSON", "Yes — more than one person"], ["NO", "No"], ["NOT_SURE", "Not sure"]], app.data.answers.context_subject_answer, chooseContext);
    ui.continueButton.disabled = !app.data.answers.context_subject_answer;
    return;
  }
  if (part.kind === "uncertain_focus_path") {
    setQuestion("What would you like to do with the uncertain yellow box?", "You do not need to force a guess.", "SAFE UNCERTAIN PATH", true);
    answerCards([["TRY_DIFFERENT_FRAME", "Try a different frame"], ["UNCERTAIN_SUBJECT_A", "Continue with an uncertain Subject A"], ["NO_SUBJECT", "No subject to follow"]], app.data.answers.uncertain_focus_path, chooseUncertainPath);
    ui.continueButton.disabled = !app.data.answers.uncertain_focus_path || app.data.answers.uncertain_focus_path === "TRY_DIFFERENT_FRAME";
    return;
  }
  if (part.kind === "multi_subject_b") {
    setQuestion("Should another person in the yellow box become Subject B?", "Subject A is already the person you clicked. Add Subject B only when another relevant person in the yellow box should be followed.", "OPTIONAL SECOND SUBJECT", true);
    answerCards([["ADD_SUBJECT_B", "Yes — define Subject B"], ["ONLY_SUBJECT_A", "No — follow Subject A only"], ["NOT_SURE", "Not sure"]], app.data.answers.multi_subject_b, async (value) => {
      app.data.answers.multi_subject_b = value;
      if (value === "ADD_SUBJECT_B") ensureSubjects(2, "YELLOW_MULTI_PERSON_HUMAN_SELECTION");
      else app.data.subjects = app.data.subjects.slice(0, 1);
      await saveDraft(); renderQuestion();
    });
    ui.continueButton.disabled = !app.data.answers.multi_subject_b;
    return;
  }
  if (part.kind === "anchor") { renderAnchor(part.subject); return; }
  if (part.kind === "location") { renderLocation(part.subject, part.frame); return; }
  if (part.kind === "marker_review") { renderMarkerReview(part.subject); return; }
  if (part.kind === "supply") { renderSupply(part.subject, part.frame); return; }
  if (part.kind === "relationship") { renderRelationship(part.subject); return; }
  if (part.kind === "occlusion") { renderOcclusion(part.subject); return; }
  if (part.kind === "continuity") { renderContinuity(part.subject); return; }
  if (["role", "participation", "certainty"].includes(part.kind)) { renderSubjectSingle(part.subject, part.kind); return; }
  if (part.kind === "additional_subject") { renderAdditionalSubject(); return; }
  if (part.kind === "missed_check") { renderMissedCheck(); return; }
  if (part.kind === "missed_mark") { renderMissedMark(); return; }
  if (part.kind === "summary") { renderSummary(); }
}

function renderAnchor(index) {
  const subject = currentSubject(index);
  setQuestion(`Click Subject ${subjectLetter(index)} in the clearest frame`, subject.subject_definition_source.includes("YELLOW") ? `Subject ${subjectLetter(index)} is the person represented by or selected from the yellow original focus box.` : `Subject ${subjectLetter(index)} is the nearby person you chose in the blue context area.`, `DEFINE SUBJECT ${subjectLetter(index)} WITH A HUMAN CLICK`, true);
  ui.answerArea.innerHTML = `<div class="click-required"><b>${subject.anchor_source_xy ? "Human anchor recorded" : "Waiting for your click"}</b><br>${subject.anchor_source_xy ? `Frame ${subject.anchor_frame_sequence + 1} · source (${subject.anchor_source_xy.map(Math.round).join(", ")})` : "Step to the clearest frame, zoom if needed, then click the person."}</div>`;
  app.inputMode = "anchor";
  ui.continueButton.disabled = !subject.anchor_source_xy;
}

function setFrameForQuestion(frame) {
  if (app.frame !== frame) loadFrame(frame, true).catch((error) => block(`FRAME_LOAD_ERROR — ${error.message}`));
}

function renderLocation(index, frame) {
  setFrameForQuestion(frame);
  const subject = currentSubject(index);
  const row = subject.frame_observations[frame];
  setQuestion(`Where is Subject ${subjectLetter(index)} in Frame ${frame + 1}?`, frame === 4 ? "The yellow original box is visible only on this centre frame. Use your pinned subject reference." : `The original yellow box exists only on the centre frame. Follow the human-confirmed Subject ${subjectLetter(index)} marker in this frame.`, `FRAME ${frame + 1} OF 9 · HUMAN SUBJECT LOCATION`);
  answerCards(LOCATION_VALUES, row.visibility, async (value) => {
    row.visibility = value;
    row.subject_location_source_x = null;
    row.subject_location_source_y = null;
    row.human_confirmed = false;
    row.approximate_hidden_location = false;
    row.observation_supply = ["OUT_OF_FRAME_OR_LEFT_SCENE", "NOT_PRESENT"].includes(value) ? "NOT_APPLICABLE" : null;
    row.selected_candidate_ids = [];
    await saveDraft(); renderQuestion(); drawAll();
  });
  if (["VISIBLE_COMPLETE", "VISIBLE_PARTIAL"].includes(row.visibility)) {
    const note = document.createElement("div");
    note.className = "click-required";
    note.innerHTML = row.human_confirmed ? `<b>Subject ${subjectLetter(index)} location confirmed.</b> You may click again to correct it.` : `<b>Click Subject ${subjectLetter(index)} in the large frame.</b> Continue stays locked until the human point is recorded.`;
    ui.answerArea.appendChild(note);
    app.inputMode = "subject_location";
  } else if (row.visibility === "FULLY_OCCLUDED_EXPECTED_PRESENT") {
    const button = document.createElement("button");
    button.type = "button"; button.textContent = row.approximate_hidden_location ? "Approximate hidden location recorded — click to replace" : "Mark the approximate hidden location (optional)";
    button.onclick = () => { app.inputMode = "hidden_location"; button.textContent = "Now click the approximate hidden location in the large frame"; };
    ui.answerArea.appendChild(button);
  }
  ui.continueButton.disabled = !row.visibility || (["VISIBLE_COMPLETE", "VISIBLE_PARTIAL"].includes(row.visibility) && !row.human_confirmed);
}

function renderMarkerReview(index) {
  const subject = currentSubject(index);
  setQuestion(`Do these markers follow the same Subject ${subjectLetter(index)}?`, "Review all nine frames. Human-confirmed points are shown; absent and uncertain frames intentionally have no marker.", `SUBJECT ${subjectLetter(index)} MARKER REVIEW`);
  ui.answerArea.innerHTML = markerStrip(subject);
  answerCardsAppend([["SAME_SUBJECT_CONFIRMED", `These markers follow the same Subject ${subjectLetter(index)}`], ["CORRECT_FRAME", "I need to correct a frame"], ["CANNOT_TELL", "Cannot tell"]], subject.marker_continuity_confirmation, async (value) => {
    if (value === "CORRECT_FRAME") {
      subject.marker_continuity_confirmation = null;
      app.history.push(app.questionKey);
      app.questionKey = `subject_${index}_location_0`;
    } else subject.marker_continuity_confirmation = value;
    await saveDraft(); renderQuestion();
  });
  ui.continueButton.disabled = !subject.marker_continuity_confirmation;
}

function markerStrip(subject) {
  return `<div class="marker-review-strip">${subject.frame_observations.map((row, frame) => {
    const point = Number.isFinite(row.subject_location_source_x);
    return `<button type="button" class="marker-cell ${point ? "" : "empty"} ${app.frame === frame ? "current" : ""}" data-review-frame="${frame}"><b>${point ? subject.subject_token.slice(-1) : "—"}</b>F${frame + 1}<br>${LABELS[row.visibility] || "Unanswered"}</button>`;
  }).join("")}</div>`;
}

function answerCardsAppend(options, selected, onSelect) {
  const holder = document.createElement("div"); holder.className = "answer-area";
  options.forEach(([value, label], index) => {
    const button = document.createElement("button"); button.type = "button"; button.className = `answer-card${selected === value ? " selected" : ""}`; button.dataset.shortcut = String(index + 1); button.textContent = label; button.disabled = !reviewReady(); button.onclick = () => onSelect(value); holder.appendChild(button);
  });
  ui.answerArea.appendChild(holder);
  ui.answerArea.querySelectorAll("[data-review-frame]").forEach((button) => { button.onclick = () => loadFrame(Number(button.dataset.reviewFrame)); });
}

function renderSupply(index, frame) {
  setFrameForQuestion(frame);
  const subject = currentSubject(index);
  const row = subject.frame_observations[frame];
  setQuestion(`Which box evidence belongs to Subject ${subjectLetter(index)} in Frame ${frame + 1}?`, `Use the human-confirmed ${subjectLetter(index)} marker first. White rectangles are frozen model candidates available in this frame; they are not subject identity.`, `FRAME ${frame + 1} · CANDIDATE EVIDENCE`);
  answerCards(SUPPLY_VALUES, row.observation_supply, async (value) => {
    row.observation_supply = value;
    row.selected_candidate_ids = [];
    await saveDraft(); renderQuestion(); drawAll();
  });
  const candidates = frameCandidates(frame);
  const status = document.createElement("div");
  status.className = candidates.length ? "selection-status" : "candidate-empty";
  status.textContent = candidates.length ? `${row.selected_candidate_ids.length} of ${candidates.length} frozen candidate boxes selected for Subject ${subjectLetter(index)}.` : "No frozen model candidate box is available in this frame. Never infer or propagate a box from another frame.";
  ui.answerArea.appendChild(status);
  app.inputMode = "candidate";
  ui.continueButton.disabled = !row.observation_supply || (selectionRequired(row.observation_supply) && row.selected_candidate_ids.length === 0);
}

function renderRelationship(index) {
  const subject = currentSubject(index);
  setQuestion("How are the selected boxes related?", `This conditional question uses Subject ${subjectLetter(index)} markers and only the boxes you selected.`, "CONDITIONAL BOX RELATIONSHIP");
  const options = [["SAME_PERSON_DUPLICATES", "Duplicate boxes for the same person"], ["SAME_PERSON_FRAGMENTS", "Main box plus body fragments"], ["DIFFERENT_PEOPLE", "Different people"], ["CORRECT_INNER_BAD_OUTER", "Correct inner person inside a poor outer box"], ["MERGED_MULTI_PERSON", "One merged box covers several people"], ["OBJECT_OR_BACKGROUND", "Object or background fragment"], ["UNCERTAIN", "Not sure"]];
  answerCards(options, subject.candidate_relationship, async (value) => { subject.candidate_relationship = value; await saveDraft(); renderQuestion(); });
  ui.continueButton.disabled = !subject.candidate_relationship;
}

function renderOcclusion(index) {
  const subject = currentSubject(index);
  setQuestion("Does this overlap sequence look right?", "The suggested phases are editable helpers. Nothing becomes truth until you confirm.", "CONFIRM OCCLUSION ENTRY · MAINTAINED · EXIT");
  if (!subject.occlusion_confirmed) deriveOcclusion(subject);
  ui.answerArea.innerHTML = `<div class="marker-review-strip">${subject.frame_observations.map((row, frame) => `<button type="button" class="marker-cell"><b>${frame + 1}</b>${row.occlusion_phase.replaceAll("_", " ")}</button>`).join("")}</div><button id="confirmOcclusion" type="button">Confirm this sequence</button><button id="uncertainOcclusion" type="button">Mark sequence as not sure</button>`;
  $("confirmOcclusion").onclick = async () => { subject.occlusion_confirmed = true; await saveDraft(); renderQuestion(); };
  $("uncertainOcclusion").onclick = async () => { subject.frame_observations.forEach((row) => { row.occlusion_phase = "UNCERTAIN"; }); subject.occlusion_confirmed = true; await saveDraft(); renderQuestion(); };
  ui.continueButton.disabled = !subject.occlusion_confirmed;
}

function deriveOcclusion(subject) {
  subject.frame_observations.forEach((row, index, rows) => {
    if (row.visibility === "FULLY_OCCLUDED_EXPECTED_PRESENT") row.occlusion_phase = "OCCLUDED";
    else if (row.visibility === "VISIBLE_PARTIAL") row.occlusion_phase = rows.slice(0, index).some((prior) => prior.visibility === "FULLY_OCCLUDED_EXPECTED_PRESENT") ? "EXITING_OCCLUSION" : "ENTERING_OCCLUSION";
    else row.occlusion_phase = "NONE";
  });
}

function renderContinuity(index) {
  const subject = currentSubject(index);
  setQuestion(`Do your Subject ${subjectLetter(index)} markers appear to follow the same person throughout this clip?`, "Burst-local only. If you switched people, correct the locations before saving.", "BURST-LOCAL CONTINUITY");
  answerCards([["SAME_BURST_LOCAL_SUBJECT", "Yes — same person within this clip"], ["DIFFERENT_SUBJECT", "No — I switched to a different person"], ["CANNOT_TELL", "Cannot tell"]], subject.continuity, async (value) => { subject.continuity = value; await saveDraft(); renderQuestion(); });
  ui.continueButton.disabled = !subject.continuity || subject.continuity === "DIFFERENT_SUBJECT";
}

function renderSubjectSingle(index, kind) {
  const subject = currentSubject(index);
  const configs = {
    role: ["What is this person’s role?", [["OUTFIELD_PLAYER", "Player"], ["GOALKEEPER", "Goalkeeper"], ["RELEVANT_OFFICIAL", "Relevant match official"], ["OTHER_PERSON", "Other person"], ["UNKNOWN_ROLE", "Not sure"]]],
    participation: ["How are they taking part?", [["ACTIVE_IN_MATCH", "Active in the match"], ["WARMING_OR_INACTIVE", "Warming up or inactive"], ["NOT_PLAYER_OR_OFFICIAL", "Not a player or match official"], ["UNKNOWN_PARTICIPATION", "Not sure"]]],
    certainty: ["Overall, how sure are you about this subject?", [["CERTAIN", "Certain"], ["PROBABLE", "Probably"], ["NOT_SURE", "Not sure"]]],
  };
  const [title, options] = configs[kind];
  setQuestion(title, `This applies only to Subject ${subjectLetter(index)} inside this burst.`, `SUBJECT ${subjectLetter(index)} · ONE ANSWER`);
  answerCards(options, subject[kind], async (value) => { subject[kind] = value; await saveDraft(); renderQuestion(); });
  ui.continueButton.disabled = !subject[kind];
}

function renderAdditionalSubject() {
  setQuestion("Do you need to follow another person in this clip?", "Add one only when the evidence requires it. At most Subject A, B, and C are available.", "OPTIONAL BURST-LOCAL SUBJECT");
  answerCards([["CONTINUE", "No — continue"], ["ADD", `Yes — add Subject ${subjectLetter(app.data.subjects.length)}`], ["NOT_SURE", "Not sure — continue"]], app.data.answers.additional_subject, async (value) => {
    app.data.answers.additional_subject = value;
    if (value === "ADD") ensureSubjects(app.data.subjects.length + 1, "UNCERTAIN_HUMAN_SELECTION");
    await saveDraft(); renderQuestion();
  });
  ui.continueButton.disabled = !app.data.answers.additional_subject;
}

function renderMissedCheck() {
  setQuestion("Can you see any relevant person in this whole burst who has no useful model box?", "Review all nine frames. This includes a player, goalkeeper, or relevant match official.", "WHOLE-BURST MISSED-PERSON CHECK");
  answerCards([["NO", "No"], ["YES", "Yes — let me mark them"], ["NOT_SURE", "Not sure"]], app.data.answers.missed_check, async (value) => { app.data.answers.missed_check = value; if (value !== "YES") app.data.missed_person_marks = []; await saveDraft(); renderQuestion(); drawAll(); });
  ui.continueButton.disabled = !app.data.answers.missed_check;
}

function renderMissedMark() {
  setQuestion("Click the centre of each missed relevant person", "Choose a frame, zoom if useful, and click. Marks remain source-coordinate exact.", "SOURCE-COORDINATE MARKING");
  app.inputMode = "missed_mark";
  ui.answerArea.innerHTML = `<div class="click-required"><b>${app.data.missed_person_marks.length} missed ${app.data.missed_person_marks.length === 1 ? "person" : "people"} marked.</b><br>Click the large frame to add another.</div>${app.data.missed_person_marks.map((mark, index) => `<button type="button" data-remove-mark="${index}">Remove mark ${index + 1} · frame ${mark.frame_sequence + 1}</button>`).join("")}`;
  ui.answerArea.querySelectorAll("[data-remove-mark]").forEach((button) => { button.onclick = async () => { app.data.missed_person_marks.splice(Number(button.dataset.removeMark), 1); await saveDraft(); drawAll(); renderQuestion(); }; });
  ui.continueButton.disabled = app.data.missed_person_marks.length === 0;
}

function renderSummary() {
  setQuestion("Review this burst before saving", "Only the server acknowledgement makes the immutable event complete.", "PLAIN-LANGUAGE SUMMARY");
  const subjectRows = app.data.subjects.map((subject) => `<li><b>Subject ${subject.subject_token.slice(-1)}</b>: nine frame locations reviewed · ${subject.role?.replaceAll("_", " ") || "role not set"} · ${subject.certainty?.replaceAll("_", " ") || "certainty not set"}</li>`).join("");
  ui.answerArea.innerHTML = `<ul class="summary-list"><li><b>Yellow box:</b> ${app.data.answers.original_focus_box_answer?.replaceAll("_", " ")}</li>${subjectRows || "<li>No burst-local subject was followed.</li>"}<li><b>Whole-burst missed-person check:</b> ${app.data.answers.missed_check}</li><li><b>Missed-person marks:</b> ${app.data.missed_person_marks.length}</li></ul>`;
  ui.continueButton.disabled = false;
}

async function continueQuestion() {
  if (ui.continueButton.disabled || app.readOnly) return;
  if (app.questionKey === "summary") { await saveFinal(); return; }
  const sequence = questionSequence();
  const index = sequence.indexOf(app.questionKey);
  app.history.push(app.questionKey);
  app.questionKey = sequence[index + 1] || "summary";
  app.data.current_question = app.questionKey;
  await saveDraft();
  renderQuestion();
}
async function backQuestion() {
  if (!app.history.length || app.readOnly) return;
  app.questionKey = app.history.pop();
  app.data.current_question = app.questionKey;
  await saveDraft(); renderQuestion();
}

async function api(path, payload) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const body = await response.json();
  if (!response.ok || body.ok === false) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}
function draftPayload() {
  return { mode: app.mode, burst_id: app.current.burst_id, current_question: app.questionKey, current_frame_sequence: app.frame, playback_speed: app.speed, answers: app.data.answers, subjects: app.data.subjects, candidate_mappings: candidateMappings(), missed_person_marks: app.data.missed_person_marks };
}
async function saveDraft() {
  if (app.readOnly) return;
  ui.saveState.textContent = "Saving…";
  try { await api("/api/draft", draftPayload()); ui.saveState.textContent = "Progress saved to server"; }
  catch (error) { block(`DRAFT_SAVE_ERROR — ${error.message}`); throw error; }
}
function candidateMappings() {
  const result = [];
  app.data.subjects.forEach((subject) => subject.frame_observations.forEach((row, sequence) => row.selected_candidate_ids.forEach((candidateId) => {
    const candidate = frameCandidates(sequence).find((item) => item.candidate_id === candidateId);
    if (candidate) result.push({ subject_token: subject.subject_token, frame_sequence: sequence, frame_reference_id: app.current.frames[sequence].frame_reference_id, candidate_id: candidateId, source_box_xyxy: candidate.source_box_xyxy });
  })));
  return result;
}
function eventPayload() {
  return { mode: app.mode, burst_id: app.current.burst_id, original_focus_box_answer: app.data.answers.original_focus_box_answer, context_subject_answer: app.data.answers.context_subject_answer || "NOT_APPLICABLE", subjects: app.data.subjects, candidate_mappings: candidateMappings(), whole_burst_missed_person_answer: app.data.answers.missed_check, whole_burst_missed_person_marks: app.data.missed_person_marks, source_frame_hashes: app.current.frames.map((frame) => frame.source_frame_pixel_sha256), summary_confirmed: true };
}
async function saveFinal() {
  ui.continueButton.disabled = true; ui.continueButton.textContent = "Saving safely…"; ui.saveState.textContent = "Persisting immutable event…";
  try {
    const result = await api("/api/save", eventPayload());
    ui.saveState.textContent = `SAVED — SERVER ACKNOWLEDGED · ${result.event_id}`;
    await loadMode(app.mode, app.current.tranche_id);
  } catch (error) { block(`FINAL_SAVE_ERROR — ${error.message}`); }
}

async function sha256Blob(blob) {
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}
async function verifiedImage(url, expectedHash) {
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) throw new Error(`asset HTTP ${response.status}`);
  const blob = await response.blob();
  if (await sha256Blob(blob) !== expectedHash) throw new Error("derivative hash mismatch");
  const objectUrl = URL.createObjectURL(blob); const image = new Image();
  await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error("image decode failed")); image.src = objectUrl; });
  image._objectUrl = objectUrl;
  return image;
}

async function loadFrame(index, preservePlayback = false) {
  if (!preservePlayback) stopPlayback();
  const next = Math.max(0, Math.min(8, index));
  if (!ui.lockViewToggle.checked && next !== app.frame) resetViews();
  app.frame = next; app.data.current_frame_sequence = next; app.assetReady = false; app.mappingVerified = false;
  ui.mappingState.textContent = "CHECKING"; ui.assetState.classList.remove("hidden"); ui.focusAssetState.classList.remove("hidden");
  const frame = app.current.frames[next];
  try {
    const [panorama, focus] = await Promise.all([verifiedImage(frame.panorama_url, frame.panorama_sha256), verifiedImage(frame.focus_url, frame.focus_sha256)]);
    if (app.image?._objectUrl) URL.revokeObjectURL(app.image._objectUrl);
    if (app.focusImage?._objectUrl) URL.revokeObjectURL(app.focusImage._objectUrl);
    app.image = panorama; app.focusImage = focus;
    app.assetReady = panorama.naturalWidth > 0 && panorama.naturalHeight > 0 && focus.naturalWidth > 0 && focus.naturalHeight > 0;
    resizeCanvases();
    drawPanorama();
    drawFocus();
    app.mappingVerified = app.assetReady && verifyMapping();
    ui.mappingState.textContent = app.mappingVerified ? "VERIFIED" : "FAILED";
    pixelContentGate(ui.panoramaCanvas);
    if (!app.mappingVerified) throw new Error("source/display mapping verification failed");
    ui.assetState.classList.add("hidden"); ui.focusAssetState.classList.add("hidden");
    renderTimeline(); renderQuestion(); updateSubjectReference(activeSubjectIndex()); prefetchNext();
  } catch (error) { block(`ASSET_LOAD_ERROR — ${error.message}`); }
}

function viewMetrics(view, canvas, crop = null) {
  const rect = canvas.getBoundingClientRect();
  const frame = app.current.frames[app.frame];
  const bounds = crop || [0, 0, frame.source_width, frame.source_height];
  const width = bounds[2] - bounds[0], height = bounds[3] - bounds[1];
  const fit = Math.min(rect.width / width, rect.height / height);
  const scale = fit * view.zoom;
  const sourceCenterX = bounds[0] + view.centerX * width;
  const sourceCenterY = bounds[1] + view.centerY * height;
  return { rect, bounds, width, height, fit, scale, left: rect.width / 2 - (sourceCenterX - bounds[0]) * scale, top: rect.height / 2 - (sourceCenterY - bounds[1]) * scale, dpr: window.devicePixelRatio || 1 };
}
function sourceToDisplay(point, target = "panorama") {
  const focus = target === "focus"; const crop = focus ? app.current.focus_crop_source_xyxy : null; const metrics = viewMetrics(focus ? app.focusView : app.view, focus ? ui.focusCanvas : ui.panoramaCanvas, crop); const bounds = metrics.bounds;
  return [metrics.left + (point[0] - bounds[0]) * metrics.scale, metrics.top + (point[1] - bounds[1]) * metrics.scale];
}
function displayToSource(point, target = "panorama") {
  const focus = target === "focus"; const crop = focus ? app.current.focus_crop_source_xyxy : null; const metrics = viewMetrics(focus ? app.focusView : app.view, focus ? ui.focusCanvas : ui.panoramaCanvas, crop); const bounds = metrics.bounds;
  return [bounds[0] + (point[0] - metrics.left) / metrics.scale, bounds[1] + (point[1] - metrics.top) / metrics.scale];
}
function verifyMapping() {
  const frame = app.current.frames[app.frame];
  const points = [[0, 0], [frame.source_width / 2, frame.source_height / 2], [frame.source_width, frame.source_height]];
  for (const point of points) {
    const restored = displayToSource(sourceToDisplay(point));
    if (Math.max(Math.abs(restored[0] - point[0]), Math.abs(restored[1] - point[1])) > 0.5) return false;
  }
  const crop = app.current.focus_crop_source_xyxy;
  for (const point of [[crop[0], crop[1]], [(crop[0] + crop[2]) / 2, (crop[1] + crop[3]) / 2], [crop[2], crop[3]]]) {
    const restored = displayToSource(sourceToDisplay(point, "focus"), "focus");
    if (Math.max(Math.abs(restored[0] - point[0]), Math.abs(restored[1] - point[1])) > 0.5) return false;
  }
  return true;
}
function pixelContentGate(canvas) {
  const context = canvas.getContext("2d"); const width = Math.min(canvas.width, 160), height = Math.min(canvas.height, 96); const x = Math.max(0, Math.floor((canvas.width - width) / 2)), y = Math.max(0, Math.floor((canvas.height - height) / 2)); const sample = context.getImageData(x, y, width, height).data;
  let min = 765, max = 0, nonBlack = 0;
  for (let index = 0; index < sample.length; index += 4) { const value = sample[index] + sample[index + 1] + sample[index + 2]; min = Math.min(min, value); max = Math.max(max, value); if (value > 48) nonBlack += 1; }
  if (max - min < 24 || nonBlack < sample.length / 32) throw new Error("blank rendered football frame");
}
function resizeCanvas(canvas, wrap) {
  const rect = wrap.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr)); canvas.height = Math.max(1, Math.round(rect.height * dpr)); canvas.style.width = `${rect.width}px`; canvas.style.height = `${rect.height}px`;
}
function resizeCanvases() { resizeCanvas(ui.panoramaCanvas, ui.panoramaWrap); resizeCanvas(ui.focusCanvas, ui.focusWrap); drawAll(); }
function requestDraw() {
  if (app.drawQueued) return;
  app.drawQueued = true;
  requestAnimationFrame((timestamp) => { const start = performance.now(); app.drawQueued = false; drawPanorama(); drawFocus(); app.performance.push({ timestamp, duration_ms: performance.now() - start }); if (app.performance.length > 200) app.performance.shift(); });
}
function drawAll() { requestDraw(); }
function canvasContext(canvas, metrics) { const context = canvas.getContext("2d"); context.setTransform(metrics.dpr, 0, 0, metrics.dpr, 0, 0); context.clearRect(0, 0, metrics.rect.width, metrics.rect.height); context.fillStyle = "#080e1d"; context.fillRect(0, 0, metrics.rect.width, metrics.rect.height); return context; }
function drawPanorama() {
  if (!app.image || !app.current) return;
  const metrics = viewMetrics(app.view, ui.panoramaCanvas); const context = canvasContext(ui.panoramaCanvas, metrics); const frame = app.current.frames[app.frame];
  context.drawImage(app.image, metrics.left, metrics.top, frame.source_width * metrics.scale, frame.source_height * metrics.scale);
  drawOverlays(context, "panorama"); ui.zoomPercent.textContent = `${Math.round(app.view.zoom * 100)}%`;
}
function drawFocus() {
  if (!app.focusImage || !app.current) return;
  const crop = app.current.focus_crop_source_xyxy; const metrics = viewMetrics(app.focusView, ui.focusCanvas, crop); const context = canvasContext(ui.focusCanvas, metrics);
  context.drawImage(app.focusImage, metrics.left, metrics.top, metrics.width * metrics.scale, metrics.height * metrics.scale);
  drawOverlays(context, "focus"); ui.focusZoomPercent.textContent = `${Math.round(app.focusView.zoom * 100)}%`;
}
function drawOverlays(context, target) {
  const crop = target === "focus" ? app.current.focus_crop_source_xyxy : null;
  const visible = (box) => !crop || !(box[2] < crop[0] || box[0] > crop[2] || box[3] < crop[1] || box[1] > crop[3]);
  drawBox(context, app.current.focus_crop_source_xyxy, "#55a9ff", [10, 7], "CONTEXT AREA", target);
  if (app.frame === 4) app.current.candidates.filter((candidate) => visible(candidate.source_box_xyxy)).forEach((candidate) => drawBox(context, candidate.source_box_xyxy, "#ffd329", [], "ORIGINAL FOCUS CANDIDATE", target));
  if (ui.overlayToggle.checked) frameCandidates().filter((candidate) => visible(candidate.source_box_xyxy)).forEach((candidate) => drawBox(context, candidate.source_box_xyxy, "#f7f9ff", [], ui.idToggle.checked ? candidate.candidate_id : "", target, 2));
  if (ui.subjectToggle.checked && app.data) app.data.subjects.forEach((subject, index) => { const row = subject.frame_observations[app.frame]; if (Number.isFinite(row.subject_location_source_x)) drawSubjectMarker(context, [row.subject_location_source_x, row.subject_location_source_y], index, row.approximate_hidden_location, target); });
  if (target === "panorama") app.data?.missed_person_marks.filter((mark) => mark.frame_sequence === app.frame).forEach((mark, index) => drawNumberMark(context, mark.source_xy, index + 1, target));
}
function drawBox(context, box, colour, dash, label, target, width = 4) {
  const a = sourceToDisplay([box[0], box[1]], target), b = sourceToDisplay([box[2], box[3]], target); context.save(); context.strokeStyle = colour; context.lineWidth = width; context.setLineDash(dash); context.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
  if (label) { context.setLineDash([]); context.font = "700 12px Segoe UI"; const labelWidth = Math.max(88, context.measureText(label).width + 16); context.fillStyle = "#111a33e8"; context.fillRect(a[0], Math.max(0, a[1] - 24), labelWidth, 23); context.fillStyle = "#fff"; context.fillText(label, a[0] + 7, Math.max(16, a[1] - 8)); }
  context.restore();
}
function drawSubjectMarker(context, point, index, approximate, target) {
  const p = sourceToDisplay(point, target); context.save(); context.fillStyle = approximate ? "#fff" : SUBJECT_COLOURS[index]; context.strokeStyle = SUBJECT_COLOURS[index]; context.lineWidth = 4; if (approximate) context.setLineDash([5, 4]); context.beginPath(); context.arc(p[0], p[1], 16, 0, Math.PI * 2); context.fill(); context.stroke(); context.setLineDash([]); context.fillStyle = approximate ? SUBJECT_COLOURS[index] : "#111a33"; context.font = "900 14px Segoe UI"; context.textAlign = "center"; context.fillText(subjectLetter(index), p[0], p[1] + 5); context.restore();
}
function drawNumberMark(context, point, number, target) { const p = sourceToDisplay(point, target); context.save(); context.fillStyle = "#e96767"; context.strokeStyle = "#fff"; context.lineWidth = 3; context.beginPath(); context.arc(p[0], p[1], 13, 0, Math.PI * 2); context.fill(); context.stroke(); context.fillStyle = "#fff"; context.font = "bold 12px Arial"; context.textAlign = "center"; context.fillText(String(number), p[0], p[1] + 4); context.restore(); }

function setZoom(target, zoom, anchor = null) {
  const focus = target === "focus", view = focus ? app.focusView : app.view, canvas = focus ? ui.focusCanvas : ui.panoramaCanvas, crop = focus ? app.current.focus_crop_source_xyxy : null;
  const metrics = viewMetrics(view, canvas, crop); const point = anchor || [metrics.rect.width / 2, metrics.rect.height / 2]; const source = displayToSource(point, target); const nextZoom = Math.max(1, Math.min(12, zoom)); const bounds = metrics.bounds; const width = metrics.width, height = metrics.height; const nextScale = metrics.fit * nextZoom;
  view.zoom = nextZoom; view.centerX = Math.max(0, Math.min(1, (source[0] - bounds[0] - (point[0] - metrics.rect.width / 2) / nextScale) / width)); view.centerY = Math.max(0, Math.min(1, (source[1] - bounds[1] - (point[1] - metrics.rect.height / 2) / nextScale) / height)); requestDraw();
}
function resetViews() { app.view = { zoom: 1, centerX: 0.5, centerY: 0.5 }; app.focusView = { zoom: 1, centerX: 0.5, centerY: 0.5 }; requestDraw(); }
function zoomToSubject(target = "panorama") {
  const subject = currentSubject(); if (!subject) return;
  const row = subject.frame_observations[app.frame]; const point = Number.isFinite(row.subject_location_source_x) ? [row.subject_location_source_x, row.subject_location_source_y] : subject.anchor_source_xy; if (!point) return;
  const focus = target === "focus", view = focus ? app.focusView : app.view, bounds = focus ? app.current.focus_crop_source_xyxy : [0, 0, app.current.source_width, app.current.source_height]; view.centerX = Math.max(0, Math.min(1, (point[0] - bounds[0]) / (bounds[2] - bounds[0]))); view.centerY = Math.max(0, Math.min(1, (point[1] - bounds[1]) / (bounds[3] - bounds[1]))); view.zoom = Math.max(view.zoom, 4); requestDraw();
}
function installViewport(canvas, target) {
  canvas.addEventListener("pointerdown", (event) => { const rect = canvas.getBoundingClientRect(); canvas.focus(); app.pointer = { target, pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, lastX: event.clientX, lastY: event.clientY, moved: false, rect }; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", (event) => {
    if (!app.pointer || app.pointer.pointerId !== event.pointerId || app.pointer.target !== target) return;
    const dx = event.clientX - app.pointer.lastX, dy = event.clientY - app.pointer.lastY; if (Math.hypot(event.clientX - app.pointer.startX, event.clientY - app.pointer.startY) > 4) app.pointer.moved = true;
    if (app.pointer.moved) { const focus = target === "focus", view = focus ? app.focusView : app.view, metrics = viewMetrics(view, canvas, focus ? app.current.focus_crop_source_xyxy : null); view.centerX = Math.max(0, Math.min(1, view.centerX - dx / (metrics.scale * metrics.width))); view.centerY = Math.max(0, Math.min(1, view.centerY - dy / (metrics.scale * metrics.height))); requestDraw(); }
    app.pointer.lastX = event.clientX; app.pointer.lastY = event.clientY;
  });
  canvas.addEventListener("pointerup", async (event) => { if (!app.pointer || app.pointer.pointerId !== event.pointerId || app.pointer.target !== target) return; const pointer = app.pointer; app.pointer = null; if (pointer.moved || target === "focus") return; const rect = canvas.getBoundingClientRect(); const source = displayToSource([event.clientX - rect.left, event.clientY - rect.top], target); await handleSourceClick(source); });
  canvas.addEventListener("wheel", (event) => { event.preventDefault(); const rect = canvas.getBoundingClientRect(); const view = target === "focus" ? app.focusView : app.view; setZoom(target, view.zoom * (event.deltaY < 0 ? 1.2 : 1 / 1.2), [event.clientX - rect.left, event.clientY - rect.top]); }, { passive: false });
}
async function handleSourceClick(source) {
  const frame = app.current.frames[app.frame];
  if (source[0] < 0 || source[1] < 0 || source[0] > frame.source_width || source[1] > frame.source_height) return;
  const rounded = source.map((value) => Number(value.toFixed(3)));
  if (app.inputMode === "anchor") { const subject = currentSubject(app.inputSubject); subject.anchor_frame_sequence = app.frame; subject.anchor_source_xy = rounded; await saveDraft(); updateSubjectReference(app.inputSubject); renderQuestion(); drawAll(); return; }
  if (["subject_location", "hidden_location"].includes(app.inputMode)) { const subject = currentSubject(app.inputSubject); const row = subject.frame_observations[app.frame]; row.subject_location_source_x = rounded[0]; row.subject_location_source_y = rounded[1]; row.human_confirmed = app.inputMode === "subject_location"; row.approximate_hidden_location = app.inputMode === "hidden_location"; await saveDraft(); renderQuestion(); drawAll(); return; }
  if (app.inputMode === "candidate") { await toggleCandidate(source, app.inputSubject); return; }
  if (app.inputMode === "missed_mark") { app.data.missed_person_marks.push({ mark_id: crypto.randomUUID(), frame_reference_id: frame.frame_reference_id, frame_sequence: app.frame, source_xy: rounded, role: "UNKNOWN_ROLE", certainty: "NOT_SURE" }); await saveDraft(); renderQuestion(); drawAll(); }
}
async function toggleCandidate(source, subjectIndex) {
  const candidates = frameCandidates(); const hits = candidates.filter((candidate) => { const box = candidate.source_box_xyxy; return source[0] >= box[0] && source[0] <= box[2] && source[1] >= box[1] && source[1] <= box[3]; }).sort((a, b) => boxArea(a.source_box_xyxy) - boxArea(b.source_box_xyxy));
  if (!hits.length) return;
  const row = currentSubject(subjectIndex).frame_observations[app.frame]; const id = hits[0].candidate_id; const position = row.selected_candidate_ids.indexOf(id); if (position >= 0) row.selected_candidate_ids.splice(position, 1); else row.selected_candidate_ids.push(id); await saveDraft(); renderQuestion(); drawAll();
}
function boxArea(box) { return (box[2] - box[0]) * (box[3] - box[1]); }

function renderTimeline() {
  ui.timeline.innerHTML = "";
  app.current.frames.forEach((frame, index) => { const button = document.createElement("button"); button.type = "button"; button.className = `frame-thumb ${index === app.frame ? "current" : ""} ${index === 4 ? "centre" : ""}`; const subject = currentSubject(); const row = subject?.frame_observations[index]; button.innerHTML = `<img src="${frame.panorama_url}" alt="Frame ${index + 1}" loading="${Math.abs(index - app.frame) > 2 ? "lazy" : "eager"}"><span>${row?.human_confirmed ? `${subject.subject_token.slice(-1)} ✓ · ` : ""}${frame.relative_offset_seconds > 0 ? "+" : ""}${frame.relative_offset_seconds.toFixed(1)}s</span>`; button.onclick = () => loadFrame(index); ui.timeline.appendChild(button); });
}
function updateSubjectReference(index) {
  const subject = app.data?.subjects[index];
  ui.subjectReference.classList.toggle("hidden", !subject?.anchor_source_xy);
  if (!subject?.anchor_source_xy || !app.focusImage) return;
  const letter = subjectLetter(index); ui.subjectReferenceToken.textContent = letter; ui.subjectReferenceToken.style.background = SUBJECT_COLOURS[index]; ui.subjectReferenceTitle.textContent = `Subject ${letter} reference`; ui.subjectReferenceMeta.textContent = `Human anchor · frame ${subject.anchor_frame_sequence + 1} · role ${subject.role ? subject.role.replaceAll("_", " ").toLowerCase() : "not asked yet"}`;
  const canvas = ui.subjectReferenceCanvas, context = canvas.getContext("2d"), crop = app.current.focus_crop_source_xyxy, anchor = subject.anchor_source_xy; context.fillStyle = "#080e1d"; context.fillRect(0, 0, canvas.width, canvas.height); const radiusX = 80, radiusY = 56; const sx = Math.max(0, anchor[0] - crop[0] - radiusX), sy = Math.max(0, anchor[1] - crop[1] - radiusY), sw = Math.min(app.focusImage.naturalWidth - sx, radiusX * 2), sh = Math.min(app.focusImage.naturalHeight - sy, radiusY * 2); if (sw > 0 && sh > 0) context.drawImage(app.focusImage, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height); context.strokeStyle = SUBJECT_COLOURS[index]; context.lineWidth = 3; context.beginPath(); context.arc(canvas.width / 2, canvas.height / 2, 11, 0, Math.PI * 2); context.stroke(); context.fillStyle = SUBJECT_COLOURS[index]; context.font = "900 12px Segoe UI"; context.fillText(letter, canvas.width / 2 + 14, canvas.height / 2 + 4);
  ui.zoomSubjectButton.textContent = `Zoom to Subject ${letter}`;
}

function stopPlayback() { if (app.timer) clearTimeout(app.timer); app.timer = null; app.playing = false; ui.playButton.textContent = "Play"; }
function schedulePlayback() { if (!app.playing) return; app.timer = setTimeout(async () => { await loadFrame((app.frame + 1) % 9, true); schedulePlayback(); }, 200 / app.speed); }
function togglePlayback() { if (app.playing) { stopPlayback(); return; } app.playing = true; ui.playButton.textContent = "Pause"; schedulePlayback(); }
function prefetchNext() { if (app.mode !== "real") return; const index = app.cases.findIndex((item) => item.burst_id === app.current.burst_id), next = app.cases[index + 1]; if (!next) return; const link = document.createElement("link"); link.rel = "prefetch"; link.href = next.frames[4].panorama_url; document.head.appendChild(link); }

async function loadCase(caseRow, draft = null) {
  app.current = caseRow; app.frame = draft?.current_frame_sequence ?? 4; app.speed = draft?.playback_speed ?? 1; app.data = draft ? structuredClone(draft) : blankData(caseRow); app.questionKey = draft?.current_question || "original_focus"; app.history = []; app.readOnly = false; resetViews();
  ui.caseEyebrow.textContent = `MATCH ${caseRow.match_id} · ${caseRow.half.replaceAll("_", " ")} · ${caseRow.frames[4].resolved_timestamp_seconds.toFixed(2)} SECONDS`;
  ui.caseTitle.textContent = `Burst ${caseRow.tranche_position || caseRow.practice_position} · nine-frame review`;
  ui.reviewShell.classList.remove("hidden"); ui.welcomeScreen.classList.add("hidden"); ui.completionScreen.classList.add("hidden"); updateProgress(); renderQuestion(); renderTimeline(); resizeCanvases(); await loadFrame(app.frame);
}
async function loadMode(mode, tranche = null) {
  app.mode = mode; stopPlayback(); ui.saveState.textContent = "Loading server state…";
  const response = await fetch(`/api/bootstrap?mode=${mode}${tranche ? `&tranche=${tranche}` : ""}`); const payload = await response.json(); if (!response.ok) throw new Error(payload.error || "bootstrap failed");
  if (payload.state.review_revision !== REVISION) throw new Error(`review revision mismatch: ${payload.state.review_revision}`);
  app.allCases = payload.cases; app.serverState = payload.state; ui.practiceBanner.classList.toggle("hidden", mode !== "practice"); ui.modePill.textContent = mode === "practice" ? "Practice · isolated" : `${payload.state.tranche_id?.replace("_", " ")} of 6`;
  if (payload.state.incompatible_draft) { ui.legacyDraftNotice.classList.remove("hidden"); ui.reviewShell.classList.add("hidden"); ui.welcomeScreen.classList.remove("hidden"); ui.saveState.textContent = "Old practice draft rejected safely"; return; }
  ui.legacyDraftNotice.classList.add("hidden");
  if (mode === "real" && payload.state.tranche_complete) { renderCompletion(payload.state); return; }
  if (mode === "practice" && payload.state.all_practice_complete) { renderPracticeComplete(); return; }
  app.cases = mode === "practice" ? payload.cases : payload.cases.filter((item) => item.tranche_id === payload.state.tranche_id);
  const id = payload.state.first_incomplete_burst_id || app.cases[0].burst_id; const row = app.cases.find((item) => item.burst_id === id); await loadCase(row, payload.state.draft); ui.saveState.textContent = payload.state.draft ? "Draft restored from server" : "Ready · server-backed";
}
function updateProgress() { const state = app.serverState, done = state?.completed_count || 0, total = state?.total_count || (app.mode === "practice" ? 3 : 20); ui.progressText.textContent = `${done} of ${total}`; ui.progressFill.style.width = `${100 * done / total}%`; }
function renderCompletion(state) { ui.reviewShell.classList.add("hidden"); ui.welcomeScreen.classList.add("hidden"); ui.completionScreen.classList.remove("hidden"); const number = Number(state.tranche_id.split("_")[1]); ui.modePill.textContent = `Tranche ${number} complete`; ui.progressText.textContent = "20 of 20"; ui.progressFill.style.width = "100%"; ui.completionTitle.textContent = `TRANCHE ${number} COMPLETE`; ui.trancheReceipt.textContent = state.tranche_completion_receipt_id; ui.lastEvent.textContent = state.last_event_id; ui.globalReceiptRow.classList.toggle("hidden", !state.all_cases_complete); ui.globalReceipt.textContent = state.global_completion_receipt_id || "—"; ui.nextTrancheButton.classList.toggle("hidden", number === 6); ui.nextTrancheButton.textContent = `Start Tranche ${number + 1}`; ui.saveState.textContent = "Completion restored from server"; }
function renderPracticeComplete() { ui.reviewShell.classList.add("hidden"); ui.welcomeScreen.classList.remove("hidden"); ui.completionScreen.classList.add("hidden"); ui.startPracticeButton.textContent = "Practice complete — reset or try again"; ui.saveState.textContent = "Practice saved outside human truth"; }
function block(message) { stopPlayback(); ui.blockingError.textContent = message; ui.blockingError.classList.remove("hidden"); ui.continueButton.disabled = true; ui.saveState.textContent = "Stopped safely"; }

ui.continueButton.onclick = continueQuestion; ui.backButton.onclick = backQuestion; ui.playButton.onclick = togglePlayback;
ui.firstFrameButton.onclick = () => loadFrame(0); ui.previousFrameButton.onclick = () => loadFrame(app.frame - 1); ui.nextFrameButton.onclick = () => loadFrame(app.frame + 1); ui.centreFrameButton.onclick = () => loadFrame(4); ui.lastFrameButton.onclick = () => loadFrame(8);
ui.fitButton.onclick = () => { app.view = { zoom: 1, centerX: 0.5, centerY: 0.5 }; requestDraw(); }; ui.resetViewButton.onclick = ui.fitButton.onclick; ui.zoomInButton.onclick = () => setZoom("panorama", app.view.zoom * 1.25); ui.zoomOutButton.onclick = () => setZoom("panorama", app.view.zoom / 1.25); ui.zoomSubjectButton.onclick = () => zoomToSubject("panorama"); ui.fullScreenButton.onclick = () => ui.evidenceCard.requestFullscreen();
ui.focusFitButton.onclick = () => { app.focusView = { zoom: 1, centerX: 0.5, centerY: 0.5 }; requestDraw(); }; ui.focusResetButton.onclick = ui.focusFitButton.onclick; ui.focusZoomInButton.onclick = () => setZoom("focus", app.focusView.zoom * 1.25); ui.focusZoomOutButton.onclick = () => setZoom("focus", app.focusView.zoom / 1.25); ui.focusZoomSubjectButton.onclick = () => zoomToSubject("focus");
[ui.overlayToggle, ui.subjectToggle, ui.idToggle].forEach((control) => { control.onchange = requestDraw; });
document.querySelectorAll("[data-speed]").forEach((button) => { button.onclick = () => { app.speed = Number(button.dataset.speed); document.querySelectorAll("[data-speed]").forEach((item) => item.classList.toggle("active", item === button)); if (app.playing) { stopPlayback(); togglePlayback(); } saveDraft(); }; });
installViewport(ui.panoramaCanvas, "panorama"); installViewport(ui.focusCanvas, "focus");
ui.startRealButton.onclick = () => loadMode("real").catch((error) => block(`BOOT_ERROR — ${error.message}`)); ui.startPracticeButton.onclick = () => loadMode("practice").catch((error) => block(`PRACTICE_BOOT_ERROR — ${error.message}`));
async function resetPractice() { await api("/api/practice/reset", { mode: "practice" }); ui.legacyDraftNotice.classList.add("hidden"); await loadMode("practice"); }
ui.resetPractice.onclick = resetPractice; ui.resetLegacyPractice.onclick = resetPractice;
ui.helpButton.onclick = () => { ui.helpDrawer.classList.add("open"); ui.helpDrawer.setAttribute("aria-hidden", "false"); }; ui.closeHelp.onclick = () => { ui.helpDrawer.classList.remove("open"); ui.helpDrawer.setAttribute("aria-hidden", "true"); };
ui.nextTrancheButton.onclick = () => { const number = Number(app.serverState.tranche_id.split("_")[1]); ui.confirmTitle.textContent = `Begin Tranche ${number + 1}?`; ui.confirmDialog.showModal(); };
ui.confirmDialog.addEventListener("close", async () => { if (ui.confirmDialog.returnValue !== "confirm") return; const result = await api("/api/tranche/start-next", { tranche_id: app.serverState.tranche_id }); await loadMode("real", result.next_tranche_id); });
ui.reviewCompletedButton.onclick = async () => { const response = await fetch(`/api/completed?tranche=${app.serverState.tranche_id}`), payload = await response.json(); if (!response.ok) throw new Error(payload.error || "completed view unavailable"); app.readOnly = true; ui.saveState.textContent = `${payload.events.length} completed answers loaded read-only`; ui.reviewCompletedButton.disabled = true; };
window.addEventListener("resize", resizeCanvases); document.addEventListener("fullscreenchange", resizeCanvases);
document.addEventListener("keydown", (event) => {
  if (event.key.toLowerCase() === "h") { ui.helpButton.click(); return; }
  if (event.code === "Space" && app.current) { event.preventDefault(); togglePlayback(); return; }
  if (event.key === "ArrowLeft" && app.current) loadFrame(app.frame - 1); if (event.key === "ArrowRight" && app.current) loadFrame(app.frame + 1);
  if ((event.key === "+" || event.key === "=") && document.activeElement === ui.panoramaCanvas) { event.preventDefault(); setZoom("panorama", app.view.zoom * 1.25); }
  if (event.key === "-" && document.activeElement === ui.panoramaCanvas) { event.preventDefault(); setZoom("panorama", app.view.zoom / 1.25); }
  const number = Number(event.key); if (number >= 1 && number <= 9) { const button = ui.answerArea.querySelector(`[data-shortcut="${number}"]`); if (button) button.click(); }
});

window.__G7E_B_R1__ = { app, loadMode, loadCase, loadFrame, renderQuestion, continueQuestion, saveDraft, sourceToDisplay, displayToSource, setZoom, zoomToSubject, resetViews, verifyMapping, eventPayload, frameCandidates, updateSubjectReference, requestDraw, questionSequence, chooseOriginalFocus, chooseContext, chooseUncertainPath };
const params = new URLSearchParams(location.search); if (params.get("preview") === "1") ui.previewBanner.classList.remove("hidden"); if (params.get("autostart") === "1") loadMode(params.get("mode") || "practice").catch((error) => block(`BOOT_ERROR — ${error.message}`));
