"use strict";

const REVIEW_ID = "G7D_C1_TARGETED_VISUAL_TRANSFER_DIAGNOSIS";
const REVISION = "G7D_C1_R1_NOVICE_GUIDED_VISUAL_DIAGNOSIS_REVIEW_V1";
const EVENT_SCHEMA = "football_intelligence.g7d_c1.human_visual_diagnosis_event.v1";
const DRAFT_SCHEMA = "football_intelligence.g7d_c1_r1.server_progress_draft.v1";
const PROGRESS_SAVED = "Progress saved";
const $ = (selector) => document.querySelector(selector);

const candidateLabels = {
  proposal_validity: {
    CLEAN_SINGLE_PERSON: "One person",
    MERGES_MULTIPLE_PEOPLE: "More than one person",
    NO_PERSON_BACKGROUND_OR_OBJECT: "No person",
    DUPLICATE_OF_ANOTHER_CANDIDATE: "Same person as another box",
    UNCERTAIN: "Not sure",
  },
  role: {
    OUTFIELD_PLAYER: "Outfield player",
    GOALKEEPER: "Goalkeeper",
    REFEREE: "Referee",
    OTHER_OFFICIAL: "Other match official",
    STAFF_OR_SPECTATOR: "Staff or spectator",
    UNKNOWN_PERSON_ROLE: "Not sure",
  },
  team: { TEAM_1: "Team 1", TEAM_2: "Team 2", UNKNOWN_TEAM: "Can't tell" },
  participation: { ACTIVE: "Yes, playing", WARMING_UP: "Warming up", NON_PLAYER: "Not playing", UNKNOWN: "Can't tell" },
  pitch_state: { ON_PITCH: "On the pitch", OFF_PITCH: "Off the pitch", BOUNDARY: "On the line", UNCERTAIN: "Can't tell" },
  occlusion: {
    NONE: "Fully visible",
    PARTIAL: "Partly blocked",
    SEVERE: "Mostly hidden",
    FULLY_OCCLUDED_PERSON_EXPECTED_HERE: "A person should be here but is hidden",
    UNCERTAIN: "Can't tell",
  },
  box_quality: {
    GOOD_SINGLE_PERSON_BOX: "Good fit",
    TOO_LOOSE: "Too big",
    TOO_TIGHT_OR_TRUNCATED: "Too small or cuts them off",
    MISLOCALIZED: "Wrong place",
    UNCERTAIN: "Can't tell",
  },
  certainty: { CERTAIN: "Sure", PROBABLE: "Probably", UNCERTAIN: "Not sure" },
};

const questionBank = {
  inside: {
    field: "proposal_validity",
    title: "What is inside the highlighted box?",
    hint: "Look at the yellow box in both pictures. Choose the best simple description.",
    choices: [
      ["CLEAN_SINGLE_PERSON", "One person", "The box mostly covers one person."],
      ["MERGES_MULTIPLE_PEOPLE", "More than one person", "The same box covers two or more people."],
      ["NO_PERSON_BACKGROUND_OR_OBJECT", "No person", "It is grass, equipment, background or another object."],
      ["DUPLICATE_OF_ANOTHER_CANDIDATE", "Same person as another box", "Another box in this frame shows this same person."],
      ["UNCERTAIN", "Not sure", "The picture is too unclear to decide."],
    ],
  },
  role: {
    field: "role",
    title: "What kind of person is this?",
    hint: "Use their kit and what they are doing. Choose Not sure if the view is unclear.",
    choices: [
      ["OUTFIELD_PLAYER", "Outfield player", "A player who is not the goalkeeper."],
      ["GOALKEEPER", "Goalkeeper", "A goalkeeper, usually in a different kit."],
      ["REFEREE", "Referee", "The main referee."],
      ["OTHER_OFFICIAL", "Other match official", "For example, an assistant referee."],
      ["STAFF_OR_SPECTATOR", "Staff or spectator", "A coach, staff member or person watching."],
      ["UNKNOWN_PERSON_ROLE", "Not sure", "The person's role is unclear."],
    ],
  },
  team: {
    field: "team",
    title: "Which team are they on?",
    hint: "Match the person's main shirt colour. Team numbers belong only to this match.",
    choices: [],
  },
  participation: {
    field: "participation",
    title: "Are they taking part in the match right now?",
    hint: "Playing means they are currently on the pitch as part of the match.",
    choices: [
      ["ACTIVE", "Yes, playing", "They are currently playing."],
      ["WARMING_UP", "Warming up", "They are preparing beside the pitch."],
      ["NON_PLAYER", "Not playing", "They are not currently taking part."],
      ["UNKNOWN", "Can't tell", "The picture does not make this clear."],
    ],
  },
  pitch: {
    field: "pitch_state",
    title: "Where are their feet?",
    hint: "Judge the ground under their feet, not where most of their body appears.",
    choices: [
      ["ON_PITCH", "On the pitch", "Their feet are inside the playing area."],
      ["OFF_PITCH", "Off the pitch", "Their feet are outside the playing area."],
      ["BOUNDARY", "On the line", "Their feet touch or sit very close to the boundary."],
      ["UNCERTAIN", "Can't tell", "Their feet or the line are unclear."],
    ],
  },
  occlusion: {
    field: "occlusion",
    title: "How clearly can you see them?",
    hint: "Think about how much of the person is blocked by another person or object.",
    choices: [
      ["NONE", "Fully visible", "Almost all of the person can be seen."],
      ["PARTIAL", "Partly blocked", "A smaller part of them is hidden."],
      ["SEVERE", "Mostly hidden", "Most of them is hidden."],
      ["FULLY_OCCLUDED_PERSON_EXPECTED_HERE", "A person should be here but is hidden", "The scene clearly suggests a person, though they cannot be seen."],
      ["UNCERTAIN", "Can't tell", "The image is too unclear."],
    ],
  },
  box: {
    field: "box_quality",
    title: "How well does the box fit the person?",
    hint: "A good box includes the whole person with little empty space.",
    choices: [
      ["GOOD_SINGLE_PERSON_BOX", "Good fit", "The box fits one whole person closely."],
      ["TOO_LOOSE", "Too big", "There is lots of extra space around them."],
      ["TOO_TIGHT_OR_TRUNCATED", "Too small or cuts them off", "The box misses part of their body."],
      ["MISLOCALIZED", "Wrong place", "The box is shifted away from the person."],
      ["UNCERTAIN", "Can't tell", "The fit is hard to judge."],
    ],
  },
  certainty: {
    field: "certainty",
    title: "How sure are you?",
    hint: "This is about your whole answer for this highlighted box.",
    choices: [
      ["CERTAIN", "Sure", "The picture gives clear evidence."],
      ["PROBABLE", "Probably", "This is likely, but not completely clear."],
      ["UNCERTAIN", "Not sure", "The picture is too unclear for confidence."],
    ],
  },
};

const sceneQuestions = {
  missed: {
    field: "missed_answer",
    title: "Can you see anyone important who has no useful box?",
    hint: "Look across the whole frame for a player, goalkeeper or referee without a useful box.",
    choices: [["NO", "No", "Every clearly visible relevant person has a useful box."], ["YES", "Yes, let me mark them", "I can see at least one missed person."], ["UNCERTAIN", "Not sure", "The frame is too unclear to be certain."]],
  },
  off_pitch: {
    field: "off_pitch_proposal_burden",
    title: "How many boxes looked useless or were outside the action?",
    hint: "Think about boxes on background, objects or people who are not part of the action.",
    choices: [["LOW", "Very few", "Almost none."], ["MODERATE", "Some", "A noticeable number."], ["HIGH", "A lot", "Many boxes."], ["UNCERTAIN", "Not sure", "Hard to judge."]],
  },
  duplicate: {
    field: "duplicate_or_overlap_burden",
    title: "How often did boxes overlap or repeat the same person?",
    hint: "Count obvious repeated or strongly overlapping boxes.",
    choices: [["LOW", "Never or almost never", "None or very few."], ["MODERATE", "Sometimes", "A noticeable number."], ["HIGH", "Often", "This happened many times."], ["UNCERTAIN", "Not sure", "Hard to judge."]],
  },
  hidden: {
    field: "occlusion_burden",
    title: "How much were people hidden behind each other?",
    hint: "Think about the whole scene, especially crowded groups.",
    choices: [["NONE", "Not at all", "People are separate and clear."], ["LOW", "A little", "A few people overlap."], ["MODERATE", "Quite a lot", "Several people are blocked."], ["HIGH", "A lot", "Crowding makes many people hard to see."], ["UNCERTAIN", "Not sure", "Hard to judge."]],
  },
};

const bottleneckCards = [
  ["PROPOSAL_MISS", "Person missed"], ["OFF_PITCH_OR_BACKGROUND_CLUTTER", "Background or off-pitch boxes"],
  ["DUPLICATE_PROPOSALS", "Duplicate boxes"], ["MERGED_OR_OVERSIZED_BOXES", "One box covering several people"],
  ["PARTIAL_OR_TRUNCATED_BOXES", "Boxes cutting people off"], ["SCALE_OR_PERSPECTIVE", "Distance or perspective"],
  ["ROLE_SEMANTICS", "Role was hard"], ["TEAM_SEMANTICS", "Team was hard"],
  ["PARTICIPATION_SEMANTICS", "Playing or warm-up status was hard"], ["PITCH_STATE", "Pitch position was hard"],
  ["OCCLUSION", "People hidden behind others"], ["NO_CLEAR_BOTTLENECK", "No clear problem"], ["UNCERTAIN", "Not sure"],
];

let serverState;
let activeCase;
let activeTarget;
let mode = "candidate";
let stepIndex = 0;
let answers = {};
let sceneAnswers = {};
let missedPoints = [];
let saveKey = null;
let image = new Image();
let zoom = 1;
let pan = { x: 0, y: 0 };
let dragging = false;
let dragStart = null;
let marking = false;

function currentSceneIndex() { return serverState.cases.findIndex((item) => item.scene_id === activeCase.scene_id); }
function currentTargetIndex() { return activeCase.targets.findIndex((item) => item.target_id === activeTarget?.target_id); }
function latestSavedTarget(target) { return serverState.saved_candidates[target.target_id]; }
function savedTargetCount(scene) { return scene.targets.filter(latestSavedTarget).length; }
function answeredCount() { return Object.keys(serverState.saved_candidates).length + Object.keys(serverState.saved_scenes).length; }

function candidateFlow() {
  const flow = ["inside"];
  if (answers.proposal_validity === "CLEAN_SINGLE_PERSON") {
    flow.push("role");
    if (["OUTFIELD_PLAYER", "GOALKEEPER"].includes(answers.role)) flow.push("team", "participation");
    if (answers.role) flow.push("pitch", "occlusion", "box");
  }
  if (answers.proposal_validity === "DUPLICATE_OF_ANOTHER_CANDIDATE") flow.push("duplicatePicker");
  if (answers.proposal_validity && answers.proposal_validity !== "CLEAN_SINGLE_PERSON") flow.push("certainty");
  if (answers.proposal_validity === "CLEAN_SINGLE_PERSON" && answers.box_quality) flow.push("certainty");
  if (answers.certainty) flow.push("summary");
  return flow;
}

function sceneFlow() {
  const flow = ["missed"];
  if (sceneAnswers.missed_answer === "YES") flow.push("mark", "missedRole", "missedCertainty");
  if (sceneAnswers.missed_answer) flow.push("off_pitch", "duplicate", "hidden", "bottlenecksA", "bottlenecksB", "bottlenecksC");
  if (sceneAnswers.bottlenecks?.length) flow.push("sceneSummary");
  return flow;
}

function teamChoices() {
  const colours = activeCase.team_colours;
  return [
    ["TEAM_1", `Team 1 — ${titleCase(colours.TEAM_1)}`, "Use the approved match colour.", colours.TEAM_1],
    ["TEAM_2", `Team 2 — ${titleCase(colours.TEAM_2)}`, "Use the approved match colour.", colours.TEAM_2],
    ["UNKNOWN_TEAM", "Can't tell", "The kit colour is unclear."],
  ];
}

function titleCase(value) { return String(value).toLowerCase().replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function showToast(message, isError = false) {
  const toast = $("#toast"); toast.textContent = message; toast.className = `toast${isError ? " error" : ""}`; toast.hidden = false;
  window.clearTimeout(showToast.timer); showToast.timer = window.setTimeout(() => { toast.hidden = true; }, 3200);
}
function setSaveState(text, kind = "") { $("#saveState").textContent = text; $("#saveState").className = `save-state ${kind}`; }

async function post(path, payload) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const body = await response.json();
  if (!response.ok || !body.ok) throw new Error(body.message || "The server could not save this.");
  return body;
}

async function saveDraft() {
  setSaveState("Saving progress…");
  const payload = {
    schema_version: DRAFT_SCHEMA, review_id: REVIEW_ID, revision: REVISION, draft_type: mode,
    scene_id: activeCase.scene_id, target_id: mode === "candidate" ? activeTarget.target_id : null,
    step_index: stepIndex, answers: mode === "candidate" ? answers : sceneAnswers,
    missed_people_source_xy: missedPoints, idempotency_key: saveKey,
  };
  try {
    const response = await post("/api/draft", payload);
    setSaveState(response.status === PROGRESS_SAVED ? PROGRESS_SAVED : response.status, "saved");
  } catch (error) {
    setSaveState("Progress not saved", "error"); showToast(error.message, true); throw error;
  }
}

function candidateDefaults(value) {
  const base = { proposal_validity: value, notes: "" };
  if (value === "NO_PERSON_BACKGROUND_OR_OBJECT") return { ...base, role: "NOT_A_PERSON", team: "NOT_APPLICABLE", participation: "NOT_APPLICABLE", pitch_state: "UNCERTAIN", occlusion: "NOT_APPLICABLE", box_quality: "NO_PERSON" };
  if (value === "MERGES_MULTIPLE_PEOPLE") return { ...base, role: "UNKNOWN_PERSON_ROLE", team: "UNKNOWN_TEAM", participation: "UNKNOWN", pitch_state: "UNCERTAIN", occlusion: "UNCERTAIN", box_quality: "MERGED_BOX" };
  if (value === "DUPLICATE_OF_ANOTHER_CANDIDATE") return { ...base, role: "UNKNOWN_PERSON_ROLE", team: "UNKNOWN_TEAM", participation: "UNKNOWN", pitch_state: "UNCERTAIN", occlusion: "UNCERTAIN", box_quality: "UNCERTAIN" };
  if (value === "UNCERTAIN") return { ...base, role: "UNKNOWN_PERSON_ROLE", team: "UNKNOWN_TEAM", participation: "UNKNOWN", pitch_state: "UNCERTAIN", occlusion: "UNCERTAIN", box_quality: "UNCERTAIN" };
  return base;
}

function normalizeRole(value) {
  answers.role = value;
  if (["REFEREE", "OTHER_OFFICIAL", "STAFF_OR_SPECTATOR"].includes(value)) {
    answers.team = "NO_TEAM"; answers.participation = "NON_PLAYER";
  } else if (value === "UNKNOWN_PERSON_ROLE") {
    answers.team = "UNKNOWN_TEAM"; answers.participation = "UNKNOWN";
  }
}

async function choose(value, field) {
  if (mode === "candidate") {
    if (field === "proposal_validity") answers = candidateDefaults(value);
    else if (field === "role") normalizeRole(value);
    else answers[field] = value;
  } else {
    sceneAnswers[field] = value;
    if (field === "missed_answer" && value !== "YES") missedPoints = [];
  }
  saveKey ||= crypto.randomUUID();
  await saveDraft();
  renderQuestion();
}

function answerCards(question, selected) {
  const choices = question.field === "team" ? teamChoices() : question.choices;
  return choices.map((choice, index) => {
    const [value, label, detail, colour] = choice;
    const swatch = colour ? `<span class="swatch" style="background:${cssColour(colour)}" aria-label="${label} colour"></span>` : "";
    const unsure = /not sure|can't tell/i.test(label) ? " not-sure" : "";
    return `<button class="answer-card${selected === value ? " selected" : ""}${unsure}" data-value="${value}" role="radio" aria-checked="${selected === value}"><span class="number">${index + 1}</span><span><strong>${label}</strong><small>${detail}</small></span>${swatch}</button>`;
  }).join("");
}

function cssColour(colour) {
  return { GREY: "#81858d", BLUE: "#2465d4", WHITE: "#ffffff", RED: "#df3248", YELLOW: "#f1cf2f" }[colour] || "#888";
}

function renderQuestion() {
  const flow = mode === "candidate" ? candidateFlow() : sceneFlow();
  stepIndex = Math.min(stepIndex, flow.length - 1);
  const key = flow[stepIndex];
  updateStatus(flow.length);
  $("#specialArea").innerHTML = "";
  $("#answers").innerHTML = "";
  $("#continueButton").textContent = "Continue";
  $("#continueButton").disabled = false;
  if (key === "summary") return renderCandidateSummary();
  if (key === "duplicatePicker") return renderDuplicatePicker();
  if (key === "mark") return renderMarking();
  if (key === "missedRole") return renderMissedDetail("role");
  if (key === "missedCertainty") return renderMissedDetail("certainty");
  if (key.startsWith("bottlenecks")) return renderBottlenecks(key.at(-1));
  if (key === "sceneSummary") return renderSceneSummary();
  const question = mode === "candidate" ? questionBank[key] : sceneQuestions[key];
  const values = mode === "candidate" ? answers : sceneAnswers;
  $("#questionStep").textContent = mode === "candidate" ? `Question ${stepIndex + 1}` : "Whole-scene check";
  $("#questionTitle").textContent = question.title;
  $("#questionHint").textContent = question.hint;
  $("#answers").innerHTML = answerCards(question, values[question.field]);
  $("#answers").querySelectorAll(".answer-card").forEach((card) => card.addEventListener("click", () => choose(card.dataset.value, question.field)));
  $("#continueButton").disabled = !values[question.field];
}

function renderDuplicatePicker() {
  $("#questionStep").textContent = "Choose the matching box";
  $("#questionTitle").textContent = "Which other box shows the same person?";
  $("#questionHint").textContent = "Other boxes are now shown. Pick one different box. Use Not sure if you cannot match it.";
  $("#showOthers").checked = true; drawViews();
  const options = activeCase.targets.filter((target) => target.target_id !== activeTarget.target_id);
  $("#specialArea").innerHTML = `<div class="duplicate-picker">${options.map((target) => `<button class="duplicate-option${answers.duplicate_of_target_id === target.target_id ? " selected" : ""}" data-id="${target.target_id}">Box ${activeCase.targets.indexOf(target) + 1}<br><small>${target.target_id}</small></button>`).join("")}<button class="duplicate-option${answers.duplicate_of_target_id === "UNCERTAIN" ? " selected" : ""}" data-id="UNCERTAIN">Not sure</button></div>`;
  $("#specialArea").querySelectorAll(".duplicate-option").forEach((button) => button.addEventListener("click", async () => {
    if (button.dataset.id === "UNCERTAIN") {
      answers = candidateDefaults("UNCERTAIN");
    } else {
      answers.duplicate_of_target_id = button.dataset.id;
    }
    await saveDraft(); renderQuestion();
  }));
  $("#continueButton").disabled = !answers.duplicate_of_target_id;
}

function renderCandidateSummary() {
  $("#questionStep").textContent = "Check before saving";
  $("#questionTitle").textContent = "Does this summary look right?";
  $("#questionHint").textContent = "Nothing becomes final until the server confirms your save.";
  const rows = Object.entries(answers).filter(([key, value]) => value && !["notes", "duplicate_of_target_id"].includes(key));
  $("#specialArea").innerHTML = `<div class="summary">${rows.map(([field, value]) => `<div class="summary-row"><span>${plainField(field)}</span><strong>${plainValue(field, value)}</strong></div>`).join("")}</div>`;
  $("#continueButton").textContent = "Save and next";
}

function plainField(field) { return { proposal_validity: "Inside the box", role: "Person type", team: "Team", participation: "Taking part", pitch_state: "Feet", occlusion: "Visibility", box_quality: "Box fit", certainty: "Confidence" }[field] || field; }
function plainValue(field, value) { return candidateLabels[field]?.[value] || titleCase(value.replaceAll("_", " ")); }

function renderMarking() {
  marking = true;
  $("#questionStep").textContent = "Mark anyone missed";
  $("#questionTitle").textContent = "Click the centre of each missed person";
  $("#questionHint").textContent = "Use the large scene view. Each new mark starts as 'person type unclear' and 'probably'; you can remove a mistaken mark.";
  $("#specialArea").innerHTML = `<div class="mark-note">${missedPoints.length} missed ${missedPoints.length === 1 ? "person" : "people"} marked.</div><div class="duplicate-picker">${missedPoints.map((point, index) => `<button class="duplicate-option" data-remove="${index}">Remove mark ${index + 1}</button>`).join("")}</div>`;
  $("#specialArea").querySelectorAll("[data-remove]").forEach((button) => button.addEventListener("click", async () => { missedPoints.splice(Number(button.dataset.remove), 1); await saveDraft(); drawViews(); renderQuestion(); }));
  $("#continueButton").disabled = missedPoints.length === 0;
}

function renderMissedDetail(field) {
  marking = false;
  const point = missedPoints[missedPoints.length - 1];
  const choices = field === "role"
    ? [["OUTFIELD_PLAYER", "Outfield player", "A player who is not the goalkeeper."], ["GOALKEEPER", "Goalkeeper", "The goalkeeper."], ["RELEVANT_OFFICIAL", "Referee or official", "A match official."], ["UNKNOWN_RELEVANT_PERSON", "Not sure", "The type is unclear."]]
    : [["CERTAIN", "Sure", "The missed person is clearly visible."], ["PROBABLE", "Probably", "Likely, but not completely clear."], ["UNCERTAIN", "Not sure", "The picture is unclear."]];
  $("#questionStep").textContent = "Missed-person details";
  $("#questionTitle").textContent = field === "role" ? "What kind of person did you mark?" : "How sure are you about this missed person?";
  $("#questionHint").textContent = `This is mark ${missedPoints.length}. You can go Back to move or remove it.`;
  const selected = point?.[field];
  $("#answers").innerHTML = answerCards({ field, choices }, selected);
  $("#answers").querySelectorAll(".answer-card").forEach((card) => card.addEventListener("click", async () => {
    point[field] = card.dataset.value; await saveDraft(); renderQuestion();
  }));
  if (field === "certainty" && selected) {
    $("#specialArea").innerHTML = '<button id="markAnother" class="secondary">Mark another missed person</button>';
    $("#markAnother").onclick = async () => { stepIndex = 1; await saveDraft(); renderQuestion(); };
  }
  $("#continueButton").disabled = !selected;
}

function renderBottlenecks(page) {
  marking = false;
  const pages = {
    A: [...bottleneckCards.slice(0, 5), bottleneckCards.at(-1)],
    B: [...bottleneckCards.slice(5, 10), bottleneckCards.at(-1)],
    C: bottleneckCards.slice(10),
  };
  $("#questionStep").textContent = `Choose up to three · page ${page === "A" ? 1 : page === "B" ? 2 : 3} of 3`;
  $("#questionTitle").textContent = "What were the main problems in this frame?";
  $("#questionHint").textContent = "Choose up to three across these short pages. Not sure is always available.";
  const selected = sceneAnswers.bottlenecks || [];
  $("#answers").innerHTML = pages[page].map(([value, label], index) => `<button class="answer-card${selected.includes(value) ? " selected" : ""}" data-value="${value}"><span class="number">${index + 1}</span><span><strong>${label}</strong></span></button>`).join("");
  $("#answers").querySelectorAll(".answer-card").forEach((button) => button.addEventListener("click", async () => {
    const value = button.dataset.value; const current = sceneAnswers.bottlenecks || [];
    if (["UNCERTAIN", "NO_CLEAR_BOTTLENECK"].includes(value)) sceneAnswers.bottlenecks = [value];
    else {
      const usable = current.filter((item) => !["UNCERTAIN", "NO_CLEAR_BOTTLENECK"].includes(item));
      sceneAnswers.bottlenecks = usable.includes(value) ? usable.filter((item) => item !== value) : usable.length < 3 ? [...usable, value] : usable;
    }
    await saveDraft(); renderQuestion();
  }));
  $("#continueButton").disabled = selected.length === 0;
}

function renderSceneSummary() {
  marking = false;
  $("#questionStep").textContent = "Check the scene";
  $("#questionTitle").textContent = "Ready to finish this scene?";
  $("#questionHint").textContent = "The server will acknowledge the scene only after all eight box answers are safe.";
  $("#specialArea").innerHTML = `<div class="summary"><div class="summary-row"><span>Missed people marked</span><strong>${missedPoints.length}</strong></div><div class="summary-row"><span>Useless or outside boxes</span><strong>${scenePlain("off_pitch_proposal_burden")}</strong></div><div class="summary-row"><span>Repeated boxes</span><strong>${scenePlain("duplicate_or_overlap_burden")}</strong></div><div class="summary-row"><span>People hidden</span><strong>${scenePlain("occlusion_burden")}</strong></div><div class="summary-row"><span>Main problems</span><strong>${sceneAnswers.bottlenecks.map((value) => titleCase(value.replaceAll("_", " "))).join(", ")}</strong></div></div>`;
  $("#continueButton").textContent = "Save scene and continue";
}

function scenePlain(field) { return titleCase(String(sceneAnswers[field] || "Not sure").replaceAll("_", " ")); }

async function continueWizard() {
  const flow = mode === "candidate" ? candidateFlow() : sceneFlow();
  const key = flow[stepIndex];
  if (key === "summary") return saveCandidate();
  if (key === "sceneSummary") return saveScene();
  stepIndex = Math.min(stepIndex + 1, flow.length - 1); await saveDraft(); renderQuestion();
}

async function backWizard() {
  if (stepIndex === 0) return;
  stepIndex -= 1; await saveDraft(); renderQuestion();
}

async function saveCandidate() {
  setSaveState("Saving final answer…");
  const decision = { ...answers };
  if (decision.proposal_validity === "CLEAN_SINGLE_PERSON" && decision.box_quality === "TOO_LOOSE") decision.proposal_validity = "LOOSE_BACKGROUND_AROUND_PERSON";
  if (decision.proposal_validity === "CLEAN_SINGLE_PERSON" && decision.box_quality === "TOO_TIGHT_OR_TRUNCATED") decision.proposal_validity = "PARTIAL_SINGLE_PERSON";
  const payload = { schema_version: EVENT_SCHEMA, review_id: REVIEW_ID, revision: REVISION, event_type: "candidate", scene_id: activeCase.scene_id, target_id: activeTarget.target_id, idempotency_key: saveKey || crypto.randomUUID(), decision };
  try {
    const result = await post("/api/save", payload);
    setSaveState(`SAVED — SERVER ACKNOWLEDGED · ${result.event_id}`, "saved"); showToast("Answer safe. Moving to the next box.");
    await refreshState(); advanceAfterCandidate();
  } catch (error) { setSaveState("Final answer not saved", "error"); showToast(error.message, true); }
}

async function saveScene() {
  setSaveState("Saving scene…");
  const review = { full_frame_coverage_confirmed: true, missed_people_source_xy: missedPoints, off_pitch_proposal_burden: sceneAnswers.off_pitch_proposal_burden, duplicate_or_overlap_burden: sceneAnswers.duplicate_or_overlap_burden, occlusion_burden: sceneAnswers.occlusion_burden, bottlenecks: sceneAnswers.bottlenecks };
  const payload = { schema_version: EVENT_SCHEMA, review_id: REVIEW_ID, revision: REVISION, event_type: "scene", scene_id: activeCase.scene_id, idempotency_key: saveKey || crypto.randomUUID(), review };
  try {
    const result = await post("/api/save", payload);
    setSaveState(`SAVED — SERVER ACKNOWLEDGED · ${result.event_id}`, "saved"); showToast("Scene safe. Moving on.");
    await refreshState(); advanceScene();
  } catch (error) { setSaveState("Scene not saved", "error"); showToast(error.message, true); }
}

function advanceAfterCandidate() {
  const next = activeCase.targets.find((target) => !latestSavedTarget(target));
  if (next) selectTarget(next); else startSceneReview();
}
function advanceScene() {
  const next = serverState.cases.find((scene) => !serverState.saved_scenes[scene.scene_id]);
  if (next) selectCase(next); else completeReview();
}
async function completeReview() {
  try { const result = await post("/api/complete", { review_id: REVIEW_ID, revision: REVISION }); showToast(`${result.status}. Receipt: ${result.completion_receipt_id}`); }
  catch (error) { showToast(error.message, true); }
}

function startSceneReview() {
  mode = "scene"; activeTarget = activeCase.targets[7]; stepIndex = 0; sceneAnswers = {}; missedPoints = []; saveKey = crypto.randomUUID();
  const draft = serverState.drafts[activeCase.scene_id];
  if (draft) { sceneAnswers = draft.answers || {}; missedPoints = draft.missed_people_source_xy || []; stepIndex = draft.step_index || 0; saveKey = draft.idempotency_key || saveKey; }
  drawViews(); renderQuestion(); renderNavigator();
}

function selectTarget(target) {
  mode = "candidate"; activeTarget = target; stepIndex = 0; answers = {}; missedPoints = []; marking = false; saveKey = crypto.randomUUID();
  const saved = latestSavedTarget(target);
  const draft = serverState.drafts[target.target_id];
  if (saved) { answers = structuredClone(saved.payload.decision); stepIndex = Math.max(0, candidateFlow().length - 1); saveKey = crypto.randomUUID(); }
  else if (draft) { answers = draft.answers || {}; stepIndex = draft.step_index || 0; saveKey = draft.idempotency_key || saveKey; }
  zoom = 1; pan = { x: 0, y: 0 }; loadImage(); renderQuestion(); renderNavigator();
}

function selectCase(scene) {
  activeCase = scene;
  const next = scene.targets.find((target) => !latestSavedTarget(target));
  if (next) selectTarget(next); else if (!serverState.saved_scenes[scene.scene_id]) startSceneReview(); else selectTarget(scene.targets[0]);
  $("#navigator").hidden = true; $("#navigatorButton").setAttribute("aria-expanded", "false");
}

async function refreshState() { serverState = await fetch("/api/state").then((response) => response.json()); activeCase = serverState.cases.find((item) => item.scene_id === activeCase.scene_id); }

function loadImage() {
  image = new Image(); image.onload = drawViews; image.onerror = () => showToast("The source picture could not be loaded.", true); image.src = `/assets/${activeCase.asset_name}`;
  $("#matchName").textContent = `Match ${activeCase.match_id} · ${titleCase(activeCase.half.replaceAll("_", " "))} · ${activeCase.timestamp_seconds.toFixed(2)} seconds`;
  $("#targetName").textContent = mode === "candidate" ? `Box ${currentTargetIndex() + 1} · ${activeTarget.target_id}` : "Check the whole scene";
}

function canvasBox(canvas, target = activeTarget) {
  const [left, top, right, bottom] = target.source_box_xyxy; return { left, top, right, bottom, width: right - left, height: bottom - top, canvas };
}
function cropAround(box, factor, aspect) {
  const width = Math.max(box.width * factor, 240); const height = Math.max(box.height * factor, width / aspect);
  const centerX = (box.left + box.right) / 2 + pan.x; const centerY = (box.top + box.bottom) / 2 + pan.y;
  const scaledWidth = width / zoom; const scaledHeight = height / zoom;
  return { x: Math.max(0, Math.min(image.width - scaledWidth, centerX - scaledWidth / 2)), y: Math.max(0, Math.min(image.height - scaledHeight, centerY - scaledHeight / 2)), width: Math.min(image.width, scaledWidth), height: Math.min(image.height, scaledHeight) };
}
function setupCanvas(canvas) { const ratio = window.devicePixelRatio || 1; const rect = canvas.getBoundingClientRect(); canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio); return { context: canvas.getContext("2d"), width: canvas.width, height: canvas.height }; }
function drawCrop(canvas, crop, closeup = false) {
  const { context, width, height } = setupCanvas(canvas); context.clearRect(0, 0, width, height); context.imageSmoothingEnabled = true; context.drawImage(image, crop.x, crop.y, crop.width, crop.height, 0, 0, width, height);
  if (!closeup || $("#showOthers").checked) activeCase.targets.forEach((target) => drawBox(context, crop, width, height, target, target.target_id === activeTarget.target_id));
  else drawBox(context, crop, width, height, activeTarget, true);
  if (marking) missedPoints.forEach((point, index) => drawPoint(context, crop, width, height, point.source_xy, index + 1));
}
function drawBox(context, crop, width, height, target, selected) {
  if (!selected && !$("#showOthers").checked) return;
  const [left, top, right, bottom] = target.source_box_xyxy; const x = (left - crop.x) * width / crop.width; const y = (top - crop.y) * height / crop.height;
  const w = (right - left) * width / crop.width; const h = (bottom - top) * height / crop.height;
  context.save(); context.strokeStyle = selected ? "#ffcf33" : "rgba(185,198,220,.72)"; context.lineWidth = selected ? Math.max(8, width / 190) : Math.max(2, width / 600); context.shadowColor = selected ? "rgba(255,207,51,.95)" : "transparent"; context.shadowBlur = selected ? 16 : 0; context.strokeRect(x, y, w, h); context.fillStyle = selected ? "#172034" : "rgba(23,32,52,.78)"; context.font = `${selected ? 20 : 14}px sans-serif`; context.fillText(selected ? `${currentTargetIndex() + 1} · ${target.target_id}` : `${activeCase.targets.indexOf(target) + 1}`, x + 4, Math.max(20, y - 8)); context.restore();
}
function drawPoint(context, crop, width, height, point, label) { const x = (point[0] - crop.x) * width / crop.width; const y = (point[1] - crop.y) * height / crop.height; context.save(); context.fillStyle = "#ff466c"; context.beginPath(); context.arc(x, y, 12, 0, Math.PI * 2); context.fill(); context.fillStyle = "white"; context.font = "bold 15px sans-serif"; context.fillText(String(label), x - 4, y + 5); context.restore(); }
function drawViews() {
  if (!image.complete || !image.naturalWidth || !activeTarget) return;
  const contextCanvas = $("#contextCanvas"); const closeupCanvas = $("#closeupCanvas");
  const contextAspect = Math.max(.5, contextCanvas.clientWidth / contextCanvas.clientHeight); const closeAspect = Math.max(.5, closeupCanvas.clientWidth / closeupCanvas.clientHeight);
  const box = canvasBox(contextCanvas); const contextCrop = mode === "scene" ? { x: 0, y: 0, width: image.width, height: image.height } : cropAround(box, 14, contextAspect); const closeCrop = cropAround(box, 4.5, closeAspect);
  drawCrop(contextCanvas, contextCrop); drawCrop(closeupCanvas, closeCrop, true);
  const orientation = $("#orientationCanvas"); const setup = setupCanvas(orientation); setup.context.drawImage(image, 0, 0, setup.width, setup.height); activeCase.targets.forEach((target) => drawBox(setup.context, { x: 0, y: 0, width: image.width, height: image.height }, setup.width, setup.height, target, target.target_id === activeTarget.target_id));
}

function markMissedPerson(event) {
  if (!marking) return;
  const canvas = $("#contextCanvas"); const rect = canvas.getBoundingClientRect(); const crop = { x: 0, y: 0, width: image.width, height: image.height };
  const x = crop.x + (event.clientX - rect.left) * crop.width / rect.width; const y = crop.y + (event.clientY - rect.top) * crop.height / rect.height;
  missedPoints.push({ source_xy: [x, y], role: null, certainty: null }); saveDraft().then(() => { drawViews(); renderQuestion(); });
}

function updateStatus(questionTotal) {
  $("#scenePosition").textContent = `Scene ${currentSceneIndex() + 1} of 24`;
  $("#boxPosition").textContent = mode === "candidate" ? `Box ${currentTargetIndex() + 1} of 8` : "Whole-scene check";
  $("#questionPosition").textContent = `Question ${stepIndex + 1} of ${questionTotal}`;
  const percent = Math.round(answeredCount() / 216 * 100); $("#overallText").textContent = `Overall progress ${percent}%`; $("#overallBar").style.width = `${percent}%`;
  $("#backButton").disabled = stepIndex === 0;
}

function renderNavigator() {
  $("#sceneList").innerHTML = serverState.cases.map((scene, index) => `<button class="scene-button${scene.scene_id === activeCase.scene_id ? " current" : ""}" data-scene="${scene.scene_id}"><strong>Scene ${index + 1}</strong><small>Match ${scene.match_id} · ${savedTargetCount(scene)}/8 boxes${serverState.saved_scenes[scene.scene_id] ? " · done" : ""}</small></button>`).join("");
  $("#sceneList").querySelectorAll(".scene-button").forEach((button) => button.addEventListener("click", () => selectCase(serverState.cases.find((scene) => scene.scene_id === button.dataset.scene))));
}

function setupTutorial() {
  const slides = [
    ["We highlight one box.", "The thick yellow outline shows exactly what to review."],
    ["Answer one simple question at a time.", "Large answer cards guide you through the picture."],
    ["Use Not sure whenever the picture is unclear.", "A careful uncertain answer is always better than a guess."],
    ["After eight boxes, check the whole frame.", "You can mark any clearly visible relevant person who was missed."],
  ];
  let index = 0; const dialog = $("#tutorial");
  function render() { $("#tutorialTitle").textContent = slides[index][0]; $("#tutorialText").textContent = slides[index][1]; $("#tutorialDots").innerHTML = slides.map((_, slide) => `<span class="${slide === index ? "active" : ""}"></span>`).join(""); $("#nextTutorial").textContent = index === slides.length - 1 ? "Start" : "Next"; }
  $("#nextTutorial").onclick = () => { if (index < slides.length - 1) { index += 1; render(); } else { dialog.close(); localStorage.setItem("g7d-c1-r1-tutorial", "seen"); } };
  $("#skipTutorial").onclick = () => { dialog.close(); localStorage.setItem("g7d-c1-r1-tutorial", "seen"); };
  $("#showTutorial").onclick = () => { index = 0; render(); dialog.showModal(); $("#helpDrawer").hidden = true; };
  render(); if (!localStorage.getItem("g7d-c1-r1-tutorial")) dialog.showModal();
}

function bindControls() {
  $("#continueButton").onclick = continueWizard; $("#backButton").onclick = backWizard;
  $("#showOthers").onchange = drawViews; $("#fitTarget").onclick = () => { zoom = 1; pan = { x: 0, y: 0 }; drawViews(); };
  $("#zoomIn").onclick = () => { zoom = Math.min(4, zoom * 1.25); drawViews(); }; $("#zoomOut").onclick = () => { zoom = Math.max(.55, zoom / 1.25); drawViews(); };
  $("#resetView").onclick = () => { zoom = 1; pan = { x: 0, y: 0 }; drawViews(); };
  $("#navigatorButton").onclick = () => { const nav = $("#navigator"); nav.hidden = !nav.hidden; $("#navigatorButton").setAttribute("aria-expanded", String(!nav.hidden)); };
  $("#closeNavigator").onclick = () => { $("#navigator").hidden = true; }; $("#helpButton").onclick = () => { $("#helpDrawer").hidden = false; };
  $("#closeHelp").onclick = () => { $("#helpDrawer").hidden = true; };
  const canvas = $("#contextCanvas"); canvas.addEventListener("click", markMissedPerson);
  canvas.addEventListener("pointerdown", (event) => { if (marking) return; dragging = true; dragStart = { x: event.clientX, y: event.clientY, pan: { ...pan } }; canvas.classList.add("dragging"); canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", (event) => { if (!dragging) return; pan.x = dragStart.pan.x - (event.clientX - dragStart.x) * 3 / zoom; pan.y = dragStart.pan.y - (event.clientY - dragStart.y) * 3 / zoom; drawViews(); });
  canvas.addEventListener("pointerup", () => { dragging = false; canvas.classList.remove("dragging"); });
  window.addEventListener("resize", drawViews);
  window.addEventListener("keydown", (event) => {
    if (["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if (/^[1-6]$/.test(event.key)) $("#answers").querySelectorAll(".answer-card")[Number(event.key) - 1]?.click();
    else if (event.key === "Enter") $("#continueButton").click();
    else if (event.key === "Backspace") { event.preventDefault(); backWizard(); }
    else if (event.key.toLowerCase() === "z") $("#fitTarget").click();
    else if (event.key.toLowerCase() === "o") { $("#showOthers").checked = !$("#showOthers").checked; drawViews(); }
    else if (event.key.toLowerCase() === "h") $("#helpButton").click();
  });
}

async function start() {
  try {
    serverState = await fetch("/api/state").then((response) => response.json());
    if (serverState.review_revision !== REVISION || serverState.cases.length !== 24) throw new Error("The reviewer package does not match this guided revision.");
    activeCase = serverState.cases.find((scene) => !serverState.saved_scenes[scene.scene_id]) || serverState.cases[0];
    bindControls(); setupTutorial(); renderNavigator(); selectCase(activeCase); setSaveState("Ready");
  } catch (error) { setSaveState("Reviewer unavailable", "error"); showToast(error.message, true); }
}

start();
