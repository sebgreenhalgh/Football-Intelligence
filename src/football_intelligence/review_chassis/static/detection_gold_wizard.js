(function () {
  "use strict";

  const GLOBAL_STEPS = [
    "Mark what you can see",
    "Answer a few questions",
    "Check the machine boxes",
    "Review and save",
  ];

  const ROLE_CHOICES = [
    ["Player", "PLAYER"],
    ["Goalkeeper", "GOALKEEPER"],
    ["Referee", "REFEREE"],
    ["Other match official", "OFFICIAL"],
    ["Staff or spectator", "STAFF_OR_SPECTATOR"],
    ["I can't tell", "UNKNOWN"],
  ];
  const VISIBILITY_CHOICES = [
    ["All or nearly all visible", "VISIBLE"],
    ["Some is hidden", "PARTIALLY_VISIBLE"],
    ["Only a small part is visible", "HEAVILY_OCCLUDED"],
    ["I can't tell", "UNRESOLVED"],
  ];
  const OCCLUDER_CHOICES = [
    ["Another person", "PERSON"],
    ["Image edge", "FRAME_EDGE"],
    ["Goal or equipment", "EQUIPMENT"],
    ["Something else", "SCENE_STRUCTURE"],
    ["I can't tell", "UNKNOWN"],
  ];
  const OCCLUSION_FRACTIONS = [
    ["A little", 0.2],
    ["About half", 0.5],
    ["Most", 0.75],
    ["Almost all", 0.9],
    ["I can't tell", 0.5],
  ];
  const FOOT_UNCERTAINTY = [
    ["Very sure", 3],
    ["A little unsure", 8],
    ["Very unsure", 20],
  ];
  const PITCH_CHOICES = [
    ["Inside the playing field", "ON_PITCH"],
    ["Outside the playing field", "OFF_PITCH"],
    ["On or very close to the boundary", "BOUNDARY_UNCERTAIN"],
    ["I can't tell", "BOUNDARY_UNCERTAIN"],
  ];
  const CANDIDATE_RELATIONS = [
    ["One person - the box is useful", "CLEAN_SINGLE_INSTANCE"],
    ["An extra box for the same person", "DUPLICATE_OF_INSTANCE"],
    ["Two or more people joined together", "MERGED_MULTIPLE_INSTANCES"],
    ["Only part of one person", "PARTIAL_INSTANCE"],
    ["Not a person", "BACKGROUND"],
    ["I can't tell", "AMBIGUOUS"],
  ];
  const TEMPORAL_STATES = [
    ["Yes, clearly", "OBSERVED"],
    ["Yes, but nearby frames help", "OBSERVED_WITH_TEMPORAL_REFINEMENT"],
    ["No - I can only predict where they are", "OCCLUDED_PREDICTED"],
    ["No, they are not visible", "NOT_VISIBLE"],
    ["I can't tell", "UNRESOLVED"],
  ];
  const FOOTBALL_STATES = [
    ["Yes, clearly", "VISIBLE_CLEAR"],
    ["Yes, but blurred", "VISIBLE_BLURRED"],
    ["Partly visible", "PARTIALLY_OCCLUDED_VISIBLE"],
    ["No - only predicted from nearby frames", "FULLY_OCCLUDED_PREDICTED"],
    ["No football visible", "NOT_VISIBLE"],
    ["It is outside the image", "OUT_OF_FRAME"],
    ["I can't tell", "UNRESOLVED"],
  ];
  const FAILURE_CHOICES = [
    ["There was never a useful early box", "NO_VALID_RAW_PROPOSAL"],
    ["The early box was aimed at the person but badly placed", "BAD_RAW_LOCALIZATION"],
    ["A useful box disappeared at the confidence step", "VALID_PROPOSAL_LOW_CONFIDENCE"],
    ["Separate people had boxes before suppression but not afterward", "VALID_PROPOSALS_NMS_COLLAPSED"],
    ["The same person still has extra final boxes", "DUPLICATED_AFTER_VIEW_FUSION"],
    ["The field-boundary decision is wrong", "PITCH_GATE_ERROR"],
    ["The displayed box does not match its stored source", "RENDERER_OR_PROVENANCE_ERROR"],
    ["I don't know", "UNRESOLVED"],
  ];
  const HARD_NEGATIVES = [
    ["A pitch marking", "PITCH_MARKING"],
    ["A line crossing", "LINE_INTERSECTION"],
    ["A logo or text", "LOGO_OR_TEXT"],
    ["Equipment or a cone", "EQUIPMENT_OR_CONE"],
    ["A shoe or sock", "SHOE_OR_SOCK"],
    ["A person's head", "HEAD"],
    ["A bright reflection", "HIGHLIGHT_OR_REFLECTION"],
    ["A compression mark", "COMPRESSION_ARTEFACT"],
    ["I can't tell", "UNKNOWN"],
  ];

  const clone = (value) => JSON.parse(JSON.stringify(value));
  const escapeHtml = (value) => String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

  function answerButtons(question, choices, key, {columns = 2, hint = ""} = {}) {
    return `<section class="nwQuestionCard" data-nw-question="${key}">
      <span class="nwQuestionLabel">ONE SHORT QUESTION</span>
      <h3>${question}</h3>
      ${hint ? `<p class="nwQuestionHint">${hint}</p>` : ""}
      <div class="nwChoices nwColumns${columns}">
        ${choices.map(([label, value]) => `<button type="button" data-nw-answer-key="${key}" data-nw-answer-value="${escapeHtml(value)}">${label}</button>`).join("")}
      </div>
    </section>`;
  }

  class NoviceWizard {
    constructor(host) {
      this.host = host;
      this.states = {};
    }

    defaultState(caseData) {
      const r3 = this.host.incrementalR3?.() === true;
      const authoritative = r3 ? this.host.authoritativeBinding?.(caseData) : null;
      return {
        schema_version: r3
          ? "football_intelligence.m5_5g1a_r3.wizard_state.v1"
          : "football_intelligence.m5_5g1a_r2.wizard_state.v1",
        case_id: caseData.case_id,
        step: 1,
        drawing_complete: false,
        current_object_uuid: null,
        question_index: 0,
        completed_object_uuids: [],
        footpoint_placed_uuids: [],
        footpoint_reviews: {},
        pending_footpoint_decision: null,
        candidate_index: 0,
        candidate_phase: "relation",
        candidate_relation: null,
        candidate_targets: [],
        candidate_answered_uuids: [],
        candidate_answer_records: {},
        human_truth_revision: 0,
        person_question_revision: 0,
        candidate_answer_revision: 0,
        summary_revision: 0,
        person_question_completion_revisions: {},
        summary_validity: "UNANSWERED",
        summary_human_truth_revision: null,
        invalidation_notice: null,
        frame_answered_sequences: [],
        frame_phase: "visibility",
        desired_frame_state: null,
        pitch_footpoint_set: false,
        pitch_question_index: 0,
        pitch_answers: [],
        football_candidate_answers: {},
        failure_reviewed: false,
        help_opened: false,
        active_tranche_id: r3 ? this.host.currentTrancheId() : null,
        authoritative_frame_sequence: authoritative?.frame_sequence ?? null,
        authoritative_source_frame_sha256: authoritative?.source_frame_sha256 ?? null,
        primary_canvas_frame_sequence: authoritative?.frame_sequence ?? null,
        primary_canvas_source_frame_sha256: authoritative?.source_frame_sha256 ?? null,
        candidate_queue_binding_hash: authoritative?.candidate_queue_binding_hash ?? null,
      };
    }

    state(caseData = this.host.caseData()) {
      if (!this.states[caseData.case_id]) this.states[caseData.case_id] = this.defaultState(caseData);
      return this.ensureRevisionState(this.states[caseData.case_id]);
    }

    restore(caseId, value) {
      if (value && value.case_id === caseId) this.states[caseId] = this.ensureRevisionState(clone(value));
    }

    snapshot(caseId = this.host.caseData().case_id) {
      return clone(this.states[caseId] || this.defaultState(this.host.caseData()));
    }

    replace(caseId, value) {
      this.states[caseId] = this.ensureRevisionState(clone(value));
    }

    revisionAware() {
      return this.host.revisionAware?.() === true;
    }

    ensureRevisionState(state) {
      state.candidate_answer_records ||= {};
      state.person_question_completion_revisions ||= {};
      state.human_truth_revision = Number.isInteger(state.human_truth_revision) ? state.human_truth_revision : 0;
      state.person_question_revision = Number.isInteger(state.person_question_revision) ? state.person_question_revision : 0;
      state.candidate_answer_revision = Number.isInteger(state.candidate_answer_revision) ? state.candidate_answer_revision : 0;
      state.summary_revision = Number.isInteger(state.summary_revision) ? state.summary_revision : 0;
      state.summary_validity ||= "UNANSWERED";
      state.summary_human_truth_revision ??= null;
      state.invalidation_notice ??= null;
      state.candidate_answered_uuids ||= [];
      return state;
    }

    reset(caseId = this.host.caseData().case_id) {
      this.states[caseId] = this.defaultState(this.host.caseData());
      return this.state();
    }

    initialFrameIndex(caseData) {
      if (["detection_gold_player_static", "detection_gold_dense_region", "detection_gold_pitch_boundary"].includes(caseData.task_type)) {
        const index = (caseData.visible_metadata.frame_records || []).findIndex(
          (row) => Number(row.frame_sequence) === Number(caseData.source_frame_sequence)
        );
        return Math.max(0, index);
      }
      return 0;
    }

    candidates(caseData = this.host.caseData()) {
      const required = new Set(
        this.host.incrementalR3?.() === true
          ? this.host.authoritativeCandidateUuids(caseData)
          : (caseData.visible_metadata.candidate_uuids || [])
      );
      const rows = [];
      const seen = new Set();
      (caseData.visible_metadata.frame_records || []).forEach((record, frameIndex) => {
        if (this.host.incrementalR3?.() === true
          && ["detection_gold_player_static", "detection_gold_dense_region"].includes(caseData.task_type)
          && frameIndex !== this.host.authoritativeFrameIndex(caseData)) return;
        for (const candidate of record.candidates || []) {
          if (!required.has(candidate.diagnostic_uuid) || seen.has(candidate.diagnostic_uuid)) continue;
          seen.add(candidate.diagnostic_uuid);
          rows.push({candidate, frameIndex, frameSequence: record.frame_sequence});
        }
      });
      return rows;
    }

    candidateProgress(state = this.state()) {
      const entries = this.candidates();
      const counts = {total: entries.length, valid: 0, stale: 0, unanswered: 0, invalid: 0};
      for (const entry of entries) {
        const validity = state.candidate_answer_records[entry.candidate.diagnostic_uuid]?.validity || "UNANSWERED";
        if (validity === "VALID") counts.valid += 1;
        else if (validity === "NEEDS_REVIEW") counts.stale += 1;
        else if (validity === "INVALID") counts.invalid += 1;
        else counts.unanswered += 1;
      }
      return counts;
    }

    syncCandidateAnswered(state = this.state()) {
      state.candidate_answered_uuids = this.candidates()
        .map((entry) => entry.candidate.diagnostic_uuid)
        .filter((candidateUuid) => state.candidate_answer_records[candidateUuid]?.validity === "VALID");
    }

    markSummaryStale(state, reason) {
      state.summary_validity = "NEEDS_REVIEW";
      state.summary_revision += 1;
      state.invalidation_notice = reason;
    }

    markSummaryReady(state) {
      const progress = this.candidateProgress(state);
      if (progress.stale || progress.unanswered || progress.invalid) return false;
      state.summary_validity = "VALID";
      state.summary_revision += 1;
      state.summary_human_truth_revision = state.human_truth_revision;
      state.invalidation_notice = null;
      state.step = 4;
      return true;
    }

    invalidateCandidateAnswers(state, reason, predicate = () => true) {
      for (const [candidateUuid, answer] of Object.entries(state.candidate_answer_records)) {
        if (!predicate(answer, candidateUuid) || answer.validity === "INVALID") continue;
        answer.validity = "NEEDS_REVIEW";
        answer.invalidation_reason = reason;
        answer.invalidated_at = new Date().toISOString();
      }
      this.syncCandidateAnswered(state);
      this.markSummaryStale(state, reason);
    }

    recordCandidateAnswer(state, relation, targets) {
      const entry = this.currentCandidateEntry();
      if (!entry) return;
      const candidateUuid = entry.candidate.diagnostic_uuid;
      const now = new Date().toISOString();
      const prior = state.candidate_answer_records[candidateUuid];
      state.candidate_answer_revision += 1;
      state.candidate_answer_records[candidateUuid] = {
        candidate_uuid: candidateUuid,
        relation,
        annotation_uuids: [...targets],
        answered_against_human_truth_revision: state.human_truth_revision,
        answered_person_question_revision: state.person_question_revision,
        candidate_answer_revision: state.candidate_answer_revision,
        validity: "VALID",
        invalidation_reason: null,
        answered_at: prior?.answered_at || now,
        revalidated_at: prior ? now : null,
        revalidation_event: prior ? "GUIDED_REVIEW_AFTER_INVALIDATION" : "INITIAL_REVIEW",
      };
      this.syncCandidateAnswered(state);
      state.summary_validity = "NEEDS_REVIEW";
    }

    candidateAnswerEdited(candidateUuid, reason = "A machine-box answer changed outside the guided check.") {
      const state = this.state();
      if (!this.revisionAware()) return state;
      const answer = state.candidate_answer_records[candidateUuid];
      if (answer) {
        answer.validity = "NEEDS_REVIEW";
        answer.invalidation_reason = reason;
        answer.invalidated_at = new Date().toISOString();
      }
      this.syncCandidateAnswered(state);
      this.markSummaryStale(state, reason);
      return state;
    }

    humanTruthChanged(reason, {targetUuids = null, invalidateAll = false, requireDoneDrawing = true} = {}) {
      const state = this.state();
      if (!this.revisionAware()) return state;
      state.human_truth_revision += 1;
      state.failure_reviewed = false;
      if (requireDoneDrawing) state.drawing_complete = false;
      const targets = targetUuids ? new Set(targetUuids) : null;
      this.invalidateCandidateAnswers(
        state,
        reason,
        (answer) => invalidateAll || !targets || answer.annotation_uuids.some((uuid) => targets.has(uuid)),
      );
      state.step = 1;
      state.candidate_phase = "relation";
      state.candidate_relation = null;
      state.candidate_targets = [];
      return state;
    }

    objectDeleted(annotationUuid) {
      const state = this.humanTruthChanged(
        "Some machine-box answers need checking again because a person was removed.",
        {targetUuids: [annotationUuid]},
      );
      state.completed_object_uuids = state.completed_object_uuids.filter((uuid) => uuid !== annotationUuid);
      state.footpoint_placed_uuids = state.footpoint_placed_uuids.filter((uuid) => uuid !== annotationUuid);
      delete state.footpoint_reviews[annotationUuid];
      delete state.person_question_completion_revisions[annotationUuid];
      for (const answer of Object.values(state.candidate_answer_records)) {
        answer.annotation_uuids = answer.annotation_uuids.filter((uuid) => uuid !== annotationUuid);
      }
      state.current_object_uuid = null;
      state.question_index = 0;
      return state;
    }

    allObjectsDeleted(annotationUuids) {
      const state = this.humanTruthChanged(
        "Start again by drawing the visible people. Earlier machine-box answers need review.",
        {targetUuids: annotationUuids},
      );
      state.completed_object_uuids = [];
      state.footpoint_placed_uuids = [];
      state.footpoint_reviews = {};
      state.person_question_completion_revisions = {};
      for (const answer of Object.values(state.candidate_answer_records)) answer.annotation_uuids = [];
      state.current_object_uuid = null;
      state.question_index = 0;
      return state;
    }

    objectGeometryChanged(annotationUuid) {
      return this.humanTruthChanged(
        "A visible-person box changed. Related machine-box answers need checking.",
        {targetUuids: [annotationUuid]},
      );
    }

    objectSemanticChanged(annotationUuid, {candidateRelevant = true, reopenQuestions = false} = {}) {
      const state = this.state();
      if (!this.revisionAware()) return state;
      state.person_question_revision += 1;
      delete state.person_question_completion_revisions[annotationUuid];
      state.completed_object_uuids = state.completed_object_uuids.filter((uuid) => uuid !== annotationUuid);
      if (candidateRelevant) {
        this.invalidateCandidateAnswers(
          state,
          "A person classification changed. Related machine-box answers need checking.",
          (answer) => answer.annotation_uuids.includes(annotationUuid),
        );
      } else {
        this.markSummaryStale(state, "Person details changed; review the case summary.");
      }
      if (reopenQuestions) {
        state.step = 2;
        state.current_object_uuid = annotationUuid;
        state.question_index = 0;
      }
      return state;
    }

    currentCandidateEntry() {
      const state = this.state();
      return this.candidates()[Math.max(0, Math.min(state.candidate_index, this.candidates().length - 1))] || null;
    }

    overlayPolicy() {
      const state = this.state();
      const task = this.host.caseData().task_type;
      const temporalGeometry = task === "detection_gold_temporal_player" && state.frame_phase === "geometry";
      const footballQueue = task === "detection_gold_football_burst" && state.step === 3;
      const relationQueue = ["detection_gold_player_static", "detection_gold_dense_region"].includes(task) && state.step === 3;
      const entry = relationQueue || footballQueue ? this.currentCandidateEntry() : null;
      return {
        showMachine: Boolean(entry || temporalGeometry),
        candidateUuid: entry?.candidate.diagnostic_uuid || this.host.selectedCandidate()?.diagnostic_uuid || null,
        humanInteractive: !(relationQueue || footballQueue),
        numberHumans: true,
        footpointUuids: this.host.incrementalR3?.() === true && task === "detection_gold_player_static"
          ? this.host.objects().map((row) => row.annotation_uuid)
          : [...state.footpoint_placed_uuids],
        estimatedFootpointUuids: Object.entries(state.footpoint_reviews || {})
          .filter(([, review]) => review?.estimated === true)
          .map(([annotationUuid]) => annotationUuid),
        showTemporalFootpoint: state.frame_phase !== "footpoint",
        pitchFootpointSet: state.pitch_footpoint_set,
        candidateLabel: entry ? `Machine Box ${state.candidate_index + 1}` : "Machine suggestion",
      };
    }

    syncCandidate() {
      const state = this.state();
      if (state.step !== 3) return;
      const entry = this.currentCandidateEntry();
      if (!entry) return;
      if (!(this.host.incrementalR3?.() === true
        && ["detection_gold_player_static", "detection_gold_dense_region"].includes(this.host.caseData().task_type))) {
        this.host.setFrameSilently(entry.frameIndex);
      }
      this.host.setSelectedCandidate(entry.candidate);
    }

    mutate(callback, {history = true} = {}) {
      if (history) this.host.pushHistory();
      callback(this.state(), this.host.annotation());
      this.host.persist();
      this.host.render();
    }

    objectCreated(annotationUuid) {
      const task = this.host.caseData().task_type;
      if (task === "detection_gold_temporal_player") {
        this.mutate((state) => {
          state.frame_phase = "footpoint";
        }, {history: false});
        return;
      }
      if (!["detection_gold_player_static", "detection_gold_dense_region"].includes(task)) return;
      this.mutate((state) => {
        if (this.revisionAware()) {
          const backgroundAnswers = Object.values(state.candidate_answer_records).filter(
            (answer) => answer.validity === "VALID"
              && answer.relation === "BACKGROUND"
              && answer.annotation_uuids.length === 0
          );
          this.humanTruthChanged(
            "A person was added. Existing machine-box answers need checking against the new human truth.",
            {invalidateAll: true},
          );
          if (backgroundAnswers.length && window.confirm('Keep the previous "not a person" answers?')) {
            const now = new Date().toISOString();
            for (const prior of backgroundAnswers) {
              const answer = state.candidate_answer_records[prior.candidate_uuid];
              state.candidate_answer_revision += 1;
              answer.validity = "VALID";
              answer.invalidation_reason = null;
              answer.answered_against_human_truth_revision = state.human_truth_revision;
              answer.candidate_answer_revision = state.candidate_answer_revision;
              answer.revalidated_at = now;
              answer.revalidation_event = "EXPLICIT_BACKGROUND_RETENTION";
            }
            this.syncCandidateAnswered(state);
          }
        }
        state.step = 2;
        state.current_object_uuid = annotationUuid;
        state.question_index = 0;
        state.completed_object_uuids = state.completed_object_uuids.filter((value) => value !== annotationUuid);
      }, {history: false});
    }

    footpointPlaced() {
      const task = this.host.caseData().task_type;
      if (task === "detection_gold_player_static") {
        this.mutate((state) => {
          if (state.current_object_uuid && !state.footpoint_placed_uuids.includes(state.current_object_uuid)) {
            state.footpoint_placed_uuids.push(state.current_object_uuid);
          }
          if (this.host.incrementalR3?.() === true && state.current_object_uuid) {
            const prior = state.footpoint_reviews[state.current_object_uuid];
            state.footpoint_reviews[state.current_object_uuid] = prior?.estimated
              ? {...prior, adjusted: true}
              : {decision: "MOVE_IT", estimated: false, adjusted: true};
            state.pending_footpoint_decision = null;
          }
          state.question_index += 1;
        }, {history: false});
      } else if (task === "detection_gold_temporal_player") {
        this.finishCurrentFrame();
      } else if (task === "detection_gold_pitch_boundary") {
        this.mutate((state) => {
          state.pitch_footpoint_set = true;
          state.step = 2;
          state.pitch_question_index = 0;
        }, {history: false});
      }
    }

    footballPointPlaced() {
      if (this.host.caseData().task_type !== "detection_gold_football_burst") return;
      const frame = this.host.annotation().frames[this.host.frameIndex()];
      if (frame.state === "VISIBLE_BLURRED") {
        this.mutate((state) => { state.frame_phase = "blur"; }, {history: false});
      } else {
        this.finishCurrentFrame();
      }
    }

    footballTrailDrawn() {
      if (this.host.caseData().task_type === "detection_gold_football_burst") this.finishCurrentFrame();
    }

    stepper() {
      const active = this.state().step;
      return `<ol class="nwStepper" aria-label="Case steps">${GLOBAL_STEPS.map((label, index) => {
        const step = index + 1;
        const className = step === active ? "active" : step < active ? "done" : "";
        return `<li class="${className}"><span>${step}</span><strong>${label}</strong></li>`;
      }).join("")}</ol>`;
    }

    shell(content, instruction) {
      const state = this.state();
      const repairControls = this.revisionAware()
        && ["detection_gold_player_static", "detection_gold_dense_region"].includes(this.host.caseData().task_type)
        ? (() => {
          const progress = this.candidateProgress(state);
          const warning = state.invalidation_notice || (progress.stale
            ? `${progress.stale} machine-box answer(s) need checking after the human annotation changed.`
            : "");
          const returnControl = state.step > 1
            ? '<button id="nwReturnDrawing" type="button">Return to drawing people</button>'
            : "";
          const reviewControl = state.step > 1 && state.drawing_complete
            ? `<button id="nwReviewStale" type="button" ${progress.stale || progress.unanswered || progress.invalid ? "" : "disabled"}>Review answers that need checking</button>`
            : "";
          return `${warning ? `<div class="nwStaleWarning" role="alert"><strong>Answers need checking</strong><span>${escapeHtml(warning)}</span></div>` : ""}<div class="nwRepairControls">${returnControl}${reviewControl}<button id="nwRestartCase" type="button" ${this.host.isSaved?.() ? "disabled" : ""}>Restart this case</button></div>`;
        })()
        : "";
      return `<div class="nwWizard" data-nw-step="${this.state().step}">
        ${this.stepper()}
        <div class="nwTaskIntro"><span class="nwStepEyebrow">STEP ${this.state().step}</span><p>${instruction}</p><button id="nwCurrentHelp" type="button" class="nwHelpButton">How this works</button></div>
        ${repairControls}
        ${content}
      </div>`;
    }

    personQuestion(annotation, state) {
      const person = (annotation.player_instances || []).find((row) => row.annotation_uuid === state.current_object_uuid);
      if (!person) return "";
      const hidden = ["PARTIALLY_VISIBLE", "HEAVILY_OCCLUDED"].includes(person.visibility_state);
      const edgeRelevant = this.edgeQuestionRelevant(person.visible_body_box);
      const r3 = this.host.incrementalR3?.() === true;
      const questions = r3
        ? ["role", "visibility", ...(hidden ? ["occluder", "hidden_amount"] : []), "footpoint_review", "pitch", ...(edgeRelevant ? ["edge"] : [])]
        : ["role", "visibility", ...(hidden ? ["occluder", "hidden_amount"] : []), "footpoint", "foot_uncertainty", "pitch", ...(edgeRelevant ? ["edge"] : [])];
      if (state.question_index >= questions.length) {
        queueMicrotask(() => this.finishObjectQuestions(person.annotation_uuid));
        return `<div class="nwWaiting">Person ${this.host.objectIndex(person.annotation_uuid) + 1} is ready.</div>`;
      }
      const key = questions[state.question_index];
      if (key === "role") return answerButtons("Who is this?", ROLE_CHOICES, key, {columns: 2});
      if (key === "visibility") return answerButtons("How much of this person can you see?", VISIBILITY_CHOICES, key, {columns: 2});
      if (key === "occluder") return answerButtons("What is hiding them?", OCCLUDER_CHOICES, key, {columns: 2});
      if (key === "hidden_amount") return answerButtons("About how much is hidden?", OCCLUSION_FRACTIONS, key, {columns: 2});
      if (key === "footpoint_review") {
        const review = state.footpoint_reviews[person.annotation_uuid];
        return `<section class="nwQuestionCard" data-nw-question="footpoint_review"><span class="nwQuestionLabel">ONE SHORT QUESTION</span><h3>Is this roughly where their feet touch the ground?</h3><p class="nwQuestionHint">The suggested point starts at the visible-box bottom centre. Choose Move it when the feet are visible elsewhere. Hidden-foot estimates remain adjustable and carry high uncertainty.</p><div class="nwChoices nwColumns1"><button type="button" data-nw-answer-key="footpoint_review" data-nw-answer-value="YES">Yes</button><button type="button" data-nw-answer-key="footpoint_review" data-nw-answer-value="MOVE_IT">Move it</button><button type="button" data-nw-answer-key="footpoint_review" data-nw-answer-value="FEET_NOT_VISIBLE">Feet not visible</button><button type="button" data-nw-answer-key="footpoint_review" data-nw-answer-value="CANNOT_TELL">Cannot tell</button></div>${review?.estimated ? '<p class="nwEstimatedNotice">Estimated because the feet are not visible</p><button id="nwAdjustFootpoint" type="button">Adjust estimate</button>' : ""}</section>`;
      }
      if (key === "footpoint") return `<section class="nwQuestionCard" data-nw-question="footpoint"><span class="nwQuestionLabel">ONE SHORT QUESTION</span><h3>Where do their feet touch the ground?</h3><p class="nwQuestionHint">Choose the button, then click the image. You can zoom first.</p><button id="nwPlaceFootpoint" class="nwPrimary" type="button">Place the foot point</button></section>`;
      if (key === "foot_uncertainty") return answerButtons("How sure are you about that foot position?", FOOT_UNCERTAINTY, key, {columns: 1});
      if (key === "pitch") return answerButtons("Are their feet inside the playing field?", PITCH_CHOICES, key, {columns: 1});
      return answerButtons("Are they cut off by an edge of the image?", [["No", "NONE"], ["Left edge", "LEFT"], ["Top edge", "TOP"], ["Right edge", "RIGHT"], ["Bottom edge", "BOTTOM"], ["I can't tell", "UNSURE"]], key, {columns: 2});
    }

    edgeQuestionRelevant(box) {
      const bounds = this.host.record().focal_bounds;
      if (!box || !bounds) return true;
      const margin = Math.max(3, Math.min(bounds.x2 - bounds.x1, bounds.y2 - bounds.y1) * 0.015);
      return box.x1 <= bounds.x1 + margin || box.y1 <= bounds.y1 + margin
        || box.x2 >= bounds.x2 - margin || box.y2 >= bounds.y2 - margin;
    }

    finishObjectQuestions(annotationUuid) {
      this.mutate((state) => {
        if (!state.completed_object_uuids.includes(annotationUuid)) state.completed_object_uuids.push(annotationUuid);
        if (this.revisionAware()) {
          state.person_question_revision += 1;
          state.person_question_completion_revisions[annotationUuid] = state.person_question_revision;
          state.summary_validity = "NEEDS_REVIEW";
        }
        state.current_object_uuid = null;
        state.question_index = 0;
        state.step = state.drawing_complete ? 3 : 1;
        if (state.step === 3) state.candidate_index = this.nextUnansweredCandidateIndex();
      }, {history: false});
    }

    denseQuestion(annotation, state) {
      const mask = (annotation.visible_masks || []).find((row) => row.annotation_uuid === state.current_object_uuid);
      if (!mask) return "";
      const questions = ["mask_quality", "mask_front", "mask_truncation"];
      if (state.question_index >= questions.length) {
        queueMicrotask(() => this.finishObjectQuestions(mask.annotation_uuid));
        return `<div class="nwWaiting">Person ${this.host.objectIndex(mask.annotation_uuid) + 1} is ready.</div>`;
      }
      const key = questions[state.question_index];
      if (key === "mask_quality") return answerButtons("How clear is the outline you traced?", [["Clear", "PRECISE"], ["Approximate", "COARSE"], ["Too uncertain", "UNCERTAIN"], ["Ignore this outline", "IGNORE"]], key, {columns: 2});
      if (key === "mask_front") {
        const others = (annotation.visible_masks || []).filter((row) => row.annotation_uuid !== mask.annotation_uuid);
        return `<section class="nwQuestionCard" data-nw-question="mask_front"><span class="nwQuestionLabel">ONE SHORT QUESTION</span><h3>Is another marked person in front of this one?</h3><div class="nwChoices nwColumns1"><button type="button" data-nw-answer-key="mask_front" data-nw-answer-value="NONE">No</button>${others.map((row) => `<button type="button" data-nw-answer-key="mask_front" data-nw-answer-value="${row.annotation_uuid}">Person ${this.host.objectIndex(row.annotation_uuid) + 1} is in front</button>`).join("")}<button type="button" data-nw-answer-key="mask_front" data-nw-answer-value="UNSURE">I can't tell${others.length ? "" : " yet"}</button></div></section>`;
      }
      return answerButtons("Is the visible shape cut off by an image edge?", [["No", "NONE"], ["Left edge", "LEFT"], ["Top edge", "TOP"], ["Right edge", "RIGHT"], ["Bottom edge", "BOTTOM"], ["I can't tell", "UNSURE"]], key, {columns: 2});
    }

    renderStaticOrDense(annotation) {
      const state = this.state();
      const dense = this.host.caseData().task_type === "detection_gold_dense_region";
      if (state.step === 1) {
        const objects = dense ? annotation.visible_masks : annotation.player_instances;
        const label = dense ? "visible shapes" : "people";
        return this.shell(`<section class="nwActionCard"><h3>${dense ? "Trace each visible person" : "Mark every visible person"}</h3><p>${dense ? "Trace only the part of each person you can actually see. Do not draw through someone in front." : "Draw one box around each visible person in the highlighted area."}</p><p class="nwVisibleBodyRule">Box only the part you can actually see. Do not guess the hidden body.</p><p class="nwScopeNote">Label the middle frame only. Nearby frames are reference images.</p><div class="nwObjectSummary">${objects.length ? objects.map((row, index) => `<button type="button" data-nw-edit-object="${row.annotation_uuid}"><strong>Person ${index + 1}</strong><span>Edit</span></button>`).join("") : `<span>No ${label} marked yet.</span>`}</div><div class="nwActionRow"><button id="nwDrawObject" class="nwPrimary" type="button">${dense ? "Trace a person" : "Draw a person"}</button>${dense ? `<button id="nwFinishOutline" type="button" ${this.host.maskPointCount() >= 3 ? "" : "disabled"}>Finish this outline</button>` : ""}<button id="nwUndo" type="button">Undo</button>${this.revisionAware() && objects.length ? '<button id="nwDeleteAllObjects" type="button">Delete all people</button>' : ""}<button id="nwDoneDrawing" type="button">I'm done drawing people</button></div></section>`, dense ? "Trace one visible person at a time in the highlighted area." : "Draw one box around each visible person in the highlighted area.");
      }
      if (state.step === 2) {
        const label = this.host.objectIndex(state.current_object_uuid) + 1;
        const body = dense ? this.denseQuestion(annotation, state) : this.personQuestion(annotation, state);
        return this.shell(`<div class="nwCurrentObject"><strong>Person ${label}</strong><span>Question ${state.question_index + 1}</span></div>${body}<div class="nwActionRow"><button id="nwQuestionBack" type="button">Back</button><button id="nwUndo" type="button">Undo</button><button id="nwDeleteObject" type="button">Delete Person ${label}</button></div>`, `Answer one simple question at a time for Person ${label}.`);
      }
      if (state.step === 3) return this.renderCandidateQueue(annotation, dense);
      return this.renderReview(annotation);
    }

    nextUnansweredCandidateIndex({football = false} = {}) {
      const state = this.state();
      const answered = football
        ? new Set(Object.keys(state.football_candidate_answers))
        : (this.revisionAware()
          ? new Set(this.candidates()
            .filter((entry) => state.candidate_answer_records[entry.candidate.diagnostic_uuid]?.validity === "VALID")
            .map((entry) => entry.candidate.diagnostic_uuid))
          : new Set(state.candidate_answered_uuids));
      const entries = this.candidates();
      const index = entries.findIndex((entry) => !answered.has(entry.candidate.diagnostic_uuid));
      return index >= 0 ? index : Math.max(0, entries.length - 1);
    }

    renderCandidateQueue(annotation, dense) {
      const state = this.state();
      const entries = this.candidates();
      const entry = this.currentCandidateEntry();
      if (!entry) {
        if (this.revisionAware()) this.markSummaryReady(state);
        else state.step = 4;
        return this.renderReview(annotation);
      }
      const number = state.candidate_index + 1;
      const progress = this.candidateProgress(state);
      const targets = this.host.objects(annotation);
      let content;
      if (state.candidate_phase === "relation") {
        content = answerButtons("What does this machine box represent?", CANDIDATE_RELATIONS, "candidate_relation", {columns: 1, hint: "The highlighted box is selected for you. You do not need to click it."});
      } else if (state.candidate_phase === "targets") {
        const merged = state.candidate_relation === "MERGED_MULTIPLE_INSTANCES";
        content = `<section class="nwQuestionCard" data-nw-question="candidate_targets"><span class="nwQuestionLabel">CHOOSE THE ${merged ? "PEOPLE" : "PERSON"}</span><h3>${merged ? "Which people are inside this machine box?" : "Which person does this machine box belong to?"}</h3><div class="nwPersonCards">${targets.map((row, index) => `<button type="button" data-nw-target="${row.annotation_uuid}" class="${state.candidate_targets.includes(row.annotation_uuid) ? "selected" : ""}"><span>Person ${index + 1}</span><strong>${state.candidate_targets.includes(row.annotation_uuid) ? "Selected" : "Choose"}</strong></button>`).join("") || "<p>No people have been marked.</p>"}</div>${merged ? `<button id="nwConfirmTargets" class="nwPrimary" type="button" ${state.candidate_targets.length >= 2 ? "" : "disabled"}>Use these ${state.candidate_targets.length} people</button>` : ""}</section>`;
      } else if (state.candidate_phase === "coverage") {
        content = answerButtons("How much of the person's visible shape is inside this machine box?", [["Almost none", 0], ["About one quarter", 0.25], ["About half", 0.5], ["About three quarters", 0.75], ["Almost all", 1]], "candidate_coverage", {columns: 1});
      } else {
        content = answerButtons("Why might the machine have struggled here?", FAILURE_CHOICES, "failure", {columns: 1, hint: "Choose I don't know if the technical reason is not clear."});
      }
      return this.shell(`<div class="nwCandidateHeader"><span>Machine Box ${number} of ${entries.length}</span><strong>${progress.valid}/${entries.length} valid</strong></div>${this.revisionAware() ? `<div class="nwValidityProgress"><span>${progress.valid} valid</span><span>${progress.stale} need checking</span><span>${progress.unanswered} unanswered</span><span>${progress.invalid} invalid</span></div>` : ""}${content}<div class="nwActionRow"><button id="nwPreviousCandidate" type="button" ${state.candidate_index ? "" : "disabled"}>Previous machine box</button><button id="nwNextUnansweredCandidate" type="button">Next unanswered</button><button id="nwUndo" type="button">Undo answer</button></div>`, "Now check the machine's boxes. We will show them one at a time.");
    }

    renderTemporal(annotation) {
      const state = this.state();
      if (state.step === 4) return this.renderReview(annotation);
      const index = this.host.frameIndex();
      const frame = annotation.frames[index];
      const answered = new Set(state.frame_answered_sequences);
      let content;
      if (state.frame_phase === "visibility") {
        content = answerButtons("Can you see this person in the current frame?", TEMPORAL_STATES, "temporal_state", {columns: 1, hint: "Check this frame itself. Nearby frames may help, but a prediction is not an observation."});
      } else if (state.frame_phase === "geometry") {
        const candidates = this.host.currentFrameCandidates("person");
        content = `<section class="nwQuestionCard" data-nw-question="temporal_geometry"><span class="nwQuestionLabel">MARK THIS FRAME</span><h3>Where is the visible person?</h3><p class="nwQuestionHint">Use a highlighted suggestion if it is right, or draw the visible box yourself.</p><div class="nwChoices nwColumns1"><button id="nwUseTemporalCandidate" type="button" ${candidates.length ? "" : "disabled"}>Use the highlighted machine suggestion</button><button id="nwNextTemporalCandidate" type="button" ${candidates.length > 1 ? "" : "disabled"}>Show another suggestion</button><button id="nwDrawTemporalBox" class="nwPrimary" type="button">Draw the visible person</button></div></section>`;
      } else if (state.frame_phase === "footpoint") {
        content = `<section class="nwQuestionCard" data-nw-question="temporal_footpoint"><span class="nwQuestionLabel">MARK THIS FRAME</span><h3>Where do their feet touch the ground?</h3><p class="nwQuestionHint">Click the visible foot position in this frame.</p><button id="nwPlaceFootpoint" class="nwPrimary" type="button">Place the foot point</button></section>`;
      } else if (state.frame_phase === "copied_draft") {
        content = `<section class="nwQuestionCard" data-nw-question="copied_geometry"><span class="nwQuestionLabel">STARTING BOX ONLY</span><h3>Does the copied box fit this frame?</h3><p class="nwQuestionHint">The dashed box is a draft from the previous frame. It is not an observation until you confirm it here.</p><div class="nwChoices nwColumns1"><button id="nwConfirmCopiedGeometry" class="nwPrimary" type="button">Yes, confirm it on this frame</button><button id="nwRejectCopiedGeometry" type="button">No, discard the copied box</button></div></section>`;
      } else {
        content = `<section class="nwQuestionCard"><h3>All 11 frames have been checked.</h3><p>Review the frame strip before moving to the summary.</p><button id="nwConfirmTemporalRun" class="nwPrimary" type="button">Review all frames</button></section>`;
      }
      return this.shell(`<div class="nwFrameHeader"><strong>Frame ${index + 1} of ${annotation.frames.length}</strong><span>${answered.size}/${annotation.frames.length} checked</span></div>${content}<div class="nwActionRow"><button id="nwPreviousFrameQuestion" type="button" ${index ? "" : "disabled"}>Previous frame</button><button id="nwNextFrameQuestion" type="button" ${index + 1 < annotation.frames.length ? "" : "disabled"}>Next frame</button>${frame.visible_body_box ? "<button id=\"nwCopyGeometry\" type=\"button\">Copy as a starting box</button>" : ""}</div><p class="nwPredictionWarning">Predicted means no person box was observed in this frame.</p>`, "Go through all 11 frames and record only what is actually visible in each one.");
    }

    renderPitch(annotation) {
      const state = this.state();
      if (state.step === 1) return this.shell(`<section class="nwActionCard"><h3>Mark the person's foot position</h3><p>Click where their feet touch the ground. The field outline is shown to help.</p><button id="nwPlaceFootpoint" class="nwPrimary" type="button">Place the foot point</button></section>`, "Click where this person's feet touch the ground.");
      if (state.step === 4) return this.renderReview(annotation);
      const keys = ["pitch_uncertainty", "pitch_state", "pitch_role", "pitch_supply"];
      const key = keys[state.pitch_question_index];
      let content;
      if (key === "pitch_uncertainty") content = answerButtons("How sure are you about that foot position?", FOOT_UNCERTAINTY, key, {columns: 1});
      else if (key === "pitch_state") content = answerButtons("Where are their feet relative to the playing field?", PITCH_CHOICES, key, {columns: 1});
      else if (key === "pitch_role") content = answerButtons("Who is this?", ROLE_CHOICES, key, {columns: 2});
      else content = answerButtons("Should this person count as someone on the pitch?", [["Yes", "YES"], ["No", "NO"], ["I can't tell", "UNSURE"]], key, {columns: 1, hint: annotation.pitch_state === "ON_PITCH" ? "Use what you can see in this frame." : "People outside or on an uncertain boundary cannot enter the on-pitch set."});
      return this.shell(`${content}<div class="nwActionRow"><button id="nwPitchBack" type="button" ${state.pitch_question_index ? "" : "disabled"}>Back</button></div>`, "Answer one short question about the marked foot position.");
    }

    renderFootball(annotation) {
      const state = this.state();
      if (state.step === 4) return this.renderReview(annotation);
      if (state.step === 3) return this.renderFootballCandidateQueue(annotation);
      const index = this.host.frameIndex();
      const answered = new Set(state.frame_answered_sequences);
      let content;
      if (state.frame_phase === "visibility") {
        content = answerButtons("Can you see the football anywhere in this full frame?", FOOTBALL_STATES, "football_state", {columns: 1, hint: "Check the whole image, not just the suggested crop."});
      } else if (state.frame_phase === "centre") {
        content = `<section class="nwQuestionCard"><span class="nwQuestionLabel">MARK THIS FRAME</span><h3>Click the centre of the football</h3><p class="nwQuestionHint">Zoom in if needed, then click the ball itself.</p><button id="nwPlaceBall" class="nwPrimary" type="button">Mark the football centre</button></section>`;
      } else if (state.frame_phase === "blur") {
        content = `<section class="nwQuestionCard"><span class="nwQuestionLabel">BLURRED FOOTBALL</span><h3>Can you see a clear blur trail?</h3><div class="nwChoices nwColumns1"><button id="nwDrawBallTrail" type="button">Draw the blur trail</button><button id="nwSkipBallTrail" type="button">No clear trail</button></div></section>`;
      } else {
        content = `<section class="nwQuestionCard"><h3>All 9 frames have been checked.</h3><button id="nwBeginBallCandidates" class="nwPrimary" type="button">Check the machine's ball suggestions</button></section>`;
      }
      return this.shell(`<div class="nwFrameHeader"><strong>Frame ${index + 1} of ${annotation.frames.length}</strong><span>${answered.size}/${annotation.frames.length} checked</span></div>${content}<div class="nwActionRow"><button id="nwPreviousFrameQuestion" type="button" ${index ? "" : "disabled"}>Previous frame</button><button id="nwNextFrameQuestion" type="button" ${index + 1 < annotation.frames.length ? "" : "disabled"}>Next frame</button></div>`, "Check the whole image in every frame and record whether the football is visible.");
    }

    renderFootballCandidateQueue(annotation) {
      const state = this.state();
      const entries = this.candidates();
      const entry = this.currentCandidateEntry();
      if (!entry) {
        state.step = 4;
        return this.renderReview(annotation);
      }
      const answer = state.football_candidate_answers[entry.candidate.diagnostic_uuid];
      let content;
      if (state.candidate_phase === "hard_negative") {
        content = answerButtons("What might this machine suggestion be instead?", HARD_NEGATIVES, "football_hard_negative", {columns: 1});
      } else {
        content = answerButtons("Is this the football?", [["Yes", "YES"], ["No", "NO"], ["Not sure", "UNSURE"]], "football_candidate", {columns: 1, hint: "Only the current highlighted suggestion is being checked."});
      }
      return this.shell(`<div class="nwCandidateHeader"><span>Machine Ball Suggestion ${state.candidate_index + 1} of ${entries.length}</span><strong>${Object.keys(state.football_candidate_answers).length}/${entries.length} checked</strong></div>${content}<div class="nwActionRow"><button id="nwPreviousCandidate" type="button" ${state.candidate_index ? "" : "disabled"}>Previous suggestion</button><button id="nwNextUnansweredCandidate" type="button">Next unanswered</button></div>${answer ? `<p class="nwSavedAnswer">Current answer: ${answer}</p>` : ""}`, "Check one machine ball suggestion at a time.");
    }

    renderReview(annotation) {
      const task = this.host.caseData().task_type;
      const state = this.state();
      let rows = "";
      if (task === "detection_gold_player_static") {
        rows = annotation.player_instances.map((person, index) => `<li><strong>Person ${index + 1}</strong><span>${person.coarse_role.replaceAll("_", " ").toLowerCase()} - ${person.visibility_state.replaceAll("_", " ").toLowerCase()}</span><button type="button" data-nw-edit-object="${person.annotation_uuid}">Edit Person ${index + 1}</button></li>`).join("");
      } else if (task === "detection_gold_dense_region") {
        rows = annotation.visible_masks.map((mask, index) => `<li><strong>Person ${index + 1}</strong><span>${mask.mask_quality.toLowerCase()} outline - order ${mask.occlusion_order + 1}</span><button type="button" data-nw-edit-object="${mask.annotation_uuid}">Edit Person ${index + 1}</button></li>`).join("");
      } else if (task === "detection_gold_temporal_player") {
        rows = annotation.frames.map((frame, index) => `<li><strong>Frame ${index + 1}</strong><span>${frame.state.replaceAll("_", " ").toLowerCase()}</span><button type="button" data-nw-edit-frame="${index}">Return to Frame ${index + 1}</button></li>`).join("");
      } else if (task === "detection_gold_pitch_boundary") {
        rows = `<li><strong>Foot position</strong><span>${annotation.pitch_state.replaceAll("_", " ").toLowerCase()}</span><button type="button" id="nwEditPitch">Edit</button></li><li><strong>Role</strong><span>${annotation.coarse_role.replaceAll("_", " ").toLowerCase()}</span></li>`;
      } else {
        rows = annotation.frames.map((frame, index) => `<li><strong>Frame ${index + 1}</strong><span>${frame.state.replaceAll("_", " ").toLowerCase()}</span><button type="button" data-nw-edit-frame="${index}">Return to Frame ${index + 1}</button></li>`).join("");
      }
      const usesCandidateQueue = ["detection_gold_player_static", "detection_gold_dense_region", "detection_gold_football_burst"].includes(task);
      const candidateTotal = this.candidates().length;
      const revisionProgress = this.candidateProgress(state);
      const candidateDone = task === "detection_gold_football_burst"
        ? Object.keys(state.football_candidate_answers).length
        : (this.revisionAware() ? revisionProgress.valid : state.candidate_answered_uuids.length);
      const candidateSummary = usesCandidateQueue
        ? `<span><strong>${candidateDone}/${candidateTotal}</strong> machine boxes valid${this.revisionAware() && (revisionProgress.stale || revisionProgress.unanswered || revisionProgress.invalid) ? `; ${revisionProgress.stale} need checking, ${revisionProgress.unanswered} unanswered` : ""}</span>`
        : "";
      const candidateLinks = usesCandidateQueue && candidateTotal
        ? `<details class="nwCandidateReviewLinks"><summary>Review a machine-box answer</summary><div>${this.candidates().map((entry, index) => `<button type="button" data-nw-edit-candidate="${index}">Review Machine Box ${index + 1}</button>`).join("")}</div></details>`
        : "";
      const saveBlocked = this.revisionAware() && ["detection_gold_player_static", "detection_gold_dense_region"].includes(task)
        && (revisionProgress.stale || revisionProgress.unanswered || revisionProgress.invalid || state.summary_validity !== "VALID");
      return this.shell(`<section class="nwReviewCard"><h3>Check your work</h3><div class="nwReviewStats"><span><strong>${task.includes("temporal") || task.includes("football") ? state.frame_answered_sequences.length : this.host.objects(annotation).length}</strong> ${task.includes("temporal") || task.includes("football") ? "frames checked" : "people marked"}</span>${candidateSummary}</div><ul class="nwReviewList">${rows || "<li><strong>No visible people marked</strong><span>This is allowed when none are visible.</span></li>"}</ul>${candidateLinks}<details class="nwNote"><summary>Add a note</summary><textarea id="nwNote" maxlength="1000" rows="3" placeholder="Optional">${escapeHtml(annotation.note || "")}</textarea></details><button id="nwSaveCase" class="nwPrimary nwSave" type="button" ${saveBlocked ? "disabled" : ""}>Save this case</button></section>`, "Check the plain-language summary, then save to the server.");
    }

    render(annotation) {
      const task = this.host.caseData().task_type;
      if (["detection_gold_player_static", "detection_gold_dense_region"].includes(task)) return this.renderStaticOrDense(annotation);
      if (task === "detection_gold_temporal_player") return this.renderTemporal(annotation);
      if (task === "detection_gold_pitch_boundary") return this.renderPitch(annotation);
      return this.renderFootball(annotation);
    }

    handleAnswer(key, rawValue) {
      const annotation = this.host.annotation();
      const state = this.state();
      const selected = this.host.objects(annotation).find((row) => row.annotation_uuid === state.current_object_uuid);
      this.mutate((draftState) => {
        if (this.revisionAware() && state.step === 2 && selected) {
          const candidateRelevant = new Set([
            "visibility", "occluder", "hidden_amount", "edge", "mask_quality", "mask_front", "mask_truncation",
          ]).has(key);
          this.objectSemanticChanged(selected.annotation_uuid, {candidateRelevant});
        }
        if (key === "role") selected.coarse_role = rawValue;
        else if (key === "visibility") {
          selected.visibility_state = rawValue;
          selected.ambiguity_ignore = rawValue === "UNRESOLVED";
          if (rawValue === "VISIBLE") {
            selected.occlusion_type = "NONE";
            selected.occlusion_fraction = 0;
          } else if (rawValue === "UNRESOLVED") {
            selected.occlusion_type = "UNKNOWN";
            selected.occlusion_fraction = 0.5;
          }
        } else if (key === "occluder") selected.occlusion_type = rawValue;
        else if (key === "hidden_amount") selected.occlusion_fraction = Number(rawValue);
        else if (key === "footpoint_review") {
          if (rawValue === "MOVE_IT") {
            draftState.pending_footpoint_decision = "MOVE_IT";
            this.host.setTool("footpoint");
            return;
          }
          if (rawValue === "YES") {
            const box = selected.visible_body_box;
            selected.footpoint = {x: (box.x1 + box.x2) / 2, y: box.y2};
            selected.footpoint_uncertainty_pixels = 3;
            draftState.footpoint_reviews[selected.annotation_uuid] = {
              decision: "YES",
              estimated: false,
              adjusted: false,
            };
          } else {
            this.host.estimateHiddenFootpoint(selected);
            draftState.footpoint_reviews[selected.annotation_uuid] = {
              decision: rawValue,
              estimated: true,
              adjusted: false,
            };
          }
          if (!draftState.footpoint_placed_uuids.includes(selected.annotation_uuid)) {
            draftState.footpoint_placed_uuids.push(selected.annotation_uuid);
          }
        }
        else if (key === "foot_uncertainty") selected.footpoint_uncertainty_pixels = Number(rawValue);
        else if (key === "pitch") selected.pitch_state = rawValue;
        else if (key === "edge") {
          selected.truncation_flags = ["LEFT", "TOP", "RIGHT", "BOTTOM"].includes(rawValue) ? [rawValue] : [];
          if (rawValue === "UNSURE") selected.ambiguity_ignore = true;
        } else if (key === "mask_quality") selected.mask_quality = rawValue;
        else if (key === "mask_front") {
          if (rawValue === "NONE") delete selected.occluder_uuid;
          else if (rawValue === "UNSURE") selected.mask_quality = "UNCERTAIN";
          else {
            selected.occluder_uuid = rawValue;
            selected.occlusion_order = Math.max(selected.occlusion_order, 1);
          }
        } else if (key === "mask_truncation") selected.truncation_flags = ["LEFT", "TOP", "RIGHT", "BOTTOM"].includes(rawValue) ? [rawValue] : [];
        else if (key === "candidate_relation") {
          draftState.candidate_relation = rawValue;
          draftState.candidate_targets = [];
          if (["BACKGROUND", "AMBIGUOUS"].includes(rawValue)) {
            this.commitCandidateRelation(rawValue, [], undefined, draftState, annotation);
            return;
          }
          draftState.candidate_phase = "targets";
          return;
        } else if (key === "candidate_coverage") {
          this.commitCandidateRelation(draftState.candidate_relation, draftState.candidate_targets, Number(rawValue), draftState, annotation);
          return;
        } else if (key === "failure") {
          annotation.earliest_failure_stage = rawValue;
          draftState.failure_reviewed = true;
          draftState.candidate_phase = "relation";
          if (this.revisionAware()) this.markSummaryReady(draftState);
          else draftState.step = 4;
          return;
        } else if (key === "temporal_state") {
          this.setTemporalChoice(rawValue, draftState, annotation);
          return;
        } else if (key === "pitch_uncertainty") annotation.footpoint_uncertainty_pixels = Number(rawValue);
        else if (key === "pitch_state") {
          annotation.pitch_state = rawValue;
          if (rawValue !== "ON_PITCH") annotation.primary_on_pitch_supply_eligible = false;
        } else if (key === "pitch_role") annotation.coarse_role = rawValue;
        else if (key === "pitch_supply") annotation.primary_on_pitch_supply_eligible = rawValue === "YES" && annotation.pitch_state === "ON_PITCH";
        else if (key === "football_state") {
          this.setFootballChoice(rawValue, draftState, annotation);
          return;
        } else if (key === "football_candidate") {
          this.setFootballCandidateChoice(rawValue, draftState, annotation);
          return;
        } else if (key === "football_hard_negative") {
          const entry = this.currentCandidateEntry();
          const frame = annotation.frames[entry.frameIndex];
          if (frame.state === "NOT_VISIBLE") frame.hard_negative_category = rawValue;
          draftState.football_candidate_answers[entry.candidate.diagnostic_uuid] = `NO:${rawValue}`;
          this.advanceCandidate(draftState, {football: true});
          return;
        }
        if (state.step === 2 && selected) draftState.question_index += 1;
        if (this.host.caseData().task_type === "detection_gold_pitch_boundary") {
          if (!draftState.pitch_answers.includes(key)) draftState.pitch_answers.push(key);
          draftState.pitch_question_index += 1;
          if (draftState.pitch_question_index >= 4) draftState.step = 4;
        }
      });
    }

    commitCandidateRelation(relation, targets, coverage, state, annotation) {
      const entry = this.currentCandidateEntry();
      if (!entry) return;
      this.host.upsertCandidateRelation(annotation, entry.candidate, relation, targets, coverage);
      if (this.revisionAware()) this.recordCandidateAnswer(state, relation, targets);
      else if (!state.candidate_answered_uuids.includes(entry.candidate.diagnostic_uuid)) state.candidate_answered_uuids.push(entry.candidate.diagnostic_uuid);
      this.advanceCandidate(state);
    }

    advanceCandidate(state, {football = false} = {}) {
      const entries = this.candidates();
      const answered = football
        ? new Set(Object.keys(state.football_candidate_answers))
        : (this.revisionAware()
          ? new Set(entries
            .filter((entry) => state.candidate_answer_records[entry.candidate.diagnostic_uuid]?.validity === "VALID")
            .map((entry) => entry.candidate.diagnostic_uuid))
          : new Set(state.candidate_answered_uuids));
      const next = entries.findIndex((entry) => !answered.has(entry.candidate.diagnostic_uuid));
      state.candidate_relation = null;
      state.candidate_targets = [];
      state.candidate_phase = "relation";
      if (next >= 0) {
        state.candidate_index = next;
        return;
      }
      if (this.host.caseData().task_type === "detection_gold_player_static" && !state.failure_reviewed) {
        state.candidate_phase = "failure";
        return;
      }
      if (this.revisionAware()) this.markSummaryReady(state);
      else state.step = 4;
    }

    setTemporalChoice(value, state, annotation) {
      const index = this.host.frameIndex();
      const record = this.host.record();
      const frame = annotation.frames[index];
      frame.state = value;
      frame.current_frame_pixel_support = false;
      frame.candidate_uuids = [];
      delete frame.visible_body_box;
      delete frame.footpoint;
      state.desired_frame_state = value;
      if (["OBSERVED", "OBSERVED_WITH_TEMPORAL_REFINEMENT"].includes(value)) {
        state.frame_phase = "geometry";
        this.host.setFirstFrameCandidate("person");
      } else {
        frame.frame_sequence = record.frame_sequence;
        frame.source_frame_sha256 = record.source_frame_sha256;
        this.markFrameAnswered(state, frame.frame_sequence, annotation.frames.length);
      }
    }

    setFootballChoice(value, state, annotation) {
      const frame = annotation.frames[this.host.frameIndex()];
      frame.state = value;
      for (const key of ["centre_point", "apparent_ellipse", "visible_mask_polygon", "blur_trail_endpoints", "blur_trail_width", "apparent_diameter", "geometry_uncertainty_pixels", "hard_negative_category"]) delete frame[key];
      state.desired_frame_state = value;
      if (["VISIBLE_CLEAR", "VISIBLE_BLURRED", "PARTIALLY_OCCLUDED_VISIBLE"].includes(value)) {
        state.frame_phase = "centre";
      } else {
        this.markFrameAnswered(state, frame.frame_sequence, annotation.frames.length, {football: true});
      }
    }

    markFrameAnswered(state, sequence, total, {football = false} = {}) {
      if (!state.frame_answered_sequences.includes(sequence)) state.frame_answered_sequences.push(sequence);
      if (state.frame_answered_sequences.length >= total) {
        state.frame_phase = "complete";
        if (football) this.host.annotation().full_contact_strip_reviewed = true;
        else this.host.annotation().contact_strip_reviewed = true;
        return;
      }
      const frames = this.host.annotation().frames;
      const next = frames.findIndex((row) => !state.frame_answered_sequences.includes(row.frame_sequence));
      this.host.setFrameSilently(Math.max(0, next));
      state.frame_phase = "visibility";
      state.desired_frame_state = null;
    }

    finishCurrentFrame() {
      this.mutate((state, annotation) => {
        const frame = annotation.frames[this.host.frameIndex()];
        if (this.host.caseData().task_type === "detection_gold_temporal_player") {
          frame.state = state.desired_frame_state || frame.state;
          frame.current_frame_pixel_support = true;
          this.markFrameAnswered(state, frame.frame_sequence, annotation.frames.length);
        } else {
          this.markFrameAnswered(state, frame.frame_sequence, annotation.frames.length, {football: true});
        }
      }, {history: false});
    }

    setFootballCandidateChoice(value, state, annotation) {
      const entry = this.currentCandidateEntry();
      if (!entry) return;
      const frame = annotation.frames[entry.frameIndex];
      if (value === "YES") {
        const box = entry.candidate.bbox_original_pixels;
        frame.state = frame.state.startsWith("VISIBLE") || frame.state === "PARTIALLY_OCCLUDED_VISIBLE" ? frame.state : "VISIBLE_CLEAR";
        frame.centre_point ||= {x: (box.x1 + box.x2) / 2, y: (box.y1 + box.y2) / 2};
        frame.geometry_uncertainty_pixels ||= Math.max(2, Math.min(box.x2 - box.x1, box.y2 - box.y1) / 2);
        state.football_candidate_answers[entry.candidate.diagnostic_uuid] = "YES";
        this.advanceCandidate(state, {football: true});
      } else if (value === "NO") {
        state.candidate_phase = "hard_negative";
      } else {
        state.football_candidate_answers[entry.candidate.diagnostic_uuid] = "UNSURE";
        this.advanceCandidate(state, {football: true});
      }
    }

    goToFrame(index) {
      const state = this.state();
      state.frame_phase = "visibility";
      state.desired_frame_state = null;
      this.host.setFrame(index);
    }

    geometryCopiedToFrame(index) {
      this.mutate((state) => {
        this.host.setFrameSilently(index);
        state.frame_phase = "copied_draft";
        state.desired_frame_state = "OBSERVED_WITH_TEMPORAL_REFINEMENT";
      }, {history: false});
    }

    resolveCopiedGeometry(confirmed) {
      const resolved = confirmed ? this.host.confirmGeometryDraft() : this.host.rejectGeometryDraft();
      if (!resolved) return;
      this.mutate((state, annotation) => {
        if (confirmed) {
          const frame = annotation.frames[this.host.frameIndex()];
          this.markFrameAnswered(state, frame.frame_sequence, annotation.frames.length);
        } else {
          state.frame_phase = "visibility";
          state.desired_frame_state = null;
        }
      }, {history: false});
    }

    bind() {
      const one = (selector, handler) => document.querySelector(selector)?.addEventListener("click", handler);
      document.querySelectorAll("[data-nw-answer-key]").forEach((button) => button.addEventListener("click", () => this.handleAnswer(button.dataset.nwAnswerKey, button.dataset.nwAnswerValue)));
      document.querySelectorAll("[data-nw-edit-object]").forEach((button) => button.addEventListener("click", () => {
        const annotationUuid = button.dataset.nwEditObject;
        this.host.selectObject(annotationUuid);
        this.mutate((state) => {
          state.current_object_uuid = annotationUuid;
          state.question_index = 0;
          state.step = 2;
        });
      }));
      document.querySelectorAll("[data-nw-edit-frame]").forEach((button) => button.addEventListener("click", () => this.mutate((state) => {
        this.host.setFrameSilently(Number(button.dataset.nwEditFrame));
        state.step = 1;
        state.frame_phase = "visibility";
        state.desired_frame_state = null;
      })));
      document.querySelectorAll("[data-nw-edit-candidate]").forEach((button) => button.addEventListener("click", () => this.mutate((state) => {
        state.step = 3;
        state.candidate_index = Number(button.dataset.nwEditCandidate);
        state.candidate_phase = "relation";
      })));
      one("#nwDrawObject", () => this.host.setTool(this.host.caseData().task_type === "detection_gold_dense_region" ? "mask" : "box"));
      one("#nwFinishOutline", this.host.finishMask);
      one("#nwDoneDrawing", () => this.mutate((state) => {
        const unfinished = this.host.objects().find(
          (row) => !state.completed_object_uuids.includes(row.annotation_uuid)
        );
        if (unfinished) {
          state.current_object_uuid = unfinished.annotation_uuid;
          state.question_index = 0;
          state.step = 2;
          return;
        }
        state.drawing_complete = true;
        if (this.candidates().length) {
          state.step = 3;
          state.candidate_index = this.nextUnansweredCandidateIndex();
        } else if (this.revisionAware()) this.markSummaryReady(state);
        else state.step = 4;
      }));
      one("#nwUndo", this.host.undo);
      one("#nwQuestionBack", () => this.mutate((state) => { state.question_index = Math.max(0, state.question_index - 1); }));
      one("#nwDeleteObject", this.host.removeSelected);
      one("#nwDeleteAllObjects", this.host.deleteAllObjects);
      one("#nwReturnDrawing", () => this.mutate((state) => {
        state.step = 1;
        state.current_object_uuid = null;
        state.question_index = 0;
      }, {history: false}));
      one("#nwReviewStale", () => this.mutate((state) => {
        state.step = 3;
        state.candidate_index = this.nextUnansweredCandidateIndex();
        state.candidate_phase = "relation";
        state.candidate_relation = null;
        state.candidate_targets = [];
      }, {history: false}));
      one("#nwRestartCase", this.host.restartCase);
      one("#nwPlaceFootpoint", () => this.host.setTool("footpoint"));
      one("#nwAdjustFootpoint", () => this.host.setTool("footpoint"));
      one("#nwPlaceBall", () => this.host.setTool("ball"));
      one("#nwDrawBallTrail", () => this.host.setTool("trail"));
      one("#nwSkipBallTrail", () => this.finishCurrentFrame());
      one("#nwDrawTemporalBox", () => this.host.setTool("box"));
      one("#nwUseTemporalCandidate", () => this.host.useTemporalCandidate(this.state().desired_frame_state));
      one("#nwNextTemporalCandidate", this.host.nextFrameCandidate);
      one("#nwCopyGeometry", this.host.copyGeometry);
      one("#nwConfirmCopiedGeometry", () => this.resolveCopiedGeometry(true));
      one("#nwRejectCopiedGeometry", () => this.resolveCopiedGeometry(false));
      one("#nwConfirmTemporalRun", () => this.mutate((state, annotation) => {
        annotation.stable_run_accepted = true;
        state.step = 4;
      }));
      one("#nwBeginBallCandidates", () => this.mutate((state) => {
        state.step = this.candidates().length ? 3 : 4;
        state.candidate_index = 0;
        state.candidate_phase = "relation";
      }));
      one("#nwPreviousFrameQuestion", () => this.goToFrame(Math.max(0, this.host.frameIndex() - 1)));
      one("#nwNextFrameQuestion", () => this.goToFrame(Math.min(this.host.records().length - 1, this.host.frameIndex() + 1)));
      document.querySelectorAll("[data-nw-target]").forEach((button) => button.addEventListener("click", () => this.mutate((state, annotation) => {
        const uuid = button.dataset.nwTarget;
        if (state.candidate_relation === "MERGED_MULTIPLE_INSTANCES") {
          state.candidate_targets = state.candidate_targets.includes(uuid) ? state.candidate_targets.filter((value) => value !== uuid) : [...state.candidate_targets, uuid];
          return;
        }
        state.candidate_targets = [uuid];
        if (this.host.caseData().task_type === "detection_gold_dense_region") state.candidate_phase = "coverage";
        else this.commitCandidateRelation(state.candidate_relation, state.candidate_targets, undefined, state, annotation);
      })));
      one("#nwConfirmTargets", () => this.mutate((state, annotation) => {
        if (this.host.caseData().task_type === "detection_gold_dense_region") state.candidate_phase = "coverage";
        else this.commitCandidateRelation(state.candidate_relation, state.candidate_targets, undefined, state, annotation);
      }));
      one("#nwPreviousCandidate", () => this.mutate((state) => {
        state.candidate_index = Math.max(0, state.candidate_index - 1);
        state.candidate_phase = "relation";
      }));
      one("#nwNextUnansweredCandidate", () => this.mutate((state) => {
        state.candidate_index = this.nextUnansweredCandidateIndex({
          football: this.host.caseData().task_type === "detection_gold_football_burst",
        });
        state.candidate_phase = "relation";
      }));
      one("#nwPitchBack", () => this.mutate((state) => { state.pitch_question_index = Math.max(0, state.pitch_question_index - 1); }));
      one("#nwEditPitch", () => this.mutate((state) => { state.step = 1; state.pitch_footpoint_set = false; }));
      one("#nwSaveCase", this.host.save);
      document.querySelector("#nwNote")?.addEventListener("input", (event) => {
        this.host.annotation().note = event.target.value;
        this.host.persist();
      });
      one("#nwCurrentHelp", () => this.showTour(true));
    }

    showTour(force = false) {
      const dialog = document.querySelector("#nwTour");
      if (!dialog) return;
      const key = `fi_detection_gold_tour_${this.host.reviewId()}`;
      if (!force && localStorage.getItem(key) === "done") return;
      dialog.classList.remove("isHidden");
      dialog.setAttribute("aria-hidden", "false");
      const start = dialog.querySelector("#nwTourStart");
      const skip = dialog.querySelector("#nwTourSkip");
      if (!start || !skip) return;
      start.onclick = () => {
        localStorage.setItem(key, "done");
        dialog.classList.add("isHidden");
        dialog.setAttribute("aria-hidden", "true");
      };
      skip.onclick = start.onclick;
    }

    validateForSave() {
      const state = this.state();
      const task = this.host.caseData().task_type;
      if (state.step !== 4) throw new Error("Finish the four guided steps before saving.");
      if (["detection_gold_player_static", "detection_gold_dense_region"].includes(task)) {
        const ids = this.host.objects().map((row) => row.annotation_uuid);
        if (ids.some((uuid) => !state.completed_object_uuids.includes(uuid))) throw new Error("Finish the short questions for every marked person.");
        if (this.revisionAware()) {
          const progress = this.candidateProgress(state);
          if (progress.stale || progress.invalid) throw new Error("Review every stale machine-box answer before saving.");
          if (progress.unanswered || progress.valid !== progress.total) throw new Error("Check every machine box before saving.");
          if (state.summary_validity !== "VALID") throw new Error("Review the updated case summary before saving.");
          if (!state.drawing_complete) throw new Error("Explicitly confirm that you are done drawing people before saving.");
          if (Object.keys(state.person_question_completion_revisions).length !== ids.length) {
            throw new Error("Finish the current person questions before saving.");
          }
        } else if (state.candidate_answered_uuids.length !== this.candidates().length) {
          throw new Error("Check every machine box before saving.");
        }
        if (task === "detection_gold_player_static" && !state.failure_reviewed) throw new Error("Answer the final machine-difficulty question.");
        if (this.host.incrementalR3?.() === true) {
          const binding = this.host.authoritativeBinding();
          if (state.primary_canvas_frame_sequence !== binding.frame_sequence
            || state.primary_canvas_source_frame_sha256 !== binding.source_frame_sha256
            || state.candidate_queue_binding_hash !== binding.candidate_queue_binding_hash) {
            throw new Error("The editable canvas or candidate queue is not bound to the authoritative middle frame.");
          }
          if (task === "detection_gold_player_static"
            && ids.some((uuid) => !state.footpoint_reviews?.[uuid])) {
            throw new Error("Confirm or correct the proposed footpoint for every marked person.");
          }
        }
      }
      if (["detection_gold_temporal_player", "detection_gold_football_burst"].includes(task)) {
        if (state.frame_answered_sequences.length !== this.host.annotation().frames.length) throw new Error("Check every frame before saving.");
      }
      if (task === "detection_gold_football_burst" && Object.keys(state.football_candidate_answers).length !== this.candidates().length) throw new Error("Check every machine ball suggestion before saving.");
      if (task === "detection_gold_pitch_boundary" && (!state.pitch_footpoint_set || state.pitch_answers.length < 4)) throw new Error("Finish the footpoint and boundary questions before saving.");
    }
  }

  window.DetectionGoldNoviceWizard = {
    create: (host) => new NoviceWizard(host),
    mappings: {
      roles: ROLE_CHOICES,
      visibility: VISIBILITY_CHOICES,
      occluders: OCCLUDER_CHOICES,
      occlusionFractions: OCCLUSION_FRACTIONS,
      footUncertainty: FOOT_UNCERTAINTY,
      pitch: PITCH_CHOICES,
      candidateRelations: CANDIDATE_RELATIONS,
      temporalStates: TEMPORAL_STATES,
      footballStates: FOOTBALL_STATES,
      earliestFailure: FAILURE_CHOICES,
      hardNegatives: HARD_NEGATIVES,
    },
  };
})();
