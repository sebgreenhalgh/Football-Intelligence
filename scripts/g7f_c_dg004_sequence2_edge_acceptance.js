"use strict";

const fs = require("node:fs");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const {spawn} = require("node:child_process");

const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const PRIMARY = "DG-004 strip 2 contains a clear unmasked standing person along the far touchline/track, left of person-001.";
const ACTION = "Add exactly this omitted visible person. Classify relevance only after annotating the person.";
const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

async function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
  });
}

async function waitForPage(port, expectedUrl) {
  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/json/list`);
      const pages = await response.json();
      const page = pages.find(item => item.type === "page" && item.url.startsWith(expectedUrl));
      if (page) return page;
    } catch (_) {
      // Edge is still starting.
    }
    await sleep(100);
  }
  throw new Error("Timed out waiting for Edge DevTools");
}

class DevToolsClient {
  constructor(url) {
    this.url = url;
    this.nextId = 0;
    this.pending = new Map();
  }

  async connect() {
    this.socket = new WebSocket(this.url);
    this.socket.addEventListener("message", event => {
      const message = JSON.parse(event.data);
      if (!message.id || !this.pending.has(message.id)) return;
      const {resolve, reject} = this.pending.get(message.id);
      this.pending.delete(message.id);
      if (message.error) reject(new Error(JSON.stringify(message.error)));
      else resolve(message.result);
    });
    await new Promise((resolve, reject) => {
      this.socket.addEventListener("open", resolve, {once: true});
      this.socket.addEventListener("error", reject, {once: true});
    });
  }

  send(method, params = {}) {
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, {resolve, reject});
      this.socket.send(JSON.stringify({id, method, params}));
    });
  }

  close() {
    if (this.socket) this.socket.close();
  }
}

async function stopEdge(edge, profile) {
  const exited = new Promise(resolve => edge.once("exit", resolve));
  edge.kill();
  await Promise.race([exited, sleep(3000)]);
  for (let attempt = 0; attempt < 20; attempt += 1) {
    try {
      fs.rmSync(profile, {recursive: true, force: true});
      return;
    } catch (error) {
      if (attempt === 19) throw error;
      await sleep(100);
    }
  }
}

async function main() {
  const baseUrl = process.argv[2];
  const screenshot = process.argv[3];
  if (!baseUrl) throw new Error("Usage: edge-acceptance.js <reviewer-url> [screenshot]");
  if (!fs.existsSync(EDGE)) throw new Error(`Microsoft Edge is missing: ${EDGE}`);
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "g7f-c-dg004-seq2-edge-"));
  const debugPort = await freePort();
  const edge = spawn(EDGE, [
    "--headless=new",
    "--disable-gpu",
    "--disable-extensions",
    "--no-first-run",
    "--no-default-browser-check",
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${profile}`,
    "--window-size=1600,1050",
    baseUrl,
  ], {stdio: "ignore", windowsHide: true});
  const page = await waitForPage(debugPort, baseUrl);
  const client = new DevToolsClient(page.webSocketDebuggerUrl);
  const checks = [];
  const record = (name, passed, detail) => {
    checks.push({name, passed: Boolean(passed), detail});
    if (!passed) throw new Error(`${name} failed: ${JSON.stringify(detail)}`);
  };

  try {
    await client.connect();
    await client.send("Runtime.enable");
    await client.send("Page.enable");
    async function evaluate(expression) {
      const response = await client.send("Runtime.evaluate", {expression, awaitPromise: true, returnByValue: true});
      if (response.exceptionDetails) throw new Error(response.exceptionDetails.text || "browser evaluation failed");
      return response.result.value;
    }
    async function waitUntil(expression, timeout = 20000) {
      const deadline = Date.now() + timeout;
      while (Date.now() < deadline) {
        if (await evaluate(expression)) return;
        await sleep(50);
      }
      throw new Error(`Timed out waiting for ${expression}`);
    }
    async function rejected(payloadExpression) {
      return evaluate(`(async () => {
        const payload = ${payloadExpression};
        const response = await fetch("/api/action", {
          method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload),
        });
        return {status: response.status, body: await response.json()};
      })()`);
    }

    try {
      await waitUntil('typeof state !== "undefined" && state.document !== null && state.busy === false && state.image?.complete && state.image.naturalWidth > 0');
    } catch (error) {
      const diagnostic = await evaluate(`({
        hasState:typeof state !== "undefined", documentReady:typeof state === "undefined" ? false : state.document !== null,
        imageReady:typeof state === "undefined" ? false : Boolean(state.image?.complete && state.image.naturalWidth),
        busy:typeof state === "undefined" ? null : state.busy,
        status:document.getElementById("status")?.textContent,
      })`);
      throw new Error(`${error.message}; initial diagnostic=${JSON.stringify(diagnostic)}`);
    }
    const initial = await evaluate(`(() => ({
      queue: state.bootstrap.queue.map(row => row.anonymous_dense_image_id),
      tools: state.bootstrap.tools,
      guidance: [document.getElementById("guidancePrimary").textContent, document.getElementById("guidanceAction").textContent],
      newPersonId: state.newPersonId,
      newPersonPresent: sequence2Api.targetPerson() !== null,
      people: state.document.people.length,
      strips: state.document.reviewed_exhaustiveness_strips,
      completion: state.document.completion_assertion,
      metadata: state.metadata,
      mode: state.mode,
      canvas: [document.getElementById("annotationCanvas").width, document.getElementById("annotationCanvas").height],
      forbiddenControls: ["newIgnore", "deletePerson", "deleteIgnore", "editVertices", "candidateReveal"].filter(id => document.getElementById(id)),
      revision: state.revision,
    }))()`);
    record("queue_is_exactly_dg004", JSON.stringify(initial.queue) === JSON.stringify(["DG-004"]), initial.queue);
    record("candidate_free_one_person_tools_only", initial.forbiddenControls.length === 0
      && initial.tools.visible_mask_polygon && initial.tools.multiple_components
      && !initial.tools.vertex_edit_delete && !initial.tools.ignore_region_polygon
      && initial.tools.maximum_new_people === 1 && !initial.tools.delete_person,
    {forbiddenControls: initial.forbiddenControls, tools: initial.tools});
    record("exact_omission_guidance_visible", initial.guidance[0] === PRIMARY && initial.guidance[1] === ACTION, initial.guidance);
    record("frozen_id_and_fresh_seed", /^person-\d{3}$/.test(initial.newPersonId)
      && !initial.newPersonPresent && initial.strips.length === 0 && initial.completion === null
      && initial.metadata.adjudication_assertion === null && !initial.metadata.repair_checklist_addressed,
    initial);
    record("pan_review_is_default_and_overlay_renders", initial.mode === "PAN_EDIT"
      && initial.people === 50 && initial.canvas[0] === 4096 && initial.canvas[1] === 1080, initial);

    const panSafety = await evaluate(`(() => {
      const canvas=document.getElementById("annotationCanvas"); const rect=canvas.getBoundingClientRect();
      canvas.dispatchEvent(new PointerEvent("pointerdown", {button:0, clientX:rect.left+100, clientY:rect.top+100, bubbles:true}));
      canvas.dispatchEvent(new PointerEvent("pointerup", {button:0, clientX:rect.left+100, clientY:rect.top+100, bubbles:true}));
      return {mode:state.mode, points:state.polygon.length, people:state.document.people.length};
    })()`);
    record("pan_review_never_adds_vertices", panSafety.mode === "PAN_EDIT" && panSafety.points === 0
      && panSafety.people === initial.people, panSafety);

    const spaceSafety = await evaluate(`(() => {
      document.getElementById("drawNewPerson").click();
      window.dispatchEvent(new KeyboardEvent("keydown", {code:"Space", key:" ", bubbles:true}));
      const canvas=document.getElementById("annotationCanvas"); const rect=canvas.getBoundingClientRect();
      canvas.dispatchEvent(new PointerEvent("pointerdown", {button:0, clientX:rect.left+120, clientY:rect.top+100, bubbles:true}));
      canvas.dispatchEvent(new PointerEvent("pointerup", {button:0, clientX:rect.left+140, clientY:rect.top+120, bubbles:true}));
      window.dispatchEvent(new KeyboardEvent("keyup", {code:"Space", key:" ", bubbles:true}));
      return {mode:state.mode, points:state.polygon.length, people:state.document.people.length};
    })()`);
    record("space_temporarily_pans_without_vertices", spaceSafety.mode === "DRAW_PERSON" && spaceSafety.points === 0
      && spaceSafety.people === initial.people, spaceSafety);
    await evaluate("sequence2Api.setMode('PAN_EDIT')");

    const basePayload = `({
      action_id: "crafted-" + crypto.randomUUID(), action_type: "SAVE_DRAFT",
      anonymous_dense_image_id: "DG-004", pass_kind: "CALIBRATION_ADJUDICATION",
      expected_revision: state.revision, document: JSON.parse(JSON.stringify(state.document)),
      adjudication_metadata: JSON.parse(JSON.stringify(state.metadata)),
    })`;
    const earlyFinalize = await rejected(`(() => { const p=${basePayload}; p.action_type="FINALIZE"; return p; })()`);
    record("finalization_blocked_without_new_person", earlyFinalize.status === 422, earlyFinalize);
    const geometry = await rejected(`(() => { const p=${basePayload}; p.document.people[0].visible_mask_components[0][0].x += 1; return p; })()`);
    const relevance = await rejected(`(() => { const p=${basePayload}; p.document.people[0].relevance=p.document.people[0].relevance === "MATCH_RELEVANT" ? "NON_MATCH_RELEVANT" : "MATCH_RELEVANT"; return p; })()`);
    const deletion = await rejected(`(() => { const p=${basePayload}; p.document.people.pop(); return p; })()`);
    const ignore = await rejected(`(() => { const p=${basePayload}; p.document.ignore_regions.push({ignore_region_id:"ignore-x",reason:"OTHER",polygon:[{x:1,y:1},{x:5,y:1},{x:5,y:8}]}); return p; })()`);
    record("crafted_existing_truth_and_ignore_mutations_rejected",
      [geometry, relevance, deletion, ignore].every(row => row.status === 422),
      {geometry, relevance, deletion, ignore});

    await evaluate(`(async () => {
      document.getElementById("drawNewPerson").click();
      state.polygon=[{x:1320,y:150},{x:1342,y:150},{x:1342,y:215},{x:1320,y:215}];
      await sequence2Api.finishPolygon();
    })()`);
    await waitUntil(`state.busy === false && state.revision > ${initial.revision} && sequence2Api.targetPerson() !== null`);
    const afterPerson = await evaluate(`({
      revision:state.revision, mode:state.mode, people:state.document.people.length,
      target:sequence2Api.targetPerson(), drawDisabled:document.getElementById("drawNewPerson").disabled,
    })`);
    record("one_new_person_drawn_with_frozen_id", afterPerson.people === initial.people + 1
      && afterPerson.target.instance_id === initial.newPersonId && afterPerson.target.relevance === null
      && afterPerson.mode === "PAN_EDIT" && afterPerson.drawDisabled, afterPerson);

    const beforeComponentRevision = afterPerson.revision;
    await evaluate(`(async () => {
      document.getElementById("addComponent").click();
      state.polygon=[{x:1344,y:174},{x:1351,y:174},{x:1351,y:186},{x:1344,y:186}];
      await sequence2Api.finishPolygon();
    })()`);
    await waitUntil(`state.busy === false && state.revision > ${beforeComponentRevision}`);
    record("multiple_visible_components_work", await evaluate("sequence2Api.targetPerson().visible_mask_components.length") === 2,
      await evaluate("sequence2Api.targetPerson()"));

    const guardedPayload = `({
      action_id: "crafted-after-new-" + crypto.randomUUID(), action_type: "SAVE_DRAFT",
      anonymous_dense_image_id: "DG-004", pass_kind: "CALIBRATION_ADJUDICATION",
      expected_revision: state.revision, document: JSON.parse(JSON.stringify(state.document)),
      adjudication_metadata: JSON.parse(JSON.stringify(state.metadata)),
    })`;
    const secondNew = await rejected(`(() => { const p=${guardedPayload}; p.document.people.push({instance_id:"person-999",relevance:"NON_MATCH_RELEVANT",visible_mask_components:[[{x:1400,y:150},{x:1415,y:150},{x:1415,y:190},{x:1400,y:190}]]}); return p; })()`);
    const wrongId = await rejected(`(() => { const p=${guardedPayload}; p.document.people[p.document.people.length-1].instance_id="person-999"; return p; })()`);
    record("second_person_and_wrong_new_id_rejected", secondNew.status === 422 && wrongId.status === 422,
      {secondNew, wrongId});

    let revision = await evaluate("state.revision");
    await evaluate('document.getElementById("targetNonMatch").click()');
    await waitUntil(`state.busy === false && state.revision > ${revision}`);
    record("new_person_relevance_can_be_set", await evaluate('sequence2Api.targetPerson().relevance') === "NON_MATCH_RELEVANT",
      await evaluate('sequence2Api.targetPerson().relevance'));

    for (let strip = 0; strip < 8; strip += 1) {
      revision = await evaluate("state.revision");
      await evaluate(`document.querySelector('#stripButtons button[data-strip="${strip}"]').click()`);
      await waitUntil(`state.busy === false && state.revision > ${revision}`);
    }
    for (const id of ["omissionConfirmed", "denseAssertion", "adjudicationAssertion"]) {
      revision = await evaluate("state.revision");
      await evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);
      await waitUntil(`state.busy === false && state.revision > ${revision}`);
    }
    const ready = await evaluate(`({
      strips:state.document.reviewed_exhaustiveness_strips,
      metadata:state.metadata,
      completion:state.document.completion_assertion,
      finalizeDisabled:document.getElementById("finalizeSequence2").disabled,
    })`);
    record("fresh_eight_strips_and_exact_assertions_required", ready.strips.length === 8
      && ready.metadata.repair_checklist_addressed && ready.metadata.adjudication_assertion
      && ready.completion && !ready.finalizeDisabled, ready);

    await evaluate('document.getElementById("finalizeSequence2").click()');
    try {
      await waitUntil("state.finalized === true && state.busy === false", 120000);
    } catch (error) {
      const diagnostic = await evaluate(`({
        busy:state.busy, finalized:state.finalized, revision:state.revision, mode:state.mode,
        target:sequence2Api.targetPerson(), strips:state.document.reviewed_exhaustiveness_strips,
        completion:state.document.completion_assertion, metadata:state.metadata,
        finalizeDisabled:document.getElementById("finalizeSequence2").disabled,
        status:document.getElementById("status").textContent,
      })`);
      throw new Error(`${error.message}; finalize diagnostic=${JSON.stringify(diagnostic)}`);
    }
    const finalized = await evaluate(`({
      finalized:state.finalized, mode:state.mode, revision:state.revision,
      status:document.getElementById("status").textContent,
      finalizeDisabled:document.getElementById("finalizeSequence2").disabled,
    })`);
    record("immutable_sequence2_finalized_once", finalized.finalized && finalized.mode === "PAN_EDIT"
      && finalized.finalizeDisabled && finalized.status.includes("finalized"), finalized);
    await evaluate('document.getElementById("finalizeSequence2").click()');
    await sleep(300);
    record("double_click_is_safe", await evaluate("state.revision") === finalized.revision,
      await evaluate("state.revision"));
    if (screenshot) {
      const capture = await client.send("Page.captureScreenshot", {format: "png", captureBeyondViewport: true});
      fs.writeFileSync(screenshot, Buffer.from(capture.data, "base64"));
    }
  } finally {
    client.close();
    await stopEdge(edge, profile);
  }
  process.stdout.write(JSON.stringify({passed: checks.every(row => row.passed), checks}, null, 2));
}

main().catch(error => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});
