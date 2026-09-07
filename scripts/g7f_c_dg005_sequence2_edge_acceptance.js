"use strict";

const fs = require("node:fs");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const {spawn} = require("node:child_process");

const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const PRIMARY = "person-030 is the visible flag-carrying assistant referee and is currently NON_MATCH_RELEVANT.";
const ACTION = "Confirm the official and set person-030 to MATCH_RELEVANT.";
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
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "g7f-c-dg005-seq2-edge-"));
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
    async function clickAndWait(id, previousRevision) {
      await evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);
      await waitUntil(`state.busy === false && state.revision > ${previousRevision}`);
      return evaluate("state.revision");
    }

    try {
      await waitUntil("state.document !== null && state.busy === false && state.image !== null");
    } catch (error) {
      const diagnostic = await evaluate(`({
        hasState: typeof state !== "undefined",
        hasDocument: typeof state === "undefined" ? false : state.document !== null,
        hasImage: typeof state === "undefined" ? false : state.image !== null,
        busy: typeof state === "undefined" ? null : state.busy,
        status: document.getElementById("status")?.textContent,
      })`);
      throw new Error(`${error.message}; diagnostic=${JSON.stringify(diagnostic)}`);
    }
    const initial = await evaluate(`(() => ({
      queue: state.bootstrap.queue.map(row => row.anonymous_dense_image_id),
      tools: state.bootstrap.tools,
      guidance: [document.getElementById("guidancePrimary").textContent, document.getElementById("guidanceAction").textContent],
      relevance: sequence2Api.targetPerson().relevance,
      strips: state.document.reviewed_exhaustiveness_strips,
      completion: state.document.completion_assertion,
      metadata: state.metadata,
      people: state.document.people.length,
      geometryControls: ["panEdit", "newPerson", "addComponent", "newIgnore", "finishPolygon", "cancelPolygon", "deletePerson", "deleteIgnore", "editRelevance"].filter(id => document.getElementById(id)),
      canvas: [document.getElementById("annotationCanvas").width, document.getElementById("annotationCanvas").height],
      revision: state.revision,
    }))()`);
    record("queue_is_exactly_dg005", JSON.stringify(initial.queue) === JSON.stringify(["DG-005"]), initial.queue);
    record("candidate_free_geometry_controls_absent", initial.geometryControls.length === 0
      && !initial.tools.visible_mask_polygon && !initial.tools.vertex_edit_delete
      && !initial.tools.ignore_region_polygon && !initial.tools.add_person && !initial.tools.delete_person,
    {geometryControls: initial.geometryControls, tools: initial.tools});
    record("exact_guidance_visible", initial.guidance[0] === PRIMARY && initial.guidance[1] === ACTION, initial.guidance);
    record("sequence2_starts_from_seq1_without_auto_change", initial.relevance === "NON_MATCH_RELEVANT"
      && initial.strips.length === 0 && initial.completion === null
      && initial.metadata.adjudication_assertion === null && !initial.metadata.repair_checklist_addressed,
    initial);
    record("candidate_free_overlay_renders", initial.people === 58 && initial.canvas[0] > 0 && initial.canvas[1] > 0, initial);
    process.stderr.write("edge-acceptance: initial state passed\n");

    const basePayload = `({
      action_id: "crafted-" + crypto.randomUUID(), action_type: "SAVE_DRAFT",
      anonymous_dense_image_id: "DG-005", pass_kind: "CALIBRATION_ADJUDICATION",
      expected_revision: state.revision, document: JSON.parse(JSON.stringify(state.document)),
      adjudication_metadata: JSON.parse(JSON.stringify(state.metadata)),
    })`;
    const geometry = await rejected(`(() => { const p=${basePayload}; p.document.people[0].visible_mask_components[0][0].x += 1; return p; })()`);
    const otherRelevance = await rejected(`(() => { const p=${basePayload}; const x=p.document.people.find(row => row.instance_id !== "person-030"); x.relevance=x.relevance === "MATCH_RELEVANT" ? "NON_MATCH_RELEVANT" : "MATCH_RELEVANT"; return p; })()`);
    const addPerson = await rejected(`(() => { const p=${basePayload}; p.document.people.push(JSON.parse(JSON.stringify(p.document.people[0]))); return p; })()`);
    const deletePerson = await rejected(`(() => { const p=${basePayload}; p.document.people.pop(); return p; })()`);
    const ignore = await rejected(`(() => { const p=${basePayload}; p.document.ignore_regions.push({ignore_region_id:"ignore-x",reason:"OTHER",polygon:[{x:1,y:1},{x:2,y:1},{x:2,y:2}]}); return p; })()`);
    record("crafted_geometry_and_truth_mutations_rejected_server_side",
      [geometry, otherRelevance, addPerson, deletePerson, ignore].every(row => row.status === 422),
      {geometry, otherRelevance, addPerson, deletePerson, ignore});
    record("crafted_rejections_do_not_advance_revision", await evaluate("state.revision") === initial.revision, await evaluate("state.revision"));
    process.stderr.write("edge-acceptance: server guards passed\n");

    const earlyFinalize = await rejected(`(() => { const p=${basePayload}; p.action_type="FINALIZE"; return p; })()`);
    record("finalization_blocked_before_correction_and_confirmations", earlyFinalize.status === 422, earlyFinalize);

    let revision = await clickAndWait("targetMatch", initial.revision);
    const stillEarly = await rejected(`(() => { const p=${basePayload}; p.action_type="FINALIZE"; return p; })()`);
    record("finalization_blocked_until_all_eight_strips", stillEarly.status === 422, stillEarly);
    for (let strip = 0; strip < 8; strip += 1) {
      await evaluate(`document.querySelector('#stripButtons button[data-strip="${strip}"]').click()`);
      await waitUntil(`state.busy === false && state.revision > ${revision}`);
      revision = await evaluate("state.revision");
    }
    revision = await clickAndWait("officialConfirmed", revision);
    revision = await clickAndWait("denseAssertion", revision);
    revision = await clickAndWait("adjudicationAssertion", revision);
    const ready = await evaluate(`({
      relevance: sequence2Api.targetPerson().relevance,
      strips: state.document.reviewed_exhaustiveness_strips,
      official: state.metadata.repair_checklist_addressed,
      dense: state.document.completion_assertion,
      adjudication: state.metadata.adjudication_assertion,
      finalizeDisabled: document.getElementById("finalizeSequence2").disabled,
    })`);
    record("human_correction_and_all_confirmations_enable_finalize", ready.relevance === "MATCH_RELEVANT"
      && JSON.stringify(ready.strips) === JSON.stringify([0,1,2,3,4,5,6,7])
    && ready.official && ready.dense && ready.adjudication && !ready.finalizeDisabled, ready);
    process.stderr.write("edge-acceptance: human confirmation flow passed\n");

    const finalized = await evaluate(`(async () => {
      const actionId = "double-finalize-fixed";
      const first = await sequence2Api.postAction("FINALIZE", actionId);
      const second = await sequence2Api.postAction("FINALIZE", actionId);
      await sequence2Api.loadState();
      return {first, second, finalized: state.finalized, revision: state.revision};
    })()`);
    record("double_finalize_is_idempotent", finalized.finalized && finalized.first.event_id === finalized.second.event_id
      && finalized.first.event_sha256 === finalized.second.event_sha256, finalized);
    process.stderr.write("edge-acceptance: idempotent finalization passed\n");
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
