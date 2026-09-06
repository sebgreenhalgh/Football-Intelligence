"use strict";

const fs = require("node:fs");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const {spawn} = require("node:child_process");

const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

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
  throw new Error("Timed out waiting for the Edge DevTools target");
}

class DevToolsClient {
  constructor(url) {
    this.url = url;
    this.nextId = 0;
    this.pending = new Map();
    this.socket = null;
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

async function main() {
  const baseUrl = process.argv[2];
  const screenshotPath = process.argv[3];
  if (!baseUrl || !screenshotPath) throw new Error("Usage: edge_acceptance.js <reviewer-url> <screenshot-path>");
  if (!fs.existsSync(EDGE)) throw new Error(`Microsoft Edge is missing: ${EDGE}`);
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "g7f-c-adjudication-edge-"));
  const debugPort = await freePort();
  const edge = spawn(
    EDGE,
    [
      "--headless=new",
      "--disable-gpu",
      "--disable-extensions",
      "--no-first-run",
      "--no-default-browser-check",
      `--remote-debugging-port=${debugPort}`,
      `--user-data-dir=${profile}`,
      "--window-size=1600,1000",
      baseUrl,
    ],
    {stdio: "ignore", windowsHide: true},
  );
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
    async function waitUntil(expression, timeout = 10000) {
      const deadline = Date.now() + timeout;
      while (Date.now() < deadline) {
        if (await evaluate(expression)) return;
        await sleep(50);
      }
      throw new Error(`Timed out waiting for: ${expression}`);
    }
    async function snapshot() {
      return evaluate(`({
        imageId: state.queue[state.index].anonymous_dense_image_id,
        queue: state.queue.map(row => row.anonymous_dense_image_id),
        passKind: state.passKind,
        loading: state.loading,
        finalized: state.finalized,
        mode: state.mode,
        people: state.document.people.length,
        strips: [...state.document.reviewed_exhaustiveness_strips],
        checklistItems: document.querySelectorAll("#auditChecklist li").length,
        scopePrimary: document.getElementById("scopeReminderPrimary").textContent,
        scopeSecondary: document.getElementById("scopeReminderSecondary").textContent,
        scopeVisible: !document.getElementById("scopeReminder").hidden,
        auditVisible: !document.getElementById("auditChecklistPanel").hidden,
        revealHidden: document.getElementById("reveal").hidden,
        previousDisabled: document.getElementById("previous").disabled,
        nextDisabled: document.getElementById("next").disabled,
        selectorDisabled: document.getElementById("imageSelect").disabled,
        finalEnabled: !document.getElementById("finalize").disabled,
        status: document.getElementById("status").textContent
      })`);
    }

    await waitUntil('typeof state !== "undefined" && state.document !== null && !state.loading');
    let current = await snapshot();
    record("queue_is_exactly_six_calibration_images", JSON.stringify(current.queue) === JSON.stringify(["DG-001", "DG-002", "DG-003", "DG-004", "DG-005", "DG-006"]), current);
    record("adjudication_workflow_locked", current.passKind === "CALIBRATION_ADJUDICATION", current);
    record("seeded_dg001_truth_and_zero_strips", current.imageId === "DG-001" && current.people === 24 && current.strips.length === 0, current);
    record("pan_edit_is_default", current.mode === "PAN_EDIT", current);
    record("scope_reminder_is_persistent", current.scopeVisible && current.scopePrimary.startsWith("ALL visible people count:") && current.scopeSecondary === "Annotate the person first; classify relevance second.", current);
    record("exact_dg001_audit_checklist_visible", current.auditVisible && current.checklistItems === 2, current);
    record("no_post_finalization_reveal", current.revealHidden, current);
    const forbidden = await evaluate(`(() => {
      const value = JSON.stringify(state.bootstrap).toLowerCase();
      return ["confidence", "disagreement", "run_id", "candidate_counts"].filter(token => value.includes(token));
    })()`);
    record("no_candidate_information_exposed", forbidden.length === 0, forbidden);

    const existingBefore = await evaluate("JSON.stringify(state.document.people[0])");
    await evaluate(`(() => {
      document.getElementById("relevance").value = "NON_MATCH_RELEVANT";
      document.getElementById("newPerson").click();
      const canvas = document.getElementById("canvas");
      const rect = canvas.getBoundingClientRect();
      for (const [x, y] of [[330,340],[370,340],[370,410],[330,410]]) {
        canvas.dispatchEvent(new PointerEvent("pointerdown", {clientX: rect.left + x, clientY: rect.top + y, button: 0, bubbles: true}));
      }
      document.getElementById("finishPolygon").click();
    })()`);
    current = await snapshot();
    record("new_person_can_be_added", current.people === 25, current);
    record("finish_returns_to_pan_edit", current.mode === "PAN_EDIT", current);
    record("accepted_existing_mask_is_untouched", await evaluate("JSON.stringify(state.document.people[0])") === existingBefore, {});

    await evaluate(`(() => {
      for (let index = 0; index < 8; index += 1) {
        document.querySelector('#strips button[data-strip="' + index + '"]').click();
      }
      document.getElementById("repairChecklistAddressed").click();
      document.getElementById("adjudicationAssertion").click();
    })()`);
    await evaluate("saveDraft()");
    current = await snapshot();
    record("eight_strips_and_assertions_enable_finalization", current.strips.length === 8 && current.finalEnabled, current);
    await evaluate("finalize()");
    await waitUntil("state.finalized && !state.loading");
    current = await snapshot();
    record("temp_adjudication_finalized_immutably", current.finalized && current.revealHidden, current);
    record("finalized_frame_still_navigates", !current.nextDisabled && !current.selectorDisabled, current);

    await evaluate('document.getElementById("next").click()');
    await waitUntil("state.index === 1 && !state.loading");
    await evaluate(`document.getElementById("relevance").value = "MATCH_RELEVANT"; document.getElementById("newPerson").click()`);
    await evaluate(`(() => {
      const canvas = document.getElementById("canvas");
      const rect = canvas.getBoundingClientRect();
      canvas.dispatchEvent(new PointerEvent("pointerdown", {clientX: rect.left + 500, clientY: rect.top + 400, button: 0, bubbles: true}));
    })()`);
    await evaluate('document.getElementById("next").click()');
    await sleep(100);
    current = await snapshot();
    record("unfinished_polygon_blocks_navigation", current.imageId === "DG-002" && current.status === "Finish or cancel the current polygon first.", current);
    await evaluate('document.getElementById("cancelPolygon").click(); document.getElementById("next").click()');
    await waitUntil("state.index === 2 && !state.loading");
    current = await snapshot();
    record("middle_navigation_is_enabled", !current.previousDisabled && !current.nextDisabled && !current.selectorDisabled, current);

    await evaluate(`(() => {
      const select = document.getElementById("imageSelect");
      select.value = "4";
      select.dispatchEvent(new Event("change", {bubbles: true}));
    })()`);
    await waitUntil("state.index === 4 && !state.loading");
    const relevance = await evaluate(`(() => {
      const index = state.document.people.findIndex(person => person.instance_id === "person-030");
      const before = state.document.people[index].relevance;
      state.selected = {kind: "person", i: index, c: 0};
      renderAll();
      const select = document.getElementById("editRelevance");
      select.value = "MATCH_RELEVANT";
      select.dispatchEvent(new Event("change", {bubbles: true}));
      return {before, after: state.document.people[index].relevance};
    })()`);
    record("dg005_person030_is_not_auto_changed", relevance.before === "NON_MATCH_RELEVANT", relevance);
    record("existing_relevance_can_be_changed_explicitly", relevance.after === "MATCH_RELEVANT", relevance);

    await evaluate(`(() => {
      const select = document.getElementById("imageSelect");
      select.value = "5";
      select.dispatchEvent(new Event("change", {bubbles: true}));
    })()`);
    await waitUntil("state.index === 5 && !state.loading");
    current = await snapshot();
    record("last_item_only_next_disabled", !current.previousDisabled && current.nextDisabled && !current.selectorDisabled, current);

    const screenshot = await client.send("Page.captureScreenshot", {format: "png", captureBeyondViewport: false});
    fs.writeFileSync(screenshotPath, Buffer.from(screenshot.data, "base64"));
    console.log(JSON.stringify({browser: "Microsoft Edge", passed: true, checks, screenshotPath}, null, 2));
  } finally {
    client.close();
    await stopEdge(edge, profile);
  }
}

main().catch(error => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
