"use strict";

const fs = require("node:fs");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const {spawn} = require("node:child_process");

const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";

function sleep(milliseconds) {
  return new Promise(resolve => setTimeout(resolve, milliseconds));
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
  const screenshotPath = process.argv[3] || null;
  if (!baseUrl) throw new Error("Usage: node g7f_c_r1_edge_acceptance.js <reviewer-url>");
  if (!fs.existsSync(EDGE)) throw new Error(`Microsoft Edge is missing: ${EDGE}`);

  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "g7f-c-r1-edge-"));
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
      "--window-size=1440,900",
      baseUrl,
    ],
    {stdio: "ignore", windowsHide: true},
  );
  const client = new DevToolsClient((await waitForPage(debugPort, baseUrl)).webSocketDebuggerUrl);
  const checks = [];

  const record = (name, passed, detail = null) => {
    checks.push({name, passed: Boolean(passed), detail});
    if (!passed) throw new Error(`${name} failed: ${JSON.stringify(detail)}`);
  };

  try {
    await client.connect();
    await client.send("Runtime.enable");
    await client.send("Page.enable");

    async function evaluate(expression) {
      const response = await client.send("Runtime.evaluate", {
        expression,
        awaitPromise: true,
        returnByValue: true,
      });
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
        mode: state.mode,
        effectiveMode: effectiveMode(),
        working: state.working.length,
        people: state.document.people.length,
        ignores: state.document.ignore_regions.length,
        components: state.document.people.map(person => personComponents(person).length),
        panX: state.panX,
        panY: state.panY,
        index: state.index,
        finalized: state.finalized,
        revision: state.serverRevision,
        dirty: state.dirty,
        status: document.getElementById("status").textContent,
        modeBadge: document.getElementById("modeBadge").textContent,
        addComponentDisabled: document.getElementById("addComponent").disabled,
        revealHidden: document.getElementById("reveal").hidden
      })`);
    }

    async function pointer(type, x, y, button = 0) {
      return evaluate(`(() => {
        const canvas = document.getElementById("canvas");
        const rect = canvas.getBoundingClientRect();
        const target = ${JSON.stringify(type)} === "pointerdown" ? canvas : window;
        target.dispatchEvent(new PointerEvent(${JSON.stringify(type)}, {
          clientX: rect.left + ${x}, clientY: rect.top + ${y}, button: ${button}, bubbles: true
        }));
        return true;
      })()`);
    }

    async function clickCanvas(x, y) {
      await pointer("pointerdown", x, y);
      await pointer("pointerup", x, y);
    }

    async function dragCanvas(x1, y1, x2, y2) {
      await pointer("pointerdown", x1, y1);
      await pointer("pointermove", x2, y2);
      await pointer("pointerup", x2, y2);
    }

    async function key(type, keyValue, code = keyValue) {
      return evaluate(`window.dispatchEvent(new KeyboardEvent(${JSON.stringify(type)}, {
        key: ${JSON.stringify(keyValue)}, code: ${JSON.stringify(code)}, bubbles: true
      }))`);
    }

    async function drawPoints(points) {
      for (const [x, y] of points) await clickCanvas(x, y);
    }

    await waitUntil('typeof state !== "undefined" && state.document !== null && state.image.complete');
    let current = await snapshot();
    record("image_load_defaults_to_pan_edit", current.mode === "PAN_EDIT" && current.working === 0, current);
    record("mode_badge_is_visible", current.modeBadge === "MODE: PAN / EDIT", current.modeBadge);
    const visibleText = await evaluate("document.body.innerText.toLowerCase()");
    record(
      "prefinal_ui_is_candidate_blind",
      current.revealHidden && !["confidence", "disagreement", "selection reason", "run id", "historical missed", "subject marker"]
        .some(token => visibleText.includes(token)),
      {revealHidden: current.revealHidden},
    );
    if (screenshotPath) {
      const screenshot = await client.send("Page.captureScreenshot", {format: "png", captureBeyondViewport: false});
      fs.mkdirSync(path.dirname(screenshotPath), {recursive: true});
      fs.writeFileSync(screenshotPath, Buffer.from(screenshot.data, "base64"));
    }

    await clickCanvas(720, 520);
    current = await snapshot();
    record("pan_edit_click_adds_zero_points", current.working === 0 && current.mode === "PAN_EDIT", current);

    const panBefore = current;
    await dragCanvas(720, 520, 765, 548);
    current = await snapshot();
    record(
      "pan_edit_drag_changes_view_only",
      current.working === 0 && (current.panX !== panBefore.panX || current.panY !== panBefore.panY),
      {before: panBefore, after: current},
    );

    await evaluate(`document.getElementById("relevance").value = "MATCH_RELEVANT"; document.getElementById("newPerson").click()`);
    current = await snapshot();
    record("draw_person_is_explicit_and_creates_no_empty_record", current.mode === "DRAW_PERSON" && current.people === 0, current);
    await drawPoints([[340, 300], [410, 300], [410, 390], [340, 390]]);
    current = await snapshot();
    record("draw_person_clicks_add_working_points", current.working === 4 && current.people === 0, current);
    await key("keydown", "Enter", "Enter");
    current = await snapshot();
    record(
      "finish_commits_and_returns_to_pan_edit",
      current.mode === "PAN_EDIT" && current.working === 0 && current.people === 1 && current.components[0] === 1,
      current,
    );

    await clickCanvas(620, 450);
    current = await snapshot();
    record("click_after_finish_adds_zero_points", current.working === 0 && current.mode === "PAN_EDIT", current);
    const afterFinishPan = current;
    await dragCanvas(650, 460, 690, 485);
    current = await snapshot();
    record(
      "drag_after_finish_pans_only",
      current.working === 0 && (current.panX !== afterFinishPan.panX || current.panY !== afterFinishPan.panY),
      {before: afterFinishPan, after: current},
    );
    await evaluate('document.querySelector("[data-view=resetView]").click()');
    current = await snapshot();
    record("reset_view_keeps_pan_edit", current.mode === "PAN_EDIT" && current.working === 0, current);

    await evaluate('document.getElementById("newPerson").click()');
    await drawPoints([[500, 310], [565, 310]]);
    const beforeSpace = await snapshot();
    await key("keydown", " ", "Space");
    current = await snapshot();
    record("space_enters_temporary_pan", current.mode === "DRAW_PERSON" && current.effectiveMode === "TEMPORARY_PAN", current);
    await dragCanvas(600, 460, 650, 490);
    current = await snapshot();
    record(
      "space_pan_preserves_working_polygon",
      current.working === beforeSpace.working && (current.panX !== beforeSpace.panX || current.panY !== beforeSpace.panY),
      {before: beforeSpace, after: current},
    );
    await key("keyup", " ", "Space");
    current = await snapshot();
    record("space_release_resumes_prior_draw_mode", current.mode === "DRAW_PERSON" && current.effectiveMode === "DRAW_PERSON", current);
    await drawPoints([[565, 390], [500, 390]]);
    await key("keydown", "Enter", "Enter");
    current = await snapshot();
    record("space_pan_polygon_finishes_normally", current.mode === "PAN_EDIT" && current.people === 2, current);

    await evaluate('document.getElementById("newPerson").click()');
    await drawPoints([[730, 300], [780, 300], [780, 360]]);
    await key("keydown", "Escape", "Escape");
    current = await snapshot();
    record("escape_cancels_without_committing", current.mode === "PAN_EDIT" && current.working === 0 && current.people === 2, current);

    await evaluate('document.getElementById("newPerson").click()');
    await clickCanvas(820, 330);
    await evaluate('document.getElementById("next").click()');
    await sleep(100);
    current = await snapshot();
    record(
      "next_is_blocked_by_working_polygon",
      current.index === 0 && current.working === 1 && current.status === "Finish or cancel the current polygon first.",
      current,
    );
    await evaluate(`(() => {
      const select = document.getElementById("imageSelect");
      select.value = "1";
      select.dispatchEvent(new Event("change", {bubbles: true}));
    })()`);
    await sleep(100);
    current = await snapshot();
    record("image_dropdown_is_blocked_by_working_polygon", current.index === 0 && current.working === 1, current);
    await evaluate('document.getElementById("finalize").click()');
    current = await snapshot();
    record("finalize_is_blocked_by_working_polygon", !current.finalized && current.working === 1, current);
    await key("keydown", "Escape", "Escape");

    await evaluate('document.getElementById("next").click()');
    await waitUntil("state.index === 1 && !state.loading");
    current = await snapshot();
    record("navigation_load_returns_to_pan_edit", current.mode === "PAN_EDIT" && current.working === 0, current);
    record("add_visible_part_requires_selection", current.addComponentDisabled && current.mode === "PAN_EDIT", current);

    await evaluate(`document.getElementById("ignoreReason").value = "SEVERE_VISUAL_ARTIFACT"; document.getElementById("newIgnore").click()`);
    current = await snapshot();
    record("ignore_mode_creates_no_empty_record", current.mode === "DRAW_IGNORE_REGION" && current.ignores === 0, current);
    await drawPoints([[320, 270], [390, 270], [390, 340]]);
    await key("keydown", "Escape", "Escape");
    current = await snapshot();
    record("escape_leaves_no_empty_ignore", current.mode === "PAN_EDIT" && current.ignores === 0, current);

    await evaluate('document.getElementById("newIgnore").click()');
    await drawPoints([[320, 270], [390, 270], [390, 340], [320, 340]]);
    await key("keydown", "Enter", "Enter");
    current = await snapshot();
    record("ignore_finish_commits_and_returns_to_pan_edit", current.mode === "PAN_EDIT" && current.ignores === 1, current);
    const ignoreRevision = current.revision;
    await evaluate('window.confirm = () => true; document.getElementById("deleteIgnore").click()');
    await waitUntil("!state.dirty");
    current = await snapshot();
    record(
      "explicit_ignore_delete_is_revisioned",
      current.ignores === 0 && current.mode === "PAN_EDIT" && current.revision > ignoreRevision,
      {beforeRevision: ignoreRevision, after: current},
    );

    await evaluate('document.getElementById("previous").click()');
    await waitUntil("state.index === 0 && !state.loading");
    current = await snapshot();
    const personRevision = current.revision;
    await evaluate('window.confirm = () => true; document.querySelector("#instances li[data-kind=person]").click(); document.getElementById("deletePerson").click()');
    await waitUntil("!state.dirty");
    current = await snapshot();
    record(
      "explicit_person_delete_is_revisioned",
      current.people === 1 && current.mode === "PAN_EDIT" && current.revision > personRevision,
      {beforeRevision: personRevision, after: current},
    );

    await evaluate('document.querySelector("#instances li[data-kind=person]").click(); document.getElementById("addComponent").click()');
    await drawPoints([[600, 300], [640, 300], [640, 350], [600, 350]]);
    await key("keydown", "Enter", "Enter");
    current = await snapshot();
    record(
      "selected_person_accepts_visible_component",
      current.mode === "PAN_EDIT" && current.people === 1 && current.components[0] === 2,
      current,
    );
    record("no_empty_person_or_ignore_records", current.components.every(count => count > 0) && current.ignores === 0, current);

    console.log(JSON.stringify({browser: "Microsoft Edge", passed: true, checks}, null, 2));
  } finally {
    client.close();
    edge.kill();
    await sleep(200);
    fs.rmSync(profile, {recursive: true, force: true});
  }
}

main().catch(error => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
