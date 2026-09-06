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
  if (!baseUrl) throw new Error("Usage: node g7f_c_r2_edge_navigation_acceptance.js <reviewer-url>");
  if (!fs.existsSync(EDGE)) throw new Error(`Microsoft Edge is missing: ${EDGE}`);
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "g7f-c-r2-edge-"));
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
        loading: state.loading,
        index: state.index,
        queueLength: state.queue.length,
        imageId: state.queue[state.index].anonymous_dense_image_id,
        mode: state.mode,
        finalized: state.finalized,
        working: state.working.length,
        previousDisabled: document.getElementById("previous").disabled,
        nextDisabled: document.getElementById("next").disabled,
        selectorDisabled: document.getElementById("imageSelect").disabled,
        finalizeDisabled: document.getElementById("finalize").disabled,
        revealHidden: document.getElementById("reveal").hidden,
        status: document.getElementById("status").textContent
      })`);
    }

    await waitUntil('typeof state !== "undefined" && state.document !== null && !state.loading');
    let current = await snapshot();
    record("dg001_loading_false", !current.loading, current);
    record(
      "dg001_only_previous_disabled",
      current.index === 0 && current.previousDisabled && !current.nextDisabled && !current.selectorDisabled,
      current,
    );
    record(
      "finalized_dg001_allows_navigation",
      current.finalized && current.finalizeDisabled && !current.nextDisabled && !current.selectorDisabled && !current.revealHidden,
      current,
    );

    await evaluate('document.getElementById("next").click()');
    await waitUntil("state.index === 1 && !state.loading");
    current = await snapshot();
    record(
      "dg002_middle_navigation_enabled",
      current.imageId === "DG-002" && !current.previousDisabled && !current.nextDisabled && !current.selectorDisabled,
      current,
    );
    record("image_load_preserves_pan_edit_default", current.mode === "PAN_EDIT", current);

    await evaluate(`(() => {
      const select = document.getElementById("imageSelect");
      select.value = String(state.queue.length - 1);
      select.dispatchEvent(new Event("change", {bubbles: true}));
    })()`);
    await waitUntil("state.index === state.queue.length - 1 && !state.loading");
    current = await snapshot();
    record(
      "last_item_only_next_disabled",
      !current.previousDisabled && current.nextDisabled && !current.selectorDisabled,
      current,
    );

    await evaluate(`(() => {
      const select = document.getElementById("imageSelect");
      select.value = "1";
      select.dispatchEvent(new Event("change", {bubbles: true}));
    })()`);
    await waitUntil("state.index === 1 && !state.loading");
    await evaluate(`document.getElementById("relevance").value = "MATCH_RELEVANT"; document.getElementById("newPerson").click()`);
    await evaluate(`(() => {
      const canvas = document.getElementById("canvas");
      const rect = canvas.getBoundingClientRect();
      canvas.dispatchEvent(new PointerEvent("pointerdown", {clientX: rect.left + 500, clientY: rect.top + 350, button: 0, bubbles: true}));
      window.dispatchEvent(new PointerEvent("pointerup", {clientX: rect.left + 500, clientY: rect.top + 350, button: 0, bubbles: true}));
    })()`);
    current = await snapshot();
    record("unfinished_polygon_is_active", current.mode === "DRAW_PERSON" && current.working === 1, current);
    await evaluate('document.getElementById("next").click()');
    await sleep(100);
    current = await snapshot();
    record(
      "unfinished_polygon_blocks_next",
      current.index === 1 && current.working === 1 && current.status === "Finish or cancel the current polygon first.",
      current,
    );
    await evaluate(`(() => {
      const select = document.getElementById("imageSelect");
      select.value = "2";
      select.dispatchEvent(new Event("change", {bubbles: true}));
    })()`);
    await sleep(100);
    current = await snapshot();
    record("unfinished_polygon_blocks_selector", current.index === 1 && current.working === 1, current);
    await evaluate('document.getElementById("cancelPolygon").click()');
    current = await snapshot();
    record(
      "cancel_reenables_navigation",
      current.mode === "PAN_EDIT" && current.working === 0 && !current.previousDisabled && !current.nextDisabled && !current.selectorDisabled,
      current,
    );

    console.log(JSON.stringify({browser: "Microsoft Edge", passed: true, checks}, null, 2));
  } finally {
    client.close();
    await stopEdge(edge, profile);
  }
}

main().catch(error => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
