"use strict";

const fs = require("node:fs");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const {spawn} = require("node:child_process");

const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
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

async function waitForPage(port, expectedUrl) {
  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    try {
      const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
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
      const pending = this.pending.get(message.id);
      this.pending.delete(message.id);
      if (message.error) pending.reject(new Error(JSON.stringify(message.error)));
      else pending.resolve(message.result);
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
  if (!baseUrl) throw new Error("Usage: node g7f_c_scored_first_pass_edge_acceptance.js <reviewer-url>");
  if (!fs.existsSync(EDGE)) throw new Error(`Microsoft Edge is missing: ${EDGE}`);
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "g7f-c-scored-edge-"));
  const debugPort = await freePort();
  const edge = spawn(EDGE, [
    "--headless=new",
    "--disable-gpu",
    "--disable-extensions",
    "--no-first-run",
    "--no-default-browser-check",
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${profile}`,
    "--window-size=1440,900",
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
    async function evaluate(expression) {
      const response = await client.send("Runtime.evaluate", {
        expression,
        awaitPromise: true,
        returnByValue: true,
      });
      if (response.exceptionDetails) throw new Error(response.exceptionDetails.text || "browser evaluation failed");
      return response.result.value;
    }
    async function waitUntil(expression, timeout = 15000) {
      const deadline = Date.now() + timeout;
      while (Date.now() < deadline) {
        if (await evaluate(expression)) return;
        await sleep(50);
      }
      throw new Error(`Timed out waiting for: ${expression}`);
    }
    async function snapshot() {
      return evaluate(`({
        operation: state.operation,
        index: state.index,
        queueLength: state.queue.length,
        imageId: state.currentImageId,
        revisionImageId: state.revisionImageId,
        revision: state.serverRevision,
        mode: state.mode,
        effectiveMode: effectiveMode(),
        finalized: state.finalized,
        people: state.document.people.length,
        ignores: state.document.ignore_regions.length,
        strips: state.document.reviewed_exhaustiveness_strips.length,
        assertion: state.document.completion_assertion,
        working: state.working.length,
        dirty: state.dirty,
        previousDisabled: previous.disabled,
        nextDisabled: next.disabled,
        selectorDisabled: imageSelect.disabled,
        selectorValue: imageSelect.value,
        revealHidden: document.getElementById("reveal").hidden,
        auditHidden: document.getElementById("auditChecklistPanel").hidden,
        scope: [
          document.getElementById("scopeReminderPrimary").textContent,
          document.getElementById("scopeReminderSecondary").textContent
        ],
        invariant: clientStateInvariant()
      })`);
    }

    await waitUntil('typeof state !== "undefined" && state.document !== null && state.operation === "IDLE"');
    let current = await snapshot();
    record("exact_48_scored_queue", current.queueLength === 48 && current.imageId === "DG-007", current);
    record(
      "never_started_first_image_blank",
      current.people === 0 && current.ignores === 0 && current.strips === 0 && current.assertion === null,
      current,
    );
    record(
      "first_item_navigation_and_invariant",
      current.previousDisabled && !current.nextDisabled && !current.selectorDisabled && current.invariant.ok,
      current,
    );
    record("candidate_and_audit_surfaces_hidden", current.revealHidden && current.auditHidden, current);
    record(
      "persistent_scope_reminder",
      current.scope[0].startsWith("ALL visible people count:")
        && current.scope[1] === "Annotate the person first; classify relevance second.",
      current.scope,
    );

    await evaluate(`(() => {
      const canvas = document.getElementById("canvas");
      const rect = canvas.getBoundingClientRect();
      canvas.dispatchEvent(new PointerEvent("pointerdown", {clientX: rect.left + 500, clientY: rect.top + 350, button: 0, bubbles: true}));
      window.dispatchEvent(new PointerEvent("pointerup", {clientX: rect.left + 500, clientY: rect.top + 350, button: 0, bubbles: true}));
    })()`);
    current = await snapshot();
    record("pan_edit_never_adds_vertices", current.mode === "PAN_EDIT" && current.working === 0, current);

    await evaluate(`document.getElementById("relevance").value = "MATCH_RELEVANT"; enterDrawPerson()`);
    await evaluate(`(() => {
      const canvas = document.getElementById("canvas");
      const rect = canvas.getBoundingClientRect();
      canvas.dispatchEvent(new PointerEvent("pointerdown", {clientX: rect.left + 500, clientY: rect.top + 350, button: 0, bubbles: true}));
      window.dispatchEvent(new PointerEvent("pointerup", {clientX: rect.left + 500, clientY: rect.top + 350, button: 0, bubbles: true}));
    })()`);
    await evaluate(`document.getElementById("next").click()`);
    await sleep(100);
    current = await snapshot();
    record(
      "unfinished_polygon_blocks_navigation_and_snaps_selector",
      current.index === 0 && current.selectorValue === "0" && current.working === 1,
      current,
    );
    await evaluate(`window.dispatchEvent(new KeyboardEvent("keydown", {code: "Space", key: " ", bubbles: true}))`);
    current = await snapshot();
    record(
      "space_hold_is_temporary_pan",
      current.effectiveMode === "TEMPORARY_PAN" && current.working === 1,
      current,
    );
    await evaluate(`window.dispatchEvent(new KeyboardEvent("keyup", {code: "Space", key: " ", bubbles: true})); window.dispatchEvent(new KeyboardEvent("keydown", {key: "Escape", bubbles: true}))`);
    current = await snapshot();
    record("escape_returns_to_pan_edit", current.mode === "PAN_EDIT" && current.working === 0, current);

    await evaluate(`(() => {
      window.__nativeFetch = window.fetch;
      window.__failNextSave = true;
      window.fetch = async (...args) => {
        const body = args[1]?.body ? JSON.parse(args[1].body) : null;
        if (window.__failNextSave && body?.action_type === "SAVE_DRAFT") {
          window.__failNextSave = false;
          return new Response(JSON.stringify({error_code: "INJECTED_SAVE_FAILURE", message: "TEMP failure"}), {
            status: 503,
            headers: {"Content-Type": "application/json"}
          });
        }
        return window.__nativeFetch(...args);
      };
      document.querySelector('[data-strip="0"]').click();
      document.getElementById("next").click();
    })()`);
    await waitUntil('state.operation === "IDLE" && state.statusImageId === "DG-007"');
    current = await snapshot();
    record(
      "save_failure_restores_coherent_usable_state",
      current.index === 0 && current.selectorValue === "0" && current.strips === 1 && current.dirty && current.invariant.ok,
      current,
    );

    await evaluate(`(() => {
      window.fetch = window.__nativeFetch;
      window.__delayNextSave = true;
      window.fetch = async (...args) => {
        const body = args[1]?.body ? JSON.parse(args[1].body) : null;
        if (window.__delayNextSave && body?.action_type === "SAVE_DRAFT") {
          window.__delayNextSave = false;
          await new Promise(resolve => setTimeout(resolve, 600));
        }
        return window.__nativeFetch(...args);
      };
      document.getElementById("next").click();
    })()`);
    await waitUntil('state.operation === "SAVING_FOR_NAVIGATION"');
    current = await snapshot();
    record(
      "slow_save_immediate_busy_render",
      current.index === 0 && current.previousDisabled && current.nextDisabled && current.selectorDisabled,
      current,
    );
    await waitUntil('state.index === 1 && state.operation === "IDLE"');
    current = await snapshot();
    record("atomic_navigation_to_blank_second", current.imageId === "DG-008" && current.strips === 0, current);

    await evaluate(`(() => {
      window.__wrongTarget = true;
      window.fetch = async (...args) => {
        const response = await window.__nativeFetch(...args);
        if (window.__wrongTarget && String(args[0]).includes("/api/state?image_id=DG-009")) {
          window.__wrongTarget = false;
          const payload = await response.json();
          payload.anonymous_dense_image_id = "DG-010";
          return new Response(JSON.stringify(payload), {status: 200, headers: {"Content-Type": "application/json"}});
        }
        return response;
      };
      const select = document.getElementById("imageSelect");
      select.value = "2";
      select.dispatchEvent(new Event("change", {bubbles: true}));
    })()`);
    await waitUntil('state.operation === "IDLE" && state.statusImageId === "DG-008"');
    current = await snapshot();
    record(
      "wrong_image_response_rejected_with_selector_snapback",
      current.imageId === "DG-008" && current.selectorValue === "1" && current.invariant.ok,
      current,
    );
    await evaluate(`window.fetch = window.__nativeFetch; loadImage(0)`);
    await waitUntil('state.index === 0 && state.operation === "IDLE"');
    current = await snapshot();
    record("draft_resumes_only_on_own_image", current.strips === 1 && current.people === 0, current);

    await evaluate(`(() => {
      for (let index = 0; index < 8; index += 1) {
        if (!state.document.reviewed_exhaustiveness_strips.includes(index)) {
          document.querySelector('[data-strip="' + index + '"]').click();
        }
      }
      document.getElementById("assertion").click();
    })()`);
    current = await snapshot();
    record("all_strips_and_exact_assertion_ready", current.strips === 8 && current.assertion !== null, current);
    await evaluate(`document.getElementById("finalize").click(); document.getElementById("finalize").click()`);
    await waitUntil('state.finalized && state.operation === "IDLE"', 30000);
    current = await snapshot();
    record(
      "double_click_safe_finalize_and_no_reveal",
      current.finalized && current.revealHidden && !current.nextDisabled && !current.selectorDisabled,
      current,
    );

    await evaluate(`loadImage(state.queue.length - 1)`);
    await waitUntil('state.index === state.queue.length - 1 && state.operation === "IDLE"');
    current = await snapshot();
    record(
      "last_item_only_next_disabled",
      !current.previousDisabled && current.nextDisabled && !current.selectorDisabled && current.invariant.ok,
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
