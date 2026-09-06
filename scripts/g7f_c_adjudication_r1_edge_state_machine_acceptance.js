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
  const scenario = process.argv[3];
  const screenshotPath = process.argv[4];
  if (!baseUrl || !["lifecycle", "resume"].includes(scenario)) {
    throw new Error("Usage: edge_state_machine_acceptance.js <reviewer-url> <lifecycle|resume> [screenshot]");
  }
  if (!fs.existsSync(EDGE)) throw new Error(`Microsoft Edge is missing: ${EDGE}`);

  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "g7f-c-adj-r2-edge-"));
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
      return evaluate(`(() => {
        const ids = ["panEdit", "newPerson", "addComponent", "newIgnore", "finishPolygon",
          "cancelPolygon", "assertion", "repairChecklistAddressed", "adjudicationAssertion",
          "finalize", "undo", "redo", "deletePerson", "deleteIgnore", "editRelevance",
          "relevance", "ignoreReason"];
        const mutationDisabled = Object.fromEntries(ids.map(id => [id, document.getElementById(id).disabled]));
        return {
          operation: state.operation,
          loading: state.loading,
          index: state.index,
          imageId: state.queue[state.index]?.anonymous_dense_image_id,
          currentImageId: state.currentImageId,
          revisionImageId: state.revisionImageId,
          revision: state.serverRevision,
          finalized: state.finalized,
          dirty: state.dirty,
          people: state.document?.people.length,
          strips: state.document ? [...state.document.reviewed_exhaustiveness_strips] : [],
          selectorValue: document.getElementById("imageSelect").value,
          selectorText: document.getElementById("imageSelect").selectedOptions[0]?.textContent,
          selectorDisabled: document.getElementById("imageSelect").disabled,
          previousDisabled: document.getElementById("previous").disabled,
          nextDisabled: document.getElementById("next").disabled,
          workflow: document.getElementById("workflowBadge").textContent,
          status: document.getElementById("status").textContent,
          invariant: clientStateInvariant(),
          mutationDisabled,
          stripsDisabled: [...document.querySelectorAll("#strips button")].every(button => button.disabled),
        };
      })()`);
    }

    async function navigate(index) {
      await evaluate(`loadImage(${index})`);
      await waitUntil(`state.index === ${index} && state.operation === "IDLE"`);
    }

    try {
      await waitUntil('typeof state !== "undefined" && state.document !== null && state.operation === "IDLE"');
    } catch (error) {
      const diagnostic = await evaluate(`({
        hasState: typeof state !== "undefined",
        operation: typeof state === "undefined" ? null : state.operation,
        hasDocument: typeof state === "undefined" ? false : state.document !== null,
        status: document.getElementById("status")?.textContent,
      })`);
      throw new Error(`${error.message}; initial diagnostic: ${JSON.stringify(diagnostic)}`);
    }
    let current = await snapshot();
    record(
      `${scenario}_initial_identity_coherent`,
      current.imageId === "DG-001" && current.currentImageId === "DG-001" && current.invariant.ok,
      current,
    );

    if (scenario === "resume") {
      record(
        "resume_dg001_finalized_read_only_and_navigable",
        current.finalized && Object.values(current.mutationDisabled).every(Boolean)
          && !current.nextDisabled && !current.selectorDisabled,
        current,
      );
      await navigate(1);
      current = await snapshot();
      record(
        "resume_dg002_finalized_read_only_and_navigable",
        current.imageId === "DG-002" && current.finalized
          && Object.values(current.mutationDisabled).every(Boolean)
          && !current.previousDisabled && !current.nextDisabled,
        current,
      );
      await navigate(2);
      current = await snapshot();
      record(
        "resume_large_mutable_dg003_exact_progress",
        current.imageId === "DG-003" && !current.finalized && current.people >= 75
          && JSON.stringify(current.strips) === JSON.stringify([0, 2, 4])
          && !current.previousDisabled && !current.nextDisabled && !current.selectorDisabled,
        current,
      );
      record("resume_invariant_holds", current.invariant.ok, current.invariant);
    } else {
      await navigate(2);
      current = await snapshot();
      record("large_dg003_draft_loaded", current.imageId === "DG-003" && current.people >= 75, current);

      await evaluate(`(() => {
        window.__originalFetch = window.fetch.bind(window);
        window.__saveCalls = 0;
        window.__finalizeCalls = 0;
        window.__stateCalls = 0;
        window.__delayNextSave = true;
        window.__releaseSlowSave = null;
        window.__slowSaveGate = new Promise(resolve => { window.__releaseSlowSave = resolve; });
        window.__faultNextSave = null;
        window.__wrongIdentityNextSave = false;
        window.__faultNextFinalize = null;
        window.fetch = async (url, options = {}) => {
          let action = null;
          if (String(url).includes("/api/action") && options.body) action = JSON.parse(options.body);
          if (String(url).includes("/api/state")) window.__stateCalls += 1;
          if (action?.action_type === "SAVE_DRAFT") {
            window.__saveCalls += 1;
            if (window.__delayNextSave) {
              window.__delayNextSave = false;
              await window.__slowSaveGate;
            }
            if (window.__faultNextSave) {
              const fault = window.__faultNextSave;
              window.__faultNextSave = null;
              return new Response(JSON.stringify(fault.body), {
                status: fault.status,
                headers: {"Content-Type": "application/json"},
              });
            }
            if (window.__wrongIdentityNextSave) {
              window.__wrongIdentityNextSave = false;
              return new Response(JSON.stringify({
                anonymous_dense_image_id: "DG-001",
                revision: action.expected_revision + 1,
              }), {status: 200, headers: {"Content-Type": "application/json"}});
            }
          }
          if (action?.action_type === "FINALIZE") {
            window.__finalizeCalls += 1;
            if (window.__faultNextFinalize) {
              const fault = window.__faultNextFinalize;
              window.__faultNextFinalize = null;
              return new Response(JSON.stringify(fault.body), {
                status: fault.status,
                headers: {"Content-Type": "application/json"},
              });
            }
          }
          return window.__originalFetch(url, options);
        };
        state.document.people[state.document.people.length - 1].relevance = "RELEVANCE_UNCERTAIN";
        state.mutationVersion += 1;
        state.dirty = true;
        clearTimeout(state.autosaveTimer);
        window.__navPromise = loadImage(3);
      })()`);
      await waitUntil('state.operation === "SAVING_FOR_NAVIGATION" && window.__saveCalls === 1');

      await evaluate(`(() => {
        const select = document.getElementById("imageSelect");
        select.value = "4";
        select.dispatchEvent(new Event("change", {bubbles: true}));
        loadImage(1);
        loadImage(5);
      })()`);
      current = await snapshot();
      record(
        "slow_save_busy_is_immediate_and_complete",
        current.operation === "SAVING_FOR_NAVIGATION" && current.loading
          && current.selectorDisabled && current.previousDisabled && current.nextDisabled
          && Object.values(current.mutationDisabled).every(Boolean) && current.stripsDisabled,
        current,
      );
      record(
        "rejected_navigation_snaps_selector_to_authoritative_dg003",
        current.imageId === "DG-003" && current.currentImageId === "DG-003"
          && current.selectorValue === "2" && current.selectorText.startsWith("DG-003")
          && current.status === "Please wait for the current save/load to finish.",
        current,
      );
      await evaluate("window.__releaseSlowSave()");
      await evaluate("window.__navPromise");
      await waitUntil('state.operation === "IDLE" && state.currentImageId === "DG-004"');
      current = await snapshot();
      const slowCounts = await evaluate("({saveCalls: window.__saveCalls, stateCalls: window.__stateCalls})");
      record(
        "slow_save_commits_exactly_one_intended_target",
        current.imageId === "DG-004" && current.index === 3 && current.invariant.ok
          && slowCounts.saveCalls === 1 && slowCounts.stateCalls === 1,
        {current, slowCounts},
      );

      await evaluate(`(() => {
        state.document.completion_assertion = "temporary dirty marker";
        state.mutationVersion += 1;
        state.dirty = true;
        clearTimeout(state.autosaveTimer);
        window.__beforeFailureDocument = JSON.stringify(state.document);
        window.__beforeFailureRevision = state.serverRevision;
        window.__faultNextSave = {status: 409, body: {error_code: "STALE_REVISION", message: "simulated 409"}};
      })()`);
      const result409 = await evaluate("loadImage(4)");
      current = await snapshot();
      const unchanged409 = await evaluate(
        "JSON.stringify(state.document) === window.__beforeFailureDocument && state.serverRevision === window.__beforeFailureRevision",
      );
      record(
        "save_409_restores_coherent_usable_source",
        result409 === false && unchanged409 && current.imageId === "DG-004" && current.operation === "IDLE"
          && current.selectorValue === "3" && !current.selectorDisabled && !current.previousDisabled
          && !current.nextDisabled && current.dirty && current.invariant.ok,
        current,
      );

      await evaluate(`window.__faultNextSave = {
        status: 500,
        body: {error_code: "SERVER_ERROR", message: "simulated 500"}
      }`);
      const result500 = await evaluate("loadImage(4)");
      current = await snapshot();
      const unchanged500 = await evaluate(
        "JSON.stringify(state.document) === window.__beforeFailureDocument && state.serverRevision === window.__beforeFailureRevision",
      );
      record(
        "save_500_restores_coherent_usable_source",
        result500 === false && unchanged500 && current.imageId === "DG-004" && current.operation === "IDLE"
          && current.selectorValue === "3" && !current.selectorDisabled && current.dirty && current.invariant.ok,
        current,
      );

      await evaluate(`(() => {
        window.__dg005Url = state.queue[4].image_url;
        state.queue[4].image_url = "/assets/DOES-NOT-EXIST.png";
      })()`);
      const imageFailure = await evaluate("loadImage(4)");
      current = await snapshot();
      record(
        "target_asset_failure_keeps_saved_source_authoritative",
        imageFailure === false && current.imageId === "DG-004" && current.currentImageId === "DG-004"
          && current.revisionImageId === "DG-004" && current.operation === "IDLE"
          && current.selectorValue === "3" && !current.dirty && current.invariant.ok,
        current,
      );
      await evaluate("state.queue[4].image_url = window.__dg005Url");
      await navigate(4);

      await evaluate(`(() => {
        state.document.reviewed_exhaustiveness_strips = [0,1,2,3,4,5,6,7];
        state.document.completion_assertion = ASSERTION;
        state.adjudicationMetadata.repair_checklist_addressed = true;
        state.adjudicationMetadata.adjudication_assertion = ADJUDICATION_ASSERTION;
        state.mutationVersion += 1;
        state.dirty = true;
        clearTimeout(state.autosaveTimer);
        window.__doubleFinalize = Promise.all([finalize(), finalize()]);
      })()`);
      await evaluate("window.__doubleFinalize");
      await waitUntil('state.operation === "IDLE" && state.finalized');
      current = await snapshot();
      const finalizeCalls = await evaluate("window.__finalizeCalls");
      record(
        "finalize_is_double_click_safe_and_navigation_remains_enabled",
        finalizeCalls === 1 && current.imageId === "DG-005" && current.finalized
          && !current.previousDisabled && !current.nextDisabled && !current.selectorDisabled && current.invariant.ok,
        {current, finalizeCalls},
      );

      await navigate(5);
      await evaluate(`(() => {
        state.document.reviewed_exhaustiveness_strips = [0,1,2,3,4,5,6,7];
        state.document.completion_assertion = ASSERTION;
        state.adjudicationMetadata.repair_checklist_addressed = true;
        state.adjudicationMetadata.adjudication_assertion = ADJUDICATION_ASSERTION;
        state.mutationVersion += 1;
        state.dirty = true;
        clearTimeout(state.autosaveTimer);
        window.__faultNextFinalize = {
          status: 500,
          body: {error_code: "SERVER_ERROR", message: "simulated finalize failure"}
        };
      })()`);
      const finalizeFailure = await evaluate("finalize()");
      current = await snapshot();
      record(
        "finalize_failure_recovers_saved_mutable_state",
        finalizeFailure === false && current.imageId === "DG-006" && !current.finalized
          && current.operation === "IDLE" && current.selectorValue === "5"
          && !current.previousDisabled && current.nextDisabled && !current.selectorDisabled
          && !current.dirty && current.invariant.ok,
        current,
      );

      await evaluate(`(() => {
        state.document.completion_assertion = null;
        state.mutationVersion += 1;
        state.dirty = true;
        clearTimeout(state.autosaveTimer);
        window.__beforeWrongIdentityRevision = state.serverRevision;
        window.__wrongIdentityNextSave = true;
      })()`);
      const wrongIdentity = await evaluate("loadImage(4)");
      current = await snapshot();
      record(
        "wrong_image_save_response_is_rejected_fail_closed",
        wrongIdentity === false && current.imageId === "DG-006" && current.revisionImageId === "DG-006"
          && current.revision === await evaluate("window.__beforeWrongIdentityRevision")
          && current.operation === "IDLE" && current.selectorValue === "5"
          && Object.values(current.mutationDisabled).every(Boolean)
          && current.selectorDisabled && current.previousDisabled && current.nextDisabled,
        current,
      );
    }

    if (screenshotPath) {
      const screenshot = await client.send("Page.captureScreenshot", {format: "png", captureBeyondViewport: false});
      fs.writeFileSync(screenshotPath, Buffer.from(screenshot.data, "base64"));
    }
    console.log(JSON.stringify({browser: "Microsoft Edge", scenario, passed: true, checks, screenshotPath}, null, 2));
  } finally {
    client.close();
    await stopEdge(edge, profile);
  }
}

main().catch(error => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
