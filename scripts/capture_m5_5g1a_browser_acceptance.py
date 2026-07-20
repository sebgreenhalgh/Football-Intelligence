"""Run real-browser and crash-safe acceptance for the M5.5G.1A pilot."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
import websocket
from PIL import Image, ImageStat

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 3"
    / "M5_5G1A_DETECTION_GOLD_FOUNDATION_ONTOLOGY_FREEZE_AND_PILOT_ANNOTATION_v1"
)
PACKAGE = STAGE / "10_DETECTION_GOLD_PILOT_ANNOTATION_PACKAGE"
PRODUCTION_DECISIONS = PACKAGE / "decisions"
OUT = STAGE / "11_BROWSER_PERSISTENCE_AND_VISUAL_REGRESSION"
TIMING = STAGE / "09_ANNOTATION_TIMING_AND_INTERACTION_PLAN"
RUN_ID = uuid.uuid4().hex[:10]
TMP = STAGE / "_tmp" / f"browser_acceptance_{RUN_ID}"
DECISIONS = TMP / "decisions"
PROFILE = Path(tempfile.gettempdir()) / f"m5g1a_edge_{RUN_ID}"
URL = "http://127.0.0.1:8807/"
CDP_PORT = 9300 + (int(RUN_ID[:4], 16) % 300)
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
UV = shutil.which("uv")
SESSION = "m5_5g1a_detection_gold_pilot_reviewer"
REVIEW_ID = "m5_5g1a_detection_gold_pilot_v1"
ACTIVE_PROCESSES: list[subprocess.Popen[bytes]] = []


class CDP:
    def __init__(self, socket_: websocket.WebSocket):
        self.socket = socket_
        self.counter = 0

    def close(self) -> None:
        self.socket.close()

    def command(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.counter += 1
        self.socket.send(json.dumps({"id": self.counter, "method": method, "params": params or {}}))
        while True:
            payload = json.loads(self.socket.recv())
            if payload.get("id") != self.counter:
                continue
            if "error" in payload:
                raise RuntimeError(payload["error"])
            return payload.get("result", {})

    def evaluate(self, expression: str) -> Any:
        result = self.command(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        if result.get("exceptionDetails"):
            raise RuntimeError(result["exceptionDetails"])
        remote = result.get("result", {})
        if remote.get("subtype") == "error":
            raise RuntimeError(remote.get("description") or remote)
        return remote.get("value")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"file_count": len(rows), "tree_hash": hashlib.sha256(encoded).hexdigest(), "files": rows}


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_server() -> subprocess.Popen[bytes]:
    if port_open(8807):
        raise RuntimeError("port 8807 is occupied; exact-package browser validation cannot move ports")
    if UV is None:
        raise RuntimeError("uv is not available on PATH")
    process = subprocess.Popen(
        [
            UV,
            "run",
            "fi-pipeline",
            "review-chassis",
            "serve",
            "--manifest",
            str(PACKAGE / "reviewer_manifest.json"),
            "--ui-config",
            str(PACKAGE / "ui_config.json"),
            "--evidence-root",
            str(PACKAGE / "evidence"),
            "--decisions-root",
            str(DECISIONS),
            "--host",
            "127.0.0.1",
            "--port",
            "8807",
            "--reviewer-session-id",
            SESSION,
        ],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ACTIVE_PROCESSES.append(process)
    return process


def wait_server(process: subprocess.Popen[bytes]) -> None:
    for _ in range(160):
        if process.poll() is not None:
            raise RuntimeError(f"review server exited with {process.returncode}")
        try:
            response = requests.get(URL + "api/review/state", timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.1)
    raise RuntimeError("review server did not become ready")


def start_edge(cdp_port: int = CDP_PORT) -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--disable-background-mode",
            "--hide-scrollbars",
            "--no-first-run",
            "--remote-allow-origins=*",
            f"--remote-debugging-port={cdp_port}",
            "--window-size=1440,900",
            f"--user-data-dir={PROFILE}",
            URL,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ACTIVE_PROCESSES.append(process)
    return process


def stop_tree(process: subprocess.Popen[bytes] | None) -> None:
    if process and process.poll() is None:
        subprocess.run(
            ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def connect_page(cdp_port: int = CDP_PORT) -> CDP:
    cdp_url = f"http://127.0.0.1:{cdp_port}"
    for _ in range(200):
        try:
            pages = requests.get(f"{cdp_url}/json", timeout=0.25).json()
            page = next(
                (item for item in pages if item.get("type") == "page" and str(item.get("url", "")).startswith(URL)),
                None,
            )
            if page:
                socket_ = websocket.create_connection(str(page["webSocketDebuggerUrl"]), timeout=20)
                cdp = CDP(socket_)
                cdp.command("Page.enable")
                cdp.command("Runtime.enable")
                return cdp
        except (requests.RequestException, ValueError, StopIteration, OSError):
            pass
        time.sleep(0.1)
    raise RuntimeError("Edge CDP page did not become available")


def wait_ready(cdp: CDP) -> dict[str, Any]:
    result: dict[str, Any] | None = None
    for attempt in range(5):
        try:
            result = cdp.evaluate(
                """(async () => {
          for (let i = 0; i < 240; i += 1) {
            const ready = document.body.dataset.presentation === 'detection_gold_pilot';
            const status = document.querySelector('#dgEvidenceStatus')?.textContent || '';
            if (ready && status.startsWith('Evidence verified')) break;
            await new Promise(resolve => setTimeout(resolve, 100));
          }
          await document.fonts.ready;
          await Promise.all([...document.images].filter(image => image.src).map(async image => {
            try { await image.decode(); } catch (_) {}
          }));
          await new Promise(requestAnimationFrame);
          await new Promise(requestAnimationFrame);
          return {
            presentation: document.body.dataset.presentation,
            title: document.querySelector('#dgCaseTitle')?.textContent || '',
            module: document.querySelector('#dgModuleProgress')?.textContent || '',
            evidenceStatus: document.querySelector('#dgEvidenceStatus')?.textContent || '',
            evidenceBlocked: !document.querySelector('#dgEvidenceBlocker')?.classList.contains('isHidden'),
            saveState: document.querySelector('#dgSaveState')?.textContent || '',
            serverState: document.querySelector('#dgServerState')?.textContent || '',
            completeDisabled: document.querySelector('#dgComplete')?.disabled ?? false,
            naturalWidth: document.querySelector('#dgBaseImage')?.naturalWidth || 0,
            naturalHeight: document.querySelector('#dgBaseImage')?.naturalHeight || 0,
            legacyLoadStatus: document.querySelector('#status')?.textContent || '',
            bodyText: document.body.innerText.slice(0, 1200),
          };
        })()"""
            )
            break
        except RuntimeError as error:
            if "Execution context was destroyed" not in str(error) or attempt == 4:
                raise
            time.sleep(0.5)
    if (
        not result
        or result.get("presentation") != "detection_gold_pilot"
        or result.get("evidenceBlocked")
        or not str(result.get("evidenceStatus", "")).startswith("Evidence verified")
    ):
        raise RuntimeError(f"detection-gold package did not become ready: {result}")
    return result


def wait_for(cdp: CDP, expression: str, *, timeout_seconds: float = 12) -> Any:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        value = cdp.evaluate(expression)
        if value:
            return value
        time.sleep(0.1)
    raise RuntimeError(f"browser condition timed out: {expression}")


def capture(cdp: CDP, path: Path) -> dict[str, Any]:
    payload = cdp.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(payload["data"]))
    with Image.open(path) as image:
        stats = ImageStat.Stat(image.convert("RGB"))
        spread = max(stats.stddev)
        if image.width < 900 or image.height < 600 or spread < 8:
            raise RuntimeError(f"browser screenshot failed visual-content gate: {image.size}, spread={spread}")
        return {
            "path": path.name,
            "width": image.width,
            "height": image.height,
            "rgb_standard_deviation_max": round(spread, 3),
            "sha256": sha256_file(path),
        }


def apply_viewport(cdp: CDP, profile: dict[str, Any]) -> dict[str, Any]:
    cdp.command(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": profile["css_width"],
            "height": profile["css_height"],
            "deviceScaleFactor": profile["device_scale_factor"],
            "mobile": False,
            "screenWidth": profile["physical_width"],
            "screenHeight": profile["physical_height"],
        },
    )
    time.sleep(0.3)
    wait_ready(cdp)
    audit = cdp.evaluate(
        """(() => {
          const rect = selector => {
            const node = document.querySelector(selector);
            if (!node) return null;
            const value = node.getBoundingClientRect();
            return {left: value.left, top: value.top, right: value.right, bottom: value.bottom,
              width: value.width, height: value.height};
          };
          const overlapArea = (left, right) => {
            if (!left || !right) return 0;
            return Math.max(0, Math.min(left.right, right.right) - Math.max(left.left, right.left)) *
              Math.max(0, Math.min(left.bottom, right.bottom) - Math.max(left.top, right.top));
          };
          const image = rect('#dgBaseImage');
          const overlay = rect('#dgOverlay');
          const evidence = rect('.dgEvidenceColumn');
          const annotation = rect('.dgAnnotationColumn');
          const contextImages = [...document.querySelectorAll('.dgContextStrip img')];
          const nestedHorizontalNodes = [...document.querySelectorAll('.dgMain *')].filter(node =>
            node.clientWidth > 0 && node.scrollWidth > node.clientWidth + 2 &&
              getComputedStyle(node).overflowX !== 'visible' && !node.matches('.dgContactStrip')
          );
          return {
            innerWidth: window.innerWidth,
            innerHeight: window.innerHeight,
            devicePixelRatio: window.devicePixelRatio,
            bodyHorizontalOverflowPixels: Math.max(0, document.documentElement.scrollWidth - window.innerWidth),
            evidenceAnnotationOverlapArea: overlapArea(evidence, annotation),
            primaryEvidenceWidth: image?.width || 0,
            primaryEvidenceHeight: image?.height || 0,
            imageOverlayMaxDelta: image && overlay ? Math.max(
              Math.abs(image.left - overlay.left), Math.abs(image.top - overlay.top),
              Math.abs(image.right - overlay.right), Math.abs(image.bottom - overlay.bottom)
            ) : 999,
            imageNaturalWidth: document.querySelector('#dgBaseImage')?.naturalWidth || 0,
            imageNaturalHeight: document.querySelector('#dgBaseImage')?.naturalHeight || 0,
            contextImagesDecoded: contextImages.every(node =>
              node.complete && node.naturalWidth > 0 && node.naturalHeight > 0
            ),
            evidenceBlockerVisible: !document.querySelector('#dgEvidenceBlocker')?.classList.contains('isHidden'),
            completionDisabled: document.querySelector('#dgComplete')?.disabled ?? false,
            nestedHorizontalScrollerCount: nestedHorizontalNodes.length,
            nestedHorizontalScrollerElements: nestedHorizontalNodes.map(node => ({
              tag: node.tagName, id: node.id, className: String(node.className),
              clientWidth: node.clientWidth, scrollWidth: node.scrollWidth,
              overflowX: getComputedStyle(node).overflowX,
            })),
          };
        })()"""
    )
    audit["profile"] = profile["name"]
    audit["physical_viewport"] = [profile["physical_width"], profile["physical_height"]]
    audit["effective_browser_zoom_percent"] = profile["zoom_percent"]
    checks = {
        "no_horizontal_overflow": audit["bodyHorizontalOverflowPixels"] <= 1,
        "columns_do_not_overlap": audit["evidenceAnnotationOverlapArea"] <= 1,
        "primary_evidence_not_tiny": audit["primaryEvidenceWidth"] >= 400 and audit["primaryEvidenceHeight"] >= 180,
        "image_overlay_aligned": audit["imageOverlayMaxDelta"] <= 1,
        "source_image_decoded": audit["imageNaturalWidth"] > 0 and audit["imageNaturalHeight"] > 0,
        "context_images_decoded": audit["contextImagesDecoded"],
        "evidence_not_blocked": not audit["evidenceBlockerVisible"],
        "completion_starts_disabled": audit["completionDisabled"],
        "no_nested_horizontal_scrollers": audit["nestedHorizontalScrollerCount"] == 0,
    }
    audit["checks"] = checks
    audit["passed"] = all(checks.values())
    return audit


def navigate_module(cdp: CDP, task_type: str) -> dict[str, Any]:
    started = time.perf_counter()
    clicked = cdp.evaluate(
        f"""(() => {{
          const button = document.querySelector('[data-dg-module="{task_type}"]');
          if (!button) return false;
          button.click();
          return true;
        }})()"""
    )
    if not clicked:
        raise RuntimeError(f"module button missing: {task_type}")
    ready = wait_ready(cdp)
    ready["task_type"] = task_type
    ready["navigation_seconds"] = round(time.perf_counter() - started, 3)
    return ready


def seek_proposal_case(cdp: CDP, task_type: str, max_cases: int) -> dict[str, Any]:
    result = navigate_module(cdp, task_type)
    cdp.evaluate(
        """(() => {
          for (const toggle of document.querySelectorAll('[data-dg-layer]')) toggle.checked = true;
          document.querySelector('[data-dg-layer]')?.dispatchEvent(new Event('change', {bubbles: true}));
          return true;
        })()"""
    )
    for attempt in range(max_cases):
        proposal_count = cdp.evaluate("document.querySelectorAll('.dgProposal').length")
        if proposal_count:
            result["proposal_seek_attempts"] = attempt
            result["proposal_count"] = proposal_count
            return result
        cdp.evaluate("document.querySelector('#dgNextCase')?.click(); true")
        result = wait_ready(cdp)
        if cdp.evaluate(
            f"document.querySelector('[data-dg-module=\"{task_type}\"]')?.classList.contains('active') !== true"
        ):
            break
    raise RuntimeError(f"no proposal-bearing current frame found for {task_type}")


def audit_routes_and_privacy() -> dict[str, Any]:
    manifest_response = requests.get(URL + "api/review/manifest", timeout=10)
    config_response = requests.get(URL + "api/review/ui-config", timeout=10)
    root_response = requests.get(URL, timeout=10)
    manifest = manifest_response.json()
    config = config_response.json()
    case = manifest["cases"][0]
    asset = case["evidence_assets"][0]
    relative_path = "/".join(quote(part, safe="") for part in asset["relative_path"].split("/"))
    asset_response = requests.get(f"{URL}evidence/{quote(case['case_id'], safe='')}/{relative_path}", timeout=20)
    forbidden_keys = {
        "benchmark_split",
        "split",
        "split_name",
        "sealed_holdout",
        "expected_answer",
        "gold_answer",
        "human_answer",
        "validation_label",
    }
    hits: list[str] = []

    def visit(value: Any, path: str = "root") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in forbidden_keys:
                    hits.append(f"{path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(manifest, "manifest")
    visit(config, "ui_config")
    checks = {
        "root_http_200": root_response.status_code == 200,
        "manifest_http_200": manifest_response.status_code == 200,
        "ui_config_http_200": config_response.status_code == 200,
        "review_id_exact": manifest.get("review_id") == REVIEW_ID,
        "asset_http_200": asset_response.status_code == 200,
        "asset_content_type_image": asset_response.headers.get("content-type", "").startswith("image/"),
        "asset_nonzero": len(asset_response.content) > 0,
        "asset_content_length_correct": int(asset_response.headers.get("content-length", "-1"))
        == len(asset_response.content),
        "asset_hash_exact": hashlib.sha256(asset_response.content).hexdigest() == asset["sha256"],
        "forbidden_split_or_answer_fields_absent": not hits,
    }
    return {"passed": all(checks.values()), "checks": checks, "forbidden_field_hits": hits}


def run_module_interactions(cdp: CDP) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results: dict[str, Any] = {}
    screenshots: list[dict[str, Any]] = []
    results["player_static"] = seek_proposal_case(cdp, "detection_gold_player_static", 32)
    player_interaction = cdp.evaluate(
        """(() => {
          const before = document.querySelectorAll('.dgProposal').length;
          const raw = document.querySelector('[data-dg-layer="RAW"]');
          raw.checked = true;
          raw.dispatchEvent(new Event('change', {bubbles: true}));
          const proposal = document.querySelector('.dgProposal');
          proposal?.dispatchEvent(new MouseEvent('click', {bubbles: true}));
          return {before, after: document.querySelectorAll('.dgProposal').length,
            provenancePopulated: (document.querySelector('#dgProvenance')?.textContent || '')
              .includes('Diagnostic UUID'),
            selectedProposalCount: document.querySelectorAll('.dgProposal.selected').length};
        })()"""
    )
    results["player_static"]["interaction"] = player_interaction
    screenshots.append(capture(cdp, OUT / "01_PLAYER_STATIC_ANNOTATION_UI.png"))

    results["dense_region"] = seek_proposal_case(cdp, "detection_gold_dense_region", 8)
    mask_result = cdp.evaluate(
        """(async () => {
          document.querySelector('[data-dg-tool="mask"]')?.click();
          const overlay = document.querySelector('#dgOverlay');
          const rectangle = overlay.getBoundingClientRect();
          const expected = [
            {x: rectangle.left + rectangle.width * 0.42, y: rectangle.top + rectangle.height * 0.33},
            {x: rectangle.left + rectangle.width * 0.56, y: rectangle.top + rectangle.height * 0.39},
            {x: rectangle.left + rectangle.width * 0.49, y: rectangle.top + rectangle.height * 0.67},
          ];
          for (const point of expected) {
            overlay.dispatchEvent(new MouseEvent('click', {bubbles: true, clientX: point.x, clientY: point.y}));
            await new Promise(resolve => setTimeout(resolve, 20));
          }
          document.querySelector('#dgFinishMask')?.click();
          await new Promise(resolve => setTimeout(resolve, 100));
          const polygon = document.querySelector('.dgHumanMask');
          const actual = polygon ? [...polygon.points].map(point => {
            const transformed = new DOMPoint(point.x, point.y).matrixTransform(polygon.getScreenCTM());
            return {x: transformed.x, y: transformed.y};
          }) : [];
          const maxDelta = actual.length === expected.length ? Math.max(...actual.map((point, index) =>
            Math.max(Math.abs(point.x - expected[index].x), Math.abs(point.y - expected[index].y)))) : 999;
          return {maskCount: document.querySelectorAll('.dgHumanMask').length, pointCount: actual.length,
            screenRoundTripMaxPixels: maxDelta};
        })()"""
    )
    results["dense_region"]["mask_interaction"] = mask_result
    screenshots.append(capture(cdp, OUT / "02_DENSE_VISIBLE_MASK_UI.png"))

    results["temporal_player"] = navigate_module(cdp, "detection_gold_temporal_player")
    stable_result = cdp.evaluate(
        """(async () => {
          const initialDisabled = document.querySelector('#dgAcceptStableRun')?.disabled ?? false;
          const timeline = document.querySelector('#dgTimeline');
          const count = Number(timeline.max) + 1;
          for (let index = 0; index < count; index += 1) {
            timeline.value = String(index);
            timeline.dispatchEvent(new Event('input', {bubbles: true}));
            await new Promise(resolve => setTimeout(resolve, 35));
            const state = document.querySelector('#dgTemporalState');
            state.value = 'NOT_VISIBLE';
            state.dispatchEvent(new Event('change', {bubbles: true}));
            await new Promise(resolve => setTimeout(resolve, 20));
          }
          const beforeContactDisabled = document.querySelector('#dgAcceptStableRun')?.disabled ?? false;
          const reviewed = document.querySelector('#dgContactReviewed');
          reviewed.checked = true;
          reviewed.dispatchEvent(new Event('change', {bubbles: true}));
          await new Promise(resolve => setTimeout(resolve, 80));
          const enabledAfterReview = !(document.querySelector('#dgAcceptStableRun')?.disabled ?? true);
          document.querySelector('#dgAcceptStableRun')?.click();
          return {contactFrameCount: count, initialDisabled, beforeContactDisabled, enabledAfterReview,
            contactImages: document.querySelectorAll('#dgContactStrip img').length};
        })()"""
    )
    results["temporal_player"]["stable_run_gate"] = stable_result

    results["pitch_boundary"] = navigate_module(cdp, "detection_gold_pitch_boundary")
    cdp.evaluate(
        r"""(() => {
          const note = document.querySelector('#dgNote');
          note.value = 'temporary browser acceptance draft';
          note.dispatchEvent(new Event('input', {bubbles: true}));
          return true;
        })()"""
    )
    time.sleep(0.4)
    results["football_burst"] = navigate_module(cdp, "detection_gold_football_burst")
    results["football_burst"]["contact_frame_count"] = cdp.evaluate(
        "document.querySelectorAll('#dgContactStrip img').length"
    )
    screenshots.append(capture(cdp, OUT / "03_FOOTBALL_BURST_ANNOTATION_UI.png"))
    return results, screenshots


def reload_and_wait(cdp: CDP) -> dict[str, Any]:
    cdp.evaluate("location.reload(); true")
    time.sleep(0.5)
    return wait_ready(cdp)


def browser_persistence_exercise(
    cdp: CDP,
    server: subprocess.Popen[bytes],
    edge: subprocess.Popen[bytes],
) -> tuple[dict[str, Any], CDP, subprocess.Popen[bytes], subprocess.Popen[bytes]]:
    result: dict[str, Any] = {"temporary_decisions_root": str(DECISIONS), "tests": {}}
    navigate_module(cdp, "detection_gold_pitch_boundary")
    cdp.evaluate(
        r"""(() => {
          const note = document.querySelector('#dgNote');
          note.value = 'draft survives reload and browser restart';
          note.dispatchEvent(new Event('input', {bubbles: true}));
          return true;
        })()"""
    )
    time.sleep(0.5)
    reload_and_wait(cdp)
    navigate_module(cdp, "detection_gold_pitch_boundary")
    reload_note = cdp.evaluate("document.querySelector('#dgNote')?.value || ''")
    result["tests"]["reload_draft_recovery"] = reload_note == "draft survives reload and browser restart"

    cdp.close()
    stop_tree(edge)
    for _ in range(100):
        if not port_open(CDP_PORT):
            break
        time.sleep(0.1)
    time.sleep(0.5)
    restart_cdp_port = CDP_PORT + 1
    if port_open(restart_cdp_port):
        raise RuntimeError(f"browser-restart CDP port {restart_cdp_port} is occupied")
    edge = start_edge(restart_cdp_port)
    cdp = connect_page(restart_cdp_port)
    wait_ready(cdp)
    navigate_module(cdp, "detection_gold_pitch_boundary")
    restart_note = cdp.evaluate("document.querySelector('#dgNote')?.value || ''")
    result["tests"]["browser_restart_draft_recovery"] = restart_note == "draft survives reload and browser restart"

    cdp.evaluate("document.querySelector('#dgSaveCase')?.click(); true")
    first_ack = wait_for(
        cdp,
        r"""(() => {
          const status = document.querySelector('#dgSaveState')?.textContent || '';
          const text = document.querySelector('#dgServerState')?.textContent || '';
          const match = text.match(/server\s+(\d+)/);
          return status === 'Saved to server' && Number(match?.[1] || 0) >= 1 ? {status, text} : null;
        })()""",
    )
    result["tests"]["server_acknowledged_save"] = first_ack

    stop_tree(server)
    time.sleep(0.5)
    cdp.evaluate("document.querySelector('#dgSaveCase')?.click(); true")
    offline = wait_for(
        cdp,
        """(() => {
          const status = document.querySelector('#dgSaveState')?.textContent || '';
          const text = document.querySelector('#dgServerState')?.textContent || '';
          return status.includes('Offline') && text.includes('pending 1') ? {status, text} : null;
        })()""",
    )
    result["tests"]["offline_outbox_queued"] = offline

    server = start_server()
    wait_server(server)
    recovered_page = reload_and_wait(cdp)
    recovered = wait_for(
        cdp,
        r"""(() => {
          const status = document.querySelector('#dgSaveState')?.textContent || '';
          const text = document.querySelector('#dgServerState')?.textContent || '';
          const match = text.match(/server\s+(\d+)/);
          return status === 'Saved to server' && Number(match?.[1] || 0) >= 2 && text.includes('pending 0')
            ? {status, text, sequence: Number(match[1])} : null;
        })()""",
    )
    result["tests"]["offline_outbox_replayed_after_server_restart"] = recovered
    result["tests"]["recovered_page"] = recovered_page

    stop_tree(server)
    time.sleep(0.5)
    server = start_server()
    wait_server(server)
    reload_and_wait(cdp)
    state = requests.get(URL + "api/review/state", timeout=10).json()
    result["tests"]["server_restart_materialization"] = {
        "event_sequence": state.get("event_sequence"),
        "annotation_count": len(state.get("annotations", {})),
        "passed": state.get("event_sequence") == 2 and len(state.get("annotations", {})) == 2,
    }
    incomplete = cdp.evaluate(
        """fetch('/api/review/detection-gold-complete', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({pending_outbox_events: 0, evidence_blocker_count: 0,
            unresolved_draft_count: 0, unresolved_divergence: false})
        }).then(async response => ({status: response.status, ok: response.ok, body: await response.text()}))"""
    )
    result["tests"]["incomplete_completion_blocked"] = {
        "status": incomplete["status"],
        "ok": incomplete["ok"],
        "passed": not incomplete["ok"] and incomplete["status"] >= 400,
    }
    result["tests"]["complete_button_remains_disabled"] = cdp.evaluate(
        "document.querySelector('#dgComplete')?.disabled === true"
    )
    events = (DECISIONS / "review_decision_events.jsonl").read_text(encoding="utf-8").splitlines()
    result["event_ledger"] = {
        "line_count": len([line for line in events if line.strip()]),
        "sha256": sha256_file(DECISIONS / "review_decision_events.jsonl"),
    }
    result["passed"] = all(
        bool(value.get("passed", value)) if isinstance(value, dict) else bool(value)
        for value in result["tests"].values()
    )
    return result, cdp, server, edge


def main() -> None:
    if not EDGE.exists():
        raise RuntimeError("Microsoft Edge is required for exact-package browser acceptance")
    if port_open(8807):
        raise RuntimeError("port 8807 is occupied; stop the existing review server")
    if port_open(CDP_PORT):
        raise RuntimeError(f"CDP port {CDP_PORT} is occupied")
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True)
    DECISIONS.mkdir(parents=True)
    (DECISIONS / "snapshots").mkdir()
    shutil.copy2(PRODUCTION_DECISIONS / "review_decisions.json", DECISIONS / "review_decisions.json")
    (DECISIONS / "review_decision_events.jsonl").write_bytes(b"")
    production_before = tree_manifest(PRODUCTION_DECISIONS)
    started = time.perf_counter()
    server: subprocess.Popen[bytes] | None = None
    edge: subprocess.Popen[bytes] | None = None
    cdp: CDP | None = None
    final: dict[str, Any] = {
        "schema_version": "football_intelligence.m5_5g1a.browser_acceptance.v1",
        "url": URL,
        "review_id": REVIEW_ID,
        "reviewer_session_id": SESSION,
        "exact_package_manifest": str(PACKAGE / "reviewer_manifest.json"),
        "temporary_decisions_root": str(DECISIONS),
        "real_decisions_root_opened": False,
    }
    try:
        server = start_server()
        wait_server(server)
        edge = start_edge()
        cdp = connect_page()
        final["initial_ready"] = wait_ready(cdp)
        final["route_and_privacy_audit"] = audit_routes_and_privacy()

        profiles = [
            {
                "name": "1024x768",
                "css_width": 1024,
                "css_height": 768,
                "physical_width": 1024,
                "physical_height": 768,
                "device_scale_factor": 1,
                "zoom_percent": 100,
            },
            {
                "name": "1366x768",
                "css_width": 1366,
                "css_height": 768,
                "physical_width": 1366,
                "physical_height": 768,
                "device_scale_factor": 1,
                "zoom_percent": 100,
            },
            {
                "name": "1440x900",
                "css_width": 1440,
                "css_height": 900,
                "physical_width": 1440,
                "physical_height": 900,
                "device_scale_factor": 1,
                "zoom_percent": 100,
            },
            {
                "name": "1920x1080",
                "css_width": 1920,
                "css_height": 1080,
                "physical_width": 1920,
                "physical_height": 1080,
                "device_scale_factor": 1,
                "zoom_percent": 100,
            },
            {
                "name": "2560x1440",
                "css_width": 2560,
                "css_height": 1440,
                "physical_width": 2560,
                "physical_height": 1440,
                "device_scale_factor": 1,
                "zoom_percent": 100,
            },
            {
                "name": "1440x900_at_125_percent",
                "css_width": 1152,
                "css_height": 720,
                "physical_width": 1440,
                "physical_height": 900,
                "device_scale_factor": 1.25,
                "zoom_percent": 125,
            },
        ]
        final["visual_regression"] = [apply_viewport(cdp, profile) for profile in profiles]
        apply_viewport(cdp, profiles[2])
        module_started = time.perf_counter()
        modules, screenshots = run_module_interactions(cdp)
        final["module_interactions"] = modules
        final["screenshots"] = screenshots
        final["temporary_interaction_seconds"] = round(time.perf_counter() - module_started, 3)

        persistence, cdp, server, edge = browser_persistence_exercise(cdp, server, edge)
        final["persistence"] = persistence
        final["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        final["production_decisions_preservation"] = {
            "before": production_before,
            "after": tree_manifest(PRODUCTION_DECISIONS),
        }
        final["production_decisions_preservation"]["passed"] = (
            final["production_decisions_preservation"]["before"]["tree_hash"]
            == final["production_decisions_preservation"]["after"]["tree_hash"]
        )
        final["passed"] = all(
            (
                final["route_and_privacy_audit"]["passed"],
                all(row["passed"] for row in final["visual_regression"]),
                modules["player_static"]["interaction"]["provenancePopulated"],
                modules["dense_region"]["mask_interaction"]["screenRoundTripMaxPixels"] <= 1,
                modules["temporal_player"]["stable_run_gate"]["initialDisabled"],
                modules["temporal_player"]["stable_run_gate"]["beforeContactDisabled"],
                modules["temporal_player"]["stable_run_gate"]["enabledAfterReview"],
                modules["football_burst"]["contact_frame_count"] == 9,
                persistence["passed"],
                final["production_decisions_preservation"]["passed"],
            )
        )
    finally:
        if cdp is not None:
            try:
                cdp.close()
            except OSError:
                pass
        stop_tree(server)
        stop_tree(edge)
        for process in reversed(ACTIVE_PROCESSES):
            stop_tree(process)

    visual_payload = {
        "schema_version": "football_intelligence.m5_5g1a.visual_regression.v1",
        "passed": all(row["passed"] for row in final.get("visual_regression", [])),
        "profiles": final.get("visual_regression", []),
        "screenshots": final.get("screenshots", []),
    }
    persistence_payload = final.get("persistence", {"passed": False})
    write_json(OUT / "browser_persistence_results.json", final)
    write_json(OUT / "visual_regression_results.json", visual_payload)
    write_json(OUT / "production_persistence_exercise.json", persistence_payload)
    time_estimate = json.loads((TIMING / "annotation_time_estimate.json").read_text(encoding="utf-8"))
    interaction_payload = {
        "schema_version": "football_intelligence.m5_5g1a.interaction_efficiency_validation.v1",
        "passed": bool(final.get("passed")) and time_estimate["within_budget"],
        "temporary_browser_exercise_completed": True,
        "temporary_decisions_root": str(DECISIONS),
        "real_decisions_root_opened": False,
        "automation_elapsed_seconds_not_human_annotation_time": final.get("elapsed_seconds"),
        "estimated_active_minutes": time_estimate["estimated_active_minutes"],
        "target_range_minutes": time_estimate["acceptable_range_minutes"],
        "notes_optional": True,
        "hard_cases_removed_for_time": False,
        "machine_truth_prefilled": False,
        "stable_run_requires_complete_contact_strip": True,
        "efficiency_features_exercised": [
            "module tabs",
            "next unresolved navigation",
            "candidate proposal selection",
            "original-pixel visible-mask drawing",
            "frame stepping",
            "stable-run contact-strip gate",
            "IndexedDB draft recovery",
            "offline outbox recovery",
        ],
    }
    write_json(TIMING / "interaction_efficiency_validation.json", interaction_payload)
    write_json(TIMING / "temporary_browser_timing_exercise.json", interaction_payload)
    if not final.get("passed"):
        raise RuntimeError(f"M5.5G.1A browser acceptance failed; inspect {OUT}")
    print(json.dumps({"passed": True, "elapsed_seconds": final["elapsed_seconds"], "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
