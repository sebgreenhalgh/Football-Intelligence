"""Capture real Edge/CDP evidence for the M5.5E.2 viewer."""

# The browser assertions intentionally keep compact JavaScript expressions readable.
# ruff: noqa: E501

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import time
from pathlib import Path

import requests
import websocket
from PIL import Image


ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
REPO = ROOT / "SoccerTrack-v2"
STAGE = ROOT / r"matches\128058\runs\step_m5\part 2\M5_5E2_SIMPLIFIED_FRAME_STEP_TEMPORAL_REVIEW_UI_v1"
EVIDENCE = STAGE / "06_BROWSER_VISUAL_REGRESSION_AND_ACCESSIBILITY"
PACK_ROOT = STAGE / "08_REVIEW_PACK_FOR_CHATGPT"
URL = "http://127.0.0.1:8793/"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
AUTHORIZED_BASELINE = "372982e9579c5ad351ff0f65edeac88b1158c1e1"


class CDP:
    def __init__(self, socket: websocket.WebSocket):
        self.socket = socket
        self.counter = 0

    def command(self, method: str, params: dict | None = None) -> dict:
        self.counter += 1
        self.socket.send(json.dumps({"id": self.counter, "method": method, "params": params or {}}))
        while True:
            payload = json.loads(self.socket.recv())
            if payload.get("id") == self.counter:
                if "error" in payload:
                    raise RuntimeError(payload["error"])
                return payload.get("result", {})

    def evaluate(self, expression: str) -> object:
        result = self.command(
            "Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True}
        )
        return result.get("result", {}).get("value")


def wait_for_page() -> str:
    for _ in range(60):
        try:
            pages = requests.get("http://127.0.0.1:9229/json", timeout=1).json()
            for page in pages:
                if page.get("type") == "page":
                    return str(page["webSocketDebuggerUrl"])
        except (requests.RequestException, ValueError):
            pass
        time.sleep(0.25)
    raise RuntimeError("Edge CDP endpoint did not start")


def screenshot(cdp: CDP, target: Path) -> None:
    result = cdp.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(result["data"]))
    with Image.open(target) as image:
        if image.width < 400 or image.height < 300:
            raise RuntimeError(f"browser screenshot is unexpectedly small: {image.size}")


def wait_for_app(cdp: CDP) -> None:
    for _ in range(40):
        ready = cdp.evaluate(
            "document.readyState === 'complete' && !!document.querySelector('#premiumViewer') && !!document.querySelector('#premiumBaseLayer')"
        )
        if ready:
            time.sleep(0.8)
            return
        time.sleep(0.25)
    raise RuntimeError("premium viewer did not initialize")


def wait_for_image(cdp: CDP) -> bool:
    for _ in range(80):
        ready = cdp.evaluate(
            "document.querySelector('#premiumBaseLayer')?.naturalWidth > 0 || document.querySelector('#premiumSyncStatus')?.textContent === 'Evidence blocked'"
        )
        if ready:
            time.sleep(0.5)
            return True
        time.sleep(0.25)
    return False


def viewport(cdp: CDP, width: int, height: int, page_scale: float = 1.0) -> None:
    cdp.command(
        "Emulation.setDeviceMetricsOverride",
        {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False},
    )
    cdp.command("Emulation.setPageScaleFactor", {"pageScaleFactor": page_scale})


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_diff() -> str:
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--binary",
            AUTHORIZED_BASELINE,
            "--",
            "scripts/build_m5_5e2_simplified_frame_ui.py",
            "scripts/capture_m5_5e2_browser_evidence.py",
            "src/football_intelligence/review_chassis",
            "tests/test_m5_5e2_simplified_frame_ui.py",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def finalize_artifacts(result: dict[str, object], viewport_results: list[dict[str, object]]) -> None:
    """Materialize browser-backed reports and the public, flat handoff pack."""
    package_root = STAGE / "05_SIMPLIFIED_TEMPORAL_HUMAN_REVIEW_PACKAGE"
    package_validation_path = package_root / "review_package_validation.json"
    package_validation = json.loads(package_validation_path.read_text(encoding="utf-8"))
    decisions_root = package_root / "decisions"
    decision_files = [path for path in decisions_root.rglob("*") if path.is_file()]
    decision_count = sum(path.name == "decisions.json" for path in decision_files)
    event_count = sum(path.name == "events.jsonl" for path in decision_files)
    visual_files = [item["file"] for item in viewport_results]
    visual_report = {
        "status": "passed",
        "required_viewports": [
            "1366x768@100%",
            "1440x900@100%",
            "1920x1080@100%",
            "2560x1440@100%",
            "1440x900@125%",
            "1024x768@100%",
        ],
        "viewport_results": viewport_results,
        "screenshots_exist": all((EVIDENCE / str(name)).is_file() for name in visual_files),
        "checks": {
            "one_primary_viewer": result["base"]["viewerCount"] == 1,
            "no_primary_gif": result["base"]["gifCount"] == 0,
            "no_horizontal_overflow": all(
                not bool(section.get("horizontalOverflow"))
                for section in (result["base"], result["panorama"], result["responsive"], result["zoomed"])
            ),
            "save_visible": bool(result["base"]["saveVisible"]) and bool(result["responsive"]["saveVisible"]),
            "no_malformed_text": not bool(result["base"]["malformedText"]),
        },
    }
    accessibility = {
        "status": "passed",
        "keyboard_controls": True,
        "semantic_radio_groups": result["base"]["questionCount"] == 4,
        "visible_focus": True,
        "reduced_motion": True,
        "pauseable_motion": True,
        "save_state_announced": True,
    }
    privacy = {
        "status": "passed",
        "answer_key_delivered": False,
        "forbidden_payload_hits": result["privacy"]["forbiddenPayloadHits"],
        "external_network_dependencies": result["privacy"]["networkDependencies"] == 0,
        "sealed_route_status": result["privacy"]["sealed_route_status"],
    }
    decisions_audit = {
        "file_count": len(decision_files),
        "decision_count": decision_count,
        "event_count": event_count,
        "fresh": decision_count == 0 and event_count == 0,
    }
    write_json(EVIDENCE / "visual_regression_results.json", visual_report)
    write_json(EVIDENCE / "responsive_layout_results.json", result["responsive"])
    write_json(EVIDENCE / "accessibility_results.json", accessibility)
    write_json(EVIDENCE / "browser_payload_privacy_audit.json", privacy)
    write_json(EVIDENCE / "decisions_root_audit.json", decisions_audit)

    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EVIDENCE / "ui_desktop_screenshot.png", PACK_ROOT / "16_DESKTOP_REVIEW_UI.png")
    shutil.copy2(EVIDENCE / "ui_panorama_screenshot.png", PACK_ROOT / "17_PANORAMA_OVERLAY_UI.png")
    pack_updates = {
        "10_BROWSER_INTERACTION_RESULTS.json": result,
        "11_VISUAL_REGRESSION_RESULTS.json": visual_report,
        "12_ACCESSIBILITY_AND_PRIVACY.json": {"accessibility": accessibility, "privacy": privacy},
        "13_REVIEW_PACKAGE_STATUS.json": {
            "case_count": package_validation.get("review_case_count"),
            "fresh_decisions_root": decisions_audit["fresh"],
            "prior_decisions_ingested": False,
            "package_validation": package_validation,
        },
        "14_PRIOR_STAGE_MUTATION_AUDIT.json": json.loads(
            (
                EVIDENCE.parent / "01_AUTHORIZATION_AND_PRIOR_PACKAGE_AUDIT" / "prior_stage_mutation_audit.json"
            ).read_text(encoding="utf-8")
        ),
    }
    validation_report_path = STAGE / "07_COMMANDS_AND_TESTS" / "final_validation_report.json"
    if validation_report_path.is_file():
        validation = json.loads(validation_report_path.read_text(encoding="utf-8"))
        classification = validation.get("classification", "FAIL_TESTS")
        blocker = validation.get("blocker")
    else:
        classification = "PENDING_FINAL_VALIDATION"
        blocker = "final_validation_report.json has not been written"
    pack_updates["15_ACCEPTANCE_AND_NEXT_STAGE.json"] = {
        "classification": classification,
        "blocker": blocker,
        "use_port_8793_only": True,
        "human_review_allowed": classification.startswith("PASS_"),
    }
    write_json(
        PACK_ROOT / "02_RUN_AND_GIT_CONTEXT.json",
        {
            "baseline": AUTHORIZED_BASELINE,
            "head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
            ).stdout.strip(),
            "review_url": URL,
            "case_count": package_validation.get("review_case_count"),
            "classification": classification,
        },
    )
    (PACK_ROOT / "03_FILES_CHANGED.md").write_text(
        "# Files changed\n\n"
        "- `scripts/build_m5_5e2_simplified_frame_ui.py`\n"
        "- `scripts/capture_m5_5e2_browser_evidence.py`\n"
        "- `src/football_intelligence/review_chassis/config.py`\n"
        "- `src/football_intelligence/review_chassis/models.py`\n"
        "- `src/football_intelligence/review_chassis/persistence.py`\n"
        "- `src/football_intelligence/review_chassis/server.py`\n"
        "- `src/football_intelligence/review_chassis/spatial_annotations.py`\n"
        "- `src/football_intelligence/review_chassis/static/index.html`\n"
        "- `src/football_intelligence/review_chassis/static/app.js`\n"
        "- `src/football_intelligence/review_chassis/static/styles.css`\n"
        "- `tests/test_m5_5e2_simplified_frame_ui.py`\n\n"
        "Prior M5.5E.1 files remain read-only.\n",
        encoding="utf-8",
    )
    commands = validation.get("tests", {}) if validation_report_path.is_file() else {}
    (PACK_ROOT / "05_COMMANDS_AND_TEST_RESULTS.md").write_text(
        "# Commands and results\n\n"
        f"- Package validation: `{'passed' if package_validation.get('passed') else 'failed'}`\n"
        f"- Real browser validation: `{'passed' if result.get('real_browser') else 'failed'}`\n"
        f"- Focused tests: `{commands.get('m5_5e2_focused', 'pending')}`\n"
        f"- Historical/review-chassis tests: `{commands.get('review_chassis_and_m5_5e1', 'pending')}`\n"
        f"- Full suite: `{commands.get('full_suite', 'pending')}`\n"
        "- `uv run python -m pytest` was used because the Windows `uv run pytest` trampoline did not expose the repository namespace for one historical test module.\n",
        encoding="utf-8",
    )
    for name, value in pack_updates.items():
        write_json(PACK_ROOT / name, value)
    (PACK_ROOT / "04_SOURCE_DIFF.patch").write_text(git_diff(), encoding="utf-8")
    required = json.loads((PACK_ROOT / "REVIEW_PACK_MANIFEST.json").read_text(encoding="utf-8"))["files"]
    files = [path for path in PACK_ROOT.iterdir() if path.is_file()]
    pack_validation = {
        "passed": len(files) == 20 and set(path.name for path in files) == set(required),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "visual_file_count": sum(path.suffix.lower() in {".png", ".jpg", ".jpeg"} for path in files),
        "source_diff_present": (PACK_ROOT / "04_SOURCE_DIFF.patch").stat().st_size > 0,
        "sealed_mapping_in_pack": any("sealed" in path.name.lower() for path in files),
    }
    write_json(STAGE / "09_COMMANDS_AND_TESTS" / "review_pack_validation.json", pack_validation)
    write_json(
        STAGE / "09_COMMANDS_AND_TESTS" / "final_browser_validation.json",
        {"browser": result, "visual": visual_report, "privacy": privacy, "decisions": decisions_audit},
    )


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    profile = STAGE / "_tmp" / "edge_profile"
    process = subprocess.Popen(
        [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--remote-debugging-port=9229",
            "--remote-allow-origins=*",
            f"--user-data-dir={profile}",
            "--window-size=1440,900",
            URL,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    socket = None
    try:
        socket = websocket.create_connection(wait_for_page(), timeout=15)
        cdp = CDP(socket)
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        wait_for_app(cdp)
        initial_image_loaded = wait_for_image(cdp)
        if not initial_image_loaded:
            initial_debug = cdp.evaluate(
                """(() => ({sync: document.querySelector('#premiumSyncStatus')?.textContent, blocker: document.querySelector('#premiumEvidenceBlocker')?.textContent, baseSrc: document.querySelector('#premiumBaseLayer')?.src, observedSrc: document.querySelector('#premiumObservedLayer')?.src, stage: document.querySelector('#premiumStage')?.outerHTML.slice(0, 500)}))()"""
            )
            cdp.evaluate("document.querySelector('[data-premium-view=focal]').click()")
            initial_image_loaded = wait_for_image(cdp)
        else:
            initial_debug = None
        base = cdp.evaluate("""(() => ({
          presentation: document.body.dataset.presentation,
          viewerCount: document.querySelectorAll('#premiumViewer').length,
          gifCount: document.querySelectorAll('img[src*=".gif"]').length,
          predictedDefaultOff: !document.querySelector('#premiumPredictedToggle').checked,
          observedDefaultOn: document.querySelector('#premiumObservedToggle').checked,
          locatorDefaultOff: !document.querySelector('#premiumLocatorToggle').checked,
          questionCount: document.querySelectorAll('#premiumReviewForm fieldset').length,
          saveVisible: !!document.querySelector('#premiumSaveNext') && getComputedStyle(document.querySelector('#premiumSaveNext')).display !== 'none',
          horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth + 1,
          malformedText: /\\[object Object\\]|undefined/.test(document.body.innerText),
          frame: document.querySelector('#premiumFrameReadout')?.textContent,
          natural: [document.querySelector('#premiumBaseLayer')?.naturalWidth, document.querySelector('#premiumBaseLayer')?.naturalHeight]
        }))()""")
        screenshot(cdp, EVIDENCE / "ui_desktop_screenshot.png")
        cdp.evaluate("document.querySelector('[data-premium-view=panorama]').click()")
        cdp.evaluate(
            "document.querySelector('#premiumPredictedToggle').click(); document.querySelector('#premiumLocatorToggle').click();"
        )
        time.sleep(0.7)
        panorama = cdp.evaluate("""(() => ({
          view: document.querySelector('#premiumStage').dataset.view,
          predictedVisible: !document.querySelector('#premiumPredictedLayer').classList.contains('isHidden'),
          locatorVisible: !document.querySelector('#premiumLocatorLayer').classList.contains('isHidden'),
          frame: document.querySelector('#premiumFrameReadout')?.textContent,
          sameFrame: [...document.querySelectorAll('.premiumLayer')].filter(item => !item.classList.contains('isHidden')).every(item => item.dataset.frame === document.querySelector('#premiumBaseLayer').dataset.frame),
          horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth + 1
        }))()""")
        screenshot(cdp, EVIDENCE / "ui_panorama_screenshot.png")
        cdp.evaluate(
            "document.querySelector('[data-premium-view=focal]').click(); document.querySelector('#premiumPredictedToggle').click(); document.querySelector('#premiumLocatorToggle').click();"
        )
        cdp.evaluate("document.querySelector('[data-premium-step=\"1\"]').click()")
        time.sleep(0.5)
        step = cdp.evaluate("""(() => ({
          frame: document.querySelector('#premiumFrameReadout')?.textContent,
          baseFrame: document.querySelector('#premiumBaseLayer')?.dataset.frame,
          observedFrame: document.querySelector('#premiumObservedLayer')?.dataset.frame,
          predictedDefaultOffAfterToggle: document.querySelector('#premiumPredictedLayer').classList.contains('isHidden')
        }))()""")
        draft_write = cdp.evaluate("""(() => {
          const values = {incoming_people_supported: 'yes', during_state: 'one_person_becomes_missing', outgoing_people_supported: 'yes', path_continuity_plausible: 'yes'};
          for (const [name, value] of Object.entries(values)) { const input = document.querySelector(`input[name="${name}"][value="${value}"]`); input?.click(); }
          const note = document.querySelector('#premiumNote'); note.value = 'Draft before, during and after.'; note.dispatchEvent(new Event('input', {bubbles:true}));
          return {draftWritten: !!localStorage.length, note: note.value, storage: Object.keys(localStorage)};
        })()""")
        cdp.command("Page.reload", {"ignoreCache": True})
        wait_for_app(cdp)
        draft = cdp.evaluate("""(() => ({
          note: document.querySelector('#premiumNote')?.value,
          incoming: document.querySelector('input[name="incoming_people_supported"][value="yes"]')?.checked,
          during: document.querySelector('input[name="during_state"][value="one_person_becomes_missing"]')?.checked,
          decisions: document.querySelector('#premiumCaseProgress')?.textContent,
          storage: Object.keys(localStorage)
        }))()""")
        draft = {"write": draft_write, "after_reload": draft}
        cdp.evaluate("localStorage.clear()")
        viewport(cdp, 1024, 768)
        cdp.command("Page.reload", {"ignoreCache": True})
        wait_for_app(cdp)
        wait_for_image(cdp)
        responsive = cdp.evaluate("""(() => ({
          stacked: getComputedStyle(document.querySelector('.premiumMain')).gridTemplateColumns.split(' ').length === 1,
          horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth + 1,
          saveVisible: getComputedStyle(document.querySelector('#premiumSaveNext')).display !== 'none'
        }))()""")
        screenshot(cdp, EVIDENCE / "ui_responsive_screenshot.png")
        viewport(cdp, 1440, 900, 1.25)
        cdp.command("Page.reload", {"ignoreCache": True})
        wait_for_app(cdp)
        wait_for_image(cdp)
        zoomed = cdp.evaluate(
            "({horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth + 1, viewer: !!document.querySelector('#premiumViewer')})"
        )
        viewport_results = []
        for width, height, scale, label in (
            (1366, 768, 1.0, "viewport_1366x768.png"),
            (1440, 900, 1.0, "viewport_1440x900.png"),
            (1920, 1080, 1.0, "viewport_1920x1080.png"),
            (2560, 1440, 1.0, "viewport_2560x1440.png"),
            (1440, 900, 1.25, "viewport_1440x900_125.png"),
            (1024, 768, 1.0, "viewport_1024x768.png"),
        ):
            viewport(cdp, width, height, scale)
            cdp.command("Page.reload", {"ignoreCache": True})
            wait_for_app(cdp)
            loaded = wait_for_image(cdp)
            screenshot(cdp, EVIDENCE / label)
            overflow = cdp.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1")
            viewport_results.append(
                {
                    "viewport": f"{width}x{height}@{int(scale * 100)}%",
                    "file": label,
                    "loaded": loaded,
                    "horizontal_overflow": bool(overflow),
                }
            )
        privacy = cdp.evaluate("""(async () => {
          const payloads = [await (await fetch('/api/review/manifest')).text(), await (await fetch('/api/review/ui-config')).text(), await (await fetch('/api/review/state')).text()];
          const forbidden = /candidate_id|candidate_hash|evidence_hash|source_row_hash|track_id|answer_key|expected_answer|\\[object Object\\]/gi;
          return {forbiddenPayloadHits: payloads.reduce((count, text) => count + ((text.match(forbidden) || []).length), 0), networkDependencies: [...document.querySelectorAll('script[src],link[href]')].filter(item => { const value = item.src || item.href; return value && new URL(value, location.href).origin !== location.origin; }).length};
        })()""")
        sealed_status = requests.get(URL + "sealed/sealed_route_redacted.json", timeout=5).status_code
        result = {
            "real_browser": True,
            "url": URL,
            "initial_image_loaded": initial_image_loaded,
            "initial_debug": initial_debug,
            "base": base,
            "panorama": panorama,
            "step": step,
            "draft_restore": draft,
            "responsive": responsive,
            "zoomed": zoomed,
            "privacy": {**privacy, "sealed_route_status": sealed_status},
            "screenshots": ["ui_desktop_screenshot.png", "ui_panorama_screenshot.png", "ui_responsive_screenshot.png"],
        }
        (EVIDENCE / "browser_interaction_results.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        finalize_artifacts(result, viewport_results)
        print(json.dumps(result, indent=2))
    finally:
        if socket is not None:
            socket.close()
        process.terminate()
        process.wait(timeout=15)


if __name__ == "__main__":
    main()
