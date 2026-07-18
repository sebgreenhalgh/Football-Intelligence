"""Validate the M5.5F.1A gold annotation package in a real browser."""

# ruff: noqa: E501

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
import websocket
from PIL import Image, ImageDraw

from football_intelligence.review_chassis.completion import validate_completion_bundle
from football_intelligence.review_chassis.hashing import sha256_file


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / r"matches\128058\runs\step_m5\part 2\M5_5F1A_ON_PITCH_GOLD_STRAND_BENCHMARK_AND_SPORTS_MOT_ARCHITECTURE_RESET_v1"
)
PACKAGE = STAGE / "10_GOLD_STRAND_ANNOTATION_PACKAGE"
OUT = STAGE / "13_COMMANDS_AND_TESTS" / "browser_evidence"
SMOKE_DECISIONS = STAGE / "_tmp" / "browser_smoke_decisions"
EDGE_PROFILE = STAGE / "_tmp" / "edge_profile_9240"
URL = "http://127.0.0.1:8800/"
CDP_URL = "http://127.0.0.1:9240"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
UV = shutil.which("uv")
if UV is None:
    raise RuntimeError("uv must be available on PATH for browser acceptance")
SESSION = "m5_5f1a_gold_strand_annotation_human_reviewer"


class CDP:
    def __init__(self, socket: websocket.WebSocket):
        self.socket = socket
        self.counter = 0

    def command(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.counter += 1
        self.socket.send(json.dumps({"id": self.counter, "method": method, "params": params or {}}))
        while True:
            payload = json.loads(self.socket.recv())
            if payload.get("id") == self.counter:
                if "error" in payload:
                    raise RuntimeError(payload["error"])
                return payload.get("result", {})

    def evaluate(self, expression: str) -> Any:
        result = self.command(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        return result.get("result", {}).get("value")

    def key(self, key: str, code: str, *, modifiers: int = 0) -> None:
        for event_type in ("keyDown", "keyUp"):
            self.command(
                "Input.dispatchKeyEvent",
                {"type": event_type, "key": key, "code": code, "modifiers": modifiers},
            )


def wait_for_page() -> str:
    for _ in range(200):
        try:
            pages = requests.get(f"{CDP_URL}/json", timeout=1).json()
            for page in pages:
                if page.get("type") == "page" and str(page.get("url", "")).startswith(URL):
                    return str(page["webSocketDebuggerUrl"])
        except (requests.RequestException, ValueError):
            pass
        time.sleep(0.25)
    raise RuntimeError("Edge CDP endpoint did not start")


def screenshot(cdp: CDP, path: Path) -> None:
    data = cdp.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})["data"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(data))
    with Image.open(path) as image:
        if image.width < 1000 or image.height < 600:
            raise RuntimeError(f"browser screenshot unexpectedly small: {image.size}")


def stop_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        subprocess.run(
            ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()


def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{URL.rstrip('/')}{path}", json=payload, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"{path} failed ({response.status_code}): {response.text}")
    return response.json()


def proposed_annotations(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "frame_sequence": int(record["frame_sequence"]),
            "A": dict(record["proposed_annotations"]["A"]),
            "B": dict(record["proposed_annotations"]["B"]),
        }
        for record in case["visible_metadata"]["frame_records"]
    ]


def completion_smoke(manifest: dict[str, Any]) -> dict[str, Any]:
    state = requests.get(f"{URL}api/review/state", timeout=10).json()
    for case in manifest["cases"]:
        if case["case_id"] in state.get("decisions", {}):
            continue
        if case["task_type"] == "pitch_polygon_approval":
            metadata = case["visible_metadata"]
            payload = {
                "case_id": case["case_id"],
                "decision": "PITCH_POLYGON_APPROVED",
                "structured_review": {
                    "polygon_vertices": metadata["polygon_vertices"],
                    "tolerance_pixels": metadata["tolerance_pixels"],
                    "source_frame_sha256": metadata["source_frame_sha256"],
                },
                "input_source": "browser_smoke_completion",
                "elapsed_active_seconds": 12,
            }
        else:
            payload = {
                "case_id": case["case_id"],
                "decision": "SEQUENCE_ANNOTATED",
                "structured_review": {
                    "frame_annotations": proposed_annotations(case),
                    "interaction_metrics": {
                        "clicks": 0,
                        "accepted_in_runs": len(case["visible_metadata"]["frame_records"]) * 2,
                        "manual_bbox_count": 0,
                        "active_seconds": 12,
                    },
                },
                "input_source": "browser_smoke_completion",
                "elapsed_active_seconds": 12,
            }
        state = post("/api/review/decision", payload)
    post("/api/review/complete", {"elapsed_active_seconds": 12})
    validation = validate_completion_bundle(SMOKE_DECISIONS)
    first_export = json.loads((SMOKE_DECISIONS / "completed_review.json").read_text(encoding="utf-8"))
    first_state = first_export["state"]
    hashes_before = {
        name: sha256_file(SMOKE_DECISIONS / name)
        for name in (
            "completed_review.json",
            "completed_review_events.jsonl",
            "completed_review_manifest.json",
            "completed_review_summary.json",
        )
    }
    post("/api/review/complete", {"elapsed_active_seconds": 12})
    hashes_after = {name: sha256_file(SMOKE_DECISIONS / name) for name in hashes_before}
    retry_export = json.loads((SMOKE_DECISIONS / "completed_review.json").read_text(encoding="utf-8"))
    retry_state = retry_export["state"]
    return {
        "validation": validation,
        "idempotent_retry_preserved_all_artifact_hashes": hashes_before == hashes_after,
        "first_decision_state_hash": first_export["decision_state_hash"],
        "retry_decision_state_hash": retry_export["decision_state_hash"],
        "state_value_differences": {
            key: {"first": first_state.get(key), "retry": retry_state.get(key)}
            for key in sorted(set(first_state) | set(retry_state))
            if first_state.get(key) != retry_state.get(key)
        },
        "artifact_hashes": hashes_after,
    }


def make_composite(pitch_path: Path, annotation_path: Path, output: Path) -> None:
    with Image.open(pitch_path).convert("RGB") as pitch, Image.open(annotation_path).convert("RGB") as annotation:
        width = max(pitch.width, annotation.width)
        canvas = Image.new("RGB", (width, pitch.height + annotation.height + 52), "#111513")
        canvas.paste(pitch, (0, 0))
        canvas.paste(annotation, (0, pitch.height + 52))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (20, pitch.height + 16),
            "Mandatory pitch approval above; frame-level A/B gold annotation below",
            fill="#f4f6f5",
        )
        canvas.save(output, optimize=True)


def main() -> None:
    if SMOKE_DECISIONS.exists():
        shutil.rmtree(SMOKE_DECISIONS)
    if EDGE_PROFILE.exists():
        shutil.rmtree(EDGE_PROFILE)
    OUT.mkdir(parents=True, exist_ok=True)
    SMOKE_DECISIONS.mkdir(parents=True)
    server = subprocess.Popen(
        [
            str(UV),
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
            str(SMOKE_DECISIONS),
            "--sealed-mapping",
            str(PACKAGE / "sealed" / "server_mapping.json"),
            "--host",
            "127.0.0.1",
            "--port",
            "8800",
            "--reviewer-session-id",
            SESSION,
        ],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    browser = None
    socket = None
    try:
        for _ in range(200):
            try:
                if requests.get(URL, timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.25)
        else:
            raise RuntimeError("review server did not start")
        browser = subprocess.Popen(
            [
                str(EDGE),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--remote-debugging-port=9240",
                "--remote-allow-origins=*",
                "--no-first-run",
                "--no-default-browser-check",
                f"--user-data-dir={EDGE_PROFILE}",
                "--window-size=1440,900",
                URL,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        socket = websocket.create_connection(wait_for_page(), timeout=20)
        cdp = CDP(socket)
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        for _ in range(200):
            ready = cdp.evaluate(
                "document.readyState === 'complete' && document.body.dataset.presentation === 'gold_strand_annotation' && document.querySelector('#goldPitchImage')?.naturalWidth > 0"
            )
            if ready:
                break
            time.sleep(0.25)
        else:
            raise RuntimeError("gold annotation viewer did not load the pitch approval case")
        time.sleep(2.1)
        browser_manifest = requests.get(f"{URL}api/review/manifest", timeout=10).json()
        browser_ui = requests.get(f"{URL}api/review/ui-config", timeout=10).json()
        browser_payload_text = json.dumps({"manifest": browser_manifest, "ui": browser_ui}, sort_keys=True)
        forbidden_tokens = [
            "sealed_holdout",
            "development",
            "diagnostic",
            "source_row_hash",
            "internal_sequence_id",
            "expected_answer",
            "MHSAG_PRIMARY_CANDIDATE",
        ]
        pitch_state = cdp.evaluate(
            """(() => ({
              presentation: document.body.dataset.presentation,
              caseCount: manifest.cases.length,
              pitchVisible: !document.querySelector('#goldPitchPanel').classList.contains('isHidden'),
              annotationLocked: document.querySelector('#goldAnnotationPanel').classList.contains('isHidden'),
              polygonVertexCount: document.querySelectorAll('#goldPitchSvg .goldVertex').length,
              sampleFootpointCount: document.querySelectorAll('#goldPitchSvg .goldPitchSample').length,
              activeSeconds: activeTimeNow()
            }))()"""
        )
        pitch_shot = OUT / "pitch_approval_ui.png"
        screenshot(cdp, pitch_shot)
        cdp.evaluate("document.querySelector('#goldApprovePolygon').click()")
        for _ in range(100):
            if cdp.evaluate("!document.querySelector('#goldAnnotationPanel').classList.contains('isHidden')"):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("pitch approval did not unlock sequence annotation")
        time.sleep(0.8)
        cdp.key(" ", "Space")
        time.sleep(0.3)
        after_space = cdp.evaluate(
            "({frameIndex: goldFrameIndex, a: document.querySelector('#goldAState').textContent, b: document.querySelector('#goldBState').textContent})"
        )
        cdp.key("a", "KeyA")
        cdp.key("1", "Digit1")
        state_after_missing = cdp.evaluate("document.querySelector('#goldAState').textContent")
        cdp.key("z", "KeyZ", modifiers=2)
        state_after_undo = cdp.evaluate("document.querySelector('#goldAState').textContent")
        manual_bbox = cdp.evaluate(
            """(() => {
              document.querySelector('#goldDrawManual').click();
              const svg = document.querySelector('#goldDetectionSvg');
              const r = svg.getBoundingClientRect();
              svg.dispatchEvent(new PointerEvent('pointerdown', {bubbles:true, clientX:r.left+r.width*0.42, clientY:r.top+r.height*0.30}));
              svg.dispatchEvent(new PointerEvent('pointerup', {bubbles:true, clientX:r.left+r.width*0.48, clientY:r.top+r.height*0.72}));
              return goldAnnotation(goldCase(), goldRecord().frame_sequence).A;
            })()"""
        )
        cdp.key("z", "KeyZ", modifiers=2)
        cdp.key("Enter", "Enter")
        time.sleep(0.5)
        annotation_state = cdp.evaluate(
            "({saveEnabled: !document.querySelector('#goldSaveSequence').disabled, progress: document.querySelector('#goldFrameProgress').textContent, runPercent: document.querySelector('#goldRunAccepted').textContent})"
        )
        annotation_shot = OUT / "gold_frame_annotation_ui.png"
        screenshot(cdp, annotation_shot)
        cdp.evaluate("document.querySelector('#goldSaveSequence').click()")
        time.sleep(1.0)
        smoke_state_after_ui = requests.get(f"{URL}api/review/state", timeout=10).json()
        sealed_response = requests.get(f"{URL}sealed/server_mapping.json", timeout=10)
        completion = completion_smoke(browser_manifest)
        package_state = json.loads((PACKAGE / "decisions" / "review_decisions.json").read_text(encoding="utf-8"))
        composite = OUT / "gold_annotation_ui.png"
        make_composite(pitch_shot, annotation_shot, composite)
        result = {
            "real_browser": True,
            "url": URL,
            "reviewer_session_id": SESSION,
            "pitch_state": pitch_state,
            "keyboard_space_accepts_and_advances": after_space["frameIndex"] == 1,
            "keyboard_missing_state": state_after_missing,
            "keyboard_undo_changed_state": state_after_undo != state_after_missing,
            "manual_bbox_original_pixels_stored": (
                isinstance(manual_bbox, dict)
                and manual_bbox.get("state") == "OBSERVED_MANUAL_BBOX"
                and manual_bbox.get("bbox_original_pixels", {}).get("x2", 0)
                > manual_bbox.get("bbox_original_pixels", {}).get("x1", 0)
            ),
            "run_acceptance": annotation_state,
            "ui_saved_decision_count": len(smoke_state_after_ui.get("decisions", {})),
            "active_time_nonzero": int(smoke_state_after_ui.get("elapsed_active_seconds", 0)) > 0,
            "sealed_mapping_http_status": sealed_response.status_code,
            "sealed_mapping_inaccessible": sealed_response.status_code == 404,
            "forbidden_browser_payload_hits": [token for token in forbidden_tokens if token in browser_payload_text],
            "completion": completion,
            "package_decisions_remain_empty": not package_state.get("decisions"),
            "screenshots": [pitch_shot.name, annotation_shot.name, composite.name],
        }
        result["passed"] = all(
            (
                result["pitch_state"]["pitchVisible"],
                result["pitch_state"]["annotationLocked"],
                result["pitch_state"]["polygonVertexCount"] >= 4,
                result["keyboard_space_accepts_and_advances"],
                result["keyboard_undo_changed_state"],
                result["manual_bbox_original_pixels_stored"],
                result["run_acceptance"]["saveEnabled"],
                result["active_time_nonzero"],
                result["sealed_mapping_inaccessible"],
                not result["forbidden_browser_payload_hits"],
                result["completion"]["validation"]["passed"],
                result["completion"]["idempotent_retry_preserved_all_artifact_hashes"],
                result["package_decisions_remain_empty"],
            )
        )
        (OUT / "browser_validation.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        annotation_root = STAGE / "06_GOLD_ANNOTATION_UI_AND_SCHEMA"
        completion_report = {
            "status": "PASSED" if result["completion"]["validation"]["passed"] else "FAILED",
            **result["completion"],
        }
        (annotation_root / "completion_export_browser_test.json").write_text(
            json.dumps(completion_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        accessibility = {
            "status": "PASSED" if result["passed"] else "FAILED",
            "keyboard_space": result["keyboard_space_accepts_and_advances"],
            "keyboard_a_and_1": state_after_missing,
            "keyboard_ctrl_z": result["keyboard_undo_changed_state"],
            "keyboard_enter_run": result["run_acceptance"],
            "manual_bbox_original_pixels": result["manual_bbox_original_pixels_stored"],
            "sealed_mapping_inaccessible": result["sealed_mapping_inaccessible"],
            "forbidden_browser_payload_hits": result["forbidden_browser_payload_hits"],
        }
        (annotation_root / "accessibility_and_keyboard_results.json").write_text(
            json.dumps(accessibility, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not result["passed"]:
            raise RuntimeError(f"browser acceptance failed: {result}")
        print(json.dumps(result, indent=2))
    finally:
        if socket is not None:
            socket.close()
        if browser is not None:
            stop_tree(browser)
        stop_tree(server)


if __name__ == "__main__":
    main()
