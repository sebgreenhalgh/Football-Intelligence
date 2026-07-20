"""Exercise the fresh challenge package in a real Edge browser and temporary decisions root."""

# ruff: noqa: E501

from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
import websocket
from PIL import Image

from football_intelligence.review_chassis.hashing import sha256_file


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 2"
    / "M5_5F1E_SPENT_HOLDOUT_FORENSICS_ORACLE_REACHABILITY_INVARIANTS_AND_FRESH_CHALLENGE_GOLD_ACQUISITION_v1"
)
PACKAGE = STAGE / "10_FRESH_CHALLENGE_GOLD_ANNOTATION_PACKAGE"
TMP = STAGE / "_tmp" / "browser_persistence_exercise"
DECISIONS = TMP / "decisions"
OUT = STAGE / "11_MACHINE_PREFLIGHT_AND_BROWSER_VALIDATION" / "browser_evidence"
URL = "http://127.0.0.1:8806/"
CDP_URL = "http://127.0.0.1:9266"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
UV = shutil.which("uv") or "uv"
SESSION = "m5_5f1e_fresh_challenge_gold_annotator"


class CDP:
    def __init__(self, socket: websocket.WebSocket):
        self.socket = socket
        self.counter = 0

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
        return result.get("result", {}).get("value")


def stop_tree(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    subprocess.run(
        ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def start_server() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
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
            "--sealed-mapping",
            str(PACKAGE / "sealed" / "server_mapping.json"),
            "--polygon-sidecar-root",
            str(DECISIONS / "polygon"),
            "--reviewer-session-id",
            SESSION,
            "--host",
            "127.0.0.1",
            "--port",
            "8806",
        ],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def wait_server() -> None:
    for _ in range(180):
        try:
            response = requests.get(URL + "api/review/state", timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.2)
    raise RuntimeError("port-8806 review server did not start")


def wait_page() -> str:
    for _ in range(180):
        try:
            pages = requests.get(f"{CDP_URL}/json", timeout=1).json()
            for page in pages:
                if page.get("type") == "page" and str(page.get("url", "")).startswith(URL):
                    return str(page["webSocketDebuggerUrl"])
        except (requests.RequestException, ValueError):
            pass
        time.sleep(0.2)
    raise RuntimeError("Edge CDP page did not start")


def wait_ready(cdp: CDP, *, minimum_sequence: int = 0) -> dict[str, Any]:
    expression = f"""(async () => {{
      for (let i = 0; i < 240; i += 1) {{
        const d = window.__goldPersistenceDiagnostics || {{}};
        if (document.body.classList.contains('goldPresentation')
            && Number(d.serverSequence || 0) >= {minimum_sequence}
            && Number(d.pending?.length || 0) === 0) break;
        await new Promise(resolve => setTimeout(resolve, 100));
      }}
      await document.fonts.ready;
      await Promise.all([...document.images].map(async image => {{ try {{ await image.decode(); }} catch (_) {{}} }}));
      await new Promise(requestAnimationFrame); await new Promise(requestAnimationFrame);
      const d = window.__goldPersistenceDiagnostics || {{}};
      const rect = node => {{ const value = node?.getBoundingClientRect(); return value ? {{x:value.x,y:value.y,width:value.width,height:value.height}} : null; }};
      return {{
        presentation: document.body.classList.contains('goldPresentation'),
        status: document.querySelector('#goldPersistenceStatus')?.textContent,
        serverSequence: Number(d.serverSequence || 0), pending: Number(d.pending?.length || 0),
        caseTitle: document.querySelector('#goldCaseTitle')?.textContent,
        seedVisible: !document.querySelector('#goldSeedPanel')?.classList.contains('isHidden'),
        annotationVisible: !document.querySelector('#goldAnnotationPanel')?.classList.contains('isHidden'),
        characteristics: document.querySelector('#goldChallengeCharacteristics')?.textContent || '',
        naturalWidth: document.querySelector('#goldSeedCurrentImage')?.naturalWidth || document.querySelector('#goldCurrentImage')?.naturalWidth || 0,
        naturalHeight: document.querySelector('#goldSeedCurrentImage')?.naturalHeight || document.querySelector('#goldCurrentImage')?.naturalHeight || 0,
        seedLabels: [...document.querySelectorAll('#goldSeedSvg .goldDetectionLabel')].map(node => node.textContent),
        seedImageRect: rect(document.querySelector('#goldSeedCurrentImage')),
        seedSvgRect: rect(document.querySelector('#goldSeedSvg')),
        horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth + 1,
        completeDisabled: document.querySelector('#goldComplete')?.disabled ?? true,
        saveDisabled: document.querySelector('#goldSaveSequence')?.disabled ?? true,
        runDialogOpen: document.querySelector('#goldRunDialog')?.open ?? false,
        runContactCount: document.querySelectorAll('#goldRunContactStrip figure').length,
        runSummary: document.querySelector('#goldRunSummary')?.textContent || '',
      }};
    }})()"""
    value = None
    for _ in range(8):
        try:
            value = cdp.evaluate(expression)
            break
        except RuntimeError as exc:
            if "Execution context was destroyed" not in str(exc):
                raise
            time.sleep(0.5)
    if not value or not value.get("presentation"):
        raise RuntimeError(f"fresh gold package did not load: {value}")
    if value["serverSequence"] < minimum_sequence or value["pending"]:
        raise RuntimeError(
            "gold persistence did not reach the required acknowledged state: "
            f"minimum_sequence={minimum_sequence}, observed={value}"
        )
    return value


def screenshot(cdp: CDP, path: Path) -> None:
    payload = cdp.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(payload["data"]))
    with Image.open(path) as image:
        if image.width < 1200 or image.height < 700:
            raise RuntimeError(f"browser screenshot unexpectedly small: {image.size}")


def evidence_route_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    assets = []
    for case in manifest["cases"]:
        sequence_assets = [asset for asset in case["evidence_assets"] if asset["asset_type"] == "image_sequence"]
        if sequence_assets:
            assets.extend((case["case_id"], asset) for asset in (sequence_assets[0], sequence_assets[-1]))
        elif case["evidence_assets"]:
            assets.append((case["case_id"], case["evidence_assets"][0]))
    rows = []
    for case_id, asset in assets:
        response = requests.get(URL + f"evidence/{case_id}/{asset['relative_path']}", timeout=20)
        with Image.open(io.BytesIO(response.content)) as image:
            dimensions = image.size
        rows.append(
            {
                "case_id": case_id,
                "asset_id": asset["asset_id"],
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type"),
                "content_length": len(response.content),
                "dimensions": list(dimensions),
                "hash_match": sha256_file(PACKAGE / "evidence" / case_id / asset["relative_path"]) == asset["sha256"],
            }
        )
    passed = all(
        row["status_code"] == 200
        and str(row["content_type"]).startswith("image/")
        and row["content_length"] > 0
        and row["dimensions"][0] > 0
        and row["dimensions"][1] > 0
        and row["hash_match"]
        for row in rows
    )
    return {"sample_count": len(rows), "rows": rows, "passed": passed}


def launcher_contract_audit() -> dict[str, Any]:
    launcher = (PACKAGE / "launch_review.ps1").read_text(encoding="utf-8")
    required = (
        "--manifest",
        "--ui-config",
        "--evidence-root",
        "--decisions-root",
        "--sealed-mapping",
        "--polygon-sidecar-root",
        "--reviewer-session-id m5_5f1e_fresh_challenge_gold_annotator",
        "--port $Port",
    )
    missing = [value for value in required if value not in launcher]
    return {"required_argument_count": len(required), "missing_arguments": missing, "passed": not missing}


def main() -> None:
    if not EDGE.is_file():
        raise RuntimeError("Microsoft Edge is required for production browser validation")
    if TMP.exists():
        raise RuntimeError(f"refusing to reuse browser exercise root: {TMP}")
    TMP.mkdir(parents=True)
    DECISIONS.mkdir(parents=True)
    shutil.copytree(PACKAGE / "decisions" / "polygon", DECISIONS / "polygon", dirs_exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PACKAGE / "reviewer_manifest.json").read_text(encoding="utf-8"))
    real_state_before = sha256_file(PACKAGE / "decisions" / "review_decisions.json")
    server: subprocess.Popen[bytes] | None = None
    edge: subprocess.Popen[bytes] | None = None
    socket: websocket.WebSocket | None = None
    result: dict[str, Any] = {"url": URL, "package": str(PACKAGE), "tests": {}}
    try:
        server = start_server()
        wait_server()
        sealed_response = requests.get(URL + "sealed/server_mapping.json", timeout=5)
        result["tests"]["sealed_mapping_access"] = {
            "status_code": sealed_response.status_code,
            "accessible": sealed_response.status_code == 200,
        }
        result["tests"]["launcher_contract"] = launcher_contract_audit()
        result["tests"]["evidence_routes"] = evidence_route_audit(manifest)
        edge = subprocess.Popen(
            [
                str(EDGE),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--remote-allow-origins=*",
                "--remote-debugging-port=9266",
                "--window-size=1440,900",
                f"--user-data-dir={TMP / 'edge_profile'}",
                URL,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        socket = websocket.create_connection(wait_page(), timeout=30)
        cdp = CDP(socket)
        cdp.command("Page.enable")
        cdp.command("Runtime.enable")
        cdp.command(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False},
        )
        initial = wait_ready(cdp)
        result["tests"]["initial_seed_screen"] = initial
        if (
            not initial["seedVisible"]
            or initial["annotationVisible"]
            or initial["naturalWidth"] != 2730
            or initial["naturalHeight"] != 720
            or not {"A", "B"}.issubset(initial["seedLabels"])
            or not initial["characteristics"]
            or initial["horizontalOverflow"]
            or not initial["completeDisabled"]
        ):
            raise RuntimeError(f"initial production seed screen failed: {initial}")
        screenshot(cdp, OUT / "fresh_challenge_annotation_ui.png")
        cdp.evaluate("document.querySelector('#goldSeedConfirm').click(); true")
        seeded = wait_ready(cdp, minimum_sequence=1)
        result["tests"]["seed_ack"] = seeded
        cdp.evaluate("document.querySelector('#goldAcceptFrame').click(); true")
        first_pair = wait_ready(cdp, minimum_sequence=seeded["serverSequence"] + 2)
        result["tests"]["first_pair_ack"] = first_pair
        cdp.evaluate("document.querySelector('#goldAcceptRun').click(); true")
        time.sleep(0.8)
        run_preview = wait_ready(cdp, minimum_sequence=first_pair["serverSequence"])
        result["tests"]["stable_run_preview"] = run_preview
        if not run_preview["runDialogOpen"] or run_preview["runContactCount"] != 16:
            raise RuntimeError(f"stable-run preview omitted frames: {run_preview}")
        screenshot(cdp, OUT / "stable_run_contact_strip.png")
        cdp.evaluate("document.querySelector('#goldRunConfirm').click(); true")
        stable = wait_ready(
            cdp,
            minimum_sequence=(first_pair["serverSequence"] + run_preview["runContactCount"] * 2 + 1),
        )
        result["tests"]["stable_run_ack"] = stable
        cdp.evaluate("document.querySelector('[data-gold-state=AMBIGUOUS]').click(); true")
        corrected = wait_ready(cdp, minimum_sequence=stable["serverSequence"] + 1)
        result["tests"]["correction_ack"] = corrected
        cdp.evaluate("localStorage.clear(); location.reload(); true")
        time.sleep(1)
        reloaded = wait_ready(cdp, minimum_sequence=corrected["serverSequence"])
        result["tests"]["reload_recovery"] = reloaded
        stop_tree(server)
        server = None
        server = start_server()
        wait_server()
        cdp.evaluate("location.reload(); true")
        time.sleep(1)
        restarted = wait_ready(cdp, minimum_sequence=reloaded["serverSequence"])
        result["tests"]["server_restart_recovery"] = restarted
        cdp.evaluate("document.querySelector('#goldSaveSequence').click(); true")
        finalized = wait_ready(cdp, minimum_sequence=restarted["serverSequence"] + 1)
        result["tests"]["sequence_finalization"] = finalized
        state = requests.get(URL + "api/review/state", timeout=5).json()
        events = [
            json.loads(line)
            for line in (DECISIONS / "review_decision_events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        stable_event = next(event for event in events if event.get("event_type") == "STABLE_RUN_ACCEPTED")
        explicit = [
            event
            for event in events
            if event.get("event_type") == "FRAME_STATE_SET"
            and int(event.get("frame", -1)) in set(stable_event["payload"]["frames"])
            and int(event.get("server_event_sequence", event.get("event_sequence", 0)))
            < int(stable_event.get("server_event_sequence", stable_event.get("event_sequence", 0)))
        ]
        result["tests"]["event_ledger"] = {
            "event_count": len(events),
            "server_event_sequence": state["server_sequence"],
            "stable_run_frame_count": stable_event["payload"]["frame_count"],
            "stable_run_explicit_event_count": len(explicit),
            "sequence_finalized_count": state["completion_eligibility"]["sequences_finalized"],
        }
        result["tests"]["real_decisions_root"] = {
            "hash_before": real_state_before,
            "hash_after": sha256_file(PACKAGE / "decisions" / "review_decisions.json"),
            "untouched": real_state_before == sha256_file(PACKAGE / "decisions" / "review_decisions.json"),
        }
        result["passed"] = (
            result["tests"]["sealed_mapping_access"]["accessible"] is False
            and result["tests"]["launcher_contract"]["passed"] is True
            and result["tests"]["evidence_routes"]["passed"] is True
            and seeded["status"] == "Saved to server"
            and stable["pending"] == 0
            and reloaded["serverSequence"] == corrected["serverSequence"]
            and restarted["serverSequence"] == reloaded["serverSequence"]
            and state["completion_eligibility"]["sequences_finalized"] == 1
            and len(explicit) == stable_event["payload"]["frame_count"] * 2
            and result["tests"]["real_decisions_root"]["untouched"]
        )
        if not result["passed"]:
            raise RuntimeError(f"browser persistence acceptance failed: {result}")
    finally:
        if socket is not None:
            socket.close()
        stop_tree(edge)
        stop_tree(server)
        report_path = STAGE / "11_MACHINE_PREFLIGHT_AND_BROWSER_VALIDATION" / "browser_visual_regression.json"
        report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        exercise_path = STAGE / "11_MACHINE_PREFLIGHT_AND_BROWSER_VALIDATION" / "production_persistence_exercise.json"
        if exercise_path.is_file():
            exercise = json.loads(exercise_path.read_text(encoding="utf-8"))
            direct_passed = bool(
                exercise.get(
                    "direct_persistence_passed",
                    all(
                        (
                            exercise.get("all_acknowledged"),
                            exercise.get("event_ledger_nonempty"),
                            exercise.get("pitch_polygon_migrated"),
                            exercise.get("real_package_decisions_root_untouched"),
                            exercise.get("reload_state_hash_preserved"),
                            exercise.get("sequence_finalized"),
                            exercise.get("server_restart_state_hash_preserved"),
                            exercise.get("stable_run_explicit_frame_event_count")
                            == exercise.get("stable_run_expected_frame_event_count"),
                        )
                    ),
                )
            )
            exercise["direct_persistence_passed"] = direct_passed
            exercise["browser_http_exercise_pending"] = not bool(result.get("passed"))
            exercise["browser_http_exercise_passed"] = bool(result.get("passed"))
            exercise["browser_report_path"] = str(report_path)
            exercise["passed"] = direct_passed and bool(result.get("passed"))
            exercise_path.write_text(json.dumps(exercise, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
