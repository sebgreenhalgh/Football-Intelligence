"""Run real-browser acceptance for M5.5G.6B against an isolated decisions root."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests

import capture_m5_5g1a_browser_acceptance as base
import capture_m5_5g1a_r3_r1_browser_acceptance as r1
from build_m5_5g6b_boundary_gold import PACKAGE, REVIEWER, REVIEW_ID, STAGE, read_json, tree_manifest, write_json
from football_intelligence.detection_gold.persistence import DetectionGoldPilotPersistence
from football_intelligence.review_chassis.config import load_ui_config
from football_intelligence.review_chassis.manifest import load_manifest

OUT = STAGE / "06_BROWSER_PERSISTENCE_AND_COMPLETION"
RUN_ID = uuid.uuid4().hex[:10]
TMP = STAGE / "_tmp" / f"g6b_browser_acceptance_{RUN_ID}"
DECISIONS = TMP / "decisions"
PROFILE = Path(tempfile.gettempdir()) / f"m5g6b_edge_{RUN_ID}"
URL = "http://127.0.0.1:8810/"
PORT = 8810
CDP_PORT = 11040 + (int(RUN_ID[:4], 16) % 250)
UV = r"C:\Users\sebgr\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
SCREENSHOTS = (
    OUT / "01_TARGET_ONLY_FOCAL_AND_PANORAMA.png",
    OUT / "02_BOUNDARY_UNCERTAIN_HIDDEN_FEET.png",
    OUT / "03_TARGET_REVIEW_BEFORE_SAVE.png",
)


def configure_base() -> None:
    base.STAGE = STAGE
    base.PACKAGE = PACKAGE
    base.PRODUCTION_DECISIONS = PACKAGE / "decisions"
    base.OUT = OUT
    base.DECISIONS = DECISIONS
    base.PROFILE = PROFILE
    base.SESSION = REVIEWER
    base.REVIEW_ID = REVIEW_ID
    base.CDP_PORT = CDP_PORT
    base.RUN_ID = RUN_ID
    base.TMP = TMP
    base.URL = URL
    base.UV = UV
    base.ACTIVE_PROCESSES.clear()
    r1.OUT = OUT
    r1.URL = URL
    r1.CDP_PORT = CDP_PORT


def start_server() -> subprocess.Popen[bytes]:
    if base.port_open(PORT):
        raise RuntimeError("port 8810 is occupied; exact-package browser validation cannot move ports")
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
            str(PORT),
            "--reviewer-session-id",
            REVIEWER,
        ],
        cwd=base.REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base.ACTIVE_PROCESSES.append(process)
    base.wait_server(process)
    return process


def click(cdp: base.CDP, selector: str) -> None:
    clicked = cdp.evaluate(
        f"""(() => {{
          const node = document.querySelector({json.dumps(selector)});
          if (!node || node.disabled) return false;
          node.click();
          return true;
        }})()"""
    )
    if clicked is not True:
        raise RuntimeError(f"missing or disabled browser control: {selector}")
    time.sleep(0.2)


def wait_for(cdp: base.CDP, expression: str, timeout: float = 20) -> Any:
    return base.wait_for(cdp, expression, timeout_seconds=timeout)


def reload(cdp: base.CDP) -> None:
    cdp.command("Page.navigate", {"url": f"{URL}?reload={uuid.uuid4().hex}"})
    time.sleep(1.0)
    base.wait_ready(cdp)


def viewport_audit(cdp: base.CDP, profile: dict[str, Any]) -> dict[str, Any]:
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
    time.sleep(0.25)
    base.wait_ready(cdp)
    result = cdp.evaluate(
        """(() => {
          const image = document.querySelector('#dgBaseImage');
          const overlay = document.querySelector('#dgOverlay');
          const imageRect = image?.getBoundingClientRect();
          const overlayRect = overlay?.getBoundingClientRect();
          return {
            bodyHorizontalOverflowPixels: Math.max(0, document.documentElement.scrollWidth - innerWidth),
            imageNaturalWidth: image?.naturalWidth || 0,
            imageNaturalHeight: image?.naturalHeight || 0,
            imageOverlayMaxDelta: imageRect && overlayRect ? Math.max(
              Math.abs(imageRect.left - overlayRect.left), Math.abs(imageRect.top - overlayRect.top),
              Math.abs(imageRect.right - overlayRect.right), Math.abs(imageRect.bottom - overlayRect.bottom)
            ) : 999,
            evidenceBlocked: !document.querySelector('#dgEvidenceBlocker')?.classList.contains('isHidden'),
            targetCount: document.querySelectorAll('.dgBoundaryTarget').length,
            targetLabelCount: [...document.querySelectorAll('#dgOverlay text')]
              .filter(node => node.textContent === 'TARGET').length,
            targetCopyPresent: document.body.innerText.includes(
              'Label the highlighted target person only. Other people are context.'
            ),
            completePilotDisabled: document.querySelector('#dgComplete')?.disabled ?? false,
          };
        })()"""
    )
    result["name"] = profile["name"]
    result["passed"] = (
        result["bodyHorizontalOverflowPixels"] <= 1
        and result["imageNaturalWidth"] > 0
        and result["imageNaturalHeight"] > 0
        and result["imageOverlayMaxDelta"] <= 1
        and not result["evidenceBlocked"]
        and result["targetCount"] == 1
        and result["targetLabelCount"] == 1
        and result["targetCopyPresent"]
        and result["completePilotDisabled"]
    )
    return result


def state_counts() -> dict[str, Any]:
    return requests.get(URL + "api/review/state", timeout=10).json()["counts"]


def main() -> None:
    configure_base()
    TMP.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(PACKAGE / "reviewer_manifest.json")
    ui = load_ui_config(PACKAGE / "ui_config.json")
    DetectionGoldPilotPersistence(
        manifest=manifest, ui_config=ui, decisions_root=DECISIONS, reviewer_session_id=REVIEWER
    ).ensure_state()
    production_before = tree_manifest(PACKAGE / "decisions")
    server: subprocess.Popen[bytes] | None = None
    edge: subprocess.Popen[bytes] | None = None
    cdp: base.CDP | None = None
    try:
        server = start_server()
        edge = base.start_edge(CDP_PORT)
        cdp = base.connect_page(CDP_PORT)
        cdp.socket.settimeout(60)
        ready = base.wait_ready(cdp)
        if cdp.evaluate("!document.querySelector('#nwTour')?.classList.contains('isHidden')"):
            click(cdp, "#nwTourStart")
        profiles = [viewport_audit(cdp, profile) for profile in r1.VIEWPORTS]
        viewport_audit(cdp, next(profile for profile in r1.VIEWPORTS if profile["name"] == "1440x900"))
        screenshot_rows = [base.capture(cdp, SCREENSHOTS[0])]
        initial = cdp.evaluate(
            """(() => ({
              step: Number(document.querySelector('.nwWizard')?.dataset.nwStep || 0),
              targetCount: document.querySelectorAll('.dgBoundaryTarget').length,
              targetCopy: document.body.innerText.includes(
                'Label the highlighted target person only. Other people are context.'
              ),
              crowdDrawButtonAbsent: !document.querySelector('#nwDrawObject'),
            }))()"""
        )
        click(cdp, "#nwConfirmBoundaryTarget")
        click(cdp, '[data-nw-answer-key="c2_role"][data-nw-answer-value="PLAYER"]')
        click(cdp, '[data-nw-answer-key="c2_footpoint_status"][data-nw-answer-value="FEET_NOT_VISIBLE"]')
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '3'")
        click(cdp, '[data-nw-answer-key="c2_pitch"][data-nw-answer-value="BOUNDARY_UNCERTAIN"]')
        wait_for(cdp, "Boolean(document.querySelector('[data-nw-answer-key=\"c2_pitch_certainty\"]'))")
        screenshot_rows.append(base.capture(cdp, SCREENSHOTS[1]))
        click(cdp, '[data-nw-answer-key="c2_pitch_certainty"][data-nw-answer-value="UNCERTAIN"]')
        wait_for(cdp, "Boolean(document.querySelector('[data-nw-answer-key=\"candidate_relation\"]'))")
        click(cdp, '[data-nw-answer-key="candidate_relation"][data-nw-answer-value="CLEAN_SINGLE_INSTANCE"]')
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '4'")
        screenshot_rows.append(base.capture(cdp, SCREENSHOTS[2]))
        before_reload = cdp.evaluate(
            """(() => ({
              step: Number(document.querySelector('.nwWizard')?.dataset.nwStep || 0),
              summaryHasTarget: document.querySelector('.nwReviewCard')?.innerText.includes('TARGET') || false,
              summaryHasBoundaryUncertain: document.querySelector('.nwReviewCard')?.innerText
                .toLowerCase().includes('boundary uncertain') || false,
              saveEnabled: document.querySelector('#nwSaveCase')?.disabled === false,
            }))()"""
        )
        reload(cdp)
        wait_for(cdp, "document.querySelector('.nwWizard')?.dataset.nwStep === '4'")
        reload_recovered = cdp.evaluate(
            "document.querySelector('.nwReviewCard')?.innerText.toLowerCase().includes('boundary uncertain') || false"
        )
        base.stop_tree(server)
        server = None
        wait_for(cdp, "document.querySelector('#nwSaveCase')?.disabled === false")
        click(cdp, "#nwSaveCase")
        time.sleep(1.2)
        offline = cdp.evaluate(
            """(() => ({
              saveState: document.querySelector('#dgSaveState')?.textContent || '',
              serverState: document.querySelector('#dgServerState')?.textContent || '',
              bodyText: document.body.innerText.slice(0, 2000),
            }))()"""
        )
        server = start_server()
        cdp.close()
        cdp = None
        base.stop_tree(edge)
        edge = base.start_edge(CDP_PORT)
        cdp = base.connect_page(CDP_PORT)
        cdp.socket.settimeout(60)
        base.wait_ready(cdp)
        deadline = time.time() + 25
        counts = state_counts()
        while counts.get("reviewed") != 1 and time.time() < deadline:
            time.sleep(0.4)
            counts = state_counts()
        if counts.get("reviewed") != 1:
            reconnect_debug = cdp.evaluate(
                """(async () => {
                  const request = indexedDB.open('m5_5g6b_boundary_focused_gold_outbox_v1');
                  const database = await new Promise((resolve, reject) => {
                    request.onsuccess = () => resolve(request.result);
                    request.onerror = () => reject(request.error);
                  });
                  const records = await new Promise((resolve, reject) => {
                    const transaction = database.transaction('outbox', 'readonly');
                    const read = transaction.objectStore('outbox').getAll();
                    read.onsuccess = () => resolve(read.result);
                    read.onerror = () => reject(read.error);
                  });
                  return {
                    outboxCount: records.length,
                    saveState: document.querySelector('#dgSaveState')?.textContent || '',
                    serverState: document.querySelector('#dgServerState')?.textContent || '',
                    errorText: document.querySelector('#dgError')?.textContent || '',
                    legacyStatus: document.querySelector('#status')?.textContent || '',
                    firstEvent: records[0] || null,
                  };
                })()"""
            )
            write_json(TMP / "reconnect_failure_debug.json", reconnect_debug)
            raise RuntimeError(f"offline outbox did not materialize exactly once: {counts}; debug={reconnect_debug}")
        saved_state = read_json(DECISIONS / "review_decisions.json")
        annotation = next(iter(saved_state["annotations"].values()))
        person = annotation["player_instances"][0]
        base.stop_tree(server)
        server = None
        server = start_server()
        restart_counts = state_counts()
        production_after = tree_manifest(PACKAGE / "decisions")
        scenarios = {
            "target_only_annotation": initial["targetCount"] == 1 and initial["crowdDrawButtonAbsent"],
            "no_exhaustive_crowd_requirement": initial["crowdDrawButtonAbsent"],
            "target_copy_visible": initial["targetCopy"],
            "boundary_uncertain_save": person["pitch_state"] == "BOUNDARY_UNCERTAIN",
            "hidden_feet_boundary_uncertainty": person["footpoint_status"] == "FEET_NOT_VISIBLE",
            "target_proposal_relation": len(annotation["candidate_relations"]) == 1
            and annotation["candidate_relations"][0]["relation"] == "CLEAN_SINGLE_INSTANCE",
            "reload_draft_recovery": reload_recovered is True,
            "offline_outbox_recovery": "pending" in (offline["bodyText"] + offline["saveState"]).lower(),
            "server_restart_recovery": restart_counts.get("reviewed") == 1,
            "exactly_once_materialization": int(saved_state["event_sequence"]) == 1,
            "review_summary_target_label": before_reload["summaryHasTarget"],
            "review_summary_boundary_uncertain": before_reload["summaryHasBoundaryUncertain"],
            "save_enabled_after_complete_workflow": before_reload["saveEnabled"],
            "all_viewports_pass": all(row["passed"] for row in profiles),
            "prior_decisions_unchanged": production_before == production_after,
        }
        result = {
            "schema_version": "football_intelligence.m5_5g6b.browser_persistence_results.v1",
            "status": "PASS_REAL_BROWSER_ACCEPTANCE",
            "url": URL,
            "ready_state": ready,
            "viewport_results": profiles,
            "required_scenarios": scenarios,
            "offline_state": offline,
            "server_counts_after_recovery": counts,
            "server_counts_after_restart": restart_counts,
            "saved_person_state": {
                "pitch_state": person["pitch_state"],
                "pitch_state_certainty": person["pitch_state_certainty"],
                "footpoint_status": person["footpoint_status"],
                "coarse_role": person["coarse_role"],
            },
            "screenshot_rows": screenshot_rows,
            "temporary_decisions_tree": tree_manifest(DECISIONS),
            "production_decisions_unchanged": production_before == production_after,
            "passed": all(scenarios.values()) and all(row["passed"] for row in profiles),
        }
        write_json(OUT / "browser_persistence_results.json", result)
        if not result["passed"]:
            raise RuntimeError(f"FAIL_PERSISTENCE: {scenarios}")
    finally:
        if cdp is not None:
            cdp.close()
        base.stop_tree(edge)
        base.stop_tree(server)


if __name__ == "__main__":
    main()
