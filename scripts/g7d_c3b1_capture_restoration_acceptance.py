from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
import websocket

# Exact external paths and acceptance expressions are intentionally explicit.
# ruff: noqa: E501

ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
STAGE = (
    ROOT
    / "experiments/football_observation_reasoner/part 7/G7D_C3B1_NESTED_REVIEW_FINALIZATION_AND_SAFE_RULE_SELECTION_v1"
)
PACKAGE = (
    ROOT
    / "experiments/football_observation_reasoner/part 7/G7D_C3B_NESTED_CANDIDATE_SUPPRESSION_SANDBOX_REVIEW_v1/06_NESTED_REVIEW_PACKAGE"
)
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def request(path, body=None):
    req = urllib.request.Request(
        "http://127.0.0.1:8817" + path,
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body else "GET",
    )
    return json.loads(urllib.request.urlopen(req, timeout=5).read())


def start(decisions=None):
    env = dict(os.environ)
    if decisions:
        env["G7D_C3B_DECISIONS_ROOT"] = str(decisions)
    p = subprocess.Popen(
        [str(ROOT / "SoccerTrack-v2/.venv/Scripts/python.exe"), "review_server.py"],
        cwd=PACKAGE,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
    return p


def main():
    human = PACKAGE / "human_decisions"
    before = {str(p): p.read_bytes() for p in human.rglob("*.json")}
    server = start()
    profile = STAGE / "_edge_profile"
    edge = subprocess.Popen(
        [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--remote-debugging-port=9238",
            "--remote-allow-origins=*",
            f"--user-data-dir={profile}",
            "--window-size=1600,1000",
            "http://127.0.0.1:8817/",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wsurl = None
        for _ in range(60):
            try:
                pages = json.loads(urllib.request.urlopen("http://127.0.0.1:9238/json", timeout=1).read())
                wsurl = next(x["webSocketDebuggerUrl"] for x in pages if x["type"] == "page")
                break
            except Exception:
                time.sleep(0.25)
        ws = websocket.create_connection(wsurl, timeout=10)
        i = 1
        ws.send(json.dumps({"id": i, "method": "Page.enable"}))
        ws.recv()
        time.sleep(2)
        i += 1
        ws.send(json.dumps({"id": i, "method": "Page.captureScreenshot", "params": {"format": "png"}}))
        while True:
            x = json.loads(ws.recv())
            if x.get("id") == i:
                break
        out = STAGE / "06_VISUAL_QA/02_COMPLETION_RESTORATION_AND_DECISION.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(x["result"]["data"]))
        ws.close()
        state = request("/api/review-state")
        assert (
            state["completed"]
            and state["completed_count"] == 48
            and state["completion_receipt_id"] == "completion-37401efba568571a0f627ee5"
            and state["last_event_id"] == "6b7a55ca-0da7-4af9-b36f-7376ad901dd1"
        )
    finally:
        edge.terminate()
        server.terminate()
        edge.wait(timeout=10)
        server.wait(timeout=10)
    assert before == {str(p): p.read_bytes() for p in human.rglob("*.json")}
    temp = STAGE / "_temporary_restoration_decisions"
    server = start(temp)
    try:
        assert request("/api/review-state")["first_incomplete_case_id"] == "pair_01"
        answers = {str(i): "Not sure" for i in range(6)}
        for n in range(1, 4):
            request("/api/save", {"case_id": f"pair_{n:02d}", "answers": answers})
        assert request("/api/review-state")["first_incomplete_case_id"] == "pair_04"
        request("/api/draft", {"case_id": "pair_04", "answers": {"0": "Not sure"}})
        assert request("/api/review-state")["draft"]["answers"]["0"] == "Not sure"
        for n in range(4, 49):
            last = request("/api/save", {"case_id": f"pair_{n:02d}", "answers": answers})
        complete = request("/api/review-state")
        assert last["all_cases_complete"] and complete["completed"] and complete["completed_count"] == 48
        count = len(list((temp / "events").glob("*.json")))
        request("/api/review-state")
        assert len(list((temp / "events").glob("*.json"))) == count == 48
    finally:
        server.terminate()
        server.wait(timeout=10)
        if temp.is_dir():
            shutil.rmtree(temp)
        if profile.is_dir():
            shutil.rmtree(profile)
    report = {
        "browser": "Microsoft Edge via CDP",
        "actual_human_root_completed": True,
        "actual_human_files_mutated": False,
        "completed_count": 48,
        "completion_receipt_id": "completion-37401efba568571a0f627ee5",
        "last_event_id": "6b7a55ca-0da7-4af9-b36f-7376ad901dd1",
        "zero_case_first": "pair_01",
        "partial_first_incomplete": "pair_04",
        "compatible_draft_restored": True,
        "temporary_completion_restored": True,
        "refresh_created_events": 0,
        "temporary_data_removed": True,
    }
    p = STAGE / "01_EVENT_AND_RESTORATION_CLOSURE/restoration_acceptance.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS_G7D_C3B1_RESTORATION_ACCEPTANCE")


if __name__ == "__main__":
    main()
