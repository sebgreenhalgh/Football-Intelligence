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

ROOT = Path(r"C:\Users\sebgr\Documents\football-intelligence")
STAGE = ROOT / "experiments/football_observation_reasoner/part 7/G7D_C3B_NESTED_CANDIDATE_SUPPRESSION_SANDBOX_REVIEW_v1"
PACKAGE = STAGE / "06_NESTED_REVIEW_PACKAGE"
VIS = STAGE / "07_VISUAL_QA"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


class CDP:
    def __init__(self, ws):
        self.ws = ws
        self.i = 0

    def cmd(self, m, p=None):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": m, "params": p or {}}))
        while True:
            x = json.loads(self.ws.recv())
            if x.get("id") == self.i:
                return x.get("result", {})

    def eval(self, s):
        return self.cmd("Runtime.evaluate", {"expression": s, "returnByValue": True, "awaitPromise": True})

    def shot(self, path):
        path.write_bytes(base64.b64decode(self.cmd("Page.captureScreenshot", {"format": "png"})["data"]))


def main():
    VIS.mkdir(parents=True, exist_ok=True)
    temp = STAGE / "_temporary_acceptance_decisions"
    env = {**os.environ, "G7D_C3B_DECISIONS_ROOT": str(temp)}
    server = subprocess.Popen(
        [str(ROOT / "SoccerTrack-v2/.venv/Scripts/python.exe"), "review_server.py"],
        cwd=PACKAGE,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    edge = subprocess.Popen(
        [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--remote-debugging-port=9237",
            "--remote-allow-origins=*",
            f"--user-data-dir={STAGE/'_edge_profile'}",
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
                pages = json.loads(urllib.request.urlopen("http://127.0.0.1:9237/json", timeout=1).read())
                wsurl = next(p["webSocketDebuggerUrl"] for p in pages if p["type"] == "page")
                break
            except Exception:
                time.sleep(0.25)
        c = CDP(websocket.create_connection(wsurl, timeout=10))
        c.cmd("Page.enable")
        time.sleep(2)
        c.shot(VIS / "01_NESTED_PAIR_REVIEW_READY.png")
        c.eval("i=4;q=0;answers={};render();true")
        time.sleep(2)
        c.shot(VIS / "02_LEGITIMATE_INNER_PROTECTION_PATH.png")
        assert c.eval("document.querySelectorAll('canvas').length===2 && cases.length===48")["result"]["value"]
        c.ws.close()
        cases = json.loads((PACKAGE / "cases.json").read_text())["cases"]
        for case in cases:
            body = json.dumps(
                {
                    "case_id": case["case_id"],
                    "answers": {
                        "0": "Not sure",
                        "1": "Not sure",
                        "2": "Not sure",
                        "3": "Not sure",
                        "4": "Not sure",
                        "5": "Not sure",
                    },
                }
            ).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:8817/api/save", data=body, headers={"Content-Type": "application/json"}
            )
            result = json.loads(urllib.request.urlopen(req).read())
        assert result["all_cases_complete"] and result["completion_receipt_id"]
        report = {
            "browser": "Microsoft Edge via Chrome DevTools Protocol",
            "server": "actual local reviewer on port 8817",
            "assets_loaded": 48,
            "temporary_events": 48,
            "temporary_completion": True,
            "human_root_synthetic_events": 0,
            "previews": 2,
        }
        (STAGE / "08_TESTS_AND_LOGS/live_edge_acceptance.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        edge.terminate()
        server.terminate()
        edge.wait(timeout=10)
        server.wait(timeout=10)
        if temp.is_dir():
            shutil.rmtree(temp)
        profile = STAGE / "_edge_profile"
        if profile.is_dir():
            shutil.rmtree(profile)
    print("PASS_G7D_C3B_LIVE_EDGE_ACCEPTANCE")


if __name__ == "__main__":
    main()
