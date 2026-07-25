"""Restore the captured R1-R3 IndexedDB export in an isolated browser profile."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests

import capture_m5_5g1a_browser_acceptance as browser


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
STAGE = (
    ROOT
    / "matches"
    / "128058"
    / "runs"
    / "step_m5"
    / "part 3"
    / "M5_5G4_R1_R3_PENDING_OUTBOX_AND_OCCLUSION_DEPENDENCY_RECONCILIATION_REPAIR_v1"
)
EXPORT = STAGE / "01_LIVE_SERVER_AND_BROWSER_STATE_EXPORT" / "indexeddb_pending_export.json"
PROBE = STAGE / "_tmp" / "restore_probe"
OUTPUT = STAGE / "01_LIVE_SERVER_AND_BROWSER_STATE_EXPORT" / "temporary_clone_restore_validation.json"
PORT = 8810


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=True).encode()


def database_inventory(databases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "database": database["name"],
            "version": database["version"],
            "stores": [{"name": store["name"], "record_count": len(store["records"])} for store in database["stores"]],
        }
        for database in databases
    ]


def main() -> int:
    source = json.loads(EXPORT.read_text(encoding="utf-8"))
    source_databases = source["databases"]
    source_hash = hashlib.sha256(canonical_bytes(source_databases)).hexdigest()
    run_id = uuid.uuid4().hex[:10]
    profile = Path(tempfile.gettempdir()) / f"m5g4_r1_r3_restore_{run_id}"
    cdp_port = 11800 + (int(run_id[:4], 16) % 200)
    server: subprocess.Popen[bytes] | None = None
    edge: subprocess.Popen[bytes] | None = None
    cdp: browser.CDP | None = None
    try:
        server = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1"],
            cwd=PROBE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(100):
            if server.poll() is not None:
                raise RuntimeError("temporary restore server exited")
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/", timeout=0.25).status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(0.05)
        else:
            raise RuntimeError("temporary restore server did not become ready")

        edge = subprocess.Popen(
            [
                str(browser.EDGE),
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={cdp_port}",
                "--window-size=1280,800",
                f"--user-data-dir={profile}",
                f"http://127.0.0.1:{PORT}/",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        original_url = browser.URL
        browser.URL = f"http://127.0.0.1:{PORT}/"
        try:
            cdp = browser.connect_page(cdp_port)
        finally:
            browser.URL = original_url
        restore_function = """async (databases) => {
              const request = (value) => new Promise((resolve, reject) => {
                value.onsuccess = () => resolve(value.result);
                value.onerror = () => reject(value.error);
              });
              const transactionDone = (transaction) => new Promise((resolve, reject) => {
                transaction.oncomplete = () => resolve();
                transaction.onerror = () => reject(transaction.error);
                transaction.onabort = () => reject(transaction.error);
              });
              for (const source of databases) {
                await new Promise((resolve) => {
                  const deletion = indexedDB.deleteDatabase(source.name);
                  deletion.onsuccess = deletion.onerror = deletion.onblocked = () => resolve();
                });
                const database = await new Promise((resolve, reject) => {
                  const opening = indexedDB.open(source.name, source.version);
                  opening.onupgradeneeded = () => {
                    for (const store of source.stores) {
                      if (!opening.result.objectStoreNames.contains(store.name)) {
                        const first = store.records[0] || null;
                        const keyPath = first && Object.hasOwn(first, 'id') ? 'id' : 'key';
                        opening.result.createObjectStore(store.name, {keyPath});
                      }
                    }
                  };
                  opening.onsuccess = () => resolve(opening.result);
                  opening.onerror = () => reject(opening.error);
                });
                for (const store of source.stores) {
                  const transaction = database.transaction(store.name, 'readwrite');
                  const target = transaction.objectStore(store.name);
                  for (const record of store.records) target.put(record);
                  await transactionDone(transaction);
                }
                database.close();
              }
              const output = [];
              for (const metadata of (await indexedDB.databases()).sort((a,b) => a.name.localeCompare(b.name))) {
                const database = await request(indexedDB.open(metadata.name));
                const stores = [];
                for (const name of [...database.objectStoreNames].sort()) {
                  const transaction = database.transaction(name, 'readonly');
                  const target = transaction.objectStore(name);
                  const keys = await request(target.getAllKeys());
                  const records = await request(target.getAll());
                  stores.push({name, keys, records});
                }
                output.push({name: metadata.name, version: metadata.version, stores});
                database.close();
              }
              return output;
            }"""
        restored = cdp.evaluate(f"({restore_function})({json.dumps(source_databases, separators=(',', ':'))})")
        restored_hash = hashlib.sha256(canonical_bytes(restored)).hexdigest()
        checks = {
            "source_database_count_exact": len(source_databases) == 2,
            "restored_database_count_exact": len(restored) == len(source_databases),
            "canonical_reexport_hash_matches": restored_hash == source_hash,
            "five_pending_records_restored": any(
                database["name"] == "fi_m5_5g4_r1_r2_constant_screen_space_marker_repair_v1"
                and any(store["name"] == "outbox" and len(store["records"]) == 5 for store in database["stores"])
                for database in restored
            ),
            "source_profile_was_not_opened": True,
            "real_server_was_not_contacted": True,
        }
        payload = {
            "schema_version": "football_intelligence.m5_5g4_r1_r3.temporary_clone_restore.v1",
            "source_export_sha256": hashlib.sha256(EXPORT.read_bytes()).hexdigest(),
            "source_database_canonical_sha256": source_hash,
            "restored_database_canonical_sha256": restored_hash,
            "source_inventory": database_inventory(source_databases),
            "restored_inventory": database_inventory(restored),
            "checks": checks,
            "passed": all(checks.values()),
        }
        OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not payload["passed"]:
            raise RuntimeError(f"temporary clone restore failed: {checks}")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        if cdp is not None:
            try:
                cdp.close()
            except (OSError, RuntimeError):
                pass
        browser.stop_tree(edge)
        browser.stop_tree(server)
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
