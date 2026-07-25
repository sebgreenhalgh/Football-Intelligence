(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const SVG_NS = "http://www.w3.org/2000/svg";
  const GEOMETRY_EPSILON = 1e-6;
  const MINIMUM_POLYGON_AREA = 4;
  const RECOVERY_BUILD_ID = "m5_5g4_r1_r3_pending_dependency_repair_v1";
  const DEPENDENCY_HANDSHAKE_VERSION = "m5_5g4_r1_r3_server_authoritative_v1";
  const RECOVERY_MUTATION_CONTROL_IDS = Object.freeze([
    "dcRedraw",
    "dcFinish",
    "dcUndo",
    "dcClear",
    "dcQuality",
    "dcUnreliable",
    "dcUnreliableReason",
    "dcPreviousMask",
    "dcNextMask",
  ]);
  const MARKER_SCREEN_RADIUS_CSS = Object.freeze({
    vertex: 3.5,
    crossing: 4,
  });
  const COVERAGE = [
    ["", "Choose coverage"],
    ["0", "Almost none"],
    ["0.25", "About one quarter"],
    ["0.5", "About half"],
    ["0.75", "About three quarters"],
    ["1", "Almost all"],
  ];

  const runtime = {
    manifest: null,
    uiConfig: null,
    state: null,
    api: null,
    db: null,
    items: [],
    index: 0,
    candidateIndex: 0,
    coverageValues: {},
    points: [],
    preview: null,
    previewIssue: null,
    invalidPreview: false,
    drawing: false,
    evidenceBlocked: true,
    evidenceBindingValid: false,
    activeAsset: null,
    objectUrl: null,
    view: "focal",
    transform: {
      scale: 1,
      translateX: 0,
      translateY: 0,
      naturalWidth: 1,
      naturalHeight: 1,
      viewportWidth: 1,
      viewportHeight: 1,
    },
    pan: null,
    machineRendered: false,
    machineInViewport: false,
    machineExplicitlyFocused: false,
    pendingViewState: null,
    startedAt: performance.now(),
    recovery: {
      enabled: false,
      active: false,
      rows: [],
      index: 0,
      preflight: null,
      pairIndex: 0,
      pairAnswers: {},
      migration: null,
      draft: null,
      draftRepresented: false,
      quarantined: [],
    },
  };

  function uuid() {
    return crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
    })[character]);
  }

  function current() { return runtime.items[runtime.index]; }
  function caseData() { return current().caseData; }
  function item() { return current().item; }
  function binding() { return caseData().visible_metadata.source_binding; }
  function candidateRows() { return item().affected_candidates || []; }
  function activeCandidate() { return candidateRows()[runtime.candidateIndex] || null; }
  function draftKey() { return `${caseData().case_id}:${item().original_mask_uuid}`; }
  function elapsedSeconds() { return Math.max(0, Math.round((performance.now() - runtime.startedAt) / 1000)); }
  function recoveryMutationLocked() { return runtime.recovery.enabled && runtime.recovery.active; }

  function syncRecoveryMutationLock() {
    const locked = recoveryMutationLocked();
    RECOVERY_MUTATION_CONTROL_IDS.forEach((id) => { $(id).disabled = locked; });
    if (locked || !current()) return;
    const closure = runtime.points.length >= 3 ? validateClosingSegment(runtime.points) : null;
    $("dcFinish").disabled = !runtime.drawing
      || runtime.points.length < 3
      || !closure?.valid
      || runtime.invalidPreview;
    $("dcUnreliableReason").disabled = !$("dcUnreliable").checked;
    $("dcPreviousMask").disabled = runtime.index === 0;
    $("dcNextMask").disabled = runtime.index === runtime.items.length - 1;
  }

  function openNamedDatabase(name) {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(name, 1);
      request.onupgradeneeded = () => {
        const database = request.result;
        if (!database.objectStoreNames.contains("drafts")) database.createObjectStore("drafts", {keyPath: "key"});
        if (!database.objectStoreNames.contains("outbox")) database.createObjectStore("outbox", {keyPath: "id"});
        if (!database.objectStoreNames.contains("session")) database.createObjectStore("session", {keyPath: "key"});
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  function openDatabase() {
    return openNamedDatabase(runtime.uiConfig.question_contract.indexeddb_namespace);
  }

  function idb(mode, storeName, operation) {
    return new Promise((resolve, reject) => {
      const transaction = runtime.db.transaction(storeName, mode);
      const store = transaction.objectStore(storeName);
      const request = operation(store);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  const idbGet = (store, key) => idb("readonly", store, (objectStore) => objectStore.get(key));
  const idbPut = (store, value) => idb("readwrite", store, (objectStore) => objectStore.put(value));
  const idbDelete = (store, key) => idb("readwrite", store, (objectStore) => objectStore.delete(key));
  const idbAll = (store) => idb("readonly", store, (objectStore) => objectStore.getAll());

  function databaseRequest(request) {
    return new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  function transactionDone(transaction) {
    return new Promise((resolve, reject) => {
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
  }

  function stableValue(value) {
    if (Array.isArray(value)) return value.map(stableValue);
    if (value && typeof value === "object") return Object.fromEntries(
      Object.keys(value).sort().map((key) => [key, stableValue(value[key])]),
    );
    return value;
  }

  function stableJson(value) {
    return JSON.stringify(stableValue(value));
  }

  async function sha256Text(value) {
    return sha256(new TextEncoder().encode(value));
  }

  async function databaseSnapshot(database) {
    const stores = [];
    for (const name of Array.from(database.objectStoreNames).sort()) {
      const transaction = database.transaction(name, "readonly");
      const records = await databaseRequest(transaction.objectStore(name).getAll());
      stores.push({name, records});
    }
    return {name: database.name, version: database.version, stores};
  }

  async function importSnapshot(database, snapshot, migrationRecord = null) {
    const names = snapshot.stores.map((store) => store.name);
    const transaction = database.transaction(names, "readwrite");
    for (const source of snapshot.stores) {
      const target = transaction.objectStore(source.name);
      target.clear();
      source.records.forEach((record) => target.put(structuredClone(record)));
    }
    if (migrationRecord) transaction.objectStore("session").put(migrationRecord);
    await transactionDone(transaction);
  }

  async function deleteDatabase(name) {
    await new Promise((resolve) => {
      const request = indexedDB.deleteDatabase(name);
      request.onsuccess = request.onerror = request.onblocked = () => resolve();
    });
  }

  async function sha256(buffer) {
    const digest = await crypto.subtle.digest("SHA-256", buffer);
    return Array.from(new Uint8Array(digest)).map((value) => value.toString(16).padStart(2, "0")).join("");
  }

  function canonicalPolygonKey(points) {
    const normalized = (points || []).map((point) => ({
      x: Math.round(Number(point.x) * 1e6) / 1e6,
      y: Math.round(Number(point.y) * 1e6) / 1e6,
    }));
    if (normalized.length > 1 && samePoint(normalized[0], normalized[normalized.length - 1])) normalized.pop();
    if (!normalized.length) return "[]";
    const sequences = [];
    [normalized, [...normalized].reverse()].forEach((ordered) => {
      ordered.forEach((_, index) => sequences.push([...ordered.slice(index), ...ordered.slice(0, index)]));
    });
    return sequences.map((row) => stableJson(row)).sort()[0];
  }

  async function recoverySourceSnapshot() {
    const oldName = runtime.uiConfig.question_contract.old_indexeddb_namespace;
    const known = await indexedDB.databases();
    if (!oldName || !known.some((row) => row.name === oldName)) return null;
    const database = await openNamedDatabase(oldName);
    try {
      return await databaseSnapshot(database);
    } finally {
      database.close();
    }
  }

  async function initializeRecoveryMigration() {
    if (!runtime.recovery.enabled) return;
    const expected = Number(runtime.uiConfig.question_contract.expected_legacy_pending_records || 5);
    const existing = await idbGet("session", "pending_recovery_migration");
    let migration = existing || null;
    if (!migration) {
      const source = await recoverySourceSnapshot();
      if (!source) throw new Error("The preserved R1-R2 browser database is unavailable; no data was changed.");
      const sourceHash = await sha256Text(stableJson(source.stores));
      const outbox = source.stores.find((store) => store.name === "outbox")?.records || [];
      const drafts = source.stores.find((store) => store.name === "drafts")?.records || [];
      if (outbox.length !== expected) {
        throw new Error(`Expected ${expected} preserved queue records but found ${outbox.length}; migration stopped.`);
      }
      const temporaryName = `${runtime.uiConfig.question_contract.indexeddb_namespace}_migration_probe_${sourceHash.slice(0, 12)}`;
      await deleteDatabase(temporaryName);
      const temporary = await openNamedDatabase(temporaryName);
      await importSnapshot(temporary, source);
      const restored = await databaseSnapshot(temporary);
      temporary.close();
      const restoredHash = await sha256Text(stableJson(restored.stores));
      if (restoredHash !== sourceHash) {
        throw new Error("Temporary browser restore hash mismatch; the old database remains untouched.");
      }
      await deleteDatabase(temporaryName);
      migration = {
        key: "pending_recovery_migration",
        schema_version: "football_intelligence.m5_5g4_r1_r3.browser_migration.v1",
        status: "READY",
        old_database_name: source.name,
        new_database_name: runtime.uiConfig.question_contract.indexeddb_namespace,
        old_database_retained_read_only: true,
        source_export_sha256: sourceHash,
        temporary_restore_sha256: restoredHash,
        initial_outbox_count: outbox.length,
        initial_draft_count: drafts.length,
        migrated_at: new Date().toISOString(),
        acknowledged: [],
      };
      await importSnapshot(runtime.db, source, migration);
      const importedOutbox = await idbAll("outbox");
      const importedDrafts = await idbAll("drafts");
      if (importedOutbox.length !== outbox.length || importedDrafts.length !== drafts.length) {
        throw new Error("Real recovery database count mismatch; the old database remains untouched.");
      }
    }
    runtime.recovery.migration = migration;
    runtime.recovery.rows = (await idbAll("outbox")).sort(
      (left, right) => left.createdAt.localeCompare(right.createdAt) || left.id.localeCompare(right.id),
    );
    const drafts = await idbAll("drafts");
    runtime.recovery.draft = drafts.length === 1 ? drafts[0] : null;
    const draftKey = canonicalPolygonKey(runtime.recovery.draft?.points || []);
    runtime.recovery.draftRepresented = Boolean(runtime.recovery.draft) && runtime.recovery.rows.some((row) => (
      row.draftKey === runtime.recovery.draft.key
      && canonicalPolygonKey(row.payload?.corrected_polygon_original_pixels || []) === draftKey
    ));
    runtime.recovery.active = runtime.recovery.rows.length > 0;
  }

  async function recordRecoveryAcknowledgement(row, response, disposition) {
    const migration = await idbGet("session", "pending_recovery_migration");
    const acknowledged = [...(migration?.acknowledged || [])];
    if (!acknowledged.some((item) => item.idempotency_key === row.payload.idempotency_key)) {
      acknowledged.push({
        original_event_id: row.payload.client_event_id,
        idempotency_key: row.payload.idempotency_key,
        original_created_at: row.createdAt,
        disposition,
        server_event_sequence: response.acknowledged_event_sequence || response.server_event_sequence || null,
        acknowledged_at: new Date().toISOString(),
      });
    }
    const remaining = (await idbAll("outbox")).length;
    const updated = {
      ...migration,
      key: "pending_recovery_migration",
      status: remaining === 0 ? "COMPLETE" : "IN_PROGRESS",
      acknowledged,
      remaining_outbox_count: remaining,
      completed_at: remaining === 0 ? new Date().toISOString() : null,
      updated_at: new Date().toISOString(),
    };
    await idbPut("session", updated);
    runtime.recovery.migration = updated;
  }

  async function downloadRecoveryExport() {
    const source = await recoverySourceSnapshot();
    const current = await databaseSnapshot(runtime.db);
    const payload = {
      schema_version: "football_intelligence.m5_5g4_r1_r3.user_recovery_export.v1",
      exported_at: new Date().toISOString(),
      old_database: source,
      repaired_database: current,
    };
    const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], {type: "application/json"});
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `m5_5g4_r1_r3_recovery_${Date.now()}.json`;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
  }

  function asset(type) {
    return caseData().evidence_assets.find((row) => row.asset_type === type);
  }

  function finiteNumber(value) {
    return Number.isFinite(Number(value));
  }

  function finitePoint(point) {
    return point && finiteNumber(point.x) && finiteNumber(point.y);
  }

  function screenConstantMarkerRadius(cssRadius, scale = runtime.transform.scale) {
    const desiredRadius = Number(cssRadius);
    const currentScale = Number(scale);
    if (!Number.isFinite(desiredRadius) || desiredRadius <= 0) {
      throw new RangeError("marker CSS radius must be finite and positive");
    }
    if (!Number.isFinite(currentScale) || currentScale <= 0) {
      throw new RangeError("marker scale must be finite and positive");
    }
    return desiredRadius / currentScale;
  }

  function samePoint(left, right, epsilon = GEOMETRY_EPSILON) {
    return Math.abs(Number(left.x) - Number(right.x)) <= epsilon
      && Math.abs(Number(left.y) - Number(right.y)) <= epsilon;
  }

  function orientation(a, b, c) {
    return (Number(b.x) - Number(a.x)) * (Number(c.y) - Number(a.y))
      - (Number(b.y) - Number(a.y)) * (Number(c.x) - Number(a.x));
  }

  function onSegment(a, b, point, epsilon = GEOMETRY_EPSILON) {
    return Number(point.x) >= Math.min(Number(a.x), Number(b.x)) - epsilon
      && Number(point.x) <= Math.max(Number(a.x), Number(b.x)) + epsilon
      && Number(point.y) >= Math.min(Number(a.y), Number(b.y)) - epsilon
      && Number(point.y) <= Math.max(Number(a.y), Number(b.y)) + epsilon
      && Math.abs(orientation(a, b, point)) <= epsilon;
  }

  function lineIntersection(a, b, c, d) {
    const x1 = Number(a.x); const y1 = Number(a.y);
    const x2 = Number(b.x); const y2 = Number(b.y);
    const x3 = Number(c.x); const y3 = Number(c.y);
    const x4 = Number(d.x); const y4 = Number(d.y);
    const denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (Math.abs(denominator) <= GEOMETRY_EPSILON) return null;
    return {
      x: ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator,
      y: ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator,
    };
  }

  function uniquePoints(points) {
    return points.filter((point, index) => points.findIndex((candidate) => samePoint(candidate, point)) === index);
  }

  function collinearOverlapPoint(a, b, c, d) {
    const useX = Math.abs(Number(b.x) - Number(a.x)) >= Math.abs(Number(b.y) - Number(a.y));
    const coordinate = (point) => Number(useX ? point.x : point.y);
    const low = Math.max(Math.min(coordinate(a), coordinate(b)), Math.min(coordinate(c), coordinate(d)));
    const high = Math.min(Math.max(coordinate(a), coordinate(b)), Math.max(coordinate(c), coordinate(d)));
    if (high < low - GEOMETRY_EPSILON) return {overlap: false, point: null, length: 0};
    const midpoint = (low + high) / 2;
    const start = coordinate(a);
    const end = coordinate(b);
    const ratio = Math.abs(end - start) <= GEOMETRY_EPSILON ? 0 : (midpoint - start) / (end - start);
    return {
      overlap: true,
      length: Math.max(0, high - low),
      point: {
        x: Number(a.x) + ratio * (Number(b.x) - Number(a.x)),
        y: Number(a.y) + ratio * (Number(b.y) - Number(a.y)),
      },
    };
  }

  function classifySegmentIntersection(a, b, c, d) {
    if (![a, b, c, d].every(finitePoint)) return {kind: "INVALID_INPUT", point: null};
    if (samePoint(a, b) || samePoint(c, d)) return {kind: "ZERO_LENGTH", point: samePoint(a, b) ? a : c};
    const values = [orientation(a, b, c), orientation(a, b, d), orientation(c, d, a), orientation(c, d, b)];
    const opposite = (left, right) => (left > GEOMETRY_EPSILON && right < -GEOMETRY_EPSILON)
      || (left < -GEOMETRY_EPSILON && right > GEOMETRY_EPSILON);
    if (opposite(values[0], values[1]) && opposite(values[2], values[3])) {
      return {kind: "PROPER_CROSSING", point: lineIntersection(a, b, c, d)};
    }
    if (values.every((value) => Math.abs(value) <= GEOMETRY_EPSILON)) {
      const overlap = collinearOverlapPoint(a, b, c, d);
      if (!overlap.overlap) return {kind: "NONE", point: null};
      return {
        kind: overlap.length > GEOMETRY_EPSILON ? "COLLINEAR_OVERLAP" : "TOUCH",
        point: overlap.point,
      };
    }
    const touches = uniquePoints([
      Math.abs(values[0]) <= GEOMETRY_EPSILON && onSegment(a, b, c) ? c : null,
      Math.abs(values[1]) <= GEOMETRY_EPSILON && onSegment(a, b, d) ? d : null,
      Math.abs(values[2]) <= GEOMETRY_EPSILON && onSegment(c, d, a) ? a : null,
      Math.abs(values[3]) <= GEOMETRY_EPSILON && onSegment(c, d, b) ? b : null,
    ].filter(Boolean));
    return touches.length ? {kind: "TOUCH", point: touches[0]} : {kind: "NONE", point: null};
  }

  function geometryReason(kind) {
    if (kind === "PROPER_CROSSING") return "This line would cross an earlier part of the outline.";
    if (kind === "COLLINEAR_OVERLAP") return "This line would overlap an earlier part of the outline.";
    if (kind === "TOUCH") return "This line would touch a non-adjacent part of the outline.";
    if (kind === "ZERO_LENGTH") return "Place the next point somewhere else.";
    if (kind === "OUTSIDE_IMAGE") return "This point is outside the displayed image.";
    if (kind === "OUTSIDE_FOCAL_ROI") return "This point is outside the focal review region.";
    if (kind === "INSUFFICIENT_AREA") return "This outline is too small to represent a visible person.";
    return "This segment is not valid.";
  }

  function intersectionIssue(a, b, c, d, allowedTouchPoint = null) {
    const result = classifySegmentIntersection(a, b, c, d);
    if (result.kind === "NONE") return null;
    if (result.kind === "TOUCH" && allowedTouchPoint && result.point && samePoint(result.point, allowedTouchPoint)) return null;
    return {...result, reason: geometryReason(result.kind)};
  }

  function validateOpenSegment(points, candidate, containsPoint = () => true) {
    if (!finitePoint(candidate) || !containsPoint(candidate)) {
      return {valid: false, kind: "OUTSIDE_IMAGE", point: candidate, reason: geometryReason("OUTSIDE_IMAGE")};
    }
    if (!points.length) return {valid: true, kind: "NONE", point: null, reason: ""};
    const start = points[points.length - 1];
    if (samePoint(start, candidate)) {
      return {valid: false, kind: "ZERO_LENGTH", point: candidate, reason: geometryReason("ZERO_LENGTH")};
    }
    for (let edgeIndex = 0; edgeIndex < points.length - 1; edgeIndex += 1) {
      const allowed = edgeIndex === points.length - 2 ? start : null;
      const issue = intersectionIssue(start, candidate, points[edgeIndex], points[edgeIndex + 1], allowed);
      if (issue) return {valid: false, edgeIndex, ...issue};
    }
    return {valid: true, kind: "NONE", point: null, reason: ""};
  }

  function candidateCrosses(candidate) {
    return !validateOpenSegment(runtime.points, candidate, insideDisplayedImage).valid;
  }

  function validateClosingSegment(points) {
    if (points.length < 3) return {valid: false, kind: "TOO_FEW_VERTICES", point: null, reason: "Add at least three vertices."};
    const start = points[points.length - 1];
    const end = points[0];
    if (samePoint(start, end)) return {valid: false, kind: "ZERO_LENGTH", point: end, reason: geometryReason("ZERO_LENGTH")};
    for (let edgeIndex = 0; edgeIndex < points.length - 1; edgeIndex += 1) {
      let allowed = null;
      if (edgeIndex === 0) allowed = end;
      if (edgeIndex === points.length - 2) allowed = start;
      const issue = intersectionIssue(start, end, points[edgeIndex], points[edgeIndex + 1], allowed);
      if (issue) return {valid: false, edgeIndex, ...issue};
    }
    return {valid: true, kind: "NONE", point: null, reason: ""};
  }

  function polygonArea(points) {
    if (points.length < 3) return 0;
    return Math.abs(points.reduce((total, point, index) => {
      const next = points[(index + 1) % points.length];
      return total + Number(point.x) * Number(next.y) - Number(next.x) * Number(point.y);
    }, 0)) / 2;
  }

  function validateSimplePolygon(points) {
    if (points.length < 3) return {valid: false, kind: "TOO_FEW_VERTICES", reason: "Add at least three vertices.", point: null};
    if (polygonArea(points) < MINIMUM_POLYGON_AREA) {
      return {valid: false, kind: "INSUFFICIENT_AREA", reason: geometryReason("INSUFFICIENT_AREA"), point: null};
    }
    const edgeCount = points.length;
    for (let left = 0; left < edgeCount; left += 1) {
      const leftNext = (left + 1) % edgeCount;
      if (samePoint(points[left], points[leftNext])) {
        return {valid: false, kind: "ZERO_LENGTH", reason: geometryReason("ZERO_LENGTH"), point: points[left], edges: [left, left]};
      }
      for (let right = left + 1; right < edgeCount; right += 1) {
        const rightNext = (right + 1) % edgeCount;
        let allowed = null;
        if (leftNext === right) allowed = points[right];
        if (rightNext === left) allowed = points[left];
        const issue = intersectionIssue(
          points[left], points[leftNext], points[right], points[rightNext], allowed,
        );
        if (issue) return {valid: false, edges: [left, right], ...issue};
      }
    }
    return {valid: true, kind: "NONE", reason: "", point: null};
  }

  function pointInPolygon(point, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
      const a = polygon[i];
      const b = polygon[j];
      if ((Number(a.y) > Number(point.y)) !== (Number(b.y) > Number(point.y))
        && Number(point.x) < ((Number(b.x) - Number(a.x)) * (Number(point.y) - Number(a.y)))
          / (Number(b.y) - Number(a.y)) + Number(a.x)) inside = !inside;
    }
    return inside;
  }

  function polygonsOverlap(left, right) {
    if (!left.length || !right.length) return false;
    for (let i = 0; i < left.length; i += 1) {
      for (let j = 0; j < right.length; j += 1) {
        if (classifySegmentIntersection(
          left[i], left[(i + 1) % left.length], right[j], right[(j + 1) % right.length],
        ).kind !== "NONE") return true;
      }
    }
    return pointInPolygon(left[0], right) || pointInPolygon(right[0], left);
  }

  function focalBounds() {
    const roi = binding().focal_roi_original_pixels;
    return {x1: Number(roi.x1), y1: Number(roi.y1), x2: Number(roi.x2), y2: Number(roi.y2)};
  }

  function displayedSourceBounds() {
    if (runtime.view === "focal") {
      const roi = focalBounds();
      return {
        x1: roi.x1,
        y1: roi.y1,
        x2: roi.x1 + runtime.transform.naturalWidth,
        y2: roi.y1 + runtime.transform.naturalHeight,
      };
    }
    return {x1: 0, y1: 0, x2: runtime.transform.naturalWidth, y2: runtime.transform.naturalHeight};
  }

  function insideBounds(point, bounds, epsilon = GEOMETRY_EPSILON) {
    return finitePoint(point)
      && Number(point.x) >= bounds.x1 - epsilon && Number(point.x) <= bounds.x2 + epsilon
      && Number(point.y) >= bounds.y1 - epsilon && Number(point.y) <= bounds.y2 + epsilon;
  }

  function insideDisplayedImage(point) {
    return insideBounds(point, displayedSourceBounds());
  }

  function insideFocalRoi(point) {
    return insideBounds(point, focalBounds());
  }

  function sourceToImage(point) {
    const roi = focalBounds();
    return runtime.view === "focal"
      ? {x: Number(point.x) - roi.x1, y: Number(point.y) - roi.y1}
      : {x: Number(point.x), y: Number(point.y)};
  }

  function imageToSource(point) {
    const roi = focalBounds();
    return runtime.view === "focal"
      ? {x: Number(point.x) + roi.x1, y: Number(point.y) + roi.y1}
      : {x: Number(point.x), y: Number(point.y)};
  }

  function sourceToViewport(point) {
    const imagePoint = sourceToImage(point);
    return {
      x: imagePoint.x * runtime.transform.scale + runtime.transform.translateX,
      y: imagePoint.y * runtime.transform.scale + runtime.transform.translateY,
    };
  }

  function viewportToSource(point) {
    return imageToSource({
      x: (Number(point.x) - runtime.transform.translateX) / runtime.transform.scale,
      y: (Number(point.y) - runtime.transform.translateY) / runtime.transform.scale,
    });
  }

  function sourceRoundTrip(point) {
    const restored = viewportToSource(sourceToViewport(point));
    return {point: restored, error: Math.hypot(restored.x - Number(point.x), restored.y - Number(point.y))};
  }

  function bboxFromPoints(points) {
    if (!points.length) return null;
    return {
      x1: Math.min(...points.map((point) => Number(point.x))),
      y1: Math.min(...points.map((point) => Number(point.y))),
      x2: Math.max(...points.map((point) => Number(point.x))),
      y2: Math.max(...points.map((point) => Number(point.y))),
    };
  }

  function validBbox(box) {
    return box && [box.x1, box.y1, box.x2, box.y2].every(finiteNumber)
      && Number(box.x2) > Number(box.x1) && Number(box.y2) > Number(box.y1);
  }

  function normalizeBbox(box) {
    return {x1: Number(box.x1), y1: Number(box.y1), x2: Number(box.x2), y2: Number(box.y2)};
  }

  function unionBboxes(boxes) {
    const valid = boxes.filter(validBbox).map(normalizeBbox);
    if (!valid.length) return null;
    return {
      x1: Math.min(...valid.map((box) => box.x1)),
      y1: Math.min(...valid.map((box) => box.y1)),
      x2: Math.max(...valid.map((box) => box.x2)),
      y2: Math.max(...valid.map((box) => box.y2)),
    };
  }

  function bboxesIntersect(left, right) {
    return validBbox(left) && validBbox(right)
      && Number(left.x2) >= Number(right.x1) && Number(left.x1) <= Number(right.x2)
      && Number(left.y2) >= Number(right.y1) && Number(left.y1) <= Number(right.y2);
  }

  function candidateBindingValid(candidate = activeCandidate()) {
    if (!candidate || !runtime.evidenceBindingValid || !validBbox(candidate.bbox_original_pixels)) return false;
    const box = normalizeBbox(candidate.bbox_original_pixels);
    return box.x1 >= 0 && box.y1 >= 0
      && box.x2 <= Number(binding().image_width) && box.y2 <= Number(binding().image_height)
      && Boolean(binding().source_frame_sha256) && Boolean(binding().focal_transform_hash);
  }

  function candidateIntersectsDisplayedImage(candidate = activeCandidate()) {
    return candidateBindingValid(candidate) && bboxesIntersect(candidate.bbox_original_pixels, displayedSourceBounds());
  }

  function bboxIntersectsViewport(box) {
    if (!validBbox(box)) return false;
    const topLeft = sourceToViewport({x: box.x1, y: box.y1});
    const bottomRight = sourceToViewport({x: box.x2, y: box.y2});
    return bottomRight.x >= 0 && topLeft.x <= runtime.transform.viewportWidth
      && bottomRight.y >= 0 && topLeft.y <= runtime.transform.viewportHeight;
  }

  function svgElement(name, attributes = {}) {
    const node = document.createElementNS(SVG_NS, name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  }

  function pointsAttribute(points) {
    return points.map(sourceToImage).map((point) => `${point.x},${point.y}`).join(" ");
  }

  function appendPolygon(parent, points, className) {
    if (points.length < 3) return null;
    const polygon = svgElement("polygon", {points: pointsAttribute(points), class: className, "vector-effect": "non-scaling-stroke"});
    parent.append(polygon);
    return polygon;
  }

  function appendLine(parent, left, right, className) {
    const a = sourceToImage(left);
    const b = sourceToImage(right);
    const line = svgElement("line", {
      x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: className, "vector-effect": "non-scaling-stroke",
    });
    parent.append(line);
    return line;
  }

  function compactChip(parent, sourcePoint, text, className, usedBoxes, index = 0) {
    const local = sourceToImage(sourcePoint);
    const width = Math.min(178, Math.max(48, 12 + String(text).length * 5.7));
    const height = 17;
    let screenX = local.x * runtime.transform.scale + runtime.transform.translateX + 7;
    let screenY = local.y * runtime.transform.scale + runtime.transform.translateY - height - 6 - (index % 3) * 3;
    screenX = Math.max(4, Math.min(runtime.transform.viewportWidth - width - 4, screenX));
    screenY = Math.max(4, Math.min(runtime.transform.viewportHeight - height - 4, screenY));
    let candidateBox = {x1: screenX, y1: screenY, x2: screenX + width, y2: screenY + height};
    for (let attempts = 0; attempts < 8 && usedBoxes.some((box) => bboxesIntersect(box, candidateBox)); attempts += 1) {
      screenY = Math.max(4, screenY - height - 3);
      candidateBox = {x1: screenX, y1: screenY, x2: screenX + width, y2: screenY + height};
    }
    usedBoxes.push(candidateBox);
    const imageX = (screenX - runtime.transform.translateX) / runtime.transform.scale;
    const imageY = (screenY - runtime.transform.translateY) / runtime.transform.scale;
    parent.append(svgElement("line", {
      x1: local.x,
      y1: local.y,
      x2: imageX,
      y2: imageY + height / runtime.transform.scale,
      class: "dcOverlayLeader",
      "vector-effect": "non-scaling-stroke",
    }));
    const group = svgElement("g", {
      class: `dcOverlayChip ${className || ""}`.trim(),
      transform: `translate(${imageX} ${imageY}) scale(${1 / runtime.transform.scale})`,
    });
    group.append(svgElement("rect", {x: 0, y: 0, width, height, rx: 3}));
    const label = svgElement("text", {x: 5, y: 12});
    label.textContent = text;
    group.append(label);
    parent.append(group);
  }

  function sourceBboxToViewportBox(sourceBox, padding = 0) {
    const box = normalizeBbox(sourceBox);
    const topLeft = sourceToImage({x: box.x1, y: box.y1});
    const bottomRight = sourceToImage({x: box.x2, y: box.y2});
    return {
      x1: topLeft.x * runtime.transform.scale + runtime.transform.translateX - padding,
      y1: topLeft.y * runtime.transform.scale + runtime.transform.translateY - padding,
      x2: bottomRight.x * runtime.transform.scale + runtime.transform.translateX + padding,
      y2: bottomRight.y * runtime.transform.scale + runtime.transform.translateY + padding,
    };
  }

  function candidateCoverageAvailable() {
    if (!activeCandidate() || !$("dcShowMachineBox").checked || !runtime.machineRendered) return false;
    return runtime.machineInViewport || runtime.machineExplicitlyFocused;
  }

  function materialDependencies() {
    if (runtime.recovery.enabled) return [];
    if ($("dcUnreliable").checked) return item().occlusion_dependencies || [];
    const validation = validateSimplePolygon(runtime.points);
    if (runtime.points.length < 3 || !validation.valid) return [];
    return (item().occlusion_dependencies || []).filter((dependency) => {
      const other = dependency.other_polygon_original_pixels;
      return polygonsOverlap(item().original_polygon_original_pixels, other) !== polygonsOverlap(runtime.points, other)
        || dependency.original_graph_inconsistent;
    });
  }

  function activeRecoveryRow() {
    return runtime.recovery.rows[runtime.recovery.index] || null;
  }

  function activeRecoveryPair() {
    return runtime.recovery.preflight?.occlusion_pairs?.[runtime.recovery.pairIndex] || null;
  }

  function contextMask(maskUuid) {
    return (caseData().visible_metadata.context_masks || []).find(
      (row) => row.original_mask_uuid === maskUuid,
    ) || null;
  }

  function drawRecoveryPairOverlay(parent) {
    const pair = activeRecoveryPair();
    if (!runtime.recovery.active || !pair) return;
    const personB = contextMask(pair.person_b_mask_uuid);
    const personAPoints = runtime.points.length >= 3
      ? runtime.points
      : item().original_polygon_original_pixels;
    const personBPoints = personB?.polygon_original_pixels || [];
    if (personAPoints.length < 3 || personBPoints.length < 3) return;

    appendPolygon(parent, personAPoints, "dcPairPersonA");
    appendPolygon(parent, personBPoints, "dcPairPersonB");
    const evidence = pair.material_overlap_evidence || {};
    const overlapSource = evidence.original_overlap && !evidence.corrected_overlap
      ? item().original_polygon_original_pixels
      : personAPoints;
    if (evidence.original_overlap || evidence.corrected_overlap) {
      const clipId = `dcPairClip-${runtime.recovery.index}-${runtime.recovery.pairIndex}`;
      const definitions = svgElement("defs", {});
      const clip = svgElement("clipPath", {id: clipId, clipPathUnits: "userSpaceOnUse"});
      clip.append(svgElement("polygon", {points: pointsAttribute(personBPoints)}));
      definitions.append(clip);
      parent.append(definitions);
      const overlap = appendPolygon(parent, overlapSource, "dcMaterialOverlapRegion");
      if (overlap) overlap.setAttribute("clip-path", `url(#${clipId})`);
    }
  }

  function drawOriginalIntersections(parent) {
    (item().self_intersection_edge_pairs || []).forEach((pair) => {
      [pair.left_edge_vertex_indices, pair.right_edge_vertex_indices].forEach((indices) => {
        appendLine(
          parent,
          item().original_polygon_original_pixels[indices[0]],
          item().original_polygon_original_pixels[indices[1] % item().original_polygon_original_pixels.length],
          "dcIntersectionSegment",
        );
      });
    });
  }

  function renderOverlay() {
    const svg = $("dcOverlay");
    runtime.invalidPreview = Boolean(runtime.previewIssue);
    svg.replaceChildren();
    runtime.machineRendered = false;
    runtime.machineInViewport = false;
    const contextLayer = svgElement("g", {class: "dcContextLayer", "pointer-events": "none"});
    const originalLayer = svgElement("g", {class: "dcOriginalLayer", "pointer-events": "none"});
    const machineLayer = svgElement("g", {class: "dcMachineLayer", "pointer-events": "none"});
    const correctionLayer = svgElement("g", {class: "dcCorrectionLayer", "pointer-events": "none"});
    const pairLayer = svgElement("g", {class: "dcPairLayer", "pointer-events": "none"});
    const drawingLayer = svgElement("g", {class: "dcDrawingLayer", "pointer-events": "none"});
    const errorLayer = svgElement("g", {class: "dcErrorLayer", "pointer-events": "none"});
    const labelLayer = svgElement("g", {class: "dcLabelLayer", "pointer-events": "none"});
    svg.append(contextLayer, originalLayer, machineLayer, correctionLayer, pairLayer, drawingLayer, errorLayer, labelLayer);

    const context = caseData().visible_metadata.context_masks || [];
    context.forEach((mask) => {
      if (mask.original_mask_uuid !== item().original_mask_uuid) {
        appendPolygon(contextLayer, mask.polygon_original_pixels, "dcContextMask");
      }
    });

    if ($("dcCompareOriginal").checked) {
      appendPolygon(originalLayer, item().original_polygon_original_pixels, "dcOriginalMask");
      drawOriginalIntersections(errorLayer);
    }

    const candidate = activeCandidate();
    if ($("dcShowMachineBox").checked && candidateIntersectsDisplayedImage(candidate)) {
      const box = normalizeBbox(candidate.bbox_original_pixels);
      const topLeft = sourceToImage({x: box.x1, y: box.y1});
      const bottomRight = sourceToImage({x: box.x2, y: box.y2});
      machineLayer.append(svgElement("rect", {
        x: topLeft.x,
        y: topLeft.y,
        width: bottomRight.x - topLeft.x,
        height: bottomRight.y - topLeft.y,
        class: "dcMachineBox",
        "vector-effect": "non-scaling-stroke",
        "pointer-events": "none",
      }));
      runtime.machineRendered = true;
      runtime.machineInViewport = bboxIntersectsViewport(box);
    }

    if (runtime.points.length >= 3 && !runtime.drawing) {
      appendPolygon(correctionLayer, runtime.points, "dcCorrectionMask");
    }
    drawRecoveryPairOverlay(pairLayer);
    if (runtime.points.length) {
      const draft = runtime.preview ? [...runtime.points, runtime.preview] : runtime.points;
      const validDraft = !runtime.previewIssue;
      drawingLayer.append(svgElement("polyline", {
        points: pointsAttribute(draft),
        class: validDraft ? "dcDraftLine" : "dcDraftLine",
        "vector-effect": "non-scaling-stroke",
      }));
      runtime.points.forEach((point) => {
        const local = sourceToImage(point);
        drawingLayer.append(svgElement("circle", {
          cx: local.x,
          cy: local.y,
          r: screenConstantMarkerRadius(MARKER_SCREEN_RADIUS_CSS.vertex),
          class: "dcVertex",
          "vector-effect": "non-scaling-stroke",
        }));
      });
      if (runtime.preview && runtime.previewIssue) {
        appendLine(errorLayer, runtime.points[runtime.points.length - 1], runtime.preview, "dcInvalidSegment");
        if (runtime.previewIssue.point && finitePoint(runtime.previewIssue.point)) {
          const marker = sourceToImage(runtime.previewIssue.point);
          errorLayer.append(svgElement("circle", {
            cx: marker.x,
            cy: marker.y,
            r: screenConstantMarkerRadius(MARKER_SCREEN_RADIUS_CSS.crossing),
            class: "dcCrossingMarker",
            "vector-effect": "non-scaling-stroke",
          }));
        }
      }
      if (runtime.drawing && runtime.points.length >= 3) {
        const closure = validateClosingSegment(runtime.points);
        appendLine(
          closure.valid ? drawingLayer : errorLayer,
          runtime.points[runtime.points.length - 1],
          runtime.points[0],
          closure.valid ? "dcDraftLine" : "dcInvalidSegment",
        );
        if (!closure.valid && closure.point && finitePoint(closure.point)) {
          const marker = sourceToImage(closure.point);
          errorLayer.append(svgElement("circle", {
            cx: marker.x,
            cy: marker.y,
            r: screenConstantMarkerRadius(MARKER_SCREEN_RADIUS_CSS.crossing),
            class: "dcCrossingMarker",
            "vector-effect": "non-scaling-stroke",
          }));
        }
      }
    }

    const activePersonBox = bboxFromPoints(item().original_polygon_original_pixels || []);
    const usedLabelBoxes = validBbox(activePersonBox)
      ? [sourceBboxToViewportBox(activePersonBox, 3)]
      : [];
    if (!runtime.drawing && $("dcShowContextLabels").checked) {
      context.forEach((mask, index) => {
        if (mask.original_mask_uuid === item().original_mask_uuid) return;
        const box = bboxFromPoints(mask.polygon_original_pixels || []);
        const point = validBbox(box) ? {x: box.x1, y: box.y1} : mask.label_point_original_pixels;
        if (finitePoint(point) && insideDisplayedImage(point)) {
          compactChip(labelLayer, point, mask.anonymous_label, "context", usedLabelBoxes, index);
        }
      });
    }
    if (runtime.machineRendered && candidate) {
      const box = normalizeBbox(candidate.bbox_original_pixels);
      compactChip(
        labelLayer,
        {x: box.x1, y: box.y1},
        `${candidate.anonymous_label}: ${candidate.relation_plain_language}`,
        "machine",
        usedLabelBoxes,
      );
    }

    renderOcclusionRows();
    updateCandidatePanel();
    updateGeometryFeedback();
    updateTransformInspector();
    updateSaveGate();
  }

  function stageTransform() {
    $("dcStage").style.transform = `translate(${runtime.transform.translateX}px, ${runtime.transform.translateY}px) scale(${runtime.transform.scale})`;
  }

  function constrainTransform() {
    const viewport = $("dcViewport");
    runtime.transform.viewportWidth = Math.max(1, viewport.clientWidth);
    runtime.transform.viewportHeight = Math.max(1, viewport.clientHeight);
    const displayWidth = runtime.transform.naturalWidth * runtime.transform.scale;
    const displayHeight = runtime.transform.naturalHeight * runtime.transform.scale;
    const margin = 36;
    if (displayWidth <= runtime.transform.viewportWidth) {
      runtime.transform.translateX = (runtime.transform.viewportWidth - displayWidth) / 2;
    } else {
      runtime.transform.translateX = Math.max(
        margin - displayWidth,
        Math.min(runtime.transform.viewportWidth - margin, runtime.transform.translateX),
      );
    }
    if (displayHeight <= runtime.transform.viewportHeight) {
      runtime.transform.translateY = (runtime.transform.viewportHeight - displayHeight) / 2;
    } else {
      runtime.transform.translateY = Math.max(
        margin - displayHeight,
        Math.min(runtime.transform.viewportHeight - margin, runtime.transform.translateY),
      );
    }
  }

  function fitImage() {
    const viewport = $("dcViewport");
    const width = Math.max(1, viewport.clientWidth);
    const height = Math.max(1, viewport.clientHeight);
    runtime.transform.viewportWidth = width;
    runtime.transform.viewportHeight = height;
    runtime.transform.scale = Math.min(width / runtime.transform.naturalWidth, height / runtime.transform.naturalHeight);
    runtime.transform.translateX = (width - runtime.transform.naturalWidth * runtime.transform.scale) / 2;
    runtime.transform.translateY = (height - runtime.transform.naturalHeight * runtime.transform.scale) / 2;
    runtime.machineExplicitlyFocused = false;
    stageTransform();
    renderOverlay();
  }

  function fitSourceBounds(bounds, padding = 34) {
    if (!validBbox(bounds)) return false;
    const imageBounds = unionBboxes([
      {
        x1: sourceToImage({x: bounds.x1, y: bounds.y1}).x,
        y1: sourceToImage({x: bounds.x1, y: bounds.y1}).y,
        x2: sourceToImage({x: bounds.x2, y: bounds.y2}).x,
        y2: sourceToImage({x: bounds.x2, y: bounds.y2}).y,
      },
    ]);
    if (!imageBounds) return false;
    const viewport = $("dcViewport");
    const width = Math.max(1, viewport.clientWidth);
    const height = Math.max(1, viewport.clientHeight);
    const boxWidth = Math.max(1, imageBounds.x2 - imageBounds.x1);
    const boxHeight = Math.max(1, imageBounds.y2 - imageBounds.y1);
    runtime.transform.viewportWidth = width;
    runtime.transform.viewportHeight = height;
    runtime.transform.scale = Math.max(0.2, Math.min(12, Math.min((width - 2 * padding) / boxWidth, (height - 2 * padding) / boxHeight)));
    runtime.transform.translateX = width / 2 - ((imageBounds.x1 + imageBounds.x2) / 2) * runtime.transform.scale;
    runtime.transform.translateY = height / 2 - ((imageBounds.y1 + imageBounds.y2) / 2) * runtime.transform.scale;
    constrainTransform();
    stageTransform();
    renderOverlay();
    return true;
  }

  function zoomAt(factor, clientX = null, clientY = null) {
    const viewport = $("dcViewport");
    const rect = viewport.getBoundingClientRect();
    const focalX = clientX == null ? viewport.clientWidth / 2 : clientX - rect.left;
    const focalY = clientY == null ? viewport.clientHeight / 2 : clientY - rect.top;
    const imageX = (focalX - runtime.transform.translateX) / runtime.transform.scale;
    const imageY = (focalY - runtime.transform.translateY) / runtime.transform.scale;
    const next = Math.max(0.2, Math.min(12, runtime.transform.scale * factor));
    runtime.transform.translateX = focalX - imageX * next;
    runtime.transform.translateY = focalY - imageY * next;
    runtime.transform.scale = next;
    runtime.machineExplicitlyFocused = false;
    constrainTransform();
    stageTransform();
    renderOverlay();
  }

  function eventPoint(event) {
    const rect = $("dcViewport").getBoundingClientRect();
    return viewportToSource({x: event.clientX - rect.left, y: event.clientY - rect.top});
  }

  function activePersonBounds() {
    return bboxFromPoints(runtime.points.length >= 3 ? runtime.points : item().original_polygon_original_pixels)
      || item().original_tight_visible_box;
  }

  async function switchView(view) {
    runtime.view = view;
    runtime.pendingViewState = null;
    $("dcFocalView").classList.toggle("active", view === "focal");
    $("dcPanoramaView").classList.toggle("active", view === "panorama");
    await loadEvidence();
    await persistSession();
  }

  async function ensureBoundsDisplayable(bounds) {
    if (bboxesIntersect(bounds, displayedSourceBounds())) return true;
    if (runtime.view === "focal") {
      await switchView("panorama");
      return bboxesIntersect(bounds, displayedSourceBounds());
    }
    return false;
  }

  async function focusPerson() {
    const bounds = activePersonBounds();
    if (await ensureBoundsDisplayable(bounds)) fitSourceBounds(bounds, 54);
  }

  async function focusPersonAndCandidate() {
    const candidate = activeCandidate();
    const bounds = unionBboxes([activePersonBounds(), candidate?.bbox_original_pixels]);
    if (!bounds || !await ensureBoundsDisplayable(bounds)) {
      runtime.machineExplicitlyFocused = false;
      $("dcMachineStatus").textContent = "Machine box cannot be bound to this evidence view.";
      $("dcMachineStatus").classList.add("isBlocked");
      updateSaveGate();
      return;
    }
    runtime.machineExplicitlyFocused = true;
    fitSourceBounds(bounds, 58);
    runtime.machineExplicitlyFocused = true;
    renderOverlay();
  }

  async function loadEvidence() {
    runtime.evidenceBlocked = true;
    runtime.evidenceBindingValid = false;
    updateSaveGate();
    $("dcEvidenceBlocker").classList.add("isHidden");
    $("dcEvidenceStatus").textContent = "Verifying exact image binding";
    const selected = runtime.view === "focal" ? asset("crop") : asset("image");
    if (!selected) throw new Error("required evidence asset is missing");
    const url = `/evidence/${encodeURIComponent(caseData().case_id)}/${selected.relative_path.split("/").map(encodeURIComponent).join("/")}`;
    try {
      const response = await fetch(url, {cache: "no-store"});
      if (!response.ok || !String(response.headers.get("Content-Type") || "").startsWith("image/")) {
        throw new Error(`image route failed (${response.status})`);
      }
      const bytes = await response.arrayBuffer();
      if (!bytes.byteLength || await sha256(bytes) !== selected.sha256) throw new Error("image hash mismatch");
      if (selected.metadata?.source_frame_sha256 !== binding().source_frame_sha256) throw new Error("source-frame binding mismatch");
      if (!(selected.frame_sequences || []).includes(binding().frame_sequence)) throw new Error("frame-sequence binding mismatch");
      if (runtime.objectUrl) URL.revokeObjectURL(runtime.objectUrl);
      runtime.objectUrl = URL.createObjectURL(new Blob([bytes], {type: selected.media_type}));
      const image = $("dcBaseImage");
      image.src = runtime.objectUrl;
      await image.decode();
      if (!image.naturalWidth || !image.naturalHeight) throw new Error("image decoded with zero dimensions");
      if (runtime.view === "panorama"
        && (image.naturalWidth !== Number(binding().image_width) || image.naturalHeight !== Number(binding().image_height))) {
        throw new Error("panorama dimensions do not match source binding");
      }
      const roi = focalBounds();
      if (runtime.view === "focal"
        && (Math.abs(image.naturalWidth - (roi.x2 - roi.x1)) > 1 || Math.abs(image.naturalHeight - (roi.y2 - roi.y1)) > 1)) {
        throw new Error("focal dimensions do not match crop transform");
      }
      runtime.activeAsset = selected;
      runtime.transform.naturalWidth = image.naturalWidth;
      runtime.transform.naturalHeight = image.naturalHeight;
      $("dcStage").style.width = `${image.naturalWidth}px`;
      $("dcStage").style.height = `${image.naturalHeight}px`;
      $("dcOverlay").setAttribute("viewBox", `0 0 ${image.naturalWidth} ${image.naturalHeight}`);
      runtime.evidenceBindingValid = true;
      runtime.evidenceBlocked = false;
      $("dcEvidenceStatus").textContent = `Evidence verified: ${image.naturalWidth} x ${image.naturalHeight}, frame and source hash bound`;
      fitImage();
      if (runtime.pendingViewState && runtime.pendingViewState.view === runtime.view) {
        const saved = runtime.pendingViewState;
        runtime.transform.scale = Math.max(0.2, Math.min(12, Number(saved.scale) || runtime.transform.scale));
        runtime.transform.translateX = Number(saved.translateX) || 0;
        runtime.transform.translateY = Number(saved.translateY) || 0;
        runtime.pendingViewState = null;
        constrainTransform();
        stageTransform();
        renderOverlay();
      }
      if (activeCandidate() && !candidateIntersectsDisplayedImage() && runtime.view === "focal") {
        await switchView("panorama");
        return;
      }
    } catch (error) {
      runtime.evidenceBlocked = true;
      runtime.evidenceBindingValid = false;
      $("dcEvidenceBlocker").textContent = `Evidence unavailable. ${error.message}`;
      $("dcEvidenceBlocker").classList.remove("isHidden");
      $("dcEvidenceStatus").textContent = "Evidence verification failed";
    }
    renderOverlay();
  }

  function renderCoverageRows() {
    const container = $("dcCoverageRows");
    const candidate = activeCandidate();
    if (!candidate) {
      container.innerHTML = "<p>No machine-box coverage value depends on this outline.</p>";
      updateCandidatePanel();
      return;
    }
    const selected = runtime.coverageValues[candidate.candidate_uuid];
    container.innerHTML = `<label class="dcCoverageRow"><strong>${escapeHtml(candidate.anonymous_label)}</strong><select data-dc-coverage="${escapeHtml(candidate.candidate_uuid)}">${COVERAGE.map(([choice, label]) => `<option value="${choice}" ${selected !== undefined && selected !== "" && Number(choice) === Number(selected) ? "selected" : ""}>${label}</option>`).join("")}</select></label>`;
    const select = container.querySelector("select");
    select.addEventListener("change", async () => {
      runtime.coverageValues[candidate.candidate_uuid] = select.value;
      await persistDraft();
      updateSaveGate();
      if (select.value !== "" && runtime.candidateIndex < candidateRows().length - 1) {
        window.setTimeout(() => selectCandidate(runtime.candidateIndex + 1), 180);
      }
    });
    updateCandidatePanel();
  }

  function updateCandidatePanel() {
    const rows = candidateRows();
    const candidate = activeCandidate();
    $("dcCandidateNavigator").classList.toggle("isHidden", rows.length === 0);
    $("dcPreviousCandidate").disabled = runtime.candidateIndex <= 0;
    $("dcNextCandidate").disabled = runtime.candidateIndex >= rows.length - 1;
    $("dcCandidateLabel").textContent = candidate ? `Machine box ${runtime.candidateIndex + 1} of ${rows.length}` : "No machine box";
    $("dcCandidateRelation").textContent = candidate?.relation_plain_language || "No candidate coverage review is required.";
    const available = candidateCoverageAvailable()
      && !$("dcUnreliable").checked
      && !recoveryMutationLocked();
    const select = document.querySelector("[data-dc-coverage]");
    if (select) select.disabled = !available;
    const status = $("dcMachineStatus");
    status.classList.toggle("isBlocked", Boolean(candidate) && !available && !$("dcUnreliable").checked);
    if (!candidate) status.textContent = "No active machine box for this outline.";
    else if (recoveryMutationLocked()) status.textContent = "Preserved machine-box coverage is locked during safe recovery.";
    else if (!$("dcShowMachineBox").checked) status.textContent = "Show the machine box before reviewing coverage.";
    else if (!candidateBindingValid(candidate)) status.textContent = "Machine-box source binding is invalid. Coverage is blocked.";
    else if (!runtime.machineRendered) status.textContent = "Machine box is outside this evidence view. Use Focus person + machine box.";
    else if (!runtime.machineInViewport && !runtime.machineExplicitlyFocused) status.textContent = "Machine box is outside the viewport. Focus it before reviewing coverage.";
    else status.textContent = "Machine box is rendered, source-bound and ready for coverage review.";
    $("dcCandidateAdvanced").innerHTML = candidate ? [
      ["Candidate UUID", candidate.candidate_uuid],
      ["Relation", candidate.relation],
      ["Source frame", binding().frame_sequence],
      ["Transform hash", binding().focal_transform_hash],
    ].map(([term, value]) => `<div><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("") : "";
  }

  async function selectCandidate(next) {
    const rows = candidateRows();
    runtime.candidateIndex = rows.length ? Math.max(0, Math.min(rows.length - 1, next)) : 0;
    runtime.machineExplicitlyFocused = false;
    renderCoverageRows();
    renderOverlay();
    await persistSession();
  }

  function renderOcclusionRows(saved = null) {
    if (runtime.recovery.enabled) {
      $("dcOcclusionPanel").classList.add("isHidden");
      $("dcOcclusionRows").replaceChildren();
      return;
    }
    const dependencies = materialDependencies();
    const panel = $("dcOcclusionPanel");
    panel.classList.toggle("isHidden", dependencies.length === 0);
    const liveValues = new Map(Array.from(document.querySelectorAll("[data-dc-occlusion]")).map((select) => [select.dataset.dcOcclusion, select.value]));
    const priorValues = new Map((saved?.occlusion_reviews || []).map((row) => [row.other_mask_uuid, row.status]));
    liveValues.forEach((value, key) => { if (value) priorValues.set(key, value); });
    $("dcOcclusionRows").innerHTML = dependencies.map((dependency) => `<label class="dcOcclusionRow"><strong>${escapeHtml(dependency.anonymous_label)}</strong><small>The overlap changed enough to recheck front/back order.</small><select data-dc-occlusion="${escapeHtml(dependency.other_mask_uuid)}"><option value="">Choose overlap result</option><option value="ORDER_PRESERVED" ${priorValues.get(dependency.other_mask_uuid) === "ORDER_PRESERVED" ? "selected" : ""}>Original order still applies</option><option value="ORDER_CHANGED" ${priorValues.get(dependency.other_mask_uuid) === "ORDER_CHANGED" ? "selected" : ""}>Order changed</option><option value="UNRESOLVED" ${priorValues.get(dependency.other_mask_uuid) === "UNRESOLVED" ? "selected" : ""}>Cannot tell</option></select></label>`).join("");
    $("dcOcclusionRows").querySelectorAll("select").forEach((select) => select.addEventListener("change", () => {
      persistDraft();
      updateSaveGate();
    }));
  }

  function explicitPairAnswers(row, preflight = null) {
    const allowed = new Set(["PERSON_A_IN_FRONT", "PERSON_B_IN_FRONT", "NO_MEANINGFUL_OVERLAP", "UNRESOLVED"]);
    const required = new Set(preflight?.required_occlusion_pair_review_ids || []);
    const answers = {...(row.reconciliation?.pair_answers || {})};
    (row.payload?.occlusion_reviews || []).forEach((review) => {
      const dependencyId = String(review.dependency_id || "");
      const choice = String(review.pair_choice || "");
      if (dependencyId && allowed.has(choice) && (!preflight || required.has(dependencyId))) {
        answers[dependencyId] = choice;
      }
    });
    return answers;
  }

  function recoveryPayload(row, {rawAnswers = false, final = false} = {}) {
    const payload = structuredClone(row.payload);
    const preflight = row.reconciliation?.preflight || null;
    const answers = explicitPairAnswers(row, preflight);
    payload.client_build_id = RECOVERY_BUILD_ID;
    if (rawAnswers || !preflight) {
      payload.occlusion_reviews = (row.payload.occlusion_reviews || []).filter(
        (review) => review.dependency_id && review.pair_choice,
      );
    } else {
      payload.occlusion_reviews = (preflight.occlusion_pairs || [])
        .filter((pair) => answers[pair.dependency_id])
        .map((pair) => ({
          dependency_id: pair.dependency_id,
          other_mask_uuid: pair.person_b_mask_uuid,
          pair_choice: answers[pair.dependency_id],
        }));
    }
    if (final && preflight) {
      payload.expected_server_state_hash = runtime.state.server_state_hash || null;
      payload.dependency_handshake_version = DEPENDENCY_HANDSHAKE_VERSION;
      payload.normalized_polygon_hash = preflight.normalized_polygon_hash;
      payload.dependency_set_hash = preflight.dependency_set_hash;
      payload.dependency_algorithm_version_hash = preflight.dependency_algorithm_version_hash;
      payload.dependency_answer_revision_id = row.reconciliation?.answer_revision_id || null;
      payload.pending_recovery_provenance = {
        original_client_event_id: row.payload.client_event_id,
        original_idempotency_key: row.payload.idempotency_key,
        original_created_at: row.createdAt,
        source_database_name: runtime.recovery.migration?.old_database_name || null,
        source_export_sha256: runtime.recovery.migration?.source_export_sha256 || null,
        original_payload_preserved_in_read_only_database: true,
      };
    }
    return payload;
  }

  async function persistRecoveryRow(row) {
    await idbPut("outbox", row);
    const index = runtime.recovery.rows.findIndex((candidate) => candidate.id === row.id);
    if (index >= 0) runtime.recovery.rows[index] = row;
  }

  function recoveryQuestionCount() {
    return runtime.recovery.rows.reduce((total, row) => {
      const preflight = row.reconciliation?.preflight;
      if (!preflight || preflight.already_acknowledged) return total;
      const answers = explicitPairAnswers(row, preflight);
      return total + (preflight.required_occlusion_pair_review_ids || []).filter((id) => !answers[id]).length;
    }, 0);
  }

  function recoveryStatusLabel(status) {
    return ({
      SAVED_TO_SERVER: "Saved to server",
      WAITING_FOR_SERVER: "Waiting for server",
      NEEDS_OVERLAP_ANSWERS: "Needs overlap answers",
      SERVER_REJECTED_REVIEW_REQUIRED: "Server rejected - review required",
      OFFLINE_SAFELY_STORED: "Offline - safely stored locally",
      QUARANTINED_EXPORT_AVAILABLE: "Quarantined - export available",
      PREFLIGHT_READY: "Ready for server acknowledgement",
    })[status] || "Checking preserved item";
  }

  function renderRecoveryStatus() {
    const panel = $("dcRecoveryPanel");
    panel.classList.toggle("isHidden", !runtime.recovery.enabled);
    document.body.classList.toggle("dcRecoveryLocked", runtime.recovery.active);
    const migration = runtime.recovery.migration || {};
    const acknowledged = migration.acknowledged?.length || 0;
    const pending = runtime.recovery.rows.length;
    const initial = Number(migration.initial_outbox_count || pending || 0);
    const currentOrdinal = Math.min(initial, acknowledged + 1);
    $("dcAcknowledgedCount").textContent = String(Object.keys(runtime.state?.corrections || {}).length);
    $("dcPendingCount").textContent = String(pending);
    $("dcQuestionCount").textContent = String(recoveryQuestionCount());
    $("dcRecoverySummary").textContent = pending
      ? `${Object.keys(runtime.state?.corrections || {}).length} corrections saved to server. ${pending} locally saved ${pending === 1 ? "item needs" : "items need"} reconciliation.`
      : "All preserved local events have a server acknowledgement. Normal correction can resume.";
    $("dcSaveState").textContent = pending
      ? `Locally saved work | pending ${pending}`
      : "Saved to server | pending 0";
    const row = activeRecoveryRow();
    const status = row?.reconciliation?.status || (pending ? "WAITING_FOR_SERVER" : "SAVED_TO_SERVER");
    $("dcRecoveryCurrent").textContent = row
      ? `Recovery item ${currentOrdinal} of ${initial}: ${recoveryStatusLabel(status)}.`
      : "Recovery queue complete.";
    $("dcDraftRecoveryState").textContent = runtime.recovery.draft
      ? runtime.recovery.draftRepresented
        ? "Recovered local draft: its polygon is represented in the preserved queue, so no duplicate event will be created."
        : "Recovered local draft separately. It will not be submitted without confirmation."
      : "No separate local draft was present in the preserved browser database.";
    $("dcMigrationDetails").textContent = JSON.stringify({
      old_database_name: migration.old_database_name,
      new_database_name: migration.new_database_name,
      old_database_retained_read_only: migration.old_database_retained_read_only,
      source_export_sha256: migration.source_export_sha256,
      temporary_restore_sha256: migration.temporary_restore_sha256,
      initial_outbox_count: migration.initial_outbox_count,
      acknowledged_count: acknowledged,
      remaining_outbox_count: pending,
      migration_status: migration.status,
    }, null, 2);
    $("dcPreflight").classList.toggle("isHidden", !runtime.recovery.active);
    $("dcRecoverySave").classList.toggle("isHidden", !runtime.recovery.active);
    $("dcSave").classList.toggle("isHidden", runtime.recovery.active);
    syncRecoveryMutationLock();
  }

  function renderPairReview() {
    const preflight = runtime.recovery.preflight;
    const row = activeRecoveryRow();
    const pairs = preflight?.occlusion_pairs || [];
    const pair = activeRecoveryPair();
    const panel = $("dcPairReviewPanel");
    panel.classList.toggle("isHidden", !runtime.recovery.active || !pair);
    if (!row || !preflight) {
      $("dcRecoverySave").disabled = true;
      return;
    }
    const answers = explicitPairAnswers(row, preflight);
    if (pair) {
      $("dcPairTitle").textContent = "Review Person A and Person B";
      const evidence = pair.material_overlap_evidence || {};
      const relation = pair.existing_original_relation === "PERSON_A_IN_FRONT"
        ? "The original annotation placed Person A in front."
        : pair.existing_original_relation === "PERSON_B_IN_FRONT"
          ? "The original annotation placed Person B in front."
          : "The original front/back relation was not resolved.";
      $("dcPairEvidence").textContent = `Person A is cyan and Person B is magenta. ${relation} ${evidence.overlap_changed ? "The overlap changed after the outline correction." : "The original relation still requires explicit review."}`;
      $("dcPairPosition").textContent = `Pair ${runtime.recovery.pairIndex + 1} of ${pairs.length}`;
      $("dcPreviousPair").disabled = runtime.recovery.pairIndex === 0;
      $("dcNextPair").disabled = runtime.recovery.pairIndex >= pairs.length - 1;
      $("dcPairChoices").querySelectorAll("[data-dc-pair-choice]").forEach((button) => {
        button.classList.toggle("selected", button.dataset.dcPairChoice === answers[pair.dependency_id]);
      });
    }
    const initial = row.reconciliation?.initial_preflight || preflight;
    $("dcDependencyDetails").textContent = JSON.stringify({
      required_dependency_ids: preflight.required_occlusion_pair_review_ids,
      client_submitted_ids: initial.submitted_answer_ids,
      missing_ids: preflight.missing_answer_ids,
      extra_ids: initial.extra_answer_ids,
      dependency_set_hash: preflight.dependency_set_hash,
      algorithm_version: preflight.dependency_algorithm_version,
      algorithm_version_hash: preflight.dependency_algorithm_version_hash,
      normalized_polygon_hash: preflight.normalized_polygon_hash,
    }, null, 2);
    const unanswered = (preflight.required_occlusion_pair_review_ids || []).filter((id) => !answers[id]);
    const nonPairMissing = (preflight.missing_answer_ids || []).filter(
      (id) => !(preflight.required_occlusion_pair_review_ids || []).includes(id),
    );
    $("dcRecoverySave").disabled = unanswered.length > 0 || nonPairMissing.length > 0
      || (preflight.extra_answer_ids || []).length > 0;
    $("dcPreflight").disabled = false;
    renderRecoveryStatus();
  }

  function structuredRecoveryError(error) {
    const payload = error.responsePayload || {};
    return [
      payload.message || error.message,
      payload.error_code ? `Code: ${payload.error_code}.` : null,
      payload.plain_language_next_action || null,
      payload.server_saved_corrections_safe === true ? "The 13 server-saved corrections remain safe." : null,
      payload.missing_dependency_ids?.length ? `Missing: ${payload.missing_dependency_ids.join(", ")}.` : null,
      payload.extra_dependency_ids?.length ? `Extra: ${payload.extra_dependency_ids.join(", ")}.` : null,
    ].filter(Boolean).join(" ");
  }

  async function auditRecoveryQueue() {
    for (const row of runtime.recovery.rows) {
      try {
        const initial = await runtime.api("/api/review/dense-correction-preflight", {
          method: "POST",
          body: JSON.stringify(recoveryPayload(row, {rawAnswers: true})),
        });
        const answers = explicitPairAnswers(row, initial);
        row.reconciliation = {
          ...(row.reconciliation || {}),
          initial_preflight: row.reconciliation?.initial_preflight || initial,
          preflight: initial,
          pair_answers: answers,
          status: initial.already_acknowledged
            ? "SAVED_TO_SERVER"
            : initial.missing_answer_ids.length
              ? "NEEDS_OVERLAP_ANSWERS"
              : "PREFLIGHT_READY",
          audited_at: new Date().toISOString(),
        };
        await persistRecoveryRow(row);
      } catch (error) {
        row.reconciliation = {
          ...(row.reconciliation || {}),
          status: error.httpStatus ? "SERVER_REJECTED_REVIEW_REQUIRED" : "OFFLINE_SAFELY_STORED",
          structured_error: error.responsePayload || {message: error.message},
          audited_at: new Date().toISOString(),
        };
        await persistRecoveryRow(row);
      }
    }
  }

  async function hydrateRecoveryRow() {
    const row = activeRecoveryRow();
    if (!row) return;
    const nextIndex = runtime.items.findIndex((candidate) => (
      candidate.caseData.case_id === row.payload.case_id
      && candidate.item.original_mask_uuid === row.payload.original_mask_uuid
    ));
    if (nextIndex < 0) {
      row.reconciliation = {
        ...(row.reconciliation || {}),
        status: "QUARANTINED_EXPORT_AVAILABLE",
        structured_error: {message: "The pending event is outside the immutable repair manifest."},
      };
      runtime.recovery.quarantined.push(row.id);
      await persistRecoveryRow(row);
      renderRecoveryStatus();
      return;
    }
    runtime.index = nextIndex;
    runtime.candidateIndex = 0;
    runtime.points = (row.payload.corrected_polygon_original_pixels || []).map((point) => ({
      x: Number(point.x), y: Number(point.y),
    }));
    runtime.drawing = false;
    runtime.preview = null;
    runtime.previewIssue = null;
    runtime.coverageValues = Object.fromEntries((row.payload.candidate_coverage_reviews || []).map((review) => [
      review.candidate_uuid,
      review.candidate_visible_mask_coverage == null ? "" : String(review.candidate_visible_mask_coverage),
    ]));
    runtime.view = "focal";
    runtime.pendingViewState = null;
    $("dcQuality").value = row.payload.mask_quality || item().original_mask_quality;
    $("dcUnreliable").checked = row.payload.decision === "UNRELIABLE_OUTLINE";
    $("dcUnreliableReason").disabled = true;
    $("dcUnreliableReason").value = row.payload.unreliable_reason || "";
    $("dcFocalView").classList.add("active");
    $("dcPanoramaView").classList.remove("active");
    updateProgress();
    renderCoverageRows();
    renderOcclusionRows();
    runtime.recovery.preflight = row.reconciliation?.preflight || null;
    runtime.recovery.pairIndex = 0;
    await loadEvidence();
    renderPairReview();
    renderRecoveryStatus();
  }

  async function refreshRecoveryRows() {
    runtime.recovery.rows = (await idbAll("outbox")).sort(
      (left, right) => left.createdAt.localeCompare(right.createdAt) || left.id.localeCompare(right.id),
    );
    runtime.recovery.index = 0;
    runtime.recovery.active = runtime.recovery.rows.length > 0;
  }

  async function removeAcknowledgedRecoveryRow(row, response, disposition) {
    await idbDelete("outbox", row.id);
    const remaining = await idbAll("outbox");
    const draftPolygon = canonicalPolygonKey(runtime.recovery.draft?.points || []);
    if (runtime.recovery.draft && !remaining.some((candidate) => (
      candidate.draftKey === runtime.recovery.draft.key
      && canonicalPolygonKey(candidate.payload?.corrected_polygon_original_pixels || []) === draftPolygon
    ))) {
      await idbDelete("drafts", runtime.recovery.draft.key);
      runtime.recovery.draft = null;
    }
    await recordRecoveryAcknowledgement(row, response, disposition);
    await refreshRecoveryRows();
    if (runtime.recovery.active) {
      await hydrateRecoveryRow();
      await runCurrentRecoveryPreflight();
    } else {
      runtime.recovery.preflight = null;
      renderPairReview();
      renderRecoveryStatus();
      runtime.state = await runtime.api("/api/review/state");
      await selectIndex(nextIncompleteIndex());
    }
  }

  async function runCurrentRecoveryPreflight() {
    const row = activeRecoveryRow();
    if (!row) return;
    $("dcError").classList.add("isHidden");
    row.reconciliation = {...(row.reconciliation || {}), status: "WAITING_FOR_SERVER"};
    await persistRecoveryRow(row);
    renderRecoveryStatus();
    try {
      const preflight = await runtime.api("/api/review/dense-correction-preflight", {
        method: "POST",
        body: JSON.stringify(recoveryPayload(row)),
      });
      row.reconciliation = {
        ...(row.reconciliation || {}),
        preflight,
        pair_answers: explicitPairAnswers(row, preflight),
        status: preflight.already_acknowledged
          ? "SAVED_TO_SERVER"
          : preflight.missing_answer_ids.length
            ? "NEEDS_OVERLAP_ANSWERS"
            : "PREFLIGHT_READY",
        preflighted_at: new Date().toISOString(),
      };
      await persistRecoveryRow(row);
      runtime.recovery.preflight = preflight;
      if (preflight.already_acknowledged) {
        await removeAcknowledgedRecoveryRow(row, preflight, "ALREADY_ACKNOWLEDGED");
        return;
      }
      renderPairReview();
      renderOverlay();
    } catch (error) {
      const rejected = Boolean(error.httpStatus);
      row.reconciliation = {
        ...(row.reconciliation || {}),
        status: rejected ? "SERVER_REJECTED_REVIEW_REQUIRED" : "OFFLINE_SAFELY_STORED",
        structured_error: error.responsePayload || {message: error.message},
      };
      await persistRecoveryRow(row);
      $("dcError").textContent = structuredRecoveryError(error);
      $("dcError").classList.remove("isHidden");
      renderPairReview();
      renderRecoveryStatus();
    }
  }

  async function chooseRecoveryPair(choice) {
    const row = activeRecoveryRow();
    const pair = activeRecoveryPair();
    if (!row || !pair) return;
    row.reconciliation = {
      ...(row.reconciliation || {}),
      answer_revision_id: row.reconciliation?.answer_revision_id || uuid(),
      pair_answers: {
        ...explicitPairAnswers(row, runtime.recovery.preflight),
        [pair.dependency_id]: choice,
      },
      status: "WAITING_FOR_SERVER",
      answer_updated_at: new Date().toISOString(),
    };
    await persistRecoveryRow(row);
    await runCurrentRecoveryPreflight();
  }

  async function saveReconciledRecoveryItem() {
    const row = activeRecoveryRow();
    if (!row) return;
    $("dcRecoverySave").disabled = true;
    try {
      runtime.state = await runtime.api("/api/review/state");
      const preflight = await runtime.api("/api/review/dense-correction-preflight", {
        method: "POST",
        body: JSON.stringify(recoveryPayload(row)),
      });
      row.reconciliation = {...(row.reconciliation || {}), preflight};
      await persistRecoveryRow(row);
      runtime.recovery.preflight = preflight;
      if (preflight.already_acknowledged) {
        await removeAcknowledgedRecoveryRow(row, preflight, "ALREADY_ACKNOWLEDGED");
        return;
      }
      if (!preflight.saveable) {
        row.reconciliation.status = "NEEDS_OVERLAP_ANSWERS";
        await persistRecoveryRow(row);
        renderPairReview();
        return;
      }
      const response = await runtime.api("/api/review/dense-correction-event", {
        method: "POST",
        body: JSON.stringify(recoveryPayload(row, {final: true})),
      });
      runtime.state = response;
      await removeAcknowledgedRecoveryRow(
        row,
        response,
        response.duplicate_event ? "DUPLICATE_ACKNOWLEDGED" : "SERVER_ACKNOWLEDGED",
      );
    } catch (error) {
      row.reconciliation = {
        ...(row.reconciliation || {}),
        status: error.httpStatus ? "SERVER_REJECTED_REVIEW_REQUIRED" : "OFFLINE_SAFELY_STORED",
        structured_error: error.responsePayload || {message: error.message},
      };
      await persistRecoveryRow(row);
      $("dcError").textContent = structuredRecoveryError(error);
      $("dcError").classList.remove("isHidden");
      renderPairReview();
      renderRecoveryStatus();
    }
  }

  function focalContainmentValid() {
    return runtime.points.length >= 3 && runtime.points.every(insideFocalRoi);
  }

  function correctionValid() {
    if ($("dcUnreliable").checked) return Boolean($("dcUnreliableReason").value);
    return !runtime.drawing && runtime.points.length >= 3
      && validateSimplePolygon(runtime.points).valid && focalContainmentValid();
  }

  function coverageComplete() {
    if ($("dcUnreliable").checked) return true;
    return candidateRows().every((row) => runtime.coverageValues[row.candidate_uuid] !== undefined
      && runtime.coverageValues[row.candidate_uuid] !== "");
  }

  function coverageBindingsValid() {
    return candidateRows().every((row) => candidateBindingValid(row));
  }

  function occlusionComplete() {
    if (runtime.recovery.enabled) return true;
    if ($("dcUnreliable").checked) return true;
    return Array.from(document.querySelectorAll("[data-dc-occlusion]")).every((select) => select.value !== "");
  }

  function updateGeometryFeedback() {
    $("dcVertexCount").textContent = `${runtime.points.length} ${runtime.points.length === 1 ? "vertex" : "vertices"}`;
    const closure = runtime.points.length >= 3 ? validateClosingSegment(runtime.points) : null;
    const reason = runtime.previewIssue?.reason
      || (runtime.drawing && closure && !closure.valid ? closure.reason : null)
      || (runtime.drawing ? "Continue in one direction, then finish the outline." : "Outline drawing is closed.");
    $("dcGeometryReason").textContent = reason;
    $("dcGeometryReason").classList.toggle("isInvalid", Boolean(runtime.previewIssue || (runtime.drawing && closure && !closure.valid)));

    const polygon = validateSimplePolygon(runtime.points);
    const checks = [
      ["Simple polygon", !runtime.drawing && polygon.valid],
      ["Inside focal ROI", focalContainmentValid()],
      ["Source hash valid", runtime.evidenceBindingValid],
      ["Machine coverage reviewed", coverageComplete() && coverageBindingsValid()],
    ];
    const ready = correctionValid() && coverageComplete() && coverageBindingsValid() && occlusionComplete() && !runtime.evidenceBlocked;
    $("dcValidationSummary").innerHTML = `<strong>Validation</strong>${checks.map(([label, passed]) => `<span class="${passed ? "pass" : "fail"}">${passed ? "Pass" : "Pending"}: ${escapeHtml(label)}</span>`).join("")}<span class="${ready ? "pass" : "fail"}">${ready ? "Ready to save" : "Not ready to save"}</span>`;
  }

  function saveBlockers() {
    const blockers = [];
    if (runtime.evidenceBlocked || !runtime.evidenceBindingValid) blockers.push("verified source evidence is unavailable");
    if (!correctionValid()) blockers.push($("dcUnreliable").checked ? "choose an unresolved-outline reason" : "finish a valid simple outline inside the focal ROI");
    if (!coverageBindingsValid()) blockers.push("a machine-box source binding is invalid");
    if (!coverageComplete()) blockers.push("review every machine-box coverage value");
    if (!occlusionComplete()) blockers.push("finish the required overlap review");
    return blockers;
  }

  function updateSaveGate() {
    if (!runtime.state || !current()) return;
    const closure = runtime.points.length >= 3 ? validateClosingSegment(runtime.points) : null;
    $("dcFinish").disabled = recoveryMutationLocked()
      || !runtime.drawing
      || runtime.points.length < 3
      || !closure?.valid
      || runtime.invalidPreview;
    const blockers = saveBlockers();
    $("dcSave").disabled = blockers.length !== 0;
    $("dcSaveReason").textContent = blockers.length ? `Save disabled: ${blockers[0]}.` : "All checks pass. This outline is ready to save.";
    $("dcSaveReason").classList.toggle("isBlocked", blockers.length !== 0);
    const complete = Object.keys(runtime.state.corrections || {}).length === runtime.items.length;
    $("dcComplete").disabled = !complete || runtime.evidenceBlocked || runtime.recovery.active;
    if (runtime.recovery.enabled) {
      $("dcSave").classList.toggle("isHidden", runtime.recovery.active);
      $("dcRecoverySave").classList.toggle("isHidden", !runtime.recovery.active);
      $("dcPreflight").classList.toggle("isHidden", !runtime.recovery.active);
    }
  }

  function viewState() {
    return {
      view: runtime.view,
      scale: runtime.transform.scale,
      translateX: runtime.transform.translateX,
      translateY: runtime.transform.translateY,
    };
  }

  async function persistSession() {
    if (!runtime.db || !current()) return;
    await idbPut("session", {
      key: "navigation",
      itemIndex: runtime.index,
      candidateIndex: runtime.candidateIndex,
      caseId: caseData().case_id,
      maskUuid: item().original_mask_uuid,
      updatedAt: new Date().toISOString(),
    });
  }

  async function persistDraft() {
    if (!runtime.db || !current()) return;
    if (runtime.recovery.active) {
      await persistSession();
      $("dcSaveState").textContent = "Recovered polygon locked until server acknowledgement";
      return;
    }
    await idbPut("drafts", {
      key: draftKey(),
      points: runtime.points,
      drawing: runtime.drawing,
      quality: $("dcQuality").value,
      unreliable: $("dcUnreliable").checked,
      unreliableReason: $("dcUnreliableReason").value,
      coverage: {...runtime.coverageValues},
      occlusion: Object.fromEntries(Array.from(document.querySelectorAll("[data-dc-occlusion]")).map((select) => [select.dataset.dcOcclusion, select.value])),
      candidateIndex: runtime.candidateIndex,
      viewState: viewState(),
      savedAt: new Date().toISOString(),
    });
    await persistSession();
    $("dcSaveState").textContent = "Draft stored locally";
  }

  async function hydrateItem() {
    const saved = runtime.state.corrections?.[item().original_mask_uuid] || null;
    const draft = await idbGet("drafts", draftKey());
    const source = draft || (saved ? {
      points: saved.corrected_polygon_original_pixels || [],
      drawing: false,
      quality: saved.mask_quality,
      unreliable: saved.decision === "UNRELIABLE_OUTLINE",
      unreliableReason: saved.unreliable_reason || "",
      coverage: Object.fromEntries((saved.candidate_coverage_reviews || []).map((row) => [row.candidate_uuid, row.candidate_visible_mask_coverage == null ? "" : String(row.candidate_visible_mask_coverage)])),
      occlusion: Object.fromEntries((saved.occlusion_reviews || []).map((row) => [row.other_mask_uuid, row.status])),
      candidateIndex: 0,
      viewState: {view: "focal"},
    } : null);
    runtime.points = (source?.points || []).map((point) => ({x: Number(point.x), y: Number(point.y)}));
    runtime.drawing = Boolean(source?.drawing);
    runtime.preview = null;
    runtime.previewIssue = null;
    runtime.coverageValues = {...(source?.coverage || {})};
    runtime.candidateIndex = Math.max(0, Math.min(candidateRows().length - 1, Number(source?.candidateIndex) || 0));
    runtime.view = source?.viewState?.view === "panorama" ? "panorama" : "focal";
    runtime.pendingViewState = source?.viewState || null;
    $("dcFocalView").classList.toggle("active", runtime.view === "focal");
    $("dcPanoramaView").classList.toggle("active", runtime.view === "panorama");
    $("dcQuality").value = source?.quality || item().original_mask_quality;
    $("dcUnreliable").checked = Boolean(source?.unreliable);
    $("dcUnreliableReason").disabled = recoveryMutationLocked() || !$("dcUnreliable").checked;
    $("dcUnreliableReason").value = source?.unreliableReason || "";
    renderCoverageRows();
    renderOcclusionRows(saved);
    Object.entries(source?.occlusion || {}).forEach(([other, value]) => {
      const select = document.querySelector(`[data-dc-occlusion="${CSS.escape(other)}"]`);
      if (select) select.value = value;
    });
  }

  function updateProgress() {
    const corrections = runtime.state.corrections || {};
    const savedMasks = Object.keys(corrections).length;
    const caseIndex = runtime.manifest.cases.findIndex((row) => row.case_id === caseData().case_id);
    const completedCases = runtime.manifest.cases.filter((row) => row.visible_metadata.repair_items.every(
      (repairItem) => Boolean(corrections[repairItem.original_mask_uuid]),
    )).length;
    const remainingGeometryReviews = runtime.items.reduce((total, row) => {
      if (corrections[row.item.original_mask_uuid]) return total;
      const coverage = (row.item.affected_candidates || []).length;
      const inconsistentOcclusion = (row.item.occlusion_dependencies || []).filter(
        (dependency) => dependency.original_graph_inconsistent,
      ).length;
      return total + coverage + inconsistentOcclusion;
    }, 0);
    const inCase = caseData().visible_metadata.repair_items.findIndex((row) => row.original_mask_uuid === item().original_mask_uuid);
    $("dcCaseProgress").textContent = `${completedCases} of ${runtime.manifest.cases.length} affected cases complete`;
    $("dcMaskProgress").textContent = `${savedMasks} of ${runtime.items.length} outlines saved | ${remainingGeometryReviews} geometry reviews remain`;
    $("dcProgressBar").style.width = `${100 * savedMasks / runtime.items.length}%`;
    $("dcCaseTitle").textContent = `Dense case ${caseIndex + 1}`;
    $("dcMaskLabel").textContent = `Outline ${inCase + 1} of ${caseData().visible_metadata.repair_items.length}`;
    $("dcPreviousMask").disabled = recoveryMutationLocked() || runtime.index === 0;
    $("dcNextMask").disabled = recoveryMutationLocked() || runtime.index === runtime.items.length - 1;
  }

  async function selectIndex(next) {
    if (recoveryMutationLocked()) return;
    runtime.index = Math.max(0, Math.min(runtime.items.length - 1, next));
    runtime.candidateIndex = 0;
    runtime.preview = null;
    runtime.previewIssue = null;
    runtime.machineExplicitlyFocused = false;
    updateProgress();
    await hydrateItem();
    await loadEvidence();
    await persistSession();
  }

  function nextIncompleteIndex() {
    const corrections = runtime.state.corrections || {};
    for (let offset = 1; offset <= runtime.items.length; offset += 1) {
      const index = (runtime.index + offset) % runtime.items.length;
      if (!corrections[runtime.items[index].item.original_mask_uuid]) return index;
    }
    return runtime.index;
  }

  function coveragePayload(unreliable) {
    return candidateRows().map((row) => ({
      candidate_uuid: row.candidate_uuid,
      review_status: unreliable ? "EVIDENCE_UNRESOLVED" : "REVALIDATED",
      candidate_visible_mask_coverage: unreliable ? null : Number(runtime.coverageValues[row.candidate_uuid]),
    }));
  }

  function occlusionPayload(unreliable) {
    return materialDependencies().map((row) => ({
      other_mask_uuid: row.other_mask_uuid,
      status: unreliable ? "UNRESOLVED" : document.querySelector(`[data-dc-occlusion="${CSS.escape(row.other_mask_uuid)}"]`).value,
    }));
  }

  function eventPayload() {
    const unreliable = $("dcUnreliable").checked;
    const id = uuid();
    return {
      event_type: "DENSE_MASK_CORRECTION_SAVED",
      review_id: runtime.manifest.review_id,
      reviewer_session_id: runtime.uiConfig.question_contract.reviewer_session_id,
      case_id: caseData().case_id,
      original_mask_uuid: item().original_mask_uuid,
      source_frame_sha256: binding().source_frame_sha256,
      focal_transform_hash: binding().focal_transform_hash,
      original_polygon_hash: item().original_polygon_hash,
      decision: unreliable ? "UNRELIABLE_OUTLINE" : "CORRECTED_OUTLINE",
      corrected_polygon_original_pixels: unreliable ? null : runtime.points,
      mask_quality: unreliable ? "UNCERTAIN" : $("dcQuality").value,
      correction_reason: "SELF_INTERSECTION_REPAIR",
      unreliable_reason: unreliable ? $("dcUnreliableReason").value : null,
      candidate_coverage_reviews: coveragePayload(unreliable),
      occlusion_reviews: occlusionPayload(unreliable),
      client_event_id: id,
      idempotency_key: `${runtime.manifest.review_id}:${item().original_mask_uuid}:${id}`,
      expected_server_state_hash: runtime.state.server_state_hash || null,
      elapsed_active_seconds: elapsedSeconds(),
      input_source: "dense_mask_correction_ui",
      client_build_id: runtime.uiConfig.question_contract.client_build_id || null,
    };
  }

  async function deliverOutbox(row) {
    const response = await runtime.api("/api/review/dense-correction-event", {method: "POST", body: JSON.stringify(row.payload)});
    runtime.state = response;
    await idbDelete("outbox", row.id);
    await idbDelete("drafts", row.draftKey);
    $("dcSaveState").textContent = `Saved to server | pending ${(await idbAll("outbox")).length}`;
  }

  async function flushOutbox() {
    const rows = (await idbAll("outbox")).sort((left, right) => left.createdAt.localeCompare(right.createdAt));
    for (const row of rows) await deliverOutbox(row);
  }

  async function saveCorrection() {
    updateSaveGate();
    if ($("dcSave").disabled) return;
    $("dcError").classList.add("isHidden");
    const payload = eventPayload();
    const row = {id: payload.idempotency_key, draftKey: draftKey(), payload, createdAt: new Date().toISOString()};
    await idbPut("outbox", row);
    $("dcSaveState").textContent = "Queued locally | pending 1";
    if (runtime.recovery.enabled) {
      await refreshRecoveryRows();
      await auditRecoveryQueue();
      await hydrateRecoveryRow();
      await runCurrentRecoveryPreflight();
      return;
    }
    try {
      await deliverOutbox(row);
      updateProgress();
      await selectIndex(nextIncompleteIndex());
    } catch (error) {
      $("dcError").textContent = `Saved locally; server acknowledgement pending. ${error.message}`;
      $("dcError").classList.remove("isHidden");
      $("dcSaveState").textContent = `Offline queue | pending ${(await idbAll("outbox")).length}`;
    }
  }

  async function completeRepair() {
    try {
      if (runtime.recovery.enabled && runtime.recovery.active) {
        throw new Error("locally preserved events still require server acknowledgement");
      }
      if (!runtime.recovery.enabled) await flushOutbox();
      const pending = (await idbAll("outbox")).length;
      const drafts = (await idbAll("drafts")).length;
      const id = uuid();
      runtime.state = await runtime.api("/api/review/dense-correction-complete", {
        method: "POST",
        body: JSON.stringify({
          review_id: runtime.manifest.review_id,
          client_event_id: id,
          idempotency_key: `${runtime.manifest.review_id}:complete:${id}`,
          expected_server_state_hash: runtime.state.server_state_hash,
          pending_outbox_events: pending,
          unresolved_draft_count: drafts,
          elapsed_active_seconds: elapsedSeconds(),
          input_source: "dense_mask_correction_ui",
          client_build_id: runtime.uiConfig.question_contract.client_build_id || null,
        }),
      });
      $("dcSaveState").textContent = "Repair completed and validated";
      $("dcComplete").disabled = true;
    } catch (error) {
      $("dcError").textContent = `Completion blocked: ${error.message}`;
      $("dcError").classList.remove("isHidden");
    }
  }

  function beginRedraw() {
    if (recoveryMutationLocked()) return;
    runtime.points = [];
    runtime.preview = null;
    runtime.previewIssue = null;
    runtime.drawing = true;
    $("dcUnreliable").checked = false;
    $("dcUnreliableReason").disabled = true;
    renderOverlay();
    persistDraft();
  }

  function removeLastPoint() {
    if (recoveryMutationLocked()) return;
    runtime.points.pop();
    runtime.drawing = true;
    runtime.preview = null;
    runtime.previewIssue = null;
    renderOverlay();
    persistDraft();
  }

  function clearOutline() {
    if (recoveryMutationLocked()) return;
    runtime.points = [];
    runtime.preview = null;
    runtime.previewIssue = null;
    runtime.drawing = true;
    renderOverlay();
    persistDraft();
  }

  function updateTransformInspector() {
    if (!current()) return;
    const probe = activePersonBounds();
    const point = probe ? {x: (probe.x1 + probe.x2) / 2, y: (probe.y1 + probe.y2) / 2} : {x: 0, y: 0};
    const roundTrip = sourceRoundTrip(point);
    $("dcTransformInspector").textContent = JSON.stringify({
      client_build_id: runtime.uiConfig.question_contract.client_build_id || null,
      coordinate_space: "SOURCE_IMAGE_PIXELS",
      view: runtime.view,
      source_origin: runtime.view === "focal" ? {x: focalBounds().x1, y: focalBounds().y1} : {x: 0, y: 0},
      image_size: {width: runtime.transform.naturalWidth, height: runtime.transform.naturalHeight},
      viewport_size: {width: runtime.transform.viewportWidth, height: runtime.transform.viewportHeight},
      scale: runtime.transform.scale,
      translate: {x: runtime.transform.translateX, y: runtime.transform.translateY},
      roundtrip_error_pixels: roundTrip.error,
      source_frame_sha256: binding().source_frame_sha256,
    }, null, 2);
  }

  function toggleMachineBox() {
    runtime.machineExplicitlyFocused = false;
    renderOverlay();
  }

  function bind() {
    $("dcFocalView").addEventListener("click", () => switchView("focal"));
    $("dcPanoramaView").addEventListener("click", () => switchView("panorama"));
    $("dcZoomOut").addEventListener("click", () => zoomAt(0.8));
    $("dcZoomIn").addEventListener("click", () => zoomAt(1.25));
    $("dcFit").addEventListener("click", fitImage);
    $("dcFocusPerson").addEventListener("click", focusPerson);
    $("dcFocusTogether").addEventListener("click", focusPersonAndCandidate);
    $("dcShowMachineBox").addEventListener("change", toggleMachineBox);
    $("dcCompareOriginal").addEventListener("change", renderOverlay);
    $("dcShowContextLabels").addEventListener("change", renderOverlay);
    $("dcRedraw").addEventListener("click", beginRedraw);
    $("dcFinish").addEventListener("click", () => {
      if (recoveryMutationLocked()) return;
      const closure = validateClosingSegment(runtime.points);
      const polygon = validateSimplePolygon(runtime.points);
      if (!closure.valid || !polygon.valid) {
        runtime.previewIssue = closure.valid ? polygon : closure;
        renderOverlay();
        return;
      }
      runtime.drawing = false;
      runtime.preview = null;
      runtime.previewIssue = null;
      renderOverlay();
      persistDraft();
    });
    $("dcUndo").addEventListener("click", removeLastPoint);
    $("dcClear").addEventListener("click", clearOutline);
    $("dcQuality").addEventListener("change", () => {
      if (!recoveryMutationLocked()) persistDraft();
    });
    $("dcUnreliable").addEventListener("change", () => {
      if (recoveryMutationLocked()) return;
      $("dcUnreliableReason").disabled = !$("dcUnreliable").checked;
      renderCoverageRows();
      renderOcclusionRows();
      persistDraft();
      renderOverlay();
    });
    $("dcUnreliableReason").addEventListener("change", () => {
      if (recoveryMutationLocked()) return;
      persistDraft();
      updateSaveGate();
    });
    $("dcPreviousMask").addEventListener("click", () => selectIndex(runtime.index - 1));
    $("dcNextMask").addEventListener("click", () => selectIndex(runtime.index + 1));
    $("dcPreviousCandidate").addEventListener("click", () => selectCandidate(runtime.candidateIndex - 1));
    $("dcNextCandidate").addEventListener("click", () => selectCandidate(runtime.candidateIndex + 1));
    $("dcSave").addEventListener("click", saveCorrection);
    $("dcPreflight").addEventListener("click", runCurrentRecoveryPreflight);
    $("dcRecoverySave").addEventListener("click", saveReconciledRecoveryItem);
    $("dcExportRecovery").addEventListener("click", downloadRecoveryExport);
    $("dcPairChoices").querySelectorAll("[data-dc-pair-choice]").forEach((button) => {
      button.addEventListener("click", () => chooseRecoveryPair(button.dataset.dcPairChoice));
    });
    $("dcPreviousPair").addEventListener("click", () => {
      runtime.recovery.pairIndex = Math.max(0, runtime.recovery.pairIndex - 1);
      renderPairReview();
      renderOverlay();
    });
    $("dcNextPair").addEventListener("click", () => {
      const count = runtime.recovery.preflight?.occlusion_pairs?.length || 0;
      runtime.recovery.pairIndex = Math.min(Math.max(0, count - 1), runtime.recovery.pairIndex + 1);
      renderPairReview();
      renderOverlay();
    });
    $("dcComplete").addEventListener("click", completeRepair);
    $("dcShortcutHelp").addEventListener("click", () => $("dcHelpDialog").showModal());
    $("dcViewport").addEventListener("wheel", (event) => {
      event.preventDefault();
      zoomAt(event.deltaY < 0 ? 1.12 : 0.89, event.clientX, event.clientY);
    }, {passive: false});
    $("dcViewport").addEventListener("pointerdown", (event) => {
      if (recoveryMutationLocked() && event.button === 0 && !event.shiftKey) return;
      if (event.button === 1 || event.shiftKey || !runtime.drawing) {
        runtime.pan = {
          x: event.clientX,
          y: event.clientY,
          tx: runtime.transform.translateX,
          ty: runtime.transform.translateY,
        };
        runtime.machineExplicitlyFocused = false;
        $("dcViewport").classList.add("isPanning");
        $("dcViewport").setPointerCapture(event.pointerId);
        return;
      }
      if (event.button !== 0 || $("dcUnreliable").checked) return;
      const point = eventPoint(event);
      const validation = validateOpenSegment(runtime.points, point, insideDisplayedImage);
      if (!insideFocalRoi(point)) {
        runtime.preview = point;
        runtime.previewIssue = {valid: false, kind: "OUTSIDE_FOCAL_ROI", point, reason: geometryReason("OUTSIDE_FOCAL_ROI")};
      } else if (!validation.valid) {
        runtime.preview = point;
        runtime.previewIssue = validation;
      } else {
        runtime.points.push(point);
        runtime.preview = null;
        runtime.previewIssue = null;
        persistDraft();
      }
      renderOverlay();
    });
    $("dcViewport").addEventListener("pointermove", (event) => {
      if (runtime.pan) {
        runtime.transform.translateX = runtime.pan.tx + event.clientX - runtime.pan.x;
        runtime.transform.translateY = runtime.pan.ty + event.clientY - runtime.pan.y;
        constrainTransform();
        stageTransform();
        renderOverlay();
        return;
      }
      if (!runtime.drawing || !runtime.points.length) return;
      runtime.preview = eventPoint(event);
      runtime.previewIssue = !insideFocalRoi(runtime.preview)
        ? {valid: false, kind: "OUTSIDE_FOCAL_ROI", point: runtime.preview, reason: geometryReason("OUTSIDE_FOCAL_ROI")}
        : validateOpenSegment(runtime.points, runtime.preview, insideDisplayedImage).valid
          ? null
          : validateOpenSegment(runtime.points, runtime.preview, insideDisplayedImage);
      renderOverlay();
    });
    const stopPan = () => { runtime.pan = null; $("dcViewport").classList.remove("isPanning"); };
    $("dcViewport").addEventListener("pointerup", stopPan);
    $("dcViewport").addEventListener("pointercancel", stopPan);
    window.addEventListener("resize", () => {
      fitImage();
      updateTransformInspector();
    });
    document.addEventListener("keydown", (event) => {
      const target = event.target;
      if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement) return;
      if (event.key === "?") $("dcHelpDialog").showModal();
      else if (event.key === "Backspace") {
        event.preventDefault();
        if (!recoveryMutationLocked()) removeLastPoint();
      } else if (event.key.toLowerCase() === "c") {
        if (!recoveryMutationLocked()) clearOutline();
      }
      else if (event.key.toLowerCase() === "f" && event.shiftKey) focusPersonAndCandidate();
      else if (event.key.toLowerCase() === "f") focusPerson();
      else if (event.key.toLowerCase() === "m") {
        $("dcShowMachineBox").checked = !$("dcShowMachineBox").checked;
        toggleMachineBox();
      } else if (event.key.toLowerCase() === "l") {
        $("dcShowContextLabels").checked = !$("dcShowContextLabels").checked;
        renderOverlay();
      } else if (event.key === "[") selectCandidate(runtime.candidateIndex - 1);
      else if (event.key === "]") selectCandidate(runtime.candidateIndex + 1);
    });
  }

  async function mount({manifest, uiConfig, state, api}) {
    runtime.manifest = manifest;
    runtime.uiConfig = uiConfig;
    runtime.state = state;
    runtime.api = api;
    runtime.recovery.enabled = uiConfig.question_contract.client_build_id === RECOVERY_BUILD_ID
      || uiConfig.question_contract.pending_recovery_mode === true;
    runtime.items = manifest.cases.flatMap((row) => row.visible_metadata.repair_items.map((repairItem) => ({caseData: row, item: repairItem})));
    runtime.db = await openDatabase();
    bind();
    if (runtime.recovery.enabled) {
      try {
        await initializeRecoveryMigration();
        runtime.state = await api("/api/review/state");
        await auditRecoveryQueue();
        await refreshRecoveryRows();
        if (runtime.recovery.active) {
          await hydrateRecoveryRow();
          await runCurrentRecoveryPreflight();
          return;
        }
      } catch (error) {
        runtime.recovery.active = true;
        $("dcError").textContent = `Recovery stopped safely. ${structuredRecoveryError(error)}`;
        $("dcError").classList.remove("isHidden");
        renderRecoveryStatus();
        return;
      }
    }
    try {
      await flushOutbox();
      runtime.state = await api("/api/review/state");
    } catch (error) {
      $("dcSaveState").textContent = `Offline recovery | ${error.message}`;
    }
    const serverResume = Object.keys(runtime.state.corrections || {}).length < runtime.items.length
      ? runtime.items.findIndex((row) => !runtime.state.corrections?.[row.item.original_mask_uuid])
      : 0;
    const session = await idbGet("session", "navigation");
    const sessionIndex = Number(session?.itemIndex);
    const resume = Number.isInteger(sessionIndex) && sessionIndex >= 0 && sessionIndex < runtime.items.length
      ? sessionIndex
      : Math.max(0, serverResume);
    await selectIndex(resume);
  }

  window.DenseMaskCorrection = {
    mount,
    debug: {
      GEOMETRY_EPSILON,
      MARKER_SCREEN_RADIUS_CSS,
      classifySegmentIntersection,
      candidateCrosses,
      screenConstantMarkerRadius: (cssRadius, scale) => screenConstantMarkerRadius(cssRadius, scale),
      validateOpenSegment,
      validateClosingSegment,
      validateSimplePolygon,
      sourceRoundTrip: (point) => sourceRoundTrip(point),
      sourceToViewport: (point) => sourceToViewport(point),
      viewportToSource: (point) => viewportToSource(point),
      selectIndex: (index) => selectIndex(index),
      selectCandidate: (index) => selectCandidate(index),
      focusPerson: () => focusPerson(),
      focusPersonAndCandidate: () => focusPersonAndCandidate(),
      fitImage: () => fitImage(),
      setZoomScaleForTest: (scale) => {
        const requestedScale = Number(scale);
        if (!Number.isFinite(requestedScale) || requestedScale <= 0) {
          throw new RangeError("test zoom scale must be finite and positive");
        }
        zoomAt(requestedScale / runtime.transform.scale);
        return runtime.transform.scale;
      },
      snapshot: () => ({
        index: runtime.index,
        candidateIndex: runtime.candidateIndex,
        coverageValues: {...runtime.coverageValues},
        points: runtime.points.map((point) => ({...point})),
        drawing: runtime.drawing,
        view: runtime.view,
        transform: {...runtime.transform},
        machineRendered: runtime.machineRendered,
        machineInViewport: runtime.machineInViewport,
        coverageAvailable: candidateCoverageAvailable(),
        evidenceBindingValid: runtime.evidenceBindingValid,
        recovery: {
          enabled: runtime.recovery.enabled,
          active: runtime.recovery.active,
          pendingCount: runtime.recovery.rows.length,
          questionCount: recoveryQuestionCount(),
          pairIndex: runtime.recovery.pairIndex,
          preflight: runtime.recovery.preflight ? structuredClone(runtime.recovery.preflight) : null,
          migration: runtime.recovery.migration ? structuredClone(runtime.recovery.migration) : null,
          draftRepresented: runtime.recovery.draftRepresented,
          currentStatus: activeRecoveryRow()?.reconciliation?.status || null,
        },
      }),
      runRecoveryPreflight: () => runCurrentRecoveryPreflight(),
      saveRecoveredItem: () => saveReconciledRecoveryItem(),
      chooseRecoveryPair: (choice) => chooseRecoveryPair(choice),
      downloadRecoveryExport: () => downloadRecoveryExport(),
    },
  };
})();
