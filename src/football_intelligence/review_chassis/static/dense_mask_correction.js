(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const SVG_NS = "http://www.w3.org/2000/svg";
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
    points: [],
    preview: null,
    invalidPreview: false,
    drawing: false,
    evidenceBlocked: true,
    view: "focal",
    scale: 1,
    translateX: 0,
    translateY: 0,
    naturalWidth: 1,
    naturalHeight: 1,
    pan: null,
    startedAt: performance.now(),
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
  function draftKey() { return `${caseData().case_id}:${item().original_mask_uuid}`; }
  function elapsedSeconds() { return Math.max(0, Math.round((performance.now() - runtime.startedAt) / 1000)); }

  function openDatabase() {
    return new Promise((resolve, reject) => {
      const name = runtime.uiConfig.question_contract.indexeddb_namespace;
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

  async function sha256(buffer) {
    const digest = await crypto.subtle.digest("SHA-256", buffer);
    return Array.from(new Uint8Array(digest)).map((value) => value.toString(16).padStart(2, "0")).join("");
  }

  function asset(type) {
    return caseData().evidence_assets.find((row) => row.asset_type === type);
  }

  async function loadEvidence() {
    runtime.evidenceBlocked = true;
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
      const objectUrl = URL.createObjectURL(new Blob([bytes], {type: selected.media_type}));
      const image = $("dcBaseImage");
      image.src = objectUrl;
      await image.decode();
      if (!image.naturalWidth || !image.naturalHeight) throw new Error("image decoded with zero dimensions");
      runtime.naturalWidth = image.naturalWidth;
      runtime.naturalHeight = image.naturalHeight;
      $("dcStage").style.width = `${image.naturalWidth}px`;
      $("dcStage").style.height = `${image.naturalHeight}px`;
      $("dcOverlay").setAttribute("viewBox", `0 0 ${image.naturalWidth} ${image.naturalHeight}`);
      runtime.evidenceBlocked = false;
      $("dcEvidenceStatus").textContent = `Evidence verified: ${image.naturalWidth} x ${image.naturalHeight}, source hash bound`;
      fitImage();
      renderOverlay();
    } catch (error) {
      runtime.evidenceBlocked = true;
      $("dcEvidenceBlocker").textContent = `Evidence unavailable. ${error.message}`;
      $("dcEvidenceBlocker").classList.remove("isHidden");
      $("dcEvidenceStatus").textContent = "Evidence verification failed";
    }
    updateSaveGate();
  }

  function sourceToView(point) {
    const roi = binding().focal_roi_original_pixels;
    return runtime.view === "focal"
      ? {x: Number(point.x) - Number(roi.x1), y: Number(point.y) - Number(roi.y1)}
      : {x: Number(point.x), y: Number(point.y)};
  }

  function viewToSource(point) {
    const roi = binding().focal_roi_original_pixels;
    return runtime.view === "focal"
      ? {x: point.x + Number(roi.x1), y: point.y + Number(roi.y1)}
      : point;
  }

  function svgElement(name, attributes = {}) {
    const node = document.createElementNS(SVG_NS, name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  }

  function pointsAttribute(points) {
    return points.map(sourceToView).map((point) => `${point.x},${point.y}`).join(" ");
  }

  function orientation(a, b, c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  }

  function onSegment(a, b, c) {
    const epsilon = 1e-7;
    return c.x >= Math.min(a.x, b.x) - epsilon && c.x <= Math.max(a.x, b.x) + epsilon
      && c.y >= Math.min(a.y, b.y) - epsilon && c.y <= Math.max(a.y, b.y) + epsilon
      && Math.abs(orientation(a, b, c)) <= epsilon;
  }

  function segmentsIntersect(a, b, c, d) {
    const epsilon = 1e-7;
    const values = [orientation(a, b, c), orientation(a, b, d), orientation(c, d, a), orientation(c, d, b)];
    if (((values[0] > epsilon && values[1] < -epsilon) || (values[0] < -epsilon && values[1] > epsilon))
      && ((values[2] > epsilon && values[3] < -epsilon) || (values[2] < -epsilon && values[3] > epsilon))) return true;
    return (Math.abs(values[0]) <= epsilon && onSegment(a, b, c))
      || (Math.abs(values[1]) <= epsilon && onSegment(a, b, d))
      || (Math.abs(values[2]) <= epsilon && onSegment(c, d, a))
      || (Math.abs(values[3]) <= epsilon && onSegment(c, d, b));
  }

  function selfIntersectionPairs(points) {
    const pairs = [];
    if (points.length < 4) return pairs;
    for (let left = 0; left < points.length; left += 1) {
      const leftNext = (left + 1) % points.length;
      for (let right = left + 1; right < points.length; right += 1) {
        const rightNext = (right + 1) % points.length;
        if (left === right || leftNext === right || rightNext === left) continue;
        if (segmentsIntersect(points[left], points[leftNext], points[right], points[rightNext])) pairs.push([left, right]);
      }
    }
    return pairs;
  }

  function candidateCrosses(candidate) {
    if (runtime.points.length < 2) return false;
    const start = runtime.points[runtime.points.length - 1];
    for (let index = 0; index < runtime.points.length - 2; index += 1) {
      if (segmentsIntersect(start, candidate, runtime.points[index], runtime.points[index + 1])) return true;
    }
    return false;
  }

  function pointInPolygon(point, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
      const a = polygon[i];
      const b = polygon[j];
      if ((a.y > point.y) !== (b.y > point.y)
        && point.x < ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
    }
    return inside;
  }

  function polygonsOverlap(left, right) {
    if (!left.length || !right.length) return false;
    for (let i = 0; i < left.length; i += 1) {
      for (let j = 0; j < right.length; j += 1) {
        if (segmentsIntersect(left[i], left[(i + 1) % left.length], right[j], right[(j + 1) % right.length])) return true;
      }
    }
    return pointInPolygon(left[0], right) || pointInPolygon(right[0], left);
  }

  function materialDependencies() {
    if ($("dcUnreliable").checked) return item().occlusion_dependencies || [];
    if (runtime.points.length < 3 || selfIntersectionPairs(runtime.points).length) return [];
    return (item().occlusion_dependencies || []).filter((dependency) => {
      const other = dependency.other_polygon_original_pixels;
      return polygonsOverlap(item().original_polygon_original_pixels, other) !== polygonsOverlap(runtime.points, other)
        || dependency.original_graph_inconsistent;
    });
  }

  function appendPolygon(svg, points, className) {
    if (points.length < 3) return;
    svg.append(svgElement("polygon", {points: pointsAttribute(points), class: className}));
  }

  function renderOverlay() {
    const svg = $("dcOverlay");
    svg.replaceChildren();
    const context = caseData().visible_metadata.context_masks || [];
    context.forEach((mask) => {
      if (mask.original_mask_uuid === item().original_mask_uuid) return;
      appendPolygon(svg, mask.polygon_original_pixels, "dcContextMask");
      const local = sourceToView(mask.label_point_original_pixels);
      const label = svgElement("text", {x: local.x, y: local.y, class: "dcMaskNumber"});
      label.textContent = mask.anonymous_label;
      svg.append(label);
    });
    if ($("dcCompareOriginal").checked) {
      appendPolygon(svg, item().original_polygon_original_pixels, "dcOriginalMask");
      (item().self_intersection_edge_pairs || []).forEach((pair) => {
        [pair.left_edge_vertex_indices, pair.right_edge_vertex_indices].forEach((indices) => {
          const left = sourceToView(item().original_polygon_original_pixels[indices[0]]);
          const right = sourceToView(item().original_polygon_original_pixels[indices[1] % item().original_polygon_original_pixels.length]);
          svg.append(svgElement("line", {x1: left.x, y1: left.y, x2: right.x, y2: right.y, class: "dcIntersectionSegment"}));
        });
      });
    }
    if (runtime.points.length >= 3 && !runtime.drawing) appendPolygon(svg, runtime.points, "dcCorrectionMask");
    if (runtime.points.length) {
      const draft = runtime.preview ? [...runtime.points, runtime.preview] : runtime.points;
      svg.append(svgElement("polyline", {points: pointsAttribute(draft), class: "dcDraftLine"}));
      runtime.points.forEach((point) => {
        const local = sourceToView(point);
        svg.append(svgElement("circle", {cx: local.x, cy: local.y, r: 3.5 / runtime.scale, class: "dcVertex"}));
      });
      if (runtime.preview && runtime.invalidPreview) {
        const left = sourceToView(runtime.points[runtime.points.length - 1]);
        const right = sourceToView(runtime.preview);
        svg.append(svgElement("line", {x1: left.x, y1: left.y, x2: right.x, y2: right.y, class: "dcInvalidSegment"}));
      }
      if (runtime.drawing && runtime.points.length >= 3 && selfIntersectionPairs(runtime.points).length) {
        const left = sourceToView(runtime.points[runtime.points.length - 1]);
        const right = sourceToView(runtime.points[0]);
        svg.append(svgElement("line", {x1: left.x, y1: left.y, x2: right.x, y2: right.y, class: "dcInvalidSegment"}));
      }
    }
    renderOcclusionRows();
    updateSaveGate();
  }

  function stageTransform() {
    $("dcStage").style.transform = `translate(${runtime.translateX}px, ${runtime.translateY}px) scale(${runtime.scale})`;
  }

  function fitImage() {
    const viewport = $("dcViewport");
    const width = Math.max(1, viewport.clientWidth);
    const height = Math.max(1, viewport.clientHeight);
    runtime.scale = Math.min(width / runtime.naturalWidth, height / runtime.naturalHeight);
    runtime.translateX = (width - runtime.naturalWidth * runtime.scale) / 2;
    runtime.translateY = (height - runtime.naturalHeight * runtime.scale) / 2;
    stageTransform();
    renderOverlay();
  }

  function zoomAt(factor, clientX, clientY) {
    const viewport = $("dcViewport");
    const rect = viewport.getBoundingClientRect();
    const focalX = clientX == null ? viewport.clientWidth / 2 : clientX - rect.left;
    const focalY = clientY == null ? viewport.clientHeight / 2 : clientY - rect.top;
    const imageX = (focalX - runtime.translateX) / runtime.scale;
    const imageY = (focalY - runtime.translateY) / runtime.scale;
    const next = Math.max(0.2, Math.min(12, runtime.scale * factor));
    runtime.translateX = focalX - imageX * next;
    runtime.translateY = focalY - imageY * next;
    runtime.scale = next;
    stageTransform();
    renderOverlay();
  }

  function eventPoint(event) {
    const rect = $("dcViewport").getBoundingClientRect();
    const local = {
      x: (event.clientX - rect.left - runtime.translateX) / runtime.scale,
      y: (event.clientY - rect.top - runtime.translateY) / runtime.scale,
    };
    return viewToSource(local);
  }

  function insideActiveImage(point) {
    const roi = binding().focal_roi_original_pixels;
    return point.x >= Number(roi.x1) && point.x <= Number(roi.x2)
      && point.y >= Number(roi.y1) && point.y <= Number(roi.y2);
  }

  function renderCoverageRows(saved = null) {
    const container = $("dcCoverageRows");
    const rows = item().affected_candidates || [];
    container.innerHTML = rows.length ? rows.map((row, index) => {
      const value = saved?.candidate_coverage_reviews?.find((entry) => entry.candidate_uuid === row.candidate_uuid)?.candidate_visible_mask_coverage;
      return `<label class="dcCoverageRow"><strong>${escapeHtml(row.anonymous_label)}</strong><small>${escapeHtml(row.relation_plain_language)}</small><select data-dc-coverage="${escapeHtml(row.candidate_uuid)}">${COVERAGE.map(([choice, label]) => `<option value="${choice}" ${value != null && Number(choice) === Number(value) ? "selected" : ""}>${label}</option>`).join("")}</select></label>`;
    }).join("") : "<p>No machine-box coverage value depends on this outline.</p>";
    container.querySelectorAll("select").forEach((select) => select.addEventListener("change", () => {
      persistDraft();
      updateSaveGate();
    }));
  }

  function renderOcclusionRows(saved = null) {
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

  function correctionValid() {
    if ($("dcUnreliable").checked) return Boolean($("dcUnreliableReason").value);
    return !runtime.drawing && runtime.points.length >= 3 && selfIntersectionPairs(runtime.points).length === 0;
  }

  function coverageComplete() {
    if ($("dcUnreliable").checked) return true;
    return Array.from(document.querySelectorAll("[data-dc-coverage]")).every((select) => select.value !== "");
  }

  function occlusionComplete() {
    if ($("dcUnreliable").checked) return true;
    return Array.from(document.querySelectorAll("[data-dc-occlusion]")).every((select) => select.value !== "");
  }

  function updateSaveGate() {
    $("dcFinish").disabled = !runtime.drawing
      || runtime.points.length < 3
      || runtime.invalidPreview
      || selfIntersectionPairs(runtime.points).length !== 0;
    $("dcSave").disabled = runtime.evidenceBlocked || !correctionValid() || !coverageComplete() || !occlusionComplete();
    const complete = Object.keys(runtime.state.corrections || {}).length === runtime.items.length;
    $("dcComplete").disabled = !complete || runtime.evidenceBlocked;
  }

  async function persistDraft() {
    if (!runtime.db || !current()) return;
    await idbPut("drafts", {
      key: draftKey(),
      points: runtime.points,
      drawing: runtime.drawing,
      quality: $("dcQuality").value,
      unreliable: $("dcUnreliable").checked,
      unreliableReason: $("dcUnreliableReason").value,
      coverage: Object.fromEntries(Array.from(document.querySelectorAll("[data-dc-coverage]")).map((select) => [select.dataset.dcCoverage, select.value])),
      occlusion: Object.fromEntries(Array.from(document.querySelectorAll("[data-dc-occlusion]")).map((select) => [select.dataset.dcOcclusion, select.value])),
      savedAt: new Date().toISOString(),
    });
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
    } : null);
    runtime.points = (source?.points || []).map((point) => ({x: Number(point.x), y: Number(point.y)}));
    runtime.drawing = Boolean(source?.drawing);
    $("dcQuality").value = source?.quality || item().original_mask_quality;
    $("dcUnreliable").checked = Boolean(source?.unreliable);
    $("dcUnreliableReason").disabled = !$("dcUnreliable").checked;
    $("dcUnreliableReason").value = source?.unreliableReason || "";
    renderCoverageRows(saved);
    Object.entries(source?.coverage || {}).forEach(([candidate, value]) => {
      const select = document.querySelector(`[data-dc-coverage="${CSS.escape(candidate)}"]`);
      if (select && value !== "") select.value = String(value);
    });
    renderOcclusionRows(saved);
    Object.entries(source?.occlusion || {}).forEach(([other, value]) => {
      const select = document.querySelector(`[data-dc-occlusion="${CSS.escape(other)}"]`);
      if (select) select.value = value;
    });
    renderOverlay();
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
    $("dcPreviousMask").disabled = runtime.index === 0;
    $("dcNextMask").disabled = runtime.index === runtime.items.length - 1;
  }

  async function selectIndex(next) {
    runtime.index = Math.max(0, Math.min(runtime.items.length - 1, next));
    runtime.preview = null;
    runtime.invalidPreview = false;
    updateProgress();
    await hydrateItem();
    await loadEvidence();
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
    return item().affected_candidates.map((row) => ({
      candidate_uuid: row.candidate_uuid,
      review_status: unreliable ? "EVIDENCE_UNRESOLVED" : "REVALIDATED",
      candidate_visible_mask_coverage: unreliable ? null : Number(document.querySelector(`[data-dc-coverage="${CSS.escape(row.candidate_uuid)}"]`).value),
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
    $("dcError").classList.add("isHidden");
    const payload = eventPayload();
    const row = {id: payload.idempotency_key, draftKey: draftKey(), payload, createdAt: new Date().toISOString()};
    await idbPut("outbox", row);
    $("dcSaveState").textContent = "Queued locally | pending 1";
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
      await flushOutbox();
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
    runtime.points = [];
    runtime.preview = null;
    runtime.invalidPreview = false;
    runtime.drawing = true;
    $("dcUnreliable").checked = false;
    $("dcUnreliableReason").disabled = true;
    renderOverlay();
    persistDraft();
  }

  function bind() {
    $("dcFocalView").addEventListener("click", async () => {
      runtime.view = "focal";
      $("dcFocalView").classList.add("active");
      $("dcPanoramaView").classList.remove("active");
      await loadEvidence();
    });
    $("dcPanoramaView").addEventListener("click", async () => {
      runtime.view = "panorama";
      $("dcPanoramaView").classList.add("active");
      $("dcFocalView").classList.remove("active");
      await loadEvidence();
    });
    $("dcZoomOut").addEventListener("click", () => zoomAt(0.8));
    $("dcZoomIn").addEventListener("click", () => zoomAt(1.25));
    $("dcFit").addEventListener("click", fitImage);
    $("dcCompareOriginal").addEventListener("change", renderOverlay);
    $("dcRedraw").addEventListener("click", beginRedraw);
    $("dcFinish").addEventListener("click", () => {
      if (runtime.points.length < 3 || selfIntersectionPairs(runtime.points).length) return;
      runtime.drawing = false;
      runtime.preview = null;
      runtime.invalidPreview = false;
      renderOverlay();
      persistDraft();
    });
    $("dcUndo").addEventListener("click", () => {
      runtime.points.pop();
      runtime.drawing = true;
      renderOverlay();
      persistDraft();
    });
    $("dcRestart").addEventListener("click", beginRedraw);
    $("dcQuality").addEventListener("change", persistDraft);
    $("dcUnreliable").addEventListener("change", () => {
      $("dcUnreliableReason").disabled = !$("dcUnreliable").checked;
      renderCoverageRows();
      renderOcclusionRows();
      persistDraft();
      updateSaveGate();
    });
    $("dcUnreliableReason").addEventListener("change", () => { persistDraft(); updateSaveGate(); });
    $("dcPreviousMask").addEventListener("click", () => selectIndex(runtime.index - 1));
    $("dcNextMask").addEventListener("click", () => selectIndex(runtime.index + 1));
    $("dcSave").addEventListener("click", saveCorrection);
    $("dcComplete").addEventListener("click", completeRepair);
    $("dcViewport").addEventListener("wheel", (event) => {
      event.preventDefault();
      zoomAt(event.deltaY < 0 ? 1.12 : 0.89, event.clientX, event.clientY);
    }, {passive: false});
    $("dcViewport").addEventListener("pointerdown", (event) => {
      if (event.button === 1 || event.shiftKey || !runtime.drawing) {
        runtime.pan = {x: event.clientX, y: event.clientY, tx: runtime.translateX, ty: runtime.translateY};
        $("dcViewport").classList.add("isPanning");
        $("dcViewport").setPointerCapture(event.pointerId);
        return;
      }
      if (event.button !== 0 || $("dcUnreliable").checked) return;
      const point = eventPoint(event);
      if (!insideActiveImage(point) || candidateCrosses(point)) {
        runtime.preview = point;
        runtime.invalidPreview = true;
        renderOverlay();
        return;
      }
      runtime.points.push(point);
      runtime.preview = null;
      runtime.invalidPreview = false;
      renderOverlay();
      persistDraft();
    });
    $("dcViewport").addEventListener("pointermove", (event) => {
      if (runtime.pan) {
        runtime.translateX = runtime.pan.tx + event.clientX - runtime.pan.x;
        runtime.translateY = runtime.pan.ty + event.clientY - runtime.pan.y;
        stageTransform();
        return;
      }
      if (!runtime.drawing || !runtime.points.length) return;
      runtime.preview = eventPoint(event);
      runtime.invalidPreview = !insideActiveImage(runtime.preview) || candidateCrosses(runtime.preview);
      renderOverlay();
    });
    const stopPan = () => { runtime.pan = null; $("dcViewport").classList.remove("isPanning"); };
    $("dcViewport").addEventListener("pointerup", stopPan);
    $("dcViewport").addEventListener("pointercancel", stopPan);
    window.addEventListener("resize", fitImage);
  }

  async function mount({manifest, uiConfig, state, api}) {
    runtime.manifest = manifest;
    runtime.uiConfig = uiConfig;
    runtime.state = state;
    runtime.api = api;
    runtime.items = manifest.cases.flatMap((row) => row.visible_metadata.repair_items.map((repairItem) => ({caseData: row, item: repairItem})));
    runtime.db = await openDatabase();
    bind();
    try {
      await flushOutbox();
      runtime.state = await api("/api/review/state");
    } catch (error) {
      $("dcSaveState").textContent = `Offline recovery | ${error.message}`;
    }
    const resumeMask = Object.keys(runtime.state.corrections || {}).length < runtime.items.length
      ? runtime.items.findIndex((row) => !runtime.state.corrections?.[row.item.original_mask_uuid])
      : 0;
    await selectIndex(Math.max(0, resumeMask));
  }

  window.DenseMaskCorrection = {mount};
})();
