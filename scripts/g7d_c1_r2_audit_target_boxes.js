"use strict";

const crypto = require("crypto");
const fs = require("fs");
const path = require("path");
const calibration = require("../src/football_intelligence/g7d_c1_r2_static/calibration.js");

const [packageDirectory, evidenceDirectory] = process.argv.slice(2);
if (!packageDirectory || !evidenceDirectory) {
  throw new Error("Usage: node scripts/g7d_c1_r2_audit_target_boxes.js <package> <evidence>");
}

const REVISION = "G7D_C1_R2_CALIBRATED_TARGET_BOX_NOVICE_REVIEW_V1";
const source = JSON.parse(fs.readFileSync(path.join(packageDirectory, "review_cases.json"), "utf8"));
if (source.review_revision !== REVISION) throw new Error("The installed package is not the R2 revision.");

function canonical(value) { return `${JSON.stringify(value, Object.keys(value).sort())}\n`; }
function sha256(file) { return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex"); }
function writeJson(file, value) { fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`, "utf8"); }
function band(area) { return area < 256 ? "TINY" : area < 1024 ? "SMALL" : area < 4096 ? "MEDIUM" : "LARGE"; }
function edge(record) {
  const [left, top, right, bottom] = record.source_box_xyxy;
  const [width, height] = record.source_frame_dimensions;
  return left <= 1 || top <= 1 || right >= width - 1 || bottom >= height - 1 ? "EDGE" : "NON_EDGE";
}

const records = [];
const failures = [];
for (const scene of source.cases) {
  const asset = path.join(packageDirectory, "assets", scene.asset_name);
  if (!fs.existsSync(asset) || sha256(asset) !== scene.frame_sha256) {
    failures.push({ scene_id: scene.scene_id, failure: "FRAME_HASH_MISMATCH" });
    continue;
  }
  for (const target of scene.targets) {
    const record = calibration.auditTarget(scene, target);
    records.push(record);
    if (!record.passed) failures.push({ scene_id: scene.scene_id, target_id: target.target_id, record });
  }
}

const fixtures = [
  ["central_normal", [1600, 360, 1710, 640]], ["tiny_far_side", [3978, 112, 3985, 122]],
  ["left_edge", [0, 480, 30, 580]], ["right_edge", [4060, 480, 4096, 580]],
  ["top_edge", [1880, 0, 1950, 50]], ["bottom_edge", [1880, 1030, 1950, 1080]],
  ["wide", [1000, 300, 1800, 360]], ["tall", [2010, 120, 2050, 850]],
  ["one_pixel_near_boundary", [4094, 1078, 4095, 1079]],
];
const fixtureScene = { scene_id: "fixture", source_width: 4096, source_height: 1080 };
const fixtureResults = fixtures.map(([fixture_id, source_box_xyxy]) => {
  const record = calibration.auditTarget(fixtureScene, { target_id: fixture_id, source_box_xyxy });
  return { fixture_id, passed: record.passed, source_to_display_to_source_error_px: record.source_to_display_to_source_error_px, display_to_source_to_display_error_css_px: record.display_to_source_to_display_error_css_px, crop_containment: record.crop_containment };
});
const dprAndResize = [1, 2].flatMap((dpr) => [
  calibration.cssCanvasMetrics(1280, 720, dpr), calibration.cssCanvasMetrics(880, 540, dpr),
]);
const resizedRecord = calibration.auditTarget(fixtureScene, { target_id: "resize", source_box_xyxy: [1600, 360, 1710, 640] }, {
  full: { css_width: 880, css_height: 540 }, context: { css_width: 1260, css_height: 620 }, closeup: { css_width: 560, css_height: 620 },
});
if (!resizedRecord.passed || fixtureResults.some((result) => !result.passed)) {
  failures.push({ failure: "FIXTURE_OR_RESIZE_CALIBRATION_FAILURE", fixture_results: fixtureResults, resized_record: resizedRecord });
}

const summary = {};
for (const record of records) {
  const key = [record.scene_id, band(record.box_area_source_px2), edge(record)].join("|");
  summary[key] = (summary[key] || 0) + 1;
}
const audit = {
  schema_version: "football_intelligence.g7d_c1_r2.target_box_calibration_audit.v1",
  review_revision: REVISION,
  target_count: records.length,
  failure_count: failures.length,
  source_roundtrip_limit_px: 0.5,
  css_roundtrip_limit_px: 1.0,
  layouts: calibration.DEFAULT_LAYOUTS,
  fixtures: fixtureResults,
  dpr_and_resize_metrics: dprAndResize,
  resized_layout_passed: resizedRecord.passed,
  records,
  summary_by_scene_size_band_and_edge: summary,
};
const status = {
  schema_version: "football_intelligence.g7d_c1_r2.target_box_calibration_status.v1",
  review_revision: REVISION,
  verified: records.length === 192 && failures.length === 0,
  target_count: records.length,
  failure_count: failures.length,
  target_ids: records.map((record) => record.target_id).sort(),
  audit_sha256: crypto.createHash("sha256").update(`${JSON.stringify(audit)}\n`).digest("hex"),
};
fs.mkdirSync(evidenceDirectory, { recursive: true });
writeJson(path.join(evidenceDirectory, "target_box_calibration_audit.json"), audit);
writeJson(path.join(evidenceDirectory, "target_box_calibration_failures.json"), failures);
writeJson(path.join(evidenceDirectory, "target_box_calibration_status.json"), status);
writeJson(path.join(packageDirectory, "target_box_calibration_status.json"), status);
if (!status.verified) throw new Error(`Target box calibration failed for ${failures.length} item(s).`);
