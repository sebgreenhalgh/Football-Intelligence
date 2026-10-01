"""Freeze R2 source provenance and compact engineering handoff. Never annotate/ingest.

All publication uses immutable_write. Existing differing release/report bytes
cause a hold, not an overwrite. Browser acceptance must already have completed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from football_intelligence.gold.corpus import canonical, digest, file_hash, immutable_write, require
from football_intelligence.gold.temporal_reviewer_r2 import RELEASE, REPO, STATIC, TemporalReviewer
try:
    from scripts import g7g_b_r1_run_temporal_gold_pilot as gate
except ModuleNotFoundError:
    import g7g_b_r1_run_temporal_gold_pilot as gate

r1 = gate.r1
STAGE_NAME = 'G7G_B_R1_TEMPORAL_REVIEWER_USABILITY_AND_SEQUENTIAL_FLOW_REPAIR_v1'
STAGE = r1.STAGE.parent / STAGE_NAME
START_HEAD = '48df70256b12dd0b89f7a0c9043474d1417368b6'
PASS = 'PASS_G7G_B_R1_TEMPORAL_REVIEWER_R2_READY_FOR_HUMAN_PILOT'


def publish(name, value):
    data = value.encode('utf-8') if isinstance(value, str) else canonical(value)
    immutable_write(STAGE / name, data)


def real_inventory():
    files = [{'path': p.relative_to(r1.REAL).as_posix(), 'bytes': p.stat().st_size, 'sha256': file_hash(p)}
             for p in sorted(r1.REAL.rglob('*')) if p.is_file()]
    return {'root': str(r1.REAL), 'file_count': len(files), 'files': files}


def freeze():
    reviewer = TemporalReviewer(r1.PILOT, r1.REAL, gold_root=r1.GOLD)
    files = [REPO / 'src/football_intelligence/gold/temporal_reviewer_r2.py',
             *(STATIC / name for name in ('index.html', 'app.js', 'styles.css', 'view.js'))]
    # Match the accepted reviewer's Windows-native bundle-key encoding exactly.
    bundle = {str(p.relative_to(REPO)): file_hash(p) for p in files}
    require(reviewer.reviewer_sha256 == digest(canonical(bundle)), 'Unexpected reviewer bundle')
    dependencies = ['scripts/g7g_b_r1_run_temporal_gold_pilot.py', 'scripts/g7g_b_run_temporal_gold_pilot.py',
                    'src/football_intelligence/gold/temporal.py', 'src/football_intelligence/gold/temporal_ingest.py',
                    'src/football_intelligence/gold/temporal_reviewer.py',
                    *(f'src/football_intelligence/gold/temporal_reviewer_static/{name}' for name in ('index.html', 'app.js', 'styles.css')),
                    *(f'schemas/gold/{name}' for name in r1.SCHEMA_NAMES.values())]
    binding = {'schema_version': 'football_intelligence.g7g_b.reviewer_release.v2',
               'reviewer_release': RELEASE, 'reviewer_sha256': reviewer.reviewer_sha256,
               'bundle_source_sha256': bundle,
               'source_sha256': {**bundle, **{p: file_hash(REPO / p) for p in dependencies}},
               'pilot_selection_sha256': r1.PILOT_SHA, 'r1_reviewer_sha256': gate.FROZEN_REVIEWER_SHA256,
               'starting_code_commit': START_HEAD, 'candidate_blind': True, 'context_video_ui': False,
               'truth_schemas_changed': False, 'truth_builders_changed': False, 'gold_adapters_changed': False,
               'canonical_gold_mutation': False, 'production_ready': False}
    immutable_write(gate.RELEASE_CONFIG, canonical(binding))
    publish('02_R2_REVIEWER_RELEASE.json', {**binding, 'config_path': str(gate.RELEASE_CONFIG), 'config_sha256': file_hash(gate.RELEASE_CONFIG)})
    return binding


def run(command):
    result = subprocess.run(command, cwd=REPO, env={**os.environ, 'PYTHONPATH': str(REPO / 'src')}, capture_output=True, text=True)
    require(result.returncode == 0, f'Test failed: {command}\n{result.stdout}\n{result.stderr}')
    return {'command': command, 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def handoff():
    require(Path(sys.executable).resolve() == Path(r'C:\Users\sebgr\anaconda3\envs\fi-reviewer\python.exe').resolve(), 'Approved Python required')
    binding = freeze()
    before = json.loads((STAGE / 'REAL_ROOT_BEFORE_INVENTORY.json').read_bytes())
    require(real_inventory() == before, 'Real files changed')
    reviewer, sequence = gate.validate_release()
    lifecycle = gate.validate_lifecycle(reviewer, sequence)
    require(lifecycle['lifecycle_state'] == 'NOT_STARTED', 'Unexpected substantive human work; preserve and audit')
    tests = run([sys.executable, '-m', 'unittest', 'tests.test_g7g_b_r1_reviewer_r2.R2Tests',
                 'tests.test_g7g_b_pilot_release', 'tests.test_g7g_a_temporal_foundation',
                 'tests.test_g7g_a_frozen_selection', 'tests.test_g7f_f_provisional_stdlib', '-v'])
    metadata = run([sys.executable, 'scripts/check_project_metadata.py'])
    coordinates = run(['node', 'tests/g7g_b_r1_view_test.js'])
    gold_cli = run([sys.executable, '-m', 'football_intelligence.gold', 'validate'])
    edge_path = STAGE / 'TEMP_EDGE/browser_acceptance.json'
    edge = json.loads(edge_path.read_bytes())
    require(edge['passed'] and not edge['uncaught_errors'] and not edge['real_decisions_used'], 'TEMP Edge acceptance failed')
    require({c['number'] for c in edge['checks'] if isinstance(c['number'], int)} == set(range(1, 40)) - {37, 38}, 'Browser matrix incomplete')
    after = real_inventory()
    require(after == before, 'Real decision bytes changed during tests')
    gold = r1.inventory()
    require(gold == json.loads(r1.BEFORE.read_bytes()), 'Full Gold inventory changed')
    publish('01_REAL_ROOT_BEFORE_AFTER_INTEGRITY.json', {'before': before, 'after': after, 'unchanged': True,
        'lifecycle': lifecycle, 'real_reviewer_launched': False, 'read_time_writes': False,
        'r1_compatibility': 'TEMP empty, partial draft, mixed-release and complete-event tests; exact byte preservation'})
    publish('04_VIEW_TRANSFORM_AND_COORDINATE_TESTS.json', {'result': json.loads(coordinates['stdout']),
        'cases': ['FIT', '2x', '5x', 'horizontal pan', 'vertical pan', 'combined pan/zoom', 'four viewports'],
        'browser_pointer_geometry_and_ball_tested': True, 'display_state_in_truth_events': False})
    summary = {}
    for key, values in edge['performance_ms'].items():
        summary[key] = {'count': len(values), 'max': max(values), 'mean': sum(values) / len(values)}
    publish('05_TEMP_EDGE_ACCEPTANCE.json', {'browser': edge['browser'], 'layout': edge['layout'],
        'matrix': edge['checks'] + [{'number': 37, 'name': 'Full canonical Gold unchanged', 'passed': True},
                                    {'number': 38, 'name': 'Real decision root unchanged', 'passed': True}],
        'required_matrix_checks': 39, 'all_passed': True,
        'raw_evidence_path': str(edge_path), 'raw_evidence_sha256': file_hash(edge_path),
        'screenshots': {p.name: file_hash(p) for p in sorted((STAGE / 'TEMP_EDGE').glob('*.png'))},
        'screenshots_visually_inspected': True, 'no_page_scroll': True,
        'r1_viewport_height_at_1080': 702, 'r2_viewport_height_at_1080': 798,
        'r1_additional_video_ui_removed': True, 'performance_ms': summary,
        'physical_trackpad_automated': False, 'wheel_ctrl_wheel_and_pointer_semantics_tested': True,
        'server_restart_coverage': 'fresh backend instance plus persisted TEMP drafts; no real server launched',
        'real_decisions_used': False, 'production_ready': False})
    publish('06_GOLD_INTEGRITY.json', {'unchanged': True, 'gold_release': 'gold-v0.1.0',
        'corpus_manifest_sha256': r1.GOLD_SHA, 'release_manifest_sha256': r1.RELEASE_SHA,
        'file_count': gold['file_count'], 'tree_sha256': gold['tree_sha256'],
        'before_inventory_path': str(r1.BEFORE), 'before_inventory_sha256': file_hash(r1.BEFORE),
        'all_objects_registries_releases_identical': True, 'validation': json.loads(gold_cli['stdout']),
        'ingestion_performed': False, 'new_release_created': False})
    publish('TEST_RESULTS.json', {'focused_and_retained': tests, 'metadata': metadata, 'coordinate_test': coordinates, 'gold_cli': gold_cli})
    instructions = (r1.STAGE / '07_HUMAN_PILOT_INSTRUCTIONS.md').read_bytes()
    immutable_write(STAGE / '07_HUMAN_PILOT_INSTRUCTIONS.md', instructions)
    publish('03_SEQUENTIAL_WORKFLOW.md', '# R2 sequential workflow\n\nPhase A: each frame People → Ball → Match state → acknowledged Save & Next. The canonical anchor skips editable People.\n\nPhase B: explicitly start, navigate, select, add and end tracklets; no auto-linking. Gaps and POSSIBLY_SAME_PERSON retain the accepted semantics.\n\nPhase C: inspect summaries and exact assertions; explicitly finalize TRACKLET, BALL and MATCH_STATE, then SEQUENCE. BALL and MATCH_STATE remain single sequence events.\n\nOne task panel; manual Previous/Next and frame strip. Cursor-anchored zoom and two-axis pan transform display only. Browser-local workflow state never enters finalized truth. Immutable events and acknowledgements retain their original builders and schemas.\n')
    publish('00_EXECUTIVE_SUMMARY.json', {'decision': PASS, 'stage': STAGE_NAME, 'status': 'READY_FOR_HUMAN_PILOT',
        'reviewer_release': RELEASE, 'reviewer_sha256': binding['reviewer_sha256'],
        'gold_sequence_id': sequence['gold_sequence_id'], 'pilot_sequence_unchanged': True,
        'real_root_state': lifecycle['lifecycle_state'], 'real_files_preserved': True,
        'real_sequences_annotated': 0, 'real_temporal_events_created': 0,
        'context_video_displayed': False, 'context_video_requested': False, 'manual_previous_next': True,
        'trackpad_pan': True, 'cursor_anchored_zoom': True, 'sequential_wizard': True,
        'candidate_blind': True, 'gold_manifest_unchanged': True, 'gold_v0_1_0_unchanged': True,
        'r1_release_unchanged': True, 'production_ready': False,
        'next_action': 'COMPLETE_THE_AUTHORIZED_G7G_B_SINGLE_SEQUENCE_HUMAN_PILOT',
        'launcher_path': str(r1.STAGE / 'launch_g7g_b_temporal_gold_pilot.ps1')})
    publish('08_DECISION.md', f'# Decision\n\n{PASS}\n\nR2 is ready for the same single-sequence human pilot. Gold and the empty real root remain byte-identical. R1 release, schemas, builders and adapters are unchanged. No real reviewer was launched; no real annotations, inference, ingestion or Gold release occurred.\n\nPhysical trackpad hardware was not automated; Edge wheel/pinch-equivalent and pointer events were tested. Screenshots show synthetic TEMP truth only. The existing ingestion adapter remains R1-pinned; accepting frozen R2 events requires the later separately authorized ingestion stage.\n\nThe original pilot launcher and two instruction files were explicitly superseded; their exact bytes are recoverable from SUPERSEDED_PILOT_TEXT_BYTES.json. Historical handoff manifests were not rewritten.\n\nCommit/push is authorized after full diff and integrity checks. Publication commit is bound separately after push to avoid a self-referential manifest. production_ready=false. Stop before human annotation.\n')
    names = [f'{i:02d}_{suffix}' for i, suffix in enumerate(['EXECUTIVE_SUMMARY.json', 'REAL_ROOT_BEFORE_AFTER_INTEGRITY.json', 'R2_REVIEWER_RELEASE.json', 'SEQUENTIAL_WORKFLOW.md', 'VIEW_TRANSFORM_AND_COORDINATE_TESTS.json', 'TEMP_EDGE_ACCEPTANCE.json', 'GOLD_INTEGRITY.json', 'HUMAN_PILOT_INSTRUCTIONS.md', 'DECISION.md'])]
    names += ['REAL_ROOT_BEFORE_INVENTORY.json', 'SUPERSEDED_PILOT_TEXT_BYTES.json', 'TEST_RESULTS.json']
    publish('09_MANIFEST.json', {'schema_version': 'football_intelligence.g7g_b_r1.handoff.v1', 'decision': PASS,
        'starting_head': START_HEAD, 'files': [{'path': name, 'bytes': (STAGE/name).stat().st_size, 'sha256': file_hash(STAGE/name)} for name in names],
        'launcher_sha256': file_hash(r1.STAGE/'launch_g7g_b_temporal_gold_pilot.ps1'), 'production_ready': False})
    return {'decision': PASS, 'handoff': str(STAGE), 'reviewer_sha256': binding['reviewer_sha256']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'handoff'))
    args = parser.parse_args()
    print(json.dumps(freeze() if args.phase == 'freeze' else handoff(), indent=2, sort_keys=True))
