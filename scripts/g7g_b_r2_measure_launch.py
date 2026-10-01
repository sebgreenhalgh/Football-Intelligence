"""Read-only workstation measurement; reports only outside Git, never serves."""
import argparse
import contextlib
import io
import json
import time
from unittest.mock import patch

from football_intelligence import dense_person_gold
from football_intelligence.gold import corpus, dense, temporal_reviewer_r2
try:
    from scripts import g7g_b_r1_run_temporal_gold_pilot as gate
except ModuleNotFoundError:
    import g7g_b_r1_run_temporal_gold_pilot as gate

STAGE = gate.r1.STAGE.parent / 'G7G_B_R2_FAST_LAUNCH_GATE_AND_GOLD_VALIDATION_PERFORMANCE_REPAIR_v1'


def measure(mode):
    started=time.perf_counter(); paths=set(); calls=0; deep_calls=0; mask_calls=0
    original_hash=corpus.file_hash; original_get=corpus.GoldCorpus.get
    original_validate=corpus.GoldCorpus.validate; original_mask=dense_person_gold.canonical_mask_geometry
    def hash_file(path):
        nonlocal calls
        calls+=1;paths.add(str(path.resolve()));return original_hash(path)
    def get_object(instance,sha):
        nonlocal calls
        calls+=1;paths.add(str(instance.object_path(sha).resolve()));return original_get(instance,sha)
    def deep(instance,*args,**kwargs):
        nonlocal deep_calls
        deep_calls+=1
        if mode=='fast':raise AssertionError('Fast check called GoldCorpus.validate')
        return original_validate(instance,*args,**kwargs)
    def mask(*args,**kwargs):
        nonlocal mask_calls
        mask_calls+=1
        if mode=='fast':raise AssertionError('Fast check rerasterized a mask')
        return original_mask(*args,**kwargs)
    before={p.relative_to(gate.r1.REAL).as_posix():original_hash(p) for p in gate.r1.REAL.rglob('*') if p.is_file()}
    # This measurement is allowed only on the verified empty real staging root.
    corpus.require(not before, 'Use a TEMP root if real human work has started')
    with contextlib.ExitStack() as stack:
        for module in (gate,corpus,temporal_reviewer_r2):stack.enter_context(patch.object(module,'file_hash',hash_file))
        stack.enter_context(patch.object(corpus.GoldCorpus,'get',get_object))
        stack.enter_context(patch.object(corpus.GoldCorpus,'validate',deep))
        stack.enter_context(patch.object(dense_person_gold,'canonical_mask_geometry',mask))
        if mode=='fast':stack.enter_context(patch.object(dense,'_geometry_valid',side_effect=AssertionError('Historical geometry')))
        else:dense._geometry_valid.cache_clear()
        output=io.StringIO()
        with contextlib.redirect_stdout(output):result=gate.check() if mode=='fast' else gate.audit()
    elapsed=time.perf_counter()-started
    after={p.relative_to(gate.r1.REAL).as_posix():original_hash(p) for p in gate.r1.REAL.rglob('*') if p.is_file()}
    corpus.require(before==after,'Real root changed during measurement')
    report={'mode':mode,'elapsed_seconds':elapsed,'unique_files_hashed':len(paths),'file_hash_operations':calls,
            'GoldCorpus_validate_calls':deep_calls,'canonical_mask_geometry_calls':mask_calls,
            'frame_assets_decoded_by_fast_gate':result['frame_assets_decoded'],'real_root_unchanged':True,
            'messages':output.getvalue(),'result':result,'production_ready':False}
    corpus.immutable_write(STAGE / (mode.upper()+'_MEASUREMENT.json'),corpus.canonical(report))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('fast','audit'))
    args=parser.parse_args()
    if args.mode=='audit':print('Performing expensive deep Gold semantic audit (instrumented read-only).',flush=True)
    try:print(json.dumps(measure(args.mode),indent=2,sort_keys=True))
    except KeyboardInterrupt:
        print('AUDIT_INTERRUPTED' if args.mode=='audit' else 'CHECK_INTERRUPTED',flush=True)
        raise SystemExit(130)
