"""Frozen-weight, full controller-by-plant evaluation; no optimization or training."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'flight_control_corrected_2026-10-24/experiments/controller_route_20261024'
sys.path.insert(0, str(ROOT / 'frozen_source'))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def checkpoint(seed, kind, source):
    if kind == 'teacher':
        return SOURCE / f'teachers/seed_{seed}/{source}/best_validation_checkpoint.pt'
    return SOURCE / f'students/seed_{seed}/best_validation.pt'


def compare_archive(seed, kind, target, rows, ckpt_hash):
    archive = SOURCE / f'evaluations/{kind}_seed_{seed}/{target}/metrics.json'
    old = read(archive)
    assert old['checkpoint_sha256'] == ckpt_hash
    indexed = {(r['case_id'], r['duration_s']): r for r in old['rows']}
    max_rmse_difference = 0.
    pass_mismatches = []
    for row in rows:
        original = indexed[(row['case_id'], row['duration_s'])]
        if row['failed'] != original['failed'] or row['passed'] != original['passed']:
            pass_mismatches.append([row['case_id'], row['duration_s']])
        if original['rmse_deg_s'] is not None and row['rmse_deg_s'] is not None:
            max_rmse_difference = max(max_rmse_difference, abs(original['rmse_deg_s'] - row['rmse_deg_s']))
    return {'archive': str(archive), 'max_absolute_rmse_difference_deg_s': max_rmse_difference,
            'pass_fail_mismatches': pass_mismatches,
            'reproduced': not pass_mismatches and max_rmse_difference <= 1e-4}


def initialize_worker():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def run_job(job):
    from route_common import load_teacher, load_student
    from evaluation import evaluate, scenarios, long_scenarios, aggregate

    protocol = read(ROOT / 'protocol.json')
    seed, kind, source = job
    path = checkpoint(seed, kind, source)
    model_hash = digest(path)
    model = load_teacher(path) if kind == 'teacher' else load_student(path)
    rows = protocol['source_objects']
    if kind == 'teacher':
        rows = sorted(rows, key=lambda r: r['plant_id'] != source)
    progress = ROOT / f'progress/seed_{seed}_{kind}_{source}.json'
    completed = []
    started = time.monotonic()
    for target in rows:
        target_id = target['plant_id']
        output = ROOT / f'results/seed_{seed}/{kind}_{source}/{target_id}'
        metrics_file = output / 'metrics.json'
        if metrics_file.exists():
            assert read(metrics_file)['checkpoint_sha256'] == model_hash
            completed.append(target_id)
            continue
        short = evaluate(model, target, scenarios(), teacher=kind == 'teacher',
                         save_trace=output / 'closed_loop_8s.npz')
        long = evaluate(model, target, long_scenarios(), teacher=kind == 'teacher',
                        save_trace=output / 'closed_loop_30s.npz')
        for record in short + long:
            record.update(seed=seed, controller_kind=kind, teacher_source=source if kind == 'teacher' else None)
        replication = None
        if kind == 'student' or source == target_id:
            replication = compare_archive(seed, kind, target_id, short + long, model_hash)
        write(metrics_file, {'seed': seed, 'controller_kind': kind, 'teacher_source': source,
                            'target_plant': target_id, 'checkpoint_sha256': model_hash,
                            'summary_commands_8s': aggregate([r for r in short if r['category'] == 'command']),
                            'archive_replication': replication, 'rows': short + long})
        completed.append(target_id)
        write(progress, {'job': job, 'completed_targets': completed, 'target_count': 12,
                         'elapsed_s': time.monotonic() - started, 'last_target': target_id})
    assert digest(path) == model_hash
    return {'job': job, 'completed_pairs': len(completed), 'elapsed_s': time.monotonic() - started}


def create_protocol():
    import torch
    source = read(SOURCE / 'protocol.json')
    seeds = source['policy_seeds']
    objects = source['source_objects']
    verified = []
    for seed in seeds:
        hashes = [digest(checkpoint(seed, 'teacher', p['plant_id'])) for p in objects]
        student = checkpoint(seed, 'student', 'unified')
        metadata = torch.load(student, map_location='cpu', weights_only=False)['metadata']
        assert metadata['teacher_bank_sha256'] == hashes
        assert metadata['source_protocol_sha256'] == digest(SOURCE / 'protocol.json')
        verified.append({'seed': seed, 'teacher_sha256_in_plant_order': hashes,
                         'student_sha256': digest(student), 'selected_student_round': metadata['round']})
    return {'version': 'cross-object-frozen-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'source_protocol_sha256': digest(SOURCE / 'protocol.json'), 'source_objects': objects,
            'seeds': seeds, 'verified_checkpoints': verified, 'training_steps': 0,
            'controller_plant_pairs': 3 * 13 * 12, 'episodes_per_pair': 28,
            'primary_metric': '18 command cases at 8 s, per-target pass fraction; RMSE/peak/TV thresholds 1.5/5/12',
            'recovery': '8 s initial deviation +/-5 deg/s and external force +/-1 N; repeated at 30 s with +/-15 deg/s steps',
            'teacher_privileged_input': 'Always current target plant parameters and its current state/history; never source plant A parameters on B.',
            'student_coverage': 'All 12 evaluation plants are sources distilled into the same-seed student; evaluate multi-object control retention.',
            'reference_protocol': 'Identical to the archived frozen cohort: fixed reference_mode and same-time observation alignment.',
            'failures': 'Censor at nonfinite or |p|>300 deg/s; retain all cases in pass fraction denominator.',
            'trace_examples_predeclared': [
                {'target': 'train_core-0750', 'foreign_teacher': 'train_core-0148'},
                {'target': 'train_core-0148', 'foreign_teacher': 'train_core-0401'},
                {'target': 'train_core-0401', 'foreign_teacher': 'train_core-0750'}],
            'trace_seed': seeds[0], 'cpu_workers': 16,
            'cohort_note': 'Matched 12-object controller-route-v1 cohort; separate from the original 49-teacher delivery.'}


def main():
    if not (ROOT / 'protocol.json').exists():
        write(ROOT / 'protocol.json', create_protocol())
    protocol = read(ROOT / 'protocol.json')
    jobs = [(seed, kind, source) for seed in protocol['seeds']
            for kind, source in [('teacher', p['plant_id']) for p in protocol['source_objects']] + [('student', 'unified')]]
    start = time.monotonic()
    completed = []
    def status(state, **extra):
        write(ROOT / 'status.json', {'state': state, 'pid': os.getpid(), 'total_jobs': len(jobs),
                                    'completed_jobs': len(completed), 'elapsed_s': time.monotonic() - start,
                                    'jobs': completed, **extra})
    status('running')
    print(f'Start {len(jobs)} jobs, {protocol["controller_plant_pairs"]} pairs, frozen weights only.', flush=True)
    try:
        with ProcessPoolExecutor(max_workers=protocol['cpu_workers'], mp_context=mp.get_context('spawn'),
                                 initializer=initialize_worker) as pool:
            futures = {pool.submit(run_job, job): job for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                status('running')
                print(json.dumps(result), flush=True)
        files = list((ROOT / 'results').rglob('metrics.json'))
        assert len(files) == protocol['controller_plant_pairs']
        replications = [read(f)['archive_replication'] for f in files if read(f)['archive_replication'] is not None]
        write(ROOT / 'replication_audit.json', {'checked_pairs': len(replications),
              'all_reproduced': all(r['reproduced'] for r in replications), 'comparisons': replications})
        status('complete', result_pairs=len(files), all_archive_pairs_reproduced=all(r['reproduced'] for r in replications))
    except Exception as exc:
        status('error', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
