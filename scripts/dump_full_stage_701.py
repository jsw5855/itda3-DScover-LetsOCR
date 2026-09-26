"""Reusable raw four-stage OCR dataset. Default: read-only preflight, NO OCR."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STAGES = ('original_512', 'rotation_270', 'highres_1024', 'clahe')
IDS_SHA256 = '4a1c04e97c285caa12a00841616ff56d3eaf8f374238c45105a5ae139a8f0e4b'
MODELS = ('PP-OCRv6_medium_det', 'korean_PP-OCRv5_mobile_rec')
PACKAGES = ('paddleocr', 'paddlepaddle', 'numpy', 'pillow', 'paddlex', 'opencv-contrib-python')
ENGINE = {'cpu_threads': 2, 'enable_mkldnn': True, 'recognition_batch_size': 6}
SOURCES = {
    'early_stop_26_run1': 'a1615ef92e052553138071f0853753a87a59d371d8b4310dc4bf407a99528e50',
    'correct_controls_50_run1': 'e24f83ff0f99dcb711c5053fd1aa6ec088a657f4051d1e98b7a97ef3eabc37cb',
    'triggered_correct_controls_9_run1': 'bd822d4c42b16ea974e052ada26120336730ff69cdde923ed0016554092578df',
    'frozen_b_shadow_independent_run1': '8ff0fc2236e89701858676ea9448d0438e884b5e349949238d35b1a37e3b4c6c',
}


def sha(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
        return h.hexdigest()


def digest(value):
    # Also compatible with the shadow runner's record checksum.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f'Duplicate JSON key: {key}')
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding='utf8'), object_pairs_hook=unique)


def atomic_text(path, text):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf8', newline='\n') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def exact_ids(rows):
    ids = []
    for row in rows:
        value = row['image_id'].strip()
        if not value.isascii() or not value.isdigit():
            raise ValueError(f'Invalid ID: {value!r}')
        ids.append(f'{int(value):06d}')
    ids.sort()
    if len(ids) != 701 or len(set(ids)) != 701:
        raise ValueError('Require exactly 701 unique IDs')
    if hashlib.sha256('\n'.join(ids).encode()).hexdigest() != IDS_SHA256:
        raise ValueError('Missing/unexpected IDs: frozen 701 membership mismatch')
    return ids


def stage_metadata(stage):
    return {
        'decode': 'PIL ImageOps.exif_transpose, RGB',
        'resize': 'shrink only; rounded dimensions; PIL LANCZOS',
        'transform': {'original_512': 'resize decoded RGB to max side 512',
                      'rotation_270': 'PIL rotate(270, expand=True) of original_512',
                      'highres_1024': 'resize decoded RGB to max side 1024',
                      'clahe': 'original_512 RGB->LAB; L CLAHE clipLimit=2 tileGridSize=(8,8); LAB->RGB'}[stage],
        'predict': {'text_det_limit_side_len': 1024 if stage == 'highres_1024' else 512,
                    'text_det_limit_type': 'max', 'text_det_box_thresh': 0.7},
        'coordinates': 'polygon pixel coordinates in transformed stage input, not original image',
        'raw_schema': 'ordered paddle_to_common detections: text, confidence, bbox; no parser filtering',
    }


def validate_raw(raw):
    if not isinstance(raw.get('detections'), list):
        raise ValueError('Missing raw detections')
    def number(x):
        return type(x) in (int, float) and math.isfinite(x)
    for d in raw['detections']:
        if not isinstance(d.get('text'), str) or not number(d.get('confidence')) or not 0 <= d['confidence'] <= 1:
            raise ValueError('Invalid raw text/confidence')
        box = d.get('bbox')
        if not isinstance(box, list) or len(box) < 4 or any(
                not isinstance(p, list) or len(p) != 2 or not all(number(v) for v in p) for p in box):
            raise ValueError('Invalid polygon')
    for field in ('ocr_sec', 'stage_sec'):
        if not number(raw.get(field)) or raw[field] < 0:
            raise ValueError('Invalid timing')
    if raw['stage_sec'] < raw['ocr_sec']:
        raise ValueError('Stage time shorter than OCR time')


def load_record(path, key, stage, run_id):
    raw = read(path)
    checksum = raw.get('record_sha256')
    if digest({k: v for k, v in raw.items() if k != 'record_sha256'}) != checksum:
        raise ValueError(f'Cache checksum mismatch: {path}')
    if (raw.get('image_id'), raw.get('stage'), raw.get('run_id')) != (key, stage, run_id):
        raise ValueError(f'Cache identity/provenance mismatch: {path}')
    validate_raw(raw)
    if raw.get('stage_metadata') != stage_metadata(stage):
        raise ValueError('Cache stage metadata mismatch')
    return raw


def scan_cache(output, manifest, complete=False):
    run_id = digest(manifest)
    expected = {f'{key}_{stage}.json': (key, stage)
                for key in manifest['image_ids'] for stage in STAGES}
    result = {}
    directory = output / 'cache'
    if directory.exists():
        for path in directory.iterdir():
            # A .tmp file is never a committed record; only known atomic-write names allowed.
            if path.name.endswith('.tmp') and path.name[:-4] in expected:
                continue
            if path.name not in expected or not path.is_file():
                raise ValueError(f'Unexpected/duplicate cache file: {path}')
            key, stage = expected[path.name]
            result[key, stage] = load_record(path, key, stage, run_id)
    if complete and len(result) != len(expected):
        raise ValueError(f'Missing stages: {len(result)}/{len(expected)}; consolidation requires all four per image')
    return result


def audit_sources(images, packages, weights):
    accepted, reports, measured = {}, {}, {s: [] for s in STAGES}
    for name, pin in SOURCES.items():
        directory = ROOT / 'docs' / name
        mp = directory / 'manifest.json'
        if not mp.exists():
            reports[name] = {'reusable': 0, 'reason': 'source absent'}
            continue
        if sha(mp) != pin:
            raise ValueError(f'Inspected historical manifest changed: {mp}')
        m = read(mp)
        missing = set(PACKAGES) - {p.lower() for p in m['packages']}
        issues = []
        for p, version in m['packages'].items():
            if packages.get(p.lower()) != version:
                issues.append(f'package drift: {p}')
        for p, checksum in m['code_sha256'].items():
            if sha(ROOT / p) != checksum:
                issues.append(f'code drift: {p}')
        if {p.replace('\\', '/'): v for p, v in m['weights_sha256'].items()} != weights:
            issues.append('model weight mismatch')
        if m.get('engine', ENGINE if m.get('cpu_threads') == 2 else None) != ENGINE:
            issues.append('engine mismatch')
        # Side length is per stage; the remaining detection options must equal ours.
        common = {k: v for k, v in stage_metadata(STAGES[0])['predict'].items() if k != 'text_det_limit_side_len'}
        if not isinstance(m.get('predict'), dict) or {
                k: v for k, v in m['predict'].items() if k != 'text_det_limit_side_len'} != common:
            issues.append('detection option mismatch')
        # Only the inspected shadow schema meets the complete runtime provenance contract
        # (per-image hashes, per-record identity/checksum bound to its manifest).
        if name != 'frozen_b_shadow_independent_run1':
            issues.append('schema lacks per-record checksum/run binding')
        if missing:
            issues.append('missing historical package versions: ' + ', '.join(sorted(missing)))
        count = 0
        for stage in STAGES:
            for path in sorted(directory.glob(f'*_{stage}.json')):
                raw = read(path)
                validate_raw(raw)
                key = path.name.split('_')[0]
                if raw['stage'] != stage or raw.get('image_id', key) != key or key not in images:
                    raise ValueError(f'Historical cache identity mismatch: {path}')
                measured[stage].append(raw['ocr_sec'])
                count += 1
                if issues:
                    continue
                if stage not in m['stages']:
                    raise ValueError('Unsupported historical schema')
                if m['images'][key]['sha256'] != images[key]['sha256']:
                    raise ValueError(f'Historical image changed: {key}')
                checksum = raw.pop('record_sha256')
                if digest(raw) != checksum or raw['run_id'] != digest(m):
                    raise ValueError(f'Historical record integrity mismatch: {path}')
                pair = f'{key}_{stage}'
                if pair in accepted:
                    raise ValueError(f'Duplicate historical stage: {pair}')
                accepted[pair] = {'path': str(path.resolve()), 'sha256': sha(path),
                                  'manifest_sha256': pin}
        reports[name] = {'records': count, 'reusable': 0 if issues else count,
                         'reason': '; '.join(issues) if issues else 'matching complete provenance and validated raw schema'}
    return accepted, reports, measured


def preflight(output):
    labels_path = ROOT / 'tmp_labels_701.csv'
    review_path = ROOT / 'labels/review/label_review.csv'
    labels = csv_rows(labels_path)
    ids = exact_ids(labels)
    if exact_ids(csv_rows(review_path)) != ids:
        raise ValueError('Review and analysis membership differ')
    images = {}
    for path in (ROOT / 'data').iterdir():
        if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png') and path.stem.isdigit():
            key = f'{int(path.stem):06d}'
            if key in ids:
                if key in images:
                    raise ValueError(f'Duplicate image: {key}')
                images[key] = {'path': str(path.resolve()), 'sha256': sha(path)}
    if set(images) != set(ids):
        raise ValueError(f'Missing images: {set(ids) - set(images)}')
    packages = {p: importlib.metadata.version(p) for p in PACKAGES}
    weights = {(Path('weights/paddleocr') / model / file).as_posix():
               sha(ROOT / 'weights/paddleocr' / model / file) for model in MODELS
               for file in ('inference.json', 'inference.pdiparams', 'inference.yml')}
    reuse, audit, measured = audit_sources(images, packages, weights)
    manifest = {'schema_version': 1, 'image_ids': ids, 'ids_sha256': IDS_SHA256,
                'images': images, 'models': list(MODELS), 'weights_sha256': weights,
                'packages': packages, 'python': sys.version, 'engine': ENGINE,
                'stages': {s: stage_metadata(s) for s in STAGES},
                'code_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in
                               (Path(__file__), ROOT / 'ocr_pipeline.py')},
                'dataset_sources': {str(p.relative_to(ROOT)): sha(p) for p in (labels_path, review_path)},
                'label_role': 'verbatim metadata only; never interpreted or passed to OCR',
                'historical_reuse': reuse, 'cache_audit': audit}
    if output.exists():
        if not (output / 'manifest.json').exists() or read(output / 'manifest.json') != manifest:
            raise ValueError('Output manifest/provenance mismatch; use a new directory')
    cached = scan_cache(output, manifest)
    reusable = {(key, stage) for key in ids for stage in STAGES if f'{key}_{stage}' in reuse}
    remaining = Counter(stage for key in ids for stage in STAGES
                        if (key, stage) not in cached and (key, stage) not in reusable)
    means = {s: sum(v) / len(v) if v else None for s, v in measured.items()}
    report = {'preflight': 'passed', 'ocr_imported': False, 'image_count': len(ids),
              'maximum_attempts': 701 * 4, 'existing_output_records': len(cached),
              'safely_reusable_historical_attempts': len(reusable),
              'remaining_attempts': sum(remaining.values()), 'remaining_by_stage': dict(remaining),
              'cache_audit': audit, 'timing_sample_counts': {s: len(v) for s, v in measured.items()},
              'measured_historical_mean_ocr_sec': means,
              # None when a remaining stage has no historical timing sample to estimate from.
              'rough_remaining_ocr_sec': (None if any(remaining[s] and means[s] is None for s in STAGES)
                                          else sum(remaining[s] * (means[s] or 0) for s in STAGES)),
              'estimate_caveat': 'Selected cohorts; OCR/conversion only, excludes init/decode/preparation/writes; not measured full-run wall time'}
    return manifest, labels, report


class OCRRunner:
    """Only instantiated with --run-ocr; no parser/trigger/acceptance calls."""
    def __init__(self, manifest):
        self.manifest = manifest
        self.engine = None
        self.key = None
        self.initialization_sec = 0.0

    def __call__(self, key, stage):
        if self.engine is None:
            for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                os.environ[name] = '2'
            import ocr_pipeline as pipeline
            self.pipeline = pipeline
            start = time.perf_counter()
            self.engine = pipeline.initialize_engine(**ENGINE)
            self.initialization_sec = time.perf_counter() - start
        p = self.pipeline
        decode_sec = 0.0
        if self.key != key:
            start = time.perf_counter()
            self.rgb = p.decode_image(self.manifest['images'][key]['path'])
            self.key = key
            decode_sec = time.perf_counter() - start
        start = time.perf_counter()
        if stage == 'highres_1024':
            array = p.resize_image(self.rgb, 1024)
        else:
            array = p.resize_image(self.rgb, 512)
            if stage == 'rotation_270':
                import numpy as np
                from PIL import Image
                array = np.asarray(Image.fromarray(array).rotate(270, expand=True))
            elif stage == 'clahe':
                array = p.apply_clahe(array)
        ocr_start = time.perf_counter()
        detections = p.paddle_to_common(self.engine.predict(array, **stage_metadata(stage)['predict']))
        end = time.perf_counter()
        return {'detections': detections, 'decode_sec': decode_sec,
                'prepare_sec': ocr_start - start, 'ocr_sec': end - ocr_start, 'stage_sec': end - start,
                'input_shape_hwc': list(array.shape), 'decoded_size_wh': list(self.rgb.size)}


def seal(raw, key, stage, run_id, source):
    result = {**raw, 'image_id': key, 'stage': stage, 'run_id': run_id,
              'stage_metadata': stage_metadata(stage), 'source': source}
    result.pop('record_sha256', None)
    validate_raw(result)
    result['record_sha256'] = digest(result)
    return result


def consolidate(output, manifest):
    records = scan_cache(output, manifest, complete=True)
    lines = [records[key, stage] for key in manifest['image_ids'] for stage in STAGES]
    path = output / 'raw_ocr.jsonl'
    atomic_text(path, ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in lines))
    # Read the published artifact back and verify exact order, membership and checksums.
    # Split on LF only: splitlines() also breaks on U+0085/U+2028, which json.dumps
    # leaves unescaped with ensure_ascii=False and OCR text may contain.
    loaded = [json.loads(s) for s in path.read_text(encoding='utf8').split('\n')[:-1]]
    if loaded != lines:
        raise ValueError('Consolidation readback mismatch')
    integrity = {'records': len(lines), 'images': len(manifest['image_ids']),
                 'stages_per_image': 4, 'raw_ocr_sha256': sha(path), 'run_id': digest(manifest),
                 'record_checksums': {f'{r["image_id"]}_{r["stage"]}': r['record_sha256'] for r in lines}}
    write(output / 'integrity.json', integrity)
    timing = {s: {'count': len(manifest['image_ids']),
                  'ocr_sec': sum(r['ocr_sec'] for r in lines if r['stage'] == s),
                  'stage_sec': sum(r['stage_sec'] for r in lines if r['stage'] == s)} for s in STAGES}
    write(output / 'timing_summary.json', {'by_stage': timing,
          'scope': 'Original measured durations, including imported caches; not invocation wall time. Stage includes preparation except where source timing says otherwise.'})
    return integrity


def execute(output, manifest, labels, runner):
    output.mkdir(parents=True, exist_ok=True)
    # OS-managed exclusive lock releases on process death; the lock file is intentionally retained.
    with (output / 'writer.lock').open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            if lock.seek(0, 2) == 0:
                lock.write(b'0')
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mp = output / 'manifest.json'
        if mp.exists() and read(mp) != manifest:
            raise ValueError('Output manifest mismatch')
        if not mp.exists():
            write(mp, manifest)
        (output / 'cache').mkdir(exist_ok=True)
        write(output / 'labels_metadata.json', labels)
        cached = scan_cache(output, manifest)
        fresh = imported = 0
        start = time.perf_counter()
        progress = {}
        try:
            for key in manifest['image_ids']:
                for stage in STAGES:  # Unconditional Cartesian product, no early stopping.
                    if (key, stage) in cached:
                        continue
                    source = manifest['historical_reuse'].get(f'{key}_{stage}')
                    if source:
                        if sha(source['path']) != source['sha256']:
                            raise ValueError('Historical cache changed after preflight')
                        raw = read(source['path'])
                        record = seal(raw, key, stage, digest(manifest), {'kind': 'historical', **source})
                    else:
                        record = seal(runner(key, stage), key, stage, digest(manifest), {'kind': 'fresh'})
                    write(output / 'cache' / f'{key}_{stage}.json', record)
                    cached[key, stage] = record
                    imported += bool(source)
                    fresh += not bool(source)
                    progress = {'status': 'running', 'completed_stages': len(cached),
                                'fresh_attempts_this_invocation': fresh, 'imported_this_invocation': imported,
                                'cached_at_start': len(cached) - fresh - imported}
                    write(output / 'progress.json', progress)
                print(f'{key}: four stages saved; {len(cached)}/{len(manifest["image_ids"]) * 4}', flush=True)
            integrity = consolidate(output, manifest)
            progress.update(status='complete', completed_stages=integrity['records'],
                            fresh_attempts_this_invocation=fresh, imported_this_invocation=imported,
                            cached_at_start=len(cached) - fresh - imported)
            write(output / 'summary.json', {**progress, 'wall_sec_this_invocation': time.perf_counter() - start,
                  'initialization_sec_this_invocation': getattr(runner, 'initialization_sec', 0.0),
                  'raw_ocr_sha256': integrity['raw_ocr_sha256']})
        finally:
            write(output / 'progress.json', {**progress, 'completed_stages': len(cached),
                  'status': progress.get('status', 'interrupted') if progress.get('status') == 'complete' else 'interrupted',
                  'fresh_attempts_this_invocation': fresh, 'imported_this_invocation': imported})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'docs/full_stage_701_run1')
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--run-ocr', action='store_true', help='Run/resume every missing stage, without early stopping')
    group.add_argument('--consolidate-only', action='store_true', help='Validate and consolidate a complete cache; no OCR')
    args = ap.parse_args()
    manifest, labels, report = preflight(args.output)
    print(json.dumps(report, indent=2), flush=True)
    if args.run_ocr:
        execute(args.output, manifest, labels, OCRRunner(manifest))
    elif args.consolidate_only:
        # execute obtains the writer lock; complete scan guarantees runner cannot be called.
        scan_cache(args.output, manifest, complete=True)
        def forbidden(*args):
            raise AssertionError('Consolidation must never invoke OCR')
        execute(args.output, manifest, labels, forbidden)
    else:
        print('Read-only preflight: no OCR imports, initialization, inference, or output writes.')


if __name__ == '__main__':
    main()
