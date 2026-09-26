"""No real OCR: injected runner and mocked engine, including 701 x 4 coverage."""
import contextlib
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import dump_full_stage_701 as dump


def raw():
    return {'detections': [{'text': '2020.04.30', 'confidence': .98,
                            'bbox': [[0, 0], [10, 0], [10, 5], [0, 5]]}],
            'ocr_sec': .1, 'stage_sec': .2}


def make_fake_repo(root):
    # Placeholder bytes only: preflight hashes image files but never decodes them.
    (root / 'labels/review').mkdir(parents=True)
    (root / 'scripts').mkdir()
    for name in ('tmp_labels_701.csv', 'labels/review/label_review.csv', 'ocr_pipeline.py',
                 'scripts/dump_full_stage_701.py'):
        (root / name).write_bytes((ROOT / name).read_bytes())
    (root / 'data').mkdir()
    for key in dump.exact_ids(dump.csv_rows(ROOT / 'tmp_labels_701.csv')):
        (root / 'data' / f'{key}.jpg').write_bytes(key.encode())
    (root / 'data/999999.jpg').write_bytes(b'outside the frozen membership')
    for model in dump.MODELS:
        (root / 'weights/paddleocr' / model).mkdir(parents=True)
        for name in ('inference.json', 'inference.pdiparams', 'inference.yml'):
            (root / 'weights/paddleocr' / model / name).write_bytes(f'{model}/{name}'.encode())


class FullStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / 'run'
        self.manifest = {'image_ids': ['000007', '000015'], 'historical_reuse': {}}

    def run_dump(self, runner=None):
        with contextlib.redirect_stdout(io.StringIO()):
            dump.execute(self.output, self.manifest, [], runner or (lambda *args: raw()))

    def test_exact_701_universe(self):
        rows = dump.csv_rows(ROOT / 'tmp_labels_701.csv')
        self.assertEqual(dump.exact_ids(rows), dump.exact_ids(dump.csv_rows(ROOT / 'labels/review/label_review.csv')))
        for changed in (rows[:-1], rows + [rows[0]], [dict(rows[0], image_id='999999')] + rows[1:]):
            with self.assertRaises(ValueError):
                dump.exact_ids(changed)

    @unittest.skipUnless((ROOT / 'data').is_dir() and (ROOT / 'weights/paddleocr').is_dir()
                         and (ROOT / 'docs/frozen_b_shadow_independent_run1/manifest.json').is_file(),
                         'Real preflight needs local data/, weights/ and the shadow run artifacts')
    def test_readonly_preflight_imports_no_ocr(self):
        code = """
import pathlib, sys
sys.path.insert(0, 'scripts')
import dump_full_stage_701 as d
output = pathlib.Path(sys.argv[1])
m, labels, report = d.preflight(output)
assert report['image_count'] == 701
assert report['maximum_attempts'] == 2804
assert report['safely_reusable_historical_attempts'] == 269
assert report['remaining_attempts'] == 2535
assert not output.exists()
assert not any(n in sys.modules for n in ('ocr_pipeline', 'paddleocr', 'paddle', 'cv2', 'numpy'))
"""
        result = subprocess.run([sys.executable, '-B', '-c', code, str(self.output)],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_synthetic_preflight_no_ocr_no_writes(self):
        """Full preflight on a synthetic repository (701 placeholder image files, fake weights)."""
        fake = Path(self.tmp.name) / 'repo'
        make_fake_repo(fake)
        code = """
import importlib.metadata, pathlib, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / 'scripts'))
import dump_full_stage_701 as d
assert d.ROOT == pathlib.Path(sys.argv[1]).resolve()
d.importlib.metadata.version = lambda name: '1.0'
output = d.ROOT / 'docs/full_stage_701_run1'
m, labels, report = d.preflight(output)
assert report['preflight'] == 'passed' and report['image_count'] == 701, report
assert report['maximum_attempts'] == 2804 and report['remaining_attempts'] == 2804, report
assert report['safely_reusable_historical_attempts'] == 0, report
assert report['rough_remaining_ocr_sec'] is None, report
assert len(m['images']) == 701 and len(labels) == 701
assert not output.exists()
assert not any(n in sys.modules for n in ('ocr_pipeline', 'paddleocr', 'paddle', 'cv2', 'numpy'))
"""
        result = subprocess.run([sys.executable, '-B', '-c', code, str(fake)],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        (fake / 'data/002066.jpg').unlink()
        result = subprocess.run([sys.executable, '-B', '-c', code, str(fake)],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertIn('Missing images', result.stderr)

    def test_historical_audit_accepts_only_matching_provenance(self):
        fake = Path(self.tmp.name) / 'repo'
        make_fake_repo(fake)
        weights = {f'weights/paddleocr/{m}/{f}': dump.sha(fake / 'weights/paddleocr' / m / f)
                   for m in dump.MODELS for f in ('inference.json', 'inference.pdiparams', 'inference.yml')}
        images = {'000007': {'sha256': dump.sha(fake / 'data/000007.jpg')}}
        packages = {p: '1.0' for p in dump.PACKAGES}
        source = fake / 'docs/frozen_b_shadow_independent_run1'
        source.mkdir(parents=True)

        def build(manifest_changes=None, record_changes=None):
            m = {'packages': packages, 'code_sha256': {}, 'weights_sha256': weights,
                 'engine': dump.ENGINE, 'predict': {'text_det_limit_type': 'max', 'text_det_box_thresh': 0.7},
                 'stages': ['original_512', 'rotation_270', 'highres_1024'], 'images': images,
                 **(manifest_changes or {})}
            dump.write(source / 'manifest.json', m)
            r = {'image_id': '000007', 'stage': 'original_512', 'run_id': dump.digest(m), **raw()}
            r['record_sha256'] = dump.digest(r)
            r.update(record_changes or {})
            dump.write(source / '000007_original_512.json', r)
            return {'frozen_b_shadow_independent_run1': dump.sha(source / 'manifest.json')}

        def audit(pins, image_sha=images['000007']['sha256']):
            with patch.object(dump, 'ROOT', fake), patch.object(dump, 'SOURCES', pins):
                return dump.audit_sources({'000007': {'sha256': image_sha}}, packages, weights)

        accepted, reports, _ = audit(build())
        self.assertEqual(list(accepted), ['000007_original_512'])
        self.assertEqual(reports['frozen_b_shadow_independent_run1']['reusable'], 1)
        for changes in ({'packages': dict(packages, paddleocr='0.9')},
                        {'engine': dict(dump.ENGINE, cpu_threads=4)},
                        {'predict': {'text_det_limit_type': 'max', 'text_det_box_thresh': 0.6}},
                        {'weights_sha256': {}}):
            accepted, reports, _ = audit(build(changes))
            self.assertEqual(accepted, {}, changes)
            self.assertEqual(reports['frozen_b_shadow_independent_run1']['reusable'], 0)
        with self.assertRaisesRegex(ValueError, 'integrity'):
            audit(build(record_changes={'ocr_sec': .05}))
        with self.assertRaisesRegex(ValueError, 'image changed'):
            audit(build(), image_sha='0' * 64)
        pins = build()
        (source / 'manifest.json').write_text('{}', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            audit(pins)

    def test_full_701_four_stages_no_early_stop_and_consolidation(self):
        self.manifest['image_ids'] = dump.exact_ids(dump.csv_rows(ROOT / 'tmp_labels_701.csv'))
        calls = []
        def fake(key, stage):
            calls.append((key, stage))
            return raw()  # Clear date in every original: still all four calls required.
        self.run_dump(fake)
        self.assertEqual(calls, [(k, s) for k in self.manifest['image_ids'] for s in dump.STAGES])
        self.assertEqual(len(calls), 2804)
        rows = [json.loads(s) for s in (self.output / 'raw_ocr.jsonl').read_text(encoding='utf8').split('\n')[:-1]]
        self.assertEqual(len(rows), 2804)
        self.assertEqual(dump.read(self.output / 'integrity.json')['raw_ocr_sha256'], dump.sha(self.output / 'raw_ocr.jsonl'))

    def test_interruption_resume_never_repeats_committed_stage(self):
        calls = []
        def interrupted(key, stage):
            if len(calls) == 3:
                raise KeyboardInterrupt()
            calls.append((key, stage))
            return raw()
        with self.assertRaises(KeyboardInterrupt):
            self.run_dump(interrupted)
        prior = set(calls)
        self.assertEqual(len(dump.scan_cache(self.output, self.manifest)), 3)
        def resume(key, stage):
            self.assertNotIn((key, stage), prior)
            calls.append((key, stage))
            return raw()
        self.run_dump(resume)
        self.assertEqual(len(calls), 8)
        self.assertEqual(dump.read(self.output / 'summary.json')['cached_at_start'], 3)
        self.run_dump(lambda *args: self.fail('Complete cache reran OCR'))
        self.assertEqual(dump.read(self.output / 'summary.json')['fresh_attempts_this_invocation'], 0)

    def test_missing_and_duplicate_stage_rejected(self):
        self.run_dump()
        p = self.output / 'cache/000007_clahe.json'
        p.rename(p.with_name('duplicate.json'))
        with self.assertRaisesRegex(ValueError, 'Unexpected/duplicate'):
            dump.consolidate(self.output, self.manifest)
        p.with_name('duplicate.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing stages'):
            dump.consolidate(self.output, self.manifest)

    def test_malformed_cache_rejected_before_runner(self):
        self.run_dump()
        p = self.output / 'cache/000007_original_512.json'
        p.write_text('{truncated', encoding='utf8')
        with self.assertRaises(ValueError):
            self.run_dump(lambda *args: self.fail('OCR invoked before cache validation'))

    def test_checksum_identity_and_provenance_rejected(self):
        self.run_dump()
        p = self.output / 'cache/000007_original_512.json'
        original = dump.read(p)
        for field, value in [('run_id', 'wrong'), ('image_id', '000015'), ('stage', 'clahe')]:
            changed = dict(original, **{field: value})
            changed['record_sha256'] = dump.digest({k: v for k, v in changed.items() if k != 'record_sha256'})
            dump.write(p, changed)
            with self.assertRaises(ValueError):
                dump.scan_cache(self.output, self.manifest)
        dump.write(p, dict(original, ocr_sec=.05))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            dump.scan_cache(self.output, self.manifest)
        dump.write(p, original)
        with self.assertRaisesRegex(ValueError, 'manifest'):
            dump.execute(self.output, dict(self.manifest, changed=True), [], lambda *args: raw())

    def test_unicode_line_separators_in_ocr_text_consolidate(self):
        texts = ['A\x85B', 'C\u2028D', 'E\u2029F', 'G\x1cH']
        def fake(key, stage):
            r = raw()
            r['detections'][0]['text'] = texts[0]
            texts.append(texts.pop(0))
            return r
        self.run_dump(fake)
        path = self.output / 'raw_ocr.jsonl'
        with path.open(encoding='utf8') as stream:
            rows = [json.loads(line) for line in stream]
        self.assertEqual(len(rows), 8)
        self.assertEqual({r['detections'][0]['text'] for r in rows}, {'A\x85B', 'C\u2028D', 'E\u2029F', 'G\x1cH'})
        self.assertEqual(dump.read(self.output / 'integrity.json')['records'], 8)

    def test_invalid_raw_schema(self):
        for mutate in (lambda r: r.update(ocr_sec=float('nan')),
                       lambda r: r['detections'][0].update(confidence=2),
                       lambda r: r['detections'][0].update(bbox=[]),
                       lambda r: r.update(detections=None)):
            r = raw()
            mutate(r)
            with self.assertRaises(ValueError):
                dump.validate_raw(r)
        dump.validate_raw(dict(raw(), detections=[]))

    def test_uncommitted_temporary_ignored_and_replaced(self):
        self.run_dump()
        p = self.output / 'cache/000007_clahe.json'
        p.unlink()
        p.with_name(p.name + '.tmp').write_text('{interrupted', encoding='utf8')
        calls = []
        self.run_dump(lambda k, s: calls.append((k, s)) or raw())
        self.assertEqual(calls, [('000007', 'clahe')])

    def test_historical_import_and_mutation_rejection(self):
        source = Path(self.tmp.name) / 'source.json'
        dump.write(source, raw())
        self.manifest['historical_reuse']['000007_original_512'] = {'path': str(source), 'sha256': dump.sha(source)}
        calls = []
        self.run_dump(lambda k, s: calls.append((k, s)) or raw())
        self.assertNotIn(('000007', 'original_512'), calls)
        self.assertEqual(dump.read(self.output / 'summary.json')['imported_this_invocation'], 1)
        (self.output / 'cache/000007_original_512.json').unlink()
        dump.write(source, dict(raw(), ocr_sec=.05))
        with self.assertRaisesRegex(ValueError, 'changed after preflight'):
            self.run_dump(lambda *args: self.fail('Unexpected OCR'))

    def test_atomic_replace_failure_preserves_old_record(self):
        p = Path(self.tmp.name) / 'record.json'
        dump.write(p, {'old': True})
        with patch.object(dump.os, 'replace', side_effect=OSError('simulated interruption')):
            with self.assertRaises(OSError):
                dump.write(p, {'new': True})
        self.assertEqual(dump.read(p), {'old': True})

    def test_duplicate_json_keys_rejected(self):
        p = Path(self.tmp.name) / 'bad.json'
        p.write_text('{"stage": "clahe", "stage": "original_512"}')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON'):
            dump.read(p)

    def test_runner_exact_preprocessing_with_fake_engine(self):
        import numpy as np
        from PIL import Image
        import ocr_pipeline as pipeline
        image = Path(self.tmp.name) / 'image.png'
        pixels = np.arange(900 * 600 * 3, dtype=np.uint8).reshape(900, 600, 3)
        Image.fromarray(pixels).save(image)
        rgb = pipeline.decode_image(image)
        base = pipeline.resize_image(rgb, 512)
        expected = [base, np.asarray(Image.fromarray(base).rotate(270, expand=True)),
                    pipeline.resize_image(rgb, 1024), pipeline.apply_clahe(base)]
        class FakeEngine:
            def predict(self, array, **kwargs):
                np.testing.assert_array_equal(array, expected.pop(0))
                self.kwargs.append(kwargs)
                return [{'rec_texts': ['2020.04.30'], 'rec_scores': [.98],
                         'rec_polys': [[[0, 0], [10, 0], [10, 5], [0, 5]]]}]
        engine = FakeEngine()
        engine.kwargs = []
        with patch.object(pipeline, 'initialize_engine', return_value=engine) as init, \
             patch.object(pipeline, 'parse_expiration_date', side_effect=AssertionError('Parser called')), \
             patch.object(pipeline, 'has_candidate', side_effect=AssertionError('Trigger called')):
            runner = dump.OCRRunner({'images': {'000007': {'path': str(image)}}})
            for stage in dump.STAGES:
                dump.validate_raw(runner('000007', stage))
            init.assert_called_once_with(**dump.ENGINE)
        self.assertFalse(expected)
        self.assertEqual(engine.kwargs, [dump.stage_metadata(s)['predict'] for s in dump.STAGES])


if __name__ == '__main__':
    unittest.main()
