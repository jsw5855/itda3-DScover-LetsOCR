import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import ocr_pipeline as pipeline


class FakeEngine:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.sides = []

    def predict(self, image, **kwargs):
        self.sides.append(kwargs['text_det_limit_side_len'])
        return [{'rec_texts': [next(self.texts)], 'rec_scores': np.array([0.99]),
                 'rec_polys': np.array([[[0, 0], [100, 0], [100, 20], [0, 20]]])}]


@pytest.mark.parametrize('stage', range(4))
def test_cascade_stops_at_first_candidate(tmp_path, monkeypatch, stage):
    path = tmp_path / '000018.jpg'
    Image.new('RGB', (1200, 600)).save(path)
    monkeypatch.setattr(pipeline, 'apply_clahe', lambda image: image)
    engine = FakeEngine(['nothing'] * stage + ['2026.04.24'])
    result, method = pipeline.predict_image(engine, path)
    assert result['final_date'] == '2026-04-24'
    assert method == ['original_512', 'rotation_270', 'highres_1024', 'clahe'][stage]
    assert engine.sides == [512, 512, 1024, 512][:stage + 1]


def test_no_candidate(tmp_path, monkeypatch):
    path = tmp_path / '1.png'
    Image.new('RGB', (20, 10)).save(path)
    monkeypatch.setattr(pipeline, 'apply_clahe', lambda image: image)
    result, method = pipeline.predict_image(FakeEngine(['nothing'] * 4), path)
    assert result == pipeline.parse_expiration_date([])
    assert method == 'original_no_candidate'


def test_notebook_run_all_uses_environment_and_writes_csv(tmp_path, monkeypatch):
    images = tmp_path / 'images'
    images.mkdir()
    Image.new('RGB', (20, 10)).save(images / '000018.JPG')
    output = tmp_path / 'result' / 'submission.csv'
    monkeypatch.setenv('ITDA_INPUT_DIR', str(images))
    monkeypatch.setenv('ITDA_OUTPUT_PATH', str(output))
    monkeypatch.setenv('ITDA_OCR_PROCESSES', '1')
    monkeypatch.setattr(pipeline, 'initialize_engine', lambda **kwargs: FakeEngine(['2026.04.24']))
    notebook = json.loads((pipeline.ROOT / 'predict.ipynb').read_text(encoding='utf-8'))
    namespace = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), 'predict.ipynb', 'exec'), namespace)
    with output.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == pipeline.COLUMNS
        assert list(reader) == [dict(image_id='000018', year='2026', month='04', day='24', final_date='2026-04-24')]


def test_duplicate_ids_fail_before_model_initialization(tmp_path):
    for name in ('01.jpg', '01.png'):
        (tmp_path / name).touch()
    with pytest.raises(ValueError, match='Duplicate'):
        pipeline.run_submission(tmp_path, tmp_path / 'out.csv')
