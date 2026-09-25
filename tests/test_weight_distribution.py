import hashlib
import io
from pathlib import Path
import tarfile

import pytest

from scripts import download_weights as download


def model_bundle(tmp_path):
    content = b"verified model fixture"
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as bundle:
        item = tarfile.TarInfo("../../outside/inference.json")
        item.size = len(content)
        bundle.addfile(item, io.BytesIO(content))
    model = {"model_name": "fixture", "url": "https://official.invalid/model.tar",
             "archive_sha256": download.digest(archive),
             "files": {"inference.json": {"bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest()}}}
    return archive, model


def test_download_verifies_and_flattens_without_path_traversal(tmp_path, monkeypatch):
    archive, model = model_bundle(tmp_path)
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: archive.open("rb"))
    weights = tmp_path / "weights"
    download.prepare(weights, [model])
    download.verify(weights / "fixture", model)
    assert not (tmp_path / "outside").exists()
    assert sorted(p.name for p in weights.iterdir()) == ["fixture"]
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected network"))
    download.prepare(weights, [model])


def test_corrupt_archive_never_installs_partial_model(tmp_path, monkeypatch):
    archive, model = model_bundle(tmp_path)
    model["archive_sha256"] = "0" * 64
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: archive.open("rb"))
    weights = tmp_path / "weights"
    with pytest.raises(RuntimeError, match="Archive checksum mismatch"):
        download.prepare(weights, [model])
    assert list(weights.iterdir()) == []


def test_existing_mismatch_is_never_overwritten(tmp_path, monkeypatch):
    _, model = model_bundle(tmp_path)
    weights = tmp_path / "weights"
    destination = weights / "fixture"
    destination.mkdir(parents=True)
    existing = destination / "inference.json"
    existing.write_bytes(b"preserve this")
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected network"))
    with pytest.raises(RuntimeError, match="not overwritten"):
        download.prepare(weights, [model])
    assert existing.read_bytes() == b"preserve this"
