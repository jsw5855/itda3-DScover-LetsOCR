"""Prepare pinned official CPU inference artifacts before going offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts/weights_manifest.json"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify(directory, model):
    for name, expected in model["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise RuntimeError(f"Missing or mismatched model file: {path}; existing files were not overwritten")


def prepare(weights, models):
    weights.mkdir(parents=True, exist_ok=True)
    # Validate every existing destination before doing any downloads.
    for model in models:
        destination = weights / model["model_name"]
        if destination.exists():
            verify(destination, model)
    for model in models:
        destination = weights / model["model_name"]
        if destination.exists():
            print(f"Verified existing {destination.name}; unchanged", flush=True)
            continue
        with tempfile.TemporaryDirectory(prefix=".download-", dir=weights) as temporary:
            temporary = Path(temporary)
            archive = temporary / "model.tar"
            print(f"Downloading {model['url']}", flush=True)
            with urllib.request.urlopen(model["url"], timeout=120) as source, archive.open("xb") as target:
                shutil.copyfileobj(source, target)
            if digest(archive) != model["archive_sha256"]:
                raise RuntimeError(f"Archive checksum mismatch: {model['model_name']}")
            staged = temporary / "model"
            staged.mkdir()
            with tarfile.open(archive) as bundle:
                for name in model["files"]:
                    members = [m for m in bundle.getmembers() if m.isfile() and Path(m.name).name == name]
                    if len(members) != 1:
                        raise RuntimeError(f"Expected exactly one {name} in archive")
                    # Copy explicitly named regular files; never extract archive paths.
                    with bundle.extractfile(members[0]) as source, (staged / name).open("xb") as target:
                        shutil.copyfileobj(source, target)
            verify(staged, model)
            if destination.exists():
                raise FileExistsError(f"Refusing existing destination: {destination}")
            staged.rename(destination)
            print(f"Installed and verified {destination.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    models = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]
    weights = ROOT / "weights/paddleocr"
    if args.verify_only:
        for model in models:
            verify(weights / model["model_name"], model)
        print("Verified all six inference files (SHA256)")
    else:
        prepare(weights, models)


if __name__ == "__main__":
    main()
