#!/usr/bin/env python3
"""Download the five pinned public data artifacts, checking bytes before install."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def destination(data_dir: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or '..' in path.parts or '\\' in relative:
        raise ValueError('Unsafe artifact path')
    root = data_dir.resolve()
    target = root.joinpath(*path.parts)
    target.resolve().relative_to(root)
    return target


def valid_file(path: Path, entry: dict) -> bool:
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == entry['sha256']


def fetch_one(entry: dict, data_dir: Path, *, timeout: float = 90,
              opener=urllib.request.urlopen) -> dict:
    """A failed/partial download never replaces an existing artifact."""
    target = destination(data_dir, entry['path'])
    if valid_file(target, entry):
        return {'path': entry['path'], 'status': 'verified-existing', 'bytes': entry['bytes']}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + '.',
                                         suffix='.part', delete=False) as out:
            temporary = Path(out.name)
            digest, size = hashlib.sha256(), 0
            with opener(entry['url'], timeout=timeout) as response:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > entry['bytes']:
                        raise ValueError('Artifact exceeds pinned size: ' + entry['path'])
                    digest.update(block)
                    out.write(block)
            if size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
                raise ValueError('Artifact size/SHA256 mismatch: ' + entry['path'])
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, target)
        return {'path': entry['path'], 'status': 'downloaded-verified', 'bytes': size}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_manifest() -> dict:
    manifest = json.loads((ROOT / 'data_manifest.json').read_text())
    prefix = ('https://huggingface.co/datasets/' + manifest['repository']
              + '/resolve/' + manifest['revision'] + '/')
    for entry in manifest['files']:
        if entry['url'] != prefix + entry['path']:
            raise ValueError('URL does not match pinned data revision')
        if not (0 < entry['bytes'] <= 100 * 1024 * 1024):
            raise ValueError('Unexpected artifact size')
        destination(ROOT / 'data', entry['path'])
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.timeout <= 0 or not 1 <= args.workers <= 8:
        parser.error('timeout must be positive; workers must be between 1 and 8')
    manifest = load_manifest()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_one, entry, args.data_dir, timeout=args.timeout)
                   for entry in manifest['files']]
        for future in concurrent.futures.as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    print(json.dumps({'status': 'complete', 'files': len(manifest['files']),
                      'revision': manifest['revision']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
