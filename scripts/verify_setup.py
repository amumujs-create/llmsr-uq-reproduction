#!/usr/bin/env python3
"""Verify pinned source/data and imports; optionally list endpoint models (no generation)."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import urllib.parse
import urllib.request

from fetch_data import ROOT, destination, load_manifest, valid_file

DEPENDENCIES = {
    'numpy': 'numpy', 'scipy': 'scipy', 'torch': 'torch', 'sympy': 'sympy',
    'scikit-learn': 'sklearn', 'pyarrow': 'pyarrow', 'datasets': 'datasets', 'openai': 'openai',
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Do not contact any model endpoint (default).')
    parser.add_argument('--require-data', action='store_true', help='Fail if any pinned data file is missing.')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--check-endpoint', metavar='BASE_URL', help='Check an OpenAI-compatible /v1 base URL.')
    parser.add_argument('--model', help='Require this exact model ID in the endpoint model list.')
    args = parser.parse_args()
    if args.offline and args.check_endpoint:
        parser.error('--offline and --check-endpoint are mutually exclusive')
    if args.model and not args.check_endpoint:
        parser.error('--model requires --check-endpoint')
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'errors': [], 'warnings': [], 'dependencies': {}}
    if sys.version_info < (3, 11):
        report['errors'].append('Use Python 3.11 or newer; Python 3.11 is the documented environment.')
    manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    verified = 0
    for entry in manifest['files']:
        path = destination(ROOT / 'upstream', entry['path'])
        if not path.is_file():
            report['errors'].append('Missing upstream source: ' + entry['path'])
            continue
        data = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if len(data) != entry['bytes'] or blob != entry['git_blob_sha']:
            report['errors'].append('Upstream source differs: ' + entry['path'])
        else:
            verified += 1
    report['upstream'] = {'commit': manifest['commit'], 'verified_files': verified,
                          'expected_files': len(manifest['files'])}
    for package, module in DEPENDENCIES.items():
        try:
            version = importlib.metadata.version(package)
            importlib.import_module(module)
            report['dependencies'][package] = version
        except Exception as exc:
            report['errors'].append('Dependency import failed: ' + package + ' (' + type(exc).__name__ + ')')
    numpy_version = report['dependencies'].get('numpy')
    if numpy_version and int(numpy_version.split('.')[0]) >= 2:
        report['errors'].append('Upstream requires NumPy < 2; reinstall from requirements.txt.')
    data_manifest = load_manifest()
    verified_data, missing_data = [], []
    for entry in data_manifest['files']:
        path = destination(args.data_dir, entry['path'])
        if not path.exists():
            missing_data.append(entry['path'])
        elif not valid_file(path, entry):
            report['errors'].append('Data size/SHA256 mismatch: ' + entry['path'])
        else:
            verified_data.append(entry['path'])
    report['data'] = {'revision': data_manifest['revision'], 'verified': verified_data, 'missing': missing_data}
    if missing_data:
        target = report['errors'] if args.require_data else report['warnings']
        target.append('Missing data; run python scripts/fetch_data.py before experiments.')
    if args.check_endpoint:
        parsed = urllib.parse.urlsplit(args.check_endpoint)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            parser.error('Use an http(s) base URL without credentials, query, or fragment.')
        headers = {}
        key = os.environ.get('LOCAL_LLM_API_KEY')
        if key:
            headers['Authorization'] = 'Bearer ' + key
        request = urllib.request.Request(args.check_endpoint.rstrip('/') + '/models', headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.load(response)
            models = [entry['id'] for entry in payload['data']]
            report['endpoint'] = {'available_model_ids': models, 'generation_performed': False}
            if args.model and args.model not in models:
                report['errors'].append('Requested model ID is not served by the endpoint.')
        except Exception as exc:
            report['errors'].append('Endpoint listing failed (' + type(exc).__name__ + ').')
    report['passed'] = not report['errors']
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
