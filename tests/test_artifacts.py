"""Offline checks for integrity-preserving installation of benchmark data."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('fetch_data', Path(__file__).resolve().parents[1] / 'scripts/fetch_data.py')
FETCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCH)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.content = b'valid benchmark content'
        self.entry = {'path': 'subset/data.parquet', 'bytes': len(self.content),
                      'sha256': hashlib.sha256(self.content).hexdigest(), 'url': 'https://example.invalid/data'}

    def install(self, content):
        return FETCH.fetch_one(self.entry, self.root, opener=lambda *a, **k: io.BytesIO(content))

    def test_valid_download_is_installed_and_skipped_without_network(self):
        self.assertEqual(self.install(self.content)['status'], 'downloaded-verified')
        result = FETCH.fetch_one(self.entry, self.root, opener=lambda *a, **k: self.fail('Must not redownload valid file'))
        self.assertEqual(result['status'], 'verified-existing')

    def test_hash_mismatch_does_not_replace_existing_file(self):
        target = self.root / self.entry['path']
        target.parent.mkdir()
        target.write_bytes(b'previous file')
        with self.assertRaises(ValueError):
            self.install(b'x' * len(self.content))
        self.assertEqual(target.read_bytes(), b'previous file')
        self.assertEqual(list(self.root.rglob('*.part')), [])

    def test_truncated_and_oversized_responses_are_not_installed(self):
        for data in [self.content[:-1], self.content + b'extra']:
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                self.install(data)
            self.assertFalse((self.root / self.entry['path']).exists())
            self.assertEqual(list(self.root.rglob('*.part')), [])

    def test_path_traversal_is_rejected(self):
        for path in ['../escape', '/absolute', 'a/../../escape', 'a\\..\\escape']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                FETCH.destination(self.root, path)

    def test_pinned_manifest_has_four_domains_and_readme(self):
        manifest = FETCH.load_manifest()
        self.assertEqual(len(manifest['files']), 5)
        self.assertEqual(sum(e['path'].endswith('.parquet') for e in manifest['files']), 4)


if __name__ == '__main__':
    unittest.main()
