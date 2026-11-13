import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from habibench.lucas.download import NOVEL_MODEL, NOVEL_REVISION, fetch_model


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.shard = "model-00001-of-00001.safetensors"
        self.files = {
            "model.safetensors.index.json": json.dumps({"weight_map": {"w": self.shard}}).encode(),
            self.shard: b"synthetic model for a file integrity test",
            "preprocessor_config.json": b'{"image_processor_type":"vision","processor_class":"Qwen"}',
            "chat_template.jinja": b"{{ messages }}",
            "tokenizer_config.json": b"{}", "tokenizer.json": b"{}", "config.json": b"{}",
        }

    def hub(self, downloader=None):
        siblings = []
        for name, data in self.files.items():
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()
            lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if name == self.shard else None
            siblings.append(SimpleNamespace(rfilename=name, size=len(data), blob_id=blob, lfs=lfs))
        hub = Mock()
        hub.HfApi.return_value.model_info.return_value = SimpleNamespace(sha=NOVEL_REVISION, siblings=siblings)

        def write_files(*, local_dir, allow_patterns, **kwargs):
            self.assertEqual(kwargs, {"revision": NOVEL_REVISION, "repo_id": NOVEL_MODEL})
            for name in allow_patterns:
                (local_dir / name).write_bytes(self.files[name])
        hub.snapshot_download.side_effect = downloader or write_files
        return patch.dict("sys.modules", {"huggingface_hub": hub})

    def test_atomic_download_then_offline_verification(self):
        with self.hub():
            result = fetch_model(self.root)
        self.assertEqual(result, self.root / NOVEL_REVISION)
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            self.assertEqual(fetch_model(self.root), result)
            shard = result / self.shard
            shard.write_bytes(b"x" * shard.stat().st_size)
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                fetch_model(self.root)

    def test_interrupted_download_keeps_partial_for_retry(self):
        def interrupt(*, local_dir, **kwargs):
            (local_dir / "config.json").write_bytes(self.files["config.json"])
            raise ConnectionError("interrupted")
        with self.hub(interrupt), self.assertRaises(ConnectionError):
            fetch_model(self.root)
        self.assertFalse((self.root / NOVEL_REVISION).exists())
        self.assertTrue((self.root / f".{NOVEL_REVISION}.partial" / "config.json").exists())
        with self.hub():
            self.assertTrue((fetch_model(self.root) / "config.json").is_file())

    def test_reject_incomplete_index_and_invalid_processor(self):
        self.files["preprocessor_config.json"] = b"{}"
        with self.hub(), self.assertRaisesRegex(ValueError, "image processor"):
            fetch_model(self.root)
        self.assertFalse((self.root / NOVEL_REVISION).exists())
        del self.files[self.shard]
        with self.hub(), self.assertRaisesRegex(ValueError, "weight files disagree"):
            fetch_model(self.root)
        self.assertFalse((self.root / NOVEL_REVISION).exists())
