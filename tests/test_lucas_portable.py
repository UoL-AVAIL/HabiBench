import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from habibench.lucas.portable import CuratedDataset, export_dataset, verify_export


class PortableReleaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.raw = self.base / "raw"
        self.run_dir = self.base / "run"
        self.raw.mkdir()
        self.run_dir.mkdir()
        im = self.raw / "field.jpg"
        Image.new("RGB", (12, 10), "green").save(im)
        sha = hashlib.sha256(im.read_bytes()).hexdigest()
        np.savez(self.raw / "emb.npz", np.arange(128, dtype=np.float32))
        self.rows, self.records = [], {}
        for pid, view, action, status in [
            ("12345678", "P", "keep", "valid"), ("12345678", "E", "keep", "valid"),
            ("11223344", "C", "exclude", "valid"), ("99887766", "N", "keep", "object_placeholder")
        ]:
            photo_id = pid + view
            self.rows.append(dict(photo_id=photo_id, point_id=pid, view=view, image_path="field.jpg",
                                  vector_path="emb.npz", vector_status=status, sha256=sha,
                                  broad_class="E", lc1="E20", habitat="Grassland",
                                  latitude=53.5, longitude=0.0))
            self.records[photo_id] = dict(photo_id=photo_id, sha256=sha, decision=action,
                                          observation={"context": "ambiguous" if view == "E" else "compatible", "strength": "high", "reason": "Grass."})

    def publish(self):
        return export_dataset(self.raw, self.run_dir, self.rows, self.records, {"model": "test"})

    def test_export_load_after_move_and_raw_removal(self):
        folder = self.publish()
        metadata_inode = (folder / "release.json").stat().st_ino
        folder = self.publish()
        self.assertEqual(metadata_inode, (folder / "release.json").stat().st_ino)
        destination = self.base / "relocated"
        shutil.move(folder, destination)
        shutil.rmtree(self.raw)
        result = verify_export(destination)
        self.assertEqual(result, {"photos": 2, "points": 1, "reviewed": 4, "sample_batch": 2})
        samples = CuratedDataset(destination)
        self.assertEqual(samples[0]["vector_path"], samples[1]["vector_path"])
        self.assertEqual(samples[0]["image"].size, (12, 10))
        np.testing.assert_array_equal(samples[1]["embedding"], np.arange(128, dtype=np.float32))
        self.assertEqual(len((destination / "excluded.jsonl").read_text().splitlines()), 1)
        self.assertEqual(len(list((destination / "vectors").iterdir())), 1)
        self.assertFalse(any(p.is_symlink() for p in destination.rglob("*")))
        selected = json.loads((destination / "retention.json").read_text())["total"]
        self.assertEqual(selected["kept_missing_vector"], 1)
        self.assertEqual(selected["ambiguous_kept"], 1)
        self.assertEqual(samples[1]["context"], "ambiguous")

    def test_rebuild_replaces_both_manifest_and_files(self):
        folder = self.publish()
        self.records["12345678E"]["decision"] = "exclude"
        self.publish()
        self.assertEqual(verify_export(folder)["photos"], 1)
        self.assertFalse((folder / "images/E/12345678E.jpg").exists())
        previous = (folder / "release.json").read_bytes()
        self.rows[0]["image_path"] = "../escape.jpg"
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual((folder / "release.json").read_bytes(), previous)

    def test_tampering_external_symlink_and_corrupt_vector(self):
        folder = self.publish()
        source = folder / "images/E/12345678P.jpg"
        source.unlink()
        source.symlink_to(self.raw / "field.jpg")
        with self.assertRaisesRegex(ValueError, "escapes dataset"):
            verify_export(folder)
        source.unlink()
        shutil.copyfile(self.raw / "field.jpg", source)
        (folder / "vectors/12345678.npz").write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "Embedding checksum"):
            verify_export(folder)

    def test_existing_classification_release_is_preserved(self):
        folder = self.run_dir / "curated"
        folder.mkdir()
        path = folder / "release.json"
        path.write_text('{"schema": 1}')
        with self.assertRaisesRegex(ValueError, "fresh run directory"):
            self.publish()
        self.assertEqual(path.read_text(), '{"schema": 1}')

    def test_late_audit_changes_identity_and_checksums(self):
        folder = self.publish()
        (self.run_dir / "evaluation-audit.json").write_text('{"labelled": 48}')
        self.publish()
        target = folder / "evaluation-audit.json"
        self.assertEqual(json.loads(target.read_text()), {"labelled": 48})
        target.write_text('{"labelled": 2}')
        with self.assertRaisesRegex(ValueError, "Metadata checksum mismatch"):
            verify_export(folder)
