import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from habibench.lucas.pilot import SHEET_FIELDS, evaluate, write_pilot
from habibench.lucas.pipeline import policy_identity, write_json, write_lines


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.raw, self.run = root / "data", root / "run"
        self.raw.mkdir()
        (self.run / "curated").mkdir(parents=True)
        self.rows, self.pilot, ledger = [], [], []
        for i, prediction in enumerate(["exclude", "exclude", "error", "keep"]):
            photo_id, blind_id = f"3232323{i}N", f"blind{i}"
            filename = photo_id + ".jpg"
            exif = Image.Exif()
            exif[270] = "Woodland, private coordinate"
            Image.new("RGB", (3, 3), "green").save(self.raw / filename, exif=exif)
            self.rows.append({"photo_id": photo_id, "image_path": filename, "view": "N", "broad_class": "C", "habitat": "Woodland"})
            self.pilot.append({"photo_id": photo_id, "blind_id": blind_id, "split": "tune"})
            ledger.append({"photo_id": photo_id, "decision": prediction, "error": "bad" if i == 2 else None})
        write_lines(self.run / "ledger.jsonl", ledger)
        write_lines(self.run / "index.jsonl", self.rows)
        write_json(self.run / "pilot.json", self.pilot)
        write_json(self.run / "identity.json", {"model": "test-8b"})
        write_pilot(self.raw, self.run, self.rows, self.pilot)
        self.sheet = self.run / "pilot/tune/ratings.csv"

    def test_blinding_persistence_and_partial_report(self):
        with self.sheet.open(newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(tuple(rows[0]), SHEET_FIELDS)
        for row in rows:
            with Image.open(self.sheet.parent / row["image"]) as img:
                self.assertFalse(img.getexif())
            self.assertEqual(row["view"], "compass")
            self.assertEqual((row["broad_class"], row["habitat"]), ("C", "Woodland"))
        for row, decision in zip(rows[:3], ("exclude", "keep", "keep")):
            row["human_decision"] = decision
        with self.sheet.open("w", newline="") as file:
            writer = csv.DictWriter(file, SHEET_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        # Preserve original CSV bytes, including BOM and CRLF, for an exact replay.
        raw = b"\xef\xbb\xbf" + self.sheet.read_bytes()
        self.sheet.write_bytes(raw)
        write_pilot(self.raw, self.run, self.rows, self.pilot)
        preview = evaluate(self.run, self.sheet, persist=False)
        self.assertFalse((self.run / "evaluation-tune.json").exists())
        self.assertFalse((self.run / "ratings-tune.csv").exists())
        report = evaluate(self.run, self.sheet)
        self.assertEqual(report, preview)
        self.assertEqual(report, json.loads((self.run / "evaluation-tune.json").read_text()))
        self.assertEqual(report["ratings_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(report["policy_fingerprint"], policy_identity())
        self.assertEqual((self.run / "ratings-tune.csv").read_bytes(), raw)
        self.assertEqual(report["model"], "test-8b")
        self.assertEqual(report["split"], "tune")
        self.assertEqual([report[k] for k in ("expected_labels", "labelled", "unlabelled")], [4, 3, 1])
        self.assertEqual(report["exclusion_precision"], 0.5)
        self.assertEqual(report["good_photo_removal_rate"], 0.5)
        self.assertEqual(report["inference_coverage"], 2 / 3)
        self.assertEqual(report["label_coverage"], 3 / 4)
        self.assertEqual(report["usable_with_inference_error"], 1)
        self.assertEqual(len(report["disagreements"]), 2)
        self.assertFalse((self.run / "curated/evaluation-tune.json").exists())
        write_json(self.run / "policy.json", {"name": "different", "sha256": "obsolete"})
        with self.assertRaisesRegex(ValueError, "Decision policy differs"):
            evaluate(self.run, self.sheet)

    def test_empty_duplicate_unmatched_and_wrong_phase(self):
        with self.assertRaisesRegex(ValueError, "No completed"):
            evaluate(self.run, self.sheet)
        with self.assertRaisesRegex(ValueError, "outside"):
            evaluate(self.run, self.sheet, "audit")
        contents = self.sheet.read_text()
        self.sheet.write_text(contents + contents.splitlines()[1] + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate or unmatched"):
            evaluate(self.run, self.sheet)
        self.sheet.write_text(contents.replace("blind3", "extra"))
        with self.assertRaisesRegex(ValueError, "Duplicate or unmatched"):
            evaluate(self.run, self.sheet)

    def test_reject_relabelled_target_and_legacy_ratings(self):
        contents = self.sheet.read_text()
        self.sheet.write_text(contents.replace(",C,Woodland,", ",D,Woodland,", 1))
        with self.assertRaisesRegex(ValueError, "target does not match"):
            evaluate(self.run, self.sheet)
        self.sheet.write_text("blind_id,human_dominant\nblind0,C\n")
        with self.assertRaisesRegex(ValueError, "Earlier classification ratings"):
            evaluate(self.run, self.sheet)
        with self.assertRaisesRegex(ValueError, "another task"):
            write_pilot(self.raw, self.run, self.rows, self.pilot)
