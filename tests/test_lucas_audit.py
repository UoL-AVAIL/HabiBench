import csv
import json
import tempfile
import unittest
from pathlib import Path

from habibench.lucas.audit import attach_evaluation
from habibench.lucas.pilot import RATING_FIELDS, evaluate
from habibench.lucas.pipeline import digest, policy_identity, write_json, write_lines


class AuditAttachmentTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.full, self.source = root / "full", root / "audit"
        self.identity = {"model": "fixture-model", "rubric": "fixture"}
        self.rows = [
            {"photo_id": str(i) + "P", "point_id": str(i), "view": "P",
             "broad_class": "G", "lc1": "G11", "sha256": f"{i:064x}"}
            for i in (1, 2, 3)
        ]
        for run, scope in ((self.full, "all"), (self.source, "audit")):
            pilot = [
                {"photo_id": r["photo_id"], "blind_id": "b" + r["point_id"],
                 "split": "tune" if r["point_id"] == "3" else "audit"}
                for r in self.rows if scope == "all" or r["point_id"] != "3"
            ]
            rows = self.rows[:len(pilot)]
            write_json(run / "selection.json", {"scope": scope, "profile": "novel"})
            write_json(run / "summary.json", {"reviewed": len(rows)})
            write_json(run / "identity.json", self.identity)
            write_json(run / "policy.json", policy_identity())
            write_json(run / "pilot.json", pilot)
            write_lines(run / "index.jsonl", rows)
            write_lines(run / "ledger.jsonl", [
                {"photo_id": r["photo_id"], "sha256": r["sha256"], "identity_key": digest(self.identity),
                 "decision": "exclude" if run == self.full and r["point_id"] == "2" else "keep"}
                for r in rows
            ])
        self.labels = root / "human.csv"
        self.write_ratings(2)

    def write_ratings(self, count):
        with self.labels.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, ["blind_id", "broad_class", *RATING_FIELDS])
            writer.writeheader()
            for i in range(1, count + 1):
                writer.writerow({"blind_id": f"b{i}", "broad_class": "G",
                                 "human_decision": "keep", "reason": "Open lake"})
        evaluate(self.source, self.labels, "audit")

    def test_reports_full_predictions_and_preserves_existing_release(self):
        curated = self.full / "curated"
        curated.mkdir()
        published = curated / "evaluation-audit.json"
        published.write_text("prior release")
        report = attach_evaluation(self.full, self.source, "audit")
        self.assertEqual(report["labelled"], 2)
        self.assertEqual(report["good_photos_removed"], 1)
        self.assertEqual(report["pilot_provenance"]["changed_decision_blind_ids"], ["b2"])
        self.assertEqual(report["ratings_sha256"], json.loads(
            (self.source / "evaluation-audit.json").read_text())["ratings_sha256"])
        self.assertEqual(json.loads((self.full / "evaluation-audit.json").read_text()), report)
        self.assertEqual(published.read_text(), "prior release")
        self.assertEqual(attach_evaluation(self.full, self.source, "audit"), report)

    def test_mismatched_policy_identity_hash_scope_and_assignments(self):
        changes = [
            ("identity.json", {"model": "other"}),
            ("policy.json", {"name": "drift"}),
            ("selection.json", {"scope": "pilot", "profile": "novel"}),
            ("pilot.json", [{"blind_id": "changed", "photo_id": "1P", "split": "audit"}]),
        ]
        for filename, wrong in changes:
            with self.subTest(filename=filename):
                path = self.source / filename
                original = path.read_text()
                write_json(path, wrong)
                with self.assertRaises(ValueError):
                    attach_evaluation(self.full, self.source, "audit")
                path.write_text(original)
        write_lines(self.full / "index.jsonl", [{**self.rows[0], "sha256": "diff"}, *self.rows[1:]])
        with self.assertRaisesRegex(ValueError, "checksum"):
            attach_evaluation(self.full, self.source, "audit")
        self.assertFalse((self.full / "evaluation-audit.json").exists())

    def test_incomplete_and_edited_source_report_rejected(self):
        self.write_ratings(1)
        with self.assertRaisesRegex(ValueError, "Complete every"):
            attach_evaluation(self.full, self.source, "audit")
        self.write_ratings(2)
        path = self.source / "evaluation-audit.json"
        report = json.loads(path.read_text())
        report["good_photos_removed"] = 1
        write_json(path, report)
        with self.assertRaisesRegex(ValueError, "differs from replay"):
            attach_evaluation(self.full, self.source, "audit")
        self.assertFalse((self.full / "evaluation-audit.json").exists())
