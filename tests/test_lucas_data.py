import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from habibench.lucas.data import CLASSES, LOCAL_CASES, build_index, load_vector, select_pilot


def make_survey(root):
    (root / "geospatial").mkdir()
    with (root / "UK_2018_20200213.csv").open("w", newline="") as points, \
            (root / "habitats.csv").open("w", newline="") as habitats:
        survey = csv.writer(points)
        survey.writerow(["POINT_ID", "LC1", "TH_LAT", "TH_LONG"])
        lookup = csv.writer(habitats)
        lookup.writerow(["code", "habitat"])
        for i, code in enumerate(CLASSES):
            lookup.writerow([code + "10", code])
            folder = root / "lucas-dataset" / f"{code}_test"
            folder.mkdir(parents=True)
            ids = [f"{30000000 + 100 * i + j:08}" for j in range(15)]
            ids.extend(dict.fromkeys(entry[:-1] for entry in LOCAL_CASES.get(code, ())))
            for point in ids:
                survey.writerow([point, code + "10", 53, -1])
                vector = None if point == "32323620" else np.ones(128, dtype=np.float32)
                np.savez(root / "geospatial" / f"{point}.npz", vector)
                for view in "PNESWC":
                    photo = point + view
                    payload = b"duplicate" if photo in {"35983886C", "35963888C"} else photo.encode()
                    (folder / f"{point[:3]}_{point[3:6]}_{photo}.jpg").write_bytes(payload)
        survey.writerow(["17.253 rows selected."])


class IndexTests(unittest.TestCase):
    def test_local_sampling_validates_only_selected_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_survey(root)
            with patch("habibench.lucas.data.load_vector", wraps=load_vector) as validated:
                sample = build_index(root, "local-small")
        self.assertEqual(Counter(r["broad_class"] for r in sample), dict.fromkeys(CLASSES, 4))
        self.assertEqual(validated.call_count, len({r["point_id"] for r in sample}))
        by_id = {r["photo_id"]: r for r in sample}
        self.assertEqual(by_id["32323620N"]["vector_status"], "object_placeholder")
        self.assertTrue({"31623628E", "31643632N", "32523662W"} <= by_id.keys())
        self.assertEqual(by_id["35983886C"]["sha256"], by_id["35963888C"]["sha256"])
        self.assertTrue(all(name in by_id for entries in LOCAL_CASES.values() for name in entries))

    def test_novel_pilot_reuses_all_local_ratings_and_is_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_survey(root)
            complete = build_index(root, "novel")
            local = build_index(root, "local-small")
        pilot = select_pilot(complete, "novel")
        self.assertEqual(pilot, select_pilot(list(reversed(complete)), "novel"))
        self.assertTrue(all(rating in pilot for rating in select_pilot(local, "local-small")))
        by_id = {r["photo_id"]: r for r in complete}
        counts = Counter((r["split"], by_id[r["photo_id"]]["broad_class"]) for r in pilot)
        self.assertEqual(counts, {(phase, c): 6 for phase in ("tune", "audit") for c in CLASSES})
        self.assertEqual(len({by_id[r["photo_id"]]["point_id"] for r in pilot}), 96)
        self.assertEqual(len({by_id[r["photo_id"]]["sha256"] for r in pilot}), 96)
        audit = [by_id[r["photo_id"]] for r in pilot if r["split"] == "audit"]
        for field in ("point_id", "sha256"):
            self.assertTrue({r[field] for r in local}.isdisjoint(r[field] for r in audit))

    def test_reject_invalid_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point.npz"
            for array in (np.ones(128), np.ones((1, 128), np.float32),
                          np.full(128, np.inf, np.float32), None):
                with self.subTest(array_type=type(array)):
                    np.savez(path, array)
                    with self.assertRaises(ValueError):
                        load_vector(path)
