"""Self-contained, point-indexed image/embedding exports of a LUCAS review run."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict

from PIL import Image

from .data import CLASSES, VIEWS, load_vector


SCHEMA_VERSION = 2
STATUSES = ("keep", "exclude", "error")
V1_STATUSES = ("core", "secondary", "mismatch", "uncertain", "error")
POINT_FIELDS = ("point_id", "lc1", "broad_class", "habitat", "latitude", "longitude")
MANIFESTS = ("photos.jsonl", "points.jsonl", "reviews.jsonl", "excluded.jsonl")
PROVENANCE_FILES = ("selection.json", "pilot.json", "freeze.json", "policy.json", "freeze-policy.json")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected a relative path inside {root}: {name}")
    target = (root / relative).resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError(f"File escapes dataset root or is not regular: {name}")
    return target


def _json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True) + "\n")


def _read(root: Path, name: str) -> list[dict]:
    with _file(root, name).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


class CuratedDataset:
    """Load `dataset[index]` as metadata plus a PIL RGB image and float32 embedding."""

    def __init__(self, directory):
        self.root = Path(directory).resolve(strict=True)
        self.photos = _read(self.root, "photos.jsonl")

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, index):
        row = self.photos[index]
        with Image.open(_file(self.root, row["image_path"])) as image:
            pixels = image.convert("RGB")
        vector = load_vector(_file(self.root, row["vector_path"]))
        return {**row, "image": pixels, "embedding": vector}


def _outcome_counts(reviews, schema=SCHEMA_VERSION):
    by_class = {code: Counter() for code in CLASSES}
    by_view = {view: Counter() for view in VIEWS}
    by_class_view = defaultdict(Counter)
    total = Counter()
    for row in reviews:
        status = row["decision"]
        keep = status == ("core" if schema == 1 else "keep")
        included = keep and row["vector_status"] == "valid"
        for bucket in (total, by_class[row["broad_class"]], by_view[row["view"]],
                       by_class_view[row["broad_class"] + "/" + row["view"]]):
            bucket["reviewed"] += 1
            bucket[status] += 1
            if schema == 1:
                bucket["paired_core"] += int(included)
                bucket["core_missing_vector"] += int(keep and not included)
            else:
                bucket["paired_keep"] += int(included)
                bucket["kept_missing_vector"] += int(keep and not included)
                bucket["ambiguous_kept"] += int(keep and row["observation"]["context"] == "ambiguous")
    return {"total": total, "by_class": by_class, "by_view": by_view, "by_class_view": by_class_view}


def _exchange(old: Path, new: Path):
    """Atomically exchange two complete directories without exposing a partial release."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        result = libc.renamex_np(os.fsencode(old), os.fsencode(new), 2)  # RENAME_SWAP
    elif sys.platform.startswith("linux"):
        result = libc.renameat2(-100, os.fsencode(old), -100, os.fsencode(new), 2)  # RENAME_EXCHANGE
    else:
        raise OSError("Atomic replacement of a published dataset requires Linux or macOS")
    if result:
        error = ctypes.get_errno()
        raise OSError(error, f"Cannot exchange release directories: {os.strerror(error)}")


def _build(root, stage, run, rows, records, identity, input_digest):
    photos, reviews, excluded = [], [], []
    points = {}
    seen = set()
    for row in rows:
        photo_id = row["photo_id"]
        point_id = row["point_id"]
        if (not re.fullmatch(r"[0-9]{8}", point_id) or row["view"] not in tuple(VIEWS)
                or row["broad_class"] not in tuple(CLASSES)
                or photo_id != point_id + row["view"] or photo_id in seen):
            raise ValueError(f"Invalid or duplicate association: {photo_id}")
        seen.add(photo_id)
        rec = records[photo_id]
        if rec["sha256"] != row["sha256"] or rec["decision"] not in STATUSES or rec["photo_id"] != photo_id:
            raise ValueError(f"Invalid or stale review for {photo_id}")
        audit = {**row, **rec, "source_image_path": row["image_path"], "source_vector_path": row["vector_path"],
                 "duplicate_group": row["sha256"]}
        del audit["image_path"], audit["vector_path"]
        reviews.append(audit)
        if rec["decision"] == "exclude":
            excluded.append(audit)
        if rec["decision"] != "keep" or row["vector_status"] != "valid":
            continue
        original = _file(root, row["image_path"])
        photo_path = f"images/{row['broad_class']}/{photo_id}.jpg"
        dest = stage / photo_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, dest)
        point = {k: row[k] for k in POINT_FIELDS}
        vector_path = f"vectors/{point_id}.npz"
        if point_id not in points:
            vector_dest = stage / vector_path
            vector_dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(_file(root, row["vector_path"]), vector_dest)
            points[point_id] = {**point, "vector_path": vector_path,
                                "vector_sha256": _sha(vector_dest), "photo_ids": []}
        elif any(points[point_id][key] != value for key, value in point.items()):
            raise ValueError(f"Survey metadata conflict at {point_id}")
        points[point_id]["photo_ids"].append(photo_id)
        photos.append({**point, "photo_id": photo_id, "view": row["view"], "image_path": photo_path,
                       "vector_path": vector_path, "sha256": row["sha256"], "duplicate_group": row["sha256"],
                       "context": rec["observation"]["context"], "strength": rec["observation"]["strength"],
                       "reason": rec["observation"]["reason"]})
    if seen != records.keys():
        raise ValueError("Index and review journal contain different photo IDs")
    _jsonl(stage / "photos.jsonl", photos)
    _jsonl(stage / "points.jsonl", points.values())
    _jsonl(stage / "reviews.jsonl", reviews)
    _jsonl(stage / "excluded.jsonl", excluded)
    retention = _outcome_counts(reviews)
    _json(stage / "retention.json", retention)
    _json(stage / "identity.json", identity)
    for name in PROVENANCE_FILES:
        if (run / name).is_file():
            shutil.copyfile(_file(run, name), stage / name)
    for p in sorted(run.glob("evaluation*.json")):
        if p.is_file():
            shutil.copyfile(_file(run, p.name), stage / p.name)
    (stage / "DATASET_CARD.md").write_text(
        f"# UK LUCAS 2018 landscape photographs paired with Tessera\n\n"
        f"{len(photos):,} original JPEGs from {len(points):,} survey points were retained by the context filter "
        f"and have finite 128-component float32 embeddings. The reviewer assessed {len(reviews):,} "
        f"photos with {identity.get('model', 'the model recorded in identity.json')}.\n\n"
        "Photos carry the original point's LC1 and broad A–H class. Qwen saw the broad survey class and assessed "
        "whether the view was extremely out of context. Exclusion requires a reported obstruction, "
        "high evidence of an extreme context failure and no usable landscape context. Ambiguous and "
        "mixed natural landscapes remain eligible. Evidence strength is ordinal. The extraction coordinates "
        "are the survey TH_LAT and TH_LONG. The source does not document the Tessera embedding year.\n\n"
        "Load the folder with `from habibench.lucas.portable import CuratedDataset`. "
        "`CuratedDataset(folder)[i]` contains a PIL RGB image, an embedding and survey metadata. "
        "Sampling by point can limit the weight of points with many views. Group by point and duplicate "
        "hash when defining splits. No splits are assigned.\n\n"
        "photos.jsonl and points.jsonl index the included assets. reviews.jsonl records all decisions "
        "and original relative source paths. excluded.jsonl indexes clear outliers against the "
        "raw dataset. retention.json breaks down every decision by class and view. identity.json "
        "records the inference settings.\n", encoding="utf-8")
    metadata = {p.name: _sha(p) for p in stage.iterdir() if p.is_file()}
    _json(stage / "release.json", {"schema": SCHEMA_VERSION, "photos": len(photos), "points": len(points),
                                   "reviewed": len(reviews), "metadata_sha256": metadata,
                                   "input_sha256": input_digest})


def _input_digest(root, run, rows, records, identity):
    digest = hashlib.sha256()
    vectors = set()

    def update(value):
        digest.update(json.dumps(value, sort_keys=True).encode() + b"\n")

    update([SCHEMA_VERSION, identity])
    if len(records) != len(rows):
        raise ValueError("Index and review journal have different lengths")
    for row in rows:
        record = records[row["photo_id"]]
        update([row, record])
        if record["decision"] == "keep" and row["vector_status"] == "valid":
            vectors.add(row["vector_path"])
    for name in sorted(vectors):
        update([name, _sha(_file(root, name))])
    for name in (*PROVENANCE_FILES, *(p.name for p in sorted(run.glob("evaluation*.json")))):
        if (run / name).is_file():
            update([name, _sha(_file(run, name))])
    return digest.hexdigest()


def export_dataset(root: Path, run: Path, rows: list[dict], records: dict, identity: dict) -> Path:
    """Publish original assets and relative manifests as a single movable directory."""
    root = Path(root).resolve(strict=True)
    run = Path(run).resolve(strict=True)
    target = run / "curated"
    if (target.is_symlink() or (target.exists() and not (target / "release.json").is_file())):
        raise ValueError(f"Refusing to replace an unrelated export path: {target}")
    input_digest = _input_digest(root, run, rows, records, identity)
    if target.exists():
        release = json.loads(_file(target, "release.json").read_text())
        if release.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"Existing export uses an older policy. Choose a fresh run directory: {target}")
        if release.get("input_sha256") == input_digest:
            verify_export(target)
            return target
    with tempfile.TemporaryDirectory(dir=run, prefix=".curated-build-") as temporary:
        stage = Path(temporary) / "release"
        stage.mkdir()
        _build(root, stage, run, rows, records, identity, input_digest)
        verify_export(stage)
        if target.exists():
            _exchange(stage, target)
        else:
            stage.rename(target)
    return target


def verify_export(export_dir: Path) -> dict:
    """Verify checksums, point/view associations and usable training assets using only the release."""
    root = Path(export_dir).resolve(strict=True)
    release = json.loads(_file(root, "release.json").read_text())
    metadata = release["metadata_sha256"]
    schema = release["schema"]
    if schema not in (1, SCHEMA_VERSION):
        raise ValueError("Unknown HabiBench release version")
    kept, dropped, pool, statuses = ("core", "secondary", "secondary.jsonl", V1_STATUSES) if schema == 1 else (
        "keep", "exclude", "excluded.jsonl", STATUSES)
    if (not set((*MANIFESTS[:3], pool, "retention.json", "identity.json", "DATASET_CARD.md")) <= metadata.keys()):
        raise ValueError("Unknown or incomplete HabiBench release")
    for name, checksum in metadata.items():
        if Path(name).name != name or _sha(_file(root, name)) != checksum:
            raise ValueError(f"Metadata checksum mismatch: {name}")
    if {p.name for p in root.glob("evaluation*.json")} != {k for k in metadata if k.startswith("evaluation") and k.endswith(".json")}:
        raise ValueError("Untracked or missing evaluation report")
    dataset = CuratedDataset(root)
    photos = dataset.photos
    points = _read(root, "points.jsonl")
    reviews = _read(root, "reviews.jsonl")
    dropped_rows = _read(root, pool)
    expected = {(r["point_id"], r["photo_id"]) for r in reviews if r["decision"] == kept and r["vector_status"] == "valid"}
    if expected != {(r["point_id"], r["photo_id"]) for r in photos} or len(expected) != len(photos):
        raise ValueError("Paired training associations differ from the ledger")
    if dropped_rows != [r for r in reviews if r["decision"] == dropped]:
        raise ValueError(f"{pool} differs from the ledger")
    if len(reviews) != len({r["photo_id"] for r in reviews}) or any(r["decision"] not in statuses for r in reviews):
        raise ValueError("Duplicate or invalid review")
    if any(release[k] != n for k, n in (("photos", len(photos)), ("points", len(points)), ("reviewed", len(reviews)))):
        raise ValueError("Release counts differ from manifest counts")
    if _outcome_counts(reviews, schema) != json.loads(_file(root, "retention.json").read_text()):
        raise ValueError("Retention counts differ from the ledger")
    review_by_id = {r["photo_id"]: r for r in reviews}
    photo_groups = defaultdict(list)
    for row in photos:
        review = review_by_id[row["photo_id"]]
        if any(row[k] != review[k] for k in (*POINT_FIELDS, "view", "sha256", "duplicate_group")):
            raise ValueError(f"Photo metadata differs from review: {row['photo_id']}")
        if row["image_path"] != f"images/{row['broad_class']}/{row['photo_id']}.jpg":
            raise ValueError(f"Invalid release image path for {row['photo_id']}")
        photo_groups[row["point_id"]].append(row)
        img = _file(root, row["image_path"])
        if _sha(img) != row["sha256"] or row["duplicate_group"] != row["sha256"]:
            raise ValueError(f"JPEG checksum mismatch: {row['photo_id']}")
        with Image.open(img) as image:
            if image.format != "JPEG":
                raise ValueError(f"Expected JPEG: {row['image_path']}")
            image.verify()
    if len(points) != len(photo_groups) or set(photo_groups) != {p["point_id"] for p in points}:
        raise ValueError("Point and photo manifests contain different points")
    for point in points:
        group = photo_groups[point["point_id"]]
        if (point["vector_path"] != f"vectors/{point['point_id']}.npz"
                or point["photo_ids"] != [r["photo_id"] for r in group]
                or any(r[k] != point[k] for r in group for k in (*POINT_FIELDS, "vector_path"))):
            raise ValueError(f"Point relationships differ: {point['point_id']}")
        vector = _file(root, point["vector_path"])
        if _sha(vector) != point["vector_sha256"]:
            raise ValueError(f"Embedding checksum mismatch: {point['point_id']}")
        load_vector(vector)
    batch = [dataset[i] for i in range(min(4, len(dataset)))]
    return {"photos": len(photos), "points": len(points), "reviewed": len(reviews), "sample_batch": len(batch)}
