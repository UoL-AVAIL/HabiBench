"""Join the UK 2018 LUCAS photographs, survey records and point-level Tessera vectors."""

import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


CLASSES = "ABCDEFGH"
VIEWS = "PNESWC"
PHOTO_NAME = re.compile(r"\d{3}_\d{3}_(\d{8})([PNESWC])\.jpg", re.IGNORECASE)
SURVEY_FOOTER = re.compile(r"[\d.,]+ rows selected\.")
LOCAL_CASES = {
    "A": ("32323620N",),  # Photo with a placeholder Tessera vector.
    "C": ("31783608P",),
    "E": ("31623628E", "31643632N", "35963888C", "35983886C"),
    "G": ("34063818P", "34063818W"),
    "H": ("32523662W",),
}


def load_vector(path: Path) -> np.ndarray:
    """Read a native float32 vector, rejecting pickle placeholders and malformed data."""
    with np.load(path, allow_pickle=False) as archive:
        vector = archive["arr_0"]
    if vector.shape != (128,):
        raise ValueError("wrong_shape")
    if vector.dtype != np.float32:
        raise ValueError("wrong_dtype")
    if not np.isfinite(vector).all():
        raise ValueError("nonfinite")
    return vector


def _vector_status(path: Path) -> str:
    try:
        load_vector(path)
    except FileNotFoundError:
        return "missing_file"
    except KeyError:
        return "missing_arr_0"
    except ValueError as exc:
        detail = str(exc)
        if "Object arrays cannot be loaded" in detail:
            return "object_placeholder"
        if detail in {"wrong_shape", "wrong_dtype", "nonfinite"}:
            return detail
        return "invalid_archive"
    except (BadZipFile, EOFError, TypeError):
        return "invalid_archive"
    return "valid"


def _table_by_key(path: Path, key: str) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        table = {}
        for row in csv.DictReader(stream):
            value = row[key]
            if key == "POINT_ID" and SURVEY_FOOTER.fullmatch(value):
                continue
            if key == "POINT_ID" and not re.fullmatch(r"\d{8}", value):
                raise ValueError(f"Unexpected survey point ID in {path}: {value!r}")
            if value in table:
                raise ValueError(f"Duplicate {key} in {path}: {value}")
            table[value] = row
    return table


def _rank(purpose: str, row: dict) -> bytes:
    return hashlib.sha256(f"lucas-2018-v1:{purpose}:{row['photo_id']}".encode()).digest()


def _select_local(rows: list[dict]) -> list[dict]:
    """Select associations using names alone; read photo bytes only after this stage."""
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["broad_class"]].append(row)
    selection = []
    for code in CLASSES:
        candidates = sorted(by_class[code], key=lambda row: _rank("local", row))
        by_id = {row["photo_id"]: row for row in candidates}
        required = LOCAL_CASES.get(code, ())
        absent = set(required) - by_id.keys()
        if absent:
            raise ValueError(f"Local regression photos absent from class {code}: {sorted(absent)}")
        picked = [by_id[photo_id] for photo_id in required]
        seen_points = {row["point_id"] for row in picked}
        for view in "PNCEWS":
            if len(picked) == 4:
                break
            if any(row["view"] == view for row in picked):
                continue
            row = next((r for r in candidates if r["view"] == view and r["point_id"] not in seen_points), None)
            if row is not None:
                picked.append(row)
                seen_points.add(row["point_id"])
        if len(picked) != 4:
            raise ValueError(f"Need four local photos for class {code}, found {len(picked)}")
        selection.extend(picked)
    return selection


def _photo_rows(root: Path, profile: str) -> list[dict]:
    if profile not in {"local-small", "novel"}:
        raise ValueError(f"Unknown profile: {profile}")
    root = Path(root)
    survey = _table_by_key(root / "UK_2018_20200213.csv", "POINT_ID")
    habitats = _table_by_key(root / "habitats.csv", "code")
    rows = []
    seen = set()
    for path in (root / "lucas-dataset").glob("*/*"):
        if path.suffix.lower() != ".jpg":
            continue
        match = PHOTO_NAME.fullmatch(path.name)
        if not match:
            raise ValueError(f"Could not parse LUCAS photo filename: {path}")
        point_id, view = match.group(1), match.group(2).upper()
        photo_id = point_id + view
        if photo_id in seen:
            raise ValueError(f"Duplicate point/view association: {photo_id}")
        seen.add(photo_id)
        if point_id not in survey:
            raise ValueError(f"Photo {photo_id} has no survey row")
        metadata = survey[point_id]
        lc1 = metadata["LC1"]
        if lc1 not in habitats or lc1[0] not in CLASSES:
            raise ValueError(f"Unknown LC1 {lc1!r} for {photo_id}")
        if path.parent.name.partition("_")[0] != lc1[0]:
            raise ValueError(f"Photo folder conflicts with survey LC1 {lc1}: {path}")
        rows.append({
            "photo_id": photo_id, "point_id": point_id, "view": view,
            "image_path": path.relative_to(root).as_posix(),
            "lc1": lc1, "broad_class": lc1[0], "habitat": habitats[lc1]["habitat"],
            "latitude": float(metadata["TH_LAT"]), "longitude": float(metadata["TH_LONG"]),
            "vector_path": f"geospatial/{point_id}.npz",
        })
    if not rows:
        raise ValueError(f"No LUCAS JPEGs under {root / 'lucas-dataset'}")
    if profile == "local-small":
        rows = _select_local(rows)
    return sorted(rows, key=lambda r: r["photo_id"])


def _photo_hash(root, row):
    if "sha256" not in row:
        with (root / row["image_path"]).open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        row["sha256"] = digest.hexdigest()
    return row["sha256"]


def _validate_rows(root, rows):
    vector_status = {}
    for index, row in enumerate(rows, start=1):
        _photo_hash(root, row)
        point_id = row["point_id"]
        if point_id not in vector_status:
            vector_status[point_id] = _vector_status(root / row["vector_path"])
        row["vector_status"] = vector_status[point_id]
        if index % 5000 == 0:
            print(f"Checked {index}/{len(rows)} photos", flush=True)

def build_index(root: Path, profile: str) -> list[dict]:
    """Return the validated index for the whole profile."""
    root = Path(root)
    rows = _photo_rows(root, profile)
    _validate_rows(root, rows)
    return rows


def build_selection(root, profile, scope):
    root = Path(root)
    print("Reading LUCAS survey records and photo filenames...", flush=True)
    rows = _photo_rows(root, profile)
    print(f"Indexed {len(rows)} photo filenames. Selecting the {scope} sample.", flush=True)
    pilot = select_pilot(rows, profile, root=root, include_audit=scope != "tune")
    if scope != "all":
        pilot = [item for item in pilot if scope == "pilot" or item["split"] == scope]
        ids = {item["photo_id"] for item in pilot}
        rows = [row for row in rows if row["photo_id"] in ids]
    print(f"Checking image hashes and point vectors for {len(rows)} photos...", flush=True)
    _validate_rows(root, rows)
    return rows, pilot


def select_pilot(rows: list[dict], profile: str, *, root=None, include_audit=True) -> list[dict]:
    """Return blinded identities for the local 8 or the point/hash-disjoint 48+48 pilot."""
    if profile not in {"local-small", "novel"}:
        raise ValueError(f"Unknown profile: {profile}")
    local = _select_local(rows)
    seen_points, seen_hashes = set(), set()
    pilot = []

    def add(row, split):
        seen_points.add(row["point_id"])
        seen_hashes.add(_photo_hash(root, row))
        pilot.append({
            "photo_id": row["photo_id"], "split": split,
            "blind_id": _rank("blind", row).hex()[:20],
        })

    by_class = defaultdict(list)
    for row in rows:
        by_class[row["broad_class"]].append(row)
    tune_start = {}
    for index, code in enumerate(CLASSES):
        # Vary the viewing conditions represented in the local annotation sheet.
        choices = sorted((r for r in local if r["broad_class"] == code),
                         key=lambda r: (r["view"] != "PNCE"[index % 4], _rank("pilot", r)))
        choice = next(r for r in choices if r["point_id"] not in seen_points and _photo_hash(root, r) not in seen_hashes)
        tune_start[code] = choice
        add(choice, "tune")
    if profile == "novel":
        for split in (("tune", "audit") if include_audit else ("tune",)):
            if split == "audit":
                # Keep the entire local development gallery, including the known
                # anomalies, their other views and identical copies, out of the audit.
                seen_points.update(r["point_id"] for r in local)
                seen_hashes.update(_photo_hash(root, r) for r in local)
            for code in CLASSES:
                candidates = sorted(by_class[code], key=lambda r: _rank(split, r))
                view_counts = Counter([tune_start[code]["view"]] if split == "tune" else [])
                while view_counts.total() < 6:
                    # Stable ordering preserves the split rank for equally represented
                    # views. Stop reading JPEGs as soon as a unique candidate wins.
                    ordered = sorted(candidates, key=lambda r: view_counts[r["view"]])
                    choice = next((r for r in ordered if r["point_id"] not in seen_points
                                   and _photo_hash(root, r) not in seen_hashes), None)
                    if choice is None:
                        raise ValueError(f"Too few distinct points/photos for {split} class {code}")
                    add(choice, split)
                    view_counts[choice["view"]] += 1
    return sorted(pilot, key=lambda r: (r["split"] != "tune", r["blind_id"]))
