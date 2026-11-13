"""Ground-truth-conditioned human ratings, blinded to automated decisions."""

import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from .pipeline import atomic_text, exclusive, policy_identity, read_lines, write_json


RATING_FIELDS = ("human_decision", "reason")
SHEET_FIELDS = ("blind_id", "image", "view", "broad_class", "habitat", *RATING_FIELDS)


def write_pilot(root, run, rows, pilot):
    """Publish anonymised images and empty templates without changing any existing ratings."""
    by_id = {r["photo_id"]: r for r in rows}
    for split in ("tune", "audit"):
        selected = [p for p in pilot if p["split"] == split]
        if not selected:
            continue
        folder = run / "pilot" / split
        (folder / "images").mkdir(parents=True, exist_ok=True)
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        for item in sorted(selected, key=lambda p: p["blind_id"]):
            row = by_id[item["photo_id"]]
            blind = item["blind_id"]
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", blind):
                raise ValueError(f"Invalid blind ID: {blind!r}")
            image_path = f"images/{blind}.jpg"
            target = folder / image_path
            if not target.exists():
                with Image.open(root / row["image_path"]) as source:
                    photo = ImageOps.exif_transpose(source).convert("RGB")
                photo.info.clear()
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
                    tmp = Path(out.name)
                    try:
                        photo.save(out, format="JPEG", quality=94)
                    except BaseException:
                        tmp.unlink(missing_ok=True)
                        raise
                os.replace(tmp, target)
            view = {"P": "point", "C": "close-up"}.get(row["view"], "compass")
            writer.writerow({"blind_id": blind, "image": image_path, "view": view,
                             "broad_class": row["broad_class"], "habitat": row["habitat"]})
        sheet = folder / "ratings.csv"
        if sheet.exists():
            with sheet.open(encoding="utf-8-sig", newline="") as stream:
                if tuple(csv.DictReader(stream).fieldnames or ()) != SHEET_FIELDS:
                    raise ValueError(f"Existing sheet belongs to another task. Use a fresh run: {sheet}")
        else:
            atomic_text(sheet, output.getvalue())
        atomic_text(folder / "instructions.txt",
                    "Open each image and consider the broad landscape class recorded at the survey point. "
                    "A compass view may show the surroundings. A close-up intentionally shows detail.\n\n"
                    "Enter keep for usable, mixed or uncertain views. A grassy-looking wetland, a distant "
                    "house beside a field and a facade on artificial land can all be useful. Enter exclude "
                    "only for an obvious context failure, such as a van or building filling a grassland "
                    "frame and obscuring the landscape. Leave human_decision blank for pending review. "
                    "A short reason is optional. Work independently of Qwen's decisions.\n")


def _metrics(pairs):
    counts = Counter(pairs)
    correct = counts["exclude", "exclude"]
    removed_good = counts["keep", "exclude"]
    proposed = correct + removed_good
    good = sum(n for (human, _), n in counts.items() if human == "keep")
    outliers = len(pairs) - good
    failures = counts["keep", "error"] + counts["exclude", "error"]
    return {
        "labelled": len(pairs), "inference_success": len(pairs) - failures, "inference_errors": failures,
        "inference_coverage": (len(pairs) - failures) / len(pairs) if pairs else None,
        "proposed_exclusions": proposed, "correct_exclusions": correct,
        "exclusion_precision": correct / proposed if proposed else None,
        "human_keep": good, "good_photos_removed": removed_good,
        "good_photo_removal_rate": removed_good / good if good else None,
        "human_exclude": outliers, "outliers_kept": counts["exclude", "keep"],
        "outliers_with_inference_error": counts["exclude", "error"],
        "usable_with_inference_error": counts["keep", "error"],
        "outlier_recall": correct / outliers if outliers else None,
    }


def _save_ratings(path, contents):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


def evaluate(run, labels, split=None, *, persist=True):
    """Evaluate the supplied human ratings independently for a single pilot phase."""
    run, labels = Path(run), Path(labels)
    pilot = json.loads((run / "pilot.json").read_text())
    all_items = {i["blind_id"]: i for i in pilot}
    if len(all_items) != len(pilot):
        raise ValueError("Duplicate blind ID in pilot")
    input_bytes = labels.read_bytes()
    with io.StringIO(input_bytes.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"blind_id", "broad_class", "human_decision"} <= set(reader.fieldnames or []):
            raise ValueError("Anomaly ratings require blind_id, broad_class and human_decision. "
                             "Earlier classification ratings cannot be reused.")
        ratings = list(reader)
    submitted = {}
    for row in ratings:
        key = (row["blind_id"] or "").strip()
        if key in submitted or key not in all_items:
            raise ValueError(f"Duplicate or unmatched blind_id: {key!r}")
        submitted[key] = row
    if split is None:
        phases = {all_items[k]["split"] for k in submitted}
        if len(phases) != 1:
            raise ValueError("Use one tune or audit rating sheet per evaluation")
        split = phases.pop()
    selected = {k: i for k, i in all_items.items() if i["split"] == split}
    if not selected or not submitted.keys() <= selected.keys():
        raise ValueError(f"Rating sheet contains items outside the requested {split} phase")
    policy = policy_identity()
    saved_policy = run / "policy.json"
    if saved_policy.exists() and json.loads(saved_policy.read_text()) != policy:
        raise ValueError("Decision policy differs from the finalized run. Use the matching policy version.")
    index = {r["photo_id"]: r for r in read_lines(run / "index.jsonl")}
    ledger = {r["photo_id"]: r for r in read_lines(run / "ledger.jsonl")}
    pairs, disagreements = [], []
    per_class, per_view = defaultdict(list), defaultdict(list)
    for key, rating in submitted.items():
        try:
            photo_id = selected[key]["photo_id"]
            row = index[photo_id]
            if (rating["broad_class"] or "").strip().upper() != row["broad_class"]:
                raise ValueError("Rating target does not match the survey broad class")
            human = (rating["human_decision"] or "").strip().lower()
            if not human:
                continue
            if human not in ("keep", "exclude"):
                raise ValueError("human_decision must be keep, exclude or blank")
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid human rating for {key}: {exc}") from exc
        if photo_id not in ledger:
            raise ValueError(f"No inference record for {key}; finalize the review first")
        rec = ledger[photo_id]
        predicted = rec["decision"]
        if predicted not in ("keep", "exclude", "error"):
            raise ValueError("Review ledger belongs to another policy. Prepare a fresh anomaly run.")
        pairs.append((human, predicted))
        per_class[row["broad_class"]].append((human, predicted))
        per_view[row["view"]].append((human, predicted))
        if human != predicted:
            disagreements.append({"blind_id": key, "photo_id": photo_id, "human": human,
                                  "predicted": predicted, "class": row["broad_class"], "view": row["view"],
                                  "human_reason": (rating.get("reason") or "").strip(),
                                  "model_reason": (rec.get("observation") or {}).get("reason"),
                                  "inference_error": rec.get("error")})
    if not pairs:
        raise ValueError("No completed human ratings in the sheet")
    identity = json.loads((run / "identity.json").read_text())
    report = {
        "split": split, "expected_labels": len(selected), "unlabelled": len(selected) - len(pairs),
        "label_coverage": len(pairs) / len(selected),
        "model": identity["model"], "identity": identity, "policy_fingerprint": policy,
        "ratings_sha256": hashlib.sha256(input_bytes).hexdigest(),
        **_metrics(pairs),
        "by_class": {k: _metrics(v) for k, v in sorted(per_class.items())},
        "by_view": {k: _metrics(v) for k, v in sorted(per_view.items())},
        "disagreements": disagreements,
    }
    if persist:
        name = f"evaluation-{split}.json"
        with exclusive(run / f".evaluate-{split}.lock"):
            _save_ratings(run / f"ratings-{split}.csv", input_bytes)
            write_json(run / name, report)
    return report
