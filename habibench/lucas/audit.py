"""Attach pilot evidence to the predictions made by a finalized production run."""

import json
from pathlib import Path

from .pilot import evaluate
from .pipeline import digest, exclusive, policy_identity, write_json


def _json(run, filename):
    return json.loads((run / filename).read_text())


def _pilot_assignments(run, split):
    items = [p for p in _json(run, "pilot.json") if p["split"] == split]
    assignments = {p["photo_id"]: p["blind_id"] for p in items}
    if not items or len(assignments) != len(items) or len(set(assignments.values())) != len(items):
        raise ValueError(f"Missing or duplicate pilot assignments in {run}")
    return assignments


def _pilot_rows(run, filename, wanted):
    rows = {}
    with (run / filename).open() as stream:
        for line in stream:
            row = json.loads(line)
            photo_id = row["photo_id"]
            if photo_id in wanted:
                if photo_id in rows:
                    raise ValueError(f"Duplicate {photo_id} in {run / filename}")
                rows[photo_id] = row
    if rows.keys() != wanted.keys():
        raise ValueError(f"Incomplete pilot coverage in {run / filename}")
    return rows


def attach_evaluation(full_run: Path, source_run: Path, split: str) -> dict:
    """Replay archived human ratings against both ledgers, publishing the full-run result."""
    full, source = Path(full_run).resolve(), Path(source_run).resolve()
    if split not in ("tune", "audit") or full == source:
        raise ValueError("Supply a distinct pilot run and its tune or audit split")
    for run, scope in ((full, "all"), (source, split)):
        spec = _json(run, "selection.json")
        if (spec["profile"], spec["scope"]) != ("novel", scope):
            raise ValueError(f"Expected a Novel {scope} run at {run}")
        if not (run / "summary.json").is_file():
            raise ValueError(f"Finalize {run} before attaching its evaluation")
    with (exclusive(full / ".finalize.lock"), exclusive(source / ".finalize.lock"),
          exclusive(full / f".evaluate-{split}.lock"), exclusive(source / f".evaluate-{split}.lock")):
        policy = _json(full, "policy.json")
        if policy != _json(source, "policy.json") or policy != policy_identity():
            raise ValueError("Current, source and full decision policies must match")
        identity = _json(full, "identity.json")
        if identity != _json(source, "identity.json"):
            raise ValueError("Source and full model identities must match")
        assignments = _pilot_assignments(source, split)
        if assignments != _pilot_assignments(full, split):
            raise ValueError("Pilot photos or blind IDs differ between runs")

        identity_key = digest(identity)
        indexes, ledgers = [], []
        for run in (source, full):
            index = _pilot_rows(run, "index.jsonl", assignments)
            ledger = _pilot_rows(run, "ledger.jsonl", assignments)
            for photo_id, row in index.items():
                record = ledger[photo_id]
                if record["sha256"] != row["sha256"] or record["identity_key"] != identity_key:
                    raise ValueError(f"Ledger image checksum or model identity differs for {photo_id}")
            indexes.append(index)
            ledgers.append(ledger)
        for photo_id, row in indexes[0].items():
            for field in ("sha256", "point_id", "view", "broad_class", "lc1"):
                if row[field] != indexes[1][photo_id][field]:
                    raise ValueError(f"Pilot photo {photo_id} differs in {field}")

        filename = f"evaluation-{split}.json"
        ratings = source / f"ratings-{split}.csv"
        saved = _json(source, filename)
        replay = evaluate(source, ratings, split, persist=False)
        if replay != saved:
            raise ValueError("Saved pilot evaluation differs from replay with its archived ratings")
        if saved["labelled"] != len(assignments) or saved["unlabelled"]:
            raise ValueError("Complete every pilot rating before attaching the evaluation")
        report = evaluate(full, ratings, split, persist=False)
        if report["ratings_sha256"] != replay["ratings_sha256"]:
            raise ValueError("Archived ratings changed during evaluation")
        changed = sorted(
            assignments[photo_id] for photo_id in assignments
            if ledgers[0][photo_id]["decision"] != ledgers[1][photo_id]["decision"]
        )
        report["pilot_provenance"] = {
            "source_run": source.name,
            "source_scope": split,
            "source_evaluation_sha256": digest(saved),
            "source_pilot_ledger_sha256": digest(ledgers[0]),
            "full_pilot_ledger_sha256": digest(ledgers[1]),
            "decision_policy": policy,
            "metrics_predictions": "full_run",
            "changed_decision_blind_ids": changed,
        }
        write_json(full / filename, report)
        return report
