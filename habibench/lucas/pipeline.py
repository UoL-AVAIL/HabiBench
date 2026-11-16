"""Resumable review of independently identifiable photographs."""
import fcntl
import hashlib
import inspect
import json
import os
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from .data import build_selection
from .reviewer import decision, make_backend


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def policy_identity():
    return {"name": "lucas-landscape-context-v2", "sha256": digest(inspect.getsource(decision))}


def atomic_text(path, contents):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".tmp-", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_lines(path, rows):
    atomic_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def read_lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def check_selection(run, profile, scope):
    spec = json.loads((run / "selection.json").read_text())
    if spec.get("task") != "context-screen":
        raise ValueError("This run uses an earlier review task. Prepare a fresh context-screen run.")
    if spec["profile"] != profile or spec["scope"] != scope:
        raise ValueError(f"Run was prepared as {spec['profile']}/{spec['scope']}, not {profile}/{scope}")


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using {path.parent} ({path.name})") from exc
        yield


def prepare(root, run, profile, scope):
    rows, pilot = build_selection(root, profile, scope)
    if not rows:
        raise ValueError(f"No photos in {profile} scope {scope}")
    spec = {"profile": profile, "scope": scope, "index_digest": digest(rows), "task": "context-screen"}
    path = run / "selection.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Selection or raw dataset changed. Use a fresh --run-dir.")
    write_lines(run / "index.jsonl", rows)
    write_json(run / "pilot.json", pilot)
    write_json(path, spec)
    print(f"Selected {len(rows)} photos, {len(pilot)} blinded review items at {run}", flush=True)


def register_identity(run, value):
    path = run / "identity.json"
    frozen = run / "freeze.json"
    if frozen.exists() and json.loads(frozen.read_text()) != value:
        raise ValueError("Inference settings differ from the frozen tuning run.")
    with (run / ".identity.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError("Model, rubric or generation settings changed. Use a fresh --run-dir.")
        else:
            write_json(path, value)
    return digest(value)


def read_journal(path):
    if not path.exists():
        return {}
    records = {}
    with path.open("rb") as stream:
        for line in stream:
            if line.endswith(b"\n"):
                rec = json.loads(line)
                records[rec["photo_id"]] = rec
    return records


def repair_tail(path):
    """Discard an incomplete last append following interruption (under shard lock)."""
    if not path.exists() or not path.stat().st_size:
        return
    with path.open("rb+") as stream:
        stream.seek(-1, 2)
        if stream.read(1) == b"\n":
            return
        pos = stream.tell()
        while pos:
            start = max(0, pos - 65536)
            stream.seek(start)
            newline = stream.read(pos - start).rfind(b"\n")
            if newline >= 0:
                stream.truncate(start + newline + 1)
                return
            pos = start
        stream.truncate(0)


def setup_shards(run, count):
    path = run / "shards.json"
    with (run / ".shards.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and json.loads(path.read_text())["count"] != count:
            raise ValueError("Keep the original shard count when resuming this run.")
        write_json(path, {"count": count})


def review(root, run, profile, model, model_path, rank, count, batch_size):
    if not 0 <= rank < count or batch_size < 1:
        raise ValueError("Invalid shard or batch size")
    if json.loads((run / "selection.json").read_text())["profile"] != profile:
        raise ValueError("Requested profile differs from prepared selection")
    rows = read_lines(run / "index.jsonl")[rank::count]
    setup_shards(run, count)
    journal = run / f"shard-{rank:03d}.jsonl"
    with exclusive(run / f".shard-{rank:03d}.lock"):
        repair_tail(journal)
        backend = make_backend(profile, model, model_path)
        key = register_identity(run, backend.identity)
        write_json(run / f"worker-{rank}.json", {"rank": rank, "workers": count, "batch_size": batch_size})
        existing = read_journal(journal)
        pending = [r for r in rows if not matches(r, existing.get(r["photo_id"]), key)]
        start = time.monotonic()
        processed = 0
        benchmark_start = None
        with journal.open("a") as stream:
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset:offset + batch_size]
                outputs = backend.review_batch([(root / row["image_path"], row["view"], row["broad_class"])
                                                for row in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError("Backend returned the wrong number of observations")
                for row, output in zip(batch, outputs):
                    action = "error" if output.get("error") else decision(output["observation"], row["broad_class"])
                    rec = {**output, "photo_id": row["photo_id"], "sha256": row["sha256"], "target": row["broad_class"],
                           "decision": action, "identity_key": key}
                    stream.write(json.dumps(rec) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                processed += len(batch)
                if processed == len(batch):
                    benchmark_start = time.monotonic()
                    warm_count = len(batch)
                elif processed - warm_count >= (500 + count - 1) // count and not (run / f"timing-{rank}.json").exists():
                    duration = time.monotonic() - benchmark_start
                    write_json(run / f"timing-{rank}.json", {"timed_photos": processed-warm_count,
                               "seconds": duration, "shard_photos": len(rows),
                               "projected_shard_hours": len(rows)*duration/(processed-warm_count)/3600})
                if profile == "local-small" or processed % 128 < len(batch):
                    print(f"Shard {rank}: {processed}/{len(pending)} new reviews", flush=True)
        print(f"Shard {rank}: {processed} inference requests, {len(rows)-len(pending)} cached, "
              f"{time.monotonic()-start:.1f}s", flush=True)


def matches(row, rec, key):
    return (rec is not None and rec["identity_key"] == key and rec["sha256"] == row["sha256"]
            and rec.get("target") == row["broad_class"]
            and rec["decision"] != "error")


def finalize(root, run):
    from .portable import export_dataset, verify_export
    from .pilot import write_pilot
    rows = read_lines(run / "index.jsonl")
    identity = json.loads((run / "identity.json").read_text())
    key = digest(identity)
    policy = policy_identity()
    frozen_policy = run / "freeze-policy.json"
    if frozen_policy.exists() and json.loads(frozen_policy.read_text()) != policy:
        raise ValueError("Decision policy differs from the frozen tuning run")
    count = json.loads((run / "shards.json").read_text())["count"]
    records = {}
    for rank in range(count):
        journal = read_journal(run / f"shard-{rank:03d}.jsonl")
        for row in rows[rank::count]:
            rec = journal.get(row["photo_id"])
            if (not rec or rec.get("identity_key") != key or rec.get("sha256") != row["sha256"]
                    or rec.get("target") != row["broad_class"]):
                raise ValueError(f"Incomplete or stale shard {rank} at {row['photo_id']}; resume review.")
            if rec.get("observation") and not rec.get("error"):
                rec["decision"] = decision(rec["observation"], row["broad_class"])
            records[row["photo_id"]] = rec
    write_lines(run / "ledger.jsonl", (records[r["photo_id"]] for r in rows))
    write_pilot(root, run, rows, json.loads((run / "pilot.json").read_text()))
    release_identity = {**identity, "decision_policy": policy}
    write_json(run / "policy.json", policy)
    folder = export_dataset(root, run, rows, records, release_identity)
    checked = verify_export(folder)
    decisions = dict(Counter(r["decision"] for r in records.values()))
    summary = {"reviewed": len(rows), "decisions": decisions, "export": str(folder),
               "photos_without_valid_vector": sum(r["vector_status"] != "valid" for r in rows),
               "kept_without_valid_vector": sum(r["vector_status"] != "valid" and
                   records[r["photo_id"]]["decision"] == "keep" for r in rows), "verified": checked}
    if json.loads((run / "selection.json").read_text())["profile"] == "local-small":
        from .gallery import write_gallery
        summary["gallery"] = str(write_gallery(root, run, rows, records, identity))
    times = [json.loads(p.read_text()) for p in sorted(run.glob("timing-*.json"))]
    if len(times) == count:
        summary["projected_inference_hours"] = max(t["projected_shard_hours"] for t in times)
    write_json(run / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return decisions.get("error", 0)
