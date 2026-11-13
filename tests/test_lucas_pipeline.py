import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from habibench.lucas import pipeline


def test_restart_and_incomplete_append(tmp_path):
    rows = [{"photo_id": f"point{i}P", "sha256": str(i), "view": "P", "image_path": f"{i}.jpg",
             "broad_class": "E"} for i in range(3)]
    pipeline.write_lines(tmp_path / "index.jsonl", rows)
    pipeline.write_json(tmp_path / "selection.json", {"profile": "local-small", "scope": "all"})
    observation = dict(context="ambiguous", obstruction="none", landscape_visible=True,
                       strength="high", reason="Grasses are visible, but the cover is difficult to distinguish.")
    ok = {"observation": observation, "raw": json.dumps(observation)}
    backend = Mock(identity={"backend": "ollama", "model": "test"}, review_batch=Mock(
        side_effect=[[ok, {"raw": "broken", "error": "invalid"}, ok], [ok]]))
    def run():
        pipeline.review(tmp_path, tmp_path, "local-small", "test", None, 0, 1, 3)
    with patch("habibench.lucas.pipeline.make_backend", return_value=backend):
        run()
        with (tmp_path / "shard-000.jsonl").open("ab") as stream:
            stream.write(b'{"photo_id":')
        run()
        run()
        assert backend.review_batch.call_count == 2
        assert len(backend.review_batch.call_args_list[1].args[0]) == 1
        pipeline.write_json(tmp_path / "freeze.json", {"model": "something else"})
        with pytest.raises(ValueError, match="frozen"):
            run()
    ledger = pipeline.read_journal(tmp_path / "shard-000.jsonl")
    assert len(ledger) == 3
    assert all(r["decision"] == "keep" for r in ledger.values())


def test_shard_merge_rejects_stale_row(tmp_path):
    pipeline.write_lines(tmp_path / "index.jsonl", [{"photo_id":"p", "sha256":"new"}])
    pipeline.write_json(tmp_path / "identity.json", {"model":"test"})
    pipeline.write_json(tmp_path / "shards.json", {"count": 1})
    rec = {"photo_id": "p", "decision": "core", "sha256": "old",
           "identity_key": pipeline.digest({"model": "test"})}
    pipeline.write_lines(tmp_path / "shard-000.jsonl", [rec])
    with pytest.raises(ValueError, match="stale shard"):
        pipeline.finalize(tmp_path, tmp_path)
