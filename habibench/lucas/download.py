"""Stage and verify the pinned Novel snapshot on a shared POSIX filesystem."""

import fcntl
import hashlib
import json
from pathlib import Path

from .reviewer import NOVEL_MODEL, NOVEL_REVISION


MANIFEST = "HABIBENCH_FILES.json"
REQUIRED = {
    "config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
    "chat_template.jinja", "model.safetensors.index.json",
}


def _metadata_and_download(staging):
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub>=0.32 on the connected staging host.") from exc
    info = HfApi().model_info(NOVEL_MODEL, revision=NOVEL_REVISION, files_metadata=True)
    if info.sha != NOVEL_REVISION:
        raise ValueError(f"Unexpected upstream revision: {info.sha}")
    files = {}
    for entry in info.siblings:
        name = entry.rfilename
        if name == "LICENSE" or (Path(name).name == name and
                                 Path(name).suffix in {".safetensors", ".json", ".jinja", ".txt"}):
            if entry.lfs:
                files[name] = {"size": entry.size, "sha256": entry.lfs.sha256}
            else:
                files[name] = {"size": entry.size, "git_blob": entry.blob_id}
    if not REQUIRED <= files.keys():
        raise ValueError(f"Upstream snapshot lacks required files: {sorted(REQUIRED - files.keys())}")
    snapshot_download(repo_id=NOVEL_MODEL, revision=NOVEL_REVISION, local_dir=staging,
                      allow_patterns=list(files))
    return files


def _verify(directory, files):
    if not REQUIRED <= files.keys():
        raise ValueError(f"Incomplete snapshot manifest at {directory}")
    for name, expected in files.items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid snapshot filename: {name}")
        path = directory / name
        if not path.is_file() or path.stat().st_size != expected["size"] or not expected["size"]:
            raise ValueError(f"Missing, truncated or empty snapshot file: {path}")
        if "sha256" in expected:
            digest = hashlib.sha256()
            checksum = expected["sha256"]
        else:
            digest = hashlib.sha1(f"blob {expected['size']}\0".encode(), usedforsecurity=False)
            checksum = expected["git_blob"]
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != checksum:
            raise ValueError(f"Checksum mismatch: {path}")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    listed = {name for name in files if name.endswith(".safetensors")}
    if not shards or shards != listed:
        raise ValueError("Safetensors index and verified weight files disagree")
    processor = json.loads((directory / "preprocessor_config.json").read_text())
    if not processor.get("image_processor_type") or not processor.get("processor_class"):
        raise ValueError("Snapshot lacks the Qwen image processor configuration")


def fetch_model(model_dir: Path) -> Path:
    """Return model_dir/revision, verifying integrity even on an offline repeat."""
    root = Path(model_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    published = root / NOVEL_REVISION
    staging = root / f".{NOVEL_REVISION}.partial"
    with (root / f".{NOVEL_REVISION}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if published.exists():
            marker = published / "HABIBENCH_REVISION"
            if not marker.is_file() or marker.read_text().strip() != NOVEL_REVISION:
                raise ValueError(f"Unverified existing model directory: {published}")
            files = json.loads((published / MANIFEST).read_text())
            _verify(published, files)
            return published
        staging.mkdir(exist_ok=True)
        files = _metadata_and_download(staging)
        _verify(staging, files)
        (staging / MANIFEST).write_text(json.dumps(files, sort_keys=True, indent=2) + "\n")
        (staging / "HABIBENCH_REVISION").write_text(NOVEL_REVISION + "\n")
        staging.rename(published)
        return published
