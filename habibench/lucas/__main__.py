"""Command-line entry point for HabiBench LUCAS curation."""
import argparse
import json
import sys
import time
from pathlib import Path

from . import pipeline
from .reviewer import NOVEL_MODEL, prompt_preview


def parser():
    cli = argparse.ArgumentParser(prog="python -m habibench.lucas", description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("run", "prepare", "worker", "finalize"):
        p = commands.add_parser(name)
        p.add_argument("--profile", choices=("local-small", "novel"), default="local-small")
        p.add_argument("--data-root", type=Path, default=Path("data/lucas_tessera"))
        p.add_argument("--run-dir", type=Path)
        p.add_argument("--scope", choices=("all", "tune", "audit", "pilot"), default="all")
        p.add_argument("--model")
        p.add_argument("--model-path", type=Path)
        p.add_argument("--batch-size", type=int)
        p.add_argument("--freeze-from", type=Path, help="Require the inference configuration of a completed tuning run")
        if name == "worker":
            p.add_argument("--shard-count", type=int, default=1)
            p.add_argument("--shard-index", type=int, default=0)
    p = commands.add_parser("evaluate")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--split", choices=("tune", "audit"))
    p = commands.add_parser("attach-evaluation", help="Validate a completed pilot evaluation against a full run")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--source-run", required=True, type=Path)
    p.add_argument("--split", required=True, choices=("tune", "audit"))
    p = commands.add_parser("fetch-model")
    p.add_argument("--model-dir", required=True, type=Path)
    p = commands.add_parser("verify")
    p.add_argument("export_dir", type=Path)
    p = commands.add_parser("prompt")
    p.add_argument("--view", choices=list("PNESWC"), default="P")
    p.add_argument("--class", dest="broad_class", required=True, choices=list("ABCDEFGH"))
    return cli


def main(argv=None):
    a = parser().parse_args(argv)
    if a.command == "prompt":
        print(json.dumps(prompt_preview(a.view, a.broad_class), indent=2))
        return 0
    if a.command == "fetch-model":
        from .download import fetch_model
        print(fetch_model(a.model_dir))
        return 0
    if a.command == "verify":
        from .portable import verify_export
        print(json.dumps(verify_export(a.export_dir), indent=2))
        return 0
    if a.command == "evaluate":
        from .pilot import evaluate
        print(json.dumps(evaluate(a.run, a.labels, a.split), indent=2))
        return 0
    if a.command == "attach-evaluation":
        from .audit import attach_evaluation
        attach_evaluation(a.run, a.source_run, a.split)
        print(f"Attached {a.split} evaluation to {a.run}. Run finalize to include it in the export.")
        return 0
    root = a.data_root.resolve()
    run = (a.run_dir or Path("runs") / f"{a.profile}-context").resolve()
    if root == run or root in run.parents:
        raise ValueError("Choose a run directory outside the raw dataset")
    run.mkdir(parents=True, exist_ok=True)
    model = a.model or ("qwen3-vl:8b" if a.profile == "local-small" else NOVEL_MODEL)
    batch = a.batch_size if a.batch_size is not None else (1 if a.profile == "local-small" else 8)
    start = time.monotonic()
    if a.command in ("run", "prepare"):
        with pipeline.exclusive(run / ".prepare.lock"):
            pipeline.prepare(root, run, a.profile, a.scope)
            if a.command == "prepare":
                from .pilot import write_pilot
                write_pilot(root, run, pipeline.read_lines(run / "index.jsonl"),
                            json.loads((run / "pilot.json").read_text()))
            if a.freeze_from:
                baseline = a.freeze_from
                base_spec = json.loads((baseline / "selection.json").read_text())
                base_stats = json.loads((baseline / "summary.json").read_text())
                if (base_spec["profile"] != "novel" or base_spec["scope"] != "tune"
                        or base_stats["decisions"].get("error", 0)):
                    raise ValueError("Freezing requires a completed error-free Novel tuning run.")
                expected = json.loads((baseline / "identity.json").read_text())
                frozen = run / "freeze.json"
                if frozen.exists() and json.loads(frozen.read_text()) != expected:
                    raise ValueError("Run already references a different frozen identity.")
                pipeline.write_json(frozen, expected)
                frozen_policy = json.loads((baseline / "policy.json").read_text())
                if frozen_policy != pipeline.policy_identity():
                    raise ValueError("Decision policy differs from the frozen tuning run")
                pipeline.write_json(run / "freeze-policy.json", frozen_policy)
    if a.command in ("finalize", "worker"):
        pipeline.check_selection(run, a.profile, a.scope)
    if a.command in ("run", "worker"):
        pipeline.review(root, run, a.profile, model, a.model_path, getattr(a, "shard_index", 0),
                        getattr(a, "shard_count", 1), batch)
    if a.command in ("run", "finalize"):
        with pipeline.exclusive(run / ".finalize.lock"):
            errors = pipeline.finalize(root, run)
        print(f"Elapsed: {time.monotonic()-start:.1f}s. Run directory: {run}", flush=True)
        return 1 if errors else 0
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(f"LUCAS: {exc}")
