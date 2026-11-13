"""Exercise the actual 27B processor, grammar and GPU kernels on one real photo."""

import json
import sys
from pathlib import Path

from habibench.lucas.reviewer import NOVEL_MODEL, make_backend


def main():
    snapshot = Path(sys.argv[1])
    path = Path("/raw/lucas-dataset/C_woodland/317_836_31783608P.jpg")
    backend = make_backend("novel", NOVEL_MODEL, snapshot)
    result, = backend.review_batch([(path, "P", "C")])
    print(json.dumps({"identity": backend.identity, "result": result}, indent=2), flush=True)
    if "observation" not in result:
        raise SystemExit("Qwen failed the one-photo smoke test")


if __name__ == "__main__":
    main()
