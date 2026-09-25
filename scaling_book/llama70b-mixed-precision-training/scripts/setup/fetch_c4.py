#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

REPO_ID = "allenai/c4"
REVISION = "main"
REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("DATA_ROOT", str(REPO / "data")))
DEST = DATA / "c4-en"


def digest(path: Path, limit: int = 1 << 20) -> str:
    value = hashlib.sha256()
    size = path.stat().st_size
    value.update(str(size).encode())
    with path.open("rb") as handle:
        value.update(handle.read(limit))
        if size > limit:
            handle.seek(max(0, size - limit))
            value.update(handle.read(limit))
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", type=int, default=10)
    parser.add_argument("--eval-shards", type=int, default=2)
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    jobs = [
        (f"en/c4-train.{index:05d}-of-01024.json.gz", "train")
        for index in range(args.train_shards)
    ]
    jobs += [
        (f"en/c4-validation.{index:05d}-of-00008.json.gz", "validation")
        for index in range(args.eval_shards)
    ]

    files: dict[str, dict[str, int | str]] = {}
    manifest: dict[str, object] = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "files": files,
    }
    for name, split in jobs:
        output = args.dest / split / Path(name).name
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            cached = hf_hub_download(
                REPO_ID, name, repo_type="dataset", revision=REVISION
            )
            shutil.copyfile(cached, output)
        files[f"{split}/{output.name}"] = {
            "bytes": output.stat().st_size,
            "digest": digest(output),
        }
        print(output)

    blob = json.dumps(files, sort_keys=True).encode()
    manifest["manifest_sha256"] = hashlib.sha256(blob).hexdigest()
    manifest["total_bytes"] = sum(int(item["bytes"]) for item in files.values())
    path = args.dest / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
