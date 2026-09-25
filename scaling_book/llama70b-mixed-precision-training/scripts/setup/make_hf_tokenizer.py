#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAXTEXT = Path(os.environ.get("MAXTEXT_ROOT", "/workspace/maxtext"))
DATA = Path(os.environ.get("DATA_ROOT", str(REPO / "data")))
SENTENCEPIECE = MAXTEXT / "src" / "maxtext" / "assets" / "tokenizers" / "tokenizer.llama2"
OUTPUT = DATA / "tokenizer-llama2-hf"
C4_TRAIN = DATA / "c4-en" / "train"
MIRROR = "NousResearch/Llama-2-7b-hf"
PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
]


def sample_c4(limit: int) -> list[str]:
    documents: list[str] = []
    for shard in sorted(C4_TRAIN.glob("*.json.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                documents.append(json.loads(line)["text"])
                if len(documents) >= limit:
                    return documents
    return documents


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-docs", type=int, default=5000)
    parser.add_argument("--mirror", default=MIRROR)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()

    import sentencepiece as spm
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    documents = sample_c4(args.verify_docs)
    if not SENTENCEPIECE.is_file():
        raise SystemExit(f"missing reference tokenizer: {SENTENCEPIECE}")
    if not documents:
        raise SystemExit(f"no C4 documents found under {C4_TRAIN}")

    reference = spm.SentencePieceProcessor(model_file=str(SENTENCEPIECE))
    cached = Path(snapshot_download(args.mirror, allow_patterns=PATTERNS))
    tokenizer = AutoTokenizer.from_pretrained(str(cached))
    mismatches = [
        index
        for index, text in enumerate(documents)
        if reference.EncodeAsIds(text)
        != tokenizer(text, add_special_tokens=False)["input_ids"]
    ]
    if mismatches or reference.GetPieceSize() != tokenizer.vocab_size:
        raise SystemExit("candidate tokenizer does not match the reference vocabulary")

    args.out.mkdir(parents=True, exist_ok=True)
    for path in cached.iterdir():
        if path.is_file():
            shutil.copyfile(path, args.out / path.name)
    provenance = {
        "mirror": args.mirror,
        "verified_against": str(SENTENCEPIECE),
        "reference_sha256": hashlib.sha256(SENTENCEPIECE.read_bytes()).hexdigest(),
        "verified_documents": len(documents),
        "vocab_size": tokenizer.vocab_size,
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.out.iterdir())
            if path.is_file() and path.name != "PROVENANCE.json"
        },
    }
    (args.out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
