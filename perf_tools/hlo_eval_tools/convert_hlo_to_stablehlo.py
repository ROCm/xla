#!/usr/bin/env python3
"""Convert the collected `before_optimizations` HLO text modules to StableHLO
portable artifacts (versioned VHLO bytecode, `.mlirbc`).

HLO text has no cross-version compatibility guarantee, so a corpus dumped by one
JAX/XLA release may not parse on an older or newer `multihost_hlo_runner`.
StableHLO portable artifacts carry 5 years of backward and 2 years of forward
compatibility, but only when serialized as bytecode for a target version that is
<= the StableHLO version of the oldest XLA that has to read them. Textual
StableHLO MLIR carries no such guarantee, which is why this script always
serializes.

This only works for *pre-optimization* HLO: optimized HLO contains fusions,
backend custom-calls and layouts that have no StableHLO equivalent
(openxla/xla#36492).

For every `<stem>.txt` / `<stem>.hlo` module this writes, next to it:
  <stem>.mlirbc        the portable artifact, readable by
                       `hlo_runner_main --input_format=stablehlo`;
  <stem>.mlirbc.meta   `key=value` sidecar: num_partitions (the bytecode cannot
                       be grepped for it, run_hlo_eval.sh reads it from here),
                       target version, strip_debuginfo and the source module's
                       sha256.

Pipeline per module:
  1. xla-translate --hlo-text-to-mlir-hlo         HLO text -> MHLO
  2. mlir-hlo-opt --hlo-legalize-to-stablehlo     MHLO -> StableHLO; ops StableHLO
     lacks (e.g. topk) become `stablehlo.custom_call @mhlo.<op>`, which XLA's
     StableHLO -> HLO legalization turns back into the original op
  3. stablehlo-translate --serialize --target=T   -> VHLO bytecode at version T
  4. stablehlo-translate --deserialize            check the artifact reads back
  5. unless --no-verify-roundtrip, lower it back to HLO with
     mlir-hlo-opt --stablehlo-legalize-to-hlo + xla-translate --mlir-hlo-to-hlo-text
     (the lowering of `xla-translate --stablehlo-to-hlo`, as text) and compare
     the distinct computation bodies (per-opcode counts of live instructions,
     minus tuple plumbing) with the original HLO.

Build the tools at the XLA revision that produced the HLO, so its own parser
and importer read the text (for the current corpus rocm-jaxlib-v0.10.2,
7b5ecf1c; see README "Provenance"):
  bazel build -c opt //xla/hlo/translate:xla-translate //xla/mlir_hlo:mlir-hlo-opt \
      @stablehlo//:stablehlo-translate
The serialized artifacts' compatibility depends only on --target, so any
stablehlo-translate whose own version is >= the target works.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import collections
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# StableHLO version pinned by the oldest XLA the artifacts must load on
# (rocm-jaxlib-v0.8.0 -> third_party/stablehlo commit 0a4440a5 -> 1.13.1).
# Lower this if an even older runner has to read the artifacts.
DEFAULT_TARGET = "1.13.1"

HLO_SUFFIXES = (".txt", ".hlo")
NUM_PARTITIONS_RE = re.compile(r"num_partitions=(\d+)")
# `[ROOT ]%name = <shape> opcode(` - the first lowercase `word(` after ` = `.
# Shapes never contain one (tuple shapes open with a bare `(`, layout tiles are
# `T(`).
INSTRUCTION_RE = re.compile(
    r"^\s*(ROOT\s+)?(%?[\w.\-]+)\s*=\s*.*?(?<![\w.\-])([a-z][a-z0-9\-]*)\("
)
# Operand names, with or without the `%` (short print format). Other tokens
# (shapes, literals) are dropped by the lookup in computation_signatures.
OPERAND_RE = re.compile(r"%?([\w.\-]+)")
COMPUTATION_RE = re.compile(r"^(?:ENTRY\s+)?%?([\w.\-]+)\s.*\{\s*$")
# Plumbing that legitimately changes in the round trip: the importer flattens
# tuple parameters/results and the exporter re-tuples the entry.
ROUNDTRIP_IGNORED_OPCODES = {"tuple", "get-tuple-element", "parameter"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="HLO module files and/or directories to search recursively.",
    )
    parser.add_argument("--xla-translate", required=True, type=Path)
    parser.add_argument("--mlir-hlo-opt", required=True, type=Path)
    parser.add_argument("--stablehlo-translate", required=True, type=Path)
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"StableHLO version to serialize for (default {DEFAULT_TARGET}).",
    )
    parser.add_argument(
        "--strip-debuginfo",
        action="store_true",
        help="Drop op_name/source locations (smaller artifacts).",
    )
    parser.add_argument(
        "--no-verify-roundtrip",
        dest="verify_roundtrip",
        action="store_false",
        help="Skip the StableHLO -> HLO opcode-count comparison.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing .mlirbc files."
    )
    parser.add_argument("-j", "--jobs", type=int, default=8)
    return parser.parse_args()


def find_modules(paths: list[Path]) -> list[Path]:
    modules = set()
    for path in paths:
        if path.is_file():
            modules.add(path.resolve())
        elif path.is_dir():
            modules.update(
                p.resolve()
                for p in path.rglob("*")
                if p.is_file() and p.suffix in HLO_SUFFIXES
            )
        else:
            sys.exit(f"error: not found: {path}")
    # <stem>.txt and <stem>.hlo would both write <stem>.mlirbc.
    by_stem: dict[Path, list[Path]] = collections.defaultdict(list)
    for module in modules:
        by_stem[module.with_suffix("")].append(module)
    clashes = sorted(
        sorted(group) for group in by_stem.values() if len(group) > 1
    )
    if clashes:
        sys.exit(
            "error: modules that would write the same .mlirbc:\n"
            + "\n".join("  " + ", ".join(map(str, group)) for group in clashes)
        )
    return sorted(modules)


def computation_signatures(hlo_text: str) -> set[tuple[tuple[str, int], ...]]:
    """Distinct per-computation counts of live opcodes, minus tuple plumbing.

    Only instructions reachable from the ROOT (or with side effects) count:
    the round trip duplicates dead instructions (e.g. the unused output
    `reshape ..., sharding=` JAX emits), which DCE drops anyway. A set rather
    than a list because the round trip gives every reduce/while its own copy
    of shared regions and keeps dead copies of the originals.
    """
    signatures = set()
    # name -> (opcode, operand names); None outside a computation.
    instructions: dict[str, tuple[str, list[str]]] | None = None
    roots: list[str] = []
    for line in hlo_text.splitlines():
        if instructions is None:
            if COMPUTATION_RE.match(line):
                instructions, roots = {}, []
        elif line.startswith("}"):
            live, stack = set(), list(roots)
            while stack:
                name = stack.pop()
                if name in instructions and name not in live:
                    live.add(name)
                    stack.extend(instructions[name][1])
            counts = collections.Counter(
                instructions[name][0] for name in live
            )
            for opcode in ROUNDTRIP_IGNORED_OPCODES:
                counts.pop(opcode, None)
            signatures.add(tuple(sorted(counts.items())))
            instructions = None
        else:
            match = INSTRUCTION_RE.match(line)
            if not match:
                continue
            is_root, name, opcode = match.groups()
            name = name.lstrip("%")
            # Operands run up to the paren that closes `opcode(`.
            depth, end = 1, match.end()
            while depth and end < len(line):
                depth += {"(": 1, ")": -1}.get(line[end], 0)
                end += 1
            operands = OPERAND_RE.findall(line[match.end():end])
            instructions[name] = (opcode, operands)
            if is_root or "custom_call_has_side_effect=true" in line:
                roots.append(name)
    return signatures


def format_signature(signature: tuple[tuple[str, int], ...]) -> str:
    return "{" + ", ".join(f"{op}:{n}" for op, n in signature) + "}"


def run(command: list[str], stdin: Path | None = None) -> None:
    """Run a tool, raising with the tail of stderr on failure."""
    with open(stdin or os.devnull, "rb") as input_file:
        proc = subprocess.run(
            command, stdin=input_file, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace").strip().splitlines()[-15:]
        raise RuntimeError(
            f"{Path(command[0]).name} exited {proc.returncode}:\n    "
            + "\n    ".join(tail)
        )


def convert(module: Path, args: argparse.Namespace) -> tuple[str, str]:
    """Convert one module. Returns (status, detail)."""
    stem = module.with_suffix("")
    artifact = stem.with_name(stem.name + ".mlirbc")
    meta = artifact.with_name(artifact.name + ".meta")
    hlo_bytes = module.read_bytes()
    source_sha256 = hashlib.sha256(hlo_bytes).hexdigest()
    strip_debuginfo = str(args.strip_debuginfo).lower()
    if artifact.exists() and meta.exists() and not args.force:
        recorded = dict(
            line.split("=", 1)
            for line in meta.read_text().splitlines()
            if "=" in line
        )
        # Redo it if the module or any serialization option changed.
        if (recorded.get("source_sha256") == source_sha256
                and recorded.get("stablehlo_target") == args.target
                and recorded.get("strip_debuginfo") == strip_debuginfo):
            return "SKIP", "already converted (use --force to redo)"

    hlo_text = hlo_bytes.decode(errors="replace")
    header = hlo_text.split("\n", 1)[0]
    match = NUM_PARTITIONS_RE.search(header)
    num_partitions = int(match.group(1)) if match else 1

    with tempfile.TemporaryDirectory(prefix="hlo2shlo.") as tmp:
        tmp = Path(tmp)
        mhlo_mlir = tmp / "mhlo.mlir"
        stablehlo_mlir = tmp / "stablehlo.mlir"
        tmp_artifact = tmp / "module.mlirbc"
        readback_mlir = tmp / "readback.mlir"
        roundtrip_mhlo = tmp / "roundtrip.mhlo.mlir"
        roundtrip_hlo = tmp / "roundtrip.hlo"
        try:
            # ConvertHloToStablehlo's import options, but via MHLO: a direct
            # StableHLO import keeps ops like topk as MHLO, which VHLO cannot
            # serialize. The intermediates are printed with their locations
            # (the HLO op_name/source metadata); otherwise the next tool would
            # locate every op in the temp file instead.
            run([
                str(args.xla_translate), "--hlo-text-to-mlir-hlo",
                "--hlo-import-all-computations",
                "--hlo-flatten-computation-args-result",
                "--mlir-print-debuginfo",
                str(module), "-o", str(mhlo_mlir),
            ])
            run([
                str(args.mlir_hlo_opt), "--hlo-legalize-to-stablehlo",
                "--mlir-print-debuginfo",
                str(mhlo_mlir), "-o", str(stablehlo_mlir),
            ])
            # From stdin: the artifact records its input's file name, and the
            # temp dir's would make every conversion's bytes differ.
            serialize = [
                str(args.stablehlo_translate), "--serialize",
                f"--target={args.target}", "-o", str(tmp_artifact),
            ]
            if args.strip_debuginfo:
                serialize.insert(2, "--strip-debuginfo")
            run(serialize, stdin=stablehlo_mlir)
            run([
                str(args.stablehlo_translate), "--deserialize",
                str(tmp_artifact), "-o", str(readback_mlir),
            ])
            detail = ""
            if args.verify_roundtrip:
                run([
                    str(args.mlir_hlo_opt), "--stablehlo-legalize-to-hlo",
                    str(readback_mlir), "-o", str(roundtrip_mhlo),
                ])
                run([
                    str(args.xla_translate), "--mlir-hlo-to-hlo-text",
                    str(roundtrip_mhlo), "-o", str(roundtrip_hlo),
                ])
                before = computation_signatures(hlo_text)
                after = computation_signatures(
                    roundtrip_hlo.read_text(errors="replace")
                )
                diffs = [
                    f"only in {side}: {format_signature(signature)}"
                    for side, signatures in (
                        ("original", before - after),
                        ("round trip", after - before),
                    )
                    for signature in sorted(signatures)
                ]
                detail = "\n       ".join(diffs)
        except RuntimeError as e:
            return "FAIL", str(e)

        # Write both files next to their targets and rename them into place,
        # so neither is ever seen half-written. The old .meta goes first and
        # the new one comes last, so a .meta never describes another artifact;
        # a run that dies in between leaves none, and the next run redoes it.
        meta.unlink(missing_ok=True)
        partial = artifact.with_name(artifact.name + ".partial")
        shutil.copyfile(tmp_artifact, partial)
        os.replace(partial, artifact)
    partial = meta.with_name(meta.name + ".partial")
    partial.write_text(
        f"num_partitions={num_partitions}\n"
        f"stablehlo_target={args.target}\n"
        f"strip_debuginfo={strip_debuginfo}\n"
        f"source={module.name}\n"
        f"source_sha256={source_sha256}\n"
    )
    os.replace(partial, meta)
    size_mb = artifact.stat().st_size / 1e6
    return ("DIFF" if detail else "OK"), detail or f"{size_mb:.1f} MB"


def main() -> int:
    args = parse_args()
    for tool in (args.xla_translate, args.mlir_hlo_opt,
                 args.stablehlo_translate):
        if not os.access(tool, os.X_OK):
            sys.exit(f"error: not executable: {tool}")
    modules = find_modules(args.paths)
    if not modules:
        sys.exit("error: no .txt/.hlo modules found")
    print(f"converting {len(modules)} module(s) to StableHLO v{args.target} "
          f"with {args.jobs} job(s)")

    statuses: collections.Counter[str] = collections.Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(convert, m, args): m for m in modules}
        for future in concurrent.futures.as_completed(futures):
            module = futures[future]
            # File-system errors fail only their module.
            try:
                status, detail = future.result()
            except OSError as e:
                status, detail = "FAIL", str(e)
            statuses[status] += 1
            print(f"[{status:4}] {module}" + (f"\n       {detail}" if detail else ""),
                  flush=True)

    print("\n==== summary ====")
    for status in ("OK", "DIFF", "SKIP", "FAIL"):
        if statuses[status]:
            print(f"{status:4}: {statuses[status]}")
    if statuses["DIFF"]:
        print("DIFF modules converted, but some computations differ after the "
              "round trip; inspect before trusting A/B numbers against .txt.")
    return 1 if statuses["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
