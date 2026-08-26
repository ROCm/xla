#!/usr/bin/env python3
"""Extracts the decision points of an RCCL build, and of XLA's use of it.

The sizes a case is run at are only meaningful relative to the thresholds the
library actually branches on, and those move. Between the two RCCL checkouts
consulted while this suite was designed, the default of RCCL_WARP_SPEED_AUTO
flipped and the protocol range table differed per architecture. A matrix with
thresholds written into it keeps running after the library stops agreeing with
them, and reports the same green while covering less.

So the matrix is generated from this extractor rather than hand-written, and
the extractor's output is compared against the previous run. A change in the
output is itself a finding: it means the library reorganized its decisions and
the coverage needs revisiting.

Usage:
  extract_thresholds.py --rccl-src PATH [--xla-src PATH] [-o thresholds.json]
  extract_thresholds.py --rccl-src PATH --diff previous.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any


# --------------------------------------------------------------------------
# Generic C source helpers
# --------------------------------------------------------------------------

def strip_comments(text: str) -> str:
  """Removes C comments. The tuning tables carry their labels in comments."""
  text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
  text = re.sub(r"//[^\n]*", " ", text)
  return text


def parse_braced_ints(text: str, start: int) -> tuple[Any, int]:
  """Parses a brace-nested initializer of integers starting at text[start]=='{'.

  Returns the nested list and the index just past the closing brace. Values that
  are not plain integers (expressions, enum names) become None rather than
  aborting: the tables are read for the numbers, and a symbolic entry should be
  visible as a gap instead of silently dropping the whole table.
  """
  assert text[start] == "{"
  result: list[Any] = []
  i = start + 1
  token = ""

  def flush() -> None:
    nonlocal token
    literal = token.strip()
    token = ""
    if not literal:
      return
    try:
      result.append(int(literal, 0))
    except ValueError:
      result.append(None)

  while i < len(text):
    char = text[i]
    if char == "{":
      nested, i = parse_braced_ints(text, i)
      result.append(nested)
      continue
    if char == "}":
      flush()
      return result, i + 1
    if char == ",":
      flush()
      i += 1
      continue
    token += char
    i += 1
  raise ValueError("unterminated initializer")


def find_field(text: str, field: str, search_from: int = 0) -> Any:
  """Returns the initializer of `.field = { ... }`, or None if absent."""
  match = re.search(rf"\.{re.escape(field)}\s*=\s*\{{", text[search_from:])
  if match is None:
    return None
  brace = search_from + match.end() - 1
  value, _ = parse_braced_ints(text, brace)
  return value


# --------------------------------------------------------------------------
# RCCL
# --------------------------------------------------------------------------

PARAM_RE = re.compile(
    r"\b(?:NCCL|RCCL)_PARAM\(\s*(\w+)\s*,\s*\"([^\"]+)\"\s*,\s*([^)]*)\)"
)


def extract_params(rccl_src: pathlib.Path) -> dict[str, Any]:
  """Every tunable the build exposes, with the default it ships with.

  These are the knobs that decide which implementation runs. Recording all of
  them, not just the ones a case uses today, is what makes the diff useful: a
  new tunable appearing is the earliest signal that a new branch exists.
  """
  params: dict[str, Any] = {}
  for path in sorted(rccl_src.rglob("*.cc")) + sorted(rccl_src.rglob("*.h")):
    try:
      text = path.read_text(errors="ignore")
    except OSError:
      continue
    for symbol, env_name, default in PARAM_RE.findall(text):
      params.setdefault(
          env_name,
          {
              "symbol": symbol,
              "default": default.strip(),
              "defined_in": str(path.relative_to(rccl_src)),
          },
      )
  return params


# Order of the per-collective rows in the tuning tables, from the RccclTunableColls
# enum the tables are documented against.
LL_PROTO_COLLECTIVES = [
    "ReduceScatter", "AllGather", "AllReduce", "Reduce", "Broadcast",
]
CHANNEL_COLLECTIVES = ["ReduceScatter", "AllGather", "AllReduce"]


def extract_tuning_models(tuning_cc: pathlib.Path) -> dict[str, Any]:
  """Protocol ranges and channel-count buckets, per tuning model.

  llProtoRanges rows are [min, max, factor, thread_threshold] for LL and then
  LL128; anything above the LL128 maximum uses Simple. channelThresholds rows
  are [min, max, channels] against the per-rank transfer size.
  """
  raw = tuning_cc.read_text(errors="ignore")
  text = strip_comments(raw)
  models: dict[str, Any] = {}

  for match in re.finditer(r"struct\s+tuningModel\s+(tuning_model_\d+)\s*\{",
                           text):
    name = match.group(1)
    brace = match.end() - 1
    body, _ = None, None
    # Slice out just this model's initializer so field lookups cannot leak into
    # the next model.
    depth = 0
    end = brace
    for i in range(brace, len(text)):
      if text[i] == "{":
        depth += 1
      elif text[i] == "}":
        depth -= 1
        if depth == 0:
          end = i + 1
          break
    body = text[brace:end]

    ll_ranges = find_field(body, "llProtoRanges")
    channels = find_field(body, "channelThresholds")

    entry: dict[str, Any] = {}
    if ll_ranges:
      entry["ll_proto_ranges"] = {
          collective: {"LL": rows[0], "LL128": rows[1]}
          for collective, rows in zip(LL_PROTO_COLLECTIVES, ll_ranges)
          if isinstance(rows, list) and len(rows) >= 2
      }
    if channels:
      entry["channel_thresholds"] = {
          collective: rows
          for collective, rows in zip(CHANNEL_COLLECTIVES, channels)
      }
    if entry:
      models[name] = entry

  return models


def extract_arch_tuning_index(tuning_cc: pathlib.Path) -> dict[str, int]:
  """Which tuning model each architecture selects."""
  text = tuning_cc.read_text(errors="ignore")
  match = re.search(r"rcclGetTuningIndexForArch\s*\([^)]*\)\s*\{(.*?)\n\}", text,
                    flags=re.S)
  if match is None:
    return {}
  return {
      arch: int(index)
      for arch, index in re.findall(r"\{\s*\"(gfx\w+)\"\s*,\s*(\d+)\s*\}",
                                    match.group(1))
  }


# Per-collective activation thresholds. Present as tunables in some builds and
# as a single hard-coded constant in others, which is precisely why they are
# read out of the source instead of written into the cases.
WARP_SPEED_THRESHOLD_ENV = [
    "WARP_SPEED_AG_THRESHOLD",
    "WARP_SPEED_AR_THRESHOLD",
    "WARP_SPEED_RS_THRESHOLD",
    "WARP_SPEED_AUTO",
    "WARP_SPEED_ENABLE",
    "WARP_SPEED_FORCE_ENABLE",
    "WARP_SPEED_CU_COUNT",
]


def extract_warp_speed(rccl_src: pathlib.Path) -> dict[str, Any]:
  """Activation conditions for WarpSpeed.

  Reported as source facts rather than as a decision: whether the feature is
  reachable at all depends on ENABLE_WARP_SPEED at build time, which the source
  defaults to OFF, so a library can contain none of this.
  """
  wrap = rccl_src / "src" / "rccl_wrap.cc"
  info: dict[str, Any] = {"source_present": wrap.exists()}
  if not wrap.exists():
    return info

  text = wrap.read_text(errors="ignore")
  min_bytes = re.search(
      r"#define\s+RCCL_WARP_SPEED_MIN_BYTES\s+\(([^)]*)\)", text)
  if min_bytes:
    expression = min_bytes.group(1).strip()
    info["min_bytes_expression"] = expression
    shift = re.match(r"1ULL\s*<<\s*(\d+)", expression)
    if shift:
      info["min_bytes"] = 1 << int(shift.group(1))

  # Scope the condition scrape to the auto-mode function. The file contains
  # other architecture checks, and attributing them to this feature would
  # overstate where it can activate. The return type is deliberately not
  # pinned: it differs between checkouts, and a regex that assumed one silently
  # reported "no conditions found" against the other.
  auto_mode = re.search(
      r"\w[\w:<>*&\s]*\brcclSetWarpSpeedAuto\s*\([^)]*\)\s*\{(.*?)\n\}", text,
      flags=re.S)
  if auto_mode is not None:
    body = auto_mode.group(1)
    info["auto_mode_only_on"] = sorted(set(re.findall(
        r"IsArchMatch\(\s*comm->archName\s*,\s*\"(\w+)\"\s*\)", body)))
    info["requires_single_node"] = "comm->nNodes == 1" in body
    info["requires_ring"] = "NCCL_ALGO_RING" in body

  cmake = rccl_src / "CMakeLists.txt"
  if cmake.exists():
    option = re.search(r"option\(ENABLE_WARP_SPEED\s+\"[^\"]*\"\s+(\w+)\)",
                       cmake.read_text(errors="ignore"))
    if option:
      info["cmake_option_default"] = option.group(1)
  return info


DISPATCH_ENV = [
    "MSCCLPP_ENABLE",
    "MSCCLPP_THRESHOLD",
    "ROCSHMEM_ENABLE",
    "ROCSHMEM_THRESHOLD",
    "PIVOT_ALLTOALL_ENABLE",
    "INTRANET_THRESHOLD",
    "P2P_LL_THRESHOLD",
    "P2P_NET_THRESHOLD",
]


def extract_dispatch(params: dict[str, Any]) -> dict[str, Any]:
  """The tunables that select between whole alternative implementations.

  Separated from the rest because crossing one of these does not change how a
  collective is executed, it changes which code executes it.
  """
  return {name: params[name] for name in DISPATCH_ENV if name in params}


# --------------------------------------------------------------------------
# XLA
# --------------------------------------------------------------------------

def extract_xla_combiner_defaults(xla_src: pathlib.Path) -> dict[str, Any]:
  """Combiner limits, which decide how many operations share one RCCL group.

  The byte threshold also decides which RCCL branches XLA can reach at all: a
  combined collective that stays under it cannot produce a transfer large
  enough to cross the library's larger feature thresholds.
  """
  header = xla_src / "xla" / "service" / "collective_utils.h"
  if not header.exists():
    return {}
  text = strip_comments(header.read_text(errors="ignore"))
  found: dict[str, Any] = {}
  for name, expression in re.findall(
      r"constexpr\s+int64_t\s+(kDefault\w*Combine\w*)\s*=\s*([^;]+);", text):
    expression = " ".join(expression.split())
    entry: dict[str, Any] = {"expression": expression}
    try:
      # The expressions in this header are plain arithmetic on literals.
      entry["value"] = int(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:  # pylint: disable=broad-except
      pass
    found[name] = entry
  return found


def extract_xla_rccl_symbols(xla_src: pathlib.Path) -> list[str]:
  """Which RCCL entry points XLA actually calls.

  This is the list the suite claims to cover. When it grows, something in XLA
  started using a part of the library nothing here exercises yet.
  """
  directory = xla_src / "xla" / "backends" / "gpu" / "collectives"
  if not directory.exists():
    return []
  symbols: set[str] = set()
  for path in sorted(directory.glob("rccl_*.cc")) + sorted(
      directory.glob("rccl_*.h")):
    text = strip_comments(path.read_text(errors="ignore"))
    symbols.update(re.findall(r"\bnccl[A-Z]\w*", text))
  return sorted(symbols)


# --------------------------------------------------------------------------
# Assembly and diffing
# --------------------------------------------------------------------------

def build_report(rccl_src: pathlib.Path,
                 xla_src: pathlib.Path | None) -> dict[str, Any]:
  tuning_cc = rccl_src / "src" / "graph" / "tuning.cc"
  params = extract_params(rccl_src)

  warp_speed = extract_warp_speed(rccl_src)
  warp_speed["thresholds"] = {
      name: params.get(name, "<absent in this build>")
      for name in WARP_SPEED_THRESHOLD_ENV
  }

  report: dict[str, Any] = {
      "rccl": {
          "source": str(rccl_src),
          "parameter_count": len(params),
          "parameters": params,
          "dispatch": extract_dispatch(params),
          "warp_speed": warp_speed,
      }
  }
  if tuning_cc.exists():
    report["rccl"]["tuning_models"] = extract_tuning_models(tuning_cc)
    report["rccl"]["arch_tuning_index"] = extract_arch_tuning_index(tuning_cc)

  if xla_src is not None:
    report["xla"] = {
        "source": str(xla_src),
        "combiner_defaults": extract_xla_combiner_defaults(xla_src),
        "rccl_symbols_used": extract_xla_rccl_symbols(xla_src),
    }
  return report


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
  if isinstance(value, dict):
    out: dict[str, Any] = {}
    for key, item in value.items():
      out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    return out
  return {prefix: value if not isinstance(value, list) else json.dumps(value)}


def diff_reports(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
  """Reports what moved. Paths that only differ in the source directory are
  ignored, since those change with the checkout rather than with the library."""
  ignored = {"rccl.source", "xla.source"}
  before = flatten(previous)
  after = flatten(current)
  lines: list[str] = []
  for key in sorted(set(before) | set(after)):
    if key in ignored:
      continue
    old = before.get(key, "<absent>")
    new = after.get(key, "<absent>")
    if old != new:
      lines.append(f"{key}: {old} -> {new}")
  return lines


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--rccl-src", required=True, type=pathlib.Path,
                      help="RCCL source root (the directory holding src/)")
  parser.add_argument("--xla-src", type=pathlib.Path, default=None,
                      help="XLA source root")
  parser.add_argument("-o", "--output", type=pathlib.Path, default=None)
  parser.add_argument("--diff", type=pathlib.Path, default=None,
                      help="compare against a previously written report")
  args = parser.parse_args()

  rccl_src = args.rccl_src
  if not (rccl_src / "src").is_dir() and (rccl_src / "projects" / "rccl" /
                                          "src").is_dir():
    # Accept a rocm-systems checkout as well as a bare RCCL one.
    rccl_src = rccl_src / "projects" / "rccl"

  report = build_report(rccl_src, args.xla_src)
  text = json.dumps(report, indent=2, sort_keys=True)

  if args.output is not None:
    args.output.write_text(text + "\n")
    print(f"wrote {args.output}")
  elif args.diff is None:
    print(text)

  if args.diff is not None:
    previous = json.loads(args.diff.read_text())
    changes = diff_reports(previous, report)
    if not changes:
      print("no change against " + str(args.diff))
      return 0
    print(f"{len(changes)} change(s) against {args.diff}:")
    for line in changes:
      print("  " + line)
    # A non-zero exit so CI surfaces the change. Thresholds moving is not an
    # error, but continuing to run the old matrix against them would be.
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
