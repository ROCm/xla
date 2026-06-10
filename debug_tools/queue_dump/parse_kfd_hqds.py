#!/usr/bin/env python3
"""
Parser for /sys/kernel/debug/kfd/hqds
Source: kfd_device_queue_manager.c :: dqm_debugfs_hqds()
        amdgpu_amdkfd_gfx_v9.c    :: kgd_gfx_v9_hqd_dump()

The hqds file contains LIVE hardware register values (read via RREG32 after
acquiring the HQD slot), unlike /mqds which holds the saved MQD.

Register set: 56 consecutive regs from mmCP_MQD_BASE_ADDR to
mmCP_HQD_PQ_WPTR_HI, corresponding to struct v9_mqd dwords 128-183.

Output format from dqm_debugfs_hqds + seq_reg_dump:

  Device <header>
   Inst <xcc>, HIQ on MEC <mec> Pipe <pipe> Queue <queue>
   <byte_addr>: <val> [<val> ...]     <- up to 8 values per line
   Inst <xcc>, CP Pipe <pipe>, Queue <queue>
   <byte_addr>: ...
   SDMA Engine <e>, RLC <q>           <- skipped (user request)
   ...

Usage:
  sudo cat /sys/kernel/debug/kfd/hqds > hqds.txt
  python3 parse_kfd_hqds.py hqds.txt         # reads from a previously captured file
"""

import sys
import re

DEBUGFS_HQDS = "/sys/kernel/debug/kfd/hqds"

# ---------------------------------------------------------------------------
# The 56 registers dumped by kgd_gfx_v9_hqd_dump, in order.
# Corresponds exactly to struct v9_mqd dwords 128-183.
# Source: for (reg = mmCP_MQD_BASE_ADDR; reg <= mmCP_HQD_PQ_WPTR_HI; reg++)
# ---------------------------------------------------------------------------
HQD_REGS = [
    # offset  register name (gc_9_4_3_offset.h)
    "cp_mqd_base_addr_lo",           # 0  0x1245
    "cp_mqd_base_addr_hi",           # 1  0x1246
    "cp_hqd_active",                 # 2  0x1247
    "cp_hqd_vmid",                   # 3  0x1248
    "cp_hqd_persistent_state",       # 4  0x1249
    "cp_hqd_pipe_priority",          # 5  0x124a
    "cp_hqd_queue_priority",         # 6  0x124b
    "cp_hqd_quantum",                # 7  0x124c
    "cp_hqd_pq_base_lo",             # 8  0x124d
    "cp_hqd_pq_base_hi",             # 9  0x124e
    "cp_hqd_pq_rptr",                # 10 0x124f
    "cp_hqd_pq_rptr_report_addr_lo", # 11 0x1250
    "cp_hqd_pq_rptr_report_addr_hi", # 12 0x1251
    "cp_hqd_pq_wptr_poll_addr_lo",   # 13 0x1252
    "cp_hqd_pq_wptr_poll_addr_hi",   # 14 0x1253
    "cp_hqd_pq_doorbell_control",    # 15 0x1254
    "unknown_0x1255",                # 16 0x1255  (gap - no entry in gc_9_4_3_offset.h)
    "cp_hqd_pq_control",             # 17 0x1256
    "cp_hqd_ib_base_addr_lo",        # 18 0x1257
    "cp_hqd_ib_base_addr_hi",        # 19 0x1258
    "cp_hqd_ib_rptr",                # 20 0x1259
    "cp_hqd_ib_control",             # 21 0x125a
    "cp_hqd_iq_timer",               # 22 0x125b
    "cp_hqd_iq_rptr",                # 23 0x125c
    "cp_hqd_dequeue_request",        # 24 0x125d
    "cp_hqd_dma_offload",            # 25 0x125e  (alias: cp_hqd_offload)
    "cp_hqd_sema_cmd",               # 26 0x125f
    "cp_hqd_msg_type",               # 27 0x1260
    "cp_hqd_atomic0_preop_lo",       # 28 0x1261
    "cp_hqd_atomic0_preop_hi",       # 29 0x1262
    "cp_hqd_atomic1_preop_lo",       # 30 0x1263
    "cp_hqd_atomic1_preop_hi",       # 31 0x1264
    "cp_hqd_hq_scheduler0",          # 32 0x1265  (alias: cp_hqd_hq_status0)
    "cp_hqd_hq_control0",            # 33 0x1266  (alias: cp_hqd_hq_scheduler1)
    "cp_mqd_control",                # 34 0x1267
    "cp_hqd_hq_status1",             # 35 0x1268
    "cp_hqd_hq_control1",            # 36 0x1269
    "cp_hqd_eop_base_addr_lo",       # 37 0x126a
    "cp_hqd_eop_base_addr_hi",       # 38 0x126b
    "cp_hqd_eop_control",            # 39 0x126c
    "cp_hqd_eop_rptr",               # 40 0x126d
    "cp_hqd_eop_wptr",               # 41 0x126e
    "cp_hqd_eop_events",             # 42 0x126f
    "cp_hqd_ctx_save_base_addr_lo",  # 43 0x1270
    "cp_hqd_ctx_save_base_addr_hi",  # 44 0x1271
    "cp_hqd_ctx_save_control",       # 45 0x1272
    "cp_hqd_cntl_stack_offset",      # 46 0x1273
    "cp_hqd_cntl_stack_size",        # 47 0x1274
    "cp_hqd_wg_state_offset",        # 48 0x1275
    "cp_hqd_ctx_save_size",          # 49 0x1276
    "cp_hqd_gds_resource_state",     # 50 0x1277
    "cp_hqd_error",                  # 51 0x1278
    "cp_hqd_eop_wptr_mem",           # 52 0x1279
    "cp_hqd_aql_control",            # 53 0x127a
    "cp_hqd_pq_wptr_lo",             # 54 0x127b
    "cp_hqd_pq_wptr_hi",             # 55 0x127c
]

assert len(HQD_REGS) == 56

def _reg(regs, name):
    try:
        return regs[HQD_REGS.index(name)]
    except (ValueError, IndexError):
        return 0

# ---------------------------------------------------------------------------
# Decoders (same logic as parse_kfd_mqds.py)
# ---------------------------------------------------------------------------
def u64(lo, hi):
    return (hi << 32) | lo

def pq_base_addr(lo, hi):
    return ((hi << 32) | lo) << 8

def eop_base_addr(lo, hi):
    return ((hi << 32) | lo) << 8

def ctx_save_addr(lo, hi):
    return u64(lo, hi)

def pq_queue_size_bytes(pq_ctrl):
    field = pq_ctrl & 0x3F
    return 4 * (1 << (field + 1))

def doorbell_offset(doorbell_ctrl):
    return (doorbell_ctrl >> 21) & 0x3FF

def doorbell_enabled(doorbell_ctrl):
    return (doorbell_ctrl >> 30) & 1

def decode_pq_control(val):
    queue_size_enc = val & 0x3F
    unord = (val >> 14) & 1
    no_upd_rptr = (val >> 27) & 1
    aql_wptr = (val >> 28) & 3
    priv = (val >> 23) & 1
    kmd = (val >> 24) & 1
    return (f"size_enc={queue_size_enc} unord={unord} "
            f"no_upd_rptr={no_upd_rptr} aql_wptr_mode={aql_wptr} "
            f"priv={priv} kmd={kmd}")

def decode_persistent_state(val):
    preload_req = val & 1
    preload_size = (val >> 8) & 0xFF
    qswitch_mode = (val >> 14) & 1
    return (f"preload_req={preload_req} preload_size=0x{preload_size:x} "
            f"qswitch_mode={qswitch_mode}")

def decode_hqd_error(val):
    if val == 0:
        return "none"
    bits = []
    names = {0: "SUA_ERROR", 1: "PRIV_VIOLATION", 2: "DMA_WRITE",
             3: "DMA_READ", 4: "FATAL"}
    for b, name in names.items():
        if val & (1 << b):
            bits.append(name)
    return f"0x{val:08x} ({', '.join(bits) or 'unknown'})"

def decode_hq_status0(val):
    # Bit 14: CP setup DISPATCH_PTR (set by init_mqd via cp_hqd_hq_status0 |= 1<<14)
    dispatch_ptr = (val >> 14) & 1
    return f"0x{val:08x} (dispatch_ptr_setup={dispatch_ptr})"

# ---------------------------------------------------------------------------
# Main HQD entry decoder
# ---------------------------------------------------------------------------
def decode_hqd(regs, xcc_id, pipe, queue, queue_type, device_label):
    def r(name):
        return _reg(regs, name)

    # Helpers that print using the exact register name.
    # p()   - single 32-bit register
    # p64() - lo/hi pair; prints each separately, combined value as comment on hi line
    #         shift > 0 means the hardware shifts the combined value left (e.g. <<8 for addresses)
    W = 44  # label column width
    def p(name, comment=""):
        v = r(name)
        c = f"  # {comment}" if comment else ""
        print(f"    {name:<{W}} 0x{v:08x}{c}")

    def p64(lo_name, hi_name, shift=0):
        lo = r(lo_name)
        hi = r(hi_name)
        combined = ((hi << 32) | lo) << shift
        shift_str = f" <<{shift}" if shift else ""
        print(f"    {lo_name:<{W}} 0x{lo:08x}")
        print(f"    {hi_name:<{W}} 0x{hi:08x}  # => 0x{combined:016x}{shift_str}")

    print(f"\n[LIVE] xcc={xcc_id}  pipe={pipe}  queue={queue}  "
          f"type={queue_type}  device={device_label}")
    print()

    # Print all 56 HQD registers in order (0x1245 .. 0x127c)
    p64("cp_mqd_base_addr_lo",           "cp_mqd_base_addr_hi")
    p("cp_hqd_active")
    p("cp_hqd_vmid")
    p("cp_hqd_persistent_state")
    p("cp_hqd_pipe_priority")
    p("cp_hqd_queue_priority")
    p("cp_hqd_quantum")
    p64("cp_hqd_pq_base_lo",             "cp_hqd_pq_base_hi",             shift=8)
    p("cp_hqd_pq_rptr")
    p64("cp_hqd_pq_rptr_report_addr_lo", "cp_hqd_pq_rptr_report_addr_hi")
    p64("cp_hqd_pq_wptr_poll_addr_lo",   "cp_hqd_pq_wptr_poll_addr_hi")
    p("cp_hqd_pq_doorbell_control")
    p("unknown_0x1255")                  # 0x1255: gap in gc_9_4_3_offset.h
    p("cp_hqd_pq_control")
    p64("cp_hqd_ib_base_addr_lo",        "cp_hqd_ib_base_addr_hi")
    p("cp_hqd_ib_rptr")
    p("cp_hqd_ib_control")
    p("cp_hqd_iq_timer")
    p("cp_hqd_iq_rptr")
    p("cp_hqd_dequeue_request")
    p("cp_hqd_dma_offload",              "alias: cp_hqd_offload")
    p("cp_hqd_sema_cmd")
    p("cp_hqd_msg_type")
    p64("cp_hqd_atomic0_preop_lo",       "cp_hqd_atomic0_preop_hi")
    p64("cp_hqd_atomic1_preop_lo",       "cp_hqd_atomic1_preop_hi")
    p("cp_hqd_hq_scheduler0",            "alias: cp_hqd_hq_status0")
    p("cp_hqd_hq_control0",              "alias: cp_hqd_hq_scheduler1")
    p("cp_mqd_control")
    p("cp_hqd_hq_status1")
    p("cp_hqd_hq_control1")
    p64("cp_hqd_eop_base_addr_lo",       "cp_hqd_eop_base_addr_hi",       shift=8)
    p("cp_hqd_eop_control")
    p("cp_hqd_eop_rptr")
    p("cp_hqd_eop_wptr")
    p("cp_hqd_eop_events")
    p64("cp_hqd_ctx_save_base_addr_lo",  "cp_hqd_ctx_save_base_addr_hi")
    p("cp_hqd_ctx_save_control")
    p("cp_hqd_cntl_stack_offset")
    p("cp_hqd_cntl_stack_size")
    p("cp_hqd_wg_state_offset")
    p("cp_hqd_ctx_save_size")
    p("cp_hqd_gds_resource_state")
    p("cp_hqd_error")
    p("cp_hqd_eop_wptr_mem")
    p("cp_hqd_aql_control")
    p64("cp_hqd_pq_wptr_lo",             "cp_hqd_pq_wptr_hi")


# ---------------------------------------------------------------------------
# Line parsers
# ---------------------------------------------------------------------------
# Header patterns from dqm_debugfs_hqds
_HIQ_RE  = re.compile(r'^\s+Inst\s+(\d+),\s+HIQ on MEC\s+(\d+)\s+Pipe\s+(\d+)\s+Queue\s+(\d+)')
_CP_RE   = re.compile(r'^\s+Inst\s+(\d+),\s+CP Pipe\s+(\d+),\s+Queue\s+(\d+)')
_SDMA_RE = re.compile(r'^\s+SDMA Engine\s+(\d+),\s+RLC\s+(\d+)')

# kfd_debugfs_hqds_by_device outer device header (various forms encountered):
_DEV_RE  = re.compile(r'^[A-Za-z].*:')

# seq_reg_dump line: " <addr8>: <val8> [<val8> ...]"
_REG_RE  = re.compile(r'^\s+([0-9a-fA-F]{1,8}):\s+((?:[0-9a-fA-F]{8}\s*)+)')


def parse_reg_dump_line(line):
    """Return (byte_addr, [val, ...]) or None."""
    m = _REG_RE.match(line)
    if not m:
        return None
    addr = int(m.group(1), 16)
    vals = [int(v, 16) for v in m.group(2).split()]
    return addr, vals


def parse_hqds_file(lines):
    current_device = None
    current_type   = None   # "hiq" | "cp" | "sdma" | None
    current_xcc    = None
    current_pipe   = None
    current_queue  = None
    skip_section   = False

    # base byte-address of the first register in this section's dump
    base_addr      = None
    reg_values     = {}     # relative_reg_index -> value

    def flush_section():
        if current_type in ("hiq", "cp") and reg_values and base_addr is not None:
            regs = [reg_values.get(i, 0) for i in range(56)]
            decode_hqd(regs, current_xcc, current_pipe, current_queue,
                       current_type.upper(), current_device or "")

    for line in lines:
        line = line.rstrip('\n')

        # Device header
        m = _DEV_RE.match(line)
        if m and not _HIQ_RE.match(line) and not _CP_RE.match(line) and not _SDMA_RE.match(line):
            flush_section()
            current_type = None; base_addr = None; reg_values = {}; skip_section = False
            current_device = line.strip()
            print(f"\n{'='*70}")
            print(f"Device: {current_device}")
            print(f"{'='*70}")
            continue

        # HIQ header
        m = _HIQ_RE.match(line)
        if m:
            flush_section()
            current_xcc   = int(m.group(1))
            current_pipe  = int(m.group(3))
            current_queue = int(m.group(4))
            current_type  = "hiq"
            skip_section  = False
            base_addr     = None
            reg_values    = {}
            continue

        # CP compute queue header
        m = _CP_RE.match(line)
        if m:
            flush_section()
            current_xcc   = int(m.group(1))
            current_pipe  = int(m.group(2))
            current_queue = int(m.group(3))
            current_type  = "cp"
            skip_section  = False
            base_addr     = None
            reg_values    = {}
            continue

        # SDMA header - skip
        m = _SDMA_RE.match(line)
        if m:
            flush_section()
            current_type = "sdma"
            skip_section = True
            base_addr    = None
            reg_values   = {}
            continue

        if line.strip() == "Device is stopped":
            print("  [NOTE] Device scheduler is stopped")
            continue

        # Register dump line
        parsed = parse_reg_dump_line(line)
        if parsed and not skip_section and current_type in ("hiq", "cp"):
            addr, vals = parsed
            if base_addr is None:
                base_addr = addr
            for j, v in enumerate(vals):
                reg_idx = (addr - base_addr) // 4 + j
                reg_values[reg_idx] = v

    flush_section()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Parse a captured /sys/kernel/debug/kfd/hqds dump file.")
    ap.add_argument("file", nargs="?",
                    help="Path to the captured hqds file (default: stdin)")
    args = ap.parse_args()

    if args.file:
        try:
            with open(args.file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"File not found: {args.file}")
            sys.exit(1)
        except PermissionError:
            print(f"Permission denied: {args.file}")
            sys.exit(1)
    elif not sys.stdin.isatty():
        lines = sys.stdin.readlines()
    else:
        ap.print_help()
        sys.exit(0)

    if not lines:
        print("Input is empty.")
        return

    parse_hqds_file(lines)


if __name__ == "__main__":
    main()
