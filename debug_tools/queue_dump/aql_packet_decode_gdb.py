import gdb
import struct

outfile = open('aql_packet_decode_gdb.txt', 'w')
outfile.write("=== AQL + SDMA Packet Decode via plain gdb ===\n")
outfile.write("This extracts kernel names, signal addresses from AQL packets,\n")
outfile.write("and decodes SDMA (DMA) ring packets.\n")
outfile.write("Queues are discovered by walking the ROCR runtime singleton\n")
outfile.write("(no rocgdb 'info queues' / 'info dispatch' commands required;\n")
outfile.write("librocr debug symbols must be loaded).\n\n")

AQL_PACKET_SIZE = 64

# AQL packet types (lower 8 bits of header)
PACKET_TYPES = {
    0: "VENDOR_SPECIFIC",
    1: "INVALID",
    2: "KERNEL_DISPATCH",
    3: "BARRIER_AND",
    4: "AGENT_DISPATCH",
    5: "BARRIER_OR",
}

AMD_VENDOR_PACKET_BARRIER_VALUE = 2  # hsa_amd_barrier_value_packet_s

AMD_SIGNAL_VALUE_OFFSET = 8

HSA_SIGNAL_CONDITION_NAMES = {
    0: "EQ",
    1: "NE",
    2: "LT",
    3: "GTE",
}

# ---------------------------------------------------------------------------
# SDMA packet opcodes we decode
# ---------------------------------------------------------------------------
SDMA_OP_COPY        = 1   # 28 bytes
SDMA_OP_FENCE       = 5   # 16 bytes
SDMA_OP_TRAP        = 6   # 8  bytes
SDMA_OP_POLL_REGMEM = 8   # 24 bytes
SDMA_OP_ATOMIC      = 10  # 32 bytes
SDMA_OP_TIMESTAMP   = 13  # 12 bytes

SDMA_OP_NAMES = {
    SDMA_OP_COPY:        "SDMA_OP_COPY",
    SDMA_OP_FENCE:       "SDMA_OP_FENCE",
    SDMA_OP_TRAP:        "SDMA_OP_TRAP",
    SDMA_OP_POLL_REGMEM: "SDMA_OP_POLL_REGMEM",
    SDMA_OP_ATOMIC:      "SDMA_OP_ATOMIC",
    SDMA_OP_TIMESTAMP:   "SDMA_OP_TIMESTAMP",
}

SDMA_TIMESTAMP_SUBOP = {
    0: "SET_LOCAL_TIMESTAMP",
    1: "GET_LOCAL_TIMESTAMP",
    2: "GET_GLOBAL_TIMESTAMP",
}

# POLL_REGMEM func bits[30:28] of header
SDMA_POLL_REGMEM_FUNC = {
    0: "ALWAYS",
    1: "LT",
    2: "LTE",
    3: "EQ",
    4: "NE",
    5: "GTE",
    6: "GT",
    7: "RSVD",
}

# Offsets of 4-byte words inside amd_signal_s. Used to identify which field
# an SDMA packet targets and recover the encompassing signal.
# (amd_signal_s layout, see hsa_ext_amd.h.)
SIGNAL_FIELD_NAMES = {
    0:  "kind (lo32)",
    4:  "kind (hi32)",
    8:  "value (lo32)",
    12: "value (hi32)",
    16: "event_mailbox_ptr (lo32)",
    20: "event_mailbox_ptr (hi32)",
    24: "event_id",
    28: "reserved1",
    32: "start_ts (lo32)",
    36: "start_ts (hi32)",
    40: "end_ts (lo32)",
    44: "end_ts (hi32)",
    48: "queue_ptr (lo32)",
    52: "queue_ptr (hi32)",
}

# Aliases for 8-byte-wide field lookups (timestamps, value, etc.) where the
# SDMA packet addresses the whole 64-bit field by its base offset.
SIGNAL_FIELD_NAMES_64 = {
    0:  "kind",
    8:  "value",
    16: "event_mailbox_ptr",
    24: "event_id + reserved1",
    32: "start_ts",
    40: "end_ts",
    48: "queue_ptr",
}

VALID_SIGNAL_KINDS = {0, 1, -1, -2}

# amd_signal_s is declared with __attribute__((aligned(64))) in hsa_ext_amd.h,
# so every signal allocation is 64-byte aligned. We use this to recover the
# signal base from any address that falls inside it.
SIGNAL_ALIGNMENT = 64


def read_memory(addr, size):
    """Read memory from inferior process."""
    try:
        inferior = gdb.selected_inferior()
        return inferior.read_memory(addr, size).tobytes()
    except Exception as e:
        outfile.write(f"  [read_memory] Failed at 0x{addr:x} size={size}: {e}\n")
        return None


def read_amd_signal(signal_handle):
    """Read fields from an amd_signal_s structure."""
    if signal_handle == 0:
        return None
    try:
        data = read_memory(signal_handle, 56)
        if not data or len(data) < 56:
            return None

        kind = struct.unpack('<q', data[0:8])[0]
        value = struct.unpack('<q', data[8:16])[0]
        event_mailbox_ptr = struct.unpack('<Q', data[16:24])[0]
        event_id = struct.unpack('<I', data[24:28])[0]
        start_ts = struct.unpack('<Q', data[32:40])[0]
        end_ts = struct.unpack('<Q', data[40:48])[0]
        queue_ptr = struct.unpack('<Q', data[48:56])[0]

        kind_name = {
            0: "INVALID",
            1: "USER",
            -1: "DOORBELL",
            -2: "LEGACY_DOORBELL",
        }.get(kind, f"UNKNOWN({kind})")

        return {
            'kind': kind,
            'kind_name': kind_name,
            'value': value,
            'event_mailbox_ptr': event_mailbox_ptr,
            'event_id': event_id,
            'start_ts': start_ts,
            'end_ts': end_ts,
            'queue_ptr': queue_ptr,
        }
    except Exception as e:
        outfile.write(f"  [read_amd_signal] Failed for 0x{signal_handle:x}: {e}\n")
        return None


def format_signal_info(sig_info, indent="       "):
    """Detailed amd_signal_s dump - now always prints every meaningful field."""
    if sig_info is None:
        return f"{indent}<failed to read>\n"

    lines = []
    lines.append(f"{indent}  kind:      {sig_info['kind_name']} ({sig_info['kind']})")
    lines.append(f"{indent}  value:     {sig_info['value']} (0x{sig_info['value'] & 0xFFFFFFFFFFFFFFFF:016x})")

    if sig_info['value'] > 0:
        lines.append(f"{indent}  status:    PENDING (value={sig_info['value']})")
    elif sig_info['value'] == 0:
        lines.append(f"{indent}  status:    COMPLETED")
    else:
        lines.append(f"{indent}  status:    COMPLETED (negative: {sig_info['value']})")

    lines.append(f"{indent}  event_mb:  0x{sig_info['event_mailbox_ptr']:x}")
    lines.append(f"{indent}  event_id:  {sig_info['event_id']}")
    lines.append(f"{indent}  start_ts:  {sig_info['start_ts']}"
                 + ("" if sig_info['start_ts'] == 0 else f"  (0x{sig_info['start_ts']:x})"))
    lines.append(f"{indent}  end_ts:    {sig_info['end_ts']}"
                 + ("" if sig_info['end_ts'] == 0 else f"  (0x{sig_info['end_ts']:x})"))
    if sig_info['start_ts'] != 0 and sig_info['end_ts'] != 0 and sig_info['end_ts'] > sig_info['start_ts']:
        dur = sig_info['end_ts'] - sig_info['start_ts']
        lines.append(f"{indent}  duration:  {dur} ns ({dur / 1e6:.3f} ms)")
    lines.append(f"{indent}  queue_ptr: 0x{sig_info['queue_ptr']:x}")

    return '\n'.join(lines) + '\n'


def find_encompassing_signal(addr):
    """Round `addr` down to the 64-byte signal alignment, validate that the
    base actually looks like an amd_signal_s, and return the offset within it.

    Works regardless of which field inside the signal the SDMA packet was
    pointing at - start_ts (+32), end_ts (+40), value (+8), etc. all live in
    the same 64-byte cache-line-aligned block.

    Returns (offset_in_signal, signal_base, sig_info_dict) or (None, None, None).
    """
    base = addr & ~(SIGNAL_ALIGNMENT - 1)
    off = addr - base
    sig = read_amd_signal(base)
    if sig is None:
        return None, None, None
    if sig['kind'] not in VALID_SIGNAL_KINDS:
        return None, None, None
    return off, base, sig


def write_encompassing_signal(addr, indent="       "):
    """Look up & pretty-print the amd_signal_s that `addr` lives inside.

    Returns True if a signal was identified, False otherwise.
    """
    off, sig_base, sig = find_encompassing_signal(addr)
    if sig is None:
        outfile.write(f"{indent}encompassing signal: "
                      f"<not found - 0x{addr:x} & ~0x3f does not look like an amd_signal_s>\n")
        return False

    field64 = SIGNAL_FIELD_NAMES_64.get(off)
    field32 = SIGNAL_FIELD_NAMES.get(off, f"+0x{off:x}")
    field = field64 if field64 else field32

    outfile.write(f"{indent}encompassing signal:\n")
    outfile.write(f"{indent}  base:      0x{sig_base:x}\n")
    outfile.write(f"{indent}  field:     {field}  (signal+{off})\n")
    outfile.write(format_signal_info(sig, indent=indent + "  "))
    return True


def resolve_kernel_entry_point(kernel_object):
    """Return (absolute_entry_addr, symbol_str, kernarg_preload_bias) for a kernel descriptor.

    AMD amdhsa kernel_descriptor_t layout (64 bytes total):
      +0   uint32  group_segment_fixed_size
      +4   uint32  private_segment_fixed_size
      +8   uint32  kernarg_size
      +12  uint32  reserved0
      +16  int64   kernel_code_entry_byte_offset  (signed, delta from descriptor start)
      +24  uint8[20] reserved1
      +44  uint32  compute_pgm_rsrc3
      +48  uint32  compute_pgm_rsrc1
      +52  uint32  compute_pgm_rsrc2
      +56  uint16  kernel_code_properties
      +58  uint16  kernarg_preload_spec   <-- non-zero => kernarg preload enabled
      +60  uint32  reserved2

    Base entry = kernel_object + kernel_code_entry_byte_offset.
    When kernarg_preload_spec != 0 the compiler inserts a 256-byte preload
    trampoline at base entry; the real ISA code follows immediately after, so:
      entry_point = base_entry + 256  (with bias applied)
    """
    if not kernel_object:
        return None, None, False
    desc = read_memory(kernel_object, 60)
    if not desc or len(desc) < 60:
        return None, None, False
    entry_offset = struct.unpack('<q', desc[16:24])[0]        # signed int64
    kernarg_preload_spec = struct.unpack('<H', desc[58:60])[0]  # uint16
    base_entry = kernel_object + entry_offset
    preload_bias = (kernarg_preload_spec != 0)
    entry_addr = base_entry + (256 if preload_bias else 0)
    entry_sym = f"<unresolved @ 0x{entry_addr:x}>"
    try:
        result = gdb.execute(f"info symbol {entry_addr}", to_string=True)
        entry_sym = result.strip()
    except Exception:
        pass
    return entry_addr, entry_sym, preload_bias


def decode_kernel_dispatch(data, pkt_addr):
    if len(data) < 64:
        return None

    header = struct.unpack('<H', data[0:2])[0]
    setup = struct.unpack('<H', data[2:4])[0]
    workgroup_x = struct.unpack('<H', data[4:6])[0]
    workgroup_y = struct.unpack('<H', data[6:8])[0]
    workgroup_z = struct.unpack('<H', data[8:10])[0]
    grid_x = struct.unpack('<I', data[12:16])[0]
    grid_y = struct.unpack('<I', data[16:20])[0]
    grid_z = struct.unpack('<I', data[20:24])[0]
    private_segment_size = struct.unpack('<I', data[24:28])[0]
    group_segment_size = struct.unpack('<I', data[28:32])[0]
    kernel_object = struct.unpack('<Q', data[32:40])[0]
    kernarg_address = struct.unpack('<Q', data[40:48])[0]
    completion_signal = struct.unpack('<Q', data[56:64])[0]

    kernel_name = "??"
    if kernel_object != 0:
        try:
            result = gdb.execute(f"info symbol {kernel_object}", to_string=True)
            kernel_name = result.strip()
        except Exception:
            kernel_name = f"<unresolved @ 0x{kernel_object:x}>"

    entry_point, entry_symbol, entry_preload_bias = resolve_kernel_entry_point(kernel_object)

    return {
        'header': header,
        'setup': setup,
        'workgroup': (workgroup_x, workgroup_y, workgroup_z),
        'grid': (grid_x, grid_y, grid_z),
        'private_segment_size': private_segment_size,
        'group_segment_size': group_segment_size,
        'kernel_object': kernel_object,
        'kernel_name': kernel_name,
        'entry_point': entry_point,
        'entry_symbol': entry_symbol,
        'entry_preload_bias': entry_preload_bias,
        'kernarg': kernarg_address,
        'completion_signal': completion_signal,
    }


def decode_barrier_packet(data, pkt_addr, ptype):
    if len(data) < 64:
        return None

    header = struct.unpack('<H', data[0:2])[0]
    dep_signals = []
    for i in range(5):
        offset = 8 + i * 8
        sig = struct.unpack('<Q', data[offset:offset + 8])[0]
        dep_signals.append(sig)
    completion_signal = struct.unpack('<Q', data[56:64])[0]

    return {
        'header': header,
        'type_name': 'BARRIER_AND' if ptype == 3 else 'BARRIER_OR',
        'dep_signals': dep_signals,
        'completion_signal': completion_signal,
    }


def decode_barrier_value_packet(data, pkt_addr):
    if len(data) < 64:
        return None

    header = struct.unpack('<H', data[0:2])[0]
    # AmdFormat is a uint16_t at offset [2:4] (hsa_amd_vendor_packet_header_t).
    amd_format = struct.unpack('<H', data[2:4])[0]
    watch_signal = struct.unpack('<Q', data[8:16])[0]
    compare_value = struct.unpack('<q', data[16:24])[0]
    mask = struct.unpack('<q', data[24:32])[0]
    cond = struct.unpack('<I', data[32:36])[0]
    completion_signal = struct.unpack('<Q', data[56:64])[0]

    cond_name = HSA_SIGNAL_CONDITION_NAMES.get(cond, f"UNKNOWN({cond})")

    return {
        'header': header,
        'amd_format': amd_format,
        'watch_signal': watch_signal,
        'compare_value': compare_value,
        'mask': mask,
        'cond': cond,
        'cond_name': cond_name,
        'completion_signal': completion_signal,
    }


# ---------------------------------------------------------------------------
# SDMA packet decoding (DMA queues)
# ---------------------------------------------------------------------------
def decode_sdma_packet(pkt_addr):
    """Read and decode one SDMA packet starting at pkt_addr.

    Returns (info_dict, size_in_bytes). info_dict['op'] == 0 signals end of
    stream. Returns (None, 0) on read failure; (info, 0) for unknown opcodes
    whose length we can't determine safely.
    """
    head = read_memory(pkt_addr, 4)
    if not head or len(head) < 4:
        return None, 0

    header = struct.unpack('<I', head)[0]
    op = header & 0xFF
    sub_op = (header >> 8) & 0xFF

    if op == 0:
        return {'op': 0, 'sub_op': sub_op, 'header': header, 'name': 'END'}, 0

    if op == SDMA_OP_COPY:
        data = read_memory(pkt_addr, 28)
        if not data or len(data) < 28:
            return None, 0
        _hdr, count, parameter, src_addr, dst_addr = struct.unpack('<IIIQQ', data)
        return ({
            'op': op, 'sub_op': sub_op, 'header': header, 'name': SDMA_OP_NAMES[op],
            'count': count, 'parameter': parameter,
            'src_addr': src_addr, 'dst_addr': dst_addr,
        }, 28)

    if op == SDMA_OP_FENCE:
        data = read_memory(pkt_addr, 16)
        if not data or len(data) < 16:
            return None, 0
        _hdr, addr, value = struct.unpack('<IQI', data)
        return ({
            'op': op, 'sub_op': sub_op, 'header': header, 'name': SDMA_OP_NAMES[op],
            'addr': addr, 'data': value,
        }, 16)

    if op == SDMA_OP_TRAP:
        data = read_memory(pkt_addr, 8)
        if not data or len(data) < 8:
            return None, 0
        _hdr, context = struct.unpack('<II', data)
        return ({
            'op': op, 'sub_op': sub_op, 'header': header, 'name': SDMA_OP_NAMES[op],
            'context': context,
        }, 8)

    if op == SDMA_OP_POLL_REGMEM:
        # header(4) + addr(8) + value(4) + mask(4) + dw5(4) = 24
        data = read_memory(pkt_addr, 24)
        if not data or len(data) < 24:
            return None, 0
        _hdr, addr, value, mask, dw5 = struct.unpack('<IQIII', data)
        # Header:
        #   [7:0]   op
        #   [15:8]  sub_op
        #   [26]    hdp_flush
        #   [30:28] func
        #   [31]    mem_poll (0=register, 1=memory)
        func     = (header >> 28) & 0x7
        mem_poll = (header >> 31) & 0x1
        interval    = dw5 & 0xFFFF
        retry_count = (dw5 >> 16) & 0xFFF
        return ({
            'op': op, 'sub_op': sub_op, 'header': header, 'name': SDMA_OP_NAMES[op],
            'addr': addr, 'value': value, 'mask': mask,
            'func': func, 'func_name': SDMA_POLL_REGMEM_FUNC.get(func, f"UNK({func})"),
            'mem_poll': mem_poll, 'interval': interval, 'retry_count': retry_count,
        }, 24)

    if op == SDMA_OP_ATOMIC:
        # header(4) + addr(8) + src_data(8) + cmp_data(8) + loop_interval(4) = 32
        data = read_memory(pkt_addr, 32)
        if not data or len(data) < 32:
            return None, 0
        _hdr, addr, src_data, cmp_data, loop_interval = struct.unpack('<IQQQI', data)
        atomic_op = (header >> 25) & 0x7F
        return ({
            'op': op, 'sub_op': sub_op, 'header': header, 'name': SDMA_OP_NAMES[op],
            'atomic_op': atomic_op, 'addr': addr,
            'src_data': src_data, 'cmp_data': cmp_data,
            'loop_interval': loop_interval,
        }, 32)

    if op == SDMA_OP_TIMESTAMP:
        # header(4) + value_or_addr(8) = 12
        # sub_op 0 SET_LOCAL_TIMESTAMP: 64-bit timestamp value
        # sub_op 1 GET_LOCAL_TIMESTAMP: 64-bit dest address
        # sub_op 2 GET_GLOBAL_TIMESTAMP: 64-bit dest address (used by rocr)
        data = read_memory(pkt_addr, 12)
        if not data or len(data) < 12:
            return None, 0
        _hdr, value_or_addr = struct.unpack('<IQ', data)
        return ({
            'op': op, 'sub_op': sub_op, 'header': header, 'name': SDMA_OP_NAMES[op],
            'sub_op_name': SDMA_TIMESTAMP_SUBOP.get(sub_op, f"UNKNOWN({sub_op})"),
            'value_or_addr': value_or_addr,
        }, 12)

    return ({
        'op': op, 'sub_op': sub_op, 'header': header, 'name': f"UNKNOWN({op})",
    }, 0)


def _eval_poll(func, masked, ref):
    if   func == 0: return True
    elif func == 1: return masked <  ref
    elif func == 2: return masked <= ref
    elif func == 3: return masked == ref
    elif func == 4: return masked != ref
    elif func == 5: return masked >= ref
    elif func == 6: return masked >  ref
    return None


def write_sdma_packet(i, pkt_addr, pkt):
    op = pkt['op']
    outfile.write(
        f"  [{i}] @ 0x{pkt_addr:x}: {pkt['name']} "
        f"(op=0x{op:x}, sub_op=0x{pkt['sub_op']:x}, header=0x{pkt['header']:08x})\n"
    )

    if op == SDMA_OP_COPY:
        outfile.write(f"       count:     {pkt['count']} (bytes_to_copy = count+1 = {pkt['count'] + 1})\n")
        outfile.write(f"       parameter: 0x{pkt['parameter']:08x}\n")
        outfile.write(f"       src_addr:  0x{pkt['src_addr']:x}\n")
        outfile.write(f"       dst_addr:  0x{pkt['dst_addr']:x}\n")

    elif op == SDMA_OP_FENCE:
        outfile.write(f"       addr:      0x{pkt['addr']:x}\n")
        outfile.write(f"       data:      0x{pkt['data']:08x} ({pkt['data']})\n")
        mem = read_memory(pkt['addr'], 4)
        if mem:
            cur = struct.unpack('<I', mem)[0]
            outfile.write(f"       *addr:     0x{cur:08x} "
                          f"{'<fence written>' if cur == pkt['data'] else '<not yet written>'}\n")

    elif op == SDMA_OP_TRAP:
        outfile.write(f"       context:   0x{pkt['context']:08x}\n")

    elif op == SDMA_OP_POLL_REGMEM:
        target = "MEM" if pkt['mem_poll'] else "REG"
        outfile.write(f"       target:    {target} ({'memory' if pkt['mem_poll'] else 'register'})\n")
        outfile.write(f"       addr:      0x{pkt['addr']:x}\n")
        outfile.write(f"       func:      {pkt['func_name']} ({pkt['func']})\n")
        outfile.write(f"       ref_value: 0x{pkt['value']:08x} ({pkt['value']})\n")
        outfile.write(f"       mask:      0x{pkt['mask']:08x}\n")
        outfile.write(f"       interval:  {pkt['interval']} (poll cycles)\n")
        outfile.write(f"       retry:     {pkt['retry_count']} "
                      f"({'infinite' if pkt['retry_count'] == 0 else 'count'})\n")

        if pkt['mem_poll']:
            mem = read_memory(pkt['addr'], 4)
            if mem:
                cur = struct.unpack('<I', mem)[0]
                masked = cur & pkt['mask']
                ref    = pkt['value'] & pkt['mask']
                outfile.write(f"       *addr:     0x{cur:08x} (masked=0x{masked:08x})\n")
                satisfied = _eval_poll(pkt['func'], masked, ref)
                if satisfied is not None:
                    outfile.write(f"       eval:      masked {pkt['func_name']} 0x{ref:08x} -> "
                                  f"{'SATISFIED' if satisfied else 'BLOCKED (queue waiting here)'}\n")
            write_encompassing_signal(pkt['addr'])

    elif op == SDMA_OP_ATOMIC:
        outfile.write(f"       atomic_op: {pkt['atomic_op']}\n")
        outfile.write(f"       addr:      0x{pkt['addr']:x}\n")
        outfile.write(f"       src_data:  0x{pkt['src_data']:x} ({pkt['src_data']})\n")
        outfile.write(f"       cmp_data:  0x{pkt['cmp_data']:x} ({pkt['cmp_data']})\n")
        outfile.write(f"       loop_interval: {pkt['loop_interval']} "
                      f"(>>1 = {pkt['loop_interval'] >> 1})\n")
        # SDMA atomic typically modifies a signal's `value` field (+8).
        write_encompassing_signal(pkt['addr'])

    elif op == SDMA_OP_TIMESTAMP:
        outfile.write(f"       sub_op:    {pkt['sub_op_name']} ({pkt['sub_op']})\n")
        if pkt['sub_op'] == 0:
            outfile.write(f"       timestamp: 0x{pkt['value_or_addr']:x} ({pkt['value_or_addr']})\n")
        else:
            outfile.write(f"       dst_addr:  0x{pkt['value_or_addr']:x}\n")
            ts_mem = read_memory(pkt['value_or_addr'], 8)
            if ts_mem:
                ts_val = struct.unpack('<Q', ts_mem)[0]
                outfile.write(f"       *dst:      0x{ts_val:x} ({ts_val}) "
                              f"{'<written>' if ts_val != 0 else '<not yet written>'}\n")

    outfile.write("\n")


def decode_sdma_ring(ring_base, ring_size_bytes, max_packets=2048):
    """Walk an SDMA ring buffer starting at ring_base, stop at op==0 or end."""
    end = ring_base + ring_size_bytes
    addr = ring_base
    i = 0
    decoded = 0

    while addr + 4 <= end and i < max_packets:
        pkt, size = decode_sdma_packet(addr)
        if pkt is None:
            outfile.write(f"  [{i}] @ 0x{addr:x}: FAILED TO READ - stopping\n\n")
            return

        if pkt['op'] == 0:
            outfile.write(f"  [{i}] @ 0x{addr:x}: end-of-stream (op=0), "
                          f"decoded {decoded} packet(s)\n\n")
            return

        if size == 0:
            outfile.write(f"  [{i}] @ 0x{addr:x}: {pkt['name']} "
                          f"(header=0x{pkt['header']:08x}) - cannot continue past unknown op\n")
            sample = read_memory(addr, 64)
            if sample:
                outfile.write("       raw[64]: " +
                              ' '.join(f'{b:02x}' for b in sample) + "\n")
            outfile.write("\n")
            return

        write_sdma_packet(i, addr, pkt)
        decoded += 1
        addr += size
        i += 1

    outfile.write(f"  (stopped after {i} packet(s); reached "
                  f"{'max_packets' if i >= max_packets else 'end of buffer'})\n\n")


# ---------------------------------------------------------------------------
# Runtime singleton queue enumeration (plain gdb, no 'info queues')
#
# Walks rocr::core::Runtime::runtime_singleton_->gpu_agents_ to enumerate
# every HSA AQL queue (user-created + 3 internal per agent) and every SDMA
# blit ring (BlitSdma<...>) without going through any rocgdb-specific
# command. Requires librocr debug symbols loaded in gdb.
#
# Returned dicts use the same shape as the previous 'info queues' parser:
#   id, device, queue_num, qid, target_id, type, read, write, size, address
# so the existing decode_hsa_queue / decode_dma_queue functions consume them
# unchanged.
# ---------------------------------------------------------------------------
def _vec_iter(vec):
    """Yield each element of a libstdc++ std::vector<T> as a gdb.Value."""
    impl = vec['_M_impl']
    start = impl['_M_start']
    end = impl['_M_finish']
    elem_size = start.type.target().sizeof
    if elem_size == 0:
        return
    n = (int(end) - int(start)) // elem_size
    for i in range(n):
        yield (start + i).dereference()


def _unique_ptr_get(up):
    """Return the raw T* stored inside a std::unique_ptr<T>, as gdb.Value."""
    for path in (
        ('_M_t', '_M_t', '_M_head_impl'),    # libstdc++ (most versions)
        ('_M_t', '_M_head_impl'),            # libstdc++ (older)
        ('__ptr_',),                         # libc++
        ('_M_ptr',),                         # fallback
    ):
        try:
            v = up
            for p in path:
                v = v[p]
            return v
        except gdb.error:
            continue
        except Exception:
            continue
    return None


def _lazy_ptr_get(lp, target_type=None):
    """Return the raw T* held by a rocr::lazy_ptr<T>, as gdb.Value.

    Tries the typed std::unique_ptr internal-field path first; if that fails
    (older/newer libstdc++ layouts, missing debug info), falls back to
    reading the first 8 bytes of the lazy_ptr - safe because `obj` is the
    first member and libstdc++'s std::unique_ptr with an empty deleter is
    layout-compatible with a raw pointer.
    """
    try:
        v = _unique_ptr_get(lp['obj'])
        if v is not None:
            return v
    except Exception:
        pass

    if target_type is None:
        return None
    try:
        addr = int(lp.address)
        raw = read_memory(addr, 8)
        if not raw:
            return None
        ptr_val = struct.unpack('<Q', raw)[0]
        if ptr_val == 0:
            return None
        return gdb.Value(ptr_val).cast(target_type.pointer())
    except Exception:
        return None


def _hsa_queue_dict(core_q, dev_idx, q_idx, queue_id, internal=False):
    """Build the queue dict for a core::Queue / AqlQueue gdb.Value.

    Returns None for the runtime's internal 4 KB PM4 queues - they don't
    carry AQL packets, so we don't want them in the enumeration at all.
    """
    try:
        amd_q = core_q['amd_queue_']
        hsa_q = amd_q['hsa_queue']
        base = int(hsa_q['base_address'])
        slots = int(hsa_q['size'])
        qid = int(hsa_q['id'])
        rptr_field = amd_q['read_dispatch_id']
        wptr_field = amd_q['write_dispatch_id']
        read_idx = int(rptr_field)
        write_idx = int(wptr_field)
        rptr_addr = int(rptr_field.address) if rptr_field.address else None
        wptr_addr = int(wptr_field.address) if wptr_field.address else None
    except Exception as e:
        outfile.write(f"  [enum] cannot read amd_queue_ fields: {e}\n")
        return None

    size_bytes = slots * AQL_PACKET_SIZE
    if size_bytes == 4096:
        return None    # internal PM4 queue, not AQL - ignore.

    label = f"AMDGPU Queue {dev_idx}:{q_idx} (QID {qid})"
    if internal:
        label += " [internal]"
    return {
        'id': queue_id,
        'device': dev_idx,
        'queue_num': q_idx,
        'qid': qid,
        'target_id': label,
        'type': 'HSA',
        'read': read_idx,
        'write': write_idx,
        'rptr_addr': rptr_addr,
        'wptr_addr': wptr_addr,
        'size': size_bytes,
        'address': base,
    }


def _sdma_queue_dict(blit_obj, dev_idx, q_idx, queue_id):
    """Build the queue dict for a BlitSdma<...> gdb.Value pointer.

    blit_obj is a gdb.Value pointer (e.g. core::Blit*). dynamic_type is used
    to recover the concrete BlitSdma<...> instantiation, which is where the
    queue_start_addr_ / queue_resource_ / queue_(r|w)ptr_ fields live.
    Returns None if the object isn't an SDMA blit or hasn't been initialised.
    """
    try:
        real_type = blit_obj.dynamic_type
    except Exception:
        return None
    if real_type is None:
        return None
    type_name = real_type.name or str(real_type)
    if 'BlitSdma' not in type_name:
        return None    # likely a kernel blit (BlitKernel*), skip

    # dynamic_type may return either a pointer type or the bare object type
    # depending on gdb version. Normalize to a pointer and dereference.
    try:
        if real_type.code == gdb.TYPE_CODE_PTR:
            concrete = blit_obj.cast(real_type).dereference()
        else:
            concrete = blit_obj.cast(real_type.pointer()).dereference()
    except Exception as e:
        outfile.write(f"  [enum] cannot cast Blit to {type_name}: {e}\n")
        return None

    try:
        ring_base = int(concrete['queue_start_addr_'])
    except Exception:
        return None
    if ring_base == 0:
        return None    # lazy_ptr created but BlitSdma not yet Initialize()d

    qid = 0
    try:
        qid = int(concrete['queue_resource_']['QueueId'])
    except Exception:
        pass

    try:
        ring_size = int(gdb.parse_and_eval("'rocr::AMD::BlitSdmaBase'::kQueueSize"))
    except Exception:
        ring_size = 4 * 1024 * 1024   # historical default

    read_idx = None
    write_idx = None
    try:
        rptr = concrete['queue_rptr_']
        if int(rptr) != 0:
            read_idx = int(rptr.dereference())
    except Exception:
        pass
    try:
        wptr = concrete['queue_wptr_']
        if int(wptr) != 0:
            write_idx = int(wptr.dereference())
    except Exception:
        pass

    return {
        'id': queue_id,
        'device': dev_idx,
        'queue_num': q_idx,
        'qid': qid,
        'target_id': f"AMDGPU Queue {dev_idx}:{q_idx} (QID {qid})  <{type_name}>",
        'type': 'DMA',
        'read': read_idx,
        'write': write_idx,
        'size': ring_size,
        'address': ring_base,
    }


def enumerate_runtime_queues():
    """Walk rocr::core::Runtime::runtime_singleton_ and return a list of
    queue dicts compatible with decode_hsa_queue / decode_dma_queue.

    Plain gdb only - no rocgdb commands. Needs librocr debug symbols.
    """
    queues = []

    # Quote the class so gdb treats it as a single symbol; the last "::"
    # is otherwise ambiguous with the static-member separator.
    try:
        rt = gdb.parse_and_eval("'rocr::core::Runtime'::runtime_singleton_")
    except Exception as e:
        outfile.write(f"[enum] cannot read 'rocr::core::Runtime'::runtime_singleton_: {e}\n")
        outfile.write("       (librocr debug symbols not loaded?)\n\n")
        return queues

    if int(rt) == 0:
        outfile.write("[enum] runtime_singleton_ is null - runtime not initialised.\n\n")
        return queues

    try:
        runtime = rt.dereference()
        gpu_agents_vec = runtime['gpu_agents_']
    except Exception as e:
        outfile.write(f"[enum] cannot access gpu_agents_: {e}\n\n")
        return queues

    try:
        gpu_agent_ptr_t = gdb.lookup_type('rocr::AMD::GpuAgent').pointer()
    except Exception as e:
        outfile.write(f"[enum] cannot find type rocr::AMD::GpuAgent: {e}\n\n")
        return queues

    try:
        aql_queue_ptr_t = gdb.lookup_type('rocr::AMD::AqlQueue').pointer()
    except Exception:
        aql_queue_ptr_t = None

    try:
        core_blit_t = gdb.lookup_type('rocr::core::Blit')
    except Exception:
        core_blit_t = None

    try:
        core_queue_t = gdb.lookup_type('rocr::core::Queue')
    except Exception:
        core_queue_t = None

    qid_counter = 0

    for dev_idx, agent_ptr_v in enumerate(_vec_iter(gpu_agents_vec)):
        if int(agent_ptr_v) == 0:
            continue
        try:
            gpu_ptr = agent_ptr_v.cast(gpu_agent_ptr_t)
            gpu = gpu_ptr.dereference()
        except Exception as e:
            outfile.write(f"  [enum] device {dev_idx}: cannot cast to GpuAgent: {e}\n")
            continue

        # ---- (1) user-created AQL queues: GpuAgent::aql_queues_
        try:
            aql_vec = gpu['aql_queues_']
            for q_idx, aq_ptr_v in enumerate(_vec_iter(aql_vec)):
                if int(aq_ptr_v) == 0:
                    continue
                try:
                    if aql_queue_ptr_t is not None:
                        aq = aq_ptr_v.cast(aql_queue_ptr_t).dereference()
                    else:
                        aq = aq_ptr_v.dereference()
                    qd = _hsa_queue_dict(aq, dev_idx, q_idx, qid_counter)
                    if qd:
                        queues.append(qd)
                        qid_counter += 1
                except Exception as e:
                    outfile.write(f"  [warn] device {dev_idx} aql_queues_[{q_idx}]: {e}\n")
        except Exception as e:
            outfile.write(f"  [enum] device {dev_idx}: cannot read aql_queues_: {e}\n")

        # ---- (2) internal AQL queues: GpuAgent::queues_[QueueCount]
        # (QueueUtility, QueueBlitOnly, QueuePCSampling)
        try:
            internal = gpu['queues_']
            atype = internal.type
            try:
                rng = atype.range()
                nelems = int(rng[1] - rng[0]) + 1
            except Exception:
                nelems = int(atype.sizeof // atype.target().sizeof)
            for sub in range(nelems):
                lp = internal[sub]
                obj = _lazy_ptr_get(lp, core_queue_t)
                if obj is None or int(obj) == 0:
                    continue
                try:
                    core_q = obj.dereference()
                    qd = _hsa_queue_dict(core_q, dev_idx, 100 + sub, qid_counter,
                                          internal=True)
                    if qd:
                        queues.append(qd)
                        qid_counter += 1
                except Exception as e:
                    outfile.write(f"  [warn] device {dev_idx} queues_[{sub}]: {e}\n")
        except Exception as e:
            outfile.write(f"  [enum] device {dev_idx}: cannot read internal queues_: {e}\n")

        # ---- (3) SDMA blit rings: GpuAgent::blits_
        #
        # _vec_iter() uses Python integer arithmetic to compute the element
        # count (end - start) / elem_size, and elem_size may be wrong when
        # gdb cannot fully resolve the lazy_ptr<Blit> template instantiation
        # (it often returns pointer-size = 8 instead of the true ~80 bytes).
        # Instead, use gdb.parse_and_eval with the typed agent pointer so
        # gdb's own type-aware arithmetic computes the correct count and
        # index, bypassing the elem_size problem entirely.
        try:
            agent_expr = f"(('rocr::AMD::GpuAgent'*)0x{int(gpu_ptr):x})"
            n_blits = int(gdb.parse_and_eval(
                f"{agent_expr}->blits_._M_impl._M_finish"
                f" - {agent_expr}->blits_._M_impl._M_start"))
            null_ptrs = kernel_blits = uninit = added = 0
            for b_idx in range(n_blits):
                try:
                    lp = gdb.parse_and_eval(
                        f"{agent_expr}->blits_._M_impl._M_start[{b_idx}]")
                except Exception as e:
                    outfile.write(f"  [warn] device {dev_idx} blits_[{b_idx}]: {e}\n")
                    null_ptrs += 1
                    continue

                # Extract raw Blit* from lazy_ptr<Blit>::obj (unique_ptr<Blit>).
                # Try the typed path first; fall back to reading 8 bytes directly
                # from the start of the unique_ptr (which stores the raw pointer
                # as its first word in every known libstdc++/libc++ ABI).
                obj = None
                try:
                    obj = _unique_ptr_get(lp['obj'])
                except Exception:
                    pass
                if obj is None:
                    try:
                        up_addr = int(lp['obj'].address)
                        raw = read_memory(up_addr, 8)
                        if raw:
                            ptr_val = struct.unpack('<Q', raw)[0]
                            if ptr_val:
                                blit_t = core_blit_t or gdb.lookup_type('rocr::core::Blit')
                                obj = gdb.Value(ptr_val).cast(blit_t.pointer())
                    except Exception:
                        pass

                if obj is None or int(obj) == 0:
                    null_ptrs += 1
                    continue

                try:
                    dt = obj.dynamic_type
                    dt_name = (dt.name or str(dt)) if dt else "<unknown>"
                except Exception as e:
                    dt_name = f"<dynamic_type failed: {e}>"

                if 'BlitSdma' not in dt_name:
                    kernel_blits += 1
                    continue
                qd = _sdma_queue_dict(obj, dev_idx, 200 + b_idx, qid_counter)
                if qd is None:
                    uninit += 1
                    continue
                queues.append(qd)
                qid_counter += 1
                added += 1
            pass
        except Exception as e:
            outfile.write(f"  [enum] device {dev_idx}: cannot read blits_: {e}\n")

    return queues


# ---------------------------------------------------------------------------
# Per-queue dispatchers
# ---------------------------------------------------------------------------
def _write_signal_block(addr, label, indent="       "):
    """Print 'label: 0xADDR' followed by the full signal struct dump."""
    outfile.write(f"{indent}{label}: 0x{addr:x}\n")
    if addr != 0:
        sig = read_amd_signal(addr)
        outfile.write(f"{indent}{' ' * len(label)}  signal info:\n")
        outfile.write(format_signal_info(sig, indent=indent + ' ' * (len(label) + 2)))


# Per-packet-body printers, shared between live and inferred-consumed decoding.
def _print_kernel_dispatch_body(pkt, barrier_bit):
    gx, gy, gz = pkt['grid']
    wx, wy, wz = pkt['workgroup']
    outfile.write(f"       kernel:      {pkt['kernel_name']}\n")
    outfile.write(f"       kernel_obj:  0x{pkt['kernel_object']:x}\n")
    if pkt.get('entry_point') is not None:
        bias_note = "  (+256 kernarg_preload bias)" if pkt.get('entry_preload_bias') else ""
        outfile.write(f"       entry_point: 0x{pkt['entry_point']:x}{bias_note}  [{pkt['entry_symbol']}]\n")
    else:
        outfile.write(f"       entry_point: <unreadable>\n")
    outfile.write(f"       grid:        [{gx}, {gy}, {gz}]\n")
    outfile.write(f"       workgroup:   [{wx}, {wy}, {wz}]\n")
    outfile.write(f"       kernarg:     0x{pkt['kernarg']:x}\n")
    outfile.write(f"       private_seg: {pkt['private_segment_size']}\n")
    outfile.write(f"       group_seg:   {pkt['group_segment_size']}\n")
    if barrier_bit is not None:
        outfile.write(f"       barrier_bit: {barrier_bit}\n")
    _write_signal_block(pkt['completion_signal'], "completion ")
    outfile.write("\n")


def _print_barrier_body(pkt):
    for j, dep in enumerate(pkt['dep_signals']):
        if dep != 0:
            outfile.write(f"       dep[{j}]:     0x{dep:x}\n")
            sig = read_amd_signal(dep)
            outfile.write(format_signal_info(sig, indent="         "))
    _write_signal_block(pkt['completion_signal'], "completion ")
    outfile.write("\n")


def _print_barrier_value_body(pkt):
    outfile.write(f"       amd_format:  {pkt['amd_format']}\n")
    outfile.write(f"       compare_val: {pkt['compare_value']}\n")
    outfile.write(f"       mask:        0x{pkt['mask'] & 0xFFFFFFFFFFFFFFFF:016x}\n")
    outfile.write(f"       condition:   {pkt['cond_name']} ({pkt['cond']})\n")
    outfile.write(f"       watch_sig:   0x{pkt['watch_signal']:x}\n")
    if pkt['watch_signal'] != 0:
        ws = read_amd_signal(pkt['watch_signal'])
        outfile.write(format_signal_info(ws, indent="         "))
        if ws:
            masked = ws['value'] & pkt['mask']
            cmp_val = pkt['compare_value']
            cond = pkt['cond']
            if cond == 0:
                satisfied = (masked == cmp_val)
            elif cond == 1:
                satisfied = (masked != cmp_val)
            elif cond == 2:
                satisfied = (masked < cmp_val)
            elif cond == 3:
                satisfied = (masked >= cmp_val)
            else:
                satisfied = None
            if satisfied is not None:
                note = ""
                if pkt['mask'] == 0:
                    note = "  [mask=0: (value & 0) is always 0]"
                outfile.write(
                    f"       eval:        (sig_val & mask) = {masked}, "
                    f"{pkt['cond_name']} {cmp_val} -> "
                    f"{'SATISFIED' if satisfied else 'BLOCKED'}{note}\n"
                )
    _write_signal_block(pkt['completion_signal'], "completion ")
    outfile.write("\n")


def infer_consumed_packet_type(data):
    """Given a 64-byte AQL packet whose header has been INVALIDated by the
    consumer (per HSA spec - consumer overwrites the header type with
    HSA_PACKET_TYPE_INVALID), infer the original packet type from body
    content that is left intact.

    Discriminators (in priority order):
      1) workgroup_size_x (uint16_t @ +4) >= 1 AND kernel_object (+32) resolves
         to a symbol -> KERNEL_DISPATCH.  workgroup_size_x must be >= 1 per
         HSA spec for a valid kernel dispatch packet; reserved/dep-signal
         positions in the other packet types leave that field at 0.
      2) AmdFormat (uint16_t @ +2) == 2 AND workgroup_size_x == 0
         -> BARRIER_VALUE.
      3) workgroup_size_x == 0 AND at least one of the 5 dep_signal slots
         (+8 .. +47) is a valid amd_signal_t (64-byte aligned, valid kind)
         -> BARRIER_AND_OR (cannot distinguish AND from OR without header).

    Returns (type_name, original_ptype) where original_ptype is the AQL type
    byte the header WOULD have had.
    """
    if len(data) < 64:
        return ("UNKNOWN", 1)

    # Per HSA spec, these two fields disambiguate the three packet types we
    # decode:
    #
    #   +2..+3 (uint16)  : KERNEL_DISPATCH.setup (>=1, dimension count)
    #                      BARRIER_AND/OR.reserved0 (== 0)
    #                      BARRIER_VALUE.AmdFormat (== 2)
    #   +4..+7 (uint32)  : KERNEL_DISPATCH.workgroup_size_x/y (lo16 >= 1)
    #                      BARRIER_AND/OR.reserved1 (== 0)
    #                      BARRIER_VALUE.reserved0 (== 0)
    #
    # Order of checks: workgroup_size_x first - it cleanly separates
    # KERNEL_DISPATCH from the two barrier flavors. Then AmdFormat picks
    # between BARRIER_VALUE and BARRIER_AND/OR.
    w2_uint16  = struct.unpack('<H', data[2:4])[0]
    w4_uint32  = struct.unpack('<I', data[4:8])[0]
    workgroup_size_x = w4_uint32 & 0xFFFF
    amd_format       = w2_uint16

    kernel_obj = struct.unpack('<Q', data[32:40])[0]

    # Both barrier flavors require the uint32 @ +4 to be 0
    # (KERNEL_DISPATCH.workgroup_size_x/y vs BARRIER.reserved1 / BARRIER_VALUE.reserved0).
    # If +4 is non-zero, the only candidate is KERNEL_DISPATCH.
    if w4_uint32 != 0:
        if workgroup_size_x >= 1 and kernel_obj != 0:
            return ("KERNEL_DISPATCH", 2)
        return ("UNKNOWN", 1)

    # w4_uint32 == 0 here, so KERNEL_DISPATCH is ruled out (wg_x must be >= 1).
    if amd_format == AMD_VENDOR_PACKET_BARRIER_VALUE:
        return ("BARRIER_VALUE", 0)

    if w2_uint16 == 0:  # BARRIER_AND/OR.reserved0
        return ("BARRIER_AND_OR", 3)

    return ("UNKNOWN", 1)


def _dump_consumed_aql_packets(ring_base, num_slots, read_idx, max_count=5):
    """Dump the last few packets in an idle (rptr == wptr) but used queue.

    The consumer overwrites only the header byte with INVALID; the body
    remains intact, so we can infer the original packet type and replay the
    structured decode against it.
    """
    n = min(max_count, read_idx, num_slots)
    if n == 0:
        return

    outfile.write(f"  (Queue is empty but has been used; dumping last {n} "
                  f"consumed packet(s) - header is INVALID, original type inferred from body)\n\n")

    for offset in range(1, n + 1):
        pkt_idx = read_idx - offset
        slot = pkt_idx % num_slots
        pkt_addr = ring_base + slot * AQL_PACKET_SIZE

        data = read_memory(pkt_addr, AQL_PACKET_SIZE)
        if not data:
            outfile.write(f"  [-{offset}] slot={slot} @ 0x{pkt_addr:x}: FAILED TO READ\n\n")
            continue

        header = struct.unpack('<H', data[0:2])[0]
        type_name, inferred_ptype = infer_consumed_packet_type(data)

        outfile.write(
            f"  [-{offset}] slot={slot} @ 0x{pkt_addr:x}: CONSUMED "
            f"(header=0x{header:04x}); inferred: {type_name}\n"
        )

        hex_dump = ' '.join(f'{b:02x}' for b in data)
        if inferred_ptype == 2:
            pkt = decode_kernel_dispatch(data, pkt_addr)
            if pkt:
                # barrier_bit is in the (overwritten) header - we no longer
                # know its original value, so omit.
                _print_kernel_dispatch_body(pkt, barrier_bit=None)
        elif inferred_ptype == 3:
            pkt = decode_barrier_packet(data, pkt_addr, 3)
            if pkt:
                _print_barrier_body(pkt)
        elif inferred_ptype == 0:
            pkt = decode_barrier_value_packet(data, pkt_addr)
            if pkt:
                _print_barrier_value_body(pkt)
        outfile.write(f"       raw: {hex_dump}\n\n")

    outfile.write("\n")


def decode_hsa_queue(queue):
    read_idx = queue['read']
    write_idx = queue['write']
    ring_size_bytes = queue['size']
    ring_base = queue['address']

    num_slots = ring_size_bytes // AQL_PACKET_SIZE
    if num_slots == 0:
        outfile.write(f"Queue {queue['target_id']}: ring size {ring_size_bytes} too small\n\n")
        return

    # 4096-byte HSA queues are the runtime's internal PM4 queues, not AQL.
    # Skip them - their packets don't follow the AQL layout we decode.
    if ring_size_bytes == 4096:
        return

    if write_idx < read_idx:
        outfile.write(
            f"Queue {queue['target_id']}: write ({write_idx}) < read ({read_idx}), "
            f"unexpected - skipping\n\n"
        )
        return

    pending = write_idx - read_idx
    # Empty + virgin queue: nothing to show.
    if pending == 0 and read_idx == 0:
        return

    outfile.write(f"{'=' * 74}\n")
    outfile.write(f"Queue {queue['target_id']} (device {queue['device']}, QID {queue['qid']}) [HSA]\n")
    outfile.write(f"  Ring base:   0x{ring_base:x}\n")
    outfile.write(f"  Ring size:   {ring_size_bytes} bytes ({num_slots} packet slots)\n")
    rptr_addr = queue.get('rptr_addr')
    wptr_addr = queue.get('wptr_addr')
    rptr_addr_str = f"  [@ 0x{rptr_addr:x}]" if rptr_addr else ""
    wptr_addr_str = f"  [@ 0x{wptr_addr:x}]" if wptr_addr else ""
    outfile.write(f"  Read index:  {read_idx} (slot {read_idx % num_slots}){rptr_addr_str}\n")
    outfile.write(f"  Write index: {write_idx} (slot {write_idx % num_slots}){wptr_addr_str}\n")
    outfile.write(f"  Pending:     {pending} packet(s)\n")
    outfile.write(f"{'=' * 74}\n\n")

    # Empty but used queue: dump the last few consumed packets.
    if pending == 0:
        _dump_consumed_aql_packets(ring_base, num_slots, read_idx, max_count=5)
        return

    max_decode = min(pending, 512)
    if pending > max_decode:
        outfile.write(f"  (Decoding first {max_decode} of {pending} pending packets)\n\n")

    for i in range(max_decode):
        pkt_idx = read_idx + i
        slot = pkt_idx % num_slots
        pkt_addr = ring_base + slot * AQL_PACKET_SIZE

        data = read_memory(pkt_addr, AQL_PACKET_SIZE)
        if not data:
            outfile.write(f"  [{i}] slot={slot} @ 0x{pkt_addr:x}: FAILED TO READ\n\n")
            continue

        header = struct.unpack('<H', data[0:2])[0]
        ptype = header & 0xFF
        barrier_bit = (header >> 8) & 1
        type_name = PACKET_TYPES.get(ptype, f"UNKNOWN({ptype})")

        hex_dump = ' '.join(f'{b:02x}' for b in data)

        if ptype == 2:
            pkt = decode_kernel_dispatch(data, pkt_addr)
            if pkt:
                outfile.write(f"  [{i}] slot={slot} @ 0x{pkt_addr:x}: KERNEL_DISPATCH\n")
                _print_kernel_dispatch_body(pkt, barrier_bit)
                outfile.write(f"       raw: {hex_dump}\n\n")

        elif ptype in (3, 5):
            pkt = decode_barrier_packet(data, pkt_addr, ptype)
            if pkt:
                outfile.write(f"  [{i}] slot={slot} @ 0x{pkt_addr:x}: {pkt['type_name']}\n")
                _print_barrier_body(pkt)
                outfile.write(f"       raw: {hex_dump}\n\n")

        elif ptype == 0:
            amd_format = struct.unpack('<H', data[2:4])[0]
            if amd_format == AMD_VENDOR_PACKET_BARRIER_VALUE:
                pkt = decode_barrier_value_packet(data, pkt_addr)
                if pkt:
                    outfile.write(
                        f"  [{i}] slot={slot} @ 0x{pkt_addr:x}: "
                        f"VENDOR_SPECIFIC / AMD BARRIER_VALUE\n"
                    )
                    _print_barrier_value_body(pkt)
                    outfile.write(f"       raw: {hex_dump}\n\n")
            else:
                outfile.write(
                    f"  [{i}] slot={slot} @ 0x{pkt_addr:x}: "
                    f"VENDOR_SPECIFIC (amd_format={amd_format})\n"
                )
                outfile.write(f"       raw: {hex_dump}\n\n")

        elif ptype == 1:
            type_name_inf, inferred_ptype = infer_consumed_packet_type(data)
            outfile.write(
                f"  [{i}] slot={slot} @ 0x{pkt_addr:x}: INVALID (consumed); "
                f"inferred: {type_name_inf}\n"
            )
            if inferred_ptype == 2:
                pkt = decode_kernel_dispatch(data, pkt_addr)
                if pkt:
                    _print_kernel_dispatch_body(pkt, barrier_bit=None)
            elif inferred_ptype == 3:
                pkt = decode_barrier_packet(data, pkt_addr, 3)
                if pkt:
                    _print_barrier_body(pkt)
            elif inferred_ptype == 0:
                pkt = decode_barrier_value_packet(data, pkt_addr)
                if pkt:
                    _print_barrier_value_body(pkt)
            outfile.write(f"       raw: {hex_dump}\n\n")

        else:
            outfile.write(
                f"  [{i}] slot={slot} @ 0x{pkt_addr:x}: "
                f"{type_name} (header=0x{header:04x})\n"
            )
            outfile.write(f"       raw: {hex_dump}\n\n")

    outfile.write("\n")


def decode_dma_queue(queue):
    ring_base = queue['address']
    ring_size_bytes = queue['size']
    rptr = queue.get('read')
    wptr = queue.get('write')

    outfile.write(f"{'=' * 74}\n")
    outfile.write(f"Queue {queue['target_id']} (device {queue['device']}, "
                  f"QID {queue['qid']}) [{queue['type']} / SDMA]\n")
    outfile.write(f"  Ring base:   0x{ring_base:x}\n")
    outfile.write(f"  Ring size:   {ring_size_bytes} bytes\n")
    if rptr is not None:
        outfile.write(f"  RPTR:        {rptr} (ring offset {rptr % ring_size_bytes})\n")
    if wptr is not None:
        outfile.write(f"  WPTR:        {wptr} (ring offset {wptr % ring_size_bytes})\n")
    if rptr is not None and wptr is not None:
        if wptr >= rptr:
            outfile.write(f"  Pending:     {wptr - rptr} bytes\n")
        else:
            outfile.write(f"  Pending:     <wptr < rptr, unexpected>\n")
    outfile.write(f"{'=' * 74}\n\n")

    # When we know RPTR & WPTR, decode just the live (unconsumed) region.
    if (rptr is not None and wptr is not None and wptr > rptr
            and ring_size_bytes > 0):
        pending = wptr - rptr
        start_off = rptr % ring_size_bytes
        if start_off + pending <= ring_size_bytes:
            decode_sdma_ring(ring_base + start_off, pending)
        else:
            # Wraps around the end of the ring: BlitSdma pads to the boundary
            # with NOOPs (op==0), then writes the rest at offset 0.
            tail = ring_size_bytes - start_off
            outfile.write(f"  (Region wraps; decoding tail then head)\n")
            outfile.write(f"  -- tail {tail} bytes @ ring+{start_off}:\n\n")
            decode_sdma_ring(ring_base + start_off, tail)
            outfile.write(f"  -- head {pending - tail} bytes @ ring+0:\n\n")
            decode_sdma_ring(ring_base, pending - tail)
        return

    # No RPTR/WPTR available (or queue empty) - linear scan from the start.
    if rptr is not None and wptr is not None and rptr == wptr:
        outfile.write("  (Queue empty: no pending SDMA packets.)\n\n")
        return
    outfile.write("  (No RPTR/WPTR available; scanning linearly until op=0.)\n\n")
    decode_sdma_ring(ring_base, ring_size_bytes)


# ---------------------------------------------------------------------------
# Threads waiting in InterruptSignal::WaitRelaxed
# ---------------------------------------------------------------------------
# rocr layout (from ROCR-Runtime sources, core/inc/signal.h):
#
#   class InterruptSignal : private LocalSignal, public Signal { ... };
#
#   class Signal {
#     ...
#     amd_signal_t& signal_;   // public, stored as a pointer
#   };
#
# hsa_signal_t.handle == &signal->signal_ (i.e. the amd_signal_t address
# directly), so once we recover the value of `signal_`, we can hand it
# straight to read_amd_signal().
def _extract_this_from_frame(frame):
    """Return the `this` address for a member-function frame, best-effort."""
    try:
        v = frame.read_var('this')
        return int(v), "frame.read_var"
    except Exception:
        pass
    # x86_64 SysV ABI: `this` is in rdi at call. May be clobbered, but worth
    # trying before giving up.
    try:
        v = frame.read_register('rdi')
        return int(v), "rdi (may be stale)"
    except Exception:
        pass
    return None, None


def _write_backtrace(start_frame, max_frames=10, indent="    "):
    """Write a short backtrace starting at `start_frame`, walking older frames."""
    frame = start_frame
    n = 0
    while frame is not None and n < max_frames:
        try:
            fn = frame.name() or "??"
        except Exception:
            fn = "??"
        loc = ""
        try:
            sal = frame.find_sal()
            if sal and sal.symtab and sal.line:
                loc = f"  at {sal.symtab.filename}:{sal.line}"
            else:
                pc = frame.pc()
                if pc:
                    loc = f"  pc=0x{int(pc):x}"
        except Exception:
            pass
        outfile.write(f"{indent}#{n}  {fn}{loc}\n")
        try:
            frame = frame.older()
        except Exception:
            break
        n += 1
    if frame is not None:
        outfile.write(f"{indent}    (... more frames omitted)\n")


def _locate_signal_ref_from_this(this_addr):
    """Find `&this->signal_` given the InterruptSignal* `this`.

    Strategy 1: ask gdb to evaluate it directly (requires librocr debug syms).
    Strategy 2: scan the object's first ~12 8-byte slots for a pointer that
                lands on a 64-byte-aligned amd_signal_t with a valid `kind`.
    """
    try:
        v = gdb.parse_and_eval(
            f"&((rocr::core::Signal*){this_addr:#x})->signal_"
        )
        return int(v), "gdb type lookup"
    except Exception:
        pass

    for off in range(0, 96, 8):
        ptr_bytes = read_memory(this_addr + off, 8)
        if not ptr_bytes:
            continue
        ptr = struct.unpack('<Q', ptr_bytes)[0]
        if ptr == 0 or (ptr & (SIGNAL_ALIGNMENT - 1)):
            continue
        sig = read_amd_signal(ptr)
        if sig and sig['kind'] in VALID_SIGNAL_KINDS:
            return ptr, f"scan @ this+{off}"
    return None, None


def find_and_dump_interrupt_signal_waiters():
    """Walk every thread, find frames in InterruptSignal::WaitRelaxed or
    BusyWaitSignal::WaitRelaxed, and dump the amd_signal_t being waited on.
    """
    _dump_wait_relaxed_waiters(
        class_names=("InterruptSignal", "BusyWaitSignal"),
    )


def _dump_wait_relaxed_waiters(class_names):
    label = " / ".join(f"{c}::WaitRelaxed" for c in class_names)
    outfile.write(f"--- Threads waiting in {label} ---\n\n")

    inferior = gdb.selected_inferior()
    try:
        original_thread = gdb.selected_thread()
    except Exception:
        original_thread = None

    found = 0
    skipped = 0

    for thread in inferior.threads():
        try:
            thread.switch()
        except Exception:
            skipped += 1
            continue

        try:
            frame = gdb.newest_frame()
        except Exception:
            skipped += 1
            continue

        depth = 0
        match = None
        while frame is not None and depth < 256:
            try:
                fn = frame.name() or ""
            except Exception:
                fn = ""
            if "WaitRelaxed" in fn and any(c in fn for c in class_names):
                match = (depth, frame, fn)
                break
            try:
                frame = frame.older()
            except Exception:
                break
            depth += 1

        if not match:
            continue

        found += 1
        fdepth, fframe, fname = match
        ptid = thread.ptid
        lwp = ptid[1] if len(ptid) >= 2 else "?"
        outfile.write(f"Thread {thread.num} (LWP {lwp}): frame #{fdepth}: {fname}\n")

        def _emit_backtrace():
            outfile.write(f"  backtrace (max 20 frames from WaitRelaxed):\n")
            _write_backtrace(fframe, max_frames=20, indent="    ")
            outfile.write("\n")

        this_addr, this_src = _extract_this_from_frame(fframe)
        if this_addr is None:
            outfile.write(f"  Failed to extract `this` from frame.\n")
            _emit_backtrace()
            continue
        outfile.write(f"  this:        0x{this_addr:x}  [{this_src}]\n")

        for arg in ('condition', 'compare_value', 'timeout', 'wait_hint'):
            try:
                v = fframe.read_var(arg)
                outfile.write(f"  {arg:14s} {v}\n")
            except Exception:
                pass

        sig_addr, sig_src = _locate_signal_ref_from_this(this_addr)
        if sig_addr is None:
            outfile.write(f"  Could not locate signal_ in this object.\n")
            _emit_backtrace()
            continue

        off, sig_base, sig = find_encompassing_signal(sig_addr)
        if sig is None:
            outfile.write(f"  signal_:     0x{sig_addr:x}  [{sig_src}]  "
                          f"<not a valid amd_signal_t>\n")
            _emit_backtrace()
            continue

        outfile.write(f"  signal:      0x{sig_base:x}  [{sig_src}]\n")
        if off != 0:
            outfile.write(f"               (signal_+{off})\n")
        outfile.write(format_signal_info(sig, indent="    "))
        _emit_backtrace()

    if original_thread is not None:
        try:
            original_thread.switch()
        except Exception:
            pass

    if found == 0:
        outfile.write(f"No threads found waiting in {label}.\n")
    else:
        outfile.write(f"Total: {found} thread(s) in WaitRelaxed.\n")
    if skipped:
        outfile.write(f"({skipped} thread(s) skipped due to errors)\n")
    outfile.write("\n")


HSA_EVENTTYPE_NAMES = {
    0: "SIGNAL",
    1: "NODECHANGE",
    2: "DEVICESTATECHANGE",
    3: "HW_EXCEPTION",
    4: "SYSTEM_EVENT",
    5: "DEBUG_EVENT",
    6: "PROFILE_EVENT",
    7: "QUEUE_EVENT",
    8: "MEMORY",
}


def find_and_dump_kmt_waiters():
    """Walk every thread, find frames in hsaKmtWaitOnMultipleEvents_Ext,
    decode each HsaEvent* in the Events[] array, and resolve the associated
    amd_signal_s via the SyncVar.UserDataPtrValue pointer (64-byte aligned).
    """
    outfile.write("--- Threads in hsaKmtWaitOnMultipleEvents_Ext ---\n\n")

    inferior = gdb.selected_inferior()
    try:
        original_thread = gdb.selected_thread()
    except Exception:
        original_thread = None

    found = 0

    for thread in inferior.threads():
        try:
            thread.switch()
            frame = gdb.newest_frame()
        except Exception:
            continue

        # Find the outermost (highest-numbered) frame with this name —
        # there may be inlined/recursive instances; we want the last one.
        depth = 0
        target = None
        while frame is not None and depth < 256:
            try:
                fn = frame.name() or ""
            except Exception:
                fn = ""
            if "hsaKmtWaitOnMultipleEvents_Ext" in fn:
                target = (depth, frame, fn)  # keep updating; last wins
            try:
                frame = frame.older()
            except Exception:
                break
            depth += 1

        if not target:
            continue

        found += 1
        fdepth, fframe, fname = target
        ptid = thread.ptid
        lwp = ptid[1] if len(ptid) >= 2 else "?"
        outfile.write(f"Thread {thread.num} (LWP {lwp}): frame #{fdepth}: {fname}\n")

        # ------------------------------------------------------------------
        # Extract Events (HsaEvent**) and NumEvents.
        # Try debug-info variable names first; fall back to registers.
        # x86-64 SysV: rdi=Events, rsi=NumEvents.
        # ------------------------------------------------------------------
        events_ptr = None
        num_events = None
        for name in ('Events', 'events'):
            try:
                events_ptr = int(fframe.read_var(name))
                break
            except Exception:
                pass
        for name in ('NumEvents', 'num_events'):
            try:
                num_events = int(fframe.read_var(name))
                break
            except Exception:
                pass
        if events_ptr is None:
            try:
                events_ptr = int(fframe.read_register('rdi'))
            except Exception:
                pass
        if num_events is None:
            try:
                num_events = int(fframe.read_register('rsi'))
            except Exception:
                pass

        if events_ptr is None or num_events is None:
            outfile.write("  Could not read Events/NumEvents arguments.\n\n")
            continue
        if not (0 < num_events <= 1024):
            outfile.write(f"  NumEvents={num_events} looks invalid.\n\n")
            continue

        outfile.write(f"  Events=0x{events_ptr:x}  NumEvents={num_events}\n\n")

        # ------------------------------------------------------------------
        # Decode each HsaEvent*.
        # HsaEvent layout (hsakmttypes.h):
        #   +0  HSA_EVENTID EventId          (uint32)
        #   +4  HsaEventData.EventType       (uint32)
        #   +8  HsaEventData.EventData.SyncVar.UserDataPtrValue (uint64)
        #           -> for SIGNAL events this points into the amd_signal_s
        #              (value field or mailbox); use 64-byte alignment to
        #              recover the signal base.
        #   +16 HsaEventData.EventData.SyncVar.SyncVarSize (uint64)
        # ------------------------------------------------------------------
        for i in range(num_events):
            ptr_bytes = read_memory(events_ptr + i * 8, 8)
            if not ptr_bytes:
                outfile.write(f"  Events[{i}]: cannot read pointer\n")
                continue
            event_ptr = struct.unpack('<Q', ptr_bytes)[0]
            if event_ptr == 0:
                outfile.write(f"  Events[{i}]: null\n")
                continue

            # HsaEvent layout (with compiler padding on x86-64):
            #   +0  EventId          (uint32)
            #   +4  <padding>
            #   +8  EventData.EventType  (uint32)
            #   +12 <padding>
            #   +16 EventData.SyncVar.UserDataPtrValue (uint64)
            #   +24 EventData.SyncVar.SyncVarSize      (uint64)
            #   +32 EventData.HWData1  (uint64)
            #   +40 EventData.HWData2  (uint64)
            #   +48 EventData.HWData3  (uint32)
            hdr = read_memory(event_ptr, 52)
            if not hdr or len(hdr) < 52:
                outfile.write(f"  Events[{i}] @ 0x{event_ptr:x}: cannot read struct\n")
                continue

            event_id   = struct.unpack('<I', hdr[0:4])[0]
            event_type = struct.unpack('<I', hdr[8:12])[0]
            user_data  = struct.unpack('<Q', hdr[16:24])[0]
            type_name  = HSA_EVENTTYPE_NAMES.get(event_type, f"UNKNOWN({event_type})")

            outfile.write(f"  Events[{i}] @ 0x{event_ptr:x}:\n")
            outfile.write(f"    EventId:   {event_id}\n")
            outfile.write(f"    EventType: {type_name}\n")
            outfile.write(f"    UserData:  0x{user_data:x}\n")

            if user_data == 0:
                outfile.write("    (no UserData - cannot resolve signal)\n\n")
                continue

            # Use 64-byte signal alignment to find the amd_signal_s base.
            off, sig_base, sig = find_encompassing_signal(user_data)
            if sig is None:
                outfile.write(f"    (UserData 0x{user_data:x} does not resolve "
                              f"to a valid amd_signal_s)\n\n")
                continue

            field = SIGNAL_FIELD_NAMES_64.get(off) or SIGNAL_FIELD_NAMES.get(off, f"+0x{off:x}")
            outfile.write(f"    signal @ 0x{sig_base:x}  "
                          f"(UserData is signal.{field})\n")
            outfile.write(format_signal_info(sig, indent="      "))
            outfile.write("\n")

    if original_thread is not None:
        try:
            original_thread.switch()
        except Exception:
            pass

    if found == 0:
        outfile.write("No threads found in hsaKmtWaitOnMultipleEvents_Ext.\n")
    outfile.write("\n")


def find_and_decode_queues():
    outfile.write("--- Enumerating HSA / DMA queues via ROCR runtime singleton ---\n\n")

    queues = enumerate_runtime_queues()

    n_hsa  = sum(1 for q in queues if q['type'] == 'HSA')
    n_dma  = sum(1 for q in queues if q['type'] == 'DMA')
    n_xgmi = sum(1 for q in queues if q['type'] == 'XGMI')
    outfile.write(f"Enumerated {len(queues)} queue(s): "
                  f"{n_hsa} HSA, {n_dma} DMA, {n_xgmi} XGMI\n\n")

    if not queues:
        outfile.write("No queues found (runtime not initialised or debug "
                      "symbols missing for librocr).\n")
        return

    # DMA/XGMI first, then HSA in original enumeration order.
    queues.sort(key=lambda q: 0 if q['type'] in ('DMA', 'XGMI') else 1)

    # Brief table-style summary so the output is comparable to old 'info queues'.
    outfile.write(
        f"{'Id':>3}  {'Target':<48} {'Type':<5} "
        f"{'Read':>12} {'Write':>12} {'Size':>10} {'Address':>18}\n"
    )
    for q in queues:
        r = '-' if q['read']  is None else str(q['read'])
        w = '-' if q['write'] is None else str(q['write'])
        outfile.write(
            f"{q['id']:>3}  {q['target_id']:<48} {q['type']:<5} "
            f"{r:>12} {w:>12} {q['size']:>10}  0x{q['address']:016x}\n"
        )
    outfile.write("\n")

    for queue in queues:
        if queue['type'] == 'HSA':
            decode_hsa_queue(queue)
        elif queue['type'] in ('DMA', 'XGMI'):
            decode_dma_queue(queue)
        else:
            outfile.write(f"Queue {queue['target_id']}: unsupported type "
                          f"{queue['type']}, skipping\n\n")


# ============================================================
# Main
# ============================================================
try:
    find_and_decode_queues()
except Exception as e:
    outfile.write(f"\nFATAL ERROR in find_and_decode_queues: {e}\n")
    import traceback
    outfile.write(traceback.format_exc())

try:
    find_and_dump_interrupt_signal_waiters()
except Exception as e:
    outfile.write(f"\nFATAL ERROR in find_and_dump_interrupt_signal_waiters: {e}\n")
    import traceback
    outfile.write(traceback.format_exc())

# try:
#     find_and_dump_kmt_waiters()
# except Exception as e:
#     outfile.write(f"\nFATAL ERROR in find_and_dump_kmt_waiters: {e}\n")
#     import traceback
#     outfile.write(traceback.format_exc())

outfile.close()
print("AQL + SDMA packet decode complete - see aql_packet_decode_gdb.txt")




