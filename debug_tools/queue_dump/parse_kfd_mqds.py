#!/usr/bin/env python3
"""
Parser for /sys/kernel/debug/kfd/mqds
Targets struct v9_mqd (gfx9 / gfx950 compute queues) and struct v9_sdma_mqd.
Source: https://github.com/ROCm/amdgpu/blob/master/drivers/gpu/drm/amd/include/v9_structs.h#L159

Output format from kernel (kfd_mqd_manager_v9.c debugfs_show_mqd):
  seq_hex_dump(m, " ", DUMP_PREFIX_OFFSET, 32, 4, mqd, sizeof(struct v9_mqd), false)

Each line:
  " 00000000: c0310800 00000001 ..."
  (8 dwords per line, offset in hex, dwords in little-endian native order)

Usage:
  sudo cat /sys/kernel/debug/kfd/mqds > mqds.txt
  python3 parse_kfd_mqds.py mqds.txt          # parse all processes
  python3 parse_kfd_mqds.py mqds.txt --pid 1234  # filter to one PID
"""

import sys
import re
import struct

# ---------------------------------------------------------------------------
# struct v9_mqd field table: (byte_offset, field_name)
# Source: v9_structs.h, every field is uint32_t (4 bytes)
# ---------------------------------------------------------------------------

# Build the field name list in order; index == dword index, value == name.
_V9_MQD_FIELDS = [
    # dword 0
    "header",
    "compute_dispatch_initiator",
    "compute_dim_x",
    "compute_dim_y",
    "compute_dim_z",
    "compute_start_x",
    "compute_start_y",
    "compute_start_z",
    "compute_num_thread_x",
    "compute_num_thread_y",
    "compute_num_thread_z",
    "compute_pipelinestat_enable",
    "compute_perfcount_enable",
    "compute_pgm_lo",
    "compute_pgm_hi",
    "compute_tba_lo",
    "compute_tba_hi",
    "compute_tma_lo",
    "compute_tma_hi",
    "compute_pgm_rsrc1",
    "compute_pgm_rsrc2",           # 20
    "compute_vmid",
    "compute_resource_limits",
    "compute_static_thread_mgmt_se0",
    "compute_static_thread_mgmt_se1",
    "compute_tmpring_size",
    "compute_static_thread_mgmt_se2",
    "compute_static_thread_mgmt_se3",
    "compute_restart_x",
    "compute_restart_y",
    "compute_restart_z",           # 30
    "compute_thread_trace_enable",
    "compute_misc_reserved",
    "compute_dispatch_id",
    "compute_threadgroup_id",
    "compute_relaunch",
    "compute_wave_restore_addr_lo",
    "compute_wave_restore_addr_hi",
    "compute_wave_restore_control",
    # union dwords 39-42: se4-7 OR xcc-specific (gfx950 uses xcc variant)
    "compute_static_thread_mgmt_se4__OR__compute_current_logic_xcc_id",  # 39
    "compute_static_thread_mgmt_se5__OR__compute_restart_cg_tg_id",      # 40
    "compute_static_thread_mgmt_se6__OR__compute_tg_chunk_size",          # 41
    "compute_static_thread_mgmt_se7__OR__compute_restore_tg_chunk_size",  # 42
] + [f"reserved_{i}" for i in range(43, 65)] + [   # 43-64
    "compute_user_data_0",   # 65
    "compute_user_data_1",
    "compute_user_data_2",
    "compute_user_data_3",
    "compute_user_data_4",
    "compute_user_data_5",
    "compute_user_data_6",
    "compute_user_data_7",
    "compute_user_data_8",
    "compute_user_data_9",
    "compute_user_data_10",
    "compute_user_data_11",
    "compute_user_data_12",
    "compute_user_data_13",
    "compute_user_data_14",
    "compute_user_data_15",          # 80
    "cp_compute_csinvoc_count_lo",   # 81
    "cp_compute_csinvoc_count_hi",
    "reserved_83",
    "reserved_84",
    "reserved_85",
    "cp_mqd_query_time_lo",          # 86
    "cp_mqd_query_time_hi",
    "cp_mqd_connect_start_time_lo",
    "cp_mqd_connect_start_time_hi",
    "cp_mqd_connect_end_time_lo",    # 90
    "cp_mqd_connect_end_time_hi",
    "cp_mqd_connect_end_wf_count",
    "cp_mqd_connect_end_pq_rptr",
    "cp_mqd_connect_end_pq_wptr",
    "cp_mqd_connect_end_ib_rptr",
    "cp_mqd_readindex_lo",           # 96
    "cp_mqd_readindex_hi",
    "cp_mqd_save_start_time_lo",
    "cp_mqd_save_start_time_hi",
    "cp_mqd_save_end_time_lo",       # 100
    "cp_mqd_save_end_time_hi",
    "cp_mqd_restore_start_time_lo",
    "cp_mqd_restore_start_time_hi",
    "cp_mqd_restore_end_time_lo",
    "cp_mqd_restore_end_time_hi",    # 105
    "disable_queue",
    "reserved_107",
    "gds_cs_ctxsw_cnt0",
    "gds_cs_ctxsw_cnt1",
    "gds_cs_ctxsw_cnt2",             # 110
    "gds_cs_ctxsw_cnt3",
    "reserved_112",
    "reserved_113",
    "cp_pq_exe_status_lo",
    "cp_pq_exe_status_hi",
    "cp_packet_id_lo",
    "cp_packet_id_hi",
    "cp_packet_exe_status_lo",
    "cp_packet_exe_status_hi",
    "gds_save_base_addr_lo",         # 120
    "gds_save_base_addr_hi",
    "gds_save_mask_lo",
    "gds_save_mask_hi",
    "ctx_save_base_addr_lo",
    "ctx_save_base_addr_hi",
    "dynamic_cu_mask_addr_lo",
    "dynamic_cu_mask_addr_hi",
    "cp_mqd_base_addr_lo",           # 128
    "cp_mqd_base_addr_hi",
    "cp_hqd_active",                 # 130
    "cp_hqd_vmid",
    "cp_hqd_persistent_state",
    "cp_hqd_pipe_priority",
    "cp_hqd_queue_priority",
    "cp_hqd_quantum",
    "cp_hqd_pq_base_lo",             # 136
    "cp_hqd_pq_base_hi",
    "cp_hqd_pq_rptr",
    "cp_hqd_pq_rptr_report_addr_lo",
    "cp_hqd_pq_rptr_report_addr_hi", # 140
    "cp_hqd_pq_wptr_poll_addr_lo",
    "cp_hqd_pq_wptr_poll_addr_hi",
    "cp_hqd_pq_doorbell_control",
    "reserved_144",
    "cp_hqd_pq_control",             # 145
    "cp_hqd_ib_base_addr_lo",
    "cp_hqd_ib_base_addr_hi",
    "cp_hqd_ib_rptr",
    "cp_hqd_ib_control",
    "cp_hqd_iq_timer",               # 150
    "cp_hqd_iq_rptr",
    "cp_hqd_dequeue_request",
    "cp_hqd_dma_offload",
    "cp_hqd_sema_cmd",
    "cp_hqd_msg_type",
    "cp_hqd_atomic0_preop_lo",
    "cp_hqd_atomic0_preop_hi",
    "cp_hqd_atomic1_preop_lo",
    "cp_hqd_atomic1_preop_hi",       # 159
    "cp_hqd_hq_status0",             # 160
    "cp_hqd_hq_control0",
    "cp_mqd_control",
    "cp_hqd_hq_status1",
    "cp_hqd_hq_control1",
    "cp_hqd_eop_base_addr_lo",       # 165
    "cp_hqd_eop_base_addr_hi",
    "cp_hqd_eop_control",
    "cp_hqd_eop_rptr",
    "cp_hqd_eop_wptr",
    "cp_hqd_eop_done_events",        # 170
    "cp_hqd_ctx_save_base_addr_lo",
    "cp_hqd_ctx_save_base_addr_hi",
    "cp_hqd_ctx_save_control",
    "cp_hqd_cntl_stack_offset",
    "cp_hqd_cntl_stack_size",        # 175
    "cp_hqd_wg_state_offset",
    "cp_hqd_ctx_save_size",
    "cp_hqd_gds_resource_state",
    "cp_hqd_error",
    "cp_hqd_eop_wptr_mem",           # 180
    "cp_hqd_aql_control",
    "cp_hqd_pq_wptr_lo",
    "cp_hqd_pq_wptr_hi",             # 183
] + [f"reserved_{i}" for i in range(184, 192)] + [  # 184-191
    "iqtimer_pkt_header",            # 192
] + [f"iqtimer_pkt_dw{i}" for i in range(32)] + [   # 192-223
    # dword 224: union (pm4_target_xcc_in_xcp or reserved_225)
    "pm4_target_xcc_in_xcp__OR__reserved_225",
    # dword 225: union (cp_mqd_stride_size or reserved_226)
    "cp_mqd_stride_size__OR__reserved_226",          # 225
    "reserved_227",                                  # 226
    "set_resources_header",                          # 227
    "set_resources_dw1",
    "set_resources_dw2",
    "set_resources_dw3",
    "set_resources_dw4",
    "set_resources_dw5",
    "set_resources_dw6",
    "set_resources_dw7",                             # 235
] + [f"reserved_{i}" for i in range(236, 240)] + [  # 236-239
    "queue_doorbell_id0",                            # 240
    "queue_doorbell_id1",
    "queue_doorbell_id2",
    "queue_doorbell_id3",
    "queue_doorbell_id4",
    "queue_doorbell_id5",
    "queue_doorbell_id6",
    "queue_doorbell_id7",
    "queue_doorbell_id8",
    "queue_doorbell_id9",
    "queue_doorbell_id10",
    "queue_doorbell_id11",
    "queue_doorbell_id12",
    "queue_doorbell_id13",
    "queue_doorbell_id14",
    "queue_doorbell_id15",                           # 255
] + [f"reserved_{i}" for i in range(256, 512)]       # 256-511

# Sanity: struct v9_mqd is 512 dwords
assert len(_V9_MQD_FIELDS) == 512, f"Field count mismatch: {len(_V9_MQD_FIELDS)}"

# Fix the iqtimer list: 192 is iqtimer_pkt_header, then dw0..dw31 = dwords 192-223
# The list above over-counts because iqtimer_pkt_header + dw0..dw31 = 33 entries.
# Re-build correctly:
_V9_MQD_FIELDS_CORRECTED = (
    _V9_MQD_FIELDS[:192]  # up to and including iqtimer_pkt_header (index 192 = dword 192... wait)
)
# Actually the iqtimer section: iqtimer_pkt_header is one field, then dw0..dw31 = 32 fields = 33 total
# That spans dwords 192..224. But the union starts at dword 224 (pm4_target_xcc...).
# Let me recount: after dword 191 (reserved_191), iqtimer starts:
#   192: iqtimer_pkt_header
#   193-224: iqtimer_pkt_dw0..dw31  (32 entries)
# So iqtimer ends at dword 224. Then union starts at 225.
# The code above has iqtimer_pkt_header at index 192, then dw0..dw31 (32 entries)
# = indices 192..224, which is 33 entries covering dwords 192-224. That's correct.
# Then pm4_target_xcc starts at dword 225 (index 225 in the list).
# But the struct comments say pm4_target_xcc is at offset 225 (0xE1) and cp_mqd_stride_size at 226 (0xE2).
# set_resources_header is at offset 228 (0xE4) per the struct... wait, let me recount from the struct.

# Re-reading the struct v9_mqd carefully:
# After iqtimer_pkt_dw31 (32 dwords for dw0-31 + 1 for header = 33), then union:
#   { reserved_225, reserved_226 } or { pm4_target_xcc_in_xcp, cp_mqd_stride_size }
# at struct comment "offset: 225 (0xE1)" and "offset: 226 (0xE2)"
# So iqtimer spans dwords 192-224 (33 dwords), and the union is dwords 225-226.
# But in my list above I built:
#   index 192: iqtimer_pkt_header
#   index 193..224: iqtimer_pkt_dw0..dw31  (32 entries, indices 193-224)
#   index 225: pm4_target_xcc...
#   index 226: cp_mqd_stride...
#   index 227: set_resources_header  (but struct says it's at 228, offset 0xE4...)
# Hmm, the struct comments say:
#   iqtimer_pkt_dw31 = last of 32 dwords + header = dword 192+32 = 224
#   then union: offset 225, 226 (0xE1, 0xE2)
#   then reserved_227 at index 226 in 0-based, i.e., dword 227 in the header comment (0xE3)
#   set_resources_header at dword 228 (struct says "after reserved_227")
# Wait, let me look at the struct again. The comments in the struct say:
#   compute_current_logic_xcc_id // offset: 39 (0x27)
#   pm4_target_xcc_in_xcp // offset: 225 (0xE1)
#   cp_mqd_stride_size // offset: 226 (0xE2)
# These are dword offsets. So pm4_target_xcc is at dword 225, cp_mqd_stride at dword 226.
# After iqtimer_pkt_header (dword 192) + iqtimer_pkt_dw0..dw31 (32 dwords = dwords 193-224)
# = total iqtimer range: dwords 192-224 (33 dwords)
# union dwords 225-226; reserved_227 at dword 227; set_resources at dwords 228-235; 
# reserved_236-239 at dwords 236-239; queue_doorbell_id0-15 at dwords 240-255.
# Then reserved_256..511.
# So my list has the correct structure. Let me just verify total count.

# ---------------------------------------------------------------------------
# struct v9_sdma_mqd field table
# ---------------------------------------------------------------------------
_V9_SDMA_MQD_FIELDS = [
    "sdmax_rlcx_rb_cntl",          # 0
    "sdmax_rlcx_rb_base",
    "sdmax_rlcx_rb_base_hi",
    "sdmax_rlcx_rb_rptr",
    "sdmax_rlcx_rb_rptr_hi",
    "sdmax_rlcx_rb_wptr",
    "sdmax_rlcx_rb_wptr_hi",
    "sdmax_rlcx_rb_wptr_poll_cntl",
    "sdmax_rlcx_rb_rptr_addr_hi",
    "sdmax_rlcx_rb_rptr_addr_lo",
    "sdmax_rlcx_ib_cntl",          # 10
    "sdmax_rlcx_ib_rptr",
    "sdmax_rlcx_ib_offset",
    "sdmax_rlcx_ib_base_lo",
    "sdmax_rlcx_ib_base_hi",
    "sdmax_rlcx_ib_size",
    "sdmax_rlcx_skip_cntl",
    "sdmax_rlcx_context_status",
    "sdmax_rlcx_doorbell",
    "sdmax_rlcx_status",
    "sdmax_rlcx_doorbell_log",     # 20
    "sdmax_rlcx_watermark",
    "sdmax_rlcx_doorbell_offset",
    "sdmax_rlcx_csa_addr_lo",
    "sdmax_rlcx_csa_addr_hi",
    "sdmax_rlcx_ib_sub_remain",
    "sdmax_rlcx_preempt",
    "sdmax_rlcx_dummy_reg",
    "sdmax_rlcx_rb_wptr_poll_addr_hi",
    "sdmax_rlcx_rb_wptr_poll_addr_lo",
    "sdmax_rlcx_rb_aql_cntl",     # 30
    "sdmax_rlcx_minor_ptr_update",
    "sdmax_rlcx_midcmd_data0",
    "sdmax_rlcx_midcmd_data1",
    "sdmax_rlcx_midcmd_data2",
    "sdmax_rlcx_midcmd_data3",
    "sdmax_rlcx_midcmd_data4",
    "sdmax_rlcx_midcmd_data5",
    "sdmax_rlcx_midcmd_data6",
    "sdmax_rlcx_midcmd_data7",
    "sdmax_rlcx_midcmd_data8",    # 40
    "sdmax_rlcx_midcmd_cntl",
] + [f"reserved_{i}" for i in range(42, 126)] + [  # 42-125
    "sdma_engine_id",              # 126
    "sdma_queue_id",               # 127
]

assert len(_V9_SDMA_MQD_FIELDS) == 128, f"SDMA field count: {len(_V9_SDMA_MQD_FIELDS)}"


# ---------------------------------------------------------------------------
# Field lookup by name -> dword index
# ---------------------------------------------------------------------------
def _field_index(name):
    for i, f in enumerate(_V9_MQD_FIELDS):
        if f == name or f.startswith(name + "__OR__"):
            return i
    raise KeyError(f"Unknown v9_mqd field: {name}")

def _sdma_field_index(name):
    for i, f in enumerate(_V9_SDMA_MQD_FIELDS):
        if f == name:
            return i
    raise KeyError(f"Unknown v9_sdma_mqd field: {name}")


# ---------------------------------------------------------------------------
# Hex dump line parser
# Line format (from kernel seq_hex_dump DUMP_PREFIX_OFFSET, groupsize=4, rowsize=32):
#   " 00000000: c0310800 00000001 ..."
# ---------------------------------------------------------------------------
_HEX_LINE_RE = re.compile(r'^\s+([0-9a-fA-F]+):\s+((?:[0-9a-fA-F]{8}\s*)+)$')

def parse_hex_dump_lines(lines):
    """Parse seq_hex_dump lines into a flat list of uint32 dwords.

    Uses the byte offset printed on each line to place dwords at the correct
    dword index.  This avoids mis-mapping when the dump starts mid-struct or
    has gaps (e.g. multi-XCC MQD blocks each restart at offset 0x000).
    """
    sparse = {}   # dword_index -> value
    for line in lines:
        m = _HEX_LINE_RE.match(line)
        if not m:
            continue
        byte_off = int(m.group(1), 16)
        for j, dw in enumerate(m.group(2).split()):
            dword_idx = byte_off // 4 + j
            sparse[dword_idx] = int(dw, 16)
    if not sparse:
        return []
    max_idx = max(sparse)
    return [sparse.get(i, 0) for i in range(max_idx + 1)]


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------
def u64(lo, hi):
    return (hi << 32) | lo

def pq_base_addr(lo, hi):
    """Queue ring buffer GPU address from cp_hqd_pq_base_lo/hi (addr >> 8 stored)."""
    return ((hi << 32) | lo) << 8

def rptr_report_addr(lo, hi):
    """Physical address of the CPU-visible rptr location."""
    return u64(lo, hi)

def wptr_poll_addr(lo, hi):
    """Physical address of the CPU-visible wptr location."""
    return u64(lo, hi)

def pq_queue_size_bytes(pq_control):
    """Decode ring size from cp_hqd_pq_control bits[5:0].
    Encoding: order_base_2(queue_size/4) - 1
    So: size = 4 * 2^(field+1) dwords = 4 * 2^(field+1) * 4 bytes.
    Wait: field = order_base_2(queue_size/4) - 1, so queue_size/4 = 2^(field+1)
    => queue_size = 4 * 2^(field+1) dwords? No: queue_size is in bytes in the
    kernel (queue_properties.queue_size). Let's compute in dwords:
    field = order_base_2(bytes/4) - 1 => bytes = 4 * 2^(field+1)
    """
    field = pq_control & 0x3F
    return 4 * (1 << (field + 1))

def doorbell_offset(doorbell_ctrl):
    """Extract doorbell slot offset from cp_hqd_pq_doorbell_control.
    CP_HQD_PQ_DOORBELL_CONTROL__DOORBELL_OFFSET__SHIFT = 21 (gfx9)
    """
    return (doorbell_ctrl >> 21) & 0x3FF

def pgm_addr(lo, hi):
    """compute_pgm_lo/hi encode shader PC >> 8."""
    return ((hi << 32) | lo) << 8

def mqd_base_addr(lo, hi):
    return u64(lo, hi)

def ctx_save_addr(lo, hi):
    return u64(lo, hi)

def eop_base_addr(lo, hi):
    """cp_hqd_eop_base_addr_lo/hi encode EOP ring >> 8."""
    return ((hi << 32) | lo) << 8

def decode_pgm_rsrc2(val):
    """Decode some useful bits of COMPUTE_PGM_RSRC2."""
    trap_present = (val >> 6) & 1
    user_sgpr = (val >> 1) & 0x1F
    tgid_x = (val >> 7) & 1
    tgid_y = (val >> 8) & 1
    tgid_z = (val >> 9) & 1
    lds_size = (val >> 15) & 0x1FF  # in 256-dword units
    return (f"trap={trap_present} user_sgpr={user_sgpr} "
            f"tgid_xyz={tgid_x}{tgid_y}{tgid_z} lds_size={lds_size*256*4}B")

def decode_pq_control(val):
    queue_size_enc = val & 0x3F
    rptr_block = (val >> 8) & 0xF
    unord_dispatch = (val >> 14) & 1
    no_update_rptr = (val >> 27) & 1
    aql_wptr = (val >> 28) & 3  # SLOT_BASED_WPTR
    priv = (val >> 23) & 1
    kmd_queue = (val >> 24) & 1
    return (f"size_enc={queue_size_enc} rptr_block={rptr_block} "
            f"unord={unord_dispatch} no_upd_rptr={no_update_rptr} "
            f"aql_wptr_mode={aql_wptr} priv={priv} kmd={kmd_queue}")

def decode_persistent_state(val):
    preload_req = (val >> 0) & 1
    preload_size = (val >> 8) & 0xFF
    qswitch_mode = (val >> 14) & 1
    disp_obj_id = (val >> 18) & 0x1FFF
    return (f"preload_req={preload_req} preload_size=0x{preload_size:x} "
            f"qswitch_mode={qswitch_mode} disp_obj_id=0x{disp_obj_id:x}")

def decode_hqd_error(val):
    if val == 0:
        return "none"
    bits = []
    if val & (1 << 0): bits.append("SUA_ERROR")
    if val & (1 << 1): bits.append("PRIV_STATE_VIOLATION")
    if val & (1 << 2): bits.append("DMA_WRITE_ERROR")
    if val & (1 << 3): bits.append("DMA_READ_ERROR")
    if val & (1 << 4): bits.append("FATAL_ERROR")
    return f"0x{val:08x} ({', '.join(bits) or 'unknown bits'})"


# ---------------------------------------------------------------------------
# Main decode functions
# ---------------------------------------------------------------------------
def decode_compute_mqd(dwords, xcc_idx, device_id):
    if len(dwords) < 512:
        print(f"    [WARNING] Only {len(dwords)} dwords, expected 512 for v9_mqd")

    def dw(name):
        idx = _field_index(name)
        return dwords[idx] if idx < len(dwords) else 0

    # ----------------------------------------------------------------
    # Print helpers — exact field names, no aliases.
    # p()   prints a single 32-bit field.
    # p64() prints a lo/hi pair each on its own line; the hi line also
    #       shows the combined 64-bit value as a comment.
    #       shift > 0: hardware left-shifts the combined value (e.g. <<8
    #       for base-address registers stored as addr>>8).
    # ----------------------------------------------------------------
    W = 52  # label column width
    def p(name, comment=""):
        v = dw(name)
        c = f"  # {comment}" if comment else ""
        print(f"    {name:<{W}} 0x{v:08x}{c}")

    def p64(lo_name, hi_name, shift=0):
        lo = dw(lo_name)
        hi = dw(hi_name)
        combined = ((hi << 32) | lo) << shift
        shift_str = f" <<{shift}" if shift else ""
        print(f"    {lo_name:<{W}} 0x{lo:08x}")
        print(f"    {hi_name:<{W}} 0x{hi:08x}  # => 0x{combined:016x}{shift_str}")

    header = dw("header")

    print(f"\n--- XCC {xcc_idx} (device 0x{device_id:x}) ---")
    print()

    # dwords 0-38: compute configuration
    p("header")
    p("compute_dispatch_initiator")
    p("compute_dim_x")
    p("compute_dim_y")
    p("compute_dim_z")
    p("compute_start_x")
    p("compute_start_y")
    p("compute_start_z")
    p("compute_num_thread_x")
    p("compute_num_thread_y")
    p("compute_num_thread_z")
    p("compute_pipelinestat_enable")
    p("compute_perfcount_enable")
    p64("compute_pgm_lo",                "compute_pgm_hi",                shift=8)
    p64("compute_tba_lo",                "compute_tba_hi",                shift=8)
    p64("compute_tma_lo",                "compute_tma_hi",                shift=8)
    p("compute_pgm_rsrc1")
    p("compute_pgm_rsrc2")
    p("compute_vmid")
    p("compute_resource_limits")
    p("compute_static_thread_mgmt_se0")
    p("compute_static_thread_mgmt_se1")
    p("compute_tmpring_size")
    p("compute_static_thread_mgmt_se2")
    p("compute_static_thread_mgmt_se3")
    p("compute_restart_x")
    p("compute_restart_y")
    p("compute_restart_z")
    p("compute_thread_trace_enable")
    p("compute_misc_reserved")
    p("compute_dispatch_id")
    p("compute_threadgroup_id")
    p("compute_relaunch")
    p64("compute_wave_restore_addr_lo",  "compute_wave_restore_addr_hi",  shift=8)
    p("compute_wave_restore_control")
    print()

    # dwords 39-42: union (se4-7 on gfx9, XCC-specific on gfx950)
    p("compute_static_thread_mgmt_se4__OR__compute_current_logic_xcc_id")
    p("compute_static_thread_mgmt_se5__OR__compute_restart_cg_tg_id")
    p("compute_static_thread_mgmt_se6__OR__compute_tg_chunk_size")
    p("compute_static_thread_mgmt_se7__OR__compute_restore_tg_chunk_size")
    print()

    # dwords 43-64: reserved (print only if non-zero)
    for _i in range(43, 65):
        _v = dw(f"reserved_{_i}")
        if _v:
            print(f"    {'reserved_' + str(_i):<{W}} 0x{_v:08x}  # non-zero reserved")
    print()

    # dwords 65-80: user data SGPRs
    for _i in range(16):
        p(f"compute_user_data_{_i}")
    print()

    # dwords 81-85
    p64("cp_compute_csinvoc_count_lo",   "cp_compute_csinvoc_count_hi")
    p("reserved_83")
    p("reserved_84")
    p("reserved_85")
    print()

    # dwords 86-107: CP timing / connect snapshot + counters
    p64("cp_mqd_query_time_lo",          "cp_mqd_query_time_hi")
    p64("cp_mqd_connect_start_time_lo",  "cp_mqd_connect_start_time_hi")
    p64("cp_mqd_connect_end_time_lo",    "cp_mqd_connect_end_time_hi")
    p("cp_mqd_connect_end_wf_count")
    p("cp_mqd_connect_end_pq_rptr")
    p("cp_mqd_connect_end_pq_wptr")
    p("cp_mqd_connect_end_ib_rptr")
    p64("cp_mqd_readindex_lo",           "cp_mqd_readindex_hi")
    p64("cp_mqd_save_start_time_lo",     "cp_mqd_save_start_time_hi")
    p64("cp_mqd_save_end_time_lo",       "cp_mqd_save_end_time_hi")
    p64("cp_mqd_restore_start_time_lo",  "cp_mqd_restore_start_time_hi")
    p64("cp_mqd_restore_end_time_lo",    "cp_mqd_restore_end_time_hi")
    p("disable_queue")
    p("reserved_107")
    p("gds_cs_ctxsw_cnt0")
    p("gds_cs_ctxsw_cnt1")
    p("gds_cs_ctxsw_cnt2")
    p("gds_cs_ctxsw_cnt3")
    p("reserved_112")
    p("reserved_113")
    print()

    # dwords 114-127: CP packet state, GDS, ctx save base, dynamic CU mask
    p64("cp_pq_exe_status_lo",           "cp_pq_exe_status_hi")
    p64("cp_packet_id_lo",               "cp_packet_id_hi")
    p64("cp_packet_exe_status_lo",       "cp_packet_exe_status_hi")
    p64("gds_save_base_addr_lo",         "gds_save_base_addr_hi")
    p64("gds_save_mask_lo",              "gds_save_mask_hi")
    p64("ctx_save_base_addr_lo",         "ctx_save_base_addr_hi")
    p64("dynamic_cu_mask_addr_lo",       "dynamic_cu_mask_addr_hi")
    print()

    # dwords 128-183: HQD register image
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
    p("reserved_144")
    p("cp_hqd_pq_control")
    p64("cp_hqd_ib_base_addr_lo",        "cp_hqd_ib_base_addr_hi")
    p("cp_hqd_ib_rptr")
    p("cp_hqd_ib_control")
    p("cp_hqd_iq_timer")
    p("cp_hqd_iq_rptr")
    p("cp_hqd_dequeue_request")
    p("cp_hqd_dma_offload")
    p("cp_hqd_sema_cmd")
    p("cp_hqd_msg_type")
    p64("cp_hqd_atomic0_preop_lo",       "cp_hqd_atomic0_preop_hi")
    p64("cp_hqd_atomic1_preop_lo",       "cp_hqd_atomic1_preop_hi")
    p("cp_hqd_hq_status0")
    p("cp_hqd_hq_control0")
    p("cp_mqd_control")
    p("cp_hqd_hq_status1")
    p("cp_hqd_hq_control1")
    p64("cp_hqd_eop_base_addr_lo",       "cp_hqd_eop_base_addr_hi",       shift=8)
    p("cp_hqd_eop_control")
    p("cp_hqd_eop_rptr")
    p("cp_hqd_eop_wptr")
    p("cp_hqd_eop_done_events")
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
    print()

    # dwords 184-191: reserved
    for _i in range(184, 192):
        _v = dw(f"reserved_{_i}")
        if _v:
            print(f"    {'reserved_' + str(_i):<{W}} 0x{_v:08x}  # non-zero reserved")
    print()

    # dwords 192-224: IQ timer packet
    p("iqtimer_pkt_header")
    for _i in range(32):
        p(f"iqtimer_pkt_dw{_i}")
    print()

    # dwords 224-235: XCC-specific and SET_RESOURCES
    p("pm4_target_xcc_in_xcp__OR__reserved_225")
    p("cp_mqd_stride_size__OR__reserved_226")
    p("reserved_227")
    p("set_resources_header")
    p("set_resources_dw1")
    p("set_resources_dw2")
    p("set_resources_dw3")
    p("set_resources_dw4")
    p("set_resources_dw5")
    p("set_resources_dw6")
    p("set_resources_dw7")
    print()

    # dwords 236-239: reserved
    for _i in range(236, 240):
        _v = dw(f"reserved_{_i}")
        if _v:
            print(f"    {'reserved_' + str(_i):<{W}} 0x{_v:08x}  # non-zero reserved")

    # dwords 240-255: doorbell IDs (print all)
    for _i in range(16):
        p(f"queue_doorbell_id{_i}")
    print()

    # dwords 256-511: reserved upper half — print only non-zero entries
    _nz = [(i, dwords[i]) for i in range(256, min(512, len(dwords))) if dwords[i]]
    if _nz:
        for _i, _v in _nz:
            print(f"    {'reserved_' + str(_i):<{W}} 0x{_v:08x}  # non-zero reserved")

    pass  # all output produced by p() / p64() calls above


def decode_sdma_mqd(dwords, device_id):
    if len(dwords) < 128:
        print(f"    [WARNING] Only {len(dwords)} dwords, expected 128 for v9_sdma_mqd")

    def dw(name):
        idx = _sdma_field_index(name)
        return dwords[idx] if idx < len(dwords) else 0

    rb_cntl = dw("sdmax_rlcx_rb_cntl")
    rb_base_lo = dw("sdmax_rlcx_rb_base")
    rb_base_hi = dw("sdmax_rlcx_rb_base_hi")
    rptr = dw("sdmax_rlcx_rb_rptr")
    rptr_hi = dw("sdmax_rlcx_rb_rptr_hi")
    wptr = dw("sdmax_rlcx_rb_wptr")
    wptr_hi = dw("sdmax_rlcx_rb_wptr_hi")
    rptr_addr_hi = dw("sdmax_rlcx_rb_rptr_addr_hi")
    rptr_addr_lo = dw("sdmax_rlcx_rb_rptr_addr_lo")
    doorbell = dw("sdmax_rlcx_doorbell")
    doorbell_offset_val = dw("sdmax_rlcx_doorbell_offset")
    context_status = dw("sdmax_rlcx_context_status")
    status = dw("sdmax_rlcx_status")
    engine_id = dw("sdma_engine_id")
    queue_id = dw("sdma_queue_id")
    dummy_reg = dw("sdmax_rlcx_dummy_reg")

    ring_addr = ((u64(rb_base_lo, rb_base_hi)) << 8)
    rptr_addr = u64(rptr_addr_lo, rptr_addr_hi)
    rptr_full = u64(rptr, rptr_hi)
    wptr_full = u64(wptr, wptr_hi)

    # rb_cntl: bits[5:1] = ring size (order_base_2(size/4))
    rb_size_enc = (rb_cntl >> 1) & 0x1F
    rb_size_bytes = (4 << rb_size_enc) if rb_size_enc else 0
    rb_vmid = (rb_cntl >> 7) & 0xF
    rptr_writeback = (rb_cntl >> 12) & 1
    rb_enable = (rb_cntl >> 0) & 1

    doorbell_slot = (doorbell_offset_val >> 2) & 0x7FFFF  # bits[21:2] / SDMA DOORBELL_OFFSET field

    indent = "    "
    print(f"\n{indent}--- SDMA engine={engine_id} queue={queue_id} (device 0x{device_id:x}) ---")
    print(f"{indent}ring_addr (GPU):     0x{ring_addr:016x}  (rb_base<<8)")
    print(f"{indent}ring_size:           0x{rb_size_bytes:x} bytes  vmid={rb_vmid}  enable={rb_enable}")
    print(f"{indent}rptr:                0x{rptr_full:016x}")
    print(f"{indent}wptr:                0x{wptr_full:016x}")
    if rptr_full == wptr_full:
        print(f"{indent}queue_idle:          yes (rptr==wptr)")
    else:
        print(f"{indent}queue_idle:          no  (rptr!=wptr, {wptr_full - rptr_full} dwords pending)")
    print(f"{indent}rptr_addr:           0x{rptr_addr:016x}  (CPU-visible)")
    print(f"{indent}rptr_writeback_en:   {rptr_writeback}")
    print(f"{indent}doorbell_offset:     0x{doorbell_slot:x}")
    print(f"{indent}context_status:      0x{context_status:08x}")
    print(f"{indent}status:              0x{status:08x}")
    print(f"{indent}dummy_reg:           0x{dummy_reg:08x}  (expected 0xf)")


# ---------------------------------------------------------------------------
# File parser
# ---------------------------------------------------------------------------
def parse_mqds_file(lines, pid_filter=None):
    """Parse the full /sys/kernel/debug/kfd/mqds output.

    If pid_filter is not None, only queues for that tgid are decoded.
    """

    # State machine
    current_process = None   # (tgid, pasid)
    active = True            # whether current process passes the filter
    current_queue_type = None  # "compute" | "sdma"
    current_device_id = None
    hex_lines_buf = []
    xcc_dwords_list = []     # list of dword lists, one per XCC
    V9_MQD_DWORDS = 512
    V9_SDMA_MQD_DWORDS = 128

    def flush_queue():
        nonlocal xcc_dwords_list, hex_lines_buf
        if not xcc_dwords_list and not hex_lines_buf:
            return
        # Accumulate remaining hex lines
        if hex_lines_buf:
            raw = parse_hex_dump_lines(hex_lines_buf)
            hex_lines_buf = []
            if raw:
                xcc_dwords_list.append(raw)

        if current_queue_type == "compute":
            for xcc_idx, dw_list in enumerate(xcc_dwords_list):
                decode_compute_mqd(dw_list[:V9_MQD_DWORDS], xcc_idx, current_device_id)
        elif current_queue_type == "sdma":
            for xcc_idx, dw_list in enumerate(xcc_dwords_list):
                decode_sdma_mqd(dw_list[:V9_SDMA_MQD_DWORDS], current_device_id)
        xcc_dwords_list = []

    process_re = re.compile(r'^Process\s+(\d+)\s+PASID\s+(\d+):')
    compute_re = re.compile(r'^\s+Compute queue on device\s+(\w+)')
    sdma_re    = re.compile(r'^\s+SDMA queue on device\s+(\w+)')
    diq_re     = re.compile(r'^\s+DIQ on device\s+(\w+)')
    bad_re     = re.compile(r'^\s+Bad')

    for line in lines:
        line = line.rstrip('\n')

        m = process_re.match(line)
        if m:
            flush_queue()
            current_queue_type = None
            current_process = (int(m.group(1)), int(m.group(2)))
            active = (pid_filter is None or current_process[0] == pid_filter)
            if active:
                print(f"\n{'='*70}")
                print(f"Process tgid={current_process[0]}  PASID={current_process[1]}")
                print(f"{'='*70}")
            continue

        if not active:
            continue

        m = compute_re.match(line)
        if m:
            flush_queue()
            current_queue_type = "compute"
            current_device_id = int(m.group(1), 16)
            print(f"\n  [Compute Queue] device=0x{current_device_id:x}")
            hex_lines_buf = []
            xcc_dwords_list = []
            continue

        m = sdma_re.match(line)
        if m:
            flush_queue()
            current_queue_type = "sdma"
            current_device_id = int(m.group(1), 16)
            print(f"\n  [SDMA Queue] device=0x{current_device_id:x}")
            hex_lines_buf = []
            xcc_dwords_list = []
            continue

        m = diq_re.match(line)
        if m:
            flush_queue()
            current_queue_type = "diq"
            current_device_id = int(m.group(1), 16)
            print(f"\n  [DIQ] device=0x{current_device_id:x}  (skipping decode)")
            hex_lines_buf = []
            xcc_dwords_list = []
            continue

        if bad_re.match(line):
            flush_queue()
            current_queue_type = None
            print(f"  [WARN] {line.strip()}")
            continue

        # Hex dump line?
        if _HEX_LINE_RE.match(line):
            # Check if we're starting a new XCC: kernel resets offset to 0 for each XCC MQD.
            m2 = _HEX_LINE_RE.match(line)
            offset = int(m2.group(1), 16)
            if offset == 0 and hex_lines_buf:
                # Flush current buffer as one XCC's worth of data
                raw = parse_hex_dump_lines(hex_lines_buf)
                if raw:
                    xcc_dwords_list.append(raw)
                hex_lines_buf = []
            hex_lines_buf.append(line)
            continue

    # Flush last queue
    flush_queue()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Parse a captured /sys/kernel/debug/kfd/mqds dump file.")
    ap.add_argument("file", nargs="?",
                    help="Path to the captured mqds file (default: stdin)")
    ap.add_argument("--pid", type=int, default=None,
                    help="Show only queues for this PID (default: show all)")
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

    parse_mqds_file(lines, pid_filter=args.pid)


if __name__ == "__main__":
    main()
