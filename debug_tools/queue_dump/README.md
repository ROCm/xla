# ROCm Queue / MQD / HQD Inspection Tools

A small set of scripts for inspecting AMD GPU compute/DMA queues on ROCm
(gfx9 / gfx94x / gfx950 — MI100 / MI200 / MI300 class GPUs). They let you:

- decode the **AQL/SDMA packets** in flight on a running HIP/ROCm process,
- dump the **live HQD hardware registers** for every resident queue, and
- decode the **saved MQD images** the driver keeps per queue.

Together they make it possible to correlate a user-space HSA queue (ring base,
QID) with the hardware slot it landed on (`xcc` / `pipe` / `queue`).

| Script | Source of truth | What it shows |
| --- | --- | --- |
| `aql_packet_decode_gdb.py` | live process memory (via gdb) | Enumerated HSA/SDMA queues, ring base, read/write indices, decoded AQL kernel-dispatch / barrier packets and SDMA ring packets |
| `parse_kfd_hqds.py` | `/sys/kernel/debug/kfd/hqds` | **Live** hardware HQD register image (56 regs) per `xcc`/`pipe`/`queue` — i.e. the actual pipe a queue is mapped to |
| `parse_kfd_mqds.py` | `/sys/kernel/debug/kfd/mqds` | **Saved** `struct v9_mqd` / `v9_sdma_mqd` for every queue of every process (per-XCC) |

## Requirements

- An AMD GPU + ROCm installation (gfx9 family).
- **Debug symbols (required):**

  ```bash
  sudo apt install hip-runtime-amd-dbgsym hsa-rocr-dbgsym
  ```

  These are mandatory. `aql_packet_decode_gdb.py` walks the ROCr runtime
  singleton (`rocr::core::Runtime::runtime_singleton_`) and reads internal
  queue structures by symbol, which is impossible without the
  `hsa-rocr-dbgsym` / `hip-runtime-amd-dbgsym` debug info loaded.
- `gdb` with Python 3 support (for `aql_packet_decode_gdb.py`).
- Python 3 (for the two parsers — standard library only, no pip packages).
- `root` / `sudo` to read the KFD debugfs nodes.

> The `*-dbgsym` packages are published in the same AMD ROCm apt repository you
> used to install ROCm. If `apt` can't find them, make sure that repo (and, on
> some distros, the matching `ddebs`/debug repo) is enabled.

## Usage

### 1. Decode AQL / SDMA packets of a running process

`aql_packet_decode_gdb.py` is a gdb Python script. Attach gdb to your JAX
application and source it; it writes `aql_packet_decode_gdb.txt` in the current
directory.

```bash
# PID of your running JAX application (python3 process executing the script)
gdb -p <PID> -batch -x aql_packet_decode_gdb.py

# output:
#   aql_packet_decode_gdb.txt
```

The header of the output lists every enumerated queue with its device index,
QID and ring base, followed by a per-queue decode of the packets still in the
ring. Debug symbols for librocr must be resolvable for the enumeration to work.

### 2. Dump live HQD hardware registers (May not work under VM)

```bash
sudo cat /sys/kernel/debug/kfd/hqds > hqds.txt
python3 parse_kfd_hqds.py hqds.txt
# or pipe directly:
sudo cat /sys/kernel/debug/kfd/hqds | python3 parse_kfd_hqds.py
```

Each resident queue is printed as a `[LIVE] xcc=.. pipe=.. queue=.. type=..`
block with all 56 HQD registers decoded (ring base, priority, doorbell, etc.).
SDMA (RLC) sections are skipped.

### 3. Decode saved MQDs

```bash
sudo cat /sys/kernel/debug/kfd/mqds > mqds.txt
python3 parse_kfd_mqds.py mqds.txt            # all processes
python3 parse_kfd_mqds.py mqds.txt --pid 1234 # filter to one PID
```

Decodes the full `struct v9_mqd` (512 dwords) for compute queues and
`struct v9_sdma_mqd` for SDMA queues, one block per XCC.

## Notes

- `hqds` / `mqds` are `CONFIG_HSA_AMD` debugfs nodes; they only exist when KFD
  debugfs is mounted and require root to read.
- A queue is only visible in `hqds` while it is actually resident on a CP
  hardware slot. Under the hardware scheduler (MES/HWS) user queues may not be
  pinned to fixed pipes; capture while the queues are busy to maximize the
  chance they are mapped.
- The MQD/HQD field tables target gfx9 / gfx950 (`v9_structs.h`). Other ASIC
  generations have different layouts and are not decoded by these parsers.

## Files

```
aql_packet_decode_gdb.py   # gdb script: enumerate queues + decode AQL/SDMA packets
parse_kfd_hqds.py          # parse /sys/kernel/debug/kfd/hqds (live HQD regs)
parse_kfd_mqds.py          # parse /sys/kernel/debug/kfd/mqds (saved MQDs)
```
